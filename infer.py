"""
GEODiffusion inference script.

Runs the trained flow-matching model on a directory of satellite images
and writes predicted road segments as GeoJSON files.

Usage:
    conda activate trace_geo
    python infer.py \
        --checkpoint runs/checkpoints/flow_matching/best.ckpt \
        --input_dir  /path/to/sat_images \
        --output_dir ./predictions \
        [--euler_steps 20] \
        [--active_threshold 0.3] \
        [--device cuda:0]

Input:
    Directory containing *_sat.jpg files (512x512 uint8 RGB).
    Optionally paired *_roads.geojson for GT overlay figures.

Output:
    <output_dir>/<stem>_pred.geojson  — predicted road segments
    <output_dir>/<stem>_viz.png       — overlay figure (if --visualize)
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch
from PIL import Image
from omegaconf import OmegaConf

# ── project imports ────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))
from geodiffusion.model.transformer import TransformerModel  # noqa: F401  (needed for hydra instantiation)
from geodiffusion.model.flow import FlowMatching
from geodiffusion.anchors.spoke_wheel import SpokeWheelAnchors
from geodiffusion.lightning.lightning_module import VectorFlowLightningModule
from geodiffusion.utils.snap import snap_endpoints_inplace, merge_collinear_segments


# ── helpers ────────────────────────────────────────────────────────────────────

def load_image(path: str, size: int = 512) -> torch.Tensor:
    """Load a JPEG satellite image → uint8 [3, H, W] tensor."""
    img = Image.open(path).convert("RGB")
    if img.size != (size, size):
        img = img.resize((size, size), Image.BILINEAR)
    return torch.from_numpy(np.array(img)).permute(2, 0, 1)  # [3, H, W] uint8


def segs_to_geojson(segs: np.ndarray, img_size: int = 512) -> dict:
    """Convert predicted segments from [-1,1] coords to pixel-space GeoJSON.

    Each segment (x1,y1,x2,y2) in [-1,1] becomes a LineString feature
    with pixel coordinates (useful for overlay / visual inspection).
    """
    features = []
    for x1, y1, x2, y2 in segs:
        px1 = float((x1 + 1) / 2 * img_size)
        py1 = float((y1 + 1) / 2 * img_size)
        px2 = float((x2 + 1) / 2 * img_size)
        py2 = float((y2 + 1) / 2 * img_size)
        features.append({
            "type": "Feature",
            "geometry": {"type": "LineString", "coordinates": [[px1, py1], [px2, py2]]},
            "properties": {}
        })
    return {"type": "FeatureCollection", "features": features}


def make_viz(img_np: np.ndarray, pred_segs: np.ndarray,
             gt_segs: np.ndarray | None, out_path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    size = img_np.shape[0]
    ncols = 3 if gt_segs is not None else 2
    fig, axes = plt.subplots(1, ncols, figsize=(ncols * 4, 4), dpi=120)

    # Satellite
    axes[0].imshow(img_np)
    axes[0].set_title("Satellite", fontsize=9)
    axes[0].axis("off")

    # Predictions
    axes[1].imshow(img_np)
    for x1, y1, x2, y2 in pred_segs:
        xs = [(x1 + 1) / 2 * size, (x2 + 1) / 2 * size]
        ys = [(y1 + 1) / 2 * size, (y2 + 1) / 2 * size]
        axes[1].plot(xs, ys, color="#FF00FF", linewidth=1.5, alpha=0.85, solid_capstyle="round")
    axes[1].set_title(f"Predicted ({len(pred_segs)} segs)", fontsize=9)
    axes[1].axis("off")

    # GT overlay (if available)
    if gt_segs is not None:
        axes[2].imshow(img_np)
        for x1, y1, x2, y2 in gt_segs:
            xs = [(x1 + 1) / 2 * size, (x2 + 1) / 2 * size]
            ys = [(y1 + 1) / 2 * size, (y2 + 1) / 2 * size]
            axes[2].plot(xs, ys, color="#00FFFF", linewidth=1.5, alpha=0.85, solid_capstyle="round")
        axes[2].set_title(f"Ground Truth ({len(gt_segs)} segs)", fontsize=9)
        axes[2].axis("off")

    plt.tight_layout(pad=0.5)
    plt.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


# ── main ───────────────────────────────────────────────────────────────────────

def run(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load checkpoint
    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location="cpu")

    # Reconstruct config from checkpoint hyper_parameters
    cfg = OmegaConf.create(ckpt["hyper_parameters"]["cfg"])

    # Override inference-time settings (disable struct so new keys can be added)
    OmegaConf.set_struct(cfg, False)
    cfg.training.euler_steps_eval = args.euler_steps
    cfg.training.active_threshold = args.active_threshold

    # Instantiate module and load weights
    module = VectorFlowLightningModule(cfg)
    # Strip "model." prefix if saved with Lightning wrapper
    state = {k.replace("model.", "", 1) if k.startswith("model.") else k: v
             for k, v in ckpt["state_dict"].items()}
    # Use strict=False to tolerate minor key mismatches
    missing, unexpected = module.load_state_dict(ckpt["state_dict"], strict=False)
    if missing:
        print(f"  Missing keys: {missing[:5]}{'...' if len(missing) > 5 else ''}")
    if unexpected:
        print(f"  Unexpected keys: {unexpected[:5]}{'...' if len(unexpected) > 5 else ''}")

    module.eval().to(device)
    anchor_mode = str(cfg.anchors.get("mode", "grid"))
    if anchor_mode == "gt_seeded":
        raise RuntimeError(
            "Checkpoint config uses anchors.mode=gt_seeded, but infer.py has no GT anchors at test time. "
            "Run inference with a checkpoint trained using anchors.mode=grid, or retrain with grid anchors."
        )
    print(f"  Active threshold : {module.active_threshold}")
    print(f"  Euler steps      : {module.euler_steps_eval}")
    print(f"  Anchor mode      : {anchor_mode}")

    # Collect input images
    input_dir = args.input_dir
    image_files = sorted(f for f in os.listdir(input_dir) if f.endswith("_sat.jpg"))
    if not image_files:
        # Also accept plain .jpg
        image_files = sorted(f for f in os.listdir(input_dir) if f.lower().endswith(".jpg"))
    if not image_files:
        print(f"No *_sat.jpg files found in {input_dir}")
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Running inference on {len(image_files)} images → {args.output_dir}")

    for fname in image_files:
        stem = fname.replace("_sat.jpg", "").replace(".jpg", "")
        img_path = os.path.join(input_dir, fname)

        # Load image
        img_tensor = load_image(img_path).to(device)           # [3, H, W] uint8
        img_float  = img_tensor.float().unsqueeze(0) / 255.0   # [1, 3, H, W]
        img_np     = img_tensor.permute(1, 2, 0).cpu().numpy() # H, W, 3  uint8

        # Generate anchors
        x0 = module.anchors.generate(1, device)  # [1, N, 5]

        # Euler integration
        with torch.no_grad():
            x1pred = module.flow.euler_integrate(
                x0, module.model, img_float,
                steps=module.euler_steps_eval,
                device=device,
            )  # [1, N, 5]

        if getattr(module, "snap_endpoints", True) and getattr(module, "snap_mode", "self") == "self":
            active_mask = x1pred[:, :, 4] > module.active_threshold
            x1pred = snap_endpoints_inplace(
                x1pred,
                threshold=getattr(module, "snap_threshold", 0.02),
                valid_mask=active_mask,
                active_scores=x1pred[:, :, 4],
            )

        if args.collinear_merge:
            active_mask_cm = x1pred[:, :, 4] > module.active_threshold
            x1pred, _, _ = merge_collinear_segments(
                x1pred,
                threshold=args.collinear_merge_threshold,
                angle_tol_deg=args.collinear_merge_angle_tol,
                valid_mask=active_mask_cm,
                active_scores=x1pred[:, :, 4],
            )

        # Apply active threshold
        active_mask = x1pred[0, :, 4] > module.active_threshold
        pred_segs   = x1pred[0, active_mask, :4].cpu().numpy()  # [M, 4]

        print(f"  {fname}: {len(pred_segs)} active segments (threshold={module.active_threshold})")

        # Write GeoJSON
        geojson_path = os.path.join(args.output_dir, f"{stem}_pred.geojson")
        with open(geojson_path, "w") as f:
            json.dump(segs_to_geojson(pred_segs), f)

        # Optional visualisation
        if args.visualize:
            gt_segs = None
            gt_path = os.path.join(input_dir, f"{stem}_osm.geojson")
            if os.path.exists(gt_path):
                # Use the dataset to convert geo coords → normalised [-1,1] pixel coords
                from geodiffusion.dataloader.dataset import VectorRoadDataset
                ds = VectorRoadDataset(input_dir, split=None, augment=False)
                # Find this image in the dataset
                for idx in range(len(ds)):
                    if os.path.basename(ds.sat_images[idx]) == fname:
                        sample = ds[idx]
                        valid = sample["road_data"][~sample["invalid_mask"]]
                        gt_segs = valid.numpy() if len(valid) else None
                        break

            viz_path = os.path.join(args.output_dir, f"{stem}_viz.png")
            make_viz(img_np, pred_segs, gt_segs, viz_path)

    print(f"\nDone. Results written to {args.output_dir}")


def main():
    parser = argparse.ArgumentParser(description="GEODiffusion inference")
    parser.add_argument("--checkpoint",       required=True,
                        help="Path to .ckpt file")
    parser.add_argument("--input_dir",        required=True,
                        help="Directory with *_sat.jpg files")
    parser.add_argument("--output_dir",       default="./predictions",
                        help="Where to write *_pred.geojson [./predictions]")
    parser.add_argument("--euler_steps",      type=int,   default=20,
                        help="Euler integration steps (more = more accurate) [20]")
    parser.add_argument("--active_threshold", type=float, default=0.3,
                        help="Active channel cutoff [0.3]")
    parser.add_argument("--device",           default="cuda:0",
                        help="Torch device [cuda:0]")
    parser.add_argument("--visualize",        action="store_true",
                        help="Save *_viz.png overlay figures alongside GeoJSON")
    parser.add_argument("--collinear_merge",   action="store_true",
                        help="Merge collinear adjacent segments after snap")
    parser.add_argument("--collinear_merge_threshold", type=float, default=0.02,
                        help="Endpoint gap threshold for collinear merge [0.02]")
    parser.add_argument("--collinear_merge_angle_tol", type=float, default=15.0,
                        help="Angle tolerance (degrees) for collinear merge [15.0]")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
