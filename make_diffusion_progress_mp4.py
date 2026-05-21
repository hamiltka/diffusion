"""
Build an MP4 that shows GEODiffusion predictions progressing over checkpoints.

Example:
    conda run -n trace_geo python make_diffusion_progress_mp4.py \
        --checkpoints_dir runs/checkpoints/flow_matching_subset \
        --data_root /path/to/usgs_crops_512_trace_2_NAIP \
        --split val \
        --sample_index 0 \
        --epoch_start 0 \
        --epoch_end 520 \
        --output runs/inference/flow_matching_subset_epoch0_520.mp4
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

import subprocess
import tempfile

import cv2
import numpy as np
import torch
from omegaconf import OmegaConf

# Project imports
sys.path.insert(0, os.path.dirname(__file__))
from geodiffusion.dataloader.dataset import VectorRoadDataset
from geodiffusion.lightning.lightning_module import VectorFlowLightningModule


def _extract_epoch_from_name(path: Path) -> int | None:
    """Best-effort parse for epoch number in checkpoint filename."""
    m = re.search(r"epoch(?:=|_)?(\d+)", path.stem)
    if m:
        return int(m.group(1))
    return None


def _read_epoch_from_ckpt(path: Path) -> int | None:
    """Fallback epoch read directly from checkpoint metadata."""
    try:
        ckpt = torch.load(str(path), map_location="cpu")
    except Exception:
        return None
    ep = ckpt.get("epoch", None)
    if ep is None:
        return None
    try:
        return int(ep)
    except Exception:
        return None


def _discover_checkpoints(checkpoints_dir: Path, epoch_start: int, epoch_end: int) -> list[tuple[int, Path]]:
    """Return sorted (epoch, path) checkpoint pairs within [epoch_start, epoch_end]."""
    candidates = sorted(checkpoints_dir.glob("*.ckpt"))
    out: list[tuple[int, Path]] = []

    for ckpt_path in candidates:
        epoch = _extract_epoch_from_name(ckpt_path)
        if epoch is None:
            epoch = _read_epoch_from_ckpt(ckpt_path)
        if epoch is None:
            continue
        if epoch_start <= epoch <= epoch_end:
            out.append((epoch, ckpt_path))

    # Keep one checkpoint per epoch (first one encountered after sort).
    dedup: dict[int, Path] = {}
    for epoch, ckpt_path in out:
        dedup.setdefault(epoch, ckpt_path)

    return sorted(dedup.items(), key=lambda x: x[0])


def _norm_to_px(segs_norm: np.ndarray, image_size: int) -> np.ndarray:
    """Convert [M,4] segments from [-1,1] to pixel coordinates."""
    if segs_norm.size == 0:
        return np.zeros((0, 4), dtype=np.float32)
    half = image_size / 2.0
    px = segs_norm.copy().astype(np.float32)
    px[:, [0, 2]] = (px[:, [0, 2]] + 1.0) * half
    px[:, [1, 3]] = (px[:, [1, 3]] + 1.0) * half
    return px


def _draw_segments_bgr(
    img_bgr: np.ndarray,
    segs_px: np.ndarray,
    color: tuple[int, int, int],
    thickness: int = 1,
    endpoint_color: tuple[int, int, int] | None = None,
    endpoint_radius: int = 4,
) -> None:
    for x1, y1, x2, y2 in segs_px:
        p1 = (int(round(x1)), int(round(y1)))
        p2 = (int(round(x2)), int(round(y2)))
        cv2.line(img_bgr, p1, p2, color, thickness=thickness, lineType=cv2.LINE_AA)
        if endpoint_color is not None:
            cv2.circle(img_bgr, p1, endpoint_radius, endpoint_color, -1, lineType=cv2.LINE_AA)
            cv2.circle(img_bgr, p2, endpoint_radius, endpoint_color, -1, lineType=cv2.LINE_AA)


def _load_sample(data_root: str, split: str, sample_index: int, image_size: int) -> tuple[np.ndarray, np.ndarray, str]:
    """Load one dataset sample and return image RGB, GT segments in norm coords, and sample id."""
    ds = VectorRoadDataset(
        data_root=data_root,
        split=split,
        densify=True,
        max_segment_length=0.06,
        augment=False,
        image_size=image_size,
        use_exclusion_csv=False,
    )
    if sample_index < 0 or sample_index >= len(ds):
        raise IndexError(f"sample_index {sample_index} out of range [0, {len(ds)-1}]")

    sample = ds[sample_index]
    sample_id = ds.sample_ids[sample_index]
    img_rgb = sample["image"].permute(1, 2, 0).numpy().copy()  # uint8 H,W,3
    gt_norm = sample["road_data"][~sample["invalid_mask"]].numpy().astype(np.float32)
    return img_rgb, gt_norm, sample_id


def _load_module(ckpt_path: Path, device: torch.device) -> VectorFlowLightningModule:
    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    cfg = OmegaConf.create(ckpt["hyper_parameters"]["cfg"])

    # Allow inference-time overrides if needed.
    OmegaConf.set_struct(cfg, False)

    module = VectorFlowLightningModule(cfg)
    module.load_state_dict(ckpt["state_dict"], strict=False)
    module.eval().to(device)
    return module


def _predict_segments(
    module: VectorFlowLightningModule,
    img_rgb: np.ndarray,
    device: torch.device,
    euler_steps: int | None,
    active_threshold: float | None,
) -> np.ndarray:
    """Run one forward sampling pass and return active predicted segments in norm coords."""
    img_t = torch.from_numpy(img_rgb).permute(2, 0, 1).float().unsqueeze(0).to(device) / 255.0

    steps = int(euler_steps) if euler_steps is not None else int(module.euler_steps_eval)
    thr = float(active_threshold) if active_threshold is not None else float(module.active_threshold)

    with torch.no_grad():
        x0 = module.anchors.generate(1, device)
        x1pred = module.flow.euler_integrate(x0, module.model, img_t, steps=steps, device=device)

    active = x1pred[0, :, 4] > thr
    return x1pred[0, active, :4].detach().cpu().numpy().astype(np.float32)


def _make_frame(
    img_rgb: np.ndarray,
    gt_norm: np.ndarray,
    pred_norm: np.ndarray,
    epoch: int,
    sample_id: str,
    image_size: int,
) -> np.ndarray:
    """Create a side-by-side frame: satellite | GT overlay | prediction overlay."""
    sat = img_rgb.copy()
    gt = img_rgb.copy()
    pred = img_rgb.copy()

    gt_px = _norm_to_px(gt_norm, image_size)
    pred_px = _norm_to_px(pred_norm, image_size)

    # Colors are BGR in OpenCV.
    gt_bgr = cv2.cvtColor(gt, cv2.COLOR_RGB2BGR)
    pred_bgr = cv2.cvtColor(pred, cv2.COLOR_RGB2BGR)
    sat_bgr = cv2.cvtColor(sat, cv2.COLOR_RGB2BGR)

    # GT: cyan lines, yellow endpoints.  Pred: magenta lines (active only), bright-orange endpoints.
    _draw_segments_bgr(gt_bgr,   gt_px,   color=(255, 255, 0), thickness=2, endpoint_color=(0, 255, 255), endpoint_radius=4)
    _draw_segments_bgr(pred_bgr, pred_px, color=(255, 0, 255), thickness=2, endpoint_color=(0, 165, 255), endpoint_radius=4)

    panel = np.concatenate([sat_bgr, gt_bgr, pred_bgr], axis=1)

    # Header strip with epoch/sample metadata.
    header_h = 56
    canvas = np.zeros((panel.shape[0] + header_h, panel.shape[1], 3), dtype=np.uint8)
    canvas[header_h:] = panel

    title = f"GEODiffusion progression | sample={sample_id} | epoch={epoch:04d}"
    subtitle = f"GT segments={len(gt_norm)} | Pred segments={len(pred_norm)}"
    cv2.putText(canvas, title, (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (245, 245, 245), 2, cv2.LINE_AA)
    cv2.putText(canvas, subtitle, (12, 47), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (190, 190, 190), 1, cv2.LINE_AA)

    # Column labels.
    w = image_size
    cv2.putText(canvas, "Satellite", (w // 2 - 40, header_h + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1, cv2.LINE_AA)
    cv2.putText(canvas, "Ground Truth", (w + w // 2 - 55, header_h + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 1, cv2.LINE_AA)
    cv2.putText(canvas, "Prediction", (2 * w + w // 2 - 45, header_h + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 0, 255), 1, cv2.LINE_AA)

    return canvas


def main() -> None:
    parser = argparse.ArgumentParser(description="Create MP4 of prediction progression across checkpoints.")
    parser.add_argument("--checkpoints_dir", required=True, help="Directory with .ckpt files.")
    parser.add_argument("--data_root", required=True, help="Dataset root used to load a fixed sample.")
    parser.add_argument("--split", default="val", help="Dataset split for fixed sample [val].")
    parser.add_argument("--sample_index", type=int, default=0, help="Index of sample inside split [0].")
    parser.add_argument("--epoch_start", type=int, default=0, help="First epoch to include [0].")
    parser.add_argument("--epoch_end", type=int, default=520, help="Last epoch to include [520].")
    parser.add_argument("--epoch_stride", type=int, default=1, help="Keep every Nth epoch [1].")
    parser.add_argument("--output", required=True, help="Output MP4 path.")
    parser.add_argument("--fps", type=int, default=10, help="Video frame rate [10].")
    parser.add_argument("--image_size", type=int, default=512, help="Image size [512].")
    parser.add_argument("--euler_steps", type=int, default=None, help="Override Euler steps.")
    parser.add_argument("--active_threshold", type=float, default=None, help="Override active threshold.")
    parser.add_argument("--device", default="cuda:0", help="Torch device [cuda:0].")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    checkpoints_dir = Path(args.checkpoints_dir)
    output_path = Path(args.output)

    if not checkpoints_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoints_dir}")

    ckpts = _discover_checkpoints(checkpoints_dir, args.epoch_start, args.epoch_end)
    if args.epoch_stride > 1:
        ckpts = [pair for i, pair in enumerate(ckpts) if i % args.epoch_stride == 0]
    if not ckpts:
        raise RuntimeError("No checkpoints found in the requested epoch range.")

    print(f"Found {len(ckpts)} checkpoints in [{args.epoch_start}, {args.epoch_end}].")

    img_rgb, gt_norm, sample_id = _load_sample(
        data_root=args.data_root,
        split=args.split,
        sample_index=args.sample_index,
        image_size=args.image_size,
    )

    # Build first frame to determine output dimensions.
    first_epoch, first_ckpt = ckpts[0]
    first_module = _load_module(first_ckpt, device=device)
    first_pred = _predict_segments(
        module=first_module,
        img_rgb=img_rgb,
        device=device,
        euler_steps=args.euler_steps,
        active_threshold=args.active_threshold,
    )
    first_frame = _make_frame(
        img_rgb=img_rgb,
        gt_norm=gt_norm,
        pred_norm=first_pred,
        epoch=first_epoch,
        sample_id=sample_id,
        image_size=args.image_size,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    h, w = first_frame.shape[:2]

    # Write frames to a temporary mp4v file, then re-encode to H.264 with ffmpeg.
    # H.264 + yuv420p is natively supported by Google Slides without re-compression.
    tmp_file = tempfile.NamedTemporaryFile(suffix="_raw.mp4", delete=False, dir=output_path.parent)
    tmp_path = Path(tmp_file.name)
    tmp_file.close()

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(tmp_path), fourcc, float(args.fps), (w, h))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open VideoWriter for temp output: {tmp_path}")

    try:
        writer.write(first_frame)
        print(f"[1/{len(ckpts)}] epoch={first_epoch:04d} -> {first_ckpt.name}  pred={len(first_pred)}")

        for i, (epoch, ckpt_path) in enumerate(ckpts[1:], start=2):
            module = _load_module(ckpt_path, device=device)
            pred = _predict_segments(
                module=module,
                img_rgb=img_rgb,
                device=device,
                euler_steps=args.euler_steps,
                active_threshold=args.active_threshold,
            )
            frame = _make_frame(
                img_rgb=img_rgb,
                gt_norm=gt_norm,
                pred_norm=pred,
                epoch=epoch,
                sample_id=sample_id,
                image_size=args.image_size,
            )
            writer.write(frame)
            print(f"[{i}/{len(ckpts)}] epoch={epoch:04d} -> {ckpt_path.name}  pred={len(pred)}")
    finally:
        writer.release()

    # Re-encode to H.264 (libopenh264) for Google Slides / browser compatibility.
    # High bitrate (8M) keeps quality sharp; yuv420p is required for wide support.
    print("Re-encoding to H.264 (ffmpeg libopenh264)...")
    # Use the ffmpeg binary from the active conda environment if on PATH.
    import shutil
    ffmpeg_bin = shutil.which("ffmpeg") or "ffmpeg"
    ffmpeg_cmd = [
        ffmpeg_bin, "-y",
        "-i", str(tmp_path),
        "-vcodec", "libopenh264",
        "-b:v", "8M",
        "-pix_fmt", "yuv420p",
        str(output_path),
    ]
    result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True)
    tmp_path.unlink(missing_ok=True)  # clean up temp file
    if result.returncode != 0:
        print(f"ffmpeg failed (returncode={result.returncode}). Raw temp file was at {tmp_path}")
        print(result.stderr[-2000:])
        raise RuntimeError("ffmpeg re-encode failed")

    print(f"Saved MP4 (H.264): {output_path}")


if __name__ == "__main__":
    main()
