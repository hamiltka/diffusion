"""
PyTorch Lightning module for vector road-network **flow matching** training.

Training loop
─────────────
1. Batch arrives: satellite image + GT road segments (x1,y1,x2,y2).
2. Spoke-wheel anchors are sampled as source distribution x₀.
3. Hungarian matching assigns each GT segment to a unique anchor → targets x₁.
4. Random t ~ U(0,1); interpolate x_t = (1-t)·x₀ + t·x₁.
5. Velocity field v_θ(x_t, t, image) is predicted.
6. Three losses: segment (perm-invariant MSE on matched endpoints),
   active (regression of active channel to ±1), connectivity (shared nodes).

TensorBoard signals
───────────────────
Scalars — every training step:
    Train/seg_loss        matched-anchor endpoint velocity MSE
    Train/active_loss     active-channel regression MSE
    Train/conn_loss       shared-node convergence penalty
    Train/match_fraction  fraction of anchors matched to a GT segment
    Train/active_accuracy fraction of anchors with correct active sign

Scalars — every epoch:
    Loss/train            total weighted training loss
    Loss/val              total weighted val loss  (teacher-forced, all batches)
    Val/seg_loss          val segment loss component
    Val/active_loss       val active loss component
    Val/conn_loss         val connectivity loss component
    Val/match_fraction    val match fraction
    Val/active_accuracy   val active accuracy
    Train/lr              current learning rate

Scalars — every val_image_log_every_n_epochs epochs (full Euler integration):
    Metrics/precision_05        segment precision @ τ=0.05
    Metrics/recall_05           segment recall    @ τ=0.05
    Metrics/F1_05               segment F1        @ τ=0.05
    Metrics/precision_10        segment precision @ τ=0.10
    Metrics/recall_10           segment recall    @ τ=0.10
    Metrics/F1_10               segment F1        @ τ=0.10
    Metrics/mean_active_count   mean predicted active segments per image
    Metrics/mean_gt_count       mean GT segments per image

Images — every val_image_log_every_n_epochs epochs:
    Val/qual_grid             4 samples × 5 cols: Sat|GT|Predicted|Confidence|Error
    Val/flow_trajectory       2 samples × 6 cols: t=0→0.25→0.5→0.75→1.0|GT
    Val/snap_closeups         5 close-ups: original endpoints → snapped midpoint
    Val/pr_curve              P/R/F1 vs threshold + Precision-Recall curve
    Val/soft_road_map         sigmoid-weighted segment rasterization heatmap
    Val/velocity_field        per-endpoint displacement arrows x₀→x₁, coloured by active score
    Val/endpoint_density      2-D endpoint density histogram across all eval images
    Val/calibration_scatter   active score vs. distance-to-GT scatter (hexbin)
    Val/snap_distance_hist    histogram of merged endpoint gap sizes
"""
from __future__ import annotations

import csv
import os
import random
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf

from geodiffusion.model.transformer import TransformerModel
from geodiffusion.model.flow import FlowMatching
from geodiffusion.anchors.spoke_wheel import SpokeWheelAnchors
from geodiffusion.matching.matcher import build_targets
from geodiffusion.losses.losses import DistanceLoss, ActiveLoss, ConnectivityLoss
from geodiffusion.metrics.segment_metrics import (
    segment_precision_recall_f1,
    pi_dist_matrix,
    segment_density_bucket,
    DENSITY_BUCKETS,
)

# ── colour palette  (no green, no white — poor contrast on satellite imagery) ──
_GT_COLOR       = "#FF8C00"   # orange        — GT segments
_GT_EP_COLOR    = "#FFE000"   # bright yellow — GT endpoints  (≠ orange)
_ANC_COLOR      = "#4DBBFF"   # sky-blue      — x₀ anchor segments
_ANC_EP_COLOR   = "#FF69B4"   # hot-pink      — x₀ anchor endpoints  (≠ sky-blue)
_ACT_COLOR      = "#FFE000"   # yellow        — active predicted segments
_ACT_EP_COLOR   = "#00FFFF"   # cyan          — active pred endpoints  (≠ yellow)
_INACT_COLOR    = "#FF00FF"   # magenta       — inactive predicted segments
_INACT_EP_COLOR = "#FF6600"   # deep-orange   — inactive pred endpoints  (≠ magenta)
_ARROW_COLOR    = "#00FFFF"   # cyan          — flow arrows on dark background


class VectorFlowLightningModule(pl.LightningModule):
    """
    Lightning module for conditional flow-matching road network generation.
    See module docstring for full description of all logged signals.
    """

    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.save_hyperparameters()
        self.cfg = cfg
        tc = cfg.training
        ac = cfg.anchors
        lc = cfg.loss

        # ── velocity-field model ─────────────────────────────────────────────
        n_anchors = ac.grid_size * ac.grid_size * ac.n_spokes
        self.model = TransformerModel(
            max_segments=n_anchors,
            img_feature_dim=int(cfg.model.get("img_feature_dim", 512)),
        )

        # ── flow & anchor modules ─────────────────────────────────────────────
        self.flow = FlowMatching()
        self.anchors = SpokeWheelAnchors(ac)

        # ── losses ───────────────────────────────────────────────────────────
        self.dist_loss_fn = DistanceLoss()
        self.act_loss_fn  = ActiveLoss(pos_weight=float(lc.get("active_pos_weight", 6.0)))
        self.conn_loss_fn = ConnectivityLoss(node_eps=float(lc.get("node_eps", 0.02)))
        self.w_seg  = float(lc.get("segment_weight",      1.0))
        self.w_act  = float(lc.get("active_weight",       0.5))
        self.w_conn = float(lc.get("connectivity_weight", 0.1))

        # ── eval / visualisation config ───────────────────────────────────────
        self.val_image_log_every_n_epochs: int = int(
            cfg.get("val_image_log_every_n_epochs", 2)
        )
        self.n_metric_samples: int  = int(tc.get("n_metric_samples",  64))
        self.euler_steps_eval: int  = int(tc.get("euler_steps_eval",  10))
        self.active_threshold: float = float(tc.get("active_threshold", 0.3))

        # ── per-epoch loss history (rank-0 only, for loss-curve figure) ────────
        self._train_loss_hist: list[tuple[int, float]] = []
        self._val_loss_hist:   list[tuple[int, float]] = []
        self._step_train_losses: list[float] = []  # accumulate within each epoch
        # ── per-step active diagnostics (written to CSV at epoch end) ─────────
        self._step_active_pred_means: list[float] = []
        self._step_gt_active_fracs:   list[float] = []

        # ── locked grid-sample indices (set once, then fixed for all epochs) ─
        # Each dict: {"sparse": [i, j], "medium": [i, j], "dense": [i, j]}
        self._train_grid_idx: dict | None = None
        self._val_grid_idx:   dict | None = None

        # ── locked eval indices for comprehensive eval ────────────────────────
        self._eval_indices: list[int] | None = None
        self._viz_indices:  list[int] | None = None

    # ──────────────────────────────────────────────────────────────────────────
    # Forward
    # ──────────────────────────────────────────────────────────────────────────

    def forward(self, xt, t, image=None):
        return self.model(xt, t, image)

    # ──────────────────────────────────────────────────────────────────────────
    # Training step
    # ──────────────────────────────────────────────────────────────────────────

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor | None:
        import os
        import math
        # ...existing code...
        if batch is None or batch.get("_skip_batch", False):
            z = torch.zeros(1, device=self.device).squeeze()
            self.log("Train/seg_loss",       z, on_step=True, on_epoch=False, prog_bar=False, sync_dist=True)
            self.log("Train/active_loss",    z, on_step=True, on_epoch=False, prog_bar=False, sync_dist=True)
            self.log("Train/conn_loss",      z, on_step=True, on_epoch=False, prog_bar=False, sync_dist=True)
            self.log("Loss/train",           z, on_step=True, on_epoch=True,  prog_bar=True,  sync_dist=True)
            self.log("Train/match_fraction", z, on_step=True, on_epoch=False, prog_bar=False, sync_dist=True)
            self.log("Train/active_accuracy",z, on_step=True, on_epoch=False, prog_bar=False, sync_dist=True)
            return None

        images  = batch["image"].float() / 255.0   # [B, 3, H, W]
        gt_segs = batch["road_data"]               # [B, max_gt, 4]
        invalid = batch["invalid_mask"]            # [B, max_gt]
        B = images.shape[0]
        if self.cfg.anchors.get("mode", "grid") == "gt_seeded":
            x0 = self.anchors.generate_from_gt(gt_segs, invalid)    # [B, N, 5]
        else:
            x0 = self.anchors.generate(B, self.device)              # [B, N, 5]
        targets, matched_indices, junction_pairs = build_targets(
            x0, gt_segs, invalid, node_eps=self.conn_loss_fn.node_eps
        )  # [B,N,5], [M_total,2], [P_total,5]

        t = torch.rand(B, device=self.device)
        xt      = self.flow.interpolate(x0, targets, t)         # [B, N, 5]
        v_gt    = self.flow.velocity(x0, targets)               # [B, N, 5]
        v_pred  = self.model(xt, t, image=images)               # [B, N, 5]

        dist_loss = self.dist_loss_fn(v_pred[:, :, :4], v_gt[:, :, :4], matched_indices)
        act_loss  = self.act_loss_fn(v_pred[:, :, 4],  v_gt[:, :, 4])
        x1pred    = x0 + v_pred  # [B, N, 5]
        conn_loss = self.conn_loss_fn(x1pred, junction_pairs)
        total     = (self.w_seg * dist_loss + self.w_act * act_loss
                        + self.w_conn * conn_loss)

        match_frac = matched_indices.shape[0] / (B * v_pred.shape[1])
        active_acc = ((v_gt[:, :, 4] > 0) == (v_pred[:, :, 4] > 0)).float().mean()
        pred_active_mean = float(v_pred[:, :, 4].detach().mean().item())
        gt_active_frac   = float((v_gt[:, :, 4] > 0).float().mean().item())

        # ...existing code...

        # per-component losses
        self.log("Train/seg_loss",         dist_loss,         on_step=True,  on_epoch=False, prog_bar=False, sync_dist=True)
        self.log("Train/active_loss",      act_loss,          on_step=True,  on_epoch=False, prog_bar=False, sync_dist=True)
        self.log("Train/conn_loss",        conn_loss,         on_step=True,  on_epoch=False, prog_bar=False, sync_dist=True)
        self.log("Loss/train",             total,             on_step=True,  on_epoch=True,  prog_bar=True,  sync_dist=True)

        # matching diagnostics
        self.log("Train/match_fraction",   match_frac,        on_step=True, on_epoch=False, prog_bar=False, sync_dist=True)
        self.log("Train/active_accuracy",  active_acc,        on_step=True, on_epoch=False, prog_bar=False, sync_dist=True)
        self.log("Train/pred_active_mean", pred_active_mean,  on_step=True, on_epoch=False, prog_bar=False, sync_dist=False)
        self.log("Train/gt_active_frac",   gt_active_frac,    on_step=True, on_epoch=False, prog_bar=False, sync_dist=False)

        # accumulate for CSV diagnostics
        self._step_active_pred_means.append(pred_active_mean)
        self._step_gt_active_fracs.append(gt_active_frac)

        if isinstance(total, torch.Tensor) and total.numel() > 0:
            self._step_train_losses.append(float(total.detach().item()))
            return total
        return None

    def on_train_epoch_start(self) -> None:
        opt = self.optimizers()
        lr = opt.param_groups[0]["lr"]
        self.log("Train/lr", lr, on_step=False, on_epoch=True, prog_bar=False, sync_dist=False)

    def on_train_epoch_end(self) -> None:
        """Average step losses into an epoch loss and store for loss-curve figure."""
        if self.trainer.is_global_zero:
            ep = self.current_epoch
            avg_loss          = float(np.mean(self._step_train_losses))         if self._step_train_losses         else float("nan")
            avg_active_pred   = float(np.mean(self._step_active_pred_means))    if self._step_active_pred_means    else float("nan")
            avg_gt_active_frac= float(np.mean(self._step_gt_active_fracs))      if self._step_gt_active_fracs      else float("nan")

            if self._step_train_losses:
                self._train_loss_hist.append((ep, avg_loss))

            # ── human-readable stdout summary ─────────────────────────────────
            print(
                f"[Epoch {ep:04d}]  loss={avg_loss:.4f}  "
                f"pred_active_mean={avg_active_pred:.4f}  "
                f"gt_active_frac={avg_gt_active_frac:.4f}",
                flush=True,
            )

            # ── per-epoch diagnostics CSV ──────────────────────────────────────
            log_dir  = self.cfg.paths.logs_folder
            run_name = self.cfg.get("name", "run")
            csv_path = os.path.join(log_dir, run_name, "diagnostics.csv")
            os.makedirs(os.path.dirname(csv_path), exist_ok=True)
            write_header = not os.path.exists(csv_path)
            row = {
                "epoch":            ep,
                "avg_loss":         avg_loss,
                "pred_active_mean": avg_active_pred,
                "gt_active_frac":   avg_gt_active_frac,
            }
            with open(csv_path, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(row.keys()))
                if write_header:
                    writer.writeheader()
                writer.writerow(row)

        self._step_train_losses.clear()
        self._step_active_pred_means.clear()
        self._step_gt_active_fracs.clear()

    # ──────────────────────────────────────────────────────────────────────────
    # Validation step  (teacher-forced, all batches — for checkpoint monitoring)
    # ──────────────────────────────────────────────────────────────────────────

    def validation_step(self, batch: dict, batch_idx: int) -> None:
        if batch is None or batch.get("_skip_batch", False):
            # Must still participate in every sync_dist collective so that DDP
            # all-reduces are symmetric across ranks; log zeros and skip.
            z = torch.zeros(1, device=self.device).squeeze()
            self.log("Val/seg_loss",        z, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
            self.log("Val/active_loss",     z, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
            self.log("Val/conn_loss",       z, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
            self.log("Loss/val",            z, on_step=False, on_epoch=True, prog_bar=True,  sync_dist=True)
            self.log("Val/match_fraction",  z, on_step=False, on_epoch=True,                sync_dist=True)
            self.log("Val/active_accuracy", z, on_step=False, on_epoch=True,                sync_dist=True)
            return

        z = torch.zeros(1, device=self.device).squeeze()
        try:
            images  = batch["image"].float() / 255.0
            gt_segs = batch["road_data"]
            invalid = batch["invalid_mask"]
            B = images.shape[0]

            if self.cfg.anchors.get("mode", "grid") == "gt_seeded":
                x0 = self.anchors.generate_from_gt(gt_segs, invalid)    # [B, N, 5]
            else:
                x0 = self.anchors.generate(B, self.device)
            targets, matched_indices, junction_pairs = build_targets(
                x0, gt_segs, invalid, node_eps=self.conn_loss_fn.node_eps
            )  # [B,N,5], [M_total,2], [P_total,5]

            t      = torch.full((B,), 0.5, device=self.device)
            xt     = self.flow.interpolate(x0, targets, t)
            v_gt   = self.flow.velocity(x0, targets)
            v_pred = self.model(xt, t, image=images)

            dist_loss = self.dist_loss_fn(v_pred[:, :, :4], v_gt[:, :, :4], matched_indices)
            act_loss  = self.act_loss_fn(v_pred[:, :, 4],  v_gt[:, :, 4])
            conn_loss = self.conn_loss_fn(x0 + v_pred, junction_pairs)
            total     = self.w_seg * dist_loss + self.w_act * act_loss + self.w_conn * conn_loss

            match_frac = matched_indices.shape[0] / (B * v_pred.shape[1])
            active_acc = ((v_gt[:, :, 4] > 0) == (v_pred[:, :, 4] > 0)).float().mean()
        except Exception as e:
            # print(f"[validation_step] batch_idx={batch_idx} failed: {e}", flush=True)
            seg_loss = act_loss = conn_loss = total = match_frac = active_acc = z

        self.log("Val/seg_loss",    dist_loss,  on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        self.log("Val/active_loss", act_loss,  on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        self.log("Val/conn_loss",   conn_loss, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        self.log("Loss/val",        total,     on_step=False, on_epoch=True, prog_bar=True,  sync_dist=True)
        self.log("Val/match_fraction",  match_frac, on_step=False, on_epoch=True, sync_dist=True)
        self.log("Val/active_accuracy", active_acc, on_step=False, on_epoch=True, sync_dist=True)

    def on_fit_start(self) -> None:
        """Populate locked eval indices once from the val dataset."""
        dm = self.trainer.datamodule
        if dm is None:
            return
        ds = getattr(dm, "val_dataset", None)
        if ds is None:
            return
        n_eval = int(self.cfg.training.get("n_eval_samples", 20))
        n_eval = min(n_eval, len(ds))
        self._eval_indices = list(range(n_eval))
        self._viz_indices  = list(range(min(4, n_eval)))
        # print(f"[on_fit_start] eval_indices set: {n_eval} samples", flush=True)

    def on_validation_epoch_end(self) -> None:
        # Always log images every validation epoch
        if hasattr(self, '_run_comprehensive_eval'):
            try:
                self._run_comprehensive_eval()
            except Exception as e:
                # print(f"[on_validation_epoch_end] Image logging failed: {e}", flush=True)
                pass

    # ──────────────────────────────────────────────────────────────────────────
    # Optimiser
    # ──────────────────────────────────────────────────────────────────────────

    def configure_optimizers(self):
        tc = self.cfg.training
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=float(tc.lr),
            weight_decay=float(tc.get("weight_decay", 1e-5)),
        )
        return optimizer

    # ──────────────────────────────────────────────────────────────────────────
    # Comprehensive evaluation  (Euler-integrated, runs every N epochs)
    # ──────────────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def _run_comprehensive_eval(self) -> None:
        # Ensure datamodule is available
        if not hasattr(self, 'trainer') or not hasattr(self.trainer, 'datamodule') or self.trainer.datamodule is None:
            print("[on_validation_epoch_end] Image logging failed: datamodule is not available", flush=True)
            return
        dm = self.trainer.datamodule
        dataset = getattr(dm, "val_dataset", None)
        if dataset is None:
            print("[_run_comprehensive_eval] val_dataset not available, skipping.", flush=True)
            return
        eval_indices = self._eval_indices if self._eval_indices is not None else []
        n_samples = len(eval_indices)
        viz_set = set(range(min(4, n_samples)))
        self.model.eval()

        pred_list_raw:  list[torch.Tensor] = []  # [M_i, 4] active preds per image
        gt_list:        list[torch.Tensor] = []  # [N_i, 4] GT per image
        active_counts_raw:  list[int]      = []
        gt_counts:      list[int]          = []
        viz_data:       list[dict]         = []  # first 4 samples for figures
        all_score_dist_pairs: list[tuple[float, float]] = []  # (active_score, dist_to_gt)

        # Density-bucketed samples for per-bucket metrics and grid
        # { bucket: list of dict(pred, gt, img_np, x0_b, x1pred_b) }
        density_samples: dict[str, list[dict]] = {b: [] for b in DENSITY_BUCKETS}

        # Active-channel confusion matrix accumulators (across all eval batches)
        cm_tp = cm_fp = cm_tn = cm_fn = 0
        all_active_scores: list[float] = []  # raw active scores for histogram

        bsz = 4
        for batch_start in range(0, n_samples, bsz):
            batch_positions = list(range(batch_start, min(batch_start + bsz, n_samples)))
            samples = [dataset[eval_indices[pos]] for pos in batch_positions]
            B       = len(samples)

            imgs    = torch.stack([s["image"] for s in samples]).float().to(self.device) / 255.0
            gt_b    = torch.stack([s["road_data"]    for s in samples]).to(self.device)
            inv_b   = torch.stack([s["invalid_mask"] for s in samples]).to(self.device)

            if self.cfg.anchors.get("mode", "grid") == "gt_seeded":
                x0 = self.anchors.generate_from_gt(gt_b, inv_b)
            else:
                x0 = self.anchors.generate(B, self.device)
            x1pred_raw = self.flow.euler_integrate(
                x0, self.model, imgs, steps=self.euler_steps_eval, device=self.device
            )
            x1pred = x1pred_raw.clone()

            anchor_status = torch.zeros(
                (B, x1pred.shape[1]), dtype=torch.int8, device=x1pred.device
            )

            # ── active-channel confusion matrix accumulation ──────────────────
            _, cm_idx, _ = build_targets(x0, gt_b, inv_b)
            # Convert matched index pairs [M,2] → boolean mask [B, N]
            cm_mm = torch.zeros(B, x0.shape[1], dtype=torch.bool, device=x0.device)
            if cm_idx.shape[0] > 0:
                cm_mm[cm_idx[:, 0], cm_idx[:, 1]] = True
            pred_act_flag = x1pred_raw[:, :, 4] > self.active_threshold
            cm_tp += int(( cm_mm &  pred_act_flag).sum())
            cm_fp += int((~cm_mm &  pred_act_flag).sum())
            cm_tn += int((~cm_mm & ~pred_act_flag).sum())
            cm_fn += int(( cm_mm & ~pred_act_flag).sum())
            all_active_scores.extend(x1pred_raw[:, :, 4].cpu().float().numpy().flatten().tolist())

            for b in range(B):
                active_mask_raw  = x1pred_raw[b, :, 4] > self.active_threshold
                pred_active_raw  = x1pred_raw[b, active_mask_raw, :4].cpu()
                gt_valid     = gt_b[b, ~inv_b[b]].cpu()

                pred_list_raw.append(pred_active_raw)
                gt_list.append(gt_valid)
                active_counts_raw.append(pred_active_raw.shape[0])
                gt_counts.append(gt_valid.shape[0])

                # Bucket by GT road density
                bucket = segment_density_bucket(gt_valid)
                sample_record = dict(
                    img_np   = samples[b]["image"].float().numpy().transpose(1, 2, 0) / 255.0,
                    x0_b     = x0[b].cpu(),
                    x1pred_b = x1pred[b].cpu(),
                    gt_segs  = gt_valid,
                    pred     = pred_active_raw,
                )
                density_samples[bucket].append(sample_record)

                if batch_positions[b] in viz_set:
                    viz_data.append(dict(
                        img_np    = samples[b]["image"].float().numpy().transpose(1, 2, 0) / 255.0,
                        x0_b      = x0[b].cpu(),       # [N, 5]
                        x1pred_b  = x1pred[b].cpu(),   # [N, 5]
                        gt_segs   = gt_valid,           # [K, 4]
                        status    = anchor_status[b].cpu(),  # [N] int8 status
                    ))

                # Score-distance calibration pairs (sub-sampled per image)
                if gt_valid.shape[0] > 0:
                    n_cal    = min(x1pred.shape[1], 128)
                    cal_idx  = torch.randperm(x1pred.shape[1])[:n_cal]
                    cal_scores = x1pred_raw[b, cal_idx, 4].cpu()
                    cal_coords = x1pred_raw[b, cal_idx, :4].cpu()
                    D_cal      = pi_dist_matrix(cal_coords, gt_valid)
                    min_dists  = D_cal.min(dim=1).values
                    for si in range(n_cal):
                        all_score_dist_pairs.append(
                            (float(cal_scores[si]), float(min_dists[si]))
                        )

        # ── scalar metrics ────────────────────────────────────────────────────
        def _collect_prf(preds: list[torch.Tensor], gts: list[torch.Tensor]):
            p05, r05, f05 = [], [], []
            p10, r10, f10 = [], [], []
            for pred, gt in zip(preds, gts):
                if pred.shape[0] == 0 or gt.shape[0] == 0:
                    continue
                _p, _r, _f = segment_precision_recall_f1(pred, gt, 0.05)
                p05.append(_p.item()); r05.append(_r.item()); f05.append(_f.item())
                _p, _r, _f = segment_precision_recall_f1(pred, gt, 0.10)
                p10.append(_p.item()); r10.append(_r.item()); f10.append(_f.item())
            return p05, r05, f05, p10, r10, f10

        p05_raw, r05_raw, f05_raw, p10_raw, r10_raw, f10_raw = _collect_prf(pred_list_raw, gt_list)

        def _m(v):
            return float(np.nanmean(v)) if v else float("nan")

        tb   = self.logger.experiment
        step = self.current_epoch

        # Backward-compatible aliases: keep historical Metrics/* as RAW model metrics.
        tb.add_scalar("Metrics/precision_05",     _m(p05_raw), step)
        tb.add_scalar("Metrics/recall_05",        _m(r05_raw), step)
        tb.add_scalar("Metrics/F1_05",            _m(f05_raw), step)
        tb.add_scalar("Metrics/precision_10",     _m(p10_raw), step)
        tb.add_scalar("Metrics/recall_10",        _m(r10_raw), step)
        tb.add_scalar("Metrics/F1_10",            _m(f10_raw), step)
        tb.add_scalar("Metrics/mean_active_count", _m(active_counts_raw), step)
        tb.add_scalar("Metrics/mean_gt_count",     _m(gt_counts),     step)

        # ── per-density-bucket metrics ────────────────────────────────────────
        # (Per-density-bucket metrics removed for clarity in TensorBoard)

        # ── visualisations ────────────────────────────────────────────────────
        # ── active-channel confusion matrix scalars ───────────────────────────
        _pos = cm_tp + cm_fn
        _neg = cm_tn + cm_fp
        act_recall    = cm_tp / (_pos + 1e-8)
        act_precision = cm_tp / (cm_tp + cm_fp + 1e-8)
        act_f1        = 2 * act_precision * act_recall / (act_precision + act_recall + 1e-8)
        tb.add_scalar("Active/recall",    act_recall,    step)
        tb.add_scalar("Active/precision", act_precision, step)
        tb.add_scalar("Active/F1",        act_f1,        step)
        tb.add_scalar("Active/FP_rate",   cm_fp / (_neg + 1e-8), step)
        tb.add_scalar("Active/TP",        cm_tp, step)
        tb.add_scalar("Active/FP",        cm_fp, step)
        tb.add_scalar("Active/TN",        cm_tn, step)
        tb.add_scalar("Active/FN",        cm_fn, step)

        # ── fixed-crop sample grids (Train + Val) ─────────────────────────────
        for split, ds in [("Train", getattr(dm, "train_dataset", None)),
                   ("Val",   getattr(dm, "val_dataset",   None))]:
            if ds is None:
                continue
            idx_dict = self._get_locked_grid_idx(split, ds)
            if idx_dict is None:
                continue
            try:
                grid_samps = self._collect_grid_samples(ds, idx_dict)
                fig = self._make_sample_grid(
                    grid_samps,
                    title=(f"Epoch {step}  —  {split}  "
                           f"|  sparse×2  ·  medium×2  ·  dense×2"),
                )
                if fig is not None:
                    tb.add_image(f"{split}/sample_grid",
                                 _fig_to_img_tensor(fig), step)
                    plt.close(fig)
            except Exception as _eg:
                import traceback as _tb
                print(f"[_run_comprehensive_eval] {split} grid failed: {_eg}")
                _tb.print_exc()

        self.model.train()

    # ──────────────────────────────────────────────────────────────────────────
    # Fixed-crop sample grids
    # ──────────────────────────────────────────────────────────────────────────

    def _get_locked_grid_idx(self, split: str, dataset) -> dict | None:
        """
        Return (and permanently cache) 2 sample indices per density tier.
        Scans up to 500 items to find 2 sparse / 2 medium / 2 dense samples.
        Returns None if any tier is entirely empty in the dataset.
        """
        attr = f"_{split.lower()}_grid_idx"
        if getattr(self, attr) is not None:
            return getattr(self, attr)

        tiers: dict[str, list[int]] = {"sparse": [], "medium": [], "dense": []}
        n_scan = min(500, len(dataset))
        for i in range(n_scan):
            try:
                s = dataset[i]
            except Exception:
                continue
            n_gt = int((~s["invalid_mask"]).sum())
            tier = "sparse" if n_gt < 20 else ("medium" if n_gt < 75 else "dense")
            if len(tiers[tier]) < 2:
                tiers[tier].append(i)
            if all(len(v) >= 2 for v in tiers.values()):
                break

        # Drop empty tiers; if ALL are empty, give up
        tiers = {k: v for k, v in tiers.items() if v}
        if not tiers:
            return None

        setattr(self, attr, tiers)
        return tiers

    @torch.no_grad()
    def _collect_grid_samples(self, dataset, idx_dict: dict) -> list[dict]:
        """
        Run Euler integration for each of the 6 fixed crops and return a list
        ordered: sparse×2 → medium×2 → dense×2.
        """
        samples = []
        for tier in ["sparse", "medium", "dense"]:
            for idx in idx_dict.get(tier, []):
                s = dataset[idx]
                # Strictly skip any sample with no valid segments or missing sample_id
                if s is None or s.get("sample_id", None) is None:
                    continue
                img     = s["image"].float().to(self.device).unsqueeze(0) / 255.0
                gt_segs = s["road_data"].to(self.device).unsqueeze(0)
                inv     = s["invalid_mask"].to(self.device).unsqueeze(0)
                pre_densify_count = s.get("pre_densify_count", None)
                post_densify_count = s.get("post_densify_count", None)
                sample_id = s.get("sample_id", None)

                if self.cfg.anchors.get("mode", "grid") == "gt_seeded":
                    x0 = self.anchors.generate_from_gt(gt_segs, inv)
                else:
                    x0 = self.anchors.generate(1, self.device)

                targets, matched_idx, _ = build_targets(
                    x0, gt_segs, inv, node_eps=self.conn_loss_fn.node_eps
                )
                x1pred = self.flow.euler_integrate(
                    x0, self.model, img,
                    steps=self.euler_steps_eval, device=self.device,
                )

                if matched_idx.shape[0] > 0:
                    b_mask  = matched_idx[:, 0] == 0
                    anc_idx = matched_idx[b_mask, 1].cpu()
                else:
                    anc_idx = torch.zeros(0, dtype=torch.long)

                samples.append(dict(
                    img_np      = s["image"].float().numpy().transpose(1, 2, 0) / 255.0,
                    gt_segs     = gt_segs[0, ~inv[0]].cpu(),   # [K, 4]
                    x0_b        = x0[0].cpu(),                  # [N, 5]
                    targets_b   = targets[0].cpu(),             # [N, 5]
                    x1pred_b    = x1pred[0].cpu(),              # [N, 5]
                    anc_matched = anc_idx,                      # [M] anchor indices
                    sample_idx  = idx,
                    tier        = tier,
                    pre_densify_count = pre_densify_count,
                    post_densify_count = post_densify_count,
                    sample_id = sample_id,
                ))
        return samples

    def _make_sample_grid(self, samples: list[dict], title: str) -> plt.Figure | None:
        """
        6 rows × 7 columns fixed-crop diagnostic grid.

        Rows 0–1  sparse   (< 20 GT segs)
        Rows 2–3  medium   (20–74 GT segs)
        Rows 4–5  dense    (≥ 75 GT segs)

        Col 0  Sat + GT overlay      orange segs / yellow endpoints
        Col 1  Degradation (dark)    GT (orange/yellow) + cyan arrows: GT-ep → x₀-ep
        Col 2  Anchor overlay        Sat + x₀ (sky-blue segs / hot-pink endpoints)
        Col 3  Flow →x₁  (dark)     x₀-ep → x₁pred-ep, plasma colourmap
        Col 4  All predictions       active=yellow/cyan-ep  inactive=magenta/deep-orange-ep
        Col 5  Active only           yellow segs / cyan endpoints on sat
        Col 6  Residual (dark)       x₁pred-ep → nearest GT-ep, cyan/orange/red by dist
        """
        if not samples:
            return None

        n_rows = len(samples)   # 6
        n_cols = 7
        _S     = samples[0]["img_np"].shape[0]   # image pixel size (typically 512)

        fig, axes = plt.subplots(
            n_rows, n_cols,
            figsize=(n_cols * 3.0, n_rows * 3.0),
            dpi=110,
        )
        if n_rows == 1:
            axes = axes[None, :]

        # (header, color-key subtitle or "" if none)
        col_headers = [
            ("Satellite + GT",         ""),
            ("Source Noise  GT→x₀",    "arrows: GT endpoint → x₀ endpoint"),
            ("Initial Anchors  x₀",    "matched: bright  ·  unmatched: faint"),
            ("Predicted Flow  x₀→x₁",  "active: blue  ·  inactive: red"),
            ("Active (by score)",        "plasma: low conf → high conf"),
            ("Active Predictions",      ""),
            ("Endpoint Residual",       "active pred endpoint → nearest GT endpoint"),
        ]
        for c, (hdr, sub) in enumerate(col_headers):
            label = f"{hdr}\n{sub}" if sub else hdr
            axes[0, c].set_title(label, fontsize=7, pad=4, linespacing=1.5)

        tier_labels = {"sparse": "Sparse", "medium": "Medium", "dense": "Dense"}
        _white = np.ones((_S, _S, 3), dtype=np.float32)

        def _p(v: float) -> float:
            """Normalised [-1, 1] → pixel [0, S]."""
            return (v + 1.0) / 2.0 * _S

        def _draw_segs_ep(ax, segs, seg_c, ep_c, lw=2.0, alpha=0.88, ep_s=22):
            """Segments then endpoint dots; seg colour is always ≠ ep colour."""
            if segs is None or len(segs) == 0:
                return
            for s in segs:
                x1, y1, x2, y2 = s
                ax.plot([_p(x1), _p(x2)], [_p(y1), _p(y2)],
                        color=seg_c, lw=lw, alpha=alpha, solid_capstyle="round", zorder=4)
                ax.scatter([_p(x1), _p(x2)], [_p(y1), _p(y2)],
                           color=ep_c, s=ep_s, zorder=5, linewidths=0)

        for row_i, data in enumerate(samples):
            img      = np.clip(data["img_np"], 0, 1)
            gt_segs  = data["gt_segs"]       # [K, 4]
            x0_b     = data["x0_b"]          # [N, 5]
            tgt_b    = data["targets_b"]     # [N, 5]  tgt_b[i,:4] = GT ep for matched i
            x1p      = data["x1pred_b"]      # [N, 5]
            anc_m    = data["anc_matched"]   # [M] matched anchor indices
            sid      = data["sample_idx"]
            tier     = data["tier"]

            # Debug: log stats of predicted active channel
            active_vals = x1p[:, 4].cpu().numpy()
            print(f"[QualGrid] sample {sid} active min={active_vals.min():.3f} max={active_vals.max():.3f} mean={active_vals.mean():.3f} above0={(active_vals > 0).sum()} total={active_vals.size}")

            # Crop ID label in the top-left corner of col 0, with segment counts
            pre_count = data.get("pre_densify_count", None)
            post_count = data.get("post_densify_count", None)
            sample_id = data.get("sample_id", None)
            label = f"{tier_labels[tier]}  #{sid}"
            if sample_id is not None:
                label += f"\nID: {sample_id}"
            if pre_count is not None and post_count is not None:
                label += f"\nGT segs: {pre_count} → {post_count}"
            axes[row_i, 0].text(
                4, 10,
                label,
                fontsize=6, color="black",
                bbox=dict(boxstyle="round,pad=0.2", fc="black", alpha=0.55, lw=0),
                va="top", zorder=10,
            )
            # --- Improved: single-line, small, black font, only in 'Active Predictions' (col 5) ---
            active_threshold = getattr(self, 'active_threshold', 0.7)
            n_active = int((x1p[:, 4] > active_threshold).sum().item())
            n_inactive = int((x1p[:, 4] <= active_threshold).sum().item())
            n_gt = len(gt_segs)
            # Place counts as a single line at the bottom of the title area in col 5
            counts_str = f"A: {n_active} | I: {n_inactive} | GT: {n_gt}"
            axes[row_i, 5].set_title(counts_str, fontsize=7, pad=2, color="black", loc="left")

            active_mask = x1p[:, 4] > self.active_threshold
            pred_active = x1p[active_mask]   # [Ma, 5]

            # ── initialise all axes ──────────────────────────────────────────
            for c in range(n_cols):
                _ax = axes[row_i, c]
                _ax.imshow(_white if c in (1, 3, 6) else img)
                _ax.set_xlim(0, _S); _ax.set_ylim(_S, 0)
                _ax.axis("off"); _ax.autoscale(False)

            # ── Col 0: Sat + GT overlay ─────────────────────────────────────
            _draw_segs_ep(axes[row_i, 0], gt_segs,
                          seg_c=_GT_COLOR, ep_c=_GT_EP_COLOR)

            # ── Col 1: Degradation (white bg) ───────────────────────────────
            # Show GT (for reference) then arrows from GT-ep → x₀-ep (degradation)
            _draw_segs_ep(axes[row_i, 1], gt_segs,
                          seg_c=_GT_COLOR, ep_c=_GT_EP_COLOR,
                          lw=1.6, alpha=0.80, ep_s=16)
            if anc_m.shape[0] > 0:
                src_xs, src_ys, dxs, dys = [], [], [], []
                for ep_off in (0, 2):          # p1 then p2
                    for ai in anc_m.tolist():
                        tx = _p(float(tgt_b[ai, ep_off]))
                        ty = _p(float(tgt_b[ai, ep_off + 1]))
                        anc_px = _p(float(x0_b[ai, ep_off]))
                        anc_py = _p(float(x0_b[ai, ep_off + 1]))
                        src_xs.append(tx);  src_ys.append(ty)
                        dxs.append(anc_px - tx); dys.append(anc_py - ty)
                if src_xs:
                    axes[row_i, 1].quiver(
                        src_xs, src_ys, dxs, dys,
                        color="#005EA6", alpha=0.85,
                        scale=1, scale_units="xy", angles="xy",
                        width=0.004, headwidth=5, headlength=6,
                    )

            # ── Col 2: Anchor overlay (Sat + x₀) ───────────────────────────
            matched_set = set(anc_m.tolist())
            for i in range(x0_b.shape[0]):
                x1, y1, x2, y2 = x0_b[i, :4].tolist()
                is_m = i in matched_set
                axes[row_i, 2].plot(
                    [_p(x1), _p(x2)], [_p(y1), _p(y2)],
                    color="#8833CC",
                    lw=1.5 if is_m else 0.5,
                    alpha=0.80 if is_m else 0.15,
                    solid_capstyle="round", zorder=3,
                )
            for ai in anc_m.tolist():
                x1, y1, x2, y2 = x0_b[ai, :4].tolist()
                axes[row_i, 2].scatter(
                    [_p(x1), _p(x2)], [_p(y1), _p(y2)],
                    color="#FF99CC", s=18, zorder=5, linewidths=0,
                )

            # ── Col 3: Predicted Flow x₀→x₁ (white bg) ────────────────────────
            # x₀ anchors (purple/pink) then flow arrows: active=blue, inactive=red.
            _flow_ms = set(anc_m.tolist())
            for i in range(x0_b.shape[0]):
                _fx1, _fy1, _fx2, _fy2 = x0_b[i, :4].tolist()
                _fis_m = i in _flow_ms
                axes[row_i, 3].plot(
                    [_p(_fx1), _p(_fx2)], [_p(_fy1), _p(_fy2)],
                    color="#8833CC",
                    lw=1.5 if _fis_m else 0.5,
                    alpha=0.75 if _fis_m else 0.12,
                    solid_capstyle="round", zorder=3,
                )
                # pink endpoints
                for _epx, _epy in ((_fx1, _fy1), (_fx2, _fy2)):
                    axes[row_i, 3].scatter(
                        _p(_epx), _p(_epy),
                        s=14 if _fis_m else 4,
                        c="#FF99CC",
                        alpha=0.85 if _fis_m else 0.15,
                        zorder=4, linewidths=0,
                    )
            _fscores = x1p[:, 4].numpy()
            _fact    = _fscores > self.active_threshold
            _rng3    = np.random.RandomState(0)
            _CAP3    = 150
            for ep_off in (0, 2):
                for _farrow_c, _fmask, _fal in [
                    ("#0066CC", _fact,  0.85),
                    ("#CC4400", ~_fact, 0.07),
                ]:
                    _fi = np.where(_fmask)[0]
                    if len(_fi) == 0:
                        continue
                    if len(_fi) > _CAP3:
                        _fi = _rng3.choice(_fi, _CAP3, replace=False)
                    _fsx = np.array([_p(float(x0_b[i, ep_off]))   for i in _fi])
                    _fsy = np.array([_p(float(x0_b[i, ep_off+1])) for i in _fi])
                    _fex = np.array([_p(float(x1p[i, ep_off]))    for i in _fi])
                    _fey = np.array([_p(float(x1p[i, ep_off+1]))  for i in _fi])
                    axes[row_i, 3].quiver(
                        _fsx, _fsy, _fex - _fsx, _fey - _fsy,
                        color=_farrow_c, alpha=_fal,
                        scale=1, scale_units="xy", angles="xy",
                        width=0.003, headwidth=4, headlength=5,
                    )

            # ── Col 4: Active predictions (score-colored) ──────────────────
            # Color each active segment by its confidence score (blue→yellow).
            _cmap4 = plt.cm.plasma
            for i in range(x1p.shape[0]):
                if x1p[i, 4] <= self.active_threshold:
                    continue
                _sc4 = float(np.clip((float(x1p[i, 4]) - self.active_threshold)
                                     / (1.0 - self.active_threshold), 0.0, 1.0))
                _c4  = _cmap4(_sc4)
                _x1c, _y1c, _x2c, _y2c = x1p[i, :4].tolist()
                axes[row_i, 4].plot(
                    [_p(_x1c), _p(_x2c)], [_p(_y1c), _p(_y2c)],
                    color=_c4, lw=1.5, alpha=0.85,
                    solid_capstyle="round", zorder=3,
                )
                axes[row_i, 4].scatter(
                    [_p(_x1c), _p(_x2c)], [_p(_y1c), _p(_y2c)],
                    color=_c4, s=12, zorder=5, linewidths=0,
                )

            # ── Col 5: Active only ───────────────────────────────────────────
            _draw_segs_ep(
                axes[row_i, 5],
                pred_active[:, :4].tolist() if pred_active.shape[0] > 0 else [],
                seg_c=_ACT_COLOR, ep_c=_ACT_EP_COLOR,
            )

            # ── Col 6: Endpoint Residual (white bg) ─────────────────────────
            # GT (orange) + active preds (yellow) for context, then residual
            # arrows (pred endpoint → nearest GT endpoint), capped at 150.
            _draw_segs_ep(axes[row_i, 6], gt_segs,
                          seg_c=_GT_COLOR, ep_c=_GT_EP_COLOR,
                          lw=1.2, alpha=0.55, ep_s=8)
            _draw_segs_ep(
                axes[row_i, 6],
                pred_active[:, :4].tolist() if pred_active.shape[0] > 0 else [],
                seg_c="#4DBBFF", ep_c="#FF4444",   # sky-blue segs / red eps (distinct from orange/yellow GT)
                lw=1.2, alpha=0.65, ep_s=8,
            )
            if pred_active.shape[0] > 0 and gt_segs.shape[0] > 0:
                _gt_ep6  = torch.cat([gt_segs[:, :2], gt_segs[:, 2:]], dim=0)
                _CAP6    = 150
                _rng6    = np.random.RandomState(0)
                _pa6_idx = np.arange(len(pred_active))
                if len(_pa6_idx) > _CAP6:
                    _pa6_idx = _rng6.choice(_pa6_idx, _CAP6, replace=False)
                _pa6 = pred_active[_pa6_idx]
                for ep_off in (0, 2):
                    _pe6 = _pa6[:, ep_off:ep_off + 2]
                    _ne6 = _gt_ep6[torch.cdist(_pe6, _gt_ep6).min(dim=1).indices]
                    _sx6 = [_p(float(_pe6[i, 0])) for i in range(len(_pe6))]
                    _sy6 = [_p(float(_pe6[i, 1])) for i in range(len(_pe6))]
                    _ex6 = [_p(float(_ne6[i, 0])) for i in range(len(_ne6))]
                    _ey6 = [_p(float(_ne6[i, 1])) for i in range(len(_ne6))]
                    axes[row_i, 6].quiver(
                        _sx6, _sy6,
                        [_ex6[k] - _sx6[k] for k in range(len(_sx6))],
                        [_ey6[k] - _sy6[k] for k in range(len(_sy6))],
                        color="#1A1A1A", alpha=0.85,
                        scale=1, scale_units="xy", angles="xy",
                        width=0.004, headwidth=5, headlength=6,
                    )

        fig.suptitle(title, fontsize=10, y=1.003)
        plt.tight_layout(pad=0.1, h_pad=0.15, w_pad=0.08, rect=[0, 0, 1, 0.99])
        return fig

    @staticmethod
    def _fig_to_img_tensor(fig: plt.Figure) -> torch.Tensor:
        """Convert a matplotlib Figure to a [3, H, W] uint8 tensor."""
        fig.canvas.draw()
        w, h = fig.canvas.get_width_height()
        buf = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8).reshape(h, w, 4)
        return torch.from_numpy(buf[:, :, 1:]).permute(2, 0, 1)   # [3, H, W]


# ── module-level helper (not a class method so it's importable) ────────────────

def _draw_segs(
    ax,
    segs: torch.Tensor,
    color,
    lw: float = 2.5,
    alpha: float = 0.85,
    img_size: int = 512,
) -> None:
    """Draw (x1,y1,x2,y2) segments on ax.  Coords in [-1,1]; maps to pixel space."""
    if segs is None or len(segs) == 0:
        return
    if torch.is_tensor(segs):
        segs = segs.tolist()
    for x1, y1, x2, y2 in segs:
        xs = [(x1+1)/2 * img_size, (x2+1)/2 * img_size]
        ys = [(y1+1)/2 * img_size, (y2+1)/2 * img_size]
        ax.plot(xs, ys, color=color, linewidth=lw, alpha=alpha, solid_capstyle='round')
        ax.scatter(xs, ys, color=color, s=8, alpha=alpha, zorder=5, linewidths=0)


def _fig_to_img_tensor(fig: plt.Figure) -> torch.Tensor:
    """Convenience wrapper (mirrors static method for use outside the class)."""
    return VectorFlowLightningModule._fig_to_img_tensor(fig)


# Backward-compat alias
VectorDiffusionLightningModule = VectorFlowLightningModule
