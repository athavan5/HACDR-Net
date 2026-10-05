"""
test_all_epochs_aupr.py  —  HACDR-Net edition
==============================================
Iterates over every checkpoint saved during HACDR-Net training,
runs inference on the test split, and writes a CSV of per-class
AUPR (EX, MA, SE, HE) plus mean AUPR for every epoch/iteration.

Usage
-----
python test_all_epochs_aupr.py \
    --config   configs/HACDR_1440x960_idrid.py \
    --weights-dir  ./save_dir/HACDRNet_idrid/ \
    --output-csv   ./results/HACDRNet_idrid/all_iters_aupr.csv \
    [--gpu-id 0] \
    [--data-root ./data/idrid_1440x960/]

The script expects checkpoints to follow the mmseg default naming
convention produced by IterBasedRunner with checkpoint_config:
    iter_1000.pth, iter_2000.pth, ..., latest.pth  (latest is skipped)

If your run used EpochBasedRunner the filenames will be
    epoch_1.pth, epoch_2.pth, ...
Both patterns are detected automatically.

Class index mapping (from HACDR_1440x960_idrid.py, num_classes=5):
    0 = Background   (excluded from AUPR summary)
    1 = EX  Hard Exudates
    2 = MA  Microaneurysms
    3 = SE  Soft Exudates
    4 = HE  Hemorrhages
"""

import argparse
import csv
import os
import re
import warnings

import mmcv
import numpy as np
import torch
from mmcv.cnn.utils import revert_sync_batchnorm
from mmcv.runner import load_checkpoint
from sklearn.metrics import average_precision_score

from mmseg.datasets import build_dataloader, build_dataset
from mmseg.models import build_segmentor
from mmseg.utils import build_dp, get_device


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate all HACDR-Net checkpoints and save per-class AUPR CSV."
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the mmseg model config file (e.g. configs/HACDR_1440x960_idrid.py).",
    )
    parser.add_argument(
        "--weights-dir",
        dest="weights_dir",
        type=str,
        default=None,
        help=(
            "Directory containing .pth checkpoints. "
            "Defaults to work_dir defined inside the config file."
        ),
    )
    parser.add_argument(
        "--output-csv",
        dest="output_csv",
        type=str,
        default=None,
        help="Output CSV path. Defaults to <weights_dir>/all_iters_aupr.csv",
    )
    parser.add_argument(
        "--gpu-id",
        dest="gpu_id",
        type=int,
        default=0,
        help="GPU id to use for inference (default: 0). Use -1 for CPU.",
    )
    parser.add_argument(
        "--data-root",
        dest="data_root",
        type=str,
        default=None,
        help=(
            "Optional override for cfg.data.test.data_root. "
            "Useful if data lives in a different location than the config specifies."
        ),
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Checkpoint discovery
# ---------------------------------------------------------------------------

# Matches:  iter_1000.pth   epoch_5.pth
_CKPT_RE = re.compile(r"^(iter|epoch)_(\d+)\.pth$")


def list_checkpoints(weights_dir):
    """Return [(step_int, path), ...] sorted by step, skipping 'latest.pth'."""
    if not os.path.isdir(weights_dir):
        raise FileNotFoundError(f"Weights directory not found: {weights_dir}")

    ckpts = []
    for name in os.listdir(weights_dir):
        m = _CKPT_RE.match(name)
        if m is None:
            continue
        step = int(m.group(2))
        ckpts.append((step, os.path.join(weights_dir, name)))

    if not ckpts:
        raise FileNotFoundError(
            f"No iter_N.pth / epoch_N.pth checkpoints found in: {weights_dir}"
        )

    ckpts.sort(key=lambda x: x[0])
    return ckpts


# ---------------------------------------------------------------------------
# AUPR computation  (mirrors compute_aupr from your metrics.py)
# ---------------------------------------------------------------------------

def compute_aupr(prob_maps, gt_seg_maps, num_classes, ignore_index=255):
    """
    Compute per-class AUPR from soft probability maps.

    Parameters
    ----------
    prob_maps : list[np.ndarray]
        Per-image softmax arrays, each shape (num_classes, H, W).
    gt_seg_maps : list[np.ndarray]
        Per-image ground-truth integer masks, each shape (H, W).
    num_classes : int
    ignore_index : int
        Pixels with this label value are excluded from scoring.

    Returns
    -------
    np.ndarray  shape (num_classes,)  — per-class AUPR; NaN where no positives exist.
    """
    all_probs  = [[] for _ in range(num_classes)]
    all_labels = [[] for _ in range(num_classes)]

    for prob_map, gt in zip(prob_maps, gt_seg_maps):
        valid_mask = gt != ignore_index
        gt_flat    = gt[valid_mask]
        for c in range(num_classes):
            prob_flat      = prob_map[c][valid_mask]
            binary_label   = (gt_flat == c).astype(np.float32)
            all_probs[c].append(prob_flat)
            all_labels[c].append(binary_label)

    aupr = np.full(num_classes, np.nan)
    for c in range(num_classes):
        y_true  = np.concatenate(all_labels[c])
        y_score = np.concatenate(all_probs[c])
        if y_true.sum() > 0:
            aupr[c] = average_precision_score(y_true, y_score)
    return aupr


# ---------------------------------------------------------------------------
# Single-checkpoint evaluation
# ---------------------------------------------------------------------------

def evaluate_checkpoint(ckpt_path, model, data_loader, dataset, device):
    """
    Load weights into *model*, run inference over *data_loader*, return
    (EX, MA, SE, HE, mean_AUPR) as floats.

    The model is mutated in-place (weights replaced) but its architecture
    is never rebuilt, so successive calls are efficient.
    """
    # Load checkpoint weights (silences the 'keys missing' spam)
    load_checkpoint(model, ckpt_path, map_location="cpu", logger=mmcv.get_logger("mmseg"))
    model.eval()

    # Reset AUPR accumulators on the dataset object
    if hasattr(dataset, "_aupr_probs"):
        del dataset._aupr_probs, dataset._aupr_gts

    prob_maps   = []
    gt_seg_maps = []

    loader_indices = data_loader.batch_sampler
    for batch_indices, data in zip(loader_indices, data_loader):
        # Argmax segmentation prediction (needed by dataset.pre_eval)
        with torch.no_grad():
            result = model(return_loss=False, **data)

        # Soft probability map for AUPR
        img       = data["img"][0].to(device)
        img_metas = data["img_metas"][0].data[0]
        with torch.no_grad():
            prob = model.module.inference(img, img_metas, rescale=True)
            prob = prob.cpu().numpy()          # (1, num_classes, H, W)

        for p in prob:
            prob_maps.append(p)                # (num_classes, H, W)

        # Collect ground-truth for these indices
        for idx in (batch_indices if isinstance(batch_indices, list) else [batch_indices]):
            gt_seg_maps.append(dataset.get_gt_seg_map_by_idx(idx))

    num_classes = len(dataset.CLASSES)
    aupr = compute_aupr(prob_maps, gt_seg_maps, num_classes)

    # Class layout: 0=BG, 1=EX, 2=MA, 3=SE, 4=HE
    ex = float(aupr[1])
    ma = float(aupr[2])
    se = float(aupr[3])
    he = float(aupr[4])
    mean_aupr = float(np.nanmean([ex, ma, se, he]))
    return ex, ma, se, he, mean_aupr


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # ------------------------------------------------------------------
    # Load config
    # ------------------------------------------------------------------
    cfg = mmcv.Config.fromfile(args.config)

    # Force test mode settings (mirrors test_config.py)
    cfg.model.pretrained = None
    cfg.data.test.test_mode = True
    if args.data_root is not None:
        cfg.data.test.data_root = args.data_root

    # ------------------------------------------------------------------
    # Resolve paths
    # ------------------------------------------------------------------
    weights_dir = args.weights_dir or cfg.get("work_dir", "./work_dirs")
    output_csv  = args.output_csv  or os.path.join(weights_dir, "all_iters_aupr.csv")
    os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)

    # ------------------------------------------------------------------
    # Device
    # ------------------------------------------------------------------
    if args.gpu_id >= 0 and torch.cuda.is_available():
        cfg.gpu_ids = [args.gpu_id]
        device_str  = f"cuda:{args.gpu_id}"
    else:
        cfg.gpu_ids = []
        device_str  = "cpu"
    cfg.device = get_device()
    device     = torch.device(device_str)

    # ------------------------------------------------------------------
    # Build dataset & dataloader  (built once, reused for every checkpoint)
    # ------------------------------------------------------------------
    print(f"[INFO] Building test dataset from config: {cfg.data.test.data_root}")
    dataset = build_dataset(cfg.data.test)

    loader_cfg = dict(
        num_gpus  = max(1, len(cfg.gpu_ids)),
        dist      = False,
        shuffle   = False,
    )
    loader_cfg.update({
        k: v for k, v in cfg.data.items()
        if k not in ["train", "val", "test",
                     "train_dataloader", "val_dataloader", "test_dataloader"]
    })
    test_loader_cfg = {**loader_cfg, "samples_per_gpu": 1, "shuffle": False,
                       **cfg.data.get("test_dataloader", {})}
    data_loader = build_dataloader(dataset, **test_loader_cfg)
    print(f"[INFO] Test set size: {len(dataset)} images")

    # ------------------------------------------------------------------
    # Build model  (architecture built once; weights swapped per checkpoint)
    # ------------------------------------------------------------------
    cfg.model.train_cfg = None
    model = build_segmentor(cfg.model, test_cfg=cfg.get("test_cfg"))

    warnings.warn(
        "SyncBN is only supported with DDP. Converting to BN for single-GPU eval.",
        stacklevel=1,
    )
    model = revert_sync_batchnorm(model)
    model = build_dp(model, cfg.device, device_ids=cfg.gpu_ids if cfg.gpu_ids else [0])
    model.eval()

    # ------------------------------------------------------------------
    # Discover checkpoints
    # ------------------------------------------------------------------
    checkpoints = list_checkpoints(weights_dir)
    print(f"[INFO] Found {len(checkpoints)} checkpoints in: {weights_dir}")

    # ------------------------------------------------------------------
    # Evaluate each checkpoint
    # ------------------------------------------------------------------
    rows = []
    for step, ckpt_path in checkpoints:
        print(f"[INFO] Evaluating step {step}: {ckpt_path}")
        ex, ma, se, he, mean_aupr = evaluate_checkpoint(
            ckpt_path, model, data_loader, dataset, device
        )
        rows.append({
            "iter":      step,
            "EX":        round(ex,        6),
            "MA":        round(ma,        6),
            "SE":        round(se,        6),
            "HE":        round(he,        6),
            "Mean_AUPR": round(mean_aupr, 6),
        })
        print(
            f"[INFO] AUPR — "
            f"EX: {ex:.6f}, MA: {ma:.6f}, "
            f"SE: {se:.6f}, HE: {he:.6f}, Mean: {mean_aupr:.6f}"
        )

    # ------------------------------------------------------------------
    # Write CSV
    # ------------------------------------------------------------------
    fieldnames = ["iter", "EX", "MA", "SE", "HE", "Mean_AUPR"]
    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"[INFO] Saved per-iteration AUPR CSV to: {output_csv}")


if __name__ == "__main__":
    main()
