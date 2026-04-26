#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
evaluate.py

Per-batch validation generator and training-log plotting utilities.

Exports:
  - evaluate_tensor_data : yields one dict of metrics per validation batch
  - compute_classification_metrics: batch metrics from probs/preds/labels
  - plot_training_metrics: reads a val_log.jsonl and saves a PNG

Can also be run standalone to regenerate plots:
    python evaluate.py --jsonl Results/run_seed42/val_log_YYYYMMDD_HHMMSS.jsonl
"""

import argparse
import json
import math
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torch.nn import BCEWithLogitsLoss
from tqdm import tqdm

from sklearn.metrics import precision_recall_curve, auc, roc_auc_score


# ============================================================================
# Per-batch evaluation generator
# ============================================================================
def evaluate_tensor_data(model, test_loader, epoch, pos_weight=None,
                         device=None, threshold=0.9):
    """Batch-level evaluation generator.

    Yields one dict per batch containing that batch's metrics. The caller is
    responsible for logging / aggregating (or not aggregating) these.

    Note: batch-level AUPRC/AUROC are defined only for batches containing
    both classes. Single-class batches yield NaN for those two metrics.
    These per-batch numbers are NOT comparable to global AUPRC/AUROC
    reported by benchmarks like CLEAN or Squidly.
    """
    if device is not None:
        model.to(device)
    if (pos_weight is not None and isinstance(pos_weight, torch.Tensor)
            and device is not None):
        pos_weight = pos_weight.to(device)

    model.eval()
    criterion = BCEWithLogitsLoss(pos_weight=pos_weight)

    with torch.no_grad():
        pbar = tqdm(test_loader, total=len(test_loader),
                    desc="Evaluating", unit="batch")

        for batch_idx, (data_tensor, target, sample_ids) in enumerate(pbar):
            if device is not None:
                target = target.to(device)
                data_tensor = data_tensor.to(device)
            data_tensor = data_tensor.float()

            logits = model(data_tensor).view(-1)
            loss = criterion(logits, target)

            probs = torch.sigmoid(logits)
            preds = (probs > threshold).float()

            tp = (preds * target).sum().item()
            tn = ((1 - preds) * (1 - target)).sum().item()
            fp = (preds * (1 - target)).sum().item()
            fn = ((1 - preds) * target).sum().item()

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1_score = (2 * precision * recall / (precision + recall)
                        if (precision + recall) > 0 else 0.0)
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

            batch_size = target.size(0)
            batch_correct = (preds == target.float()).sum().item()
            batch_acc = batch_correct / batch_size if batch_size > 0 else 0.0
            batch_loss = loss.item()

            target_np = target.detach().cpu().numpy()
            probs_np = probs.detach().cpu().numpy()

            if len(np.unique(target_np)) > 1:
                p_curve, r_curve, _ = precision_recall_curve(target_np, probs_np)
                auprc = float(auc(r_curve, p_curve))
                auc_score = float(roc_auc_score(target_np, probs_np))
            else:
                auprc = float("nan")
                auc_score = float("nan")

            pbar.set_postfix(
                loss="{:.3f}".format(batch_loss),
                auprc=("{:.3f}".format(auprc) if not math.isnan(auprc) else "NaN"),
            )

            yield {
                "batch_idx": batch_idx,
                "batch_size": batch_size,
                "loss": batch_loss,
                "acc": batch_acc,
                "tp": tp, "fp": fp, "tn": tn, "fn": fn,
                "precision": precision,
                "recall": recall,
                "f1": f1_score,
                "sensitivity": sensitivity,
                "specificity": specificity,
                "auprc": auprc,
                "auc": auc_score,
                "probs": probs.detach().cpu(),
                "preds": preds.detach().cpu(),
                "target": target.detach().cpu(),
                "sample_ids": list(sample_ids),
            }


def compute_classification_metrics(probs, preds, labels, threshold=0.9):
    """Compute the standard classification metrics from raw arrays.

    Used by train.py for both train-batch and val-batch logging without
    re-running forward passes. probs / preds / labels are 1D torch tensors
    or numpy arrays (any framework, will be converted).
    """
    _ = threshold  # preds are already binarized; kept for call-site clarity
    # Normalize to numpy
    if isinstance(probs, torch.Tensor):
        probs = probs.detach().cpu().numpy()
    if isinstance(preds, torch.Tensor):
        preds = preds.detach().cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().numpy()

    probs = np.asarray(probs).ravel()
    preds = np.asarray(preds).ravel()
    labels = np.asarray(labels).ravel()

    tp = float(((preds == 1) & (labels == 1)).sum())
    tn = float(((preds == 0) & (labels == 0)).sum())
    fp = float(((preds == 1) & (labels == 0)).sum())
    fn = float(((preds == 0) & (labels == 1)).sum())

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) > 0 else 0.0)
    sensitivity = recall
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    n = len(labels)
    acc = (tp + tn) / n if n > 0 else 0.0

    if len(np.unique(labels)) > 1:
        p_curve, r_curve, _ = precision_recall_curve(labels, probs)
        auprc = float(auc(r_curve, p_curve))
        auroc = float(roc_auc_score(labels, probs))
    else:
        auprc = float("nan")
        auroc = float("nan")

    return {
        "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
        "acc": float(acc),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "auprc": auprc,
        "auroc": auroc,
    }


# ============================================================================
# Training log plotting
# ============================================================================
def plot_training_metrics(jsonl_path):
    """Read a val_log.jsonl and save a multi-panel PNG of validation metrics.

    Every row in the jsonl is one validation batch (the training code writes
    per-batch rows), so the x-axis of the plots is "batch count over all
    epochs". The rolling mean window is 30 to smooth the small-batch noise.
    For train-curve plots, use train_log.jsonl separately.
    """
    log_path = Path(jsonl_path)
    assert log_path.exists(), "File not found: {}".format(log_path)

    records = []
    with log_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue

            rec = {
                "datetime": obj.get("datetime"),
                "precision": obj.get("precision"),
                "recall": obj.get("recall"),
                "f1_score": obj.get("f1"),
                "sensitivity": obj.get("sensitivity"),
                "specificity": obj.get("specificity"),
                "auprc": obj.get("auprc"),
                "auc_score": obj.get("auroc"),
                "test_loss": obj.get("loss"),
                "val_acc": obj.get("acc"),
            }
            if any(v is not None for v in rec.values()):
                records.append(rec)

    if not records:
        print("No valid metric records found in the log file.")
        return

    df = pd.DataFrame(records).reset_index().rename(columns={"index": "step"})
    if "datetime" in df.columns:
        try:
            df["datetime"] = pd.to_datetime(df["datetime"])
        except Exception:
            pass

    sns.set(style="whitegrid", context="talk")
    metrics = [
        "precision", "recall", "f1_score",
        "sensitivity", "specificity", "auprc",
        "auc_score", "test_loss", "val_acc",
    ]

    ncols = 3
    nrows = math.ceil(len(metrics) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(18, 4 * nrows),
                             constrained_layout=True)
    axes = axes.ravel()

    i = 0
    for i, metric in enumerate(metrics):
        ax = axes[i]
        series = pd.to_numeric(df[metric], errors="coerce")
        ax.plot(df["step"], series, color="#1f77b4", alpha=0.4, label="raw")
        ax.plot(df["step"], series.rolling(window=30, min_periods=1).mean(),
                color="#d62728", linewidth=2.0, label="rolling mean (30)")
        ax.set_title(metric)
        ax.set_xlabel("batch step")
        if metric in {"precision", "recall", "f1_score", "sensitivity",
                      "specificity", "auprc", "auc_score", "val_acc"}:
            ax.set_ylim(0, 1)
            ax.set_ylabel("score (0-1)")
        else:
            ax.set_ylabel(metric)

    for j in range(i + 1, len(axes)):
        axes[j].axis("off")
    axes[i].legend(loc="best")

    try:
        ts = pd.to_datetime(df["datetime"].iloc[-1]).strftime("%Y%m%d-%H%M%S")
    except Exception:
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")

    out_path = log_path.parent / "metrics_over_time_{}.png".format(ts)
    fig.suptitle("Metrics over time", y=1.02, fontsize=16)
    plt.savefig(out_path, dpi=200)
    plt.close(fig)
    print("Saved figure to: {}".format(out_path))


# ============================================================================
# CLI entry point: python evaluate.py --jsonl path/to/log.jsonl
# ============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Plot validation metrics from a val_log.jsonl file."
    )
    parser.add_argument("--jsonl", type=str, required=True,
                        help="Path to a val_log_*.jsonl file.")
    args = parser.parse_args()
    plot_training_metrics(args.jsonl)


if __name__ == "__main__":
    main()
