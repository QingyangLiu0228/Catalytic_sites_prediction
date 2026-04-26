#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
test.py

Multi-seed test driver for Catalytic Site Prediction (Set A 3D-CNN).

For each training seed listed in the config, this script
  * resolves the seed's best-model checkpoint via results/summary.json
    (falls back to the newest best_model_*.pt in run_seed<seed>/ if needed),
  * builds the setB_test dataset (and optionally a 1:k_neg balanced subset),
  * runs evaluate.evaluate_tensor_data and writes one jsonl row per batch
    (same schema as val_log_*.jsonl produced by training),
  * aggregates dataset-level AUPRC / AUROC / F1 / acc / precision / recall
    by concatenating per-batch probs/labels (more reliable than per-batch
    AUPRC, which is the noisy signal used at training time).

After all seeds finish, this script writes results/test/test_summary.json
with per-seed metrics + mean/std across seeds, and saves PR-curve PNGs
(one per mode) overlaying the 3 seeds.

Usage:
    python test.py --config configs/test.yaml
"""

import argparse
import json
import os
import random
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import precision_recall_curve, auc
from torch.utils.data import DataLoader

from cnn3d import CNN3D
from data import make_balanced_subset_from_csv
from data_test import ReadfromCSV_SetB
from evaluate import compute_classification_metrics, evaluate_tensor_data
from utils import dict_to_namespace, load_config, namespace_to_dict


# ============================================================================
# Reproducibility (mirrors train.py)
# ============================================================================
def setup_reproducibility(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def worker_init_fn(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# ============================================================================
# I/O helpers
# ============================================================================
def write_jsonl_line(path, entry):
    with open(path, "a", encoding="utf-8") as f:
        json.dump(entry, f, ensure_ascii=False)
        f.write("\n")


def find_best_model_for_seed(seed, results_dir):
    """Return (checkpoint_path, training_run_stamp) for the given training seed.

    Prefer results_dir/summary.json (written by train.generate_summary). If
    summary is missing, fall back to the most recently modified best_model_*.pt
    under run_seed<seed>/.
    """
    summary_path = Path(results_dir) / "summary.json"
    seed_dir = Path(results_dir) / "run_seed{}".format(seed)

    if summary_path.exists():
        with summary_path.open("r", encoding="utf-8") as f:
            summary = json.load(f)
        per_seed = summary.get("per_seed", {})
        info = per_seed.get(str(seed))
        if info and info.get("best_model_basename"):
            ckpt = seed_dir / info["best_model_basename"]
            if ckpt.exists():
                return ckpt, info.get("run_stamp")

    candidates = sorted(seed_dir.glob("best_model_*.pt"),
                        key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        raise FileNotFoundError(
            "No best_model_*.pt for seed {} under {}. "
            "Either train this seed first or fix paths in config.".format(
                seed, seed_dir))
    ckpt = candidates[0]
    stem = ckpt.stem.replace("best_model_", "")
    return ckpt, stem


# ============================================================================
# Per-seed evaluation
# ============================================================================
def build_test_dataset(cfg):
    """Construct the full setB_test dataset (shared across all seeds)."""
    return ReadfromCSV_SetB(
        csv_path=cfg.test_csv_path,
        pdb_dir=cfg.test_pdb_dir,
        esm_dir=cfg.test_esm_dir,
        esm_layer=cfg.esm_layer,
        box_size=cfg.box_size,
        voxel_size=cfg.voxel_size,
        debug_max_samples=cfg.debug_max_samples,
    )


def build_loader(dataset, cfg, mode):
    """Return a DataLoader for either 'balanced' (1:k_neg) or 'full' mode."""
    if mode == "balanced":
        balanced = make_balanced_subset_from_csv(
            dataset,
            k_neg=cfg.k_neg,
            seed=cfg.balanced_seed,
            shuffle=False,
        )
        if balanced is None:
            raise RuntimeError(
                "Could not build balanced test subset (need both classes "
                "in the test CSV).")
        loader = DataLoader(
            balanced,
            batch_size=cfg.test_batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
            worker_init_fn=worker_init_fn,
        )
        return loader
    elif mode == "full":
        loader = DataLoader(
            dataset,
            batch_size=cfg.test_batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
            worker_init_fn=worker_init_fn,
        )
        return loader
    else:
        raise ValueError("Unknown test mode: {}".format(mode))


def run_one_test(model, loader, *, mode, cfg, seed, model_run_stamp,
                 run_dir, run_stamp, device):
    """Evaluate `model` on `loader`, stream jsonl rows, return dataset-level
    metrics and the concatenated probs/labels for plotting."""
    jsonl_path = run_dir / "test_log_{}_{}.jsonl".format(mode, run_stamp)
    if jsonl_path.exists():
        # Avoid silently appending to a previous test run with the same stamp.
        jsonl_path.unlink()

    all_probs = []
    all_labels = []
    all_preds = []
    all_sample_ids = []

    print("\n[seed {} | mode {}] writing per-batch log to {}".format(
        seed, mode, jsonl_path.name))

    for batch_metrics in evaluate_tensor_data(
        model,
        loader,
        epoch=0,
        device=device,
        pos_weight=None,
        threshold=cfg.threshold,
    ):
        log_entry = {
            "datetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "phase": "test",
            "mode": mode,
            "seed": int(seed),
            "model_run_stamp": model_run_stamp,
            "batch_idx": int(batch_metrics["batch_idx"]),
            "batch_size": int(batch_metrics["batch_size"]),
            "k_neg": int(cfg.k_neg) if mode == "balanced" else None,
            "threshold": float(cfg.threshold),
            "loss": float(batch_metrics["loss"]),
            "acc": float(batch_metrics["acc"]),
            "precision": float(batch_metrics["precision"]),
            "recall": float(batch_metrics["recall"]),
            "f1": float(batch_metrics["f1"]),
            "sensitivity": float(batch_metrics["sensitivity"]),
            "specificity": float(batch_metrics["specificity"]),
            "auprc": float(batch_metrics["auprc"]),
            "auroc": float(batch_metrics["auc"]),
            "tp": int(batch_metrics["tp"]),
            "tn": int(batch_metrics["tn"]),
            "fp": int(batch_metrics["fp"]),
            "fn": int(batch_metrics["fn"]),
            "probs": batch_metrics["probs"].tolist(),
            "preds": batch_metrics["preds"].tolist(),
            "labels": batch_metrics["target"].tolist(),
            "sample_ids": batch_metrics["sample_ids"],
        }
        write_jsonl_line(jsonl_path, log_entry)

        all_probs.append(batch_metrics["probs"].numpy())
        all_preds.append(batch_metrics["preds"].numpy())
        all_labels.append(batch_metrics["target"].numpy())
        all_sample_ids.extend(batch_metrics["sample_ids"])

    probs = np.concatenate(all_probs) if all_probs else np.array([])
    labels = np.concatenate(all_labels) if all_labels else np.array([])
    preds = np.concatenate(all_preds) if all_preds else np.array([])

    metrics = compute_classification_metrics(probs, preds, labels,
                                             threshold=cfg.threshold)
    metrics["n_samples"] = int(probs.size)
    metrics["n_positive"] = int(labels.sum()) if labels.size else 0
    metrics["n_negative"] = int(labels.size - labels.sum()) if labels.size else 0
    metrics["jsonl_file"] = jsonl_path.name

    print("[seed {} | mode {}] dataset-level: AUPRC={:.4f} AUROC={:.4f} "
          "F1@{:.2f}={:.4f} P={:.4f} R={:.4f} acc={:.4f}  (n={})".format(
              seed, mode, metrics["auprc"], metrics["auroc"],
              cfg.threshold, metrics["f1"], metrics["precision"],
              metrics["recall"], metrics["acc"], metrics["n_samples"]))

    return metrics, probs, labels


# ============================================================================
# PR-curve plotting
# ============================================================================
def plot_pr_curves(per_seed_curves, mode, cfg, save_path):
    """Overlay PR curves for each seed in `mode`, plus random baseline (pos rate)."""
    _ = cfg  # reserved for future plot options
    sns_palette = ["#1f77b4", "#d62728", "#2ca02c",
                   "#9467bd", "#ff7f0e", "#8c564b"]

    fig, ax = plt.subplots(figsize=(7.5, 6.5), constrained_layout=True)

    plotted = 0
    for i, (seed, (probs, labels)) in enumerate(per_seed_curves.items()):
        if probs.size == 0 or len(np.unique(labels)) < 2:
            continue
        precision, recall, _ = precision_recall_curve(labels, probs)
        seed_auprc = float(auc(recall, precision))
        ax.plot(recall, precision,
                color=sns_palette[i % len(sns_palette)],
                lw=2.0,
                label="seed {} (AUPRC={:.4f})".format(seed, seed_auprc))
        plotted += 1

    # Random / majority baseline: constant precision = positive class rate.
    pos_rate = None
    for _, (probs, labels) in per_seed_curves.items():
        if labels.size:
            pos_rate = float(labels.sum() / labels.size)
            break
    if pos_rate is not None:
        ax.axhline(pos_rate, color="grey", linestyle=":", lw=1.0,
                   label="random (P=pos rate={:.3f})".format(pos_rate))

    if plotted:
        title = "PR curves on setB_test ({} mode)".format(mode)
    else:
        title = "PR curves on setB_test ({} mode) - no valid seeds".format(mode)
    ax.set_title(title)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower left", fontsize=9)

    fig.savefig(save_path, dpi=200)
    plt.close(fig)
    print("Saved PR-curve plot to: {}".format(save_path))


# ============================================================================
# Multi-seed driver
# ============================================================================
def run_multi_seed_test(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    print("Seeds:", cfg.seeds)
    print("Modes:", cfg.modes)

    test_root = Path(cfg.results_dir) / cfg.test_save_subdir
    test_root.mkdir(parents=True, exist_ok=True)
    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Snapshot the resolved config alongside the run
    cfg_snapshot_path = test_root / "config_test_{}.yaml".format(run_stamp)
    import yaml
    with cfg_snapshot_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(namespace_to_dict(cfg), f, default_flow_style=False,
                       sort_keys=False)
    print("Test config snapshot:", cfg_snapshot_path)

    # Build the full test dataset ONCE; both 'balanced' and 'full' modes
    # reuse it (balanced just selects a Subset).
    print("\n=== Loading setB_test dataset ===")
    full_dataset = build_test_dataset(cfg)
    if len(full_dataset) == 0:
        raise RuntimeError(
            "Empty setB_test dataset after filtering. Check paths in "
            "configs/test.yaml.")

    # per_seed_curves[mode][seed] = (probs, labels)
    per_seed_curves = {m: {} for m in cfg.modes}
    per_seed_metrics = {}      # metrics_by_seed[seed][mode] = metrics_dict
    checkpoints_used = {}

    for seed in cfg.seeds:
        seed_dir = test_root / "run_seed{}".format(seed)
        seed_dir.mkdir(parents=True, exist_ok=True)

        print("\n" + "=" * 70)
        print("SEED {}".format(seed))
        print("=" * 70)

        ckpt_path, model_run_stamp = find_best_model_for_seed(
            seed, cfg.results_dir)
        print("Loading checkpoint:", ckpt_path)
        checkpoints_used[str(seed)] = ckpt_path.name

        setup_reproducibility(seed)
        model = CNN3D(dropout=0.5).to(device)
        ckpt = torch.load(ckpt_path, map_location=device)
        state = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
        model.load_state_dict(state)
        model.eval()

        per_seed_metrics[str(seed)] = {}

        for mode in cfg.modes:
            loader = build_loader(full_dataset, cfg, mode)
            metrics, probs, labels = run_one_test(
                model, loader,
                mode=mode, cfg=cfg, seed=seed,
                model_run_stamp=model_run_stamp,
                run_dir=seed_dir, run_stamp=run_stamp, device=device,
            )
            per_seed_metrics[str(seed)][mode] = metrics
            per_seed_curves[mode][seed] = (probs, labels)

        # Free memory before the next checkpoint
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ---- mean/std across seeds, per mode -----------------------------------
    mean_std = {}
    for mode in cfg.modes:
        per_mode = {}
        for key in ("auprc", "auroc", "f1", "precision", "recall", "acc"):
            vals = []
            for s in cfg.seeds:
                v = per_seed_metrics[str(s)][mode].get(key)
                if v is not None and not (isinstance(v, float) and np.isnan(v)):
                    vals.append(v)
            if vals:
                per_mode[key] = {
                    "mean": float(np.mean(vals)),
                    "std": float(np.std(vals)),
                    "n": len(vals),
                }
        mean_std[mode] = per_mode

    summary = {
        "seeds": list(cfg.seeds),
        "modes": list(cfg.modes),
        "checkpoints": checkpoints_used,
        "per_seed": per_seed_metrics,
        "mean_std": mean_std,
        "test_csv_path": cfg.test_csv_path,
        "threshold": cfg.threshold,
        "k_neg": cfg.k_neg,
        "balanced_seed": cfg.balanced_seed,
        "run_stamp": run_stamp,
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    summary_path = test_root / "test_summary_{}.json".format(run_stamp)
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    # also keep a stable filename for easy lookup of the latest run
    latest_summary_path = test_root / "test_summary.json"
    with latest_summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print("\nSummary saved to: {}\n              and {}".format(
        summary_path, latest_summary_path))

    print("\n=== Mean +/- std across seeds ===")
    for mode in cfg.modes:
        print("[{}]".format(mode))
        for k, st in mean_std[mode].items():
            print("  {:10s} {:.4f} +/- {:.4f}  (n={})".format(
                k, st["mean"], st["std"], st["n"]))

    # ---- PR plots ----------------------------------------------------------
    for mode in cfg.modes:
        plot_path = test_root / "pr_curve_{}_{}.png".format(mode, run_stamp)
        plot_pr_curves(per_seed_curves[mode], mode, cfg, plot_path)

    return summary


# ============================================================================
# Entry point
# ============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Test setA-trained 3D-CNN on setB_test "
                    "(multi-seed, balanced + full).")
    parser.add_argument("--config", type=str,
                        default="configs/test.yaml",
                        help="Path to YAML test config (defaults to "
                             "configs/test.yaml).")
    args = parser.parse_args()

    cfg_dict = load_config(args.config)
    cfg = dict_to_namespace(cfg_dict)

    print("Catalytic Sites Prediction - TEST")
    print("=" * 50)
    print("Config:", cfg_dict)
    print("=" * 50)

    run_multi_seed_test(cfg)
    print("\nTesting completed!")


if __name__ == "__main__":
    main()
