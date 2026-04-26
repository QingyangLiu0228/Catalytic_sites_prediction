#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
train.py

Training entry point for Set A catalytic site prediction.

Supports single-GPU training and 8-GPU DistributedDataParallel on RTX8000.
Launch:
    # Single-GPU (debug):
    python train.py --config configs/debug.yaml

    # Multi-GPU DDP (production):
    torchrun --standalone --nproc_per_node=8 train.py --config configs/default.yaml

See README.md for SLURM usage.
"""

import argparse
import copy
import json
import logging
import math
import os
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler

from data import ReadfromCSV, make_balanced_subset_from_csv
from cnn3d import CNN3D
from evaluate import evaluate_tensor_data, compute_classification_metrics
from utils import load_config, dict_to_namespace, save_config_snapshot, namespace_to_dict


# ============================================================================
# Reproducibility
# ============================================================================
def setup_reproducibility(seed):
    """Set all random seeds for reproducible training."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def worker_init_fn(worker_id):
    """Worker init for reproducible DataLoader subprocesses."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# ============================================================================
# I/O helpers
# ============================================================================
def setup_logging(run_dir, run_stamp):
    """Configure a file+stream logger rooted at run_dir/training_<stamp>.log."""
    log_file = run_dir / "training_{}.log".format(run_stamp)
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(),
        ],
    )
    return logging.getLogger(__name__)


def create_run_directory(base_dir, seed):
    """Create and return Results/run_seed<seed>/ ."""
    run_dir = Path(base_dir) / "run_seed{}".format(seed)
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def write_jsonl_line(path, entry):
    """Append one JSON object as a line to `path`. Creates file if missing."""
    with open(path, "a", encoding="utf-8") as f:
        json.dump(entry, f, ensure_ascii=False)
        f.write("\n")


# ============================================================================
# DDP helpers
# ============================================================================
def is_dist_enabled():
    """True only if DDP is initialized."""
    return dist.is_available() and dist.is_initialized()


def get_rank():
    """Current process rank. Returns 0 when DDP is not initialized."""
    return dist.get_rank() if is_dist_enabled() else 0


def get_world_size():
    """Number of DDP processes. Returns 1 when DDP is not initialized."""
    return dist.get_world_size() if is_dist_enabled() else 1


def is_main_process():
    """Only rank 0 writes logs, saves checkpoints, and prints."""
    return get_rank() == 0


def main_print(*args, **kwargs):
    """print() guarded by rank check."""
    if is_main_process():
        print(*args, **kwargs)


def setup_ddp(backend="nccl"):
    """Initialize DDP from torchrun environment variables.

    Returns (local_rank, device). Raises RuntimeError if torchrun didn't set
    the expected env vars.
    """
    if "LOCAL_RANK" not in os.environ:
        raise RuntimeError(
            "setup_ddp() called but LOCAL_RANK is not set. "
            "Launch this script with `torchrun --nproc_per_node=N ...`."
        )
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend=backend)
    device = torch.device("cuda", local_rank)
    return local_rank, device


def cleanup_ddp():
    if is_dist_enabled():
        dist.destroy_process_group()


def new_run_artifact_stem():
    """Shared timestamp (rank 0 wall clock; broadcast in DDP) for this run's files."""
    if is_main_process():
        obj = [datetime.now().strftime("%Y%m%d_%H%M%S")]
    else:
        obj = [None]
    if is_dist_enabled():
        dist.broadcast_object_list(obj, src=0)
    return obj[0]


# ============================================================================
# Single-seed training loop
# ============================================================================
def run_single_seed(cfg, seed, base_save_dir, local_rank, device):
    """Run training for a single seed.

    local_rank is -1 when not in DDP mode (single-GPU).
    """
    main_print("\n" + "=" * 60)
    main_print("Starting training with seed {}".format(seed))
    main_print("=" * 60)

    setup_reproducibility(seed)

    # Output directory (rank 0 creates, all ranks use)
    run_dir = create_run_directory(base_save_dir, seed)
    # Same stem for config / training log / jsonl / checkpoint; broadcast in DDP
    run_stamp = new_run_artifact_stem()
    if is_main_process():
        setup_logging(run_dir, run_stamp)
        save_config_snapshot(namespace_to_dict(cfg), run_dir, run_stamp)
        main_print("Run artifacts (shared stem {}): {}/config_{}.yaml, training_{}.log, "
                   "train_log_{}.jsonl, val_log_{}.jsonl, best_model_{}.pt".format(
                       run_stamp, run_dir, run_stamp, run_stamp, run_stamp, run_stamp, run_stamp))

    # Make sure all ranks wait until rank 0 has created the run dir / files
    if is_dist_enabled():
        dist.barrier()

    train_jsonl_path = run_dir / "train_log_{}.jsonl".format(run_stamp)
    val_jsonl_path = run_dir / "val_log_{}.jsonl".format(run_stamp)

    # ---- Load dataset --------------------------------------------------
    main_print("\n=== Loading dataset ===")
    full_dataset = ReadfromCSV(
        csv_path=cfg.csv_path,
        pdb_dir=cfg.pdb_dir,
        esm_dir=cfg.esm_dir,
        esm_layer=cfg.esm_layer,
        box_size=cfg.box_size,
        voxel_size=cfg.voxel_size,
        debug_max_samples=cfg.debug_max_samples,
    )
    if len(full_dataset) == 0:
        raise RuntimeError(
            "No samples after PDB/ESM validation. Check pdb_dir, esm_dir, and that "
            "ESM files match the pattern 'sp|<pdb4>_<chain>|esm.pt' (same as get_pos_feature)."
        )

    # ---- Train/Val split at protein level ------------------------------
    main_print("\n=== Creating train/val split ===")
    unique_pdbs = full_dataset.df["pdb_file"].unique()
    main_print("Total proteins: {}".format(len(unique_pdbs)))

    rng = np.random.RandomState(seed)
    shuffled_pdbs = rng.permutation(unique_pdbs)
    split_idx = int(len(shuffled_pdbs) * cfg.train_ratio)
    train_pdbs = set(shuffled_pdbs[:split_idx])
    val_pdbs = set(shuffled_pdbs[split_idx:])
    main_print("Train proteins: {}, Val proteins: {}".format(
        len(train_pdbs), len(val_pdbs)))

    train_indices = full_dataset.df[
        full_dataset.df["pdb_file"].isin(train_pdbs)
    ].index.tolist()
    val_indices = full_dataset.df[
        full_dataset.df["pdb_file"].isin(val_pdbs)
    ].index.tolist()
    main_print("Train residues: {:,}, Val residues: {:,}".format(
        len(train_indices), len(val_indices)))

    # ---- Balanced 1:k_neg subsets --------------------------------------
    main_print("\n=== Creating balanced subsets ===")
    train_balanced_subset = make_balanced_subset_from_csv(
        Subset(full_dataset, train_indices),
        k_neg=cfg.k_neg, seed=seed, shuffle=True,
    )
    val_balanced_subset = make_balanced_subset_from_csv(
        Subset(full_dataset, val_indices),
        k_neg=cfg.k_neg, seed=seed + 1000, shuffle=True,
    )
    if train_balanced_subset is None or val_balanced_subset is None:
        raise RuntimeError(
            "Could not build balanced train/val subsets (need both positive and "
            "negative labels in each split). Common causes: empty dataset after "
            "PDB/ESM filtering, or wrong esm_dir / ESM filenames. "
            "Check training log warnings and paths in the config."
        )
    main_print("Train balanced: {:,}, Val balanced: {:,}".format(
        len(train_balanced_subset), len(val_balanced_subset)))

    # ---- DataLoaders ---------------------------------------------------
    if is_dist_enabled():
        # Each rank sees a disjoint shard of the training data
        train_sampler = DistributedSampler(
            train_balanced_subset,
            num_replicas=get_world_size(),
            rank=get_rank(),
            shuffle=True,
            drop_last=True,
        )
        train_loader = DataLoader(
            train_balanced_subset,
            batch_size=cfg.train_batch_size,
            sampler=train_sampler,
            shuffle=False,              # sampler handles shuffling
            num_workers=cfg.num_workers,
            drop_last=True,
            worker_init_fn=worker_init_fn,
        )
        # Only rank 0 evaluates, so it needs the full val set (no sampler)
        val_loader = DataLoader(
            val_balanced_subset,
            batch_size=cfg.val_batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
            worker_init_fn=worker_init_fn,
        )
    else:
        train_sampler = None
        train_generator = torch.Generator().manual_seed(seed)
        train_loader = DataLoader(
            train_balanced_subset,
            batch_size=cfg.train_batch_size,
            shuffle=True,
            num_workers=cfg.num_workers,
            drop_last=True,
            worker_init_fn=worker_init_fn,
            generator=train_generator,
        )
        val_loader = DataLoader(
            val_balanced_subset,
            batch_size=cfg.val_batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
            worker_init_fn=worker_init_fn,
        )

    # ---- Model / optimizer / loss --------------------------------------
    model = CNN3D(dropout=0.5).to(device)

    if is_dist_enabled():
        if getattr(cfg, "sync_bn", False):
            # Convert all BatchNormXd to SyncBatchNorm for more stable stats
            # across GPUs. Slight per-step communication overhead.
            model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = DDP(model, device_ids=[local_rank])

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    pos_weight = torch.tensor(cfg.pos_weight, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # ---- Training state ------------------------------------------------
    best_val_auprc = 0.0
    best_model_state = None
    best_epoch = 0
    best_metrics = {}

    main_print("\n=== Starting training ===")

    for epoch in range(cfg.num_epochs):
        main_print("\nEpoch {}/{}".format(epoch + 1, cfg.num_epochs))
        model.train()

        # Re-shuffle DDP sampler per epoch
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        epoch_train_loss = 0.0
        epoch_train_acc = 0.0
        for batch_idx, (feats, labels, sample_ids) in enumerate(train_loader):
            feats = feats.float().to(device, non_blocking=True)
            labels = labels.float().to(device, non_blocking=True)

            optimizer.zero_grad()
            logits = model(feats).squeeze(-1)
            loss = criterion(logits, labels)
            probs = torch.sigmoid(logits)
            preds = (probs > cfg.threshold).float()

            loss.backward()
            optimizer.step()

            batch_loss = loss.item()
            train_batch_acc = (preds == labels).float().mean().item()
            epoch_train_loss += batch_loss
            epoch_train_acc += train_batch_acc

            if is_main_process():
                probs_cpu = probs.detach().cpu()
                preds_cpu = preds.detach().cpu()
                labels_cpu = labels.detach().cpu()

                batch_metrics = compute_classification_metrics(
                    probs_cpu, preds_cpu, labels_cpu, threshold=cfg.threshold,
                )

                train_log_entry = {
                    "datetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "phase": "train",
                    "seed": seed,
                    "epoch": epoch,
                    "batch_idx": batch_idx,
                    "batch_size": int(labels.size(0)),
                    "k_neg": int(cfg.k_neg),
                    "loss": float(batch_loss),
                    "acc": float(train_batch_acc),
                    "precision": batch_metrics["precision"],
                    "recall": batch_metrics["recall"],
                    "f1": batch_metrics["f1"],
                    "sensitivity": batch_metrics["sensitivity"],
                    "specificity": batch_metrics["specificity"],
                    "auprc": batch_metrics["auprc"],
                    "auroc": batch_metrics["auroc"],
                    "tp": batch_metrics["tp"],
                    "tn": batch_metrics["tn"],
                    "fp": batch_metrics["fp"],
                    "fn": batch_metrics["fn"],
                    "probs": probs_cpu.tolist(),
                    "preds": preds_cpu.tolist(),
                    "labels": labels_cpu.tolist(),
                    "sample_ids": list(sample_ids),
                }
                write_jsonl_line(train_jsonl_path, train_log_entry)

            if (batch_idx + 1) % 10 == 0 and is_main_process():
                print("  [Train] Epoch {}, Batch {}/{}: loss={:.4f}, acc={:.4f}".format(
                    epoch + 1, batch_idx + 1, len(train_loader),
                    batch_loss, train_batch_acc))

        n_batches = max(1, len(train_loader))
        epoch_train_loss /= n_batches
        epoch_train_acc /= n_batches
        main_print("  [Train] Epoch {}: avg_loss={:.4f}, avg_acc={:.4f}".format(
            epoch + 1, epoch_train_loss, epoch_train_acc))

        # ---- Validation: only on rank 0 --------------------------------
        # Other ranks wait at the barrier at the bottom of the epoch.
        if is_main_process():
            print("  [Validation] Evaluating full validation set...")

            # evaluate_tensor_data expects a plain model, not DDP-wrapped
            eval_model = model.module if is_dist_enabled() else model

            # Accumulate per-batch outputs to compute ONE epoch-level AUPRC/AUROC
            # over the entire concatenated val set. This is the standard
            # "epoch-level" metric: per-batch AUPRC depends on each batch's
            # class composition, so picking the best batch across an epoch is
            # noisy and biased toward batches with few positives.
            epoch_probs_chunks = []
            epoch_preds_chunks = []
            epoch_labels_chunks = []
            epoch_loss_sum = 0.0
            epoch_n_samples = 0

            for batch_metrics in evaluate_tensor_data(
                eval_model,
                val_loader,
                epoch=epoch,
                device=device,
                pos_weight=pos_weight,
                threshold=cfg.threshold,
            ):
                val_log_entry = {
                    "datetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "phase": "val",
                    "seed": seed,
                    "epoch": epoch,
                    "batch_idx": batch_metrics["batch_idx"],
                    "batch_size": batch_metrics["batch_size"],
                    "k_neg": int(cfg.k_neg),
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
                write_jsonl_line(val_jsonl_path, val_log_entry)

                epoch_probs_chunks.append(batch_metrics["probs"])
                epoch_preds_chunks.append(batch_metrics["preds"])
                epoch_labels_chunks.append(batch_metrics["target"])
                bs = int(batch_metrics["batch_size"])
                epoch_loss_sum += float(batch_metrics["loss"]) * bs
                epoch_n_samples += bs

            if epoch_n_samples == 0:
                print("  [Validation] empty val_loader; skipping checkpoint.")
            else:
                all_probs = torch.cat(epoch_probs_chunks).detach().cpu()
                all_preds = torch.cat(epoch_preds_chunks).detach().cpu()
                all_labels = torch.cat(epoch_labels_chunks).detach().cpu()

                epoch_metrics = compute_classification_metrics(
                    all_probs, all_preds, all_labels, threshold=cfg.threshold,
                )
                epoch_loss = epoch_loss_sum / epoch_n_samples
                epoch_val_auprc = float(epoch_metrics["auprc"])

                # Per-epoch summary line in val_log.jsonl (phase="val_epoch")
                # so downstream tools can find epoch-level numbers without
                # re-aggregating per-batch rows.
                val_epoch_entry = {
                    "datetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "phase": "val_epoch",
                    "seed": seed,
                    "epoch": epoch,
                    "n_samples": epoch_n_samples,
                    "loss": float(epoch_loss),
                    "acc": float(epoch_metrics["acc"]),
                    "precision": float(epoch_metrics["precision"]),
                    "recall": float(epoch_metrics["recall"]),
                    "f1": float(epoch_metrics["f1"]),
                    "sensitivity": float(epoch_metrics["sensitivity"]),
                    "specificity": float(epoch_metrics["specificity"]),
                    "auprc": epoch_val_auprc,
                    "auroc": float(epoch_metrics["auroc"]),
                    "tp": int(epoch_metrics["tp"]),
                    "tn": int(epoch_metrics["tn"]),
                    "fp": int(epoch_metrics["fp"]),
                    "fn": int(epoch_metrics["fn"]),
                }
                write_jsonl_line(val_jsonl_path, val_epoch_entry)

                print(
                    "  [Validation] epoch {} (0-based): "
                    "epoch_val_auprc={:.4f} (n={}), "
                    "auroc={:.4f}, f1={:.4f}".format(
                        epoch, epoch_val_auprc, epoch_n_samples,
                        float(epoch_metrics["auroc"]),
                        float(epoch_metrics["f1"]),
                    )
                )

                # Checkpoint on epoch-level AUPRC over the whole val set.
                if (not math.isnan(epoch_val_auprc)
                        and epoch_val_auprc > best_val_auprc):
                    best_val_auprc = epoch_val_auprc
                    state_to_save = (model.module.state_dict()
                                     if is_dist_enabled() else model.state_dict())
                    best_model_state = copy.deepcopy(state_to_save)
                    best_epoch = epoch
                    best_metrics = {
                        "epoch": epoch,
                        "val_auprc": epoch_val_auprc,
                        "val_auc": float(epoch_metrics["auroc"]),
                        "val_f1": float(epoch_metrics["f1"]),
                        "val_acc": float(epoch_metrics["acc"]) * 100.0,
                        "val_precision": float(epoch_metrics["precision"]),
                        "val_recall": float(epoch_metrics["recall"]),
                        "val_loss": float(epoch_loss),
                        "val_n_samples": int(epoch_n_samples),
                    }
                    best_model_path = run_dir / "best_model_{}.pt".format(run_stamp)
                    save_dict = {
                        "model_state_dict": best_model_state,
                        "seed": seed,
                        "run_stamp": run_stamp,
                    }
                    save_dict.update(best_metrics)
                    torch.save(save_dict, best_model_path)
                    print("  New best model! epoch (0-based)={}, "
                          "epoch_val_auprc={:.4f}".format(
                              epoch, epoch_val_auprc))

        # All ranks wait here before starting next epoch
        if is_dist_enabled():
            dist.barrier()

    main_print("\nSeed {} training complete".format(seed))
    main_print("Best epoch-level val AUPRC: {:.4f} at epoch {} (0-based)".format(
        best_val_auprc, best_epoch))
    out = dict(best_metrics) if best_metrics else {}
    out["run_stamp"] = run_stamp
    out["config_file"] = "config_{}.yaml".format(run_stamp)
    out["train_log_file"] = "train_log_{}.jsonl".format(run_stamp)
    out["val_log_file"] = "val_log_{}.jsonl".format(run_stamp)
    out["best_model_basename"] = "best_model_{}.pt".format(run_stamp)
    return out


# ============================================================================
# Multi-seed driver
# ============================================================================
def run_multi_seed_training(cfg, local_rank, device):
    main_print("Starting multi-seed training...")
    main_print("Seeds: {}".format(cfg.seeds))
    main_print("Save directory: {}".format(cfg.save_dir))
    main_print("World size (GPUs for this seed): {}".format(get_world_size()))

    base_save_dir = Path(cfg.save_dir)
    if is_main_process():
        base_save_dir.mkdir(parents=True, exist_ok=True)
    if is_dist_enabled():
        dist.barrier()

    all_results = {}
    for seed in cfg.seeds:
        main_print("\n" + "=" * 80)
        main_print("RUNNING SEED {}".format(seed))
        main_print("=" * 80)
        try:
            best_metrics = run_single_seed(cfg, seed, base_save_dir,
                                           local_rank, device)
            all_results[seed] = best_metrics
            main_print("Seed {} completed successfully".format(seed))
        except Exception as e:
            main_print("Seed {} failed: {}".format(seed, e))
            if is_main_process():
                logging.error("Seed {} failed: {}".format(seed, e))
            continue

    if is_main_process():
        if all_results:
            generate_summary(all_results, cfg.seeds, base_save_dir)
        else:
            print("No successful runs to summarize")

    main_print("\nMulti-seed training complete!")
    return all_results


def generate_summary(all_results, seeds, save_dir):
    """Write summary.json with per-seed best metrics + mean/std across seeds."""
    print("\n=== Generating summary ===")

    metrics_by_seed = {}
    all_auprcs, all_aucs, all_f1s = [], [], []

    for seed in seeds:
        if seed in all_results:
            result = all_results[seed]
            metrics_by_seed[str(seed)] = {
                "run_stamp": result.get("run_stamp"),
                "config_file": result.get("config_file"),
                "train_log_file": result.get("train_log_file"),
                "val_log_file": result.get("val_log_file"),
                "best_model_basename": result.get("best_model_basename"),
                "best_val_auprc": result.get("val_auprc"),
                "best_val_auc": result.get("val_auc"),
                "best_val_f1": result.get("val_f1"),
                "best_val_loss": result.get("val_loss"),
                "best_val_n_samples": result.get("val_n_samples"),
                "best_epoch": result.get("epoch"),
            }
            if "val_auprc" in result and result.get("val_auprc") is not None:
                all_auprcs.append(result["val_auprc"])
                all_aucs.append(result["val_auc"])
                all_f1s.append(result["val_f1"])

    mean_std = {}
    if all_auprcs:
        mean_std = {
            "val_auprc": {"mean": float(np.mean(all_auprcs)),
                          "std": float(np.std(all_auprcs))},
            "val_auc": {"mean": float(np.mean(all_aucs)),
                        "std": float(np.std(all_aucs))},
            "val_f1": {"mean": float(np.mean(all_f1s)),
                       "std": float(np.std(all_f1s))},
        }

    summary = {
        "seeds": list(seeds),
        "per_seed": metrics_by_seed,
        "mean_std": mean_std,
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    summary_path = Path(save_dir) / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("Summary saved to: {}".format(summary_path))
    print("\nResults summary:")
    for metric, stats in mean_std.items():
        print("  {}: {:.4f} +/- {:.4f}".format(metric, stats["mean"], stats["std"]))
    return summary


# ============================================================================
# Entry point
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description="Train catalytic site predictor")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to YAML config (e.g. configs/default.yaml)")
    args = parser.parse_args()

    cfg_dict = load_config(args.config)
    cfg = dict_to_namespace(cfg_dict)

    # ---- Choose execution mode ----
    # DDP when requested AND torchrun set LOCAL_RANK.
    use_ddp = bool(getattr(cfg, "distributed", False)) and "LOCAL_RANK" in os.environ

    if use_ddp:
        local_rank, device = setup_ddp(backend=getattr(cfg, "backend", "nccl"))
    else:
        local_rank = -1
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if getattr(cfg, "distributed", False) and "LOCAL_RANK" not in os.environ:
            print("[train] distributed=true in config but LOCAL_RANK not set. "
                  "Falling back to single-GPU. Launch with `torchrun` to enable DDP.")

    main_print("Catalytic Sites Prediction Training")
    main_print("====================================")
    if is_main_process():
        print("Configuration: {}".format(cfg_dict))

    try:
        run_multi_seed_training(cfg, local_rank, device)
    finally:
        cleanup_ddp()

    main_print("\nTraining completed!")


if __name__ == "__main__":
    main()
