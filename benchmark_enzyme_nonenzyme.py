#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
benchmark_enzyme_nonenzyme.py

Multi-seed protein-level Enzyme vs Non-Enzyme classification benchmark for
the 3D-CNN (Set A) repo.

Mirrors the GNN-side script
    /data/data3/conglab/s441865/code/Catalytic_site_prediction_setA_gnn/benchmark_enzyme_nonenzyme.py
adapted to the 3D-CNN model: every residue gets a (box_size/voxel_size)^3
voxel grid carrying ESM-2 3B neighbour embeddings, instead of a per-residue
GNN micro-environment subgraph.

Pipeline per seed:
    1. Locate best_model_*.pt under <results_dir>/run_seed<seed>/ (latest
       by stamp, or via results_dir/summary.json["per_seed"][seed]).
    2. Iterate enzyme PDBs (default: setB_test/) and non-enzyme PDBs
       (default: step17_pdbs/), building per-residue voxel features
       PDB-by-PDB (no big up-front cache; non-enzyme set is large).
    3. Run inference -> per-residue sigmoid(prob of catalytic).
    4. Save per-residue predictions to two CSVs inside the run dir:
         bench_enzyme_<stamp>.csv
         bench_nonenzyme_<stamp>.csv
       (columns: accession, res_num, res_type, prob)
    5. Aggregate to protein-level via max(prob); compute AUROC, AUPRC,
       and a TP/FP/TN/FN/Prec/Rec/F1/Acc threshold table.
    6. Write per-seed artifacts:
         bench_enzyme_nonenzyme_metrics_<stamp>.json
         bench_enzyme_nonenzyme_threshold_table_<stamp>.txt
         bench_enzyme_nonenzyme_curves_<stamp>.png

After all seeds finish, write to <results_dir>/:
    benchmark_enzyme_nonenzyme_summary.json   # per-seed + mean/std
    benchmark_enzyme_nonenzyme_overlay.png    # multi-seed ROC + PRC overlay

Usage:
    python benchmark_enzyme_nonenzyme.py --config configs/benchmark.yaml
    python benchmark_enzyme_nonenzyme.py --config configs/benchmark.yaml \
        --run_dirs results/run_seed42 results/run_seed123 results/run_seed2026
    python benchmark_enzyme_nonenzyme.py --config configs/benchmark_explicit_ckpts.yaml
    python benchmark_enzyme_nonenzyme.py --config configs/benchmark.yaml \
        --checkpoints /abs/run_seed42/best_model_....pt /abs/run_seed123/...

    Explicit ``--checkpoints`` / YAML ``checkpoint_paths:`` overrides automatic
    selection (which otherwise picks newest ``best_model_*.pt`` per run).

    sbatch submit_benchmark_enzyme_nonenzyme.slurm
"""

import argparse
import csv
import json
import logging
import os
import re
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from Bio.PDB import PDBParser

from cnn3d import CNN3D
from data import AA_3to1_dict, get_residue_center
from utils import dict_to_namespace, load_config, namespace_to_dict


# ============================================================================
# Run discovery
# ============================================================================

_SEED_DIR_RE = re.compile(r"^run_seed(-?\d+)$")


def _find_latest_best_model(run_dir: Path) -> Path:
    """Return the most recently mtime-sorted best_model_*.pt under run_dir.

    Mirrors test.find_best_model_for_seed's fallback behaviour.
    """
    candidates = sorted(run_dir.glob("best_model_*.pt"),
                        key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        raise FileNotFoundError(
            "No best_model_*.pt under {}".format(run_dir))
    return candidates[0]


def discover_seed_runs(results_dir: Path) -> list:
    """Return [(seed:int, run_dir:Path, ckpt:Path), ...] sorted by seed.

    Skips run dirs without a best_model_*.pt (with a warning).
    """
    results_dir = Path(results_dir)
    if not results_dir.is_dir():
        raise FileNotFoundError(
            "Results dir not found: {}".format(results_dir))

    out = []
    for p in sorted(results_dir.iterdir()):
        if not p.is_dir():
            continue
        m = _SEED_DIR_RE.match(p.name)
        if m is None:
            continue
        seed = int(m.group(1))
        try:
            ckpt = _find_latest_best_model(p)
        except FileNotFoundError as e:
            print("  [WARN] {}: {}".format(p.name, e))
            continue
        out.append((seed, p, ckpt))
    out.sort(key=lambda t: t[0])
    return out


def run_dirs_to_seed_runs(run_dirs: list) -> list:
    """Convert explicit --run_dirs into the same (seed, run_dir, ckpt) tuples."""
    out = []
    for r in run_dirs:
        run_dir = Path(r)
        m = _SEED_DIR_RE.match(run_dir.name)
        seed = int(m.group(1)) if m else -1
        ckpt = _find_latest_best_model(run_dir)
        out.append((seed, run_dir, ckpt))
    out.sort(key=lambda t: (t[0], str(t[1])))
    return out


def seeds_to_seed_runs(seeds: list, results_dir: Path) -> list:
    """Resolve an explicit list of seeds against <results_dir>/run_seed<seed>/.

    Used when the YAML config provides `seeds: [42, 123, 2026]` (mirrors
    the existing test.py contract).
    """
    out = []
    for s in seeds:
        run_dir = Path(results_dir) / "run_seed{}".format(s)
        if not run_dir.is_dir():
            print("  [WARN] missing run dir: {}".format(run_dir))
            continue
        try:
            ckpt = _find_latest_best_model(run_dir)
        except FileNotFoundError as e:
            print("  [WARN] {}: {}".format(run_dir.name, e))
            continue
        out.append((int(s), run_dir, ckpt))
    out.sort(key=lambda t: t[0])
    return out


def explicit_checkpoint_paths_to_seed_runs(paths: list) -> list:
    """Build (seed, run_dir, ckpt) tuples from explicit best_model paths.

    ``seed`` is parsed from the parent directory name ``run_seed<seed>``.
    """
    out = []
    for p in paths:
        ckpt = Path(p).expanduser().resolve()
        if not ckpt.is_file():
            raise FileNotFoundError(
                "Checkpoint path does not exist: {}".format(ckpt))
        run_dir = ckpt.parent
        m = _SEED_DIR_RE.match(run_dir.name)
        seed = int(m.group(1)) if m else -1
        if seed == -1:
            print("[WARN] {}: parent '{}' is not run_seed<number> — seed=-1 in logs".format(
                ckpt.name, run_dir.name))
        out.append((seed, run_dir, ckpt))
    out.sort(key=lambda t: (t[0], str(t[1])))
    return out


# ============================================================================
# Accession listing
# ============================================================================

def parse_fasta_accessions(fasta_path) -> list:
    """Return the list of accessions from a FASTA file, in order.
    Accepts both `>sp|<acc>|<rest>` and `><acc>...` headers.
    """
    accs = []
    with open(fasta_path, "r") as f:
        for line in f:
            if not line.startswith(">"):
                continue
            h = line[1:].strip()
            if "|" in h:
                parts = h.split("|")
                acc = parts[1] if len(parts) >= 2 else parts[0]
            else:
                acc = h.split()[0]
            accs.append(acc)
    return accs


def list_accessions_for_set(set_cfg, max_proteins=None) -> list:
    """Resolve which accessions to score for one of the benchmark sets.

    If `set_cfg.fasta` is set: read accessions from FASTA, keep only those
    with both a PDB and an ESM file.
    Otherwise: list every <acc>.pdb in pdb_dir whose ESM file also exists.
    """
    pdb_dir = str(set_cfg.pdb_dir).rstrip("/") + "/"
    esm_dir = str(set_cfg.esm_dir).rstrip("/") + "/"

    fasta = getattr(set_cfg, "fasta", None)
    if fasta:
        cand = parse_fasta_accessions(fasta)
    else:
        cand = sorted(
            f[:-4] for f in os.listdir(pdb_dir) if f.endswith(".pdb")
        )

    kept = []
    for acc in cand:
        pdb_path = pdb_dir + "{}.pdb".format(acc)
        esm_path = esm_dir + "sp|{}|esm.pt".format(acc)
        if os.path.exists(pdb_path) and os.path.exists(esm_path):
            kept.append(acc)

    if max_proteins is not None:
        kept = kept[: int(max_proteins)]
    return kept


# ============================================================================
# Model loading
# ============================================================================

def load_model_from_ckpt(ckpt_path: Path, device,
                         dropout: float = 0.5) -> torch.nn.Module:
    """Load a CNN3D model from a checkpoint produced by train.py / test.py."""
    model = CNN3D(dropout=dropout).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state)
    model.eval()
    return model


# ============================================================================
# Per-PDB inference: build all residue voxels for one PDB once,
# then run every loaded model on them.
# ============================================================================

def _load_esm_embeddings(esm_file: str, esm_layer: int) -> torch.Tensor:
    """Load the (L, D) ESM embedding tensor for a setB-style file
    (sp|<acc>|esm.pt, no chain suffix). Raises on failure.
    """
    record = torch.load(esm_file, map_location="cpu")
    if "representations" in record:
        return record["representations"][esm_layer]
    first = next(iter(record.values()))
    return first["representations"][esm_layer]


@torch.no_grad()
def infer_one_pdb_for_all_models(
    pdb_file: str,
    set_cfg,
    cfg,
    models: list,
    device,
    batch_size: int,
):
    """Build per-residue voxels for one PDB and run every model on them.

    Returns:
        (res_info, [probs_per_model])
        res_info: list[(res_num:int, res_name:str)]
        probs_per_model: list[ list[float] ], one inner list per model
    Returns (None, None) if the protein is skipped (parse error, missing
    file, no usable residues, ...).
    """
    chain_id = getattr(set_cfg, "chain", "A")
    box_size = float(cfg.box_size)
    voxel_size = float(cfg.voxel_size)
    numbox = int(box_size / voxel_size)
    half_voxel = numbox // 2
    half_dist = box_size / 2.0  # neighbour cutoff in Angstroms (per axis)

    pdb_path = str(Path(set_cfg.pdb_dir) / pdb_file)
    pdb_name = pdb_file.split(".")[0]

    # ---- PDB ----------------------------------------------------------
    parser = PDBParser(QUIET=True)
    try:
        structure = parser.get_structure(pdb_name, pdb_path)
    except Exception as e:
        print("[SKIP] {}: PDB parse error -- {}".format(pdb_name, e))
        return None, None

    if chain_id not in structure[0]:
        print("[SKIP] {}: chain '{}' not found".format(pdb_name, chain_id))
        return None, None
    chain_obj = structure[0][chain_id]

    # ---- ESM ----------------------------------------------------------
    esm_file = str(Path(set_cfg.esm_dir) / "sp|{}|esm.pt".format(pdb_name))
    try:
        esm_embeddings = _load_esm_embeddings(esm_file, int(cfg.esm_layer))
    except FileNotFoundError:
        print("[SKIP] ESM file not found: {}".format(esm_file))
        return None, None
    except Exception as e:
        print("[SKIP] {}: ESM load error -- {}".format(pdb_name, e))
        return None, None

    esm_len = int(esm_embeddings.shape[0])
    embedding_dim = int(esm_embeddings.shape[1])
    # Convert once to float32 numpy for fast row indexing into the voxel grid
    esm_np = esm_embeddings.detach().to(torch.float32).cpu().numpy()

    # ---- Residue list (focal candidates: standard residues with CA) ----
    # `residue_to_order_index` maps PDB resid -> sequential order across
    # the SAME chain's standard residues. Mirrors data_test.get_pos_feature_setB.
    focal_residues = []  # [(resid:int, res_name:str, center:np.ndarray)]
    residue_to_order_index = {}
    for residue in chain_obj:
        het, resid, _icode = residue.get_id()
        if het != ' ':
            continue
        center = get_residue_center(residue)
        if center is None:
            continue
        if residue.get_resname() not in AA_3to1_dict:
            # Non-standard amino acid; original code keeps these but warns.
            # We score them too (to mirror per-residue pipeline) but only
            # if they have a CA.
            pass
        residue_to_order_index[resid] = len(focal_residues)
        focal_residues.append((int(resid), residue.get_resname(),
                               np.asarray(center, dtype=np.float32)))

    if not focal_residues:
        print("[SKIP] {}: no usable standard residues".format(pdb_name))
        return None, None

    # ---- Pre-compute all neighbour candidates ONCE ---------------------
    # We iterate every residue in the structure (any chain), keep those
    # whose resid matches a focal residue from chain_obj. This matches
    # data_test.get_pos_feature_setB.
    neighbours = []  # [(resid:int, center:np.ndarray)]
    for nb_residue in structure.get_residues():
        nb_center = get_residue_center(nb_residue)
        if nb_center is None:
            continue
        _, nb_resid, _ = nb_residue.get_id()
        if nb_resid in residue_to_order_index:
            neighbours.append((nb_resid,
                               np.asarray(nb_center, dtype=np.float32)))

    # ---- Stream voxel features for every focal residue, batched -------
    res_info = []
    probs_per_model = [[] for _ in models]
    feat_buf = []
    info_buf = []

    def _flush():
        if not feat_buf:
            return
        feats = np.stack(feat_buf, axis=0)  # [B, X, Y, Z, D]
        feats_t = torch.from_numpy(feats).to(device, dtype=torch.float32)
        for mi, m in enumerate(models):
            logits = m(feats_t).view(-1)
            p = torch.sigmoid(logits).detach().cpu().numpy().tolist()
            probs_per_model[mi].extend(p)
        res_info.extend(info_buf)
        feat_buf.clear()
        info_buf.clear()

    for resid, resname, center in focal_residues:
        feat = np.zeros((numbox, numbox, numbox, embedding_dim),
                        dtype=np.float32)

        for nb_resid, nb_center in neighbours:
            dx = float(nb_center[0] - center[0])
            dy = float(nb_center[1] - center[1])
            dz = float(nb_center[2] - center[2])
            if abs(dx) >= half_dist or abs(dy) >= half_dist or abs(dz) >= half_dist:
                continue
            order_idx = residue_to_order_index[nb_resid]
            if not (0 <= order_idx < esm_len):
                continue
            voxel_x = int(dx / voxel_size) + half_voxel
            voxel_y = int(dy / voxel_size) + half_voxel
            voxel_z = int(dz / voxel_size) + half_voxel
            if (0 <= voxel_x < numbox
                    and 0 <= voxel_y < numbox
                    and 0 <= voxel_z < numbox):
                feat[voxel_x, voxel_y, voxel_z, :] = esm_np[order_idx]

        feat_buf.append(feat)
        info_buf.append((int(resid), str(resname)))

        if len(feat_buf) >= batch_size:
            _flush()

    _flush()

    return res_info, probs_per_model


# ============================================================================
# CSV writer helpers
# ============================================================================

def open_csv_writer(path: Path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    f = open(path, "w", newline="")
    w = csv.writer(f)
    w.writerow(["accession", "res_num", "res_type", "prob"])
    return f, w


# ============================================================================
# Protein-level metrics (mirrors the GNN-side notebook + benchmark script)
# ============================================================================

def aggregate_protein_level(per_residue: dict, label_value: int) -> list:
    """{accession: [prob_per_residue,...]} -> [(acc, max_prob, label)]."""
    out = []
    for acc, probs in per_residue.items():
        if not probs:
            continue
        out.append((acc, float(np.max(probs)), int(label_value)))
    return out


def compute_curves_and_metrics(y_true: np.ndarray, y_score: np.ndarray) -> dict:
    """ROC + PRC curves, AUROC + AUPRC. NaN for either if a class is absent."""
    out = {}
    try:
        fpr, tpr, _ = roc_curve(y_true, y_score)
        out["fpr"] = fpr.tolist()
        out["tpr"] = tpr.tolist()
        out["auroc"] = float(roc_auc_score(y_true, y_score))
    except Exception as e:
        print("  [WARN] ROC failed: {}".format(e))
        out["fpr"], out["tpr"], out["auroc"] = [], [], float("nan")
    try:
        prec, rec, _ = precision_recall_curve(y_true, y_score)
        out["precision"] = prec.tolist()
        out["recall"] = rec.tolist()
        out["auprc"] = float(average_precision_score(y_true, y_score))
    except Exception as e:
        print("  [WARN] PRC failed: {}".format(e))
        out["precision"], out["recall"], out["auprc"] = [], [], float("nan")
    out["baseline"] = float(np.mean(y_true)) if len(y_true) else float("nan")
    return out


def threshold_table(y_true: np.ndarray, y_score: np.ndarray, thresholds: list) -> list:
    rows = []
    for thr in thresholds:
        preds = (y_score >= thr).astype(int)
        tp = int(((preds == 1) & (y_true == 1)).sum())
        fp = int(((preds == 1) & (y_true == 0)).sum())
        tn = int(((preds == 0) & (y_true == 0)).sum())
        fn = int(((preds == 0) & (y_true == 1)).sum())
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        acc = (tp + tn) / max(1, len(y_true))
        rows.append((float(thr), tp, fp, tn, fn, prec, rec, f1, acc))
    return rows


def format_threshold_table(rows) -> str:
    header = ("{:>6s}  {:>5s}  {:>5s}  {:>5s}  {:>5s}  "
              "{:>6s}  {:>6s}  {:>6s}  {:>6s}").format(
                  "Thresh", "TP", "FP", "TN", "FN",
                  "Prec", "Recall", "F1", "Acc")
    sep = "-" * len(header)
    lines = [header, sep]
    for thr, tp, fp, tn, fn, prec, rec, f1, acc in rows:
        lines.append(
            "{:6.2f}  {:5d}  {:5d}  {:5d}  {:5d}  "
            "{:6.3f}  {:6.3f}  {:6.3f}  {:6.3f}".format(
                thr, tp, fp, tn, fn, prec, rec, f1, acc)
        )
    return "\n".join(lines)


# ============================================================================
# Plotting
# ============================================================================

def plot_per_seed_curves(curves: dict, seed: int, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    if curves["fpr"] and curves["tpr"]:
        axes[0].plot(curves["fpr"], curves["tpr"], color="#2563EB", lw=2,
                     label="AUROC = {:.4f}".format(curves["auroc"]))
    axes[0].plot([0, 1], [0, 1], color="gray", lw=1, linestyle="--", label="Random")
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].set_title("Seed {} - ROC".format(seed))
    axes[0].set_xlim(-0.01, 1.01)
    axes[0].set_ylim(-0.01, 1.01)
    axes[0].legend(loc="lower right")
    axes[0].grid(True, alpha=0.3)

    if curves["recall"] and curves["precision"]:
        axes[1].plot(curves["recall"], curves["precision"], color="#DC2626", lw=2,
                     label="AUPRC = {:.4f}".format(curves["auprc"]))
    axes[1].axhline(y=curves["baseline"], color="gray", lw=1, linestyle="--",
                    label="Baseline = {:.3f}".format(curves["baseline"]))
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_title("Seed {} - PRC".format(seed))
    axes[1].set_xlim(-0.01, 1.01)
    axes[1].set_ylim(-0.01, 1.01)
    axes[1].legend(loc="upper right")
    axes[1].grid(True, alpha=0.3)

    fig.suptitle("Protein-level Enzyme vs Non-Enzyme - seed {}".format(seed),
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print("  Saved per-seed curves: {}".format(out_path))


def plot_overlay(per_seed_curves: list, mean_std: dict, out_path: Path) -> None:
    """Overlay every seed's ROC and PRC curves on one figure."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    cmap = plt.get_cmap("tab10")

    for i, item in enumerate(per_seed_curves):
        seed = item["seed"]
        c = item["curves"]
        color = cmap(i % 10)

        if c["fpr"] and c["tpr"]:
            axes[0].plot(c["fpr"], c["tpr"], lw=1.8, alpha=0.85, color=color,
                         label="seed={}  AUROC={:.4f}".format(seed, c["auroc"]))
        if c["recall"] and c["precision"]:
            axes[1].plot(c["recall"], c["precision"], lw=1.8, alpha=0.85, color=color,
                         label="seed={}  AUPRC={:.4f}".format(seed, c["auprc"]))

    axes[0].plot([0, 1], [0, 1], color="gray", lw=1, linestyle="--")
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].set_title("ROC overlay")
    axes[0].set_xlim(-0.01, 1.01)
    axes[0].set_ylim(-0.01, 1.01)
    axes[0].legend(loc="lower right", fontsize=9)
    axes[0].grid(True, alpha=0.3)

    if per_seed_curves:
        baseline = per_seed_curves[0]["curves"]["baseline"]
        axes[1].axhline(y=baseline, color="gray", lw=1, linestyle="--",
                        label="Baseline = {:.3f}".format(baseline))
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_title("PRC overlay")
    axes[1].set_xlim(-0.01, 1.01)
    axes[1].set_ylim(-0.01, 1.01)
    axes[1].legend(loc="upper right", fontsize=9)
    axes[1].grid(True, alpha=0.3)

    title = "Protein-level Enzyme vs Non-Enzyme - multi-seed overlay (3D-CNN)"
    parts = []
    if mean_std and "auroc" in mean_std:
        parts.append("AUROC = {:.4f} +/- {:.4f}".format(
            mean_std["auroc"]["mean"], mean_std["auroc"]["std"]))
    if mean_std and "auprc" in mean_std:
        parts.append("AUPRC = {:.4f} +/- {:.4f}".format(
            mean_std["auprc"]["mean"], mean_std["auprc"]["std"]))
    if parts:
        title += "\n" + "   |   ".join(parts)
    fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print("Saved multi-seed overlay: {}".format(out_path))


# ============================================================================
# Main pipeline
# ============================================================================

def benchmark_one_set(
    set_cfg,
    accessions: list,
    cfg,
    models: list,
    device,
    batch_size: int,
    csv_writers: list,
    csv_files: list,
    per_residue_per_model: list,
    set_label: int,
    log_prefix: str,
):
    """Iterate all PDBs in `accessions`, write per-residue probabilities into
    each model's CSV, and accumulate per-residue dicts for protein-level
    aggregation downstream.
    """
    _ = set_label  # reserved for future per-set summary logging
    pdb_dir = str(set_cfg.pdb_dir).rstrip("/") + "/"
    n_total = len(accessions)
    n_done = 0
    n_skipped = 0

    for idx, acc in enumerate(accessions):
        pdb_file = "{}.pdb".format(acc)
        if not os.path.exists(pdb_dir + pdb_file):
            n_skipped += 1
            continue

        try:
            res_info, probs_per_model = infer_one_pdb_for_all_models(
                pdb_file=pdb_file,
                set_cfg=set_cfg,
                cfg=cfg,
                models=models,
                device=device,
                batch_size=batch_size,
            )
        except Exception as e:
            print("  [SKIP] {}: inference failed -- {}".format(acc, e))
            n_skipped += 1
            continue

        if res_info is None:
            n_skipped += 1
            continue

        for mi, probs in enumerate(probs_per_model):
            for (res_num, res_name), prob in zip(res_info, probs):
                csv_writers[mi].writerow(
                    [acc, res_num, res_name, "{:.6f}".format(prob)])
            per_residue_per_model[mi].setdefault(acc, []).extend(probs)

        n_done += 1
        if (idx + 1) % 50 == 0 or (idx + 1) == n_total:
            print("  [{}] {}/{}  done={}  skipped={}".format(
                log_prefix, idx + 1, n_total, n_done, n_skipped),
                flush=True)
            for f in csv_files:
                f.flush()

    print("  [{}] FINAL: done={}  skipped={}  total={}".format(
        log_prefix, n_done, n_skipped, n_total))
    if n_total == 0:
        print("  [{}] [WARN] No accessions to score. Check set_cfg.fasta / "
              "set_cfg.pdb_dir / set_cfg.esm_dir paths and that <acc>.pdb + "
              "sp|<acc>|esm.pt both exist on disk.".format(log_prefix))
    elif n_done == 0:
        print(("  [{}] [WARN] All {} accessions were skipped. Common causes: "
               "ESM file is missing the requested esm_layer (cfg.esm_layer={}); "
               "wrong chain id (set_cfg.chain={}); PDB parse error.").format(
                  log_prefix, n_total,
                  int(cfg.esm_layer), getattr(set_cfg, "chain", "A")))


def aggregate_summary(seed_results: list) -> dict:
    """Build the JSON summary across seeds."""
    aurocs, auprcs = [], []
    per_seed = {}
    for r in seed_results:
        per_seed[str(r["seed"])] = {
            "run_dir": str(r["run_dir"]),
            "ckpt": str(r["ckpt"]),
            "stamp": r["stamp"],
            "auroc": r["curves"]["auroc"],
            "auprc": r["curves"]["auprc"],
            "n_proteins": r["n_proteins"],
            "n_enzyme": r["n_enzyme"],
            "n_non_enzyme": r["n_non_enzyme"],
            "metrics_file": r["metrics_file"],
            "threshold_table_file": r["threshold_table_file"],
            "curves_plot_file": r["curves_plot_file"],
            "csv_enzyme": r["csv_enzyme"],
            "csv_non_enzyme": r["csv_non_enzyme"],
        }
        if not np.isnan(r["curves"]["auroc"]):
            aurocs.append(r["curves"]["auroc"])
        if not np.isnan(r["curves"]["auprc"]):
            auprcs.append(r["curves"]["auprc"])

    mean_std = {}
    if aurocs:
        mean_std["auroc"] = {"mean": float(np.mean(aurocs)),
                             "std": float(np.std(aurocs))}
    if auprcs:
        mean_std["auprc"] = {"mean": float(np.mean(auprcs)),
                             "std": float(np.std(auprcs))}

    return {
        "seeds": [r["seed"] for r in seed_results],
        "per_seed": per_seed,
        "mean_std": mean_std,
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }


def run_benchmark(cfg, seed_runs: list, results_dir: Path, device) -> dict:
    print("Running enzyme/non-enzyme benchmark on {} seed(s)".format(len(seed_runs)))
    for seed, run_dir, ckpt in seed_runs:
        print("  seed={}  run_dir={}  ckpt={}".format(seed, run_dir, ckpt.name))

    bench_cfg = cfg.benchmark
    enzyme_cfg = bench_cfg.enzyme
    nonenzyme_cfg = bench_cfg.non_enzyme

    max_p = getattr(bench_cfg, "max_proteins_per_set", None)

    print("\n=== Resolving accessions ===")
    enzyme_accs = list_accessions_for_set(enzyme_cfg, max_proteins=max_p)
    nonenz_accs = list_accessions_for_set(nonenzyme_cfg, max_proteins=max_p)
    print("  Enzyme accessions:     {}".format(len(enzyme_accs)))
    print("  Non-enzyme accessions: {}".format(len(nonenz_accs)))

    # ---- Models -------------------------------------------------------
    print("\n=== Loading checkpoints ===")
    dropout = float(getattr(cfg, "dropout", 0.5))
    models = []
    for seed, run_dir, ckpt in seed_runs:
        print("  seed={}: {}".format(seed, ckpt))
        models.append(load_model_from_ckpt(ckpt, device, dropout=dropout))

    # ---- One stamp per benchmark run, applied to every seed dir -------
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # ---- Open per-seed CSV writers (one per dataset) ------------------
    enzyme_files, enzyme_writers = [], []
    nonenz_files, nonenz_writers = [], []
    enzyme_csv_paths, nonenz_csv_paths = [], []
    for seed, run_dir, _ in seed_runs:
        ep = run_dir / "bench_enzyme_{}.csv".format(stamp)
        np_ = run_dir / "bench_nonenzyme_{}.csv".format(stamp)
        f1, w1 = open_csv_writer(ep)
        f2, w2 = open_csv_writer(np_)
        enzyme_files.append(f1); enzyme_writers.append(w1); enzyme_csv_paths.append(ep)
        nonenz_files.append(f2); nonenz_writers.append(w2); nonenz_csv_paths.append(np_)

    enz_per_residue_per_model = [dict() for _ in seed_runs]
    non_per_residue_per_model = [dict() for _ in seed_runs]

    # ---- Inference: enzyme + non-enzyme ------------------------------
    batch_size = int(getattr(bench_cfg, "batch_size", 8))

    print("\n=== Inference: ENZYME set ===")
    benchmark_one_set(
        set_cfg=enzyme_cfg,
        accessions=enzyme_accs,
        cfg=cfg,
        models=models,
        device=device,
        batch_size=batch_size,
        csv_writers=enzyme_writers,
        csv_files=enzyme_files,
        per_residue_per_model=enz_per_residue_per_model,
        set_label=1,
        log_prefix="ENZ",
    )
    for f in enzyme_files:
        f.close()

    print("\n=== Inference: NON-ENZYME set ===")
    benchmark_one_set(
        set_cfg=nonenzyme_cfg,
        accessions=nonenz_accs,
        cfg=cfg,
        models=models,
        device=device,
        batch_size=batch_size,
        csv_writers=nonenz_writers,
        csv_files=nonenz_files,
        per_residue_per_model=non_per_residue_per_model,
        set_label=0,
        log_prefix="NON",
    )
    for f in nonenz_files:
        f.close()

    # ---- Per-seed protein-level metrics ------------------------------
    thresholds = list(getattr(bench_cfg, "thresholds",
                              [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4,
                               0.5, 0.6, 0.7, 0.8, 0.9, 0.95]))

    print("\n=== Computing per-seed protein-level metrics ===")
    per_seed_results = []
    for i, (seed, run_dir, ckpt) in enumerate(seed_runs):
        enz_pl = aggregate_protein_level(enz_per_residue_per_model[i], label_value=1)
        non_pl = aggregate_protein_level(non_per_residue_per_model[i], label_value=0)

        if not enz_pl and not non_pl:
            print("  [WARN] seed={}: no proteins evaluated - skipping.".format(seed))
            continue

        all_pl = enz_pl + non_pl
        accs = [t[0] for t in all_pl]
        y_score = np.array([t[1] for t in all_pl], dtype=np.float64)
        y_true = np.array([t[2] for t in all_pl], dtype=np.int64)

        curves = compute_curves_and_metrics(y_true, y_score)
        thr_rows = threshold_table(y_true, y_score, thresholds)

        # ---- Per-seed artifacts ---------------------------------------
        metrics_path = run_dir / "bench_enzyme_nonenzyme_metrics_{}.json".format(stamp)
        thr_path = run_dir / "bench_enzyme_nonenzyme_threshold_table_{}.txt".format(stamp)
        plot_path = run_dir / "bench_enzyme_nonenzyme_curves_{}.png".format(stamp)

        per_protein = [{"accession": a, "score": float(s), "label": int(l)}
                       for (a, s, l) in zip(accs, y_score, y_true)]
        with open(metrics_path, "w") as f:
            json.dump({
                "seed": seed,
                "run_dir": str(run_dir),
                "ckpt": str(ckpt),
                "stamp": stamp,
                "n_enzyme": int((y_true == 1).sum()),
                "n_non_enzyme": int((y_true == 0).sum()),
                "n_proteins": int(len(y_true)),
                "auroc": curves["auroc"],
                "auprc": curves["auprc"],
                "baseline": curves["baseline"],
                "threshold_table": [
                    {"thr": thr, "tp": tp, "fp": fp, "tn": tn, "fn": fn,
                     "precision": prec, "recall": rec, "f1": f1, "acc": acc}
                    for thr, tp, fp, tn, fn, prec, rec, f1, acc in thr_rows
                ],
                "per_protein": per_protein,
            }, f, indent=2)

        thr_text = ("=== Seed {} - Protein-level Enzyme vs Non-Enzyme (3D-CNN) ===\n"
                    "AUROC = {:.4f}\n"
                    "AUPRC = {:.4f}\n"
                    "n_enzyme = {}, n_non_enzyme = {}, baseline = {:.3f}\n\n").format(
                        seed,
                        curves["auroc"], curves["auprc"],
                        int((y_true == 1).sum()),
                        int((y_true == 0).sum()),
                        curves["baseline"])
        thr_text += format_threshold_table(thr_rows) + "\n"
        thr_path.write_text(thr_text)

        plot_per_seed_curves(curves, seed=seed, out_path=plot_path)

        print("  seed={}: AUROC={:.4f}  AUPRC={:.4f}  (enz={}, non={})".format(
            seed, curves["auroc"], curves["auprc"],
            int((y_true == 1).sum()), int((y_true == 0).sum())))

        per_seed_results.append({
            "seed": seed,
            "run_dir": str(run_dir),
            "ckpt": str(ckpt),
            "stamp": stamp,
            "curves": curves,
            "n_proteins": int(len(y_true)),
            "n_enzyme": int((y_true == 1).sum()),
            "n_non_enzyme": int((y_true == 0).sum()),
            "metrics_file": str(metrics_path),
            "threshold_table_file": str(thr_path),
            "curves_plot_file": str(plot_path),
            "csv_enzyme": str(enzyme_csv_paths[i]),
            "csv_non_enzyme": str(nonenz_csv_paths[i]),
        })

    if not per_seed_results:
        print("No seed produced any metrics - nothing to summarise.")
        return {}

    # ---- Aggregate ----------------------------------------------------
    summary = aggregate_summary(per_seed_results)
    summary_path = results_dir / "benchmark_enzyme_nonenzyme_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print("\nSummary saved to: {}".format(summary_path))

    # Snapshot the resolved config alongside the summary so we know the
    # exact ESM layer / paths / batch sizes that produced these numbers.
    cfg_snapshot = results_dir / "config_benchmark_{}.yaml".format(stamp)
    import yaml
    with cfg_snapshot.open("w") as f:
        yaml.safe_dump(namespace_to_dict(cfg), f, default_flow_style=False,
                       sort_keys=False)
    print("Config snapshot:  {}".format(cfg_snapshot))

    overlay_path = results_dir / "benchmark_enzyme_nonenzyme_overlay.png"
    plot_overlay(per_seed_results, summary["mean_std"], overlay_path)

    print("\n=== Aggregate ===")
    for k, v in summary["mean_std"].items():
        print("  {}: {:.4f} +/- {:.4f}".format(k, v["mean"], v["std"]))

    return summary


# ============================================================================
# Entry point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Multi-seed protein-level enzyme vs non-enzyme benchmark "
                    "for the Catalytic_sites_setA_3dcnn repo."
    )
    parser.add_argument("--config", type=str, required=True,
                        help="YAML config (e.g. configs/benchmark.yaml)")
    parser.add_argument("--checkpoints", type=str, nargs="*", default=None,
                        help="Optional explicit best_model_*.pt paths (same order "
                             "as models to score). Overrides run_dirs / seeds / globs.")

    parser.add_argument("--run_dirs", type=str, nargs="*", default=None,
                        help="Optional explicit list of <results_dir>/run_seed*/. "
                             "If omitted, seeds from the config are used; "
                             "if those are empty, every <results_dir>/run_seed*/ "
                             "with a best_model_*.pt is used.")
    parser.add_argument("--results_dir", type=str, default=None,
                        help="Override cfg.results_dir (rare).")
    args = parser.parse_args()

    cfg_dict = load_config(args.config)
    cfg = dict_to_namespace(cfg_dict)

    # Quiet matplotlib + Bio.PDB chatter under SLURM
    logging.getLogger("matplotlib").setLevel(logging.WARNING)

    results_dir = Path(args.results_dir or cfg.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    # Checkpoint resolution priority:
    #   CLI --checkpoints > YAML checkpoint_paths > --run_dirs > seeds > globs.
    ck_arg = getattr(args, "checkpoints", None)
    ck_yaml = getattr(cfg, "checkpoint_paths", None)
    path_list = None
    if ck_arg:
        path_list = list(ck_arg)
    elif ck_yaml:
        path_list = list(ck_yaml)

    if path_list:
        print("Using explicit checkpoints ({} file(s)):".format(len(path_list)))
        for p in path_list:
            print("  {}".format(Path(p).resolve()))
        seed_runs = explicit_checkpoint_paths_to_seed_runs(path_list)
    elif args.run_dirs:
        seed_runs = run_dirs_to_seed_runs(args.run_dirs)
    elif getattr(cfg, "seeds", None):
        seed_runs = seeds_to_seed_runs(list(cfg.seeds), results_dir)
        if not seed_runs:
            print("[INFO] cfg.seeds yielded no run dirs; "
                  "falling back to glob over {}".format(results_dir))
            seed_runs = discover_seed_runs(results_dir)
    else:
        seed_runs = discover_seed_runs(results_dir)

    if not seed_runs:
        raise SystemExit(
            "No seed runs found under {}. Train first or pass "
            "--run_dirs explicitly.".format(results_dir))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device: {}".format(device))
    print("Results dir: {}".format(Path(results_dir).resolve()))

    run_benchmark(cfg, seed_runs, results_dir, device)
    print("\nBenchmark completed.")


if __name__ == "__main__":
    main()
