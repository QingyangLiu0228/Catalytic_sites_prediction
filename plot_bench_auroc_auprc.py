#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot ROC and precision–recall curves (protein-level max residue prob) for
multiple enzyme/non-enzyme benchmark CSV pairs on one figure (curves overlaid).

Example:
  python plot_bench_auroc_auprc.py \\
    -o results/bench_roc_prc_3seeds.png \\
    --pair 42 results/run_seed42/bench_enzyme_20260429_191934_delete.csv \\
           results/run_seed42/bench_nonenzyme_20260429_191934_delete.csv \\
    --pair 123 results/run_seed123/bench_enzyme_20260429_191934_delete.csv \\
           results/run_seed123/bench_nonenzyme_20260429_191934_delete.csv \\
    --pair 2026 results/run_seed2026/bench_enzyme_20260429_191934_delete.csv \\
           results/run_seed2026/bench_nonenzyme_20260429_191934_delete.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from typing import List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)


def max_prob_per_acc(path: str) -> dict:
    m: dict = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            acc = row["accession"]
            p = float(row["prob"])
            o = m.get(acc)
            if o is None or p > o:
                m[acc] = p
    return m


def protein_level_arrays(
    enz_path: str, non_path: str
) -> Tuple[np.ndarray, np.ndarray, int, int]:
    enz = max_prob_per_acc(enz_path)
    non = max_prob_per_acc(non_path)
    y_true = np.array([1] * len(enz) + [0] * len(non), dtype=np.int8)
    y_score = np.array(list(enz.values()) + list(non.values()), dtype=np.float64)
    return y_true, y_score, len(enz), len(non)


def plot_roc_prc_overlay(
    pairs: List[Tuple[str, str, str]],
    title: str,
    save_path: str | None,
) -> None:
    colors = ["#2563EB", "#059669", "#D97706", "#7C3AED", "#DC2626"]
    fig, (ax_roc, ax_pr) = plt.subplots(
        1,
        2,
        figsize=(12, 5.2),
        dpi=150,
        gridspec_kw={"width_ratios": [1, 1], "wspace": 0.28},
    )

    y0, _, _, _ = protein_level_arrays(pairs[0][1], pairs[0][2])
    prevalence = float(np.mean(y0))

    aurocs_list: List[float] = []
    auprcs_list: List[float] = []
    for i, (seed_label, enz_p, non_p) in enumerate(pairs):
        y_true, y_score, _, _ = protein_level_arrays(enz_p, non_p)
        auroc = roc_auc_score(y_true, y_score)
        auprc = average_precision_score(y_true, y_score)
        aurocs_list.append(float(auroc))
        auprcs_list.append(float(auprc))
        fpr, tpr, _ = roc_curve(y_true, y_score)
        prec, rec, _ = precision_recall_curve(y_true, y_score)
        c = colors[i % len(colors)]
        ax_roc.plot(
            fpr,
            tpr,
            color=c,
            lw=2,
            label="seed {} (AUROC = {:.4f})".format(seed_label, auroc),
        )
        ax_pr.plot(
            rec,
            prec,
            color=c,
            lw=2,
            label="seed {} (AUPRC = {:.4f})".format(seed_label, auprc),
        )
        print(
            "seed={}  n_pos={}  n_neg={}  AUROC={:.6f}  AUPRC={:.6f}".format(
                seed_label, int(y_true.sum()), int(len(y_true) - y_true.sum()),
                auroc, auprc,
            ),
            file=sys.stderr,
        )

    ax_roc.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.45, label="chance")
    ax_roc.set_xlabel("False positive rate")
    ax_roc.set_ylabel("True positive rate")
    ax_roc.set_title("ROC")
    ax_roc.set_xlim(-0.02, 1.02)
    ax_roc.set_ylim(-0.02, 1.02)
    ax_roc.legend(loc="lower right", fontsize=8, framealpha=0.92)
    ax_roc.grid(True, linestyle=":", alpha=0.45)

    ax_pr.axhline(
        prevalence,
        color="k",
        linestyle="--",
        lw=1,
        alpha=0.45,
        label="random (P = {:.4f})".format(prevalence),
    )
    ax_pr.set_xlabel("Recall")
    ax_pr.set_ylabel("Precision")
    ax_pr.set_title("PRC")
    ax_pr.set_xlim(-0.02, 1.02)
    ax_pr.set_ylim(-0.02, 1.02)
    ax_pr.legend(loc="lower left", fontsize=8, framealpha=0.92)
    ax_pr.grid(True, linestyle=":", alpha=0.45)

    au = np.array(aurocs_list, dtype=np.float64)
    ap = np.array(auprcs_list, dtype=np.float64)
    n_seeds = len(au)
    std_au = float(au.std(ddof=1)) if n_seeds > 1 else 0.0
    std_ap = float(ap.std(ddof=1)) if n_seeds > 1 else 0.0
    stats_line = (
        "mean ± std  AUROC: {:.4f} ± {:.4f}    "
        "AUPRC: {:.4f} ± {:.4f}    (n_seeds = {})".format(
            float(au.mean()), std_au, float(ap.mean()), std_ap, n_seeds
        )
    )

    fig.suptitle(title, fontsize=11, y=1.02)
    fig.text(
        0.5,
        0.965,
        stats_line,
        ha="center",
        fontsize=10,
        transform=fig.transFigure,
    )
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.88])
    fig.savefig(save_path, bbox_inches="tight", pad_inches=0.25)
    print(stats_line, file=sys.stderr)
    print("Saved: {}".format(save_path), file=sys.stderr)


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--pair",
        action="append",
        nargs=3,
        metavar=("SEED_LABEL", "BENCH_ENZYME_CSV", "BENCH_NONENZYME_CSV"),
        required=True,
        help="Triplet per seed; repeat --pair for each run.",
    )
    p.add_argument("-o", "--output", required=True, help="Output PNG path.")
    p.add_argument("--title", default="", help="Figure suptitle (optional).")
    args = p.parse_args()

    pairs: List[Tuple[str, str, str]] = [(t[0], t[1], t[2]) for t in args.pair]
    tit = args.title.strip() or (
        "Enzyme vs non-enzyme (protein-level max residue prob)"
    )
    plot_roc_prc_overlay(pairs, tit, args.output)


if __name__ == "__main__":
    main()
