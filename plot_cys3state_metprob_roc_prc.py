#!/usr/bin/env python3
"""
Match cys3state predictions.tsv (Met_prob, residue = FASTA 1-based Cys index)
to residue-level labels in test_log_*full*.jsonl, then plot ROC + PR curves.

Scores are continuous Met_prob; optional markers show operating points at
fixed Met_prob thresholds.

Example:
  python plot_cys3state_metprob_roc_prc.py \\
    --jsonl results/test/run_seed42/test_log_full_20260424_201416.delete.jsonl \\
    --predictions-tsv /data/data3/conglab/s441865/cys3state/predictions.tsv \\
    -o Results/cys3state_metprob_vs_test_roc_prc.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)


def _parse_sample_id(sid: str) -> tuple[str, int]:
    parts = str(sid).split("|")
    acc = parts[0].replace(".pdb", "").replace(".PDB", "")
    resid = int(parts[2])
    return acc, resid


def load_met_probs(tsv_path: Path) -> dict[tuple[str, int], float]:
    df = pd.read_csv(tsv_path, sep="\t")
    out: dict[tuple[str, int], float] = {}
    for row in df.itertuples(index=False):
        key = (str(row.Protein), int(row.Residue))
        out[key] = float(row.Met_prob)
    return out


def collect_labels_scores(
    jsonl_path: Path,
    met: dict[tuple[str, int], float],
) -> tuple[np.ndarray, np.ndarray, list[tuple[str, int]], list[tuple[str, int]]]:
    """Return y_true, y_score for keys present in ``met`` and jsonl."""
    y_true: list[float] = []
    y_score: list[float] = []
    matched_keys: list[tuple[str, int]] = []

    present_in_jsonl: set[tuple[str, int]] = set()

    with jsonl_path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            for sid, lb in zip(row["sample_ids"], row["labels"], strict=True):
                key = _parse_sample_id(sid)
                if key in met:
                    present_in_jsonl.add(key)
                    y_true.append(float(lb))
                    y_score.append(met[key])
                    matched_keys.append(key)

    missing_from_jsonl = sorted(set(met.keys()) - present_in_jsonl)
    return (
        np.asarray(y_true, dtype=np.float64),
        np.asarray(y_score, dtype=np.float64),
        matched_keys,
        missing_from_jsonl,
    )


def metrics_at_threshold(y_true: np.ndarray, y_score: np.ndarray, t: float) -> dict:
    y_pred = (y_score >= t).astype(np.int32)
    yt = (y_true >= 0.5).astype(np.int32)
    tp = int(np.sum((yt == 1) & (y_pred == 1)))
    fp = int(np.sum((yt == 0) & (y_pred == 1)))
    tn = int(np.sum((yt == 0) & (y_pred == 0)))
    fn = int(np.sum((yt == 1) & (y_pred == 0)))
    sens = tp / (tp + fn) if (tp + fn) else float("nan")
    spec = tn / (tn + fp) if (tn + fp) else float("nan")
    prec = tp / (tp + fp) if (tp + fp) else float("nan")
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn, "sens": sens, "spec": spec, "prec": prec}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jsonl", type=Path, required=True, help="test_log_*full*.jsonl")
    parser.add_argument(
        "--predictions-tsv",
        type=Path,
        required=True,
        help="cys3state predictions.tsv",
    )
    parser.add_argument("-o", "--output", type=Path, required=True, help="Output PNG path")
    parser.add_argument(
        "--thresholds",
        type=str,
        default="0.01,0.05,0.1,0.2,0.3,0.5,0.7,0.9",
        help="Comma-separated Met_prob thresholds for scatter markers",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="Cys3state Met_prob vs test labels (matched Cys residues)",
    )
    args = parser.parse_args()

    met = load_met_probs(args.predictions_tsv)
    y_true, y_score, _, missing = collect_labels_scores(args.jsonl, met)

    n_pos = int(np.sum(y_true >= 0.5))
    n_neg = int(np.sum(y_true < 0.5))
    print(
        "Matched cys3state ↔ jsonl residues: {}  (pos={}, neg={})".format(
            y_true.size, n_pos, n_neg
        )
    )
    print("cys3state sites not found in jsonl: {}".format(len(missing)))
    if n_pos == 0 or n_neg == 0:
        raise SystemExit("Need both classes for ROC/PR (check labels).")

    auroc = float(roc_auc_score(y_true, y_score))
    auprc = float(average_precision_score(y_true, y_score))
    print("AUROC={:.4f}  AUPRC={:.4f}".format(auroc, auprc))

    thresh_list = [float(x.strip()) for x in args.thresholds.split(",") if x.strip()]
    print("\nMet_prob threshold  |  P  |  R  |  FPR  |  F1")
    print("-" * 52)
    for t in thresh_list:
        m = metrics_at_threshold(y_true, y_score, t)
        fpr_p = m["fp"] / (m["fp"] + m["tn"]) if (m["fp"] + m["tn"]) else float("nan")
        pr, rc = m["prec"], m["sens"]
        f1 = (
            2 * pr * rc / (pr + rc)
            if not np.isnan(pr) and not np.isnan(rc) and (pr + rc) > 0
            else float("nan")
        )
        print(
            "{:17.4f} | {:5.3f} | {:5.3f} | {:6.4f} | {:6.3f}".format(
                t, pr, rc, fpr_p, f1
            )
        )
    fpr, tpr, _ = roc_curve(y_true, y_score)
    prec, rec, _ = precision_recall_curve(y_true, y_score)
    prevalence = float(np.mean(y_true))

    fig, (ax_roc, ax_pr) = plt.subplots(
        1,
        2,
        figsize=(12, 5.2),
        dpi=150,
        gridspec_kw={"width_ratios": [1, 1], "wspace": 0.28},
    )

    ax_roc.plot(fpr, tpr, color="#2563EB", lw=2, label="Met_prob (AUROC={:.4f})".format(auroc))
    ax_roc.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.45)
    ax_roc.set_xlabel("False positive rate")
    ax_roc.set_ylabel("True positive rate")
    ax_roc.set_title("ROC")
    ax_roc.set_xlim(-0.02, 1.02)
    ax_roc.set_ylim(-0.02, 1.02)
    ax_roc.grid(True, linestyle=":", alpha=0.45)

    colors = plt.cm.plasma(np.linspace(0.15, 0.85, len(thresh_list)))
    for t, c in zip(thresh_list, colors):
        m = metrics_at_threshold(y_true, y_score, t)
        # ROC point: (FPR, TPR) = (FP/(FP+TN), TP/(TP+FN))
        fpr_p = m["fp"] / (m["fp"] + m["tn"]) if (m["fp"] + m["tn"]) else float("nan")
        tpr_p = m["sens"]
        if not np.isnan(fpr_p) and not np.isnan(tpr_p):
            ax_roc.scatter(
                [fpr_p],
                [tpr_p],
                color=[c],
                s=36,
                zorder=5,
                edgecolors="white",
                linewidths=0.6,
            )

    ax_roc.legend(loc="lower right", fontsize=8, framealpha=0.92)

    ax_pr.plot(
        rec,
        prec,
        color="#059669",
        lw=2,
        label="Met_prob (AUPRC={:.4f})".format(auprc),
    )
    ax_pr.axhline(prevalence, color="k", linestyle="--", lw=1, alpha=0.45)
    for t, c in zip(thresh_list, colors):
        m = metrics_at_threshold(y_true, y_score, t)
        if not np.isnan(m["prec"]) and not np.isnan(m["sens"]):
            ax_pr.scatter(
                [m["sens"]],
                [m["prec"]],
                color=[c],
                s=36,
                zorder=5,
                edgecolors="white",
                linewidths=0.6,
            )

    ax_pr.set_xlabel("Recall (sensitivity)")
    ax_pr.set_ylabel("Precision")
    ax_pr.set_title("Precision–recall")
    ax_pr.set_xlim(-0.02, 1.02)
    ax_pr.set_ylim(-0.02, 1.02)
    ax_pr.grid(True, linestyle=":", alpha=0.45)
    ax_pr.legend(loc="upper right", fontsize=8, framealpha=0.92)

    thr_txt = "Threshold markers: " + ", ".join("{:.2g}".format(t) for t in thresh_list)
    fig.suptitle(args.title, fontsize=11, y=0.995)
    fig.text(
        0.5,
        0.93,
        "n={} Cys sites  |  positives={}  |  {}".format(y_true.size, n_pos, thr_txt),
        ha="center",
        fontsize=9,
        transform=fig.transFigure,
    )
    plt.subplots_adjust(left=0.07, right=0.98, top=0.86, bottom=0.12)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, bbox_inches="tight", pad_inches=0.25)
    plt.close()
    print("Saved:", args.output)


if __name__ == "__main__":
    main()
