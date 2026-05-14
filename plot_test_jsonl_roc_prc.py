#!/usr/bin/env python3
"""
ROC & precision-recall curves from test_log_*_{mode}_*.jsonl files.

Mirrors styling and logic spirit of plot_bench_auroc_auprc.ipynb:
  - residue-level: every (prob, label) pair across batches;
  - protein-level: score = max(prob) within each accession (sample_ids prefix before
    the first '|'); label = 1 if any residue in that protein is positive, else 0.

Examples:
  python plot_test_jsonl_roc_prc.py \\
    results/test/run_seed42/test_log_full_20260424_201416.delete.jsonl \\
    -o roc_prc_full_seed42.png

  python plot_test_jsonl_roc_prc.py --level protein \\
    full_seed42.jsonl full_seed123.jsonl \\
    --names seed42 seed123 -o roc_prc_overlay.png

Note: ``protein`` matches the bench notebook (max prob per accession). It requires both
positive and negative proteins in the JSONL; pure enzyme sets yield undefined AUROC.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)


def _jsonl_rows(path: Path) -> Iterable[dict]:
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def residue_level_arrays(jsonl_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    ys: List[float] = []
    scores: List[float] = []
    for row in _jsonl_rows(jsonl_path):
        for pl, tl in zip(row["probs"], row["labels"], strict=True):
            scores.append(float(pl))
            ys.append(float(tl))
    y_true = np.asarray(ys, dtype=np.float64)
    y_score = np.asarray(scores, dtype=np.float64)
    return y_true, y_score


def protein_level_arrays(jsonl_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """One score per protein (max prob); label OR across residues."""
    max_prob: Dict[str, float] = {}
    pos: Dict[str, bool] = {}
    for row in _jsonl_rows(jsonl_path):
        for sid, pl, tl in zip(
            row["sample_ids"], row["probs"], row["labels"], strict=True
        ):
            acc = str(sid).split("|", 1)[0]
            p = float(pl)
            prev = max_prob.get(acc)
            if prev is None or p > prev:
                max_prob[acc] = p
            if float(tl) >= 0.5:
                pos[acc] = True
    accs = sorted(max_prob.keys())
    y_true = np.array([1.0 if pos.get(a, False) else 0.0 for a in accs])
    y_score = np.array([max_prob[a] for a in accs], dtype=np.float64)
    return y_true, y_score


def _metrics_safe(y_true: np.ndarray, y_score: np.ndarray) -> Tuple[float, float]:
    if y_true.size == 0:
        return float("nan"), float("nan")
    n_pos = int(np.sum(y_true >= 0.5))
    n_neg = int(np.sum(y_true < 0.5))
    if n_pos == 0 or n_neg == 0:
        return float("nan"), float("nan")
    return (
        float(roc_auc_score(y_true, y_score)),
        float(average_precision_score(y_true, y_score)),
    )


def plot_roc_prc_overlay(
    pairs: List[Tuple[str, Path]],
    *,
    level: str,
    title: str,
    save_path: Path | None,
) -> None:
    colors = ["#2563EB", "#059669", "#D97706", "#7C3AED", "#DC2626"]
    fig, (ax_roc, ax_pr) = plt.subplots(
        1,
        2,
        figsize=(12, 5.2),
        dpi=150,
        gridspec_kw={"width_ratios": [1, 1], "wspace": 0.28},
        constrained_layout=False,
    )

    load = residue_level_arrays if level == "residue" else protein_level_arrays

    y0, _ = load(pairs[0][1])
    prevalence = float(np.mean(y0)) if y0.size else float("nan")

    aurocs_list: List[float] = []
    auprcs_list: List[float] = []

    for i, (curve_label, jp) in enumerate(pairs):
        y_true, y_score = load(jp)
        auroc, auprc = _metrics_safe(y_true, y_score)
        aurocs_list.append(auroc)
        auprcs_list.append(auprc)

        if np.isnan(auroc):
            print(
                "[skip curves] {}  n={}, n_pos={}, n_neg={} — need both classes".format(
                    curve_label,
                    y_true.size,
                    int(np.sum(y_true >= 0.5)),
                    int(np.sum(y_true < 0.5)),
                )
            )
            continue

        fpr, tpr, _ = roc_curve(y_true, y_score)
        prec, rec, _ = precision_recall_curve(y_true, y_score)
        c = colors[i % len(colors)]
        ax_roc.plot(
            fpr,
            tpr,
            color=c,
            lw=2,
            label="{} (AUROC = {:.4f})".format(curve_label, auroc),
        )
        ax_pr.plot(
            rec,
            prec,
            color=c,
            lw=2,
            label="{} (AUPRC = {:.4f})".format(curve_label, auprc),
        )

    ax_roc.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.45, label="chance")
    ax_roc.set_xlabel("False positive rate")
    ax_roc.set_ylabel("True positive rate")
    ax_roc.set_title("ROC")
    ax_roc.set_xlim(-0.02, 1.02)
    ax_roc.set_ylim(-0.02, 1.02)
    ax_roc.legend(loc="lower right", fontsize=8, framealpha=0.92)
    ax_roc.grid(True, linestyle=":", alpha=0.45)

    if not np.isnan(prevalence):
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
    ax_pr.set_title("Precision–recall")
    ax_pr.set_xlim(-0.02, 1.02)
    ax_pr.set_ylim(-0.02, 1.02)
    ax_pr.legend(loc="lower left", fontsize=8, framealpha=0.92)
    ax_pr.grid(True, linestyle=":", alpha=0.45)

    au = np.array([x for x in aurocs_list if not np.isnan(x)], dtype=np.float64)
    ap = np.array([x for x in auprcs_list if not np.isnan(x)], dtype=np.float64)
    n_valid = au.size
    if n_valid == 0:
        stats_line = "AUROC/AUPRC: n/a (no valid curves)"
    else:
        std_au = float(au.std(ddof=1)) if n_valid > 1 else 0.0
        std_ap = float(ap.std(ddof=1)) if n_valid > 1 else 0.0
        stats_line = (
            "mean ± std  AUROC: {:.4f} ± {:.4f}    "
            "AUPRC: {:.4f} ± {:.4f}    (n_curves = {})"
        ).format(float(au.mean()), std_au, float(ap.mean()), std_ap, n_valid)

    lvl_note = "residue-level" if level == "residue" else "protein-level max(prob)"
    fig.suptitle("{} — {}".format(title, lvl_note), fontsize=11, y=0.995)
    fig.text(
        0.5,
        0.94,
        stats_line,
        ha="center",
        fontsize=10,
        transform=fig.transFigure,
    )
    plt.subplots_adjust(left=0.07, right=0.98, top=0.86, bottom=0.12, wspace=0.28)

    if save_path:
        fig.savefig(save_path, bbox_inches="tight", pad_inches=0.25)
        print("Saved:", save_path)
    print(stats_line)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "jsonl",
        nargs="+",
        type=Path,
        help="One or more test_log_*.jsonl paths",
    )
    parser.add_argument(
        "--names",
        nargs="*",
        default=None,
        help="Legend labels (same length as jsonl paths; default: stems)",
    )
    parser.add_argument(
        "--level",
        choices=("residue", "protein"),
        default="residue",
        help=(
            "residue = per-residue ROC/PR (default); protein = max(prob) per "
            "accession (needs mixed pos/neg proteins, cf. bench notebook)"
        ),
    )
    parser.add_argument(
        "--title",
        type=str,
        default="Test set — 3D-CNN",
        help="Figure title prefix",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="PNG path (omit to only display interactively)",
    )
    args = parser.parse_args()

    paths = args.jsonl
    if args.names is None:
        names = [p.stem for p in paths]
    else:
        if len(args.names) != len(paths):
            parser.error("--names length must match number of jsonl files")
        names = args.names

    pairs = list(zip(names, paths, strict=True))

    for _, jp in pairs:
        if not jp.is_file():
            raise SystemExit("missing file: {}".format(jp))

    plot_roc_prc_overlay(
        pairs,
        level=args.level,
        title=args.title,
        save_path=args.output,
    )

    if args.output is None:
        plt.show()
    else:
        plt.close()


if __name__ == "__main__":
    main()
