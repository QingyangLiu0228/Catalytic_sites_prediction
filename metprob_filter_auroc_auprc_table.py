#!/usr/bin/env python3
"""
For each Met_prob threshold: remove residues that appear in ``predictions.tsv``
with ``Met_prob > threshold`` (predicted-metal cysteines only — TSV lists Cys).

AUROC / AUPRC are computed on **all remaining residues** in each test JSONL (every
``sample_ids`` row), using JSONL ``probs`` vs ``labels`` unless
``--score-source metprob``.

Multiple ``--jsonl`` runs can be passed (e.g. different random seeds / logs); the
script reports per-seed metrics plus **AUROC_mean**, **AUPRC_mean** (and std).

Examples
--------
  python metprob_filter_auroc_auprc_table.py \\
    --jsonl results/test/run_seed42/test_log_full_20260424_201416.delete.jsonl \\
            results/test/run_seed123/test_log_balanced_20260424_201416.delete.jsonl \\
            results/test/run_seed2026/test_log_full_20260502_020921.delete.jsonl \\
    --seed-labels seed42_full seed123_balanced seed2026_full \\
    --predictions-tsv /data/data3/conglab/s441865/cys3state/predictions.tsv \\
    --thresholds 0,0.01,0.05,0.1,0.2,0.3,0.5,0.7,0.9,1.0 \\
    -o Results/metprob_filtered_auroc_auprc_multi_seed.csv
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


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


def collect_full_jsonl_arrays(
    jsonl_path: Path,
    met: dict[tuple[str, int], float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Full test set: ``met_prob[j]`` is nan if residue not in TSV (most residues)."""
    mp_list: list[float] = []
    yt_list: list[float] = []
    y_cnn_list: list[float] = []

    with jsonl_path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            for sid, lb, pb in zip(
                row["sample_ids"], row["labels"], row["probs"], strict=True
            ):
                key = _parse_sample_id(sid)
                if key in met:
                    mp_list.append(met[key])
                else:
                    mp_list.append(float("nan"))
                yt_list.append(float(lb))
                y_cnn_list.append(float(pb))

    return (
        np.asarray(mp_list, dtype=np.float64),
        np.asarray(yt_list, dtype=np.float64),
        np.asarray(y_cnn_list, dtype=np.float64),
    )


def _safe_auroc_auprc(y_true: np.ndarray, y_score: np.ndarray) -> tuple[float, float]:
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


def default_seed_label(jsonl_path: Path) -> str:
    """Use parent ``run_seedXX`` folder name if present, else JSON stem."""
    parent = jsonl_path.parent.name
    m = re.match(r"^(run_seed\d+)$", parent)
    if m:
        return m.group(1)
    return jsonl_path.stem


def compute_rows_for_arrays(
    met_prob: np.ndarray,
    y_true: np.ndarray,
    y_cnn: np.ndarray,
    thr_list: list[float],
    score_source: str,
) -> list[dict]:
    rows: list[dict] = []
    for t in thr_list:
        excluded = np.isfinite(met_prob) & (met_prob > t)
        keep = ~excluded
        n_kept = int(np.sum(keep))
        n_drop = int(np.sum(excluded))

        if score_source == "catalytic":
            yt = y_true[keep]
            ys = y_cnn[keep]
        else:
            sub = np.isfinite(met_prob) & keep
            yt = y_true[sub]
            ys = met_prob[sub]

        auroc, auprc = _safe_auroc_auprc(yt, ys)
        n_pos_k = int(np.sum(yt >= 0.5))
        n_neg_k = int(yt.size - n_pos_k)

        rows.append(
            {
                "metprob_threshold": t,
                "n_removed_metprob_gt_threshold": n_drop,
                "n_kept": n_kept,
                "n_pos_kept": n_pos_k,
                "n_neg_kept": n_neg_k,
                "AUROC": auroc,
                "AUPRC": auprc,
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--jsonl",
        type=Path,
        nargs="+",
        required=True,
        help="One or more test JSONL files (e.g. different seeds)",
    )
    parser.add_argument(
        "--seed-labels",
        type=str,
        nargs="*",
        default=None,
        help="Labels for each --jsonl (default: parent run_seed* name or file stem)",
    )
    parser.add_argument("--predictions-tsv", type=Path, required=True)
    parser.add_argument(
        "--thresholds",
        type=str,
        required=True,
        help="Comma-separated Met_prob cutoffs (TSV sites with Met_prob > t drop out)",
    )
    parser.add_argument(
        "--score-source",
        choices=("catalytic", "metprob"),
        default="catalytic",
        help=(
            "catalytic: JSONL probs on all kept residues (default). "
            "metprob: Met_prob only on TSV residues that pass the filter"
        ),
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Optional CSV path for the summary table",
    )
    args = parser.parse_args()

    labels = args.seed_labels
    if labels is None:
        labels = [default_seed_label(p) for p in args.jsonl]
    elif len(labels) != len(args.jsonl):
        parser.error("--seed-labels length must match number of --jsonl paths")

    # Unique labels for CSV columns
    seen: set[str] = set()
    uniq_labels: list[str] = []
    for lb in labels:
        base = lb
        i = 2
        while base in seen:
            base = "{}_{}".format(lb, i)
            i += 1
        seen.add(base)
        uniq_labels.append(base)

    thr_list: list[float] = []
    for x in args.thresholds.split(","):
        x = x.strip()
        if not x:
            continue
        thr_list.append(float(x))

    met = load_met_probs(args.predictions_tsv)

    print(
        "Score for AUROC/AUPRC: {}\n".format(
            "JSONL probs (catalytic model)"
            if args.score_source == "catalytic"
            else "Met_prob (TSV residues only)"
        )
    )

    per_seed_tables: list[pd.DataFrame] = []
    for jp, lbl in zip(args.jsonl, uniq_labels, strict=True):
        if not jp.is_file():
            raise SystemExit("missing JSONL: {}".format(jp))
        met_prob, y_true, y_cnn = collect_full_jsonl_arrays(jp, met)
        n_all = met_prob.size
        n_in_tsv = int(np.sum(np.isfinite(met_prob)))
        n_pos_all = int(np.sum(y_true >= 0.5))
        print(
            "[{}] {}  |  n={} (pos={}, neg={})  |  TSV Cys keys hit: {}".format(
                lbl, jp.name, n_all, n_pos_all, n_all - n_pos_all, n_in_tsv
            )
        )
        rows = compute_rows_for_arrays(
            met_prob, y_true, y_cnn, thr_list, args.score_source
        )
        df_s = pd.DataFrame(rows)
        df_s = df_s.rename(
            columns={
                "n_removed_metprob_gt_threshold": "n_removed__{}".format(lbl),
                "n_kept": "n_kept__{}".format(lbl),
                "n_pos_kept": "n_pos_kept__{}".format(lbl),
                "n_neg_kept": "n_neg_kept__{}".format(lbl),
                "AUROC": "AUROC__{}".format(lbl),
                "AUPRC": "AUPRC__{}".format(lbl),
            }
        )
        per_seed_tables.append(df_s)

    merged = per_seed_tables[0]
    for df_next in per_seed_tables[1:]:
        merged = merged.merge(
            df_next,
            on="metprob_threshold",
            how="outer",
            validate="one_to_one",
        )

    auroc_cols = [c for c in merged.columns if c.startswith("AUROC__")]
    auprc_cols = [c for c in merged.columns if c.startswith("AUPRC__")]
    merged["AUROC_mean"] = merged[auroc_cols].mean(axis=1, skipna=True)
    merged["AUPRC_mean"] = merged[auprc_cols].mean(axis=1, skipna=True)
    if len(auroc_cols) > 1:
        merged["AUROC_std"] = merged[auroc_cols].std(axis=1, ddof=1, skipna=True)
        merged["AUPRC_std"] = merged[auprc_cols].std(axis=1, ddof=1, skipna=True)
    else:
        merged["AUROC_std"] = np.nan
        merged["AUPRC_std"] = np.nan

    pd.set_option("display.max_rows", 200)
    pd.set_option("display.width", 200)
    pd.set_option("display.float_format", lambda v: "{:.6g}".format(v))

    display_cols = ["metprob_threshold"] + auroc_cols + auprc_cols
    display_cols += ["AUROC_mean", "AUROC_std", "AUPRC_mean", "AUPRC_std"]
    display_cols = [c for c in display_cols if c in merged.columns]
    print("\n" + merged[display_cols].to_string(index=False))

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        merged.to_csv(args.output, index=False)
        print("\nWrote:", args.output)


if __name__ == "__main__":
    main()
