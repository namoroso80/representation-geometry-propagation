#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Regenerate only 04_decision_vs_xai_concordance.csv from existing outputs.

This does NOT recompute XAI. It reads:
  - 03_xai_concordance_vs_p1.csv
  - 09_decision_summary_by_p.csv

and writes:
  - 04_decision_vs_xai_concordance.csv

NaN correlations are excluded from XAI aggregation and their counts are reported.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--xai-dir", type=Path, required=True)
    p.add_argument("--decision-dir", type=Path, required=True)
    return p.parse_args()


def pick_col(df: pd.DataFrame, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(f"None of these columns found: {candidates}. Available: {list(df.columns)}")


def main():
    args = parse_args()

    xai_path = args.xai_dir / "03_xai_concordance_vs_p1.csv"
    decision_path = args.decision_dir / "09_decision_summary_by_p.csv"
    out_path = args.xai_dir / "04_decision_vs_xai_concordance.csv"

    xdf = pd.read_csv(xai_path)
    ddf = pd.read_csv(decision_path)

    p_col_x = pick_col(xdf, ["p"])
    spear_x = pick_col(
        xdf,
        ["spearman_vs_p1", "spearman", "xai_spearman_vs_p1", "subject_spearman_vs_p1"]
    )
    pear_x = pick_col(
        xdf,
        ["pearson_vs_p1", "pearson", "xai_pearson_vs_p1", "subject_pearson_vs_p1"]
    )

    rows = []
    for pval, g in xdf.groupby(p_col_x, sort=True):
        sx = g[spear_x].to_numpy(dtype=float)
        px = g[pear_x].to_numpy(dtype=float)

        sx_f = sx[np.isfinite(sx)]
        px_f = px[np.isfinite(px)]

        rows.append({
            "p": float(pval),
            "xai_spearman_median": float(np.median(sx_f)) if sx_f.size else np.nan,
            "xai_spearman_q25": float(np.quantile(sx_f, 0.25)) if sx_f.size else np.nan,
            "xai_spearman_q75": float(np.quantile(sx_f, 0.75)) if sx_f.size else np.nan,
            "xai_pearson_median": float(np.median(px_f)) if px_f.size else np.nan,
            "xai_pearson_q25": float(np.quantile(px_f, 0.25)) if px_f.size else np.nan,
            "xai_pearson_q75": float(np.quantile(px_f, 0.75)) if px_f.size else np.nan,
            "xai_n_total": int(len(g)),
            "xai_spearman_n_valid": int(sx_f.size),
            "xai_spearman_n_invalid": int(len(sx) - sx_f.size),
            "xai_pearson_n_valid": int(px_f.size),
            "xai_pearson_n_invalid": int(len(px) - px_f.size),
        })

    xsum = pd.DataFrame(rows)

    # Preserve all decision summary columns and append corrected XAI summaries.
    out = ddf.merge(xsum, on="p", how="left")
    out.to_csv(out_path, index=False)

    print(f"[SAVE] {out_path}")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
