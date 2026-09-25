#!/usr/bin/env python3
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

COLS = ["auc_heldout","gamma_star","Q_typical","decision_concordance","xai_concordance","leading_mode_share","decision_loss","xai_loss"]

def main():
    ap=argparse.ArgumentParser(description="Compare two geometry-pipeline summary fingerprints")
    ap.add_argument("reference", type=Path)
    ap.add_argument("candidate", type=Path)
    ap.add_argument("--atol", type=float, default=1e-10)
    ap.add_argument("--rtol", type=float, default=1e-8)
    a=ap.parse_args()
    ref=pd.read_csv(a.reference).sort_values("p").reset_index(drop=True)
    cur=pd.read_csv(a.candidate).sort_values("p").reset_index(drop=True)
    if not np.allclose(ref.p,cur.p,rtol=0,atol=1e-12):
        raise SystemExit("FAIL: p grids differ")
    rows=[]; ok=True
    for c in COLS:
        if c not in ref or c not in cur: continue
        x=ref[c].to_numpy(float); y=cur[c].to_numpy(float)
        finite=np.isfinite(x)&np.isfinite(y)
        max_abs=float(np.max(np.abs(x[finite]-y[finite]))) if finite.any() else np.nan
        good=np.allclose(x,y,rtol=a.rtol,atol=a.atol,equal_nan=True)
        rows.append((c,max_abs,good)); ok &= good
    print("column,max_abs_diff,pass")
    for c,d,g in rows: print(f"{c},{d:.12g},{g}")
    print("PASS" if ok else "FAIL")
    raise SystemExit(0 if ok else 1)
if __name__ == "__main__": main()
