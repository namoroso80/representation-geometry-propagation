#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
import argparse
from pathlib import Path

try:
    import anndata as ad
except ImportError as exc:
    raise RuntimeError("Install anndata with: pip install anndata") from exc

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--h5ad", type=Path, required=True)
    p.add_argument("--top-obs", type=int, default=80)
    args = p.parse_args()

    print(f"[OPEN] {args.h5ad}")
    a = ad.read_h5ad(args.h5ad, backed="r")

    print(f"[SHAPE] cells={a.n_obs:,} genes={a.n_vars:,}")
    print(f"[X] type={type(a.X).__name__}")
    print(f"[LAYERS] {list(a.layers.keys())}")
    print(f"[OBSM] {list(a.obsm.keys())}")
    print(f"[VARM] {list(a.varm.keys())}")
    print(f"[RAW] {'present' if a.raw is not None else 'absent'}")

    print("\n[OBS COLUMNS]")
    for c in list(a.obs.columns)[:args.top_obs]:
        s = a.obs[c]
        nunique = s.nunique(dropna=True)
        print(f"{c}\tdtype={s.dtype}\tnunique={nunique}")
        if nunique <= 20:
            vals = s.astype("string").value_counts(dropna=False).head(20)
            print("  " + " | ".join(f"{k}:{v}" for k, v in vals.items()))

    donor_terms = ("donor", "patient", "subject", "individual", "sample")
    donor_cols = [c for c in a.obs.columns if any(t in c.lower() for t in donor_terms)]
    print("\n[CANDIDATE DONOR COLUMNS]")
    print(donor_cols)

    label_terms = ("disease", "diagnosis", "condition", "phenotype", "status", "sle")
    label_cols = [c for c in a.obs.columns if any(t in c.lower() for t in label_terms)]
    print("\n[CANDIDATE LABEL COLUMNS]")
    print(label_cols)

    for c in donor_cols[:10]:
        vc = a.obs[c].astype("string").value_counts(dropna=False)
        print(f"\n[DONOR DISTRIBUTION] {c}")
        print(f"n_unique={vc.size}")
        print(vc.describe().to_string())
        print("smallest:")
        print(vc.sort_values().head(10).to_string())
        print("largest:")
        print(vc.sort_values(ascending=False).head(10).to_string())

    for c in label_cols[:10]:
        print(f"\n[LABEL DISTRIBUTION] {c}")
        print(a.obs[c].astype("string").value_counts(dropna=False).head(30).to_string())

    print("\n[VAR COLUMNS]")
    for c in a.var.columns:
        print(c)

    if a.file is not None:
        a.file.close()

if __name__ == "__main__":
    main()
