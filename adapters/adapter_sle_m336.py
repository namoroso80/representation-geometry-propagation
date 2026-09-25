#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build donor-level SLE gene-gene correlation matrices for Geometry 2.0.

Frozen methodology
------------------
- donor is the item
- raw counts from adata.raw
- library-size normalization to 1e4 counts per cell
- log1p transform
- genes ranked by donor-balanced variance from the prior unsupervised audit
- top m=336 genes
- no disease label used in feature selection
- one Pearson gene-gene correlation matrix per donor
"""

from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.sparse as sp
import anndata as ad


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--h5ad", type=Path, required=True)
    p.add_argument("--gene-rankings", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--donor-col", default="donor_id")
    p.add_argument("--label-col", default="disease")
    p.add_argument("--method", default="variance")
    p.add_argument("--m", type=int, default=336)
    p.add_argument("--target-sum", type=float, default=1e4)
    p.add_argument("--rank-tol", type=float, default=1e-10)
    p.add_argument("--psd-tol", type=float, default=1e-8)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def normalize_log1p(X, target_sum):
    if sp.issparse(X):
        X = X.tocsr().astype(np.float64)
        lib = np.asarray(X.sum(axis=1)).ravel()
        scale = np.divide(target_sum, lib, out=np.zeros_like(lib), where=lib > 0)
        Z = X.multiply(scale[:, None]).tocsr()
        Z.data = np.log1p(Z.data)
        return Z
    X = np.asarray(X, dtype=np.float64)
    lib = X.sum(axis=1)
    scale = np.divide(target_sum, lib, out=np.zeros_like(lib), where=lib > 0)
    return np.log1p(X * scale[:, None])


def effective_rank(eigvals):
    eigvals = np.clip(np.asarray(eigvals, float), 0, None)
    s = eigvals.sum()
    if s <= 0:
        return 0.0
    p = eigvals / s
    p = p[p > 0]
    return float(np.exp(-np.sum(p * np.log(p))))


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if not args.overwrite and any(args.output_dir.glob("adjacency_*.csv")):
        raise FileExistsError(f"{args.output_dir} already contains matrices; use --overwrite")

    ranks = pd.read_csv(args.gene_rankings)
    rr = ranks[ranks["method"] == args.method].sort_values("rank")
    if len(rr) < args.m:
        raise ValueError(f"Only {len(rr)} ranked genes available for method={args.method}")

    chosen = rr.iloc[:args.m].copy()
    gene_indices = chosen["gene_index"].to_numpy(dtype=int)
    gene_names_ranked = chosen["gene_name"].astype(str).to_numpy()
    gene_ids_ranked = chosen["gene_id"].astype(str).to_numpy()

    a = ad.read_h5ad(args.h5ad, backed="r")
    if a.raw is None:
        raise ValueError("Expected raw counts in adata.raw")
    if args.donor_col not in a.obs.columns:
        raise KeyError(args.donor_col)
    if args.label_col not in a.obs.columns:
        raise KeyError(args.label_col)

    donors = a.obs[args.donor_col].astype("string")
    labels = a.obs[args.label_col].astype("string")
    donor_names = pd.Index(pd.unique(donors))

    # HDF5 column slicing is safest in sorted index order.
    sorted_genes = np.sort(gene_indices)
    col_lookup = {g: j for j, g in enumerate(sorted_genes)}
    ranked_cols = np.array([col_lookup[g] for g in gene_indices], dtype=int)

    info_rows = []
    qc_rows = []

    for k, donor in enumerate(donor_names, start=1):
        row_idx = np.flatnonzero((donors == donor).to_numpy())
        donor_labels = pd.unique(labels.iloc[row_idx].dropna())
        if len(donor_labels) != 1:
            raise ValueError(f"Donor {donor}: inconsistent labels {donor_labels}")
        label = str(donor_labels[0])

        X = a.raw.X[row_idx, sorted_genes]
        Z = normalize_log1p(X, args.target_sum)
        if sp.issparse(Z):
            Z = Z.toarray()
        Z = np.asarray(Z, dtype=np.float64)[:, ranked_cols]

        sd = Z.std(axis=0, ddof=1)
        if np.any(~np.isfinite(sd)) or np.any(sd <= 1e-12):
            bad = np.flatnonzero((~np.isfinite(sd)) | (sd <= 1e-12))
            raise ValueError(f"Donor {donor}: constant/invalid selected genes {bad[:10].tolist()}")

        C = np.corrcoef(Z, rowvar=False)
        C = 0.5 * (C + C.T)
        np.fill_diagonal(C, 1.0)

        eig = np.linalg.eigvalsh(C)
        rank = int(np.sum(eig > args.rank_tol))
        min_eig = float(eig[0])
        max_eig = float(eig[-1])
        pos = eig[eig > args.rank_tol]
        cond = float(max_eig / pos[0]) if len(pos) else np.inf

        if min_eig < -args.psd_tol:
            raise ValueError(f"Donor {donor}: PSD check failed, min eig={min_eig:.3e}")

        out_name = f"adjacency_{k:04d}.csv"
        np.savetxt(args.output_dir / out_name, C, delimiter=",", fmt="%.10g")

        info_rows.append({
            "sample_id": f"SLE_{k:04d}",
            "donor_id": str(donor),
            "diagnosis": label,
            "matrix_file": out_name,
            "n_cells": int(len(row_idx)),
        })
        qc_rows.append({
            "sample_id": f"SLE_{k:04d}",
            "donor_id": str(donor),
            "diagnosis": label,
            "n_cells": int(len(row_idx)),
            "n_genes": int(args.m),
            "matrix_rank": rank,
            "effective_rank": effective_rank(eig),
            "min_eigenvalue": min_eig,
            "max_eigenvalue": max_eig,
            "condition_number": cond,
            "trace": float(np.trace(C)),
        })

        if k == 1 or k % 20 == 0 or k == len(donor_names):
            print(f"[SLE] donors {k}/{len(donor_names)}")

    info = pd.DataFrame(info_rows)
    qc = pd.DataFrame(qc_rows)
    info.to_csv(args.output_dir / "info.csv", index=False)
    qc.to_csv(args.output_dir / "adapter_qc.csv", index=False)

    chosen[["rank","gene_index","gene_id","gene_name","score"]].to_csv(
        args.output_dir / "selected_genes_m336.csv", index=False
    )

    print("[DONE] SLE matrices built")
    print(f"[DATA] donors={len(info)} | matrix=({args.m},{args.m})")
    print("[LABELS]")
    print(info["diagnosis"].value_counts().to_string())
    print(
        f"[QC] rank range={qc['matrix_rank'].min()}-{qc['matrix_rank'].max()} | "
        f"min eigenvalue={qc['min_eigenvalue'].min():.3e} | "
        f"trace range={qc['trace'].min():.6f}-{qc['trace'].max():.6f}"
    )
    print(f"[SAVE] {args.output_dir}")

    if a.file is not None:
        a.file.close()


if __name__ == "__main__":
    main()
