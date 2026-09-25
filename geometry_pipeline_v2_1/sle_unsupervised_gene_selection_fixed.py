#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SLE unsupervised gene-selection audit
======================================

Purpose
-------
Evaluate, without using disease labels, whether a stable and defensible
unsupervised gene-filtering criterion exists before constructing donor-level
gene-gene correlation matrices.

Pipeline
--------
1. Open the CELLxGENE H5AD in backed mode.
2. Use donor_id as the experimental unit.
3. Read expression in row chunks.
4. If the chosen matrix appears count-like, normalize each cell to 1e4 counts
   and apply log1p before computing variability statistics.
5. Accumulate donor x gene sufficient statistics:
      sum(x), sum(x^2), number of detected cells, number of cells.
6. Apply label-free eligibility filters.
7. Rank eligible genes with three unsupervised criteria:
      - donor-balanced variance
      - donor-balanced Fano factor
      - mean-adjusted variance residual (HVG-like)
8. Recompute rankings on repeated random subsets of donors and quantify
   top-m stability by Jaccard overlap with the full-data ranking.
9. For the most stable ranking method, perform a small matrix-dimensionality
   audit on sampled donors for several m values.

No disease/diagnosis field is used anywhere in feature selection.

Outputs
-------
gene_statistics.csv
gene_rankings.csv
ranking_stability.csv
matrix_dimensionality_qc.csv
selection_recommendation.txt
01_mean_variance.png
02_stability_vs_m.png
03_matrix_effective_rank_vs_m.png
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd

try:
    import anndata as ad
except ImportError as exc:
    raise RuntimeError("Install anndata with: pip install anndata") from exc

try:
    import scipy.sparse as sp
except ImportError as exc:
    raise RuntimeError("Install scipy with: pip install scipy") from exc

import matplotlib.pyplot as plt


# ----------------------------- CLI ---------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--h5ad", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--donor-col", default="donor_id")
    p.add_argument("--matrix-source", choices=["auto", "X", "raw"], default="auto")
    p.add_argument("--chunk-size", type=int, default=5000)
    p.add_argument("--target-sum", type=float, default=1e4)
    p.add_argument("--min-donor-prevalence", type=float, default=0.90,
                   help="Fraction of donors in which gene must be detected at least once.")
    p.add_argument("--min-balanced-detection", type=float, default=0.02,
                   help="Minimum donor-balanced cell detection rate.")
    p.add_argument("--n-resamples", type=int, default=30)
    p.add_argument("--subject-fraction", type=float, default=0.80)
    p.add_argument("--m-values", default="64,128,256,512,1000")
    p.add_argument("--qc-donors", type=int, default=20)
    p.add_argument("--random-state", type=int, default=42)
    return p.parse_args()


# -------------------------- Utilities -------------------------------

def parse_m_values(text: str) -> list[int]:
    vals = sorted({int(x.strip()) for x in text.split(",") if x.strip()})
    if not vals or vals[0] <= 1:
        raise ValueError("--m-values must contain integers > 1")
    return vals


def get_gene_names_from_var(var_names, var_df) -> tuple[np.ndarray, np.ndarray]:
    gene_id = np.asarray(pd.Index(var_names).astype(str))
    if "feature_name" in var_df.columns:
        gene_name = var_df["feature_name"].astype(str).to_numpy()
    else:
        gene_name = gene_id.copy()
    return gene_id, gene_name


def choose_matrix(a, source: str):
    if source == "raw":
        if a.raw is None:
            raise ValueError("--matrix-source raw requested but adata.raw is absent")
        return a.raw.X, "raw"
    if source == "X":
        return a.X, "X"

    # auto: prefer raw if present, but inspect count-likeness below
    if a.raw is not None:
        return a.raw.X, "raw"
    return a.X, "X"


def inspect_count_like(matrix, n_rows: int = 2000) -> tuple[bool, dict]:
    n = min(n_rows, matrix.shape[0])
    block = matrix[:n]
    if sp.issparse(block):
        vals = block.data
    else:
        vals = np.asarray(block).ravel()
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return False, {"sample_values": 0, "max": np.nan, "integer_fraction": np.nan}
    if vals.size > 200000:
        vals = vals[:200000]
    integer_fraction = float(np.mean(np.isclose(vals, np.round(vals), atol=1e-8)))
    nonnegative = bool(np.nanmin(vals) >= -1e-12)
    vmax = float(np.nanmax(vals))
    count_like = bool(nonnegative and integer_fraction > 0.995 and vmax > 1.0)
    return count_like, {
        "sample_values": int(vals.size),
        "max": vmax,
        "integer_fraction": integer_fraction,
        "nonnegative": nonnegative,
    }


def transform_block(block, count_like: bool, target_sum: float):
    """
    Returns:
      Z: sparse/dense transformed expression used for variability
      detected: binary sparse/dense detection from original expression
    """
    if sp.issparse(block):
        X = block.tocsr().astype(np.float64)
        detected = X.copy()
        detected.data = np.ones_like(detected.data, dtype=np.float64)

        if count_like:
            lib = np.asarray(X.sum(axis=1)).ravel()
            scale = np.divide(target_sum, lib, out=np.zeros_like(lib), where=lib > 0)
            Z = X.multiply(scale[:, None]).tocsr()
            Z.data = np.log1p(Z.data)
        else:
            Z = X
        return Z, detected

    X = np.asarray(block, dtype=np.float64)
    detected = (X > 0).astype(np.float64)
    if count_like:
        lib = X.sum(axis=1)
        scale = np.divide(target_sum, lib, out=np.zeros_like(lib), where=lib > 0)
        Z = np.log1p(X * scale[:, None])
    else:
        Z = X
    return Z, detected


def aggregate_chunk_by_donor(
    Z, detected, donor_codes: np.ndarray, n_donors: int, n_genes: int,
    sums: np.ndarray, sumsqs: np.ndarray, detects: np.ndarray, ncells: np.ndarray
):
    # Chunk donor groups are small enough to aggregate one donor at a time.
    unique = np.unique(donor_codes)
    for d in unique:
        mask = donor_codes == d
        ncells[d] += int(mask.sum())

        if sp.issparse(Z):
            Zd = Z[mask]
            sums[d] += np.asarray(Zd.sum(axis=0)).ravel()
            sq = Zd.copy()
            sq.data **= 2
            sumsqs[d] += np.asarray(sq.sum(axis=0)).ravel()
            detects[d] += np.asarray(detected[mask].sum(axis=0)).ravel()
        else:
            Zd = Z[mask]
            sums[d] += Zd.sum(axis=0)
            sumsqs[d] += np.square(Zd).sum(axis=0)
            detects[d] += detected[mask].sum(axis=0)


def donor_balanced_stats(
    sums: np.ndarray, sumsqs: np.ndarray, detects: np.ndarray, ncells: np.ndarray,
    donor_subset: np.ndarray | None = None
) -> Dict[str, np.ndarray]:
    if donor_subset is None:
        donor_subset = np.arange(len(ncells))
    ds = np.asarray(donor_subset, dtype=int)

    n = ncells[ds].astype(np.float64)
    if np.any(n <= 0):
        raise ValueError("Encountered donor with zero cells")

    mean_d = sums[ds] / n[:, None]
    ex2_d = sumsqs[ds] / n[:, None]
    var_d = np.maximum(ex2_d - mean_d**2, 0.0)
    det_d = detects[ds] / n[:, None]

    mean_bal = mean_d.mean(axis=0)
    var_bal = var_d.mean(axis=0)
    det_bal = det_d.mean(axis=0)
    donor_prev = (detects[ds] > 0).mean(axis=0)

    return {
        "mean": mean_bal,
        "variance": var_bal,
        "detection": det_bal,
        "donor_prevalence": donor_prev,
    }


def mean_adjusted_residual_score(mean: np.ndarray, variance: np.ndarray, eligible: np.ndarray) -> np.ndarray:
    """
    HVG-like score without extra dependencies:
    regress log10 variance on log10 mean using robust binned medians,
    then score genes by positive/negative residual from the trend.
    """
    eps = 1e-12
    x = np.log10(np.maximum(mean, eps))
    y = np.log10(np.maximum(variance, eps))
    score = np.full_like(mean, np.nan, dtype=np.float64)

    idx = np.flatnonzero(eligible & np.isfinite(x) & np.isfinite(y))
    if len(idx) < 100:
        return score

    xi = x[idx]
    yi = y[idx]
    # Quantile bins preserve enough genes across expression levels.
    qs = np.linspace(0, 1, 41)
    edges = np.unique(np.quantile(xi, qs))
    if len(edges) < 5:
        pred = np.full_like(yi, np.median(yi))
    else:
        bin_id = np.clip(np.digitize(xi, edges[1:-1], right=True), 0, len(edges)-2)
        centers, meds = [], []
        for b in np.unique(bin_id):
            m = bin_id == b
            centers.append(float(np.median(xi[m])))
            meds.append(float(np.median(yi[m])))
        centers = np.asarray(centers)
        meds = np.asarray(meds)
        order = np.argsort(centers)
        pred = np.interp(xi, centers[order], meds[order])

    residual = yi - pred
    score[idx] = residual
    return score


def ranking_scores(stats: Dict[str, np.ndarray], eligible: np.ndarray) -> Dict[str, np.ndarray]:
    eps = 1e-12
    variance = stats["variance"].copy()
    fano = variance / np.maximum(stats["mean"], eps)
    residual = mean_adjusted_residual_score(stats["mean"], variance, eligible)

    for arr in (variance, fano, residual):
        arr[~eligible] = np.nan

    return {
        "variance": variance,
        "fano": fano,
        "hvg_residual": residual,
    }


def rank_indices(score: np.ndarray) -> np.ndarray:
    idx = np.flatnonzero(np.isfinite(score))
    return idx[np.argsort(score[idx])[::-1]]


def jaccard_top(a: np.ndarray, b: np.ndarray, m: int) -> float:
    aa = set(a[:m].tolist())
    bb = set(b[:m].tolist())
    if not aa and not bb:
        return np.nan
    return len(aa & bb) / len(aa | bb)


def effective_rank(eigvals: np.ndarray) -> float:
    vals = np.clip(np.asarray(eigvals, dtype=float), 0, None)
    s = vals.sum()
    if s <= 0:
        return 0.0
    p = vals / s
    p = p[p > 0]
    return float(np.exp(-np.sum(p * np.log(p))))


# ---------------------------- Main ---------------------------------

def main():
    args = parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    m_values = parse_m_values(args.m_values)

    print(f"[OPEN] {args.h5ad}")
    a = ad.read_h5ad(args.h5ad, backed="r")

    if args.donor_col not in a.obs.columns:
        raise KeyError(f"Missing donor column {args.donor_col!r}")

    donors = a.obs[args.donor_col].astype("string")
    donor_categories = pd.Index(pd.unique(donors))
    donor_map = {d: i for i, d in enumerate(donor_categories)}
    donor_codes_all = donors.map(donor_map).to_numpy(dtype=np.int32)

    matrix, matrix_name = choose_matrix(a, args.matrix_source)
    n_cells, n_genes = matrix.shape

    if matrix_name == "X":
        gene_id, gene_name = get_gene_names_from_var(a.var_names, a.var)
    else:
        gene_id, gene_name = get_gene_names_from_var(a.raw.var_names, a.raw.var)

    if len(gene_id) != n_genes:
        raise ValueError("Gene metadata length does not match selected matrix")

    count_like, inspection = inspect_count_like(matrix)
    print(f"[DATA] cells={n_cells:,} genes={n_genes:,} donors={len(donor_categories)}")
    print(f"[SOURCE] {matrix_name} | count_like={count_like} | inspection={inspection}")
    if count_like:
        print(f"[TRANSFORM] library-size normalize to {args.target_sum:g} + log1p")
    else:
        print("[TRANSFORM] matrix treated as already transformed; no additional normalization")

    nd = len(donor_categories)
    # float32 keeps donor sufficient-statistic memory modest; aggregation is robust enough here.
    sums = np.zeros((nd, n_genes), dtype=np.float32)
    sumsqs = np.zeros((nd, n_genes), dtype=np.float32)
    detects = np.zeros((nd, n_genes), dtype=np.float32)
    ncells = np.zeros(nd, dtype=np.int64)

    for start in range(0, n_cells, args.chunk_size):
        stop = min(start + args.chunk_size, n_cells)
        block = matrix[start:stop]
        Z, detected = transform_block(block, count_like, args.target_sum)
        aggregate_chunk_by_donor(
            Z, detected, donor_codes_all[start:stop], nd, n_genes,
            sums, sumsqs, detects, ncells
        )
        if start == 0 or stop == n_cells or (start // args.chunk_size) % 20 == 0:
            print(f"[ACCUMULATE] {stop:,}/{n_cells:,} cells")

    print(f"[DONORS] cells/donor min={ncells.min()} median={int(np.median(ncells))} max={ncells.max()}")

    full_stats = donor_balanced_stats(sums, sumsqs, detects, ncells)
    eligible = (
        (full_stats["donor_prevalence"] >= args.min_donor_prevalence)
        & (full_stats["detection"] >= args.min_balanced_detection)
        & np.isfinite(full_stats["variance"])
        & (full_stats["variance"] > 0)
    )
    print(f"[ELIGIBILITY] retained {eligible.sum():,}/{n_genes:,} genes")

    scores = ranking_scores(full_stats, eligible)
    rankings = {k: rank_indices(v) for k, v in scores.items()}

    stats_df = pd.DataFrame({
        "gene_id": gene_id,
        "gene_name": gene_name,
        "donor_prevalence": full_stats["donor_prevalence"],
        "balanced_detection_rate": full_stats["detection"],
        "balanced_mean": full_stats["mean"],
        "balanced_variance": full_stats["variance"],
        "variance_score": scores["variance"],
        "fano_score": scores["fano"],
        "hvg_residual_score": scores["hvg_residual"],
        "eligible": eligible,
    })
    stats_df.to_csv(out / "gene_statistics.csv", index=False)

    rank_rows = []
    max_rank_save = min(max(max(m_values), 2000), int(eligible.sum()))
    for method, order in rankings.items():
        for r, idx in enumerate(order[:max_rank_save], start=1):
            rank_rows.append({
                "method": method,
                "rank": r,
                "gene_index": int(idx),
                "gene_id": gene_id[idx],
                "gene_name": gene_name[idx],
                "score": float(scores[method][idx]),
            })
    pd.DataFrame(rank_rows).to_csv(out / "gene_rankings.csv", index=False)

    # Stability under donor subsampling
    rng = np.random.default_rng(args.random_state)
    n_sub = max(2, int(round(args.subject_fraction * nd)))
    stability_records = []

    for rep in range(args.n_resamples):
        ds = np.sort(rng.choice(nd, size=n_sub, replace=False))
        st = donor_balanced_stats(sums, sumsqs, detects, ncells, ds)
        elig_sub = (
            (st["donor_prevalence"] >= args.min_donor_prevalence)
            & (st["detection"] >= args.min_balanced_detection)
            & np.isfinite(st["variance"])
            & (st["variance"] > 0)
        )
        sc = ranking_scores(st, elig_sub)
        for method in rankings:
            rr = rank_indices(sc[method])
            for m in m_values:
                if m <= len(rankings[method]) and m <= len(rr):
                    stability_records.append({
                        "repeat": rep + 1,
                        "method": method,
                        "m": m,
                        "jaccard_vs_full": jaccard_top(rankings[method], rr, m),
                    })
        if (rep + 1) % 5 == 0 or rep == 0:
            print(f"[STABILITY] {rep+1}/{args.n_resamples} donor resamples")

    stab = pd.DataFrame(stability_records)
    stab_summary = (
        stab.groupby(["method", "m"], as_index=False)["jaccard_vs_full"]
        .agg(["mean", "median", "std", "min", "max"])
        .reset_index()
    )
    stab.to_csv(out / "ranking_stability_resamples.csv", index=False)
    stab_summary.to_csv(out / "ranking_stability.csv", index=False)

    # Choose method by mean stability averaged across requested m values.
    method_score = stab_summary.groupby("method")["mean"].mean().sort_values(ascending=False)
    best_method = str(method_score.index[0])
    print(f"[SELECTION] most stable method={best_method}")

    # Matrix-dimensionality QC on a sample of donors using the best ranking.
    # We reread only selected donor rows and top max(m) genes.
    qc_n = min(args.qc_donors, nd)
    qc_donor_idx = np.sort(rng.choice(nd, size=qc_n, replace=False))
    top_order = rankings[best_method]
    max_m = min(max(m_values), len(top_order))
    genes_max = np.sort(top_order[:max_m])
    gene_pos = {g: i for i, g in enumerate(genes_max)}

    # Collect selected-gene expression donor-by-donor, chunked.
    donor_blocks = {d: [] for d in qc_donor_idx.tolist()}
    qc_set = set(qc_donor_idx.tolist())

    for start in range(0, n_cells, args.chunk_size):
        stop = min(start + args.chunk_size, n_cells)
        codes = donor_codes_all[start:stop]
        if not np.isin(codes, qc_donor_idx).any():
            continue
        block = matrix[start:stop, genes_max]
        Z, _ = transform_block(block, count_like, args.target_sum)
        if sp.issparse(Z):
            Z = Z.toarray()
        for d in np.unique(codes):
            if int(d) in qc_set:
                donor_blocks[int(d)].append(np.asarray(Z[codes == d], dtype=np.float64))

    qc_rows = []
    for d in qc_donor_idx:
        Xd_max = np.vstack(donor_blocks[int(d)])
        # genes_max is sorted, while top_order is ranked. Map ranked genes into sorted columns.
        col_map = {g: j for j, g in enumerate(genes_max)}
        ranked_cols = np.array([col_map[g] for g in top_order[:max_m]], dtype=int)

        for m in m_values:
            if m > max_m:
                continue
            Xm = Xd_max[:, ranked_cols[:m]]
            # Remove residual constant columns only for QC; report how many.
            sd = Xm.std(axis=0, ddof=1)
            valid = np.isfinite(sd) & (sd > 1e-12)
            Xv = Xm[:, valid]
            if Xv.shape[1] < 2:
                continue
            C = np.corrcoef(Xv, rowvar=False)
            C = 0.5 * (C + C.T)
            np.fill_diagonal(C, 1.0)
            eig = np.linalg.eigvalsh(C)
            qc_rows.append({
                "donor_id": str(donor_categories[d]),
                "n_cells": int(Xm.shape[0]),
                "m_requested": int(m),
                "m_nonconstant": int(Xv.shape[1]),
                "matrix_rank": int(np.linalg.matrix_rank(C, tol=1e-10)),
                "effective_rank": effective_rank(eig),
                "min_eigenvalue": float(eig[0]),
                "max_eigenvalue": float(eig[-1]),
            })

    qc = pd.DataFrame(qc_rows)
    qc.to_csv(out / "matrix_dimensionality_qc.csv", index=False)

    # Plots
    fig, ax = plt.subplots(figsize=(7, 5))
    mask = eligible & (full_stats["mean"] > 0) & (full_stats["variance"] > 0)
    ax.scatter(full_stats["mean"][mask], full_stats["variance"][mask], s=4, alpha=0.25)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Donor-balanced mean expression")
    ax.set_ylabel("Donor-balanced variance")
    ax.set_title("SLE unsupervised mean-variance structure")
    fig.tight_layout()
    fig.savefig(out / "01_mean_variance.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    for method, g in stab_summary.groupby("method"):
        ax.plot(g["m"], g["mean"], marker="o", label=method)
    ax.set_xscale("log", base=2)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Top-m genes")
    ax.set_ylabel("Mean Jaccard vs full-data ranking")
    ax.set_title("Donor-resampling stability")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "02_stability_vs_m.png", dpi=180)
    plt.close(fig)

    if not qc.empty:
        qsum = qc.groupby("m_requested")["effective_rank"].agg(["median", "min", "max"]).reset_index()
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.plot(qsum["m_requested"], qsum["median"], marker="o")
        ax.fill_between(qsum["m_requested"], qsum["min"], qsum["max"], alpha=0.2)
        ax.set_xscale("log", base=2)
        ax.set_xlabel("Requested number of genes")
        ax.set_ylabel("Correlation-matrix effective rank")
        ax.set_title(f"Matrix dimensionality QC ({best_method})")
        fig.tight_layout()
        fig.savefig(out / "03_matrix_effective_rank_vs_m.png", dpi=180)
        plt.close(fig)

    # Recommendation: choose the smallest m with mean stability >= 0.75 for best method,
    # otherwise choose m with highest mean stability. Final scientific choice remains explicit.
    bs = stab_summary[stab_summary["method"] == best_method].sort_values("m")
    stable = bs[bs["mean"] >= 0.75]
    if len(stable):
        suggested_m = int(stable.iloc[0]["m"])
        reason = "smallest tested m with mean donor-resampling Jaccard >= 0.75"
    else:
        row = bs.sort_values("mean", ascending=False).iloc[0]
        suggested_m = int(row["m"])
        reason = "tested m with highest mean donor-resampling Jaccard (threshold 0.75 not reached)"

    rec = [
        "SLE UNSUPERVISED GENE-SELECTION AUDIT",
        f"matrix_source={matrix_name}",
        f"count_like={count_like}",
        f"n_cells={n_cells}",
        f"n_genes={n_genes}",
        f"n_donors={nd}",
        f"eligible_genes={int(eligible.sum())}",
        f"min_donor_prevalence={args.min_donor_prevalence}",
        f"min_balanced_detection={args.min_balanced_detection}",
        f"most_stable_method={best_method}",
        f"suggested_m={suggested_m}",
        f"suggestion_basis={reason}",
        "",
        "IMPORTANT: disease labels were not used in any ranking, filtering,",
        "stability calculation, or dimensionality diagnostic.",
        "The suggested m is a data-driven starting point, not an automatic",
        "scientific decision; inspect ranking_stability.csv and",
        "matrix_dimensionality_qc.csv before fixing the final representation.",
    ]
    (out / "selection_recommendation.txt").write_text("\n".join(rec), encoding="utf-8")

    print("\n".join(f"[RESULT] {x}" for x in rec if x))
    print(f"[DONE] outputs -> {out}")

    if a.file is not None:
        a.file.close()


if __name__ == "__main__":
    main()
