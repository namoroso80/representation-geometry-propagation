#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PEMS-SF exact low-rank Schatten benchmark.

Goal
----
Benchmark exact pairwise Schatten distances for the full valid sensor representation,
without materializing or diagonalizing 958 x 958 pairwise difference matrices.

No labels are read.

Mathematics
-----------
For each day s, after retaining sensors nonconstant in every day, let Z_s be the
sensor x time matrix whose sensor profiles are centered and standardized over time.

The correlation density operator is

    rho_s = Z_s Z_s^T / ((T-1) * n_valid_sensors) = B_s B_s^T,

with B_s = Z_s / sqrt((T-1) * n_valid_sensors).

Because each row of Z_s is centered, rank(rho_s) <= T-1 = 143.

For a pair (s,t),

    Delta = rho_s - rho_t = U J U^T,
    U = [B_s, B_t],
    J = diag(I, -I).

Let G = U^T U. The nonzero eigenvalues of Delta are exactly the eigenvalues of

    H = G^(1/2) J G^(1/2),

restricted to the positive-eigenvalue subspace of G. H is symmetric and has
dimension at most 2*(T-1) = 286.

This benchmark:
- parses PEMS_train + PEMS_test;
- removes only sensors constant in at least one day;
- constructs exact low-rank factors for all 440 days;
- samples random subject pairs;
- computes all requested Schatten-p distances from ONE reduced eigensystem/pair;
- measures pair throughput and extrapolates full 96,580-pair ETA;
- optionally validates a few pairs against dense 958 x 958 eigendecomposition.

Outputs
-------
benchmark_pairs.csv
benchmark_summary.txt
factor_qc.csv
"""

from __future__ import annotations
import argparse
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.linalg import eigh
from tqdm.auto import tqdm

N_SENSORS = 963
N_TIME = 144
P_DEFAULT = [1, 1.25, 1.5, 1.75, 2, 3, 4, 8, 16]
_float_re = re.compile(r"[-+]?(?:\d*\.\d+|\d+\.?)(?:[eE][-+]?\d+)?")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--n-pairs", type=int, default=300)
    p.add_argument("--n-dense-validation", type=int, default=3)
    p.add_argument("--random-state", type=int, default=42)
    p.add_argument("--p-values", nargs="+", type=float, default=P_DEFAULT)
    p.add_argument("--gram-tol", type=float, default=1e-11)
    return p.parse_args()


def parse_pems_file(path: Path) -> np.ndarray:
    days = []
    print(f"[OPEN] {path}")
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            s = line.strip()
            if s.startswith("["):
                s = s[1:]
            if s.endswith("]"):
                s = s[:-1]

            parts = s.split(";")
            if len(parts) == N_SENSORS:
                rows = []
                good = True
                for part in parts:
                    v = np.fromstring(part, sep=" ", dtype=np.float64)
                    if v.size != N_TIME:
                        v = np.array([float(x) for x in _float_re.findall(part)], dtype=np.float64)
                    if v.size != N_TIME:
                        good = False
                        break
                    rows.append(v)
                if good:
                    days.append(np.vstack(rows).astype(np.float32))
                    continue

            vals = np.array([float(x) for x in _float_re.findall(s)], dtype=np.float64)
            if vals.size != N_SENSORS * N_TIME:
                raise ValueError(
                    f"{path.name}, line {line_no}: {vals.size} numeric values; "
                    f"expected {N_SENSORS*N_TIME}"
                )
            days.append(vals.reshape(N_SENSORS, N_TIME).astype(np.float32))

    arr = np.stack(days)
    print(f"[PARSE] {path.name}: {arr.shape}")
    return arr


def build_factors(X: np.ndarray):
    # Globally valid = nonconstant and finite in every day.
    sensor_sd = X.astype(np.float64).std(axis=2, ddof=1)
    valid = np.all(np.isfinite(sensor_sd) & (sensor_sd > 1e-12), axis=0)
    idx = np.flatnonzero(valid)
    n = len(idx)
    print(f"[FILTER] globally valid sensors: {n}/{X.shape[1]}")

    factors = []
    qc = []

    denom = np.sqrt((N_TIME - 1) * n)

    for d, day in enumerate(X):
        Y = day[idx].astype(np.float64)              # sensors x time
        mu = Y.mean(axis=1, keepdims=True)
        sd = Y.std(axis=1, ddof=1, keepdims=True)
        Z = (Y - mu) / sd
        B = Z / denom                               # rho = B B^T

        # Drop the exact centering null direction via thin SVD.
        # This also gives a numerically stable factor with <=143 columns.
        U, s, _ = np.linalg.svd(B, full_matrices=False)
        keep = s > 1e-12
        F = U[:, keep] * s[keep][None, :]           # F F^T = B B^T exactly numerically
        factors.append(F.astype(np.float64, copy=False))

        qc.append({
            "day": d,
            "rank": int(keep.sum()),
            "trace_from_singular_values": float(np.sum(s[keep]**2)),
            "largest_singular_value": float(s[0]),
            "smallest_retained_singular_value": float(s[keep][-1]),
        })

        if d == 0 or (d+1) % 40 == 0 or d+1 == len(X):
            print(f"[FACTORS] {d+1}/{len(X)}")

    return idx, factors, pd.DataFrame(qc)


def exact_reduced_eigenvalues(Fs: np.ndarray, Ft: np.ndarray, gram_tol: float):
    rs = Fs.shape[1]
    rt = Ft.shape[1]

    # Gram of U=[Fs,Ft], built blockwise.
    Gss = Fs.T @ Fs
    Gtt = Ft.T @ Ft
    Gst = Fs.T @ Ft
    G = np.block([[Gss, Gst], [Gst.T, Gtt]])
    G = 0.5 * (G + G.T)

    # Positive subspace of G.
    g, V = eigh(G, check_finite=False, overwrite_a=True)
    scale = max(float(g[-1]), 1.0)
    keep = g > gram_tol * scale
    gp = g[keep]
    Vp = V[:, keep]

    # G^(1/2) represented in the retained eigenspace.
    # H_red = sqrt(g) * (V^T J V) * sqrt(g)
    signs = np.concatenate([np.ones(rs), -np.ones(rt)])
    JV = signs[:, None] * Vp
    K = Vp.T @ JV
    sg = np.sqrt(gp)
    H = (sg[:, None] * K) * sg[None, :]
    H = 0.5 * (H + H.T)

    lam = eigh(H, eigvals_only=True, check_finite=False, overwrite_a=True)
    return lam


def schatten_from_eigs(lam: np.ndarray, p_values):
    a = np.abs(lam)
    out = {}
    for p in p_values:
        if np.isinf(p):
            out[p] = float(a.max())
        else:
            out[p] = float(np.sum(a**p)**(1.0/p))
    return out


def dense_eigenvalues(Fs, Ft):
    D = Fs @ Fs.T - Ft @ Ft.T
    D = 0.5 * (D + D.T)
    return eigh(D, eigvals_only=True, check_finite=False, overwrite_a=True)


def main():
    a = parse_args()
    a.output_dir.mkdir(parents=True, exist_ok=True)

    print("[LOAD] labels are NOT read")
    X = np.concatenate([
        parse_pems_file(a.data_dir / "PEMS_train"),
        parse_pems_file(a.data_dir / "PEMS_test")
    ], axis=0)
    print(f"[DATA] days={X.shape[0]}, sensors={X.shape[1]}, time={X.shape[2]}")

    valid_idx, factors, qc = build_factors(X)
    qc.to_csv(a.output_dir / "factor_qc.csv", index=False)

    n = len(factors)
    all_pairs = [(i, j) for i in range(n-1) for j in range(i+1, n)]
    rng = np.random.default_rng(a.random_state)
    take = min(a.n_pairs, len(all_pairs))
    sel = rng.choice(len(all_pairs), size=take, replace=False)
    pairs = [all_pairs[k] for k in sel]

    rows = []
    print(f"[BENCH] exact reduced eigensystems on {take} random pairs")
    t0 = time.perf_counter()

    for q, (i, j) in enumerate(tqdm(pairs, desc="low-rank pairs")):
        tp = time.perf_counter()
        lam = exact_reduced_eigenvalues(factors[i], factors[j], a.gram_tol)
        vals = schatten_from_eigs(lam, a.p_values)
        elapsed = time.perf_counter() - tp

        row = {
            "pair_index": q,
            "day_i": i,
            "day_j": j,
            "rank_i": factors[i].shape[1],
            "rank_j": factors[j].shape[1],
            "reduced_nonzero_eigs": len(lam),
            "seconds": elapsed,
        }
        for p, v in vals.items():
            row[f"d_p_{p:g}"] = v
        rows.append(row)

    total = time.perf_counter() - t0
    df = pd.DataFrame(rows)
    df.to_csv(a.output_dir / "benchmark_pairs.csv", index=False)

    # Dense validation on first few benchmark pairs.
    nval = min(a.n_dense_validation, take)
    val_rows = []
    if nval > 0:
        print(f"[VALIDATE] dense 958x958 eigendecomposition for {nval} pairs")
    for q in range(nval):
        i, j = pairs[q]

        t1 = time.perf_counter()
        lr = exact_reduced_eigenvalues(factors[i], factors[j], a.gram_tol)
        low_t = time.perf_counter() - t1

        t2 = time.perf_counter()
        de = dense_eigenvalues(factors[i], factors[j])
        dense_t = time.perf_counter() - t2

        lv = schatten_from_eigs(lr, a.p_values)
        dv = schatten_from_eigs(de, a.p_values)

        for p in a.p_values:
            denom = max(abs(dv[p]), 1e-15)
            val_rows.append({
                "day_i": i,
                "day_j": j,
                "p": p,
                "low_rank": lv[p],
                "dense": dv[p],
                "relative_error": abs(lv[p] - dv[p]) / denom,
                "low_rank_seconds": low_t,
                "dense_seconds": dense_t,
            })

    val = pd.DataFrame(val_rows)
    val.to_csv(a.output_dir / "dense_validation.csv", index=False)

    pair_rate = take / total
    full_pairs = n * (n - 1) // 2
    eta_s = full_pairs / pair_rate
    eta_h = eta_s / 3600

    med_s = float(df["seconds"].median())
    q25_s = float(df["seconds"].quantile(.25))
    q75_s = float(df["seconds"].quantile(.75))
    max_rel = float(val["relative_error"].max()) if len(val) else np.nan
    dense_med = float(val["dense_seconds"].median()) if len(val) else np.nan
    low_val_med = float(val["low_rank_seconds"].median()) if len(val) else np.nan

    report = [
        "PEMS-SF EXACT LOW-RANK SCHATTEN BENCHMARK",
        "Labels were not read.",
        f"n_days={n}",
        f"valid_sensors={len(valid_idx)}/{N_SENSORS}",
        f"factor_rank_min={qc['rank'].min()}",
        f"factor_rank_median={qc['rank'].median():.1f}",
        f"factor_rank_max={qc['rank'].max()}",
        f"benchmark_pairs={take}",
        f"median_seconds_per_pair={med_s:.6f}",
        f"iqr_seconds_per_pair=[{q25_s:.6f},{q75_s:.6f}]",
        f"throughput_pairs_per_second={pair_rate:.3f}",
        f"full_unique_pairs={full_pairs}",
        f"estimated_full_geometry_hours_single_process={eta_h:.3f}",
        f"dense_validation_pairs={nval}",
        f"max_relative_error_lowrank_vs_dense={max_rel:.3e}",
        f"median_lowrank_validation_seconds={low_val_med:.6f}",
        f"median_dense_validation_seconds={dense_med:.6f}",
    ]

    (a.output_dir / "benchmark_summary.txt").write_text("\n".join(report), encoding="utf-8")
    print("\n" + "\n".join("[RESULT] " + x for x in report))
    print(f"[DONE] {a.output_dir}")


if __name__ == "__main__":
    main()
