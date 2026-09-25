#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PEMS-SF unsupervised sensor-selection and dimensionality audit.

Scientific design
-----------------
- PEMS_train + PEMS_test are merged only for the UNSUPERVISED representation audit.
- PEMS_*labels are never read.
- Each day is expected to contain a 963 x 144 sensor-by-time matrix.
- Sensor ranking is label-free and day-balanced.
- Candidate rankings:
    1) temporal_variance: mean within-day temporal variance, equal weight per day
    2) interday_mean_variance: variance across days of each sensor's daily mean
    3) combined_z: sum of z-scored versions of the two rankings
- Stability is assessed by repeated 80% day resampling.
- Dimensionality is assessed from day-specific sensor correlation matrices.

Outputs
-------
audit_summary.txt
sensor_statistics.csv
sensor_rankings.csv
ranking_stability.csv
dimensionality_sweep_by_day.csv
dimensionality_sweep_summary.csv
selection_recommendation.txt
"""

from __future__ import annotations
import argparse
import re
from pathlib import Path
import numpy as np
import pandas as pd

N_SENSORS = 963
N_TIME = 144


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--m-values", default="32,48,64,80,96,104,108,112,120,128")
    p.add_argument("--n-resamples", type=int, default=30)
    p.add_argument("--day-fraction", type=float, default=0.8)
    p.add_argument("--random-state", type=int, default=42)
    p.add_argument("--q-threshold", type=float, default=0.75)
    p.add_argument("--stability-threshold", type=float, default=0.90)
    p.add_argument("--condition-threshold", type=float, default=1e4)
    p.add_argument("--rank-tol", type=float, default=1e-10)
    p.add_argument("--eig-tol", type=float, default=1e-8)
    return p.parse_args()


_float_re = re.compile(r"[-+]?(?:\d*\.\d+|\d+\.?)(?:[eE][-+]?\d+)?")


def parse_pems_file(path: Path) -> np.ndarray:
    days = []
    print(f"[OPEN] {path}")
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue

            # First try MATLAB row separators: 963 rows separated by ';'.
            clean = line.strip()
            if clean.startswith("["):
                clean = clean[1:]
            if clean.endswith("]"):
                clean = clean[:-1]

            row_parts = clean.split(";")
            if len(row_parts) == N_SENSORS:
                rows = []
                ok = True
                for rp in row_parts:
                    vals = np.fromstring(rp, sep=" ", dtype=np.float64)
                    if vals.size != N_TIME:
                        # fallback to regex in case of odd delimiters
                        vals = np.array([float(x) for x in _float_re.findall(rp)], dtype=np.float64)
                    if vals.size != N_TIME:
                        ok = False
                        break
                    rows.append(vals)
                if ok:
                    day = np.vstack(rows)
                    days.append(day.astype(np.float32))
                    continue

            # Fallback: extract all numeric values from a line and reshape.
            vals = np.array([float(x) for x in _float_re.findall(clean)], dtype=np.float64)
            expected = N_SENSORS * N_TIME
            if vals.size != expected:
                raise ValueError(
                    f"{path.name}: line {line_no} contains {vals.size} numeric values; "
                    f"expected {expected} (=963x144). "
                    f"Semicolon-separated rows detected={len(row_parts)}."
                )
            day = vals.reshape(N_SENSORS, N_TIME)
            days.append(day.astype(np.float32))

            if len(days) == 1:
                print(f"[PARSE] first day shape={day.shape}, min={day.min():.6g}, max={day.max():.6g}")

    if not days:
        raise ValueError(f"No days parsed from {path}")
    arr = np.stack(days, axis=0)
    print(f"[PARSE] {path.name}: days={arr.shape[0]} shape/day={arr.shape[1:]}")
    return arr


def zscore(v):
    v = np.asarray(v, float)
    s = np.nanstd(v)
    return (v - np.nanmean(v)) / s if s > 0 else np.zeros_like(v)


def rank_desc(score):
    order = np.argsort(-score, kind="mergesort")
    ranks = np.empty_like(order)
    ranks[order] = np.arange(1, len(score)+1)
    return order, ranks


def jaccard(a, b):
    a, b = set(map(int, a)), set(map(int, b))
    return len(a & b) / len(a | b) if (a or b) else 1.0


def effective_rank(e):
    e = np.clip(np.asarray(e, float), 0, None)
    s = e.sum()
    if s <= 0:
        return 0.0
    p = e / s
    p = p[p > 0]
    return float(np.exp(-np.sum(p * np.log(p))))


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    mvals = sorted({int(x) for x in args.m_values.split(",")})
    max_m = max(mvals)

    # IMPORTANT: labels are deliberately not opened.
    train = parse_pems_file(args.data_dir / "PEMS_train")
    test = parse_pems_file(args.data_dir / "PEMS_test")
    X = np.concatenate([train, test], axis=0)
    del train, test

    n_days, n_sensors, n_time = X.shape
    if (n_sensors, n_time) != (N_SENSORS, N_TIME):
        raise ValueError(X.shape)
    print(f"[DATA] days={n_days} sensors={n_sensors} timepoints={n_time}")
    if n_days != 440:
        print(f"[WARN] Expected 440 total days, found {n_days}")

    finite_fraction = np.isfinite(X).mean(axis=(0,2))
    constant_day_fraction = np.mean(np.nanstd(X, axis=2) <= 1e-12, axis=0)

    # Equal weight per day.
    within_day_var = np.nanvar(X, axis=2, ddof=1)            # day x sensor
    temporal_variance = np.nanmean(within_day_var, axis=0)  # sensor

    daily_mean = np.nanmean(X, axis=2)                       # day x sensor
    interday_mean_variance = np.nanvar(daily_mean, axis=0, ddof=1)

    combined_z = zscore(temporal_variance) + zscore(interday_mean_variance)

    stats = pd.DataFrame({
        "sensor_index_0based": np.arange(n_sensors),
        "sensor_index_1based": np.arange(1, n_sensors+1),
        "finite_fraction": finite_fraction,
        "constant_day_fraction": constant_day_fraction,
        "temporal_variance_day_balanced": temporal_variance,
        "interday_daily_mean_variance": interday_mean_variance,
        "combined_z_score": combined_z,
    })
    stats.to_csv(args.output_dir / "sensor_statistics.csv", index=False)

    methods = {
        "temporal_variance": temporal_variance,
        "interday_mean_variance": interday_mean_variance,
        "combined_z": combined_z,
    }

    ranking_rows = []
    full_orders = {}
    for method, score in methods.items():
        order, ranks = rank_desc(score)
        full_orders[method] = order
        for j in range(n_sensors):
            ranking_rows.append({
                "method": method,
                "sensor_index_0based": j,
                "sensor_index_1based": j+1,
                "score": float(score[j]),
                "rank": int(ranks[j]),
            })
    pd.DataFrame(ranking_rows).to_csv(args.output_dir / "sensor_rankings.csv", index=False)

    # Stability: recompute the ranking statistic after resampling DAYS.
    rng = np.random.default_rng(args.random_state)
    stab_rows = []
    n_sub = max(2, int(round(args.day_fraction * n_days)))

    for r in range(args.n_resamples):
        idx = rng.choice(n_days, size=n_sub, replace=False)
        Xr = X[idx]

        temp_r = np.nanmean(np.nanvar(Xr, axis=2, ddof=1), axis=0)
        mean_r = np.nanvar(np.nanmean(Xr, axis=2), axis=0, ddof=1)
        comb_r = zscore(temp_r) + zscore(mean_r)
        rmethods = {
            "temporal_variance": temp_r,
            "interday_mean_variance": mean_r,
            "combined_z": comb_r,
        }

        for method, score_r in rmethods.items():
            order_r, _ = rank_desc(score_r)
            ref = full_orders[method]
            for m in mvals:
                stab_rows.append({
                    "resample": r,
                    "method": method,
                    "m": m,
                    "jaccard_vs_full": jaccard(ref[:m], order_r[:m]),
                })
        if (r+1) % 5 == 0 or r == 0:
            print(f"[STABILITY] {r+1}/{args.n_resamples}")

    stab_long = pd.DataFrame(stab_rows)
    stab_summary = (stab_long.groupby(["method","m"])["jaccard_vs_full"]
                    .agg(["mean","median","std","min","max"]).reset_index())
    stab_summary.to_csv(args.output_dir / "ranking_stability.csv", index=False)

    # Choose the most stable method by average Jaccard over m.
    method_stability = stab_summary.groupby("method")["mean"].mean().sort_values(ascending=False)
    best_method = str(method_stability.index[0])
    print(f"[RANKING] most_stable_method={best_method}")
    order = full_orders[best_method]

    # Dimensionality sweep on ALL 440 days.
    detail = []
    for d in range(n_days):
        day = X[d]
        for m in mvals:
            sel = order[:m]
            Z = np.asarray(day[sel, :], dtype=np.float64).T  # time x sensors
            sd = Z.std(axis=0, ddof=1)
            valid = np.isfinite(sd) & (sd > 1e-12)
            Zv = Z[:, valid]

            if Zv.shape[1] < 2:
                detail.append({
                    "day": d, "m": m, "m_nonconstant": Zv.shape[1],
                    "rank": 0, "effective_rank": np.nan,
                    "min_eigenvalue": np.nan, "max_eigenvalue": np.nan,
                    "condition_number": np.inf
                })
                continue

            C = np.corrcoef(Zv, rowvar=False)
            C = 0.5 * (C + C.T)
            np.fill_diagonal(C, 1.0)
            eig = np.linalg.eigvalsh(C)
            pos = eig[eig > args.rank_tol]
            cond = float(eig[-1] / pos[0]) if len(pos) else np.inf

            detail.append({
                "day": d,
                "m": m,
                "m_nonconstant": int(Zv.shape[1]),
                "rank": int(np.sum(eig > args.rank_tol)),
                "effective_rank": effective_rank(eig),
                "min_eigenvalue": float(eig[0]),
                "max_eigenvalue": float(eig[-1]),
                "condition_number": cond,
            })
        if (d+1) % 40 == 0 or d == 0 or d+1 == n_days:
            print(f"[DIM] days {d+1}/{n_days}")

    detail = pd.DataFrame(detail)
    detail.to_csv(args.output_dir / "dimensionality_sweep_by_day.csv", index=False)

    rows = []
    best_stab = stab_summary[stab_summary["method"] == best_method].set_index("m")["mean"].to_dict()
    for m, g in detail.groupby("m"):
        q = m / n_time
        nonconstant_all = bool(np.all(g["m_nonconstant"].to_numpy() == m))
        full_rank_all = bool(np.all(g["rank"].to_numpy() == m))
        min_eig = float(np.nanmin(g["min_eigenvalue"]))
        cond_max = float(np.nanmax(g["condition_number"]))
        stability = float(best_stab.get(m, np.nan))
        passes = (
            q <= args.q_threshold
            and stability >= args.stability_threshold
            and nonconstant_all
            and full_rank_all
            and min_eig >= -args.eig_tol
            and cond_max <= args.condition_threshold
        )
        rows.append({
            "m": int(m),
            "n_timepoints": n_time,
            "q": q,
            "ranking_method": best_method,
            "stability_mean_jaccard": stability,
            "nonconstant_all_days": nonconstant_all,
            "full_rank_all_days": full_rank_all,
            "effective_rank_median": float(np.nanmedian(g["effective_rank"])),
            "effective_rank_fraction_median": float(np.nanmedian(g["effective_rank"] / m)),
            "min_eigenvalue_worst": min_eig,
            "condition_number_median": float(np.nanmedian(g["condition_number"])),
            "condition_number_max": cond_max,
            "passes_prespecified_rules": passes,
        })

    summary = pd.DataFrame(rows).sort_values("m")
    summary.to_csv(args.output_dir / "dimensionality_sweep_summary.csv", index=False)

    passing = summary[summary["passes_prespecified_rules"]]
    rec = int(passing["m"].max()) if len(passing) else None

    audit = [
        "PEMS-SF UNSUPERVISED SENSOR-SELECTION AUDIT",
        f"n_days={n_days}",
        f"n_sensors={n_sensors}",
        f"n_timepoints={n_time}",
        f"most_stable_method={best_method}",
        f"q_threshold={args.q_threshold}",
        f"stability_threshold={args.stability_threshold}",
        f"condition_number_threshold={args.condition_threshold}",
        "IMPORTANT: PEMS_trainlabels and PEMS_testlabels were not read.",
    ]
    (args.output_dir / "audit_summary.txt").write_text("\n".join(audit), encoding="utf-8")

    recommendation = audit + [
        "",
        f"recommended_m={rec if rec is not None else 'NONE'}",
        "Selection rule: largest tested m satisfying simultaneously:",
        "1) q=m/144 <= q_threshold;",
        "2) mean ranking Jaccard >= stability_threshold;",
        "3) all selected sensors nonconstant in every day;",
        "4) full-rank day-level correlation matrices;",
        "5) PSD within numerical tolerance;",
        "6) max condition number <= threshold.",
    ]
    (args.output_dir / "selection_recommendation.txt").write_text(
        "\n".join(recommendation), encoding="utf-8"
    )

    print("\n[RESULT]")
    print("\n".join(recommendation))
    print("\n[DIMENSIONALITY SUMMARY]")
    print(summary.to_string(index=False))
    print(f"\n[DONE] outputs -> {args.output_dir}")


if __name__ == "__main__":
    main()
