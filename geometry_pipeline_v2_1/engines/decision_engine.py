#!/usr/bin/env python3
# %%
"""AD Schatten-p decision-space analysis v1.

Post-hoc analysis of the decision functions saved by
ad_schatten_gamma_dynamics_v4.py. No classifier is refit.

Scientific role
---------------
Study the propagation K_p -> f_p after the geometry/kernel analysis.
Primary inference uses held-out (OOF) decision scores only. Training scores
are reconstructed exclusively as a diagnostic.

For every subject and p, repeated nested CV provides one held-out score in
each of the 20 outer-CV repetitions. The primary subject trajectory is the
median of these 20 unbiased scores; all repeat-level scores are retained.

Outputs
-------
00_oof_scores_long.csv
01_subject_oof_summary.csv
02_spaghetti_oof_subject_trajectories.png/pdf
03_spaghetti_oof_delta_from_p1.png/pdf
04_oof_score_distributions_by_class.png/pdf
05_oof_subject_by_p_heatmap.png/pdf
06_oof_score_concordance_vs_p1.png/pdf
07_oof_label_agreement_vs_p1.png/pdf
08_train_vs_oof_class_separation.png/pdf
09_decision_summary_by_p.csv
10_subject_crossings_summary.csv

The script deliberately remains exploratory: no decision susceptibility is
introduced at this stage.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import roc_auc_score

P_VALUES = np.array([1.0, 1.25, 1.5, 1.75, 2.0, 3.0, 4.0, 8.0, 16.0], dtype=float)
N_REPEATS = 20
N_SPLITS = 5
REFERENCE_P = 1.0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Explore OOF decision trajectories along Schatten p.")
    p.add_argument("--results-source", required=True, type=Path,
                   help="v4 result directory containing fold checkpoints.")
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--positive-name", default="AD")
    p.add_argument("--negative-name", default="NEGATIVE")
    p.add_argument("--p-values", nargs="+", type=float, default=P_VALUES.tolist())
    p.add_argument("--reference-p", type=float, default=1.0)
    p.add_argument("--n-repeats", type=int, default=20)
    p.add_argument("--n-splits", type=int, default=5)
    p.add_argument("--dpi", default=220, type=int)
    return p.parse_args()


def checkpoint_path(root: Path, p_index: int, fold_id: int) -> Path:
    candidates = [
        root / "checkpoints" / f"p{p_index:02d}_fold{fold_id:03d}.npz",
        root / "checkpoints" / f"p{p_index:02d}_fold_{fold_id:03d}.npz",
        root / f"p{p_index:02d}_fold{fold_id:03d}.npz",
    ]
    for c in candidates:
        if c.exists():
            return c
    # Robust fallback: exact p-index and fold search under source.
    hits = sorted(root.rglob(f"*p{p_index:02d}*fold*{fold_id:03d}*.npz"))
    if hits:
        return hits[0]
    raise FileNotFoundError(
        f"Cannot locate checkpoint for p_index={p_index}, fold={fold_id} under {root}"
    )


def load_oof_and_train(source: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    oof_rows: list[dict] = []
    train_rows: list[dict] = []

    for p_index, p in enumerate(P_VALUES):
        for fold_id in range(1, N_REPEATS * N_SPLITS + 1):
            path = checkpoint_path(source, p_index, fold_id)
            z = np.load(path, allow_pickle=False)
            repeat = (fold_id - 1) // N_SPLITS + 1
            fold_within_repeat = (fold_id - 1) % N_SPLITS + 1

            te = z["test_idx"].astype(int)
            yte = z["y_test"].astype(int)
            ste = z["test_scores"].astype(float)
            pte = z["test_pred"].astype(int)
            for idx, y, score, pred in zip(te, yte, ste, pte):
                oof_rows.append({
                    "p": float(p), "p_index": p_index,
                    "repeat": repeat, "fold_within_repeat": fold_within_repeat,
                    "outer_fold": fold_id, "subject_idx": int(idx), "y": int(y),
                    "score": float(score), "prediction": int(pred),
                })

            tr = z["train_idx"].astype(int)
            ytr = z["y_train"].astype(int)
            strn = z["train_scores"].astype(float)
            ptrn = z["train_pred"].astype(int)
            for idx, y, score, pred in zip(tr, ytr, strn, ptrn):
                train_rows.append({
                    "p": float(p), "p_index": p_index,
                    "repeat": repeat, "fold_within_repeat": fold_within_repeat,
                    "outer_fold": fold_id, "subject_idx": int(idx), "y": int(y),
                    "score": float(score), "prediction": int(pred),
                })

    oof = pd.DataFrame(oof_rows).sort_values(["p", "repeat", "subject_idx"]).reset_index(drop=True)
    train = pd.DataFrame(train_rows).sort_values(["p", "repeat", "subject_idx"]).reset_index(drop=True)

    # Every subject must be held out exactly once per repeat and p.
    counts = oof.groupby(["p", "repeat", "subject_idx"]).size()
    if not np.all(counts.to_numpy() == 1):
        raise RuntimeError("OOF provenance error: a subject is not held out exactly once per (p, repeat).")
    return oof, train


def subject_summary(oof: pd.DataFrame) -> pd.DataFrame:
    g = oof.groupby(["p", "subject_idx", "y"], sort=True)["score"]
    out = g.agg(
        score_median="median",
        score_mean="mean",
        score_sd="std",
        score_q25=lambda x: np.quantile(x, 0.25),
        score_q75=lambda x: np.quantile(x, 0.75),
        n_repeats="size",
    ).reset_index()
    out["pred_from_median_score"] = (out["score_median"] >= 0).astype(int)
    return out


def repeat_class_summary(df: pd.DataFrame, source: str) -> pd.DataFrame:
    rows = []
    for (p, rep), g in df.groupby(["p", "repeat"], sort=True):
        for y in (0, 1):
            x = g.loc[g.y == y, "score"].to_numpy(float)
            rows.append({
                "source": source, "p": float(p), "repeat": int(rep), "y": y,
                "median": float(np.median(x)),
                "q25": float(np.quantile(x, 0.25)),
                "q75": float(np.quantile(x, 0.75)),
                "mean": float(np.mean(x)), "sd": float(np.std(x, ddof=1)),
            })
    return pd.DataFrame(rows)


def safe_corr(x: np.ndarray, y: np.ndarray, method: str) -> float:
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return np.nan
    return float(spearmanr(x, y).statistic if method == "spearman" else pearsonr(x, y).statistic)


def decision_summary(oof: pd.DataFrame, subj: pd.DataFrame) -> pd.DataFrame:
    rows = []
    ref_subj = subj[np.isclose(subj.p, REFERENCE_P)].set_index("subject_idx")
    for p in P_VALUES:
        sg = subj[np.isclose(subj.p, p)].set_index("subject_idx")
        common = ref_subj.index.intersection(sg.index)
        a = ref_subj.loc[common, "score_median"].to_numpy(float)
        b = sg.loc[common, "score_median"].to_numpy(float)
        pa = (a >= 0).astype(int)
        pb = (b >= 0).astype(int)

        # Pooled OOF AUC separately within each repeat; summarize across 20 repeats.
        aucs = []
        for rep, rg in oof[np.isclose(oof.p, p)].groupby("repeat"):
            aucs.append(roc_auc_score(rg.y.to_numpy(int), rg.score.to_numpy(float)))

        rows.append({
            "p": float(p),
            "spearman_vs_p1_subject_median": safe_corr(a, b, "spearman"),
            "pearson_vs_p1_subject_median": safe_corr(a, b, "pearson"),
            "label_agreement_vs_p1_subject_median": float(np.mean(pa == pb)),
            "n_label_changes_vs_p1_subject_median": int(np.sum(pa != pb)),
            "median_abs_delta_score_vs_p1": float(np.median(np.abs(b - a))),
            "oof_auc_repeat_median": float(np.median(aucs)),
            "oof_auc_repeat_q25": float(np.quantile(aucs, 0.25)),
            "oof_auc_repeat_q75": float(np.quantile(aucs, 0.75)),
        })
    return pd.DataFrame(rows)


def crossings_summary(subj: pd.DataFrame) -> pd.DataFrame:
    rows = []
    wide = subj.pivot(index=["subject_idx", "y"], columns="p", values="score_median")
    wide = wide.reindex(columns=P_VALUES)
    for (idx, y), r in wide.iterrows():
        s = r.to_numpy(float)
        signs = np.signbit(s)
        crossings = int(np.sum(signs[1:] != signs[:-1]))
        first_cross = np.nan
        if crossings:
            j = int(np.where(signs[1:] != signs[:-1])[0][0] + 1)
            first_cross = float(P_VALUES[j])
        rows.append({
            "subject_idx": int(idx), "y": int(y),
            "n_adjacent_boundary_crossings": crossings,
            "ever_crosses_boundary": bool(crossings > 0),
            "first_p_after_crossing": first_cross,
        })
    return pd.DataFrame(rows)


def savefig(fig: plt.Figure, outbase: Path, dpi: int) -> None:
    fig.savefig(outbase.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    fig.savefig(outbase.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_spaghetti(subj: pd.DataFrame, out: Path, dpi: int, class_names: dict[int, str]) -> None:
    fig, ax = plt.subplots(figsize=(9.2, 6.2))
    # Use default matplotlib colors, one per class; no hard-coded colors.
    line_handles = []
    for y in (0, 1):
        gy = subj[subj.y == y]
        pivot = gy.pivot(index="subject_idx", columns="p", values="score_median").reindex(columns=P_VALUES)
        color = f"C{y}"
        for _, row in pivot.iterrows():
            ax.plot(P_VALUES, row.to_numpy(float), color=color, alpha=0.14, lw=0.8)
        med = gy.groupby("p")["score_median"].median().reindex(P_VALUES)
        h, = ax.plot(P_VALUES, med.to_numpy(float), color=color, lw=2.6, marker="o",
                     label=f"{class_names[y]} median")
        line_handles.append(h)
    ax.axhline(0, ls="--", lw=1.2, color="0.25")
    ax.axvspan(3, 4, alpha=0.07, color="0.5")
    ax.set_xscale("log")
    ax.set_xticks(P_VALUES)
    ax.set_xticklabels([str(int(p)) if float(p).is_integer() else str(p) for p in P_VALUES])
    ax.set_xlabel("Schatten p")
    ax.set_ylabel("OOF SVM decision score (subject median across 20 repeats)")
    ax.set_title("Subject-specific held-out decision trajectories")
    ax.legend(handles=line_handles, frameon=False)
    ax.grid(alpha=0.18)
    fig.tight_layout()
    savefig(fig, out / "02_spaghetti_oof_subject_trajectories", dpi)


def plot_delta_spaghetti(subj: pd.DataFrame, out: Path, dpi: int, class_names: dict[int, str]) -> None:
    wide = subj.pivot(index=["subject_idx", "y"], columns="p", values="score_median").reindex(columns=P_VALUES)
    ref = wide[REFERENCE_P]
    delta = wide.subtract(ref, axis=0)

    fig, ax = plt.subplots(figsize=(9.2, 6.2))
    handles = []
    for y in (0, 1):
        dy = delta.loc[pd.IndexSlice[:, y], :]
        color = f"C{y}"
        for _, row in dy.iterrows():
            ax.plot(P_VALUES, row.to_numpy(float), color=color, alpha=0.14, lw=0.8)
        med = np.nanmedian(dy.to_numpy(float), axis=0)
        h, = ax.plot(P_VALUES, med, color=color, lw=2.6, marker="o",
                     label=f"{class_names[y]} median")
        handles.append(h)
    ax.axhline(0, ls="--", lw=1.2, color="0.25")
    ax.axvspan(3, 4, alpha=0.07, color="0.5")
    ax.set_xscale("log")
    ax.set_xticks(P_VALUES)
    ax.set_xticklabels([str(int(p)) if float(p).is_integer() else str(p) for p in P_VALUES])
    ax.set_xlabel("Schatten p")
    ax.set_ylabel(r"$\Delta f_s(p)=f_s(p)-f_s(1)$")
    ax.set_title("Subject-specific change in held-out decision score")
    ax.legend(handles=handles, frameon=False)
    ax.grid(alpha=0.18)
    fig.tight_layout()
    savefig(fig, out / "03_spaghetti_oof_delta_from_p1", dpi)


def plot_distributions(oof: pd.DataFrame, out: Path, dpi: int, class_names: dict[int, str]) -> None:
    # Repeat-level class medians + IQR across repeats, rather than treating all fold scores as independent.
    rep = repeat_class_summary(oof, "OOF")
    fig, ax = plt.subplots(figsize=(9.2, 6.2))
    for y in (0, 1):
        gy = rep[rep.y == y]
        med, lo, hi = [], [], []
        for p in P_VALUES:
            x = gy.loc[np.isclose(gy.p, p), "median"].to_numpy(float)
            med.append(np.median(x)); lo.append(np.quantile(x, .25)); hi.append(np.quantile(x, .75))
        med, lo, hi = map(np.asarray, (med, lo, hi))
        ax.errorbar(P_VALUES, med, yerr=np.vstack([med-lo, hi-med]), marker="o", capsize=3,
                    lw=1.8, label=class_names[y])
    ax.axhline(0, ls="--", lw=1.2, color="0.25")
    ax.axvspan(3, 4, alpha=0.07, color="0.5")
    ax.set_xscale("log")
    ax.set_xticks(P_VALUES)
    ax.set_xticklabels([str(int(p)) if float(p).is_integer() else str(p) for p in P_VALUES])
    ax.set_xlabel("Schatten p")
    ax.set_ylabel("Within-class median OOF decision score")
    ax.set_title("Held-out class separation along geometric deformation")
    ax.legend(frameon=False)
    ax.grid(alpha=0.18)
    fig.tight_layout()
    savefig(fig, out / "04_oof_score_distributions_by_class", dpi)


def plot_heatmap(subj: pd.DataFrame, out: Path, dpi: int) -> None:
    wide = subj.pivot(index=["subject_idx", "y"], columns="p", values="score_median").reindex(columns=P_VALUES)
    # Order by class, then score at p=1; keeps subject identity fixed across p.
    meta = wide.reset_index()[["subject_idx", "y"]]
    vals = wide.reset_index(drop=True)
    order = np.lexsort((vals[REFERENCE_P].to_numpy(float), meta.y.to_numpy(int)))
    mat = vals.iloc[order].to_numpy(float)
    sorted_y = meta.iloc[order].y.to_numpy(int)
    boundary = int(np.sum(sorted_y == 0))

    fig, ax = plt.subplots(figsize=(8.2, 8.0))
    vmax = np.nanpercentile(np.abs(mat), 98)
    im = ax.imshow(mat, aspect="auto", interpolation="nearest", cmap="coolwarm",
                   vmin=-vmax, vmax=vmax)
    if 0 < boundary < len(mat):
        ax.axhline(boundary - 0.5, color="k", lw=1.1)
    ax.set_xticks(np.arange(len(P_VALUES)))
    ax.set_xticklabels([str(int(p)) if float(p).is_integer() else str(p) for p in P_VALUES])
    ax.set_xlabel("Schatten p")
    ax.set_ylabel("Subjects (ordered by class and p=1 score)")
    ax.set_title("Held-out subject-by-p decision map")
    fig.colorbar(im, ax=ax, label="Median OOF decision score")
    fig.tight_layout()
    savefig(fig, out / "05_oof_subject_by_p_heatmap", dpi)


def plot_concordance(summary: pd.DataFrame, out: Path, dpi: int) -> None:
    fig, ax = plt.subplots(figsize=(8.4, 5.8))
    ax.plot(summary.p, summary.spearman_vs_p1_subject_median, marker="o", label="Spearman")
    ax.plot(summary.p, summary.pearson_vs_p1_subject_median, marker="s", label="Pearson")
    ax.axvspan(3, 4, alpha=0.07, color="0.5")
    ax.set_xscale("log")
    ax.set_xticks(P_VALUES)
    ax.set_xticklabels([str(int(p)) if float(p).is_integer() else str(p) for p in P_VALUES])
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("Schatten p")
    ax.set_ylabel("Correlation with subject scores at p=1")
    ax.set_title("Held-out decision-score concordance")
    ax.legend(frameon=False)
    ax.grid(alpha=0.18)
    fig.tight_layout()
    savefig(fig, out / "06_oof_score_concordance_vs_p1", dpi)


def plot_agreement(summary: pd.DataFrame, out: Path, dpi: int) -> None:
    fig, ax = plt.subplots(figsize=(8.4, 5.8))
    ax.plot(summary.p, summary.label_agreement_vs_p1_subject_median, marker="o")
    ax.axvspan(3, 4, alpha=0.07, color="0.5")
    ax.set_xscale("log")
    ax.set_xticks(P_VALUES)
    ax.set_xticklabels([str(int(p)) if float(p).is_integer() else str(p) for p in P_VALUES])
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Schatten p")
    ax.set_ylabel("Label agreement with p=1")
    ax.set_title("Stability of held-out decision sign")
    ax.grid(alpha=0.18)
    fig.tight_layout()
    savefig(fig, out / "07_oof_label_agreement_vs_p1", dpi)


def plot_train_vs_oof(oof: pd.DataFrame, train: pd.DataFrame, out: Path, dpi: int, class_names: dict[int, str]) -> None:
    ro = repeat_class_summary(oof, "OOF")
    rt = repeat_class_summary(train, "train")
    fig, ax = plt.subplots(figsize=(9.2, 6.2))
    linestyles = {"OOF": "-", "train": "--"}
    for source, rdf in [("train", rt), ("OOF", ro)]:
        for y in (0, 1):
            med = []
            for p in P_VALUES:
                x = rdf.loc[np.isclose(rdf.p, p) & (rdf.y == y), "median"].to_numpy(float)
                med.append(np.median(x))
            ax.plot(P_VALUES, med, marker="o", ls=linestyles[source], color=f"C{y}",
                    label=f"{class_names[y]} {source}")
    ax.axhline(0, ls=":", lw=1.1, color="0.25")
    ax.axvspan(3, 4, alpha=0.07, color="0.5")
    ax.set_xscale("log")
    ax.set_xticks(P_VALUES)
    ax.set_xticklabels([str(int(p)) if float(p).is_integer() else str(p) for p in P_VALUES])
    ax.set_xlabel("Schatten p")
    ax.set_ylabel("Median decision score")
    ax.set_title("Training versus held-out decision separation (diagnostic)")
    ax.legend(frameon=False, ncol=2)
    ax.grid(alpha=0.18)
    fig.tight_layout()
    savefig(fig, out / "08_train_vs_oof_class_separation", dpi)


def main() -> None:
    global P_VALUES, N_REPEATS, N_SPLITS, REFERENCE_P
    args = parse_args()
    P_VALUES = np.asarray(args.p_values, dtype=float)
    N_REPEATS = int(args.n_repeats)
    N_SPLITS = int(args.n_splits)
    REFERENCE_P = float(args.reference_p)
    if not np.any(np.isclose(P_VALUES, REFERENCE_P)):
        raise ValueError("reference-p must be present in p-values")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    class_names = {0: args.negative_name, 1: args.positive_name}

    print("[LOAD] Reading v4 fold checkpoints...")
    oof, train = load_oof_and_train(args.results_source)
    n_subjects = oof.subject_idx.nunique()
    print(f"[DATA] OOF rows={len(oof):,}; subjects={n_subjects}; p={len(P_VALUES)}; repeats={N_REPEATS}")

    expected = n_subjects * len(P_VALUES) * N_REPEATS
    if len(oof) != expected:
        raise RuntimeError(f"Expected {expected:,} OOF rows, found {len(oof):,}.")

    oof.to_csv(args.output_dir / "00_oof_scores_long.csv", index=False)
    subj = subject_summary(oof)
    subj.to_csv(args.output_dir / "01_subject_oof_summary.csv", index=False)

    summary = decision_summary(oof, subj)
    summary.to_csv(args.output_dir / "09_decision_summary_by_p.csv", index=False)
    crossings = crossings_summary(subj)
    crossings.to_csv(args.output_dir / "10_subject_crossings_summary.csv", index=False)

    plot_spaghetti(subj, args.output_dir, args.dpi, class_names)
    plot_delta_spaghetti(subj, args.output_dir, args.dpi, class_names)
    plot_distributions(oof, args.output_dir, args.dpi, class_names)
    plot_heatmap(subj, args.output_dir, args.dpi)
    plot_concordance(summary, args.output_dir, args.dpi)
    plot_agreement(summary, args.output_dir, args.dpi)
    plot_train_vs_oof(oof, train, args.output_dir, args.dpi, class_names)

    print("[SAVE] Decision-space analysis written to:", args.output_dir)
    print("[NEXT] Inspect 02_spaghetti_oof_subject_trajectories.png first.")


if __name__ == "__main__":
    main()
