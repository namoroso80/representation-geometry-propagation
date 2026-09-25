#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Support-vector layer analysis for Geometry 2.0.

Purpose
-------
Target the empirically localized transition K_p -> f_p without modifying the
production pipeline. The analysis uses the exact outer-fold checkpoints already
saved by the classifier.

Prespecified diagnostics
------------------------
For every dataset, p, repeat and outer fold:

1. Support-vector set overlap vs p=1:
       J_SV = |SV_p ∩ SV_1| / |SV_p ∪ SV_1|

2. Reference-SV kernel preservation:
   Compare test-to-SV_1 kernel relations under p and p=1, using the actual
   fold-specific gamma selected at each p.

3. Dual-coefficient preservation on the union of support vectors:
   cosine similarity between signed SVC dual-weight vectors after embedding
   both models on SV_p ∪ SV_1 and filling absent entries with zero.

4. Decision-contribution preservation:
   Compare the flattened per-test-subject contribution fields
       alpha_i K(x_i, x)
   on the same SV union. This is immediately upstream of the SVC sum.

The analysis reports all four diagnostics; none is selected based on which one
best matches decision drift.
"""

from __future__ import annotations
import argparse, json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from sklearn.svm import SVC


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--reference-p", type=float, default=1.0)
    ap.add_argument("--dpi", type=int, default=250)
    return ap.parse_args()


def checkpoint_path(root: Path, p_index: int, fold_id: int) -> Path:
    candidates = [
        root / "checkpoints" / f"p{p_index:02d}_fold{fold_id:03d}.npz",
        root / "checkpoints" / f"p{p_index:02d}_fold_{fold_id:03d}.npz",
        root / f"p{p_index:02d}_fold{fold_id:03d}.npz",
        root / "fold_checkpoints" / f"p{p_index:02d}_fold{fold_id:03d}.npz",
    ]
    for q in candidates:
        if q.exists():
            return q
    raise FileNotFoundError(
        f"Checkpoint not found for p_index={p_index}, fold_id={fold_id} under {root}"
    )


def dpath(root, p):
    root = Path(root).expanduser()
    for s in [str(float(p)).replace(".", "p"), f"{p:g}", str(p).replace(".", "p")]:
        for q in [root / f"distance_p{s}.npy", root / f"distance_{s}.npy"]:
            if q.exists():
                return q
    raise FileNotFoundError(f"distance p={p} under {root}")


def distance_from_spectra(source, p, n_subjects):
    source = Path(source).expanduser()
    sv = np.load(source / "pairwise_singular_values.npy", mmap_mode="r")
    z = np.load(source / "pair_indices.npz")
    ii = z["pair_s"].astype(int)
    jj = z["pair_t"].astype(int)
    block = np.asarray(sv, dtype=float)
    if float(p) == 1.0:
        vals = np.sum(block, axis=1)
    elif float(p) == 2.0:
        vals = np.sqrt(np.sum(block * block, axis=1))
    else:
        vals = np.sum(np.abs(block) ** float(p), axis=1) ** (1.0 / float(p))
    D = np.zeros((n_subjects, n_subjects), dtype=float)
    D[ii, jj] = vals
    D[jj, ii] = vals
    return D


def load_distance(sp, p, n_subjects):
    if sp.get("distance_dir"):
        return np.load(dpath(sp["distance_dir"], p))
    if sp.get("spectra_source"):
        return distance_from_spectra(sp["spectra_source"], p, n_subjects)
    raise ValueError(f'{sp["name"]}: provide distance_dir or spectra_source')


def reconstruct_model(D, z):
    tr = z["train_idx"].astype(int)
    te = z["test_idx"].astype(int)
    ytr = z["y_train"].astype(int)
    gamma = float(z["gamma"])
    C = float(z["C"])
    Ktrain = np.exp(-gamma * D[np.ix_(tr, tr)] ** 2)
    clf = SVC(C=C, kernel="precomputed")
    clf.fit(Ktrain, ytr)
    support_local = clf.support_.astype(int)
    support_global = tr[support_local]
    dual = clf.dual_coef_.reshape(-1).astype(float)
    return tr, te, gamma, C, support_global, dual, clf


def cosine(a, b):
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    den = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / den) if den > 0 else np.nan


def jaccard(a, b):
    a, b = set(map(int, a)), set(map(int, b))
    u = a | b
    return len(a & b) / len(u) if u else 1.0


def safe_spearman(a, b):
    a = np.asarray(a, float).ravel()
    b = np.asarray(b, float).ravel()
    if len(a) < 3 or np.allclose(a, a[0]) or np.allclose(b, b[0]):
        return np.nan
    return float(spearmanr(a, b).statistic)


def embed_dual(union, support, dual):
    pos = {int(s): i for i, s in enumerate(union)}
    v = np.zeros(len(union), float)
    for s, w in zip(support, dual):
        v[pos[int(s)]] = float(w)
    return v


def contribution_field(D, te, union, support, dual, gamma):
    """
    Signed support-vector contribution field on a common union:
       contribution(x, i) = dual_i * K(x, SV_i)
    with zero columns for subjects not supporting the model.
    """
    pos = {int(s): i for i, s in enumerate(union)}
    W = np.zeros(len(union), float)
    for s, w in zip(support, dual):
        W[pos[int(s)]] = float(w)
    K = np.exp(-gamma * D[np.ix_(te, union)] ** 2)
    return K * W[None, :]


def plot_trajectory_scatter(df, x, y, path, title, xlabel, ylabel, dpi):
    fig, ax = plt.subplots(figsize=(8.0, 5.8))
    for name, g in df.groupby("dataset", sort=False):
        g = g.sort_values("p")
        ax.plot(g[x], g[y], marker="o", lw=1.4, ms=5, label=name)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=.2)
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(path.with_suffix(".png"), dpi=dpi)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    specs = json.loads(args.config.read_text())["datasets"]

    all_rows = []

    for sp in specs:
        name = sp["name"]
        oof = pd.read_csv(Path(sp["oof"]).expanduser())
        fs = pd.read_csv(Path(sp["fold_summary"]).expanduser())

        ps = sorted(oof["p"].astype(float).unique())
        n_subjects = int(oof["subject_idx"].nunique())
        n_splits = int(oof["fold_within_repeat"].max())
        n_repeats = int(oof["repeat"].max())

        # Normalize fold_summary provenance.
        if "repeat" not in fs.columns or "fold_within_repeat" not in fs.columns:
            if "outer_fold" not in fs.columns:
                raise ValueError(f"{name}: fold_summary lacks repeat/fold and outer_fold")
            fs = fs.copy()
            fs["repeat"] = ((fs["outer_fold"].astype(int) - 1) // n_splits) + 1
            fs["fold_within_repeat"] = ((fs["outer_fold"].astype(int) - 1) % n_splits) + 1

        gamma_col = next((c for c in ["gamma","gamma_best","best_gamma","gamma_selected"] if c in fs.columns), None)
        C_col = next((c for c in ["C","c","C_best","best_C","C_selected"] if c in fs.columns), None)
        if gamma_col is None or C_col is None:
            raise ValueError(f"{name}: need gamma and C in fold_summary; columns={list(fs.columns)}")

        # One global label per subject, recovered from OOF records.
        yy = (oof[["subject_idx","y"]].drop_duplicates()
              .sort_values("subject_idx"))
        if len(yy) != n_subjects or not np.array_equal(yy.subject_idx.to_numpy(), np.arange(n_subjects)):
            # General mapping if subject IDs are not contiguous.
            subject_ids = np.sort(oof.subject_idx.unique().astype(int))
            y_map = dict(zip(yy.subject_idx.astype(int), yy.y.astype(int)))
        else:
            subject_ids = np.arange(n_subjects)
            y_map = dict(zip(subject_ids, yy.y.astype(int)))

        print(f"[SV] {name}: subjects={n_subjects}, repeats={n_repeats}, splits={n_splits}")

        D1 = load_distance(sp, args.reference_p, n_subjects)

        # Reconstruct reference fitted model for every outer fold from frozen
        # fold_summary + OOF split provenance. No checkpoint files required.
        ref_models = {}
        for rep in range(1, n_repeats + 1):
            for fold in range(1, n_splits + 1):
                test_ids = np.sort(
                    oof[(oof.repeat == rep) &
                        (oof.fold_within_repeat == fold) &
                        np.isclose(oof.p.astype(float), args.reference_p)]
                    .subject_idx.unique().astype(int)
                )
                train_ids = np.setdiff1d(subject_ids, test_ids)
                row = fs[(fs["repeat"] == rep) &
                         (fs["fold_within_repeat"] == fold) &
                         np.isclose(fs["p"].astype(float), args.reference_p)]
                if len(row) != 1:
                    raise ValueError(f"{name}: reference model lookup failed rep={rep}, fold={fold}, rows={len(row)}")
                g1 = float(row.iloc[0][gamma_col])
                C1 = float(row.iloc[0][C_col])
                ytr = np.array([y_map[int(s)] for s in train_ids], dtype=int)
                Ktr = np.exp(-g1 * D1[np.ix_(train_ids, train_ids)] ** 2)
                clf1 = SVC(C=C1, kernel="precomputed")
                clf1.fit(Ktr, ytr)
                sv1 = train_ids[clf1.support_.astype(int)]
                dual1 = clf1.dual_coef_.reshape(-1).astype(float)
                ref_models[(rep, fold)] = (train_ids, test_ids, g1, C1, sv1, dual1)

        # Process each p once; crucial for large MIMII distance reconstruction.
        for pi, p in enumerate(ps, start=1):
            print(f"[SV] {name}: p={p:g} ({pi}/{len(ps)})")
            Dp = D1 if np.isclose(p, args.reference_p) else load_distance(sp, p, n_subjects)

            for rep in range(1, n_repeats + 1):
                for fold in range(1, n_splits + 1):
                    tr1, te1, g1, C1, sv1, dual1 = ref_models[(rep, fold)]

                    test_p = np.sort(
                        oof[(oof.repeat == rep) &
                            (oof.fold_within_repeat == fold) &
                            np.isclose(oof.p.astype(float), p)]
                        .subject_idx.unique().astype(int)
                    )
                    train_p = np.setdiff1d(subject_ids, test_p)
                    if not np.array_equal(tr1, train_p) or not np.array_equal(te1, test_p):
                        raise RuntimeError(f"{name}: outer split changed rep={rep}, fold={fold}, p={p}")

                    row = fs[(fs["repeat"] == rep) &
                             (fs["fold_within_repeat"] == fold) &
                             np.isclose(fs["p"].astype(float), p)]
                    if len(row) != 1:
                        raise ValueError(f"{name}: model lookup failed rep={rep}, fold={fold}, p={p}, rows={len(row)}")
                    gp = float(row.iloc[0][gamma_col])
                    Cp = float(row.iloc[0][C_col])

                    ytr = np.array([y_map[int(s)] for s in train_p], dtype=int)
                    Ktrp = np.exp(-gp * Dp[np.ix_(train_p, train_p)] ** 2)
                    clfp = SVC(C=Cp, kernel="precomputed")
                    clfp.fit(Ktrp, ytr)
                    svp = train_p[clfp.support_.astype(int)]
                    dualp = clfp.dual_coef_.reshape(-1).astype(float)

                    union = np.union1d(sv1, svp).astype(int)

                    sv_j = jaccard(sv1, svp)

                    K1_refsv = np.exp(-g1 * D1[np.ix_(te1, sv1)] ** 2)
                    Kp_refsv = np.exp(-gp * Dp[np.ix_(test_p, sv1)] ** 2)
                    refsv_rho = safe_spearman(K1_refsv, Kp_refsv)

                    a1 = embed_dual(union, sv1, dual1)
                    ap = embed_dual(union, svp, dualp)
                    dual_cos = cosine(a1, ap)
                    dual_rho = safe_spearman(a1, ap)

                    Q1 = contribution_field(D1, te1, union, sv1, dual1, g1)
                    Qp = contribution_field(Dp, test_p, union, svp, dualp, gp)
                    contrib_cos = cosine(Q1.ravel(), Qp.ravel())
                    contrib_rho = safe_spearman(Q1, Qp)

                    # Use frozen OOF scores rather than reconstructed checkpoint scores.
                    s1 = (oof[(oof.repeat == rep) &
                              (oof.fold_within_repeat == fold) &
                              np.isclose(oof.p.astype(float), args.reference_p)]
                          .sort_values("subject_idx")["score"].to_numpy(float))
                    spv = (oof[(oof.repeat == rep) &
                               (oof.fold_within_repeat == fold) &
                               np.isclose(oof.p.astype(float), p)]
                           .sort_values("subject_idx")["score"].to_numpy(float))
                    decision_rho_fold = safe_spearman(s1, spv)

                    all_rows.append({
                        "dataset": name, "p": float(p), "repeat": rep,
                        "fold_within_repeat": fold,
                        "outer_fold": (rep - 1) * n_splits + fold,
                        "n_train": len(tr1), "n_test": len(te1),
                        "n_sv_p1": len(sv1), "n_sv_p": len(svp),
                        "sv_fraction_p1": len(sv1) / len(tr1),
                        "sv_fraction_p": len(svp) / len(train_p),
                        "sv_jaccard_vs_p1": sv_j,
                        "sv_loss": 1.0 - sv_j,
                        "reference_sv_kernel_spearman": refsv_rho,
                        "reference_sv_kernel_loss": 1.0 - refsv_rho if np.isfinite(refsv_rho) else np.nan,
                        "dual_cosine_vs_p1": dual_cos,
                        "dual_spearman_vs_p1": dual_rho,
                        "dual_loss_cosine": 1.0 - dual_cos if np.isfinite(dual_cos) else np.nan,
                        "contribution_cosine_vs_p1": contrib_cos,
                        "contribution_spearman_vs_p1": contrib_rho,
                        "contribution_loss_cosine": 1.0 - contrib_cos if np.isfinite(contrib_cos) else np.nan,
                        "contribution_loss_spearman": 1.0 - contrib_rho if np.isfinite(contrib_rho) else np.nan,
                        "decision_spearman_fold_vs_p1": decision_rho_fold,
                        "decision_loss_fold": 1.0 - decision_rho_fold if np.isfinite(decision_rho_fold) else np.nan,
                        "gamma_p1": g1, "gamma_p": gp, "C_p1": C1, "C_p": Cp,
                    })

            if not np.isclose(p, args.reference_p):
                del Dp

    fold_df = pd.DataFrame(all_rows)
    fold_df.to_csv(args.output_dir / "10_support_vector_layer_fold.csv", index=False)

    metrics = [
        "sv_jaccard_vs_p1","sv_loss",
        "reference_sv_kernel_spearman","reference_sv_kernel_loss",
        "dual_cosine_vs_p1","dual_spearman_vs_p1","dual_loss_cosine",
        "contribution_cosine_vs_p1","contribution_spearman_vs_p1",
        "contribution_loss_cosine","contribution_loss_spearman",
        "decision_spearman_fold_vs_p1","decision_loss_fold",
        "sv_fraction_p1","sv_fraction_p",
    ]
    rows = []
    for (name, p), g in fold_df.groupby(["dataset","p"]):
        row = {"dataset": name, "p": float(p), "n_folds": len(g)}
        for m in metrics:
            v = g[m].to_numpy(float)
            v = v[np.isfinite(v)]
            row[m+"_mean"] = np.mean(v) if len(v) else np.nan
            row[m+"_median"] = np.median(v) if len(v) else np.nan
            row[m+"_sem"] = np.std(v, ddof=1)/np.sqrt(len(v)) if len(v)>1 else np.nan
        rows.append(row)
    agg = pd.DataFrame(rows)
    agg.to_csv(args.output_dir / "11_support_vector_layer_summary.csv", index=False)

    joined = agg.copy()
    prior = args.output_dir.parent / "outputs_v3_2" / "03b_decision_repeat_loss.csv"
    if prior.exists():
        d = pd.read_csv(prior)
        ds = d.groupby(["dataset","p"])["decision_loss_repeat"].mean().reset_index(name="decision_loss_repeat_mean")
        joined = joined.merge(ds, on=["dataset","p"], how="left")
    joined.to_csv(args.output_dir / "12_support_vector_propagation_joined.csv", index=False)

    ycol = "decision_loss_repeat_mean" if "decision_loss_repeat_mean" in joined.columns else "decision_loss_fold_mean"

    plot_trajectory_scatter(joined,"sv_loss_mean",ycol,args.output_dir/"10_sv_overlap_vs_decision",
        "Support-vector reorganization vs decision drift",
        "Support-vector loss (1 - Jaccard vs p=1)","Decision loss",args.dpi)
    plot_trajectory_scatter(joined,"reference_sv_kernel_loss_mean",ycol,args.output_dir/"11_reference_sv_kernel_vs_decision",
        "Task-anchor kernel deformation vs decision drift",
        "Kernel loss toward p=1 support vectors","Decision loss",args.dpi)
    plot_trajectory_scatter(joined,"dual_loss_cosine_mean",ycol,args.output_dir/"12_dual_weights_vs_decision",
        "Dual-weight reorganization vs decision drift",
        "Dual-weight cosine loss","Decision loss",args.dpi)
    plot_trajectory_scatter(joined,"contribution_loss_cosine_mean",ycol,args.output_dir/"13_contribution_field_vs_decision",
        "Support-vector contribution reorganization vs decision drift",
        "Contribution-field cosine loss","Decision loss",args.dpi)

    print(f"[DONE] {args.output_dir}")

if __name__ == "__main__":
    main()
