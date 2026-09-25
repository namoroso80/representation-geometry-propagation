#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AD Schatten-p -> gamma -> kernel -> decision dynamics (v4)
===========================================================

Purpose
-------
Study, step by step on the real AD/NC morphometric dataset, how a controlled
geometric deformation induced by the Schatten-p family propagates from
subject-space distances to an RBF-kernel SVM.

Scientific chain
----------------
    p -> D_p -> gamma*(p, outer_fold) -> K_p -> f_p

This is an exploratory/mechanistic analysis. It saves rich fold-level outputs,
but all scientific visualization of hyperparameter behaviour is aggregated at
the level of the 20 repeated outer-CV repetitions rather than treating 100
outer folds as independent experimental units.

Main design choices
-------------------
- AD/NC morphometric MRI only (pilot/discovery dataset).
- Density operators built exactly as in the previous pipeline:
  symmetrize, native/unit diagonal according to configuration, PSD check,
  trace normalization.
- Schatten grid identical to the simulation:
  p = 1, 1.25, 1.5, 1.75, 2, 3, 4, 8, 16.
- Repeated nested CV identical to the previous paper:
  20 repetitions x stratified 5-fold outer CV = 100 held-out evaluations;
  stratified 5-fold inner CV; ROC-AUC model selection.
- Precomputed RBF-kernel SVM.
- C grid preserved from the previous pipeline: logspace(-3, 3, 7).
- gamma grid is now ABSOLUTE and identical for every p and every fold.
  No median-distance rescaling is used to construct the search grid.
- Full pairwise eigenspectra of rho_s-rho_t are computed once and reused for
  every p.
- Saves, for every (p, outer fold):
    S_train = ||D_train||_F
    median(d_train^2) and 1/median(d_train^2) as diagnostics only
    best gamma and C
    inner-CV ROC-AUC
    K_train and K_test
    train/test decision scores and predictions
    train/test subject indices and labels
- Out-of-sample scores are the only unbiased predictive outputs.
  Training scores are saved for mechanistic diagnostics only.

The script is intentionally modular and cell-oriented (# %%).
"""

from __future__ import annotations

# %% 0. IMPORTS AND THREAD SETUP
import argparse
import json
import math
import os
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

# Prevent accidental oversubscription when NumPy/Accelerate is already threaded.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.linalg import eigvalsh
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    roc_auc_score,
)
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
from sklearn.svm import SVC
from tqdm.auto import tqdm


# %% 1. SCIENTIFIC AND COMPUTATIONAL CONFIGURATION
@dataclass(frozen=True)
class ScientificConfig:
    p_values: tuple[float, ...] = (
        1.0,
        1.25,
        1.5,
        1.75,
        2.0,
        3.0,
        4.0,
        8.0,
        16.0,
    )

    # Same regularization grid as previous pipeline.
    C_values: tuple[float, ...] = tuple(np.logspace(-3, 3, 7).tolist())

    # Absolute gamma grid: SAME values for every p and every fold.
    # Broad by design; boundary hits are explicitly diagnosed.
    gamma_values: tuple[float, ...] = tuple((10.0 ** np.arange(-6.0, 8.0001, 0.5)).tolist())

    n_repeats: int = 20
    n_splits: int = 5
    inner_splits: int = 5
    random_state: int = 42

    positive_label: str = "POSITIVE"
    negative_label: str = "NEGATIVE"


@dataclass(frozen=True)
class ComputeConfig:
    dtype: str = "float64"
    kernel_save_dtype: str = "float32"
    show_progress: bool = True
    save_distance_matrices: bool = True
    save_pairwise_spectra: bool = False
    skip_existing: bool = True


SCI = ScientificConfig()
COMP = ComputeConfig()
DTYPE = np.dtype(COMP.dtype)
KERNEL_DTYPE = np.dtype(COMP.kernel_save_dtype)


# %% 2. I/O UTILITIES — COMPATIBLE WITH PREVIOUS PIPELINE
def natural_sort_key(path: Path) -> Tuple:
    parts = re.split(r"(\d+)", path.name)
    return tuple(int(p) if p.isdigit() else p.lower() for p in parts)


def read_info(path: Path, delimiter: Optional[str] = None) -> pd.DataFrame:
    if delimiter is None:
        return pd.read_csv(path, sep=None, engine="python")
    return pd.read_csv(path, sep=delimiter)


def infer_format(path: Path, fmt: str) -> str:
    if fmt != "auto":
        return fmt
    suffix = path.suffix.lower()
    if suffix == ".npy":
        return "npy"
    if suffix in {".csv", ".txt", ".tsv"}:
        return "csv"
    raise ValueError(f"Cannot infer matrix format from {path}")


def load_matrix(path: Path, fmt: str = "auto") -> np.ndarray:
    fmt = infer_format(path, fmt)
    if fmt == "npy":
        A = np.load(path)
    elif fmt == "csv":
        A = pd.read_csv(path, header=None).to_numpy(dtype=float)
    else:
        raise ValueError(f"Unsupported matrix format: {fmt}")

    A = np.asarray(A, dtype=np.float64)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError(f"Matrix {path.name} is not square: shape={A.shape}")
    if not np.all(np.isfinite(A)):
        raise ValueError(f"Matrix {path.name} contains non-finite values")
    return A


def list_matrix_files(data_dir: Path, matrix_glob: str) -> List[Path]:
    files = sorted(data_dir.glob(matrix_glob), key=natural_sort_key)
    if not files:
        raise FileNotFoundError(
            f"No matrices found with pattern {matrix_glob!r} in {data_dir}"
        )
    return files


def align_info_and_files(
    info: pd.DataFrame,
    files: List[Path],
    id_col: Optional[str],
) -> pd.DataFrame:
    if len(info) != len(files):
        raise ValueError(
            f"Number of info rows ({len(info)}) does not match number of "
            f"matrices ({len(files)}). This pipeline assumes the same order."
        )

    out = info.copy()
    out["matrix_file"] = [p.name for p in files]
    out["matrix_path"] = [str(p) for p in files]

    if id_col and id_col in out.columns:
        out["sample_id"] = out[id_col].astype(str)
    else:
        out["sample_id"] = [p.stem for p in files]

    return out


# %% 3. DENSITY-OPERATOR CONSTRUCTION
def preprocess_matrix(
    A: np.ndarray,
    diagonal: str = "unit",
    symmetrize: bool = True,
) -> np.ndarray:
    A = np.asarray(A, dtype=np.float64).copy()

    if diagonal == "unit":
        np.fill_diagonal(A, 1.0)
    elif diagonal == "keep":
        pass
    else:
        raise ValueError("diagonal must be 'unit' or 'keep'")

    if symmetrize:
        A = 0.5 * (A + A.T)

    return np.ascontiguousarray(A, dtype=np.float64)


def make_density(A: np.ndarray, tol: float) -> np.ndarray:
    tr = float(np.trace(A))
    if not np.isfinite(tr) or abs(tr) <= tol:
        raise ValueError("Cannot trace-normalize matrix with zero/non-finite trace")
    return A / tr


def build_density_stack(
    data_dir: Path,
    info_file: Path,
    matrix_glob: str,
    matrix_format: str,
    id_col: Optional[str],
    label_col: str,
    delimiter: Optional[str],
    diagonal: str,
    symmetrize: bool,
    psd_tol: float,
    expected_nodes: Optional[int],
) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:
    files = list_matrix_files(data_dir, matrix_glob)
    info = read_info(info_file, delimiter)
    manifest = align_info_and_files(info, files, id_col)

    labels = manifest[label_col].astype(str).str.strip()
    keep = labels.isin([SCI.negative_label, SCI.positive_label])
    manifest = manifest.loc[keep].reset_index(drop=True)

    if manifest.empty:
        raise ValueError("No samples remain after binary label filtering")

    densities: list[np.ndarray] = []
    kept_rows = []
    stats = []
    excluded = []
    target_shape = None

    print(f"[DATA] Candidate binary-class subjects: {len(manifest)}")

    iterator = manifest.iterrows()
    if COMP.show_progress:
        iterator = tqdm(
            iterator,
            total=len(manifest),
            desc="Density operators",
            leave=False,
        )

    for _, row in iterator:
        path = Path(row["matrix_path"])
        try:
            A_raw = load_matrix(path, matrix_format)

            if expected_nodes is not None:
                wanted = (expected_nodes, expected_nodes)
                if A_raw.shape != wanted:
                    raise ValueError(f"shape={A_raw.shape}; expected {wanted}")

            if target_shape is None:
                target_shape = A_raw.shape
            if A_raw.shape != target_shape:
                raise ValueError(
                    f"shape={A_raw.shape}; first retained shape={target_shape}"
                )

            A = preprocess_matrix(
                A_raw,
                diagonal=diagonal,
                symmetrize=symmetrize,
            )

            evals = eigvalsh(A, check_finite=False)
            min_eval = float(evals.min())
            if min_eval < -psd_tol:
                raise ValueError(f"not PSD: min eigenvalue={min_eval:.3e}")

            rho = make_density(A, psd_tol)
            densities.append(rho)
            kept_rows.append(row.to_dict())
            stats.append(
                {
                    "sample_id": row["sample_id"],
                    "label": row[label_col],
                    "min_eigenvalue": min_eval,
                    "trace_raw": float(np.trace(A)),
                    "trace_density": float(np.trace(rho)),
                }
            )
        except Exception as exc:
            excluded.append(
                {
                    "sample_id": row.get("sample_id", path.stem),
                    "matrix_file": path.name,
                    "label": row.get(label_col, ""),
                    "reason": str(exc),
                }
            )

    if not densities:
        raise RuntimeError("No valid density operators were constructed")

    retained = pd.DataFrame(kept_rows).reset_index(drop=True)
    R = np.stack(densities).astype(DTYPE, copy=False)
    stats_df = pd.DataFrame(stats)

    if excluded:
        print(f"[DATA] Excluded subjects: {len(excluded)}")
    print(f"[DATA] Retained subjects: {len(retained)} | matrix shape={R.shape[1:]}")

    return retained, R, stats_df


# %% 4. LABEL ENCODING
def encode_labels(manifest: pd.DataFrame, label_col: str) -> np.ndarray:
    labels = manifest[label_col].astype(str).str.strip().to_numpy()
    y = np.full(len(labels), -1, dtype=int)
    y[labels == SCI.negative_label] = 0
    y[labels == SCI.positive_label] = 1

    if np.any(y < 0):
        bad = np.unique(labels[y < 0])
        raise ValueError(f"Unexpected labels after filtering: {bad}")

    print(
        f"[LABEL] {SCI.negative_label}=0: {(y == 0).sum()} | "
        f"{SCI.positive_label}=1: {(y == 1).sum()}"
    )
    return y


# %% 5. PAIRWISE EIGENSPECTRA — COMPUTE ONCE, REUSE FOR ALL p
def upper_triangle_pairs(n_subjects: int):
    return np.triu_indices(n_subjects, k=1)


def pairwise_singular_values(
    R: np.ndarray,
    show_progress: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    For Delta_st = rho_s-rho_t, Delta is symmetric, therefore its singular
    values are |eigvalsh(Delta)|. These are computed once for each unique pair.
    """
    pair_s, pair_t = upper_triangle_pairs(len(R))
    n_pairs = len(pair_s)
    n_nodes = R.shape[1]

    singvals = np.empty((n_pairs, n_nodes), dtype=DTYPE)

    iterator = range(n_pairs)
    if show_progress:
        iterator = tqdm(
            iterator,
            total=n_pairs,
            desc="Pair spectra",
            leave=False,
        )

    t0 = time.perf_counter()
    for q in iterator:
        delta = R[pair_s[q]] - R[pair_t[q]]
        singvals[q] = np.abs(eigvalsh(delta, check_finite=False))

    print(
        f"[GEOM] Pair spectra computed: {n_pairs:,} pairs "
        f"in {time.perf_counter() - t0:.1f} s"
    )
    return pair_s, pair_t, singvals


def schatten_pair_distances(singvals: np.ndarray, p: float) -> np.ndarray:
    if p < 1:
        raise ValueError("Schatten p must satisfy p >= 1")
    if np.isinf(p):
        return singvals.max(axis=1)
    return np.power(np.power(singvals, p).sum(axis=1), 1.0 / p)


def pair_vector_to_matrix(
    values: np.ndarray,
    pair_s: np.ndarray,
    pair_t: np.ndarray,
    n_subjects: int,
) -> np.ndarray:
    D = np.zeros((n_subjects, n_subjects), dtype=DTYPE)
    D[pair_s, pair_t] = values
    D[pair_t, pair_s] = values
    return D


def build_all_distance_matrices(
    singvals: np.ndarray,
    pair_s: np.ndarray,
    pair_t: np.ndarray,
    n_subjects: int,
) -> dict[float, np.ndarray]:
    out: dict[float, np.ndarray] = {}
    print(f"[GEOM] Building {len(SCI.p_values)} Schatten distance matrices")
    for p in SCI.p_values:
        pair_d = schatten_pair_distances(singvals, p)
        out[p] = pair_vector_to_matrix(pair_d, pair_s, pair_t, n_subjects)
        print(
            f"       p={p:g} | ||D_p||_F={np.linalg.norm(out[p], 'fro'):.6g}"
        )
    return out


# %% 6. CV SPLITS — CREATED ONCE AND SHARED ACROSS ALL p
def make_outer_splits(y: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    outer = RepeatedStratifiedKFold(
        n_splits=SCI.n_splits,
        n_repeats=SCI.n_repeats,
        random_state=SCI.random_state,
    )
    splits = [(tr.copy(), te.copy()) for tr, te in outer.split(np.zeros(len(y)), y)]
    expected = SCI.n_repeats * SCI.n_splits
    if len(splits) != expected:
        raise RuntimeError(f"Expected {expected} outer folds, got {len(splits)}")
    print(f"[CV] Outer folds prepared once and shared across p: {len(splits)}")
    return splits


# %% 7. KERNEL AND FOLD-LEVEL GEOMETRY DIAGNOSTICS
def rbf_from_distance(D: np.ndarray, gamma: float) -> np.ndarray:
    return np.exp(-float(gamma) * np.square(D))


def train_distance_diagnostics(Dtrain: np.ndarray) -> dict[str, float]:
    vals = Dtrain[np.triu_indices_from(Dtrain, k=1)]
    vals = vals[np.isfinite(vals) & (vals > 0)]

    S_train = float(np.linalg.norm(Dtrain, ord="fro"))
    if vals.size == 0:
        return {
            "S_train": S_train,
            "median_d": np.nan,
            "median_d2": np.nan,
            "gamma_scale_diagnostic": np.nan,
        }

    median_d = float(np.median(vals))
    median_d2 = float(np.median(vals ** 2))
    gamma_scale = 1.0 / median_d2 if median_d2 > 0 else np.nan

    return {
        "S_train": S_train,
        "median_d": median_d,
        "median_d2": median_d2,
        # Diagnostic only. NEVER used to build the gamma grid.
        "gamma_scale_diagnostic": float(gamma_scale),
    }


# %% 8. INNER CV — ABSOLUTE gamma GRID
def select_hyperparameters(
    Dtrain: np.ndarray,
    ytrain: np.ndarray,
    outer_fold: int,
) -> dict:
    """Select (gamma, C) and retain the complete inner-CV response surface.

    The scientific selection rule is unchanged: mean inner-fold ROC-AUC, with
    deterministic first-best tie handling.  v4 additionally stores the ROC-AUC
    attained in every inner fold for every point of the absolute (gamma, C) grid.
    """
    inner = StratifiedKFold(
        n_splits=SCI.inner_splits,
        shuffle=True,
        random_state=SCI.random_state + outer_fold,
    )

    inner_splits = [
        (itr.copy(), ival.copy())
        for itr, ival in inner.split(np.zeros(len(ytrain)), ytrain)
    ]

    gammas = np.asarray(SCI.gamma_values, dtype=float)
    Cs = np.asarray(SCI.C_values, dtype=float)
    auc_folds = np.full((len(gammas), len(Cs), len(inner_splits)), np.nan, dtype=np.float32)

    best_auc = -np.inf
    best_gamma = None
    best_C = None

    for ig, gamma in enumerate(gammas):
        Kfull = rbf_from_distance(Dtrain, float(gamma))

        for ic, C in enumerate(Cs):
            for ii, (itr, ival) in enumerate(inner_splits):
                K_i = Kfull[np.ix_(itr, itr)]
                K_v = Kfull[np.ix_(ival, itr)]

                clf = SVC(kernel="precomputed", C=float(C))
                clf.fit(K_i, ytrain[itr])
                scores = clf.decision_function(K_v)
                auc_folds[ig, ic, ii] = roc_auc_score(ytrain[ival], scores)

            mean_auc = float(np.mean(auc_folds[ig, ic, :]))

            # Preserve deterministic first-best tie handling.
            if mean_auc > best_auc:
                best_auc = mean_auc
                best_gamma = float(gamma)
                best_C = float(C)

    if best_gamma is None or best_C is None:
        raise RuntimeError("Inner CV failed to select hyperparameters")

    auc_mean = np.mean(auc_folds, axis=2, dtype=np.float64).astype(np.float32)
    auc_sd = np.std(auc_folds, axis=2, ddof=1, dtype=np.float64).astype(np.float32)

    gamma_min = float(gammas.min())
    gamma_max = float(gammas.max())

    return {
        "gamma": best_gamma,
        "C": best_C,
        "inner_auc": best_auc,
        "gamma_at_lower_boundary": bool(np.isclose(best_gamma, gamma_min)),
        "gamma_at_upper_boundary": bool(np.isclose(best_gamma, gamma_max)),
        "inner_gamma_grid": gammas,
        "inner_C_grid": Cs,
        "inner_auc_folds": auc_folds,
        "inner_auc_mean_surface": auc_mean,
        "inner_auc_sd_surface": auc_sd,
    }


# %% 9. OUTER-FOLD RUN — SAVE TRAIN AND TEST BEHAVIOUR
def metric_dict(y: np.ndarray, scores: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    return {
        "roc_auc": float(roc_auc_score(y, scores)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "accuracy": float(accuracy_score(y, pred)),
    }


def fold_result_path(output_dir: Path, p_index: int, fold_id: int) -> Path:
    return output_dir / "folds" / f"p{p_index:02d}_fold{fold_id:03d}.npz"


def run_outer_fold(
    D: np.ndarray,
    y: np.ndarray,
    p: float,
    p_index: int,
    fold_id: int,
    tr: np.ndarray,
    te: np.ndarray,
    output_dir: Path,
    skip_existing: bool,
) -> dict:
    out = fold_result_path(output_dir, p_index, fold_id)
    out.parent.mkdir(parents=True, exist_ok=True)

    if skip_existing and out.exists():
        data = np.load(out, allow_pickle=False)
        return {
            "p": float(data["p"]),
            "p_index": int(data["p_index"]),
            "outer_fold": int(data["outer_fold"]),
            "gamma": float(data["gamma"]),
            "C": float(data["C"]),
            "inner_auc": float(data["inner_auc"]),
            "S_train": float(data["S_train"]),
            "median_d": float(data["median_d"]),
            "median_d2": float(data["median_d2"]),
            "gamma_scale_diagnostic": float(data["gamma_scale_diagnostic"]),
            "gamma_at_lower_boundary": bool(data["gamma_at_lower_boundary"]),
            "gamma_at_upper_boundary": bool(data["gamma_at_upper_boundary"]),
            "test_roc_auc": float(data["test_roc_auc"]),
            "test_balanced_accuracy": float(data["test_balanced_accuracy"]),
            "test_accuracy": float(data["test_accuracy"]),
            "train_roc_auc": float(data["train_roc_auc"]),
            "train_balanced_accuracy": float(data["train_balanced_accuracy"]),
            "train_accuracy": float(data["train_accuracy"]),
            "source": "checkpoint",
        }

    Dtrain = D[np.ix_(tr, tr)]
    Dtest = D[np.ix_(te, tr)]
    ytrain = y[tr]
    ytest = y[te]

    diag = train_distance_diagnostics(Dtrain)
    best = select_hyperparameters(Dtrain, ytrain, outer_fold=fold_id)

    gamma = best["gamma"]
    C = best["C"]

    Ktrain = rbf_from_distance(Dtrain, gamma)
    Ktest = rbf_from_distance(Dtest, gamma)

    clf = SVC(kernel="precomputed", C=C)
    clf.fit(Ktrain, ytrain)

    train_scores = clf.decision_function(Ktrain)
    test_scores = clf.decision_function(Ktest)
    train_pred = clf.predict(Ktrain)
    test_pred = clf.predict(Ktest)

    train_metrics = metric_dict(ytrain, train_scores, train_pred)
    test_metrics = metric_dict(ytest, test_scores, test_pred)

    np.savez_compressed(
        out,
        p=float(p),
        p_index=int(p_index),
        outer_fold=int(fold_id),
        train_idx=tr.astype(np.int32),
        test_idx=te.astype(np.int32),
        y_train=ytrain.astype(np.int8),
        y_test=ytest.astype(np.int8),
        S_train=float(diag["S_train"]),
        median_d=float(diag["median_d"]),
        median_d2=float(diag["median_d2"]),
        gamma_scale_diagnostic=float(diag["gamma_scale_diagnostic"]),
        gamma=float(gamma),
        C=float(C),
        inner_auc=float(best["inner_auc"]),
        inner_gamma_grid=np.asarray(best["inner_gamma_grid"], dtype=np.float64),
        inner_C_grid=np.asarray(best["inner_C_grid"], dtype=np.float64),
        inner_auc_folds=np.asarray(best["inner_auc_folds"], dtype=np.float32),
        inner_auc_mean_surface=np.asarray(best["inner_auc_mean_surface"], dtype=np.float32),
        inner_auc_sd_surface=np.asarray(best["inner_auc_sd_surface"], dtype=np.float32),
        gamma_at_lower_boundary=bool(best["gamma_at_lower_boundary"]),
        gamma_at_upper_boundary=bool(best["gamma_at_upper_boundary"]),
        K_train=Ktrain.astype(KERNEL_DTYPE),
        K_test=Ktest.astype(KERNEL_DTYPE),
        train_scores=np.asarray(train_scores, dtype=np.float64),
        test_scores=np.asarray(test_scores, dtype=np.float64),
        train_pred=np.asarray(train_pred, dtype=np.int8),
        test_pred=np.asarray(test_pred, dtype=np.int8),
        train_roc_auc=train_metrics["roc_auc"],
        train_balanced_accuracy=train_metrics["balanced_accuracy"],
        train_accuracy=train_metrics["accuracy"],
        test_roc_auc=test_metrics["roc_auc"],
        test_balanced_accuracy=test_metrics["balanced_accuracy"],
        test_accuracy=test_metrics["accuracy"],
    )

    return {
        "p": float(p),
        "p_index": int(p_index),
        "outer_fold": int(fold_id),
        "gamma": float(gamma),
        "C": float(C),
        "inner_auc": float(best["inner_auc"]),
        **diag,
        "gamma_at_lower_boundary": bool(best["gamma_at_lower_boundary"]),
        "gamma_at_upper_boundary": bool(best["gamma_at_upper_boundary"]),
        "test_roc_auc": test_metrics["roc_auc"],
        "test_balanced_accuracy": test_metrics["balanced_accuracy"],
        "test_accuracy": test_metrics["accuracy"],
        "train_roc_auc": train_metrics["roc_auc"],
        "train_balanced_accuracy": train_metrics["balanced_accuracy"],
        "train_accuracy": train_metrics["accuracy"],
        "source": "computed",
    }


# %% 10. FULL p x OUTER-FOLD GRID
def run_nested_cv_all_p(
    distances: dict[float, np.ndarray],
    y: np.ndarray,
    outer_splits: list[tuple[np.ndarray, np.ndarray]],
    output_dir: Path,
    skip_existing: bool = True,
) -> pd.DataFrame:
    rows = []
    total = len(SCI.p_values) * len(outer_splits)
    done = 0

    print(
        f"[MODEL] Starting p x outer-fold grid: "
        f"{len(SCI.p_values)} x {len(outer_splits)} = {total} models"
    )

    for p_index, p in enumerate(SCI.p_values):
        print(f"\n[P] {p_index + 1}/{len(SCI.p_values)} | p={p:g}")
        D = distances[p]

        for fold_id, (tr, te) in enumerate(outer_splits, start=1):
            t0 = time.perf_counter()
            row = run_outer_fold(
                D=D,
                y=y,
                p=p,
                p_index=p_index,
                fold_id=fold_id,
                tr=tr,
                te=te,
                output_dir=output_dir,
                skip_existing=skip_existing,
            )
            rows.append(row)
            done += 1

            print(
                f"[FOLD] p={p:g} | {fold_id:03d}/{len(outer_splits)} | "
                f"gamma={row['gamma']:.3g} | C={row['C']:.3g} | "
                f"AUC(test)={row['test_roc_auc']:.3f} | "
                f"{row['source']} | {time.perf_counter() - t0:.1f}s | "
                f"progress={done}/{total}"
            )

    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "fold_summary.csv", index=False)
    print(f"[SAVE] {output_dir / 'fold_summary.csv'}")
    return df


# %% 11. OOF SCORE ASSEMBLY — UNBIASED ONLY
def assemble_oof_scores(
    source_dir: Path,
    save_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Each subject is held out exactly once per repeat and p.

    Read held-out scores from fold checkpoints in source_dir. If save_dir is
    provided, save a long table there. No averaging is performed at this stage.
    """
    rows = []

    for p_index, p in enumerate(SCI.p_values):
        for fold_id in range(1, SCI.n_repeats * SCI.n_splits + 1):
            path = fold_result_path(source_dir, p_index, fold_id)
            if not path.exists():
                continue

            data = np.load(path, allow_pickle=False)
            te = data["test_idx"].astype(int)
            scores = data["test_scores"].astype(float)
            pred = data["test_pred"].astype(int)
            ytest = data["y_test"].astype(int)

            repeat_id = (fold_id - 1) // SCI.n_splits + 1
            fold_within_repeat = (fold_id - 1) % SCI.n_splits + 1

            for subject_idx, yy, score, pp in zip(te, ytest, scores, pred):
                rows.append(
                    {
                        "p": float(p),
                        "p_index": p_index,
                        "outer_fold": fold_id,
                        "repeat": repeat_id,
                        "fold_within_repeat": fold_within_repeat,
                        "subject_idx": int(subject_idx),
                        "y": int(yy),
                        "score": float(score),
                        "prediction": int(pp),
                    }
                )

    df = pd.DataFrame(rows)
    if save_dir is not None and not df.empty:
        save_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(save_dir / "oof_scores_long.csv", index=False)
        print(f"[SAVE] {save_dir / 'oof_scores_long.csv'}")
    return df


# %% 12. REPEAT-LEVEL SUMMARIES AND EXPLORATORY PLOTS
def add_repeat_coordinates(fold_df: pd.DataFrame) -> pd.DataFrame:
    """Attach repeat and fold-within-repeat provenance to fold summaries."""
    df = fold_df.copy()
    f = df["outer_fold"].astype(int).to_numpy()
    df["repeat"] = (f - 1) // SCI.n_splits + 1
    df["fold_within_repeat"] = (f - 1) % SCI.n_splits + 1
    return df


def summarize_repeat_level(fold_df: pd.DataFrame) -> pd.DataFrame:
    """Reduce the five outer folds within each repeated-CV repetition."""
    df = add_repeat_coordinates(fold_df)
    rows = []

    for (p, repeat), g in df.groupby(["p", "repeat"], sort=True):
        gamma = g["gamma"].to_numpy(float)
        med_d2 = g["median_d2"].to_numpy(float)
        gamma_scale = g["gamma_scale_diagnostic"].to_numpy(float)
        q_kernel = gamma * med_d2

        rows.append(
            {
                "p": float(p),
                "repeat": int(repeat),
                "n_outer_folds": int(len(g)),
                # Hyperparameter behaviour: median over the five outer folds.
                "gamma_repeat": float(np.nanmedian(gamma)),
                "log10_gamma_repeat": float(np.nanmedian(np.log10(gamma))),
                "C_repeat": float(np.nanmedian(g["C"].to_numpy(float))),
                "gamma_scale_repeat": float(np.nanmedian(gamma_scale)),
                "median_d2_repeat": float(np.nanmedian(med_d2)),
                # Natural RBF exponent at a typical squared distance.
                "Q_kernel_repeat": float(np.nanmedian(q_kernel)),
                # S is retained only as a diagnostic reference.
                "S_train_repeat": float(np.nanmedian(g["S_train"].to_numpy(float))),
                "inner_auc_repeat": float(np.nanmean(g["inner_auc"].to_numpy(float))),
                "train_auc_repeat": float(np.nanmean(g["train_roc_auc"].to_numpy(float))),
                "test_fold_auc_mean_repeat": float(np.nanmean(g["test_roc_auc"].to_numpy(float))),
                "gamma_lower_boundary_fraction": float(
                    np.mean(g["gamma_at_lower_boundary"].astype(bool))
                ),
                "gamma_upper_boundary_fraction": float(
                    np.mean(g["gamma_at_upper_boundary"].astype(bool))
                ),
            }
        )

    return pd.DataFrame(rows).sort_values(["p", "repeat"]).reset_index(drop=True)


def add_pooled_oof_auc(repeat_df: pd.DataFrame, oof_df: pd.DataFrame) -> pd.DataFrame:
    """Add one unbiased pooled held-out ROC-AUC for each (p, repeat)."""
    out = repeat_df.copy()
    if oof_df.empty:
        out["test_oof_auc_repeat"] = out["test_fold_auc_mean_repeat"]
        return out

    auc_map = {}
    for (p, repeat), g in oof_df.groupby(["p", "repeat"], sort=True):
        y = g["y"].to_numpy(int)
        scores = g["score"].to_numpy(float)
        if np.unique(y).size == 2:
            auc_map[(float(p), int(repeat))] = float(roc_auc_score(y, scores))

    out["test_oof_auc_repeat"] = [
        auc_map.get((float(r.p), int(r.repeat)), np.nan)
        for r in out.itertuples(index=False)
    ]
    return out


def median_iqr(values: np.ndarray) -> tuple[float, float, float]:
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return np.nan, np.nan, np.nan
    q25, med, q75 = np.percentile(vals, [25, 50, 75])
    return float(med), float(q25), float(q75)


def summarize_across_repeats(repeat_df: pd.DataFrame) -> pd.DataFrame:
    """Describe the 20 repeat-level experimental values for each p."""
    cols = [
        "gamma_repeat",
        "log10_gamma_repeat",
        "gamma_scale_repeat",
        "median_d2_repeat",
        "Q_kernel_repeat",
        "S_train_repeat",
        "inner_auc_repeat",
        "train_auc_repeat",
        "test_oof_auc_repeat",
    ]
    rows = []
    for p, g in repeat_df.groupby("p", sort=True):
        row = {"p": float(p), "n_repeats": int(len(g))}
        for col in cols:
            med, q25, q75 = median_iqr(g[col].to_numpy(float))
            row[f"{col}_median"] = med
            row[f"{col}_q25"] = q25
            row[f"{col}_q75"] = q75
        rows.append(row)
    return pd.DataFrame(rows).sort_values("p").reset_index(drop=True)


def _asym_iqr(med: np.ndarray, q25: np.ndarray, q75: np.ndarray) -> np.ndarray:
    return np.vstack([med - q25, q75 - med])


def plot_exploratory(
    fold_df: pd.DataFrame,
    oof_df: pd.DataFrame,
    output_dir: Path,
) -> None:
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    diag_dir = output_dir / "diagnostics"
    diag_dir.mkdir(exist_ok=True)

    repeat_df = summarize_repeat_level(fold_df)
    repeat_df = add_pooled_oof_auc(repeat_df, oof_df)
    p_summary = summarize_across_repeats(repeat_df)

    repeat_df.to_csv(output_dir / "repeat_summary.csv", index=False)
    p_summary.to_csv(output_dir / "summary_by_p_repeat_level.csv", index=False)
    print(f"[SAVE] {output_dir / 'repeat_summary.csv'}")
    print(f"[SAVE] {output_dir / 'summary_by_p_repeat_level.csv'}")

    pvals = p_summary["p"].to_numpy(float)

    # D1. Geometry scale retained as a diagnostic only.
    smed = p_summary["S_train_repeat_median"].to_numpy(float)
    sq25 = p_summary["S_train_repeat_q25"].to_numpy(float)
    sq75 = p_summary["S_train_repeat_q75"].to_numpy(float)
    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    ax.errorbar(pvals, smed, yerr=_asym_iqr(smed, sq25, sq75), marker="o", capsize=3)
    ax.set_xlabel("Schatten p")
    ax.set_ylabel(r"Training distance scale $S=\|D_{train}\|_F$")
    ax.set_title("Diagnostic: geometry scale across p")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(diag_dir / "D01_geometry_scale_vs_p.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    # 2. gamma*(p): 20 repeat-level values, not 100 outer folds.
    grouped = [
        repeat_df.loc[np.isclose(repeat_df["p"], pp), "gamma_repeat"].to_numpy(float)
        for pp in pvals
    ]
    fig, ax = plt.subplots(figsize=(8.0, 5.4))
    positions = np.arange(len(pvals))
    ax.boxplot(grouped, positions=positions, widths=0.55, showfliers=True)
    ax.set_yscale("log")
    ax.set_xticks(positions)
    ax.set_xticklabels([f"{pp:g}" for pp in pvals])
    ax.set_xlabel("Schatten p")
    ax.set_ylabel(r"Repeat-level selected $\gamma^*$")
    ax.set_title(r"Kernel scale selected across geometric deformation")
    ax.grid(alpha=0.2, axis="y")
    fig.tight_layout()
    fig.savefig(output_dir / "02_gamma_vs_p_repeat_boxplots.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    # 3. Main geometry-kernel relation: 9 experimental points, p encoded by color.
    xmed = p_summary["gamma_scale_repeat_median"].to_numpy(float)
    xq25 = p_summary["gamma_scale_repeat_q25"].to_numpy(float)
    xq75 = p_summary["gamma_scale_repeat_q75"].to_numpy(float)
    ymed = p_summary["gamma_repeat_median"].to_numpy(float)
    yq25 = p_summary["gamma_repeat_q25"].to_numpy(float)
    yq75 = p_summary["gamma_repeat_q75"].to_numpy(float)

    fig, ax = plt.subplots(figsize=(7.7, 5.6))
    cmap = plt.cm.viridis
    norm = plt.Normalize(vmin=float(np.min(pvals)), vmax=float(np.max(pvals)))
    for pp, xx, yy, xl, xu, yl, yu in zip(pvals, xmed, ymed, xq25, xq75, yq25, yq75):
        ax.errorbar(
            xx,
            yy,
            xerr=np.array([[xx - xl], [xu - xx]]),
            yerr=np.array([[yy - yl], [yu - yy]]),
            fmt="o",
            capsize=3,
            color=cmap(norm(pp)),
            markersize=7,
        )
    finite = np.isfinite(xmed) & np.isfinite(ymed) & (xmed > 0) & (ymed > 0)
    if np.any(finite):
        lo = min(float(np.min(xmed[finite])), float(np.min(ymed[finite])))
        hi = max(float(np.max(xmed[finite])), float(np.max(ymed[finite])))
        ax.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1.0)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"Natural geometric scale $1/\mathrm{median}(d^2)$")
    ax.set_ylabel(r"Selected absolute $\gamma^*$")
    ax.set_title("Geometry-kernel scale relation")
    ax.grid(alpha=0.2)
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.02)
    cbar.set_label("Schatten p")
    fig.tight_layout()
    fig.savefig(output_dir / "03_gamma_vs_scale_repeat_uncertainty.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    # 4. Natural RBF exponent Q = gamma*median(d^2), repeat-level median + IQR.
    qmed = p_summary["Q_kernel_repeat_median"].to_numpy(float)
    qq25 = p_summary["Q_kernel_repeat_q25"].to_numpy(float)
    qq75 = p_summary["Q_kernel_repeat_q75"].to_numpy(float)
    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    ax.errorbar(pvals, qmed, yerr=_asym_iqr(qmed, qq25, qq75), marker="o", capsize=3)
    ax.set_xlabel("Schatten p")
    ax.set_ylabel(r"$Q(p)=\gamma^*\,\mathrm{median}(d_p^2)$")
    ax.set_title("Effective RBF exponent at a typical distance")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_dir / "04_effective_rbf_exponent_Q_vs_p.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    # 5. Prediction: one value per repeated CV repetition.
    trmed = p_summary["train_auc_repeat_median"].to_numpy(float)
    trq25 = p_summary["train_auc_repeat_q25"].to_numpy(float)
    trq75 = p_summary["train_auc_repeat_q75"].to_numpy(float)
    temed = p_summary["test_oof_auc_repeat_median"].to_numpy(float)
    teq25 = p_summary["test_oof_auc_repeat_q25"].to_numpy(float)
    teq75 = p_summary["test_oof_auc_repeat_q75"].to_numpy(float)

    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    ax.errorbar(
        pvals, trmed, yerr=_asym_iqr(trmed, trq25, trq75),
        marker="o", capsize=3, label="Training (diagnostic)"
    )
    ax.errorbar(
        pvals, temed, yerr=_asym_iqr(temed, teq25, teq75),
        marker="s", capsize=3, label="Held-out OOF"
    )
    ax.set_xlabel("Schatten p")
    ax.set_ylabel("ROC-AUC")
    ax.set_title("Decision-space behaviour across p")
    ax.grid(alpha=0.2)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_dir / "05_train_test_auc_repeat_level.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    print("[PLOT] Repeat-level exploratory figures saved")

# %% 13. INNER-CV LANDSCAPE DIAGNOSTICS
def analyze_inner_cv_landscapes(source_dir: Path, output_dir: Path, near_optimal_tol: float = 0.005) -> None:
    """Aggregate and visualize the complete inner-CV AUC(C, gamma) landscapes.

    For each p and outer fold, the gamma profile is obtained by maximizing the
    mean inner-CV AUC over C.  A gamma value is called near-optimal when its
    profiled AUC lies within `near_optimal_tol` ROC-AUC units of the fold's best
    value.  This is descriptive and does not alter model selection.
    """
    import matplotlib.pyplot as plt

    diag_dir = output_dir / "diagnostics" / "inner_cv_landscape"
    diag_dir.mkdir(parents=True, exist_ok=True)

    surface_records = []
    profile_records = []
    span_records = []

    for p_index, pval in enumerate(SCI.p_values):
        fold_surfaces = []
        fold_profiles = []
        gamma_grid = None
        C_grid = None

        for fold_id in range(1, SCI.n_repeats * SCI.n_splits + 1):
            path = fold_result_path(source_dir, p_index, fold_id)
            if not path.exists():
                continue
            data = np.load(path, allow_pickle=False)
            required = {"inner_gamma_grid", "inner_C_grid", "inner_auc_mean_surface"}
            if not required.issubset(set(data.files)):
                raise RuntimeError(
                    f"Checkpoint {path} lacks v4 inner-CV surfaces. "
                    "A full v4 nested-CV rerun is required."
                )

            gammas = data["inner_gamma_grid"].astype(float)
            Cs = data["inner_C_grid"].astype(float)
            surf = data["inner_auc_mean_surface"].astype(float)
            gamma_grid = gammas
            C_grid = Cs
            fold_surfaces.append(surf)

            profile = np.nanmax(surf, axis=1)
            fold_profiles.append(profile)
            best = float(np.nanmax(profile))
            delta = best - profile
            near = delta <= float(near_optimal_tol) + 1e-12
            if np.any(near):
                logs = np.log10(gammas[near])
                span = float(np.max(logs) - np.min(logs))
                n_near = int(np.sum(near))
            else:
                span = 0.0
                n_near = 0

            repeat_id = (fold_id - 1) // SCI.n_splits + 1
            fold_within_repeat = (fold_id - 1) % SCI.n_splits + 1
            span_records.append({
                "p": float(pval),
                "outer_fold": int(fold_id),
                "repeat": int(repeat_id),
                "fold_within_repeat": int(fold_within_repeat),
                "best_inner_auc": best,
                "near_optimal_tol_auc": float(near_optimal_tol),
                "near_optimal_gamma_log10_span": span,
                "near_optimal_gamma_n_grid_points": n_near,
                "selected_gamma": float(data["gamma"]),
                "selected_C": float(data["C"]),
            })

            for ig, gg in enumerate(gammas):
                profile_records.append({
                    "p": float(pval),
                    "outer_fold": int(fold_id),
                    "repeat": int(repeat_id),
                    "gamma": float(gg),
                    "profile_auc_max_over_C": float(profile[ig]),
                    "delta_from_fold_best_auc": float(delta[ig]),
                })

        if not fold_surfaces:
            continue

        stack = np.stack(fold_surfaces, axis=0)
        med_surface = np.nanmedian(stack, axis=0)
        q25_surface = np.nanpercentile(stack, 25, axis=0)
        q75_surface = np.nanpercentile(stack, 75, axis=0)

        for ig, gg in enumerate(gamma_grid):
            for ic, cc in enumerate(C_grid):
                surface_records.append({
                    "p": float(pval),
                    "gamma": float(gg),
                    "C": float(cc),
                    "inner_auc_median_outer_folds": float(med_surface[ig, ic]),
                    "inner_auc_q25_outer_folds": float(q25_surface[ig, ic]),
                    "inner_auc_q75_outer_folds": float(q75_surface[ig, ic]),
                })

        # One heatmap per p: median inner-CV AUC over the 100 outer training sets.
        fig, ax = plt.subplots(figsize=(7.2, 5.4))
        im = ax.imshow(
            med_surface.T,
            origin="lower",
            aspect="auto",
            interpolation="nearest",
            extent=[np.log10(gamma_grid[0]), np.log10(gamma_grid[-1]),
                    np.log10(C_grid[0]), np.log10(C_grid[-1])],
        )
        ax.set_xlabel(r"$\log_{10}\gamma$")
        ax.set_ylabel(r"$\log_{10}C$")
        ax.set_title(f"Median inner-CV ROC-AUC surface | p={pval:g}")
        cbar = fig.colorbar(im, ax=ax, pad=0.02)
        cbar.set_label("Median inner-CV ROC-AUC")
        fig.tight_layout()
        fig.savefig(diag_dir / f"surface_median_p{p_index:02d}_p{pval:g}.png", dpi=220, bbox_inches="tight")
        plt.close(fig)

        # Profile over gamma after maximizing over C: median + IQR across outer folds.
        prof_stack = np.stack(fold_profiles, axis=0)
        pmed = np.nanmedian(prof_stack, axis=0)
        pq25 = np.nanpercentile(prof_stack, 25, axis=0)
        pq75 = np.nanpercentile(prof_stack, 75, axis=0)
        fig, ax = plt.subplots(figsize=(7.2, 5.0))
        x = np.log10(gamma_grid)
        ax.plot(x, pmed, marker="o", markersize=3)
        ax.fill_between(x, pq25, pq75, alpha=0.2)
        ax.set_xlabel(r"$\log_{10}\gamma$")
        ax.set_ylabel("Inner-CV ROC-AUC maximized over C")
        ax.set_title(f"Gamma identifiability profile | p={pval:g}")
        ax.grid(alpha=0.2)
        fig.tight_layout()
        fig.savefig(diag_dir / f"gamma_profile_p{p_index:02d}_p{pval:g}.png", dpi=220, bbox_inches="tight")
        plt.close(fig)

    surface_df = pd.DataFrame(surface_records)
    profile_df = pd.DataFrame(profile_records)
    span_df = pd.DataFrame(span_records)
    surface_df.to_csv(diag_dir / "inner_cv_surface_median_iqr.csv", index=False)
    profile_df.to_csv(diag_dir / "inner_cv_gamma_profile_foldwise.csv", index=False)
    span_df.to_csv(diag_dir / "inner_cv_gamma_near_optimal_span_foldwise.csv", index=False)

    if not span_df.empty:
        # First reduce the five folds within each repetition, then summarize 20 repeats.
        rep = (
            span_df.groupby(["p", "repeat"], as_index=False)
            .agg(
                gamma_log10_span_repeat=("near_optimal_gamma_log10_span", "median"),
                best_inner_auc_repeat=("best_inner_auc", "mean"),
            )
        )
        rows = []
        for pp, g in rep.groupby("p", sort=True):
            med, q25, q75 = median_iqr(g["gamma_log10_span_repeat"].to_numpy(float))
            rows.append({"p": float(pp), "median_log10_span": med, "q25_log10_span": q25, "q75_log10_span": q75})
        span_summary = pd.DataFrame(rows)
        span_summary.to_csv(diag_dir / "inner_cv_gamma_near_optimal_span_repeat_summary.csv", index=False)

        fig, ax = plt.subplots(figsize=(7.5, 5.0))
        pp = span_summary["p"].to_numpy(float)
        med = span_summary["median_log10_span"].to_numpy(float)
        q25 = span_summary["q25_log10_span"].to_numpy(float)
        q75 = span_summary["q75_log10_span"].to_numpy(float)
        ax.errorbar(pp, med, yerr=_asym_iqr(med, q25, q75), marker="o", capsize=3)
        ax.set_xlabel("Schatten p")
        ax.set_ylabel(r"Near-optimal $\gamma$ span (decades)")
        ax.set_title(f"Inner-CV gamma identifiability | tolerance={near_optimal_tol:.3f} AUC")
        ax.grid(alpha=0.2)
        fig.tight_layout()
        fig.savefig(diag_dir / "gamma_near_optimal_span_vs_p.png", dpi=220, bbox_inches="tight")
        plt.close(fig)

    print(f"[INNER-CV] Landscape diagnostics saved -> {diag_dir}")


# %% 14. BOUNDARY AUDIT
def gamma_boundary_audit(fold_df: pd.DataFrame) -> None:
    lower = fold_df["gamma_at_lower_boundary"].astype(bool)
    upper = fold_df["gamma_at_upper_boundary"].astype(bool)
    frac = float(np.mean(lower | upper))

    print(
        f"[GAMMA] Boundary selections: {(lower | upper).sum()}/{len(fold_df)} "
        f"({100 * frac:.1f}%)"
    )

    if frac > 0.05:
        print(
            "[GAMMA][WARNING] More than 5% of outer-fold optima hit a gamma "
            "boundary. Inspect summary_by_p_repeat_level.csv before scientific interpretation; "
            "the absolute gamma grid may need to be extended."
        )


# %% 14. SAVE CONFIGURATION / MANIFEST
def save_run_metadata(
    output_dir: Path,
    args: argparse.Namespace,
    manifest: pd.DataFrame,
    density_stats: pd.DataFrame,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "scientific_config": asdict(SCI),
        "compute_config": asdict(COMP),
        "runtime_arguments": {
            k: str(v) if isinstance(v, Path) else v
            for k, v in vars(args).items()
        },
        "important_note": (
            "gamma_values is an absolute grid shared across all p and folds. "
            "gamma_scale_diagnostic=1/median(d_train^2) is saved for comparison "
            "only and is not used for hyperparameter-grid construction."
        ),
    }

    (output_dir / "config.json").write_text(json.dumps(payload, indent=2))
    manifest.to_csv(output_dir / "retained_manifest.csv", index=False)
    density_stats.to_csv(output_dir / "density_stats.csv", index=False)

    print(f"[SAVE] {output_dir / 'config.json'}")
    print(f"[SAVE] {output_dir / 'retained_manifest.csv'}")


# %% 15. COMPLETE WORKFLOW
def configure_science_from_args(args: argparse.Namespace) -> None:
    global SCI
    if args.positive_label == args.negative_label:
        raise ValueError("Positive and negative labels must differ")
    SCI = ScientificConfig(
        p_values=tuple(float(x) for x in args.p_values),
        C_values=tuple(float(x) for x in args.c_values),
        gamma_values=tuple(float(x) for x in args.gamma_values),
        n_repeats=int(args.n_repeats),
        n_splits=int(args.n_splits),
        inner_splits=int(args.inner_splits),
        random_state=int(args.random_state),
        positive_label=str(args.positive_label),
        negative_label=str(args.negative_label),
    )

def run_experiment(args: argparse.Namespace) -> None:
    configure_science_from_args(args)
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print("[START] Schatten-p -> gamma -> kernel -> decision dynamics")
    print(f"        data={args.data_dir.expanduser().resolve()}")
    print(f"        output={output_dir}")
    print(f"        p grid={SCI.p_values}")
    print(f"        absolute gamma grid={SCI.gamma_values}")
    print(
        f"        CV={SCI.n_repeats}x{SCI.n_splits} outer, "
        f"{SCI.inner_splits}-fold inner"
    )

    manifest, R, density_stats = build_density_stack(
        data_dir=args.data_dir.expanduser().resolve(),
        info_file=args.info_file.expanduser().resolve(),
        matrix_glob=args.matrix_glob,
        matrix_format=args.matrix_format,
        id_col=args.id_col,
        label_col=args.label_col,
        delimiter=args.delimiter,
        diagonal=args.diagonal,
        symmetrize=not args.no_symmetrize,
        psd_tol=args.psd_tol,
        expected_nodes=args.expected_nodes,
    )

    y = encode_labels(manifest, args.label_col)
    save_run_metadata(output_dir, args, manifest, density_stats)

    spectra_path = output_dir / "pairwise_singular_values.npy"
    pairs_path = output_dir / "pair_indices.npz"

    source_spectra_path = spectra_path
    source_pairs_path = pairs_path
    if args.spectra_source is not None:
        src = args.spectra_source.expanduser().resolve()
        source_spectra_path = src / "pairwise_singular_values.npy"
        source_pairs_path = src / "pair_indices.npz"

    if (args.reuse_spectra or args.spectra_source is not None) and source_spectra_path.exists() and source_pairs_path.exists():
        print(f"[GEOM] Reusing cached pairwise spectra from {source_spectra_path.parent}")
        singvals = np.load(source_spectra_path)
        pair_data = np.load(source_pairs_path)
        pair_s = pair_data["pair_s"]
        pair_t = pair_data["pair_t"]
        if args.cache_spectra and source_spectra_path.resolve() != spectra_path.resolve():
            np.save(spectra_path, singvals)
            np.savez_compressed(pairs_path, pair_s=pair_s, pair_t=pair_t)
            print(f"[SAVE] Pairwise spectra copied -> {spectra_path}")
    else:
        pair_s, pair_t, singvals = pairwise_singular_values(
            R,
            show_progress=COMP.show_progress,
        )
        if COMP.save_pairwise_spectra or args.cache_spectra:
            np.save(spectra_path, singvals)
            np.savez_compressed(pairs_path, pair_s=pair_s, pair_t=pair_t)
            print(f"[SAVE] Pairwise spectra -> {spectra_path}")

    distances = build_all_distance_matrices(
        singvals,
        pair_s,
        pair_t,
        len(R),
    )

    if COMP.save_distance_matrices:
        dist_dir = output_dir / "distances"
        dist_dir.mkdir(exist_ok=True)
        for p_index, p in enumerate(SCI.p_values):
            np.save(dist_dir / f"D_p{p_index:02d}_p{p:g}.npy", distances[p])
        print(f"[SAVE] Distance matrices -> {dist_dir}")

    outer_splits = make_outer_splits(y)

    fold_df = run_nested_cv_all_p(
        distances=distances,
        y=y,
        outer_splits=outer_splits,
        output_dir=output_dir,
        skip_existing=(COMP.skip_existing and not args.no_skip_existing),
    )

    gamma_boundary_audit(fold_df)
    oof_df = assemble_oof_scores(output_dir, save_dir=output_dir)
    plot_exploratory(fold_df, oof_df, output_dir)
    analyze_inner_cv_landscapes(output_dir, output_dir)

    print("[DONE] geometry-to-decision experiment completed")


# %% 16. PLOT-ONLY / REASSEMBLY WORKFLOW
def load_fold_summaries_from_checkpoints(output_dir: Path) -> pd.DataFrame:
    rows = []
    for p_index, p in enumerate(SCI.p_values):
        for fold_id in range(1, SCI.n_repeats * SCI.n_splits + 1):
            path = fold_result_path(output_dir, p_index, fold_id)
            if not path.exists():
                continue
            data = np.load(path, allow_pickle=False)
            rows.append(
                {
                    "p": float(data["p"]),
                    "p_index": int(data["p_index"]),
                    "outer_fold": int(data["outer_fold"]),
                    "gamma": float(data["gamma"]),
                    "C": float(data["C"]),
                    "inner_auc": float(data["inner_auc"]),
                    "S_train": float(data["S_train"]),
                    "median_d": float(data["median_d"]),
                    "median_d2": float(data["median_d2"]),
                    "gamma_scale_diagnostic": float(data["gamma_scale_diagnostic"]),
                    "gamma_at_lower_boundary": bool(data["gamma_at_lower_boundary"]),
                    "gamma_at_upper_boundary": bool(data["gamma_at_upper_boundary"]),
                    "test_roc_auc": float(data["test_roc_auc"]),
                    "test_balanced_accuracy": float(data["test_balanced_accuracy"]),
                    "test_accuracy": float(data["test_accuracy"]),
                    "train_roc_auc": float(data["train_roc_auc"]),
                    "train_balanced_accuracy": float(data["train_balanced_accuracy"]),
                    "train_accuracy": float(data["train_accuracy"]),
                    "source": "checkpoint",
                }
            )

    return pd.DataFrame(rows)


def run_plot_only(output_dir: Path, analysis_source: Optional[Path] = None) -> None:
    output_dir = output_dir.expanduser().resolve()
    source_dir = (analysis_source or output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    fold_df = load_fold_summaries_from_checkpoints(source_dir)
    if fold_df.empty:
        raise RuntimeError(f"No fold checkpoints found in {source_dir}")

    fold_df.to_csv(output_dir / "fold_summary.csv", index=False)
    gamma_boundary_audit(fold_df)
    oof_df = assemble_oof_scores(source_dir, save_dir=output_dir)
    plot_exploratory(fold_df, oof_df, output_dir)
    analyze_inner_cv_landscapes(source_dir, output_dir)
    print(f"[DONE] Analysis-only workflow completed from {source_dir}")


# %% 17. COMMAND-LINE INTERFACE
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "AD real-data Schatten-p geometry -> absolute gamma -> kernel -> "
            "decision exploratory experiment (v4 dense-gamma + inner-CV landscape analysis)"
        )
    )

    p.add_argument("--data-dir", required=False, type=Path)
    p.add_argument("--info-file", required=False, type=Path)
    p.add_argument("--output-dir", required=True, type=Path)

    p.add_argument("--matrix-glob", default="adjacency_*.csv")
    p.add_argument("--matrix-format", default="auto", choices=["auto", "csv", "npy"])
    p.add_argument("--id-col", default=None)
    p.add_argument("--label-col", default="class")
    p.add_argument("--positive-label", default="POSITIVE")
    p.add_argument("--negative-label", default="NEGATIVE")
    p.add_argument("--p-values", nargs="+", type=float, default=list(ScientificConfig().p_values))
    p.add_argument("--n-repeats", type=int, default=20)
    p.add_argument("--n-splits", type=int, default=5)
    p.add_argument("--inner-splits", type=int, default=5)
    p.add_argument("--random-state", type=int, default=42)
    p.add_argument("--c-values", nargs="+", type=float, default=list(ScientificConfig().C_values))
    p.add_argument("--gamma-values", nargs="+", type=float, default=list(ScientificConfig().gamma_values))
    p.add_argument("--delimiter", default=None)
    p.add_argument("--diagonal", default="unit", choices=["unit", "keep"])
    p.add_argument("--no-symmetrize", action="store_true")
    p.add_argument("--psd-tol", default=1e-8, type=float)
    p.add_argument("--expected-nodes", default=None, type=int)

    p.add_argument(
        "--cache-spectra",
        action="store_true",
        help="Persist pairwise singular values for future reruns",
    )
    p.add_argument(
        "--reuse-spectra",
        action="store_true",
        help="Reuse cached pairwise spectra if present in --output-dir",
    )
    p.add_argument(
        "--spectra-source",
        default=None,
        type=Path,
        help=(
            "Optional previous results directory containing pairwise_singular_values.npy "
            "and pair_indices.npz. This reuses geometry preprocessing while the v4 nested CV "
            "is recomputed with the denser absolute gamma grid."
        ),
    )
    p.add_argument(
        "--no-skip-existing",
        action="store_true",
        help="Recompute existing (p, outer-fold) checkpoints",
    )
    p.add_argument(
        "--plot-only",
        action="store_true",
        help="Regenerate summaries/plots from existing fold checkpoints",
    )
    p.add_argument(
        "--analysis-source",
        default=None,
        type=Path,
        help=(
            "Optional directory containing existing fold checkpoints. With "
            "--plot-only, read models from this directory and write the new "
            "v4 summaries/plots to --output-dir."
        ),
    )

    args = p.parse_args()

    if not args.plot_only:
        if args.data_dir is None or args.info_file is None:
            p.error("--data-dir and --info-file are required unless --plot-only is used")

    return args


# %% 18. SHELL ENTRYPOINT
if __name__ == "__main__":
    ARGS = parse_args()
    if ARGS.plot_only:
        run_plot_only(ARGS.output_dir, ARGS.analysis_source)
    else:
        run_experiment(ARGS)
