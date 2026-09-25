#!/usr/bin/env python3
# %%
"""AD Schatten-p XAI propagation analysis v1.

Scientific aim
--------------
Complete the propagation chain

    p -> spectral concentration -> D_p -> K_p -> f_p -> I_p

and test whether geometry dependence is amplified when moving from held-out
SVM decisions to perturbation-based explanations.

The classifier is NOT re-tuned. This script reuses the outer-fold models
selected by ad_schatten_gamma_dynamics_v4.py and reconstructs the same SVCs
from saved K_train, y_train, C and gamma.

For every held-out subject s, node r and Schatten p, importance is

    I_{r,p}(s) = |f_p(s) - f_p^{(-r)}(s)|,

where node suppression is propagated to the Schatten distance through a
first-order local sensitivity. The same first-order construction is used for
all p so the continuous p comparison is methodologically uniform.

For Delta = rho_s-rho_t, d_p=||Delta||_p and a row/column suppression of node r,
the first-order contribution is

    c_r = <grad ||Delta||_p, M_r(Delta)>,

where M_r keeps row r and column r (counting the diagonal once). For symmetric
Delta=U diag(lambda) U^T this can be computed for ALL nodes without forming the
full gradient matrix:

    c_r = 2 diag(G_p Delta)_r - diag(G_p)_r Delta_rr,

    G_p = U diag(sign(lambda)|lambda|^(p-1)/d_p^(p-1)) U^T.

At p=1 this reduces to the trace-norm subgradient used in the previous work
(without the old 1/2 trace-distance convention, because the present family
uses the Schatten-1 norm consistently).

Primary outputs
---------------
00_excluded_matrices.csv                matrices excluded by v4-compatible QC
00_retained_after_qc.csv                  retained subjects after XAI-side QC
00_xai_repeat_tensor.npz                 complete OOF importance tensor
01_subject_xai_profiles.npz              subject median profiles across repeats
02_population_importance_profiles.csv    global/AD/NC profiles and ranks
03_xai_concordance_vs_p1.csv              subject-level explanation concordance
04_decision_vs_xai_concordance.csv        direct decision/XAI comparison
05_xai_heatmap_node_by_p.png/pdf          HOW explanations reorganize
06_top_node_rank_trajectories.png/pdf     HOW leading explanations reorder
07_subject_xai_concordance_vs_p1.png/pdf  HOW MUCH XAI changes
08_decision_vs_xai_dependence.png/pdf     amplification test
09_class_specific_xai_profiles.png/pdf    AD/NC explanatory structure

a diagnostics/ folder also contains validation of the first-order p=2
approximation against the exact Frobenius leave-node-out distance on sampled
subject pairs.

Computational note
------------------
This engine uses a streaming implementation designed for large datasets. It
never materializes the full (p x subject-pair x node) contribution tensor.
Instead, it reconstructs the exact outer-fold SVC support vectors once, streams
subject pairs in chunks, computes each pair eigensystem at most once, and
accumulates only the support-vector-weighted decision perturbations required by
the XAI definition. This preserves the scientific quantity I_{r,p}(s) while
keeping memory and disk use bounded.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.linalg import eigh, eigvalsh
from scipy.stats import pearsonr, spearmanr
from sklearn.svm import SVC
from threadpoolctl import threadpool_limits
from tqdm.auto import tqdm

# %% 1. SCIENTIFIC CONFIGURATION
P_VALUES = np.array([1.0, 1.25, 1.5, 1.75, 2.0, 3.0, 4.0, 8.0, 16.0], dtype=float)
N_REPEATS = 20
N_SPLITS = 5
REFERENCE_P = 1.0
NEGATIVE_LABEL = "NC"
POSITIVE_LABEL = "AD"
CONTRIB_DTYPE = np.float32
IMPORTANCE_DTYPE = np.float32


# %% 2. CLI
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Propagate Schatten-p geometry from decisions to perturbation XAI.")
    p.add_argument("--results-source", required=True, type=Path,
                   help="ad_schatten_gamma_results_v4 directory.")
    p.add_argument("--data-dir", required=False, type=Path)
    p.add_argument("--factors-file", required=True, type=Path,
                   help="PEMS low-rank factors.npz containing factors[subject,node,rank].")
    p.add_argument("--precomputed-distance-dir", required=True, type=Path,
                   help="Directory containing the exact distance_p*.npy matrices used by the classifier.")
    p.add_argument("--info-file", required=True, type=Path)
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--matrix-glob", default="adjacency_*.csv")
    p.add_argument("--matrix-format", default="auto", choices=["auto", "csv", "npy"])
    p.add_argument("--id-col", default=None)
    p.add_argument("--label-col", default="class")
    p.add_argument("--positive-label", default="POSITIVE")
    p.add_argument("--negative-label", default="NEGATIVE")
    p.add_argument("--p-values", nargs="+", type=float, default=P_VALUES.tolist())
    p.add_argument("--reference-p", type=float, default=1.0)
    p.add_argument("--n-repeats", type=int, default=20)
    p.add_argument("--n-splits", type=int, default=5)
    p.add_argument("--delimiter", default=None)
    p.add_argument("--diagonal", default="unit", choices=["unit", "keep"])
    p.add_argument("--no-symmetrize", action="store_true")
    p.add_argument("--psd-tol", default=1e-8, type=float)
    p.add_argument("--expected-nodes", default=None, type=int)
    p.add_argument("--reuse-contributions", action="store_true",
                   help="Legacy compatibility flag. The streaming engine does not build the old full contribution cache.")
    p.add_argument("--xai-pair-chunk-size", default=4096, type=int,
                   help="Number of subject pairs processed per streaming accumulation chunk.")
    p.add_argument("--distance-chunk-size", default=50000, type=int,
                   help="Number of cached pair spectra read at once when reconstructing exact distance matrices.")
    p.add_argument("--xai-parallel", default="auto", choices=["auto", "serial", "threads"],
                   help="Pairwise contribution backend. auto benchmarks this machine before the full run.")
    p.add_argument("--xai-workers", default=0, type=int,
                   help="Worker threads for pairwise XAI geometry. 0 lets auto choose candidates from CPU count.")
    p.add_argument("--xai-autotune-pairs", default=48, type=int,
                   help="Number of pairs used by auto backend/worker calibration.")
    p.add_argument("--top-n-ranks", default=20, type=int)
    p.add_argument("--validation-pairs", default=300, type=int,
                   help="Sampled pairs for p=2 first-order vs exact Frobenius validation.")
    p.add_argument("--random-state", default=42, type=int)
    p.add_argument("--dpi", default=220, type=int)
    return p.parse_args()


# %% 3. I/O AND DENSITY CONSTRUCTION (same conventions as v4)
def natural_sort_key(path: Path) -> Tuple:
    parts = re.split(r"(\d+)", path.name)
    return tuple(int(x) if x.isdigit() else x.lower() for x in parts)


def read_info(path: Path, delimiter: Optional[str]) -> pd.DataFrame:
    if delimiter is None:
        return pd.read_csv(path, sep=None, engine="python")
    return pd.read_csv(path, sep=delimiter)


def infer_format(path: Path, fmt: str) -> str:
    if fmt != "auto":
        return fmt
    if path.suffix.lower() == ".npy":
        return "npy"
    if path.suffix.lower() in {".csv", ".txt", ".tsv"}:
        return "csv"
    raise ValueError(f"Cannot infer matrix format from {path}")


def load_matrix(path: Path, fmt: str) -> np.ndarray:
    """Load one matrix using the same hard-QC convention as gamma-dynamics v4.

    Invalid shape and non-finite entries are raised here and caught by
    build_density_stack(), which records and excludes the corresponding subject.
    """
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


def preprocess_matrix(A: np.ndarray, diagonal: str, symmetrize: bool) -> np.ndarray:
    """Apply exactly the matrix preprocessing conventions used in v4."""
    A = np.asarray(A, dtype=np.float64).copy()
    if diagonal == "unit":
        np.fill_diagonal(A, 1.0)
    elif diagonal != "keep":
        raise ValueError("diagonal must be 'unit' or 'keep'")
    if symmetrize:
        A = 0.5 * (A + A.T)
    return np.ascontiguousarray(A, dtype=np.float64)


def _manifest_identity(df: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in ["sample_id", "matrix_file"] if c in df.columns]
    if not cols:
        raise ValueError("Manifest contains neither sample_id nor matrix_file")
    return df[cols].astype(str).reset_index(drop=True)


def verify_against_v4_manifest(source: Path, retained: pd.DataFrame) -> None:
    """Require exact subject identity/order agreement with the v4 checkpoints.

    The XAI code indexes the v4 pairwise spectra and outer-fold checkpoints by
    subject position. Merely obtaining the same number of subjects is therefore
    insufficient: identity and order must be identical.
    """
    ref_path = source / "retained_manifest.csv"
    if not ref_path.exists():
        raise FileNotFoundError(
            f"Missing {ref_path}. The XAI run requires the v4 retained_manifest.csv "
            "to verify subject identity/order before reusing v4 checkpoints."
        )
    ref = pd.read_csv(ref_path)
    a = _manifest_identity(ref)
    b = _manifest_identity(retained)
    if len(a) != len(b):
        raise ValueError(
            f"XAI QC retained {len(b)} subjects, but v4 retained {len(a)}. "
            "The subject sets differ, so v4 checkpoints cannot be reused safely."
        )
    if list(a.columns) != list(b.columns):
        common = [c for c in ["sample_id", "matrix_file"] if c in a.columns and c in b.columns]
        a, b = a[common], b[common]
    mismatch = np.where(np.any(a.to_numpy() != b.to_numpy(), axis=1))[0]
    if mismatch.size:
        i = int(mismatch[0])
        raise ValueError(
            "Subject identity/order mismatch with v4 retained_manifest.csv at "
            f"row {i}: v4={a.iloc[i].to_dict()} vs XAI={b.iloc[i].to_dict()}. "
            "Stopping before XAI because checkpoint indices would be invalid."
        )
    print(f"[QC] Exact identity/order match with v4 retained_manifest.csv: {len(b)} subjects")


def build_density_stack(args: argparse.Namespace) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Reconstruct the density stack with v4-compatible exclusion/QC.

    Candidate AD/NC subjects whose matrices are corrupt or scientifically invalid
    are recorded in 00_excluded_matrices.csv and skipped. The final retained list
    is then required to match the v4 retained manifest exactly.
    """
    files = sorted(args.data_dir.expanduser().resolve().glob(args.matrix_glob), key=natural_sort_key)
    if not files:
        raise FileNotFoundError(f"No matrices found in {args.data_dir} with {args.matrix_glob!r}")
    info = read_info(args.info_file.expanduser().resolve(), args.delimiter)
    if len(info) != len(files):
        raise ValueError(
            f"info rows={len(info)} but matrices={len(files)}; exact row/file alignment is required"
        )

    manifest = info.copy()
    manifest["matrix_file"] = [x.name for x in files]
    manifest["matrix_path"] = [str(x) for x in files]
    manifest["sample_id"] = (
        manifest[args.id_col].astype(str)
        if args.id_col and args.id_col in manifest.columns
        else [x.stem for x in files]
    )
    labels = manifest[args.label_col].astype(str).str.strip()
    manifest = manifest.loc[labels.isin([args.negative_label, args.positive_label])].reset_index(drop=True)
    if manifest.empty:
        raise ValueError("No samples remain after binary label filtering")

    densities: list[np.ndarray] = []
    kept: list[dict] = []
    excluded: list[dict] = []
    target_shape = None

    iterator = tqdm(manifest.iterrows(), total=len(manifest), desc="Density operators", leave=False)
    for _, row in iterator:
        path = Path(row["matrix_path"])
        try:
            A_raw = load_matrix(path, args.matrix_format)

            if args.expected_nodes is not None:
                wanted = (args.expected_nodes, args.expected_nodes)
                if A_raw.shape != wanted:
                    raise ValueError(f"shape={A_raw.shape}; expected {wanted}")

            if target_shape is None:
                target_shape = A_raw.shape
            if A_raw.shape != target_shape:
                raise ValueError(f"shape={A_raw.shape}; first retained shape={target_shape}")

            A = preprocess_matrix(A_raw, diagonal=args.diagonal, symmetrize=not args.no_symmetrize)
            evals = eigvalsh(A, check_finite=False)
            min_eval = float(evals.min())
            if min_eval < -args.psd_tol:
                raise ValueError(f"not PSD: min eigenvalue={min_eval:.3e}")

            tr = float(np.trace(A))
            if not np.isfinite(tr) or abs(tr) <= args.psd_tol:
                raise ValueError(f"invalid trace={tr}")

            densities.append(A / tr)
            kept.append(row.to_dict())
        except Exception as exc:
            excluded.append({
                "sample_id": row.get("sample_id", path.stem),
                "matrix_file": path.name,
                "label": row.get(args.label_col, ""),
                "reason": str(exc),
            })

    if not densities:
        raise RuntimeError("No valid density operators were constructed")

    retained = pd.DataFrame(kept).reset_index(drop=True)
    R = np.stack(densities).astype(np.float64, copy=False)
    lab = retained[args.label_col].astype(str).str.strip().to_numpy()
    y = np.where(lab == args.positive_label, 1, 0).astype(np.int8)

    out = args.output_dir.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    excluded_df = pd.DataFrame(excluded, columns=["sample_id", "matrix_file", "label", "reason"])
    excluded_df.to_csv(out / "00_excluded_matrices.csv", index=False)
    retained[["sample_id", args.label_col, "matrix_file", "matrix_path"]].to_csv(
        out / "00_retained_after_qc.csv", index=False
    )

    print(f"[DATA] Candidate binary-class subjects: {len(manifest)}")
    print(f"[DATA] Excluded subjects: {len(excluded_df)}")
    if len(excluded_df):
        print("[DATA] Exclusion reasons:")
        for reason, n in excluded_df["reason"].value_counts().items():
            print(f"       {n:3d}  {reason}")
    print(f"[DATA] retained={len(y)} | {args.negative_label}={(y==0).sum()} | {args.positive_label}={(y==1).sum()} | nodes={R.shape[1]}")

    verify_against_v4_manifest(args.results_source.expanduser().resolve(), retained)
    return retained, R, y



def build_pems_lowrank_stack(args: argparse.Namespace) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Load the frozen PEMS full-sensor low-rank representation without rebuilding dense matrices."""
    info = read_info(args.info_file.expanduser().resolve(), args.delimiter).copy()
    labels = info[args.label_col].astype(str).str.strip()
    keep = labels.isin([str(args.negative_label), str(args.positive_label)])
    manifest = info.loc[keep].reset_index(drop=True)
    if manifest.empty:
        raise ValueError("No PEMS subjects remain after binary-label filtering")

    if args.id_col and args.id_col in manifest.columns:
        manifest["sample_id"] = manifest[args.id_col].astype(str)
    elif "sample_id" not in manifest.columns:
        manifest["sample_id"] = [f"PEMS_{i+1:04d}" for i in range(len(manifest))]

    y = np.where(
        manifest[args.label_col].astype(str).str.strip().to_numpy() == str(args.positive_label),
        1, 0
    ).astype(int)

    z = np.load(args.factors_file.expanduser().resolve())
    F = np.asarray(z["factors"], dtype=np.float64)
    if F.ndim != 3:
        raise ValueError(f"Expected factors[subject,node,rank], got {F.shape}")
    if F.shape[0] != len(manifest):
        raise ValueError(f"Factor subjects={F.shape[0]} but manifest subjects={len(manifest)}")
    if args.expected_nodes is not None and F.shape[1] != int(args.expected_nodes):
        raise ValueError(f"Factor nodes={F.shape[1]} but expected-nodes={args.expected_nodes}")

    # Trace(rho)=||F||_F^2 must be one.
    traces = np.einsum("snr,snr->s", F, F)
    if not np.allclose(traces, 1.0, rtol=1e-5, atol=1e-6):
        raise ValueError(
            f"Low-rank factors are not unit-trace: range={traces.min():.9f}-{traces.max():.9f}"
        )

    print(
        f"[DATA] retained={len(y)} | {args.negative_label}={(y==0).sum()} | "
        f"{args.positive_label}={(y==1).sum()} | nodes={F.shape[1]} | rank={F.shape[2]}"
    )
    verify_against_v4_manifest(args.results_source.expanduser().resolve(), manifest)
    return manifest, F, y


def _lowrank_pair_eigensystem(Fs: np.ndarray, Ft: np.ndarray, gram_tol: float = 1e-11):
    """Exact nonzero eigensystem of Fs Fs' - Ft Ft' in sensor space."""
    rs, rt = Fs.shape[1], Ft.shape[1]
    Gss = Fs.T @ Fs
    Gtt = Ft.T @ Ft
    Gst = Fs.T @ Ft
    G = np.block([[Gss, Gst], [Gst.T, Gtt]])
    G = 0.5 * (G + G.T)
    g, V = eigh(G, check_finite=False)
    keep = g > gram_tol * max(float(g[-1]), 1.0)
    gp = g[keep]
    Vp = V[:, keep]
    signs = np.concatenate([np.ones(rs), -np.ones(rt)])
    K = Vp.T @ (signs[:, None] * Vp)
    sg = np.sqrt(gp)
    H = (sg[:, None] * K) * sg[None, :]
    H = 0.5 * (H + H.T)
    lam, W = eigh(H, check_finite=False)
    Ucat = np.concatenate([Fs, Ft], axis=1)
    Q = Ucat @ (Vp / np.sqrt(gp)[None, :]) @ W
    return lam, Q


def _pair_node_contribution_lowrank(
    factors: np.ndarray, pair_s: np.ndarray, pair_t: np.ndarray, q: int
) -> tuple[int, np.ndarray]:
    """Exact low-rank backend for the frozen first-order Schatten node contribution."""
    i, j = int(pair_s[q]), int(pair_t[q])
    Fs, Ft = factors[i], factors[j]
    diag_delta = np.sum(Fs * Fs, axis=1) - np.sum(Ft * Ft, axis=1)
    lam, U = _lowrank_pair_eigensystem(Fs, Ft)
    U2 = U * U
    abs_lam = np.abs(lam)
    sign_lam = np.sign(lam)

    p_row = P_VALUES[None, :]
    powers = abs_lam[:, None] ** p_row
    sums = np.sum(powers, axis=0)
    d = np.where(sums > 0, sums ** (1.0 / P_VALUES), 0.0)
    denom = np.where(d > 0, d ** (P_VALUES - 1.0), 1.0)

    aa = powers / denom[None, :]
    bb = sign_lam[:, None] * (abs_lam[:, None] ** (p_row - 1.0)) / denom[None, :]
    bb[abs_lam == 0, :] = 0.0

    diag_g_delta = U2 @ aa
    diag_g = U2 @ bb
    contrib = 2.0 * diag_g_delta - diag_delta[:, None] * diag_g
    contrib[:, d == 0] = 0.0
    return q, contrib.T.astype(CONTRIB_DTYPE, copy=False)


def _time_pair_kernel_lowrank(
    factors: np.ndarray, pair_s: np.ndarray, pair_t: np.ndarray,
    q_indices: list[int], workers: int
) -> float:
    t0 = time.perf_counter()
    if workers <= 1:
        for q in q_indices:
            _pair_node_contribution_lowrank(factors, pair_s, pair_t, q)
    else:
        with threadpool_limits(limits=1):
            with ThreadPoolExecutor(max_workers=workers) as ex:
                list(ex.map(
                    lambda q: _pair_node_contribution_lowrank(factors, pair_s, pair_t, q),
                    q_indices
                ))
    return time.perf_counter() - t0


def _choose_pair_workers_lowrank(
    factors: np.ndarray, pair_s: np.ndarray, pair_t: np.ndarray,
    backend: str, requested_workers: int, autotune_pairs: int
) -> tuple[str, int, float]:
    cpu = max(1, os.cpu_count() or 1)
    if backend == "serial":
        candidates = [1]
    elif backend == "threads" and requested_workers > 0:
        candidates = [requested_workers]
    elif requested_workers > 0:
        candidates = sorted(set([1, requested_workers]))
    else:
        candidates = sorted(set([1, 2, min(4, cpu), min(6, cpu), min(8, cpu)]))
    candidates = [w for w in candidates if 1 <= w <= cpu]

    ntest = min(max(8, autotune_pairs), len(pair_s))
    idx = np.linspace(0, len(pair_s)-1, ntest, dtype=int).tolist()
    timings = {}
    print(f"[XAI-PERF] Auto-calibrating LOW-RANK pair kernel on {ntest} pairs; candidates={candidates}")
    for w in candidates:
        elapsed = _time_pair_kernel_lowrank(factors, pair_s, pair_t, idx, w)
        timings[w] = elapsed
        print(f"[XAI-PERF] workers={w}: {elapsed:.3f} s ({ntest/elapsed:.2f} pairs/s calibration)")
    best = min(timings, key=timings.get)
    if best != 1 and 1 in timings and timings[best] > 0.95 * timings[1]:
        best = 1
    chosen = "serial" if best == 1 else "threads"
    rate = ntest / timings[best]
    print(f"[XAI-PERF] Selected backend={chosen}, workers={best}")
    return chosen, best, rate




def load_exact_precomputed_distances(distance_dir: Path, n_subjects: int) -> np.ndarray:
    """Load the exact matrices used by the precomputed-distance classifier: p x subject x subject."""
    mats = []
    for p in P_VALUES:
        tag = str(float(p)).replace(".", "p")
        candidates = [
            distance_dir / f"distance_p{tag}.npy",
            distance_dir / f"distance_p{p:g}.npy",
        ]
        path = next((q for q in candidates if q.exists()), None)
        if path is None:
            raise FileNotFoundError(f"Missing exact classifier distance matrix for p={p}: {candidates}")
        D = np.asarray(np.load(path), dtype=np.float64)
        if D.shape != (n_subjects, n_subjects):
            raise ValueError(f"{path.name}: shape={D.shape}, expected {(n_subjects,n_subjects)}")
        mats.append(D)
        print(f"[GEOM] exact classifier distance p={p:g} loaded from {path.name}")
    return np.stack(mats, axis=0)


def pair_distance_cache_from_matrices(
    distance_stack: np.ndarray, pair_s: np.ndarray, pair_t: np.ndarray
) -> np.ndarray:
    """Return exact classifier distances in pair x p order."""
    out = np.empty((len(pair_s), len(P_VALUES)), dtype=np.float64)
    for pi in range(len(P_VALUES)):
        out[:, pi] = distance_stack[pi, pair_s, pair_t]
    return out


# %% 4. V4 CHECKPOINT ACCESS
def checkpoint_path(root: Path, p_index: int, fold_id: int) -> Path:
    candidates = [
        root / "checkpoints" / f"p{p_index:02d}_fold{fold_id:03d}.npz",
        root / "checkpoints" / f"p{p_index:02d}_fold_{fold_id:03d}.npz",
        root / f"p{p_index:02d}_fold{fold_id:03d}.npz",
    ]
    for c in candidates:
        if c.exists():
            return c
    hits = sorted(root.rglob(f"*p{p_index:02d}*fold*{fold_id:03d}*.npz"))
    if hits:
        return hits[0]
    raise FileNotFoundError(f"Missing checkpoint p_index={p_index}, fold={fold_id} under {root}")


def load_pair_geometry(source: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    spectra_path = source / "pairwise_singular_values.npy"
    pairs_path = source / "pair_indices.npz"
    if not spectra_path.exists() or not pairs_path.exists():
        raise FileNotFoundError("v4 pairwise_singular_values.npy / pair_indices.npz are required")
    singvals = np.load(spectra_path, mmap_mode="r")
    z = np.load(pairs_path)
    pair_s = z["pair_s"].astype(np.int32)
    pair_t = z["pair_t"].astype(np.int32)
    if len(pair_s) != singvals.shape[0] or len(pair_t) != singvals.shape[0]:
        raise ValueError("Pair-index / spectrum length mismatch")
    print(f"[GEOM] pair spectra mmap: pairs={len(pair_s):,}, modes={singvals.shape[1]}")
    return singvals, pair_s, pair_t


def _distance_values_from_spectra_block(sv: np.ndarray, p: float) -> np.ndarray:
    sv = np.asarray(sv, dtype=np.float64)
    if p == 1.0:
        return np.sum(sv, axis=1)
    if p == 2.0:
        return np.sqrt(np.sum(sv * sv, axis=1))
    return np.sum(np.power(sv, p), axis=1) ** (1.0 / p)


def distance_matrix_from_spectra_chunked(
    singvals: np.ndarray,
    pair_s: np.ndarray,
    pair_t: np.ndarray,
    p: float,
    n_subjects: int,
    chunk_size: int,
) -> np.ndarray:
    """Build one exact distance matrix without materializing the full spectra file."""
    D = np.zeros((n_subjects, n_subjects), dtype=np.float64)
    n_pairs = len(pair_s)
    for start in range(0, n_pairs, chunk_size):
        stop = min(start + chunk_size, n_pairs)
        vals = _distance_values_from_spectra_block(singvals[start:stop], p)
        ii = pair_s[start:stop]
        jj = pair_t[start:stop]
        D[ii, jj] = vals
        D[jj, ii] = vals
    return D


def pair_distances_for_indices(singvals: np.ndarray, qidx: np.ndarray) -> np.ndarray:
    """Return cached Schatten distances for selected pairs, shape pair x p."""
    sv = np.asarray(singvals[qidx], dtype=np.float64)
    out = np.empty((len(qidx), len(P_VALUES)), dtype=np.float64)
    for pi, p in enumerate(P_VALUES):
        out[:, pi] = _distance_values_from_spectra_block(sv, float(p))
    return out


# %% 5. FIRST-ORDER PAIRWISE NODE CONTRIBUTIONS -- STREAMING KERNEL
def _pair_node_contribution(R: np.ndarray, pair_s: np.ndarray, pair_t: np.ndarray, q: int) -> tuple[int, np.ndarray]:
    """Compute first-order Schatten node contributions for one subject pair.

    The returned block has shape (n_p, n_nodes). No global pair cache is built.
    """
    i, j = int(pair_s[q]), int(pair_t[q])
    delta = R[i] - R[j]
    diag_delta = np.diagonal(delta).copy()
    lam, U = eigh(delta, check_finite=False, overwrite_a=True)
    U2 = U * U
    abs_lam = np.abs(lam)
    sign_lam = np.sign(lam)

    p_row = P_VALUES[None, :]
    powers = abs_lam[:, None] ** p_row
    sums = np.sum(powers, axis=0)
    d = np.where(sums > 0, sums ** (1.0 / P_VALUES), 0.0)
    denom = np.where(d > 0, d ** (P_VALUES - 1.0), 1.0)

    a = powers / denom[None, :]
    b = sign_lam[:, None] * (abs_lam[:, None] ** (p_row - 1.0)) / denom[None, :]
    b[abs_lam == 0, :] = 0.0

    diag_g_delta = U2 @ a
    diag_g = U2 @ b
    contrib = 2.0 * diag_g_delta - diag_delta[:, None] * diag_g
    contrib[:, d == 0] = 0.0
    return q, contrib.T.astype(CONTRIB_DTYPE, copy=False)


def _time_pair_kernel(R: np.ndarray, pair_s: np.ndarray, pair_t: np.ndarray,
                      q_indices: list[int], workers: int) -> float:
    t0 = time.perf_counter()
    if workers <= 1:
        for q in q_indices:
            _pair_node_contribution(R, pair_s, pair_t, q)
    else:
        with threadpool_limits(limits=1):
            with ThreadPoolExecutor(max_workers=workers) as ex:
                list(ex.map(lambda q: _pair_node_contribution(R, pair_s, pair_t, q), q_indices))
    return time.perf_counter() - t0


def _choose_pair_workers(R: np.ndarray, pair_s: np.ndarray, pair_t: np.ndarray,
                         backend: str, requested_workers: int,
                         autotune_pairs: int) -> tuple[str, int, float]:
    if backend == "serial":
        ntest = min(max(8, autotune_pairs), len(pair_s))
        idx = np.linspace(0, len(pair_s) - 1, ntest, dtype=int).tolist()
        elapsed = _time_pair_kernel(R, pair_s, pair_t, idx, 1)
        return "serial", 1, ntest / elapsed
    if backend == "threads" and requested_workers > 0:
        ntest = min(max(8, autotune_pairs), len(pair_s))
        idx = np.linspace(0, len(pair_s) - 1, ntest, dtype=int).tolist()
        elapsed = _time_pair_kernel(R, pair_s, pair_t, idx, requested_workers)
        return "threads", requested_workers, ntest / elapsed

    cpu = max(1, os.cpu_count() or 1)
    if requested_workers > 0:
        candidates = [1, requested_workers]
    else:
        candidates = sorted(set([1, 2, min(4, cpu), min(6, cpu), min(8, cpu)]))
    candidates = [w for w in candidates if 1 <= w <= cpu]

    ntest = min(max(8, autotune_pairs), len(pair_s))
    idx = np.linspace(0, len(pair_s) - 1, ntest, dtype=int).tolist()
    timings = {}
    print(f"[XAI-PERF] Auto-calibrating pair kernel on {ntest} pairs; candidates={candidates}")
    for w in candidates:
        elapsed = _time_pair_kernel(R, pair_s, pair_t, idx, w)
        timings[w] = elapsed
        print(f"[XAI-PERF] workers={w}: {elapsed:.3f} s ({ntest/elapsed:.2f} pairs/s calibration)")
    best = min(timings, key=timings.get)
    serial_t = timings.get(1)
    if best != 1 and serial_t is not None and timings[best] > 0.95 * serial_t:
        best = 1
    chosen_backend = "serial" if best == 1 else "threads"
    rate = ntest / timings[best]
    print(f"[XAI-PERF] Selected backend={chosen_backend}, workers={best}")
    return chosen_backend, best, rate


# %% 6. EXACT OUTER-MODEL RECONSTRUCTION AND SUPPORT-VECTOR PREFLIGHT
def reconstruct_svc(Ktrain: np.ndarray, ytrain: np.ndarray, C: float) -> SVC:
    clf = SVC(kernel="precomputed", C=float(C))
    clf.fit(np.asarray(Ktrain, dtype=np.float64), ytrain.astype(int))
    return clf


def reconstruct_model_metadata(
    source: Path,
    y: np.ndarray,
    distance_stack: np.ndarray,
    out: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    """Reconstruct the exact saved outer models once and retain only SV metadata.

    Returns
    -------
    alpha : p x repeat x fold x global_subject
        Signed SVC dual coefficients; zero for non-support-vectors.
    gamma : p x repeat x fold
    fold_of_subject : repeat x global_subject
        Outer test-fold membership.
    F : p x repeat x global_subject
        Saved held-out decision scores.
    stats : DataFrame
        Support-vector counts and fractions for preflight reporting.
    """
    n_p = len(P_VALUES)
    n_subjects = len(y)
    alpha = np.zeros((n_p, N_REPEATS, N_SPLITS, n_subjects), dtype=np.float64)
    gamma = np.full((n_p, N_REPEATS, N_SPLITS), np.nan, dtype=np.float64)
    fold_of_subject = np.full((N_REPEATS, n_subjects), -1, dtype=np.int16)
    F = np.full((n_p, N_REPEATS, n_subjects), np.nan, dtype=np.float64)
    rows = []

    print("[XAI-PREFLIGHT] Reconstructing exact outer SVCs and support-vector maps")
    for p_index, p in enumerate(P_VALUES):
        print(f"[XAI-PREFLIGHT] p={p:g}: reusing EXACT classifier distance matrix")
        D = np.asarray(distance_stack[p_index], dtype=np.float64)
        for fold_id in tqdm(
            range(1, N_REPEATS * N_SPLITS + 1),
            desc=f"p={p:g} model reconstruction",
            leave=False,
        ):
            z = np.load(checkpoint_path(source, p_index, fold_id), allow_pickle=False)
            tr = z["train_idx"].astype(int)
            te = z["test_idx"].astype(int)
            ytr = z["y_train"].astype(int)
            gam = float(z["gamma"])
            C = float(z["C"])
            rep = (fold_id - 1) // N_SPLITS
            fold = (fold_id - 1) % N_SPLITS

            existing = fold_of_subject[rep, te]
            if np.any((existing >= 0) & (existing != fold)):
                raise RuntimeError("Outer-fold assignment changed across p/checkpoints")
            fold_of_subject[rep, te] = fold

            Dtrain = D[np.ix_(tr, tr)]
            Dtest = D[np.ix_(te, tr)]
            Ktrain = np.exp(-gam * Dtrain * Dtrain)
            Ktest = np.exp(-gam * Dtest * Dtest)
            clf = reconstruct_svc(Ktrain, ytr, C)
            support_local = clf.support_.astype(int)
            support_global = tr[support_local]
            dual = clf.dual_coef_.reshape(-1).astype(np.float64)
            alpha[p_index, rep, fold, support_global] = dual
            gamma[p_index, rep, fold] = gam

            saved_test_scores = z["test_scores"].astype(float)
            check = clf.decision_function(Ktest)
            if not np.allclose(check, saved_test_scores, rtol=1e-6, atol=2e-6):
                raise RuntimeError(
                    f"SVC reconstruction mismatch at p={p}, fold={fold_id}; "
                    f"max_abs_diff={np.max(np.abs(check-saved_test_scores)):.3e}"
                )
            F[p_index, rep, te] = saved_test_scores
            rows.append({
                "p": float(p),
                "p_index": int(p_index),
                "outer_fold": int(fold_id),
                "repeat": int(rep + 1),
                "fold_within_repeat": int(fold + 1),
                "n_train": int(len(tr)),
                "n_test": int(len(te)),
                "n_support": int(len(support_global)),
                "support_fraction_train": float(len(support_global) / len(tr)),
                "gamma": gam,
                "C": C,
            })

    if np.any(fold_of_subject < 0):
        raise RuntimeError("Incomplete outer-fold membership map")
    if np.isnan(F).any() or np.isnan(gamma).any():
        raise RuntimeError("Incomplete reconstructed model metadata")

    stats = pd.DataFrame(rows)
    diag = out / "diagnostics"
    diag.mkdir(exist_ok=True)
    stats.to_csv(diag / "support_vector_preflight.csv", index=False)
    print(
        "[XAI-PREFLIGHT] Support vectors/train: "
        f"mean={stats['support_fraction_train'].mean():.3f}, "
        f"median={stats['support_fraction_train'].median():.3f}, "
        f"min={stats['support_fraction_train'].min():.3f}, "
        f"max={stats['support_fraction_train'].max():.3f}"
    )
    return alpha, gamma, fold_of_subject, F, stats


# %% 7. VALIDATION AND STREAMING XAI ACCUMULATION
def validate_p2_first_order_streaming(
    factors: np.ndarray,
    pair_s: np.ndarray,
    pair_t: np.ndarray,
    out: Path,
    n_pairs: int,
    seed: int,
) -> pd.DataFrame:
    """Validate low-rank first-order p=2 contributions against exact Frobenius node removal."""
    rng = np.random.default_rng(seed)
    qsel = rng.choice(len(pair_s), size=min(n_pairs, len(pair_s)), replace=False)
    rows = []
    hits = np.where(np.isclose(P_VALUES, 2.0))[0]
    if len(hits) == 0:
        print("[VALIDATE] p=2 not present in p-grid; skipping exact Frobenius validation")
        return pd.DataFrame()
    p2_idx = int(hits[0])

    for q in qsel:
        i, j = int(pair_s[q]), int(pair_t[q])
        Fs, Ft = factors[i], factors[j]
        # Exact Frobenius norm from factors without forming the dense operator.
        nss = np.sum((Fs.T @ Fs) ** 2)
        ntt = np.sum((Ft.T @ Ft) ** 2)
        nst = np.sum((Fs.T @ Ft) ** 2)
        D2 = max(float(nss + ntt - 2.0 * nst), 0.0)
        D = np.sqrt(D2)
        if D <= 0:
            continue

        # For symmetric Delta, removed squared Frobenius mass of node r is
        # 2*sum_k Delta[r,k]^2 - Delta[r,r]^2.
        # Compute row norms using factors only:
        # row_r(Delta) = Fs[r,:] Fs' - Ft[r,:] Ft'.
        Ass = Fs.T @ Fs
        Att = Ft.T @ Ft
        Ast = Fs.T @ Ft
        row_sq = (
            np.einsum("nr,rs,ns->n", Fs, Ass, Fs)
            + np.einsum("nr,rs,ns->n", Ft, Att, Ft)
            - 2.0 * np.einsum("nr,rs,ns->n", Fs, Ast, Ft)
        )
        diag_delta = np.sum(Fs*Fs, axis=1) - np.sum(Ft*Ft, axis=1)
        exact_removed_sq = 2.0 * row_sq - diag_delta * diag_delta
        D_exact_pert = np.sqrt(np.maximum(D2 - exact_removed_sq, 0.0))
        exact_change = D - D_exact_pert

        _, block = _pair_node_contribution_lowrank(factors, pair_s, pair_t, int(q))
        approx_change = np.asarray(block[p2_idx], dtype=np.float64)
        rows.append({
            "pair_index": int(q),
            "subject_i": i,
            "subject_j": j,
            "spearman_node_change": float(spearmanr(exact_change, approx_change).statistic),
            "pearson_node_change": float(pearsonr(exact_change, approx_change).statistic),
            "median_relative_abs_error": float(
                np.median(np.abs(approx_change-exact_change) / np.maximum(np.abs(exact_change),1e-12))
            ),
        })

    df = pd.DataFrame(rows)
    diag = out / "diagnostics"
    df.to_csv(diag / "p2_first_order_vs_exact_pair_validation.csv", index=False)
    if len(df):
        df[["spearman_node_change","pearson_node_change","median_relative_abs_error"]].agg(
            ["median","mean","std"]
        ).to_csv(diag / "p2_first_order_vs_exact_summary.csv")
        print(
            "[VALIDATE p=2] median Spearman exact vs first-order = "
            f"{df['spearman_node_change'].median():.4f}"
        )
    return df


def _needed_pair_mask_for_chunk(
    ii: np.ndarray,
    jj: np.ndarray,
    fold_of_subject: np.ndarray,
    support_any: np.ndarray,
) -> np.ndarray:
    """Cheap prefilter: pair is needed if it participates in any SV test relation."""
    needed = np.zeros(len(ii), dtype=bool)
    for rep in range(N_REPEATS):
        fi = fold_of_subject[rep, ii]
        fj = fold_of_subject[rep, jj]
        diff = fi != fj
        if not np.any(diff):
            continue
        needed |= diff & support_any[rep, fi, jj]
        needed |= diff & support_any[rep, fj, ii]
    return needed


def _accumulate_direction(
    accumulator: np.ndarray,
    contrib: np.ndarray,
    distances: np.ndarray,
    targets: np.ndarray,
    support_subjects: np.ndarray,
    folds: np.ndarray,
    valid: np.ndarray,
    alpha: np.ndarray,
    gamma: np.ndarray,
    rep: int,
) -> None:
    loc0 = np.flatnonzero(valid)
    if loc0.size == 0:
        return
    for pi in range(len(P_VALUES)):
        coeff0 = alpha[pi, rep, folds[loc0], support_subjects[loc0]]
        active0 = coeff0 != 0.0
        if not np.any(active0):
            continue
        loc = loc0[active0]
        coeff = coeff0[active0]
        gam = gamma[pi, rep, folds[loc]]
        d = distances[loc, pi]
        c = np.asarray(contrib[loc, pi, :], dtype=np.float64)
        k_full = np.exp(-gam * d * d)
        d_pert = np.maximum(d[:, None] - c, 0.0)
        k_pert = np.exp(-gam[:, None] * d_pert * d_pert)
        signed_delta = coeff[:, None] * (k_full[:, None] - k_pert)
        np.add.at(accumulator[pi, rep], targets[loc], signed_delta)


def run_xai_streaming(
    factors: np.ndarray,
    pair_distance_cache: np.ndarray,
    pair_s: np.ndarray,
    pair_t: np.ndarray,
    alpha: np.ndarray,
    gamma: np.ndarray,
    fold_of_subject: np.ndarray,
    F: np.ndarray,
    out: Path,
    parallel_backend: str,
    workers: int,
    autotune_pairs: int,
    pair_chunk_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    n_p = len(P_VALUES)
    n_subjects = factors.shape[0]
    n_nodes = factors.shape[1]
    n_pairs = len(pair_s)

    backend, n_workers, calibrated_rate = _choose_pair_workers_lowrank(
        factors, pair_s, pair_t, parallel_backend, workers, autotune_pairs
    )
    raw_eta_h = n_pairs / max(calibrated_rate, 1e-12) / 3600.0
    print(
        f"[XAI-PREFLIGHT] Full-pair geometry upper-bound ETA from calibration: "
        f"{raw_eta_h:.2f} h at {calibrated_rate:.1f} pairs/s"
    )
    print(
        f"[XAI-STREAM] No full contribution tensor will be written. "
        f"Streaming chunk={pair_chunk_size:,} pairs, backend={backend}, workers={n_workers}."
    )

    support_any = np.any(alpha != 0.0, axis=0)  # repeat x fold x global subject
    accumulator = np.zeros((n_p, N_REPEATS, n_subjects, n_nodes), dtype=np.float64)
    n_geom = 0
    n_skipped = 0
    t0 = time.time()

    executor = None
    limit_ctx = None
    try:
        if backend == "threads":
            limit_ctx = threadpool_limits(limits=1)
            limit_ctx.__enter__()
            executor = ThreadPoolExecutor(max_workers=n_workers)

        starts = range(0, n_pairs, pair_chunk_size)
        for start in tqdm(starts, total=math.ceil(n_pairs / pair_chunk_size), desc="XAI streaming pair chunks"):
            stop = min(start + pair_chunk_size, n_pairs)
            q_all = np.arange(start, stop, dtype=np.int64)
            ii_all = pair_s[start:stop]
            jj_all = pair_t[start:stop]
            needed = _needed_pair_mask_for_chunk(ii_all, jj_all, fold_of_subject, support_any)
            if not np.any(needed):
                n_skipped += len(q_all)
                continue

            qidx = q_all[needed]
            ii = ii_all[needed]
            jj = jj_all[needed]
            n_geom += len(qidx)
            n_skipped += len(q_all) - len(qidx)

            if executor is None:
                results = [_pair_node_contribution_lowrank(factors, pair_s, pair_t, int(q)) for q in qidx]
            else:
                results = list(executor.map(
                    lambda q: _pair_node_contribution_lowrank(factors, pair_s, pair_t, int(q)),
                    qidx,
                    chunksize=1,
                ))
            blocks = np.stack([b for _, b in results], axis=0)  # pair x p x node, float32
            distances = np.asarray(pair_distance_cache[qidx], dtype=np.float64)

            for rep in range(N_REPEATS):
                fi = fold_of_subject[rep, ii]
                fj = fold_of_subject[rep, jj]
                diff = fi != fj
                if not np.any(diff):
                    continue
                # i is held out in fi; j is a candidate support vector of model fi.
                _accumulate_direction(
                    accumulator, blocks, distances,
                    targets=ii, support_subjects=jj, folds=fi, valid=diff,
                    alpha=alpha, gamma=gamma, rep=rep,
                )
                # Symmetric reverse role for the fold in which j is held out.
                _accumulate_direction(
                    accumulator, blocks, distances,
                    targets=jj, support_subjects=ii, folds=fj, valid=diff,
                    alpha=alpha, gamma=gamma, rep=rep,
                )
    finally:
        if executor is not None:
            executor.shutdown(wait=True)
        if limit_ctx is not None:
            limit_ctx.__exit__(None, None, None)

    elapsed = time.time() - t0
    print(
        f"[XAI-STREAM] Geometry evaluated for {n_geom:,}/{n_pairs:,} unique pairs; "
        f"prefilter skipped {n_skipped:,}. Elapsed={elapsed/3600:.2f} h."
    )

    I = np.abs(accumulator).astype(IMPORTANCE_DTYPE, copy=False)
    if np.isnan(I).any() or np.isnan(F).any():
        raise RuntimeError("Incomplete streaming OOF XAI tensor")
    return I, F


# %% 8. SUMMARIES
def subject_profiles(I: np.ndarray) -> np.ndarray:
    return np.median(I, axis=1)  # p x subject x node


def population_profiles(subject_I: np.ndarray, y: np.ndarray) -> pd.DataFrame:
    rows = []
    for pi, p in enumerate(P_VALUES):
        global_mean = np.mean(subject_I[pi], axis=0)
        nc_mean = np.mean(subject_I[pi, y == 0], axis=0)
        ad_mean = np.mean(subject_I[pi, y == 1], axis=0)
        global_norm = global_mean / max(np.sum(global_mean), 1e-15)
        nc_norm = nc_mean / max(np.sum(nc_mean), 1e-15)
        ad_norm = ad_mean / max(np.sum(ad_mean), 1e-15)
        rank = pd.Series(global_mean).rank(ascending=False, method="min").to_numpy(int)
        for r in range(subject_I.shape[2]):
            rows.append({
                "p": float(p), "node_1based": r + 1,
                "importance_global_mean": float(global_mean[r]),
                f"importance_{NEGATIVE_LABEL}_mean": float(nc_mean[r]),
                f"importance_{POSITIVE_LABEL}_mean": float(ad_mean[r]),
                "importance_global_normalized": float(global_norm[r]),
                f"importance_{NEGATIVE_LABEL}_normalized": float(nc_norm[r]),
                f"importance_{POSITIVE_LABEL}_normalized": float(ad_norm[r]),
                "global_rank": int(rank[r]),
            })
    return pd.DataFrame(rows)


def xai_concordance(subject_I: np.ndarray, y: np.ndarray) -> pd.DataFrame:
    ref = subject_I[0]
    rows = []
    for pi, p in enumerate(P_VALUES):
        for s in range(subject_I.shape[1]):
            a, b = ref[s], subject_I[pi, s]
            rows.append({
                "p": float(p), "subject_idx": s, "y": int(y[s]),
                "spearman_vs_p1": float(spearmanr(a, b).statistic),
                "pearson_vs_p1": float(pearsonr(a, b).statistic),
            })
    return pd.DataFrame(rows)


def decision_xai_comparison(F: np.ndarray, xai_corr: pd.DataFrame) -> pd.DataFrame:
    subject_F = np.median(F, axis=1)  # p x subject
    rows = []
    for pi, p in enumerate(P_VALUES):
        fd = float(spearmanr(subject_F[0], subject_F[pi]).statistic)
        g = xai_corr[np.isclose(xai_corr.p, p)]["spearman_vs_p1"].to_numpy(float)
        finite = np.isfinite(g)
        gv = g[finite]
        if gv.size == 0:
            med = q25 = q75 = np.nan
        else:
            med = float(np.median(gv))
            q25 = float(np.quantile(gv, 0.25))
            q75 = float(np.quantile(gv, 0.75))
        rows.append({
            "p": float(p),
            "decision_spearman_vs_p1": fd,
            "decision_dependence_1_minus_rho": 1.0 - fd,
            "xai_subject_spearman_median_vs_p1": med,
            "xai_subject_spearman_q25_vs_p1": q25,
            "xai_subject_spearman_q75_vs_p1": q75,
            "xai_n_valid": int(gv.size),
            "xai_n_invalid": int(g.size - gv.size),
            "xai_dependence_1_minus_median_rho": 1.0 - med if np.isfinite(med) else np.nan,
        })
    return pd.DataFrame(rows)


# %% 9. PLOTTING
def savefig(fig: plt.Figure, base: Path, dpi: int) -> None:
    fig.savefig(base.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def p_labels() -> list[str]:
    return [str(int(x)) if float(x).is_integer() else str(x) for x in P_VALUES]


def plot_heatmap(pop: pd.DataFrame, out: Path, dpi: int) -> None:
    mat = pop.pivot(index="node_1based", columns="p", values="importance_global_normalized").reindex(columns=P_VALUES)
    # Sort nodes by their importance at p=1; this makes reorganization visually explicit.
    order = np.argsort(-mat[REFERENCE_P].to_numpy(float))
    arr = mat.to_numpy(float)[order]
    fig, ax = plt.subplots(figsize=(8.2, 9.2))
    im = ax.imshow(arr, aspect="auto", interpolation="nearest")
    ax.set_xticks(np.arange(len(P_VALUES)))
    ax.set_xticklabels(p_labels())
    ax.set_xlabel("Schatten p")
    ax.set_ylabel("Nodes ordered by importance at p=1")
    ax.set_title("Reorganization of population XAI profile along geometric deformation")
    cb = fig.colorbar(im, ax=ax)
    cb.set_label("Normalized mean perturbation importance")
    fig.tight_layout()
    savefig(fig, out / "05_xai_heatmap_node_by_p", dpi)


def plot_rank_trajectories(pop: pd.DataFrame, out: Path, dpi: int, top_n: int) -> None:
    ref = pop[np.isclose(pop.p, REFERENCE_P)].nsmallest(top_n, "global_rank")
    nodes = ref["node_1based"].to_numpy(int)
    fig, ax = plt.subplots(figsize=(9.2, 6.4))
    for node in nodes:
        g = pop[pop.node_1based == node].set_index("p").reindex(P_VALUES)
        ax.plot(P_VALUES, g["global_rank"].to_numpy(float), marker="o", lw=1.0, alpha=0.7)
    ax.set_xscale("log")
    ax.set_xticks(P_VALUES)
    ax.set_xticklabels(p_labels())
    ax.invert_yaxis()
    ax.set_xlabel("Schatten p")
    ax.set_ylabel("Global XAI rank (1 = highest)")
    ax.set_title(f"Rank trajectories of the top {top_n} nodes at p=1")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    savefig(fig, out / "06_top_node_rank_trajectories", dpi)


def plot_xai_concordance(xc: pd.DataFrame, out: Path, dpi: int) -> None:
    rows = []
    for p in P_VALUES:
        v = xc[np.isclose(xc.p, p)]["spearman_vs_p1"].to_numpy(float)
        rows.append((np.median(v), np.quantile(v, 0.25), np.quantile(v, 0.75)))
    a = np.asarray(rows)
    fig, ax = plt.subplots(figsize=(8.8, 5.8))
    ax.errorbar(P_VALUES, a[:, 0], yerr=np.vstack([a[:,0]-a[:,1], a[:,2]-a[:,0]]),
                fmt="o-", capsize=4)
    ax.set_xscale("log")
    ax.set_xticks(P_VALUES)
    ax.set_xticklabels(p_labels())
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("Schatten p")
    ax.set_ylabel("Subject-level XAI Spearman concordance vs p=1")
    ax.set_title("Geometry dependence of subject-specific explanations")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    savefig(fig, out / "07_subject_xai_concordance_vs_p1", dpi)


def plot_amplification(comp: pd.DataFrame, out: Path, dpi: int) -> None:
    fig, ax = plt.subplots(figsize=(8.8, 5.8))
    ax.plot(comp.p, comp.decision_dependence_1_minus_rho, marker="o", lw=2.0,
            label="Decision: 1 - Spearman vs p=1")
    ax.plot(comp.p, comp.xai_dependence_1_minus_median_rho, marker="o", lw=2.0,
            label="XAI: 1 - median subject Spearman vs p=1")
    ax.set_xscale("log")
    ax.set_xticks(P_VALUES)
    ax.set_xticklabels(p_labels())
    ax.set_xlabel("Schatten p")
    ax.set_ylabel("Geometry dependence (loss of concordance)")
    ax.set_title("Propagation from decision reorganization to explanation reorganization")
    ax.legend(frameon=False)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    savefig(fig, out / "08_decision_vs_xai_dependence", dpi)


def plot_class_profiles(pop: pd.DataFrame, out: Path, dpi: int) -> None:
    # Rather than 549 spaghetti lines, show rank correlation of each class profile to p=1.
    rows = []
    for col, name in [(f"importance_{NEGATIVE_LABEL}_normalized", NEGATIVE_LABEL), (f"importance_{POSITIVE_LABEL}_normalized", POSITIVE_LABEL)]:
        ref = pop[np.isclose(pop.p, 1.0)].sort_values("node_1based")[col].to_numpy(float)
        for p in P_VALUES:
            cur = pop[np.isclose(pop.p, p)].sort_values("node_1based")[col].to_numpy(float)
            rows.append({"p": p, "class": name, "spearman": float(spearmanr(ref, cur).statistic)})
    df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(8.8, 5.8))
    for name, g in df.groupby("class"):
        ax.plot(g.p, g.spearman, marker="o", label=name)
    ax.set_xscale("log")
    ax.set_xticks(P_VALUES)
    ax.set_xticklabels(p_labels())
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("Schatten p")
    ax.set_ylabel("Population XAI profile Spearman vs p=1")
    ax.set_title("Class-specific explanatory reorganization")
    ax.legend(frameon=False)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    savefig(fig, out / "09_class_specific_xai_profiles", dpi)


# %% 10. MAIN
def main() -> None:
    global P_VALUES, N_REPEATS, N_SPLITS, REFERENCE_P, NEGATIVE_LABEL, POSITIVE_LABEL
    args = parse_args()
    P_VALUES = np.asarray(args.p_values, dtype=float)
    N_REPEATS = int(args.n_repeats)
    N_SPLITS = int(args.n_splits)
    REFERENCE_P = float(args.reference_p)
    NEGATIVE_LABEL = str(args.negative_label)
    POSITIVE_LABEL = str(args.positive_label)
    if POSITIVE_LABEL == NEGATIVE_LABEL:
        raise ValueError("Positive and negative labels must differ")
    if not np.any(np.isclose(P_VALUES, REFERENCE_P)):
        raise ValueError("reference-p must be present in p-values")
    source = args.results_source.expanduser().resolve()
    out = args.output_dir.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    (out / "diagnostics").mkdir(exist_ok=True)

    print("[START] Schatten-p decision -> XAI propagation")
    print(f"        source={source}")
    print(f"        output={out}")
    print(f"        factors={args.factors_file.expanduser().resolve()}")
    print(f"        exact_classifier_distances={args.precomputed_distance_dir.expanduser().resolve()}")
    print("        backend=exact low-rank PEMS")

    manifest, factors, y = build_pems_lowrank_stack(args)
    singvals, pair_s, pair_t = load_pair_geometry(source)
    if len(pair_s) != len(pair_t):
        raise ValueError("Pair-index length mismatch")
    distance_stack = load_exact_precomputed_distances(
        args.precomputed_distance_dir.expanduser().resolve(), len(y)
    )
    pair_distance_cache = pair_distance_cache_from_matrices(distance_stack, pair_s, pair_t)
    print("[GEOM] XAI kernel distances are now byte-for-byte sourced from the classifier distance files.")
    if args.reuse_contributions:
        print("[XAI-STREAM] --reuse-contributions is a legacy flag; the large pairwise contribution cache is no longer used.")

    alpha, gamma, fold_of_subject, F, sv_stats = reconstruct_model_metadata(
        source, y, distance_stack, out
    )
    validate_p2_first_order_streaming(
        factors, pair_s, pair_t, out, n_pairs=args.validation_pairs, seed=args.random_state
    )
    I, F = run_xai_streaming(
        factors, pair_distance_cache, pair_s, pair_t, alpha, gamma, fold_of_subject, F, out,
        parallel_backend=args.xai_parallel, workers=args.xai_workers,
        autotune_pairs=args.xai_autotune_pairs, pair_chunk_size=args.xai_pair_chunk_size,
    )
    # Preserve the exact v1 output schema, including labels.
    np.savez_compressed(out / "00_xai_repeat_tensor.npz",
                        p_values=P_VALUES, importance=I, decision_scores=F, y=y)
    print(f"[SAVE] {out / '00_xai_repeat_tensor.npz'}")
    subject_I = subject_profiles(I)
    np.savez_compressed(out / "01_subject_xai_profiles.npz",
                        p_values=P_VALUES, importance_subject_median=subject_I,
                        decision_subject_median=np.median(F, axis=1), y=y,
                        subject_ids=manifest["sample_id"].astype(str).to_numpy())

    pop = population_profiles(subject_I, y)
    pop.to_csv(out / "02_population_importance_profiles.csv", index=False)
    xc = xai_concordance(subject_I, y)
    xc.to_csv(out / "03_xai_concordance_vs_p1.csv", index=False)
    comp = decision_xai_comparison(F, xc)
    comp.to_csv(out / "04_decision_vs_xai_concordance.csv", index=False)

    plot_heatmap(pop, out, args.dpi)
    plot_rank_trajectories(pop, out, args.dpi, args.top_n_ranks)
    plot_xai_concordance(xc, out, args.dpi)
    plot_amplification(comp, out, args.dpi)
    plot_class_profiles(pop, out, args.dpi)

    keep_cols = [c for c in ["sample_id", args.label_col, "source_split", "original_day_label", "factor_index"] if c in manifest.columns]
    manifest[keep_cols].to_csv(out / "retained_manifest_xai.csv", index=False)
    print("[DONE] XAI propagation analysis complete.")


if __name__ == "__main__":
    main()
