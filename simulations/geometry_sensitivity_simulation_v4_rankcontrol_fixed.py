#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Geometry Sensitivity Simulation v3 - normalized cumulative metrics
==================================

Canonical v3 simulation for producing normalized, baseline-referenced quantities
for the main Figure 2. The generative model and Schatten grid are unchanged from v2.

Scientific changes relative to v1
---------------------------------
1. Extended generative-variance grid:
   sigma^2 = 0, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 5e-1, 1.
2. Realized population heterogeneity is computed within every simulation cell:
       H_F_mean = mean_{s<t} ||rho_s-rho_t||_F
       H_F_rms  = RMS_{s<t}  ||rho_s-rho_t||_F
       H_center = RMS_s ||rho_s-rho_bar||_F
3. Existing v1 results can be reused for the original sigma^2 points when
   plotting/aggregating, so the expensive old conditions do not need to be
   recomputed.
4. New v2 results are stored separately in geometry_sensitivity_results_v3_normalized.

The geometry/susceptibility definitions are unchanged from v1.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
from scipy.stats import spearmanr
from numpy.typing import NDArray
from tqdm.auto import tqdm


# %% 0. CONFIGURATION
@dataclass(frozen=True)
class ScientificConfig:
    n_nodes: int = 100
    n_subjects: int = 200

    sigma2_values: tuple[float, ...] = (
        0.0,
        1e-4,
        5e-4,
        1e-3,
        5e-3,
        1e-2,
        5e-2,
        1e-1,
        5e-1,
        1.0,
    )

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

    n_replicates: int = 100
    random_seed: int = 20260812


@dataclass(frozen=True)
class ComputeConfig:
    dtype: str = "float64"
    cache_pairwise_singular_values: bool = False
    show_progress: bool = True
    output_dir: str = "geometry_sensitivity_results_v4_rankcontrol"

    # Optional legacy reuse for aggregation/plotting.
    legacy_v1_dir: str = "geometry_sensitivity_results_v1"


SCI = ScientificConfig()
COMP = ComputeConfig()

DTYPE = np.dtype(COMP.dtype)
OUTPUT_DIR = Path(COMP.output_dir)
LEGACY_V1_DIR = Path(COMP.legacy_v1_dir)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ORIGINAL_V1_SIGMA2 = (0.0, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2)
NEW_SIGMA2 = (5e-2, 1e-1, 5e-1, 1.0)

print("[SETUP] Geometry Sensitivity Simulation v3 - normalized cumulative metrics")
print(
    f"        nodes={SCI.n_nodes}, subjects={SCI.n_subjects}, "
    f"replicates={SCI.n_replicates}"
)
print(f"        output={OUTPUT_DIR.resolve()}")
print(f"        sigma^2 grid={SCI.sigma2_values}")


# %% 1. REPRODUCIBILITY / CONFIG
def make_rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def child_seed(base_seed: int, sigma_index: int, replicate: int) -> int:
    """
    Same deterministic rule as v1.

    IMPORTANT:
    sigma_index is the index in the FULL v2 sigma^2 grid.
    """
    return int(base_seed + 100_000 * sigma_index + replicate)


def save_config() -> None:
    payload = {
        "scientific_config": asdict(SCI),
        "compute_config": asdict(COMP),
        "original_v1_sigma2": ORIGINAL_V1_SIGMA2,
        "new_sigma2": NEW_SIGMA2,
    }
    path = OUTPUT_DIR / "config.json"
    path.write_text(json.dumps(payload, indent=2))
    print(f"[CONFIG] Saved -> {path}")


save_config()


# %% 2. BASE DENSITY OPERATOR
def make_reference_factor(
    n_nodes: int,
    rng: np.random.Generator,
    dtype: np.dtype = np.float64,
) -> NDArray[np.floating]:
    return rng.normal(
        loc=0.0,
        scale=1.0,
        size=(n_nodes, n_nodes),
    ).astype(dtype)


def factor_to_density(x: NDArray[np.floating]) -> NDArray[np.floating]:
    rho = x @ x.T
    tr = np.trace(rho)
    if not np.isfinite(tr) or tr <= 0:
        raise ValueError("Invalid trace while constructing density operator.")
    return rho / tr


def validate_density(rho: NDArray[np.floating], atol: float = 1e-10) -> None:
    if not np.allclose(rho, rho.T, atol=atol):
        raise ValueError("Density operator is not symmetric.")
    if not np.isclose(np.trace(rho), 1.0, atol=atol):
        raise ValueError("Density operator is not trace-normalized.")
    eig_min = np.linalg.eigvalsh(rho).min()
    if eig_min < -atol:
        raise ValueError(
            f"Density operator is not PSD: min eigenvalue={eig_min:.3e}"
        )


BASE_RNG = make_rng(SCI.random_seed)
X0 = make_reference_factor(SCI.n_nodes, BASE_RNG, DTYPE)
RHO0 = factor_to_density(X0)
validate_density(RHO0)

print("[BASE] Reference density operator generated")


# %% 3. SYNTHETIC POPULATION
def generate_population(
    x0: NDArray[np.floating],
    sigma2: float,
    n_subjects: int,
    rng: np.random.Generator,
    dtype: np.dtype = np.float64,
) -> NDArray[np.floating]:
    """
    X_s = X_0 + sigma G_s
    rho_s = X_s X_s^T / Tr(X_s X_s^T)
    """
    if sigma2 < 0:
        raise ValueError("sigma2 must be non-negative.")

    sigma = math.sqrt(sigma2)
    n_nodes = x0.shape[0]
    population = np.empty(
        (n_subjects, n_nodes, n_nodes),
        dtype=dtype,
    )

    if sigma2 == 0.0:
        rho0 = factor_to_density(x0)
        population[:] = rho0
        return population

    for s in range(n_subjects):
        noise = rng.normal(
            loc=0.0,
            scale=1.0,
            size=(n_nodes, n_nodes),
        ).astype(dtype)
        xs = x0 + sigma * noise
        population[s] = factor_to_density(xs)

    return population


# %% 4. REALIZED HETEROGENEITY
def realized_heterogeneity(
    population: NDArray[np.floating],
) -> dict[str, float]:
    """
    p-independent realized heterogeneity.

    H_F_mean = mean pairwise Frobenius distance
    H_F_rms  = RMS pairwise Frobenius distance
    H_center = RMS Frobenius dispersion around rho_bar
    """
    n_subjects = population.shape[0]
    flat = population.reshape(n_subjects, -1)

    norms2 = np.einsum("ij,ij->i", flat, flat)
    gram = flat @ flat.T
    d2 = norms2[:, None] + norms2[None, :] - 2.0 * gram
    d2 = np.maximum(d2, 0.0)

    iu = np.triu_indices(n_subjects, k=1)
    pair_d2 = d2[iu]
    pair_d = np.sqrt(pair_d2)

    rho_bar = np.mean(population, axis=0)
    centered = population - rho_bar

    return {
        "H_F_mean": float(np.mean(pair_d)),
        "H_F_rms": float(np.sqrt(np.mean(pair_d2))),
        "H_center": float(
            np.sqrt(np.mean(np.sum(centered * centered, axis=(1, 2))))
        ),
    }


# %% 5. PAIR INDEXING
def upper_triangle_pairs(
    n_subjects: int,
) -> Tuple[NDArray[np.int64], NDArray[np.int64]]:
    return np.triu_indices(n_subjects, k=1)


PAIR_S, PAIR_T = upper_triangle_pairs(SCI.n_subjects)
N_PAIRS = len(PAIR_S)

print(f"[PAIRS] Unique subject pairs: {N_PAIRS:,}")


# %% 6. PAIRWISE SINGULAR VALUES
def pairwise_singular_values(
    population: NDArray[np.floating],
    pair_s: NDArray[np.int64],
    pair_t: NDArray[np.int64],
    show_progress: bool = True,
) -> NDArray[np.floating]:
    """
    Delta_st is symmetric, so singular values are |eigvalsh(Delta_st)|.
    """
    n_pairs = len(pair_s)
    n_nodes = population.shape[1]
    singvals = np.empty((n_pairs, n_nodes), dtype=population.dtype)

    iterator = range(n_pairs)
    if show_progress:
        iterator = tqdm(
            iterator,
            total=n_pairs,
            desc="Pair spectra",
            leave=False,
        )

    for q in iterator:
        delta = population[pair_s[q]] - population[pair_t[q]]
        eigvals = np.linalg.eigvalsh(delta)
        singvals[q] = np.abs(eigvals)

    return singvals


# %% 7. SCHATTEN DISTANCES
def schatten_pair_distances(
    singvals: NDArray[np.floating],
    p: float,
) -> NDArray[np.floating]:
    if p < 1:
        raise ValueError("Schatten p must satisfy p >= 1.")
    if np.isinf(p):
        return singvals.max(axis=1)
    return np.power(
        np.power(singvals, p).sum(axis=1),
        1.0 / p,
    )


def pair_vector_to_distance_matrix(
    pair_values: NDArray[np.floating],
    pair_s: NDArray[np.int64],
    pair_t: NDArray[np.int64],
    n_subjects: int,
) -> NDArray[np.floating]:
    dmat = np.zeros((n_subjects, n_subjects), dtype=pair_values.dtype)
    dmat[pair_s, pair_t] = pair_values
    dmat[pair_t, pair_s] = pair_values
    return dmat


# %% 8. SCALE / SHAPE REPRESENTATIONS
def frobenius_scale(dmat: NDArray[np.floating]) -> float:
    return float(np.linalg.norm(dmat, ord="fro"))


def normalize_distance_shape(
    dmat: NDArray[np.floating],
) -> NDArray[np.floating]:
    scale = frobenius_scale(dmat)
    if scale == 0.0:
        return np.zeros_like(dmat)
    return dmat / scale


# %% 9. GLOBAL SUSCEPTIBILITIES
def finite_difference_shape_response(
    dmat_a: NDArray[np.floating],
    dmat_b: NDArray[np.floating],
    p_a: float,
    p_b: float,
) -> float:
    dp = p_b - p_a
    if dp <= 0:
        raise ValueError("Require p_b > p_a.")

    shape_a = normalize_distance_shape(dmat_a)
    shape_b = normalize_distance_shape(dmat_b)

    return float(
        np.linalg.norm(shape_b - shape_a, ord="fro") / dp
    )


def finite_difference_scale_response(
    dmat_a: NDArray[np.floating],
    dmat_b: NDArray[np.floating],
    p_a: float,
    p_b: float,
) -> float:
    dp = p_b - p_a
    if dp <= 0:
        raise ValueError("Require p_b > p_a.")

    sa = frobenius_scale(dmat_a)
    sb = frobenius_scale(dmat_b)

    if sa == 0.0 or sb == 0.0:
        return np.nan

    return float(
        abs((np.log(sb) - np.log(sa)) / dp)
    )


# %% 10. SUBJECT-LEVEL SUSCEPTIBILITY
def subject_shape_profiles(
    dmat: NDArray[np.floating],
) -> NDArray[np.floating]:
    norms = np.linalg.norm(dmat, axis=1, keepdims=True)
    out = np.zeros_like(dmat)
    nonzero = norms[:, 0] > 0
    out[nonzero] = dmat[nonzero] / norms[nonzero]
    return out


def finite_difference_subject_shape_response(
    dmat_a: NDArray[np.floating],
    dmat_b: NDArray[np.floating],
    p_a: float,
    p_b: float,
) -> NDArray[np.floating]:
    dp = p_b - p_a
    if dp <= 0:
        raise ValueError("Require p_b > p_a.")

    prof_a = subject_shape_profiles(dmat_a)
    prof_b = subject_shape_profiles(dmat_b)

    return np.linalg.norm(prof_b - prof_a, axis=1) / dp


# %% 11. SINGLE CONDITION
def run_single_condition(
    sigma2: float,
    sigma_index: int,
    replicate: int,
) -> Dict[str, NDArray[np.floating] | float | int]:
    seed = child_seed(
        SCI.random_seed,
        sigma_index,
        replicate,
    )
    rng = make_rng(seed)

    print(
        f"[RUN] sigma^2={sigma2:.6g} | "
        f"replicate={replicate + 1}/{SCI.n_replicates}"
    )

    t0 = time.perf_counter()

    population = generate_population(
        x0=X0,
        sigma2=sigma2,
        n_subjects=SCI.n_subjects,
        rng=rng,
        dtype=DTYPE,
    )

    heterogeneity = realized_heterogeneity(population)
    print(
        f"      population generated | "
        f"H_F={heterogeneity['H_F_mean']:.6g}"
    )

    singvals = pairwise_singular_values(
        population=population,
        pair_s=PAIR_S,
        pair_t=PAIR_T,
        show_progress=COMP.show_progress,
    )

    print("      pairwise spectra computed")

    p_values = np.asarray(SCI.p_values, dtype=float)

    # Legacy local susceptibilities are retained for audit/continuity.
    global_shape = np.full(len(p_values) - 1, np.nan)
    global_scale = np.full(len(p_values) - 1, np.nan)
    subject_mean = np.full(len(p_values) - 1, np.nan)
    subject_sd = np.full(len(p_values) - 1, np.nan)

    # Main-paper cumulative quantities, all referenced to p=1.
    # Percent units are used for direct interpretation.
    cumulative_scale_contraction_pct = np.full(len(p_values), np.nan)
    cumulative_shape_change_pct = np.full(len(p_values), np.nan)
    cumulative_subject_mean_pct = np.full(len(p_values), np.nan)
    cumulative_subject_sd_pct = np.full(len(p_values), np.nan)
    cumulative_subject_cv_pct = np.full(len(p_values), np.nan)
    cumulative_pairwise_rank_loss_pct = np.full(len(p_values), np.nan)

    previous_p = None
    previous_dmat = None
    baseline_dmat = None
    baseline_scale = None
    baseline_shape = None
    baseline_subject_profiles = None
    baseline_pair_dist = None

    for i, p in enumerate(
        tqdm(
            p_values,
            desc="Schatten sweep",
            leave=False,
            disable=not COMP.show_progress,
        )
    ):
        pair_dist = schatten_pair_distances(singvals, p)
        dmat = pair_vector_to_distance_matrix(
            pair_dist,
            PAIR_S,
            PAIR_T,
            SCI.n_subjects,
        )

        # Define p=1 as the common geometric baseline. For sigma^2=0 all
        # pairwise distances vanish, so downstream normalized quantities are
        # intentionally left undefined and that condition is used only in
        # the heterogeneity panel.
        if i == 0:
            baseline_pair_dist = pair_dist.copy()
            baseline_dmat = dmat.copy()
            baseline_scale = frobenius_scale(baseline_dmat)
            if baseline_scale > 0:
                baseline_shape = normalize_distance_shape(baseline_dmat)
                baseline_subject_profiles = subject_shape_profiles(baseline_dmat)
                cumulative_scale_contraction_pct[i] = 0.0
                cumulative_shape_change_pct[i] = 0.0
                cumulative_subject_mean_pct[i] = 0.0
                cumulative_subject_sd_pct[i] = 0.0
                cumulative_subject_cv_pct[i] = np.nan
                cumulative_pairwise_rank_loss_pct[i] = 0.0
        elif baseline_scale is not None and baseline_scale > 0:
            current_scale = frobenius_scale(dmat)
            cumulative_scale_contraction_pct[i] = 100.0 * (
                1.0 - current_scale / baseline_scale
            )

            current_shape = normalize_distance_shape(dmat)
            cumulative_shape_change_pct[i] = 100.0 * (
                np.linalg.norm(current_shape - baseline_shape, ord="fro")
                / np.sqrt(2.0)
            )

            rho_rank = spearmanr(baseline_pair_dist, pair_dist).statistic
            cumulative_pairwise_rank_loss_pct[i] = 100.0 * (1.0 - rho_rank)

            current_profiles = subject_shape_profiles(dmat)
            subject_cumulative = 100.0 * (
                np.linalg.norm(
                    current_profiles - baseline_subject_profiles, axis=1
                )
                / np.sqrt(2.0)
            )
            cumulative_subject_mean_pct[i] = float(np.mean(subject_cumulative))
            cumulative_subject_sd_pct[i] = float(
                np.std(subject_cumulative, ddof=1)
            )
            if cumulative_subject_mean_pct[i] > 0:
                cumulative_subject_cv_pct[i] = 100.0 * (
                    cumulative_subject_sd_pct[i]
                    / cumulative_subject_mean_pct[i]
                )

        if previous_dmat is not None and previous_p is not None:
            j = i - 1

            global_shape[j] = finite_difference_shape_response(
                previous_dmat,
                dmat,
                previous_p,
                p,
            )
            global_scale[j] = finite_difference_scale_response(
                previous_dmat,
                dmat,
                previous_p,
                p,
            )

            subject_response = finite_difference_subject_shape_response(
                previous_dmat,
                dmat,
                previous_p,
                p,
            )

            subject_mean[j] = float(np.mean(subject_response))
            subject_sd[j] = float(
                np.std(subject_response, ddof=1)
            )

        previous_p = p
        previous_dmat = dmat

    elapsed = time.perf_counter() - t0
    print(f"      completed in {elapsed:.1f} s")

    result = {
        "sigma2": float(sigma2),
        "sigma_index": int(sigma_index),
        "replicate": int(replicate),
        "seed": int(seed),
        "p_left": p_values[:-1],
        "p_right": p_values[1:],
        "shape_response": global_shape,
        "scale_response": global_scale,
        "subject_shape_mean": subject_mean,
        "subject_shape_sd": subject_sd,
        "cumulative_scale_contraction_pct": cumulative_scale_contraction_pct,
        "cumulative_shape_change_pct": cumulative_shape_change_pct,
        "cumulative_subject_mean_pct": cumulative_subject_mean_pct,
        "cumulative_subject_sd_pct": cumulative_subject_sd_pct,
        "cumulative_subject_cv_pct": cumulative_subject_cv_pct,
        "cumulative_pairwise_rank_loss_pct": cumulative_pairwise_rank_loss_pct,
        "H_F_mean": heterogeneity["H_F_mean"],
        "H_F_normalized_pct": 100.0 * heterogeneity["H_F_mean"] / np.sqrt(2.0),
        "H_F_rms": heterogeneity["H_F_rms"],
        "H_center": heterogeneity["H_center"],
        "elapsed_seconds": float(elapsed),
    }

    if COMP.cache_pairwise_singular_values:
        np.save(
            OUTPUT_DIR
            / f"singvals_sigma{sigma_index:03d}_rep{replicate:03d}.npy",
            singvals,
        )

    return result


# %% 12. SAVE / LOAD
def result_path(
    sigma_index: int,
    replicate: int,
) -> Path:
    return (
        OUTPUT_DIR
        / f"result_sigma{sigma_index:03d}_rep{replicate:03d}.npz"
    )


def save_condition_result(
    result: dict,
    sigma_index: int,
    replicate: int,
) -> Path:
    path = result_path(sigma_index, replicate)
    np.savez_compressed(path, **result)
    return path


def condition_result_exists(
    sigma_index: int,
    replicate: int,
) -> bool:
    return result_path(sigma_index, replicate).exists()


# %% 13. RUN SELECTED SIGMA^2 VALUES
def sigma_index_from_value(sigma2: float) -> int:
    values = np.asarray(SCI.sigma2_values, dtype=float)
    matches = np.where(
        np.isclose(values, sigma2, rtol=0.0, atol=1e-15)
    )[0]
    if len(matches) != 1:
        raise ValueError(
            f"sigma^2={sigma2} not uniquely present in v2 grid."
        )
    return int(matches[0])


def run_sigma_subset(
    sigma2_subset: tuple[float, ...],
    skip_existing: bool = True,
) -> None:
    total = len(sigma2_subset) * SCI.n_replicates
    completed = 0

    print(
        f"[GRID] Running {len(sigma2_subset)} sigma^2 conditions "
        f"x {SCI.n_replicates} replicates"
    )

    for sigma2 in sigma2_subset:
        sigma_index = sigma_index_from_value(sigma2)

        print(
            f"\n[SIGMA] full-grid index={sigma_index} "
            f"| sigma^2={sigma2:g}"
        )

        for replicate in range(SCI.n_replicates):
            if (
                skip_existing
                and condition_result_exists(sigma_index, replicate)
            ):
                completed += 1
                print(
                    f"[SKIP] sigma^2={sigma2:g}, "
                    f"replicate={replicate + 1} "
                    f"({completed}/{total})"
                )
                continue

            result = run_single_condition(
                sigma2=sigma2,
                sigma_index=sigma_index,
                replicate=replicate,
            )
            path = save_condition_result(
                result,
                sigma_index,
                replicate,
            )

            completed += 1
            print(
                f"[SAVE] {path.name} | "
                f"progress={completed}/{total}"
            )

    print("\n[DONE] Requested sigma^2 subset completed")


def run_full_grid(skip_existing: bool = True) -> None:
    run_sigma_subset(
        tuple(SCI.sigma2_values),
        skip_existing=skip_existing,
    )


def run_new_points_only(skip_existing: bool = True) -> None:
    run_sigma_subset(
        NEW_SIGMA2,
        skip_existing=skip_existing,
    )


# %% 14. LEGACY V1 RESULT REUSE
def _legacy_v1_index(sigma2: float) -> int | None:
    for i, value in enumerate(ORIGINAL_V1_SIGMA2):
        if np.isclose(sigma2, value, rtol=0.0, atol=1e-15):
            return i
    return None


def _load_npz_result(path: Path) -> dict:
    data = np.load(path)
    result = {
        "sigma2": float(data["sigma2"]),
        "replicate": int(data["replicate"]),
        "seed": int(data["seed"]),
        "p_left": data["p_left"],
        "p_right": data["p_right"],
        "shape_response": data["shape_response"],
        "scale_response": data["scale_response"],
        "subject_shape_mean": data["subject_shape_mean"],
        "subject_shape_sd": data["subject_shape_sd"],
        "elapsed_seconds": float(data["elapsed_seconds"]),
    }

    for key in (
        "cumulative_scale_contraction_pct",
        "cumulative_shape_change_pct",
        "cumulative_subject_mean_pct",
        "cumulative_subject_sd_pct",
        "cumulative_subject_cv_pct",
        "cumulative_pairwise_rank_loss_pct",
        "H_F_normalized_pct",
    ):
        result[key] = data[key] if key in data.files else np.nan

    for key in ("H_F_mean", "H_F_rms", "H_center"):
        result[key] = (
            float(data[key])
            if key in data.files
            else np.nan
        )

    return result


def load_all_results(
    include_legacy_v1: bool = True,
) -> list[dict]:
    """
    Prefer v2 files. For original sigma^2 points not yet recomputed in v2,
    optionally fall back to the existing v1 NPZ files.
    """
    results: list[dict] = []

    for sigma2 in SCI.sigma2_values:
        v2_index = sigma_index_from_value(sigma2)

        for replicate in range(SCI.n_replicates):
            v2_path = result_path(v2_index, replicate)

            if v2_path.exists():
                result = _load_npz_result(v2_path)
                result["source"] = "v2"
                results.append(result)
                continue

            if include_legacy_v1:
                v1_index = _legacy_v1_index(sigma2)
                if v1_index is not None:
                    v1_path = (
                        LEGACY_V1_DIR
                        / f"result_sigma{v1_index:03d}_rep{replicate:03d}.npz"
                    )

                    if v1_path.exists():
                        result = _load_npz_result(v1_path)
                        result["source"] = "v1"
                        results.append(result)

    print(
        f"[LOAD] Loaded {len(results)} simulation cells "
        f"(v2 preferred, v1 fallback enabled={include_legacy_v1})"
    )
    return results


# %% 15. HETEROGENEITY FOR LEGACY V1 CONDITIONS
def compute_missing_heterogeneity(
    results: list[dict],
) -> None:
    """
    Legacy v1 NPZ files do not contain H_F. Reconstruct ONLY the population
    (no spectra, no Schatten sweep) for those cells and fill H in memory.

    This is cheap relative to the full susceptibility computation.
    """
    missing = [
        r for r in results
        if not np.isfinite(r["H_F_mean"])
    ]

    if not missing:
        print("[HET] All loaded results already contain heterogeneity metrics")
        return

    print(
        f"[HET] Reconstructing realized heterogeneity for "
        f"{len(missing)} legacy cells only"
    )

    for r in tqdm(
        missing,
        desc="Legacy heterogeneity",
    ):
        sigma2 = float(r["sigma2"])

        # IMPORTANT: legacy v1 seed must use the legacy sigma index.
        legacy_index = _legacy_v1_index(sigma2)
        if legacy_index is None:
            raise RuntimeError(
                "Missing H for a non-legacy condition."
            )

        replicate = int(r["replicate"])
        seed = child_seed(
            SCI.random_seed,
            legacy_index,
            replicate,
        )
        rng = make_rng(seed)

        population = generate_population(
            x0=X0,
            sigma2=sigma2,
            n_subjects=SCI.n_subjects,
            rng=rng,
            dtype=DTYPE,
        )

        h = realized_heterogeneity(population)
        r.update(h)

    print("[HET] Legacy heterogeneity reconstruction completed")


# %% 16. AGGREGATION
def _mean_sd_ci(stack: NDArray[np.floating]):
    mean = np.nanmean(stack, axis=0)
    sd = np.nanstd(stack, axis=0, ddof=1)
    n = np.sum(np.isfinite(stack), axis=0)

    sem = np.divide(
        sd,
        np.sqrt(n),
        out=np.full_like(sd, np.nan),
        where=n > 0,
    )

    return (
        mean,
        sd,
        mean - 1.96 * sem,
        mean + 1.96 * sem,
    )


def aggregate_grid(results: list[dict]) -> dict:
    sigma2 = np.asarray(SCI.sigma2_values, dtype=float)
    p_left = np.asarray(SCI.p_values[:-1], dtype=float)
    p_right = np.asarray(SCI.p_values[1:], dtype=float)
    p_mid = 0.5 * (p_left + p_right)

    shape = {}
    scale = {}
    subj_mean = {}
    subj_sd = {}
    ratio = {}

    h_mean = np.full(len(sigma2), np.nan)
    h_sd = np.full(len(sigma2), np.nan)
    h_rms_mean = np.full(len(sigma2), np.nan)
    h_center_mean = np.full(len(sigma2), np.nan)

    n_reps = np.zeros(len(sigma2), dtype=int)

    for i, s2 in enumerate(sigma2):
        subset = [
            r for r in results
            if np.isclose(
                r["sigma2"],
                s2,
                rtol=0.0,
                atol=1e-15,
            )
        ]

        if not subset:
            continue

        n_reps[i] = len(subset)

        a = np.vstack(
            [r["shape_response"] for r in subset]
        )
        b = np.vstack(
            [r["scale_response"] for r in subset]
        )
        c = np.vstack(
            [r["subject_shape_mean"] for r in subset]
        )
        d = np.vstack(
            [r["subject_shape_sd"] for r in subset]
        )
        e = np.divide(
            d,
            c,
            out=np.full_like(d, np.nan),
            where=c > 0,
        )

        for stack, store in [
            (a, shape),
            (b, scale),
            (c, subj_mean),
            (e, ratio),
        ]:
            m, sd, lo, hi = _mean_sd_ci(stack)

            store.setdefault(
                "mean",
                np.full(
                    (len(sigma2), len(p_mid)),
                    np.nan,
                ),
            )[i] = m

            store.setdefault(
                "sd",
                np.full(
                    (len(sigma2), len(p_mid)),
                    np.nan,
                ),
            )[i] = sd

            store.setdefault(
                "ci_low",
                np.full(
                    (len(sigma2), len(p_mid)),
                    np.nan,
                ),
            )[i] = lo

            store.setdefault(
                "ci_high",
                np.full(
                    (len(sigma2), len(p_mid)),
                    np.nan,
                ),
            )[i] = hi

        subj_sd.setdefault(
            "mean",
            np.full(
                (len(sigma2), len(p_mid)),
                np.nan,
            ),
        )[i] = np.nanmean(d, axis=0)

        subj_sd.setdefault(
            "sd",
            np.full(
                (len(sigma2), len(p_mid)),
                np.nan,
            ),
        )[i] = np.nanstd(d, axis=0, ddof=1)

        h = np.asarray(
            [r["H_F_mean"] for r in subset],
            dtype=float,
        )
        h_rms = np.asarray(
            [r["H_F_rms"] for r in subset],
            dtype=float,
        )
        h_center = np.asarray(
            [r["H_center"] for r in subset],
            dtype=float,
        )

        h_mean[i] = np.mean(h)
        h_sd[i] = np.std(h, ddof=1)
        h_rms_mean[i] = np.mean(h_rms)
        h_center_mean[i] = np.mean(h_center)

    return {
        "sigma2": sigma2,
        "p_left": p_left,
        "p_right": p_right,
        "p_mid": p_mid,
        "n_reps": n_reps,
        "shape": shape,
        "scale": scale,
        "subj_mean": subj_mean,
        "subj_sd": subj_sd,
        "ratio": ratio,
        "H_F_mean": h_mean,
        "H_F_sd": h_sd,
        "H_F_rms_mean": h_rms_mean,
        "H_center_mean": h_center_mean,
    }


# %% 17. PRIMARY COLLAPSE PLOTS
def _plot_collapse(
    agg: dict,
    block: dict,
    ylabel: str,
    title: str,
    filename: str,
):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8.5, 5.8))
    p = agg["p_mid"]

    for i, s2 in enumerate(agg["sigma2"]):
        if s2 == 0 or agg["n_reps"][i] == 0:
            continue

        ax.plot(
            p,
            block["mean"][i],
            marker="o",
            markersize=4,
            linewidth=1.5,
            label=rf"$\sigma^2={s2:g}$",
        )

    ax.set_xlabel("Schatten p")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.2)
    ax.legend(
        title=r"Population heterogeneity",
        frameon=False,
        fontsize=8,
        ncol=2,
    )

    fig.tight_layout()
    out = OUTPUT_DIR / filename
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.show()
    return out


def plot_shape_collapse(agg: dict):
    return _plot_collapse(
        agg,
        agg["shape"],
        r"Global shape susceptibility $\chi_{\mathrm{shape}}$",
        "Geometry-response collapse across extended heterogeneity",
        "v2_collapse_global_shape_susceptibility.png",
    )


def plot_scale_collapse(agg: dict):
    return _plot_collapse(
        agg,
        agg["scale"],
        r"Global scale susceptibility $\chi_{\mathrm{scale}}$",
        "Scale response across extended heterogeneity",
        "v2_collapse_global_scale_susceptibility.png",
    )


def plot_subject_pipeline_ratio(agg: dict):
    return _plot_collapse(
        agg,
        agg["ratio"],
        r"$R=\mathrm{SD}_s[\chi_s]/\langle\chi_s\rangle_s$",
        "Relative subject contribution across extended heterogeneity",
        "v2_subject_pipeline_ratio.png",
    )


# %% 18. SHAPE COLLAPSE + RELATIVE RESIDUALS
def plot_shape_with_residuals(
    agg: dict,
    sigma_ref: float = 1e-4,
):
    import matplotlib.pyplot as plt

    sigma = agg["sigma2"]
    p = agg["p_mid"]
    curves = agg["shape"]["mean"]

    matches = np.where(
        np.isclose(
            sigma,
            sigma_ref,
            rtol=0.0,
            atol=1e-15,
        )
    )[0]

    if len(matches) != 1:
        raise ValueError(
            f"Reference sigma^2={sigma_ref} not found."
        )

    ref_i = int(matches[0])
    ref = curves[ref_i]

    fig = plt.figure(figsize=(9.0, 8.0))
    gs = fig.add_gridspec(
        2,
        1,
        height_ratios=[2.2, 1.0],
        hspace=0.08,
    )

    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1], sharex=ax1)

    markers = [
        "o",
        "s",
        "^",
        "D",
        "v",
        "P",
        "X",
        "<",
        ">",
    ]

    j = 0
    for i, s2 in enumerate(sigma):
        if s2 == 0 or agg["n_reps"][i] == 0:
            continue

        marker = markers[j % len(markers)]
        j += 1

        ax1.plot(
            p,
            curves[i],
            marker=marker,
            linewidth=1.5,
            markersize=4.5,
            label=rf"$\sigma^2={s2:g}$",
        )

        residual = np.divide(
            curves[i] - ref,
            ref,
            out=np.zeros_like(ref),
            where=ref != 0,
        )

        ax2.plot(
            p,
            residual,
            marker=marker,
            linewidth=1.3,
            markersize=4,
        )

    ax1.set_ylabel(
        r"Global shape susceptibility $\chi_{\mathrm{shape}}$"
    )
    ax1.set_title(
        "Geometry-response collapse and deviations "
        "across extended heterogeneity"
    )
    ax1.grid(alpha=0.2)
    ax1.legend(
        title=r"Population heterogeneity",
        frameon=False,
        fontsize=8,
        ncol=2,
    )
    ax1.tick_params(labelbottom=False)

    ax2.axhline(0.0, linewidth=1.0)
    ax2.set_xlabel("Schatten p")
    ax2.set_ylabel(
        r"Relative residual vs $\sigma^2_{\rm ref}=10^{-4}$"
    )
    ax2.grid(alpha=0.2)

    fig.tight_layout()

    out = OUTPUT_DIR / "v2_shape_collapse_with_residuals.png"
    fig.savefig(out, dpi=250, bbox_inches="tight")
    plt.show()
    return out


# %% 19. REALIZED HETEROGENEITY PLOT
def plot_realized_heterogeneity(agg: dict):
    import matplotlib.pyplot as plt

    sigma = agg["sigma2"]
    h = agg["H_F_mean"]
    h_sd = agg["H_F_sd"]

    fig, ax = plt.subplots(figsize=(7.5, 5.2))

    pos = (
        (sigma > 0)
        & np.isfinite(h)
        & (agg["n_reps"] > 0)
    )

    ax.errorbar(
        sigma[pos],
        h[pos],
        yerr=h_sd[pos],
        marker="o",
        linewidth=1.5,
        capsize=3,
    )
    ax.set_xscale("log")

    if (
        len(sigma) > 0
        and sigma[0] == 0
        and np.isfinite(h[0])
        and np.any(pos)
    ):
        xzero = sigma[pos].min() / 2.5
        ax.scatter(
            [xzero],
            [h[0]],
            marker="x",
            s=55,
            label=r"$\sigma^2=0$",
        )
        ax.legend(frameon=False)

    ax.set_xlabel(r"Generative variance $\sigma^2$")
    ax.set_ylabel(r"Realized heterogeneity $H_F$")
    ax.set_title(
        "Realized population heterogeneity "
        "across the extended regime"
    )
    ax.grid(alpha=0.2)

    fig.tight_layout()
    out = OUTPUT_DIR / "v2_realized_heterogeneity.png"
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.show()
    return out


# %% 20. SUMMARY TABLE
def save_summary_csv(agg: dict) -> Path:
    import csv

    path = OUTPUT_DIR / "v2_summary.csv"

    with path.open("w", newline="") as f:
        writer = csv.writer(f)

        writer.writerow(
            [
                "sigma2",
                "n_replicates",
                "H_F_mean",
                "H_F_sd",
                "H_F_rms_mean",
                "H_center_mean",
            ]
        )

        for i, s2 in enumerate(agg["sigma2"]):
            writer.writerow(
                [
                    s2,
                    int(agg["n_reps"][i]),
                    agg["H_F_mean"][i],
                    agg["H_F_sd"][i],
                    agg["H_F_rms_mean"][i],
                    agg["H_center_mean"][i],
                ]
            )

    print(f"[SAVE] {path}")
    return path


# %% 21. NORMALIZED MAIN-FIGURE AGGREGATION

def aggregate_normalized_main_figure(results: list[dict]) -> dict:
    sigma2 = np.asarray(SCI.sigma2_values, dtype=float)
    p_values = np.asarray(SCI.p_values, dtype=float)

    out = {
        "sigma2": sigma2,
        "p": p_values,
        "n_reps": np.zeros(len(sigma2), dtype=int),
        "H_pct_mean": np.full(len(sigma2), np.nan),
        "H_pct_sd": np.full(len(sigma2), np.nan),
    }

    metric_keys = [
        "cumulative_scale_contraction_pct",
        "cumulative_shape_change_pct",
        "cumulative_subject_mean_pct",
        "cumulative_subject_cv_pct",
        "cumulative_pairwise_rank_loss_pct",
    ]
    for key in metric_keys:
        out[key] = {
            "mean": np.full((len(sigma2), len(p_values)), np.nan),
            "sd": np.full((len(sigma2), len(p_values)), np.nan),
            "ci_low": np.full((len(sigma2), len(p_values)), np.nan),
            "ci_high": np.full((len(sigma2), len(p_values)), np.nan),
        }

    for i, s2 in enumerate(sigma2):
        subset = [
            r for r in results
            if np.isclose(r["sigma2"], s2, rtol=0.0, atol=1e-15)
            and np.ndim(r.get("cumulative_scale_contraction_pct", np.nan)) == 1
        ]
        if not subset:
            continue

        out["n_reps"][i] = len(subset)

        h = 100.0 * np.asarray([r["H_F_mean"] for r in subset]) / np.sqrt(2.0)
        out["H_pct_mean"][i] = np.mean(h)
        out["H_pct_sd"][i] = np.std(h, ddof=1) if len(h) > 1 else 0.0

        for key in metric_keys:
            stack = np.vstack([np.asarray(r[key], dtype=float) for r in subset])
            m, sd, lo, hi = _mean_sd_ci(stack)
            out[key]["mean"][i] = m
            out[key]["sd"][i] = sd
            out[key]["ci_low"][i] = lo
            out[key]["ci_high"][i] = hi

    return out


def save_normalized_summary_csv(agg: dict) -> Path:
    import csv
    path = OUTPUT_DIR / "figure2_normalized_summary.csv"
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "sigma2", "p", "n_replicates",
            "H_pct_mean", "H_pct_sd",
            "scale_contraction_pct_mean", "scale_contraction_pct_ci_low", "scale_contraction_pct_ci_high",
            "shape_change_pct_mean", "shape_change_pct_ci_low", "shape_change_pct_ci_high",
            "subject_change_pct_mean", "subject_change_pct_ci_low", "subject_change_pct_ci_high",
            "subject_cv_pct_mean", "subject_cv_pct_ci_low", "subject_cv_pct_ci_high",
        ])
        for i, s2 in enumerate(agg["sigma2"]):
            for j, pval in enumerate(agg["p"]):
                w.writerow([
                    s2, pval, int(agg["n_reps"][i]),
                    agg["H_pct_mean"][i], agg["H_pct_sd"][i],
                    agg["cumulative_scale_contraction_pct"]["mean"][i, j],
                    agg["cumulative_scale_contraction_pct"]["ci_low"][i, j],
                    agg["cumulative_scale_contraction_pct"]["ci_high"][i, j],
                    agg["cumulative_shape_change_pct"]["mean"][i, j],
                    agg["cumulative_shape_change_pct"]["ci_low"][i, j],
                    agg["cumulative_shape_change_pct"]["ci_high"][i, j],
                    agg["cumulative_subject_mean_pct"]["mean"][i, j],
                    agg["cumulative_subject_mean_pct"]["ci_low"][i, j],
                    agg["cumulative_subject_mean_pct"]["ci_high"][i, j],
                    agg["cumulative_subject_cv_pct"]["mean"][i, j],
                    agg["cumulative_subject_cv_pct"]["ci_low"][i, j],
                    agg["cumulative_subject_cv_pct"]["ci_high"][i, j],
                ])
    print(f"[SAVE] {path}")
    return path


def plot_normalized_figure2(agg: dict) -> Path:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(12.5, 9.2))
    axa, axb, axc, axd = axes.ravel()

    # a) Realized heterogeneity on an absolute theoretical scale.
    sigma = agg["sigma2"]
    axa.errorbar(
        sigma,
        agg["H_pct_mean"],
        yerr=agg["H_pct_sd"],
        marker="o",
        linewidth=1.4,
        capsize=2.5,
    )
    axa.set_xscale("symlog", linthresh=1e-4)
    axa.set_xlabel(r"Generative variance $\sigma^2$")
    axa.set_ylabel("Population heterogeneity\n(% of theoretical pairwise diameter)")
    axa.set_title("a  Controlled population heterogeneity", loc="left")

    # For b-d, sigma=0 is undefined because all pairwise distances are zero.
    valid_sigma = [i for i, s2 in enumerate(sigma) if s2 > 0 and agg["n_reps"][i] > 0]
    pvals = agg["p"]

    def draw_curves(ax, block, ylabel, title, skip_p1=False):
        for i in valid_sigma:
            y = block["mean"][i]
            lo = block["ci_low"][i]
            hi = block["ci_high"][i]
            mask = np.isfinite(y)
            if skip_p1:
                mask &= pvals > 1
            ax.plot(pvals[mask], y[mask], marker="o", linewidth=1.2,
                    label=rf"$\sigma^2={sigma[i]:g}$")
            ax.fill_between(pvals[mask], lo[mask], hi[mask], alpha=0.12)
        ax.set_xscale("log", base=2)
        ax.set_xlabel("Schatten order $p$")
        ax.set_ylabel(ylabel)
        ax.set_title(title, loc="left")

    draw_curves(
        axb,
        agg["cumulative_scale_contraction_pct"],
        "Distance-scale contraction\nfrom $p=1$ (%)",
        "b  Global scale response",
    )
    draw_curves(
        axc,
        agg["cumulative_shape_change_pct"],
        "Normalized geometric reorganization\nfrom $p=1$ (%)",
        "c  Global relational response",
    )
    draw_curves(
        axd,
        agg["cumulative_subject_cv_pct"],
        "Relative inter-observation variability\n(CV, %)",
        "d  Observation-specific response",
        skip_p1=True,
    )

    axb.legend(title=r"$\sigma^2$", frameon=False, fontsize=7, ncol=2)
    for ax in axes.ravel():
        ax.grid(alpha=0.2)

    fig.tight_layout()
    out = OUTPUT_DIR / "Figure2_normalized_main.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.show()
    print(f"[SAVE] {out}")
    return out


# %% 22. V3 WORKFLOW

def plot_v3_normalized_results():
    # Cumulative metrics require v3 NPZ files; legacy v1/v2 summaries are not sufficient.
    results = load_all_results(include_legacy_v1=False)
    if not results:
        raise RuntimeError("No v3 simulation results available. Run the v3 grid first.")
    missing = [r for r in results if np.ndim(r.get("cumulative_scale_contraction_pct", np.nan)) != 1]
    if missing:
        raise RuntimeError("Some loaded files do not contain v3 cumulative metrics. Recompute those cells.")
    agg = aggregate_normalized_main_figure(results)
    fig = plot_normalized_figure2(agg)
    csv = save_normalized_summary_csv(agg)
    return {"figure": fig, "summary_csv": csv}



def plot_rank_reorganization_control(agg):
    import matplotlib.pyplot as plt
    p_values = np.asarray(SCI.p_values, dtype=float)
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    vals = agg["cumulative_pairwise_rank_loss_pct"]
    for i, sigma2 in enumerate(SCI.sigma2_values):
        if sigma2 == 0.0:
            continue
        mean = vals["mean"][i]
        lo = vals["ci_low"][i]
        hi = vals["ci_high"][i]
        ax.plot(p_values, mean, marker="o", linewidth=1.4, label=rf"$\sigma^2={sigma2:g}$")
        ax.fill_between(p_values, lo, hi, alpha=0.12)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Schatten order $p$")
    ax.set_ylabel("Pairwise-ordering loss from $p=1$ (%)")
    ax.set_title("Ordinal reorganization of pairwise distances")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    out = OUTPUT_DIR / "rank_reorganization_control.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out

def save_rank_control_csv(agg):
    import csv
    out = OUTPUT_DIR / "rank_reorganization_summary.csv"
    vals = agg["cumulative_pairwise_rank_loss_pct"]
    with out.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sigma2","p","mean_rank_loss_pct","ci_low","ci_high"])
        for i,sigma2 in enumerate(SCI.sigma2_values):
            for j,p in enumerate(SCI.p_values):
                w.writerow([sigma2,p,vals["mean"][i,j],vals["ci_low"][i,j],vals["ci_high"][i,j]])
    return out

def aggregate_existing_v4_results():
    """Aggregate and plot the already-computed v4 NPZ files only.

    This function performs NO simulation and NO SVD/Schatten recomputation.
    It reads exclusively from OUTPUT_DIR (geometry_sensitivity_results_v4_rankcontrol).
    """
    print("[V4-AGG] Aggregating existing v4 rank-control results only")
    results = load_all_results(include_legacy_v1=False)
    if len(results) != len(SCI.sigma2_values) * SCI.n_replicates:
        raise RuntimeError(
            f"Expected {len(SCI.sigma2_values) * SCI.n_replicates} v4 cells, "
            f"found {len(results)} in {OUTPUT_DIR}."
        )
    missing_rank = [
        r for r in results
        if np.ndim(r.get("cumulative_pairwise_rank_loss_pct", np.nan)) != 1
    ]
    if missing_rank:
        raise RuntimeError(
            f"{len(missing_rank)} v4 files are missing cumulative_pairwise_rank_loss_pct."
        )

    agg = aggregate_normalized_main_figure(results)
    fig = plot_normalized_figure2(agg)
    summary_csv = save_normalized_summary_csv(agg)
    rank_fig = plot_rank_reorganization_control(agg)
    rank_csv = save_rank_control_csv(agg)

    print(f"[V4-AGG] Rank figure -> {rank_fig}")
    print(f"[V4-AGG] Rank summary -> {rank_csv}")
    print("[V4-AGG] Done. No simulation was rerun.")
    return {
        "figure": fig,
        "summary_csv": summary_csv,
        "rank_figure": rank_fig,
        "rank_summary_csv": rank_csv,
    }


# %% 23. SHELL ENTRYPOINT
if __name__ == "__main__":
    aggregate_existing_v4_results()
