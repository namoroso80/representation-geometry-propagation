#!/usr/bin/env python3
"""
Auxiliary heterogeneity audit for Geometry Sensitivity v1.

Purpose
-------
Verify that the generative control parameter sigma^2 produces the intended
increase in realized population heterogeneity.

This script DOES NOT rerun the Schatten-p susceptibility pipeline.
It reconstructs exactly the same synthetic populations used by the v1 main
script, using the same base seed and child-seed rule, and computes independent
heterogeneity measures.

Inputs
------
geometry_sensitivity_results_v1/config.json

Outputs
-------
geometry_sensitivity_results_v1/heterogeneity_audit.csv
geometry_sensitivity_results_v1/heterogeneity_audit_summary.csv
geometry_sensitivity_results_v1/heterogeneity_vs_sigma2.png
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm


# %% 0. PATHS
RESULTS_DIR = Path("geometry_sensitivity_results_v1")
CONFIG_FILE = RESULTS_DIR / "config.json"

DETAIL_CSV = RESULTS_DIR / "heterogeneity_audit.csv"
SUMMARY_CSV = RESULTS_DIR / "heterogeneity_audit_summary.csv"
PLOT_FILE = RESULTS_DIR / "heterogeneity_vs_sigma2.png"


# %% 1. LOAD CONFIGURATION
def load_config() -> dict:
    if not CONFIG_FILE.exists():
        raise FileNotFoundError(
            f"Missing configuration file: {CONFIG_FILE.resolve()}"
        )

    config = json.loads(CONFIG_FILE.read_text())

    sci = config["scientific_config"]
    comp = config["compute_config"]

    print("[SETUP] Heterogeneity audit")
    print(
        f"        nodes={sci['n_nodes']}, "
        f"subjects={sci['n_subjects']}, "
        f"replicates={sci['n_replicates']}"
    )
    print(f"        source={CONFIG_FILE.resolve()}")

    return config


# %% 2. EXACT REPRODUCTION OF THE V1 GENERATIVE MODEL
def make_rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def child_seed(base_seed: int, sigma_index: int, replicate: int) -> int:
    """
    Must match the v1 main script exactly.
    """
    return int(base_seed + 100_000 * sigma_index + replicate)


def make_reference_factor(
    n_nodes: int,
    rng: np.random.Generator,
    dtype: np.dtype,
) -> np.ndarray:
    """
    Must match the v1 main script exactly.
    """
    return rng.normal(
        loc=0.0,
        scale=1.0,
        size=(n_nodes, n_nodes),
    ).astype(dtype)


def factor_to_density(x: np.ndarray) -> np.ndarray:
    rho = x @ x.T
    return rho / np.trace(rho)


def generate_population(
    x0: np.ndarray,
    sigma2: float,
    n_subjects: int,
    rng: np.random.Generator,
    dtype: np.dtype,
) -> np.ndarray:
    """
    Must match the v1 main script exactly.

    X_s = X_0 + sigma G_s
    rho_s = X_s X_s^T / Tr(X_s X_s^T)
    """
    sigma = math.sqrt(sigma2)
    n_nodes = x0.shape[0]

    population = np.empty(
        (n_subjects, n_nodes, n_nodes),
        dtype=dtype,
    )

    if sigma2 == 0.0:
        population[:] = factor_to_density(x0)
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


# %% 3. REALIZED HETEROGENEITY MEASURES
def realized_heterogeneity(population: np.ndarray) -> dict[str, float]:
    """
    Compute p-independent realized population heterogeneity.

    Primary measure
    ---------------
    H_F_mean:
        mean pairwise Frobenius distance

        H_F = 2/[N(N-1)] sum_{s<t} ||rho_s-rho_t||_F

    Controls
    --------
    H_F_rms:
        root mean square pairwise Frobenius distance

    H_center:
        RMS Frobenius dispersion around the arithmetic population mean

        sqrt[ mean_s ||rho_s-rho_bar||_F^2 ]

    These measures do not use Schatten p as a control parameter.
    """
    n_subjects = population.shape[0]

    # Squared Frobenius norms of each matrix.
    flat = population.reshape(n_subjects, -1)
    norms2 = np.einsum("ij,ij->i", flat, flat)

    # Pairwise squared Frobenius distances:
    # ||A-B||_F^2 = ||A||_F^2 + ||B||_F^2 - 2 <A,B>
    gram = flat @ flat.T
    d2 = norms2[:, None] + norms2[None, :] - 2.0 * gram
    d2 = np.maximum(d2, 0.0)

    iu = np.triu_indices(n_subjects, k=1)
    pair_d2 = d2[iu]
    pair_d = np.sqrt(pair_d2)

    h_f_mean = float(np.mean(pair_d))
    h_f_rms = float(np.sqrt(np.mean(pair_d2)))

    rho_bar = np.mean(population, axis=0)
    centered = population - rho_bar
    h_center = float(
        np.sqrt(
            np.mean(
                np.sum(centered * centered, axis=(1, 2))
            )
        )
    )

    return {
        "H_F_mean": h_f_mean,
        "H_F_rms": h_f_rms,
        "H_center": h_center,
    }


# %% 4. RUN AUDIT
def run_audit() -> list[dict]:
    config = load_config()
    sci = config["scientific_config"]
    comp = config["compute_config"]

    n_nodes = int(sci["n_nodes"])
    n_subjects = int(sci["n_subjects"])
    sigma2_values = tuple(float(x) for x in sci["sigma2_values"])
    n_replicates = int(sci["n_replicates"])
    base_seed = int(sci["random_seed"])
    dtype = np.dtype(comp["dtype"])

    # Reproduce X0 exactly as in the main script.
    base_rng = make_rng(base_seed)
    x0 = make_reference_factor(n_nodes, base_rng, dtype)

    rows: list[dict] = []
    total = len(sigma2_values) * n_replicates

    print(f"[AUDIT] Conditions to reconstruct: {total}")

    progress = tqdm(total=total, desc="Heterogeneity audit")

    for sigma_index, sigma2 in enumerate(sigma2_values):
        print(
            f"[SIGMA] {sigma_index + 1}/{len(sigma2_values)} "
            f"| sigma^2={sigma2:g}"
        )

        for replicate in range(n_replicates):
            seed = child_seed(
                base_seed,
                sigma_index,
                replicate,
            )
            rng = make_rng(seed)

            population = generate_population(
                x0=x0,
                sigma2=sigma2,
                n_subjects=n_subjects,
                rng=rng,
                dtype=dtype,
            )

            measures = realized_heterogeneity(population)

            rows.append(
                {
                    "sigma2": sigma2,
                    "replicate": replicate,
                    "seed": seed,
                    **measures,
                }
            )

            progress.update(1)

    progress.close()
    print("[DONE] Heterogeneity reconstruction completed")
    return rows


# %% 5. SAVE DETAIL AND SUMMARY
def save_detail(rows: list[dict]) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "sigma2",
        "replicate",
        "seed",
        "H_F_mean",
        "H_F_rms",
        "H_center",
    ]

    with DETAIL_CSV.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"[SAVE] {DETAIL_CSV.resolve()}")


def summarize(rows: list[dict]) -> list[dict]:
    sigma_values = sorted({row["sigma2"] for row in rows})
    summary: list[dict] = []

    measures = ["H_F_mean", "H_F_rms", "H_center"]

    for sigma2 in sigma_values:
        subset = [row for row in rows if row["sigma2"] == sigma2]

        record = {
            "sigma2": sigma2,
            "n_replicates": len(subset),
        }

        for measure in measures:
            values = np.asarray(
                [row[measure] for row in subset],
                dtype=float,
            )

            record[f"{measure}_mean"] = float(np.mean(values))
            record[f"{measure}_sd"] = float(
                np.std(values, ddof=1)
                if len(values) > 1
                else 0.0
            )

        summary.append(record)

    return summary


def save_summary(summary: list[dict]) -> None:
    fieldnames = list(summary[0].keys())

    with SUMMARY_CSV.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(summary)

    print(f"[SAVE] {SUMMARY_CSV.resolve()}")


# %% 6. DIAGNOSTIC PLOT
def plot_summary(summary: list[dict]) -> None:
    import matplotlib.pyplot as plt

    sigma2 = np.asarray(
        [row["sigma2"] for row in summary],
        dtype=float,
    )

    h_mean = np.asarray(
        [row["H_F_mean_mean"] for row in summary],
        dtype=float,
    )

    h_sd = np.asarray(
        [row["H_F_mean_sd"] for row in summary],
        dtype=float,
    )

    fig, ax = plt.subplots(figsize=(7.5, 5.2))

    # sigma^2 = 0 is shown separately because log scale cannot display zero.
    zero_mask = sigma2 == 0
    pos_mask = sigma2 > 0

    if np.any(pos_mask):
        ax.errorbar(
            sigma2[pos_mask],
            h_mean[pos_mask],
            yerr=h_sd[pos_mask],
            marker="o",
            linewidth=1.5,
            capsize=3,
        )
        ax.set_xscale("log")

    if np.any(zero_mask):
        ax.scatter(
            [sigma2[pos_mask].min() / 2.5 if np.any(pos_mask) else 1e-5],
            [h_mean[zero_mask][0]],
            marker="x",
            s=55,
            label=r"$\sigma^2=0$",
        )
        ax.legend(frameon=False)

    ax.set_xlabel(r"Generative variance $\sigma^2$")
    ax.set_ylabel(r"Realized heterogeneity $H_F$")
    ax.set_title("Realized population heterogeneity")
    ax.grid(alpha=0.2)

    fig.tight_layout()
    fig.savefig(
        PLOT_FILE,
        dpi=220,
        bbox_inches="tight",
    )

    print(f"[SAVE] {PLOT_FILE.resolve()}")
    plt.show()


# %% 7. MAIN
if __name__ == "__main__":
    rows = run_audit()
    save_detail(rows)

    summary = summarize(rows)
    save_summary(summary)

    print("\n[SUMMARY]")
    for row in summary:
        print(
            f"  sigma^2={row['sigma2']:g} | "
            f"H_F={row['H_F_mean_mean']:.6g} "
            f"+/- {row['H_F_mean_sd']:.3g}"
        )

    plot_summary(summary)
