#!/usr/bin/env python3

import math
import numpy as np

N_NODES = 100
N_SUBJECTS = 200
N_REPLICATES = 100
SIGMA2 = 1.0
BASE_SEED = 20260812


def factor_to_density(X):
    rho = X @ X.T
    return rho / np.trace(rho)


def mean_pairwise_frobenius(population):
    n = population.shape[0]
    total = 0.0
    count = 0
    for i in range(n - 1):
        diff = population[i + 1:] - population[i]
        d = np.linalg.norm(diff, axis=(1, 2))
        total += d.sum()
        count += len(d)
    return total / count


def generate_shared_population(rng, sigma2=1.0):
    sigma = math.sqrt(sigma2)
    X0 = rng.normal(0.0, 1.0, size=(N_NODES, N_NODES))

    population = np.empty(
        (N_SUBJECTS, N_NODES, N_NODES),
        dtype=np.float64,
    )

    for s in range(N_SUBJECTS):
        Gs = rng.normal(0.0, 1.0, size=(N_NODES, N_NODES))
        Xs = X0 + sigma * Gs
        population[s] = factor_to_density(Xs)

    return population


def generate_independent_population(rng):
    population = np.empty(
        (N_SUBJECTS, N_NODES, N_NODES),
        dtype=np.float64,
    )

    for s in range(N_SUBJECTS):
        Gs = rng.normal(0.0, 1.0, size=(N_NODES, N_NODES))
        population[s] = factor_to_density(Gs)

    return population


def main():
    shared_values = []
    independent_values = []

    for rep in range(N_REPLICATES):
        rng_shared = np.random.default_rng(BASE_SEED + rep)
        rng_independent = np.random.default_rng(
            BASE_SEED + 1_000_000 + rep
        )

        shared_pop = generate_shared_population(
            rng_shared,
            sigma2=SIGMA2,
        )
        independent_pop = generate_independent_population(
            rng_independent
        )

        H_shared = mean_pairwise_frobenius(shared_pop)
        H_independent = mean_pairwise_frobenius(independent_pop)

        shared_values.append(H_shared)
        independent_values.append(H_independent)

        print(
            f"[{rep + 1:03d}/{N_REPLICATES}] "
            f"H_shared={H_shared:.6f} | "
            f"H_ind={H_independent:.6f} | "
            f"ratio={100 * H_shared / H_independent:.2f}%"
        )

    shared_values = np.asarray(shared_values)
    independent_values = np.asarray(independent_values)

    mean_shared = shared_values.mean()
    mean_independent = independent_values.mean()
    ratio_of_means = 100.0 * mean_shared / mean_independent

    replicate_ratios = 100.0 * shared_values / independent_values
    mean_ratio = replicate_ratios.mean()
    sd_ratio = replicate_ratios.std(ddof=1)

    print("\n----------------------------------------")
    print("FINAL INDEPENDENT-REFERENCE AUDIT")
    print("----------------------------------------")
    print(
        f"Shared-reference heterogeneity (sigma^2={SIGMA2:g}): "
        f"{mean_shared:.6f} +/- {shared_values.std(ddof=1):.6f}"
    )
    print(
        f"Independent-reference heterogeneity: "
        f"{mean_independent:.6f} +/- "
        f"{independent_values.std(ddof=1):.6f}"
    )
    print(f"\nRatio of ensemble means: {ratio_of_means:.2f}%")
    print(
        f"Mean replicate-wise ratio: "
        f"{mean_ratio:.2f}% +/- {sd_ratio:.2f}%"
    )


if __name__ == "__main__":
    main()
