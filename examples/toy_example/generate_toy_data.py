#!/usr/bin/env python3

from pathlib import Path
import numpy as np
import pandas as pd

SEED = 20260925
N_PER_CLASS = 20
N_NODES = 12
LATENT_RANK = 4

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"
INFO_FILE = HERE / "info.csv"


def make_psd_operator(rng, label):
    """Generate a synthetic PSD, unit-trace operator."""
    X = rng.normal(size=(N_NODES, LATENT_RANK))

    if label == "POSITIVE":
        X[:4, 0] += 0.8
        X[4:8, 1] -= 0.5
    else:
        X[:4, 0] -= 0.3
        X[4:8, 1] += 0.2

    X += 0.25 * rng.normal(size=X.shape)

    A = X @ X.T
    A += 1e-3 * np.eye(N_NODES)
    A /= np.trace(A)

    return A


def main():
    rng = np.random.default_rng(SEED)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    labels = ["NEGATIVE"] * N_PER_CLASS + ["POSITIVE"] * N_PER_CLASS
    rows = []

    for idx, label in enumerate(labels):
        A = make_psd_operator(rng, label)

        filename = f"adjacency_{idx:03d}.csv"
        np.savetxt(
            DATA_DIR / filename,
            A,
            delimiter=",",
            fmt="%.12g",
        )

        rows.append({
            "sample_id": f"toy_{idx:03d}",
            "label": label,
        })

    pd.DataFrame(rows).to_csv(INFO_FILE, index=False)

    print(f"Generated {len(labels)} synthetic observations")
    print(f"Matrix dimension: {N_NODES} x {N_NODES}")
    print(f"Data directory: {DATA_DIR}")
    print(f"Metadata: {INFO_FILE}")


if __name__ == "__main__":
    main()
