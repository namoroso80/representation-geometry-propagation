#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Convert PEMS exact low-rank pair spectra to the standard Geometry 2.0 spectral-engine cache format.

Input directory must contain:
  pair_spectra.npz   with array 'eigenvalues' (signed eigenvalues, NaN padded)
  pair_manifest.csv  with columns i,j

Output directory will contain:
  pairwise_singular_values.npy
  pair_indices.npz

For symmetric operator differences, singular values are abs(eigenvalues), so this conversion is exact.
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--geometry-dir", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    z = np.load(args.geometry_dir / "pair_spectra.npz")
    eig = np.asarray(z["eigenvalues"], dtype=np.float32)

    # NaN padding denotes absent zero modes. Convert padding to 0 and take absolute values.
    sing = np.nan_to_num(np.abs(eig), nan=0.0, posinf=0.0, neginf=0.0)

    pairs = pd.read_csv(args.geometry_dir / "pair_manifest.csv")
    if not {"i", "j"}.issubset(pairs.columns):
        raise ValueError("pair_manifest.csv must contain columns i,j")
    if len(pairs) != sing.shape[0]:
        raise ValueError(f"pairs={len(pairs)} spectra={sing.shape[0]}")

    pair_s = pairs["i"].to_numpy(dtype=np.int32)
    pair_t = pairs["j"].to_numpy(dtype=np.int32)

    np.save(args.output_dir / "pairwise_singular_values.npy", sing)
    np.savez_compressed(args.output_dir / "pair_indices.npz", pair_s=pair_s, pair_t=pair_t)

    print(f"[DATA] pairs={len(pairs)} singular_values_shape={sing.shape}")
    print(f"[QC] min={sing.min():.6g} max={sing.max():.6g} finite={np.isfinite(sing).all()}")
    print(f"[SAVE] {args.output_dir / 'pairwise_singular_values.npy'}")
    print(f"[SAVE] {args.output_dir / 'pair_indices.npz'}")
    print("[DONE] Exact conversion: singular values = abs(eigenvalues) for symmetric differences.")

if __name__ == "__main__":
    main()
