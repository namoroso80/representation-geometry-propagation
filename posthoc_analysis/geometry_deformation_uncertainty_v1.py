#!/usr/bin/env python3
"""
geometry_deformation_uncertainty_v1.py

Post-hoc subject-resampling uncertainty for empirical geometry deformation.
No classifiers, kernels, SVDs or XAI are recomputed.

For each dataset and Schatten order p, the script loads D_1 and D_p using the
same distance-loading logic as geometry_propagation_mechanism_analysis_v3_2.py,
then repeatedly resamples observations and recomputes:

  scale contraction (%) = 100 * [1 - median(D_p) / median(D_1)]
  ordering loss (%)      = 100 * [1 - Spearman(vec(D_p), vec(D_1))]

To keep the very large MIMII analysis tractable, resampling uses at most
--max-subjects observations per replicate. For smaller datasets all subjects
are sampled. Sampling is with replacement; pairs corresponding to two copies
of the same original observation are excluded so that artificial zero-distance
duplicate pairs do not enter the statistics.

Outputs:
  geometry_deformation_uncertainty.csv
  geometry_deformation_uncertainty_replicates.csv
"""

import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import rankdata

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--reference-p", type=float, default=1.0)
    p.add_argument("--n-resamples", type=int, default=500)
    p.add_argument("--max-subjects", type=int, default=1000,
                   help="Cap only for resampling; full-data point estimates remain unchanged.")
    p.add_argument("--seed", type=int, default=20260823)
    return p.parse_args()

def dpath(root, p):
    root = Path(root).expanduser()
    for s in [str(float(p)).replace(".","p"), f"{p:g}", str(p).replace(".","p")]:
        for q in [root/f"distance_p{s}.npy", root/f"distance_{s}.npy"]:
            if q.exists():
                return q
    raise FileNotFoundError(f"distance p={p} under {root}")

def distance_from_spectra(source, p, n_subjects):
    source = Path(source).expanduser()
    sv = np.load(source/"pairwise_singular_values.npy", mmap_mode="r")
    z = np.load(source/"pair_indices.npz")
    ii = z["pair_s"].astype(int)
    jj = z["pair_t"].astype(int)
    block = np.asarray(sv, dtype=float)
    if float(p) == 1.0:
        vals = np.sum(block, axis=1)
    elif float(p) == 2.0:
        vals = np.sqrt(np.sum(block*block, axis=1))
    else:
        vals = np.sum(np.abs(block)**float(p), axis=1)**(1.0/float(p))
    D = np.zeros((n_subjects, n_subjects), float)
    D[ii,jj] = vals
    D[jj,ii] = vals
    return D

def load_distance(sp, p, n_subjects):
    if sp.get("distance_dir"):
        return np.load(dpath(sp["distance_dir"], p), mmap_mode="r")
    if sp.get("spectra_source"):
        return distance_from_spectra(sp["spectra_source"], p, n_subjects)
    raise ValueError(f'{sp.get("name","dataset")}: provide distance_dir or spectra_source')

def upper_from_indices(D, idx):
    m = len(idx)
    a,b = np.triu_indices(m, 1)
    keep = idx[a] != idx[b]  # remove pairs made of duplicate bootstrap copies
    return np.asarray(D[idx[a[keep]], idx[b[keep]]], dtype=float)

def spearman_fast(x, y):
    # Exact Spearman with average ranks for ties.
    rx = rankdata(x, method="average")
    ry = rankdata(y, method="average")
    rx -= rx.mean()
    ry -= ry.mean()
    den = np.sqrt(np.dot(rx,rx) * np.dot(ry,ry))
    return np.dot(rx,ry)/den if den > 0 else np.nan

def point_estimate(D1, Dp):
    iu = np.triu_indices(D1.shape[0], 1)
    v1 = np.asarray(D1[iu], dtype=float)
    vp = np.asarray(Dp[iu], dtype=float)
    scale = 100.0 * (1.0 - np.median(vp)/np.median(v1))
    order = 100.0 * (1.0 - spearman_fast(v1, vp))
    return scale, order

def main():
    a = parse_args()
    a.output_dir.mkdir(parents=True, exist_ok=True)
    specs = json.loads(a.config.read_text())["datasets"]
    rng = np.random.default_rng(a.seed)

    rep_rows = []
    sum_rows = []

    for sp in specs:
        name = sp["name"]
        print(f"[DATASET] {name}")
        summ = pd.read_csv(Path(sp["summary"]).expanduser())
        ps = summ["p"].astype(float).tolist()

        # Infer n from OOF, exactly as original posthoc script.
        oof = pd.read_csv(Path(sp["oof"]).expanduser())
        n = int(oof.subject_idx.nunique())
        m = min(n, a.max_subjects)

        print(f"  n={n}; resample size={m}; B={a.n_resamples}")
        D1 = load_distance(sp, a.reference_p, n)

        for p in ps:
            Dp = D1 if np.isclose(p, a.reference_p) else load_distance(sp, p, n)
            full_scale, full_order = point_estimate(D1, Dp)

            bs_scale = np.empty(a.n_resamples, float)
            bs_order = np.empty(a.n_resamples, float)

            for b in range(a.n_resamples):
                idx = rng.integers(0, n, size=m)
                v1 = upper_from_indices(D1, idx)
                vp = upper_from_indices(Dp, idx)

                bs_scale[b] = 100.0 * (1.0 - np.median(vp)/np.median(v1))
                bs_order[b] = 100.0 * (1.0 - spearman_fast(v1, vp))

                rep_rows.append({
                    "dataset": name, "p": p, "resample": b,
                    "n_subjects_full": n, "n_subjects_resample": m,
                    "scale_contraction_pct": bs_scale[b],
                    "ordering_loss_pct": bs_order[b],
                })

            sl, sh = np.nanpercentile(bs_scale, [2.5,97.5])
            ol, oh = np.nanpercentile(bs_order, [2.5,97.5])

            sum_rows.append({
                "dataset": name, "p": p,
                "n_subjects_full": n, "n_subjects_resample": m,
                "n_resamples": a.n_resamples,
                "scale_contraction_pct": full_scale,
                "scale_ci95_low": sl, "scale_ci95_high": sh,
                "ordering_loss_pct": full_order,
                "ordering_ci95_low": ol, "ordering_ci95_high": oh,
            })
            print(f"  p={p:g}: scale={full_scale:.3f}% [{sl:.3f},{sh:.3f}] "
                  f"order={full_order:.3f}% [{ol:.3f},{oh:.3f}]")

            if Dp is not D1:
                del Dp

        del D1

    reps = pd.DataFrame(rep_rows)
    sums = pd.DataFrame(sum_rows)
    reps.to_csv(a.output_dir/"geometry_deformation_uncertainty_replicates.csv", index=False)
    sums.to_csv(a.output_dir/"geometry_deformation_uncertainty.csv", index=False)

    meta = {
        "n_resamples": a.n_resamples,
        "max_subjects": a.max_subjects,
        "seed": a.seed,
        "reference_p": a.reference_p,
        "note": ("Percentile subject-bootstrap intervals. For datasets with n > max_subjects, "
                 "each bootstrap replicate contains max_subjects sampled observations; "
                 "full-data point estimates always use all observations. Duplicate-copy self-pairs "
                 "are excluded.")
    }
    (a.output_dir/"uncertainty_metadata.json").write_text(json.dumps(meta, indent=2))
    print("[DONE]", a.output_dir/"geometry_deformation_uncertainty.csv")

if __name__ == "__main__":
    main()
