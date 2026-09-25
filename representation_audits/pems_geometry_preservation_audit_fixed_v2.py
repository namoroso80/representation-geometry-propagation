#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PEMS-SF geometry-preservation audit for choosing sensor subset size m.

No labels are read.

Reference:
  all 963 sensors.

Candidate subsets:
  top-m sensors from the previously selected unsupervised temporal-variance ranking.

For each representation, build one trace-normalized sensor correlation operator per day.
Compare candidate-vs-full inter-day geometries at p=1 and p=2 using Spearman correlation.

For p=2, distances are computed exactly from Frobenius norms.
For p=1, trace distances require eigvalsh of pairwise operator differences.
"""

from __future__ import annotations
import argparse, re
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from scipy.spatial.distance import squareform
from tqdm.auto import tqdm

N_SENSORS = 963
N_TIME = 144
_float_re = re.compile(r"[-+]?(?:\d*\.\d+|\d+\.?)(?:[eE][-+]?\d+)?")


def args():
    p=argparse.ArgumentParser()
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--rankings", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--m-values", default="16,24,32,48,64,80,96,108")
    p.add_argument("--method", default="temporal_variance")
    return p.parse_args()


def parse(path):
    days=[]
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line_no,line in enumerate(f,1):
            if not line.strip(): continue
            s=line.strip()
            if s.startswith("["): s=s[1:]
            if s.endswith("]"): s=s[:-1]
            parts=s.split(";")
            rows=[]
            if len(parts)==N_SENSORS:
                for part in parts:
                    v=np.fromstring(part,sep=" ",dtype=np.float64)
                    if v.size != N_TIME:
                        v=np.array([float(x) for x in _float_re.findall(part)])
                    rows.append(v)
                if all(len(v)==N_TIME for v in rows):
                    days.append(np.vstack(rows).astype(np.float32)); continue
            v=np.array([float(x) for x in _float_re.findall(s)])
            if v.size != N_SENSORS*N_TIME:
                raise ValueError(f"{path.name} line {line_no}: {v.size}")
            days.append(v.reshape(N_SENSORS,N_TIME).astype(np.float32))
    return np.stack(days)


def operators(X, idx, require_nonconstant=True):
    R=[]
    for day in X:
        Y=day[idx].astype(np.float64)
        sd=Y.std(axis=1,ddof=1)
        if require_nonconstant and np.any(sd<=1e-12):
            raise ValueError("constant selected sensor")
        if not require_nonconstant:
            keep=np.isfinite(sd) & (sd>1e-12)
            Y=Y[keep]
        C=np.corrcoef(Y)
        C=0.5*(C+C.T)
        np.fill_diagonal(C,1.0)
        C/=np.trace(C)
        R.append(C)
    return np.stack(R)


def frob_vec(R):
    n=len(R)
    norms=np.einsum("nij,nij->n",R,R)
    gram=np.einsum("nij,mij->nm",R,R)
    d2=np.maximum(norms[:,None]+norms[None,:]-2*gram,0)
    return squareform(np.sqrt(d2),checks=False)


def trace_vec(R):
    n=len(R); out=np.empty(n*(n-1)//2,float); z=0
    for i in tqdm(range(n-1), desc=f"trace pairs n={n}"):
        for j in range(i+1,n):
            e=np.linalg.eigvalsh(R[i]-R[j])
            out[z]=np.sum(np.abs(e)); z+=1
    return out


def main():
    a=args(); a.output_dir.mkdir(parents=True,exist_ok=True)
    mvals=[int(x) for x in a.m_values.split(",")]

    print("[LOAD] PEMS train+test; labels are NOT read")
    X=np.concatenate([parse(a.data_dir/"PEMS_train"),parse(a.data_dir/"PEMS_test")])
    print(f"[DATA] {X.shape}")

    r=pd.read_csv(a.rankings)
    rr=r[r.method==a.method].sort_values("rank")

    # Sensors must be nonconstant in EVERY day, both for the full reference
    # and for every candidate top-m subset.
    sensor_sd = X.astype(np.float64).std(axis=2, ddof=1)
    globally_valid = np.all(np.isfinite(sensor_sd) & (sensor_sd > 1e-12), axis=0)
    full_idx = np.flatnonzero(globally_valid)
    valid_set = set(map(int, full_idx))
    rr_valid = rr[rr.sensor_index_0based.astype(int).isin(valid_set)].copy()
    order = rr_valid.sensor_index_0based.to_numpy(int)

    print(f"[FILTER] globally valid sensors: {len(full_idx)}/{N_SENSORS}")
    print(f"[FILTER] valid sensors available in {a.method} ranking: {len(order)}")
    if len(order)<max(mvals):
        raise ValueError(f"Only {len(order)} globally valid ranked sensors; need {max(mvals)}")

    pd.DataFrame({
        "sensor_index_0based": full_idx,
        "sensor_index_1based": full_idx + 1
    }).to_csv(a.output_dir/"full_reference_valid_sensors.csv", index=False)

    rr_valid.to_csv(a.output_dir/"filtered_sensor_ranking.csv", index=False)

    # Full reference. p=2 is cheap. Full p=1 on 963x963 would be prohibitive.
    # Because rank <=143, full p=1 is computed in the 144-dimensional temporal dual
    # using AB/BA nonzero-spectrum equivalence after standardized Gram construction.
    # Here we explicitly build full operators only for p=2; for p=1 reference use
    # a rank-reduced eigenspace representation per day.
    print(f"[FULL] valid sensors nonconstant in every day: {len(full_idx)}/{N_SENSORS}")
    if len(full_idx) < 2:
        raise ValueError("Fewer than two globally valid sensors.")

    print("[FULL] building p=2 reference on globally valid sensors")
    Rfull=operators(X,full_idx)
    D2full=frob_vec(Rfull)

    # Exact full p=1 via dense 963 differences is too expensive for an audit.
    # Use m=108 as high-dimensional anchor for trace geometry, and full 963 for Frobenius.
    # This is stated explicitly in outputs rather than silently approximated.
    anchor=max(mvals)
    print(f"[TRACE REFERENCE] using m={anchor} high-dimensional unsupervised anchor")
    Ranchor=operators(X,order[:anchor])
    D1anchor=trace_vec(Ranchor)

    rows=[]
    for m in mvals:
        print(f"[M] {m}")
        R=operators(X,order[:m])
        d2=frob_vec(R)
        if m==anchor:
            d1=D1anchor
        else:
            d1=trace_vec(R)
        rho2=float(spearmanr(d2,D2full).statistic)
        rho1=float(spearmanr(d1,D1anchor).statistic)
        rows.append({
            "m":m,
            "p2_spearman_vs_full963":rho2,
            "p1_spearman_vs_m108_anchor":rho1,
            "p1_reference_note":"m108 anchor; full963 exact trace audit computationally prohibitive",
        })
        print(f"[RESULT] m={m} | p2 vs full963={rho2:.5f} | p1 vs m108={rho1:.5f}")

    out=pd.DataFrame(rows)
    out.to_csv(a.output_dir/"geometry_preservation_by_m.csv",index=False)

    # Smallest m meeting a transparent high-concordance criterion on both audits.
    ok=out[(out.p2_spearman_vs_full963>=.95)&(out.p1_spearman_vs_m108_anchor>=.95)]
    rec=int(ok.m.min()) if len(ok) else None
    text=[
        "PEMS-SF GEOMETRY-PRESERVATION AUDIT",
        "Labels were not read.",
        f"p=2 reference: all sensors nonconstant in every day ({len(full_idx)}/{N_SENSORS}).",
        "p=1 reference: top-108 temporal-variance sensors (explicit computational anchor).",
        "Selection diagnostic threshold: Spearman >= 0.95 for both comparisons.",
        f"smallest_m_meeting_both={rec if rec is not None else 'NONE'}",
        "",
        out.to_string(index=False)
    ]
    (a.output_dir/"geometry_preservation_report.txt").write_text("\n".join(text),encoding="utf-8")
    print("\n".join("[RESULT] "+x for x in text[:6]))
    print(f"[DONE] {a.output_dir}")


if __name__=="__main__":
    main()
