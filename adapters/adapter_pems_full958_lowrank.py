#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PEMS-SF full-sensor low-rank adapter for Geometry 2.0.

Frozen representation:
- 440 days (PEMS_train + PEMS_test)
- retain sensors nonconstant in every day (expected 958/963)
- each sensor profile standardized across the 144 within-day time points
- rho_s = Z_s Z_s^T / ((T-1) * n_valid), trace one
- save exact thin factor F_s such that rho_s = F_s F_s^T, rank <= 143
- binary task: weekday (labels 1..5) vs weekend (labels 6..7)
- original 7-class labels are preserved in metadata
"""

from __future__ import annotations
import argparse, re
from pathlib import Path
import numpy as np
import pandas as pd

N_SENSORS=963
N_TIME=144
_float_re=re.compile(r"[-+]?(?:\d*\.\d+|\d+\.?)(?:[eE][-+]?\d+)?")


def parse_args():
    p=argparse.ArgumentParser()
    p.add_argument("--data-dir",type=Path,required=True)
    p.add_argument("--output-dir",type=Path,required=True)
    p.add_argument("--overwrite",action="store_true")
    return p.parse_args()


def parse_matrix_file(path):
    days=[]
    with path.open("r",encoding="utf-8",errors="replace") as f:
        for ln,line in enumerate(f,1):
            if not line.strip(): continue
            s=line.strip()
            if s.startswith("["): s=s[1:]
            if s.endswith("]"): s=s[:-1]
            parts=s.split(";")
            if len(parts)==N_SENSORS:
                rows=[]
                good=True
                for part in parts:
                    v=np.fromstring(part,sep=" ",dtype=np.float64)
                    if v.size!=N_TIME:
                        v=np.array([float(x) for x in _float_re.findall(part)])
                    if v.size!=N_TIME:
                        good=False; break
                    rows.append(v)
                if good:
                    days.append(np.vstack(rows).astype(np.float32)); continue
            v=np.array([float(x) for x in _float_re.findall(s)])
            if v.size!=N_SENSORS*N_TIME:
                raise ValueError(f"{path.name} line {ln}: {v.size} values")
            days.append(v.reshape(N_SENSORS,N_TIME).astype(np.float32))
    return np.stack(days)


def parse_labels(path):
    txt=path.read_text(encoding="utf-8").strip().strip("[]")
    return np.fromstring(txt,sep=" ",dtype=int)


def main():
    a=parse_args()
    a.output_dir.mkdir(parents=True,exist_ok=True)
    if not a.overwrite and (a.output_dir/"factors.npz").exists():
        raise FileExistsError("Output exists; use --overwrite")

    print("[LOAD] matrices")
    tr=parse_matrix_file(a.data_dir/"PEMS_train")
    te=parse_matrix_file(a.data_dir/"PEMS_test")
    X=np.concatenate([tr,te],axis=0)
    split=np.array(["train"]*len(tr)+["test"]*len(te))

    print("[LOAD] labels (only after representation design was frozen)")
    y7=np.concatenate([
        parse_labels(a.data_dir/"PEMS_trainlabels"),
        parse_labels(a.data_dir/"PEMS_testlabels")
    ])
    if len(y7)!=len(X):
        raise ValueError(f"labels={len(y7)} days={len(X)}")
    ybin=np.where(np.isin(y7,[6,7]),"weekend","weekday")

    # Fixed unsupervised validity filter.
    sd_all=X.astype(np.float64).std(axis=2,ddof=1)
    valid=np.all(np.isfinite(sd_all)&(sd_all>1e-12),axis=0)
    idx=np.flatnonzero(valid)
    n=len(idx)
    print(f"[FILTER] valid sensors={n}/{N_SENSORS}")

    factors=[]
    qc=[]
    denom=np.sqrt((N_TIME-1)*n)

    for d,day in enumerate(X):
        Y=day[idx].astype(np.float64)
        mu=Y.mean(axis=1,keepdims=True)
        sd=Y.std(axis=1,ddof=1,keepdims=True)
        Z=(Y-mu)/sd
        B=Z/denom
        U,s,_=np.linalg.svd(B,full_matrices=False)
        keep=s>1e-12
        F=(U[:,keep]*s[keep][None,:]).astype(np.float32)
        factors.append(F)
        qc.append({
            "sample_id":f"PEMS_{d+1:04d}",
            "rank":int(keep.sum()),
            "trace":float(np.sum(s[keep]**2)),
            "largest_eigenvalue":float(s[0]**2),
            "smallest_nonzero_eigenvalue":float(s[keep][-1]**2),
        })
        if d==0 or (d+1)%40==0 or d+1==len(X):
            print(f"[FACTORS] {d+1}/{len(X)}")

    # Same rank is expected, permitting a compact tensor.
    ranks=[f.shape[1] for f in factors]
    if len(set(ranks))!=1:
        raise ValueError(f"Variable ranks found: {sorted(set(ranks))}")
    F=np.stack(factors,axis=0)  # days x sensors x rank

    np.savez_compressed(
        a.output_dir/"factors.npz",
        factors=F,
        valid_sensor_index_0based=idx.astype(np.int32),
        valid_sensor_index_1based=(idx+1).astype(np.int32),
    )

    info=pd.DataFrame({
        "sample_id":[f"PEMS_{i+1:04d}" for i in range(len(X))],
        "source_split":split,
        "original_day_label":y7,
        "diagnosis":ybin,
        "factor_index":np.arange(len(X),dtype=int),
    })
    info.to_csv(a.output_dir/"info.csv",index=False)
    pd.DataFrame(qc).to_csv(a.output_dir/"adapter_qc.csv",index=False)
    pd.DataFrame({
        "sensor_index_0based":idx,
        "sensor_index_1based":idx+1
    }).to_csv(a.output_dir/"valid_sensors.csv",index=False)

    print("[DONE] PEMS low-rank adapter")
    print(f"[DATA] days={len(X)} valid_sensors={n} factor_shape={F.shape}")
    print("[LABELS]")
    print(info["diagnosis"].value_counts().to_string())
    print("[ORIGINAL LABELS]")
    print(info["original_day_label"].value_counts().sort_index().to_string())
    q=pd.DataFrame(qc)
    print(f"[QC] rank={q['rank'].min()}-{q['rank'].max()} trace={q['trace'].min():.9f}-{q['trace'].max():.9f}")
    print(f"[SAVE] {a.output_dir}")


if __name__=="__main__":
    main()
