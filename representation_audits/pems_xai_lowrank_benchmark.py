#!/usr/bin/env python3
# PEMS low-rank XAI kernel benchmark against the frozen xai_engine first-order formula.
import argparse, time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.linalg import eigh
from scipy.stats import spearmanr

P=np.array([1,1.25,1.5,1.75,2,3,4,8,16.],float)

def reduced(Fs,Ft,tol=1e-11):
    rs,rt=Fs.shape[1],Ft.shape[1]
    Gss=Fs.T@Fs; Gtt=Ft.T@Ft; Gst=Fs.T@Ft
    G=np.block([[Gss,Gst],[Gst.T,Gtt]]); G=(G+G.T)/2
    g,V=eigh(G,check_finite=False)
    keep=g>tol*max(float(g[-1]),1.)
    g=g[keep]; V=V[:,keep]
    signs=np.r_[np.ones(rs),-np.ones(rt)]
    K=V.T@(signs[:,None]*V); sg=np.sqrt(g)
    H=(sg[:,None]*K)*sg[None,:]; H=(H+H.T)/2
    lam,W=eigh(H,check_finite=False)
    # Eigenvectors of Delta in sensor space: Q = U V g^-1/2 W
    U=np.concatenate([Fs,Ft],axis=1)
    Q=U@(V/np.sqrt(g)[None,:])@W
    return lam,Q

def contrib_from_eig(lam,Q,diag_delta):
    Q2=Q*Q; al=np.abs(lam); sl=np.sign(lam)
    prow=P[None,:]; powers=al[:,None]**prow; sums=powers.sum(0)
    d=np.where(sums>0,sums**(1/P),0); den=np.where(d>0,d**(P-1),1)
    aa=powers/den[None,:]
    bb=sl[:,None]*(al[:,None]**(prow-1))/den[None,:]
    bb[al==0,:]=0
    dgd=Q2@aa; dg=Q2@bb
    c=2*dgd-diag_delta[:,None]*dg
    c[:,d==0]=0
    return c.T

def dense(Fs,Ft):
    D=Fs@Fs.T-Ft@Ft.T; D=(D+D.T)/2
    diag=np.diag(D).copy()
    lam,Q=eigh(D,check_finite=False)
    return contrib_from_eig(lam,Q,diag)

def lowrank(Fs,Ft):
    diag=(Fs*Fs).sum(1)-(Ft*Ft).sum(1)
    lam,Q=reduced(Fs,Ft)
    return contrib_from_eig(lam,Q,diag)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--factors",type=Path,required=True)
    ap.add_argument("--output-dir",type=Path,required=True)
    ap.add_argument("--n-pairs",type=int,default=48)
    ap.add_argument("--n-dense-validation",type=int,default=3)
    ap.add_argument("--random-state",type=int,default=42)
    a=ap.parse_args(); a.output_dir.mkdir(parents=True,exist_ok=True)
    F=np.load(a.factors)["factors"].astype(np.float64)
    n=len(F); rng=np.random.default_rng(a.random_state)
    pairs=[]
    while len(pairs)<a.n_pairs:
        i,j=sorted(rng.choice(n,2,replace=False))
        if (i,j) not in pairs:pairs.append((i,j))
    rows=[]; t0=time.perf_counter()
    for q,(i,j) in enumerate(pairs):
        t=time.perf_counter(); C=lowrank(F[i],F[j]); sec=time.perf_counter()-t
        rows.append({"q":q,"i":i,"j":j,"seconds":sec,"max_abs_contribution":float(np.max(np.abs(C)))})
    total=time.perf_counter()-t0
    vals=[]
    for q,(i,j) in enumerate(pairs[:a.n_dense_validation]):
        t=time.perf_counter(); L=lowrank(F[i],F[j]); tl=time.perf_counter()-t
        t=time.perf_counter(); D=dense(F[i],F[j]); td=time.perf_counter()-t
        for k,p in enumerate(P):
            err=np.max(np.abs(L[k]-D[k]))
            scale=max(np.max(np.abs(D[k])),1e-15)
            rho=spearmanr(L[k],D[k]).statistic
            vals.append({"pair":q,"p":p,"max_abs_error":err,"max_relative_error":err/scale,
                         "spearman_nodes":rho,"lowrank_seconds":tl,"dense_seconds":td})
    pd.DataFrame(rows).to_csv(a.output_dir/"xai_lowrank_benchmark_pairs.csv",index=False)
    V=pd.DataFrame(vals); V.to_csv(a.output_dir/"xai_lowrank_dense_validation.csv",index=False)
    rate=a.n_pairs/total
    report=[
      "PEMS LOW-RANK XAI KERNEL BENCHMARK",
      f"n_pairs={a.n_pairs}",f"nodes={F.shape[1]}",f"rank={F.shape[2]}",
      f"median_seconds_per_pair={pd.DataFrame(rows).seconds.median():.6f}",
      f"throughput_pairs_per_second={rate:.3f}",
      f"geometry_all_pairs_eta_hours={n*(n-1)/2/rate/3600:.3f}",
      f"dense_validation_pairs={a.n_dense_validation}",
      f"max_relative_error={V.max_relative_error.max():.3e}",
      f"min_spearman_nodes={V.spearman_nodes.min():.12f}",
      f"median_lowrank_validation_seconds={V.lowrank_seconds.median():.6f}",
      f"median_dense_validation_seconds={V.dense_seconds.median():.6f}",
      "NOTE: ETA above is pair-kernel geometry only; full support-vector-weighted XAI streaming may differ."
    ]
    (a.output_dir/"xai_lowrank_benchmark_summary.txt").write_text("\n".join(report))
    print("\n".join("[RESULT] "+x for x in report))
if __name__=="__main__": main()
