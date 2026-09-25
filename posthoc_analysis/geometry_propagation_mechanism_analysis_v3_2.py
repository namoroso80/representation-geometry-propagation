#!/usr/bin/env python3
import argparse,json
from pathlib import Path
import numpy as np,pandas as pd,matplotlib.pyplot as plt
from scipy.stats import spearmanr

def args():
 p=argparse.ArgumentParser()
 p.add_argument("--config",type=Path,required=True);p.add_argument("--output-dir",type=Path,required=True)
 p.add_argument("--reference-p",type=float,default=1.0);p.add_argument("--boundary-z",type=float,default=.5)
 p.add_argument("--small-gap-quantile",type=float,default=.10);p.add_argument("--dpi",type=int,default=250)
 return p.parse_args()

def dpath(root,p):
 root=Path(root).expanduser()
 for s in [str(float(p)).replace(".","p"),f"{p:g}",str(p).replace(".","p")]:
  for q in [root/f"distance_p{s}.npy",root/f"distance_{s}.npy"]:
   if q.exists(): return q
 raise FileNotFoundError(f"distance p={p} under {root}")

def distance_from_spectra(source,p,n_subjects):
 source=Path(source).expanduser()
 sv=np.load(source/"pairwise_singular_values.npy",mmap_mode="r")
 z=np.load(source/"pair_indices.npz")
 ii=z["pair_s"].astype(int);jj=z["pair_t"].astype(int)
 block=np.asarray(sv,dtype=float)
 if float(p)==1.0: vals=np.sum(block,axis=1)
 elif float(p)==2.0: vals=np.sqrt(np.sum(block*block,axis=1))
 else: vals=np.sum(np.abs(block)**float(p),axis=1)**(1.0/float(p))
 D=np.zeros((n_subjects,n_subjects),float)
 D[ii,jj]=vals;D[jj,ii]=vals
 return D

def load_distance(sp,p,n_subjects):
 if sp.get("distance_dir"):
  return np.load(dpath(sp["distance_dir"],p))
 if sp.get("spectra_source"):
  return distance_from_spectra(sp["spectra_source"],p,n_subjects)
 raise ValueError(f'{sp.get("name","dataset")}: provide distance_dir or spectra_source')

def upper(D):
 i,j=np.triu_indices(D.shape[0],1);return D[i,j].astype(float)

def robust_z(x):
 x=np.asarray(x,float);m=np.median(x);s=1.4826*np.median(np.abs(x-m))
 if not np.isfinite(s) or s<=1e-12:s=np.std(x,ddof=1)
 return (x-m)/s if np.isfinite(s) and s>1e-12 else np.zeros_like(x)

def kernel_fold_diagnostics(name,sp,oof,ref_p):
 fs=pd.read_csv(Path(sp["fold_summary"]).expanduser())
 gamma_col=next((c for c in ["gamma","gamma_best","best_gamma","gamma_selected"] if c in fs.columns),None)
 if gamma_col is None:
  raise ValueError(f"{name}: gamma column not found in fold_summary: {list(fs.columns)}")

 if "repeat" not in fs.columns or "fold_within_repeat" not in fs.columns:
  if "outer_fold" not in fs.columns:
   raise ValueError(f"{name}: need either repeat+fold_within_repeat or outer_fold in fold_summary")
  n_splits=int(oof["fold_within_repeat"].max())
  fs=fs.copy()
  fs["repeat"]=((fs["outer_fold"].astype(int)-1)//n_splits)+1
  fs["fold_within_repeat"]=((fs["outer_fold"].astype(int)-1)%n_splits)+1
  print(f"[KERNEL] {name}: reconstructed repeat/fold from outer_fold using n_splits={n_splits}")

 nsub=int(oof.subject_idx.nunique())
 all_ids=np.sort(oof.subject_idx.unique().astype(int))
 ps=sorted(oof.p.astype(float).unique())
 rows=[]

 fold_specs=[]
 for rep in sorted(oof.repeat.unique()):
  for fold in sorted(oof[oof.repeat==rep].fold_within_repeat.unique()):
   test_ref=oof[(oof.repeat==rep)&(oof.fold_within_repeat==fold)&np.isclose(oof.p,ref_p)].subject_idx.unique().astype(int)
   train_ref=np.setdiff1d(all_ids,test_ref)
   rr1=fs[(fs["repeat"]==rep)&(fs["fold_within_repeat"]==fold)&np.isclose(fs.p.astype(float),ref_p)]
   if len(rr1)!=1:
    raise ValueError(f"{name}: reference gamma lookup failed rep={rep} fold={fold}; rows={len(rr1)}")
   g1=float(rr1.iloc[0][gamma_col])
   fold_specs.append((int(rep),int(fold),train_ref,g1))

 print(f"[KERNEL] {name}: loading reference distance matrix p={ref_p:g}")
 D1=load_distance(sp,ref_p,nsub)

 ref_kernel={}
 for rep,fold,train_ref,g1 in fold_specs:
  a=upper(D1[np.ix_(train_ref,train_ref)])
  ref_kernel[(rep,fold)]=(train_ref,g1,np.exp(-g1*a*a))

 for p_index,p in enumerate(ps,1):
  print(f"[KERNEL] {name}: p={p:g} ({p_index}/{len(ps)})")
  Dp=D1 if np.isclose(p,ref_p) else load_distance(sp,p,nsub)

  for rep,fold,train_ref,g1 in fold_specs:
   rr=fs[(fs["repeat"]==rep)&(fs["fold_within_repeat"]==fold)&np.isclose(fs.p.astype(float),p)]
   if len(rr)!=1:
    raise ValueError(f"{name}: gamma lookup failed rep={rep} fold={fold} p={p}; rows={len(rr)}")
   gp=float(rr.iloc[0][gamma_col])

   test_p=oof[(oof.repeat==rep)&(oof.fold_within_repeat==fold)&np.isclose(oof.p,p)].subject_idx.unique().astype(int)
   train_p=np.setdiff1d(all_ids,test_p)
   if np.array_equal(train_ref,train_p):
    train=train_ref
    ka=ref_kernel[(rep,fold)][2]
   else:
    train=np.intersect1d(train_ref,train_p)
    aa=upper(D1[np.ix_(train,train)])
    ka=np.exp(-g1*aa*aa)

   b=upper(Dp[np.ix_(train,train)])
   kb=np.exp(-gp*b*b)
   rho=float(spearmanr(ka,kb).statistic)

   rows.append(dict(
      dataset=name,repeat=rep,fold_within_repeat=fold,p=float(p),
      gamma_p1=g1,gamma_p=gp,n_train=len(train),
      kernel_spearman_vs_p1=rho,
      kernel_relational_loss=1-rho,
      kernel_mean_abs_change=float(np.mean(np.abs(kb-ka)))
   ))

  if not np.isclose(p,ref_p):
   del Dp

 return pd.DataFrame(rows)

def scatter(df,x,y,path,title,xlab,ylab,dpi):
 fig,ax=plt.subplots(figsize=(7.5,5.5))
 for n,g in df.groupby("dataset",sort=False):
  g=g.sort_values("p");ax.plot(g[x],g[y],marker="o",lw=1.3,ms=5,label=n)
 ax.set(xlabel=xlab,ylabel=ylab,title=title);ax.grid(alpha=.2);ax.legend(frameon=False,ncol=2)
 fig.tight_layout();fig.savefig(path.with_suffix(".png"),dpi=dpi);fig.savefig(path.with_suffix(".pdf"));plt.close(fig)

def main():
 a=args();a.output_dir.mkdir(parents=True,exist_ok=True)
 specs=json.loads(a.config.read_text())["datasets"]; GG=[];FF=[];DD=[];KK=[];JJ=[]
 for sp in specs:
  name=sp["name"];print("[DATASET]",name)
  summ=pd.read_csv(Path(sp["summary"]).expanduser());oof=pd.read_csv(Path(sp["oof"]).expanduser())
  ps=summ.p.astype(float).tolist();nsub=int(oof.subject_idx.nunique());Dr=load_distance(sp,a.reference_p,nsub);vr=upper(Dr)
  mr=np.median(vr);ar=np.mean(vr);gr=[]
  for p in ps:
   D=load_distance(sp,p,nsub);v=upper(D);rho=spearmanr(vr,v).statistic
   gr.append(dict(dataset=name,p=p,n_subjects=D.shape[0],n_pairs=len(v),median_distance=np.median(v),
    mean_distance=np.mean(v),median_scale_ratio_vs_p1=np.median(v)/mr,mean_scale_ratio_vs_p1=np.mean(v)/ar,
    log_median_scale_ratio_vs_p1=np.log(np.median(v)/mr),distance_spearman_vs_p1=rho,relational_loss=1-rho))
  geom=pd.DataFrame(gr);GG.append(geom)
  fr=[]
  ref=oof[np.isclose(oof.p,a.reference_p)]
  for rep,g in ref.groupby("repeat"):
   z=robust_z(g.score.to_numpy());y=g.y.to_numpy(int);gaps=np.abs(z[:,None]-z[None,:]);ii,jj=np.triu_indices(len(z),1);gv=gaps[ii,jj]
   fr.append(dict(dataset=name,repeat=rep,n_subjects=len(z),score_sd_raw=np.std(g.score,ddof=1),
    score_iqr_raw=np.subtract(*np.percentile(g.score,[75,25])),
    class_separation_z=abs(np.mean(z[y==1])-np.mean(z[y==0])),
    near_boundary_fraction_z=np.mean(np.abs(z)<a.boundary_z),
    pairwise_gap_q=a.small_gap_quantile,pairwise_gap_q_value_z=np.quantile(gv,a.small_gap_quantile),
    pairwise_gap_median_z=np.median(gv),inverse_small_gap_z=1/max(np.quantile(gv,a.small_gap_quantile),1e-12)))
  frag=pd.DataFrame(fr)
  if sp.get("repeat_summary"):
   rr=pd.read_csv(Path(sp["repeat_summary"]).expanduser());r1=rr[np.isclose(rr.p,a.reference_p)][["repeat","test_oof_auc_repeat"]].rename(columns={"test_oof_auc_repeat":"auc_p1"})
   frag=frag.merge(r1,on="repeat",how="left")
  FF.append(frag)
  dr=[]
  for rep,g in oof.groupby("repeat"):
   r=g[np.isclose(g.p,a.reference_p)].sort_values("subject_idx")
   for p in sorted(g.p.unique()):
    c=g[np.isclose(g.p,p)].sort_values("subject_idx");rho=spearmanr(r.score,c.score).statistic
    dr.append(dict(dataset=name,repeat=rep,p=p,decision_concordance_repeat=rho,decision_loss_repeat=1-rho))
  dec=pd.DataFrame(dr);DD.append(dec)
  kern=kernel_fold_diagnostics(name,sp,oof,a.reference_p);KK.append(kern)
  ks=(kern.groupby(["dataset","p"]).agg(
      kernel_relational_loss=("kernel_relational_loss","mean"),
      kernel_relational_loss_sem=("kernel_relational_loss",lambda s:s.std(ddof=1)/np.sqrt(len(s))),
      kernel_mean_abs_change=("kernel_mean_abs_change","mean")).reset_index())
  ds=dec.groupby(["dataset","p"]).decision_loss_repeat.agg(decision_loss="mean",decision_loss_sem=lambda s:s.std(ddof=1)/np.sqrt(len(s))).reset_index()
  xp=Path(sp["xai"]).expanduser() if sp.get("xai") else None
  if xp and xp.exists():
   x=pd.read_csv(xp);xx=x.groupby("p").spearman_vs_p1.median().reset_index(name="xai_concordance");xx["xai_loss"]=1-xx.xai_concordance
  else: xx=summ[["p","xai_concordance","xai_loss"]]
  join=geom.merge(ds,on=["dataset","p"]).merge(ks,on=["dataset","p"],how="left").merge(xx,on="p",how="left")
  for c in ["auc_heldout","gamma_star","Q_typical","leading_mode_share"]:
   if c in summ: join=join.merge(summ[["p",c]],on="p",how="left")
  fs=frag.drop(columns=["repeat"]).groupby("dataset").median(numeric_only=True).reset_index()
  join=join.merge(fs,on="dataset",how="left");JJ.append(join)
 G=pd.concat(GG);F=pd.concat(FF);D=pd.concat(DD);K=pd.concat(KK);J=pd.concat(JJ)
 G.to_csv(a.output_dir/"01_geometry_deformation.csv",index=False);F.to_csv(a.output_dir/"02_decision_fragility.csv",index=False)
 J.to_csv(a.output_dir/"03_propagation_joined.csv",index=False);D.to_csv(a.output_dir/"03b_decision_repeat_loss.csv",index=False);K.to_csv(a.output_dir/"04_kernel_deformation.csv",index=False)
 scatter(J,"log_median_scale_ratio_vs_p1","relational_loss",a.output_dir/"04_scale_vs_relational","Scale vs relational deformation","log median distance ratio vs p=1","Relational loss",a.dpi)
 scatter(J,"relational_loss","decision_loss",a.output_dir/"05_relational_vs_decision","Relational reorganization vs decision drift","Relational loss","Decision loss",a.dpi)
 scatter(J,"relational_loss","xai_loss",a.output_dir/"06_relational_vs_xai","Relational reorganization vs explanation drift","Relational loss","XAI loss",a.dpi)
 scatter(J,"inverse_small_gap_z","decision_loss",a.output_dir/"07_fragility_vs_decision","Baseline rank fragility vs decision drift","Inverse 10th-percentile pairwise score gap","Decision loss",a.dpi)
 scatter(J,"relational_loss","kernel_relational_loss",a.output_dir/"08_relational_vs_kernel","Distance relational loss vs kernel relational loss","Distance relational loss","Kernel relational loss",a.dpi)
 scatter(J,"kernel_relational_loss","decision_loss",a.output_dir/"09_kernel_vs_decision","Kernel relational loss vs decision drift","Kernel relational loss","Decision loss",a.dpi)
 (a.output_dir/"mechanism_summary.txt").write_text("POST-HOC MECHANISM ANALYSIS\nPrespecified: scale vs relational deformation; baseline task/rank fragility.\nNo diagnostic is selected by downstream association.\n")
 print("[DONE]",a.output_dir)
if __name__=="__main__":main()
