#!/usr/bin/env python3
import argparse, json
from pathlib import Path
import numpy as np, pandas as pd

def paired_bootstrap(a,b,n_resamples=5000,seed=42):
    a=np.asarray(a,float); b=np.asarray(b,float)
    if a.shape!=b.shape: raise ValueError('paired arrays must have same shape')
    d=a-b; effect=float(d.mean()); rng=np.random.default_rng(seed); n=len(d)
    idx=rng.integers(0,n,size=(n_resamples,n)); means=d[idx].mean(axis=1); se=float(means.std(ddof=1))
    centered=d-effect; null_means=centered[idx].mean(axis=1)
    p=(np.count_nonzero(np.abs(null_means)>=abs(effect))+1)/(n_resamples+1)
    return {'mean_difference':effect,'bootstrap_se':se,'p_two_sided':float(p),'n_users':n,'n_resamples':n_resamples}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--isdsrec',required=True); ap.add_argument('--comparator',required=True); ap.add_argument('--metric',default='ndcg'); ap.add_argument('--resamples',type=int,default=5000); ap.add_argument('--seed',type=int,default=42); args=ap.parse_args()
    def load(p):
        rows=[json.loads(x) for x in open(p) if x.strip()]; return {int(r['user_id']):float(r[args.metric]) for r in rows}
    A=load(args.isdsrec); B=load(args.comparator); users=sorted(set(A)&set(B)); res=paired_bootstrap([A[u] for u in users],[B[u] for u in users],args.resamples,args.seed); print(json.dumps(res,indent=2))
if __name__=='__main__': main()
