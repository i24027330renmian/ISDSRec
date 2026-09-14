#!/usr/bin/env python3
import argparse, json, math
import numpy as np, pandas as pd

def aggregate_noise_sd(matrix):
    """matrix shape [checkpoints, noise_draws] of full-test-set scores."""
    x=np.asarray(matrix,float); variances=np.var(x,axis=1,ddof=1); return float(np.sqrt(np.mean(variances)))

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--input',required=True,help='CSV with checkpoint,draw,hr,ndcg'); args=ap.parse_args(); df=pd.read_csv(args.input)
    hr=[]; nd=[]
    for _,g in df.groupby('checkpoint'):
        hr.append(g.hr.to_numpy()); nd.append(g.ndcg.to_numpy())
    print(json.dumps({'HR@10_noise_induced_SD':aggregate_noise_sd(np.vstack(hr)),'NDCG@10_noise_induced_SD':aggregate_noise_sd(np.vstack(nd))},indent=2))
if __name__=='__main__': main()
