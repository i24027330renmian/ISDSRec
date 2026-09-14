#!/usr/bin/env python3
import argparse, json
import pandas as pd, numpy as np

def holm_adjust(pvalues):
    p=np.asarray(pvalues,float); m=len(p); order=np.argsort(p); adj=np.empty(m,float); running=0.0
    for rank,idx in enumerate(order):
        val=(m-rank)*p[idx]; running=max(running,val); adj[idx]=min(1.0,running)
    return adj

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--input',required=True); ap.add_argument('--p-column',default='p_two_sided'); ap.add_argument('--output',required=True); args=ap.parse_args()
    df=pd.read_csv(args.input); df['p_holm']=holm_adjust(df[args.p_column].to_numpy()); df.to_csv(args.output,index=False)
if __name__=='__main__': main()
