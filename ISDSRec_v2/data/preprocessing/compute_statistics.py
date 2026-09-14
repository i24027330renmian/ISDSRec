#!/usr/bin/env python3
import argparse, json, math
from pathlib import Path
import numpy as np
import pandas as pd

DAY = 86400.0

def load_jsonl(path):
    with open(path, encoding='utf-8') as f:
        for line in f:
            if line.strip(): yield json.loads(line)

def local_gaps(ts):
    if len(ts) < 2: return np.array([], dtype=float)
    return np.maximum(0.0, np.diff(np.asarray(ts, dtype=float))/DAY)

def recencies(ts):
    if len(ts) == 0: return np.array([], dtype=float)
    a=np.asarray(ts,dtype=float); return np.maximum(0.0,(a[-1]-a)/DAY)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--train-instances',required=True); ap.add_argument('--split-file',required=True)
    ap.add_argument('--out-dir',required=True); ap.add_argument('--max-seq-len',type=int,default=50)
    ap.add_argument('--gamma',type=float,required=True,help='Dataset-selected recency decay for final-config behavior-stat normalization')
    args=ap.parse_args(); out=Path(args.out_dir); out.mkdir(parents=True,exist_ok=True)
    instances=list(load_jsonl(args.train_instances)); log_gaps=[]; log_rec=[]
    for x in instances:
        log_gaps.extend(np.log1p(local_gaps(x['history_timestamps'])).tolist())
        log_rec.extend(np.log1p(recencies(x['history_timestamps'])).tolist())
    gap_mu=float(np.mean(log_gaps)) if log_gaps else 0.; gap_std=float(np.std(log_gaps,ddof=0)) if log_gaps else 1.
    rec_std=float(np.std(log_rec,ddof=0)) if log_rec else 1.; rec_std=max(rec_std,1e-8)

    behavior=[]
    for x in instances:
        ts=np.asarray(x['history_timestamps'][-args.max_seq_len:],dtype=float); L=len(ts)
        span=0. if L<2 else max(0.,(ts[-1]-ts[0])/DAY); gaps=local_gaps(ts)
        mg=float(gaps.mean()) if len(gaps) else 0.; valid=1. if len(gaps) else 0.
        rec=recencies(ts); norm=np.log1p(rec)/rec_std; r=np.exp(-args.gamma*norm)
        rmean=float(r.mean()) if len(r) else 0.; rstd=float(np.sqrt(np.mean((r-rmean)**2))) if len(r) else 0.
        behavior.append([math.log1p(L),math.log1p(span),mg,valid,rmean,rstd])
    q=np.asarray(behavior,float) if behavior else np.zeros((1,6))
    cont=q[:,[0,1,2,4,5]]
    stats={
      'day_seconds':DAY,'gap_log_mean':gap_mu,'gap_log_std':max(gap_std,1e-8),'recency_log_std':rec_std,
      'behavior_continuous_mean':cont.mean(axis=0).tolist(),
      'behavior_continuous_std':np.maximum(cont.std(axis=0,ddof=0),1e-8).tolist(),
      'behavior_stats_gamma':args.gamma,'max_seq_len':args.max_seq_len,
    }
    (out/'normalization_stats.json').write_text(json.dumps(stats,indent=2),encoding='utf-8')
    split=pd.read_csv(args.split_file); train=split[split.split=='train']
    ds={'users':int(split.user_id.nunique()),'items':int(split.item_id.nunique()),'interactions':int(len(split)),
        'train_interactions':int(len(train)),'validation_interactions':int((split.split=='validation').sum()),'test_interactions':int((split.split=='test').sum())}
    (out/'dataset_stats.json').write_text(json.dumps(ds,indent=2),encoding='utf-8'); print(json.dumps({'normalization':stats,'dataset':ds},indent=2))
if __name__=='__main__': main()
