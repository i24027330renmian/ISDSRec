#!/usr/bin/env bash
set -euo pipefail
SEEDS=(42 123 2024 3407 5678)
DATASETS=(beauty sports toys movielens1m)
for ds in "${DATASETS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    python src/evaluate.py --config "configs/isdsrec/${ds}.yaml" --checkpoint "outputs/${ds}/seed_${seed}/best.pt" --split test --output "outputs/${ds}/seed_${seed}/test_per_user.jsonl"
  done
done
python - <<'PY2'
import json, math
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import t
TH={
 'beauty': {'gap':(14,62),'activity':(5,8),'pop':(7,15)},
 'sports': {'gap':(18,78),'activity':(5,8),'pop':(7,14)},
 'toys': {'gap':(16,68),'activity':(5,8),'pop':(7,15)},
 'movielens1m': {'gap':(3,14),'activity':(62,153),'pop':(43,188)},
}
def group(v,cuts,names):
 a,b=cuts
 return names[0] if v<=a else (names[1] if v<=b else names[2])
def ci(x):
 x=np.asarray(x,float); m=x.mean(); s=x.std(ddof=1); h=t.ppf(.975,len(x)-1)*s/np.sqrt(len(x)); return m,m-h,m+h
for ds in TH:
 split=pd.read_csv(f'data/processed/{ds}/interactions_split.csv'); tr=split[split.split=='train']; activity=tr.groupby('user_id').size().to_dict(); pop=tr.groupby('item_id').size().to_dict()
 runs=[]
 for seed in [42,123,2024,3407,5678]:
  p=Path(f'outputs/{ds}/seed_{seed}/test_per_user.jsonl')
  if not p.exists(): continue
  rows=[json.loads(x) for x in p.open() if x.strip()]
  for r in rows:
   r['seed']=seed; r['target_gap_days']=max(0,(r['target_timestamp']-r['last_history_timestamp'])/86400); r['activity']=activity[r['user_id']]; r['popularity']=pop.get(r['target_item'],0); runs.append(r)
 if not runs: continue
 df=pd.DataFrame(runs); specs=[('target_gap_days','gap',['Short','Medium','Long']),('activity','activity',['Sparse','Medium','Active']),('popularity','pop',['Tail','Mid-Popularity','Head'])]
 out=[]
 for col,key,names in specs:
  df['group']=df[col].map(lambda v:group(v,TH[ds][key],names))
  per=df.groupby(['seed','group']).ndcg.mean().reset_index()
  for g in names:
   vals=per[per.group==g].ndcg.to_numpy(); m,lo,hi=ci(vals); out.append({'dimension':key,'group':g,'mean':m,'ci_low':lo,'ci_high':hi,'runs':len(vals)})
 pd.DataFrame(out).to_csv(f'outputs/{ds}/stratified_ndcg10.csv',index=False)
PY2
