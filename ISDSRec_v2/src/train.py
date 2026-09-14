#!/usr/bin/env python3
import argparse,json,random,sys
from pathlib import Path
import numpy as np,torch,yaml
from torch.utils.data import Dataset,DataLoader
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.model.isdsrec import ISDSRec
from src.model.temporal_encoding import build_temporal_tensors

def set_seed(s): random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)
def load_jsonl(p): return [json.loads(x) for x in open(p,encoding='utf-8') if x.strip()]
def parse_value(s):
    try: return json.loads(s)
    except Exception: return s
def apply_set(cfg,sets):
    for spec in sets or []:
        key,val=spec.split('=',1); cur=cfg
        parts=key.split('.')
        for p in parts[:-1]: cur=cur.setdefault(p,{})
        cur[parts[-1]]=parse_value(val)
    return cfg
class PrefixDataset(Dataset):
    def __init__(self,rows,num_items,nneg=100,seed=42): self.rows=rows; self.num_items=num_items; self.nneg=nneg; self.rng=np.random.default_rng(seed)
    def __len__(self): return len(self.rows)
    def __getitem__(self,i):
        x=self.rows[i]; forbidden=set(x['history_items'])|{x['target_item']}; pool=np.array([j for j in range(self.num_items) if j not in forbidden],dtype=np.int64)
        return x,[x['target_item']]+self.rng.choice(pool,size=self.nneg,replace=False).tolist()
def collate(samples,L,stats,device):
    xs,cands=zip(*samples); ids=torch.zeros(len(xs),L,dtype=torch.long,device=device); times=[]
    for b,x in enumerate(xs):
        h=x['history_items'][-L:]; ids[b,L-len(h):]=torch.tensor(h,device=device); times.append(x['history_timestamps'])
    return {'history_items':ids,'temporal':build_temporal_tensors(times,L,stats,device),'candidate_items':torch.tensor(cands,dtype=torch.long,device=device)}
@torch.no_grad()
def val_ndcg(model,rows,stats,cfg,device):
    model.eval(); allz=model.all_item_representations(); vals=[]; L=cfg['model']['max_seq_len']; draws=cfg.get('evaluation',{}).get('noise_realizations',20) if cfg['model'].get('t_inf',0)>0 else 1
    for x in rows:
        ids=torch.zeros(1,L,dtype=torch.long,device=device); h=x['history_items'][-L:]; ids[0,L-len(h):]=torch.tensor(h,device=device); b={'history_items':ids,'temporal':build_temporal_tensors([x['history_timestamps']],L,stats,device)}; per=[]
        for _ in range(draws):
            u,_=model.user_representation(b); scores=(allz@u[0]).cpu().numpy();
            for seen in x['history_items']:
                if seen!=x['target_item']: scores[seen]=-np.inf
            idx=np.argpartition(-scores,min(10,len(scores)-1))[:10]; idx=idx[np.argsort(-scores[idx])]
            per.append(1/np.log2(int(np.where(idx==x['target_item'])[0][0])+2) if x['target_item'] in idx else 0.)
        vals.append(float(np.mean(per)))
    model.train(); return float(np.mean(vals))
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--config',required=True); ap.add_argument('--seed',type=int); ap.add_argument('--ablation'); ap.add_argument('--output-dir'); ap.add_argument('--set',action='append',default=[]); args=ap.parse_args()
    cfg=apply_set(yaml.safe_load(open(args.config)),args.set); seed=args.seed if args.seed is not None else cfg['training'].get('seed',42); set_seed(seed); device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    proc=Path(cfg['dataset']['processed_dir']); stats=json.load(open(proc/'normalization_stats.json')); tr=load_jsonl(proc/'train_instances.jsonl'); va=load_jsonl(proc/'validation_instances.jsonl'); sem=torch.from_numpy(np.load(cfg['dataset']['semantic_cache'])).to(device)
    ab=yaml.safe_load(open(args.ablation)) if args.ablation else {}; model=ISDSRec(len(sem),sem,stats,cfg,ab).to(device); opt=torch.optim.Adam(model.parameters(),lr=cfg['training']['learning_rate'],weight_decay=cfg['training']['weight_decay'])
    ds=PrefixDataset(tr,len(sem),cfg['training'].get('negative_samples',100),seed); loader=DataLoader(ds,batch_size=cfg['training']['batch_size'],shuffle=True,num_workers=0,collate_fn=lambda s:collate(s,cfg['model']['max_seq_len'],stats,device))
    out=Path(args.output_dir or f"outputs/{cfg['dataset']['name']}/seed_{seed}"); out.mkdir(parents=True,exist_ok=True); best=-1.; bad=0
    for epoch in range(1,cfg['training']['max_epochs']+1):
        model.train(); total=0.; n=0
        for batch in loader:
            opt.zero_grad(set_to_none=True); res=model(batch); res['loss'].backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),cfg['training'].get('gradient_clip',5.)); opt.step(); total+=res['loss'].item(); n+=1
        val=val_ndcg(model,va,stats,cfg,device); print(json.dumps({'epoch':epoch,'train_loss':total/max(n,1),'validation_ndcg10':val}))
        if val>best: best=val; bad=0; torch.save({'model':model.state_dict(),'config':cfg,'seed':seed,'best_validation_ndcg10':best},out/'best.pt')
        else:
            bad+=1
            if bad>=cfg['training'].get('patience',10): break
    (out/'summary.json').write_text(json.dumps({'seed':seed,'best_validation_ndcg10':best},indent=2))
if __name__=='__main__': main()
