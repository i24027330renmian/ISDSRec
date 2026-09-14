#!/usr/bin/env python3
import argparse,json,sys,time
from pathlib import Path
import numpy as np,torch,yaml
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.model.isdsrec import ISDSRec
from src.model.temporal_encoding import build_temporal_tensors
from src.metrics import hr_ndcg_at_k

def load_jsonl(p): return [json.loads(x) for x in open(p,encoding='utf-8') if x.strip()]
def parse_value(s):
    try: return json.loads(s)
    except Exception: return s
def apply_set(cfg,sets):
    for spec in sets or []:
        key,val=spec.split('=',1); cur=cfg; parts=key.split('.')
        for p in parts[:-1]: cur=cur.setdefault(p,{})
        cur[parts[-1]]=parse_value(val)
    return cfg
def batch_one(x,L,stats,device):
    ids=torch.zeros(1,L,dtype=torch.long,device=device); h=x['history_items'][-L:]; ids[0,L-len(h):]=torch.tensor(h,device=device); return {'history_items':ids,'temporal':build_temporal_tensors([x['history_timestamps']],L,stats,device)}
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--config',required=True); ap.add_argument('--checkpoint',required=True); ap.add_argument('--split',choices=['validation','test'],default='test'); ap.add_argument('--ablation'); ap.add_argument('--output'); ap.add_argument('--noise-realizations',type=int); ap.add_argument('--set',action='append',default=[]); ap.add_argument('--benchmark',action='store_true'); args=ap.parse_args()
    cfg=apply_set(yaml.safe_load(open(args.config)),args.set); device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); proc=Path(cfg['dataset']['processed_dir']); stats=json.load(open(proc/'normalization_stats.json')); rows=load_jsonl(proc/f"{args.split}_instances.jsonl"); sem=torch.from_numpy(np.load(cfg['dataset']['semantic_cache'])).to(device); ab=yaml.safe_load(open(args.ablation)) if args.ablation else {}
    model=ISDSRec(len(sem),sem,stats,cfg,ab).to(device); ck=torch.load(args.checkpoint,map_location=device); model.load_state_dict(ck['model'],strict=False); model.eval(); allz=model.all_item_representations(); L=cfg['model']['max_seq_len']; t=cfg['model'].get('t_inf',40); draws=args.noise_realizations or (cfg.get('evaluation',{}).get('noise_realizations',20) if t>0 else 1)
    if args.benchmark:
        sample=rows[:min(256,len(rows))];
        def bench(tinf,repeats=20):
            if device.type=='cuda': torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize()
            start=time.perf_counter(); count=0
            with torch.no_grad():
                for _ in range(repeats):
                    for x in sample:
                        b=batch_one(x,L,stats,device); u,_=model.user_representation(b,t_inf=tinf); _=allz@u[0]; count+=1
            if device.type=='cuda': torch.cuda.synchronize()
            sec=time.perf_counter()-start; mem=torch.cuda.max_memory_allocated()/1024**3 if device.type=='cuda' else 0.
            return {'latency_ms':1000*sec/count,'throughput_req_s':count/sec,'peak_memory_GB':mem}
        print(json.dumps({'clean':bench(0),'one_pass':bench(t)},indent=2)); return
    output=[]
    with torch.no_grad():
        for x in rows:
            b=batch_one(x,L,stats,device); hrs=[]; nd=[]
            for _ in range(draws):
                u,_=model.user_representation(b,t_inf=t); scores=(allz@u[0]).cpu().numpy();
                for seen in x['history_items']:
                    if seen!=x['target_item']: scores[seen]=-np.inf
                idx=np.argpartition(-scores,min(10,len(scores)-1))[:10]; idx=idx[np.argsort(-scores[idx])]; a,c=hr_ndcg_at_k(idx.tolist(),x['target_item'],10); hrs.append(a); nd.append(c)
            output.append({'user_id':x['user_id'],'target_item':x['target_item'],'target_timestamp':x['target_timestamp'],'history_length':len(x['history_items']),'last_history_timestamp':x['history_timestamps'][-1], 'hr':float(np.mean(hrs)),'ndcg':float(np.mean(nd)),'hr_noise_sd':float(np.std(hrs,ddof=1)) if len(hrs)>1 else 0.,'ndcg_noise_sd':float(np.std(nd,ddof=1)) if len(nd)>1 else 0.})
    summary={'HR@10':float(np.mean([r['hr'] for r in output])),'NDCG@10':float(np.mean([r['ndcg'] for r in output])),'n_users':len(output),'noise_realizations':draws}; print(json.dumps(summary,indent=2))
    if args.output: Path(args.output).parent.mkdir(parents=True,exist_ok=True); Path(args.output).write_text('\n'.join(json.dumps(r) for r in output)+'\n',encoding='utf-8')
if __name__=='__main__': main()
