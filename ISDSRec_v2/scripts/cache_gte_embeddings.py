#!/usr/bin/env python3
import argparse, json
from pathlib import Path
import numpy as np, pandas as pd, torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

@torch.no_grad()
def last_token_pool(last_hidden, attention_mask):
    lengths=attention_mask.sum(dim=1)-1
    return last_hidden[torch.arange(last_hidden.size(0),device=last_hidden.device),lengths]

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--items',required=True); ap.add_argument('--output',required=True)
    ap.add_argument('--model',default='Alibaba-NLP/gte-Qwen2-7B-instruct'); ap.add_argument('--batch-size',type=int,default=8)
    ap.add_argument('--max-length',type=int,default=512); ap.add_argument('--placeholder',default='[MISSING]')
    ap.add_argument('--title-marker',default='Title:'); ap.add_argument('--category-marker',default='Category:'); ap.add_argument('--description-marker',default='Description:')
    args=ap.parse_args(); device='cuda' if torch.cuda.is_available() else 'cpu'
    df=pd.read_csv(args.items).fillna('').sort_values('item_id'); n=int(df.item_id.max())+1
    tok=AutoTokenizer.from_pretrained(args.model,trust_remote_code=True); model=AutoModel.from_pretrained(args.model,trust_remote_code=True).to(device).eval()
    texts=[]
    for _,r in df.iterrows():
        title=r.title or args.placeholder; cat=r.category or args.placeholder; desc=r.description or args.placeholder
        texts.append(f"{args.title_marker} {title}\n{args.category_marker} {cat}\n{args.description_marker} {desc}")
    arr=None
    for i in tqdm(range(0,len(texts),args.batch_size)):
        batch=tok(texts[i:i+args.batch_size],padding=True,truncation=True,max_length=args.max_length,return_tensors='pt').to(device)
        out=model(**batch); emb=last_token_pool(out.last_hidden_state,batch['attention_mask']); emb=torch.nn.functional.normalize(emb,p=2,dim=-1).cpu().numpy().astype('float32')
        if arr is None: arr=np.zeros((n,emb.shape[1]),dtype='float32')
        ids=df.iloc[i:i+len(emb)].item_id.to_numpy(int); arr[ids]=emb
    Path(args.output).parent.mkdir(parents=True,exist_ok=True); np.save(args.output,arr)
    Path(str(args.output)+'.meta.json').write_text(json.dumps({'model':args.model,'max_length':args.max_length,'pooling':'last_valid_token','l2_normalized':True,'shape':list(arr.shape)},indent=2))

if __name__=='__main__': main()
