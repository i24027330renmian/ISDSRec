#!/usr/bin/env python3
import argparse, json
from pathlib import Path
import pandas as pd


def iterative_k_core(df, k=5):
    cur = df.copy()
    while True:
        uc = cur.groupby('user_raw').size()
        ic = cur.groupby('item_raw').size()
        nxt = cur[cur.user_raw.isin(uc[uc >= k].index) & cur.item_raw.isin(ic[ic >= k].index)]
        if len(nxt) == len(cur):
            return nxt.copy()
        cur = nxt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--interactions', required=True)
    ap.add_argument('--items', required=True)
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--k', type=int, default=5)
    args = ap.parse_args()
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

    raw = pd.read_csv(args.interactions, dtype={'user_raw':str,'item_raw':str})
    items = pd.read_csv(args.items, dtype={'item_raw':str}).fillna('')
    raw_n = len(raw)
    dedup = raw.drop_duplicates(['user_raw','item_raw','timestamp'], keep='first').copy()
    dedup_n = len(dedup)
    core = iterative_k_core(dedup, args.k)

    users = sorted(core.user_raw.unique().tolist())
    item_ids = sorted(core.item_raw.unique().tolist())
    user_map = {u:i for i,u in enumerate(users)}
    item_map = {v:i for i,v in enumerate(item_ids)}
    core['user_id'] = core.user_raw.map(user_map).astype(int)
    core['item_id'] = core.item_raw.map(item_map).astype(int)
    core = core.sort_values(['user_id','timestamp','source_order'], kind='mergesort').reset_index(drop=True)

    parts = []
    for uid, g in core.groupby('user_id', sort=False):
        g = g.copy()
        g['split'] = 'train'
        if len(g) < 3:
            raise RuntimeError('A retained user has fewer than 3 interactions after k-core.')
        g.loc[g.index[-2], 'split'] = 'validation'
        g.loc[g.index[-1], 'split'] = 'test'
        parts.append(g)
    split = pd.concat(parts).sort_values(['user_id','timestamp','source_order'], kind='mergesort')
    split.to_csv(out/'interactions_split.csv', index=False)

    mapped_items = items[items.item_raw.isin(item_ids)].copy()
    mapped_items['item_id'] = mapped_items.item_raw.map(item_map).astype(int)
    mapped_items = mapped_items.sort_values('item_id')
    mapped_items.to_csv(out/'items_mapped.csv', index=False)

    summary = {
        'raw_interactions': raw_n,
        'deduplicated_interactions': dedup_n,
        'after_5core_interactions': int(len(core)),
        'users': len(users), 'items': len(item_ids),
        'train_interactions': int((split.split=='train').sum()),
        'validation_interactions': int((split.split=='validation').sum()),
        'test_interactions': int((split.split=='test').sum()),
        'duplicates_removed': raw_n-dedup_n,
        'k_core': args.k,
    }
    (out/'id_maps.json').write_text(json.dumps({'user_raw_to_id':user_map,'item_raw_to_id':item_map}, indent=2), encoding='utf-8')
    (out/'preprocessing_counts.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2))

if __name__ == '__main__': main()
