#!/usr/bin/env python3
import argparse, json
from pathlib import Path
import pandas as pd


def dump_jsonl(path, rows):
    with open(path, 'w', encoding='utf-8') as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')


def pack(uid, history, target):
    return {
        'user_id': int(uid),
        'history_items': [int(x) for x in history.item_id.tolist()],
        'history_timestamps': [int(x) for x in history.timestamp.tolist()],
        'target_item': int(target.item_id),
        'target_timestamp': int(target.timestamp),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--split-file', required=True)
    ap.add_argument('--out-dir', required=True)
    args = ap.parse_args()
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.split_file)
    df = df.sort_values(['user_id','timestamp','source_order'], kind='mergesort')

    train_rows, val_rows, test_rows = [], [], []
    for uid, g in df.groupby('user_id', sort=False):
        tr = g[g.split=='train'].reset_index(drop=True)
        va = g[g.split=='validation'].reset_index(drop=True)
        te = g[g.split=='test'].reset_index(drop=True)
        for i in range(1, len(tr)):
            train_rows.append(pack(uid, tr.iloc[:i], tr.iloc[i]))
        if len(va) == 1:
            val_rows.append(pack(uid, tr, va.iloc[0]))
        if len(te) == 1:
            hist = pd.concat([tr, va], ignore_index=True)
            test_rows.append(pack(uid, hist, te.iloc[0]))

    dump_jsonl(out/'train_instances.jsonl', train_rows)
    dump_jsonl(out/'validation_instances.jsonl', val_rows)
    dump_jsonl(out/'test_instances.jsonl', test_rows)
    print({'train_instances':len(train_rows),'validation_instances':len(val_rows),'test_instances':len(test_rows)})

if __name__ == '__main__': main()
