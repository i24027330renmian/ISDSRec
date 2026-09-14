#!/usr/bin/env python3
import argparse, ast, gzip, json
from pathlib import Path
import pandas as pd


def read_json_lines(path):
    opener = gzip.open if str(path).endswith('.gz') else open
    with opener(path, 'rt', encoding='utf-8', errors='replace') as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                obj = ast.literal_eval(line)
            yield i, obj


def first_present(d, keys, default=''):
    for k in keys:
        if k in d and d[k] not in (None, ''):
            return d[k]
    return default


def normalize_category(value):
    if isinstance(value, list):
        flat = []
        def walk(x):
            if isinstance(x, list):
                for y in x: walk(y)
            elif x not in (None, ''):
                flat.append(str(x))
        walk(value)
        return ' > '.join(flat)
    return '' if value is None else str(value)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--reviews', required=True)
    ap.add_argument('--metadata', required=True)
    ap.add_argument('--out-dir', required=True)
    args = ap.parse_args()
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

    rows = []
    for source_order, r in read_json_lines(args.reviews):
        user = first_present(r, ['reviewerID', 'user_id', 'user'])
        item = first_present(r, ['asin', 'item_id', 'item'])
        ts = first_present(r, ['unixReviewTime', 'timestamp', 'time'], None)
        if user == '' or item == '' or ts is None:
            continue
        rows.append((str(user), str(item), int(ts), source_order))
    pd.DataFrame(rows, columns=['user_raw','item_raw','timestamp','source_order']).to_csv(
        out/'interactions_raw.csv', index=False)

    meta = []
    seen = set()
    for _, m in read_json_lines(args.metadata):
        item = first_present(m, ['asin', 'item_id', 'item'])
        if item == '' or str(item) in seen:
            continue
        seen.add(str(item))
        title = first_present(m, ['title', 'name'])
        category = normalize_category(first_present(m, ['categories', 'category'], ''))
        desc = first_present(m, ['description', 'desc'], '')
        if isinstance(desc, list): desc = ' '.join(map(str, desc))
        meta.append((str(item), str(title), str(category), str(desc)))
    pd.DataFrame(meta, columns=['item_raw','title','category','description']).to_csv(
        out/'items_raw.csv', index=False)

    print(f'Wrote {len(rows):,} interactions and {len(meta):,} metadata rows to {out}')

if __name__ == '__main__': main()
