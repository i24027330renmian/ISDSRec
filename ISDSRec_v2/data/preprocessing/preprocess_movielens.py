#!/usr/bin/env python3
import argparse
from pathlib import Path
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ratings', required=True)
    ap.add_argument('--movies', required=True)
    ap.add_argument('--out-dir', required=True)
    args = ap.parse_args()
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

    ratings = pd.read_csv(args.ratings, sep='::', engine='python', header=None,
                          names=['user_raw','item_raw','rating','timestamp'], encoding='latin-1')
    ratings['source_order'] = range(len(ratings))
    ratings[['user_raw','item_raw','timestamp','source_order']].to_csv(out/'interactions_raw.csv', index=False)

    movies = pd.read_csv(args.movies, sep='::', engine='python', header=None,
                         names=['item_raw','title','genres'], encoding='latin-1')
    movies['category'] = movies['genres'].fillna('').str.replace('|', ' > ', regex=False)
    movies['description'] = ''
    movies[['item_raw','title','category','description']].to_csv(out/'items_raw.csv', index=False)
    print(f'Wrote {len(ratings):,} interactions and {len(movies):,} item rows to {out}')

if __name__ == '__main__': main()
