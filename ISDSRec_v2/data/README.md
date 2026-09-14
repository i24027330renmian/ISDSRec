# Data

## Amazon Product Data (2014)

The manuscript uses the Beauty, Sports and Outdoors, and Toys and Games categories from the 2014 Amazon Product Data collection. Place the downloaded review and metadata files anywhere locally and pass their paths to `preprocess_amazon.py`.

source-file names:

- `reviews_Beauty.json.gz`, `meta_Beauty.json.gz`
- `reviews_Sports_and_Outdoors.json.gz`, `meta_Sports_and_Outdoors.json.gz`
- `reviews_Toys_and_Games.json.gz`, `meta_Toys_and_Games.json.gz`

## MovieLens-1M

The manuscript treats all ratings as implicit interactions and uses `movies.dat` titles and genres as title/category metadata. Descriptions are unavailable and are represented by the shared missing-field placeholder during semantic encoding.

files:

- `ratings.dat`
- `movies.dat`

## Preprocessing order

1. Parse source files while retaining source-file order.
2. Remove exact duplicate `(user, item, timestamp)` records, keeping the first occurrence.
3. Apply iterative 5-core filtering until every retained user and item has at least five interactions.
4. Stable chronological sort within user; source order breaks timestamp ties.
5. Assign last interaction to test, penultimate to validation, all earlier interactions to train.
6. Build all valid next-item prefixes from the training partition.
7. Compute local-gap/recency variables from the complete observable prefix before truncating to the most recent 50 interactions.
8. Estimate normalization statistics from training instances only.
