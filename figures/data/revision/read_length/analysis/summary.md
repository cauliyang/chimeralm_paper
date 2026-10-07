# Read-length / truncation analysis

## PromethION (p2) WGA chimeric reads

- chimeric reads in BAM: 12,963,576; predictions: 12,963,576; matched: 12,963,576
- read length: min 170, median 2,100, mean 2,869, N50 4,053, max 126,974
- reads > 32,768 bp: 1,211 (0.009%)
  - of which ALL junctions lie beyond 32,768 bp (model window junction-free): 2 (0.2% of >32 k; 0.0000% of all chimeric reads)
  - of which SOME but not all junctions lie beyond the window: 841 (69.4% of >32 k)

| length bin | n | % called artifact | mean #segments |
|---|---|---|---|
| ≤2 k | 6,201,521 | 95.4 | 2.10 |
| 2–8 k | 6,206,445 | 94.6 | 2.38 |
| 8–16 k | 521,707 | 76.6 | 3.19 |
| 16–32 k | 32,692 | 33.0 | 5.38 |
| >32 k | 1,211 | 3.1 | 19.99 |

Retained (pred 0): n 769,743, median 3,087, mean 4,727, N50 7,989, >32 k 0.153%
Removed  (pred 1): n 12,193,833, median 2,067, mean 2,752, N50 3,836, >32 k 0.000%

## MinION (mk1c) WGA chimeric reads

- chimeric reads in BAM: 1,666,427; predictions: 1,666,427; matched: 1,666,427
- read length: min 182, median 1,469, mean 1,901, N50 2,333, max 42,118
- reads > 32,768 bp: 6 (0.000%)
  - of which ALL junctions lie beyond 32,768 bp (model window junction-free): 0 (0.0% of >32 k; 0.0000% of all chimeric reads)
  - of which SOME but not all junctions lie beyond the window: 0 (0.0% of >32 k)

| length bin | n | % called artifact | mean #segments |
|---|---|---|---|
| ≤2 k | 1,134,711 | 96.4 | 2.08 |
| 2–8 k | 515,485 | 93.3 | 2.31 |
| 8–16 k | 15,360 | 57.2 | 2.87 |
| 16–32 k | 865 | 14.8 | 3.34 |
| >32 k | 6 | 0.0 | 3.50 |

Retained (pred 0): n 82,734, median 2,031, mean 3,238, N50 5,560, >32 k 0.007%
Removed  (pred 1): n 1,583,693, median 1,458, mean 1,832, N50 2,220, >32 k 0.000%

## Held-out test split (76,512 reads), predictions from PromethION WGA run

- covered by WGA predictions: 58,636 (76.6%); label 1 (artifact) among covered: 29,318
- uncovered (bulk-sampled genuine reads, not in WGA BAM): 17,876, of which label 0: 17,876
- seq_len (parquet) vs read_len (BAM) mismatches: 0

| bin | n | n artifact | precision | recall | F1 |
|---|---|---|---|---|---|
| all | 58,636 | 29,318 | 0.775 | 0.956 | 0.856 |
| ≤2 k | 33,611 | 14,067 | 0.694 | 0.976 | 0.811 |
| 2–8 k | 23,561 | 13,952 | 0.866 | 0.956 | 0.909 |
| 8–16 k | 1,377 | 1,217 | 0.990 | 0.757 | 0.858 |
| 16–32 k | 85 | 80 | 1.000 | 0.312 | 0.476 |
| >32 k | 2 | 2 | nan | 0.000 | nan |

- test reads > 32 k: 2; junction-free window: 0
