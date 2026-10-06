# Residual unsupported SV calls after ChimeraLM

## PromethION: unsupported n = 4,332; supported n = 4,490

- unsupported SV type: DEL 1,710 (39.5%), INS 1,510 (34.9%), INV 1,039 (24.0%), DUP 73 (1.7%)
- unsupported size: median 189 bp, <500 bp 70.2%
- unsupported SUPPORT: median 4, ≤5 reads 65.3%
- supported SV type: DEL 2,258 (50.3%), INS 2,221 (49.5%), DUP 10 (0.2%), INV 1 (0.0%)
- supported size: median 146 bp, <500 bp 87.5%
- supported SUPPORT: median 10, ≤5 reads 27.0%

| SUPPORT ≥ | unsupported kept | unsupported removed (%) | supported kept | supported removed (%) | unsupported:supported |
|---|---|---|---|---|---|
| 3 | 4,332 | 0.0 | 4,490 | 0.0 | 0.96 : 1 |
| 5 | 1,893 | 56.3 | 3,586 | 20.1 | 0.53 : 1 |
| 10 | 853 | 80.3 | 2,309 | 48.6 | 0.37 : 1 |
| 20 | 399 | 90.8 | 1,161 | 74.1 | 0.34 : 1 |

## MinION: unsupported n = 606; supported n = 1,450

- unsupported SV type: DEL 268 (44.2%), INS 228 (37.6%), INV 100 (16.5%), DUP 10 (1.7%)
- unsupported size: median 124 bp, <500 bp 81.5%
- unsupported SUPPORT: median 3, ≤5 reads 89.6%
- supported SV type: DEL 840 (57.9%), INS 609 (42.0%), DUP 1 (0.1%)
- supported size: median 114 bp, <500 bp 95.8%
- supported SUPPORT: median 4, ≤5 reads 81.2%

| SUPPORT ≥ | unsupported kept | unsupported removed (%) | supported kept | supported removed (%) | unsupported:supported |
|---|---|---|---|---|---|
| 3 | 606 | 0.0 | 1,450 | 0.0 | 0.42 : 1 |
| 5 | 115 | 81.0 | 473 | 67.4 | 0.24 : 1 |
| 10 | 15 | 97.5 | 36 | 97.5 | 0.42 : 1 |
| 20 | 3 | 99.5 | 3 | 99.8 | 1.00 : 1 |

