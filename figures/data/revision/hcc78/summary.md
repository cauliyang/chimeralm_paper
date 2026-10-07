# External evaluation (frozen ChimeraLM)

## MDA_R10.4
- chimeric reads predicted: 3,033,649; called artifact: 2,478,587 (81.7%)
- mapped primary reads: 4,552,015; chimeric: 3,033,649 (66.64%); after ChimeraLM: 555,062 chimeric of 2,073,428 reads (26.77%)
- labelled (bulk-matched) reads: 3,033,649; artifact (support 0): 3,026,631 (99.8%); genuine: 7,018
- precision 0.997  recall 0.817  F1 0.898  (TP 2,471,862 FP 6,725 FN 554,769 TN 293)

## MALBAC
- chimeric reads predicted: 26,916; called artifact: 23,488 (87.3%)
- mapped primary reads: 394,385; chimeric: 26,916 (6.82%); after ChimeraLM: 3,428 chimeric of 370,897 reads (0.92%)
- labelled (bulk-matched) reads: 26,916; artifact (support 0): 25,691 (95.4%); genuine: 1,225
- precision 0.955  recall 0.873  F1 0.912  (TP 22,427 FP 1,061 FN 3,264 TN 164)

