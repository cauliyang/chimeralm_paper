"""External validation on HCC78 (PRJNA875576, Ni et al. 2023) with the frozen ChimeraLM model.

(a) chimeric vs non-chimeric mapped primary reads for bulk, WGA and WGA + ChimeraLM (Fig. 2b style);
(b) precision / recall / F1 against bulk-support labels (positive class = artifact).
Counts/metrics: figures/data/revision/hcc78/metrics.tsv + RUNLOG (samtools flagstat, dbp count-chimeric,
chimeralm predict, annotate --ovr-threshold 1000).
Writes figures/final_figures/sf_hcc78.pdf and single panels sf_hcc78_{a,b}.pdf.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures" / "final_figures"
T1, T2 = "#00a087", "#9fc8c8"
C_P, C_R, C_F = "#4c72b0", "#dd8452", "#55a868"

# (label, mapped primary reads, chimeric reads) -- see RUNLOG 2026-10-07
BARS = [
    ("Bulk\nR10.4", 3_114_766, 654_895),
    ("Bulk\nR9.4.1", 4_555_737, 359_374),
    ("MDA\nsingle cell", 4_552_015, 3_033_649),
    ("MDA\n+ ChimeraLM", 4_552_015 - 2_478_587, 3_033_649 - 2_478_587),
    ("MALBAC\nsingle cell", 394_385, 26_916),
    ("MALBAC\n+ ChimeraLM", 394_385 - 23_488, 26_916 - 23_488),
]
METRICS = pd.read_csv(ROOT / "figures/data/revision/hcc78/metrics.tsv", sep="\t")
SAMPLE_LABEL = {"MDA_R10.4": "MDA single cell\n(R10.4)", "MALBAC": "MALBAC single cell\n(R9.4.1)"}


def panel_bars(ax):
    x = range(len(BARS))
    non = [t - c for _, t, c in BARS]
    chim = [c for _, _, c in BARS]
    ax.bar(x, non, color=T2, edgecolor="k", linewidth=0.6, label="Non-chimeric reads")
    ax.bar(x, chim, bottom=non, color=T1, edgecolor="k", linewidth=0.6, label="Chimeric reads")
    for i, (_, t, c) in enumerate(BARS):
        ax.text(i, t * 1.02, f"{c / t * 100:.1f}%", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(list(x), [b[0] for b in BARS], fontsize=8.5)
    ax.set_ylabel("Mapped reads")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v/1e6:.0f}M" if v >= 1e6 else f"{v/1e3:.0f}k"))
    ax.set_ylim(0, max(t for _, t, _ in BARS) * 1.15)
    ax.legend(frameon=False, fontsize=8.5, loc="upper right")
    ax.spines[["top", "right"]].set_visible(False)


def panel_metrics(ax):
    m = METRICS.set_index("sample").loc[list(SAMPLE_LABEL)]
    x = np.arange(len(m)); w = 0.26
    for j, (col, c, lab) in enumerate([("precision", C_P, "Precision"), ("recall", C_R, "Recall"), ("f1", C_F, "F1")]):
        bars = ax.bar(x + (j - 1) * w, m[col], w, color=c, edgecolor="k", linewidth=0.6, label=lab)
        for b, v in zip(bars, m[col], strict=True):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.2f}", ha="center", va="bottom", fontsize=8)
    for xi, (_, r) in zip(x, m.iterrows(), strict=True):
        ax.text(xi, 1.08, f"n = {int(r['n_labelled']):,}\nartifact {r['n_label_artifact']/r['n_labelled']*100:.1f}%", ha="center", fontsize=7.5)
    ax.set_xticks(x, [SAMPLE_LABEL[s] for s in m.index], fontsize=8.5)
    ax.set_ylim(0, 1.25)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_ylabel("Score (artifact = positive class)")
    ax.legend(frameon=False, fontsize=8.5, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.22))
    ax.spines[["top", "right"]].set_visible(False)


fig, (a, b) = plt.subplots(1, 2, figsize=(11.5, 4.2), gridspec_kw={"width_ratios": [1.6, 1]})
panel_bars(a); a.set_title("a  HCC78: chimeric read fraction", loc="left", fontweight="bold", fontsize=10)
panel_metrics(b); b.set_title("b  HCC78: classification vs bulk-support labels", loc="left", fontweight="bold", fontsize=10)
fig.tight_layout()
fig.savefig(OUT / "sf_hcc78.pdf"); fig.savefig(OUT / "sf_hcc78.png", dpi=170); plt.close(fig)
for letter, fn, size in (("a", panel_bars, (7.2, 4.0)), ("b", panel_metrics, (4.6, 4.0))):
    fig, ax = plt.subplots(figsize=size); fn(ax); fig.tight_layout(); fig.savefig(OUT / f"sf_hcc78_{letter}.pdf"); plt.close(fig)
print("wrote", OUT / "sf_hcc78.pdf", "+ sf_hcc78_{a,b}.pdf")
