"""Fig. 2b-style stacked bars for the external HCC78 dataset (PRJNA875576): chimeric vs non-chimeric
mapped primary reads for bulk, WGA and WGA + ChimeraLM (frozen model). Counts from
revision/external/hcc78 (samtools flagstat, dbp count-chimeric, chimeralm predict); see RUNLOG.
Style: draw_figures.ipynb cell 33 (Fig. 2b) with the manuscript teal palette.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parents[1] / "figures" / "final_figures"
T1, T2 = "#00a087", "#9fc8c8"  # chimeric (dark teal), non-chimeric (light teal) as in Fig. 2b

# (label, mapped primary reads, chimeric reads)
BARS = [
    ("Bulk\nR10.4", 3_114_766, 654_895),
    ("Bulk\nR9.4.1", 4_555_737, 359_374),
    ("MDA\nsingle cell", 4_552_015, 3_033_649),
    ("MDA\n+ ChimeraLM", 4_552_015 - 2_478_587, 3_033_649 - 2_478_587),
    ("MALBAC\nsingle cell", 394_385, 26_916),
    ("MALBAC\n+ ChimeraLM", 394_385 - 23_488, 26_916 - 23_488),
]


def draw(ax, bars, title=None):
    x = range(len(bars))
    non = [t - c for _, t, c in bars]
    chim = [c for _, _, c in bars]
    ax.bar(x, non, color=T2, edgecolor="k", linewidth=0.6, label="Non-chimeric reads")
    ax.bar(x, chim, bottom=non, color=T1, edgecolor="k", linewidth=0.6, label="Chimeric reads")
    for i, (_, t, c) in enumerate(bars):
        ax.text(i, t * 1.02, f"{c / t * 100:.1f}%", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(list(x), [b[0] for b in bars], fontsize=8.5)
    ax.set_ylabel("Mapped reads")
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v/1e6:.0f}M" if v >= 1e6 else f"{v/1e3:.0f}k"))
    ax.set_ylim(0, max(t for _, t, _ in bars) * 1.15)
    ax.legend(frameon=False, fontsize=8.5, loc="upper right")
    ax.spines[["top", "right"]].set_visible(False)
    if title:
        ax.set_title(title, loc="left", fontsize=10)


fig, ax = plt.subplots(figsize=(7.2, 4.0))
draw(ax, BARS, "HCC78 (Ni et al. 2023): chimeric read fraction")
fig.tight_layout()
fig.savefig(OUT / "sf_hcc78_chimeric_bars.pdf"); fig.savefig(OUT / "sf_hcc78_chimeric_bars.png", dpi=170)
for b in BARS:
    print(f"{b[0].replace(chr(10), ' '):22s} mapped {b[1]:>9,}  chimeric {b[2]:>9,}  ({b[2]/b[1]*100:5.1f}%)")
print("wrote", OUT / "sf_hcc78_chimeric_bars.pdf")
