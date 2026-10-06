"""Ext Data Fig: P2 depth titration of cross-platform SV concordance (R3.Q4). Data from
Qingxiang's 20260420_R3Q4 analysis (concordance_per_depth.tsv): bulk P2 BAM downsampled with
samtools view -s, Sniffles2 SUPPORT>=3, OctopuSV three-way merge with bulk Mk1c and PacBio."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

root = Path(__file__).resolve().parents[1]
df = pd.read_csv(root / "figures/data/revision/concordance_per_depth.tsv", sep="\t")
T1, GREY = "#00a087", "#7f7f7f"
fig, ax = plt.subplots(figsize=(5.2, 4.0))
ax.plot(df["depth"], df["p2_total"], "o-", color=GREY, lw=1.8, label="Bulk PromethION SVs (SUPPORT ≥ 3)")
for i, (x, y) in enumerate(zip(df["depth"], df["p2_total"], strict=True)):
    last = i == len(df) - 1
    ax.annotate(f"{y:,}", (x, y), textcoords="offset points", xytext=(-8, -4) if last else (0, -14),
                ha="right" if last else "center", fontsize=7.5, color=GREY)
ax.set_xlabel("PromethION sequencing depth (×)")
ax.set_ylabel("SV calls", color=GREY)
ax.set_ylim(0, df["p2_total"].max() * 1.25)
ax2 = ax.twinx()
ax2.plot(df["depth"], df["p2_supported_pct"], "s-", color=T1, lw=1.8, label="Supported by ≥1 other platform (%)")
for x, y in zip(df["depth"], df["p2_supported_pct"], strict=True):
    ax2.annotate(f"{y:.1f}%", (x, y), textcoords="offset points", xytext=(0, 7), ha="center", fontsize=7.5, color=T1)
ax2.set_ylabel("PromethION SVs supported by\nMinION or PacBio (%)", color=T1)
ax2.set_ylim(0, 105)
ax.set_xticks(df["depth"], [f"{d:g}×" for d in df["depth"]])
h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
ax.legend(h1 + h2, l1 + l2, frameon=False, fontsize=8, loc="lower right")
ax.spines["top"].set_visible(False); ax2.spines["top"].set_visible(False)
ax.set_title("Cross-platform concordance vs depth", loc="left")
fig.tight_layout()
fig.savefig(root / "figures/final_figures/sf_bulk_concordance_b.pdf")
fig.savefig(root / "figures/final_figures/sf_bulk_concordance_b.png", dpi=170)
print("ok")
