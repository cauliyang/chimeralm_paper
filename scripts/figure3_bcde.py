"""Redraw Fig. 3b-e (revision R1, R2.Q1) as single-panel PDFs.

Numbers: Truvari v4.2.2 summary.json (TP-comp = Supported, FP = No support) from
Qingxiang's SUPPORT>=3 benchmark
(/gpfs/projects/b1171/qgn1237/4_single_cell_SV_chimera/20260401_R2Q1_alternative_strict_GT_benchmark_minsupport_3_final).
Layout follows the rebuttal slide (grouped bars, shared log scale, counts and FP:TP printed),
replacing the two-scale design of the submitted figure. Palette = draw_figures.ipynb (t1/t2).

Usage: python scripts/figure3_bcde.py  (writes figures/final_figures/figure3_{b,c,d,e}.pdf)
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

T1, T2 = "#00a087", "#9fc8c8"  # supported (dark teal), no support (light teal)
OUT = Path(__file__).resolve().parents[1] / "figures" / "final_figures"

# panel: (platform, reference label, {set: (supported, no_support)})
PANELS = {
    "b": ("PromethION", "High-confidence SVs", {"WGA": (5147, 263790), "WGA + ChimeraLM": (4490, 4332)}),
    "c": ("MinION", "High-confidence SVs", {"WGA": (1940, 6977), "WGA + ChimeraLM": (1450, 606)}),
    "d": ("PromethION", "SVs of bulk", {"WGA": (7019, 261918), "WGA + ChimeraLM": (6048, 2774)}),
    "e": ("MinION", "SVs of bulk", {"WGA": (1880, 7037), "WGA + ChimeraLM": (1413, 643)}),
}


def draw(ax, platform: str, ref: str, data: dict[str, tuple[int, int]]) -> None:
    sets = list(data)
    x = range(len(sets))
    w = 0.36
    for i, s in enumerate(sets):
        sup, nosup = data[s]
        b1 = ax.bar(i - w / 2, nosup, w, color=T2, edgecolor="k", linewidth=0.8, label="No support" if i == 0 else None)
        b2 = ax.bar(i + w / 2, sup, w, color=T1, edgecolor="k", linewidth=0.8, label=f"Supported by {ref}" if i == 0 else None)
        for b, v in ((b1[0], nosup), (b2[0], sup)):
            ax.text(b.get_x() + b.get_width() / 2, v * 1.15, f"{v:,}", ha="center", va="bottom", fontsize=9)
        ax.text(i, -0.17, f"unsupported : supported\n{nosup / sup:.2f} : 1", transform=ax.get_xaxis_transform(),
                ha="center", va="top", fontsize=8, color="#444444")
    ax.set_yscale("log")
    ymax = max(v for pair in data.values() for v in pair)
    ax.set_ylim(100, ymax * 6)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{int(v):,}"))
    ax.set_xticks(list(x), sets)
    ax.set_ylabel("SV calls")
    ax.set_title(f"{platform} vs {ref}", loc="left", fontsize=11)
    ax.legend(frameon=False, fontsize=8.5, loc="upper right")
    ax.spines[["top", "right"]].set_visible(False)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for letter, (platform, ref, data) in PANELS.items():
        fig, ax = plt.subplots(figsize=(4.6, 3.8))
        draw(ax, platform, ref, data)
        fig.tight_layout()
        fig.savefig(OUT / f"figure3_{letter}.pdf")
        plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(9.4, 7.6))
    for ax, (letter, (platform, ref, data)) in zip(axes.flat, PANELS.items(), strict=True):
        draw(ax, platform, ref, data)
        ax.set_title(f"{letter}  {platform} vs {ref}", loc="left", fontsize=11, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "figure3_bcde.pdf")
    fig.savefig(OUT / "figure3_bcde.png", dpi=170)
    print(f"wrote {OUT}/figure3_{{b,c,d,e}}.pdf and figure3_bcde.pdf")


if __name__ == "__main__":
    main()
