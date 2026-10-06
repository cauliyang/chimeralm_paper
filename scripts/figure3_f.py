"""Redraw Fig. 3f (SV-type composition) at SUPPORT >= 3 (revision R1, R2.Q1).

Counts: figures/data/revision/fig3f_svtype_support3.txt (Sniffles2 v2.5 --minsupport 1 output filtered
to SUPPORT >= 3; WGA Set A / WGA + ChimeraLM Set C are the Fig. 3b-e call sets; bulk filtered identically).
Donut style = draw_figures.ipynb `draw_pie_chart_for_sv_type_distribution` (colours, labels, leader lines).

Usage: python scripts/figure3_f.py  -> figures/final_figures/figure3_f.pdf + figure3_f_<platform>_<set>.pdf
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures" / "final_figures"
SV_TYPES = ["TRA", "DEL", "INV", "DUP", "INS"]
SV_COLORS = {"INV": "#C98BB1", "DEL": "#6AA995", "DUP": "#D68B6D", "INS": "#8A97B3", "TRA": "#96B85E"}
COLORS = [SV_COLORS[s] for s in SV_TYPES]
SETS = [("bulk", "Bulk"), ("wga", "WGA"), ("chimeralm", "WGA+ChimeraLM")]
PLATFORMS = [("P2", "PromethION"), ("Mk1c", "MinION")]


def parse_counts(path: Path) -> dict[tuple[str, str], list[int]]:
    data, key = {}, None
    for line in path.read_text().splitlines():
        if line.startswith("## "):
            h = line[3:]
            plat = "P2" if h.startswith("P2") else "Mk1c"
            s = "bulk" if "bulk" in h else ("chimeralm" if "ChimeraLM" in h else "wga")
            key = (plat, s)
            data[key] = {}
        elif line.strip() and key:
            t, n = line.split("\t")
            data[key][t] = int(n)
    return {k: [v.get(t, 0) for t in SV_TYPES] for k, v in data.items()}


def create_donut(ax, values, title=None):
    total = sum(values)
    wedges, texts, autotexts = ax.pie(
        values, colors=COLORS, autopct=lambda p: f"{p:.1f}%" if p > 8 else "", startangle=90, pctdistance=0.82,
        wedgeprops=dict(width=0.6, edgecolor="white", linewidth=3), textprops={"fontsize": 14, "fontweight": "bold"},
    )
    for a in autotexts:
        a.set_color("white"); a.set_fontsize(13); a.set_fontweight("bold")
    for t in texts:
        t.set_fontsize(0)
    for i, (w, v) in enumerate(zip(wedges, values, strict=True)):
        pct = v / total * 100
        if 0.1 < pct <= 8:
            ang = (w.theta2 - w.theta1) / 2.0 + w.theta1
            x, y = np.cos(np.deg2rad(ang)), np.sin(np.deg2rad(ang))
            y_off = (i % 3 - 1) * 0.2 if (45 < ang < 135 or 225 < ang < 315) else 0
            ax.annotate(f"{SV_TYPES[i]}: {pct:.1f}%", xy=(x * 0.75, y * 0.75), xytext=(x * 1.5, y * 1.5 + y_off),
                        ha="left" if x >= 0 else "right", va="center", fontsize=11, fontweight="bold", color="#2c3e50",
                        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor=COLORS[i], linewidth=2, alpha=0.95),
                        arrowprops=dict(arrowstyle="-", color=COLORS[i], linewidth=2, alpha=0.8))
    ax.text(0, 0, f"{total:,}\nSVs", ha="center", va="center", fontsize=17, fontweight="bold", color="#333333")
    if title:
        ax.set_title(title, fontsize=18, fontweight="bold", pad=25, color="#2c3e50")
    ax.set_xlim(-1.9, 1.9); ax.set_ylim(-1.9, 1.9)


def main() -> None:
    counts = parse_counts(ROOT / "figures/data/revision/fig3f_svtype_support3.txt")
    for (plat, pname) in PLATFORMS:
        print(pname)
        for s, sname in SETS:
            v = counts[(plat, s)]; tot = sum(v)
            print(f"  {sname:15s} total {tot:>9,}  " + "  ".join(f"{t} {n:,} ({n/tot*100:.1f}%)" for t, n in zip(SV_TYPES, v, strict=True)))
    fig, axes = plt.subplots(2, 3, figsize=(25, 17))
    for r, (plat, pname) in enumerate(PLATFORMS):
        for c, (s, sname) in enumerate(SETS):
            create_donut(axes[r, c], counts[(plat, s)], f"{pname} {sname}")
    handles = [Rectangle((0, 0), 1, 1, facecolor=SV_COLORS[t], edgecolor="white", linewidth=2, label=t) for t in SV_TYPES]
    fig.legend(handles=handles, labels=SV_TYPES, loc="lower center", ncol=5, frameon=False, fontsize=16, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    fig.savefig(OUT / "figure3_f.pdf", bbox_inches="tight"); fig.savefig(OUT / "figure3_f.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    for (plat, _), in zip(PLATFORMS, strict=True):
        for s, _ in SETS:
            fig, ax = plt.subplots(figsize=(6, 6))
            create_donut(ax, counts[(plat, s)])
            fig.savefig(OUT / f"figure3_f_{plat.lower()}_{s}.pdf", bbox_inches="tight"); plt.close(fig)
    print(f"wrote {OUT}/figure3_f.pdf and 6 single donuts")


if __name__ == "__main__":
    main()
