"""Ext Data Fig: features of residual unsupported SV calls after ChimeraLM vs supported calls
(R3.Q6/R3.Q11). Data: revision/residual_sv/residual_sv_features.py output (Truvari FP vs TP-comp,
WGA + ChimeraLM call sets vs gold standard, SUPPORT >= 3)."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

root = Path(__file__).resolve().parents[1]
t = pd.read_csv(root / "figures/data/revision/residual_sv/residual_sv_tables.tsv", sep="\t", dtype={"bin": str})
C_UNS, C_SUP = "#9fc8c8", "#00a087"
TYPE_ORDER = ["DEL", "INS", "INV", "DUP", "TRA"]
SIZE_ORDER = ["50–100 bp", "100–500 bp", "500 bp–1 kb", "1–5 kb", "5–10 kb", "10–50 kb", ">50 kb"]
SUP_ORDER = ["3", "4", "5", "6–10", "11–20", "21–50", ">50"]
PANELS = [("svtype", TYPE_ORDER, "SV type"), ("size", SIZE_ORDER, "SV size"), ("support", SUP_ORDER, "Sniffles2 SUPPORT (reads)")]
OUT = root / "figures/final_figures"


def draw(ax, dset: str, feature: str, order: list[str], xlabel: str) -> None:
    sub = t[(t["set"] == dset) & (t["feature"] == feature) & (t["class"].isin(["unsupported", "supported"]))]
    n = {c: int(sub[sub["class"] == c]["count"].sum()) for c in ("unsupported", "supported")}
    x = np.arange(len(order)); w = 0.4
    for i, (cls, col) in enumerate((("unsupported", C_UNS), ("supported", C_SUP))):
        s = sub[sub["class"] == cls].set_index("bin")["pct"].reindex(order).fillna(0)
        bars = ax.bar(x + (i - 0.5) * w, s.values, w, color=col, edgecolor="k", linewidth=0.6,
                      label=f"{cls.capitalize()} (n = {n[cls]:,})")
        for b, v in zip(bars, s.values, strict=True):
            if v > 0:
                ax.text(b.get_x() + b.get_width() / 2, v + 1, f"{v:.0f}", ha="center", va="bottom", fontsize=6.5)
    ax.set_xticks(x, order, rotation=30 if feature != "svtype" else 0, ha="right" if feature != "svtype" else "center", fontsize=8)
    ax.set_ylabel("Calls (%)")
    ax.set_xlabel(xlabel)
    ax.set_ylim(0, 105)
    ax.legend(frameon=False, fontsize=7.5, loc="upper right")
    ax.spines[["top", "right"]].set_visible(False)


letters = iter("abcdef")
fig, axes = plt.subplots(2, 3, figsize=(12, 7))
for r, dset in enumerate(["PromethION", "MinION"]):
    for c, (feature, order, xlabel) in enumerate(PANELS):
        ax = axes[r, c]
        draw(ax, dset, feature, order, xlabel)
        ax.set_title(f"{next(letters)}  {dset}: {xlabel}", loc="left", fontweight="bold", fontsize=10)
fig.tight_layout()
fig.savefig(OUT / "sf_residual_sv.pdf"); fig.savefig(OUT / "sf_residual_sv.png", dpi=170); plt.close(fig)
letters = iter("abcdef")
for dset in ["PromethION", "MinION"]:
    for feature, order, xlabel in PANELS:
        fig, ax = plt.subplots(figsize=(4.4, 3.4))
        draw(ax, dset, feature, order, xlabel)
        ax.set_title(f"{dset}: {xlabel}", loc="left", fontsize=10)
        fig.tight_layout(); fig.savefig(OUT / f"sf_residual_sv_{next(letters)}.pdf"); plt.close(fig)
print("ok")
