import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from math import pi, cos, sin
import seaborn as sns
from matplotlib.patches import Wedge


# /// script
# dependencies = [
#   "matplotlib",
#   "plotly",
#   "pandas",
#   "numpy",
#   "seaborn",
#   "plotly",
# ]
# ///


# Your dataframe
data = {
    "Model": [
        "BulkMk1c",
        "BulkP2",
        "BulkPacBio",
        "HyenaMk1c",
        "HyenaP2",
        "OriginalMk1c",
        "OriginalP2",
    ],
    "Dataset": [
        "Reference",
        "Reference",
        "Reference",
        "Set_C",
        "Set_C",
        "Set_A",
        "Set_A",
    ],
    "TRA_count": [552, 5293, 588, 276, 6324, 2892, 5791],
    "DEL_count": [12942, 72178, 18413, 13480, 65667, 28471, 26252],
    "INV_count": [4143, 11089, 1730, 18121, 190835, 406305, 306148],
    "DUP_count": [176, 1027, 768, 788, 6622, 10892, 15569],
    "INS_count": [11842, 29655, 22119, 6530, 43073, 11613, 15139],
}

df = pd.DataFrame(data)

# SV types and colors
sv_types = ["TRA", "DEL", "INV", "DUP", "INS"]
sv_colors = ["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FFEAA7"]

# Group models by sequencing technology
mk1c_models = ["BulkMk1c", "HyenaMk1c", "OriginalMk1c"]
p2_models = ["BulkP2", "HyenaP2", "OriginalP2"]

print("Creating Rose Diagrams grouped by sequencing technology...")


# 1. Classic Rose Diagram - Side by Side
def create_rose_diagrams():
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(20, 10), subplot_kw=dict(projection="polar")
    )

    # Calculate angular positions for each SV type (like compass directions)
    angles = np.linspace(0, 2 * pi, len(sv_types), endpoint=False)
    width = 2 * pi / len(sv_types)  # Width of each sector

    # Mk1c Rose Diagram
    mk1c_data = df[df["Model"].isin(mk1c_models)]

    # Calculate total counts for each SV type in Mk1c
    mk1c_totals = [mk1c_data[f"{sv}_count"].sum() for sv in sv_types]
    max_mk1c = max(mk1c_totals)

    # Normalize for visualization (log scale)
    mk1c_log = [np.log10(count + 1) for count in mk1c_totals]
    max_log_mk1c = max(mk1c_log)

    # Create sectors
    for i, (sv_type, angle, count, log_count) in enumerate(
        zip(sv_types, angles, mk1c_totals, mk1c_log)
    ):
        # Draw the sector
        bars = ax1.bar(
            angle,
            log_count,
            width=width * 0.9,
            bottom=0,
            color=sv_colors[i],
            alpha=0.8,
            edgecolor="white",
            linewidth=2,
        )

        # Add count labels
        label_angle = angle
        label_radius = log_count + 0.2
        ax1.text(
            label_angle,
            label_radius,
            f"{sv_type}\n{count:,}",
            ha="center",
            va="center",
            fontsize=11,
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9),
        )

    # Customize Mk1c diagram
    ax1.set_ylim(0, max_log_mk1c * 1.3)
    ax1.set_theta_zero_location("N")  # Start from top like compass
    ax1.set_theta_direction(-1)  # Clockwise
    ax1.set_title(
        "Mk1c Models Rose Diagram\n(Log Scale)", size=18, fontweight="bold", pad=30
    )
    ax1.set_rticks([1, 2, 3, 4, 5, 6])
    ax1.set_rticklabels(["10¹", "10²", "10³", "10⁴", "10⁵", "10⁶"])
    ax1.grid(True, alpha=0.3)

    # Add model breakdown as concentric rings
    for model_idx, (_, row) in enumerate(mk1c_data.iterrows()):
        radius_offset = 0.15 * (model_idx + 1)
        for sv_idx, sv_type in enumerate(sv_types):
            count = row[f"{sv_type}_count"]
            log_count = np.log10(count + 1)
            angle = angles[sv_idx]

            # Draw smaller bars for individual models
            ax1.bar(
                angle,
                log_count,
                width=width * 0.25,
                bottom=radius_offset,
                color=sv_colors[sv_idx],
                alpha=0.6,
                edgecolor="black",
                linewidth=0.5,
            )

    # P2 Rose Diagram
    p2_data = df[df["Model"].isin(p2_models)]

    # Calculate total counts for each SV type in P2
    p2_totals = [p2_data[f"{sv}_count"].sum() for sv in sv_types]
    max_p2 = max(p2_totals)

    # Normalize for visualization (log scale)
    p2_log = [np.log10(count + 1) for count in p2_totals]
    max_log_p2 = max(p2_log)

    # Create sectors
    for i, (sv_type, angle, count, log_count) in enumerate(
        zip(sv_types, angles, p2_totals, p2_log)
    ):
        # Draw the sector
        bars = ax2.bar(
            angle,
            log_count,
            width=width * 0.9,
            bottom=0,
            color=sv_colors[i],
            alpha=0.8,
            edgecolor="white",
            linewidth=2,
        )

        # Add count labels
        label_angle = angle
        label_radius = log_count + 0.2
        ax2.text(
            label_angle,
            label_radius,
            f"{sv_type}\n{count:,}",
            ha="center",
            va="center",
            fontsize=11,
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9),
        )

    # Customize P2 diagram
    ax2.set_ylim(0, max_log_p2 * 1.3)
    ax2.set_theta_zero_location("N")
    ax2.set_theta_direction(-1)
    ax2.set_title(
        "P2 Models Rose Diagram\n(Log Scale)", size=18, fontweight="bold", pad=30
    )
    ax2.set_rticks([1, 2, 3, 4, 5, 6])
    ax2.set_rticklabels(["10¹", "10²", "10³", "10⁴", "10⁵", "10⁶"])
    ax2.grid(True, alpha=0.3)

    # Add model breakdown as concentric rings
    for model_idx, (_, row) in enumerate(p2_data.iterrows()):
        radius_offset = 0.15 * (model_idx + 1)
        for sv_idx, sv_type in enumerate(sv_types):
            count = row[f"{sv_type}_count"]
            log_count = np.log10(count + 1)
            angle = angles[sv_idx]

            # Draw smaller bars for individual models
            ax2.bar(
                angle,
                log_count,
                width=width * 0.25,
                bottom=radius_offset,
                color=sv_colors[sv_idx],
                alpha=0.6,
                edgecolor="black",
                linewidth=0.5,
            )

    plt.suptitle(
        "Rose Diagrams: SV Counts by Sequencing Technology",
        fontsize=22,
        fontweight="bold",
        y=0.95,
    )

    # Add legends
    legend_elements = [
        plt.Rectangle((0, 0), 1, 1, facecolor=sv_colors[i], label=sv_types[i])
        for i in range(len(sv_types))
    ]
    fig.legend(
        handles=legend_elements,
        loc="center",
        bbox_to_anchor=(0.5, 0.02),
        ncol=len(sv_types),
        fontsize=12,
    )

    plt.tight_layout()
    return fig


# 2. Layered Rose Diagram (Wind Rose Style)
def create_layered_rose_diagram():
    fig, ax = plt.subplots(figsize=(16, 16), subplot_kw=dict(projection="polar"))

    angles = np.linspace(0, 2 * pi, len(sv_types), endpoint=False)
    width = 2 * pi / len(sv_types)

    # Get all data
    mk1c_data = df[df["Model"].isin(mk1c_models)]
    p2_data = df[df["Model"].isin(p2_models)]

    # Layer 1: P2 models (outer layer)
    p2_totals = [p2_data[f"{sv}_count"].sum() for sv in sv_types]
    p2_log = [np.log10(count + 1) for count in p2_totals]

    for i, (sv_type, angle, log_count) in enumerate(zip(sv_types, angles, p2_log)):
        ax.bar(
            angle,
            log_count,
            width=width * 0.8,
            bottom=0,
            color=sv_colors[i],
            alpha=0.6,
            edgecolor="white",
            linewidth=3,
            label=f"{sv_type} (P2)" if i < len(sv_types) else "",
        )

    # Layer 2: Mk1c models (inner layer)
    mk1c_totals = [mk1c_data[f"{sv}_count"].sum() for sv in sv_types]
    mk1c_log = [np.log10(count + 1) for count in mk1c_totals]

    for i, (sv_type, angle, log_count) in enumerate(zip(sv_types, angles, mk1c_log)):
        ax.bar(
            angle,
            log_count,
            width=width * 0.6,
            bottom=0,
            color=sv_colors[i],
            alpha=0.9,
            edgecolor="black",
            linewidth=2,
            hatch="///",
            label=f"{sv_type} (Mk1c)" if i < len(sv_types) else "",
        )

    # Add directional labels
    for i, (sv_type, angle) in enumerate(zip(sv_types, angles)):
        # Calculate label position
        max_radius = max(max(p2_log), max(mk1c_log)) * 1.15
        x = max_radius * cos(angle - pi / 2)  # Adjust for polar coordinates
        y = max_radius * sin(angle - pi / 2)

        ax.annotate(
            sv_type,
            (angle, max_radius),
            ha="center",
            va="center",
            fontsize=16,
            fontweight="bold",
            bbox=dict(boxstyle="circle,pad=0.3", facecolor=sv_colors[i], alpha=0.8),
        )

    # Customize
    ax.set_ylim(0, max(max(p2_log), max(mk1c_log)) * 1.3)
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_title(
        "Layered Rose Diagram\nP2 (outer) vs Mk1c (inner, hatched)",
        size=20,
        fontweight="bold",
        pad=50,
    )
    ax.set_rticks([1, 2, 3, 4, 5, 6])
    ax.set_rticklabels(["10¹", "10²", "10³", "10⁴", "10⁵", "10⁶"], fontsize=12)
    ax.grid(True, alpha=0.3)

    # Add custom legend
    legend_elements = [
        plt.Rectangle((0, 0), 1, 1, facecolor="gray", alpha=0.6, label="P2 Models"),
        plt.Rectangle(
            (0, 0), 1, 1, facecolor="gray", alpha=0.9, hatch="///", label="Mk1c Models"
        ),
    ]
    ax.legend(
        handles=legend_elements, loc="center", bbox_to_anchor=(1.3, 0.5), fontsize=14
    )

    return fig


# 3. Multi-ring Rose Diagram (Each model as separate ring)
def create_multiring_rose_diagram():
    fig, ax = plt.subplots(figsize=(18, 18), subplot_kw=dict(projection="polar"))

    angles = np.linspace(0, 2 * pi, len(sv_types), endpoint=False)
    width = 2 * pi / len(sv_types)

    # Define ring positions
    ring_positions = [0, 1, 2, 3, 4, 5]  # 6 models = 6 rings
    ring_height = 0.8

    all_models = mk1c_models + p2_models
    model_colors = ["#FF4444", "#44FF44", "#4444FF", "#FFAA44", "#AA44FF", "#44AAFF"]

    for model_idx, model in enumerate(all_models):
        model_data = df[df["Model"] == model].iloc[0]
        ring_bottom = ring_positions[model_idx]

        for sv_idx, sv_type in enumerate(sv_types):
            count = model_data[f"{sv_type}_count"]
            # Normalize count for ring height
            log_count = np.log10(count + 1) / 6 * ring_height  # Scale to ring height

            ax.bar(
                angles[sv_idx],
                log_count,
                width=width * 0.9,
                bottom=ring_bottom,
                color=sv_colors[sv_idx],
                alpha=0.8,
                edgecolor=model_colors[model_idx],
                linewidth=2,
            )

            # Add count labels for significant values
            if count > 1000:
                label_radius = ring_bottom + log_count / 2
                ax.text(
                    angles[sv_idx],
                    label_radius,
                    f"{count:,}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    fontweight="bold",
                    color="white",
                )

    # Add model labels on the side
    for model_idx, model in enumerate(all_models):
        ring_center = ring_positions[model_idx] + ring_height / 2
        tech_type = "(Mk1c)" if model in mk1c_models else "(P2)"
        ax.text(
            0,
            ring_center,
            f"{model}\n{tech_type}",
            ha="left",
            va="center",
            fontsize=11,
            fontweight="bold",
            bbox=dict(
                boxstyle="round,pad=0.3", facecolor=model_colors[model_idx], alpha=0.7
            ),
            transform=ax.transData,
        )

    # Add SV type labels
    for sv_idx, sv_type in enumerate(sv_types):
        label_radius = max(ring_positions) + ring_height + 0.3
        ax.text(
            angles[sv_idx],
            label_radius,
            sv_type,
            ha="center",
            va="center",
            fontsize=14,
            fontweight="bold",
            bbox=dict(
                boxstyle="circle,pad=0.3", facecolor=sv_colors[sv_idx], alpha=0.8
            ),
        )

    ax.set_ylim(0, max(ring_positions) + ring_height + 1)
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_title(
        "Multi-Ring Rose Diagram\n(Each ring = one model)",
        size=20,
        fontweight="bold",
        pad=50,
    )
    ax.set_yticks([])  # Hide radial ticks for clarity
    ax.grid(True, alpha=0.2)

    return fig


# 4. Interactive Plotly Rose Diagram
def create_plotly_rose_diagram():
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=["Mk1c Models", "P2 Models"],
        specs=[[{"type": "polar"}, {"type": "polar"}]],
    )

    # Mk1c
    mk1c_data = df[df["Model"].isin(mk1c_models)]
    mk1c_totals = [mk1c_data[f"{sv}_count"].sum() for sv in sv_types]

    fig.add_trace(
        go.Barpolar(
            r=[np.log10(count + 1) for count in mk1c_totals],
            theta=sv_types,
            name="Mk1c Total",
            marker_color=sv_colors,
            hovertemplate="<b>%{theta}</b><br>Count: %{customdata:,}<br>Log: %{r:.2f}<extra></extra>",
            customdata=mk1c_totals,
            opacity=0.8,
        ),
        row=1,
        col=1,
    )

    # P2
    p2_data = df[df["Model"].isin(p2_models)]
    p2_totals = [p2_data[f"{sv}_count"].sum() for sv in sv_types]

    fig.add_trace(
        go.Barpolar(
            r=[np.log10(count + 1) for count in p2_totals],
            theta=sv_types,
            name="P2 Total",
            marker_color=sv_colors,
            hovertemplate="<b>%{theta}</b><br>Count: %{customdata:,}<br>Log: %{r:.2f}<extra></extra>",
            customdata=p2_totals,
            opacity=0.8,
        ),
        row=1,
        col=2,
    )

    fig.update_layout(
        title_text="Interactive Rose Diagrams: SV Counts by Technology",
        showlegend=False,
        height=800,
        width=1200,
    )

    return fig


# Create and display all rose diagrams
print("Creating classic rose diagrams...")
rose_fig = create_rose_diagrams()
plt.show()

print("Creating layered rose diagram...")
layered_fig = create_layered_rose_diagram()
plt.show()

print("Creating multi-ring rose diagram...")
multiring_fig = create_multiring_rose_diagram()
plt.show()

print("Creating interactive rose diagram...")
plotly_rose_fig = create_plotly_rose_diagram()
plotly_rose_fig.show()

# Analysis summary
print("\n" + "=" * 60)
print("ROSE DIAGRAM ANALYSIS: Mk1c vs P2 COMPARISON")
print("=" * 60)

mk1c_data = df[df["Model"].isin(mk1c_models)]
p2_data = df[df["Model"].isin(p2_models)]

mk1c_totals = [mk1c_data[f"{sv}_count"].sum() for sv in sv_types]
p2_totals = [p2_data[f"{sv}_count"].sum() for sv in sv_types]

print(f"\nTotal SV Counts by Technology:")
print(f"Mk1c: {sum(mk1c_totals):,}")
print(f"P2:   {sum(p2_totals):,}")
print(f"P2 detects {(sum(p2_totals) / sum(mk1c_totals) - 1) * 100:.1f}% more SVs")

print(f"\nSV Distribution by Technology:")
for i, sv in enumerate(sv_types):
    mk1c_pct = (mk1c_totals[i] / sum(mk1c_totals)) * 100
    p2_pct = (p2_totals[i] / sum(p2_totals)) * 100
    ratio = p2_totals[i] / mk1c_totals[i] if mk1c_totals[i] > 0 else float("inf")
    print(
        f"  {sv}: Mk1c {mk1c_pct:5.1f}% ({mk1c_totals[i]:6,}) | P2 {p2_pct:5.1f}% ({p2_totals[i]:6,}) | Ratio: {ratio:.2f}x"
    )

print(f"\nDominant SV Type:")
mk1c_dominant = sv_types[mk1c_totals.index(max(mk1c_totals))]
p2_dominant = sv_types[p2_totals.index(max(p2_totals))]
print(f"  Mk1c: {mk1c_dominant} ({(max(mk1c_totals) / sum(mk1c_totals)) * 100:.1f}%)")
print(f"  P2:   {p2_dominant} ({(max(p2_totals) / sum(p2_totals)) * 100:.1f}%)")

print(f"\nRequired libraries:")
print("pip install matplotlib plotly pandas numpy seaborn")
