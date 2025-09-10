#!/usr/bin/env -S uv run --script

# /// script
# dependencies = [
#  "graphviz",
#  "matplotlib",
# ]
# ///


import matplotlib.pyplot as plt
import matplotlib.patches as patches


def draw_model_architecture():
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.axis("off")

    # ====================
    # Colors (Nature Methods style, pastel & muted)
    # ====================
    color_input = "#4C78A8"  # muted teal-blue
    color_backbone = "#F2C572"  # soft sand
    color_pool = "#72B7B2"  # sage green
    color_mlp = "#E28E5C"  # muted orange
    color_output = "#D47474"  # soft red

    # ====================
    # Input tokens
    # ====================
    for i, base in enumerate(["A", "T", "G", "C", "..."]):
        ax.add_patch(
            patches.Rectangle(
                (0, i * 0.6), 0.8, 0.5, facecolor=color_input, edgecolor="black", lw=0.8
            )
        )
        ax.text(
            0.4,
            i * 0.6 + 0.25,
            base,
            ha="center",
            va="center",
            fontsize=9,
            color="white",
            weight="bold",
        )

    ax.text(0.4, 3.5, "DNA sequence tokens", ha="center", fontsize=12)

    # ====================
    # Backbone
    # ====================
    ax.add_patch(
        patches.FancyBboxPatch(
            (2, 0.5),
            2.5,
            2.5,
            boxstyle="round,pad=0.1",
            facecolor=color_backbone,
            edgecolor="black",
            lw=1,
        )
    )
    ax.text(
        3.25,
        1.75,
        "HyenaDNA\nBackbone\n[batch, seq_len, 256]",
        ha="center",
        va="center",
        fontsize=11,
    )

    # Arrow to pooling
    ax.annotate(
        "",
        xy=(4.7, 1.75),
        xytext=(4.5, 1.75),
        arrowprops=dict(arrowstyle="->", lw=1.5, color="black"),
    )

    # ====================
    # Attention Pooling
    # ====================
    ax.add_patch(
        patches.Polygon(
            [[5, 1], [6, 1.75], [5, 2.5]],
            closed=True,
            facecolor=color_pool,
            edgecolor="black",
            lw=1,
        )
    )
    ax.text(
        5.5,
        1.75,
        "Attention\nPooling\n[batch,256]",
        ha="center",
        va="center",
        fontsize=11,
    )

    # Arrow to MLP
    ax.annotate(
        "",
        xy=(6.7, 1.75),
        xytext=(6, 1.75),
        arrowprops=dict(arrowstyle="->", lw=1.5, color="black"),
    )

    # ====================
    # MLP Classifier as nodes
    # ====================
    layer_sizes = [5, 4, 3]  # schematic only
    layer_x = [7.5, 8.5, 9.5]

    for li, size in enumerate(layer_sizes):
        for ni in range(size):
            y = 1.75 + (ni - size / 2) * 0.5
            ax.add_patch(
                patches.Circle(
                    (layer_x[li], y),
                    0.15,
                    facecolor=color_mlp,
                    edgecolor="black",
                    lw=0.8,
                )
            )
            if li > 0:
                prev_size = layer_sizes[li - 1]
                for pj in range(prev_size):
                    y_prev = 1.75 + (pj - prev_size / 2) * 0.5
                    ax.plot(
                        [layer_x[li - 1] + 0.15, layer_x[li] - 0.15],
                        [y_prev, y],
                        color="gray",
                        lw=0.6,
                        alpha=0.7,
                    )

    ax.text(
        8.5, 3.2, "MLP Classifier\nResidual + Dropout + GELU", ha="center", fontsize=11
    )

    # ====================
    # Output Layer
    # ====================
    ax.add_patch(
        patches.Circle(
            (10.8, 2.2), 0.25, facecolor=color_output, edgecolor="black", lw=0.8
        )
    )
    ax.add_patch(
        patches.Circle(
            (10.8, 1.3), 0.25, facecolor=color_output, edgecolor="black", lw=0.8
        )
    )
    ax.text(11.3, 2.2, "Class 0", va="center", fontsize=11)
    ax.text(11.3, 1.3, "Class 1", va="center", fontsize=11)

    # Arrow from last hidden layer to outputs
    ax.annotate(
        "",
        xy=(10.5, 1.75),
        xytext=(9.65, 1.75),
        arrowprops=dict(arrowstyle="->", lw=1.5, color="black"),
    )

    # ====================
    # Layout
    # ====================
    ax.set_xlim(-0.5, 12)
    ax.set_ylim(-0.5, 4)
    plt.tight_layout()
    plt.savefig("model_architecture_fancy.png", dpi=600)  # high-res raster
    plt.savefig("model_architecture_fancy.pdf")  # vector version
    plt.show()


if __name__ == "__main__":
    draw_model_architecture()
