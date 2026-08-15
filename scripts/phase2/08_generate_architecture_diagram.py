"""Generate ResearchPilot final architecture diagram for M8 / DA2."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[2]
OUT_DIRS = [
    ROOT / "reports" / "figures" / "final",
    ROOT / "DA2" / "figures" / "final",
]


def box(ax, xy, w, h, text, fc="#e8f1f8", ec="#2c3e50", fontsize=8, bold=False):
    x, y = xy
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.08",
        linewidth=1.2,
        facecolor=fc,
        edgecolor=ec,
    )
    ax.add_patch(patch)
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        fontweight="bold" if bold else "normal",
        color="#1a1a1a",
    )
    return x + w / 2, y, y + h  # cx, bottom, top


def arrow(ax, x1, y1, x2, y2):
    ax.add_patch(
        FancyArrowPatch(
            (x1, y1),
            (x2, y2),
            arrowstyle="-|>",
            mutation_scale=12,
            linewidth=1.1,
            color="#34495e",
        )
    )


def main() -> None:
    for d in OUT_DIRS:
        d.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(11, 14), dpi=160)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 16.2)
    ax.axis("off")
    ax.set_title(
        "ResearchPilot — Final Architecture (Phase 1 + Phase 2)",
        fontsize=13,
        fontweight="bold",
        pad=12,
    )

    ax.add_patch(
        FancyBboxPatch(
            (0.3, 8.55),
            9.4,
            7.0,
            boxstyle="round,pad=0.02,rounding_size=0.05",
            facecolor="#f7fbf7",
            edgecolor="#5d8a5d",
            linewidth=1.4,
            alpha=0.55,
        )
    )
    ax.text(0.5, 15.25, "PHASE 1 — Data Foundation", fontsize=10, fontweight="bold", color="#2d5a2d")

    ax.add_patch(
        FancyBboxPatch(
            (0.3, 0.3),
            9.4,
            8.05,
            boxstyle="round,pad=0.02,rounding_size=0.05",
            facecolor="#f5f8fc",
            edgecolor="#3d6a9a",
            linewidth=1.4,
            alpha=0.55,
        )
    )
    ax.text(0.5, 8.05, "PHASE 2 — Intelligence Layer", fontsize=10, fontweight="bold", color="#2a4d73")

    w, h = 4.2, 0.68
    x = 2.9
    p1 = [
        (14.25, "OpenAlex API", "#d6eaf8"),
        (13.35, "Raw Research Dataset\n(data/raw/)", "#d6eaf8"),
        (12.45, "Preprocessing\n(clean + normalize)", "#d5f5e3"),
        (11.55, "Feature Extraction\n(concepts, keywords, OA fields)", "#d5f5e3"),
        (10.65, "Feature Engineering\n(final_dataset.csv)", "#d5f5e3"),
        (9.75, "R EDA + Python EDA", "#fcf3cf"),
        (8.85, "SQLite Database\n(database/researchpilot.db)", "#fdebd0"),
    ]
    coords = []
    for y, label, color in p1:
        cx, bot, top = box(ax, (x, y), w, h, label, fc=color, fontsize=8)
        coords.append((cx, bot, top))
    for i in range(len(coords) - 1):
        arrow(ax, coords[i][0], coords[i][1] - 0.02, coords[i + 1][0], coords[i + 1][2] + 0.02)

    p2 = [
        (7.15, "Feature Selection (M3)\nconsensus + scaling", "#d6eaf8"),
        (6.25, "ML Feature Matrix\n(train / val / test)", "#d6eaf8"),
        (5.35, "M4 — Multiple ML Models\n(14 algorithms, OA task)", "#d5f5e3"),
        (4.45, "M5 — Hyperparameter Tuning\n(RandomizedSearchCV)", "#d5f5e3"),
        (3.55, "M6 — Comparative Evaluation\n(metrics + visualizations)", "#fcf3cf"),
    ]
    coords2 = []
    for y, label, color in p2:
        cx, bot, top = box(ax, (x, y), w, h, label, fc=color, fontsize=8)
        coords2.append((cx, bot, top))

    arrow(ax, coords[-1][0], coords[-1][1] - 0.02, coords2[0][0], coords2[0][2] + 0.02)
    for i in range(len(coords2) - 1):
        arrow(ax, coords2[i][0], coords2[i][1] - 0.02, coords2[i + 1][0], coords2[i + 1][2] + 0.02)

    lx, lb, lt = box(
        ax,
        (0.7, 1.75),
        3.6,
        0.95,
        "OA Classification\n(M4/M5/M6 · AdaBoost champion)",
        fc="#aed6f1",
        fontsize=8,
        bold=True,
    )
    rx, rb, rt = box(
        ax,
        (5.7, 1.75),
        3.6,
        0.95,
        "Impact-Tier Prediction (M7)\n+ Topic Clustering (k=8)",
        fc="#a9dfbf",
        fontsize=8,
        bold=True,
    )
    ix, ib, it = box(
        ax,
        (2.9, 0.5),
        4.2,
        0.7,
        "Research Insights\n(relative impact · exploratory topics)",
        fc="#f9e79f",
        fontsize=8,
        bold=True,
    )

    arrow(ax, coords2[-1][0], coords2[-1][1] - 0.02, lx, lt + 0.02)
    arrow(ax, coords2[-1][0], coords2[-1][1] - 0.02, rx, rt + 0.02)
    arrow(ax, lx, lb - 0.02, ix - 0.5, it + 0.02)
    arrow(ax, rx, rb - 0.02, ix + 0.5, it + 0.02)

    ax.text(
        5.0,
        15.7,
        "OpenAlex → Dataset → Features → Database → ML → Tuning → Comparison → Impact / Topics",
        ha="center",
        fontsize=7.5,
        style="italic",
        color="#555555",
    )

    fig.tight_layout()
    for d in OUT_DIRS:
        out = d / "researchpilot_final_architecture.png"
        fig.savefig(out, dpi=160, bbox_inches="tight", facecolor="white")
        print("Wrote", out)
    plt.close(fig)


if __name__ == "__main__":
    main()
