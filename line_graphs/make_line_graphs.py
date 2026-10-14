"""Line graphs of intra/inter collaboration metrics by 5-year block (college and department level)."""
import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA_DIR = Path("topics_NC/topics/line_graph")
OUT_DIR = Path("line_graphs")
METRICS = ["SNCI", "PNCI", "FNCI", "CIR"]
COLORS = {"SNCI": "#5bc0d8", "PNCI": "#d9534f", "FNCI": "#2e8b3d", "CIR": "#6f8a96"}
MARKERS = {"intra": "o", "inter": "D"}

LEVELS = {
    "college": dict(
        file="table8_college_metrics_summary_by_block.csv",
        title="College Level Collaboration Metrics (2006-2025)",
        bg="#ffffff", border="#3a7a85"),
    "department": dict(
        file="table9_department_metrics_summary_by_block.csv",
        title="Department Level Collaboration Metrics (2006-2025)",
        bg="#fcf1d0", border="#b0306e"),
}


def load(path):
    blocks, data = [], {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            if row["block"] not in blocks:
                blocks.append(row["block"])
            for m in METRICS:
                data.setdefault((m, row["colab_type"]), {})[row["block"]] = float(row[m])
    return blocks, data


def plot(level, cfg, out_dir, data_dir):
    blocks, data = load(data_dir / cfg["file"])
    fig, ax = plt.subplots(figsize=(10, 6.2), facecolor=cfg["bg"])
    ax.set_facecolor(cfg["bg"])
    for m in METRICS:
        for kind in ("intra", "inter"):
            ys = [data[(m, kind)][b] for b in blocks]
            ax.plot(blocks, ys, color=COLORS[m], marker=MARKERS[kind], markersize=5,
                    linewidth=1.8, linestyle="-" if kind == "intra" else "--",
                    label=f"{m}-{kind}")
    ax.set_title(cfg["title"], fontsize=16, fontweight="bold", color="#6b6b6b", pad=34)
    ax.text(0.5, 1.04, "Intra-collaboration shows stronger performance across all metrics",
            transform=ax.transAxes, ha="center", fontsize=10, color="#7a7a7a")
    ax.set_xlabel("Year Range", color="#7a7a7a")
    ax.set_ylabel("Metric Value", color="#7a7a7a")
    ax.grid(color="#00000014", linewidth=0.8)
    ax.axhline(0, color="#888888", linewidth=1)
    ax.set_ylim(bottom=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(colors="#7a7a7a", length=0)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), frameon=False, fontsize=9,
              labelcolor="#7a7a7a")
    fig.patches.append(plt.Rectangle((0, 0), 1, 1, transform=fig.transFigure, fill=False,
                                     edgecolor=cfg["border"], linewidth=5))
    fig.tight_layout(rect=(0.01, 0.01, 0.99, 0.99))
    out_dir.mkdir(exist_ok=True)
    out = out_dir / f"{level}_collaboration_metrics.png"
    fig.savefig(out, dpi=200, facecolor=cfg["bg"])
    plt.close(fig)
    print("saved", out)


if __name__ == "__main__":
    data_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else DATA_DIR
    out_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else OUT_DIR
    for level, cfg in LEVELS.items():
        plot(level, cfg, out_dir, data_dir)
