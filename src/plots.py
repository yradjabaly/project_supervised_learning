"""Bar charts shared by several steps: one panel per metric, vertical bars."""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgb
from matplotlib.patches import Patch

from src.config import FIG_DIR


def lighter(color, amount=0.6):
    """Mix a colour with white. Used for the 'before' bars."""
    return tuple(c + (1 - c) * amount for c in to_rgb(color))


def plot_metric_bars(tables, metrics, colors, filename, best=None, ylim=None):
    """One panel per metric, one bar per group (model, calibration or sex).

    tables:  {version: DataFrame} with the groups as index and the metrics as columns.
             With two versions (e.g. default and tuned), the first one is drawn in a
             light shade and the second one in the full colour of the group.
    metrics: {column: panel title}.
    best:    {column: "max" or "min"}. The best group of the last version gets a dark
             edge and a bold label, to guide the eye.
    ylim:    common y-axis limits (e.g. (0, 1.1)); by default each panel has its own.
    """
    best = best or {}
    versions = list(tables)
    last = tables[versions[-1]]
    groups = list(last.index)
    x = np.arange(len(groups))
    width = 0.8 / len(versions)

    fig, axes = plt.subplots(1, len(metrics), figsize=(2.3 * len(metrics), 3),
                             squeeze=False)
    for ax, (metric, title) in zip(axes[0], metrics.items()):
        winner = None
        if metric in best:
            winner = last[metric].idxmax() if best[metric] == "max" else last[metric].idxmin()

        for k, version in enumerate(versions):
            is_last = k == len(versions) - 1
            values = tables[version].loc[groups, metric]
            shades = [colors[g] if is_last else lighter(colors[g]) for g in groups]
            bars = ax.bar(x + (k - (len(versions) - 1) / 2) * width, values,
                          width * 0.95, color=shades)
            for bar, group, value in zip(bars, groups, values):
                is_winner = is_last and group == winner
                if is_winner:
                    bar.set_edgecolor("black")
                    bar.set_linewidth(1.2)
                ax.annotate(f"{value:.3f}" if value < 0.1 else f"{value:.2f}",
                            (bar.get_x() + bar.get_width() / 2, value),
                            xytext=(0, 2), textcoords="offset points", ha="center",
                            va="bottom", fontsize=7,
                            fontweight="bold" if is_winner else "normal")

        top = max(tables[v][metric].max() for v in versions)
        ax.set_ylim(ylim if ylim else (0, top * 1.25))
        ax.set_title(title, fontsize=10)
        ax.set_xticks(x, [g.replace(" ", "\n", 1) for g in groups], fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)

    if len(versions) == 2:
        handles = [Patch(color=lighter("#555555"), label=versions[0]),
                   Patch(color="#555555", label=versions[1])]
        fig.legend(handles=handles, loc="upper center", ncol=2, fontsize=8, frameon=False)
        fig.tight_layout(rect=[0, 0, 1, 0.92])
    else:
        fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/{filename}")
    plt.close(fig)