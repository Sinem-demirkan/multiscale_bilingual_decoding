# Shared look and small helpers for the figure scripts (fig1.py, fig2.py, fig3.py).

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "results"            # written by scripts/01-10
OUT = REPO / "figures" / "output"     # figures are saved here
ASSETS = REPO / "figures" / "assets"  # hand-made schematics (fig1A.png, fig2A.png)

# colours used in every figure
SHARED = "#1f5fa8"     # shared dimensions
RESIDUAL = "#e07b2f"   # residual / same person
GRAY = "#8c8c8c"       # participants, different people
DARK = "#222222"       # means, full pattern
YEO = "#5f6b78"        # Yeo network bars
LOCALIZER = "#8e6bb5"  # language / multiple-demand bars

K_LIST = [1, 5, 10, 25, 50, 100, 200]

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 7,
    "axes.titlesize": 7.5,
    "axes.labelsize": 7,
    "xtick.labelsize": 6.5,
    "ytick.labelsize": 6.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "pdf.fonttype": 42,     # editable text in Illustrator
    "svg.fonttype": "none",
})


def mean_ci(values, n_boot=5000, seed=42):
    """Mean and 95% CI from resampling participants (as in the analysis scripts)."""
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    boot = values[rng.integers(0, len(values), (n_boot, len(values)))].mean(axis=1)
    return values.mean(), *np.percentile(boot, [2.5, 97.5])


def panel(fig, ax, letter, title, dx=-0.06, dy=0.03):
    """Bold panel letter and title above the top-left corner of an axis."""
    box = ax.get_position()
    fig.text(box.x0 + dx, box.y1 + dy, letter, fontsize=9, fontweight="bold", va="bottom")
    fig.text(box.x0, box.y1 + dy, title, fontsize=7.5, fontweight="bold", va="bottom")


def image_slot(ax, name, label):
    """Show a hand-made schematic from figures/assets/, or a grey placeholder if it is missing."""
    ax.set_axis_off()
    path = ASSETS / name
    if path.exists():
        ax.imshow(plt.imread(path))
    else:
        ax.add_patch(plt.Rectangle((0, 0), 1, 1, transform=ax.transAxes, fc="#f2f2f2", ec="#bbbbbb", lw=0.6))
        ax.text(0.5, 0.5, f"{label}\n(put the image in figures/assets/{name})",
                ha="center", va="center", transform=ax.transAxes, color="#777777", fontsize=7)


def log_k_axis(ax, ks=K_LIST):
    ax.set_xscale("log")
    ax.set_xticks(ks)
    ax.set_xticklabels([str(k) for k in ks])
    ax.minorticks_off()


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=300, bbox_inches="tight", pad_inches=0.03)
    print("saved", OUT / f"{name}.pdf")
