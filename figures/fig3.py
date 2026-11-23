# Figure 3: What does not transfer is a stable, person-specific signature.
#   A  decoding within each person vs k: shared dimensions only, residual, full pattern   (script 04)
#   B  similarity of English-minus-Chinese signatures: same person / different people     (script 08)
#   C  identification of individuals from their other run, with the 77 x 77 matrices
#      as insets at k = 0 and at the best k                                               (script 08)
# k = 0 (nothing removed) is drawn separately, after an axis break.
#
# Usage:  python figures/fig3.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from style import RESULTS, SHARED, RESIDUAL, GRAY, DARK, K_LIST, panel, log_k_axis, save

summary = pd.read_csv(RESULTS / "shared_dimensions_summary.csv").set_index("representation")
signatures = pd.read_csv(RESULTS / "signatures_identification_summary.csv").set_index("k_removed")
matrices = np.load(RESULTS / "signature_similarity_matrices.npz")

fig = plt.figure(figsize=(7.2, 5.6))
axA = fig.add_axes([0.09, 0.60, 0.36, 0.32])
axB = fig.add_axes([0.60, 0.60, 0.37, 0.32])
axC = fig.add_axes([0.09, 0.08, 0.88, 0.36])


def pp(x):
    return 100 * (np.asarray(x) - 0.5)


def x_of(k):
    """Position on a log axis, with k = 0 placed to the left of k = 1 after a break."""
    return np.where(np.asarray(k) == 0, 10 ** -0.55, k)


def axis_with_break(ax, ks):
    ax.set_xscale("log")
    ax.set_xticks(x_of(ks))
    ax.set_xticklabels([str(k) for k in ks])
    ax.minorticks_off()
    ax.set_xlim(10 ** -0.75, 270)
    ax.text(10 ** -0.25, 0, "//", transform=ax.get_xaxis_transform(), ha="center", va="center",
            fontsize=7, color=DARK, backgroundcolor="white")


# ---- A: decoding within each person ---------------------------------------------
full = summary.loc["full"]
axA.axhspan(pp(full["within_ci_low"]), pp(full["within_ci_high"]), color="#e6e6e6", lw=0, zorder=0)
axA.axhline(pp(full["within"]), color=DARK, lw=1, zorder=1)
axA.axhline(0, color=GRAY, ls=":", lw=0.7)
for rep, color in [("shared", SHARED), ("residual", RESIDUAL)]:
    rows = summary.loc[[f"{rep}_{k}" for k in K_LIST]]
    m = pp(rows["within"])
    axA.errorbar(K_LIST, m, yerr=[m - pp(rows["within_ci_low"]), pp(rows["within_ci_high"]) - m],
                 color=color, fmt="o--", ms=2.5, lw=0.8, capsize=1.5)
log_k_axis(axA)
axA.set_ylim(-1, max(pp(summary["within_ci_high"])) + 2.5)
axA.set_xlabel("Number of shared dimensions, k")
axA.set_ylabel("Balanced accuracy above chance\n(percentage points)")
axA.text(0.03, 0.95, "residual", color=RESIDUAL, transform=axA.transAxes, fontweight="bold", fontsize=6, va="top")
axA.text(0.55, 0.22, "shared dimensions only", color=SHARED, transform=axA.transAxes, ha="center", fontweight="bold", fontsize=6)
axA.text(0.02, pp(full["within_ci_low"]) - 0.3, "full pattern", fontsize=5.5, va="top",
         transform=axA.get_yaxis_transform())

# ---- B: signature similarity ------------------------------------------------------
ks = list(signatures.index)
for col, color in [("within", RESIDUAL), ("between", GRAY)]:
    m = signatures[f"{col}_similarity"].to_numpy()
    err = [m - signatures[f"{col}_ci_low"].to_numpy(), signatures[f"{col}_ci_high"].to_numpy() - m]
    axB.errorbar(x_of(ks), m, yerr=err, fmt="o", ms=2.5, color=color, lw=0.6, capsize=0)
    after = signatures.index > 0                       # connect only k >= 1
    axB.plot(x_of(np.array(ks)[after]), m[after], "--", color=color, lw=0.8)
axB.axhline(0, color=GRAY, ls=":", lw=0.7)
axis_with_break(axB, ks)
axB.set_xlabel("Shared dimensions removed, k")
axB.set_ylabel("Similarity of language-state\nsignatures (Pearson r)")
axB.text(0.97, 0.95, "same person, other run", color=RESIDUAL, transform=axB.transAxes, ha="right", va="top", fontweight="bold", fontsize=6)
axB.text(0.40, 0.45, "different people", color=GRAY, transform=axB.transAxes, ha="center", fontweight="bold", fontsize=6)

# ---- C: identification ----------------------------------------------------------------
ident = 100 * signatures["identification"]
after = signatures.index > 0
axC.plot(x_of(np.array(ks)[after]), ident[after], "o--", color=RESIDUAL, ms=2.5, lw=0.8)
axC.plot(x_of(0), ident[0], "o", color=RESIDUAL, ms=2.5)
n_people = matrices[f"k{ks[0]}"].shape[0]
axC.axhline(100 / n_people, color=GRAY, ls=":", lw=0.7)
axC.text(1, 100 / n_people + 1.5, "chance", transform=axC.get_yaxis_transform(), ha="right", color=GRAY, fontsize=5.5)
axis_with_break(axC, ks)
axC.set_ylim(0, 100)
axC.set_xlabel("Shared dimensions removed, k")
axC.set_ylabel("Participants identified from\ntheir other run (%)")

best_k = int(ident[after].idxmax())
for k in (0, best_k):
    axC.plot(x_of(k), ident[k], "s", ms=8, mfc="none", mec=DARK, mew=0.7)
    axC.text(x_of(k), ident[k] - 5, f"{ident[k]:.0f}%", ha="center", va="top", color=RESIDUAL, fontweight="bold", fontsize=6.5)

# insets: the 77 x 77 matrices (rows run 1, columns run 2), same colour scale
inset_places = {0: [0.10, 0.56, 0.17, 0.42], best_k: [0.70, 0.07, 0.17, 0.42]}
for k, place in inset_places.items():
    ins = axC.inset_axes(place)
    im = ins.imshow(matrices[f"k{k}"], cmap="RdBu_r", vmin=-0.3, vmax=0.3, interpolation="nearest")
    ins.set_xticks([])
    ins.set_yticks([])
    ins.set_xlabel("run 2", fontsize=5.5, labelpad=1)
    ins.set_ylabel("run 1", fontsize=5.5, labelpad=1)
    if k == 0:
        ins.set_title("full cortical pattern", fontsize=5.5, pad=2)
    else:
        ins.text(1.05, 0.5, f"{k} shared\ndimensions removed", transform=ins.transAxes, fontsize=5.5, va="center")
    # line from the boxed point to its inset
    corner = (place[0] + place[2] / 2, place[1]) if k == 0 else (place[0] + place[2] / 2, place[1] + place[3])
    axC.annotate("", xy=(x_of(k), ident[k]), xycoords="data", xytext=corner, textcoords="axes fraction",
                 arrowprops=dict(arrowstyle="-", color=DARK, lw=0.6))
    if k == 0:
        bar = fig.colorbar(im, ax=ins, fraction=0.05, pad=0.04, ticks=[-0.3, 0, 0.3])
        bar.ax.tick_params(labelsize=5)
        bar.set_label("r", fontsize=5.5)

panel(fig, axA, "A", "Decoding within each person", dy=0.03)
panel(fig, axB, "B", "Signature similarity", dx=-0.08, dy=0.03)
panel(fig, axC, "C", "Removing shared dimensions reveals each person’s signature", dy=0.03)
save(fig, "fig3")
