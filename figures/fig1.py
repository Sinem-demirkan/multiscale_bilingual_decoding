# Figure 1: Language state is expressed in a cortex-wide pattern.
#   A  schematic (image from figures/assets/fig1A.png)
#   B  decoding within each person: all trials, repeat, switch, switch status   (scripts 02, 02b)
#   C  single parcels distinguishing the languages                              (script 06)
#   D  network contributions                                                    (script 05)
#
# Usage:  python figures/fig1.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from style import RESULTS, GRAY, DARK, YEO, LOCALIZER, mean_ci, panel, image_slot, save

whole = pd.read_csv(RESULTS / "whole_cortex_decoding.csv")
switch = pd.read_csv(RESULTS / "switch_decoding.csv")
parcels = pd.read_csv(RESULTS / "parcelwise_group.csv")
networks = pd.read_csv(RESULTS / "network_importance_language_yeo7.csv")

fig = plt.figure(figsize=(7.2, 4.4))
axA = fig.add_axes([0.06, 0.66, 0.92, 0.26])
axB = fig.add_axes([0.09, 0.10, 0.27, 0.42])
axC = fig.add_axes([0.45, 0.10, 0.15, 0.42])
axD = fig.add_axes([0.72, 0.10, 0.25, 0.42])

# ---- A: schematic --------------------------------------------------------------
image_slot(axA, "fig1A.png", "Fig. 1A schematic")

# ---- B: decoding within each person --------------------------------------------
columns = [("All trials", whole["balanced_accuracy"]), ("Repeat", switch["repeat"]),
           ("Switch", switch["switch"]), ("Switch\nstatus", switch["switch_status"])]
positions = [0, 1, 2, 3.4]
rng = np.random.default_rng(0)
for x, (name, values) in zip(positions, columns):
    values = 100 * values.to_numpy()
    light = name.startswith("Switch\n")
    axB.scatter(x + rng.uniform(-0.18, 0.18, len(values)), values, s=4, lw=0,
                color="#cccccc" if light else GRAY, alpha=0.9, zorder=1)
    m, low, high = mean_ci(values)
    axB.errorbar(x, m, yerr=[[m - low], [high - m]], fmt="o", ms=4, color=DARK, capsize=2, lw=1, zorder=3)
axB.axhline(50, ls=":", lw=0.7, color=GRAY)
axB.axvline(2.7, lw=0.5, color="#dddddd")
axB.set_xticks(positions)
axB.set_xticklabels([name for name, _ in columns])
axB.set_xlim(-0.6, 4.0)
axB.set_ylabel("Balanced accuracy (%)")
axB.text(1, 1.02, "Chinese vs English", ha="center", transform=axB.get_xaxis_transform(), fontweight="bold", fontsize=6.2)
axB.text(3.4, 1.02, "Switch vs\nrepeat", ha="center", transform=axB.get_xaxis_transform(), color=GRAY, fontweight="bold", fontsize=6.2)
axB.text(3.9, 0.03, f"n = {len(whole)}", ha="right", transform=axB.get_xaxis_transform(), color=GRAY, fontsize=6)

# ---- C: single parcels -----------------------------------------------------------
labels = {"voxels": "Voxels", "demeaned": "Voxels\n− mean", "mean": "Mean"}
shades = {"voxels": "#5c6670", "demeaned": "#8a939b", "mean": "#c9ced3"}
for x, rep in enumerate(labels):
    n_sig = int(parcels.loc[parcels["representation"] == rep, "significant"].sum())
    axC.bar(x, n_sig, width=0.6, color=shades[rep])
    axC.text(x, n_sig + 3, str(n_sig), ha="center", va="bottom", fontweight="bold")
axC.set_ylim(0, 1.18 * parcels.groupby("representation")["significant"].sum().max())
axC.set_xticks(range(3))
axC.set_xticklabels(labels.values())
axC.set_ylabel("Parcels distinguishing\nChinese from English (of 800)")
axC.text(0.5, 1.03, "each parcel decoded separately", ha="center", transform=axC.transAxes, color=GRAY, fontsize=6)

# ---- D: network contributions ------------------------------------------------------
short = {"SomMot": "SM", "Default": "DMN", "Cont": "FP", "SalVentAttn": "VA",
         "DorsAttn": "DA", "Vis": "Vis", "Limbic": "Lim"}
yeo = sorted(short, key=lambda net: networks[net].mean(), reverse=True)   # largest on top
rows = [(net, short[net], YEO) for net in yeo] + [("Lang", "Lang", LOCALIZER), ("MD", "MD", LOCALIZER)]
y_positions = list(range(len(yeo))) + [len(yeo) + 0.6, len(yeo) + 1.6]
for y, (col, name, color) in zip(y_positions, rows):
    m, low, high = mean_ci(100 * networks[col].to_numpy())
    axD.barh(y, m, height=0.6, color=color)
    axD.errorbar(m, y, xerr=[[m - low], [high - m]], fmt="none", color=DARK, capsize=2, lw=0.8)
axD.set_yticks(y_positions)
axD.set_yticklabels([name for _, name, _ in rows])
axD.invert_yaxis()
axD.axvline(0, color=DARK, lw=0.6)
axD.spines["left"].set_visible(False)
axD.tick_params(axis="y", length=0)
axD.set_xlabel("Accuracy drop when network\nis shuffled (percentage points)")
axD.text(1.0, 3, "Yeo\nnetworks", transform=axD.get_yaxis_transform(), ha="right", color=YEO, fontweight="bold", fontsize=6.5)
axD.text(1.0, len(yeo) + 1.1, "Localizer\nnetworks", transform=axD.get_yaxis_transform(), ha="right",
         color=LOCALIZER, fontweight="bold", fontsize=6.5)

panel(fig, axA, "A", "", dy=-0.01)
panel(fig, axB, "B", "Decoding within each person", dy=0.06)
panel(fig, axC, "C", "Single parcels", dx=-0.09, dy=0.06)
panel(fig, axD, "D", "Network contributions", dx=-0.07, dy=0.06)
save(fig, "fig1")
