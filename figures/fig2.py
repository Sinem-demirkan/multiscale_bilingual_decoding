# Figure 2: Shared dimensions carry transfer between people, and language history
# is expressed along them.
#   A  schematic (image from figures/assets/fig2A.png)
#   B  transfer between people vs k: shared dimensions only, residual, full pattern   (script 04)
#   C  transfer with each of the first 25 shared dimensions alone                     (script 07)
#   D  prediction of age of English acquisition from learner transfer vs k, inset at k = 10   (script 09)
#   E  loading maps of PC1 and of the 3 dimensions carrying the most transfer, and their
#      network profiles (radar plots; filled dots = above the spatial null)          (scripts 07, 10)
#
# Usage:  python figures/fig2.py

import nibabel as nib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from nilearn import image, plotting, surface
from nilearn.datasets import fetch_atlas_schaefer_2018, load_fsaverage, load_fsaverage_data

from style import RESULTS, SHARED, RESIDUAL, GRAY, DARK, K_LIST, panel, image_slot, log_k_axis, save

summary = pd.read_csv(RESULTS / "shared_dimensions_summary.csv").set_index("representation")
single = pd.read_csv(RESULTS / "single_dimension_transfer_summary.csv")
aoa = pd.read_csv(RESULTS / "aoa_prediction_AoA.csv").set_index("predictor")
aoa_pred = pd.read_csv(RESULTS / "aoa_predictions_AoA.csv")
loadings = pd.read_csv(RESULTS / "pc_loadings_all_participants.csv", index_col="dimension")
enrichment = pd.read_csv(RESULTS / "pc_network_enrichment.csv")

fig = plt.figure(figsize=(7.2, 7.6))
axA = fig.add_axes([0.06, 0.82, 0.92, 0.14])
axB = fig.add_axes([0.09, 0.52, 0.23, 0.21])
axC = fig.add_axes([0.42, 0.52, 0.23, 0.21])
axD = fig.add_axes([0.76, 0.52, 0.22, 0.21])


def pp(x):   # accuracy -> percentage points above chance
    return 100 * (np.asarray(x) - 0.5)


def full_pattern_band(ax, col="transfer"):
    full = summary.loc["full"]
    ax.axhspan(pp(full[f"{col}_ci_low"]), pp(full[f"{col}_ci_high"]), color="#e6e6e6", lw=0, zorder=0)
    ax.axhline(pp(full[col]), color=DARK, lw=1, zorder=1)
    ax.axhline(0, color=GRAY, ls=":", lw=0.7)


# ---- A: schematic ----------------------------------------------------------------
image_slot(axA, "fig2A.png", "Fig. 2A schematic")

# ---- B: transfer vs k --------------------------------------------------------------
full_pattern_band(axB)
for rep, color in [("shared", SHARED), ("residual", RESIDUAL)]:
    rows = summary.loc[[f"{rep}_{k}" for k in K_LIST]]
    m = pp(rows["transfer"])
    axB.errorbar(K_LIST, m, yerr=[m - pp(rows["transfer_ci_low"]), pp(rows["transfer_ci_high"]) - m],
                 color=color, fmt="o--", ms=2.5, lw=0.8, capsize=1.5)
log_k_axis(axB)
top = max(pp(summary["transfer_ci_high"])) + 4
axB.set_ylim(-1, top)
axB.set_xlabel("Number of shared dimensions, k (cumulative)")
axB.set_ylabel("Balanced accuracy above chance\n(percentage points)")
axB.text(0.97, 0.94, "shared dimensions only", color=SHARED, transform=axB.transAxes, ha="right", fontweight="bold", fontsize=6)
axB.text(40, pp(summary.loc["residual_50", "transfer_ci_high"]) + 0.8, "residual", color=RESIDUAL,
         ha="left", va="bottom", fontweight="bold", fontsize=6)
axB.text(0.02, pp(summary.loc["full", "transfer_ci_high"]) + 0.3, "full pattern", fontsize=5.5,
         transform=axB.get_yaxis_transform())

# ---- C: each dimension alone ------------------------------------------------------------
full_pattern_band(axC)
m = pp(single["accuracy"])
colors = [SHARED if sig else "#b7c9e2" for sig in single["above_chance"]]
axC.bar(single["dimension"], m, width=0.6, color=colors)
axC.errorbar(single["dimension"], m, yerr=[m - pp(single["ci_low"]), pp(single["ci_high"]) - m],
             fmt="none", color=DARK, lw=0.5, capsize=0)
for d, top, sig in zip(single["dimension"], pp(single["ci_high"]), single["above_chance"]):
    if sig:
        axC.text(d, top + 0.2, "*", ha="center", fontsize=7)
axC.set_xticks([1, 5, 10, 15, 20, 25])
axC.set_xlabel("Shared dimension tested alone (PC number)")
axC.set_ylabel("Balanced accuracy above chance\n(percentage points)")
axC.text(0.02, pp(summary.loc["full", "transfer_ci_high"]) + 0.3, "full pattern", fontsize=5.5,
         transform=axC.get_yaxis_transform())
axC.set_ylim(axB.get_ylim())

# ---- D: age of English acquisition ------------------------------------------------------
for rep, color, label in [("shared", SHARED, "shared dimensions"), ("residual", RESIDUAL, "residual")]:
    r2 = [aoa.loc[f"learner_{rep}_{k}", "cv_r2"] for k in K_LIST]
    axD.plot(K_LIST, r2, "o--", color=color, ms=2.5, lw=0.8)
best = aoa.loc["learner_shared_10", "cv_r2"]
axD.plot(10, best, "o", ms=8, mfc="none", mec=SHARED, mew=1)
axD.axhline(0, color=GRAY, ls=":", lw=0.7)
all_r2 = aoa.loc[[f"learner_{rep}_{k}" for rep in ("shared", "residual") for k in K_LIST], "cv_r2"]
axD.set_ylim(min(all_r2.min(), 0) - 0.03, 2.3 * all_r2.max())
log_k_axis(axD)
axD.set_xlabel("Number of shared dimensions, k")
axD.set_ylabel("Prediction of age of English\nacquisition (cross-validated R²)")
axD.text(0.03, 0.84, "shared\ndimensions", color=SHARED, transform=axD.transAxes, fontweight="bold", fontsize=6)
axD.text(0.97, 0.16, "residual", color=RESIDUAL, transform=axD.transAxes, ha="right", fontweight="bold", fontsize=6)

inset = axD.inset_axes([0.66, 0.66, 0.32, 0.32])
inset.scatter(aoa_pred["observed"], aoa_pred["learner_shared_10"], s=2, color=SHARED, lw=0)
lims = [aoa_pred["observed"].min() - 1, aoa_pred["observed"].max() + 1]
inset.plot(lims, lims, ls=":", color=GRAY, lw=0.6)
inset.set_xlim(lims)
inset.set_ylim(lims)
inset.tick_params(labelsize=5, length=2)
inset.set_xlabel("Observed AoA", fontsize=5, labelpad=1)
inset.set_ylabel("Predicted", fontsize=5, labelpad=1)
inset.text(0.05, 0.92, f"R² = {best:.2f}", transform=inset.transAxes, fontsize=5)

# ---- E: loading maps and network profiles -----------------------------------------------
# PC1, then the three dimensions that carried the most transfer
top3 = single[single["dimension"] != 1].nlargest(3, "accuracy")["dimension"].tolist()
shown = [1] + sorted(top3)
transfer_pp = dict(zip(single["dimension"], pp(single["accuracy"])))

# parcel values -> volume -> left-hemisphere surface
atlas = fetch_atlas_schaefer_2018(n_rois=800, yeo_networks=7, resolution_mm=2)
atlas_img = image.load_img(atlas.maps)
atlas_ids = np.asarray(atlas_img.dataobj).astype(int)
fsaverage = load_fsaverage("fsaverage5")
sulcal = load_fsaverage_data(mesh="fsaverage5", data_type="sulcal")
vmax = 0.08


def surface_map(values):
    volume = np.zeros(atlas_ids.shape)
    volume[atlas_ids > 0] = values[atlas_ids[atlas_ids > 0] - 1]
    img = nib.Nifti1Image(volume, atlas_img.affine)
    return surface.SurfaceImage.from_volume(mesh=fsaverage["pial"], volume_img=img,
                                            interpolation="nearest_most_frequent")


order = ["Vis", "SomMot", "DorsAttn", "SalVentAttn", "Limbic", "Cont", "Default", "Lang", "MD"]
short = {"Vis": "Vis", "SomMot": "SM", "DorsAttn": "DA", "SalVentAttn": "VA", "Limbic": "Lim",
         "Cont": "FP", "Default": "DMN", "Lang": "Lang", "MD": "MD"}
angles = np.linspace(0, 2 * np.pi, len(order), endpoint=False)
r_max = 1.1 * enrichment.loc[enrichment["dimension"].isin(shown), "enrichment"].max()

width = 0.19
for i, d in enumerate(shown):
    x0 = 0.07 + i * 0.205
    # brain
    ax_brain = fig.add_axes([x0, 0.24, width, 0.16], projection="3d")
    plotting.plot_surf_stat_map(fsaverage["inflated"], surface_map(loadings.loc[d].to_numpy()),
                                hemi="left", view="lateral", bg_map=sulcal, cmap="RdBu_r",
                                vmax=vmax, colorbar=False, axes=ax_brain, figure=fig)
    ax_brain.set_title(f"PC{d}   transfer {transfer_pp[d]:+.1f}", fontsize=6.5, fontweight="bold", pad=-2)

    # radar
    ax_radar = fig.add_axes([x0 + 0.03, 0.03, width - 0.06, 0.17], projection="polar")
    g = enrichment[enrichment["dimension"] == d].set_index("network").loc[order]
    values = g["enrichment"].to_numpy()
    closed = np.append(angles, angles[0])
    ax_radar.plot(closed, np.append(values, values[0]), color=SHARED, lw=0.9)
    ax_radar.fill(closed, np.append(values, values[0]), color=SHARED, alpha=0.15, lw=0)
    sig = (g["q_bh"] < 0.05).to_numpy()
    ax_radar.scatter(angles[sig], values[sig], s=8, color=SHARED, zorder=3)
    ax_radar.scatter(angles[~sig], values[~sig], s=8, facecolor="white", edgecolor=SHARED, lw=0.6, zorder=3)
    ax_radar.plot(np.linspace(0, 2 * np.pi, 100), np.ones(100), ls=":", color=GRAY, lw=0.7)   # no preference
    ax_radar.set_theta_zero_location("N")
    ax_radar.set_theta_direction(-1)
    ax_radar.set_xticks(angles)
    ax_radar.set_xticklabels([short[n] for n in order], fontsize=5.5)
    ax_radar.tick_params(axis="x", pad=-3)
    ax_radar.set_yticklabels([])
    ax_radar.set_ylim(0, r_max)
    ax_radar.grid(lw=0.3, color="#cccccc")
    ax_radar.spines["polar"].set_visible(False)

# shared colour bar for the maps
cax = fig.add_axes([0.905, 0.27, 0.01, 0.11])
bar = fig.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(-vmax, vmax), cmap="RdBu_r"), cax=cax)
bar.set_ticks([-vmax, 0, vmax])
bar.ax.tick_params(labelsize=5.5)
bar.set_label("Loading\n(+ English > Chinese)", fontsize=5.5)
fig.text(0.885, 0.11, "● q < 0.05\n∙∙∙ no network\n     preference", fontsize=5.5, color=SHARED, va="center")

panel(fig, axA, "A", "", dy=-0.01)
panel(fig, axB, "B", "Transfer between people", dy=0.03)
panel(fig, axC, "C", "Which dimensions carry transfer", dy=0.03)
panel(fig, axD, "D", "Language history", dy=0.03)
fig.text(0.02, 0.425, "E", fontsize=9, fontweight="bold")
fig.text(0.09, 0.425, "Where the transfer-carrying dimensions lie, and their network profiles", fontsize=7.5, fontweight="bold")
save(fig, "fig2")
