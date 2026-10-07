from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize
from nilearn import datasets, image, plotting, surface
from nilearn.image import new_img_like
from sklearn.preprocessing import StandardScaler


N_PCS = 25
RENDER_PCS = [1, 5, 10, 12]
N_ROIS = 800
YEO_NETWORKS = 7
RESOLUTION_MM = 2

ROOT = Path(__file__).resolve().parents[2]
OUT_ROOT = ROOT / "paper_version" / "cross_subject_language_transfer"
OUT = OUT_ROOT / "figure2_panel_d_shared_pc_maps_networks"

FEATURES = (
    ROOT
    / "paper_version"
    / "whole_cortex_parcel_mean_aoa_prediction"
    / "whole_cortex_parcel_mean_aoa_trialwise_features.csv"
)
TRIAL_LABELS = (
    ROOT
    / "paper_version"
    / "whole_cortex_item_posthoc"
    / "whole_cortex_trialwise_predictions.csv"
)
FULL_SAMPLE_LOADINGS = OUT_ROOT / "shared_trialwise_pca_component_loadings_full_sample_pc200.csv"
YEO_LABELS = ROOT / "mvpa_schaefer800" / "latest_robustness_controls" / "schaefer800_labels.csv"
LIPKIN_LANGUAGE = (
    ROOT
    / "mvpa_schaefer800"
    / "latest_robustness_controls"
    / "lipkin_language_schaefer800_parcel_sets.csv"
)
LIPKIN_MD = (
    ROOT
    / "mvpa_schaefer800"
    / "latest_robustness_controls"
    / "lipkin_md_schaefer800_parcel_sets.csv"
)


def parcel_columns(df: pd.DataFrame) -> list[str]:
    return [col for col in df.columns if col.startswith("parcel_")]


def load_zscored_features() -> tuple[pd.DataFrame, list[str]]:
    features = pd.read_csv(FEATURES)
    labels = pd.read_csv(TRIAL_LABELS)

    parcel_cols = parcel_columns(features)
    if len(parcel_cols) != N_ROIS:
        raise ValueError(f"Expected {N_ROIS} parcel columns, found {len(parcel_cols)}")

    labels = labels.sort_values(["subject", "run", "trial_in_run"]).copy()
    labels["trial_index"] = labels.groupby("subject").cumcount() + 1

    label_cols = ["subject", "trial_index", "language", "true_y", "true_label"]
    merged = features.merge(
        labels[label_cols],
        on=["subject", "trial_index"],
        how="left",
        validate="one_to_one",
    )
    if merged["language"].isna().any():
        raise ValueError("Some trial labels could not be aligned to the feature table.")

    blocks = []
    for subject, sub in merged.groupby("subject", sort=True):
        sub = sub.sort_values("trial_index").copy()
        sub.loc[:, parcel_cols] = StandardScaler().fit_transform(sub[parcel_cols].to_numpy(float))
        blocks.append(sub)

    return pd.concat(blocks, ignore_index=True), parcel_cols


def load_full_sample_loadings(parcel_cols: list[str]) -> tuple[pd.DataFrame, np.ndarray]:
    loadings = pd.read_csv(FULL_SAMPLE_LOADINGS)
    loadings = loadings.set_index("component").loc[[f"PC{i}" for i in range(1, N_PCS + 1)]]
    vectors = loadings[parcel_cols].to_numpy(float)
    return loadings.reset_index(), vectors


def cohens_d(x_a: np.ndarray, x_b: np.ndarray) -> float:
    n_a = len(x_a)
    n_b = len(x_b)
    if n_a < 2 or n_b < 2:
        return np.nan
    pooled_var = ((n_a - 1) * np.var(x_a, ddof=1) + (n_b - 1) * np.var(x_b, ddof=1)) / (
        n_a + n_b - 2
    )
    if pooled_var <= 0 or not np.isfinite(pooled_var):
        return np.nan
    return float((np.mean(x_a) - np.mean(x_b)) / np.sqrt(pooled_var))


def sign_align_to_l2_minus_l1(
    data: pd.DataFrame,
    parcel_cols: list[str],
    vectors: np.ndarray,
) -> tuple[np.ndarray, pd.DataFrame, pd.DataFrame]:
    aligned = vectors.copy()
    by_subject_rows = []
    summary_rows = []

    language_values = sorted(data["language"].dropna().unique())
    if language_values != ["L1", "L2"]:
        raise ValueError(f"Expected language labels L1/L2, found {language_values}")

    for pc_idx in range(N_PCS):
        pc_number = pc_idx + 1
        subject_ds = []

        for subject, sub in data.groupby("subject", sort=True):
            x = sub[parcel_cols].to_numpy(float)
            scores = x @ vectors[pc_idx]
            d_value = cohens_d(
                scores[sub["language"].to_numpy() == "L2"],
                scores[sub["language"].to_numpy() == "L1"],
            )
            subject_ds.append(d_value)
            by_subject_rows.append(
                {
                    "subject": subject,
                    "pc_number": pc_number,
                    "cohens_d_L2_minus_L1_before_alignment": d_value,
                }
            )

        mean_d = float(np.nanmean(subject_ds))
        sign_multiplier = -1 if mean_d < 0 else 1
        aligned[pc_idx] *= sign_multiplier

        summary_rows.append(
            {
                "pc_number": pc_number,
                "contrast": "L2_minus_L1",
                "mean_subject_cohens_d_before_alignment": mean_d,
                "mean_subject_cohens_d_after_alignment": mean_d * sign_multiplier,
                "sign_multiplier": sign_multiplier,
                "sign_flipped": bool(sign_multiplier == -1),
                "n_subjects": int(np.isfinite(subject_ds).sum()),
            }
        )

    return aligned, pd.DataFrame(summary_rows), pd.DataFrame(by_subject_rows)


def network_masks() -> list[dict[str, object]]:
    labels = pd.read_csv(YEO_LABELS).sort_values("parcel")
    if len(labels) != N_ROIS:
        raise ValueError(f"Expected {N_ROIS} Yeo labels, found {len(labels)}")

    masks = []
    for network in ["Vis", "SomMot", "DorsAttn", "SalVentAttn", "Limbic", "Cont", "Default"]:
        mask = labels["network"].eq(network).to_numpy()
        masks.append({"network": f"Yeo7_{network}", "mask": mask})

    language = pd.read_csv(LIPKIN_LANGUAGE).sort_values("parcel")
    md = pd.read_csv(LIPKIN_MD).sort_values("parcel")
    masks.append({"network": "Lipkin_language_top10", "mask": language["lipkin_top10"].to_numpy(bool)})
    masks.append({"network": "Lipkin_MD_top10", "mask": md["lipkin_md_top10"].to_numpy(bool)})
    return masks


def summarize_network_enrichment(aligned: np.ndarray) -> pd.DataFrame:
    rows = []
    masks = network_masks()

    for pc_idx in range(N_PCS):
        pc_number = pc_idx + 1
        values = aligned[pc_idx]
        squared = values**2
        total_squared = squared.sum()
        abs_order = np.argsort(squared)[::-1]
        top_10pct_share = float(squared[abs_order[:80]].sum() / total_squared)

        for item in masks:
            mask = np.asarray(item["mask"], dtype=bool)
            n_parcels = int(mask.sum())
            parcel_fraction = n_parcels / N_ROIS
            squared_fraction = float(squared[mask].sum() / total_squared)
            rows.append(
                {
                    "pc_number": pc_number,
                    "network": item["network"],
                    "n_parcels": n_parcels,
                    "parcel_fraction": parcel_fraction,
                    "squared_loading_fraction": squared_fraction,
                    "enrichment": squared_fraction / parcel_fraction,
                    "mean_loading": float(values[mask].mean()),
                    "mean_abs_loading": float(np.abs(values[mask]).mean()),
                    "mean_squared_loading": float(squared[mask].mean()),
                    "top_10pct_squared_loading_share": top_10pct_share,
                    "spin_p": np.nan,
                    "spin_p_bh_within_pc": np.nan,
                    "spin_status": "not_computed_missing_local_spin_null",
                }
            )

    return pd.DataFrame(rows)


def component_to_img(component: np.ndarray, atlas_img, atlas_data):
    out = np.zeros_like(atlas_data, dtype=float)
    for parcel_idx, value in enumerate(component, start=1):
        out[atlas_data == parcel_idx] = value
    return new_img_like(atlas_img, out)


def render_single_views(aligned: np.ndarray, vlim: float) -> None:
    atlas = datasets.fetch_atlas_schaefer_2018(
        n_rois=N_ROIS,
        yeo_networks=YEO_NETWORKS,
        resolution_mm=RESOLUTION_MM,
    )
    atlas_img = image.load_img(atlas.maps)
    atlas_data = atlas_img.get_fdata()
    fsaverage = datasets.fetch_surf_fsaverage(mesh="fsaverage5")

    specs = [
        ("left", "lateral", fsaverage.pial_left, fsaverage.sulc_left),
        ("left", "medial", fsaverage.pial_left, fsaverage.sulc_left),
        ("right", "lateral", fsaverage.pial_right, fsaverage.sulc_right),
        ("right", "medial", fsaverage.pial_right, fsaverage.sulc_right),
    ]

    for pc_number in RENDER_PCS:
        img = component_to_img(aligned[pc_number - 1], atlas_img, atlas_data)
        img.to_filename(OUT / f"PC{pc_number:02d}_L2_minus_L1_aligned_loadings.nii.gz")

        lh = surface.vol_to_surf(img, fsaverage.pial_left)
        rh = surface.vol_to_surf(img, fsaverage.pial_right)
        texture_by_hemi = {"left": lh, "right": rh}

        for hemi, view, mesh, bg in specs:
            fig = plt.figure(figsize=(2.4, 2.4), facecolor="none")
            ax = fig.add_subplot(111, projection="3d")
            plotting.plot_surf_stat_map(
                surf_mesh=mesh,
                stat_map=texture_by_hemi[hemi],
                hemi=hemi,
                view=view,
                bg_map=bg,
                bg_on_data=False,
                cmap="RdBu_r",
                symmetric_cbar=True,
                vmin=-vlim,
                vmax=vlim,
                colorbar=False,
                axes=ax,
                figure=fig,
            )
            ax.set_axis_off()
            fig.savefig(
                OUT / f"PC{pc_number:02d}_{hemi}_{view}.png",
                dpi=600,
                bbox_inches="tight",
                pad_inches=0,
                transparent=True,
            )
            plt.close(fig)

    fig, ax = plt.subplots(figsize=(3.0, 0.34))
    norm = Normalize(vmin=-vlim, vmax=vlim)
    sm = plt.cm.ScalarMappable(norm=norm, cmap="RdBu_r")
    sm.set_array([])
    cb = fig.colorbar(sm, cax=ax, orientation="horizontal")
    cb.set_label("Signed PC loading", fontsize=8)
    cb.ax.tick_params(labelsize=7, length=2)
    fig.savefig(
        OUT / "PC01_PC05_PC10_PC12_colorbar.png",
        dpi=600,
        bbox_inches="tight",
        pad_inches=0.02,
        transparent=True,
    )
    plt.close(fig)


def write_report(
    alignment: pd.DataFrame,
    network_summary: pd.DataFrame,
    vlim: float,
) -> None:
    lines = [
        "# Figure 2 Panel D Shared Dimension Maps",
        "",
        f"Input loadings: `{FULL_SAMPLE_LOADINGS}`",
        f"Trial features: `{FEATURES}`",
        f"Trial labels: `{TRIAL_LABELS}`",
        "",
        "Sign alignment used the L2-minus-L1 contrast. In this dataset, L1 is coded as `true_y = 1` and L2 as `true_y = 0`.",
        "Each participant's parcel values were z-scored across that participant's 160 trials before PC scores were computed.",
        "",
        "Spin-test p-values were not computed in this local run because neither spin samples nor a spin-null package (`neuromaps` or `netneurotools`) is available locally.",
        "",
        f"Common render limit for PCs {', '.join(map(str, RENDER_PCS))}: +/- {vlim:.8f}",
        "",
        "## Sign Alignment",
        "",
        alignment.to_string(index=False),
        "",
        "## Strongest Observed Network Enrichment Per PC",
        "",
    ]

    strongest = (
        network_summary.sort_values(["pc_number", "enrichment"], ascending=[True, False])
        .groupby("pc_number", as_index=False)
        .head(3)
    )
    lines.append(
        strongest[
            [
                "pc_number",
                "network",
                "enrichment",
                "squared_loading_fraction",
                "top_10pct_squared_loading_share",
                "spin_status",
            ]
        ].to_string(index=False)
    )
    (OUT / "REPORT.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    data, parcel_cols = load_zscored_features()
    loadings, vectors = load_full_sample_loadings(parcel_cols)
    aligned, alignment, by_subject = sign_align_to_l2_minus_l1(data, parcel_cols, vectors)

    aligned_rows = []
    for row_idx, pc_number in enumerate(range(1, N_PCS + 1)):
        meta = loadings.loc[loadings["component"].eq(f"PC{pc_number}")].iloc[0]
        aligned_rows.append(
            {
                "component": f"PC{pc_number}",
                "explained_variance_ratio": meta["explained_variance_ratio"],
                "cumulative_explained_variance": meta["cumulative_explained_variance"],
                **{col: val for col, val in zip(parcel_cols, aligned[row_idx])},
            }
        )

    pd.DataFrame(aligned_rows).to_csv(
        OUT / "pc1_pc25_L2_minus_L1_aligned_loadings.csv",
        index=False,
    )
    alignment.to_csv(OUT / "pc1_pc25_language_alignment_summary.csv", index=False)
    by_subject.to_csv(OUT / "pc1_pc25_language_alignment_by_subject.csv", index=False)

    network_summary = summarize_network_enrichment(aligned)
    network_summary.to_csv(OUT / "pc1_pc25_network_enrichment_observed.csv", index=False)

    vlim = float(np.percentile(np.abs(aligned[np.array(RENDER_PCS) - 1].ravel()), 99))
    pd.DataFrame(
        [
            {
                "render_pcs": ",".join(map(str, RENDER_PCS)),
                "colormap": "RdBu_r",
                "symmetric_limit_99th_percentile_abs_loading": vlim,
                "spin_tests": "not_computed_missing_local_spin_null",
            }
        ]
    ).to_csv(OUT / "render_settings.csv", index=False)

    render_single_views(aligned, vlim)
    write_report(alignment, network_summary, vlim)

    print(f"Saved Figure 2 panel D files to {OUT}")
    print(alignment[["pc_number", "mean_subject_cohens_d_after_alignment", "sign_flipped"]].to_string(index=False))
    print(f"Common render limit: +/- {vlim:.8f}")


if __name__ == "__main__":
    main()
