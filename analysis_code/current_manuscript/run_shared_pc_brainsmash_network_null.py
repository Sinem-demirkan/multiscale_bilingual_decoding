from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from brainsmash.mapgen import Base
from nibabel.affines import apply_affine
from nilearn import datasets, image


N_SURROGATES = 1000
RANDOM_STATE = 42
N_ROIS = 800
YEO_NETWORKS = 7
RESOLUTION_MM = 2

ROOT = Path(__file__).resolve().parents[2]
OUT_ROOT = ROOT / "paper_version" / "cross_subject_language_transfer"
PANEL_D = OUT_ROOT / "figure2_panel_d_shared_pc_maps_networks"

ALIGNED_LOADINGS = PANEL_D / "pc1_pc25_L2_minus_L1_aligned_loadings.csv"
OBSERVED_ENRICHMENT = PANEL_D / "pc1_pc25_network_enrichment_observed.csv"
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


def bh_adjust(p_values: np.ndarray) -> np.ndarray:
    p_values = np.asarray(p_values, dtype=float)
    adjusted = np.full_like(p_values, np.nan)
    ok = np.isfinite(p_values)
    if not ok.any():
        return adjusted

    p = p_values[ok]
    order = np.argsort(p)
    ranked = p[order]
    n = len(ranked)
    raw = ranked * n / np.arange(1, n + 1)
    monotone = np.minimum.accumulate(raw[::-1])[::-1]
    monotone = np.clip(monotone, 0, 1)

    out = np.empty_like(ranked)
    out[order] = monotone
    adjusted[ok] = out
    return adjusted


def load_network_masks() -> tuple[pd.DataFrame, list[dict[str, object]]]:
    labels = pd.read_csv(YEO_LABELS).sort_values("parcel").reset_index(drop=True)
    if len(labels) != N_ROIS:
        raise ValueError(f"Expected {N_ROIS} Schaefer labels, found {len(labels)}")

    masks = []
    for network in ["Vis", "SomMot", "DorsAttn", "SalVentAttn", "Limbic", "Cont", "Default"]:
        masks.append(
            {
                "network": f"Yeo7_{network}",
                "mask": labels["network"].eq(network).to_numpy(),
            }
        )

    language = pd.read_csv(LIPKIN_LANGUAGE).sort_values("parcel").reset_index(drop=True)
    md = pd.read_csv(LIPKIN_MD).sort_values("parcel").reset_index(drop=True)
    masks.append({"network": "Lipkin_language_top10", "mask": language["lipkin_top10"].to_numpy(bool)})
    masks.append({"network": "Lipkin_MD_top10", "mask": md["lipkin_md_top10"].to_numpy(bool)})

    return labels, masks


def parcel_centroids() -> np.ndarray:
    atlas = datasets.fetch_atlas_schaefer_2018(
        n_rois=N_ROIS,
        yeo_networks=YEO_NETWORKS,
        resolution_mm=RESOLUTION_MM,
    )
    atlas_img = image.load_img(atlas.maps)
    atlas_data = atlas_img.get_fdata().astype(int)
    affine = atlas_img.affine

    centroids = np.zeros((N_ROIS, 3), dtype=float)
    for parcel in range(1, N_ROIS + 1):
        ijk = np.column_stack(np.where(atlas_data == parcel))
        if len(ijk) == 0:
            raise ValueError(f"Parcel {parcel} has no atlas voxels.")
        xyz = apply_affine(affine, ijk)
        centroids[parcel - 1] = xyz.mean(axis=0)

    return centroids


def euclidean_distance_matrix(xyz: np.ndarray) -> np.ndarray:
    diff = xyz[:, None, :] - xyz[None, :, :]
    return np.sqrt(np.sum(diff**2, axis=-1))


def enrichment(values: np.ndarray, mask: np.ndarray) -> float:
    squared = values**2
    return float((squared[mask].sum() / squared.sum()) / (mask.sum() / len(values)))


def surrogate_enrichments(surrogates: np.ndarray, masks: list[dict[str, object]]) -> dict[str, np.ndarray]:
    squared = surrogates**2
    total = squared.sum(axis=1)
    out = {}

    for item in masks:
        mask = np.asarray(item["mask"], dtype=bool)
        parcel_fraction = mask.sum() / surrogates.shape[1]
        out[item["network"]] = (squared[:, mask].sum(axis=1) / total) / parcel_fraction

    return out


def make_surrogates_for_pc(
    values: np.ndarray,
    left_idx: np.ndarray,
    right_idx: np.ndarray,
    left_distance: np.ndarray,
    right_distance: np.ndarray,
    pc_number: int,
) -> np.ndarray:
    left_gen = Base(
        values[left_idx],
        left_distance,
        seed=RANDOM_STATE + pc_number * 10 + 1,
        n_jobs=1,
    )
    right_gen = Base(
        values[right_idx],
        right_distance,
        seed=RANDOM_STATE + pc_number * 10 + 2,
        n_jobs=1,
    )

    left = np.atleast_2d(left_gen(N_SURROGATES, batch_size=100))
    right = np.atleast_2d(right_gen(N_SURROGATES, batch_size=100))

    surrogates = np.zeros((N_SURROGATES, N_ROIS), dtype=float)
    surrogates[:, left_idx] = left
    surrogates[:, right_idx] = right
    return surrogates


def main() -> None:
    labels, masks = load_network_masks()
    centroids = parcel_centroids()

    left_idx = labels.index[labels["hemisphere"].eq("LH")].to_numpy()
    right_idx = labels.index[labels["hemisphere"].eq("RH")].to_numpy()

    left_distance = euclidean_distance_matrix(centroids[left_idx])
    right_distance = euclidean_distance_matrix(centroids[right_idx])

    loadings = pd.read_csv(ALIGNED_LOADINGS)
    parcel_cols = parcel_columns(loadings)
    observed = pd.read_csv(OBSERVED_ENRICHMENT)

    rows = []
    for row in loadings.itertuples(index=False):
        pc_label = getattr(row, "component")
        pc_number = int(str(pc_label).replace("PC", ""))
        values = loadings.loc[loadings["component"].eq(pc_label), parcel_cols].iloc[0].to_numpy(float)

        print(f"BrainSMASH surrogates for PC{pc_number}", flush=True)
        surrogates = make_surrogates_for_pc(
            values,
            left_idx,
            right_idx,
            left_distance,
            right_distance,
            pc_number,
        )
        nulls = surrogate_enrichments(surrogates, masks)

        pc_rows = []
        for item in masks:
            network = item["network"]
            mask = np.asarray(item["mask"], dtype=bool)
            obs = enrichment(values, mask)
            null = nulls[network]
            p = (float(np.sum(null >= obs)) + 1.0) / (N_SURROGATES + 1.0)

            obs_row = observed.loc[
                observed["pc_number"].eq(pc_number) & observed["network"].eq(network)
            ].iloc[0]
            pc_rows.append(
                {
                    "pc_number": pc_number,
                    "network": network,
                    "n_parcels": int(mask.sum()),
                    "parcel_fraction": float(mask.sum() / N_ROIS),
                    "squared_loading_fraction": float(obs_row["squared_loading_fraction"]),
                    "enrichment": obs,
                    "mean_loading": float(obs_row["mean_loading"]),
                    "mean_abs_loading": float(obs_row["mean_abs_loading"]),
                    "mean_squared_loading": float(obs_row["mean_squared_loading"]),
                    "top_10pct_squared_loading_share": float(obs_row["top_10pct_squared_loading_share"]),
                    "brainsmash_p": p,
                    "brainsmash_null_mean": float(np.mean(null)),
                    "brainsmash_null_sd": float(np.std(null, ddof=1)),
                    "brainsmash_null_p95": float(np.percentile(null, 95)),
                    "n_surrogates": N_SURROGATES,
                    "spatial_null": "BrainSMASH_Base_MNI_centroids_separate_hemispheres",
                }
            )

        pc_frame = pd.DataFrame(pc_rows)
        pc_frame["brainsmash_p_bh_within_pc"] = bh_adjust(pc_frame["brainsmash_p"].to_numpy(float))
        rows.extend(pc_frame.to_dict("records"))

    result = pd.DataFrame(rows)
    result.to_csv(PANEL_D / "pc1_pc25_network_enrichment_brainsmash.csv", index=False)

    compact = (
        result.sort_values(["pc_number", "brainsmash_p_bh_within_pc", "enrichment"])
        .groupby("pc_number", as_index=False)
        .head(9)
    )
    compact.to_csv(PANEL_D / "pc1_pc25_network_enrichment_brainsmash_ranked.csv", index=False)

    note = [
        "# BrainSMASH Spatial Null for PC Network Enrichment",
        "",
        "Spatial null: BrainSMASH `brainsmash.mapgen.Base` (Burt et al., 2020).",
        "Surrogates were generated separately for left and right hemisphere Schaefer-800 parcels.",
        "Distance matrices were Euclidean distances between parcel centroids in MNI space.",
        f"Number of surrogates per PC: {N_SURROGATES}.",
        "One-sided p-values count surrogate enrichments greater than or equal to the observed enrichment.",
        "Benjamini-Hochberg correction was applied across the 9 networks within each PC.",
        "",
        result.sort_values(["pc_number", "brainsmash_p_bh_within_pc"])[
            [
                "pc_number",
                "network",
                "enrichment",
                "brainsmash_p",
                "brainsmash_p_bh_within_pc",
            ]
        ].to_string(index=False),
    ]
    (PANEL_D / "brainsmash_spatial_null_report.md").write_text("\n".join(note) + "\n")

    print(f"Saved {PANEL_D / 'pc1_pc25_network_enrichment_brainsmash.csv'}", flush=True)


if __name__ == "__main__":
    main()
