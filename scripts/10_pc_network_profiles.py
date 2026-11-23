# Step 10: where do the shared dimensions lie? (Methods 2.8, Fig. 2E, SI Fig. S9)
#
# 1. PCA on the pooled trials of ALL 77 participants (each z-scored on their own
#    trials, Chinese and English pooled, no labels) -> loading maps for display.
#    Check: its first 10 dimensions match those of every leave-pair-out space (step 4).
# 2. Sign of each dimension: positive = higher scores on English than Chinese trials
#    (mean over participants of Cohen's d, English - Chinese).
# 3. Network profile of each of the first 25 dimensions:
#       enrichment = network's share of the squared loadings / network's share of parcels
#    for the 7 Yeo networks and the Lipkin language and multiple-demand sets.
# 4. Spatial null: 1,000 BrainSMASH surrogate maps per dimension that keep its
#    spatial autocorrelation, made separately for each hemisphere from Euclidean
#    distances between parcel centroids. One-sided p, Benjamini-Hochberg across
#    the 9 networks within each dimension.
#
# Paper: first 10 dimensions explain 49.0% of the variance; cosine with every
#        leave-pair-out space >= 0.989; PC5 -> multiple demand + dorsal attention,
#        PC12 -> language + default.
#
# Usage:  python scripts/10_pc_network_profiles.py   (after step 4; about 1.5 hours, mostly BrainSMASH)

from pathlib import Path

import numpy as np
import pandas as pd
from brainsmash.mapgen import Base
from nilearn import image
from nilearn.datasets import fetch_atlas_schaefer_2018
from nilearn.maskers import NiftiLabelsMasker
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.multitest import multipletests

# probabilistic atlases from the EvLab resources (Lipkin et al., 2022, 2023), as in step 5
LIPKIN_LANGUAGE_MAP = Path("/path/to/LanA_n806.nii")             # language network
LIPKIN_MD_MAP = Path("/path/to/MD_probabilistic_map.nii")        # multiple-demand network

N_DIMS = 25
N_SURROGATES = 1000

# BrainSMASH 0.11 uses np.row_stack, which recent NumPy versions removed (same as np.vstack)
if not hasattr(np, "row_stack"):
    np.row_stack = np.vstack
RESULTS = Path(__file__).resolve().parents[1] / "results"


# ---- 1. PCA on all participants ----------------------------------------------
data = np.load(RESULTS / "parcels_schaefer800.npz")
subjects = sorted(key[2:] for key in data.files if key.startswith("X_"))
subject_X = {s: StandardScaler().fit_transform(data[f"X_{s}"]) for s in subjects}
subject_y = {s: np.array([t[:2] for t in data[f"trial_type_{s}"]]) for s in subjects}

pca = PCA(n_components=200, svd_solver="randomized", random_state=42)
pca.fit(np.vstack([subject_X[s] for s in subjects]))
loadings = pca.components_[:N_DIMS].copy()        # (25, 800)

pd.DataFrame({"dimension": np.arange(1, 201),
              "explained_variance": pca.explained_variance_ratio_,
              "cumulative": np.cumsum(pca.explained_variance_ratio_)}
             ).to_csv(RESULTS / "pc_variance_explained.csv", index=False)
print(f"first 10 dimensions explain {100 * pca.explained_variance_ratio_[:10].sum():.1f}% of the variance")

# same dimensions as the leave-pair-out spaces? |cosine| of dimension d with dimension d
components = np.load(RESULTS / "shared_dimensions_components.npy", mmap_mode="r")
cosines = np.array([np.abs(np.sum(np.asarray(components[p, :10]) * pca.components_[:10], axis=1))
                    for p in range(len(components))])          # (n_spaces, 10)
print(f"lowest |cosine| over all leave-pair-out spaces, dimensions 1-10: {cosines.min():.3f}")


# ---- 2. sign: positive = English > Chinese -------------------------------------
def cohens_d(a, b):
    pooled_sd = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2))
    return (a.mean() - b.mean()) / pooled_sd


for d in range(N_DIMS):
    effect = np.mean([cohens_d(subject_X[s][subject_y[s] == "L2"] @ loadings[d],
                               subject_X[s][subject_y[s] == "L1"] @ loadings[d]) for s in subjects])
    if effect < 0:
        loadings[d] *= -1

pd.DataFrame(loadings, index=pd.Index(np.arange(1, N_DIMS + 1), name="dimension"),
             columns=[f"parcel_{p}" for p in range(1, 801)]).to_csv(RESULTS / "pc_loadings_all_participants.csv")


# ---- 3. networks and enrichment ------------------------------------------------
atlas = fetch_atlas_schaefer_2018(n_rois=800, yeo_networks=7, resolution_mm=2)
labels = [str(label) for label in atlas.labels[1:]]          # labels[0] is "Background"

networks = {}
for parcel, label in enumerate(labels):                      # "7Networks_LH_SomMot_12" -> "SomMot"
    networks.setdefault(label.split("_")[2], []).append(parcel)


def top_parcels(prob_map, fraction=0.10):
    masker = NiftiLabelsMasker(labels_img=atlas.maps, strategy="mean", resampling_target="labels")
    parcel_prob = masker.fit_transform(prob_map).ravel()
    return np.sort(np.argsort(parcel_prob)[::-1][:int(round(fraction * len(parcel_prob)))])


networks["Lang"] = top_parcels(LIPKIN_LANGUAGE_MAP)
networks["MD"] = top_parcels(LIPKIN_MD_MAP)
masks = {name: np.isin(np.arange(800), cols) for name, cols in networks.items()}


def enrichment(maps):
    # maps: (n_maps, 800) -> {network: (n_maps,)}
    squared = maps ** 2
    share = squared / squared.sum(axis=1, keepdims=True)
    return {name: share[:, m].sum(axis=1) / m.mean() for name, m in masks.items()}


# ---- 4. BrainSMASH spatial null, per hemisphere --------------------------------
atlas_ids = np.asarray(image.load_img(atlas.maps).dataobj).astype(int)
affine = image.load_img(atlas.maps).affine
centroids = np.array([(affine[:3, :3] @ np.argwhere(atlas_ids == p).T).T.mean(axis=0) + affine[:3, 3]
                      for p in range(1, 801)])                 # parcel centres in mm
hemispheres = [np.array(["_LH_" in label for label in labels]),
               np.array(["_RH_" in label for label in labels])]
distances = [np.linalg.norm(centroids[h][:, None] - centroids[h][None, :], axis=-1) for h in hemispheres]

rows = []
for d in range(N_DIMS):
    surrogates = np.zeros((N_SURROGATES, 800))
    for side, (h, dist) in enumerate(zip(hemispheres, distances)):
        generator = Base(loadings[d][h], dist, seed=42 + 10 * (d + 1) + side + 1)
        surrogates[:, h] = generator(N_SURROGATES)

    observed = enrichment(loadings[[d]])
    null = enrichment(surrogates)
    p = [(np.sum(null[name] >= observed[name][0]) + 1) / (N_SURROGATES + 1) for name in masks]
    q = multipletests(p, method="fdr_bh")[1]
    for name, p_value, q_value in zip(masks, p, q):
        rows.append({"dimension": d + 1, "network": name, "enrichment": observed[name][0],
                     "p_spatial_null": p_value, "q_bh": q_value})
    print(f"dimension {d + 1} done", flush=True)

profiles = pd.DataFrame(rows)
profiles.to_csv(RESULTS / "pc_network_enrichment.csv", index=False)

print("\nnetworks above the spatial null (q < 0.05), by dimension")
for d, g in profiles.groupby("dimension"):
    sig = g[g["q_bh"] < 0.05].sort_values("enrichment", ascending=False)
    print(f"PC{d:<3}", ", ".join(f"{r.network} ({r.enrichment:.2f})" for r in sig.itertuples()) or "-")
