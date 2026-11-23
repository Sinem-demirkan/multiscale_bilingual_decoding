# Step 5: which networks does the whole-cortex decoder rely on? (Methods 2.6, Fig. 1D)
#
# Permutation feature importance on PCA scores, within each participant,
# leave-one-run-out:
#   1. split the 800 parcels into blocks (networks)
#   2. inside each block: z-score, then PCA keeping 90% of the variance
#      (both fitted on the training run only)
#   3. concatenate the component scores of all blocks, train ONE classifier
#   4. on the held-out run, shuffle one block's component scores across trials
#      (20 times) and measure how much balanced accuracy drops
# Importance = mean drop, averaged over shuffles and both folds.
#
# Two models:
#   Yeo networks   every parcel belongs to one network (7 or 17)
#   Lipkin sets    top 10% parcels (80) of the language and of the multiple-demand
#                  probabilistic maps (they do not overlap) + all remaining parcels
#                  as a third block
#
# Settings below: TARGET = "language" (Fig. 1D, S6B) or "switch" (S7B);
#                 YEO = 7 (main) or 17 (S6).
# Paper (language, Yeo 7): somatomotor largest (1.9 pp); visual lower than the others.
#
# Usage:  python scripts/05_network_importance.py   (after step 1)

from pathlib import Path

import numpy as np
import pandas as pd
from nilearn import image
from nilearn.datasets import fetch_atlas_schaefer_2018
from nilearn.maskers import NiftiLabelsMasker
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import StandardScaler

TARGET = "language"   # "language" (Chinese vs English) or "switch" (switch vs non-switch)
YEO = 7               # 7 or 17 networks

# probabilistic atlases from the EvLab resources (Lipkin et al., 2022, 2023)
LIPKIN_LANGUAGE_MAP = Path("/path/to/LanA_n806.nii")             # language network
LIPKIN_MD_MAP = Path("/path/to/MD_probabilistic_map.nii")        # multiple-demand network

N_SHUFFLES = 20
RESULTS = Path(__file__).resolve().parents[1] / "results"

data = np.load(RESULTS / "parcels_schaefer800.npz")
subjects = sorted(key[2:] for key in data.files if key.startswith("X_"))
print(len(subjects), "participants")


# ---- define the parcel sets --------------------------------------------------
# The parcel file uses the 7-network atlas, so its columns follow that atlas's order.
atlas = fetch_atlas_schaefer_2018(n_rois=800, yeo_networks=7, resolution_mm=2)


def yeo_networks(n_networks):
    """{network name: parcel columns}, labels like '7Networks_LH_SomMot_12' -> 'SomMot'."""
    atlas_n = fetch_atlas_schaefer_2018(n_rois=800, yeo_networks=n_networks, resolution_mm=2)
    names = [str(label).split("_")[2] for label in atlas_n.labels[1:]]   # labels[0] is "Background"
    if n_networks == 7:
        network_of_column = names
    else:
        # the 17-network atlas numbers the same 800 parcels in another order:
        # find, for each 7-network parcel, the 17-network parcel at the same place
        # (both atlas images are on the same 2 mm grid)
        ids7 = np.asarray(image.load_img(atlas.maps).dataobj).astype(int)
        ids_n = np.asarray(image.load_img(atlas_n.maps).dataobj).astype(int)
        network_of_column = []
        for parcel in range(1, 801):
            match = np.bincount(ids_n[ids7 == parcel]).argmax()
            network_of_column.append(names[match - 1])
    networks = {}
    for column, name in enumerate(network_of_column):
        networks.setdefault(name, []).append(column)
    return networks


yeo = yeo_networks(YEO)
print({net: len(cols) for net, cols in yeo.items()})


# Lipkin: mean probability in each parcel, keep the top 10% (80 parcels)
def top_parcels(prob_map, fraction=0.10):
    masker = NiftiLabelsMasker(labels_img=atlas.maps, strategy="mean", resampling_target="labels")
    parcel_prob = masker.fit_transform(prob_map).ravel()     # (800,)
    n_keep = int(round(fraction * len(parcel_prob)))
    return np.sort(np.argsort(parcel_prob)[::-1][:n_keep])


lipkin = {"Lang": top_parcels(LIPKIN_LANGUAGE_MAP), "MD": top_parcels(LIPKIN_MD_MAP)}
overlap = set(lipkin["Lang"]) & set(lipkin["MD"])
assert not overlap, f"language and MD sets share {len(overlap)} parcels; they should not overlap"
lipkin_other = np.setdiff1d(np.arange(800), np.concatenate([lipkin["Lang"], lipkin["MD"]]))


# ---- permutation importance --------------------------------------------------
# blocks: {name: parcel columns}; importance is returned for the blocks in `test_blocks`
def network_importance(X, y, runs, blocks, test_blocks, rng):
    drops = {name: [] for name in test_blocks}
    for test_run in np.unique(runs):
        train = runs != test_run
        test = runs == test_run

        # 1-2. z-score and PCA each block on the training run, apply to the test run
        train_parts, test_parts, block_cols = [], [], {}
        start = 0
        for name, cols in blocks.items():
            scaler = StandardScaler()
            pca = PCA(n_components=0.90, svd_solver="full")
            Z_train = pca.fit_transform(scaler.fit_transform(X[train][:, cols]))
            Z_test = pca.transform(scaler.transform(X[test][:, cols]))
            train_parts.append(Z_train)
            test_parts.append(Z_test)
            block_cols[name] = np.arange(start, start + Z_train.shape[1])
            start += Z_train.shape[1]
        Z_train = np.hstack(train_parts)
        Z_test = np.hstack(test_parts)

        # 3. one classifier on all blocks
        clf = LogisticRegression(C=0.01, solver="lbfgs", max_iter=1000, random_state=42)
        clf.fit(Z_train, y[train])
        base = balanced_accuracy_score(y[test], clf.predict(Z_test))

        # 4. shuffle one block at a time across held-out trials
        for name in test_blocks:
            cols = block_cols[name]
            for _ in range(N_SHUFFLES):
                Z_shuffled = Z_test.copy()
                order = rng.permutation(len(Z_test))
                Z_shuffled[:, cols] = Z_test[order][:, cols]
                drops[name].append(base - balanced_accuracy_score(y[test], clf.predict(Z_shuffled)))

    return {name: np.mean(d) for name, d in drops.items()}


def make_labels(trial_type, target):
    if target == "language":
        return np.array([t[:2] for t in trial_type])                                  # "L1" / "L2"
    return np.array(["nonswitch" if t.endswith("NS") else "switch" for t in trial_type])


rows = []
for s in subjects:
    X = data[f"X_{s}"]
    runs = data[f"run_{s}"]
    y = make_labels(data[f"trial_type_{s}"], TARGET)
    rng = np.random.default_rng(42)

    # Yeo: one block per network, one model, shuffle each network
    row = network_importance(X, y, runs, yeo, list(yeo), rng)

    # Lipkin: language, MD and the remaining parcels, one model, shuffle language and MD
    lipkin_blocks = {**lipkin, "other": lipkin_other}
    row.update(network_importance(X, y, runs, lipkin_blocks, ["Lang", "MD"], rng))

    rows.append({"subject": s, **row})
    print(s, "done")

df = pd.DataFrame(rows)
df.to_csv(RESULTS / f"network_importance_{TARGET}_yeo{YEO}.csv", index=False)


# ---- group summary: participant bootstrap ------------------------------------
rng = np.random.default_rng(42)
boot_samples = [rng.integers(0, len(df), len(df)) for _ in range(5000)]


def mean_ci(values):
    boot = [values[sample].mean() for sample in boot_samples]
    return values.mean(), *np.percentile(boot, [2.5, 97.5])


print(f"\n{TARGET}, Yeo {YEO}: accuracy drop when shuffled (percentage points), mean [95% CI]")
for col in list(yeo) + ["Lang", "MD"]:
    m, low, high = mean_ci(df[col].to_numpy())
    print(f"{col:<14}{100 * m:+5.2f} [{100 * low:+.2f}, {100 * high:+.2f}]")

if YEO == 7:
    other_yeo = [net for net in yeo if net != "Vis"]
    m, low, high = mean_ci((df[other_yeo].mean(axis=1) - df["Vis"]).to_numpy())
    print(f"\nmean of other Yeo networks - Vis  {100 * m:+.2f} [{100 * low:+.2f}, {100 * high:+.2f}]")
m, low, high = mean_ci((df["Lang"] - df["MD"]).to_numpy())
print(f"Lang - MD                         {100 * m:+.2f} [{100 * low:+.2f}, {100 * high:+.2f}]")
