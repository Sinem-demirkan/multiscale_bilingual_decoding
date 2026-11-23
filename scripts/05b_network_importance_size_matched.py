# Step 5b: network importance without PCA, size-matched (Methods 2.6, SI Figs. S6A, S7A).
#
# Same question as step 5, but on the parcel features directly:
#   1. train ONE classifier on all 800 parcels (z-scored on the training run),
#      leave-one-run-out, as in step 2
#   2. networks differ in size, so each network is cut down to the size of the
#      smallest one (Limbic, 54 parcels for Yeo 7): 500 random samples of parcels
#   3. for each sample, shuffle the sampled parcels across held-out trials
#      (20 times) and measure how much balanced accuracy drops
# Importance = mean drop over shuffles, samples and both folds
# (500 x 20 x 2 = 20,000 shuffles per network and participant).
#
# The Lipkin language and multiple-demand sets have the same size (80 parcels),
# so each set is shuffled as a whole (20 times per fold) in the same classifier.
#
# Settings below: TARGET = "language" (S6A) or "switch" (S7A); YEO = 7 or 17 (S6).
#
# Usage:  python scripts/05b_network_importance_size_matched.py   (after step 1)

from pathlib import Path

import numpy as np
import pandas as pd
from nilearn import image
from nilearn.datasets import fetch_atlas_schaefer_2018
from nilearn.maskers import NiftiLabelsMasker
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

TARGET = "language"   # "language" (Chinese vs English) or "switch" (switch vs non-switch)
YEO = 7               # 7 or 17 networks

# probabilistic atlases from the EvLab resources (Lipkin et al., 2022, 2023)
LIPKIN_LANGUAGE_MAP = Path("/path/to/LanA_n806.nii")             # language network
LIPKIN_MD_MAP = Path("/path/to/MD_probabilistic_map.nii")        # multiple-demand network

N_SAMPLES = 500
N_SHUFFLES = 20
RESULTS = Path(__file__).resolve().parents[1] / "results"

data = np.load(RESULTS / "parcels_schaefer800.npz")
subjects = sorted(key[2:] for key in data.files if key.startswith("X_"))
print(len(subjects), "participants")


# ---- define the parcel sets (same as step 5) ---------------------------------
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


def top_parcels(prob_map, fraction=0.10):
    masker = NiftiLabelsMasker(labels_img=atlas.maps, strategy="mean", resampling_target="labels")
    parcel_prob = masker.fit_transform(prob_map).ravel()     # (800,)
    n_keep = int(round(fraction * len(parcel_prob)))
    return np.sort(np.argsort(parcel_prob)[::-1][:n_keep])


yeo = yeo_networks(YEO)
lipkin = {"Lang": top_parcels(LIPKIN_LANGUAGE_MAP), "MD": top_parcels(LIPKIN_MD_MAP)}
assert not set(lipkin["Lang"]) & set(lipkin["MD"]), "language and MD sets should not overlap"

# the same 500 size-matched samples are used for every participant
n_matched = min(len(cols) for cols in yeo.values())
print({net: len(cols) for net, cols in yeo.items()}, "-> size-matched to", n_matched, "parcels")
rng = np.random.default_rng(42)
samples = {net: [rng.choice(cols, size=n_matched, replace=False) for _ in range(N_SAMPLES)]
           for net, cols in yeo.items()}


# ---- permutation importance --------------------------------------------------
def make_labels(trial_type, target):
    if target == "language":
        return np.array([t[:2] for t in trial_type])                                  # "L1" / "L2"
    return np.array(["nonswitch" if t.endswith("NS") else "switch" for t in trial_type])


def balanced_accuracy(y_true, y_pred):
    return np.mean([np.mean(y_pred[y_true == c] == c) for c in np.unique(y_true)])


rows = []
for s in subjects:
    X = data[f"X_{s}"]
    runs = data[f"run_{s}"]
    y = make_labels(data[f"trial_type_{s}"], TARGET)
    rng = np.random.default_rng(42)
    drops = {name: [] for name in list(yeo) + list(lipkin)}

    for test_run in np.unique(runs):
        train = runs != test_run
        test = runs == test_run

        # 1. one classifier on all 800 z-scored parcels
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X[train])
        X_test = scaler.transform(X[test])
        clf = LogisticRegression(C=0.01, solver="lbfgs", max_iter=1000, random_state=42)
        clf.fit(X_train, y[train])

        # same as clf.predict, written out so the 140,000 shuffles run fast
        w, b = clf.coef_[0], clf.intercept_[0]
        def predict(Z):
            return clf.classes_[(Z @ w + b > 0).astype(int)]

        y_test = y[test]
        base = balanced_accuracy(y_test, predict(X_test))

        # 2-3. shuffle a set of parcels across held-out trials, measure the drop
        def shuffle_drop(cols):
            X_shuffled = X_test.copy()
            X_shuffled[:, cols] = X_test[rng.permutation(len(X_test))][:, cols]
            return base - balanced_accuracy(y_test, predict(X_shuffled))

        for net in yeo:
            for cols in samples[net]:
                drops[net] += [shuffle_drop(cols) for _ in range(N_SHUFFLES)]
        for name, cols in lipkin.items():
            drops[name] += [shuffle_drop(cols) for _ in range(N_SHUFFLES)]

    rows.append({"subject": s, **{name: np.mean(d) for name, d in drops.items()}})
    print(s, "done", flush=True)

df = pd.DataFrame(rows)
df.to_csv(RESULTS / f"network_importance_size_matched_{TARGET}_yeo{YEO}.csv", index=False)


# ---- group summary: participant bootstrap ------------------------------------
rng = np.random.default_rng(42)
boot_samples = [rng.integers(0, len(df), len(df)) for _ in range(5000)]


def mean_ci(values):
    boot = [values[sample].mean() for sample in boot_samples]
    return values.mean(), *np.percentile(boot, [2.5, 97.5])


print(f"\n{TARGET}, Yeo {YEO}, size-matched: accuracy drop when shuffled (pp), mean [95% CI]")
for col in list(yeo) + ["Lang", "MD"]:
    m, low, high = mean_ci(df[col].to_numpy())
    print(f"{col:<14}{100 * m:+5.2f} [{100 * low:+.2f}, {100 * high:+.2f}]")

if YEO == 7:
    other_yeo = [net for net in yeo if net != "Vis"]
    m, low, high = mean_ci((df[other_yeo].mean(axis=1) - df["Vis"]).to_numpy())
    print(f"\nmean of other Yeo networks - Vis  {100 * m:+.2f} [{100 * low:+.2f}, {100 * high:+.2f}]")
m, low, high = mean_ci((df["Lang"] - df["MD"]).to_numpy())
print(f"Lang - MD                         {100 * m:+.2f} [{100 * low:+.2f}, {100 * high:+.2f}]")
