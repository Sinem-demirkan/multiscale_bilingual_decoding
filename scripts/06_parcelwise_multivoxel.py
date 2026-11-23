# Step 6: decoding inside each parcel separately (Methods 2.7, Fig. 1C).
#
# For each participant and each of the 800 Schaefer parcels, Chinese vs English
# with the same leave-one-run-out decoder, using three representations of the parcel:
#   voxels        every voxel is a feature (the full multivoxel pattern)
#   demeaned      voxels minus the parcel mean on each trial
#   mean          the parcel mean only (one feature)
#
# Group level: one-sided t-test of the 77 accuracies against 50% for every parcel
#              (settings T_TEST_SIDE and Q_THRESHOLD below),
#              Benjamini-Hochberg across the 800 parcels, q <= 0.001.
# Participant level: one-sided binomial test on the held-out predictions,
#              Benjamini-Hochberg across the 800 parcels within each participant, q < 0.05.
#
# Paper: 152 parcels (voxels), 135 (demeaned), 19 (mean).
#
# This script needs the voxels, so like step 1 it reads the raw single-trial images.
# Usage:  python scripts/06_parcelwise_multivoxel.py   (slow: about 1-2 hours)

from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from nilearn import image
from nilearn.datasets import fetch_atlas_schaefer_2018, load_mni152_template
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.multitest import multipletests

DATA_ROOT = Path("/home/sdemirka/fmri/duo-cogcon")   # same folder as in step 1
RESULTS = Path(__file__).resolve().parents[1] / "results"

# group-level test, as written in Methods 2.7 (the old code used a two-sided
# t-test with q < 0.05; change these two lines to compare)
T_TEST_SIDE = "greater"   # "greater" = one-sided, "two-sided"
Q_THRESHOLD = 0.001

TASK = "LanguageControl"
TRIAL_COLUMN = "trial_type"


def parse_run_number(path):
    return int(path.name.split("run-")[1].split("_")[0])


# common grid: 2 mm MNI152 template; atlas labels resampled to it (nearest neighbour)
template = load_mni152_template(resolution=2)
atlas = fetch_atlas_schaefer_2018(n_rois=800, yeo_networks=7, resolution_mm=2)
atlas_img = image.resample_to_img(atlas.maps, template, interpolation="nearest",
                                  force_resample=True, copy_header=True)
atlas_labels = np.asarray(atlas_img.dataobj).astype(int).ravel()
cortex = atlas_labels > 0                         # keep only voxels inside a parcel
parcel_of_voxel = atlas_labels[cortex]


# trials x cortical voxels for one participant (same file matching as step 1)
def load_subject_voxels(subject):
    beta_files = DATA_ROOT.glob(f"**/{subject}_task-{TASK}_run-*_singletrial-Act.nii.gz")
    event_files = DATA_ROOT.glob(f"**/{subject}_task-{TASK}_run-*_events.tsv")
    beta_by_run = {parse_run_number(p): p for p in beta_files}
    event_by_run = {parse_run_number(p): p for p in event_files}

    X_runs, y, runs = [], [], []
    for run in sorted(set(beta_by_run) & set(event_by_run)):
        events = pd.read_csv(event_by_run[run], sep="\t")
        img = nib.load(beta_by_run[run])
        if img.shape[-1] != len(events):
            print(subject, "run", run, "mismatch:", img.shape[-1], "volumes vs", len(events), "events")
            continue
        img = image.resample_to_img(img, template, interpolation="continuous",
                                    force_resample=True, copy_header=True)
        vols = np.asarray(img.dataobj, dtype=np.float32)
        vols = vols.reshape(-1, vols.shape[-1])[cortex].T          # (n_trials, n_cortical_voxels)
        vols = np.nan_to_num(vols)                                 # empty voxels at the brain edge -> 0

        trial_types = events[TRIAL_COLUMN].astype(str)
        keep = trial_types.str.startswith(("L1", "L2")).to_numpy()  # naming trials only
        X_runs.append(vols[keep])
        y += [t[:2] for t in trial_types[keep]]
        runs += [run] * keep.sum()

    if not X_runs:
        return None
    return np.vstack(X_runs), np.array(y), np.array(runs)


# leave-one-run-out; returns accuracy (mean over folds) and number of correct held-out predictions
def decode_across_runs(X, y, runs):
    scores, n_correct = [], 0
    for test_run in np.unique(runs):
        train = runs != test_run
        test = runs == test_run
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X[train])
        X_test = scaler.transform(X[test])
        clf = LogisticRegression(C=0.01, solver="lbfgs", max_iter=1000)
        clf.fit(X_train, y[train])
        y_pred = clf.predict(X_test)
        scores.append(balanced_accuracy_score(y[test], y_pred))
        n_correct += (y_pred == y[test]).sum()
    return np.mean(scores), n_correct


subjects = sorted({p.name.split("_")[0] for p in DATA_ROOT.glob(f"**/*_task-{TASK}_*_events.tsv")})
print(len(subjects), "participants")

rows = []
for s in subjects:
    result = load_subject_voxels(s)
    if result is None:
        print(s, "no files found")
        continue
    X, y, runs = result

    for parcel in range(1, 801):
        X_parcel = X[:, parcel_of_voxel == parcel]                    # (n_trials, n_voxels)
        parcel_mean = X_parcel.mean(axis=1, keepdims=True)            # (n_trials, 1)
        representations = {
            "voxels": X_parcel,
            "demeaned": X_parcel - parcel_mean,
            "mean": parcel_mean,
        }
        for rep, features in representations.items():
            acc, n_correct = decode_across_runs(features, y, runs)
            rows.append({"subject": s, "parcel": parcel, "representation": rep,
                         "n_voxels": X_parcel.shape[1], "accuracy": acc,
                         "n_correct": n_correct, "n_trials": len(y)})
    print(s, "done", flush=True)

df = pd.DataFrame(rows)

# participant level: one-sided binomial test, BH across the 800 parcels within each participant
df["p_binomial"] = stats.binom.sf(df["n_correct"] - 1, df["n_trials"], 0.5)
df["significant_in_participant"] = False
for (s, rep), idx in df.groupby(["subject", "representation"]).groups.items():
    df.loc[idx, "significant_in_participant"] = multipletests(df.loc[idx, "p_binomial"], alpha=0.05, method="fdr_bh")[0]
df.to_csv(RESULTS / "parcelwise_by_subject.csv", index=False)


# group level: one-sided t-test against 50% for each parcel, BH across the 800 parcels
group_rows = []
for (rep, parcel), g in df.groupby(["representation", "parcel"]):
    t, p = stats.ttest_1samp(g["accuracy"], 0.5, alternative=T_TEST_SIDE)
    group_rows.append({"representation": rep, "parcel": parcel,
                       "mean_accuracy": g["accuracy"].mean(), "t": t, "p": p})
group = pd.DataFrame(group_rows)
group["q"] = np.nan
for rep, idx in group.groupby("representation").groups.items():
    group.loc[idx, "q"] = multipletests(group.loc[idx, "p"], method="fdr_bh")[1]
group["significant"] = (group["q"] <= Q_THRESHOLD) & (group["t"] > 0)
group.to_csv(RESULTS / "parcelwise_group.csv", index=False)

print(f"\nparcels distinguishing Chinese from English (group level, {T_TEST_SIDE} t-test, q <= {Q_THRESHOLD})")
for rep in ["voxels", "demeaned", "mean"]:
    sig = group[(group["representation"] == rep) & group["significant"]]
    print(f"{rep:<10}{len(sig):4d} of 800   mean accuracy in these parcels = {100 * sig['mean_accuracy'].mean():.1f}%")
