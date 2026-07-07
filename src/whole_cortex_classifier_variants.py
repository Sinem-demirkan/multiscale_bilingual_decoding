"""
Analysis script for within-subject whole-cortex Schaefer decoding and
network-level follow-up analyses in the Guo et al. bilingual picture-naming
dataset (OpenNeuro ds005455).

This script supports three different analysis types. They use the same input
data and the same subject-level trial structure, but they answer different
questions.

----------------------------------------------------------------------
1. whole_cortex
----------------------------------------------------------------------

Purpose:
Estimate how well each subject can be decoded from whole-cortex Schaefer parcel
means.

What happens:
- Single-trial beta images are converted to Schaefer parcel means.
- Trials are relabeled for either:
    - language decoding: L1 vs L2
    - switch decoding: S vs NS
- A classifier is trained and tested within subject using leave-one-run-out
  cross-validation.

Main options:
- N_ROIS:
    Number of Schaefer parcels. Example: 800 for the paper setting.
- YEO_NETWORKS:
    Atlas version used to define parcels. Usually 7 for the main paper model.
- FEATURE_VARIANT:
    - "raw": use parcel means directly
    - "pca": apply PCA to the parcel-mean feature matrix before classification
- PCA_COMPONENTS:
    Only used when FEATURE_VARIANT == "pca". Can be a variance proportion such
    as 0.90.

What this analysis outputs:
- One balanced-accuracy value per subject
- A CSV file summarizing subject-level decoding scores

Use this when:
- You want the main self whole-cortex decoding model
- You want to compare atlas sizes
- You want to compare raw parcel-mean models with PCA-reduced parcel-mean models

----------------------------------------------------------------------
2. size_matched_grouped_pfi
----------------------------------------------------------------------

Purpose:
Estimate which functional networks contribute most to whole-cortex decoding,
while controlling for network size.

What happens:
- First, the subject's whole-cortex parcel-mean model is fit.
- Parcels are grouped into networks, for example Yeo7 or Yeo17 networks.
- Because different networks contain different numbers of parcels, each network
  is repeatedly downsampled to the same parcel count.
- Grouped permutation importance is then computed by permuting the selected
  parcels for one network at a time and measuring the drop in decoding accuracy.

Main options:
- GROUP_FAMILY:
    - "yeo7": use Yeo 7-network grouping
    - "yeo17": use Yeo 17-network grouping
- N_SUBSETS:
    Number of random equal-size subsets to sample per group. Larger values are
    more stable but slower.
- N_PERM_REPEATS:
    Number of permutation repeats per fold and subset.

What this analysis outputs:
- A CSV defining the sampled matched-size subsets
- A fold-level CSV of grouped permutation importance values
- A subject-level CSV averaging importance within subject
- A summary CSV averaging importance across subjects

Use this when:
- You want to compare networks fairly, without larger networks automatically
  looking more important just because they contain more parcels
- You want a network-level importance analysis that is size-controlled

Important note:
- This is not a decoding accuracy analysis.
- It is an importance analysis performed on top of the whole-cortex model.

----------------------------------------------------------------------
3. network_pca_grouped_pfi
----------------------------------------------------------------------

Purpose:
Estimate network-level importance after compressing each network into its own
lower-dimensional PCA representation.

What happens:
- Whole-cortex parcel means are extracted as usual.
- Parcels are split into networks, for example Yeo7 or Yeo17.
- PCA is run separately within each network.
- The network-specific PCs are concatenated into one combined feature matrix.
- The classifier is fit on that combined matrix.
- Grouped permutation importance is then computed by permuting all PCs from one
  network at a time.

Main options:
- GROUP_FAMILY:
    - "yeo7"
    - "yeo17"
- PCA_VARIANCE:
    Variance threshold for the within-network PCA step. Example: 0.90 means
    keep enough PCs within each network to explain 90% of that network's
    variance.
- N_PERM_REPEATS:
    Number of permutation repeats per fold.

What this analysis outputs:
- A fold-level CSV of network PCA grouped permutation importance values
- A subject-level CSV averaging importance within subject
- A summary CSV averaging importance across subjects
- A PC metadata CSV describing how many PCs each network contributed and how
  much variance they explained

Use this when:
- You want a network-level importance analysis that reduces within-network
  dimensionality before importance testing
- You want to ask whether a network remains important after representing it by
  its dominant internal components rather than all raw parcels

Important note:
- This is different from FEATURE_VARIANT == "pca" in whole_cortex.
- FEATURE_VARIANT == "pca" applies PCA to the entire whole-cortex feature
  matrix.
- network_pca_grouped_pfi applies PCA separately within each network, then
  tests network-level importance.

----------------------------------------------------------------------
Choosing the right analysis
----------------------------------------------------------------------

Use:
- ANALYSIS = "whole_cortex"
  if your goal is subject-level decoding accuracy

Use:
- ANALYSIS = "size_matched_grouped_pfi"
  if your goal is network importance with equal-size parcel groups

Use:
- ANALYSIS = "network_pca_grouped_pfi"
  if your goal is network importance after compressing each network with PCA

----------------------------------------------------------------------
Most important settings to edit
----------------------------------------------------------------------

- DATA_ROOT:
    Path to the local duo-cogcon dataset
- TASK:
    Usually "LanguageControl"
- START_SUB, END_SUB:
    Subject range to analyze
- TARGET:
    - "language" for L1 vs L2
    - "switch" for S vs NS
- ANALYSIS:
    Which analysis to run
- N_ROIS:
    Atlas size
- YEO_NETWORKS:
    Atlas version for parcel definition
- GROUP_FAMILY:
    Network definition for grouped-PFI analyses

----------------------------------------------------------------------
Examples
----------------------------------------------------------------------

Main paper-like whole-cortex decoding:
- ANALYSIS = "whole_cortex"
- TARGET = "language"
- N_ROIS = 800
- YEO_NETWORKS = 7
- FEATURE_VARIANT = "raw"

Atlas-size PCA variant:
- ANALYSIS = "whole_cortex"
- FEATURE_VARIANT = "pca"
- N_ROIS = 100, 200, 400, 800, ...

Yeo7 size-matched grouped PFI:
- ANALYSIS = "size_matched_grouped_pfi"
- GROUP_FAMILY = "yeo7"

Yeo17 network PCA grouped PFI:
- ANALYSIS = "network_pca_grouped_pfi"
- GROUP_FAMILY = "yeo17"
"""

from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from nilearn import image
from nilearn.datasets import fetch_atlas_schaefer_2018
from nilearn.maskers import NiftiLabelsMasker
from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


# Path to the root of your duo-cogcon dataset folder
# (should contain the sub-XXX single-trial beta images and events.tsv files)
DATA_ROOT = Path("/home/sdemirka/fmri/duo-cogcon")
OUT_DIR = Path("/home/sdemirka/fmri/duo-cogcon/outputs")

TASK = "LanguageControl"
START_SUB = 1
END_SUB = 77

TARGET = "language"   # "language" or "switch"
TRIAL_COLUMN = "trial_type"

ANALYSIS = "whole_cortex"   # "whole_cortex", "size_matched_grouped_pfi", "network_pca_grouped_pfi"

N_ROIS = 800
YEO_NETWORKS = 7            # 7 or 17
RESOLUTION_MM = 2

FEATURE_VARIANT = "raw"     # only used for ANALYSIS == "whole_cortex": "raw" or "pca"
PCA_COMPONENTS = 0.90       # only used for FEATURE_VARIANT == "pca"

GROUP_FAMILY = "yeo7"       # "yeo7" or "yeo17" for grouped-PFI analyses
N_SUBSETS = 500             # only used for size-matched grouped PFI
N_PERM_REPEATS = 20         # used for grouped PFI analyses
PCA_VARIANCE = 0.90         # only used for network_pca_grouped_pfi

RANDOM_STATE = 42


def subject_id(n):
    return f"sub-{n:03d}"


def relabel_trials(trial_labels, target):
    trial_labels = pd.Series(trial_labels).astype(str)

    if target == "language":
        mapping = {
            "L1S": "L1",
            "L1NS": "L1",
            "L2S": "L2",
            "L2NS": "L2",
        }
    elif target == "switch":
        mapping = {
            "L1S": "S",
            "L2S": "S",
            "L1NS": "NS",
            "L2NS": "NS",
        }
    else:
        raise ValueError("TARGET must be 'language' or 'switch'")

    y = trial_labels.map(mapping)
    keep = y.notna().to_numpy()
    return y, keep


def fetch_schaefer_atlas(n_rois, yeo_networks):
    return fetch_atlas_schaefer_2018(
        n_rois=n_rois,
        yeo_networks=yeo_networks,
        resolution_mm=RESOLUTION_MM,
    )


def make_masker(atlas):
    return NiftiLabelsMasker(
        labels_img=atlas.maps,
        standardize=False,
        strategy="mean",
        resampling_target="data",
        verbose=0,
    )


def parse_run_number(path):
    return int(path.name.split("run-")[1].split("_")[0])


def load_subject_parcel_data(subject, masker):
    beta_files = sorted(
        DATA_ROOT.glob(f"**/{subject}_task-{TASK}_run-*_singletrial-Act.nii.gz")
    )
    event_files = sorted(
        DATA_ROOT.glob(f"**/{subject}_task-{TASK}_run-*_events.tsv")
    )

    if len(beta_files) == 0 or len(event_files) == 0:
        return None, None, None

    beta_by_run = {parse_run_number(p): p for p in beta_files}
    event_by_run = {parse_run_number(p): p for p in event_files}
    runs = sorted(set(beta_by_run) & set(event_by_run))

    trial_imgs = []
    labels = []
    groups = []

    for run in runs:
        beta_file = beta_by_run[run]
        event_file = event_by_run[run]

        events = pd.read_csv(event_file, sep="\t")
        img = nib.load(beta_file)

        if TRIAL_COLUMN not in events.columns:
            print(subject, f"missing column {TRIAL_COLUMN} in {event_file.name}")
            continue

        if img.shape[-1] != len(events):
            print(subject, f"run {run} mismatch")
            continue

        y_full, keep = relabel_trials(events[TRIAL_COLUMN], TARGET)
        keep_idx = np.where(keep)[0]
        if len(keep_idx) == 0:
            continue

        for idx in keep_idx:
            trial_imgs.append(img.slicer[..., idx])

        labels.extend(y_full.iloc[keep_idx].tolist())
        groups.extend([run] * len(keep_idx))

    if len(trial_imgs) == 0:
        return None, None, None

    X = masker.fit_transform(image.concat_imgs(trial_imgs))
    y = np.array(labels)
    groups = np.array(groups)

    if len(np.unique(y)) < 2:
        print(subject, "only one class left after relabeling")
        return None, None, None

    return X, y, groups


def make_classifier():
    return LogisticRegression(
        penalty="l2",
        C=0.01,
        solver="lbfgs",
        max_iter=1000,
        random_state=RANDOM_STATE,
    )


def run_whole_cortex_decoding(X, y, groups):
    if FEATURE_VARIANT == "raw":
        clf = make_pipeline(
            StandardScaler(),
            make_classifier(),
        )
    elif FEATURE_VARIANT == "pca":
        clf = make_pipeline(
            StandardScaler(),
            PCA(n_components=PCA_COMPONENTS),
            make_classifier(),
        )
    else:
        raise ValueError("FEATURE_VARIANT must be 'raw' or 'pca'")

    logo = LeaveOneGroupOut()
    scores = []

    for tr, te in logo.split(X, y, groups=groups):
        model = clone(clf)
        model.fit(X[tr], y[tr])
        yhat = model.predict(X[te])
        scores.append(balanced_accuracy_score(y[te], yhat))

    return float(np.mean(scores))


def decode_label(x):
    if isinstance(x, bytes):
        return x.decode("utf-8")
    return str(x)


def build_group_map(n_rois, group_family):
    if group_family == "yeo7":
        atlas = fetch_schaefer_atlas(n_rois, yeo_networks=7)
    elif group_family == "yeo17":
        atlas = fetch_schaefer_atlas(n_rois, yeo_networks=17)
    else:
        raise ValueError("GROUP_FAMILY must be 'yeo7' or 'yeo17'")

    labels = [decode_label(x) for x in atlas.labels]
    group_map = {}

    for atlas_idx, label in enumerate(labels):
        if atlas_idx == 0 or label.lower() == "background":
            continue

        parts = label.split("_")
        if len(parts) < 4:
            continue

        group = parts[2]
        parcel_col = atlas_idx - 1
        group_map.setdefault(group, []).append(parcel_col)

    return {k: sorted(v) for k, v in group_map.items()}


def build_size_matched_subsets(group_map, n_subsets, random_state):
    rng = np.random.default_rng(random_state)
    target_n = min(len(v) for v in group_map.values())

    subset_defs = []
    subset_rows = []

    for subset_id in range(n_subsets):
        subset_map = {}
        for group, idxs in group_map.items():
            chosen = sorted(rng.choice(idxs, size=target_n, replace=False).tolist())
            subset_map[group] = chosen
            for parcel_idx in chosen:
                subset_rows.append({
                    "subset_id": subset_id,
                    "group": group,
                    "parcel_index": parcel_idx,
                })
        subset_defs.append(subset_map)

    return subset_defs, pd.DataFrame(subset_rows)


def permuted_group_drop(clf, X_test, y_test, cols, base_score, n_repeats, rng):
    drops = []

    for _ in range(n_repeats):
        Xp = X_test.copy()
        perm_idx = rng.permutation(Xp.shape[0])
        Xp[:, cols] = Xp[perm_idx][:, cols]
        score = balanced_accuracy_score(y_test, clf.predict(Xp))
        drops.append(base_score - score)

    return float(np.mean(drops))


def run_size_matched_grouped_pfi(subject, X, y, groups, subset_defs):
    logo = LeaveOneGroupOut()
    rng = np.random.default_rng(RANDOM_STATE)
    fold_rows = []

    for fold_id, (tr, te) in enumerate(logo.split(X, y, groups=groups)):
        scaler = StandardScaler()
        Xtr = scaler.fit_transform(X[tr])
        Xte = scaler.transform(X[te])

        clf = make_classifier()
        clf.fit(Xtr, y[tr])
        base_score = balanced_accuracy_score(y[te], clf.predict(Xte))

        for subset_id, subset_map in enumerate(subset_defs):
            for group, cols in subset_map.items():
                drop = permuted_group_drop(
                    clf=clf,
                    X_test=Xte,
                    y_test=y[te],
                    cols=cols,
                    base_score=base_score,
                    n_repeats=N_PERM_REPEATS,
                    rng=rng,
                )
                fold_rows.append({
                    "subject": subject,
                    "fold_id": fold_id,
                    "test_run": int(groups[te][0]),
                    "group_family": GROUP_FAMILY,
                    "group": group,
                    "subset_id": subset_id,
                    "base_score": base_score,
                    "importance_drop": drop,
                })

    return pd.DataFrame(fold_rows)


def run_network_pca_grouped_pfi(subject, X, y, groups, group_map):
    logo = LeaveOneGroupOut()
    rng = np.random.default_rng(RANDOM_STATE)
    fold_rows = []
    pc_meta_rows = []

    for fold_id, (tr, te) in enumerate(logo.split(X, y, groups=groups)):
        Xtr = X[tr]
        Xte = X[te]

        Ztr_blocks = []
        Zte_blocks = []
        pc_group_cols = {}
        col_start = 0

        for group, cols in group_map.items():
            scaler = StandardScaler()
            Xtr_g = scaler.fit_transform(Xtr[:, cols])
            Xte_g = scaler.transform(Xte[:, cols])

            pca = PCA(n_components=PCA_VARIANCE, svd_solver="full")
            Ztr_g = pca.fit_transform(Xtr_g)
            Zte_g = pca.transform(Xte_g)

            n_pc = Ztr_g.shape[1]
            pc_cols = list(range(col_start, col_start + n_pc))
            pc_group_cols[group] = pc_cols
            col_start += n_pc

            Ztr_blocks.append(Ztr_g)
            Zte_blocks.append(Zte_g)

            for pc_idx in range(n_pc):
                pc_meta_rows.append({
                    "subject": subject,
                    "fold_id": fold_id,
                    "group_family": GROUP_FAMILY,
                    "group": group,
                    "pc_index_within_group": pc_idx,
                    "explained_variance_ratio": float(pca.explained_variance_ratio_[pc_idx]),
                })

        Ztr = np.hstack(Ztr_blocks)
        Zte = np.hstack(Zte_blocks)

        clf = make_classifier()
        clf.fit(Ztr, y[tr])
        base_score = balanced_accuracy_score(y[te], clf.predict(Zte))

        for group, pc_cols in pc_group_cols.items():
            drop = permuted_group_drop(
                clf=clf,
                X_test=Zte,
                y_test=y[te],
                cols=pc_cols,
                base_score=base_score,
                n_repeats=N_PERM_REPEATS,
                rng=rng,
            )
            fold_rows.append({
                "subject": subject,
                "fold_id": fold_id,
                "test_run": int(groups[te][0]),
                "group_family": GROUP_FAMILY,
                "group": group,
                "base_score": base_score,
                "importance_drop": drop,
            })

    return pd.DataFrame(fold_rows), pd.DataFrame(pc_meta_rows)


atlas = fetch_schaefer_atlas(N_ROIS, YEO_NETWORKS)
masker = make_masker(atlas)
OUT_DIR.mkdir(parents=True, exist_ok=True)

if ANALYSIS == "whole_cortex":
    rows = []

    for sub in range(START_SUB, END_SUB + 1):
        subject = subject_id(sub)
        X, y, groups = load_subject_parcel_data(subject, masker)

        if X is None:
            print(subject, "missing or unusable data")
            continue

        score = run_whole_cortex_decoding(X, y, groups)
        rows.append({
            "subject": subject,
            "target": TARGET,
            "n_rois": N_ROIS,
            "yeo_networks": YEO_NETWORKS,
            "feature_variant": FEATURE_VARIANT,
            "balanced_accuracy": score,
            "n_trials": len(y),
        })
        print(subject, round(score, 5))

    results = pd.DataFrame(rows)
    out_csv = f"self_schaefer{N_ROIS}_yeo{YEO_NETWORKS}_{FEATURE_VARIANT}_{TARGET}_results.csv"
    results.to_csv(OUT_DIR / out_csv, index=False)
    print(results)

elif ANALYSIS == "size_matched_grouped_pfi":
    group_map = build_group_map(N_ROIS, GROUP_FAMILY)
    subset_defs, subset_df = build_size_matched_subsets(group_map, N_SUBSETS, RANDOM_STATE)

    all_folds = []

    for sub in range(START_SUB, END_SUB + 1):
        subject = subject_id(sub)
        X, y, groups = load_subject_parcel_data(subject, masker)

        if X is None:
            print(subject, "missing or unusable data")
            continue

        fold_df = run_size_matched_grouped_pfi(subject, X, y, groups, subset_defs)
        all_folds.append(fold_df)
        print(subject, "done")

    folds_df = pd.concat(all_folds, ignore_index=True)
    by_subject_df = (
        folds_df.groupby(["subject", "group_family", "group"], as_index=False)["importance_drop"]
        .mean()
        .rename(columns={"importance_drop": "mean_importance_drop"})
    )
    summary_df = (
        by_subject_df.groupby(["group_family", "group"], as_index=False)["mean_importance_drop"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    summary_df.columns = ["group_family", "group", "mean_importance_drop", "sd", "n_subjects"]

    base = f"size_matched_grouped_pfi_schaefer{N_ROIS}_{GROUP_FAMILY}_{TARGET}"
    subset_df.to_csv(OUT_DIR / f"{base}_subsets.csv", index=False)
    folds_df.to_csv(OUT_DIR / f"{base}_folds.csv", index=False)
    by_subject_df.to_csv(OUT_DIR / f"{base}_by_subject.csv", index=False)
    summary_df.to_csv(OUT_DIR / f"{base}_summary.csv", index=False)
    print(summary_df)

elif ANALYSIS == "network_pca_grouped_pfi":
    group_map = build_group_map(N_ROIS, GROUP_FAMILY)

    all_folds = []
    all_pc_meta = []

    for sub in range(START_SUB, END_SUB + 1):
        subject = subject_id(sub)
        X, y, groups = load_subject_parcel_data(subject, masker)

        if X is None:
            print(subject, "missing or unusable data")
            continue

        fold_df, pc_meta_df = run_network_pca_grouped_pfi(subject, X, y, groups, group_map)
        all_folds.append(fold_df)
        all_pc_meta.append(pc_meta_df)
        print(subject, "done")

    folds_df = pd.concat(all_folds, ignore_index=True)
    pc_meta_df = pd.concat(all_pc_meta, ignore_index=True)

    by_subject_df = (
        folds_df.groupby(["subject", "group_family", "group"], as_index=False)["importance_drop"]
        .mean()
        .rename(columns={"importance_drop": "mean_importance_drop"})
    )
    summary_df = (
        by_subject_df.groupby(["group_family", "group"], as_index=False)["mean_importance_drop"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    summary_df.columns = ["group_family", "group", "mean_importance_drop", "sd", "n_subjects"]

    base = f"network_pca_grouped_pfi_schaefer{N_ROIS}_{GROUP_FAMILY}_{TARGET}"
    folds_df.to_csv(OUT_DIR / f"{base}_folds.csv", index=False)
    by_subject_df.to_csv(OUT_DIR / f"{base}_by_subject.csv", index=False)
    summary_df.to_csv(OUT_DIR / f"{base}_summary.csv", index=False)
    pc_meta_df.to_csv(OUT_DIR / f"{base}_pc_meta.csv", index=False)
    print(summary_df)

else:
    raise ValueError("ANALYSIS must be 'whole_cortex', 'size_matched_grouped_pfi', or 'network_pca_grouped_pfi'")
