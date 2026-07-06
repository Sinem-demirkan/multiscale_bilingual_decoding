"""
Within-subject whole-cortex Schaefer parcel-mean decoding.

This script computes the main self-decoding measure used as whole-cortex
decoding strength. It extracts one mean feature per Schaefer parcel from each
single-trial beta image and evaluates L1-vs-L2 language decoding with
leave-one-run-out cross-validation.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from nilearn import image
from nilearn.datasets import fetch_atlas_schaefer_2018
from nilearn.maskers import NiftiLabelsMasker
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler


RANDOM_STATE = 42


def parse_args():
    parser = argparse.ArgumentParser(description="Run within-subject whole-cortex decoding.")
    parser.add_argument("--data-root", type=Path, required=True, help="Path to duo-cogcon dataset root.")
    parser.add_argument("--out-dir", type=Path, required=True, help="Directory for output CSV files.")
    parser.add_argument("--task", default="LanguageControl")
    parser.add_argument("--trial-column", default="trial_type")
    parser.add_argument("--start-sub", type=int, default=1)
    parser.add_argument("--end-sub", type=int, default=77)
    parser.add_argument("--n-rois", type=int, default=800)
    parser.add_argument("--yeo-networks", type=int, default=7, choices=[7, 17])
    parser.add_argument("--resolution-mm", type=int, default=2)
    return parser.parse_args()


def subject_id(n: int) -> str:
    return f"sub-{n:03d}"


def parse_run_number(path: Path) -> int:
    return int(path.name.split("run-")[1].split("_")[0])


def make_labels(trial_labels):
    trial_labels = pd.Series(trial_labels).astype(str)
    labels = trial_labels.map(
        lambda x: "L1" if x.startswith("L1") else ("L2" if x.startswith("L2") else np.nan)
    )
    keep = labels.notna().to_numpy()
    return labels.to_numpy(), keep


def load_subject_data(subject: str, masker, args):
    beta_files = sorted(
        args.data_root.glob(f"**/{subject}_task-{args.task}_run-*_singletrial-Act.nii.gz")
    )
    event_files = sorted(
        args.data_root.glob(f"**/{subject}_task-{args.task}_run-*_events.tsv")
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

        if args.trial_column not in events.columns:
            print(subject, f"missing column {args.trial_column} in {event_file.name}")
            continue

        if img.shape[-1] != len(events):
            print(subject, f"run {run} mismatch")
            continue

        y_full, keep = make_labels(events[args.trial_column])
        keep_idx = np.where(keep)[0]
        if len(keep_idx) == 0:
            continue

        for idx in keep_idx:
            trial_imgs.append(img.slicer[..., idx])
        labels.extend(y_full[keep_idx].tolist())
        groups.extend([run] * len(keep_idx))

    if len(trial_imgs) == 0:
        return None, None, None

    X = masker.fit_transform(image.concat_imgs(trial_imgs))
    y = np.asarray(labels)
    groups = np.asarray(groups)

    if set(np.unique(y)) != {"L1", "L2"}:
        print(subject, "missing one language class")
        return None, None, None

    return X, y, groups


def run_subject_cv(X, y, groups):
    clf = LogisticRegression(
        penalty="l2",
        C=0.01,
        solver="lbfgs",
        max_iter=1000,
        random_state=RANDOM_STATE,
    )

    logo = LeaveOneGroupOut()
    fold_rows = []
    scores = []

    for fold, (train_idx, test_idx) in enumerate(logo.split(X, y, groups=groups), start=1):
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X[train_idx])
        X_test = scaler.transform(X[test_idx])

        clf.fit(X_train, y[train_idx])
        y_pred = clf.predict(X_test)
        score = balanced_accuracy_score(y[test_idx], y_pred)
        scores.append(score)
        fold_rows.append(
            {
                "fold": fold,
                "test_run": int(groups[test_idx][0]),
                "balanced_accuracy": float(score),
            }
        )

    return float(np.mean(scores)), fold_rows


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    atlas = fetch_atlas_schaefer_2018(
        n_rois=args.n_rois,
        yeo_networks=args.yeo_networks,
        resolution_mm=args.resolution_mm,
    )

    masker = NiftiLabelsMasker(
        labels_img=atlas.maps,
        standardize=False,
        strategy="mean",
        resampling_target="data",
        verbose=0,
    )

    rows = []
    fold_rows = []
    skipped = []

    for sub in range(args.start_sub, args.end_sub + 1):
        subject = subject_id(sub)
        X, y, groups = load_subject_data(subject, masker, args)

        if X is None:
            print(subject, "missing or unusable data")
            skipped.append({"subject": subject, "reason": "missing_or_unusable_data"})
            continue

        score, folds = run_subject_cv(X, y, groups)

        rows.append(
            {
                "subject": subject,
                "balanced_accuracy": score,
                "accuracy_minus_chance": score - 0.5,
                "n_trials": int(len(y)),
                "n_L1": int(np.sum(y == "L1")),
                "n_L2": int(np.sum(y == "L2")),
                "n_runs": int(len(np.unique(groups))),
                "n_rois": args.n_rois,
                "yeo_networks": args.yeo_networks,
            }
        )
        for row in folds:
            fold_rows.append({"subject": subject, **row})
        print(subject, round(score, 5), flush=True)

    results = pd.DataFrame(rows).sort_values("subject") if rows else pd.DataFrame()
    folds = pd.DataFrame(fold_rows).sort_values(["subject", "fold"]) if fold_rows else pd.DataFrame()
    skipped_df = pd.DataFrame(skipped)

    results.to_csv(args.out_dir / "distributed_parcel_mean_decoding_by_subject.csv", index=False)
    folds.to_csv(args.out_dir / "distributed_parcel_mean_decoding_folds.csv", index=False)
    skipped_df.to_csv(args.out_dir / "distributed_parcel_mean_skipped_subjects.csv", index=False)


if __name__ == "__main__":
    main()
