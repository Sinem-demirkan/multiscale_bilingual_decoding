"""
Within-subject parcelwise Schaefer decoding.

For each participant, this script trains one parcelwise L1-vs-L2 classifier per
Schaefer parcel and summarizes local cortical extent as the proportion of
parcels with one-sided binomial z > 1.64.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from nilearn import datasets, image
from scipy.stats import binom, norm
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


RANDOM_STATE = 42


def parse_args():
    parser = argparse.ArgumentParser(description="Run parcelwise local decoding.")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--task", default="LanguageControl")
    parser.add_argument("--trial-column", default="trial_type")
    parser.add_argument("--start-sub", type=int, default=1)
    parser.add_argument("--end-sub", type=int, default=77)
    parser.add_argument("--n-rois", type=int, default=800)
    parser.add_argument("--yeo-networks", type=int, default=17)
    parser.add_argument("--resolution-mm", type=int, default=2)
    parser.add_argument("--accuracy-threshold", type=float, default=0.60)
    parser.add_argument("--z-threshold", type=float, default=1.64)
    return parser.parse_args()


def subject_id(n: int) -> str:
    return f"sub-{n:03d}"


def parse_run_number(path: Path) -> int:
    return int(path.name.split("run-")[1].split("_")[0])


def language_label(trial_type):
    trial_type = str(trial_type)
    if trial_type.startswith("L1"):
        return "L1"
    if trial_type.startswith("L2"):
        return "L2"
    return np.nan


def one_sided_binomial_p(accuracy: float, n_trials: int) -> tuple[int, float]:
    n_correct = int(np.rint(accuracy * n_trials))
    p = float(binom.sf(n_correct - 1, n_trials, 0.5))
    return n_correct, p


def cv_balanced_accuracy(X, y, run_groups, estimator):
    cv = LeaveOneGroupOut()
    scores = []

    for train_idx, test_idx in cv.split(X, y, groups=run_groups):
        clf = clone(estimator)
        clf.fit(X[train_idx], y[train_idx])
        yhat = clf.predict(X[test_idx])
        scores.append(balanced_accuracy_score(y[test_idx], yhat))

    return float(np.mean(scores))


def load_subject_data(subject: str, template, args):
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

    run_arrays = []
    run_labels = []
    run_groups = []

    for run in runs:
        beta_file = beta_by_run[run]
        event_file = event_by_run[run]

        img = image.resample_to_img(
            nib.load(beta_file),
            template,
            interpolation="continuous",
            force_resample=True,
            copy_header=True,
        )
        events = pd.read_csv(event_file, sep="\t")

        if args.trial_column not in events.columns:
            print(subject, f"missing column {args.trial_column} in {event_file.name}")
            continue

        events["language"] = events[args.trial_column].map(language_label)
        events = events.dropna(subset=["language"]).reset_index(drop=True)

        if len(events) != img.shape[-1]:
            print(subject, f"run {run} mismatch")
            continue

        data = np.asarray(img.dataobj, dtype=np.float32)
        run_arrays.append(data.reshape(-1, data.shape[-1]).T)
        run_labels.append(events["language"].to_numpy())
        run_groups.append(np.repeat(run, len(events)))

    if len(run_arrays) < 2:
        return None, None, None

    X = np.vstack(run_arrays)
    y = np.concatenate(run_labels)
    groups = np.concatenate(run_groups)

    if set(np.unique(y)) != {"L1", "L2"}:
        return None, None, None

    return X, y, groups


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    template = datasets.load_mni152_template(resolution=args.resolution_mm)
    atlas = datasets.fetch_atlas_schaefer_2018(
        n_rois=args.n_rois,
        yeo_networks=args.yeo_networks,
        resolution_mm=args.resolution_mm,
    )
    atlas_img = image.resample_to_img(
        nib.load(atlas.maps),
        template,
        interpolation="nearest",
        force_resample=True,
        copy_header=True,
    )
    atlas_data = np.asarray(atlas_img.dataobj).astype(np.int16).ravel()

    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            penalty="l2",
            C=0.01,
            solver="lbfgs",
            max_iter=1000,
            random_state=RANDOM_STATE,
        ),
    )

    rows = []
    summary_rows = []
    skipped = []

    for sub in range(args.start_sub, args.end_sub + 1):
        subject = subject_id(sub)
        X, y, groups = load_subject_data(subject, template, args)

        if X is None:
            skipped.append({"subject": subject, "reason": "missing_or_unusable_data"})
            print(subject, "missing or unusable data")
            continue

        subject_rows = []
        print(f"Running {subject}: {len(y)} trials", flush=True)

        for parcel in range(1, args.n_rois + 1):
            vox = np.flatnonzero(atlas_data == parcel)

            if len(vox) < 2:
                acc = np.nan
                n_correct = np.nan
                p_binom = np.nan
                z_one_sided = np.nan
            else:
                acc = cv_balanced_accuracy(X[:, vox], y, groups, clf)
                n_correct, p_binom = one_sided_binomial_p(acc, len(y))
                z_one_sided = float(norm.isf(np.clip(p_binom, 1e-300, 1 - 1e-16)))

            subject_rows.append(
                {
                    "subject": subject,
                    "parcel": parcel,
                    "accuracy": acc,
                    "accuracy_minus_chance": acc - 0.5 if np.isfinite(acc) else np.nan,
                    "n_voxels": int(len(vox)),
                    "n_trials": int(len(y)),
                    "n_L1": int(np.sum(y == "L1")),
                    "n_L2": int(np.sum(y == "L2")),
                    "n_correct_approx": n_correct,
                    "p_binom_one_sided": p_binom,
                    "z_one_sided": z_one_sided,
                }
            )

        subject_df = pd.DataFrame(subject_rows)
        subject_df["accuracy_gt_threshold"] = subject_df["accuracy"] > args.accuracy_threshold
        subject_df["z_gt_1p64"] = subject_df["z_one_sided"] > args.z_threshold
        rows.extend(subject_rows)

        summary_rows.append(
            {
                "subject": subject,
                "n_parcels": args.n_rois,
                "n_accuracy_gt_threshold": int(subject_df["accuracy_gt_threshold"].sum()),
                "prop_accuracy_gt_threshold": float(subject_df["accuracy_gt_threshold"].mean()),
                "n_z_gt_1p64": int(subject_df["z_gt_1p64"].sum()),
                "prop_z_gt_1p64": float(subject_df["z_gt_1p64"].mean()),
                "mean_accuracy": float(subject_df["accuracy"].mean()),
                "max_accuracy": float(subject_df["accuracy"].max()),
            }
        )

    long_df = pd.DataFrame(rows)
    summary_df = pd.DataFrame(summary_rows).sort_values("subject") if summary_rows else pd.DataFrame()
    skipped_df = pd.DataFrame(skipped)

    long_df.to_csv(args.out_dir / "schaefer800_parcelwise_multivoxel_by_subject_and_parcel.csv", index=False)
    summary_df.to_csv(args.out_dir / "local_z_extent_by_subject.csv", index=False)
    skipped_df.to_csv(args.out_dir / "schaefer800_parcelwise_multivoxel_skipped_subjects.csv", index=False)


if __name__ == "__main__":
    main()
