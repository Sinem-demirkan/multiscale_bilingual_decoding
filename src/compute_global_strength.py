from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from common import (
    balanced_accuracy_from_predictions,
    collect_subject_runs,
    l1_l2_sensitivity,
    load_subject_trials,
    load_template_and_atlas,
    make_estimator,
    cv_predictions,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Compute whole-cortex decoding strength.")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--subjects", nargs="*", default=None)
    return parser.parse_args()


def compute_parcel_means(X_voxels: np.ndarray, atlas_data: np.ndarray, n_rois: int = 800):
    X_parcels = np.zeros((X_voxels.shape[0], n_rois), dtype=np.float32)
    for parcel in range(1, n_rois + 1):
        vox = np.flatnonzero(atlas_data == parcel)
        if len(vox) == 0:
            continue
        X_parcels[:, parcel - 1] = X_voxels[:, vox].mean(axis=1)
    return X_parcels


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    template, _, atlas_data = load_template_and_atlas(n_rois=800, resolution_mm=2)
    subjects = collect_subject_runs(args.data_root)
    chosen_subjects = sorted(subjects)
    if args.subjects:
        requested = set(args.subjects)
        chosen_subjects = [s for s in chosen_subjects if s in requested]

    estimator = make_estimator(C=0.01)

    rows = []
    fold_rows = []
    skipped = []

    for subject in chosen_subjects:
        try:
            X_voxels, y, groups = load_subject_trials(args.data_root, subject, subjects[subject], template)
            X_parcels = compute_parcel_means(X_voxels, atlas_data, n_rois=800)
            y_true, y_pred, folds = cv_predictions(X_parcels, y, groups, estimator)
            bal_acc = balanced_accuracy_from_predictions(y_true, y_pred)
            l1_sens, l2_sens = l1_l2_sensitivity(y_true, y_pred)

            rows.append(
                {
                    "subject": subject,
                    "balanced_accuracy": bal_acc,
                    "accuracy_minus_chance": bal_acc - 0.5,
                    "l1_sensitivity": l1_sens,
                    "l2_sensitivity": l2_sens,
                    "n_trials": int(len(y)),
                    "n_L1": int(np.sum(y == "L1")),
                    "n_L2": int(np.sum(y == "L2")),
                    "n_features": X_parcels.shape[1],
                }
            )
            for fold in folds:
                fold_rows.append({"subject": subject, **fold})
            print(f"Finished {subject}: strength={bal_acc:.4f}", flush=True)
        except Exception as exc:
            skipped.append({"subject": subject, "error": str(exc)})
            print(f"Skipped {subject}: {exc}", flush=True)

    results = pd.DataFrame(rows).sort_values("subject")
    folds = pd.DataFrame(fold_rows).sort_values(["subject", "fold"])
    skipped_df = pd.DataFrame(skipped)

    results.to_csv(args.out_dir / "whole_cortex_strength_by_subject.csv", index=False)
    folds.to_csv(args.out_dir / "whole_cortex_strength_folds.csv", index=False)
    skipped_df.to_csv(args.out_dir / "skipped_subjects.csv", index=False)

    if len(results):
        t, p = stats.ttest_1samp(results["accuracy_minus_chance"], 0.0)
        ci = stats.t.interval(
            0.95,
            len(results) - 1,
            loc=results["balanced_accuracy"].mean(),
            scale=stats.sem(results["balanced_accuracy"]),
        )
        summary = pd.DataFrame(
            [
                {
                    "n_subjects": len(results),
                    "mean_balanced_accuracy": results["balanced_accuracy"].mean(),
                    "median_balanced_accuracy": results["balanced_accuracy"].median(),
                    "sd_balanced_accuracy": results["balanced_accuracy"].std(ddof=1),
                    "min_balanced_accuracy": results["balanced_accuracy"].min(),
                    "max_balanced_accuracy": results["balanced_accuracy"].max(),
                    "ci_low_balanced_accuracy": ci[0],
                    "ci_high_balanced_accuracy": ci[1],
                    "t_vs_chance": t,
                    "p_vs_chance": p,
                }
            ]
        )
        summary.to_csv(args.out_dir / "whole_cortex_strength_summary.csv", index=False)


if __name__ == "__main__":
    main()
