from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binom, norm

from common import (
    balanced_accuracy_from_predictions,
    collect_subject_runs,
    cv_predictions,
    load_subject_trials,
    load_template_and_atlas,
    make_estimator,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Compute parcelwise local decoding extent.")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--subjects", nargs="*", default=None)
    parser.add_argument("--accuracy-threshold", type=float, default=0.60)
    parser.add_argument("--z-threshold", type=float, default=1.64)
    return parser.parse_args()


def one_sided_binomial_p(accuracy: float, n_trials: int) -> tuple[int, float]:
    n_correct = int(np.rint(accuracy * n_trials))
    p = float(binom.sf(n_correct - 1, n_trials, 0.5))
    return n_correct, p


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

    parcel_rows = []
    summary_rows = []
    skipped = []

    for subject in chosen_subjects:
        try:
            X_voxels, y, groups = load_subject_trials(args.data_root, subject, subjects[subject], template)
            subject_rows = []
            for parcel in range(1, 801):
                vox = np.flatnonzero(atlas_data == parcel)
                if len(vox) < 2:
                    acc = np.nan
                    n_correct = np.nan
                    p_binom = np.nan
                    z_one_sided = np.nan
                else:
                    y_true, y_pred, _ = cv_predictions(X_voxels[:, vox], y, groups, estimator)
                    acc = balanced_accuracy_from_predictions(y_true, y_pred)
                    n_correct, p_binom = one_sided_binomial_p(acc, len(y_true))
                    z_one_sided = float(norm.isf(np.clip(p_binom, 1e-300, 1 - 1e-16)))

                subject_rows.append(
                    {
                        "subject": subject,
                        "parcel": parcel,
                        "accuracy": acc,
                        "accuracy_minus_chance": acc - 0.5 if np.isfinite(acc) else np.nan,
                        "n_trials": int(len(y)),
                        "n_correct_approx": n_correct,
                        "p_binom_one_sided": p_binom,
                        "z_one_sided": z_one_sided,
                        "n_voxels": int(len(vox)),
                    }
                )

            subject_df = pd.DataFrame(subject_rows)
            subject_df["accuracy_gt_threshold"] = subject_df["accuracy"] > args.accuracy_threshold
            subject_df["z_gt_threshold"] = subject_df["z_one_sided"] > args.z_threshold
            parcel_rows.append(subject_df)

            summary_rows.append(
                {
                    "subject": subject,
                    "n_accuracy_gt_threshold": int(subject_df["accuracy_gt_threshold"].sum()),
                    "prop_accuracy_gt_threshold": float(subject_df["accuracy_gt_threshold"].mean()),
                    "n_z_gt_threshold": int(subject_df["z_gt_threshold"].sum()),
                    "prop_z_gt_threshold": float(subject_df["z_gt_threshold"].mean()),
                    "mean_accuracy": float(subject_df["accuracy"].mean()),
                    "max_accuracy": float(subject_df["accuracy"].max()),
                }
            )
            print(
                f"Finished {subject}: extent(z>{args.z_threshold})="
                f"{summary_rows[-1]['prop_z_gt_threshold']:.4f}",
                flush=True,
            )
        except Exception as exc:
            skipped.append({"subject": subject, "error": str(exc)})
            print(f"Skipped {subject}: {exc}", flush=True)

    long_df = pd.concat(parcel_rows, ignore_index=True) if parcel_rows else pd.DataFrame()
    summary_df = pd.DataFrame(summary_rows).sort_values("subject") if summary_rows else pd.DataFrame()
    skipped_df = pd.DataFrame(skipped)

    long_df.to_csv(args.out_dir / "parcelwise_local_accuracy_long.csv", index=False)
    summary_df.to_csv(args.out_dir / "local_extent_by_subject.csv", index=False)
    skipped_df.to_csv(args.out_dir / "skipped_subjects.csv", index=False)

    if len(summary_df):
        overall = pd.DataFrame(
            [
                {
                    "n_subjects": len(summary_df),
                    "mean_prop_accuracy_gt_threshold": summary_df["prop_accuracy_gt_threshold"].mean(),
                    "mean_prop_z_gt_threshold": summary_df["prop_z_gt_threshold"].mean(),
                    "mean_accuracy": summary_df["mean_accuracy"].mean(),
                    "mean_max_accuracy": summary_df["max_accuracy"].mean(),
                    "accuracy_threshold": args.accuracy_threshold,
                    "z_threshold": args.z_threshold,
                }
            ]
        )
        overall.to_csv(args.out_dir / "local_extent_summary.csv", index=False)


if __name__ == "__main__":
    main()
