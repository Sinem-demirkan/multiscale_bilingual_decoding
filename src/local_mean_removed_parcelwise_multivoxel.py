from pathlib import Path
import argparse
import re

import nibabel as nib
import numpy as np
import pandas as pd
from nilearn import datasets, image
from scipy import stats
from sklearn.base import clone
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "ds005455_data"
SINGLETRIAL_DIR = DATA / "singletrial"
EVENTS_DIR = DATA / "events"
MVPA = ROOT / "mvpa_schaefer800"
OUT = MVPA / "mean_removed_local_mvpa"
OUT.mkdir(parents=True, exist_ok=True)

SUB_RE = re.compile(
    r"(sub-[0-9]+)_task-LanguageControl_run-(0[12])_singletrial-Act\.nii\.gz"
)


def canonical_subject(subject):
    return f"sub-{int(subject.split('-')[1]):03d}"


def parse_args():
    parser = argparse.ArgumentParser(description="Run mean-removed parcelwise local decoding.")
    parser.add_argument("--data-root", type=Path, default=DATA)
    parser.add_argument("--mvpa-dir", type=Path, default=MVPA)
    parser.add_argument("--out-dir", type=Path, default=OUT)
    return parser.parse_args()


def valid_events_file(path):
    if not path.exists():
        return False
    with open(path, "r") as f:
        return f.readline().startswith("onset\t")


def language_label(trial_type):
    trial_type = str(trial_type)
    if trial_type.startswith("L1"):
        return "L1"
    if trial_type.startswith("L2"):
        return "L2"
    return np.nan


def collect_subject_runs():
    subjects = {}
    for path in SINGLETRIAL_DIR.glob("*_singletrial-Act.nii.gz"):
        match = SUB_RE.match(path.name)
        if not match:
            continue
        subject, run = match.groups()
        canonical = canonical_subject(subject)
        subjects.setdefault(canonical, {}).setdefault(run, [])
        subjects[canonical][run].append(path)

    chosen = {}
    for canonical, runs in subjects.items():
        chosen[canonical] = {}
        for run, paths in runs.items():
            paths = sorted(paths, key=lambda p: (len(p.name), p.name), reverse=True)
            chosen[canonical][run] = paths[0]
    return chosen


def matching_events_path(image_path):
    return EVENTS_DIR / image_path.name.replace("_singletrial-Act.nii.gz", "_events.tsv")


def cv_balanced_accuracy(X, y, run_groups, estimator):
    cv = LeaveOneGroupOut()
    scores = []
    for train_idx, test_idx in cv.split(X, y, groups=run_groups):
        clf = clone(estimator)
        clf.fit(X[train_idx], y[train_idx])
        yhat = clf.predict(X[test_idx])
        scores.append(balanced_accuracy_score(y[test_idx], yhat))
    return float(np.mean(scores))


def load_subject(subject, run_paths, template):
    run_arrays = []
    run_labels = []
    run_groups = []

    for run in ["01", "02"]:
        img_path = run_paths.get(run)
        if img_path is None:
            continue
        events_path = matching_events_path(img_path)
        if not valid_events_file(events_path):
            continue

        img = image.resample_to_img(
            nib.load(img_path),
            template,
            interpolation="continuous",
            force_resample=True,
            copy_header=True,
        )
        events = pd.read_csv(events_path, sep="\t")
        events["language"] = events["trial_type"].map(language_label)
        events = events.dropna(subset=["language"]).reset_index(drop=True)

        if len(events) != img.shape[-1]:
            raise ValueError(f"{subject} {run}: {len(events)} events but {img.shape[-1]} image volumes")

        data = np.asarray(img.dataobj, dtype=np.float32)
        run_arrays.append(data.reshape(-1, data.shape[-1]).T)
        run_labels.append(events["language"].to_numpy())
        run_groups.append(np.repeat(run, len(events)))

    if len(run_arrays) < 2:
        raise ValueError("fewer_than_two_valid_runs")

    X = np.vstack(run_arrays)
    y = np.concatenate(run_labels)
    groups = np.concatenate(run_groups)
    if set(np.unique(y)) != {"L1", "L2"}:
        raise ValueError(f"labels_{sorted(np.unique(y))}")
    return X, y, groups


def trialwise_remove_parcel_mean(X):
    # Remove each trial's mean activation across voxels in the parcel.
    return X - X.mean(axis=1, keepdims=True)


def summarize(results):
    subj = (
        results.groupby("subject")
        .agg(
            mean_accuracy=("mean_removed_accuracy", "mean"),
            mean_accuracy_minus_chance=("mean_removed_accuracy_minus_chance", "mean"),
            max_accuracy=("mean_removed_accuracy", "max"),
            top1_mean_accuracy=("mean_removed_accuracy", lambda x: x.nlargest(max(1, int(np.ceil(len(x) * 0.01)))).mean()),
            top5_mean_accuracy=("mean_removed_accuracy", lambda x: x.nlargest(max(1, int(np.ceil(len(x) * 0.05)))).mean()),
            top10_mean_accuracy=("mean_removed_accuracy", lambda x: x.nlargest(max(1, int(np.ceil(len(x) * 0.10)))).mean()),
            n_parcels_above_60=("mean_removed_accuracy", lambda x: int((x > 0.60).sum())),
            n_parcels=("mean_removed_accuracy", "count"),
        )
        .reset_index()
    )

    rows = []
    for col in [
        "mean_accuracy_minus_chance",
        "max_accuracy",
        "top1_mean_accuracy",
        "top5_mean_accuracy",
        "top10_mean_accuracy",
        "n_parcels_above_60",
    ]:
        x = subj[col].dropna()
        test_value = 0 if col in ["mean_accuracy_minus_chance", "n_parcels_above_60"] else 0.5
        t, p = stats.ttest_1samp(x, test_value)
        ci = stats.t.interval(0.95, len(x) - 1, loc=x.mean(), scale=stats.sem(x))
        rows.append(
            {
                "measure": col,
                "n_subjects": len(x),
                "mean": x.mean(),
                "sd": x.std(ddof=1),
                "median": x.median(),
                "min": x.min(),
                "max": x.max(),
                "ci_low": ci[0],
                "ci_high": ci[1],
                "test_value": test_value,
                "t": t,
                "p": p,
            }
        )
    summary = pd.DataFrame(rows)
    return subj, summary


def compare_to_original(mean_removed):
    original = pd.read_csv(MVPA / "schaefer800_mvpa_accuracy_long.csv")
    if "parcel_index" not in original.columns and "parcel" in original.columns:
        original = original.rename(columns={"parcel": "parcel_index"})
    original = original[["subject", "parcel_index", "accuracy", "accuracy_minus_chance"]].rename(
        columns={
            "accuracy": "original_accuracy",
            "accuracy_minus_chance": "original_accuracy_minus_chance",
        }
    )
    merged = original.merge(mean_removed, on=["subject", "parcel_index"], how="inner")
    merged["accuracy_drop_after_mean_removal"] = (
        merged["original_accuracy"] - merged["mean_removed_accuracy"]
    )

    subject_compare = (
        merged.groupby("subject")
        .agg(
            original_mean_accuracy=("original_accuracy", "mean"),
            mean_removed_mean_accuracy=("mean_removed_accuracy", "mean"),
            mean_accuracy_drop=("accuracy_drop_after_mean_removal", "mean"),
            original_max_accuracy=("original_accuracy", "max"),
            mean_removed_max_accuracy=("mean_removed_accuracy", "max"),
            original_top5_mean_accuracy=("original_accuracy", lambda x: x.nlargest(max(1, int(np.ceil(len(x) * 0.05)))).mean()),
            mean_removed_top5_mean_accuracy=("mean_removed_accuracy", lambda x: x.nlargest(max(1, int(np.ceil(len(x) * 0.05)))).mean()),
            original_n_above_60=("original_accuracy", lambda x: int((x > 0.60).sum())),
            mean_removed_n_above_60=("mean_removed_accuracy", lambda x: int((x > 0.60).sum())),
        )
        .reset_index()
    )
    subject_compare["top5_accuracy_drop"] = (
        subject_compare["original_top5_mean_accuracy"]
        - subject_compare["mean_removed_top5_mean_accuracy"]
    )
    subject_compare["max_accuracy_drop"] = (
        subject_compare["original_max_accuracy"]
        - subject_compare["mean_removed_max_accuracy"]
    )

    rows = []
    for orig_col, mr_col, diff_name in [
        ("original_mean_accuracy", "mean_removed_mean_accuracy", "mean_accuracy"),
        ("original_top5_mean_accuracy", "mean_removed_top5_mean_accuracy", "top5_mean_accuracy"),
        ("original_max_accuracy", "mean_removed_max_accuracy", "max_accuracy"),
        ("original_n_above_60", "mean_removed_n_above_60", "n_parcels_above_60"),
    ]:
        diff = subject_compare[orig_col] - subject_compare[mr_col]
        t, p = stats.ttest_rel(subject_compare[orig_col], subject_compare[mr_col])
        ci = stats.t.interval(0.95, len(diff) - 1, loc=diff.mean(), scale=stats.sem(diff))
        rows.append(
            {
                "measure": diff_name,
                "original_mean": subject_compare[orig_col].mean(),
                "mean_removed_mean": subject_compare[mr_col].mean(),
                "mean_original_minus_mean_removed": diff.mean(),
                "ci_low": ci[0],
                "ci_high": ci[1],
                "paired_t": t,
                "paired_p": p,
            }
        )

    # Within-subject spatial correlation: do parcels that decode before also decode after mean removal?
    corr_rows = []
    for subject, sdf in merged.groupby("subject"):
        x = sdf["original_accuracy_minus_chance"].astype(float)
        y = sdf["mean_removed_accuracy_minus_chance"].astype(float)
        mask = np.isfinite(x) & np.isfinite(y)
        pr, pp = stats.pearsonr(x[mask], y[mask])
        sr, sp = stats.spearmanr(x[mask], y[mask])
        corr_rows.append(
            {
                "subject": subject,
                "pearson_r": pr,
                "pearson_p": pp,
                "spearman_rho": sr,
                "spearman_p": sp,
                "n_parcels": int(mask.sum()),
            }
        )
    corrs = pd.DataFrame(corr_rows)
    rows.append(
        {
            "measure": "original_vs_mean_removed_spatial_pearson_r",
            "original_mean": np.nan,
            "mean_removed_mean": np.nan,
            "mean_original_minus_mean_removed": corrs["pearson_r"].mean(),
            "ci_low": stats.t.interval(
                0.95, len(corrs) - 1, loc=corrs["pearson_r"].mean(), scale=stats.sem(corrs["pearson_r"])
            )[0],
            "ci_high": stats.t.interval(
                0.95, len(corrs) - 1, loc=corrs["pearson_r"].mean(), scale=stats.sem(corrs["pearson_r"])
            )[1],
            "paired_t": stats.ttest_1samp(np.arctanh(np.clip(corrs["pearson_r"], -0.999999, 0.999999)), 0).statistic,
            "paired_p": stats.ttest_1samp(np.arctanh(np.clip(corrs["pearson_r"], -0.999999, 0.999999)), 0).pvalue,
        }
    )
    comparison_summary = pd.DataFrame(rows)
    return merged, subject_compare, corrs, comparison_summary


def main():
    global DATA, SINGLETRIAL_DIR, EVENTS_DIR, MVPA, OUT
    args = parse_args()
    DATA = args.data_root
    SINGLETRIAL_DIR = DATA / "singletrial"
    EVENTS_DIR = DATA / "events"
    MVPA = args.mvpa_dir
    OUT = args.out_dir
    OUT.mkdir(parents=True, exist_ok=True)

    template = datasets.load_mni152_template(resolution=2)
    atlas = datasets.fetch_atlas_schaefer_2018(
        n_rois=800,
        yeo_networks=7,
        resolution_mm=2,
    )
    atlas_img = image.resample_to_img(
        nib.load(atlas.maps),
        template,
        interpolation="nearest",
        force_resample=True,
        copy_header=True,
    )
    atlas_data = np.asarray(atlas_img.dataobj).astype(np.int16).ravel()

    labels = [
        lab.decode("utf-8") if isinstance(lab, bytes) else str(lab)
        for lab in atlas.labels
    ]

    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=0.01, solver="lbfgs", max_iter=1000, random_state=42),
    )

    rows = []
    skipped = []
    subjects = collect_subject_runs()

    for subject in sorted(subjects):
        try:
            X, y, groups = load_subject(subject, subjects[subject], template)
            print(
                f"Running mean-removed local MVPA {subject}: {len(y)} trials, "
                f"L1={np.sum(y == 'L1')}, L2={np.sum(y == 'L2')}",
                flush=True,
            )

            for parcel in range(1, 801):
                vox = np.flatnonzero(atlas_data == parcel)
                if len(vox) < 2:
                    acc = np.nan
                else:
                    Xp = trialwise_remove_parcel_mean(X[:, vox])
                    acc = cv_balanced_accuracy(Xp, y, groups, clf)

                rows.append(
                    {
                        "subject": subject,
                        "parcel_index": parcel,
                        "parcel_name": labels[parcel] if parcel < len(labels) else f"Parcel_{parcel}",
                        "mean_removed_accuracy": acc,
                        "mean_removed_accuracy_minus_chance": acc - 0.5 if np.isfinite(acc) else np.nan,
                        "n_voxels": int(len(vox)),
                        "n_trials": int(len(y)),
                        "n_L1": int(np.sum(y == "L1")),
                        "n_L2": int(np.sum(y == "L2")),
                    }
                )
            pd.DataFrame(rows).to_csv(OUT / "mean_removed_local_mvpa_accuracy_long.csv", index=False)
        except Exception as e:
            print(f"SKIP {subject}: {e}", flush=True)
            skipped.append({"subject": subject, "error": str(e)})

    results = pd.DataFrame(rows)
    results.to_csv(OUT / "mean_removed_local_mvpa_accuracy_long.csv", index=False)
    pd.DataFrame(skipped).to_csv(OUT / "skipped_subjects.csv", index=False)

    subject_summary, summary = summarize(results)
    merged, subject_compare, corrs, comparison_summary = compare_to_original(results)

    subject_summary.to_csv(OUT / "mean_removed_local_mvpa_subject_summary.csv", index=False)
    summary.to_csv(OUT / "mean_removed_local_mvpa_summary_tests.csv", index=False)
    merged.to_csv(OUT / "original_vs_mean_removed_local_mvpa_long.csv", index=False)
    subject_compare.to_csv(OUT / "original_vs_mean_removed_subject_comparison.csv", index=False)
    corrs.to_csv(OUT / "original_vs_mean_removed_spatial_correlations_by_subject.csv", index=False)
    comparison_summary.to_csv(OUT / "original_vs_mean_removed_comparison_summary.csv", index=False)

    print("\nMean-removed local MVPA summary:")
    print(summary.to_string(index=False))
    print("\nOriginal vs mean-removed comparison:")
    print(comparison_summary.to_string(index=False))
    print(f"\nSaved outputs to {OUT}")


if __name__ == "__main__":
    main()
