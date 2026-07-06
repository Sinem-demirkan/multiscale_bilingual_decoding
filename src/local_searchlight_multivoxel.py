"""
Within-subject local multivoxel searchlight decoding for the Guo et al.
bilingual picture-naming dataset (OpenNeuro ds005455).

For each subject, this script runs a whole-brain searchlight analysis on the
single-trial beta images from the LanguageControl task. Trial labels are
derived from trial_type (L1 vs L2), and decoding is evaluated with
leave-one-run-out cross-validation.

After the subject-level maps are computed, the script also performs a simple
group analysis:
- smooth subject maps
- test voxelwise accuracy against chance (0.5)
- apply Benjamini-Hochberg FDR correction

Outputs:
- subject_maps/sub-XXX/searchlight_accuracy.nii.gz
- subject_maps/sub-XXX/subject_summary.csv
- subject_run_status.csv
- group_mean_accuracy_fwhm6.nii.gz
- group_tmap_vs_chance_fwhm6.nii.gz
- group_pmap_vs_chance_fwhm6.nii.gz
- group_qmap_vs_chance_fwhm6.nii.gz
- group_fdr001_mask_fwhm6.nii.gz
- group_searchlight_summary.csv
"""

from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from nilearn import datasets, image
from nilearn.decoding import SearchLight
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, make_scorer
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


DATA_ROOT = Path("duo-cogcon")
TASK = "LanguageControl"
START_SUB = 1
END_SUB = 77
TRIAL_COLUMN = "trial_type"

RADIUS_MM = 4.0
SMOOTHING_FWHM = 6.0
FDR_ALPHA = 0.001
N_JOBS = 8
RANDOM_STATE = 42

OUT_DIR = Path("outputs/local_searchlight_multivoxel")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def subject_id(n):
    return f"sub-{n:03d}"


def parse_run_number(path):
    return int(path.name.split("run-")[1].split("_")[0])


def language_label(trial_type):
    trial_type = str(trial_type)
    if trial_type.startswith("L1"):
        return "L1"
    if trial_type.startswith("L2"):
        return "L2"
    return np.nan


def valid_subject_output_dir(subject):
    out = OUT_DIR / "subject_maps" / subject
    out.mkdir(parents=True, exist_ok=True)
    return out


def build_mask_for_img(img):
    template_mask = datasets.load_mni152_brain_mask(resolution=2)
    template_mask = image.resample_to_img(
        template_mask,
        img,
        interpolation="nearest",
        force_resample=True,
        copy_header=True,
    )
    data = np.asarray(img.dataobj, dtype=np.float32)
    nonzero_mask = np.mean(np.abs(data), axis=-1) > 0
    mask_data = nonzero_mask & (np.asarray(template_mask.dataobj) > 0)
    return nib.Nifti1Image(mask_data.astype(np.uint8), img.affine, img.header)


def load_subject_data(subject):
    beta_files = sorted(
        DATA_ROOT.glob(f"**/{subject}_task-{TASK}_run-*_singletrial-Act.nii.gz")
    )
    event_files = sorted(
        DATA_ROOT.glob(f"**/{subject}_task-{TASK}_run-*_events.tsv")
    )

    if len(beta_files) == 0 or len(event_files) == 0:
        return None

    beta_by_run = {parse_run_number(p): p for p in beta_files}
    event_by_run = {parse_run_number(p): p for p in event_files}
    runs = sorted(set(beta_by_run) & set(event_by_run))

    imgs = []
    labels = []
    groups = []

    for run in runs:
        beta_file = beta_by_run[run]
        event_file = event_by_run[run]

        img = nib.load(str(beta_file))
        events = pd.read_csv(event_file, sep="\t")

        if TRIAL_COLUMN not in events.columns:
            print(subject, f"missing column {TRIAL_COLUMN} in {event_file.name}")
            continue

        events["language"] = events[TRIAL_COLUMN].map(language_label)
        events = events.dropna(subset=["language"]).reset_index(drop=True)

        if img.shape[-1] != len(events):
            print(subject, f"run {run} mismatch")
            continue

        imgs.append(img)
        labels.extend(events["language"].tolist())
        groups.extend([run] * len(events))

    if len(imgs) < 2:
        return None

    y = np.asarray(labels)
    groups = np.asarray(groups)

    if set(np.unique(y)) != {"L1", "L2"}:
        return None

    concat_img = image.concat_imgs(imgs)
    mask_img = build_mask_for_img(concat_img)
    return concat_img, y, groups, mask_img


def run_subject_searchlight(subject):
    out_dir = valid_subject_output_dir(subject)
    acc_path = out_dir / "searchlight_accuracy.nii.gz"

    loaded = load_subject_data(subject)
    if loaded is None:
        return {
            "subject": subject,
            "status": "skipped_invalid_subject",
            "map_path": "",
        }

    concat_img, y, groups, mask_img = loaded

    estimator = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            penalty="l2",
            C=0.01,
            solver="lbfgs",
            max_iter=1000,
            random_state=RANDOM_STATE,
        ),
    )

    searchlight = SearchLight(
        mask_img=mask_img,
        process_mask_img=mask_img,
        radius=RADIUS_MM,
        estimator=estimator,
        scoring=make_scorer(balanced_accuracy_score),
        cv=LeaveOneGroupOut(),
        n_jobs=N_JOBS,
        verbose=0,
    )
    searchlight.fit(concat_img, y, groups=groups)

    scores = searchlight.scores_
    score_img = nib.Nifti1Image(scores.astype(np.float32), mask_img.affine, mask_img.header)
    nib.save(score_img, str(acc_path))

    pd.DataFrame(
        [
            {
                "subject": subject,
                "n_trials": int(len(y)),
                "n_L1": int(np.sum(y == "L1")),
                "n_L2": int(np.sum(y == "L2")),
                "mean_accuracy_in_mask": float(scores[scores > 0].mean()),
                "max_accuracy": float(scores.max()),
                "min_nonzero_accuracy": float(scores[scores > 0].min()),
            }
        ]
    ).to_csv(out_dir / "subject_summary.csv", index=False)

    return {
        "subject": subject,
        "status": "ok",
        "map_path": str(acc_path),
    }


def bh_fdr(pvals):
    qvals = np.full_like(pvals, np.nan, dtype=float)
    valid = np.isfinite(pvals)
    pv = pvals[valid]

    if pv.size == 0:
        return qvals

    order = np.argsort(pv)
    ranked = pv[order]
    m = float(len(ranked))
    q_ranked = ranked * m / (np.arange(1, len(ranked) + 1))
    q_ranked = np.minimum.accumulate(q_ranked[::-1])[::-1]
    q_ranked = np.clip(q_ranked, 0, 1)
    qvals[valid] = q_ranked[np.argsort(order)]
    return qvals


def run_group_map(subjects):
    map_paths = [valid_subject_output_dir(subject) / "searchlight_accuracy.nii.gz" for subject in subjects]
    map_paths = [p for p in map_paths if p.exists()]

    if not map_paths:
        raise RuntimeError("No subject searchlight maps found for group analysis.")

    smoothed_imgs = [image.smooth_img(str(p), fwhm=SMOOTHING_FWHM) for p in map_paths]
    stack = image.concat_imgs(smoothed_imgs)
    stack_data = np.asarray(stack.dataobj, dtype=np.float32)

    valid_mask = np.any(stack_data > 0, axis=-1)
    n_subj = stack_data.shape[-1]
    vox_data = stack_data[valid_mask, :]

    tvals, pvals = stats.ttest_1samp(vox_data.T, 0.5, axis=0, alternative="greater")
    qvals = bh_fdr(pvals)
    sig = qvals < FDR_ALPHA

    t_map = np.zeros(valid_mask.shape, dtype=np.float32)
    p_map = np.ones(valid_mask.shape, dtype=np.float32)
    q_map = np.ones(valid_mask.shape, dtype=np.float32)
    mean_map = np.zeros(valid_mask.shape, dtype=np.float32)
    sig_map = np.zeros(valid_mask.shape, dtype=np.uint8)

    t_map[valid_mask] = np.nan_to_num(tvals, nan=0.0)
    p_map[valid_mask] = np.nan_to_num(pvals, nan=1.0)
    q_map[valid_mask] = np.nan_to_num(qvals, nan=1.0)
    mean_map[valid_mask] = vox_data.mean(axis=1)
    sig_map[valid_mask] = sig.astype(np.uint8)

    ref_img = smoothed_imgs[0]
    nib.save(nib.Nifti1Image(mean_map, ref_img.affine, ref_img.header), str(OUT_DIR / "group_mean_accuracy_fwhm6.nii.gz"))
    nib.save(nib.Nifti1Image(t_map, ref_img.affine, ref_img.header), str(OUT_DIR / "group_tmap_vs_chance_fwhm6.nii.gz"))
    nib.save(nib.Nifti1Image(p_map, ref_img.affine, ref_img.header), str(OUT_DIR / "group_pmap_vs_chance_fwhm6.nii.gz"))
    nib.save(nib.Nifti1Image(q_map, ref_img.affine, ref_img.header), str(OUT_DIR / "group_qmap_vs_chance_fwhm6.nii.gz"))
    nib.save(nib.Nifti1Image(sig_map, ref_img.affine, ref_img.header), str(OUT_DIR / "group_fdr001_mask_fwhm6.nii.gz"))

    sig_t = tvals[sig]
    summary = pd.DataFrame(
        [
            {
                "n_subjects": n_subj,
                "n_voxels_tested": int(valid_mask.sum()),
                "n_voxels_fdr_sig": int(sig.sum()),
                "fdr_alpha": float(FDR_ALPHA),
                "smoothing_fwhm": float(SMOOTHING_FWHM),
                "chance_level": 0.5,
                "min_significant_t": float(sig_t.min()) if sig_t.size else np.nan,
                "max_significant_t": float(sig_t.max()) if sig_t.size else np.nan,
                "mean_significant_t": float(sig_t.mean()) if sig_t.size else np.nan,
                "mean_accuracy_in_tested_voxels": float(vox_data.mean()),
                "mean_accuracy_in_sig_voxels": float(vox_data[sig].mean()) if sig.any() else np.nan,
            }
        ]
    )
    summary.to_csv(OUT_DIR / "group_searchlight_summary.csv", index=False)


rows = []
valid_subjects = []

for sub in range(START_SUB, END_SUB + 1):
    subject = subject_id(sub)
    print(f"Running subject searchlight: {subject}")
    row = run_subject_searchlight(subject)
    rows.append(row)

    if row["status"] == "ok":
        valid_subjects.append(subject)

pd.DataFrame(rows).to_csv(OUT_DIR / "subject_run_status.csv", index=False)

if not valid_subjects:
    raise RuntimeError("No valid subject maps available for group analysis.")

print(f"Running group analysis for {len(valid_subjects)} subjects")
run_group_map(valid_subjects)
print(f"Outputs written to {OUT_DIR}")
