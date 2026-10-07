from __future__ import annotations

import argparse
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from nilearn import datasets, image
from nilearn.maskers import NiftiLabelsMasker
from scipy import stats
from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "ds005455_data"
MVPA = ROOT / "mvpa_schaefer800"
ROB = MVPA / "latest_robustness_controls"
LANG = MVPA / "domain_annotation" / "lipkin_language_schaefer800_parcel_scores.csv"
OUT = ROOT / "paper_version" / "targeted_lipkin_pca_grouped_pfi"

PCA_VARIANCE = 0.90
N_PERM_REPEATS = 20
RANDOM_STATE = 42

warnings.filterwarnings("ignore", category=FutureWarning)


def parse_args():
    parser = argparse.ArgumentParser(description="Run targeted Lipkin language and MD grouped PFI.")
    parser.add_argument("--data-root", type=Path, default=DATA)
    parser.add_argument("--mvpa-dir", type=Path, default=MVPA)
    parser.add_argument("--out-dir", type=Path, default=OUT)
    return parser.parse_args()


def find_subjects() -> list[str]:
    files = sorted((DATA / "singletrial").glob("sub-*_task-LanguageControl_run-01_singletrial-Act.nii.gz"))
    subjects = sorted({f.name.split("_")[0] for f in files})
    return [s for s in subjects if len(s) == 7]


def load_subject(subject: str, masker: NiftiLabelsMasker) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    imgs = []
    events = []
    groups = []
    for run_i, run in enumerate(["01", "02"], start=1):
        img_path = DATA / "singletrial" / f"{subject}_task-LanguageControl_run-{run}_singletrial-Act.nii.gz"
        event_path = DATA / "events" / f"{subject}_task-LanguageControl_run-{run}_events.tsv"
        if not img_path.exists() or not event_path.exists():
            raise FileNotFoundError(f"Missing run data for {subject} run {run}")
        imgs.append(image.load_img(str(img_path)))
        ev = pd.read_csv(event_path, sep="\t")
        events.append(ev)
        groups.append(np.full(len(ev), run_i, dtype=int))

    y = np.concatenate([ev["trial_type"].str.contains("L1").astype(int).to_numpy() for ev in events])
    groups = np.concatenate(groups)
    X = masker.fit_transform(image.concat_imgs(imgs))
    if X.shape[0] != len(y):
        raise ValueError(f"{subject}: X has {X.shape[0]} trials but labels have {len(y)}")
    return X, y, groups


def get_lipkin_language_cols() -> np.ndarray:
    lip = pd.read_csv(LANG).sort_values("parcel_index").reset_index(drop=True)
    ranked = lip.sort_values("lipkin_language_mean", ascending=False).reset_index(drop=True)
    return np.array(sorted(ranked.head(80)["parcel_index"].astype(int).tolist()), dtype=int) - 1


def get_lipkin_md_cols() -> np.ndarray:
    md = pd.read_csv(ROB / "lipkin_md_schaefer800_parcel_sets.csv").sort_values("parcel").reset_index(drop=True)
    return np.array(sorted(md.loc[md["lipkin_md_top10"], "parcel"].astype(int).tolist()), dtype=int) - 1


def transform_targeted_block_pca(
    X_train: np.ndarray,
    X_test: np.ndarray,
    target_cols: np.ndarray,
    pca_variance: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
    all_cols = np.arange(X_train.shape[1], dtype=int)
    outside_cols = np.setdiff1d(all_cols, target_cols)

    scaler_out = StandardScaler()
    Xtr_out = scaler_out.fit_transform(X_train[:, outside_cols])
    Xte_out = scaler_out.transform(X_test[:, outside_cols])

    scaler_tgt = StandardScaler()
    Xtr_tgt = scaler_tgt.fit_transform(X_train[:, target_cols])
    Xte_tgt = scaler_tgt.transform(X_test[:, target_cols])

    pca = PCA(n_components=pca_variance, svd_solver="full", random_state=RANDOM_STATE)
    Xtr_tgt_pc = pca.fit_transform(Xtr_tgt)
    Xte_tgt_pc = pca.transform(Xte_tgt)

    Xtr_all = np.concatenate([Xtr_out, Xtr_tgt_pc], axis=1)
    Xte_all = np.concatenate([Xte_out, Xte_tgt_pc], axis=1)
    target_pc_cols = np.arange(Xtr_out.shape[1], Xtr_all.shape[1], dtype=int)

    meta = {
        "n_original_target_features": int(len(target_cols)),
        "n_outside_features": int(len(outside_cols)),
        "n_target_pcs": int(Xtr_tgt_pc.shape[1]),
        "target_explained_variance_sum": float(np.sum(pca.explained_variance_ratio_)),
    }
    return Xtr_all, Xte_all, target_pc_cols, meta


def targeted_pfi_one_subject(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    target_cols: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    estimator = LogisticRegression(
        penalty="l2",
        C=0.01,
        solver="lbfgs",
        max_iter=1000,
        random_state=RANDOM_STATE,
    )
    logo = LeaveOneGroupOut()
    rng = np.random.default_rng(RANDOM_STATE)

    fold_rows = []
    meta_rows = []

    for fold, (tr, te) in enumerate(logo.split(X, y, groups=groups), start=1):
        Xtr_all, Xte_all, target_pc_cols, meta = transform_targeted_block_pca(
            X[tr], X[te], target_cols, PCA_VARIANCE
        )

        clf = clone(estimator)
        clf.fit(Xtr_all, y[tr])
        base_acc = balanced_accuracy_score(y[te], clf.predict(Xte_all))

        drops = []
        for _ in range(N_PERM_REPEATS):
            Xperm = Xte_all.copy()
            perm = rng.permutation(Xperm.shape[0])
            Xperm[:, target_pc_cols] = Xperm[perm][:, target_pc_cols]
            perm_acc = balanced_accuracy_score(y[te], clf.predict(Xperm))
            drops.append(base_acc - perm_acc)

        fold_rows.append(
            {
                "fold": fold,
                "baseline_accuracy": base_acc,
                "importance_mean": float(np.mean(drops)),
                "importance_sd": float(np.std(drops, ddof=1)),
                "n_target_pcs": int(len(target_pc_cols)),
            }
        )

        meta["fold"] = fold
        meta["baseline_accuracy"] = base_acc
        meta_rows.append(meta)

    return pd.DataFrame(fold_rows), pd.DataFrame(meta_rows)


def run_target(
    target_name: str,
    target_cols: np.ndarray,
    feature_cache: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
) -> None:
    target_dir = OUT / target_name
    target_dir.mkdir(parents=True, exist_ok=True)

    fold_rows = []
    meta_rows = []
    subjects = sorted(feature_cache)
    for i, subject in enumerate(subjects, start=1):
        print(f"[{target_name}] {i}/{len(subjects)} {subject}", flush=True)
        X, y, g = feature_cache[subject]
        subj_fold, subj_meta = targeted_pfi_one_subject(X, y, g, target_cols)
        subj_fold["subject"] = subject
        subj_meta["subject"] = subject
        fold_rows.append(subj_fold)
        meta_rows.append(subj_meta)

    fold_df = pd.concat(fold_rows, ignore_index=True)
    fold_df.to_csv(target_dir / f"{target_name}_folds.csv", index=False)

    meta_df = pd.concat(meta_rows, ignore_index=True)
    meta_df.to_csv(target_dir / f"{target_name}_pc_meta.csv", index=False)

    subject_df = (
        fold_df.groupby("subject", as_index=False)
        .agg(
            importance_mean=("importance_mean", "mean"),
            importance_sd_across_folds=("importance_mean", "std"),
            baseline_accuracy=("baseline_accuracy", "mean"),
            mean_n_target_pcs=("n_target_pcs", "mean"),
        )
    )
    subject_df["importance_pp"] = subject_df["importance_mean"] * 100
    subject_df.to_csv(target_dir / f"{target_name}_by_subject.csv", index=False)

    vals = subject_df["importance_pp"].to_numpy()
    mean = vals.mean()
    sem = stats.sem(vals)
    lo, hi = stats.t.interval(0.95, len(vals) - 1, loc=mean, scale=sem)
    t_res = stats.ttest_1samp(vals, 0.0)
    summary_df = pd.DataFrame(
        [
            {
                "target": target_name,
                "n_subjects": len(subject_df),
                "mean_importance_pp": mean,
                "median_importance_pp": subject_df["importance_pp"].median(),
                "sd_importance_pp": subject_df["importance_pp"].std(ddof=1),
                "mean_fold_sd_pp": float(np.nanmean(subject_df["importance_sd_across_folds"])) * 100,
                "mean_baseline_accuracy": subject_df["baseline_accuracy"].mean(),
                "mean_n_target_pcs": subject_df["mean_n_target_pcs"].mean(),
                "ci_low_pp": lo,
                "ci_high_pp": hi,
                "t_vs_0": t_res.statistic,
                "p_vs_0": t_res.pvalue,
            }
        ]
    )
    summary_df.to_csv(target_dir / f"{target_name}_summary.csv", index=False)
    print(f"\n{target_name} summary:")
    print(summary_df.to_string(index=False))


def main() -> None:
    global DATA, MVPA, ROB, LANG, OUT
    args = parse_args()
    DATA = args.data_root
    MVPA = args.mvpa_dir
    ROB = MVPA / "latest_robustness_controls"
    LANG = MVPA / "domain_annotation" / "lipkin_language_schaefer800_parcel_scores.csv"
    OUT = args.out_dir
    OUT.mkdir(parents=True, exist_ok=True)

    atlas = datasets.fetch_atlas_schaefer_2018(n_rois=800, yeo_networks=7, resolution_mm=2)
    masker = NiftiLabelsMasker(
        labels_img=image.load_img(atlas.maps),
        standardize=False,
        strategy="mean",
        resampling_target="data",
        verbose=0,
    )

    subjects = find_subjects()
    feature_cache: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for i, subject in enumerate(subjects, start=1):
        print(f"[load] {i}/{len(subjects)} {subject}", flush=True)
        feature_cache[subject] = load_subject(subject, masker)

    run_target("LipkinLanguage", get_lipkin_language_cols(), feature_cache)
    run_target("LipkinMD", get_lipkin_md_cols(), feature_cache)


if __name__ == "__main__":
    main()
