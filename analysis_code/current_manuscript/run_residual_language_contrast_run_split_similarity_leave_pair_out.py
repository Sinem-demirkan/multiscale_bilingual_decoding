from pathlib import Path
import argparse
import itertools

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


RANDOM_STATE = 42
K_LIST = [0, 1, 5, 10, 25, 50, 100, 200]
N_BOOT = 5000


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Compute run-split similarity of English-Chinese residual language-state "
            "signatures using leave-both-out PCA spaces."
        )
    )
    parser.add_argument("--features-csv", type=Path, required=True)
    parser.add_argument("--labels-csv", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--k-list", type=int, nargs="+", default=K_LIST)
    parser.add_argument(
        "--positive-language-label",
        default=None,
        help=(
            "Label treated as English/L2. If omitted, string labels containing "
            "'english' or 'l2' are preferred; otherwise the larger of two numeric "
            "labels is used."
        ),
    )
    return parser.parse_args()


def infer_positive_label(y, requested):
    unique = list(pd.Series(y).dropna().unique())
    if len(unique) != 2:
        raise ValueError(f"Expected two language labels, found {unique}")
    if requested is not None:
        for value in unique:
            if str(value) == str(requested):
                return value
        raise ValueError(f"Requested positive label {requested!r} not in {unique}")

    for value in unique:
        text = str(value).lower()
        if "english" in text or "l2" in text or text == "en":
            return value

    try:
        return max(unique, key=float)
    except ValueError:
        return sorted(unique, key=str)[-1]


def load_subject_data(features_csv, labels_csv, positive_language_label):
    features = pd.read_csv(features_csv)
    parcel_cols = [c for c in features.columns if c.startswith("parcel_")]
    features["trial_index"] = features["trial_index"].astype(int)

    labels = pd.read_csv(labels_csv).sort_values(["subject", "run", "trial_in_run"]).copy()
    labels["trial_index"] = labels.groupby("subject").cumcount() + 1
    labels = labels[["subject", "trial_index", "run", "true_y"]]

    df = features.merge(labels, on=["subject", "trial_index"], how="inner")
    positive = infer_positive_label(df["true_y"], positive_language_label)
    df["language_code"] = (df["true_y"] == positive).astype(int)

    subjects = sorted(df["subject"].unique())
    subject_x = {}
    subject_y = {}
    subject_run = {}

    for subject in subjects:
        sub = df.loc[df["subject"].eq(subject)].sort_values("trial_index")
        runs = sorted(sub["run"].unique())
        if len(runs) != 2:
            raise ValueError(f"{subject} has {len(runs)} runs; expected 2.")
        subject_x[subject] = StandardScaler().fit_transform(
            sub[parcel_cols].to_numpy(float)
        )
        subject_y[subject] = sub["language_code"].to_numpy(int)
        subject_run[subject] = sub["run"].to_numpy()

    return subjects, subject_x, subject_y, subject_run, positive, parcel_cols


def residualize(x, components):
    return x - x @ components.T @ components


def language_signature(x, y, runs, run, components=None):
    work = residualize(x, components) if components is not None else x
    mask = runs == run
    pos = mask & (y == 1)
    neg = mask & (y == 0)
    if not pos.any() or not neg.any():
        raise ValueError("Each run must contain both language labels.")
    return work[pos].mean(axis=0) - work[neg].mean(axis=0)


def safe_corr(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = a - a.mean()
    b = b - b.mean()
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return np.nan
    return float(np.dot(a, b) / denom)


def bootstrap_vector_mean_ci(values, n_boot=N_BOOT, seed=RANDOM_STATE):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return np.nan, np.nan, np.nan

    mean = float(values.mean())
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(n_boot, len(values)))
    boot_means = values[indices].mean(axis=1)
    low, high = np.percentile(boot_means, [2.5, 97.5])
    return mean, float(low), float(high)


def bootstrap_pair_matrix_mean_ci(
    df,
    value_col,
    row_col="run1_subject",
    col_col="run2_subject",
    n_boot=N_BOOT,
    seed=RANDOM_STATE,
):
    work = df[[row_col, col_col, value_col]].dropna().copy()
    participants = np.unique(
        np.concatenate([work[row_col].to_numpy(), work[col_col].to_numpy()])
    )
    participant_index = {
        participant: idx for idx, participant in enumerate(participants)
    }
    row_idx = work[row_col].map(participant_index).to_numpy()
    col_idx = work[col_col].map(participant_index).to_numpy()
    values = work[value_col].to_numpy(float)

    mean = float(values.mean())
    rng = np.random.default_rng(seed)
    boot_means = []

    for _ in range(n_boot):
        sampled = rng.integers(0, len(participants), size=len(participants))
        multiplicity = np.bincount(sampled, minlength=len(participants))
        weights = (multiplicity[row_idx] * multiplicity[col_idx]).astype(float)
        total_weight = weights.sum()
        if total_weight == 0:
            continue
        boot_means.append(np.average(values, weights=weights))

    low, high = np.percentile(np.asarray(boot_means, dtype=float), [2.5, 97.5])
    return mean, float(low), float(high)


def fit_pair_components(subjects, subject_x, subject_a, subject_b, max_k):
    heldout = {subject_a, subject_b}
    pool_subjects = [s for s in subjects if s not in heldout]
    x_pool = np.vstack([subject_x[s] for s in pool_subjects])
    pca = PCA(n_components=max_k, svd_solver="randomized", random_state=RANDOM_STATE)
    pca.fit(x_pool)
    return pca.components_


def summarize_long(long_df, subjects, n_features):
    rows = []
    for k, tmp in long_df.groupby("k_removed", sort=True):
        within = tmp.loc[tmp["same_subject"]].copy()
        between = tmp.loc[~tmp["same_subject"]].copy()
        within_mean, within_low, within_high = bootstrap_vector_mean_ci(
            within["similarity"].to_numpy(float), seed=RANDOM_STATE + int(k)
        )
        between_mean, between_low, between_high = bootstrap_pair_matrix_mean_ci(
            between, "similarity", seed=RANDOM_STATE + 1000 + int(k)
        )

        between_by_subject = (
            between.groupby("run1_subject", as_index=False)["similarity"]
            .mean()
            .rename(columns={"run1_subject": "subject", "similarity": "between_mean"})
        )
        within_by_subject = within.rename(
            columns={"run1_subject": "subject", "similarity": "within_similarity"}
        )[["subject", "within_similarity"]]
        subject_diff = within_by_subject.merge(between_by_subject, on="subject", how="inner")
        diff = (
            subject_diff["within_similarity"].to_numpy(float)
            - subject_diff["between_mean"].to_numpy(float)
        )

        rows.append(
            {
                "k_removed": int(k),
                "space": "original" if int(k) == 0 else "residual_leave_both_out",
                "n_subjects": int(len(subjects)),
                "n_features": int(n_features),
                "mean_within_similarity": within_mean,
                "ci_low_within_similarity": within_low,
                "ci_high_within_similarity": within_high,
                "mean_between_similarity": between_mean,
                "ci_low_between_similarity": between_low,
                "ci_high_between_similarity": between_high,
                "mean_within_minus_subject_between": float(np.nanmean(diff)),
                "sd_within_minus_subject_between": float(np.nanstd(diff, ddof=1)),
            }
        )
    return pd.DataFrame(rows).sort_values("k_removed").reset_index(drop=True)


def run_analysis(subjects, subject_x, subject_y, subject_run, k_list):
    runs_by_subject = {s: sorted(np.unique(subject_run[s])) for s in subjects}
    if any(runs_by_subject[s] != runs_by_subject[subjects[0]] for s in subjects):
        raise ValueError("All subjects must have the same two run labels.")
    run1, run2 = runs_by_subject[subjects[0]]

    max_k = max(k for k in k_list if k > 0)
    pair_signatures = {}
    n_pairs = len(subjects) * (len(subjects) - 1) // 2
    print(f"Fitting leave-both-out PCA spaces for {n_pairs} participant pairs.", flush=True)

    original_signatures = {}
    for subject in subjects:
        original_signatures[(subject, run1)] = language_signature(
            subject_x[subject], subject_y[subject], subject_run[subject], run1
        )
        original_signatures[(subject, run2)] = language_signature(
            subject_x[subject], subject_y[subject], subject_run[subject], run2
        )

    for subject_a, subject_b in itertools.combinations(subjects, 2):
        pair_key = tuple(sorted((subject_a, subject_b)))
        components_all = fit_pair_components(
            subjects, subject_x, subject_a, subject_b, max_k
        )

        for k in k_list:
            if k == 0:
                continue
            components = components_all[:k]
            for subject in pair_key:
                for run in (run1, run2):
                    pair_signatures[(pair_key, subject, run, k)] = language_signature(
                        subject_x[subject],
                        subject_y[subject],
                        subject_run[subject],
                        run,
                        components=components,
                    )

    rows = []
    for k in k_list:
        for run1_subject in subjects:
            for run2_subject in subjects:
                if k == 0:
                    similarity = safe_corr(
                        original_signatures[(run1_subject, run1)],
                        original_signatures[(run2_subject, run2)],
                    )
                elif run1_subject == run2_subject:
                    sims = []
                    for excluded_partner in subjects:
                        if excluded_partner == run1_subject:
                            continue
                        pair_key = tuple(sorted((run1_subject, excluded_partner)))
                        sims.append(
                            safe_corr(
                                pair_signatures[(pair_key, run1_subject, run1, k)],
                                pair_signatures[(pair_key, run1_subject, run2, k)],
                            )
                        )
                    similarity = float(np.nanmean(sims))
                else:
                    pair_key = tuple(sorted((run1_subject, run2_subject)))
                    similarity = safe_corr(
                        pair_signatures[(pair_key, run1_subject, run1, k)],
                        pair_signatures[(pair_key, run2_subject, run2, k)],
                    )

                rows.append(
                    {
                        "k_removed": int(k),
                        "run1_subject": run1_subject,
                        "run2_subject": run2_subject,
                        "same_subject": bool(run1_subject == run2_subject),
                        "similarity": similarity,
                    }
                )

    long_df = pd.DataFrame(rows)
    n_features = next(iter(subject_x.values())).shape[1]
    summary = summarize_long(long_df, subjects, n_features)
    return long_df, summary


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    subjects, subject_x, subject_y, subject_run, positive, parcel_cols = load_subject_data(
        args.features_csv, args.labels_csv, args.positive_language_label
    )
    print(
        f"Loaded {len(subjects)} participants and {len(parcel_cols)} parcels; "
        f"English/L2 label is {positive!r}.",
        flush=True,
    )

    long_df, summary = run_analysis(
        subjects, subject_x, subject_y, subject_run, sorted(args.k_list)
    )
    long_path = args.out_dir / "residual_language_contrast_run_split_similarity_long.csv"
    summary_path = args.out_dir / "residual_language_contrast_run_split_fingerprint_summary.csv"
    long_df.to_csv(long_path, index=False)
    summary.to_csv(summary_path, index=False)
    print(f"Saved {long_path}")
    print(f"Saved {summary_path}")


if __name__ == "__main__":
    main()
