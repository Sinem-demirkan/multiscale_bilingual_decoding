from pathlib import Path
import argparse

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import StandardScaler


RANDOM_STATE = 42

OUT = None
FEATURES = None
LABELS = None

K_LIST = [1, 5, 10, 25, 50, 100, 200, 300, 400, 600, 799]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run leave-one-subject-out population PCA self-decoding controls."
    )
    parser.add_argument("--features-csv", type=Path, required=True)
    parser.add_argument("--labels-csv", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def configure_paths(args):
    global OUT, FEATURES, LABELS
    OUT = args.out_dir
    FEATURES = args.features_csv
    LABELS = args.labels_csv


def make_classifier():
    return LogisticRegression(
        C=0.01,
        solver="lbfgs",
        max_iter=1000,
        random_state=RANDOM_STATE,
    )


def load_subject_data():
    features = pd.read_csv(FEATURES)
    parcel_cols = [c for c in features.columns if c.startswith("parcel_")]
    features["trial_index"] = features["trial_index"].astype(int)

    labels = pd.read_csv(LABELS).sort_values(["subject", "run", "trial_in_run"]).copy()
    labels["trial_index"] = labels.groupby("subject").cumcount() + 1
    labels = labels[["subject", "trial_index", "run", "true_y"]]

    df = features.merge(labels, on=["subject", "trial_index"], how="inner")
    subjects = sorted(df["subject"].unique())

    subject_x = {}
    subject_y = {}
    subject_run = {}

    for subject in subjects:
        sub = df.loc[df["subject"] == subject].sort_values("trial_index")
        x = sub[parcel_cols].to_numpy(float)
        subject_x[subject] = StandardScaler().fit_transform(x)
        subject_y[subject] = sub["true_y"].to_numpy(int)
        subject_run[subject] = sub["run"].to_numpy()

    return subjects, subject_x, subject_y, subject_run


def residualize(x, components):
    return x - x @ components.T @ components


def loo_run_score(x, y, runs):
    preds = np.full(y.shape, np.nan)
    for run in np.unique(runs):
        test = runs == run
        train = ~test
        clf = make_classifier()
        clf.fit(x[train], y[train])
        preds[test] = clf.predict(x[test])
    return float(balanced_accuracy_score(y, preds))


def summarize(long_df):
    rows = []
    for k, tmp in long_df.groupby("n_components"):
        for metric in ["original_accuracy", "shared_space_accuracy", "residual_space_accuracy"]:
            x = tmp[metric].to_numpy(float)
            rows.append(
                {
                    "n_components": int(k),
                    "metric": metric,
                    "n_subjects": len(x),
                    "mean": float(np.mean(x)),
                    "sd": float(np.std(x, ddof=1)),
                    "ci_low": float(np.mean(x) - 1.96 * np.std(x, ddof=1) / np.sqrt(len(x))),
                    "ci_high": float(np.mean(x) + 1.96 * np.std(x, ddof=1) / np.sqrt(len(x))),
                    "t_vs_chance": float(stats.ttest_1samp(x, 0.5).statistic),
                    "p_vs_chance": float(stats.ttest_1samp(x, 0.5).pvalue),
                }
            )
    return pd.DataFrame(rows).sort_values(["n_components", "metric"]).reset_index(drop=True)


def main():
    args = parse_args()
    configure_paths(args)
    OUT.mkdir(parents=True, exist_ok=True)
    subjects, subject_x, subject_y, subject_run = load_subject_data()
    max_k = max(K_LIST)

    rows = []
    for subject in subjects:
        print(f"Leave-one-subject-out PCA self-decoding: {subject}", flush=True)
        pca_subjects = [s for s in subjects if s != subject]
        x_pool = np.vstack([subject_x[s] for s in pca_subjects])

        pca = PCA(
            n_components=max_k,
            svd_solver="randomized",
            random_state=RANDOM_STATE,
        )
        pca.fit(x_pool)

        x = subject_x[subject]
        y = subject_y[subject]
        runs = subject_run[subject]
        original_accuracy = loo_run_score(x, y, runs)

        for k in K_LIST:
            components = pca.components_[:k]
            shared_x = x @ components.T
            residual_x = residualize(x, components)

            rows.append(
                {
                    "subject": subject,
                    "n_components": k,
                    "original_accuracy": original_accuracy,
                    "shared_space_accuracy": loo_run_score(shared_x, y, runs),
                    "residual_space_accuracy": loo_run_score(residual_x, y, runs),
                }
            )

    long_df = pd.DataFrame(rows)
    long_path = OUT / "leave_one_subject_out_pca_self_decoding_by_subject.csv"
    summary_path = OUT / "leave_one_subject_out_pca_self_decoding_summary.csv"
    long_df.to_csv(long_path, index=False)

    summary = summarize(long_df)
    summary.to_csv(summary_path, index=False)

    print(f"Saved {long_path}")
    print(f"Saved {summary_path}")
    print(summary[["n_components", "metric", "mean", "ci_low", "ci_high", "p_vs_chance"]].to_string(index=False))


if __name__ == "__main__":
    main()
