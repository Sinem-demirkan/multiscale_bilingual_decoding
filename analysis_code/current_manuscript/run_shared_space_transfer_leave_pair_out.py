from pathlib import Path
import argparse

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import StandardScaler
import warnings


warnings.filterwarnings("ignore", category=FutureWarning)

RANDOM_STATE = 42
N_BOOT = 5000
BOOT_SEED = 42

OUT = None
FEATURES = None
LABELS = None
TRANSFER = None
SUBJECT_TABLE = None

K_LIST = [1, 5, 10, 25, 50, 100, 200]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run leave-pair-out population PCA transfer analyses."
    )
    parser.add_argument("--features-csv", type=Path, required=True)
    parser.add_argument("--labels-csv", type=Path, required=True)
    parser.add_argument("--transfer-csv", type=Path, required=True)
    parser.add_argument("--participant-table-csv", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def configure_paths(args):
    global OUT, FEATURES, LABELS, TRANSFER, SUBJECT_TABLE
    OUT = args.out_dir
    FEATURES = args.features_csv
    LABELS = args.labels_csv
    TRANSFER = args.transfer_csv
    SUBJECT_TABLE = args.participant_table_csv


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
    labels = labels[["subject", "trial_index", "true_y"]]

    df = features.merge(labels, on=["subject", "trial_index"], how="inner")
    subjects = sorted(df["subject"].unique())

    subject_X = {}
    subject_y = {}
    for subject in subjects:
        sub = df.loc[df["subject"] == subject].sort_values("trial_index")
        subject_X[subject] = StandardScaler().fit_transform(sub[parcel_cols].to_numpy(float))
        subject_y[subject] = sub["true_y"].to_numpy(int)

    return subjects, subject_X, subject_y


def residualize(X, components):
    return X - X @ components.T @ components


def transfer_score(X_train, y_train, X_test, y_test):
    clf = make_classifier()
    clf.fit(X_train, y_train)
    return float(balanced_accuracy_score(y_test, clf.predict(X_test)))


def participant_transfer_values(pair_df, value_col):
    teacher = pair_df.groupby("train_subject")[value_col].mean()
    learner = pair_df.groupby("test_subject")[value_col].mean()
    subjects = sorted(set(teacher.index) | set(learner.index))
    return np.asarray(
        [
            np.mean([teacher.loc[subject], learner.loc[subject]])
            for subject in subjects
        ],
        dtype=float,
    )


def bootstrap_pair_cluster_mean_ci(
    df,
    value_col,
    seed,
    train_col="train_subject",
    test_col="test_subject",
):
    work = df[[train_col, test_col, value_col]].dropna().copy()
    participants = np.unique(
        np.concatenate([work[train_col].to_numpy(), work[test_col].to_numpy()])
    )
    participant_index = {
        participant: idx for idx, participant in enumerate(participants)
    }
    train_idx = work[train_col].map(participant_index).to_numpy()
    test_idx = work[test_col].map(participant_index).to_numpy()
    values = work[value_col].to_numpy(dtype=float)

    mean = float(values.mean())
    rng = np.random.default_rng(seed)
    boot_means = []
    n_participants = len(participants)

    for _ in range(N_BOOT):
        sampled = rng.integers(0, n_participants, size=n_participants)
        multiplicity = np.bincount(sampled, minlength=n_participants)
        pair_weights = (
            multiplicity[train_idx] * multiplicity[test_idx]
        ).astype(float)
        total_weight = pair_weights.sum()
        if total_weight == 0:
            continue
        boot_means.append(np.average(values, weights=pair_weights))

    ci_low, ci_high = np.percentile(np.asarray(boot_means, dtype=float), [2.5, 97.5])
    return mean, float(ci_low), float(ci_high)


def run_leave_pair_out(subjects, subject_X, subject_y):
    rows = []
    max_k = max(K_LIST)
    component_cache = {}
    print(f"Running shared-dimension transfer for {len(subjects)} participants.", flush=True)

    for teacher in subjects:
        for learner in subjects:
            if teacher == learner:
                continue

            pair_key = tuple(sorted((teacher, learner)))
            if pair_key not in component_cache:
                pca_subjects = [s for s in subjects if s not in {teacher, learner}]
                X_pool = np.vstack([subject_X[s] for s in pca_subjects])
                pca = PCA(n_components=max_k, svd_solver="randomized", random_state=RANDOM_STATE)
                pca.fit(X_pool)
                component_cache[pair_key] = pca.components_

            for k in K_LIST:
                components = component_cache[pair_key][:k]
                rows.append(
                    {
                        "train_subject": teacher,
                        "test_subject": learner,
                        "n_components": k,
                        "shared_space_accuracy": transfer_score(
                            subject_X[teacher] @ components.T,
                            subject_y[teacher],
                            subject_X[learner] @ components.T,
                            subject_y[learner],
                        ),
                        "residual_space_accuracy": transfer_score(
                            residualize(subject_X[teacher], components),
                            subject_y[teacher],
                            residualize(subject_X[learner], components),
                            subject_y[learner],
                        ),
                    }
                )

    out = pd.DataFrame(rows)
    out.to_csv(OUT / "shared_space_transfer_leave_pair_out_by_pair.csv", index=False)
    return out


def summarize(shared_transfer):
    orig = pd.read_csv(TRANSFER)
    orig = orig.loc[
        ~orig["same_subject"].astype(str).str.lower().eq("true"),
        ["train_subject", "test_subject", "balanced_accuracy"],
    ].rename(columns={"balanced_accuracy": "original_accuracy"})

    merged = shared_transfer.merge(orig, on=["train_subject", "test_subject"], how="left")
    merged.to_csv(OUT / "shared_space_transfer_leave_pair_out_by_pair_with_original.csv", index=False)

    rows = []
    for k, tmp in merged.groupby("n_components"):
        k_index = K_LIST.index(int(k))
        for metric_i, metric in enumerate(
            ["original_accuracy", "shared_space_accuracy", "residual_space_accuracy"]
        ):
            seed = BOOT_SEED + int(k) * 10 + metric_i
            if metric == "original_accuracy":
                seed = BOOT_SEED + 1000
            elif metric == "shared_space_accuracy":
                seed = BOOT_SEED + 100 + k_index
            elif metric == "residual_space_accuracy":
                seed = BOOT_SEED + 200 + k_index
            mean, ci_low, ci_high = bootstrap_pair_cluster_mean_ci(
                tmp,
                metric,
                seed=seed,
            )
            participant_values = participant_transfer_values(tmp, metric)
            rows.append(
                {
                    "n_components": k,
                    "metric": metric,
                    "n_pairs": int(len(tmp[metric].dropna())),
                    "n_participants": int(
                        len(
                            np.unique(
                                np.concatenate(
                                    [
                                        tmp["train_subject"].to_numpy(),
                                        tmp["test_subject"].to_numpy(),
                                    ]
                                )
                            )
                        )
                    ),
                    "mean": mean,
                    "sd": float(np.std(participant_values, ddof=1)),
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                    "pair_level_mean": float(tmp[metric].mean()),
                }
            )

    summary = pd.DataFrame(rows)
    summary.to_csv(OUT / "shared_space_transfer_leave_pair_out_summary.csv", index=False)
    return summary


def run_extent_subgroups(subjects, subject_X, subject_y):
    meta = pd.read_csv(SUBJECT_TABLE).set_index("subject").loc[subjects]
    median_extent = float(meta["local_extent"].median())
    groups = {
        "low_extent": [s for s in subjects if meta.loc[s, "local_extent"] <= median_extent],
        "high_extent": [s for s in subjects if meta.loc[s, "local_extent"] > median_extent],
    }

    rows = []
    max_k = max(K_LIST)
    for group_name, group_subjects in groups.items():
        print(f"Running extent subgroup {group_name} (n={len(group_subjects)}).", flush=True)
        component_cache = {}
        original_cache = {}
        for teacher in group_subjects:
            for learner in group_subjects:
                if teacher == learner:
                    continue

                pair_key = tuple(sorted((teacher, learner)))
                if pair_key not in component_cache:
                    pca_subjects = [s for s in group_subjects if s not in {teacher, learner}]
                    X_pool = np.vstack([subject_X[s] for s in pca_subjects])
                    max_group_k = min(max_k, X_pool.shape[0] - 1, X_pool.shape[1])
                    pca = PCA(n_components=max_group_k, svd_solver="randomized", random_state=RANDOM_STATE)
                    pca.fit(X_pool)
                    component_cache[pair_key] = pca.components_

                direction_key = (teacher, learner)
                if direction_key not in original_cache:
                    original_cache[direction_key] = transfer_score(
                        subject_X[teacher],
                        subject_y[teacher],
                        subject_X[learner],
                        subject_y[learner],
                    )

                for k in K_LIST:
                    if k > component_cache[pair_key].shape[0]:
                        continue
                    components = component_cache[pair_key][:k]
                    rows.append(
                        {
                            "extent_group": group_name,
                            "train_subject": teacher,
                            "test_subject": learner,
                            "n_components": k,
                            "original_accuracy": original_cache[direction_key],
                            "shared_space_accuracy": transfer_score(
                                subject_X[teacher] @ components.T,
                                subject_y[teacher],
                                subject_X[learner] @ components.T,
                                subject_y[learner],
                            ),
                            "residual_space_accuracy": transfer_score(
                                residualize(subject_X[teacher], components),
                                subject_y[teacher],
                                residualize(subject_X[learner], components),
                                subject_y[learner],
                            ),
                        }
                    )

    out = pd.DataFrame(rows)
    out.to_csv(OUT / "shared_space_transfer_leave_pair_out_extent_subgroups_by_pair.csv", index=False)

    summary_rows = []
    for (group, k), tmp in out.groupby(["extent_group", "n_components"]):
        for metric_i, metric in enumerate(
            ["original_accuracy", "shared_space_accuracy", "residual_space_accuracy"]
        ):
            seed = BOOT_SEED + int(k) * 10 + metric_i
            if metric == "original_accuracy":
                seed = BOOT_SEED + 1000
            mean, ci_low, ci_high = bootstrap_pair_cluster_mean_ci(
                tmp,
                metric,
                seed=seed,
            )
            participant_values = participant_transfer_values(tmp, metric)
            summary_rows.append(
                {
                    "extent_group": group,
                    "n_components": k,
                    "metric": metric,
                    "n_pairs": int(len(tmp[metric].dropna())),
                    "n_participants": int(
                        len(
                            np.unique(
                                np.concatenate(
                                    [
                                        tmp["train_subject"].to_numpy(),
                                        tmp["test_subject"].to_numpy(),
                                    ]
                                )
                            )
                        )
                    ),
                    "mean": mean,
                    "sd": float(np.std(participant_values, ddof=1)),
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                    "pair_level_mean": float(tmp[metric].mean()),
                }
            )

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(OUT / "shared_space_transfer_leave_pair_out_extent_subgroups_summary.csv", index=False)
    return summary


def main():
    args = parse_args()
    configure_paths(args)
    OUT.mkdir(parents=True, exist_ok=True)
    subjects, subject_X, subject_y = load_subject_data()
    transfer = run_leave_pair_out(subjects, subject_X, subject_y)
    summary = summarize(transfer)
    subgroup_summary = run_extent_subgroups(subjects, subject_X, subject_y)

    print(f"Saved {OUT / 'shared_space_transfer_leave_pair_out_by_pair.csv'}")
    print(f"Saved {OUT / 'shared_space_transfer_leave_pair_out_summary.csv'}")
    print(f"Saved {OUT / 'shared_space_transfer_leave_pair_out_extent_subgroups_summary.csv'}")


if __name__ == "__main__":
    main()
