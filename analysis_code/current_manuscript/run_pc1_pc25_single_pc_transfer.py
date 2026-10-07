import argparse
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import StandardScaler


warnings.filterwarnings("ignore", category=FutureWarning)

RANDOM_STATE = 42
N_COMPONENTS = 25
N_BOOT = 5000
BOOT_SEED = 42

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "paper_version" / "cross_subject_language_transfer"
FEATURES = (
    ROOT
    / "paper_version"
    / "whole_cortex_parcel_mean_aoa_prediction"
    / "whole_cortex_parcel_mean_aoa_trialwise_features.csv"
)
LABELS = (
    ROOT
    / "paper_version"
    / "whole_cortex_item_posthoc"
    / "whole_cortex_trialwise_predictions.csv"
)
OUT = BASE / "pc1_pc25_single_pc_transfer"

BY_PAIR = OUT / "pc1_pc25_single_pc_transfer_by_pair.csv"
SUMMARY = OUT / "pc1_pc25_single_pc_transfer_summary.csv"
RANKED = OUT / "pc1_pc25_single_pc_transfer_ranked.csv"


def parse_args():
    parser = argparse.ArgumentParser(description="Run single-PC leave-pair-out transfer for PCs 1-25.")
    parser.add_argument("--features-csv", type=Path, default=FEATURES)
    parser.add_argument("--labels-csv", type=Path, default=LABELS)
    parser.add_argument("--out-dir", type=Path, default=OUT)
    return parser.parse_args()


def make_classifier():
    return LogisticRegression(
        C=0.01,
        solver="lbfgs",
        max_iter=1000,
        random_state=RANDOM_STATE,
    )


def load_subject_data():
    features = pd.read_csv(FEATURES)
    parcel_cols = [col for col in features.columns if col.startswith("parcel_")]
    features["trial_index"] = features["trial_index"].astype(int)

    labels = pd.read_csv(LABELS).sort_values(["subject", "run", "trial_in_run"]).copy()
    labels["trial_index"] = labels.groupby("subject").cumcount() + 1
    labels = labels[["subject", "trial_index", "true_y"]]

    df = features.merge(labels, on=["subject", "trial_index"], how="inner")
    subjects = sorted(df["subject"].unique())

    subject_x = {}
    subject_y = {}

    for subject in subjects:
        sub = df.loc[df["subject"].eq(subject)].sort_values("trial_index")
        subject_x[subject] = StandardScaler().fit_transform(sub[parcel_cols].to_numpy(float))
        subject_y[subject] = sub["true_y"].to_numpy(int)

    return subjects, subject_x, subject_y


def transfer_score(x_train, y_train, x_test, y_test):
    clf = make_classifier()
    clf.fit(x_train, y_train)
    return float(balanced_accuracy_score(y_test, clf.predict(x_test)))


def completed_pair_keys(path):
    if not path.exists():
        return set()

    done = pd.read_csv(path, usecols=["train_subject", "test_subject", "pc"])
    counts = done.groupby(["train_subject", "test_subject"])["pc"].nunique()

    completed_directions = {
        direction
        for direction, n_pc in counts.items()
        if int(n_pc) == N_COMPONENTS
    }

    completed_pairs = set()
    for train_subject, test_subject in completed_directions:
        reverse = (test_subject, train_subject)
        if reverse in completed_directions:
            completed_pairs.add(tuple(sorted((train_subject, test_subject))))

    return completed_pairs


def append_rows(rows, path):
    out = pd.DataFrame(rows)
    out.to_csv(
        path,
        mode="a",
        header=not path.exists(),
        index=False,
    )


def run_transfer(subjects, subject_x, subject_y):
    OUT.mkdir(parents=True, exist_ok=True)

    completed_pairs = completed_pair_keys(BY_PAIR)
    n_pairs = len(subjects) * (len(subjects) - 1) // 2
    pair_i = 0

    for i, subject_a in enumerate(subjects):
        for subject_b in subjects[i + 1:]:
            pair_i += 1
            pair_key = tuple(sorted((subject_a, subject_b)))

            if pair_key in completed_pairs:
                continue

            if pair_i % 25 == 0:
                print(f"pair {pair_i}/{n_pairs}", flush=True)

            pca_subjects = [s for s in subjects if s not in {subject_a, subject_b}]
            x_pool = np.vstack([subject_x[s] for s in pca_subjects])
            pca = PCA(
                n_components=N_COMPONENTS,
                svd_solver="randomized",
                random_state=RANDOM_STATE,
            )
            pca.fit(x_pool)
            components = pca.components_

            rows = []
            for teacher, learner in [(subject_a, subject_b), (subject_b, subject_a)]:
                teacher_scores = subject_x[teacher] @ components.T
                learner_scores = subject_x[learner] @ components.T
                y_teacher = subject_y[teacher]
                y_learner = subject_y[learner]

                for pc_index in range(N_COMPONENTS):
                    pc_number = pc_index + 1
                    rows.append(
                        {
                            "train_subject": teacher,
                            "test_subject": learner,
                            "representation": "single_pc",
                            "pc": pc_number,
                            "n_components": 1,
                            "accuracy": transfer_score(
                                teacher_scores[:, pc_index : pc_index + 1],
                                y_teacher,
                                learner_scores[:, pc_index : pc_index + 1],
                                y_learner,
                            ),
                        }
                    )

            append_rows(rows, BY_PAIR)


def bootstrap_ci(values, seed):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    mean = float(values.mean())
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(N_BOOT, len(values)))
    boot_means = values[indices].mean(axis=1)
    low, high = np.percentile(boot_means, [2.5, 97.5])

    return mean, float(low), float(high)


def summarize():
    transfer = pd.read_csv(BY_PAIR)
    expected_rows = 77 * 76 * N_COMPONENTS

    if len(transfer) != expected_rows:
        print(f"By-pair file has {len(transfer)} rows; expected {expected_rows}.")
        print("Summary will use currently completed rows only.", flush=True)

    rows = []
    for pc, subset in transfer.groupby("pc"):
        for role, subject_col in [("learner", "test_subject"), ("teacher", "train_subject")]:
            subject_values = (
                subset.groupby(subject_col, as_index=False)["accuracy"]
                .mean()
                .rename(columns={subject_col: "subject"})
            )
            mean, low, high = bootstrap_ci(
                subject_values["accuracy"].to_numpy(float),
                seed=BOOT_SEED + int(pc) + (0 if role == "learner" else 100),
            )
            rows.append(
                {
                    "representation": "single_pc",
                    "pc": int(pc),
                    "role": role,
                    "mean_accuracy": mean,
                    "ci_low": low,
                    "ci_high": high,
                    "sd_subject_accuracy": float(subject_values["accuracy"].std(ddof=1)),
                    "n_subjects": int(len(subject_values)),
                    "n_pair_rows": int(len(subset)),
                }
            )

    summary = pd.DataFrame(rows).sort_values(["role", "pc"])
    summary.to_csv(SUMMARY, index=False)

    ranked = (
        summary.loc[summary["role"].eq("learner")]
        .sort_values("mean_accuracy", ascending=False)
        .reset_index(drop=True)
    )
    ranked.to_csv(RANKED, index=False)

    print("\nTop learner single-PC transfer")
    print(
        ranked[["pc", "mean_accuracy", "ci_low", "ci_high", "n_subjects"]]
        .head(10)
        .to_string(index=False),
        flush=True,
    )
    print(f"\nSaved {BY_PAIR}")
    print(f"Saved {SUMMARY}")
    print(f"Saved {RANKED}")


def main():
    global FEATURES, LABELS, OUT, BY_PAIR, SUMMARY, RANKED
    args = parse_args()
    FEATURES = args.features_csv
    LABELS = args.labels_csv
    OUT = args.out_dir
    BY_PAIR = OUT / "pc1_pc25_single_pc_transfer_by_pair.csv"
    SUMMARY = OUT / "pc1_pc25_single_pc_transfer_summary.csv"
    RANKED = OUT / "pc1_pc25_single_pc_transfer_ranked.csv"

    subjects, subject_x, subject_y = load_subject_data()
    print(f"loaded {len(subjects)} subjects", flush=True)
    run_transfer(subjects, subject_x, subject_y)
    summarize()


if __name__ == "__main__":
    main()
