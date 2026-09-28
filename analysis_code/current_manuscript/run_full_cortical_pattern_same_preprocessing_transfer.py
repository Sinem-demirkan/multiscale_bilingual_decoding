from pathlib import Path
import argparse
import ast
import itertools
import warnings

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import StandardScaler


warnings.filterwarnings("ignore", category=FutureWarning)

RANDOM_STATE = 42
SUMMARY_K = [1, 5, 10, 25, 50, 100, 200]
N_BOOT = 5000
BOOT_SEED = 42


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run full-cortical-pattern cross-participant transfer using the same "
            "per-participant z-scored pipeline as leave-pair-out shared/residual transfer."
        )
    )
    parser.add_argument("--features-csv", type=Path, required=True)
    parser.add_argument("--labels-csv", type=Path, required=True)
    parser.add_argument("--shared-transfer-csv", type=Path, required=True)
    parser.add_argument("--teacher-standardized-transfer-csv", type=Path, required=True)
    parser.add_argument("--within-self-csv", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--shared-script",
        type=Path,
        default=Path(__file__).with_name("run_shared_space_transfer_leave_pair_out.py"),
    )
    parser.add_argument("--sanity-pairs", type=int, default=20)
    parser.add_argument("--sanity-tol", type=float, default=1e-12)
    return parser.parse_args()


def refuse_existing_outputs(out_dir):
    outputs = [
        "full_cortical_pattern_same_preprocessing_transfer_by_pair.csv",
        "matched_full_pattern_transfer_by_pair.csv",
        "k1_shared_residual_leave_pair_out_by_pair.csv",
        "k800_sanity_check_by_pair.csv",
        "participant_level_transfer_summary.csv",
        "paired_differences_vs_full_cortical_pattern_same_preprocessing.csv",
        "paired_differences_vs_matched_full.csv",
        "within_person_full_cortical_pattern.csv",
        "within_person_reference.csv",
        "REPORT.md",
    ]
    existing = [out_dir / name for name in outputs if (out_dir / name).exists()]
    if existing:
        joined = "\n".join(str(path) for path in existing)
        raise FileExistsError(f"Refusing to overwrite existing output files:\n{joined}")


def read_k_list(shared_script):
    tree = ast.parse(shared_script.read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "K_LIST":
                    return list(ast.literal_eval(node.value))
    raise ValueError(f"Could not find K_LIST in {shared_script}")


def inspect_shared_script(shared_script, k_list):
    text = shared_script.read_text()
    differences = []
    if "StandardScaler().fit_transform(sub[parcel_cols].to_numpy(float))" not in text:
        differences.append("Did not find per-participant StandardScaler().fit_transform in load_subject_data.")
    if "pca = PCA(n_components=max_k, svd_solver=\"randomized\", random_state=RANDOM_STATE)" not in text:
        differences.append("Did not find randomized PCA with n_components=max_k and RANDOM_STATE.")
    if "max_k = max(K_LIST)" not in text:
        differences.append("Did not find max_k = max(K_LIST).")
    if max(k_list) != 200:
        differences.append(f"max(K_LIST) is {max(k_list)}, not 200.")

    transfer_start = text.index("def transfer_score")
    transfer_end = text.index("def run_leave_pair_out")
    transfer_text = text[transfer_start:transfer_end]
    if "StandardScaler" in transfer_text:
        differences.append("transfer_score appears to apply additional scaling.")
    return differences


def make_classifier():
    return LogisticRegression(
        C=0.01,
        solver="lbfgs",
        max_iter=1000,
        random_state=RANDOM_STATE,
    )


def load_subject_data(features_csv, labels_csv):
    features = pd.read_csv(features_csv)
    parcel_cols = [c for c in features.columns if c.startswith("parcel_")]
    features["trial_index"] = features["trial_index"].astype(int)

    labels = pd.read_csv(labels_csv).sort_values(["subject", "run", "trial_in_run"]).copy()
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


def run_full_cortical_pattern_same_preprocessing(subjects, subject_X, subject_y):
    rows = []
    for teacher in subjects:
        print(f"Full cortical pattern transfer with same preprocessing, teacher {teacher}", flush=True)
        for learner in subjects:
            if teacher == learner:
                continue
            rows.append(
                {
                    "train_subject": teacher,
                    "test_subject": learner,
                    "full_cortical_pattern_accuracy": transfer_score(
                        subject_X[teacher],
                        subject_y[teacher],
                        subject_X[learner],
                        subject_y[learner],
                    ),
                }
            )
    return pd.DataFrame(rows)


def run_k800_sanity(subjects, subject_X, subject_y, full_pattern_df, n_pairs):
    rng = np.random.default_rng(RANDOM_STATE)
    unordered = list(itertools.combinations(subjects, 2))
    chosen_idx = rng.choice(len(unordered), size=n_pairs, replace=False)
    chosen_pairs = [unordered[i] for i in chosen_idx]

    full_pattern = {
        (row.train_subject, row.test_subject): float(row.full_cortical_pattern_accuracy)
        for row in full_pattern_df.itertuples(index=False)
    }

    rows = []
    for a, b in chosen_pairs:
        print(f"k=800 sanity check, pair {a}/{b}", flush=True)
        pca_subjects = [s for s in subjects if s not in {a, b}]
        X_pool = np.vstack([subject_X[s] for s in pca_subjects])
        pca = PCA(n_components=800, svd_solver="full")
        pca.fit(X_pool)
        components = pca.components_

        for teacher, learner in [(a, b), (b, a)]:
            projected = transfer_score(
                subject_X[teacher] @ components.T,
                subject_y[teacher],
                subject_X[learner] @ components.T,
                subject_y[learner],
            )
            full_pattern_accuracy = full_pattern[(teacher, learner)]
            rows.append(
                {
                    "train_subject": teacher,
                    "test_subject": learner,
                    "full_cortical_pattern_accuracy": full_pattern_accuracy,
                    "pca800_accuracy": projected,
                    "abs_difference": abs(projected - full_pattern_accuracy),
                }
            )
    return pd.DataFrame(rows)


def run_k1_shared_residual(subjects, subject_X, subject_y):
    rows = []
    component_cache = {}
    for teacher in subjects:
        print(f"k=1 shared/residual transfer, teacher {teacher}", flush=True)
        for learner in subjects:
            if teacher == learner:
                continue
            pair_key = tuple(sorted((teacher, learner)))
            if pair_key not in component_cache:
                pca_subjects = [s for s in subjects if s not in {teacher, learner}]
                X_pool = np.vstack([subject_X[s] for s in pca_subjects])
                pca = PCA(n_components=1, svd_solver="randomized", random_state=RANDOM_STATE)
                pca.fit(X_pool)
                component_cache[pair_key] = pca.components_

            components = component_cache[pair_key]
            rows.append(
                {
                    "train_subject": teacher,
                    "test_subject": learner,
                    "n_components": 1,
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
    return pd.DataFrame(rows)


def participant_values(pair_df, value_col):
    teacher = pair_df.groupby("train_subject")[value_col].mean()
    learner = pair_df.groupby("test_subject")[value_col].mean()
    subjects = sorted(set(teacher.index) | set(learner.index))
    values = pd.DataFrame(
        {
            "subject": subjects,
            "teacher_mean": [teacher.loc[s] for s in subjects],
            "learner_mean": [learner.loc[s] for s in subjects],
        }
    )
    values["participant_value"] = values[["teacher_mean", "learner_mean"]].mean(axis=1)
    return values


def bootstrap_mean_ci(values, seed):
    x = np.asarray(values, dtype=float)
    mean = float(np.mean(x))
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(x), size=(N_BOOT, len(x)))
    boot = x[indices].mean(axis=1)
    ci_low, ci_high = np.percentile(boot, [2.5, 97.5])
    return mean, float(ci_low), float(ci_high)


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


def condition_boot_seed(condition, k):
    if condition in {"full_cortical_pattern_same_preprocessing", "teacher_standardized_full"}:
        return BOOT_SEED + 1000
    k_index = SUMMARY_K.index(int(k))
    if condition == "shared":
        return BOOT_SEED + 100 + k_index
    if condition == "residual":
        return BOOT_SEED + 200 + k_index
    return BOOT_SEED


def summarize_condition(condition, k, pair_df, value_col):
    vals = participant_values(pair_df, value_col)
    mean, ci_low, ci_high = bootstrap_pair_cluster_mean_ci(
        pair_df,
        value_col,
        seed=condition_boot_seed(condition, k),
    )
    return {
        "condition": condition,
        "k": "" if k is None else int(k),
        "mean": mean,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "n_participants": int(len(vals)),
        "pair_level_mean": float(pair_df[value_col].mean()),
    }, vals[["subject", "participant_value"]].rename(columns={"participant_value": condition_key(condition, k)})


def condition_key(condition, k):
    return condition if k is None else f"{condition}_k{k}"


def off_diagonal_teacher_standardized(path):
    df = pd.read_csv(path)
    if "same_subject" in df.columns:
        df = df.loc[~df["same_subject"].astype(str).str.lower().eq("true")].copy()
    return df[["train_subject", "test_subject", "balanced_accuracy"]].rename(
        columns={"balanced_accuracy": "teacher_standardized_full_accuracy"}
    )


def build_shared_source(existing_shared, k1_df):
    parts = []
    existing = pd.read_csv(existing_shared)
    for k in SUMMARY_K:
        tmp = existing.loc[existing["n_components"].astype(int).eq(k)].copy()
        if k == 1 and k1_df is not None:
            tmp = k1_df.copy()
        if tmp.empty:
            raise ValueError(f"Missing shared/residual transfer rows for k={k}")
        parts.append(tmp)
    return pd.concat(parts, ignore_index=True)


def participant_summary(full_pattern_df, teacher_df, shared_df):
    rows = []
    value_tables = []

    row, vals = summarize_condition("full_cortical_pattern_same_preprocessing", None, full_pattern_df, "full_cortical_pattern_accuracy")
    rows.append(row)
    value_tables.append(vals)

    row, vals = summarize_condition(
        "teacher_standardized_full",
        None,
        teacher_df,
        "teacher_standardized_full_accuracy",
    )
    rows.append(row)
    value_tables.append(vals)

    for k in SUMMARY_K:
        tmp = shared_df.loc[shared_df["n_components"].astype(int).eq(k)].copy()
        row, vals = summarize_condition("shared", k, tmp, "shared_space_accuracy")
        rows.append(row)
        value_tables.append(vals)
        row, vals = summarize_condition("residual", k, tmp, "residual_space_accuracy")
        rows.append(row)
        value_tables.append(vals)

    summary = pd.DataFrame(rows)
    values = value_tables[0]
    for vals in value_tables[1:]:
        values = values.merge(vals, on="subject", how="inner")
    return summary, values


def paired_differences(full_pattern_df, shared_df, summary):
    full_pattern_mean = float(summary.loc[summary["condition"].eq("full_cortical_pattern_same_preprocessing"), "mean"].iloc[0])
    rows = []
    for condition in ["shared", "residual"]:
        for k in SUMMARY_K:
            value_col = f"{condition}_minus_full_cortical_pattern_same_preprocessing"
            metric = (
                "shared_space_accuracy"
                if condition == "shared"
                else "residual_space_accuracy"
            )
            tmp = (
                shared_df.loc[shared_df["n_components"].astype(int).eq(k)]
                .merge(
                    full_pattern_df[
                        ["train_subject", "test_subject", "full_cortical_pattern_accuracy"]
                    ],
                    on=["train_subject", "test_subject"],
                    how="inner",
                    validate="many_to_one",
                )
                .copy()
            )
            tmp[value_col] = tmp[metric] - tmp["full_cortical_pattern_accuracy"]
            mean, ci_low, ci_high = bootstrap_pair_cluster_mean_ci(
                tmp,
                value_col,
                seed=BOOT_SEED + 700 + SUMMARY_K.index(int(k)),
            )
            condition_mean = float(
                summary.loc[
                    summary["condition"].eq(condition) & summary["k"].astype(str).eq(str(k)),
                    "mean",
                ].iloc[0]
            )
            retained = float((condition_mean - 0.5) / (full_pattern_mean - 0.5))
            rows.append(
                {
                    "condition": condition,
                    "k": int(k),
                    "mean_difference": mean,
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                    "retained_fraction": retained,
                }
            )
    return pd.DataFrame(rows)


def within_full_cortical_pattern(within_self_csv, full_pattern_mean):
    df = pd.read_csv(within_self_csv)
    if "original_accuracy" not in df.columns:
        raise ValueError(f"{within_self_csv} lacks original_accuracy")
    vals = df[["subject", "original_accuracy"]].drop_duplicates("subject").copy()
    mean, ci_low, ci_high = bootstrap_mean_ci(
        vals["original_accuracy"],
        seed=BOOT_SEED + 2000,
    )
    ratio = float((full_pattern_mean - 0.5) / (mean - 0.5))
    return pd.DataFrame(
        [
            {
                "condition": "within_person_original",
                "mean": mean,
                "ci_low": ci_low,
                "ci_high": ci_high,
                "n_participants": int(len(vals)),
                "transfer_within_ratio": ratio,
            }
        ]
    )


def fmt_ci(row):
    return f"{row['mean']} [{row['ci_low']}, {row['ci_high']}]"


def make_report(
    args,
    k_list,
    differences,
    summary,
    paired,
    sanity,
    within,
    full_pattern_path,
    k1_path,
):
    lines = []
    lines.append("# Full Cortical Pattern, Same Preprocessing Transfer Report")
    lines.append("")
    lines.append("1. Local state")
    lines.append(f"- K_LIST: {k_list}")
    if differences:
        lines.append("- Differences from expected script state:")
        lines.extend(f"  - {item}" for item in differences)
    else:
        lines.append("- Differences from expected script state: none detected")
    lines.append(f"- Existing shared transfer CSV: {args.shared_transfer_csv}")
    lines.append(f"- Existing teacher-standardized transfer CSV: {args.teacher_standardized_transfer_csv}")
    lines.append(f"- Full cortical pattern, same preprocessing output CSV: {full_pattern_path}")
    if k1_path is not None:
        lines.append(f"- k=1 shared/residual output CSV: {k1_path}")
    lines.append("")

    full_pattern = summary.loc[summary["condition"].eq("full_cortical_pattern_same_preprocessing")].iloc[0]
    teacher = summary.loc[summary["condition"].eq("teacher_standardized_full")].iloc[0]
    lines.append("2. Full cortical pattern transfer with same preprocessing")
    lines.append(f"- {fmt_ci(full_pattern)}")
    lines.append("")
    lines.append("3. Teacher-standardized full-pattern transfer")
    lines.append(f"- {fmt_ci(teacher)}")
    lines.append("")
    lines.append("4. k=800 equivalence sanity check")
    lines.append(f"- maximum absolute difference: {sanity['abs_difference'].max()}")
    lines.append("")
    lines.append("5. Shared-space transfer")
    for k in SUMMARY_K:
        row = summary.loc[summary["condition"].eq("shared") & summary["k"].astype(str).eq(str(k))].iloc[0]
        lines.append(f"- k={k}: {fmt_ci(row)}")
    lines.append("")
    lines.append("6. Residual transfer")
    for k in SUMMARY_K:
        row = summary.loc[summary["condition"].eq("residual") & summary["k"].astype(str).eq(str(k))].iloc[0]
        lines.append(f"- k={k}: {fmt_ci(row)}")
    lines.append("")
    lines.append("7. Paired differences vs full cortical pattern with same preprocessing")
    for row in paired.itertuples(index=False):
        lines.append(
            f"- {row.condition} k={row.k}: mean_difference={row.mean_difference}, "
            f"95% CI=[{row.ci_low}, {row.ci_high}]"
        )
    lines.append("")
    lines.append("8. Retained fraction")
    for row in paired.itertuples(index=False):
        lines.append(f"- {row.condition} k={row.k}: {row.retained_fraction}")
    lines.append("")
    lines.append("9. Within-person full cortical pattern")
    w = within.iloc[0]
    lines.append(f"- original_accuracy: {fmt_ci(w)}")
    lines.append(f"- transfer/within ratio: {w['transfer_within_ratio']}")
    lines.append("")
    lines.append("10. Smallest k where shared-space transfer is not significantly below full cortical pattern with same preprocessing")
    candidates = paired.loc[
        paired["condition"].eq("shared")
        & ((paired["ci_low"] <= 0) | (paired["mean_difference"] >= 0))
    ].sort_values("k")
    if candidates.empty:
        lines.append("- none")
    else:
        lines.append(f"- k={int(candidates.iloc[0]['k'])}")
    lines.append("")
    return "\n".join(lines)


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    refuse_existing_outputs(args.out_dir)

    k_list = read_k_list(args.shared_script)
    differences = inspect_shared_script(args.shared_script, k_list)
    print("Step 0 local state", flush=True)
    print(f"K_LIST: {k_list}", flush=True)
    print(f"Differences from expected script state: {differences if differences else 'none detected'}", flush=True)
    print(f"Existing shared transfer CSV: {args.shared_transfer_csv}", flush=True)
    print(f"Existing teacher-standardized transfer CSV: {args.teacher_standardized_transfer_csv}", flush=True)

    subjects, subject_X, subject_y = load_subject_data(args.features_csv, args.labels_csv)
    if len(subjects) != 77:
        raise ValueError(f"Expected 77 participants, found {len(subjects)}")

    full_pattern = run_full_cortical_pattern_same_preprocessing(subjects, subject_X, subject_y)
    full_pattern_path = args.out_dir / "full_cortical_pattern_same_preprocessing_transfer_by_pair.csv"
    full_pattern.to_csv(full_pattern_path, index=False)

    sanity = run_k800_sanity(subjects, subject_X, subject_y, full_pattern, args.sanity_pairs)
    sanity_path = args.out_dir / "k800_sanity_check_by_pair.csv"
    sanity.to_csv(sanity_path, index=False)
    max_diff = float(sanity["abs_difference"].max())
    print(f"k=800 sanity maximum absolute difference: {max_diff}", flush=True)
    if max_diff > args.sanity_tol:
        raise RuntimeError(
            f"k=800 sanity check failed: maximum absolute difference {max_diff} exceeds {args.sanity_tol}"
        )

    k1 = None
    k1_path = None
    if 1 not in k_list:
        k1 = run_k1_shared_residual(subjects, subject_X, subject_y)
        k1_path = args.out_dir / "k1_shared_residual_leave_pair_out_by_pair.csv"
        k1.to_csv(k1_path, index=False)

    teacher = off_diagonal_teacher_standardized(args.teacher_standardized_transfer_csv)
    shared = build_shared_source(args.shared_transfer_csv, k1)
    summary, values = participant_summary(full_pattern, teacher, shared)
    summary_path = args.out_dir / "participant_level_transfer_summary.csv"
    summary.to_csv(summary_path, index=False)

    paired = paired_differences(full_pattern, shared, summary)
    paired_path = args.out_dir / "paired_differences_vs_full_cortical_pattern_same_preprocessing.csv"
    paired.to_csv(paired_path, index=False)

    full_pattern_mean = float(summary.loc[summary["condition"].eq("full_cortical_pattern_same_preprocessing"), "mean"].iloc[0])
    within = within_full_cortical_pattern(args.within_self_csv, full_pattern_mean)
    within_path = args.out_dir / "within_person_full_cortical_pattern.csv"
    within.to_csv(within_path, index=False)

    report = make_report(
        args,
        k_list,
        differences,
        summary,
        paired,
        sanity,
        within,
        full_pattern_path,
        k1_path,
    )
    report_path = args.out_dir / "REPORT.md"
    report_path.write_text(report + "\n")
    print(report, flush=True)


if __name__ == "__main__":
    main()
