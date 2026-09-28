from pathlib import Path
import argparse

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import LeaveOneOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


RANDOM_STATE = 42
K_LIST = [1, 5, 10, 25, 50, 100, 200]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run the manuscript AoA prediction analysis with 43 neural "
            "predictors and a max-statistic permutation test."
        )
    )
    parser.add_argument(
        "--predictor-table-csv",
        type=Path,
        default=None,
        help=(
            "Optional saved 43-predictor table. If omitted, the table is built "
            "from the source CSVs below."
        ),
    )
    parser.add_argument("--participant-table-csv", type=Path, default=None)
    parser.add_argument("--whole-cortex-by-subject-csv", type=Path, default=None)
    parser.add_argument("--self-decoding-csv", type=Path, default=None)
    parser.add_argument("--transfer-by-pair-csv", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--outcome-column", default=None)
    parser.add_argument("--ridge-alpha", type=float, default=1.0)
    parser.add_argument("--n-permutations", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=RANDOM_STATE)
    return parser.parse_args()


def require_path(path, name):
    if path is None:
        raise ValueError(f"{name} is required when --predictor-table-csv is omitted.")
    if not path.exists():
        raise FileNotFoundError(f"{name} not found: {path}")


def detect_outcome_column(df, requested):
    if requested is not None:
        if requested not in df.columns:
            raise ValueError(f"Requested outcome column {requested!r} not found.")
        return requested

    candidates = [
        "age",
        "AoA",
        "aoa",
        "age_of_acquisition",
        "AgeOfAcquisition",
        "Age_of_acquisition",
    ]
    for col in candidates:
        if col in df.columns:
            return col

    raise ValueError(
        "Could not detect the AoA outcome column. Pass --outcome-column COLUMN_NAME."
    )


def detect_subject_column(df):
    for col in ["subject", "participant_id", "participant", "sub"]:
        if col in df.columns:
            return col
    raise ValueError("Could not detect a subject column.")


def first_existing_column(df, candidates, table_name):
    for col in candidates:
        if col in df.columns:
            return col
    raise ValueError(
        f"Could not find any of {candidates} in {table_name}."
    )


def predictor_label(predictor):
    if predictor == "whole_cortex":
        return "Whole-cortex self-decoding"

    for prefix, label in [
        ("self_shared_k", "Shared-space self-decoding"),
        ("self_residual_k", "Residual self-decoding"),
        ("learner_shared_k", "Learner shared-space transfer"),
        ("teacher_shared_k", "Teacher shared-space transfer"),
        ("learner_residual_k", "Learner residual transfer"),
        ("teacher_residual_k", "Teacher residual transfer"),
    ]:
        if predictor.startswith(prefix):
            k = predictor.removeprefix(prefix)
            return f"{label} ({k} PCs)"

    return predictor


def build_predictor_table(args):
    require_path(args.participant_table_csv, "--participant-table-csv")
    require_path(args.whole_cortex_by_subject_csv, "--whole-cortex-by-subject-csv")
    require_path(args.self_decoding_csv, "--self-decoding-csv")
    require_path(args.transfer_by_pair_csv, "--transfer-by-pair-csv")

    participants = pd.read_csv(args.participant_table_csv)
    subject_col = detect_subject_column(participants)
    outcome_col = detect_outcome_column(participants, args.outcome_column)
    table = participants[[subject_col, outcome_col]].rename(
        columns={subject_col: "subject", outcome_col: "age"}
    )

    whole = pd.read_csv(args.whole_cortex_by_subject_csv)
    whole_subject_col = detect_subject_column(whole)
    whole_value_col = first_existing_column(
        whole,
        ["whole_cortex", "balanced_accuracy", "original_accuracy", "accuracy"],
        str(args.whole_cortex_by_subject_csv),
    )
    whole = whole[[whole_subject_col, whole_value_col]].rename(
        columns={whole_subject_col: "subject", whole_value_col: "whole_cortex"}
    )
    table = table.merge(whole, on="subject", how="left", validate="one_to_one")

    self_df = pd.read_csv(args.self_decoding_csv)
    required_self = {
        "subject",
        "n_components",
        "shared_space_accuracy",
        "residual_space_accuracy",
    }
    missing_self = required_self.difference(self_df.columns)
    if missing_self:
        raise ValueError(f"Self-decoding CSV missing columns: {sorted(missing_self)}")

    for k in K_LIST:
        subset = self_df.loc[self_df["n_components"].eq(k)]
        if subset.empty:
            raise ValueError(f"Self-decoding CSV has no rows for k={k}.")
        wide = subset[
            ["subject", "shared_space_accuracy", "residual_space_accuracy"]
        ].rename(
            columns={
                "shared_space_accuracy": f"self_shared_k{k}",
                "residual_space_accuracy": f"self_residual_k{k}",
            }
        )
        table = table.merge(wide, on="subject", how="left", validate="one_to_one")

    transfer = pd.read_csv(args.transfer_by_pair_csv)
    required_transfer = {
        "train_subject",
        "test_subject",
        "n_components",
        "shared_space_accuracy",
        "residual_space_accuracy",
    }
    missing_transfer = required_transfer.difference(transfer.columns)
    if missing_transfer:
        raise ValueError(
            f"Transfer CSV missing columns: {sorted(missing_transfer)}"
        )

    for k in K_LIST:
        subset = transfer.loc[transfer["n_components"].eq(k)]
        if subset.empty:
            raise ValueError(f"Transfer CSV has no rows for k={k}.")

        for role, subject_column in [
            ("teacher", "train_subject"),
            ("learner", "test_subject"),
        ]:
            means = (
                subset.groupby(subject_column, as_index=False)[
                    ["shared_space_accuracy", "residual_space_accuracy"]
                ]
                .mean()
                .rename(
                    columns={
                        subject_column: "subject",
                        "shared_space_accuracy": f"{role}_shared_k{k}",
                        "residual_space_accuracy": f"{role}_residual_k{k}",
                    }
                )
            )
            table = table.merge(means, on="subject", how="left", validate="one_to_one")

    predictors = predictor_columns(table)
    if len(predictors) != 43:
        raise ValueError(f"Expected 43 predictors, found {len(predictors)}.")

    return table


def predictor_columns(df):
    return [col for col in df.columns if col not in {"subject", "age"}]


def make_estimator(alpha):
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=alpha)),
        ]
    )


def loo_predict_single(x, y, alpha):
    x = np.asarray(x, dtype=float).reshape(-1, 1)
    y = np.asarray(y, dtype=float)
    pred = np.full(y.shape, np.nan, dtype=float)
    loo = LeaveOneOut()

    for train_idx, test_idx in loo.split(x):
        model = make_estimator(alpha)
        model.fit(x[train_idx], y[train_idx])
        pred[test_idx] = model.predict(x[test_idx])

    return pred


def summarize_prediction(y, pred, raw_predictor):
    pearson_r, pearson_p = stats.pearsonr(y, pred)
    spearman_rho, spearman_p = stats.spearmanr(y, pred)
    raw_r, raw_p = stats.pearsonr(y, raw_predictor)

    return {
        "pearson_r": float(pearson_r),
        "pearson_p_nominal": float(pearson_p),
        "spearman_rho": float(spearman_rho),
        "spearman_p_nominal": float(spearman_p),
        "cv_r2": float(r2_score(y, pred)),
        "raw_predictor_age_r": float(raw_r),
        "raw_predictor_age_p": float(raw_p),
    }


def run_observed(table, predictors, alpha):
    work = table[["subject", "age"] + predictors].copy()
    work = work.replace([np.inf, -np.inf], np.nan).dropna(subset=["age"])
    work = work.reset_index(drop=True)
    y = work["age"].to_numpy(float)

    rows = []
    predictions = {}
    for predictor in predictors:
        raw = work[predictor].to_numpy(float)
        pred = loo_predict_single(raw, y, alpha)
        row = {
            "label": predictor_label(predictor),
            "predictor": predictor,
        }
        row.update(summarize_prediction(y, pred, raw))
        rows.append(row)
        predictions[predictor] = pred

    summary = pd.DataFrame(rows).sort_values("cv_r2", ascending=False)
    return work, summary, predictions


def run_max_stat_permutation(table, predictors, alpha, n_permutations, seed):
    rng = np.random.default_rng(seed)
    work = table[["subject", "age"] + predictors].copy()
    work = work.replace([np.inf, -np.inf], np.nan).dropna(subset=["age"])
    work = work.reset_index(drop=True)
    y = work["age"].to_numpy(float)
    x_by_predictor = {
        predictor: work[predictor].to_numpy(float)
        for predictor in predictors
    }

    null_rows = []
    for permutation in range(n_permutations):
        y_perm = rng.permutation(y)
        max_cv_r2 = -np.inf

        for predictor in predictors:
            pred = loo_predict_single(x_by_predictor[predictor], y_perm, alpha)
            cv_r2 = r2_score(y_perm, pred)
            if cv_r2 > max_cv_r2:
                max_cv_r2 = cv_r2

        null_rows.append(
            {
                "permutation": int(permutation),
                "max_cv_r2": float(max_cv_r2),
            }
        )

        if (permutation + 1) % 100 == 0:
            print(f"Finished {permutation + 1} permutations", flush=True)

    return pd.DataFrame(null_rows)


def add_corrected_p(summary, null_df):
    null = null_df["max_cv_r2"].to_numpy(float)
    out = summary.copy()
    out["corrected_p_max_stat"] = [
        float((np.sum(null >= value) + 1) / (len(null) + 1))
        for value in out["cv_r2"].to_numpy(float)
    ]
    return out


def write_best_predictions(out_dir, work, summary, predictions):
    best = summary.sort_values("cv_r2", ascending=False).iloc[0]
    predictor = best["predictor"]
    out = work[["subject", "age", predictor]].copy()
    out["predicted_age"] = predictions[predictor]
    out["prediction_error"] = out["predicted_age"] - out["age"]
    out["best_label"] = best["label"]
    out.to_csv(out_dir / "age_best_predictions_43_predictors.csv", index=False)
    return best


def write_run_summary(out_dir, table, summary, null_df, best, n_permutations):
    row = {
        "n_subjects": int(table["age"].notna().sum()),
        "n_predictors": int(len(predictor_columns(table))),
        "n_permutations": int(n_permutations),
        "best_label": best["label"],
        "best_predictor": best["predictor"],
        "best_cv_r2": float(best["cv_r2"]),
        "best_pearson_r": float(best["pearson_r"]),
        "best_pearson_p_nominal": float(best["pearson_p_nominal"]),
        "best_spearman_rho": float(best["spearman_rho"]),
        "best_spearman_p_nominal": float(best["spearman_p_nominal"]),
        "best_corrected_p_max_stat": float(best["corrected_p_max_stat"]),
        "null_max_cv_r2_mean": float(null_df["max_cv_r2"].mean()),
        "null_max_cv_r2_95th": float(np.percentile(null_df["max_cv_r2"], 95)),
    }
    pd.DataFrame([row]).to_csv(
        out_dir / "age_prediction_max_stat_permutation_summary_43_predictors.csv",
        index=False,
    )


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.predictor_table_csv is None:
        table = build_predictor_table(args)
    else:
        table = pd.read_csv(args.predictor_table_csv)
        outcome_col = detect_outcome_column(table, args.outcome_column)
        if outcome_col != "age":
            table = table.rename(columns={outcome_col: "age"})

    predictors = predictor_columns(table)
    if len(predictors) != 43:
        raise ValueError(f"Expected 43 predictors, found {len(predictors)}.")

    predictor_table_path = args.out_dir / "age_prediction_predictor_table_43_predictors.csv"
    table.to_csv(predictor_table_path, index=False)

    observed_work, observed, predictions = run_observed(
        table, predictors, alpha=args.ridge_alpha
    )
    null_df = run_max_stat_permutation(
        table,
        predictors,
        alpha=args.ridge_alpha,
        n_permutations=args.n_permutations,
        seed=args.seed,
    )
    observed = add_corrected_p(observed, null_df)

    observed_path = args.out_dir / "age_prediction_max_stat_permutation_43_predictors.csv"
    null_path = args.out_dir / "age_prediction_max_stat_permutation_null_43_predictors.csv"
    observed.to_csv(observed_path, index=False)
    null_df.to_csv(null_path, index=False)

    best = write_best_predictions(args.out_dir, observed_work, observed, predictions)
    write_run_summary(args.out_dir, table, observed, null_df, best, args.n_permutations)

    print(observed.to_string(index=False))
    print(f"Saved {predictor_table_path}")
    print(f"Saved {observed_path}")
    print(f"Saved {null_path}")


if __name__ == "__main__":
    main()
