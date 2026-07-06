#!/usr/bin/env python
"""
Nested leave-one-subject-out AoA prediction using pymer4 mixed models.

For each held-out participant, the transfer mixed model is fit only on transfer
pairs among training participants. The held-out teacher and learner effects are
then estimated from heldout-to-training transfer pairs after subtracting the
fixed-effects prediction and the relevant training-participant random effect.
"""

import argparse
from pathlib import Path
from typing import Dict, List, Union

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


FORMULA = (
    "balanced_accuracy ~ "
    "teacher_selfacc_z + learner_selfacc_z + "
    "teacher_tsnr_z + learner_tsnr_z + "
    "teacher_selfacc_z:teacher_tsnr_z + "
    "learner_selfacc_z:learner_tsnr_z + "
    "(1 | teacher) + (1 | learner)"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Nested AoA prediction using pymer4 transfer effects.")
    parser.add_argument("--data-root", type=Path, help="Dataset root containing participants.tsv.")
    parser.add_argument("--participants", type=Path, help="Path to participants.tsv.")
    parser.add_argument("--aoa-column", help="AoA column name in participants.tsv.")
    parser.add_argument("--transfer", type=Path, required=True)
    parser.add_argument("--self-decoding", type=Path, required=True)
    parser.add_argument("--local-extent", type=Path)
    parser.add_argument("--qc", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--reml", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    if args.participants is None and args.data_root is None:
        parser.error("Provide either --participants or --data-root.")

    return args


def require_pymer4():
    try:
        from pymer4.models import lmer
    except ImportError as exc:
        raise ImportError(
            "This script requires pymer4 >= 0.9 and its backend dependencies."
        ) from exc
    return lmer


def as_pandas(obj) -> pd.DataFrame:
    if isinstance(obj, pd.DataFrame):
        return obj.copy()
    if hasattr(obj, "to_pandas"):
        return obj.to_pandas()
    return pd.DataFrame(obj)


def to_pymer_data(df: pd.DataFrame):
    try:
        import polars as pl
    except ImportError as exc:
        raise ImportError("pymer4 >= 0.9 requires polars. Install pymer4 and polars.") from exc
    return pl.from_pandas(df)


def read_participants(args: argparse.Namespace) -> pd.DataFrame:
    path = args.participants if args.participants is not None else args.data_root / "participants.tsv"
    participants = pd.read_csv(path, sep="\t")

    if "subject" in participants.columns:
        subject_col = "subject"
    elif "participant_id" in participants.columns:
        subject_col = "participant_id"
    else:
        raise ValueError("participants.tsv must contain either subject or participant_id.")

    if args.aoa_column:
        aoa_col = args.aoa_column
        if aoa_col not in participants.columns:
            raise ValueError(f"--aoa-column was not found in participants.tsv: {aoa_col}")
    else:
        candidates = ["AoA", "aoa", "age_of_acquisition", "AgeOfAcquisition", "age_acquisition"]
        found = [c for c in candidates if c in participants.columns]
        if not found:
            raise ValueError("Could not find an AoA column. Pass --aoa-column with the correct column name.")
        aoa_col = found[0]

    return participants[[subject_col, aoa_col]].rename(columns={subject_col: "subject", aoa_col: "AoA"})


def zscore_with(x: pd.Series, mean: float, sd: float) -> pd.Series:
    return (x - mean) / sd


# Note: uses typing.Dict instead of the built-in `dict[str, float]` subscript
# so this module doesn't require `from __future__ import annotations` or
# Python >= 3.9 to import cleanly.
def add_standardized_terms(df: pd.DataFrame, scaler: Dict[str, float]) -> pd.DataFrame:
    out = df.copy()
    out["teacher_selfacc_z"] = zscore_with(out["teacher_selfacc"], scaler["selfacc_mean"], scaler["selfacc_sd"])
    out["learner_selfacc_z"] = zscore_with(out["learner_selfacc"], scaler["selfacc_mean"], scaler["selfacc_sd"])
    out["teacher_tsnr_z"] = zscore_with(out["teacher_tsnr"], scaler["tsnr_mean"], scaler["tsnr_sd"])
    out["learner_tsnr_z"] = zscore_with(out["learner_tsnr"], scaler["tsnr_mean"], scaler["tsnr_sd"])
    return out


def train_standardizer(subject_covariates: pd.DataFrame, train_subjects: List[str]) -> Dict[str, float]:
    train_cov = subject_covariates.loc[subject_covariates["subject"].isin(train_subjects)]
    return {
        "selfacc_mean": train_cov["selfacc"].mean(),
        "selfacc_sd": train_cov["selfacc"].std(ddof=1),
        "tsnr_mean": train_cov["tsnr"].mean(),
        "tsnr_sd": train_cov["tsnr"].std(ddof=1),
    }


def prepare_inputs(args: argparse.Namespace):
    transfer = pd.read_csv(args.transfer)
    self_df = pd.read_csv(args.self_decoding)[["subject", "balanced_accuracy"]].rename(
        columns={"balanced_accuracy": "selfacc"}
    )
    qc = pd.read_csv(args.qc)[["subject", "mean_tsnr"]].rename(columns={"mean_tsnr": "tsnr"})
    participants = read_participants(args)

    subject_df = participants.merge(self_df, on="subject", how="inner").merge(qc, on="subject", how="inner")

    if args.local_extent is not None:
        local = pd.read_csv(args.local_extent)
        needed = {"subject", "prop_z_gt_1p64"}
        if not needed.issubset(local.columns):
            raise ValueError("--local-extent must contain subject and prop_z_gt_1p64")
        subject_df = subject_df.merge(local[["subject", "prop_z_gt_1p64"]], on="subject", how="inner")

    if "same_subject" in transfer.columns:
        df0 = transfer.loc[~transfer["same_subject"].isin([True, "True", "TRUE", 1])].copy()
    else:
        df0 = transfer.loc[transfer["train_subject"] != transfer["test_subject"]].copy()

    df0["teacher"] = df0["train_subject"]
    df0["learner"] = df0["test_subject"]

    df0 = df0.merge(
        self_df.rename(columns={"subject": "teacher", "selfacc": "teacher_selfacc"}),
        on="teacher",
        how="inner",
    )
    df0 = df0.merge(
        self_df.rename(columns={"subject": "learner", "selfacc": "learner_selfacc"}),
        on="learner",
        how="inner",
    )
    df0 = df0.merge(
        qc.rename(columns={"subject": "teacher", "tsnr": "teacher_tsnr"}),
        on="teacher",
        how="inner",
    )
    df0 = df0.merge(
        qc.rename(columns={"subject": "learner", "tsnr": "learner_tsnr"}),
        on="learner",
        how="inner",
    )

    subjects = sorted(set(df0["teacher"]).intersection(subject_df["subject"]))
    subject_covariates = subject_df[["subject", "selfacc", "tsnr"]].copy()

    return df0, subject_df, subject_covariates, subjects


def fit_lmer(data: pd.DataFrame, reml: bool):
    lmer = require_pymer4()
    model = lmer(FORMULA, data=to_pymer_data(data), REML=reml)
    model.fit(summary=False)
    return model


def random_effects(model, group: str) -> pd.Series:
    # model.ranef is expected to be a dict keyed by grouping factor
    # (e.g. {"teacher": DataFrame, "learner": DataFrame}) for crossed
    # random effects, rather than a single combined DataFrame.
    ranef = model.ranef
    if not isinstance(ranef, dict):
        raise ValueError("Expected crossed random effects to be returned as a dict.")

    group_df = as_pandas(ranef[group])
    level_col = "level" if "level" in group_df.columns else group_df.columns[0]
    effect_cols = [c for c in group_df.columns if c != level_col]
    intercept_col = "(Intercept)" if "(Intercept)" in effect_cols else effect_cols[0]
    return group_df.set_index(level_col)[intercept_col].astype(float)


def fixed_prediction(model, data: pd.DataFrame) -> np.ndarray:
    return np.asarray(model.predict(to_pymer_data(data), use_rfx=False), dtype=float)


def loo_linear_predictions(subject_df: pd.DataFrame, predictors: List[str], model_name: str) -> pd.DataFrame:
    dat = subject_df[["subject", "AoA", *predictors]].dropna().reset_index(drop=True)
    rows = []

    for subject in dat["subject"]:
        train = dat.loc[dat["subject"] != subject]
        test = dat.loc[dat["subject"] == subject]
        model = LinearRegression()
        model.fit(train[predictors], train["AoA"])
        pred = model.predict(test[predictors])[0]
        rows.append({"subject": subject, "observed": test["AoA"].iloc[0], "predicted": pred, "model": model_name})

    return pd.DataFrame(rows)


# Return type uses typing.Union instead of `float | str | int` so this
# module doesn't require Python >= 3.10 or the __future__ annotations import.
def score_model(model_name: str, observed: pd.Series, predicted: pd.Series) -> Dict[str, Union[float, str, int]]:
    keep = np.isfinite(observed) & np.isfinite(predicted)
    obs = np.asarray(observed[keep], dtype=float)
    pred = np.asarray(predicted[keep], dtype=float)
    pearson = stats.pearsonr(obs, pred)
    spearman = stats.spearmanr(obs, pred)
    return {
        "model": model_name,
        "n": len(obs),
        "pearson_r": pearson.statistic,
        "pearson_p": pearson.pvalue,
        "spearman_rho": spearman.statistic,
        "spearman_p": spearman.pvalue,
        "cv_r2": r2_score(obs, pred),
    }


def nested_transfer_predictions(
    df0: pd.DataFrame,
    subject_df: pd.DataFrame,
    subject_covariates: pd.DataFrame,
    subjects: List[str],
    reml: bool,
):
    prediction_rows = []
    effect_rows = []

    for target in subjects:
        train_subjects = [s for s in subjects if s != target]
        scaler = train_standardizer(subject_covariates, train_subjects)

        train_pairs = df0.loc[
            df0["teacher"].isin(train_subjects) & df0["learner"].isin(train_subjects)
        ].copy()
        train_pairs = add_standardized_terms(train_pairs, scaler)

        model = fit_lmer(train_pairs, reml=reml)
        teacher_effects = random_effects(model, "teacher")
        learner_effects = random_effects(model, "learner")

        train_effects = pd.DataFrame(
            {
                "subject": train_subjects,
                "teacher_effect": teacher_effects.loc[train_subjects].to_numpy(dtype=float),
                "learner_effect": learner_effects.loc[train_subjects].to_numpy(dtype=float),
            }
        ).merge(subject_df, on="subject", how="left")

        out_rows = df0.loc[df0["teacher"].eq(target) & df0["learner"].isin(train_subjects)].copy()
        out_rows = add_standardized_terms(out_rows, scaler)
        out_fixed = fixed_prediction(model, out_rows)
        heldout_teacher = np.mean(
            out_rows["balanced_accuracy"].to_numpy(dtype=float)
            - out_fixed
            - learner_effects.loc[out_rows["learner"]].to_numpy(dtype=float)
        )

        in_rows = df0.loc[df0["learner"].eq(target) & df0["teacher"].isin(train_subjects)].copy()
        in_rows = add_standardized_terms(in_rows, scaler)
        in_fixed = fixed_prediction(model, in_rows)
        heldout_learner = np.mean(
            in_rows["balanced_accuracy"].to_numpy(dtype=float)
            - in_fixed
            - teacher_effects.loc[in_rows["teacher"]].to_numpy(dtype=float)
        )

        target_row = subject_df.loc[subject_df["subject"].eq(target)].iloc[0]

        model_specs = {
            "pred_teacher_effect": ["teacher_effect"],
            "pred_learner_effect": ["learner_effect"],
            "pred_teacher_plus_learner": ["teacher_effect", "learner_effect"],
            "pred_learner_plus_selfdecoding": ["learner_effect", "selfacc"],
        }
        new_values = pd.DataFrame(
            [
                {
                    "teacher_effect": heldout_teacher,
                    "learner_effect": heldout_learner,
                    "selfacc": target_row["selfacc"],
                }
            ]
        )

        pred_row = {"subject": target, "observed": target_row["AoA"]}
        for col, predictors in model_specs.items():
            lm = LinearRegression()
            lm.fit(train_effects[predictors], train_effects["AoA"])
            pred_row[col] = lm.predict(new_values[predictors])[0]
        prediction_rows.append(pred_row)

        effect_rows.append(
            {
                "subject": target,
                "AoA": target_row["AoA"],
                "teacher_effect": heldout_teacher,
                "learner_effect": heldout_learner,
                "selfacc": target_row["selfacc"],
                "tsnr": target_row["tsnr"],
            }
        )
        print(f"Finished nested AoA fold for {target}", flush=True)

    return pd.DataFrame(prediction_rows), pd.DataFrame(effect_rows)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    df0, subject_df, subject_covariates, subjects = prepare_inputs(args)
    transfer_predictions, transfer_effects = nested_transfer_predictions(
        df0, subject_df, subject_covariates, subjects, reml=args.reml
    )

    long_predictions = pd.concat(
        [
            transfer_predictions[["subject", "observed", "pred_teacher_effect"]]
            .rename(columns={"pred_teacher_effect": "predicted"})
            .assign(model="Teacher effect"),
            transfer_predictions[["subject", "observed", "pred_learner_effect"]]
            .rename(columns={"pred_learner_effect": "predicted"})
            .assign(model="Learner effect"),
            transfer_predictions[["subject", "observed", "pred_teacher_plus_learner"]]
            .rename(columns={"pred_teacher_plus_learner": "predicted"})
            .assign(model="Teacher + Learner"),
            transfer_predictions[["subject", "observed", "pred_learner_plus_selfdecoding"]]
            .rename(columns={"pred_learner_plus_selfdecoding": "predicted"})
            .assign(model="Learner + self-decoding"),
        ],
        ignore_index=True,
    )

    baseline_predictions = [loo_linear_predictions(subject_df, ["selfacc"], "Self whole-cortex")]
    if "prop_z_gt_1p64" in subject_df.columns:
        baseline_predictions.append(loo_linear_predictions(subject_df, ["prop_z_gt_1p64"], "Local extent"))

    all_predictions = pd.concat([*baseline_predictions, long_predictions], ignore_index=True)
    summary = pd.DataFrame(
        [
            score_model(model, group["observed"], group["predicted"])
            for model, group in all_predictions.groupby("model")
        ]
    ).sort_values("cv_r2", ascending=False)

    transfer_predictions.to_csv(args.out_dir / "age_of_acquisition_nested_mixed_predictions_wide.csv", index=False)
    all_predictions.to_csv(args.out_dir / "age_of_acquisition_nested_mixed_predictions_long.csv", index=False)
    summary.to_csv(args.out_dir / "age_of_acquisition_nested_mixed_summary.csv", index=False)
    transfer_effects.to_csv(args.out_dir / "nested_transfer_effects_by_fold.csv", index=False)
    print(summary)


if __name__ == "__main__":
    main()
