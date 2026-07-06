#!/usr/bin/env python
"""
Crossed mixed-effects models for cross-subject transfer.

This script uses pymer4's Python interface for mixed-effects models.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


FORMULA_UNADJUSTED = "balanced_accuracy ~ 1 + (1 | teacher) + (1 | learner)"

FORMULA_ADJUSTED = (
    "balanced_accuracy ~ "
    "teacher_selfacc_z + learner_selfacc_z + "
    "teacher_tsnr_z + learner_tsnr_z + "
    "teacher_selfacc_z:teacher_tsnr_z + "
    "learner_selfacc_z:learner_tsnr_z + "
    "(1 | teacher) + (1 | learner)"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fit pymer4 mixed-effects transfer models.")
    parser.add_argument("--transfer", type=Path, required=True)
    parser.add_argument("--self-decoding", type=Path, required=True)
    parser.add_argument("--qc", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--reml", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def require_pymer4():
    # Returns the pymer4 `lmer` class. Imported lazily (rather than at module
    # level) so the script can still be imported/used for non-modeling helpers
    # (e.g. prepare_design) on machines without pymer4/R installed.
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
    # pymer4 >= 0.9 expects a polars DataFrame, not pandas, so convert here.
    try:
        import polars as pl
    except ImportError as exc:
        raise ImportError("pymer4 >= 0.9 requires polars. Install pymer4 and polars.") from exc
    return pl.from_pandas(df)


def zscore(x: pd.Series) -> pd.Series:
    return (x - x.mean()) / x.std(ddof=1)


def prepare_design(transfer_path: Path, self_path: Path, qc_path: Path) -> pd.DataFrame:
    transfer = pd.read_csv(transfer_path)
    self_df = pd.read_csv(self_path)[["subject", "balanced_accuracy"]].rename(
        columns={"balanced_accuracy": "selfacc"}
    )
    qc = pd.read_csv(qc_path)[["subject", "mean_tsnr"]].rename(columns={"mean_tsnr": "tsnr"})

    covariates = self_df.merge(qc, on="subject", how="inner")
    covariates["selfacc_z"] = zscore(covariates["selfacc"])
    covariates["tsnr_z"] = zscore(covariates["tsnr"])

    if "same_subject" in transfer.columns:
        df = transfer.loc[~transfer["same_subject"].isin([True, "True", "TRUE", 1])].copy()
    else:
        df = transfer.loc[transfer["train_subject"] != transfer["test_subject"]].copy()

    df["teacher"] = df["train_subject"]
    df["learner"] = df["test_subject"]

    df = df.merge(
        covariates.add_prefix("teacher_").rename(columns={"teacher_subject": "teacher"}),
        on="teacher",
        how="left",
    )
    df = df.merge(
        covariates.add_prefix("learner_").rename(columns={"learner_subject": "learner"}),
        on="learner",
        how="left",
    )

    required = [
        "balanced_accuracy",
        "teacher_selfacc",
        "learner_selfacc",
        "teacher_tsnr",
        "learner_tsnr",
        "teacher_selfacc_z",
        "learner_selfacc_z",
        "teacher_tsnr_z",
        "learner_tsnr_z",
    ]
    return df.dropna(subset=required).reset_index(drop=True)


def fit_lmer(formula: str, data: pd.DataFrame, reml: bool = True):
    lmer = require_pymer4()
    model = lmer(formula, data=to_pymer_data(data), REML=reml)
    model.fit(summary=False)
    return model


def variance_table(model) -> pd.DataFrame:
    ranef_var = as_pandas(model.ranef_var)
    rows = []

    for _, row in ranef_var.iterrows():
        group = str(row.get("group", ""))
        term = str(row.get("term", ""))
        estimate = float(row.get("estimate", np.nan))

        if group == "teacher":
            component = "Teacher"
        elif group == "learner":
            component = "Learner"
        elif group.lower() == "residual" or "residual" in term.lower() or "observation" in term.lower():
            component = "Pairwise residual"
        else:
            continue

        rows.append({"component": component, "variance": estimate**2})

    out = pd.DataFrame(rows).groupby("component", as_index=False)["variance"].sum()
    out["pct"] = 100.0 * out["variance"] / out["variance"].sum()
    order = ["Pairwise residual", "Teacher", "Learner"]
    out["component"] = pd.Categorical(out["component"], categories=order, ordered=True)
    return out.sort_values("component").reset_index(drop=True)


def random_effect_table(model) -> pd.DataFrame:
    # model.ranef is expected to be a dict keyed by grouping factor
    # (e.g. {"teacher": DataFrame, "learner": DataFrame}) for crossed
    # random effects, rather than a single combined DataFrame.
    ranef = model.ranef
    if not isinstance(ranef, dict):
        raise ValueError("Expected crossed random effects to be returned as a dict.")

    rows = []
    for group, role in [("teacher", "Teacher"), ("learner", "Learner")]:
        group_df = as_pandas(ranef[group])
        level_col = "level" if "level" in group_df.columns else group_df.columns[0]
        effect_cols = [c for c in group_df.columns if c != level_col]
        intercept_col = "(Intercept)" if "(Intercept)" in effect_cols else effect_cols[0]

        tmp = group_df[[level_col, intercept_col]].rename(
            columns={level_col: "subject", intercept_col: "effect"}
        )
        tmp["role"] = role
        rows.append(tmp[["subject", "role", "effect"]])

    return pd.concat(rows, ignore_index=True)


def fixed_effect_table(model) -> pd.DataFrame:
    result = as_pandas(model.result_fit)
    rename = {
        "term": "term",
        "estimate": "estimate",
        "std_error": "std_error",
        "df": "df",
        "t": "t_value",
        "t_stat": "t_value",
        "statistic": "t_value",
        "conf_low": "conf_low",
        "conf_high": "conf_high",
    }
    result = result.rename(columns={k: v for k, v in rename.items() if k in result.columns})

    keep = [c for c in ["term", "estimate", "std_error", "df", "t_value", "conf_low", "conf_high"] if c in result.columns]
    return result[keep].copy()


def write_logs(model, path: Path) -> None:
    logs = getattr(model, "r_console", [])
    summary = as_pandas(getattr(model, "result_fit", pd.DataFrame()))
    with path.open("w") as f:
        if len(logs):
            f.write("\n".join(map(str, logs)))
            f.write("\n\n")
        f.write(summary.to_string(index=False))
        f.write("\n")


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    design = prepare_design(args.transfer, args.self_decoding, args.qc)
    design.to_csv(args.out_dir / "mixed_transfer_design.csv", index=False)

    unadjusted = fit_lmer(FORMULA_UNADJUSTED, design, reml=args.reml)
    adjusted = fit_lmer(FORMULA_ADJUSTED, design, reml=args.reml)

    variance_table(unadjusted).to_csv(
        args.out_dir / "mixed_unadjusted_variance_components.csv", index=False
    )
    random_effect_table(unadjusted).to_csv(
        args.out_dir / "mixed_unadjusted_subject_effects_long.csv", index=False
    )

    variance_table(adjusted).to_csv(args.out_dir / "mixed_variance_components.csv", index=False)
    random_effect_table(adjusted).to_csv(args.out_dir / "mixed_subject_effects_long.csv", index=False)
    fixed_effect_table(adjusted).to_csv(args.out_dir / "mixed_fixed_effects.csv", index=False)

    write_logs(unadjusted, args.out_dir / "mixed_unadjusted_summary.txt")
    write_logs(adjusted, args.out_dir / "mixed_adjusted_summary.txt")


if __name__ == "__main__":
    main()
