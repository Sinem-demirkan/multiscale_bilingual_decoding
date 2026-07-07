
"""
Crossed mixed-effects models for cross-subject transfer.

This script uses pymer4's Python interface for mixed-effects models.
"""

from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from pymer4.models import lmer


# Long-format cross-subject transfer results
# (e.g. cross_subject_transfer_long.csv from the transfer decoding script)
TRANSFER = Path("/home/sdemirka/fmri/duo-cogcon/outputs/cross_subject_transfer_long.csv")

# Per-subject within-subject decoding results, used as a self-accuracy covariate
# (e.g. distributed_parcel_mean_decoding_by_subject.csv)
SELF_DECODING = Path("/home/sdemirka/fmri/duo-cogcon/outputs/distributed_parcel_mean_decoding_by_subject.csv")

# Directory containing per-run tSNR maps, named like:
# sub-001_task-LanguageControl_run-02_tsnr.nii.gz
TSNR_DIR = Path("/home/sdemirka/fmri/duo-cogcon/derivatives/tsnr")

# Where to write the design matrix and all model outputs
OUT_DIR = Path("/home/sdemirka/fmri/duo-cogcon/outputs/mixed_models")

# Must match the task name used in the tSNR filenames (sub-XXX_task-<TASK>_run-YY_tsnr.nii.gz)
TASK = "LanguageControl"

# Restricted maximum likelihood (REML=True) vs full maximum likelihood (REML=False).
# REML is standard for reporting variance components / random effects;
# switch to False only if you need likelihood-ratio tests between fixed effects.
REML = True


FORMULA_UNADJUSTED = "balanced_accuracy ~ 1 + (1 | teacher) + (1 | learner)"

FORMULA_ADJUSTED = (
    "balanced_accuracy ~ "
    "teacher_selfacc_z + learner_selfacc_z + "
    "teacher_tsnr_z + learner_tsnr_z + "
    "teacher_selfacc_z:teacher_tsnr_z + "
    "learner_selfacc_z:learner_tsnr_z + "
    "(1 | teacher) + (1 | learner)"
)


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


def build_tsnr_table(tsnr_dir: Path, task: str) -> pd.DataFrame:
    # Averages per-run tSNR maps into a single mean_tsnr value per subject.
    # Voxels <= 0 are treated as background/outside-brain and excluded from
    # the mean; adjust this if your tSNR maps use a different convention
    # (e.g. NaN for background, or a separate brain mask file).
    tsnr_files = sorted(tsnr_dir.glob(f"sub-*_task-{task}_run-*_tsnr.nii.gz"))
    if len(tsnr_files) == 0:
        raise FileNotFoundError(f"No tSNR files found in {tsnr_dir} for task {task}")

    rows = []
    for f in tsnr_files:
        subject = f.name.split("_")[0]
        run = int(f.name.split("run-")[1].split("_")[0])

        data = nib.load(f).get_fdata()
        valid = data[np.isfinite(data) & (data > 0)]
        if valid.size == 0:
            print(subject, f"run {run}", "no valid tSNR voxels, skipping")
            continue

        rows.append({"subject": subject, "run": run, "run_mean_tsnr": float(valid.mean())})

    run_level = pd.DataFrame(rows)
    subject_level = (
        run_level.groupby("subject", as_index=False)["run_mean_tsnr"]
        .mean()
        .rename(columns={"run_mean_tsnr": "mean_tsnr"})
    )
    return subject_level


def prepare_design(transfer_path: Path, self_path: Path, qc_df: pd.DataFrame) -> pd.DataFrame:
    transfer = pd.read_csv(transfer_path)
    self_df = pd.read_csv(self_path)[["subject", "balanced_accuracy"]].rename(
        columns={"balanced_accuracy": "selfacc"}
    )
    qc = qc_df[["subject", "mean_tsnr"]].rename(columns={"mean_tsnr": "tsnr"})

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


OUT_DIR.mkdir(parents=True, exist_ok=True)

qc_df = build_tsnr_table(TSNR_DIR, TASK)
qc_df.to_csv(OUT_DIR / "mean_tsnr_by_subject.csv", index=False)

design = prepare_design(TRANSFER, SELF_DECODING, qc_df)
design.to_csv(OUT_DIR / "mixed_transfer_design.csv", index=False)

unadjusted = fit_lmer(FORMULA_UNADJUSTED, design, reml=REML)
adjusted = fit_lmer(FORMULA_ADJUSTED, design, reml=REML)

variance_table(unadjusted).to_csv(
    OUT_DIR / "mixed_unadjusted_variance_components.csv", index=False
)
random_effect_table(unadjusted).to_csv(
    OUT_DIR / "mixed_unadjusted_subject_effects_long.csv", index=False
)

variance_table(adjusted).to_csv(OUT_DIR / "mixed_variance_components.csv", index=False)
random_effect_table(adjusted).to_csv(OUT_DIR / "mixed_subject_effects_long.csv", index=False)
fixed_effect_table(adjusted).to_csv(OUT_DIR / "mixed_fixed_effects.csv", index=False)

write_logs(unadjusted, OUT_DIR / "mixed_unadjusted_summary.txt")
write_logs(adjusted, OUT_DIR / "mixed_adjusted_summary.txt")

print("Unadjusted variance components:")
print(variance_table(unadjusted))
print()
print("Adjusted variance components:")
print(variance_table(adjusted))
print()
print("Adjusted fixed effects:")
print(fixed_effect_table(adjusted))
