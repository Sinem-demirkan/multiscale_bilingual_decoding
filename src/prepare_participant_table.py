"""
Build the participant-level table used by the individual-difference analyses.

Inputs are written by earlier scripts in this repository plus the Guo/OpenNeuro
participants.tsv file.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


AOA_CANDIDATES = [
    "AoA",
    "aoa",
    "age_of_english_acquisition",
    "AgeOfEnglishAcquisition",
    "english_aoa",
    "English_AoA",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare participant-level manuscript table.")
    parser.add_argument("--participants-tsv", type=Path, required=True)
    parser.add_argument("--whole-cortex-csv", type=Path, required=True)
    parser.add_argument("--cross-subject-transfer-csv", type=Path, required=True)
    parser.add_argument("--out-csv", type=Path, required=True)
    parser.add_argument("--local-extent-csv", type=Path, default=None)
    parser.add_argument("--aoa-column", default=None)
    return parser.parse_args()


def find_aoa_column(df: pd.DataFrame, requested: str | None) -> str:
    if requested:
        if requested not in df.columns:
            raise ValueError(f"Requested AoA column not found: {requested}")
        return requested

    for col in AOA_CANDIDATES:
        if col in df.columns:
            return col

    raise ValueError(
        "Could not infer AoA column. Use --aoa-column. "
        f"Available columns: {df.columns.tolist()}"
    )


def normalize_subject_column(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "participant_id" in out.columns and "subject" not in out.columns:
        out = out.rename(columns={"participant_id": "subject"})
    if "subject" not in out.columns:
        raise ValueError(f"No subject/participant_id column found: {out.columns.tolist()}")
    return out


def local_extent_table(path: Path | None) -> pd.DataFrame:
    if path is None:
        return pd.DataFrame(columns=["subject", "local_extent"])

    df = normalize_subject_column(pd.read_csv(path))
    candidates = [
        "local_extent",
        "extent",
        "extent_pct",
        "prop_z_gt_1p64",
        "prop_accuracy_gt_0p60",
        "n_significant",
    ]
    value_col = next((col for col in candidates if col in df.columns), None)
    if value_col is None:
        raise ValueError(
            "Could not infer local extent column. "
            f"Available columns: {df.columns.tolist()}"
        )

    out = df[["subject", value_col]].copy()
    out = out.rename(columns={value_col: "local_extent"})
    if "n_parcels" in df.columns and value_col == "n_significant":
        out["local_extent"] = df["n_significant"] / df["n_parcels"]
    return out


def main():
    args = parse_args()
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)

    participants = normalize_subject_column(pd.read_csv(args.participants_tsv, sep="\t"))
    aoa_col = find_aoa_column(participants, args.aoa_column)
    out = participants[["subject", aoa_col]].rename(columns={aoa_col: "AoA"}).copy()

    whole = normalize_subject_column(pd.read_csv(args.whole_cortex_csv))
    acc_col = "balanced_accuracy" if "balanced_accuracy" in whole.columns else "accuracy"
    out = out.merge(
        whole[["subject", acc_col]].rename(columns={acc_col: "whole_cortex"}),
        on="subject",
        how="left",
    )

    transfer = pd.read_csv(args.cross_subject_transfer_csv)
    if "same_subject" in transfer.columns:
        same = transfer["same_subject"].astype(str).str.lower().isin(["true", "1"])
        transfer = transfer.loc[~same].copy()
    else:
        transfer = transfer.loc[transfer["train_subject"] != transfer["test_subject"]].copy()

    grand = float(transfer["balanced_accuracy"].mean())
    teacher = (
        transfer.groupby("train_subject", as_index=False)["balanced_accuracy"]
        .mean()
        .rename(columns={"train_subject": "subject", "balanced_accuracy": "teacher_effect"})
    )
    learner = (
        transfer.groupby("test_subject", as_index=False)["balanced_accuracy"]
        .mean()
        .rename(columns={"test_subject": "subject", "balanced_accuracy": "learner_effect"})
    )
    teacher["teacher_effect"] = teacher["teacher_effect"] - grand
    learner["learner_effect"] = learner["learner_effect"] - grand

    out = out.merge(teacher, on="subject", how="left")
    out = out.merge(learner, on="subject", how="left")
    out = out.merge(local_extent_table(args.local_extent_csv), on="subject", how="left")
    out = out.sort_values("subject").reset_index(drop=True)

    out.to_csv(args.out_csv, index=False)
    print(f"Saved {args.out_csv}")


if __name__ == "__main__":
    main()
