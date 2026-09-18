from pathlib import Path
import argparse

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import balanced_accuracy_score, r2_score
from sklearn.model_selection import LeaveOneOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


RANDOM_STATE = 42

BASE = None
PARTICIPANT_TABLE = None
LEAVE_PAIR_TRANSFER = None
FEATURES = None
LABELS = None
PCA_BY_SUBJECT = None
AOA_SUMMARY = None


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run leave-one-participant-out AoA prediction analyses."
    )
    parser.add_argument("--participant-table-csv", type=Path, required=True)
    parser.add_argument("--leave-pair-transfer-csv", type=Path, required=True)
    parser.add_argument("--features-csv", type=Path, required=True)
    parser.add_argument("--labels-csv", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def configure_paths(args):
    global BASE, PARTICIPANT_TABLE, LEAVE_PAIR_TRANSFER, FEATURES, LABELS
    global PCA_BY_SUBJECT, AOA_SUMMARY
    BASE = args.out_dir
    PARTICIPANT_TABLE = args.participant_table_csv
    LEAVE_PAIR_TRANSFER = args.leave_pair_transfer_csv
    FEATURES = args.features_csv
    LABELS = args.labels_csv
    PCA_BY_SUBJECT = BASE / "whole_cortex_pca_self_decoding_by_subject.csv"
    AOA_SUMMARY = BASE / "aoa_prediction_with_whole_cortex_pca_summary.csv"


def make_decoding_classifier():
    return LogisticRegression(
        C=0.01,
        solver="lbfgs",
        max_iter=1000,
        random_state=RANDOM_STATE,
    )


def make_aoa_estimator():
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=1.0)),
        ]
    )


def compute_whole_cortex_pca_self_decoding(k_values=(1, 5, 10, 50)):
    if PCA_BY_SUBJECT.exists():
        existing = pd.read_csv(PCA_BY_SUBJECT)
        needed = [f"whole_cortex_pca{k}" for k in k_values]
        if all(col in existing.columns for col in needed):
            return existing

    features = pd.read_csv(FEATURES)
    parcel_cols = [c for c in features.columns if c.startswith("parcel_")]
    features["trial_index"] = features["trial_index"].astype(int)

    labels = pd.read_csv(LABELS).sort_values(["subject", "run", "trial_in_run"]).copy()
    labels["trial_index"] = labels.groupby("subject").cumcount() + 1
    labels = labels[["subject", "trial_index", "run", "true_y"]]

    df = features.merge(labels, on=["subject", "trial_index"], how="inner")

    rows = []
    for subject, sub in df.groupby("subject", sort=True):
        sub = sub.sort_values("trial_index")
        x = sub[parcel_cols].to_numpy(float)
        y = sub["true_y"].to_numpy(int)
        runs = sub["run"].to_numpy()
        pred_by_k = {k: np.full(y.shape, np.nan) for k in k_values}

        for run in np.unique(runs):
            test = runs == run
            train = ~test

            scaler = StandardScaler()
            x_train = scaler.fit_transform(x[train])
            x_test = scaler.transform(x[test])

            max_k = max(k_values)
            pca = PCA(n_components=max_k, svd_solver="full")
            x_train_pca = pca.fit_transform(x_train)
            x_test_pca = pca.transform(x_test)

            for k in k_values:
                clf = make_decoding_classifier()
                clf.fit(x_train_pca[:, :k], y[train])
                pred_by_k[k][test] = clf.predict(x_test_pca[:, :k])

        row = {"subject": subject}
        for k in k_values:
            row[f"whole_cortex_pca{k}"] = float(
                balanced_accuracy_score(y, pred_by_k[k])
            )
        rows.append(row)

    out = pd.DataFrame(rows)
    out.to_csv(PCA_BY_SUBJECT, index=False)
    return out


def add_leave_pair_out_shared_transfer_features(participants):
    transfer = pd.read_csv(LEAVE_PAIR_TRANSFER)
    out = participants.copy()

    for k, tmp in transfer.groupby("n_components"):
        learner = (
            tmp.groupby("test_subject", as_index=False)["shared_space_accuracy"]
            .mean()
            .rename(
                columns={
                    "test_subject": "subject",
                    "shared_space_accuracy": f"learner_shared_k{k}",
                }
            )
        )

        teacher = (
            tmp.groupby("train_subject", as_index=False)["shared_space_accuracy"]
            .mean()
            .rename(
                columns={
                    "train_subject": "subject",
                    "shared_space_accuracy": f"teacher_shared_k{k}",
                }
            )
        )

        out = out.merge(learner, on="subject", how="left")
        out = out.merge(teacher, on="subject", how="left")

    return out


def loo_predictions(df, predictors, outcome="AoA"):
    cols = ["subject"] + predictors + [outcome]
    tmp = (
        df[cols]
        .replace([np.inf, -np.inf], np.nan)
        .dropna(subset=[outcome])
        .reset_index(drop=True)
    )

    x = tmp[predictors].to_numpy(float)
    y = tmp[outcome].to_numpy(float)
    pred = np.full(len(tmp), np.nan)

    loo = LeaveOneOut()
    for train_idx, test_idx in loo.split(x, y):
        model = make_aoa_estimator()
        model.fit(x[train_idx], y[train_idx])
        pred[test_idx] = model.predict(x[test_idx])

    out = tmp[["subject"]].copy()
    out["observed"] = y
    out["predicted"] = pred
    return out


def prediction_summary(pred_df):
    observed = pred_df["observed"].to_numpy(float)
    predicted = pred_df["predicted"].to_numpy(float)

    r, p = stats.pearsonr(observed, predicted)
    rho, ps = stats.spearmanr(observed, predicted)

    return {
        "n": len(pred_df),
        "pearson_r": float(r),
        "pearson_p": float(p),
        "spearman_rho": float(rho),
        "spearman_p": float(ps),
        "cv_r2": float(r2_score(observed, predicted)),
    }


def main():
    args = parse_args()
    configure_paths(args)
    BASE.mkdir(parents=True, exist_ok=True)
    participants = pd.read_csv(PARTICIPANT_TABLE)
    pca_self = compute_whole_cortex_pca_self_decoding()
    participants = participants.merge(pca_self, on="subject", how="left")
    participants = add_leave_pair_out_shared_transfer_features(participants)

    predictor_specs = [
        ("Whole-cortex self-decoding", ["whole_cortex"]),
        ("Whole-cortex 1-PC self-decoding", ["whole_cortex_pca1"]),
        ("Whole-cortex 5-PC self-decoding", ["whole_cortex_pca5"]),
        ("Whole-cortex 10-PC self-decoding", ["whole_cortex_pca10"]),
        ("Whole-cortex 50-PC self-decoding", ["whole_cortex_pca50"]),
        ("Teacher effect", ["teacher_effect"]),
        ("Learner effect", ["learner_effect"]),
        ("Teacher shared space (10 PCs)", ["teacher_shared_k10"]),
        ("Learner shared space (10 PCs)", ["learner_shared_k10"]),
        ("Teacher shared space (200 PCs)", ["teacher_shared_k200"]),
        ("Learner shared space (200 PCs)", ["learner_shared_k200"]),
    ]

    rows = []
    for label, predictors in predictor_specs:
        pred_df = loo_predictions(participants, predictors, outcome="AoA")
        row = {"label": label, "predictors": ",".join(predictors)}
        row.update(prediction_summary(pred_df))
        rows.append(row)

    summary = pd.DataFrame(rows)
    summary.to_csv(AOA_SUMMARY, index=False)

    print(summary[
        [
            "label",
            "pearson_r",
            "pearson_p",
            "spearman_rho",
            "spearman_p",
            "cv_r2",
        ]
    ].to_string(index=False))
    print(f"Saved {PCA_BY_SUBJECT}")
    print(f"Saved {AOA_SUMMARY}")


if __name__ == "__main__":
    main()
