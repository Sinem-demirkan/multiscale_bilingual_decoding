# Step 9: predicting age of English acquisition (AoA) (Methods 2.9, Fig. 2D, SI Tables S2, S3).
#
# 43 candidate predictors per participant, all computed by earlier scripts:
#   from step 4 (shared dimensions), for k = 1, 5, 10, 25, 50, 100, 200:
#     learner transfer   mean accuracy of the 76 classifiers tested on this person
#     teacher transfer   mean accuracy of this person's classifier on the 76 others
#     within-person      leave-one-run-out accuracy, averaged over the 76 PCA spaces
#     ... each on the shared dimensions and on the residual   -> 3 x 2 x 7 = 42
#   from step 2: within-person decoding on the full cortical pattern   -> 43rd
#
# For each predictor separately: leave-one-participant-out ridge regression
# (alpha = 1; predictor z-scored on the training participants), giving a
# cross-validated R^2 and the Pearson r between observed and predicted AoA.
#
# Max-statistic permutation test: shuffle AoA across participants 10,000 times,
# redo the whole procedure for all 43 predictors, keep the largest R^2.
# Corrected p = share of permutations whose largest R^2 >= the observed R^2
# (counting the observed data as one permutation: (count + 1) / (10,000 + 1)).
# The same is done with chronological age as the outcome.
#
# Paper: best = learner transfer, 10 shared dimensions, r = 0.44, R^2 = 0.19, p < 0.001.
#
# Usage:  python scripts/09_aoa_prediction.py   (after steps 2 and 4; a few minutes)

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

PARTICIPANTS_FILE = Path("/home/sdemirka/fmri/duo-cogcon/participants.tsv")   # from ds005455
OUTCOMES = {"AoA": "AoA", "age": "age"}   # name -> column in participants.tsv
K_LIST = [1, 5, 10, 25, 50, 100, 200]
N_PERMUTATIONS = 10000
ALPHA = 1.0

RESULTS = Path(__file__).resolve().parents[1] / "results"


# ---- 1. one table: participants x 43 predictors -----------------------------
transfer = pd.read_csv(RESULTS / "shared_dimensions_transfer_by_pair.csv")
within = pd.read_csv(RESULTS / "shared_dimensions_within_by_space.csv")
whole = pd.read_csv(RESULTS / "whole_cortex_decoding.csv").set_index("subject")

table = pd.DataFrame(index=sorted(whole.index))
for k in K_LIST:
    for rep in ("shared", "residual"):
        col = f"{rep}_{k}"
        table[f"learner_{col}"] = transfer.groupby("learner")[col].mean()
        table[f"teacher_{col}"] = transfer.groupby("teacher")[col].mean()
        table[f"within_{col}"] = within.groupby("subject")[col].mean()
table["within_full"] = whole["balanced_accuracy"]
predictors = list(table.columns)
assert len(predictors) == 43, len(predictors)

participants = pd.read_csv(PARTICIPANTS_FILE, sep="\t").set_index("participant_id")
table = table.join(participants[list(OUTCOMES.values())])
table.to_csv(RESULTS / "aoa_predictor_table.csv", index_label="subject")
print(len(table), "participants,", len(predictors), "predictors")


# ---- 2. leave-one-participant-out ridge, one predictor at a time -------------
def loo_predict(x, y):
    pred = np.empty(len(y))
    for i in range(len(y)):
        train = np.arange(len(y)) != i
        scaler = StandardScaler()
        x_train = scaler.fit_transform(x[train, None])
        model = Ridge(alpha=ALPHA).fit(x_train, y[train])
        pred[i] = model.predict(scaler.transform(x[[i], None]))[0]
    return pred


# The same leave-one-out ridge, written out with numpy so that 10,000 permutations
# x 43 predictors run in seconds instead of days. For one standardized predictor,
# ridge has a closed form: slope = sum(z * y) / (n_train + alpha), intercept = mean(y).
# Y can hold many outcome vectors at once (one per row).
def loo_predict_fast(x, Y):
    n = len(x)
    others = ~np.eye(n, dtype=bool)                               # row i: everyone except i
    mean = (x.sum() - x) / (n - 1)                                # training mean for each held-out i
    std = np.sqrt(((x ** 2).sum() - x ** 2) / (n - 1) - mean ** 2)
    Z = (x[None, :] - mean[:, None]) / std[:, None] * others     # Z[i, j]: x_j standardized without i
    z_held_out = (x - mean) / std
    slope = Y @ Z.T / (n - 1 + ALPHA)                            # (n_outcomes, n)
    intercept = (Y.sum(axis=1, keepdims=True) - Y) / (n - 1)
    return intercept + slope * z_held_out


def r2_rows(Y, pred):
    return 1 - ((Y - pred) ** 2).sum(axis=1) / ((Y - Y.mean(axis=1, keepdims=True)) ** 2).sum(axis=1)


for outcome, column in OUTCOMES.items():
    data = table.dropna(subset=[column])
    y = data[column].to_numpy(float)

    # observed: R^2 and r for every predictor
    rows = []
    for p in predictors:
        x = data[p].to_numpy(float)
        pred = loo_predict(x, y)
        assert np.allclose(pred, loo_predict_fast(x, y[None, :])[0])   # fast version = sklearn
        rows.append({"predictor": p, "cv_r2": r2_score(y, pred),
                     "pearson_r": stats.pearsonr(y, pred)[0]})
    summary = pd.DataFrame(rows)

    # max-statistic permutation test
    rng = np.random.default_rng(42)
    Y_perm = np.array([rng.permutation(y) for _ in range(N_PERMUTATIONS)])
    max_r2 = np.full(N_PERMUTATIONS, -np.inf)
    for p in predictors:
        x = data[p].to_numpy(float)
        max_r2 = np.maximum(max_r2, r2_rows(Y_perm, loo_predict_fast(x, Y_perm)))
    summary["p_corrected"] = [(np.sum(max_r2 >= r2) + 1) / (N_PERMUTATIONS + 1) for r2 in summary["cv_r2"]]

    summary = summary.sort_values("cv_r2", ascending=False)
    summary.to_csv(RESULTS / f"aoa_prediction_{outcome}.csv", index=False)

    print(f"\noutcome: {outcome} (n = {len(y)}), best 5 of {len(predictors)} predictors")
    print(summary.head(5).to_string(index=False, float_format=lambda v: f"{v:.3f}"))
