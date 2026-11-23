# Step 8: is what does not transfer a person-specific signature? (Methods 2.8, Fig. 3B-C)
#
# Language-state signature, for each participant and each run:
#     mean pattern of English trials - mean pattern of Chinese trials   (800 values)
# Correlate every participant's run-1 signature with every participant's run-2
# signature -> 77 x 77 matrix:
#     diagonal       same person, other run       (within-participant similarity)
#     off-diagonal   different people             (between-participant similarity)
# Identification: give each run-1 signature to the person whose run-2 signature
# correlates most with it (and the other way round); accuracy = share of people
# matched to themselves (chance 1/77). Tested by shuffling identities 10,000 times.
#
# Done on the full pattern (k = 0) and after removing the first k shared dimensions
# (k = 1 ... 200), using the leave-pair-out PCA spaces saved by step 4:
#     off-diagonal (i, j): space fitted without i and j
#     diagonal (i, i):     averaged over the 76 spaces that exclude i
#
# Paper: identification 28% with the full pattern, 74% after removing 50 dimensions.
#
# Usage:  python scripts/08_signatures_identification.py   (after step 4)

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

K_LIST = [0, 1, 5, 10, 25, 50, 100, 200]
N_PERMUTATIONS = 10000
RESULTS = Path(__file__).resolve().parents[1] / "results"

data = np.load(RESULTS / "parcels_schaefer800.npz")
subjects = sorted(key[2:] for key in data.files if key.startswith("X_"))
n = len(subjects)
index = {s: i for i, s in enumerate(subjects)}

pairs = pd.read_csv(RESULTS / "shared_dimensions_pairs.csv")
components = np.load(RESULTS / "shared_dimensions_components.npy", mmap_mode="r")


# ---- 1. signatures: English - Chinese, per participant and run ---------------
# (z-scored per participant on all trials, as in step 4)
signature = {}   # (subject, 0 or 1) -> 800 values; 0 = first run, 1 = second run
for s in subjects:
    X = StandardScaler().fit_transform(data[f"X_{s}"])
    language = np.array([t[:2] for t in data[f"trial_type_{s}"]])
    runs = data[f"run_{s}"]
    for r, run in enumerate(np.unique(runs)):
        in_run = runs == run
        signature[s, r] = X[in_run & (language == "L2")].mean(axis=0) - X[in_run & (language == "L1")].mean(axis=0)


def residual(sig, C):
    # removing the shared dimensions from every trial and then averaging
    # is the same as removing them from the average (both are linear)
    return sig - (sig @ C.T) @ C


def corr(a, b):
    return np.corrcoef(a, b)[0, 1]


# ---- 2. the 77 x 77 run-1 x run-2 similarity matrix for every k --------------
matrices = {}
for k in K_LIST:
    R = np.zeros((n, n))
    if k == 0:
        for i, si in enumerate(subjects):
            for j, sj in enumerate(subjects):
                R[i, j] = corr(signature[si, 0], signature[sj, 1])
    else:
        diagonal_sum = np.zeros(n)
        for p, (a, b) in enumerate(zip(pairs["subject_a"], pairs["subject_b"])):
            C = np.asarray(components[p, :k])
            res = {(s, r): residual(signature[s, r], C) for s in (a, b) for r in (0, 1)}
            ia, ib = index[a], index[b]
            R[ia, ib] = corr(res[a, 0], res[b, 1])         # different people, this pair's space
            R[ib, ia] = corr(res[b, 0], res[a, 1])
            diagonal_sum[ia] += corr(res[a, 0], res[a, 1])   # same person, summed over spaces
            diagonal_sum[ib] += corr(res[b, 0], res[b, 1])
        R[np.diag_indices(n)] = diagonal_sum / (n - 1)       # average over the 76 spaces
    matrices[k] = R
    print(f"k = {k} done", flush=True)
np.savez(RESULTS / "signature_similarity_matrices.npz", subjects=subjects,
         **{f"k{k}": R for k, R in matrices.items()})


# ---- 3. summary: similarity, identification, permutation test ----------------
def identification(R):
    run1_to_run2 = np.mean(R.argmax(axis=1) == np.arange(len(R)))   # each run-1 row: best run-2 column
    run2_to_run1 = np.mean(R.argmax(axis=0) == np.arange(len(R)))   # each run-2 column: best run-1 row
    return (run1_to_run2 + run2_to_run1) / 2


rng = np.random.default_rng(42)
shuffles = [rng.permutation(n) for _ in range(N_PERMUTATIONS)]
off_diagonal = ~np.eye(n, dtype=bool)

# participant bootstrap: how many times each person is drawn, 5,000 times
counts = np.array([np.bincount(rng.integers(0, n, n), minlength=n) for _ in range(5000)])

rows = []
for k, R in matrices.items():
    observed = identification(R)
    null = np.array([identification(R[:, order]) for order in shuffles])   # shuffle run-2 identities

    # within: mean of the drawn people's diagonal; between: mean over the pairs they form
    within_boot = counts @ np.diag(R) / n
    between_boot = []
    for c in counts:
        w = np.outer(c, c) * off_diagonal
        if w.sum() > 0:
            between_boot.append((w * R).sum() / w.sum())

    rows.append({
        "k_removed": k,
        "within_similarity": np.diag(R).mean(),
        "within_ci_low": np.percentile(within_boot, 2.5),
        "within_ci_high": np.percentile(within_boot, 97.5),
        "between_similarity": R[off_diagonal].mean(),
        "between_ci_low": np.percentile(between_boot, 2.5),
        "between_ci_high": np.percentile(between_boot, 97.5),
        "identification": observed,
        "p_permutation": (np.sum(null >= observed) + 1) / (N_PERMUTATIONS + 1),
    })
summary = pd.DataFrame(rows)
summary.to_csv(RESULTS / "signatures_identification_summary.csv", index=False)

print(f"\nchance identification = {100 / n:.1f}%")
print(summary.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
