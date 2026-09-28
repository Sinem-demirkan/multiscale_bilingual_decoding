from pathlib import Path

import numpy as np
import pandas as pd


CANDIDATE_ROOTS = [
    Path("/home/sdemirka/fmri/recent"),
    Path("/Users/sdemirka/Desktop/agent"),
    Path("/Users/sdemirka/Desktop/etat civil/agent"),
]

K_LIST = [1, 5, 10, 25, 50, 100, 200]
N_BOOT = 5000
BOOT_SEED = 42


def resolve_root():
    needed = [
        "paper_version/cross_subject_language_transfer/shared_space_transfer_leave_pair_out_by_pair_with_original.csv",
        "paper_version/cross_subject_language_transfer/pair_space_self_decoding_leave_pair_out_by_subject.csv",
        "paper_version/l1l2_switch_nonswitch_whole_cortex/by_subject.csv",
        "paper_version/network_pca_grouped_pfi/network_pca_grouped_pfi_by_subject.csv",
        "paper_version/targeted_lipkin_pca_grouped_pfi/LipkinLanguage/LipkinLanguage_by_subject.csv",
        "paper_version/targeted_lipkin_pca_grouped_pfi/LipkinMD/LipkinMD_by_subject.csv",
    ]
    for root in CANDIDATE_ROOTS:
        if all((root / name).exists() for name in needed):
            return root
    raise FileNotFoundError("Could not find required manuscript result files.")


def bootstrap_mean_ci(values, seed):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    mean = float(values.mean())
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(N_BOOT, len(values)))
    boot_means = values[indices].mean(axis=1)
    ci_low, ci_high = np.percentile(boot_means, [2.5, 97.5])
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


def fmt_pp(mean, low, high):
    return f"{mean:.1f} pp [{low:.1f}, {high:.1f}]"


def add_row(rows, section, contrast, k, mean, low, high, scale=100.0):
    rows.append(
        {
            "section": section,
            "contrast": contrast,
            "k": "" if k is None else int(k),
            "mean": mean * scale,
            "ci_low": low * scale,
            "ci_high": high * scale,
            "formatted": fmt_pp(mean * scale, low * scale, high * scale),
            "ci_excludes_zero": bool((low > 0) or (high < 0)),
        }
    )


def main():
    root = resolve_root()
    paper = root / "paper_version"
    transfer_dir = paper / "cross_subject_language_transfer"

    transfer = pd.read_csv(
        transfer_dir / "shared_space_transfer_leave_pair_out_by_pair_with_original.csv"
    )
    full_pattern_candidates = [
        transfer_dir
        / "full_cortical_pattern_same_preprocessing"
        / "full_cortical_pattern_same_preprocessing_transfer_by_pair.csv",
        transfer_dir
        / "matched_full_pattern_transfer_reference"
        / "matched_full_pattern_transfer_by_pair.csv",
    ]
    full_pattern_path = next(
        (path for path in full_pattern_candidates if path.exists()),
        None,
    )
    if full_pattern_path is None:
        raise FileNotFoundError(
            "Could not find the full cortical pattern same-preprocessing transfer CSV."
        )
    full_pattern = pd.read_csv(full_pattern_path).rename(
        columns={"matched_full_accuracy": "full_cortical_pattern_accuracy"}
    )
    transfer = (
        transfer.drop(columns=["original_accuracy"], errors="ignore")
        .merge(
            full_pattern,
            on=["train_subject", "test_subject"],
            how="left",
            validate="many_to_one",
        )
    )

    self_df = pd.read_csv(
        transfer_dir / "pair_space_self_decoding_leave_pair_out_by_subject.csv"
    )
    lang_subsets = pd.read_csv(
        paper / "l1l2_switch_nonswitch_whole_cortex" / "by_subject.csv"
    )
    yeo = pd.read_csv(
        paper / "network_pca_grouped_pfi" / "network_pca_grouped_pfi_by_subject.csv"
    )
    lipkin_language = pd.read_csv(
        paper
        / "targeted_lipkin_pca_grouped_pfi"
        / "LipkinLanguage"
        / "LipkinLanguage_by_subject.csv"
    )
    lipkin_md = pd.read_csv(
        paper
        / "targeted_lipkin_pca_grouped_pfi"
        / "LipkinMD"
        / "LipkinMD_by_subject.csv"
    )

    rows = []

    for k in K_LIST:
        tmp = transfer.loc[transfer["n_components"].astype(int).eq(k)].copy()

        tmp["shared_minus_full_cortical_pattern"] = (
            tmp["shared_space_accuracy"] - tmp["full_cortical_pattern_accuracy"]
        )
        mean, low, high = bootstrap_pair_cluster_mean_ci(
            tmp,
            "shared_minus_full_cortical_pattern",
            seed=BOOT_SEED + 700 + K_LIST.index(k),
        )
        add_row(
            rows,
            "transfer",
            "shared - full cortical pattern, same preprocessing",
            k,
            mean,
            low,
            high,
        )

        tmp["residual_above_chance"] = tmp["residual_space_accuracy"] - 0.5
        mean, low, high = bootstrap_pair_cluster_mean_ci(
            tmp,
            "residual_above_chance",
            seed=BOOT_SEED + 800 + K_LIST.index(k),
        )
        add_row(rows, "transfer", "residual - chance", k, mean, low, high)

    for k in K_LIST:
        tmp = self_df.loc[self_df["n_components"].astype(int).eq(k)].copy()
        tmp["residual_minus_original"] = (
            tmp["residual_space_accuracy"] - tmp["original_accuracy"]
        )
        mean, low, high = bootstrap_mean_ci(
            tmp["residual_minus_original"].to_numpy(),
            seed=BOOT_SEED + 900 + K_LIST.index(k),
        )
        add_row(
            rows,
            "within-person",
            "residual - full cortical pattern",
            k,
            mean,
            low,
            high,
        )

    lang_wide = (
        lang_subsets.loc[
            lang_subsets["subset"].isin(["nonswitch_only", "switch_only"]),
            ["subject", "subset", "balanced_accuracy"],
        ]
        .pivot(index="subject", columns="subset", values="balanced_accuracy")
        .dropna()
    )
    repeat_minus_switch = (
        lang_wide["nonswitch_only"] - lang_wide["switch_only"]
    ).to_numpy()
    mean, low, high = bootstrap_mean_ci(
        repeat_minus_switch,
        seed=BOOT_SEED + 1000,
    )
    add_row(rows, "figure1", "repeat - switch decoding", None, mean, low, high)

    yeo_wide = (
        yeo.pivot(index="subject", columns="network", values="importance_pp")
        .dropna()
    )
    other_networks = [network for network in yeo_wide.columns if network != "Vis"]
    visual_minus_other = (
        yeo_wide["Vis"] - yeo_wide[other_networks].mean(axis=1)
    ).to_numpy()
    mean, low, high = bootstrap_mean_ci(
        visual_minus_other,
        seed=BOOT_SEED + 1001,
    )
    add_row(
        rows,
        "figure1",
        "visual - mean other Yeo networks",
        None,
        mean,
        low,
        high,
        scale=1.0,
    )
    add_row(
        rows,
        "figure1",
        "mean other Yeo networks - visual",
        None,
        -mean,
        -high,
        -low,
        scale=1.0,
    )

    lipkin_difference = (
        lipkin_language.set_index("subject")["importance_pp"]
        - lipkin_md.set_index("subject")["importance_pp"]
    ).dropna()
    mean, low, high = bootstrap_mean_ci(
        lipkin_difference.to_numpy(),
        seed=BOOT_SEED + 1002,
    )
    add_row(
        rows,
        "figure1",
        "Lipkin language - multiple demand",
        None,
        mean,
        low,
        high,
        scale=1.0,
    )

    out = pd.DataFrame(rows)
    out_path = paper / "results_bootstrap_contrasts.csv"
    out.to_csv(out_path, index=False)

    print(out[["section", "contrast", "k", "formatted", "ci_excludes_zero"]].to_string(index=False))
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
