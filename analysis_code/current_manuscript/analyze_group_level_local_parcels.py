import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from nilearn import datasets
from scipy import stats
from statsmodels.stats.multitest import multipletests


ROOT = Path(__file__).resolve().parents[2]
MVPA = ROOT / "mvpa_schaefer800"
PAPER = ROOT / "paper_version"

LOCAL = MVPA / "schaefer800_mvpa_accuracy_long.csv"
LANG_DEF = MVPA / "domain_annotation" / "lipkin_language_schaefer800_parcel_scores.csv"
MD_DEF = MVPA / "latest_robustness_controls" / "lipkin_md_schaefer800_parcel_sets.csv"

OUT_ALL = PAPER / "group_level_local_parcel_stats.csv"
OUT_SIG = PAPER / "group_level_local_parcel_stats_fdr05.csv"


def parse_args():
    parser = argparse.ArgumentParser(description="Run group-level parcel tests with BH-FDR.")
    parser.add_argument("--local-csv", type=Path, default=LOCAL)
    parser.add_argument("--mvpa-dir", type=Path, default=MVPA)
    parser.add_argument("--out-all", type=Path, default=OUT_ALL)
    parser.add_argument("--out-sig", type=Path, default=OUT_SIG)
    return parser.parse_args()


def parse_network17(label: str) -> str:
    parts = str(label).split("_")
    for hemi in ("LH", "RH"):
        if hemi in parts:
            idx = parts.index(hemi)
            return parts[idx + 1] if idx + 1 < len(parts) else "Unknown"
    return "Unknown"


def build_membership() -> pd.DataFrame:
    atlas7 = datasets.fetch_atlas_schaefer_2018(n_rois=800, yeo_networks=7, resolution_mm=2)
    atlas17 = datasets.fetch_atlas_schaefer_2018(n_rois=800, yeo_networks=17, resolution_mm=2)

    labels7 = [lab.decode("utf-8") if isinstance(lab, bytes) else lab for lab in atlas7.labels[1:]]
    labels17 = [lab.decode("utf-8") if isinstance(lab, bytes) else lab for lab in atlas17.labels[1:]]

    mem = pd.DataFrame(
        {
            "parcel": np.arange(1, 801),
            "parcel_name7": labels7,
            "parcel_name17": labels17,
        }
    )

    mem["network7"] = mem["parcel_name7"].str.split("_").str[2]
    mem["network17"] = mem["parcel_name17"].map(parse_network17)

    lang = pd.read_csv(LANG_DEF)
    ranked = lang.sort_values("lipkin_language_mean", ascending=False).reset_index(drop=True)
    top10_lang = set(ranked.head(80)["parcel_index"].astype(int).tolist())
    mem["lipkin_language_top10"] = mem["parcel"].isin(top10_lang)

    md = pd.read_csv(MD_DEF).rename(columns={"parcel_index": "parcel"})
    if "parcel" not in md.columns:
        md = pd.read_csv(MD_DEF).rename(columns={"parcel": "parcel"})
    mem = mem.merge(md[["parcel", "lipkin_md_top10"]], on="parcel", how="left")
    mem["lipkin_md_top10"] = mem["lipkin_md_top10"].fillna(False)
    return mem


def cohen_dz(values: np.ndarray) -> float:
    mean = values.mean()
    sd = values.std(ddof=1)
    return np.nan if sd == 0 else mean / sd


def main() -> None:
    global MVPA, LANG_DEF, MD_DEF
    args = parse_args()
    MVPA = args.mvpa_dir
    LANG_DEF = MVPA / "domain_annotation" / "lipkin_language_schaefer800_parcel_scores.csv"
    MD_DEF = MVPA / "latest_robustness_controls" / "lipkin_md_schaefer800_parcel_sets.csv"

    args.out_all.parent.mkdir(parents=True, exist_ok=True)
    args.out_sig.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.local_csv)
    membership = build_membership()

    rows = []
    for parcel, pdf in df.groupby("parcel"):
        vals = pdf["accuracy"].to_numpy(dtype=float)
        vals_mc = vals - 0.5
        t_stat, p_t = stats.ttest_1samp(vals, 0.5)
        try:
            w_stat, p_w = stats.wilcoxon(vals_mc, alternative="greater", zero_method="wilcox")
        except ValueError:
            w_stat, p_w = np.nan, np.nan

        rows.append(
            {
                "parcel": int(parcel),
                "n_subjects": len(vals),
                "mean_accuracy": vals.mean(),
                "mean_accuracy_minus_chance": vals_mc.mean(),
                "sd_accuracy": vals.std(ddof=1),
                "t_vs_chance": t_stat,
                "p_t_vs_chance": p_t,
                "wilcoxon_stat": w_stat,
                "p_wilcoxon_greater": p_w,
                "cohen_dz": cohen_dz(vals_mc),
            }
        )

    out = pd.DataFrame(rows).merge(membership, on="parcel", how="left")
    reject, p_fdr, _, _ = multipletests(out["p_t_vs_chance"].to_numpy(), alpha=0.05, method="fdr_bh")
    out["p_fdr_bh"] = p_fdr
    out["sig_fdr_05"] = reject
    out = out.sort_values(["sig_fdr_05", "p_fdr_bh", "mean_accuracy"], ascending=[False, True, False]).reset_index(drop=True)

    sig = out[out["sig_fdr_05"]].copy()

    out.to_csv(args.out_all, index=False)
    sig.to_csv(args.out_sig, index=False)

    print(args.out_all)
    print(args.out_sig)
    print(f"FDR-significant parcels: {len(sig)}")
    if len(sig):
        cols = ["parcel", "parcel_name17", "network17", "mean_accuracy", "mean_accuracy_minus_chance", "t_vs_chance", "p_fdr_bh"]
        print(sig[cols].head(20).to_string(index=False))


if __name__ == "__main__":
    main()
