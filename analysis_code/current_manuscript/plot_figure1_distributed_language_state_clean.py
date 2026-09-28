from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch


CANDIDATE_ROOTS = [
    Path("/home/sdemirka/fmri/recent"),
    Path("/Users/sdemirka/Desktop/agent"),
    Path("/Users/sdemirka/Desktop/etat civil/agent"),
]

ROOT = next((path for path in CANDIDATE_ROOTS if path.exists()), CANDIDATE_ROOTS[0])
PAPER = ROOT / "paper_version"

SCHEMA_PATH = ROOT / "schema.png"
RAW_800 = (
    ROOT
    / "mvpa_schaefer800"
    / "distributed_parcel_mean"
    / "distributed_parcel_mean_decoding_by_subject.csv"
)
LANG_SUBSETS = PAPER / "l1l2_switch_nonswitch_whole_cortex" / "by_subject.csv"
LANG_SWITCH = (
    ROOT
    / "mvpa_schaefer800"
    / "language_control_switch_whole_cortex"
    / "language_switch_decoding_by_subject.csv"
)

YEO_PCA_SUMMARY = (
    PAPER / "network_pca_grouped_pfi" / "network_pca_grouped_pfi_summary.csv"
)
YEO_PCA_BY_SUBJECT = (
    PAPER / "network_pca_grouped_pfi" / "network_pca_grouped_pfi_by_subject.csv"
)
LIPKIN_LANG_PCA_SUMMARY = (
    PAPER
    / "targeted_lipkin_pca_grouped_pfi"
    / "LipkinLanguage"
    / "LipkinLanguage_summary.csv"
)
LIPKIN_LANG_PCA_BY_SUBJECT = (
    PAPER
    / "targeted_lipkin_pca_grouped_pfi"
    / "LipkinLanguage"
    / "LipkinLanguage_by_subject.csv"
)
LIPKIN_MD_PCA_SUMMARY = (
    PAPER / "targeted_lipkin_pca_grouped_pfi" / "LipkinMD" / "LipkinMD_summary.csv"
)
LIPKIN_MD_PCA_BY_SUBJECT = (
    PAPER
    / "targeted_lipkin_pca_grouped_pfi"
    / "LipkinMD"
    / "LipkinMD_by_subject.csv"
)

OUT_PDF = PAPER / "figure1_distributed_language_state_clean.pdf"
OUT_PNG = PAPER / "figure1_distributed_language_state_clean.png"

N_BOOT = 5000
BOOT_SEED = 42

STYLE = {
    "font_family": "sans-serif",
    "font_sans_serif": [
        "Myriad Pro",
        "Arial",
        "Helvetica Neue",
        "Helvetica",
        "DejaVu Sans",
    ],
    "panel_label_size": 9,
    "ax_label_size": 8,
    "tick_size": 7.5,
    "legend_size": 7.5,
    "small_tick_size": 7,
}

plt.rcParams.update(
    {
        "font.family": STYLE["font_family"],
        "font.sans-serif": STYLE["font_sans_serif"],
        "font.size": 8,
        "axes.labelsize": STYLE["ax_label_size"],
        "xtick.labelsize": STYLE["tick_size"],
        "ytick.labelsize": STYLE["tick_size"],
        "legend.fontsize": STYLE["legend_size"],
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)

MULTIVOXEL_BLUE = "#2171B5"
PARCEL_MEAN_BLUE = "#BDD7E7"
DEMEANED_MULTIVOXEL_BLUE = "#6BAED6"
DARK_GRAY = "#4d4d4d"
MID_GRAY = "#7a7a7a"
LIGHT_GRAY = "#bdbdbd"
RESTING_STATE_COLOR = "#1B5E20"
FUNCTIONAL_LOCALIZER_COLOR = "#B8860B"

NETWORK_ORDER = [
    "SomMot",
    "Default",
    "Cont",
    "SalVentAttn",
    "DorsAttn",
    "Vis",
    "Limbic",
]

NETWORK_LABELS = {
    "SomMot": "Somatomotor",
    "Default": "Default",
    "Cont": "Control",
    "SalVentAttn": "Salience/\nventral attention",
    "DorsAttn": "Dorsal attention",
    "Vis": "Visual",
    "Limbic": "Limbic",
}

REFERENCE_ORDER = ["LipkinLanguage", "LipkinMD"]

REFERENCE_LABELS = {
    "LipkinLanguage": "Lipkin\nlanguage",
    "LipkinMD": "Multiple\ndemand",
}


def clean_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.9)
    ax.spines["bottom"].set_linewidth(0.9)
    ax.grid(False)
    ax.minorticks_off()
    ax.tick_params(axis="both", width=0.8, length=3)


def add_figure_panel_label(fig, label, x, y):
    fig.text(
        x,
        y,
        label,
        ha="left",
        va="bottom",
        fontsize=STYLE["panel_label_size"],
        fontweight="bold",
    )


def mean_ci95(values, seed=BOOT_SEED):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 2:
        return np.nan, np.nan, np.nan

    mean = values.mean()
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(N_BOOT, len(values)))
    boot_means = values[indices].mean(axis=1)
    low, high = np.percentile(boot_means, [2.5, 97.5])
    return mean, low, high


def center_around_chance(values):
    return (np.asarray(values, dtype=float) - 0.50) * 100.0


def crop_white_margins(image, threshold=0.95, pad_top=16, pad_other=6):
    img = np.asarray(image)
    white_level = 255.0 if img.dtype == np.uint8 else 1.0
    thresh = threshold * white_level

    if img.ndim == 2:
        content_mask = img < thresh
    else:
        rgb = img[..., :3].astype(float)
        content_mask = np.any(rgb < thresh, axis=-1)
        if img.shape[-1] == 4:
            content_mask &= img[..., 3].astype(float) > 0

    rows = np.any(content_mask, axis=1)
    cols = np.any(content_mask, axis=0)
    if not rows.any() or not cols.any():
        return img

    r = np.where(rows)[0]
    c = np.where(cols)[0]
    r0 = max(0, r[0] - pad_top)
    r1 = min(img.shape[0], r[-1] + 1 + pad_other)
    c0 = max(0, c[0] - pad_other)
    c1 = min(img.shape[1], c[-1] + 1 + pad_other)
    return img[r0:r1, c0:c1]


def add_points_mean_ci(
    ax,
    values,
    x,
    color,
    jitter_width=0.10,
    point_size=14,
    mean_size=42,
    seed=42,
):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    rng = np.random.default_rng(seed)
    jitter = rng.uniform(-jitter_width, jitter_width, size=len(values))

    ax.scatter(
        np.full(len(values), x) + jitter,
        values,
        s=point_size,
        color=color,
        alpha=0.78,
        edgecolors="white",
        linewidths=0.25,
        zorder=3,
    )

    mean, low, high = mean_ci95(values, seed=seed + 1000)

    ax.errorbar(
        x,
        mean,
        yerr=[[mean - low], [high - mean]],
        fmt="o",
        markersize=np.sqrt(mean_size),
        markerfacecolor="black",
        markeredgecolor="black",
        ecolor="black",
        elinewidth=1.0,
        capsize=3,
        capthick=1.0,
        zorder=5,
    )
    return mean, low, high


def add_accuracy_ylabel(ax):
    ax.set_ylabel("")
    ax.text(
        -0.105,
        0.52,
        "Accuracy (pp)",
        transform=ax.transAxes,
        rotation=90,
        ha="center",
        va="center",
        fontsize=STYLE["ax_label_size"],
    )
    ax.text(
        -0.067,
        0.52,
        "centered around 50%",
        transform=ax.transAxes,
        rotation=90,
        ha="center",
        va="center",
        fontsize=STYLE["small_tick_size"],
    )


def load_decoding_data():
    return (
        pd.read_csv(RAW_800)[["subject", "balanced_accuracy"]]
        .rename(columns={"balanced_accuracy": "raw_800"})
        .sort_values("raw_800")
        .reset_index(drop=True)
    )


def load_yeo_network_summary():
    summary = pd.read_csv(YEO_PCA_SUMMARY).copy()
    summary["network_key"] = summary["network"]
    summary["label"] = summary["network_key"].map(NETWORK_LABELS)
    summary["source"] = "Resting state"
    return summary


def load_lipkin_network_summary():
    language = pd.read_csv(LIPKIN_LANG_PCA_SUMMARY).copy()
    md = pd.read_csv(LIPKIN_MD_PCA_SUMMARY).copy()

    language["network_key"] = "LipkinLanguage"
    language["label"] = "Lipkin\nlanguage"
    language["source"] = "Functional localizer"

    md["network_key"] = "LipkinMD"
    md["label"] = "Multiple\ndemand"
    md["source"] = "Functional localizer"

    return pd.concat([language, md], ignore_index=True)


if not SCHEMA_PATH.exists():
    raise FileNotFoundError(f"Schematic image not found: {SCHEMA_PATH}")

schema_image = crop_white_margins(
    plt.imread(SCHEMA_PATH),
    threshold=0.95,
    pad_top=16,
    pad_other=6,
)

df = load_decoding_data()
lang_subsets = pd.read_csv(LANG_SUBSETS)
lang_switch = pd.read_csv(LANG_SWITCH)
yeo_summary = load_yeo_network_summary()
lipkin_summary = load_lipkin_network_summary()
yeo_by_subject = pd.read_csv(YEO_PCA_BY_SUBJECT)
lipkin_language_by_subject = pd.read_csv(LIPKIN_LANG_PCA_BY_SUBJECT)
lipkin_md_by_subject = pd.read_csv(LIPKIN_MD_PCA_BY_SUBJECT)

panel_c_values = {
    "All trials": df["raw_800"].to_numpy(),
    "Repeat trials": lang_subsets.loc[
        lang_subsets["subset"] == "nonswitch_only", "balanced_accuracy"
    ].to_numpy(),
    "Switch trials": lang_subsets.loc[
        lang_subsets["subset"] == "switch_only", "balanced_accuracy"
    ].to_numpy(),
    "Switch status": lang_switch["language_switch_accuracy"].to_numpy(),
}

panel_c_colors = {
    "All trials": MULTIVOXEL_BLUE,
    "Repeat trials": PARCEL_MEAN_BLUE,
    "Switch trials": DEMEANED_MULTIVOXEL_BLUE,
    "Switch status": LIGHT_GRAY,
}

panel_c_positions = {
    "All trials": 0.0,
    "Repeat trials": 1.0,
    "Switch trials": 2.0,
    "Switch status": 3.6,
}

network_rows = []

for network in NETWORK_ORDER:
    sub = yeo_summary.loc[yeo_summary["network_key"] == network].copy()
    if len(sub) == 0:
        raise ValueError(f"Network {network} was not found in {YEO_PCA_SUMMARY}")
    if len(sub) > 1:
        raise ValueError(f"Network {network} appears more than once in {YEO_PCA_SUMMARY}")

    values = yeo_by_subject.loc[
        yeo_by_subject["network"].eq(network), "importance_pp"
    ].to_numpy()
    mean, low, high = mean_ci95(
        values,
        seed=BOOT_SEED + 500 + NETWORK_ORDER.index(network),
    )
    network_rows.append(
        {
            "network_key": network,
            "label": NETWORK_LABELS[network],
            "source": "Resting state",
            "mean_importance_pp": mean,
            "ci_low_pp": low,
            "ci_high_pp": high,
        }
    )

for network in REFERENCE_ORDER:
    sub = lipkin_summary.loc[lipkin_summary["network_key"] == network].copy()
    if len(sub) == 0:
        raise ValueError(f"{network} was not found in the functional localizer summaries.")
    if len(sub) > 1:
        raise ValueError(
            f"{network} appears more than once in the functional localizer summary."
        )

    if network == "LipkinLanguage":
        values = lipkin_language_by_subject["importance_pp"].to_numpy()
        seed = BOOT_SEED + 600
    else:
        values = lipkin_md_by_subject["importance_pp"].to_numpy()
        seed = BOOT_SEED + 601

    mean, low, high = mean_ci95(values, seed=seed)
    network_rows.append(
        {
            "network_key": network,
            "label": REFERENCE_LABELS[network],
            "source": "Functional localizer",
            "mean_importance_pp": mean,
            "ci_low_pp": low,
            "ci_high_pp": high,
        }
    )

network_plot_df = pd.DataFrame(network_rows)

DISPLAY_ORDER = NETWORK_ORDER + ["__gap__", "LipkinLanguage", "LipkinMD"]
y_positions = {}
y_value = 0.0
NORMAL_GAP = 1.18
SALVENT_GAP = 1.45
BLOCK_GAP = 1.25
FUNCTIONAL_GAP = 1.90

for item in DISPLAY_ORDER:
    if item == "__gap__":
        y_value += BLOCK_GAP
        continue
    y_positions[item] = y_value
    if item == "SalVentAttn":
        y_value += SALVENT_GAP
    elif item == "LipkinLanguage":
        y_value += FUNCTIONAL_GAP
    else:
        y_value += NORMAL_GAP

network_plot_df["y"] = network_plot_df["network_key"].map(y_positions)

FIG_WIDTH = 7.25
FIG_HEIGHT = 7.8
fig = plt.figure(figsize=(FIG_WIDTH, FIG_HEIGHT), facecolor="white")

A_LEFT, A_BOTTOM, A_WIDTH, A_HEIGHT = 0.09, 0.745, 0.89, 0.205
B_LEFT, B_BOTTOM, B_WIDTH, B_HEIGHT = 0.10, 0.405, 0.27, 0.270
C_LEFT, C_BOTTOM, C_WIDTH, C_HEIGHT = 0.48, 0.405, 0.49, 0.270
D_LEFT, D_BOTTOM, D_WIDTH, D_HEIGHT = 0.25, 0.060, 0.60, 0.233

axA = fig.add_axes([A_LEFT, A_BOTTOM, A_WIDTH, A_HEIGHT])
axB = fig.add_axes([B_LEFT, B_BOTTOM, B_WIDTH, B_HEIGHT])
axC = fig.add_axes([C_LEFT, C_BOTTOM, C_WIDTH, C_HEIGHT])
axD = fig.add_axes([D_LEFT, D_BOTTOM, D_WIDTH, D_HEIGHT])

axA.imshow(schema_image, aspect="equal", interpolation="lanczos")
axA.set_anchor("C")
axA.set_xticks([])
axA.set_yticks([])
for spine in axA.spines.values():
    spine.set_visible(False)

add_points_mean_ci(
    axB,
    df["raw_800"].to_numpy() * 100.0,
    x=0.0,
    color=MULTIVOXEL_BLUE,
    jitter_width=0.13,
    point_size=14,
    mean_size=42,
    seed=42,
)
axB.axhline(50, color=MID_GRAY, linestyle=":", linewidth=1.0, zorder=1)
axB.set_xlim(-0.42, 0.42)
axB.set_ylim(40, 80)
axB.set_xticks([])
axB.set_yticks(np.arange(40, 81, 10))
axB.set_ylabel("Accuracy (%)")
axB.text(
    0.97,
    0.96,
    rf"$n = {len(df)}$",
    transform=axB.transAxes,
    ha="right",
    va="top",
    fontsize=STYLE["small_tick_size"],
)
clean_axes(axB)

for idx, (label, values) in enumerate(panel_c_values.items()):
    add_points_mean_ci(
        axC,
        center_around_chance(values),
        x=panel_c_positions[label],
        color=panel_c_colors[label],
        jitter_width=0.10,
        point_size=14,
        mean_size=42,
        seed=42 + idx,
    )

axC.axhline(0, color=MID_GRAY, linestyle=":", linewidth=1.0, zorder=1)
axC.axvline(2.8, color="#d0d0d0", linewidth=0.8, zorder=1)
axC.text(
    1.0,
    1.08,
    "Language state decoding",
    transform=axC.get_xaxis_transform(),
    ha="center",
    va="top",
    fontsize=STYLE["small_tick_size"],
    color=DARK_GRAY,
    clip_on=False,
)
axC.text(
    3.6,
    1.08,
    "Alternative target",
    transform=axC.get_xaxis_transform(),
    ha="center",
    va="top",
    fontsize=STYLE["small_tick_size"],
    color=DARK_GRAY,
    clip_on=False,
)
axC.set_xlim(-0.55, 4.15)
axC.set_ylim(-15, 30)
axC.set_yticks([-10, 0, 10, 20])
axC.set_xticks([panel_c_positions[label] for label in panel_c_values])
axC.set_xticklabels(list(panel_c_values.keys()), fontsize=STYLE["tick_size"])
for tick in axC.get_xticklabels():
    tick.set_linespacing(1.05)
add_accuracy_ylabel(axC)
clean_axes(axC)

axD.axvline(0, color=MID_GRAY, linestyle=":", linewidth=0.8, zorder=1)
axD.barh(
    network_plot_df["y"],
    network_plot_df["mean_importance_pp"],
    height=0.60,
    color=MULTIVOXEL_BLUE,
    edgecolor="none",
    zorder=2,
)
for row in network_plot_df.itertuples(index=False):
    mean = float(row.mean_importance_pp)
    lo = float(row.ci_low_pp)
    hi = float(row.ci_high_pp)
    axD.errorbar(
        mean,
        row.y,
        xerr=[[mean - lo], [hi - mean]],
        fmt="none",
        ecolor="black",
        elinewidth=0.9,
        capsize=2.5,
        capthick=0.9,
        zorder=4,
    )

display_items = NETWORK_ORDER + REFERENCE_ORDER
display_labels = [NETWORK_LABELS[item] for item in NETWORK_ORDER] + [
    REFERENCE_LABELS[item] for item in REFERENCE_ORDER
]
axD.set_yticks([y_positions[item] for item in display_items])
axD.set_yticklabels(display_labels, fontsize=STYLE["tick_size"])
for tick_label in axD.get_yticklabels():
    tick_label.set_linespacing(1.05)
for tick_label, item in zip(axD.get_yticklabels(), display_items):
    if item in NETWORK_ORDER:
        tick_label.set_color(RESTING_STATE_COLOR)
    elif item in REFERENCE_ORDER:
        tick_label.set_color(FUNCTIONAL_LOCALIZER_COLOR)

axD.set_xlabel("Network feature importance\n(percentage point accuracy drop)")
axD.set_ylabel("")
all_ci_low = network_plot_df["ci_low_pp"].to_numpy(dtype=float)
all_ci_high = network_plot_df["ci_high_pp"].to_numpy(dtype=float)
x_min = min(-0.25, np.floor((np.nanmin(all_ci_low) - 0.20) * 2) / 2)
x_max = np.ceil((np.nanmax(all_ci_high) + 0.40) * 2) / 2
axD.set_xlim(x_min, x_max)
axD.set_ylim(max(y_positions.values()) + 0.70, -0.70)
clean_axes(axD)

legend_handles = [
    Patch(facecolor=RESTING_STATE_COLOR, edgecolor="none", label="Resting state"),
    Patch(
        facecolor=FUNCTIONAL_LOCALIZER_COLOR,
        edgecolor="none",
        label="Functional localizer",
    ),
]
legend = axD.legend(
    handles=legend_handles,
    title="Networks",
    loc="center left",
    bbox_to_anchor=(0.93, 0.50),
    frameon=False,
    fontsize=STYLE["legend_size"],
    title_fontsize=STYLE["legend_size"],
    handlelength=1.0,
    handleheight=0.8,
    borderaxespad=0,
    labelspacing=0.8,
    alignment="left",
)
legend.get_title().set_color("black")
legend.get_title().set_fontweight("normal")
legend.get_title().set_ha("left")
try:
    legend._legend_box.align = "left"
except AttributeError:
    pass

LEFT_PANEL_X = 0.09
add_figure_panel_label(fig, "A", x=LEFT_PANEL_X, y=A_BOTTOM + A_HEIGHT + 0.010)
add_figure_panel_label(fig, "B", x=LEFT_PANEL_X, y=B_BOTTOM + B_HEIGHT + 0.012)
add_figure_panel_label(fig, "C", x=C_LEFT, y=C_BOTTOM + C_HEIGHT + 0.012)
add_figure_panel_label(fig, "D", x=LEFT_PANEL_X, y=D_BOTTOM + D_HEIGHT + 0.010)

fig.savefig(OUT_PDF, bbox_inches="tight", pad_inches=0.04, facecolor="white")
fig.savefig(
    OUT_PNG,
    dpi=1200,
    bbox_inches="tight",
    pad_inches=0.04,
    facecolor="white",
)
plt.show()

print("\nPanel B - whole cortex multivoxel decoding")
print(f"N participants: {len(df)}")
panel_b_mean, panel_b_low, panel_b_high = mean_ci95(
    df["raw_800"].to_numpy() * 100.0,
    seed=42 + 1000,
)
print(f"Mean [95% CI]: {panel_b_mean:.2f} [{panel_b_low:.2f}, {panel_b_high:.2f}]")

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
) * 100.0
mean, low, high = mean_ci95(repeat_minus_switch.to_numpy(), seed=BOOT_SEED + 700)
print("\nPanel C - repeat minus switch decoding")
print(f"Mean difference [95% CI]: {mean:.2f} [{low:.2f}, {high:.2f}] pp")

yeo_wide = yeo_by_subject.pivot(
    index="subject",
    columns="network",
    values="importance_pp",
).dropna()
other_networks = [network for network in NETWORK_ORDER if network != "Vis"]
visual_minus_other = yeo_wide["Vis"] - yeo_wide[other_networks].mean(axis=1)
mean, low, high = mean_ci95(visual_minus_other.to_numpy(), seed=BOOT_SEED + 701)
print("\nPanel D - visual minus mean other Yeo networks")
print(f"Mean difference [95% CI]: {mean:.2f} [{low:.2f}, {high:.2f}] pp")

mean, low, high = mean_ci95((-visual_minus_other).to_numpy(), seed=BOOT_SEED + 701)
print("\nPanel D - mean other Yeo networks minus visual")
print(f"Mean difference [95% CI]: {mean:.2f} [{low:.2f}, {high:.2f}] pp")

lipkin_difference = (
    lipkin_language_by_subject.set_index("subject")["importance_pp"]
    - lipkin_md_by_subject.set_index("subject")["importance_pp"]
).dropna()
mean, low, high = mean_ci95(lipkin_difference.to_numpy(), seed=BOOT_SEED + 702)
print("\nPanel D - Lipkin language minus multiple demand")
print(f"Mean difference [95% CI]: {mean:.2f} [{low:.2f}, {high:.2f}] pp")

print("\nPanel C colors")
print(f"All trials: {MULTIVOXEL_BLUE}")
print(f"Repeat trials: {PARCEL_MEAN_BLUE}")
print(f"Switch trials: {DEMEANED_MULTIVOXEL_BLUE}")
print(f"Switch status: {LIGHT_GRAY}")

print("\nPanel D - network feature importance")
print(
    network_plot_df[
        [
            "network_key",
            "source",
            "mean_importance_pp",
            "ci_low_pp",
            "ci_high_pp",
        ]
    ].to_string(index=False)
)

print(f"\nSaved PDF: {OUT_PDF}")
print(f"Saved PNG: {OUT_PNG}")
