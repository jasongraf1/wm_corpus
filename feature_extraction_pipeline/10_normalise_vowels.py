"""
10_normalise_vowels.py
----------------------
Applies three vowel normalisation methods to formant data and produces
a summary CSV with mean F1/F2 per vowel category per speaker:

  - Lobanov (z-score)
  - Nearey (log-mean)
  - Watt & Fabricius (S-centroid), per Watt & Fabricius (2002), as
    described at https://lingmethodshub.github.io/content/R/tidy-norm/

Prepping for normalisation (per the same tutorial's "Prepping for
normalization" section) happens once, up front, and the resulting
"clean" subset feeds ALL THREE methods -- not just Watt & Fabricius:

  1. Tokens measured during overlapping speech (overlap == True) are
     excluded outright. A formant measurement taken while another
     speaker is also talking is not a reliable measurement of either
     speaker's vowel, so it shouldn't inform anyone's statistics.
  2. Within each (speaker, phone) group, each token's Mahalanobis
     distance from that group's F1/F2 centroid is computed (using the
     midpoint/50% timepoint as the token's representative measurement,
     matching the vowel_summary.csv convention). Tokens with
     sqrt(Mahalanobis^2) > MAHAL_THRESHOLD are excluded as likely
     formant-tracking glitches (e.g. F1/F2 swaps, octave errors).

Excluded tokens are NOT deleted from the output -- they're kept with
an excluded_from_norm flag (and the reason), so you can still see them
for spot-checking, but they don't contribute to any speaker's or vowel
class's normalisation statistics. Normalised values are still computed
for excluded rows using the clean statistics, purely for diagnostic
plotting; filter on excluded_from_norm == False for real analysis.

NOTE: this brings Lobanov and Nearey in line with the same outlier
exclusion Watt & Fabricius needs (their method is far more sensitive to
a single wild measurement, since it uses raw min/max F1 rather than a
mean or z-score) -- previously this script did no outlier filtering at
all. If you want the old (unfiltered) Lobanov/Nearey behaviour for
comparison, that's just this script with MAHAL_THRESHOLD set very high
and EXCLUDE_OVERLAP set to False.

Dependencies:
    pip install pandas numpy

Usage:
    python 10_normalise_vowels.py --input output_analysis/vowel_formants.csv
                                  --output output_analysis
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------

# sqrt(Mahalanobis^2) cutoff for outlier exclusion, per the Watt & Fabricius
# "prepping for normalization" step. 2.0 matches the LingMethodsHub tutorial.
MAHAL_THRESHOLD = 2.0

# Minimum tokens in a (speaker, phone) group needed to estimate a stable 2x2
# covariance matrix. Below this, the group's tokens are kept unfiltered
# (mahal_dist = NaN) rather than risk a wild estimate from too few points.
MIN_GROUP_FOR_MAHAL = 5

# Exclude tokens measured during overlapping speech from all normalisation.
EXCLUDE_OVERLAP = True

# Timepoint (of the 5 per token) treated as each token's single
# representative measurement, for Mahalanobis filtering and the
# Watt & Fabricius triangle/scaling calculation. Matches the existing
# vowel_summary.csv convention.
REPRESENTATIVE_TIMEPOINT = 50


# ---------------------------------------------------------------------------
# PREPPING: overlap exclusion + Mahalanobis outlier flagging
# ---------------------------------------------------------------------------

def mahalanobis_sqrt_distances(f1: np.ndarray, f2: np.ndarray) -> np.ndarray:
    """
    sqrt(Mahalanobis^2) of each (f1, f2) point from the group's own mean,
    using the group's own 2x2 covariance matrix. Mirrors
    joeyr::tidy_mahalanobis() + sqrt() from the LingMethodsHub tutorial.
    """
    pts = np.column_stack([f1, f2])
    mean = pts.mean(axis=0)
    cov = np.cov(pts, rowvar=False)
    try:
        inv_cov = np.linalg.inv(cov)
    except np.linalg.LinAlgError:
        return np.full(len(pts), np.nan)  # singular covariance -- can't compute
    diff = pts - mean
    d_sq = np.einsum("ij,jk,ik->i", diff, inv_cov, diff)
    d_sq = np.clip(d_sq, 0, None)  # guard tiny negative values from float error
    return np.sqrt(d_sq)


def flag_exclusions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds columns: excluded_overlap, mahal_dist, excluded_outlier,
    excluded_from_norm. Does not drop any rows.
    """
    df = df.copy()
    df["overlap"] = df["overlap"].astype(bool)
    df["excluded_overlap"] = df["overlap"] if EXCLUDE_OVERLAP else False

    mid = df[df["timepoint"] == REPRESENTATIVE_TIMEPOINT].copy()
    # Only compute Mahalanobis distance from tokens not already excluded
    # for overlap -- an overlap-tainted token shouldn't help define what
    # "normal" looks like for its vowel class.
    mid_clean = mid[~mid["excluded_overlap"]]

    mahal_by_token = {}
    for (speaker, phone), group in mid_clean.groupby(["speaker", "phone"]):
        if len(group) < MIN_GROUP_FOR_MAHAL:
            for tok in group["token_id"]:
                mahal_by_token[tok] = np.nan
            continue
        dists = mahalanobis_sqrt_distances(group["F1"].values, group["F2"].values)
        for tok, d in zip(group["token_id"], dists):
            mahal_by_token[tok] = d

    df["mahal_dist"] = df["token_id"].map(mahal_by_token)
    df["excluded_outlier"] = df["mahal_dist"] > MAHAL_THRESHOLD
    df["excluded_from_norm"] = df["excluded_overlap"] | df["excluded_outlier"].fillna(False)

    mid = df[df["timepoint"] == REPRESENTATIVE_TIMEPOINT]  # re-derive: now has the new columns
    n_tokens_total = mid["token_id"].nunique()
    n_overlap = int(mid["excluded_overlap"].sum())
    n_outlier = int((mid["mahal_dist"] > MAHAL_THRESHOLD).sum())
    n_too_small = int(mid.loc[~mid["excluded_overlap"], "mahal_dist"].isna().sum())
    print(f"  Prepping: {n_tokens_total} tokens total")
    print(f"    excluded (overlap):         {n_overlap}")
    print(f"    excluded (Mahalanobis > {MAHAL_THRESHOLD}): {n_outlier}")
    print(f"    kept unfiltered (group < {MIN_GROUP_FOR_MAHAL} tokens): {n_too_small}")
    n_clean = n_tokens_total - int(mid["excluded_from_norm"].sum())
    print(f"    remaining clean tokens:     {n_clean}")

    return df


# ---------------------------------------------------------------------------
# NORMALISATION METHODS
# ---------------------------------------------------------------------------

def lobanov_normalise(df: pd.DataFrame) -> pd.DataFrame:
    """
    Lobanov (1971) normalisation: per-speaker z-score of F1, F2, F3.
    Statistics computed from clean (non-excluded) tokens only; the
    resulting z-score is still applied to every row.
    """
    df = df.copy()
    for speaker, group in df.groupby("speaker"):
        clean = group[~group["excluded_from_norm"]]
        for formant in ["F1", "F2", "F3"]:
            vals = clean[formant].dropna()
            if len(vals) < 2:
                df.loc[group.index, f"{formant}_lob"] = np.nan
                continue
            mean = vals.mean()
            sd = vals.std(ddof=1)
            df.loc[group.index, f"{formant}_lob"] = (df.loc[group.index, formant] - mean) / sd
    return df


def nearey_normalise(df: pd.DataFrame) -> pd.DataFrame:
    """
    Nearey (1978) individual log-mean normalisation.
    F_norm = exp(log(F) - mean_speaker(log(F1, F2, F3)))
    The grand log-mean is computed from clean (non-excluded) tokens only.
    """
    df = df.copy()
    for speaker, group in df.groupby("speaker"):
        clean = group[~group["excluded_from_norm"]]
        log_vals = []
        for formant in ["F1", "F2", "F3"]:
            log_vals.extend(np.log(clean[formant].dropna().values))
        if not log_vals:
            continue
        grand_log_mean = np.mean(log_vals)
        for formant in ["F1", "F2", "F3"]:
            df.loc[group.index, f"{formant}_nearey"] = np.exp(
                np.log(df.loc[group.index, formant]) - grand_log_mean
            )
    return df


def watt_fabricius_normalise(df: pd.DataFrame) -> pd.DataFrame:
    """
    Watt & Fabricius (2002) S-centroid normalisation.

    Per speaker, using clean (non-excluded) midpoint tokens only:
      1. Per-phone mean F1/F2 ("vowel class means").
      2. Three corner points of the vowel-space triangle:
           beet   : F1 = min class-mean F1 (whichever phone)
                    F2 = max class-mean F2 (whichever phone -- may be
                         a DIFFERENT phone than the one giving F1)
           bat    : F1 = max class-mean F1
                    F2 = the F2 of THAT SAME phone (not independently
                         maximised)
           school : F1 = beet's F1, F2 = beet's F1 (predicted point,
                    not an observed back vowel -- avoids relying on a
                    possibly-fronted GOOSE/FOOT as the back anchor)
      3. Scaling factors S1 = mean(beet_f1, bat_f1, school_f1),
                         S2 = mean(beet_f2, bat_f2, school_f2).
    Then F1_wf = F1 / S1, F2_wf = F2 / S2, applied to every row (all
    timepoints, clean or excluded) using that speaker's S1/S2.

    F3 has no equivalent in this method (it's specifically an F1/F2
    vowel-space technique) and is left untouched.
    """
    df = df.copy()
    mid_clean = df[(df["timepoint"] == REPRESENTATIVE_TIMEPOINT)
                  & (~df["excluded_from_norm"])]

    scalers = {}
    for speaker, sgroup in mid_clean.groupby("speaker"):
        class_means = sgroup.groupby("phone")[["F1", "F2"]].mean()
        if len(class_means) < 3:
            print(f"    WARNING: {speaker} has only {len(class_means)} phone "
                 f"categor(y/ies) with clean tokens -- Watt & Fabricius "
                 f"triangle will be degenerate for this speaker.")
        if class_means.empty:
            continue

        beet_f1 = class_means["F1"].min()
        beet_f2 = class_means["F2"].max()
        bat_phone = class_means["F1"].idxmax()
        bat_f1 = class_means.loc[bat_phone, "F1"]
        bat_f2 = class_means.loc[bat_phone, "F2"]
        school_f1 = beet_f1
        school_f2 = beet_f1

        s1 = np.mean([beet_f1, bat_f1, school_f1])
        s2 = np.mean([beet_f2, bat_f2, school_f2])
        scalers[speaker] = (s1, s2)

    df["_S1"] = df["speaker"].map(lambda s: scalers[s][0] if s in scalers else np.nan)
    df["_S2"] = df["speaker"].map(lambda s: scalers[s][1] if s in scalers else np.nan)
    df["F1_wf"] = df["F1"] / df["_S1"]
    df["F2_wf"] = df["F2"] / df["_S2"]
    df.drop(columns=["_S1", "_S2"], inplace=True)
    return df


# ---------------------------------------------------------------------------
# SUMMARY
# ---------------------------------------------------------------------------

def summarise_vowels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Mean F1/F2 (raw and normalised, all three methods) per vowel phone
    per speaker, at the representative (midpoint) timepoint only, using
    clean (non-excluded) tokens, with token count.
    """
    mid = df[(df["timepoint"] == REPRESENTATIVE_TIMEPOINT)
            & (~df["excluded_from_norm"])].copy()
    summary = (
        mid.groupby(["speaker", "phone"])
        .agg(
            n            = ("F1", "count"),
            F1_mean      = ("F1", "mean"),
            F2_mean      = ("F2", "mean"),
            F1_lob_mean  = ("F1_lob", "mean"),
            F2_lob_mean  = ("F2_lob", "mean"),
            F1_wf_mean   = ("F1_wf", "mean"),
            F2_wf_mean   = ("F2_wf", "mean"),
            F1_sd        = ("F1", "std"),
            F2_sd        = ("F2", "std"),
        )
        .round(2)
        .reset_index()
    )
    return summary


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Normalise vowel formants (Lobanov + Nearey + Watt & Fabricius)")
    parser.add_argument("--input", required=True, help="Path to vowel_formants.csv")
    parser.add_argument("--output", default="output_analysis", help="Output directory")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Reading {input_path}...")
    df = pd.read_csv(input_path)
    print(f"  {len(df)} rows, {df['speaker'].nunique()} speaker(s)")

    # Drop rows with missing formants
    n_before = len(df)
    df = df[~(df["F1"].isna() & df["F2"].isna())]
    print(f"  Dropped {n_before - len(df)} rows with missing F1+F2")

    print("\nPrepping for normalisation (overlap exclusion + Mahalanobis outliers)...")
    df = flag_exclusions(df)

    print("\nApplying Lobanov normalisation...")
    df = lobanov_normalise(df)
    print("Applying Nearey normalisation...")
    df = nearey_normalise(df)
    print("Applying Watt & Fabricius normalisation...")
    df = watt_fabricius_normalise(df)

    # Save full normalised data (excluded rows kept, flagged)
    norm_path = output_dir / "vowel_formants_normalised.csv"
    df.to_csv(norm_path, index=False, encoding="utf-8-sig")
    print(f"\n  Saved normalised formants -> {norm_path}")

    # Save summary (midpoint only, clean tokens only)
    summary = summarise_vowels(df)
    summary_path = output_dir / "vowel_summary.csv"
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    print(f"  Saved vowel summary -> {summary_path}")

    # Print quick vowel space overview
    # print("\n--- Vowel space summary (midpoint F1/F2, Lobanov normalised, clean tokens) ---")
    # print(summary[["speaker", "phone", "n", "F1_lob_mean", "F2_lob_mean"]].to_string(index=False))


if __name__ == "__main__":
    main()
