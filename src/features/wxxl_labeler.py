# =============================================================================
# WXXL-PAT | Label Generator
# =============================================================================
# Purpose:
#   For each confirmed pattern, determine if it succeeded or failed.
#
#   Success = price closes ABOVE the neckline within 30 bars after C2
#   Failure = price closes BELOW Cave 1 low before breaking the neckline
#           OR neckline not broken within 30 bars
#
#   This is the target variable for XGBoost training.
#   Labels use only post-C2 data — zero leakage into features.
# =============================================================================

import pandas as pd
import numpy as np
import os
import sys

sys.path.append(os.path.abspath("."))

# =============================================================================
# CONFIG
# =============================================================================
PATTERNS_PATH  = "data/patterns/confirmed_patterns.csv"
FEATURES_PATH  = "data/features/feature_matrix.csv"
PROCESSED_DIR  = "data/processed"
OUTPUT_PATH    = "data/features/labeled_features.csv"

MAX_BARS_TO_BREAKOUT = 30    # Bars after C2 to wait for neckline break
FAILURE_BUFFER       = 0.02  # 2% below C1 = confirmed failure


# =============================================================================
# LABEL SINGLE PATTERN
# =============================================================================

def label_pattern(
    df: pd.DataFrame,
    c1_idx: int,
    c2_idx: int,
    c1_price: float,
    neckline_price: float,
) -> dict:
    """
    Determine outcome of a confirmed double bottom pattern.

    Parameters
    ----------
    df             : pd.DataFrame   Full processed stock data
    c1_idx         : int            Bar index of Cave 1
    c2_idx         : int            Bar index of Cave 2
    c1_price       : float          Price of Cave 1
    neckline_price : float          Neckline resistance price

    Returns
    -------
    dict with keys:
        label           : int    1 = success, 0 = failure
        outcome         : str    'breakout', 'cave1_violated', 'timeout'
        bars_to_breakout: int    Bars from C2 to neckline break (-1 if failed)
        max_gain        : float  Maximum gain from C2 in the window
        max_loss        : float  Maximum loss from C2 in the window
    """
    result = {
        "label":            0,
        "outcome":          "timeout",
        "bars_to_breakout": -1,
        "max_gain":         0.0,
        "max_loss":         0.0,
    }

    # Look at data AFTER C2 only
    future_start = c2_idx + 1
    future_end   = min(len(df), c2_idx + MAX_BARS_TO_BREAKOUT + 1)

    if future_start >= len(df):
        result["outcome"] = "no_future_data"
        return result

    future = df.iloc[future_start:future_end]

    if len(future) == 0:
        result["outcome"] = "no_future_data"
        return result

    c2_price   = df["Close"].iloc[c2_idx]
    fail_level = c1_price * (1 - FAILURE_BUFFER)

    max_gain = 0.0
    max_loss = 0.0

    for bar_offset, (idx, bar) in enumerate(future.iterrows()):
        close = bar["Close"]
        gain  = (close - c2_price) / c2_price
        loss  = (c2_price - close) / c2_price

        max_gain = max(max_gain, gain)
        max_loss = max(max_loss, loss)

        # Success: close above neckline
        if close >= neckline_price:
            result["label"]            = 1
            result["outcome"]          = "breakout"
            result["bars_to_breakout"] = bar_offset + 1
            result["max_gain"]         = round(max_gain, 4)
            result["max_loss"]         = round(max_loss, 4)
            return result

        # Failure: close below Cave 1 low
        if close <= fail_level:
            result["label"]   = 0
            result["outcome"] = "cave1_violated"
            result["max_gain"] = round(max_gain, 4)
            result["max_loss"] = round(max_loss, 4)
            return result

    # Timeout — neckline not broken within MAX_BARS_TO_BREAKOUT
    result["outcome"]  = "timeout"
    result["max_gain"] = round(max_gain, 4)
    result["max_loss"] = round(max_loss, 4)
    return result


# =============================================================================
# MAIN
# =============================================================================

def run_labeler() -> None:
    patterns = pd.read_csv(PATTERNS_PATH, parse_dates=["c1_date"])
    features = pd.read_csv(FEATURES_PATH)

    print(f"Labeling {len(patterns)} patterns...")

    labels = []
    failed = 0

    for i, row in patterns.iterrows():
        ticker         = row["ticker"]
        c1_idx         = int(row["c1_idx"])
        c2_idx         = int(row["c2_idx"])
        c1_price       = float(row["c1_price"])
        neckline_price = float(row["neckline_price"])

        path = os.path.join(PROCESSED_DIR, f"{ticker}.csv")
        if not os.path.exists(path):
            failed += 1
            continue

        try:
            df = pd.read_csv(path, index_col=0, parse_dates=True)
            df.columns = [c.strip().title() for c in df.columns]

            label_result = label_pattern(
                df             = df,
                c1_idx         = c1_idx,
                c2_idx         = c2_idx,
                c1_price       = c1_price,
                neckline_price = neckline_price,
            )

            labels.append({
                "ticker":            ticker,
                "c1_idx":            c1_idx,
                "c2_idx":            c2_idx,
                "label":             label_result["label"],
                "outcome":           label_result["outcome"],
                "bars_to_breakout":  label_result["bars_to_breakout"],
                "max_gain":          label_result["max_gain"],
                "max_loss":          label_result["max_loss"],
            })

        except Exception as e:
            failed += 1

    df_labels = pd.DataFrame(labels)

    # Merge labels with features
    df_features = pd.read_csv(FEATURES_PATH)
    df_merged   = pd.merge(
        df_features,
        df_labels[["ticker", "c1_idx", "c2_idx", "label", "outcome",
                   "bars_to_breakout", "max_gain", "max_loss"]],
        on=["ticker", "c1_idx", "c2_idx"],
        how="inner"
    )

    df_merged.to_csv(OUTPUT_PATH, index=False)

    # Summary
    print(f"\n{'='*50}")
    print(f"Labeling complete.")
    print(f"Total labeled      : {len(df_labels)}")
    print(f"Failed             : {failed}")
    print(f"Merged with features: {len(df_merged)}")
    print(f"\nOutcome distribution:")
    print(df_labels["outcome"].value_counts())
    print(f"\nLabel distribution:")
    print(df_labels["label"].value_counts())
    success_rate = df_labels["label"].mean()
    print(f"\nOverall success rate: {success_rate:.1%}")
    print(f"Saved to: {OUTPUT_PATH}")
    print(f"{'='*50}")


# =============================================================================
if __name__ == "__main__":
    run_labeler()