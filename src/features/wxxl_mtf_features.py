# =============================================================================
# WXXL-PAT | MTF Feature Augmentor
# =============================================================================
# Purpose:
#   For each pattern in the labeled feature matrix, run MTF detection
#   and confluence scoring, then add the 9 MTF features to the matrix.
#
#   This is a one-time batch job. Results saved to:
#   data/features/labeled_features_mtf.csv
# =============================================================================

import pandas as pd
import numpy as np
import os
import sys
import warnings
warnings.filterwarnings("ignore")

sys.path.append(os.path.abspath("."))

from src.detection.wxxl_mtf_detector import detect_all_timeframes
from src.detection.wxxl_confluence   import find_best_confluence, confluence_to_features

# =============================================================================
# CONFIG
# =============================================================================
LABELED_PATH  = "data/features/labeled_features.csv"
PROCESSED_DIR = "data/processed"
OUTPUT_PATH   = "data/features/labeled_features_mtf.csv"

# Anchor windows — how far from daily C2 to search other timeframes
ANCHOR_WEEKLY_DAYS = 90
ANCHOR_H4_DAYS     = 30


# =============================================================================
# EXTRACT MTF FEATURES FOR ONE PATTERN
# =============================================================================

def get_mtf_features(
    ticker:    str,
    daily:     pd.DataFrame,
    c2_date:   pd.Timestamp,
) -> dict:
    """
    Run MTF detection anchored to a specific daily C2 date.
    Returns 9 MTF features for XGBoost.
    """
    try:
        # Slice daily data up to c2_date + 1 day — no leakage
        daily_slice = daily[daily.index <= c2_date + pd.Timedelta(days=1)]

        if len(daily_slice) < 100:
            return _empty_mtf_features()

        results = detect_all_timeframes(
            ticker                     = ticker,
            daily                      = daily_slice,
            anchor_window_weekly_days  = ANCHOR_WEEKLY_DAYS,
            anchor_window_h4_days      = ANCHOR_H4_DAYS,
        )

        confluence = find_best_confluence(
            weekly_patterns = results["weekly"],
            daily_patterns  = results["daily"],
            h4_patterns     = results["h4"],
        )

        return confluence_to_features(confluence)

    except Exception as e:
        return _empty_mtf_features()


def _empty_mtf_features() -> dict:
    return {
        "mtf_score":          0.0,
        "mtf_tfs_confirmed":  1.0,
        "mtf_aligned":        0.0,
        "mtf_weekly_present": 0.0,
        "mtf_h4_present":     0.0,
        "mtf_c2_wd_diff":     1.0,
        "mtf_nl_wd_diff":     1.0,
        "mtf_c2_dh_diff":     1.0,
        "mtf_nl_dh_diff":     1.0,
    }


# =============================================================================
# MAIN
# =============================================================================

def run_mtf_augmentation() -> None:
    print("=" * 60)
    print("WXXL-PAT | MTF Feature Augmentation")
    print("=" * 60)

    df = pd.read_csv(LABELED_PATH, parse_dates=["c1_date"])
    print(f"Patterns to process: {len(df)}")

    # Cache daily data per ticker — load once, reuse
    ticker_cache = {}
    mtf_rows     = []
    failed       = 0

    tickers_unique = df["ticker"].unique()
    print(f"Unique tickers: {len(tickers_unique)}")

    for i, row in df.iterrows():
        ticker  = row["ticker"]
        c2_idx  = int(row["c2_idx"])

        # Load daily data
        if ticker not in ticker_cache:
            path = os.path.join(PROCESSED_DIR, f"{ticker}.csv")
            if not os.path.exists(path):
                ticker_cache[ticker] = None
            else:
                try:
                    d = pd.read_csv(path, index_col=0, parse_dates=True)
                    d.columns = [c.strip().title() for c in d.columns]
                    ticker_cache[ticker] = d
                except Exception:
                    ticker_cache[ticker] = None

        daily = ticker_cache[ticker]
        if daily is None:
            mtf_rows.append(_empty_mtf_features())
            failed += 1
            continue

        # Get C2 date from index
        if c2_idx >= len(daily):
            mtf_rows.append(_empty_mtf_features())
            failed += 1
            continue

        c2_date = daily.index[c2_idx]

        # Extract MTF features
        feats = get_mtf_features(ticker, daily, c2_date)
        mtf_rows.append(feats)

        if (i + 1) % 100 == 0:
            pct = (i + 1) / len(df) * 100
            print(f"  {i + 1}/{len(df)} ({pct:.0f}%) — failed: {failed}")

    # Merge MTF features into labeled matrix
    df_mtf    = pd.DataFrame(mtf_rows)
    df_merged = pd.concat([df.reset_index(drop=True), df_mtf], axis=1)
    df_merged.to_csv(OUTPUT_PATH, index=False)

    print(f"\n{'='*60}")
    print(f"Augmentation complete.")
    print(f"Rows processed : {len(df)}")
    print(f"Failed         : {failed}")
    print(f"MTF features   : {len(df_mtf.columns)}")
    print(f"Total features : {len(df_merged.columns)}")
    print(f"Saved to       : {OUTPUT_PATH}")
    print(f"\nMTF feature summary:")
    print(df_mtf.describe().round(4).to_string())
    print(f"{'='*60}")


# =============================================================================
if __name__ == "__main__":
    run_mtf_augmentation()