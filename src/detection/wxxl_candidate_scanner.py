# =============================================================================
# WXXL-PAT | Candidate Scanner
# =============================================================================
# Purpose:
#   Scans all 497 processed stocks and applies the Downtrend Gate to find
#   valid Cave 1 candidates. Saves results to data/candidates/candidates.csv
#
# A valid Cave 1 candidate must:
#   - Have a prior decline of at least 12% within 60 bars
#   - Be within 3% of the 40-bar rolling low
# =============================================================================

import pandas as pd
import numpy as np
import os
import sys

# Allow imports from src/
sys.path.append(os.path.abspath("."))
from src.detection.wxxl_downtrend_gate import check_downtrend_gate

# =============================================================================
# CONFIG
# =============================================================================
PROCESSED_DIR   = "data/processed"
CANDIDATES_DIR  = "data/candidates"
MIN_BARS        = 100       # Minimum bars required before scanning starts
LOOKBACK        = 60        # Bars to look back for prior peak
MIN_DECLINE     = 0.12      # 12% minimum prior decline
LOW_WINDOW      = 40        # Bars to check C1 near rolling low
LOW_PROXIMITY   = 0.03      # C1 must be within 3% of rolling low
STEP            = 5         # Check every 5 bars (not every bar) for speed


# =============================================================================
# SCAN SINGLE TICKER
# =============================================================================

def scan_ticker(ticker: str, df: pd.DataFrame) -> list:
    """
    Slide through a stock's price history and find all valid C1 candidates.
    Returns a list of dicts, one per valid candidate bar.
    """
    prices    = df["Close"].reset_index(drop=True)
    dates     = df.index
    candidates = []

    for i in range(LOOKBACK + 10, len(prices) - 5, STEP):
        result = check_downtrend_gate(
            prices       = prices,
            c1_idx       = i,
            lookback     = LOOKBACK,
            min_decline_pct  = MIN_DECLINE,
            low_window   = LOW_WINDOW,
            low_proximity_pct = LOW_PROXIMITY,
        )

        if result["passed"]:
            candidates.append({
                "ticker"        : ticker,
                "c1_date"       : dates[i],
                "c1_idx"        : i,
                "c1_price"      : round(prices.iloc[i], 4),
                "prior_decline" : result["prior_decline"],
                "low_rank"      : result["low_rank"],
                "atr"           : round(df["ATR"].iloc[i], 4) if "ATR" in df.columns else None,
                "rs_rank"       : round(df["RS_rank"].iloc[i], 2) if "RS_rank" in df.columns else None,
            })

    return candidates


# =============================================================================
# MAIN
# =============================================================================

def run_scanner() -> None:
    os.makedirs(CANDIDATES_DIR, exist_ok=True)

    tickers = [
        f.replace(".csv", "")
        for f in os.listdir(PROCESSED_DIR)
        if f.endswith(".csv")
    ]
    print(f"Scanning {len(tickers)} stocks for Cave 1 candidates...")

    all_candidates = []
    failed         = 0

    for i, ticker in enumerate(tickers):
        path = os.path.join(PROCESSED_DIR, f"{ticker}.csv")
        try:
            df = pd.read_csv(path, index_col=0, parse_dates=True)
            df.columns = [c.strip().title() for c in df.columns]

            if len(df) < MIN_BARS:
                continue

            candidates = scan_ticker(ticker, df)
            all_candidates.extend(candidates)

        except Exception as e:
            print(f"  Error on {ticker}: {e}")
            failed += 1

        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(tickers)} scanned | candidates so far: {len(all_candidates)}")

    # Save
    df_out = pd.DataFrame(all_candidates)
    out_path = os.path.join(CANDIDATES_DIR, "candidates.csv")
    df_out.to_csv(out_path, index=False)

    print(f"\n{'='*50}")
    print(f"Scan complete.")
    print(f"Total candidates : {len(all_candidates)}")
    print(f"Stocks scanned   : {len(tickers) - failed}")
    print(f"Failed           : {failed}")
    print(f"Saved to         : {out_path}")
    print(f"{'='*50}")

    # Quick distribution summary
    if len(all_candidates) > 0:
        df_out = pd.DataFrame(all_candidates)
        print(f"\nPrior decline distribution:")
        print(df_out["prior_decline"].describe().round(4))
        print(f"\nCandidates per ticker (top 10):")
        print(df_out["ticker"].value_counts().head(10))


# =============================================================================
if __name__ == "__main__":
    run_scanner()