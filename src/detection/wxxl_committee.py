# =============================================================================
# WXXL-PAT | Committee Coordinator
# =============================================================================
# Purpose:
#   Runs all three detection layers on a Cave 1 candidate and requires
#   ALL THREE to vote YES before a pattern is confirmed.
#
#   Layer 2a — PIP Extractor   : finds the 7-point skeleton
#   Layer 2b — DTW Matcher     : validates shape similarity
#   Layer 2c — Shapelet        : validates geometric rules
#
#   Single disagreement = REJECT.
#   This eliminates different categories of false positives that
#   neither method catches alone.
# =============================================================================

import pandas as pd
import numpy as np
import os
import sys

sys.path.append(os.path.abspath("."))

from src.detection.wxxl_pip_extractor import extract_double_bottom_pips
from src.detection.wxxl_dtw_matcher   import dtw_match
from src.detection.wxxl_shapelet      import shapelet_vote


# =============================================================================
# COMMITTEE VOTE
# =============================================================================

def committee_vote(
    prices: pd.Series,
    c1_idx: int,
    window_before: int = 60,
    window_after:  int = 80,
    dtw_threshold: float = 0.80,
) -> dict:
    """
    Run all three committee members on a Cave 1 candidate.
    All three must vote YES for the pattern to be confirmed.

    Parameters
    ----------
    prices        : pd.Series   Full price history for one stock
    c1_idx        : int         Bar index of Cave 1 candidate
    window_before : int         Bars before C1 to include
    window_after  : int         Bars after C1 to look for Cave 2
    dtw_threshold : float       Maximum DTW distance to vote YES

    Returns
    -------
    dict with keys:
        passed          : bool    — True if ALL three voted YES
        pip_valid       : bool    — PIP extractor vote
        dtw_vote        : bool    — DTW matcher vote
        shapelet_vote   : bool    — Shapelet classifier vote
        dtw_distance    : float   — Raw DTW distance
        shapelet_conf   : float   — Shapelet confidence
        neckline_price  : float   — Detected neckline price
        c2_price        : float   — Detected Cave 2 price
        c2_idx          : int     — Absolute bar index of Cave 2
        pip_prices      : list    — 7 PIP prices
        reason          : str     — Why it failed (empty if passed)
    """
    result = {
        "passed":         False,
        "pip_valid":      False,
        "dtw_vote":       False,
        "shapelet_vote":  False,
        "dtw_distance":   999.0,
        "shapelet_conf":  0.0,
        "neckline_price": 0.0,
        "c2_price":       0.0,
        "c2_idx":         -1,
        "pip_prices":     [],
        "reason":         ""
    }

    # ── Layer 2a: PIP Extractor ──────────────────────────────────────────────
    pip_result = extract_double_bottom_pips(
        prices        = prices,
        c1_idx        = c1_idx,
        window_before = window_before,
        window_after  = window_after,
    )

    result["pip_valid"]      = pip_result["valid"]
    result["pip_prices"]     = pip_result["pip_prices"]
    result["neckline_price"] = pip_result["neckline_price"]
    result["c2_price"]       = pip_result["c2_price"]

    if not pip_result["valid"]:
        result["reason"] = f"PIP failed: {pip_result['reason']}"
        return result

    # Compute absolute C2 index
    if pip_result["c2_pip"] >= 0 and pip_result["window_start"] >= 0:
        c2_relative        = pip_result["pips"][pip_result["c2_pip"]]
        result["c2_idx"]   = pip_result["window_start"] + c2_relative

    pip_prices = pip_result["pip_prices"]

    # ── Layer 2b: DTW Matcher ────────────────────────────────────────────────
    dtw_result = dtw_match(pip_prices, threshold=dtw_threshold)

    result["dtw_vote"]     = dtw_result["vote"]
    result["dtw_distance"] = dtw_result["dtw_distance"]

    if not dtw_result["vote"]:
        result["reason"] = f"DTW failed: {dtw_result['reason']}"
        return result

    # ── Layer 2c: Shapelet ───────────────────────────────────────────────────
    shapelet_result = shapelet_vote(pip_prices)

    result["shapelet_vote"] = shapelet_result["vote"]
    result["shapelet_conf"] = shapelet_result["confidence"]

    if not shapelet_result["vote"]:
        result["reason"] = f"Shapelet failed: {shapelet_result['reason']}"
        return result

    # ── All three voted YES ──────────────────────────────────────────────────
    result["passed"] = True
    return result


# =============================================================================
# RUN COMMITTEE ON ALL CANDIDATES
# =============================================================================

def run_committee(
    candidates_path: str = "data/candidates/candidates.csv",
    processed_dir:   str = "data/processed",
    output_path:     str = "data/patterns/confirmed_patterns.csv",
    max_per_ticker:  int = 20,
    min_bar_distance: int = 30,
) -> None:
    """
    Run the committee on all Cave 1 candidates.
    Saves confirmed double bottom patterns to data/patterns/
    """
    os.makedirs("data/patterns", exist_ok=True)

    candidates = pd.read_csv(candidates_path, parse_dates=["c1_date"])
    print(f"Running committee on {len(candidates)} candidates...")

    confirmed  = []
    rejected   = 0
    ticker_counts = {}

    for i, row in candidates.iterrows():
        ticker = row["ticker"]
        c1_idx = int(row["c1_idx"])

        # Limit patterns per ticker to avoid domination
        if ticker_counts.get(ticker, 0) >= max_per_ticker:
            continue

        # Skip if too close to a previously confirmed pattern
        ticker_confirmed = [p for p in confirmed if p["ticker"] == ticker]
        too_close = any(abs(c1_idx - p["c1_idx"]) < min_bar_distance for p in ticker_confirmed)
        if too_close:
            continue

        # Skip if too close to a previously confirmed pattern
        ticker_confirmed = [p for p in confirmed if p["ticker"] == ticker]
        too_close = any(abs(c1_idx - p["c1_idx"]) < min_bar_distance for p in ticker_confirmed)
        if too_close:
            continue

        # Load processed data
        path = os.path.join(processed_dir, f"{ticker}.csv")
        if not os.path.exists(path):
            continue

        try:
            df     = pd.read_csv(path, index_col=0, parse_dates=True)
            df.columns = [c.strip().title() for c in df.columns]
            prices = df["Close"]

            result = committee_vote(prices, c1_idx)

            if result["passed"]:
                confirmed.append({
                    "ticker":         ticker,
                    "c1_date":        row["c1_date"],
                    "c1_idx":         c1_idx,
                    "c1_price":       row["c1_price"],
                    "c2_idx":         result["c2_idx"],
                    "c2_price":       result["c2_price"],
                    "neckline_price": result["neckline_price"],
                    "prior_decline":  row["prior_decline"],
                    "dtw_distance":   result["dtw_distance"],
                    "shapelet_conf":  result["shapelet_conf"],
                    "rs_rank":        row.get("rs_rank", None),
                    "atr":            row.get("atr", None),
                })
                ticker_counts[ticker] = ticker_counts.get(ticker, 0) + 1
            else:
                rejected += 1

        except Exception as e:
            rejected += 1

        if (i + 1) % 1000 == 0:
            print(f"  {i + 1}/{len(candidates)} | confirmed: {len(confirmed)} | rejected: {rejected}")

    df_out = pd.DataFrame(confirmed)
    df_out.to_csv(output_path, index=False)

    print(f"\n{'='*50}")
    print(f"Committee complete.")
    print(f"Confirmed patterns : {len(confirmed)}")
    print(f"Rejected           : {rejected}")
    print(f"Pass rate          : {len(confirmed)/(len(confirmed)+rejected):.1%}")
    print(f"Saved to           : {output_path}")
    print(f"{'='*50}")

    if len(confirmed) > 0:
        print(f"\nTop tickers by confirmed patterns:")
        print(df_out["ticker"].value_counts().head(10))


# =============================================================================
if __name__ == "__main__":
    run_committee()