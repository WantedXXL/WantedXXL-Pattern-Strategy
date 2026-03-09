# =============================================================================
# WXXL-PAT | Multi-Timeframe Detector
# =============================================================================
# Purpose:
#   Run the full detection pipeline independently on Weekly, Daily, and 4H
#   timeframes. Each timeframe finds its own C1, C2, and neckline.
#
#   Returns structured results per timeframe for the confluence engine.
# =============================================================================

import pandas as pd
import numpy as np
import os
import sys

sys.path.append(os.path.abspath("."))

from src.detection.wxxl_downtrend_gate   import check_downtrend_gate
from src.detection.wxxl_pip_extractor    import extract_double_bottom_pips
from src.detection.wxxl_dtw_matcher      import dtw_match
from src.detection.wxxl_shapelet         import shapelet_vote

# =============================================================================
# TIMEFRAME CONFIGS
# =============================================================================

TF_CONFIGS = {
    "weekly": {
        "min_decline_pct":   0.15,
        "lookback":          40,
        "low_window":        20,
        "low_proximity_pct": 0.04,
        "min_pattern_bars":  8,
        "max_pattern_bars":  60,
        "step":              2,
        "dtw_threshold":     0.90,
        "window_before":     40,
        "window_after":      50,
    },
    "daily": {
        "min_decline_pct":   0.12,
        "lookback":          60,
        "low_window":        40,
        "low_proximity_pct": 0.03,
        "min_pattern_bars":  10,
        "max_pattern_bars":  200,
        "step":              3,
        "dtw_threshold":     0.80,
        "window_before":     60,
        "window_after":      80,
    },
    "h4": {
        "min_decline_pct":   0.08,
        "lookback":          120,
        "low_window":        60,
        "low_proximity_pct": 0.03,
        "min_pattern_bars":  20,
        "max_pattern_bars":  200,
        "step":              5,
        "dtw_threshold":     0.85,
        "window_before":     80,
        "window_after":      100,
    },
}


# =============================================================================
# DETECT ON SINGLE TIMEFRAME
# =============================================================================

def detect_on_timeframe(
    df:        pd.DataFrame,
    tf:        str,
    scan_tail: int = 300,
) -> list:
    """
    Run full detection pipeline on a single timeframe dataframe.

    Parameters
    ----------
    df        : pd.DataFrame   OHLCV data for the timeframe
    tf        : str            'weekly', 'daily', or 'h4'
    scan_tail : int            Only scan the last N bars

    Returns
    -------
    list of dicts, each representing one confirmed pattern
    """
    cfg     = TF_CONFIGS[tf]
    prices  = df["Close"].reset_index(drop=True)
    dates   = df.index
    results = []

    scan_start = max(cfg["lookback"] + 10, len(prices) - scan_tail)

    for i in range(scan_start, len(prices) - 5, cfg["step"]):

        # Layer 1: Downtrend gate
        gate = check_downtrend_gate(
            prices            = prices,
            c1_idx            = i,
            lookback          = cfg["lookback"],
            min_decline_pct   = cfg["min_decline_pct"],
            low_window        = cfg["low_window"],
            low_proximity_pct = cfg["low_proximity_pct"],
        )
        if not gate["passed"]:
            continue

        # Layer 2a: PIP extraction
        pip_result = extract_double_bottom_pips(
            prices        = prices,
            c1_idx        = i,
            window_before = cfg["window_before"],
            window_after  = cfg["window_after"],
        )
        if not pip_result["valid"]:
            continue

        pip_prices = pip_result["pip_prices"]
        c2_idx     = pip_result["pips_absolute"][pip_result["c2_pip"]]
        neckline   = pip_result["neckline_price"]
        c2_price   = pip_result["c2_price"]

        pattern_len = c2_idx - i
        if not (cfg["min_pattern_bars"] <= pattern_len <= cfg["max_pattern_bars"]):
            continue

        # Layer 2b: DTW match
        if not dtw_match(pip_prices, threshold=cfg["dtw_threshold"]):
            continue

        # Layer 2c: Shapelet vote
        if not shapelet_vote(pip_prices):
            continue

        # All layers passed
        results.append({
            "tf":            tf,
            "c1_idx":        i,
            "c2_idx":        c2_idx,
            "c1_price":      float(prices.iloc[i]),
            "c2_price":      float(c2_price),
            "neckline":      float(neckline),
            "c1_date":       dates[i]        if i        < len(dates) else None,
            "c2_date":       dates[c2_idx]   if c2_idx   < len(dates) else None,
            "prior_decline": gate["prior_decline"],
            "pattern_bars":  pattern_len,
        })

    return results


# =============================================================================
# RESAMPLE DAILY TO WEEKLY
# =============================================================================

def resample_to_weekly(daily: pd.DataFrame) -> pd.DataFrame:
    return daily.resample("W").agg({
        "Open":   "first",
        "High":   "max",
        "Low":    "min",
        "Close":  "last",
        "Volume": "sum",
    }).dropna()


# =============================================================================
# LOAD 4H DATA
# =============================================================================

def load_4h(ticker: str, raw_4h_dir: str = "data/raw_4h") -> pd.DataFrame | None:
    path = os.path.join(raw_4h_dir, f"{ticker}.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df.columns = [c.strip().title() for c in df.columns]
    # Strip timezone for consistency
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    return df


# =============================================================================
# RUN ALL THREE TIMEFRAMES
# =============================================================================

def detect_all_timeframes(
    ticker:       str,
    daily:        pd.DataFrame,
    raw_4h_dir:   str = "data/raw_4h",
    scan_tail_d:  int = 300,
    anchor_window_weekly_days: int = 90,
    anchor_window_h4_days:     int = 30,
) -> dict:
    """
    Run detection on weekly, daily, and 4H independently.
    Daily is the anchor — weekly and 4H are filtered to patterns
    whose C2 date falls within a window of the most recent daily C2.

    Returns
    -------
    dict with keys 'weekly', 'daily', 'h4' — each a list of pattern dicts
    """
    results = {"weekly": [], "daily": [], "h4": []}

    # Daily first — this is the anchor
    try:
        daily_patterns = detect_on_timeframe(daily, "daily", scan_tail=scan_tail_d)
        results["daily"] = daily_patterns
    except Exception as e:
        print(f"  [{ticker}] Daily detection error: {e}")
        return results

    if not results["daily"]:
        return results

    # Anchor = most recent daily C2 date
    anchor_date = pd.Timestamp(results["daily"][-1]["c2_date"])

    # Weekly — filter to patterns whose C2 is within anchor_window_weekly_days
    try:
        weekly = resample_to_weekly(daily)
        all_weekly = detect_on_timeframe(weekly, "weekly", scan_tail=len(weekly))
        results["weekly"] = [
            p for p in all_weekly
            if abs((pd.Timestamp(p["c2_date"]) - anchor_date).days) <= anchor_window_weekly_days
        ]
    except Exception as e:
        print(f"  [{ticker}] Weekly detection error: {e}")

    # 4H — filter to patterns whose C2 is within anchor_window_h4_days
    try:
        h4 = load_4h(ticker, raw_4h_dir)
        if h4 is not None and len(h4) > 100:
            all_h4 = detect_on_timeframe(h4, "h4", scan_tail=len(h4))
            results["h4"] = [
                p for p in all_h4
                if abs((pd.Timestamp(p["c2_date"]) - anchor_date).days) <= anchor_window_h4_days
            ]
    except Exception as e:
        print(f"  [{ticker}] 4H detection error: {e}")

    return results


# =============================================================================
# QUICK TEST
# =============================================================================

if __name__ == "__main__":
    PROCESSED_DIR = "data/processed"
    ticker = "BBY"

    path = os.path.join(PROCESSED_DIR, f"{ticker}.csv")
    daily = pd.read_csv(path, index_col=0, parse_dates=True)
    daily.columns = [c.strip().title() for c in daily.columns]

    print(f"Running MTF detection on {ticker}...")
    results = detect_all_timeframes(ticker, daily)

    for tf, patterns in results.items():
        print(f"\n{tf.upper()}: {len(patterns)} pattern(s) found")
        for p in patterns[-3:]:
            print(f"  C1={p['c1_date']} C2={p['c2_date']} "
                  f"C1p={p['c1_price']:.2f} C2p={p['c2_price']:.2f} "
                  f"NL={p['neckline']:.2f}")