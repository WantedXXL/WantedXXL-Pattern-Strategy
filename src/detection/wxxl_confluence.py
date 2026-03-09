# =============================================================================
# WXXL-PAT | Confluence Engine
# =============================================================================
# Purpose:
#   Takes independent pattern detections from Weekly, Daily, and 4H
#   and determines whether they are confluent — i.e. describing the
#   same underlying W structure across timeframes.
#
#   Outputs a confluence score (0.0 to 1.0) and the best aligned
#   pattern set for visualisation and feature augmentation.
# =============================================================================

import pandas as pd
import numpy as np
from itertools import product

# =============================================================================
# CONFLUENCE RULES
# =============================================================================

# Maximum % difference between C2 prices across timeframes
C2_PRICE_TOLERANCE    = 0.06   # 6%

# Maximum % difference between necklines across timeframes
NECKLINE_TOLERANCE    = 0.10   # 10%

# Maximum days between Daily C2 and Weekly C2
WEEKLY_TIME_WINDOW    = 14     # days

# Maximum days between Daily C2 and 4H C2
H4_TIME_WINDOW        = 7      # days

# Minimum prior decline agreement across timeframes
DECLINE_TOLERANCE     = 0.10   # 10% difference allowed


# =============================================================================
# HELPERS
# =============================================================================

def pct_diff(a: float, b: float) -> float:
    """Absolute percentage difference between two prices."""
    mid = (abs(a) + abs(b)) / 2
    return abs(a - b) / mid if mid > 0 else 0.0


def days_between(d1, d2) -> float:
    """Days between two timestamps."""
    try:
        return abs((pd.Timestamp(d1) - pd.Timestamp(d2)).days)
    except Exception:
        return 9999.0


# =============================================================================
# SCORE A SINGLE TRIPLE (weekly, daily, h4)
# =============================================================================

def score_triple(
    w: dict,
    d: dict,
    h: dict,
) -> dict:
    """
    Score confluence between one weekly, one daily, one 4H pattern.

    Returns dict with:
        score       : float  0.0–1.0
        checks      : dict   individual check results
        aligned     : bool   True if minimum confluence met
    """
    checks = {}
    points = 0
    max_points = 6

    # 1. C2 price agreement: weekly vs daily
    checks["c2_wd"] = pct_diff(w["c2_price"], d["c2_price"]) <= C2_PRICE_TOLERANCE
    if checks["c2_wd"]:
        points += 1

    # 2. C2 price agreement: daily vs 4H
    checks["c2_dh"] = pct_diff(d["c2_price"], h["c2_price"]) <= C2_PRICE_TOLERANCE
    if checks["c2_dh"]:
        points += 1

    # 3. Neckline agreement: weekly vs daily
    checks["nl_wd"] = pct_diff(w["neckline"], d["neckline"]) <= NECKLINE_TOLERANCE
    if checks["nl_wd"]:
        points += 1

    # 4. Neckline agreement: daily vs 4H
    checks["nl_dh"] = pct_diff(d["neckline"], h["neckline"]) <= NECKLINE_TOLERANCE
    if checks["nl_dh"]:
        points += 1

    # 5. Time alignment: daily C2 within weekly C2 window
    checks["time_wd"] = days_between(d["c2_date"], w["c2_date"]) <= WEEKLY_TIME_WINDOW
    if checks["time_wd"]:
        points += 1

    # 6. Time alignment: 4H C2 within daily C2 window
    checks["time_dh"] = days_between(h["c2_date"], d["c2_date"]) <= H4_TIME_WINDOW
    if checks["time_dh"]:
        points += 1

    score   = points / max_points
    aligned = points >= 4   # Minimum 4/6 checks to be considered confluent

    return {
        "score":   round(score, 4),
        "points":  points,
        "checks":  checks,
        "aligned": aligned,
    }


# =============================================================================
# SCORE PARTIAL (only 2 timeframes available)
# =============================================================================

def score_partial(a: dict, b: dict, tf_a: str, tf_b: str) -> dict:
    """Score confluence between two timeframes only."""
    checks = {}
    points = 0
    max_points = 3

    checks["c2_price"] = pct_diff(a["c2_price"], b["c2_price"]) <= C2_PRICE_TOLERANCE
    if checks["c2_price"]:
        points += 1

    checks["neckline"] = pct_diff(a["neckline"], b["neckline"]) <= NECKLINE_TOLERANCE
    if checks["neckline"]:
        points += 1

    window = WEEKLY_TIME_WINDOW if "weekly" in [tf_a, tf_b] else H4_TIME_WINDOW
    checks["time"] = days_between(a["c2_date"], b["c2_date"]) <= window
    if checks["time"]:
        points += 1

    score   = points / max_points
    aligned = points >= 2

    return {
        "score":   round(score, 4),
        "points":  points,
        "checks":  checks,
        "aligned": aligned,
    }


# =============================================================================
# FIND BEST CONFLUENT SET
# =============================================================================

def find_best_confluence(
    weekly_patterns: list,
    daily_patterns:  list,
    h4_patterns:     list,
) -> dict:
    """
    Find the best-aligned combination of patterns across timeframes.

    Tries all combinations and returns the highest-scoring triple.
    Falls back to best pair if no triple found.

    Returns
    -------
    dict with:
        weekly          : best weekly pattern (or None)
        daily           : best daily pattern (or None)
        h4              : best h4 pattern (or None)
        score           : confluence score 0.0–1.0
        aligned         : bool
        tfs_confirmed   : int   number of timeframes confirmed
        checks          : dict
    """
    best = {
        "weekly":        None,
        "daily":         None,
        "h4":            None,
        "score":         0.0,
        "aligned":       False,
        "tfs_confirmed": 0,
        "checks":        {},
    }

    # ── Try all triples ────────────────────────────────────────────────────
    if weekly_patterns and daily_patterns and h4_patterns:
        for w, d, h in product(weekly_patterns, daily_patterns, h4_patterns):
            result = score_triple(w, d, h)
            if result["score"] > best["score"]:
                best.update({
                    "weekly":        w,
                    "daily":         d,
                    "h4":            h,
                    "score":         result["score"],
                    "aligned":       result["aligned"],
                    "tfs_confirmed": 3 if result["aligned"] else 2,
                    "checks":        result["checks"],
                })

        if best["aligned"]:
            return best

    # ── Try daily + weekly pair ────────────────────────────────────────────
    if daily_patterns and weekly_patterns:
        for d, w in product(daily_patterns, weekly_patterns):
            result = score_partial(d, w, "daily", "weekly")
            if result["score"] > best["score"]:
                best.update({
                    "weekly":        w,
                    "daily":         d,
                    "h4":            None,
                    "score":         result["score"],
                    "aligned":       result["aligned"],
                    "tfs_confirmed": 2 if result["aligned"] else 1,
                    "checks":        result["checks"],
                })

    # ── Try daily + 4H pair ────────────────────────────────────────────────
    if daily_patterns and h4_patterns:
        for d, h in product(daily_patterns, h4_patterns):
            result = score_partial(d, h, "daily", "h4")
            if result["score"] > best["score"]:
                best.update({
                    "weekly":        None,
                    "daily":         d,
                    "h4":            h,
                    "score":         result["score"],
                    "aligned":       result["aligned"],
                    "tfs_confirmed": 2 if result["aligned"] else 1,
                    "checks":        result["checks"],
                })

    # ── Daily only fallback ────────────────────────────────────────────────
    if not best["daily"] and daily_patterns:
        best.update({
            "daily":         daily_patterns[-1],
            "score":         0.33,
            "aligned":       False,
            "tfs_confirmed": 1,
        })

    return best


# =============================================================================
# CONFLUENCE FEATURE FOR XGBOOST
# =============================================================================

def confluence_to_features(confluence: dict) -> dict:
    """
    Convert confluence result into flat features for XGBoost.
    """
    d = confluence.get("daily")
    w = confluence.get("weekly")
    h = confluence.get("h4")

    features = {
        "mtf_score":          confluence["score"],
        "mtf_tfs_confirmed":  confluence["tfs_confirmed"],
        "mtf_aligned":        1.0 if confluence["aligned"] else 0.0,
        "mtf_weekly_present": 1.0 if w is not None else 0.0,
        "mtf_h4_present":     1.0 if h is not None else 0.0,
    }

    if w and d:
        features["mtf_c2_wd_diff"]  = pct_diff(w["c2_price"], d["c2_price"])
        features["mtf_nl_wd_diff"]  = pct_diff(w["neckline"], d["neckline"])
    else:
        features["mtf_c2_wd_diff"]  = 1.0
        features["mtf_nl_wd_diff"]  = 1.0

    if h and d:
        features["mtf_c2_dh_diff"]  = pct_diff(h["c2_price"], d["c2_price"])
        features["mtf_nl_dh_diff"]  = pct_diff(h["neckline"], d["neckline"])
    else:
        features["mtf_c2_dh_diff"]  = 1.0
        features["mtf_nl_dh_diff"]  = 1.0

    return features


# =============================================================================
# QUICK TEST
# =============================================================================

if __name__ == "__main__":
    import os
    import sys
    sys.path.append(".")

    from src.detection.wxxl_mtf_detector import detect_all_timeframes

    PROCESSED_DIR = "data/processed"
    ticker = "BBY"

    daily = pd.read_csv(
        os.path.join(PROCESSED_DIR, f"{ticker}.csv"),
        index_col=0, parse_dates=True
    )
    daily.columns = [c.strip().title() for c in daily.columns]

    print(f"Running MTF detection + confluence for {ticker}...")
    mtf     = detect_all_timeframes(ticker, daily)
    result  = find_best_confluence(mtf["weekly"], mtf["daily"], mtf["h4"])

    print(f"\nConfluence result:")
    print(f"  Score          : {result['score']:.2f}")
    print(f"  Aligned        : {result['aligned']}")
    print(f"  TFs confirmed  : {result['tfs_confirmed']}")
    print(f"  Checks         : {result['checks']}")

    if result["daily"]:
        d = result["daily"]
        print(f"\n  Daily  — C1: {d['c1_date']} C2: {d['c2_date']} NL: {d['neckline']:.2f}")
    if result["weekly"]:
        w = result["weekly"]
        print(f"  Weekly — C1: {w['c1_date']} C2: {w['c2_date']} NL: {w['neckline']:.2f}")
    if result["h4"]:
        h = result["h4"]
        print(f"  4H     — C1: {h['c1_date']} C2: {h['c2_date']} NL: {h['neckline']:.2f}")

    print(f"\nMTF features for XGBoost:")
    print(confluence_to_features(result))