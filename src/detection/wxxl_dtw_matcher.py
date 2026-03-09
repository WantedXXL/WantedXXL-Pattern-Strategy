# =============================================================================
# WXXL-PAT | Layer 2b — DTW Matcher
# =============================================================================
# Purpose:
#   Compares the extracted PIP shape against a template of real double
#   bottoms using Dynamic Time Warping (DTW).
#
#   DTW measures similarity between two time series that may vary in speed.
#   A low DTW distance = shape is close to the template = votes YES.
#
# Template:
#   Built from the canonical double bottom shape. Will be replaced with
#   a data-driven DBA (DTW Barycenter Averaging) template once we have
#   enough confirmed patterns.
# =============================================================================

import numpy as np
import pandas as pd
from dtaidistance import dtw
from typing import Optional


# =============================================================================
# CANONICAL DOUBLE BOTTOM TEMPLATE
# =============================================================================
# Normalised shape: starts high, drops to Cave 1, recovers to neckline,
# drops to Cave 2 (similar depth), then begins recovery.
# Values are normalised 0-1 where 0=lowest point, 1=highest point.

CANONICAL_TEMPLATE = np.array([
    0.80,   # P0 — Start (below the prior high)
    0.00,   # P1 — Cave 1 (lowest point)
    0.50,   # P2 — Mid recovery
    1.00,   # P3 — Neckline (highest point between caves)
    0.50,   # P4 — Mid recovery down
    0.05,   # P5 — Cave 2 (slightly above Cave 1)
    0.60,   # P6 — End (beginning of breakout)
], dtype=float)


# =============================================================================
# NORMALISE PIP SERIES
# =============================================================================

def normalise(prices: list) -> np.ndarray:
    """Normalise a price series to 0-1 range."""
    arr = np.array(prices, dtype=float)
    mn  = arr.min()
    mx  = arr.max()
    if mx == mn:
        return np.zeros_like(arr)
    return (arr - mn) / (mx - mn)


# =============================================================================
# DTW MATCH
# =============================================================================

def dtw_match(
    pip_prices: list,
    template: np.ndarray = CANONICAL_TEMPLATE,
    threshold: float = 0.80,
) -> dict:
    """
    Compare extracted PIP prices against the double bottom template using DTW.

    Parameters
    ----------
    pip_prices : list          PIP prices from the extractor (7 values)
    template   : np.ndarray    Canonical double bottom template
    threshold  : float         Maximum DTW distance to vote YES (default 0.35)

    Returns
    -------
    dict with keys:
        vote          : bool   — True if DTW distance <= threshold
        dtw_distance  : float  — Raw DTW distance (lower = more similar)
        normalised    : list   — Normalised PIP prices used for comparison
        reason        : str    — Why it failed (empty if passed)
    """
    result = {
        "vote": False,
        "dtw_distance": 999.0,
        "normalised": [],
        "reason": ""
    }

    if len(pip_prices) < 4:
        result["reason"] = "insufficient_pips"
        return result

    # Normalise both series
    norm_pips     = normalise(pip_prices)
    norm_template = normalise(template)
    result["normalised"] = [round(v, 4) for v in norm_pips]

    # Compute DTW distance
    try:
        distance = dtw.distance(norm_pips, norm_template)
        result["dtw_distance"] = round(float(distance), 4)
    except Exception as e:
        result["reason"] = f"dtw_error: {e}"
        return result

    # Vote
    if distance <= threshold:
        result["vote"] = True
    else:
        result["reason"] = f"shape_too_different: distance={distance:.4f} > threshold={threshold}"

    return result


# =============================================================================
# LOAD OR BUILD TEMPLATE FROM CONFIRMED PATTERNS
# =============================================================================

def build_template_from_patterns(pip_price_list: list) -> np.ndarray:
    """
    Build a data-driven template by averaging normalised PIP shapes.
    Call this once you have 50+ confirmed patterns.

    Parameters
    ----------
    pip_price_list : list of lists   Each inner list is 7 PIP prices

    Returns
    -------
    np.ndarray   Averaged normalised template
    """
    normalised = [normalise(pips) for pips in pip_price_list]
    template   = np.mean(normalised, axis=0)
    return template


# =============================================================================
# QUICK TEST
# =============================================================================
if __name__ == "__main__":

    # Perfect double bottom PIP prices
    perfect = [90.0, 80.0, 87.0, 95.0, 87.0, 81.0, 88.0]

    # Noisy but still valid
    noisy   = [88.0, 79.5, 86.0, 94.0, 85.0, 82.0, 89.0]

    # Bad shape — uptrend, not a double bottom
    bad     = [70.0, 75.0, 80.0, 85.0, 90.0, 95.0, 100.0]

    print("=== DTW Matcher Test ===")

    for label, pips in [("Perfect", perfect), ("Noisy", noisy), ("Bad", bad)]:
        result = dtw_match(pips)
        print(f"\n{label}:")
        print(f"  Vote         : {result['vote']}")
        print(f"  DTW distance : {result['dtw_distance']}")
        print(f"  Reason       : {result['reason'] or 'N/A'}")