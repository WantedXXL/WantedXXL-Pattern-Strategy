# =============================================================================
# WXXL-PAT | Layer 2a — PIP Extractor
# =============================================================================
# Purpose:
#   Given a Cave 1 candidate, extract the 7 Perceptually Important Points
#   that define the double bottom skeleton.
#
#   PIP algorithm: iteratively selects points of maximum perpendicular
#   distance from the line connecting already-selected neighbours.
#   Published by Chung et al. 2001, cited in 50+ quant finance papers.
#
# 7 PIPs map to:
#   P0 = Start
#   P1 = Cave 1 (lowest point of first trough)
#   P2 = Recovery begin
#   P3 = Neckline Peak (resistance level)
#   P4 = Recovery end
#   P5 = Cave 2 (second trough — our entry zone)
#   P6 = End
# =============================================================================

import pandas as pd
import numpy as np
from typing import Optional


# =============================================================================
# PIP ALGORITHM
# =============================================================================

def compute_pip(prices: np.ndarray, n_pips: int = 7) -> list:
    """
    Extract n Perceptually Important Points from a price series.

    Parameters
    ----------
    prices : np.ndarray   1D array of prices
    n_pips : int          Number of PIPs to extract (default 7)

    Returns
    -------
    list of int           Indices of the PIP points (sorted)
    """
    # Always include first and last point
    pip_indices = [0, len(prices) - 1]

    for _ in range(n_pips - 2):
        max_dist  = -1
        max_idx   = -1

        pip_indices_sorted = sorted(pip_indices)

        for j in range(len(pip_indices_sorted) - 1):
            left_idx  = pip_indices_sorted[j]
            right_idx = pip_indices_sorted[j + 1]

            # Skip adjacent points
            if right_idx - left_idx < 2:
                continue

            # Line from left to right PIP
            x1, y1 = left_idx,  prices[left_idx]
            x2, y2 = right_idx, prices[right_idx]

            # Find point of maximum perpendicular distance between them
            for k in range(left_idx + 1, right_idx):
                # Perpendicular distance from point (k, prices[k]) to line (x1,y1)-(x2,y2)
                num  = abs((y2 - y1) * k - (x2 - x1) * prices[k] + x2 * y1 - y2 * x1)
                den  = np.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2)
                dist = num / den if den > 0 else 0

                if dist > max_dist:
                    max_dist = dist
                    max_idx  = k

        if max_idx == -1:
            break

        pip_indices.append(max_idx)

    return sorted(pip_indices)


# =============================================================================
# EXTRACT DOUBLE BOTTOM SKELETON
# =============================================================================

def extract_double_bottom_pips(
    prices: pd.Series,
    c1_idx: int,
    window_before: int = 60,
    window_after:  int = 80,
    n_pips: int = 7,
) -> Optional[dict]:
    """
    Extract the 7-PIP skeleton for a double bottom pattern
    centered around a Cave 1 candidate.

    Parameters
    ----------
    prices        : pd.Series   Full price history
    c1_idx        : int         Bar index of Cave 1 candidate
    window_before : int         Bars before C1 to include (pattern start)
    window_after  : int         Bars after C1 to look for Cave 2
    n_pips        : int         Number of PIPs (default 7)

    Returns
    -------
    dict with keys:
        pips          : list of int    PIP indices (relative to window start)
        pips_absolute : list of int    PIP indices (absolute in prices series)
        pip_prices    : list of float  Price at each PIP
        c1_pip        : int            Which PIP is Cave 1 (should be P1)
        c2_pip        : int            Which PIP is Cave 2 (should be P5)
        neckline_pip  : int            Which PIP is Neckline (should be P3)
        neckline_price: float          Price of neckline
        c2_price      : float          Price of Cave 2
        window_start  : int            Absolute start of window
        window_end    : int            Absolute end of window
        valid         : bool           True if pattern looks like a double bottom
        reason        : str            Why invalid (empty if valid)
    """
    result = {
        "pips": [], "pips_absolute": [], "pip_prices": [],
        "c1_pip": -1, "c2_pip": -1, "neckline_pip": -1,
        "neckline_price": 0.0, "c2_price": 0.0,
        "window_start": 0, "window_end": 0,
        "valid": False, "reason": ""
    }

    # Define window around C1
    win_start = max(0, c1_idx - window_before)
    win_end   = min(len(prices) - 1, c1_idx + window_after)
    result["window_start"] = win_start
    result["window_end"]   = win_end

    window_prices = prices.iloc[win_start:win_end + 1].values

    if len(window_prices) < n_pips + 5:
        result["reason"] = "window_too_small"
        return result

    # Extract PIPs
    pip_indices = compute_pip(window_prices, n_pips)
    pip_prices  = [window_prices[i] for i in pip_indices]

    result["pips"]          = pip_indices
    result["pip_prices"]    = [round(p, 4) for p in pip_prices]
    result["pips_absolute"] = [win_start + i for i in pip_indices]

    if len(pip_indices) < 7:
        result["reason"] = f"insufficient_pips: got {len(pip_indices)}"
        return result

    # Identify C1 — should be near minimum in first half
    half = len(pip_indices) // 2
    first_half_prices = pip_prices[:half + 1]
    c1_pip = first_half_prices.index(min(first_half_prices))

    # Identify C2 — minimum in second half (after neckline)
    second_half_prices = pip_prices[c1_pip + 1:]
    if not second_half_prices:
        result["reason"] = "no_second_half"
        return result

    c2_pip_local = second_half_prices.index(min(second_half_prices))
    c2_pip       = c1_pip + 1 + c2_pip_local

    # Identify Neckline — maximum between C1 and C2 (exclusive of caves)
    between_prices = pip_prices[c1_pip + 1:c2_pip]
    if len(between_prices) < 1:
        result["reason"] = "no_neckline_between_caves"
        return result

    neckline_pip_local = between_prices.index(max(between_prices))
    neckline_pip       = c1_pip + 1 + neckline_pip_local

    result["c1_pip"]        = c1_pip
    result["c2_pip"]        = c2_pip
    result["neckline_pip"]  = neckline_pip
    result["neckline_price"]= round(pip_prices[neckline_pip], 4)
    result["c2_price"]      = round(pip_prices[c2_pip], 4)

    # ==========================================================================
    # VALIDATION — Does this look like a real double bottom?
    # ==========================================================================

    c1_price       = pip_prices[c1_pip]
    c2_price       = pip_prices[c2_pip]
    neckline_price = pip_prices[neckline_pip]

    # 1. Both caves must be below the neckline
    if c1_price >= neckline_price or c2_price >= neckline_price:
        result["reason"] = "caves_not_below_neckline"
        return result

    # 2. C2 must be within 5% of C1 (symmetric caves)
    cave_diff = abs(c2_price - c1_price) / c1_price
    if cave_diff > 0.05:
        result["reason"] = f"caves_too_asymmetric: {cave_diff:.1%} difference"
        return result

    # 3. Neckline must be at least 5% above the caves
    neckline_height = (neckline_price - c1_price) / c1_price
    if neckline_height < 0.05:
        result["reason"] = f"neckline_too_shallow: {neckline_height:.1%}"
        return result

    # 4. C2 must come after C1
    if c2_pip <= c1_pip:
        result["reason"] = "c2_before_c1"
        return result

    result["valid"] = True
    return result


# =============================================================================
# QUICK TEST
# =============================================================================
if __name__ == "__main__":
    # Build a clean synthetic double bottom
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    import numpy as np
    np.random.seed(42)

    # Smooth double bottom using sine curves
    t1 = np.linspace(np.pi, 2 * np.pi, 30)   # downtrend + Cave 1
    t2 = np.linspace(0, np.pi, 20)            # recovery to neckline
    t3 = np.linspace(np.pi, 2 * np.pi, 25)   # Cave 2
    t4 = np.linspace(0, np.pi / 2, 15)       # breakout

    seg1 = 90 + 10 * np.sin(t1)              # 80 to 100 arc downward
    seg2 = 80 + 15 * np.sin(t2)              # 80 up to 95
    seg3 = 95 - 13 * np.sin(t3) * (-1)       # 95 down to 82
    seg3 = 82 + 13 * (1 - np.abs(np.sin(t3)))
    seg4 = 82 + 10 * np.sin(t4)              # 82 up

    raw = np.concatenate([seg1, seg2, seg3, seg4])
    prices = pd.Series(raw, dtype=float)

    c1_idx = int(np.argmin(seg1)) + 0

    result = extract_double_bottom_pips(prices, c1_idx, window_before=10, window_after=70)

    print("=== PIP Extractor Test ===")
    print(f"Valid         : {result['valid']}")
    print(f"Reason        : {result['reason'] or 'N/A'}")
    print(f"PIP prices    : {result['pip_prices']}")
    print(f"Neckline      : {result['neckline_price']}")
    print(f"Cave 2 price  : {result['c2_price']}")