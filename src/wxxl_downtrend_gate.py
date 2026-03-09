# =============================================================================
# WXXL-PAT | Layer 1 — Downtrend Gate
# =============================================================================
# Purpose:
#   Every double bottom REQUIRES a prior downtrend.
#   This gate verifies that a Cave 1 candidate sits at the end of a real
#   downtrend — not a random dip in an uptrend or consolidation.
#
# Root cause this fixes:
#   The previous detector found any two swing lows with no awareness of
#   price structure. This produced 72,243 candidates that were random noise.
#   This gate is the foundation everything else is built on.
# =============================================================================

import pandas as pd
import numpy as np
from typing import Dict


def check_downtrend_gate(
    prices: pd.Series,
    c1_idx: int,
    lookback: int = 60,
    min_decline_pct: float = 0.12,
    low_window: int = 40,
    low_proximity_pct: float = 0.03,
) -> Dict:
    """
    Verify that a Cave 1 candidate sits at the end of a real downtrend.

    Parameters
    ----------
    prices          : pd.Series   Close prices (full stock history)
    c1_idx          : int         Bar index of Cave 1 candidate
    lookback        : int         Bars to look back for the prior peak (default 60)
    min_decline_pct : float       Minimum decline required (default 12%)
    low_window      : int         Window to check C1 is near the low (default 40)
    low_proximity_pct: float      How close C1 must be to the rolling low (default 3%)

    Returns
    -------
    dict with keys:
        passed        : bool   — True if all conditions met
        prior_decline : float  — Actual % decline found
        low_rank      : float  — How close C1 is to 40-bar low
        peak_idx      : int    — Bar index of the prior peak
        reason        : str    — Why it failed (empty string if passed)
    """
    result = {
        'passed': False,
        'prior_decline': 0.0,
        'low_rank': 0.0,
        'peak_idx': -1,
        'reason': ''
    }

    # Need enough history
    start = max(0, c1_idx - lookback)
    window = prices.iloc[start:c1_idx]

    if len(window) < 10:
        result['reason'] = 'insufficient_history'
        return result

    # 1. Find the highest close in the lookback window before c1_idx
    peak_price = window.max()
    peak_idx = window.idxmax()
    c1_price = prices.iloc[c1_idx]

    result['peak_idx'] = int(peak_idx) if hasattr(peak_idx, 'item') else peak_idx

    # 2. Check prior decline magnitude
    prior_decline = (peak_price - c1_price) / peak_price
    result['prior_decline'] = round(prior_decline, 4)

    if prior_decline < min_decline_pct:
        result['reason'] = (
            f'decline_too_small: {prior_decline:.1%} < {min_decline_pct:.1%}'
        )
        return result

    # 3. Check C1 is near the 40-bar rolling low
    low_start = max(0, c1_idx - low_window)
    rolling_low = prices.iloc[low_start:c1_idx + 1].min()
    low_rank = (c1_price - rolling_low) / rolling_low
    result['low_rank'] = round(low_rank, 4)

    if low_rank > low_proximity_pct:
        result['reason'] = (
            f'c1_not_near_low: {low_rank:.1%} above {low_window}-bar low'
        )
        return result

    # All checks passed
    result['passed'] = True
    return result


# =============================================================================
# Quick smoke test — run this file directly to verify it works
# =============================================================================
if __name__ == '__main__':
    import numpy as np
    np.random.seed(42)

    # Simulate 80 bars: uptrend to peak, then clean downtrend to Cave 1
    uptrend   = list(range(70, 101))      # 31 bars rising to 100
    downtrend = list(range(100, 79, -1))  # 21 bars falling to 80
    noise     = [80.5, 79.8, 80.2, 79.5] # 4 bars consolidating at bottom

    prices = pd.Series(uptrend + downtrend + noise, dtype=float)
    c1_idx = len(uptrend) + len(downtrend) - 1  # bar index of the 80 low

    result = check_downtrend_gate(prices, c1_idx)
    print("=== Downtrend Gate Smoke Test ===")
    print(f"Passed       : {result['passed']}")
    print(f"Prior decline: {result['prior_decline']:.1%}")
    print(f"Low rank     : {result['low_rank']:.1%}")
    print(f"Reason       : {result['reason'] or 'N/A'}")