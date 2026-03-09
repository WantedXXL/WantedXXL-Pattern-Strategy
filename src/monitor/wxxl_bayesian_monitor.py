# =============================================================================
# WXXL-PAT | Bayesian Cave 2 Monitor
# =============================================================================
# Purpose:
#   After a double bottom candidate is identified at Cave 2, this monitor
#   watches each new candle and updates P(success) using Bayes' theorem.
#
#   P(success | evidence) = P(evidence | success) * P(success) / P(evidence)
#
#   Each new candle updates the posterior using likelihood ratios derived
#   from the labeled feature matrix. When P > FIRE_THRESHOLD the monitor
#   fires E1 entry signal.
#
# Architecture:
#   - Prior = XGBoost probability at Cave 2 bar
#   - Likelihood multipliers = candle-by-candle evidence updates
#   - Physical trigger = price must also close above a trigger level
#   - E1 fires when P > 0.65 AND physical trigger confirmed
# =============================================================================

import pandas as pd
import numpy as np
import os
import sys
import json
import warnings
warnings.filterwarnings("ignore")

sys.path.append(os.path.abspath("."))

import xgboost as xgb

# =============================================================================
# CONFIG
# =============================================================================
MODEL_PATH      = "models/xgb/wxxl_xgb_v4.json"
FIRE_THRESHOLD  = 0.65    # P(success) threshold to fire E1
MAX_WATCH_BARS  = 30      # Maximum bars to watch after C2
MIN_PRIOR       = 0.40    # Minimum XGBoost prior to start monitoring

# Likelihood ratio bounds — prevent single candle dominating
LR_MIN = 0.5
LR_MAX = 2.0


# =============================================================================
# LIKELIHOOD FUNCTIONS
# =============================================================================
# Each function takes current bar data and returns a likelihood ratio.
# LR > 1.0 = bullish evidence, LR < 1.0 = bearish evidence
# These are calibrated from the labeled dataset statistics.

def lr_price_holding(bar: pd.Series, c2_price: float, c1_price: float) -> float:
    """Price holding above Cave 2 low — key confirmation."""
    close = bar["Close"]
    if close > c2_price * 1.005:
        return 1.3   # Holding above C2 — bullish
    elif close > c2_price * 0.99:
        return 1.0   # At C2 — neutral
    elif close > c1_price:
        return 0.8   # Below C2 but above C1 — mild concern
    else:
        return 0.5   # Below C1 — bearish


def lr_volume(bar: pd.Series, avg_volume: float) -> float:
    """Volume confirmation — rising volume on up bars."""
    vol   = bar["Volume"]
    close = bar["Close"]
    open_ = bar["Open"]
    rvol  = vol / avg_volume if avg_volume > 0 else 1.0

    if close > open_ and rvol > 1.2:
        return 1.25  # Up bar with above average volume — bullish
    elif close < open_ and rvol > 1.5:
        return 0.75  # Down bar with high volume — bearish
    else:
        return 1.0


def lr_momentum(bar: pd.Series, prev_closes: list) -> float:
    """Short term momentum — are recent closes trending up?"""
    if len(prev_closes) < 3:
        return 1.0
    recent = prev_closes[-3:]
    if recent[-1] > recent[-2] > recent[-3]:
        return 1.2   # Three consecutive higher closes
    elif recent[-1] < recent[-2] < recent[-3]:
        return 0.8   # Three consecutive lower closes
    else:
        return 1.0


def lr_approach_neckline(bar: pd.Series, neckline: float, c2_price: float) -> float:
    """Is price approaching the neckline?"""
    close    = bar["Close"]
    progress = (close - c2_price) / (neckline - c2_price) if neckline > c2_price else 0

    if progress > 0.60:
        return 1.35  # More than 60% of the way to neckline — very bullish
    elif progress > 0.30:
        return 1.15  # Making progress
    elif progress < 0:
        return 0.75  # Moving away from neckline
    else:
        return 1.0


def lr_candle_structure(bar: pd.Series) -> float:
    """Candle body and wick structure."""
    o, h, l, c = bar["Open"], bar["High"], bar["Low"], bar["Close"]
    rng  = h - l
    if rng < 1e-8:
        return 1.0

    body       = abs(c - o) / rng
    lower_wick = (min(o, c) - l) / rng

    if c > o and lower_wick > 0.3 and body > 0.4:
        return 1.2   # Bullish candle with lower wick — demand present
    elif c < o and body > 0.6:
        return 0.8   # Strong bearish candle
    else:
        return 1.0


# =============================================================================
# BAYESIAN UPDATE
# =============================================================================

def bayesian_update(prior: float, likelihood_ratio: float) -> float:
    """
    Single Bayesian update step.
    posterior = (prior * LR) / (prior * LR + (1 - prior))
    """
    lr      = np.clip(likelihood_ratio, LR_MIN, LR_MAX)
    num     = prior * lr
    denom   = num + (1 - prior)
    return np.clip(num / denom, 0.01, 0.99)


# =============================================================================
# MONITOR CLASS
# =============================================================================

class CaveMonitor:
    """
    Monitors a single Cave 2 candidate candle by candle.
    Updates P(success) with each new bar.
    Fires E1 when P > FIRE_THRESHOLD and physical trigger met.
    """

    def __init__(
        self,
        ticker:    str,
        c2_idx:    int,
        c2_price:  float,
        c1_price:  float,
        neckline:  float,
        prior:     float,
    ):
        self.ticker    = ticker
        self.c2_idx    = c2_idx
        self.c2_price  = c2_price
        self.c1_price  = c1_price
        self.neckline  = neckline
        self.prior     = prior

        self.probability   = prior
        self.bars_watched  = 0
        self.fired         = False
        self.fire_bar      = None
        self.fire_price    = None
        self.history       = []
        self.prev_closes   = []

    def update(self, bar: pd.Series, avg_volume: float) -> dict:
        """
        Process one new candle and update probability.

        Returns dict with current state.
        """
        if self.fired or self.bars_watched >= MAX_WATCH_BARS:
            return self._state("timeout" if not self.fired else "fired")

        # Compute likelihood ratios
        lr1 = lr_price_holding(bar, self.c2_price, self.c1_price)
        lr2 = lr_volume(bar, avg_volume)
        lr3 = lr_momentum(bar, self.prev_closes)
        lr4 = lr_approach_neckline(bar, self.neckline, self.c2_price)
        lr5 = lr_candle_structure(bar)

        # Combined LR — product of independent likelihoods
        combined_lr = lr1 * lr2 * lr3 * lr4 * lr5
        combined_lr = np.clip(combined_lr, LR_MIN, LR_MAX)

        # Bayesian update
        self.probability = bayesian_update(self.probability, combined_lr)

        # Physical trigger — price must close above midpoint of C2-neckline
        trigger_level  = self.c2_price + (self.neckline - self.c2_price) * 0.25
        physical_ok    = bar["Close"] > trigger_level

        # Record
        self.bars_watched += 1
        self.prev_closes.append(bar["Close"])
        self.history.append({
            "bar":         self.bars_watched,
            "close":       bar["Close"],
            "probability": round(self.probability, 4),
            "combined_lr": round(combined_lr, 4),
            "lr1_price":   lr1,
            "lr2_volume":  lr2,
            "lr3_momentum":lr3,
            "lr4_neckline":lr4,
            "lr5_candle":  lr5,
        })

        # Fire check
        if self.probability >= FIRE_THRESHOLD and physical_ok:
            self.fired      = True
            self.fire_bar   = self.bars_watched
            self.fire_price = bar["Close"]
            return self._state("FIRE_E1")

        # Abort check — probability collapsed
        if self.probability < 0.20:
            return self._state("aborted")

        return self._state("watching")

    def _state(self, status: str) -> dict:
        return {
            "ticker":      self.ticker,
            "status":      status,
            "probability": round(self.probability, 4),
            "bars":        self.bars_watched,
            "fire_bar":    self.fire_bar,
            "fire_price":  self.fire_price,
        }

    def summary(self) -> pd.DataFrame:
        return pd.DataFrame(self.history)


# =============================================================================
# RUN MONITOR ON HISTORICAL PATTERN
# =============================================================================

def simulate_monitor(
    df:        pd.DataFrame,
    c2_idx:    int,
    c2_price:  float,
    c1_price:  float,
    neckline:  float,
    prior:     float,
    ticker:    str = "",
) -> dict:
    """
    Simulate the Bayesian monitor on a historical pattern.
    Uses bars after C2 to replay the monitor.
    """
    monitor    = CaveMonitor(ticker, c2_idx, c2_price, c1_price, neckline, prior)
    avg_volume = df["Volume"].iloc[max(0, c2_idx - 20):c2_idx].mean()

    for i in range(c2_idx + 1, min(len(df), c2_idx + MAX_WATCH_BARS + 1)):
        bar    = df.iloc[i]
        result = monitor.update(bar, avg_volume)

        if result["status"] in ["FIRE_E1", "aborted", "timeout"]:
            break

    return {
        "ticker":      ticker,
        "fired":       monitor.fired,
        "fire_bar":    monitor.fire_bar,
        "fire_price":  monitor.fire_price,
        "final_prob":  monitor.probability,
        "bars_watched":monitor.bars_watched,
        "history":     monitor.summary(),
    }


# =============================================================================
# QUICK TEST
# =============================================================================

if __name__ == "__main__":
    PROCESSED_DIR = "data/processed"
    ticker = "BBY"

    print(f"Simulating Bayesian monitor on {ticker}...")

    daily = pd.read_csv(
        os.path.join(PROCESSED_DIR, f"{ticker}.csv"),
        index_col=0, parse_dates=True
    )
    daily.columns = [c.strip().title() for c in daily.columns]

    # Use the known BBY pattern
    c2_idx    = 4067
    c2_price  = 61.59
    c1_price  = 62.73
    neckline  = 70.41
    prior     = 0.58   # XGBoost v4 probability

    result = simulate_monitor(
        df        = daily,
        c2_idx    = c2_idx,
        c2_price  = c2_price,
        c1_price  = c1_price,
        neckline  = neckline,
        prior     = prior,
        ticker    = ticker,
    )

    print(f"\nResult:")
    print(f"  Fired      : {result['fired']}")
    print(f"  Fire bar   : {result['fire_bar']}")
    print(f"  Fire price : {result['fire_price']}")
    print(f"  Final prob : {result['final_prob']:.3f}")
    print(f"  Bars watched: {result['bars_watched']}")
    print(f"\nProbability history:")
    print(result["history"][["bar", "close", "probability", "combined_lr"]].to_string(index=False))