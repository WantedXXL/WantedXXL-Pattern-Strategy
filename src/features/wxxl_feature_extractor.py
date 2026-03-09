# =============================================================================
# WXXL-PAT | Feature Extractor
# =============================================================================
# Purpose:
#   For each confirmed double bottom pattern, extract all features
#   available at the Cave 2 bar. Zero lookahead — only data up to
#   and including Cave 2 is used.
#
# Features cover 6 pillars:
#   P1 — Structural Geometry  (Wyckoff / Raschke)
#   P2 — Fibonacci / Harmonic
#   P3 — Volume & Momentum
#   P4 — Volatility & Compression
#   P5 — Relative Strength & Regime
#   P6 — Pattern Quality
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
PROCESSED_DIR  = "data/processed"
OUTPUT_PATH    = "data/features/feature_matrix.csv"


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def safe_div(a, b, default=0.0):
    """Safe division — returns default if b is zero."""
    return a / b if b != 0 else default


def compute_rsi(closes: pd.Series, period: int = 14) -> pd.Series:
    delta  = closes.diff()
    gain   = delta.clip(lower=0).ewm(span=period, adjust=False).mean()
    loss   = (-delta.clip(upper=0)).ewm(span=period, adjust=False).mean()
    rs     = gain / loss.replace(0, 1e-10)
    return 100 - (100 / (1 + rs))


def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high, low, close = df["High"], df["Low"], df["Close"]
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low  - close.shift(1)).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()


def compute_obv(df: pd.DataFrame) -> pd.Series:
    direction = np.sign(df["Close"].diff()).fillna(0)
    return (direction * df["Volume"]).cumsum()


# =============================================================================
# FEATURE EXTRACTION — SINGLE PATTERN
# =============================================================================

def extract_features(
    df: pd.DataFrame,
    c1_idx: int,
    c2_idx: int,
    c1_price: float,
    c2_price: float,
    neckline_price: float,
) -> dict:
    """
    Extract all features for a single confirmed pattern at Cave 2 bar.
    All features use only data up to and including c2_idx.

    Returns dict of feature_name -> value
    """
    features = {}

    # Slice data up to C2 — zero leakage
    data = df.iloc[:c2_idx + 1].copy()
    if len(data) < 20:
        return {}

    close   = data["Close"]
    high    = data["High"]
    low     = data["Low"]
    volume  = data["Volume"]
    c2_bar  = data.iloc[-1]
    c1_bar  = data.iloc[c1_idx] if c1_idx < len(data) else data.iloc[-1]

    pattern_length = c2_idx - c1_idx  # bars from C1 to C2

    # ==========================================================================
    # PILLAR 1 — STRUCTURAL GEOMETRY
    # ==========================================================================

    # Cave depth and symmetry
    features["z1_cave_depth_pct"]     = safe_div(neckline_price - c1_price, neckline_price)
    features["z1_cave_symmetry"]      = 1.0 - safe_div(abs(c2_price - c1_price), c1_price)
    features["z1_c2_vs_c1"]          = safe_div(c2_price - c1_price, c1_price)
    features["z1_neckline_height"]    = safe_div(neckline_price - c1_price, c1_price)
    features["z1_pattern_length"]     = pattern_length

    # Prior decline (downtrend before C1)
    lookback_start = max(0, c1_idx - 60)
    prior_window   = df["Close"].iloc[lookback_start:c1_idx + 1]
    prior_high     = prior_window.max()
    features["z1_decline_magnitude"]  = safe_div(prior_high - c1_price, prior_high)
    features["z1_decline_bars"]       = c1_idx - lookback_start

    # Higher low check — is C2 higher than the lowest point after C1?
    between = df["Close"].iloc[c1_idx:c2_idx + 1]
    features["z1_higher_low"]         = 1.0 if c2_price > between.min() * 0.98 else 0.0

    # Spring detection — did price undercut C1 then recover?
    min_between = between.min()
    features["z4_spring_depth"]       = safe_div(c1_price - min_between, c1_price)
    features["z4_spring_present"]     = 1.0 if min_between < c1_price * 0.99 else 0.0

    # ==========================================================================
    # PILLAR 2 — FIBONACCI
    # ==========================================================================

    fib_range = neckline_price - c1_price
    fib_382   = neckline_price - 0.382 * fib_range
    fib_500   = neckline_price - 0.500 * fib_range
    fib_618   = neckline_price - 0.618 * fib_range

    features["z2_c2_vs_fib382"]  = safe_div(abs(c2_price - fib_382), fib_range)
    features["z2_c2_vs_fib500"]  = safe_div(abs(c2_price - fib_500), fib_range)
    features["z2_c2_vs_fib618"]  = safe_div(abs(c2_price - fib_618), fib_range)
    features["z2_fib_alignment"] = min(
        features["z2_c2_vs_fib382"],
        features["z2_c2_vs_fib500"],
        features["z2_c2_vs_fib618"]
    )

    # ==========================================================================
    # PILLAR 3 — VOLUME & MOMENTUM
    # ==========================================================================

    avg_vol_20 = volume.iloc[-20:].mean()
    c2_vol     = volume.iloc[-1]

    features["z3_rvol_c2"]           = safe_div(c2_vol, avg_vol_20)
    features["z3_volume_trend"]       = safe_div(
        volume.iloc[-10:].mean() - volume.iloc[-20:-10].mean(),
        volume.iloc[-20:-10].mean()
    )

    # Volume at C1 vs C2 — Wyckoff: C2 should have lower volume than C1
    c1_vol = df["Volume"].iloc[c1_idx]
    features["z3_c2_vol_vs_c1"]      = safe_div(c2_vol, c1_vol)
    features["z3_vol_declining"]      = 1.0 if c2_vol < c1_vol else 0.0

    # OBV divergence
    obv = compute_obv(data)
    features["z3_obv_slope"]         = safe_div(
        obv.iloc[-1] - obv.iloc[-20],
        abs(obv.iloc[-20]) + 1
    )

    # RSI
    rsi = compute_rsi(close)
    features["z3_rsi_c2"]            = rsi.iloc[-1]
    features["z3_rsi_c1"]            = rsi.iloc[c1_idx] if c1_idx < len(rsi) else 50.0
    features["z3_rsi_divergence"]    = features["z3_rsi_c2"] - features["z3_rsi_c1"]

    # RSI oversold at C2
    features["z3_rsi_oversold"]      = 1.0 if rsi.iloc[-1] < 35 else 0.0

    # MACD simplified
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd  = ema12 - ema26
    features["z3_macd_c2"]           = macd.iloc[-1]
    features["z3_macd_rising"]       = 1.0 if macd.iloc[-1] > macd.iloc[-5] else 0.0

    # ==========================================================================
    # PILLAR 4 — VOLATILITY & COMPRESSION
    # ==========================================================================

    atr = compute_atr(data)
    features["z4_atr_c2"]            = atr.iloc[-1]
    features["z4_atr_pct"]           = safe_div(atr.iloc[-1], close.iloc[-1])

    # ATR compression — is volatility contracting at C2?
    atr_recent = atr.iloc[-5:].mean()
    atr_prior  = atr.iloc[-20:-5].mean()
    features["z4_atr_compression"]   = safe_div(atr_recent, atr_prior)

    # NR4 — is today's range the narrowest of last 4 bars?
    ranges = (high - low).iloc[-4:]
    features["z4_nr4"]               = 1.0 if (high.iloc[-1] - low.iloc[-1]) == ranges.min() else 0.0

    # NR7
    ranges7 = (high - low).iloc[-7:]
    features["z4_nr7"]               = 1.0 if (high.iloc[-1] - low.iloc[-1]) == ranges7.min() else 0.0

    # Bollinger Band width
    sma20  = close.rolling(20).mean()
    std20  = close.rolling(20).std()
    bb_width = safe_div(4 * std20.iloc[-1], sma20.iloc[-1])
    features["z4_bb_width"]          = bb_width

    # ==========================================================================
    # PILLAR 5 — RELATIVE STRENGTH & TREND
    # ==========================================================================

    features["z5_rs_rank_c2"]        = df["RS_rank"].iloc[c2_idx] if "RS_rank" in df.columns else 50.0
    features["z5_rs_rank_c1"]        = df["RS_rank"].iloc[c1_idx] if "RS_rank" in df.columns else 50.0
    features["z5_rs_improving"]      = 1.0 if features["z5_rs_rank_c2"] > features["z5_rs_rank_c1"] else 0.0

    # Price vs moving averages
    sma50  = close.rolling(50).mean().iloc[-1]
    sma200 = close.rolling(200).mean().iloc[-1]
    features["z5_price_vs_sma50"]    = safe_div(close.iloc[-1] - sma50, sma50)
    features["z5_price_vs_sma200"]   = safe_div(close.iloc[-1] - sma200, sma200)
    features["z5_sma50_slope"]       = safe_div(
        close.rolling(50).mean().iloc[-1] - close.rolling(50).mean().iloc[-10],
        close.rolling(50).mean().iloc[-10]
    )

    # ==========================================================================
    # PILLAR 6 — PATTERN QUALITY
    # ==========================================================================

    # Candlestick at C2
    c2_body  = abs(c2_bar["Close"] - c2_bar["Open"])
    c2_range = c2_bar["High"] - c2_bar["Low"]
    features["z6_c2_body_ratio"]     = safe_div(c2_body, c2_range)
    features["z6_c2_lower_wick"]     = safe_div(
        min(c2_bar["Open"], c2_bar["Close"]) - c2_bar["Low"], c2_range
    )
    features["z6_c2_is_bullish"]     = 1.0 if c2_bar["Close"] > c2_bar["Open"] else 0.0

    # Pattern duration quality
    features["z6_pattern_bars"]      = pattern_length
    features["z6_pattern_too_short"] = 1.0 if pattern_length < 10 else 0.0
    features["z6_pattern_too_long"]  = 1.0 if pattern_length > 200 else 0.0

    # Round all to 6 decimal places
    features = {k: round(float(v), 6) if not np.isnan(float(v)) else 0.0
                for k, v in features.items()}

    return features


# =============================================================================
# MAIN — RUN ON ALL CONFIRMED PATTERNS
# =============================================================================

def run_feature_extractor() -> None:
    os.makedirs("data/features", exist_ok=True)

    patterns = pd.read_csv(PATTERNS_PATH, parse_dates=["c1_date"])
    print(f"Extracting features for {len(patterns)} confirmed patterns...")

    all_features = []
    failed       = 0

    for i, row in patterns.iterrows():
        ticker = row["ticker"]
        c1_idx = int(row["c1_idx"])
        c2_idx = int(row["c2_idx"])

        if c2_idx <= c1_idx:
            failed += 1
            continue

        path = os.path.join(PROCESSED_DIR, f"{ticker}.csv")
        if not os.path.exists(path):
            failed += 1
            continue

        try:
            df = pd.read_csv(path, index_col=0, parse_dates=True)
            df.columns = [c.strip().title() for c in df.columns]

            feats = extract_features(
                df             = df,
                c1_idx         = c1_idx,
                c2_idx         = c2_idx,
                c1_price       = float(row["c1_price"]),
                c2_price       = float(row["c2_price"]),
                neckline_price = float(row["neckline_price"]),
            )

            if feats:
                feats["ticker"]  = ticker
                feats["c1_date"] = row["c1_date"]
                feats["c1_idx"]  = c1_idx
                feats["c2_idx"]  = c2_idx
                all_features.append(feats)

        except Exception as e:
            failed += 1

        if (i + 1) % 200 == 0:
            print(f"  {i + 1}/{len(patterns)} done...")

    df_out = pd.DataFrame(all_features)
    df_out.to_csv(OUTPUT_PATH, index=False)

    print(f"\n{'='*50}")
    print(f"Feature extraction complete.")
    print(f"Patterns processed : {len(all_features)}")
    print(f"Failed             : {failed}")
    print(f"Features per row   : {len(df_out.columns) - 4}")
    print(f"Saved to           : {OUTPUT_PATH}")
    print(f"{'='*50}")
    print(f"\nFeature summary:")
    print(df_out.describe().round(4).to_string())


# =============================================================================
if __name__ == "__main__":
    run_feature_extractor()