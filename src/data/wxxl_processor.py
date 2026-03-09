# =============================================================================
# WXXL-PAT | Data Processor
# =============================================================================
# Purpose:
#   Takes raw OHLCV CSVs from data/raw/ and computes:
#   - ATR(14)        : Average True Range — measures volatility
#   - RVOL(20)       : Relative Volume — current vol vs 20-day average
#   - RS Rank        : Relative Strength rank vs all other stocks (0-100)
#   Saves clean processed files to data/processed/
# =============================================================================

import pandas as pd
import numpy as np
import os
from datetime import datetime

# =============================================================================
# CONFIG
# =============================================================================
RAW_DIR       = "data/raw"
PROCESSED_DIR = "data/processed"
ATR_PERIOD    = 14
RVOL_PERIOD   = 20
RS_PERIOD     = 63  # ~3 months for RS calculation


# =============================================================================
# INDICATORS
# =============================================================================

def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average True Range."""
    high  = df["High"]
    low   = df["Low"]
    close = df["Close"]

    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low  - close.shift(1)).abs(),
    ], axis=1).max(axis=1)

    return tr.ewm(span=period, adjust=False).mean()


def compute_rvol(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """Relative Volume: today's volume vs rolling average."""
    avg_vol = df["Volume"].rolling(period).mean()
    return df["Volume"] / avg_vol


def compute_rs(close: pd.Series, period: int = 63) -> pd.Series:
    """
    Raw RS score: percentage change over RS_PERIOD bars.
    RS Rank (0-100) is computed later across all stocks.
    """
    return close.pct_change(period) * 100


# =============================================================================
# PROCESS SINGLE TICKER
# =============================================================================

def process_ticker(ticker: str) -> pd.DataFrame | None:
    path = os.path.join(RAW_DIR, f"{ticker}.csv")
    if not os.path.exists(path):
        return None

    try:
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        df.index.name = "Date"

        # Require minimum history
        if len(df) < 100:
            return None

        # Standardise column names
        df.columns = [c.strip().title() for c in df.columns]

        # Must have OHLCV
        required = ["Open", "High", "Low", "Close", "Volume"]
        if not all(c in df.columns for c in required):
            return None

        df = df[required].copy()
        df = df.dropna(subset=["Close", "Volume"])

        # Compute indicators
        df["ATR"]     = compute_atr(df, ATR_PERIOD)
        df["RVOL"]    = compute_rvol(df, RVOL_PERIOD)
        df["RS_raw"]  = compute_rs(df["Close"], RS_PERIOD)

        return df

    except Exception as e:
        print(f"  Error processing {ticker}: {e}")
        return None


# =============================================================================
# COMPUTE RS RANK ACROSS ALL STOCKS (cross-sectional)
# =============================================================================

def compute_rs_ranks(processed: dict) -> dict:
    """
    For each date, rank all stocks by RS_raw score.
    RS_rank = percentile rank (0-100). Higher = stronger.
    """
    print("Computing cross-sectional RS ranks...")

    # Build a matrix: dates x tickers
    rs_matrix = pd.DataFrame({
        ticker: df["RS_raw"]
        for ticker, df in processed.items()
        if df is not None
    })

    # Rank each row (date) across all tickers
    rs_rank = rs_matrix.rank(axis=1, pct=True) * 100

    # Assign back
    for ticker in processed:
        if processed[ticker] is not None and ticker in rs_rank.columns:
            processed[ticker]["RS_rank"] = rs_rank[ticker]

    return processed


# =============================================================================
# MAIN
# =============================================================================

def run_processor() -> None:
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    tickers = [f.replace(".csv", "") for f in os.listdir(RAW_DIR) if f.endswith(".csv")]
    print(f"Processing {len(tickers)} tickers...")

    processed = {}
    failed    = 0

    for i, ticker in enumerate(tickers):
        df = process_ticker(ticker)
        if df is not None:
            processed[ticker] = df
        else:
            failed += 1

        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(tickers)} done...")

    print(f"  Processed: {len(processed)} | Failed: {failed}")

    # Compute RS rank cross-sectionally
    processed = compute_rs_ranks(processed)

    # Save to data/processed/
    print(f"Saving to {PROCESSED_DIR}/...")
    for ticker, df in processed.items():
        path = os.path.join(PROCESSED_DIR, f"{ticker}.csv")
        df.to_csv(path)

    print(f"\nDone. {len(processed)} files saved to {PROCESSED_DIR}/")


# =============================================================================
if __name__ == "__main__":
    run_processor()