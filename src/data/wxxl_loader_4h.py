# =============================================================================
# WXXL-PAT | 4H Data Loader
# =============================================================================
# Purpose:
#   Download 4-hour OHLCV data for all tickers.
#   yfinance supports 4H for the last 2 years only.
#   Saves to data/raw_4h/
# =============================================================================

import yfinance as yf
import pandas as pd
import os
import time

# =============================================================================
# CONFIG
# =============================================================================
RAW_4H_DIR = "data/raw_4h"
INTERVAL   = "1h"    # yfinance doesn't have 4H — we download 1H and resample
PERIOD     = "730d"  # 2 years max for intraday
BATCH_SIZE = 20


# =============================================================================
# DOWNLOAD AND RESAMPLE TO 4H
# =============================================================================

def download_4h(ticker: str) -> pd.DataFrame | None:
    try:
        df = yf.download(
            ticker,
            period   = PERIOD,
            interval = INTERVAL,
            progress = False,
            auto_adjust = True,
        )

        if df is None or len(df) < 10:
            return None

        # Flatten multi-index if present
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df.columns = [c.strip().title() for c in df.columns]

        # Resample 1H to 4H
        df_4h = df.resample("4h").agg({
            "Open":   "first",
            "High":   "max",
            "Low":    "min",
            "Close":  "last",
            "Volume": "sum",
        }).dropna()

        return df_4h

    except Exception as e:
        print(f"  Error {ticker}: {e}")
        return None


# =============================================================================
# MAIN
# =============================================================================

def run_4h_download(tickers: list) -> None:
    os.makedirs(RAW_4H_DIR, exist_ok=True)

    existing = {f.replace(".csv", "") for f in os.listdir(RAW_4H_DIR)}
    tickers  = [t for t in tickers if t not in existing]
    print(f"Downloading 4H data for {len(tickers)} tickers...")

    saved  = 0
    failed = 0

    for i, ticker in enumerate(tickers):
        df = download_4h(ticker)

        if df is not None and len(df) > 50:
            path = os.path.join(RAW_4H_DIR, f"{ticker}.csv")
            df.to_csv(path)
            saved += 1
        else:
            failed += 1

        if (i + 1) % 20 == 0:
            print(f"  {i + 1}/{len(tickers)} | saved: {saved} | failed: {failed}")
            time.sleep(1)

    print(f"\nDone. {saved} tickers saved to {RAW_4H_DIR}/")


# =============================================================================
if __name__ == "__main__":
    # Download for live scan tickers first
    tickers = [
        "HAL", "MOH", "APA", "LUV", "LVS", "CTRA", "EXPD", "F", "FCX", "PCAR",
        "NEM", "FSLR", "MOS", "DVN", "BBY", "EQT", "HPQ", "INCY", "HES",
        "AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "JPM", "BAC", "XOM", "CVX"
    ]
    run_4h_download(tickers)