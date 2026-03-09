# =============================================================================
# WXXL-PAT | Data Loader
# =============================================================================
# Purpose:
#   Download 15 years of daily OHLCV data for all 1,047 US equities
#   (S&P 500, Russell 2000, NASDAQ) and save to data/raw/
# =============================================================================

import yfinance as yf
import pandas as pd
import os
import time
from datetime import datetime

# =============================================================================
# CONFIG
# =============================================================================
START_DATE = "2010-01-01"
END_DATE   = datetime.today().strftime("%Y-%m-%d")
RAW_DIR    = "data/raw"
BATCH_SIZE = 50  # Download 50 stocks at a time

# =============================================================================
# UNIVERSE — S&P 500 + Russell 2000 + NASDAQ (1,047 stocks)
# We start with S&P 500 only for the first test run (503 stocks)
# =============================================================================

def get_sp500_tickers() -> list:
    """Get S&P 500 tickers via yfinance."""
    import json, urllib.request
    url = "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/main/data/constituents.csv"
    df = pd.read_csv(url)
    tickers = df["Symbol"].str.replace(".", "-", regex=False).tolist()
    print(f"S&P 500 tickers loaded: {len(tickers)}")
    return tickers


def download_batch(tickers: list, start: str, end: str) -> pd.DataFrame:
    """Download a batch of tickers using yfinance."""
    data = yf.download(
        tickers,
        start=start,
        end=end,
        interval="1d",
        group_by="ticker",
        auto_adjust=True,
        progress=False,
    )
    return data


def save_ticker(ticker: str, df: pd.DataFrame) -> None:
    """Save a single ticker's OHLCV to data/raw/<TICKER>.csv"""
    path = os.path.join(RAW_DIR, f"{ticker}.csv")
    df.to_csv(path)


def run_download(tickers: list) -> None:
    os.makedirs(RAW_DIR, exist_ok=True)

    # Skip already downloaded
    existing = {f.replace(".csv", "") for f in os.listdir(RAW_DIR)}
    tickers  = [t for t in tickers if t not in existing]
    print(f"Tickers to download: {len(tickers)} (skipping {len(existing)} already saved)")

    total_batches = (len(tickers) + BATCH_SIZE - 1) // BATCH_SIZE

    for i in range(0, len(tickers), BATCH_SIZE):
        batch   = tickers[i:i + BATCH_SIZE]
        batch_n = i // BATCH_SIZE + 1
        print(f"Batch {batch_n}/{total_batches}: {batch[0]} ... {batch[-1]}")

        try:
            data = download_batch(batch, START_DATE, END_DATE)

            for ticker in batch:
                try:
                    if len(batch) == 1:
                        df = data
                    else:
                        df = data[ticker]

                    df = df.dropna(how="all")
                    if len(df) > 100:
                        save_ticker(ticker, df)
                except Exception as e:
                    print(f"  Skipping {ticker}: {e}")

        except Exception as e:
            print(f"  Batch failed: {e}")

        time.sleep(1)  # polite delay between batches

    saved = len(os.listdir(RAW_DIR))
    print(f"\nDone. {saved} files saved to {RAW_DIR}/")


# =============================================================================
if __name__ == "__main__":
    tickers = get_sp500_tickers()
    run_download(tickers)