# =============================================================================
# WXXL-PAT | Pattern Visualiser
# =============================================================================
# Purpose:
#   Plot confirmed double bottom patterns with C1, C2, neckline and
#   PIP points marked on the chart. Visual QA tool.
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os
import sys

sys.path.append(os.path.abspath("."))

from src.detection.wxxl_pip_extractor import extract_double_bottom_pips

# =============================================================================
# CONFIG
# =============================================================================
PROCESSED_DIR = "data/processed"
PATTERNS_PATH = "data/patterns/confirmed_patterns.csv"
OUTPUT_DIR    = "data/patterns/charts"


# =============================================================================
# PLOT SINGLE PATTERN
# =============================================================================

def plot_pattern(
    ticker:    str,
    c1_idx:    int,
    c2_idx:    int,
    c1_price:  float,
    c2_price:  float,
    neckline:  float,
    prices:    pd.Series,
    dates:     pd.Index,
    save_path: str = None,
    show:      bool = True,
) -> None:

    # Window to display: 30 bars before C1, 30 bars after C2
    display_start = max(0, c1_idx - 30)
    display_end   = min(len(prices) - 1, c2_idx + 30)

    price_slice = prices.iloc[display_start:display_end + 1]
    date_slice  = dates[display_start:display_end + 1]
    x_range     = range(len(price_slice))

    # Convert c1_idx and c2_idx to slice-relative positions
    c1_rel = c1_idx - display_start
    c2_rel = c2_idx - display_start

    fig, ax = plt.subplots(figsize=(14, 6))
    fig.patch.set_facecolor('#0D1117')
    ax.set_facecolor('#0D1117')

    # Price line
    ax.plot(x_range, price_slice.values, color='#58A6FF', linewidth=1.5, zorder=2)

    # Neckline
    ax.axhline(y=neckline, color='#F0E68C', linewidth=1.2,
               linestyle='--', alpha=0.8, label=f'Neckline: {neckline:.2f}')

    # Cave 1
    ax.scatter(c1_rel, c1_price, color='#FF6B6B', s=120, zorder=5,
               label=f'Cave 1: {c1_price:.2f}')
    ax.annotate('C1', xy=(c1_rel, c1_price),
                xytext=(c1_rel - 2, c1_price - (price_slice.max() - price_slice.min()) * 0.08),
                color='#FF6B6B', fontsize=10, fontweight='bold')

    # Cave 2
    ax.scatter(c2_rel, c2_price, color='#51CF66', s=120, zorder=5,
               label=f'Cave 2: {c2_price:.2f}')
    ax.annotate('C2', xy=(c2_rel, c2_price),
                xytext=(c2_rel + 1, c2_price - (price_slice.max() - price_slice.min()) * 0.08),
                color='#51CF66', fontsize=10, fontweight='bold')

    # Shade the pattern zone
    ax.axvspan(c1_rel, c2_rel, alpha=0.08, color='#58A6FF')

    # X axis labels — show every 10th date
    tick_positions = list(range(0, len(date_slice), max(1, len(date_slice) // 8)))
    tick_labels    = [str(date_slice[i])[:10] for i in tick_positions]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=30, ha='right', color='#8B949E', fontsize=8)
    ax.tick_params(axis='y', colors='#8B949E')

    # Grid
    ax.grid(color='#21262D', linewidth=0.5, linestyle='-')
    ax.spines['bottom'].set_color('#30363D')
    ax.spines['top'].set_color('#30363D')
    ax.spines['left'].set_color('#30363D')
    ax.spines['right'].set_color('#30363D')

    # Labels
    ax.set_title(f'WXXL-PAT | {ticker} | Double Bottom Pattern',
                 color='#E6EDF3', fontsize=14, fontweight='bold', pad=15)
    ax.set_ylabel('Price', color='#8B949E', fontsize=10)
    ax.legend(facecolor='#161B22', edgecolor='#30363D',
              labelcolor='#E6EDF3', fontsize=9)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight',
                    facecolor='#0D1117')
        print(f"Saved: {save_path}")

    if show:
        plt.show()

    plt.close()


# =============================================================================
# PLOT MULTIPLE PATTERNS FOR A TICKER
# =============================================================================

def plot_ticker_patterns(ticker: str, max_patterns: int = 5) -> None:
    patterns = pd.read_csv(PATTERNS_PATH, parse_dates=["c1_date"])
    subset   = patterns[patterns["ticker"] == ticker].head(max_patterns)

    if len(subset) == 0:
        print(f"No confirmed patterns found for {ticker}")
        return

    path = os.path.join(PROCESSED_DIR, f"{ticker}.csv")
    df   = pd.read_csv(path, index_col=0, parse_dates=True)
    df.columns = [c.strip().title() for c in df.columns]
    prices = df["Close"]
    dates  = df.index

    print(f"Plotting {len(subset)} patterns for {ticker}...")

    for i, row in subset.iterrows():
        save_path = os.path.join(OUTPUT_DIR, f"{ticker}_{int(row['c1_idx'])}.png")
        plot_pattern(
            ticker    = ticker,
            c1_idx    = int(row["c1_idx"]),
            c2_idx    = int(row["c2_idx"]),
            c1_price  = float(row["c1_price"]),
            c2_price  = float(row["c2_price"]),
            neckline  = float(row["neckline_price"]),
            prices    = prices,
            dates     = dates,
            save_path = save_path,
            show      = False,
        )

    print(f"Done. Charts saved to {OUTPUT_DIR}/")


# =============================================================================
if __name__ == "__main__":
    # Plot top 5 patterns for HAL — highest confirmed count
    plot_ticker_patterns("HAL", max_patterns=5)
    print("Open data/patterns/charts/ to see the pattern images.")