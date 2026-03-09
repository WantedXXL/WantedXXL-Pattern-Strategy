# =============================================================================
# WXXL-PAT | Multi-Timeframe Candlestick Visualiser
# =============================================================================
# Purpose:
#   3-panel chart: Weekly context + Daily pattern + 4H entry zone
#   Candlestick charts with C1, C2, neckline marked on all timeframes
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle
import os
import sys

sys.path.append(os.path.abspath("."))

# =============================================================================
# CONFIG
# =============================================================================
PROCESSED_DIR  = "data/processed"
RAW_DIR        = "data/raw"
RAW_4H_DIR     = "data/raw_4h"
PATTERNS_PATH  = "data/patterns/confirmed_patterns.csv"
OUTPUT_DIR     = "data/patterns/charts_mtf"

# Colors
BG       = '#0D1117'
BULL     = '#26A69A'   # teal green
BEAR     = '#EF5350'   # red
WICK     = '#8B949E'
C1_COL   = '#FF6B6B'
C2_COL   = '#51CF66'
NL_COL   = '#F0E68C'
GRID     = '#21262D'


# =============================================================================
# RESAMPLE DAILY TO WEEKLY
# =============================================================================

def to_weekly(df: pd.DataFrame) -> pd.DataFrame:
    return df.resample("W").agg({
        "Open":   "first",
        "High":   "max",
        "Low":    "min",
        "Close":  "last",
        "Volume": "sum",
    }).dropna()


# =============================================================================
# LOAD 4H DATA
# =============================================================================

def load_4h(ticker: str) -> pd.DataFrame | None:
    path = os.path.join(RAW_4H_DIR, f"{ticker}.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df.columns = [c.strip().title() for c in df.columns]
    return df


# =============================================================================
# DRAW CANDLESTICKS ON AXIS
# =============================================================================

def draw_candles(ax, df: pd.DataFrame, color_bull=BULL, color_bear=BEAR) -> None:
    """Draw candlestick chart on a matplotlib axis."""
    df = df.reset_index(drop=True)

    for i, row in df.iterrows():
        o, h, l, c = row["Open"], row["High"], row["Low"], row["Close"]
        color  = color_bull if c >= o else color_bear
        body_h = abs(c - o)
        body_y = min(o, c)

        # Body
        ax.add_patch(Rectangle(
            (i - 0.3, body_y), 0.6, max(body_h, 0.001),
            facecolor=color, edgecolor=color, linewidth=0.5, zorder=3
        ))

        # Wick
        ax.plot([i, i], [l, h], color=WICK, linewidth=0.7, zorder=2)

    ax.set_xlim(-1, len(df))
    ax.set_ylim(df["Low"].min() * 0.995, df["High"].max() * 1.005)


# =============================================================================
# FORMAT X AXIS WITH DATES
# =============================================================================

def format_xaxis(ax, dates, n_ticks=6):
    indices = np.linspace(0, len(dates) - 1, n_ticks, dtype=int)
    ax.set_xticks(indices)
    ax.set_xticklabels(
        [str(dates[i])[:10] for i in indices],
        rotation=30, ha='right', color='#8B949E', fontsize=7
    )


def style_ax(ax):
    ax.set_facecolor(BG)
    ax.tick_params(axis='y', colors='#8B949E', labelsize=7)
    ax.grid(color=GRID, linewidth=0.4, linestyle='-')
    for spine in ax.spines.values():
        spine.set_color('#30363D')


# =============================================================================
# MAIN PLOT FUNCTION
# =============================================================================

def plot_mtf(
    ticker:    str,
    c1_idx:    int,
    c2_idx:    int,
    c1_price:  float,
    c2_price:  float,
    neckline:  float,
    xgb_prob:  float = None,
    save_path: str   = None,
    show:      bool  = False,
) -> None:

    # ── Load data ──────────────────────────────────────────────────────────
    daily_path = os.path.join(PROCESSED_DIR, f"{ticker}.csv")
    if not os.path.exists(daily_path):
        print(f"No daily data for {ticker}")
        return

    daily = pd.read_csv(daily_path, index_col=0, parse_dates=True)
    daily.columns = [c.strip().title() for c in daily.columns]

    weekly = to_weekly(daily)
    h4     = load_4h(ticker)

    # ── Daily window: 60 bars before C1, 30 bars after C2 ─────────────────
    d_start = max(0, c1_idx - 60)
    d_end   = min(len(daily) - 1, c2_idx + 30)
    daily_slice = daily.iloc[d_start:d_end + 1]
    daily_dates = daily_slice.index

    c1_rel_d = c1_idx - d_start
    c2_rel_d = c2_idx - d_start

    # ── Weekly window: map C1/C2 dates to weekly ───────────────────────────
    c1_date = daily.index[c1_idx]
    c2_date = daily.index[c2_idx]

    w_end   = weekly.index.searchsorted(c2_date) + 8
    w_end   = min(w_end, len(weekly))
    w_start = max(0, w_end - 60)
    weekly_slice = weekly.iloc[w_start:w_end]
    weekly_dates = weekly_slice.index

    c1_rel_w = weekly_slice.index.searchsorted(c1_date)
    c2_rel_w = weekly_slice.index.searchsorted(c2_date)

    # ── 4H window: 20 days before C2 to 10 days after ─────────────────────
    h4_slice = None
    c1_rel_4h = None
    c2_rel_4h = None

    if h4 is not None:
        h4_start = pd.Timestamp(c2_date).tz_localize("UTC") - pd.Timedelta(days=20)
        h4_end   = pd.Timestamp(c2_date).tz_localize("UTC") + pd.Timedelta(days=10)
        h4_slice = h4[(h4.index >= h4_start) & (h4.index <= h4_end)]

        if len(h4_slice) > 5:
            c1_4h = h4_slice.index.searchsorted(pd.Timestamp(c1_date).tz_localize("UTC"))
            c2_4h = h4_slice.index.searchsorted(pd.Timestamp(c2_date).tz_localize("UTC"))
            c1_rel_4h = max(0, min(c1_4h, len(h4_slice) - 1))
            c2_rel_4h = max(0, min(c2_4h, len(h4_slice) - 1))

    # ── Figure ─────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(18, 8))
    fig.patch.set_facecolor(BG)

    if h4_slice is not None and len(h4_slice) > 5:
        ax_w, ax_d, ax_4h = fig.subplots(1, 3, gridspec_kw={"width_ratios": [1, 2, 1]})
    else:
        ax_w, ax_d = fig.subplots(1, 2, gridspec_kw={"width_ratios": [1, 2]})
        ax_4h = None

    # ── PANEL 1: Weekly ────────────────────────────────────────────────────
    draw_candles(ax_w, weekly_slice)
    ax_w.axhline(y=neckline, color=NL_COL, linewidth=1.0, linestyle='--', alpha=0.8)
    if 0 <= c1_rel_w < len(weekly_slice):
        ax_w.axvline(x=c1_rel_w, color=C1_COL, linewidth=1.0, alpha=0.6)
        ax_w.scatter(c1_rel_w, c1_price, color=C1_COL, s=80, zorder=5)
    if 0 <= c2_rel_w < len(weekly_slice):
        ax_w.axvline(x=c2_rel_w, color=C2_COL, linewidth=1.0, alpha=0.6)
        ax_w.scatter(c2_rel_w, c2_price, color=C2_COL, s=80, zorder=5)
    format_xaxis(ax_w, weekly_dates)
    style_ax(ax_w)
    ax_w.set_title("Weekly", color='#E6EDF3', fontsize=10, fontweight='bold')
    ax_w.set_ylabel("Price", color='#8B949E', fontsize=8)

    # ── PANEL 2: Daily ─────────────────────────────────────────────────────
    draw_candles(ax_d, daily_slice)
    ax_d.axhline(y=neckline, color=NL_COL, linewidth=1.2, linestyle='--',
                 alpha=0.9, label=f'Neckline: {neckline:.2f}')

    # C1 marker
    ax_d.scatter(c1_rel_d, c1_price, color=C1_COL, s=120, zorder=6)
    ax_d.annotate(f'C1\n{c1_price:.2f}', xy=(c1_rel_d, c1_price),
                  xytext=(c1_rel_d - 4, c1_price - (daily_slice["High"].max() - daily_slice["Low"].min()) * 0.09),
                  color=C1_COL, fontsize=8, fontweight='bold')

    # C2 marker
    ax_d.scatter(c2_rel_d, c2_price, color=C2_COL, s=120, zorder=6)
    ax_d.annotate(f'C2\n{c2_price:.2f}', xy=(c2_rel_d, c2_price),
                  xytext=(c2_rel_d + 1, c2_price - (daily_slice["High"].max() - daily_slice["Low"].min()) * 0.09),
                  color=C2_COL, fontsize=8, fontweight='bold')

    ax_d.axvspan(c1_rel_d, c2_rel_d, alpha=0.07, color='#58A6FF')
    format_xaxis(ax_d, daily_dates)
    style_ax(ax_d)
    ax_d.legend(facecolor='#161B22', edgecolor='#30363D', labelcolor='#E6EDF3', fontsize=8)

    title = f"WXXL-PAT | {ticker} | Daily"
    if xgb_prob is not None:
        tier = "FULL+" if xgb_prob >= 0.75 else "FULL" if xgb_prob >= 0.65 else "REDUCED" if xgb_prob >= 0.55 else "WATCH"
        color = '#51CF66' if xgb_prob >= 0.75 else '#94D82D' if xgb_prob >= 0.65 else '#FFD43B' if xgb_prob >= 0.55 else '#FF6B6B'
        ax_d.text(0.02, 0.97, f'{tier}  |  XGB: {xgb_prob:.1%}',
                  transform=ax_d.transAxes, color=color, fontsize=11,
                  fontweight='bold', verticalalignment='top',
                  bbox=dict(boxstyle='round', facecolor='#161B22', edgecolor=color, alpha=0.9))

    ax_d.set_title(title, color='#E6EDF3', fontsize=11, fontweight='bold')

    # ── PANEL 3: 4H ────────────────────────────────────────────────────────
    if ax_4h is not None and h4_slice is not None and len(h4_slice) > 5:
        draw_candles(ax_4h, h4_slice)
        ax_4h.axhline(y=neckline, color=NL_COL, linewidth=1.0, linestyle='--', alpha=0.8)
        if c2_rel_4h is not None:
            ax_4h.axvline(x=c2_rel_4h, color=C2_COL, linewidth=1.2, alpha=0.8, label='C2 zone')
            ax_4h.scatter(c2_rel_4h, c2_price, color=C2_COL, s=100, zorder=5)
        format_xaxis(ax_4h, h4_slice.index, n_ticks=4)
        style_ax(ax_4h)
        ax_4h.set_title("4H Entry Zone", color='#E6EDF3', fontsize=10, fontweight='bold')
        ax_4h.legend(facecolor='#161B22', edgecolor='#30363D', labelcolor='#E6EDF3', fontsize=8)

    plt.tight_layout(pad=1.5)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor=BG)
        print(f"Saved: {save_path}")

    if show:
        plt.show()

    plt.close()


# =============================================================================
# PLOT FROM CONFIRMED PATTERNS
# =============================================================================

def plot_ticker_patterns(ticker: str, max_patterns: int = 3) -> None:
    patterns = pd.read_csv(PATTERNS_PATH, parse_dates=["c1_date"])
    subset   = patterns[patterns["ticker"] == ticker].head(max_patterns)

    if len(subset) == 0:
        print(f"No confirmed patterns for {ticker}")
        return

    print(f"Plotting {len(subset)} patterns for {ticker}...")

    for _, row in subset.iterrows():
        save_path = os.path.join(OUTPUT_DIR, f"{ticker}_{int(row['c1_idx'])}_mtf.png")
        plot_mtf(
            ticker    = ticker,
            c1_idx    = int(row["c1_idx"]),
            c2_idx    = int(row["c2_idx"]),
            c1_price  = float(row["c1_price"]),
            c2_price  = float(row["c2_price"]),
            neckline  = float(row["neckline_price"]),
            save_path = save_path,
        )


# =============================================================================
if __name__ == "__main__":
    # Plot BBY live signal
    plot_mtf(
        ticker    = "BBY",
        c1_idx    = 4059,
        c2_idx    = 4067,
        c1_price  = 62.73,
        c2_price  = 61.59,
        neckline  = 70.41,
        xgb_prob  = 0.58,
        save_path = os.path.join(OUTPUT_DIR, "BBY_live_mtf.png"),
        show      = False,
    )
    print(f"Done. Open {OUTPUT_DIR}/")