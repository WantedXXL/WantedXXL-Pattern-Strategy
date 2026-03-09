# =============================================================================
# WXXL-PAT | Multi-Timeframe Visualiser (Independent Markers)
# =============================================================================
# Purpose:
#   3-panel candlestick chart where each timeframe has its own independently
#   detected C1, C2, and neckline — not projected from daily.
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import os
import sys

sys.path.append(os.path.abspath("."))

from src.detection.wxxl_mtf_detector import (
    detect_all_timeframes,
    resample_to_weekly,
    load_4h,
)
from src.detection.wxxl_confluence import find_best_confluence, confluence_to_features

# =============================================================================
# CONFIG
# =============================================================================
PROCESSED_DIR = "data/processed"
OUTPUT_DIR    = "data/patterns/charts_mtf"

BG      = '#0D1117'
BULL    = '#26A69A'
BEAR    = '#EF5350'
WICK    = '#8B949E'
C1_COL  = '#FF6B6B'
C2_COL  = '#51CF66'
NL_COL  = '#F0E68C'
GRID    = '#21262D'


# =============================================================================
# DRAW CANDLESTICKS
# =============================================================================

def draw_candles(ax, df: pd.DataFrame) -> None:
    df = df.reset_index(drop=True)
    for i, row in df.iterrows():
        o, h, l, c = row["Open"], row["High"], row["Low"], row["Close"]
        color  = BULL if c >= o else BEAR
        body_h = max(abs(c - o), 0.001)
        body_y = min(o, c)
        ax.add_patch(Rectangle(
            (i - 0.3, body_y), 0.6, body_h,
            facecolor=color, edgecolor=color, linewidth=0.5, zorder=3
        ))
        ax.plot([i, i], [l, h], color=WICK, linewidth=0.7, zorder=2)
    ax.set_xlim(-1, len(df))
    price_range = df["High"].max() - df["Low"].min()
    ax.set_ylim(df["Low"].min() - price_range * 0.02,
                df["High"].max() + price_range * 0.02)


def format_xaxis(ax, dates, n_ticks=6):
    indices = np.linspace(0, len(dates) - 1, n_ticks, dtype=int)
    ax.set_xticks(indices)
    ax.set_xticklabels(
        [str(dates[i])[:10] for i in indices],
        rotation=30, ha='right', color='#8B949E', fontsize=7
    )


def style_ax(ax, title: str):
    ax.set_facecolor(BG)
    ax.tick_params(axis='y', colors='#8B949E', labelsize=7)
    ax.grid(color=GRID, linewidth=0.4)
    for spine in ax.spines.values():
        spine.set_color('#30363D')
    ax.set_title(title, color='#E6EDF3', fontsize=10, fontweight='bold', pad=8)


# =============================================================================
# MARK PATTERN ON AXIS
# =============================================================================

def mark_pattern(ax, df_slice, pattern: dict, label_prefix: str = "") -> None:
    """Mark C1, C2, neckline on an axis given a pattern dict."""
    if pattern is None:
        return

    dates   = df_slice.index
    prices  = df_slice["Close"].reset_index(drop=True)
    n       = len(df_slice)

    # Find relative positions by date
    c1_date = pd.Timestamp(pattern["c1_date"])
    c2_date = pd.Timestamp(pattern["c2_date"])

    c1_rel = dates.searchsorted(c1_date)
    c2_rel = dates.searchsorted(c2_date)

    c1_rel = max(0, min(c1_rel, n - 1))
    c2_rel = max(0, min(c2_rel, n - 1))

    c1_price = pattern["c1_price"]
    c2_price = pattern["c2_price"]
    neckline = pattern["neckline"]

    price_range = df_slice["High"].max() - df_slice["Low"].min()
    offset      = price_range * 0.06

    # Neckline
    ax.axhline(y=neckline, color=NL_COL, linewidth=1.1,
               linestyle='--', alpha=0.85, label=f'NL {neckline:.2f}')

    # Pattern zone
    if c1_rel < c2_rel:
        ax.axvspan(c1_rel, c2_rel, alpha=0.07, color='#58A6FF')

    # C1
    ax.scatter(c1_rel, c1_price, color=C1_COL, s=100, zorder=6)
    ax.annotate(
        f'C1\n{c1_price:.2f}',
        xy=(c1_rel, c1_price),
        xytext=(c1_rel - 1, c1_price - offset),
        color=C1_COL, fontsize=7, fontweight='bold',
        ha='right',
    )

    # C2
    ax.scatter(c2_rel, c2_price, color=C2_COL, s=100, zorder=6)
    ax.annotate(
        f'C2\n{c2_price:.2f}',
        xy=(c2_rel, c2_price),
        xytext=(c2_rel + 1, c2_price - offset),
        color=C2_COL, fontsize=7, fontweight='bold',
        ha='left',
    )

    ax.legend(facecolor='#161B22', edgecolor='#30363D',
              labelcolor='#E6EDF3', fontsize=7, loc='upper left')


# =============================================================================
# MAIN PLOT
# =============================================================================

def plot_mtf_confluence(
    ticker:     str,
    confluence: dict,
    daily:      pd.DataFrame,
    xgb_prob:   float = None,
    save_path:  str   = None,
    show:       bool  = False,
) -> None:

    weekly = resample_to_weekly(daily)
    h4     = load_4h(ticker)
    if h4 is not None and h4.index.tz is not None:
        h4.index = h4.index.tz_localize(None)

    d_pat = confluence.get("daily")
    w_pat = confluence.get("weekly")
    h_pat = confluence.get("h4")

    # ── Determine display windows ──────────────────────────────────────────

    def get_window(df, pattern, bars_before=60, bars_after=30):
        if pattern is None or df is None:
            return df
        dates   = df.index
        c2_date = pd.Timestamp(pattern["c2_date"])
        c2_pos  = dates.searchsorted(c2_date)
        start   = max(0, c2_pos - bars_before)
        end     = min(len(df) - 1, c2_pos + bars_after)
        return df.iloc[start:end + 1]

    daily_slice  = get_window(daily,  d_pat, bars_before=80,  bars_after=30)
    weekly_slice = get_window(weekly, w_pat, bars_before=40,  bars_after=10)
    h4_slice     = get_window(h4,     h_pat, bars_before=120, bars_after=40) if h4 is not None else None

    # ── Figure ─────────────────────────────────────────────────────────────
    has_h4 = h4_slice is not None and len(h4_slice) > 5 and h_pat is not None

    if has_h4:
        fig, (ax_w, ax_d, ax_4h) = plt.subplots(
            1, 3, figsize=(20, 8),
            gridspec_kw={"width_ratios": [1, 2, 1]}
        )
    else:
        fig, (ax_w, ax_d) = plt.subplots(
            1, 2, figsize=(16, 8),
            gridspec_kw={"width_ratios": [1, 2]}
        )
        ax_4h = None

    fig.patch.set_facecolor(BG)

    # ── Score box ──────────────────────────────────────────────────────────
    mtf_score = confluence.get("score", 0)
    tfs       = confluence.get("tfs_confirmed", 1)

    score_parts = []
    if xgb_prob is not None:
        tier  = "FULL+" if xgb_prob >= 0.75 else "FULL" if xgb_prob >= 0.65 else "REDUCED" if xgb_prob >= 0.55 else "WATCH"
        color = '#51CF66' if xgb_prob >= 0.75 else '#94D82D' if xgb_prob >= 0.65 else '#FFD43B' if xgb_prob >= 0.55 else '#FF6B6B'
        score_parts.append(f'{tier}  XGB:{xgb_prob:.1%}')
    else:
        color = '#FFD43B'

    score_parts.append(f'MTF:{mtf_score:.0%}  TFs:{tfs}/3')
    score_text = '  |  '.join(score_parts)

    # ── PANEL 1: Weekly ────────────────────────────────────────────────────
    if weekly_slice is not None and len(weekly_slice) > 3:
        draw_candles(ax_w, weekly_slice)
        mark_pattern(ax_w, weekly_slice, w_pat)
        format_xaxis(ax_w, weekly_slice.index)
        style_ax(ax_w, "Weekly")
        ax_w.set_ylabel("Price", color='#8B949E', fontsize=8)
    else:
        ax_w.set_facecolor(BG)
        ax_w.text(0.5, 0.5, 'No weekly pattern', transform=ax_w.transAxes,
                  color='#8B949E', ha='center', va='center')
        style_ax(ax_w, "Weekly")

    # ── PANEL 2: Daily ─────────────────────────────────────────────────────
    if daily_slice is not None and len(daily_slice) > 3:
        draw_candles(ax_d, daily_slice)
        mark_pattern(ax_d, daily_slice, d_pat)
        format_xaxis(ax_d, daily_slice.index)
        style_ax(ax_d, f"WXXL-PAT | {ticker} | Daily")
        ax_d.text(
            0.02, 0.97, score_text,
            transform=ax_d.transAxes,
            color=color, fontsize=10, fontweight='bold',
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='#161B22', edgecolor=color, alpha=0.9)
        )
    else:
        style_ax(ax_d, f"{ticker} | Daily")

    # ── PANEL 3: 4H ────────────────────────────────────────────────────────
    if ax_4h is not None and h4_slice is not None and len(h4_slice) > 5:
        draw_candles(ax_4h, h4_slice)
        mark_pattern(ax_4h, h4_slice, h_pat)
        format_xaxis(ax_4h, h4_slice.index, n_ticks=4)
        style_ax(ax_4h, "4H Entry Zone")
    elif ax_4h is not None:
        ax_4h.set_facecolor(BG)
        ax_4h.text(0.5, 0.5, 'No 4H pattern', transform=ax_4h.transAxes,
                   color='#8B949E', ha='center', va='center')
        style_ax(ax_4h, "4H Entry Zone")

    plt.tight_layout(pad=1.5)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor=BG)
        print(f"Saved: {save_path}")

    if show:
        plt.show()

    plt.close()


# =============================================================================
# CONVENIENCE RUNNER
# =============================================================================

def run_mtf_chart(ticker: str, xgb_prob: float = None) -> None:
    path = os.path.join(PROCESSED_DIR, f"{ticker}.csv")
    if not os.path.exists(path):
        print(f"No data for {ticker}")
        return

    daily = pd.read_csv(path, index_col=0, parse_dates=True)
    daily.columns = [c.strip().title() for c in daily.columns]

    print(f"Detecting MTF patterns for {ticker}...")
    mtf        = detect_all_timeframes(ticker, daily)
    confluence = find_best_confluence(mtf["weekly"], mtf["daily"], mtf["h4"])

    print(f"Confluence score: {confluence['score']:.2f} | "
          f"TFs confirmed: {confluence['tfs_confirmed']} | "
          f"Aligned: {confluence['aligned']}")

    save_path = os.path.join(OUTPUT_DIR, f"{ticker}_mtf_confluence.png")
    plot_mtf_confluence(
        ticker     = ticker,
        confluence = confluence,
        daily      = daily,
        xgb_prob   = xgb_prob,
        save_path  = save_path,
    )


# =============================================================================
if __name__ == "__main__":
    run_mtf_chart("BBY", xgb_prob=0.58)