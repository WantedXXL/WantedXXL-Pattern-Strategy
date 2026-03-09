# =============================================================================
# WXXL-PAT | Live Scanner
# =============================================================================
# Purpose:
#   Run the full pipeline on current market data.
#   For each stock in the universe, detect double bottom candidates,
#   score them with XGBoost, and visualise the top signals.
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os
import sys
import warnings
warnings.filterwarnings("ignore")

sys.path.append(os.path.abspath("."))

import xgboost as xgb
from src.detection.wxxl_downtrend_gate   import check_downtrend_gate
from src.detection.wxxl_committee        import committee_vote
from src.features.wxxl_feature_extractor import extract_features

# =============================================================================
# CONFIG
# =============================================================================
PROCESSED_DIR  = "data/processed"
MODEL_PATH     = "models/xgb/wxxl_xgb_v3.json"
OUTPUT_DIR     = "data/patterns/live_signals"
LOOKBACK       = 60
MIN_DECLINE    = 0.12
LOW_WINDOW     = 40
LOW_PROXIMITY  = 0.03
STEP           = 3
SCORE_THRESHOLD = 0.50   # Minimum XGBoost probability to report

EXCLUDE_COLS = [
    "ticker", "c1_date", "c1_idx", "c2_idx",
    "label", "outcome", "bars_to_breakout", "max_gain", "max_loss"
]

# Use top 20 tickers by confirmed patterns for the live scan
SCAN_TICKERS = [
    "HAL", "MOH", "APA", "LUV", "LVS", "CTRA", "EXPD", "F", "FCX", "PCAR",
    "NEM", "FSLR", "MOS", "DVN", "BBY", "EQT", "HPQ", "INCY", "APA", "HES"
]


# =============================================================================
# LOAD MODEL
# =============================================================================

def load_model():
    model = xgb.XGBClassifier()
    model.load_model(MODEL_PATH)
    return model


# =============================================================================
# SCAN SINGLE TICKER
# =============================================================================

def scan_ticker_live(ticker: str, df: pd.DataFrame, model) -> list:
    """
    Scan a single ticker for current double bottom signals.
    Only scans the last 300 bars — recent patterns only.
    """
    prices    = df["Close"].reset_index(drop=True)
    dates     = df.index
    signals   = []

    # Only scan recent bars
    scan_start = max(LOOKBACK + 10, len(prices) - 300)

    for i in range(scan_start, len(prices) - 5, STEP):
        # Layer 1: Downtrend gate
        gate = check_downtrend_gate(
            prices            = prices,
            c1_idx            = i,
            lookback          = LOOKBACK,
            min_decline_pct   = MIN_DECLINE,
            low_window        = LOW_WINDOW,
            low_proximity_pct = LOW_PROXIMITY,
        )
        if not gate["passed"]:
            continue

        # Layer 2: Committee
        result = committee_vote(prices, i)
        if not result["passed"]:
            continue

        c2_idx = result["c2_idx"]
        if c2_idx <= i or c2_idx >= len(prices):
            continue

        # Layer 3: Feature extraction + XGBoost score
        try:
            feats = extract_features(
                df             = df,
                c1_idx         = i,
                c2_idx         = c2_idx,
                c1_price       = float(prices.iloc[i]),
                c2_price       = float(result["c2_price"]),
                neckline_price = float(result["neckline_price"]),
            )

            if not feats:
                continue

            feat_df = pd.DataFrame([feats])
            feat_cols = [c for c in feat_df.columns if c not in EXCLUDE_COLS]
            feat_df = feat_df[feat_cols].fillna(0)

            # Align columns with model
            model_cols = model.get_booster().feature_names
            for col in model_cols:
                if col not in feat_df.columns:
                    feat_df[col] = 0.0
            feat_df = feat_df[model_cols]

            prob = model.predict_proba(feat_df)[0][1]

            if prob >= SCORE_THRESHOLD:
                signals.append({
                    "ticker":         ticker,
                    "c1_idx":         i,
                    "c1_date":        dates[i] if i < len(dates) else None,
                    "c2_idx":         c2_idx,
                    "c2_date":        dates[c2_idx] if c2_idx < len(dates) else None,
                    "c1_price":       round(float(prices.iloc[i]), 4),
                    "c2_price":       round(float(result["c2_price"]), 4),
                    "neckline":       round(float(result["neckline_price"]), 4),
                    "xgb_prob":       round(float(prob), 4),
                    "dtw_distance":   result["dtw_distance"],
                    "prior_decline":  gate["prior_decline"],
                })

        except Exception as e:
            continue

    return signals


# =============================================================================
# VISUALISE SIGNAL
# =============================================================================

def plot_signal(signal: dict, df: pd.DataFrame) -> None:
    ticker   = signal["ticker"]
    c1_idx   = signal["c1_idx"]
    c2_idx   = signal["c2_idx"]
    c1_price = signal["c1_price"]
    c2_price = signal["c2_price"]
    neckline = signal["neckline"]
    prob     = signal["xgb_prob"]

    prices = df["Close"].reset_index(drop=True)
    dates  = df.index

    display_start = max(0, c1_idx - 40)
    display_end   = min(len(prices) - 1, c2_idx + 40)

    price_slice = prices.iloc[display_start:display_end + 1]
    date_slice  = dates[display_start:display_end + 1]
    x_range     = range(len(price_slice))

    c1_rel = c1_idx - display_start
    c2_rel = c2_idx - display_start

    # Score colour
    if prob >= 0.75:
        score_color = '#51CF66'   # green
        tier = 'FULL+'
    elif prob >= 0.65:
        score_color = '#94D82D'
        tier = 'FULL'
    elif prob >= 0.55:
        score_color = '#FFD43B'
        tier = 'REDUCED'
    else:
        score_color = '#FF6B6B'
        tier = 'WATCH'

    fig, ax = plt.subplots(figsize=(14, 7))
    fig.patch.set_facecolor('#0D1117')
    ax.set_facecolor('#0D1117')

    # Price line
    ax.plot(x_range, price_slice.values, color='#58A6FF', linewidth=1.5, zorder=2)

    # Neckline
    ax.axhline(y=neckline, color='#F0E68C', linewidth=1.2,
               linestyle='--', alpha=0.9, label=f'Neckline: {neckline:.2f}')

    # Cave 1
    ax.scatter(c1_rel, c1_price, color='#FF6B6B', s=150, zorder=5)
    ax.annotate(f'C1\n{c1_price:.2f}', xy=(c1_rel, c1_price),
                xytext=(c1_rel - 3, c1_price - (price_slice.max() - price_slice.min()) * 0.10),
                color='#FF6B6B', fontsize=9, fontweight='bold')

    # Cave 2
    ax.scatter(c2_rel, c2_price, color='#51CF66', s=150, zorder=5)
    ax.annotate(f'C2\n{c2_price:.2f}', xy=(c2_rel, c2_price),
                xytext=(c2_rel + 1, c2_price - (price_slice.max() - price_slice.min()) * 0.10),
                color='#51CF66', fontsize=9, fontweight='bold')

    # Pattern zone
    ax.axvspan(c1_rel, c2_rel, alpha=0.08, color='#58A6FF')

    # Score box
    ax.text(0.02, 0.97,
            f'{tier}  |  XGBoost: {prob:.1%}',
            transform=ax.transAxes,
            color=score_color, fontsize=13, fontweight='bold',
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='#161B22', edgecolor=score_color, alpha=0.9))

    # X axis
    tick_positions = list(range(0, len(date_slice), max(1, len(date_slice) // 8)))
    tick_labels    = [str(date_slice[i])[:10] for i in tick_positions]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=30, ha='right', color='#8B949E', fontsize=8)
    ax.tick_params(axis='y', colors='#8B949E')

    ax.grid(color='#21262D', linewidth=0.5)
    ax.spines['bottom'].set_color('#30363D')
    ax.spines['top'].set_color('#30363D')
    ax.spines['left'].set_color('#30363D')
    ax.spines['right'].set_color('#30363D')

    ax.set_title(f'WXXL-PAT LIVE SIGNAL | {ticker} | Prior decline: {signal["prior_decline"]:.1%}',
                 color='#E6EDF3', fontsize=13, fontweight='bold', pad=15)
    ax.set_ylabel('Price', color='#8B949E', fontsize=10)
    ax.legend(facecolor='#161B22', edgecolor='#30363D', labelcolor='#E6EDF3', fontsize=9)

    plt.tight_layout()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    save_path = os.path.join(OUTPUT_DIR, f"{ticker}_c1{c1_idx}_prob{int(prob*100)}.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='#0D1117')
    plt.close()
    print(f"  Chart saved: {save_path}")


# =============================================================================
# MAIN
# =============================================================================

def run_live_scanner() -> None:
    print("=" * 60)
    print("WXXL-PAT | Live Scanner")
    print("=" * 60)

    model      = load_model()
    all_signals = []

    for ticker in SCAN_TICKERS:
        path = os.path.join(PROCESSED_DIR, f"{ticker}.csv")
        if not os.path.exists(path):
            print(f"  {ticker}: no data")
            continue

        try:
            df = pd.read_csv(path, index_col=0, parse_dates=True)
            df.columns = [c.strip().title() for c in df.columns]

            signals = scan_ticker_live(ticker, df, model)

            if signals:
                print(f"  {ticker}: {len(signals)} signal(s) found")
                all_signals.extend(signals)
                for sig in signals:
                    plot_signal(sig, df)
            else:
                print(f"  {ticker}: no signals")

        except Exception as e:
            print(f"  {ticker}: error — {e}")

    print(f"\n{'='*60}")
    print(f"Scan complete. Total signals: {len(all_signals)}")

    if all_signals:
        df_signals = pd.DataFrame(all_signals).sort_values("xgb_prob", ascending=False)
        print(f"\nTop signals:")
        print(df_signals[["ticker", "c2_date", "c1_price", "c2_price",
                          "neckline", "xgb_prob", "prior_decline"]].to_string(index=False))

        out_path = os.path.join(OUTPUT_DIR, "live_signals.csv")
        df_signals.to_csv(out_path, index=False)
        print(f"\nSaved to: {out_path}")

    print(f"{'='*60}")


# =============================================================================
if __name__ == "__main__":
    run_live_scanner()