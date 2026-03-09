# =============================================================================
# WXXL-PAT | Daily Evening Scanner
# =============================================================================
# Purpose:
#   Run every evening after market close.
#   Scans all 497 stocks, scores each candidate with XGBoost + MTF
#   confluence + Bayesian monitor simulation. Outputs ranked watchlist
#   and generates MTF charts for top signals.
#
# Usage:
#   python wxxl_daily_scan.py
#
# Output:
#   data/signals/daily_watchlist_YYYYMMDD.csv
#   data/signals/charts/  — MTF charts for top signals
# =============================================================================

import pandas as pd
import numpy as np
import os
import sys
import warnings
from datetime import datetime
warnings.filterwarnings("ignore")

sys.path.append(os.path.abspath("."))

import xgboost as xgb

from src.detection.wxxl_downtrend_gate   import check_downtrend_gate
from src.detection.wxxl_committee        import committee_vote
from src.features.wxxl_feature_extractor import extract_features
from src.detection.wxxl_mtf_detector     import detect_all_timeframes
from src.detection.wxxl_confluence       import find_best_confluence, confluence_to_features
from src.monitor.wxxl_bayesian_monitor   import simulate_monitor
from src.utils.wxxl_visualiser_mtf       import plot_mtf_confluence

# =============================================================================
# CONFIG
# =============================================================================
PROCESSED_DIR   = "data/processed"
MODEL_PATH      = "models/xgb/wxxl_xgb_v4.json"
OUTPUT_DIR      = "data/signals"
CHARTS_DIR      = "data/signals/charts"

LOOKBACK        = 60
MIN_DECLINE     = 0.12
LOW_WINDOW      = 40
LOW_PROXIMITY   = 0.03
STEP            = 3
SCAN_TAIL       = 300      # Only scan last 300 bars per stock

XGB_MIN         = 0.45     # Minimum XGBoost score to proceed to MTF
FINAL_MIN       = 0.50     # Minimum combined score to appear in watchlist
TOP_N_CHARTS    = 10       # Generate MTF charts for top N signals

EXCLUDE_COLS = [
    "ticker", "c1_date", "c1_idx", "c2_idx",
    "label", "outcome", "bars_to_breakout", "max_gain", "max_loss"
]

# Signal tiers
TIERS = {
    "FULL+":   0.75,
    "FULL":    0.65,
    "REDUCED": 0.55,
    "WATCH":   0.45,
}


# =============================================================================
# LOAD UNIVERSE
# =============================================================================

def load_universe() -> list:
    tickers = []
    for f in os.listdir(PROCESSED_DIR):
        if f.endswith(".csv"):
            tickers.append(f.replace(".csv", ""))
    return sorted(tickers)


# =============================================================================
# SCORE TIER
# =============================================================================

def get_tier(score: float) -> str:
    if score >= TIERS["FULL+"]:
        return "FULL+"
    elif score >= TIERS["FULL"]:
        return "FULL"
    elif score >= TIERS["REDUCED"]:
        return "REDUCED"
    elif score >= TIERS["WATCH"]:
        return "WATCH"
    return "REJECT"


# =============================================================================
# COMBINED SCORE
# =============================================================================

def combined_score(xgb_prob: float, mtf_score: float, bayes_prob: float) -> float:
    """
    Weighted combination of three signal components.
    XGBoost: 50% — primary pattern quality signal
    Bayesian: 30% — live momentum confirmation
    MTF:      20% — timeframe confluence
    """
    return round(0.50 * xgb_prob + 0.30 * bayes_prob + 0.20 * mtf_score, 4)


# =============================================================================
# SCAN SINGLE TICKER
# =============================================================================

def scan_ticker(ticker: str, df: pd.DataFrame, model) -> list:
    prices  = df["Close"].reset_index(drop=True)
    dates   = df.index
    signals = []

    scan_start = max(LOOKBACK + 10, len(prices) - SCAN_TAIL)

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

        # Layer 3: Feature extraction
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

            feat_df   = pd.DataFrame([feats])
            feat_cols = [c for c in feat_df.columns if c not in EXCLUDE_COLS]
            feat_df   = feat_df[feat_cols].fillna(0)

            model_cols = model.get_booster().feature_names
            for col in model_cols:
                if col not in feat_df.columns:
                    feat_df[col] = 0.0
            feat_df   = feat_df[model_cols]

            xgb_prob = float(model.predict_proba(feat_df)[0][1])

        except Exception:
            continue

        if xgb_prob < XGB_MIN:
            continue

        # Layer 4: MTF confluence
        try:
            mtf_results = detect_all_timeframes(ticker, df)
            confluence  = find_best_confluence(
                mtf_results["weekly"],
                mtf_results["daily"],
                mtf_results["h4"],
            )
            mtf_score   = confluence["score"]
            mtf_tfs     = confluence["tfs_confirmed"]
            mtf_aligned = confluence["aligned"]
        except Exception:
            mtf_score   = 0.33
            mtf_tfs     = 1
            mtf_aligned = False
            confluence  = {"weekly": None, "daily": None, "h4": None,
                          "score": 0.33, "tfs_confirmed": 1, "aligned": False}

        # Layer 5: Bayesian monitor simulation on last 10 bars after C2
        try:
            bayes_result = simulate_monitor(
                df        = df,
                c2_idx    = c2_idx,
                c2_price  = float(result["c2_price"]),
                c1_price  = float(prices.iloc[i]),
                neckline  = float(result["neckline_price"]),
                prior     = xgb_prob,
                ticker    = ticker,
            )
            bayes_prob  = bayes_result["final_prob"]
            bayes_fired = bayes_result["fired"]
            bayes_bar   = bayes_result["fire_bar"]
        except Exception:
            bayes_prob  = xgb_prob
            bayes_fired = False
            bayes_bar   = None

        # Combined score
        score = combined_score(xgb_prob, mtf_score, bayes_prob)

        if score < FINAL_MIN:
            continue

        signals.append({
            "ticker":        ticker,
            "c1_idx":        i,
            "c2_idx":        c2_idx,
            "c1_date":       str(dates[i])[:10] if i < len(dates) else "",
            "c2_date":       str(dates[c2_idx])[:10] if c2_idx < len(dates) else "",
            "c1_price":      round(float(prices.iloc[i]), 2),
            "c2_price":      round(float(result["c2_price"]), 2),
            "neckline":      round(float(result["neckline_price"]), 2),
            "xgb_prob":      round(xgb_prob, 4),
            "mtf_score":     round(mtf_score, 4),
            "mtf_tfs":       mtf_tfs,
            "mtf_aligned":   mtf_aligned,
            "bayes_prob":    round(bayes_prob, 4),
            "bayes_fired":   bayes_fired,
            "bayes_fire_bar":bayes_bar,
            "combined_score":score,
            "tier":          get_tier(score),
            "prior_decline": round(gate["prior_decline"], 4),
            "_confluence":   confluence,
            "_df":           df,
        })

    return signals


# =============================================================================
# MAIN
# =============================================================================

def run_daily_scan() -> None:
    today = datetime.now().strftime("%Y%m%d")
    print("=" * 65)
    print(f"WXXL-PAT | Daily Scanner | {today}")
    print("=" * 65)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(CHARTS_DIR, exist_ok=True)

    # Load model
    model = xgb.XGBClassifier()
    model.load_model(MODEL_PATH)
    print(f"Model loaded: {MODEL_PATH}")

    # Load universe
    universe = load_universe()
    print(f"Universe: {len(universe)} stocks\n")

    all_signals = []
    scanned     = 0
    errors      = 0

    for ticker in universe:
        path = os.path.join(PROCESSED_DIR, f"{ticker}.csv")
        try:
            df = pd.read_csv(path, index_col=0, parse_dates=True)
            df.columns = [c.strip().title() for c in df.columns]

            signals = scan_ticker(ticker, df, model)

            if signals:
                print(f"  {ticker}: {len(signals)} signal(s) — "
                      f"best={signals[0]['combined_score']:.3f} [{signals[0]['tier']}]")
                all_signals.extend(signals)

            scanned += 1

        except Exception as e:
            errors += 1

        if scanned % 50 == 0 and scanned > 0:
            print(f"  [{scanned}/{len(universe)}] scanned | {len(all_signals)} signals so far...")

    # Sort by combined score
    all_signals.sort(key=lambda x: x["combined_score"], reverse=True)

    print(f"\n{'='*65}")
    print(f"Scan complete.")
    print(f"Stocks scanned : {scanned}")
    print(f"Errors         : {errors}")
    print(f"Total signals  : {len(all_signals)}")

    if not all_signals:
        print("No signals found today.")
        return

    # Save watchlist — exclude internal fields
    save_cols = [k for k in all_signals[0].keys() if not k.startswith("_")]
    df_out    = pd.DataFrame([{k: s[k] for k in save_cols} for s in all_signals])
    out_path  = os.path.join(OUTPUT_DIR, f"watchlist_{today}.csv")
    df_out.to_csv(out_path, index=False)

    print(f"\nTop signals:")
    display_cols = ["ticker", "c2_date", "c1_price", "c2_price",
                    "neckline", "xgb_prob", "mtf_score", "bayes_prob",
                    "combined_score", "tier"]
    print(df_out[display_cols].head(15).to_string(index=False))
    print(f"\nWatchlist saved: {out_path}")

    # Generate MTF charts for top N
    print(f"\nGenerating MTF charts for top {TOP_N_CHARTS} signals...")
    for sig in all_signals[:TOP_N_CHARTS]:
        try:
            save_path = os.path.join(
                CHARTS_DIR,
                f"{sig['ticker']}_{sig['c2_date']}_score{int(sig['combined_score']*100)}.png"
            )
            plot_mtf_confluence(
                ticker     = sig["ticker"],
                confluence = sig["_confluence"],
                daily      = sig["_df"],
                xgb_prob   = sig["xgb_prob"],
                save_path  = save_path,
            )
        except Exception as e:
            print(f"  Chart error {sig['ticker']}: {e}")

    print(f"\nDone. Charts saved to {CHARTS_DIR}/")
    print("=" * 65)


# =============================================================================
if __name__ == "__main__":
    run_daily_scan()