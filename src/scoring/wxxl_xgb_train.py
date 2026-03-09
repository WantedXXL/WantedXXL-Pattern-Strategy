# =============================================================================
# WXXL-PAT | XGBoost Trainer
# =============================================================================
# Purpose:
#   Train an XGBoost classifier to predict which confirmed double bottom
#   patterns will break the neckline (label=1) vs fail (label=0).
#
# Validation:
#   Walk-forward expanding window — no future data ever touches training.
#   3 folds minimum. AUC > 0.60 is the go/no-go threshold.
#
# SHAP:
#   Top features must be structurally meaningful.
#   If any contaminated feature appears — investigate immediately.
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
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.preprocessing import LabelEncoder
import shap

# =============================================================================
# CONFIG
# =============================================================================
FEATURES_PATH = "data/features/labeled_features_mtf.csv"
MODEL_DIR     = "models/xgb"
MIN_TRAIN_ROWS = 200
N_FOLDS        = 5

XGB_PARAMS = {
    "n_estimators":     300,
    "max_depth":        4,
    "learning_rate":    0.05,
    "subsample":        0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 10,
    "gamma":            1.0,
    "reg_alpha":        0.1,
    "reg_lambda":       1.0,
    "scale_pos_weight": 2.0,   # handles class imbalance (2:1 neg:pos)
    "eval_metric":      "auc",
    "random_state":     42,
    "tree_method":      "hist",
}

# Features to exclude from training
EXCLUDE_COLS = [
    "ticker", "c1_date", "c1_idx", "c2_idx",
    "label", "outcome", "bars_to_breakout", "max_gain", "max_loss"
]


# =============================================================================
# LOAD AND PREPARE DATA
# =============================================================================

def load_data() -> tuple:
    df = pd.read_csv(FEATURES_PATH, parse_dates=["c1_date"])
    df = df.sort_values("c1_date").reset_index(drop=True)

    feature_cols = [c for c in df.columns if c not in EXCLUDE_COLS]
    X = df[feature_cols].fillna(0)
    y = df["label"]

    print(f"Dataset: {len(df)} rows | {len(feature_cols)} features")
    print(f"Label distribution: {y.value_counts().to_dict()}")
    print(f"Date range: {df['c1_date'].min().date()} → {df['c1_date'].max().date()}")

    return df, X, y, feature_cols


# =============================================================================
# WALK-FORWARD VALIDATION
# =============================================================================

def walk_forward_validation(df: pd.DataFrame, X: pd.DataFrame, y: pd.Series) -> list:
    """
    Expanding window walk-forward validation.
    Each fold trains on all data up to a cutoff, tests on the next window.
    """
    print(f"\nWalk-forward validation ({N_FOLDS} folds)...")

    n          = len(df)
    fold_size  = n // (N_FOLDS + 1)
    results    = []

    for fold in range(N_FOLDS):
        train_end  = fold_size * (fold + 2)
        test_start = train_end
        test_end   = min(test_start + fold_size, n)

        if train_end < MIN_TRAIN_ROWS or test_start >= n:
            continue

        X_train = X.iloc[:train_end]
        y_train = y.iloc[:train_end]
        X_test  = X.iloc[test_start:test_end]
        y_test  = y.iloc[test_start:test_end]

        if len(X_test) < 20:
            continue

        model = xgb.XGBClassifier(**XGB_PARAMS)
        model.fit(X_train, y_train, verbose=False)

        probs  = model.predict_proba(X_test)[:, 1]
        auc    = roc_auc_score(y_test, probs)

        # Top decile analysis
        threshold    = np.percentile(probs, 90)
        top_decile   = y_test[probs >= threshold]
        top_success  = top_decile.mean() if len(top_decile) > 0 else 0

        train_date = df["c1_date"].iloc[train_end - 1].date()
        test_date  = df["c1_date"].iloc[test_end - 1].date()

        print(f"  Fold {fold + 1}: train→{train_date} | test→{test_date} | "
              f"AUC={auc:.3f} | top decile={top_success:.1%} (n={len(top_decile)})")

        results.append({
            "fold":         fold + 1,
            "auc":          auc,
            "top_decile":   top_success,
            "test_size":    len(X_test),
        })

    mean_auc      = np.mean([r["auc"] for r in results])
    mean_top      = np.mean([r["top_decile"] for r in results])
    print(f"\n  Mean AUC       : {mean_auc:.3f}")
    print(f"  Mean top decile: {mean_top:.1%}")

    return results


# =============================================================================
# TRAIN FINAL MODEL
# =============================================================================

def train_final_model(X: pd.DataFrame, y: pd.Series, feature_cols: list):
    """Train on full dataset and save."""
    print("\nTraining final model on full dataset...")

    model = xgb.XGBClassifier(**XGB_PARAMS)
    model.fit(X, y, verbose=False)

    os.makedirs(MODEL_DIR, exist_ok=True)
    model_path = os.path.join(MODEL_DIR, "wxxl_xgb_v4.json")
    model.save_model(model_path)
    print(f"Model saved: {model_path}")

    return model


# =============================================================================
# SHAP ANALYSIS
# =============================================================================

def run_shap_analysis(model, X: pd.DataFrame, feature_cols: list) -> None:
    print("\nSHAP Analysis...")

    explainer  = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    # Mean absolute SHAP values
    mean_shap = pd.DataFrame({
        "feature":    feature_cols,
        "importance": np.abs(shap_values).mean(axis=0)
    }).sort_values("importance", ascending=False)

    print("\nTop 15 features by SHAP importance:")
    print(mean_shap.head(15).to_string(index=False))

    # Save importance
    importance_path = os.path.join(MODEL_DIR, "shap_importance.csv")
    mean_shap.to_csv(importance_path, index=False)
    print(f"\nSHAP importance saved: {importance_path}")

    # Sanity check
    top5 = mean_shap.head(5)["feature"].tolist()
    print(f"\nTop 5 features: {top5}")

    bad_features = ["yolo_conf", "pattern_duration", "bars_to_breakout", "max_gain", "max_loss"]
    contaminated = [f for f in top5 if any(b in f for b in bad_features)]
    if contaminated:
        print(f"WARNING: Potentially contaminated features in top 5: {contaminated}")
    else:
        print("Sanity check PASSED — no contaminated features in top 5")


# =============================================================================
# MAIN
# =============================================================================

def run_training() -> None:
    print("=" * 60)
    print("WXXL-PAT | XGBoost Training")
    print("=" * 60)

    df, X, y, feature_cols = load_data()

    # Walk-forward validation
    results = walk_forward_validation(df, X, y)

    mean_auc = np.mean([r["auc"] for r in results])
    if mean_auc < 0.55:
        print(f"\nWARNING: Mean AUC {mean_auc:.3f} is below 0.55 threshold.")
        print("Review features and labels before proceeding.")
    else:
        print(f"\nAUC {mean_auc:.3f} — proceeding to final model training.")

    # Train final model
    model = train_final_model(X, y, feature_cols)

    # SHAP analysis
    run_shap_analysis(model, X, feature_cols)

    print("\n" + "=" * 60)
    print("Training complete.")
    print("=" * 60)


# =============================================================================
if __name__ == "__main__":
    run_training()