# =============================================================================
# WXXL-PAT | Layer 2c — Shapelet Classifier
# =============================================================================
# Purpose:
#   Binary classifier that votes YES/NO on whether a PIP sequence
#   is a double bottom. Uses a different mathematical approach from DTW —
#   finds discriminative subsequences (shapelets) that separate
#   double bottoms from noise.
#
#   Both DTW AND Shapelet must vote YES for the committee to pass.
#   This eliminates different categories of false positives that
#   neither method catches alone.
#
# Training:
#   - Positives: PIP sequences from confirmed double bottoms
#   - Negatives: PIP sequences from the candidate scanner that failed
#     visual inspection
#   On first run we use rule-based logic as a proxy until we have
#   enough labeled data to train properly.
# =============================================================================

import numpy as np
import pandas as pd
from typing import Optional
import os
import pickle


# =============================================================================
# RULE-BASED SHAPELET PROXY
# =============================================================================
# Until we have 200+ labeled patterns to train on, we use geometric rules
# that approximate what a trained shapelet classifier would learn.
# These rules are replaced by the trained model in Sprint 6.

def rule_based_vote(pip_prices: list) -> dict:
    """
    Geometric rules that approximate shapelet classification.
    Assumes standard PIP order: P0=start, P1=C1, P2=mid, P3=neckline, P4=mid, P5=C2, P6=end
    """
    result = {
        "vote": False,
        "confidence": 0.0,
        "reason": ""
    }

    if len(pip_prices) < 7:
        result["reason"] = "insufficient_pips"
        return result

    p = pip_prices

    # Trust PIP positions directly
    c1_price       = p[1]
    neckline_price = max(p[2], p[3], p[4])  # neckline is highest point in middle
    c2_price       = p[5]

    # Rule 1: Both caves must be below neckline
    if c1_price >= neckline_price or c2_price >= neckline_price:
        result["reason"] = "caves_above_neckline"
        return result

    # Rule 2: Pattern must have meaningful range
    price_range = max(p) - min(p)
    if price_range < 0.03 * max(p):
        result["reason"] = "price_range_too_small"
        return result

    # Rule 3: Neckline must be at least 5% above caves
    neckline_height_c1 = (neckline_price - c1_price) / c1_price
    neckline_height_c2 = (neckline_price - c2_price) / c2_price
    if neckline_height_c1 < 0.05 or neckline_height_c2 < 0.05:
        result["reason"] = "neckline_too_shallow"
        return result

    # Rule 4: Cave symmetry — C2 within 15% of C1
    cave_diff = abs(c2_price - c1_price) / c1_price
    if cave_diff > 0.15:
        result["reason"] = f"caves_asymmetric: {cave_diff:.1%}"
        return result

    # Rule 5: Start price higher than C1 (prior downtrend)
    if p[0] <= c1_price * 1.02:
        result["reason"] = "no_downtrend_before_c1"
        return result

    # Confidence
    symmetry_score = 1.0 - (cave_diff / 0.15)
    neckline_score = min(neckline_height_c1 / 0.20, 1.0)
    confidence     = round((symmetry_score + neckline_score) / 2, 4)

    result["vote"]       = True
    result["confidence"] = confidence
    return result


# =============================================================================
# TRAINED MODEL (Sprint 6 — loaded if exists)
# =============================================================================

MODEL_PATH = "models/shapelet/shapelet_classifier.pkl"

def load_trained_model():
    """Load trained shapelet model if it exists."""
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, "rb") as f:
            return pickle.load(f)
    return None


def trained_model_vote(pip_prices: list, model) -> dict:
    """Use trained shapelet model to vote."""
    result = {"vote": False, "confidence": 0.0, "reason": ""}
    try:
        x    = np.array(pip_prices).reshape(1, -1)
        prob = model.predict_proba(x)[0][1]
        result["confidence"] = round(float(prob), 4)
        result["vote"]       = prob >= 0.5
        if not result["vote"]:
            result["reason"] = f"model_confidence_low: {prob:.2%}"
    except Exception as e:
        result["reason"] = f"model_error: {e}"
    return result


# =============================================================================
# MAIN VOTE FUNCTION
# =============================================================================

def shapelet_vote(pip_prices: list) -> dict:
    """
    Cast a shapelet vote on a PIP sequence.
    Uses trained model if available, otherwise rule-based proxy.

    Parameters
    ----------
    pip_prices : list   7 PIP prices from the extractor

    Returns
    -------
    dict with keys:
        vote       : bool   — True = double bottom
        confidence : float  — 0.0 to 1.0
        method     : str    — 'trained' or 'rule_based'
        reason     : str    — Why it failed (empty if passed)
    """
    model = load_trained_model()

    if model is not None:
        result         = trained_model_vote(pip_prices, model)
        result["method"] = "trained"
    else:
        result         = rule_based_vote(pip_prices)
        result["method"] = "rule_based"

    return result


# =============================================================================
# QUICK TEST
# =============================================================================
if __name__ == "__main__":

    perfect = [90.0, 80.0, 87.0, 95.0, 87.0, 81.0, 88.0]
    noisy   = [88.0, 79.5, 86.0, 94.0, 85.0, 82.0, 89.0]
    bad     = [70.0, 75.0, 80.0, 85.0, 90.0, 95.0, 100.0]
    uptrend = [80.0, 82.0, 85.0, 88.0, 90.0, 93.0, 97.0]

    print("=== Shapelet Classifier Test ===")

    for label, pips in [("Perfect", perfect), ("Noisy", noisy), ("Bad", bad), ("Uptrend", uptrend)]:
        result = shapelet_vote(pips)
        print(f"\n{label}:")
        print(f"  Vote       : {result['vote']}")
        print(f"  Confidence : {result['confidence']}")
        print(f"  Method     : {result['method']}")
        print(f"  Reason     : {result['reason'] or 'N/A'}")