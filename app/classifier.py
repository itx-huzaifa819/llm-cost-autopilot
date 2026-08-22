from pathlib import Path

import joblib
import numpy as np

from app.classifier.features import extract_features, features_to_vector

MODEL_PATH = Path(__file__).parent / "model.joblib"

_cached = None


def _load():
    global _cached
    if _cached is None and MODEL_PATH.exists():
        _cached = joblib.load(MODEL_PATH)
    return _cached


def _heuristic_tier(features: dict) -> int:
    
    if features["reasoning_keyword_count"] >= 1 or features["creative_marker_count"] >= 1:
        return 3
    if features["token_count"] > 40 or features["constraint_count"] >= 1:
        return 2
    return 1


def predict_tier(prompt: str, has_context: bool = False) -> tuple[int, dict]:
    
    features = extract_features(prompt, has_context=has_context)
    bundle = _load()

    if bundle is None:
        tier = _heuristic_tier(features)
        return tier, features

    vector = np.array([features_to_vector(features)], dtype=float)
    vector_scaled = bundle["scaler"].transform(vector)
    tier = int(bundle["model"].predict(vector_scaled)[0])
    return tier, features
