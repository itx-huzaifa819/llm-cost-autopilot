import os
import threading

import yaml

from app.classifier.classifier import predict_tier
from app.models import ModelConfig, get_model

_CONFIG_PATH = os.environ.get(
    "ROUTING_CONFIG_PATH",
    os.path.join(os.path.dirname(__file__), "routing_config.yaml"),
)

_lock = threading.Lock()
_config_cache: dict | None = None


def _load_config() -> dict:
    global _config_cache
    with _lock:
        with open(_CONFIG_PATH, encoding="utf-8") as f:
            _config_cache = yaml.safe_load(f)
        return _config_cache


def get_config() -> dict:
    if _config_cache is None:
        return _load_config()
    return _config_cache


def reload_config() -> dict:
    """Called by PUT /v1/routing-config after writing a new mapping to disk."""
    return _load_config()


def update_routing_map(new_map: dict[int, str]) -> dict:
    
    for model_name in new_map.values():
        get_model(model_name)  # raises if unknown

    config = get_config()
    config["routing_map"] = new_map
    with open(_CONFIG_PATH, "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False)
    return reload_config()


class RoutingDecision:
    def __init__(self, tier: int, model: ModelConfig, features: dict):
        self.tier = tier
        self.model = model
        self.features = features


def route(prompt: str, has_context: bool = False) -> RoutingDecision:
    
    tier, features = predict_tier(prompt, has_context=has_context)
    config = get_config()
    model_name = config["routing_map"][tier]
    model = get_model(model_name)
    return RoutingDecision(tier=tier, model=model, features=features)


def verifier_reference_model() -> ModelConfig:
    config = get_config()
    return get_model(config["verifier_reference_model"])


def escalation_threshold() -> float:
    config = get_config()
    return float(config["escalation_quality_threshold"])
