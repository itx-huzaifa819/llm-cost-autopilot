import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.classifier.features import extract_features
from app.models import MODEL_REGISTRY, get_model, most_expensive_model
from app.routing.router import get_config, route


def test_registry_has_all_providers():
    providers = {cfg.provider.value for cfg in MODEL_REGISTRY.values()}
    assert {"openai", "anthropic", "ollama"} <= providers


def test_cost_calculation():
    cfg = get_model("gpt-4o-mini")
    cost = cfg.cost_for(input_tokens=1_000_000, output_tokens=1_000_000)
    assert cost == cfg.cost_per_1m_input + cfg.cost_per_1m_output


def test_most_expensive_model_is_a_high_tier_model():
    cfg = most_expensive_model()
    assert cfg.quality_tier.value == "high"


def test_simple_prompt_features():
    features = extract_features("What is the date in this text?")
    assert features["token_count"] > 0
    assert features["reasoning_keyword_count"] == 0


def test_complex_prompt_features_score_higher():
    simple = extract_features("What is the total amount?")
    complex_ = extract_features(
        "Analyze the trade-offs between two architectures and compare them, "
        "explaining the reasoning step by step."
    )
    assert complex_["reasoning_keyword_count"] > simple["reasoning_keyword_count"]


def test_routing_config_loads():
    config = get_config()
    assert set(config["routing_map"].keys()) == {1, 2, 3}
    for model_name in config["routing_map"].values():
        get_model(model_name)  # should not raise


def test_route_returns_valid_model():
    decision = route("Extract the invoice number from this document.")
    assert decision.tier in (1, 2, 3)
    assert decision.model.name in MODEL_REGISTRY
