
from dataclasses import dataclass, field
from enum import Enum


class Provider(str, Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    OLLAMA = "ollama"


class QualityTier(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass(frozen=True)
class ModelConfig:
    name: str                     
    provider: Provider
    provider_model_id: str         
    cost_per_1m_input: float       
    cost_per_1m_output: float      
    quality_tier: QualityTier
    avg_latency_ms: int            
    context_window: int = 128_000

    def cost_for(self, input_tokens: int, output_tokens: int) -> float:
        return (
            input_tokens / 1_000_000 * self.cost_per_1m_input
            + output_tokens / 1_000_000 * self.cost_per_1m_output
        )


@dataclass
class LLMResponse:
    """Standardized response object every provider call is normalized into."""
    text: str
    model: str
    input_tokens: int
    output_tokens: int
    cost: float
    latency_ms: float
    raw: dict = field(default_factory=dict)



MODEL_REGISTRY: dict[str, ModelConfig] = {
    "gpt-4o": ModelConfig(
        name="gpt-4o",
        provider=Provider.OPENAI,
        provider_model_id="gpt-4o",
        cost_per_1m_input=2.50,
        cost_per_1m_output=10.00,
        quality_tier=QualityTier.HIGH,
        avg_latency_ms=1800,
        context_window=128_000,
    ),
    "gpt-4o-mini": ModelConfig(
        name="gpt-4o-mini",
        provider=Provider.OPENAI,
        provider_model_id="gpt-4o-mini",
        cost_per_1m_input=0.15,
        cost_per_1m_output=0.60,
        quality_tier=QualityTier.MEDIUM,
        avg_latency_ms=900,
        context_window=128_000,
    ),
    "claude-sonnet": ModelConfig(
        name="claude-sonnet",
        provider=Provider.ANTHROPIC,
        provider_model_id="claude-sonnet-4-6",
        cost_per_1m_input=3.00,
        cost_per_1m_output=15.00,
        quality_tier=QualityTier.HIGH,
        avg_latency_ms=1600,
        context_window=200_000,
    ),
    "claude-haiku": ModelConfig(
        name="claude-haiku",
        provider=Provider.ANTHROPIC,
        provider_model_id="claude-haiku-4-5",
        cost_per_1m_input=0.25,
        cost_per_1m_output=1.25,
        quality_tier=QualityTier.MEDIUM,
        avg_latency_ms=700,
        context_window=200_000,
    ),
    "llama3-local": ModelConfig(
        name="llama3-local",
        provider=Provider.OLLAMA,
        provider_model_id="llama3.1:8b",
        cost_per_1m_input=0.0,   
        cost_per_1m_output=0.0,
        quality_tier=QualityTier.LOW,
        avg_latency_ms=1200,
        context_window=128_000,
    ),
}


def get_model(name: str) -> ModelConfig:
    if name not in MODEL_REGISTRY:
        raise KeyError(f"Unknown model '{name}'. Known models: {list(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[name]


def most_expensive_model() -> ModelConfig:
    """Used as the 'what if we sent everything here' baseline for the dashboard."""
    return max(MODEL_REGISTRY.values(), key=lambda m: m.cost_per_1m_output)
