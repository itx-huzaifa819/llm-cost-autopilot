
import os
import time

import httpx
from anthropic import AsyncAnthropic
from openai import AsyncOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from app.models import LLMResponse, ModelConfig, Provider

_openai_client: AsyncOpenAI | None = None
_anthropic_client: AsyncAnthropic | None = None


def _get_openai() -> AsyncOpenAI:
    global _openai_client
    if _openai_client is None:
        _openai_client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    return _openai_client


def _get_anthropic() -> AsyncAnthropic:
    global _anthropic_client
    if _anthropic_client is None:
        _anthropic_client = AsyncAnthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
    return _anthropic_client


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=8))
async def _call_openai(prompt: str, cfg: ModelConfig, max_tokens: int) -> tuple[str, int, int]:
    client = _get_openai()
    resp = await client.chat.completions.create(
        model=cfg.provider_model_id,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
    )
    text = resp.choices[0].message.content or ""
    return text, resp.usage.prompt_tokens, resp.usage.completion_tokens


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=8))
async def _call_anthropic(prompt: str, cfg: ModelConfig, max_tokens: int) -> tuple[str, int, int]:
    client = _get_anthropic()
    resp = await client.messages.create(
        model=cfg.provider_model_id,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
    )
    text = "".join(block.text for block in resp.content if block.type == "text")
    return text, resp.usage.input_tokens, resp.usage.output_tokens


@retry(stop=stop_after_attempt(2), wait=wait_exponential(multiplier=1, min=1, max=4))
async def _call_ollama(prompt: str, cfg: ModelConfig, max_tokens: int) -> tuple[str, int, int]:
    base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
    async with httpx.AsyncClient(timeout=120) as client:
        resp = await client.post(
            f"{base_url}/api/generate",
            json={
                "model": cfg.provider_model_id,
                "prompt": prompt,
                "stream": False,
                "options": {"num_predict": max_tokens},
            },
        )
        resp.raise_for_status()
        data = resp.json()
    text = data.get("response", "")
    # Ollama reports token counts as eval_count / prompt_eval_count
    input_tokens = data.get("prompt_eval_count", 0)
    output_tokens = data.get("eval_count", 0)
    return text, input_tokens, output_tokens


_DISPATCH = {
    Provider.OPENAI: _call_openai,
    Provider.ANTHROPIC: _call_anthropic,
    Provider.OLLAMA: _call_ollama,
}


async def send_request(prompt: str, model_config: ModelConfig, max_tokens: int = 1024) -> LLMResponse:
    """
    The single entry point every caller in this codebase uses.
    Same signature, same return type, no matter which provider is behind it.
    """
    handler = _DISPATCH[model_config.provider]
    start = time.perf_counter()
    text, input_tokens, output_tokens = await handler(prompt, model_config, max_tokens)
    latency_ms = (time.perf_counter() - start) * 1000

    cost = model_config.cost_for(input_tokens, output_tokens)

    return LLMResponse(
        text=text,
        model=model_config.name,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cost=cost,
        latency_ms=latency_ms,
        raw={"provider": model_config.provider.value},
    )
