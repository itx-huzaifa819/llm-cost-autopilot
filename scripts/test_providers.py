import asyncio
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dotenv import load_dotenv

load_dotenv()

from app.llm_providers import send_request  # noqa: E402
from app.models import MODEL_REGISTRY  # noqa: E402

TEST_PROMPTS = [
    "What is the capital of France?",
    "Summarize the plot of Romeo and Juliet in two sentences.",
    "Extract the year from this sentence: The company was founded in 2014.",
    "Classify this as positive or negative: 'The product arrived broken.'",
    "Write a haiku about autumn.",
    "Compare REST and GraphQL in three bullet points.",
    "Convert 'hello world' to uppercase.",
    "What are the pros and cons of remote work?",
    "Given a budget of $500, suggest a laptop.",
    "Explain recursion to a 10-year-old.",
]


async def test_model(name: str):
    cfg = MODEL_REGISTRY[name]
    print(f"\n=== {name} ({cfg.provider.value}) ===")
    for i, prompt in enumerate(TEST_PROMPTS, 1):
        try:
            resp = await send_request(prompt, cfg, max_tokens=200)
            print(
                f"  [{i:02d}] {resp.latency_ms:6.0f}ms  "
                f"${resp.cost:.6f}  "
                f"({resp.input_tokens}in/{resp.output_tokens}out)  "
                f"-> {resp.text[:60].replace(chr(10), ' ')}..."
            )
        except Exception as exc:  # noqa: BLE001
            print(f"  [{i:02d}] FAILED: {exc}")


async def main():
    only = sys.argv[1] if len(sys.argv) > 1 else None
    models = [only] if only else list(MODEL_REGISTRY)
    for name in models:
        if name not in MODEL_REGISTRY:
            print(f"Unknown model '{name}', skipping.")
            continue
        await test_model(name)


if __name__ == "__main__":
    asyncio.run(main())
