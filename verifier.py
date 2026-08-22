
import asyncio
import csv
import re
from pathlib import Path

from app.db import log_request, update_verification
from app.llm_providers import send_request
from app.models import LLMResponse, ModelConfig
from app.routing.router import escalation_threshold, verifier_reference_model

FAILURE_LOG_PATH = Path(__file__).parent.parent / "classifier" / "data" / "routing_failures.csv"

JUDGE_PROMPT_TEMPLATE = """You are grading whether two AI answers to the same question agree in \
substance and quality. Score from 0.0 (completely different/wrong) to 1.0 (equivalent quality \
and correctness). Respond with ONLY a number between 0.0 and 1.0.

Question:
{prompt}

Answer A (reference, from a top-tier model):
{reference_answer}

Answer B (to grade, from a cheaper model):
{candidate_answer}

Score (0.0-1.0):"""


def _parse_score(text: str) -> float:
    match = re.search(r"(\d+\.?\d*)", text)
    if not match:
        return 0.5  # neutral default if the judge output is unparseable
    score = float(match.group(1))
    return max(0.0, min(1.0, score))


async def _score_with_llm_judge(prompt: str, reference_answer: str, candidate_answer: str) -> float:
    judge_model = verifier_reference_model()
    judge_prompt = JUDGE_PROMPT_TEMPLATE.format(
        prompt=prompt, reference_answer=reference_answer, candidate_answer=candidate_answer
    )
    judge_response: LLMResponse = await send_request(judge_prompt, judge_model, max_tokens=10)
    return _parse_score(judge_response.text)


def _append_failure_example(prompt: str, tier: int):
    """Feeds a routing failure back into the classifier's training data."""
    FAILURE_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    is_new = not FAILURE_LOG_PATH.exists()
    with open(FAILURE_LOG_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if is_new:
            writer.writerow(["prompt", "tier", "has_context"])
        # A routing failure means the tier was too low - relabel one tier up.
        corrected_tier = min(3, tier + 1)
        writer.writerow([prompt, corrected_tier, 0])


async def verify_and_maybe_escalate(
    request_id: int,
    prompt: str,
    tier: int,
    candidate_response: LLMResponse,
) -> None:
   
    try:
        reference_model: ModelConfig = verifier_reference_model()

        if candidate_response.model == reference_model.name:
            # Already the top-tier model - nothing to verify against.
            update_verification(request_id, quality_score=1.0, escalated=False,
                                 escalated_to_model=None, escalation_cost_delta=0.0)
            return

        reference_response = await send_request(prompt, reference_model)
        quality_score = await _score_with_llm_judge(
            prompt, reference_response.text, candidate_response.text
        )

        threshold = escalation_threshold()
        escalated = quality_score < threshold
        cost_delta = 0.0

        if escalated:
            _append_failure_example(prompt, tier)
            # In a live system with an active connection you'd re-run the
            # request and push the escalated result back to the caller if
            # latency budget allows. Here we log the would-be cost delta.
            cost_delta = reference_response.cost - candidate_response.cost

        update_verification(
            request_id,
            quality_score=quality_score,
            escalated=escalated,
            escalated_to_model=reference_model.name if escalated else None,
            escalation_cost_delta=cost_delta,
        )
    except Exception as exc:  # noqa: BLE001 - verification must never take the app down
        print(f"[verifier] failed for request {request_id}: {exc}")


def schedule_verification(request_id: int, prompt: str, tier: int, candidate_response: LLMResponse):
    """Fire-and-forget the verification job onto the event loop."""
    asyncio.create_task(verify_and_maybe_escalate(request_id, prompt, tier, candidate_response))
