import argparse
import asyncio
import random
import sys
import time

import httpx

SIMPLE_PROMPTS = [
    "Extract the total amount from: Invoice #4521, Total: $1,240.00",
    "What is the date in this text: Meeting scheduled for March 3rd, 2026.",
    "Convert this to a bulleted list: apples, bananas, oranges, grapes.",
    "Is this urgent? 'Please review whenever you get a chance, no rush.'",
    "What is the sender's name: 'Best regards, John from Accounting.'",
]

MODERATE_PROMPTS = [
    "Summarize this in two sentences: Our Q3 revenue grew 12% year over year, "
    "driven primarily by expansion in the EU market, though churn ticked up slightly.",
    "Classify this ticket as billing, technical, or general: 'My card was charged twice.'",
    "List the three main takeaways from a typical product launch retrospective.",
    "Tag this feedback with a sentiment: 'The app is fine but the onboarding is confusing.'",
]

COMPLEX_PROMPTS = [
    "Analyze the trade-offs between microservices and a monolith for a 5-person startup, "
    "and recommend one with justification.",
    "Design a step-by-step incident response plan for a database outage affecting billing.",
    "Critique this pricing strategy and propose two alternatives: flat $99/month for everyone.",
    "Write a short creative story about a robot learning to paint.",
    "Given competing deadlines from two stakeholders, reason through how to prioritize and why.",
]

ALL_PROMPTS = (
    [(p, False) for p in SIMPLE_PROMPTS]
    + [(p, False) for p in MODERATE_PROMPTS]
    + [(p, False) for p in COMPLEX_PROMPTS]
)


async def send_one(client: httpx.AsyncClient, base_url: str, prompt: str, has_context: bool):
    try:
        resp = await client.post(
            f"{base_url}/v1/completions",
            json={"prompt": prompt, "has_context": has_context, "verify": True},
            timeout=60,
        )
        resp.raise_for_status()
        return resp.json()
    except Exception as exc:  # noqa: BLE001
        return {"error": str(exc)}


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=500)
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--concurrency", type=int, default=10)
    args = parser.parse_args()

    sem = asyncio.Semaphore(args.concurrency)
    start = time.perf_counter()
    errors = 0
    tier_counts = {1: 0, 2: 0, 3: 0}

    async with httpx.AsyncClient() as client:
        async def bounded_send(prompt, has_context):
            nonlocal errors
            async with sem:
                result = await send_one(client, args.url, prompt, has_context)
                if "error" in result:
                    errors += 1
                else:
                    tier_counts[result["routing"]["complexity_tier"]] += 1
                return result

        tasks = []
        for _ in range(args.count):
            prompt, has_context = random.choice(ALL_PROMPTS)
            tasks.append(bounded_send(prompt, has_context))

        results = await asyncio.gather(*tasks)

    elapsed = time.perf_counter() - start
    print(f"\nSent {args.count} requests in {elapsed:.1f}s "
          f"({args.count / elapsed:.1f} req/s, {errors} errors)")
    print(f"Tier distribution: {tier_counts}")

    async with httpx.AsyncClient() as client:
        stats_resp = await client.get(f"{args.url}/v1/stats")
        stats = stats_resp.json()

    print("\n=== Cost Savings Report ===")
    print(f"Total requests logged:     {stats['total_requests']}")
    print(f"Total cost:                ${stats['total_cost_usd']:.4f}")
    print(f"Baseline (all-{stats['baseline_model']}) cost: "
          f"${stats['hypothetical_cost_if_all_on_most_expensive_model_usd']:.4f}")
    print(f"COST SAVINGS:              {stats['cost_savings_pct']:.1f}%")
    print(f"Avg quality score:         {stats['avg_quality_score']:.2f}")
    print(f"Escalation rate:           {stats['escalation_rate_pct']:.1f}%")
    print("\nRouting distribution:")
    for row in stats["routing_distribution"]:
        print(f"  {row['model']:<15} {row['count']:>5} requests   ${row['cost_usd']:.4f}")


if __name__ == "__main__":
    asyncio.run(main())
