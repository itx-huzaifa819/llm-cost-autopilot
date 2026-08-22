
import csv
import itertools
from pathlib import Path

OUT_PATH = Path(__file__).parent / "data" / "labeled_prompts.csv"

SUBJECTS = [
    "the Q3 sales report", "this customer email", "the attached invoice",
    "our onboarding docs", "this support ticket", "the meeting transcript",
    "this product spec", "the API changelog", "this resume", "the contract clause",
    "this survey data", "the incident report", "our pricing page", "this legal memo",
]

# --- Tier 1: simple reformatting / extraction / basic Q&A -------------------
TIER1_TEMPLATES = [
    "Extract the total dollar amount from {subject}.",
    "What is the date mentioned in {subject}?",
    "Convert {subject} into a bulleted list.",
    "Reformat {subject} as JSON with keys 'title' and 'body'.",
    "What is the sender's email address in {subject}?",
    "Capitalize the first letter of every sentence in {subject}.",
    "Extract all names mentioned in {subject}.",
    "What is the status field value in {subject}?",
    "Given {subject}, answer: what is the customer's name?",
    "Pull out the phone number from {subject}.",
    "Is {subject} marked as urgent? Answer yes or no.",
    "Given {subject}, what is the invoice number?",
    "Translate the greeting line of {subject} to lowercase.",
    "Count how many bullet points are in {subject}.",
]

# --- Tier 2: summarization / classification / structured analysis ----------
TIER2_TEMPLATES = [
    "Summarize {subject} in three sentences.",
    "Classify {subject} as urgent, normal, or low priority and explain briefly.",
    "Summarize the key action items from {subject}.",
    "Categorize {subject} into one of: billing, technical, general.",
    "Given {subject}, list the three most important takeaways.",
    "Summarize {subject} for a non-technical audience.",
    "What sentiment does {subject} express? Classify as positive, neutral, negative.",
    "Break {subject} down into a structured table of key fields.",
    "Summarize {subject} and flag any missing required fields.",
    "Given {subject}, identify the main request being made.",
    "Tag {subject} with relevant topics from a fixed taxonomy.",
    "Summarize {subject} in under 50 words, formatted as markdown bullet points.",
]

# --- Tier 3: multi-step reasoning / comparison / creative / judgment -------
TIER3_TEMPLATES = [
    "Analyze {subject} and compare it against last quarter's numbers, explaining the trade-offs.",
    "Given {subject}, design a step-by-step remediation plan and justify each step.",
    "Critique {subject} and propose three alternative approaches with pros and cons.",
    "Write a short creative story inspired by the themes in {subject}.",
    "Evaluate {subject} against our compliance requirements and explain your reasoning.",
    "Given {subject}, brainstorm five innovative solutions and rank them by feasibility.",
    "Analyze {subject} for logical inconsistencies and explain why each one matters.",
    "Given {subject}, architect a system design that addresses the underlying problem.",
    "Debate both sides of the decision implied in {subject}, then give a recommendation.",
    "Given {subject}, synthesize a strategy document with must-have and nice-to-have requirements.",
    "Explain the trade-offs between two possible interpretations of {subject}.",
    "Given {subject}, write a nuanced risk assessment considering at least three constraints.",
]

HAS_CONTEXT_VARIANTS = [True, False]


def build_rows() -> list[dict]:
    rows = []
    for tier, templates in ((1, TIER1_TEMPLATES), (2, TIER2_TEMPLATES), (3, TIER3_TEMPLATES)):
        for template, subject, has_context in itertools.product(
            templates, SUBJECTS, HAS_CONTEXT_VARIANTS
        ):
            prompt = template.format(subject=subject)
            rows.append({"prompt": prompt, "tier": tier, "has_context": int(has_context)})
    return rows


def main():
    rows = build_rows()
    # Subsample down to a manageable, still-diverse size (~200-300 rows)
    # while keeping tiers balanced.
    per_tier_cap = 90
    by_tier: dict[int, list[dict]] = {1: [], 2: [], 3: []}
    for row in rows:
        by_tier[row["tier"]].append(row)

    final_rows = []
    for tier, tier_rows in by_tier.items():
        step = max(1, len(tier_rows) // per_tier_cap)
        final_rows.extend(tier_rows[::step][:per_tier_cap])

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["prompt", "tier", "has_context"])
        writer.writeheader()
        writer.writerows(final_rows)

    print(f"Wrote {len(final_rows)} labeled examples to {OUT_PATH}")
    for tier in (1, 2, 3):
        count = sum(1 for r in final_rows if r["tier"] == tier)
        print(f"  Tier {tier}: {count} examples")


if __name__ == "__main__":
    main()
