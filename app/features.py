import re

REASONING_KEYWORDS = [
    "analyze", "analyse", "compare", "evaluate", "critique", "design",
    "architect", "strategy", "reasoning", "why", "explain the trade-offs",
    "pros and cons", "step by step", "debate", "synthesize",
]

CONSTRAINT_MARKERS = [
    "must", "should", "at least", "at most", "no more than", "exactly",
    "only", "within", "constraint", "requirement", "format:", "limit",
]

CREATIVE_MARKERS = [
    "write a story", "poem", "creative", "brainstorm", "imagine",
    "generate ideas", "invent",
]

OUTPUT_FORMAT_MARKERS = [
    "json", "table", "csv", "yaml", "markdown", "bullet points",
    "numbered list", "xml", "schema",
]


def _count_hits(text_lower: str, terms: list[str]) -> int:
    return sum(1 for t in terms if t in text_lower)


def extract_features(prompt: str, has_context: bool = False) -> dict:
    
    text_lower = prompt.lower()
    words = prompt.split()

    return {
        "token_count": len(words),                                   # rough proxy, good enough for routing
        "char_count": len(prompt),
        "sentence_count": max(1, len(re.split(r"[.!?]+", prompt)) - 1),
        "reasoning_keyword_count": _count_hits(text_lower, REASONING_KEYWORDS),
        "constraint_count": _count_hits(text_lower, CONSTRAINT_MARKERS),
        "creative_marker_count": _count_hits(text_lower, CREATIVE_MARKERS),
        "output_format_complexity": _count_hits(text_lower, OUTPUT_FORMAT_MARKERS),
        "has_context": int(has_context),
        "has_multiple_questions": int(text_lower.count("?") > 1),
        "avg_word_length": (sum(len(w) for w in words) / len(words)) if words else 0.0,
    }


FEATURE_ORDER = [
    "token_count", "char_count", "sentence_count", "reasoning_keyword_count",
    "constraint_count", "creative_marker_count", "output_format_complexity",
    "has_context", "has_multiple_questions", "avg_word_length",
]


def features_to_vector(features: dict) -> list[float]:
    return [features[k] for k in FEATURE_ORDER]
