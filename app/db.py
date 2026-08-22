import hashlib
import os
import sqlite3
import time
from contextlib import contextmanager

DB_PATH = os.environ.get("DATABASE_PATH", "./data/requests.db")

SCHEMA = """
CREATE TABLE IF NOT EXISTS requests (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp REAL NOT NULL,
    prompt_hash TEXT NOT NULL,
    prompt_preview TEXT NOT NULL,
    complexity_tier INTEGER NOT NULL,
    routed_model TEXT NOT NULL,
    input_tokens INTEGER NOT NULL,
    output_tokens INTEGER NOT NULL,
    cost REAL NOT NULL,
    latency_ms REAL NOT NULL,
    quality_score REAL,
    escalated INTEGER NOT NULL DEFAULT 0,
    escalated_to_model TEXT,
    escalation_cost_delta REAL DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_requests_timestamp ON requests(timestamp);
CREATE INDEX IF NOT EXISTS idx_requests_model ON requests(routed_model);
"""


@contextmanager
def get_conn():
    os.makedirs(os.path.dirname(DB_PATH) or ".", exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db():
    with get_conn() as conn:
        conn.executescript(SCHEMA)


def hash_prompt(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16]


def log_request(
    prompt: str,
    complexity_tier: int,
    routed_model: str,
    input_tokens: int,
    output_tokens: int,
    cost: float,
    latency_ms: float,
    quality_score: float | None = None,
    escalated: bool = False,
    escalated_to_model: str | None = None,
    escalation_cost_delta: float = 0.0,
) -> int:
    with get_conn() as conn:
        cur = conn.execute(
            """INSERT INTO requests
               (timestamp, prompt_hash, prompt_preview, complexity_tier, routed_model,
                input_tokens, output_tokens, cost, latency_ms, quality_score,
                escalated, escalated_to_model, escalation_cost_delta)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                time.time(),
                hash_prompt(prompt),
                prompt[:200],
                complexity_tier,
                routed_model,
                input_tokens,
                output_tokens,
                cost,
                latency_ms,
                quality_score,
                int(escalated),
                escalated_to_model,
                escalation_cost_delta,
            ),
        )
        return cur.lastrowid


def update_verification(request_id: int, quality_score: float, escalated: bool,
                         escalated_to_model: str | None, escalation_cost_delta: float):
    """Called by the async verifier once it's scored a request."""
    with get_conn() as conn:
        conn.execute(
            """UPDATE requests
               SET quality_score = ?, escalated = ?, escalated_to_model = ?,
                   escalation_cost_delta = ?
               WHERE id = ?""",
            (quality_score, int(escalated), escalated_to_model, escalation_cost_delta, request_id),
        )


def get_stats(since_seconds: float | None = None) -> dict:
    """Powers GET /v1/stats and the dashboard's headline numbers."""
    from app.models import most_expensive_model

    with get_conn() as conn:
        where = ""
        params: tuple = ()
        if since_seconds is not None:
            where = "WHERE timestamp >= ?"
            params = (time.time() - since_seconds,)

        row = conn.execute(
            f"""SELECT
                    COUNT(*) as total_requests,
                    COALESCE(SUM(cost), 0) as total_cost,
                    COALESCE(SUM(input_tokens + output_tokens), 0) as total_tokens,
                    COALESCE(AVG(latency_ms), 0) as avg_latency_ms,
                    COALESCE(AVG(quality_score), 0) as avg_quality_score,
                    COALESCE(SUM(escalated), 0) as escalation_count
                FROM requests {where}""",
            params,
        ).fetchone()

        by_model = conn.execute(
            f"""SELECT routed_model, COUNT(*) as count, COALESCE(SUM(cost), 0) as cost
                FROM requests {where}
                GROUP BY routed_model""",
            params,
        ).fetchall()

        total_input_output = conn.execute(
            f"""SELECT COALESCE(SUM(input_tokens), 0) as ti, COALESCE(SUM(output_tokens), 0) as to_
                FROM requests {where}""",
            params,
        ).fetchone()

    baseline_model = most_expensive_model()
    hypothetical_cost = baseline_model.cost_for(
        total_input_output["ti"], total_input_output["to_"]
    )
    actual_cost = row["total_cost"]
    savings_pct = (
        ((hypothetical_cost - actual_cost) / hypothetical_cost * 100)
        if hypothetical_cost > 0 else 0.0
    )

    return {
        "total_requests": row["total_requests"],
        "total_cost_usd": round(actual_cost, 4),
        "hypothetical_cost_if_all_on_most_expensive_model_usd": round(hypothetical_cost, 4),
        "cost_savings_pct": round(savings_pct, 1),
        "baseline_model": baseline_model.name,
        "avg_latency_ms": round(row["avg_latency_ms"], 1),
        "avg_quality_score": round(row["avg_quality_score"], 3),
        "escalation_count": row["escalation_count"],
        "escalation_rate_pct": round(
            (row["escalation_count"] / row["total_requests"] * 100) if row["total_requests"] else 0, 1
        ),
        "routing_distribution": [
            {"model": r["routed_model"], "count": r["count"], "cost_usd": round(r["cost"], 4)}
            for r in by_model
        ],
    }
