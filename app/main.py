from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

load_dotenv()

from app.db import get_stats, init_db, log_request  # noqa: E402
from app.llm_providers import send_request  # noqa: E402
from app.models import MODEL_REGISTRY  # noqa: E402
from app.routing.router import get_config, route, update_routing_map  # noqa: E402
from app.verifier.verifier import schedule_verification  # noqa: E402


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield


app = FastAPI(
    title="LLM Cost Autopilot",
    description="Routes each request to the cheapest model capable of handling it, "
                 "then continuously verifies the routing decisions were correct.",
    version="1.0.0",
    lifespan=lifespan,
)


class CompletionRequest(BaseModel):
    prompt: str = Field(..., description="The user's prompt.")
    has_context: bool = Field(
        default=False, description="Whether the prompt includes retrieved/provided context."
    )
    max_tokens: int = Field(default=1024, ge=1, le=8000)
    verify: bool = Field(
        default=True, description="Whether to run the async quality verification loop."
    )


class RoutingMetadata(BaseModel):
    complexity_tier: int
    model_selected: str
    provider: str
    reason: str


class CompletionResponse(BaseModel):
    text: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    latency_ms: float
    routing: RoutingMetadata


@app.post("/v1/completions", response_model=CompletionResponse)
async def create_completion(req: CompletionRequest):
    decision = route(req.prompt, has_context=req.has_context)

    try:
        response = await send_request(req.prompt, decision.model, max_tokens=req.max_tokens)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"Upstream provider error: {exc}") from exc

    request_id = log_request(
        prompt=req.prompt,
        complexity_tier=decision.tier,
        routed_model=response.model,
        input_tokens=response.input_tokens,
        output_tokens=response.output_tokens,
        cost=response.cost,
        latency_ms=response.latency_ms,
    )

    if req.verify:
        schedule_verification(request_id, req.prompt, decision.tier, response)

    reason = {
        1: "Classified as simple (reformatting/extraction/basic Q&A) - routed to cheapest model.",
        2: "Classified as moderate (summarization/classification) - routed to mid-tier model.",
        3: "Classified as complex (multi-step reasoning/creative/judgment) - routed to top-tier model.",
    }[decision.tier]

    return CompletionResponse(
        text=response.text,
        input_tokens=response.input_tokens,
        output_tokens=response.output_tokens,
        cost_usd=round(response.cost, 6),
        latency_ms=round(response.latency_ms, 1),
        routing=RoutingMetadata(
            complexity_tier=decision.tier,
            model_selected=response.model,
            provider=decision.model.provider.value,
            reason=reason,
        ),
    )



@app.get("/v1/models")
async def list_models():
    return {
        name: {
            "provider": cfg.provider.value,
            "cost_per_1m_input_usd": cfg.cost_per_1m_input,
            "cost_per_1m_output_usd": cfg.cost_per_1m_output,
            "quality_tier": cfg.quality_tier.value,
            "avg_latency_ms": cfg.avg_latency_ms,
        }
        for name, cfg in MODEL_REGISTRY.items()
    }



@app.get("/v1/stats")
async def stats(window_days: int | None = None):
    since_seconds = window_days * 86400 if window_days else None
    return get_stats(since_seconds=since_seconds)


@app.get("/v1/routing-config")
async def get_routing_config():
    return get_config()


class RoutingConfigUpdate(BaseModel):
    routing_map: dict[int, str] = Field(
        ..., description="Maps complexity tier (1/2/3) to a model name from /v1/models."
    )


@app.put("/v1/routing-config")
async def put_routing_config(update: RoutingConfigUpdate):
    try:
        new_config = update_routing_map(update.routing_map)
    except KeyError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return new_config


@app.get("/health")
async def health():
    return {"status": "ok"}
