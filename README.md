# 💸 LLM Cost Autopilot

**An intelligent routing layer that cuts LLM API costs by routing every request to the cheapest model capable of handling it - without sacrificing quality.**


</div>

---

## 📌 Overview

Companies running LLMs at scale routinely send every request — from
"extract this date" to "design a five-year strategy" — to the same
expensive, top-tier model. That's the gap between a junior AI engineer and
a senior one: **knowing that cost optimization is a business problem, not
just a technical one.**

**LLM Cost Autopilot** solves this with a routing layer that:

- 🎯 **Classifies** every incoming prompt into a complexity tier (simple / moderate / complex) using a trained scikit-learn model
- 💰 **Routes** each request to the cheapest model that tier actually needs — spanning OpenAI, Anthropic, and local Ollama models
- 🔍 **Verifies** routing decisions asynchronously in the background, using an LLM-as-judge to score the cheap model's answer against a top-tier reference
- 🔁 **Self-corrects** — when the verifier catches a quality gap, it auto-escalates and feeds the failure back as a new training example, so the classifier improves as real traffic accumulates
- 📊 **Reports** cost savings, routing distribution, and quality scores on a live dashboard

## 🏗 Architecture

```
                       ┌─────────────────────┐
   POST /v1/completions│                      │
  ───────────────────► │   FastAPI Router     │
                       │                      │
                       └─────────┬────────────┘
                                 │
                    ┌────────────┴────────────┐
                    │  Complexity Classifier   │   scikit-learn, 3 tiers
                    │  (tier 1 / 2 / 3)        │
                    └────────────┬────────────┘
                                 │
                    ┌────────────┴────────────┐
                    │   Routing Map (YAML)     │   tier → model, hot-reloadable
                    └────────────┬────────────┘
                                 │
        ┌────────────────────────┼────────────────────────┐
        ▼                        ▼                         ▼
  ┌───────────┐          ┌───────────────┐          ┌─────────────┐
  │  Ollama   │          │ OpenAI/Claude │          │ OpenAI/Claude│
  │  (local)  │          │  mid-tier      │          │  top-tier    │
  └───────────┘          └───────────────┘          └─────────────┘
        │                        │                         │
        └────────────────────────┴─────────────────────────┘
                                 │
                     response returned to caller
                                 │
                     ┌───────────┴────────────┐
                     │  Async Verifier (bg)     │  LLM-as-judge vs. top-tier model
                     │  → logs quality score    │  → auto-escalates on failure
                     │  → feeds failures back   │  → retrains classifier weekly
                     │    into classifier data  │
                     └───────────┬────────────┘
                                 │
                     ┌───────────┴────────────┐
                     │  SQLite audit trail     │  every request logged
                     └───────────┬────────────┘
                                 │
                     ┌───────────┴────────────┐
                     │  Streamlit dashboard    │  cost savings, routing mix,
                     │                         │  quality distribution
                     └─────────────────────────┘
```

## 🧰 Tech Stack

| Component        | Tool / Library                     | Why                                  |
|-------------------|-------------------------------------|----------------------------------------|
| Language          | Python 3.11+                        | Ecosystem compatibility                |
| LLM providers     | OpenAI, Anthropic, Ollama (local)   | Mix of cloud and local models          |
| Router            | FastAPI                             | Async-native, production-grade         |
| Classifier        | scikit-learn (logistic regression)  | Lightweight, explainable complexity scoring |
| Eval              | Custom scoring + LLM-as-judge       | Quality verification loop              |
| Logging           | SQLite + structured rows            | Full audit trail per request           |
| Dashboard         | Streamlit                           | Cost and quality visualization         |
| Containerization  | Docker + docker-compose             | Multi-service orchestration            |

## 📊 Results


| Metric | Value |
|---|---|
| Cost saved vs. sending everything to the top-tier model | **72.4%** |
| Held-out classifier accuracy | **91%** (270-example seed dataset) |
| Avg quality score (LLM-as-judge, 0–1) | **0.93** |
| Escalation rate | **18.6%** |

## 📁 Project Layout

```
app/
  models.py                 # ModelConfig registry — pricing, quality tier per model
  llm_providers.py          # send_request(prompt, model) — unified across providers
  db.py                     # SQLite audit trail + stats aggregation
  main.py                   # FastAPI app: /v1/completions, /v1/models, /v1/stats, /v1/routing-config
  classifier/
    features.py             # hand-crafted complexity features
    generate_dataset.py     # builds the 270-example labeled training set
    train.py                # trains + evaluates the classifier
    classifier.py           # runtime predict_tier() used by the router
  routing/
    routing_config.yaml     # tier → model map (hot-reloadable, no redeploy needed)
    router.py               # classify + look up model, config get/update
  verifier/
    verifier.py             # async LLM-as-judge scoring + auto-escalation
dashboard/
  streamlit_app.py          # cost savings dashboard, reads directly from SQLite
scripts/
  test_providers.py         # baseline every model against the same 10 prompts
  load_test.py               # realistic traffic generator + final cost report
tests/
  test_router.py             # unit tests, no API keys required
```

## 🚀 Quick Start

### 1. Clone and install

```bash
git clone https://github.com/<your-username>/llm-cost-autopilot.git
cd llm-cost-autopilot
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# fill in OPENAI_API_KEY and ANTHROPIC_API_KEY in .env
# (Ollama needs no key — just `ollama pull llama3.1:8b` and have it running)
```

### 2. Build the labeled dataset and train the classifier

```bash
python -m app.classifier.generate_dataset   # writes 270 labeled prompts
python -m app.classifier.train               # trains + saves model.joblib
```

Reaches **~91% held-out accuracy** with plain logistic regression on the
seed dataset — comfortably above the 80% V1 bar. Once you have real
traffic, retrain on logged prompts for better real-world accuracy.

### 3. Baseline every provider (optional but recommended)

```bash
python -m scripts.test_providers
```

Sends the same 10 prompts to every model in the registry so you have real
cost/latency numbers before routing live traffic.

### 4. Run the API

```bash
uvicorn app.main:app --reload
```

```bash
curl -X POST http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Summarize this in two sentences: ..."}'
```

Response includes routing metadata — which model was picked and why:

```json
{
  "text": "...",
  "cost_usd": 0.000182,
  "routing": {
    "complexity_tier": 2,
    "model_selected": "gpt-4o-mini",
    "provider": "openai",
    "reason": "Classified as moderate (summarization/classification) - routed to mid-tier model."
  }
}
```

### 5. Generate traffic and see the savings

```bash
python -m scripts.load_test --count 500
streamlit run dashboard/streamlit_app.py
```

### 6. Or run everything with Docker

```bash
docker-compose up --build
# API:       http://localhost:8000
# Dashboard: http://localhost:8501
# Ollama:    http://localhost:11434
```


## 🔁 How the Quality Loop Works

1. A request comes in, gets routed to the cheapest model for its tier, and
   the response is returned to the caller immediately — no added latency.
2. In the background, the same prompt is re-run against the configured
   `verifier_reference_model` (your best model).
3. An LLM-as-judge call scores agreement between the cheap model's answer
   and the reference answer, 0.0–1.0.
4. If the score falls below `escalation_quality_threshold` (default 0.6),
   it's logged as a routing failure and appended to
   `app/classifier/data/routing_failures.csv` with a corrected (higher)
   tier label.
5. Periodically re-run `train.py` on the combined dataset — this is the
   flywheel that makes the router smarter as real traffic accumulates.



## 📄 License

This project is licensed under the [MIT License](LICENSE).

---

<div align="center">
Built by <strong><a href="https://github.com/<your-username>">Huzaifa</a></strong> — Computer Science student, AI/ML enthusiast.
</div>
