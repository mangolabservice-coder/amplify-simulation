# AMPLIFY-clone

A small, faithful reproduction of NASA Ames' **AMPLIFY** RAG architecture
(NASA/TM–20260000162). Same component boundaries, same data flow — just
sized to run on a laptop.

## Architecture mapping

| AMPLIFY paper component                | This repo                                           |
|----------------------------------------|-----------------------------------------------------|
| Chat Interface (LibreChat)             | `app/ui/index.html`                                 |
| Unified LLM Interface (FastAPI)        | `app/main.py`, `app/api/*`                          |
| Model Router                           | `app/router.py`                                     |
| Model Context Protocol (MCP)           | `app/mcp.py`                                        |
| Self-hosted Foundational Model (Llama) | `app/llm/ollama_backend.py` (Ollama, internal)      |
| Managed Foundational Model (OpenAI)    | `app/llm/openai_backend.py` (OpenAI, external)      |
| Embeddings + RAG (LlamaIndex)          | `app/rag/ingest.py`, `app/rag/retrieve.py`          |
| Vector DB (Qdrant)                     | `docker-compose.yml` (real Qdrant container)        |
| Observability (Langfuse)               | `app/observability.py`, `/traces` endpoint          |
| File Uploads / Adaptation Raw Data     | `app/api/ingest.py`, `corpus/`                      |
| VALOR evaluation                       | *out of scope for this clone*                       |

The data flow matches Figure 1 of the paper:

```
Chat UI → /chat (FastAPI)
       → ModelRouter.decide(query) ──► picks internal (Ollama) or external (OpenAI)
       → retrieve(query) ──► Qdrant (top-k chunks via HF embeddings)
       → MCPContext.build() ──► structured (system, context, citations) payload
       → backend.chat(messages) ──► answer
       └── every step recorded as a span in `tracer` (Langfuse-style)
```

## Quickstart

### 1. Vector DB

```bash
docker compose up -d qdrant
```

Qdrant will be at `http://localhost:6333` (UI: `http://localhost:6333/dashboard`).

### 2. Python env

Use **Python 3.11+** (3.13 is fine). Prefer the venv’s interpreter for installs and for running the server so Conda does not steal `pip` / `uvicorn`:

```bash
python3 -m venv .venv && source .venv/bin/activate
python -m pip install -r requirements.txt
cp .env.example .env
# Set OPENAI_API_KEY (optional): this app loads, in order, ../.env, ../fritz-hub/.env,
# then .env here — use the same key as n8n’s OpenAI credential in Fritz Hub without duplicating it.
# Confirm OLLAMA_BASE_URL if you use Ollama.
```

### 3. (Optional) self-hosted model via Ollama

Install the **Ollama app** from [ollama.com](https://ollama.com) so the `ollama` CLI is on your `PATH`, or run Ollama via Docker (for example the `ollama` service in `fritz-hub/docker-compose.yml` on port `11434` and set `OLLAMA_BASE_URL=http://localhost:11434` in `.env`).

```bash
ollama pull llama3.2
# serve is usually automatic after install; otherwise: ollama serve
```

Without Ollama the router will fall back to OpenAI; without an OpenAI key
the router will only be able to serve internal traffic.

### 4. Ingest the demo corpus (the AMPLIFY whitepaper itself)

```bash
python -m scripts.ingest_corpus
```

The first run will download the BGE embedding model (~130 MB).

### 5. Run the API + UI

From `amplify-simulation/` with the venv activated:

```bash
python -m uvicorn app.main:app --reload --port 8000
```

If you see `ModuleNotFoundError` for `llama_index` but the traceback paths start with **conda** (`/opt/anaconda3/...`), the wrong Python is running. Use `which python` (should be inside `.venv/`) or run explicitly: `./.venv/bin/python -m uvicorn app.main:app --reload --port 8000`.

Open http://localhost:8000.

## Try it

Ask things like:

- *"What does the Model Router do, and what determines its routing decisions?"*
  → likely routes **external** (OpenAI), since no sensitive keywords hit.
- *"Summarize the SOPs for handling a non-cooperative UAV in Class B airspace."*
  → likely routes **internal** (Ollama) — `sop` and `faa` are sensitive keywords.

You'll see in the right-hand panel:
- **Citations** — `[n] file, p.X, score=…` matching the inline `[1]/[2]` markers.
- **Last trace** — every span (`router.decide`, `rag.retrieve`, `mcp.pack`,
  `llm.chat`) with timing and attributes — the same anatomy Langfuse shows
  in Figures 3–4 of the paper.

## What's deliberately *not* included

- **Production VALOR rubric scoring**: would need a labeled query/answer set.
- **Sharded multi-node Qdrant**: single-container is plenty for a demo.
- **Multi-agent (Planner / Step Definer / QA Agent)**: the paper lists this as
  *future work* (Figure 7); this clone stops at the linear RAG pipeline.

## Layout

```
amplify-clone/
├── app/
│   ├── main.py              # FastAPI app
│   ├── config.py            # Settings (.env)
│   ├── observability.py     # Langfuse-style tracer
│   ├── mcp.py               # Model Context Protocol
│   ├── router.py            # Model Router (policy)
│   ├── llm/
│   │   ├── base.py
│   │   ├── openai_backend.py
│   │   └── ollama_backend.py
│   ├── rag/
│   │   ├── ingest.py        # chunk + embed + Qdrant write
│   │   └── retrieve.py      # query → top-k chunks
│   ├── api/
│   │   ├── chat.py          # /chat
│   │   ├── ingest.py        # /ingest
│   │   └── traces.py        # /traces, /traces/{id}
│   └── ui/index.html
├── corpus/
│   └── AMPLIFY_whitepaper.pdf
├── scripts/ingest_corpus.py
├── docker-compose.yml
├── requirements.txt
└── .env.example
```
