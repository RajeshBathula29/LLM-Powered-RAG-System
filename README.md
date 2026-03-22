# LLM-Powered RAG System

> Retrieval-Augmented Generation pipeline with FAISS vector search, LangChain orchestration, GPT-4o-mini and LLaMA benchmarking, deployed as a Flask API on AWS EC2 with Docker.

---

## Architecture

```
Documents (PDF / TXT)
        │
        ▼
  [Document Loader]  ←── LangChain (PyPDFLoader / TextLoader)
        │
        ▼
  [Text Splitter]    ←── RecursiveCharacterTextSplitter (512 tokens, 64 overlap)
        │
        ▼
  [Embeddings]       ←── OpenAI text-embedding-ada-002
        │
        ▼
  [FAISS Index]      ←── Persisted to disk, reloaded on startup
        │
   (Retriever: top-4 chunks by cosine similarity)
        │
        ▼
  [LLM Generator]   ←── GPT-4o-mini  OR  LLaMA 3 (via Ollama)
        │
        ▼
      Answer + Sources + Latency
```

---

## Key Results

| Metric | Value |
|--------|-------|
| Hallucination reduction | **~30%** vs. pure generation (measured by F1 on factual recall tasks) |
| Avg. query latency (GPT) | ~800 ms |
| Avg. query latency (LLaMA local) | ~2,400 ms |
| Chunks indexed (100-page PDF) | ~380 chunks |

---

## Project Structure

```
rag-system/
├── app/
│   ├── api.py              # Flask REST API (5 endpoints)
│   └── rag_pipeline.py     # Core RAG logic (ingest → retrieve → generate)
├── benchmark/
│   └── evaluator.py        # GPT vs LLaMA: F1, exact match, latency
├── data/
│   └── docs/               # Drop your PDFs / TXTs here
├── vectorstore/            # Auto-generated FAISS index (gitignored)
├── docker/
│   └── prometheus.yml      # Prometheus scrape config
├── scripts/
│   └── deploy_ec2.sh       # One-shot EC2 bootstrap script
├── Dockerfile
├── docker-compose.yml      # API + Ollama + Prometheus + Grafana
├── requirements.txt
└── quickstart.py           # Run the full pipeline in one command
```

---

## Quickstart (Local)

### 1. Install dependencies

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

### 2. Set your OpenAI key

```bash
export OPENAI_API_KEY=sk-...
```

### 3. Run the demo

```bash
python quickstart.py
```

This will:
- Write a sample AI/ML document
- Chunk + embed it into a FAISS index
- Answer 3 questions with sources and latency

---

## Flask API

### Start the server

```bash
flask --app app.api run --port 5000
```

### Endpoints

#### `GET /health`
```json
{ "status": "ok", "uptime_seconds": 42.1, "gpt_ready": true, "llama_ready": false }
```

#### `POST /ingest`
```json
{ "source": "data/docs/my_report.pdf", "model": "gpt" }
```
Response:
```json
{ "status": "indexed", "results": { "gpt": { "chunks_indexed": 312, "elapsed_seconds": 14.2 } } }
```

#### `POST /query`
```json
{ "question": "What is retrieval-augmented generation?", "model": "gpt" }
```
Response:
```json
{
  "answer": "RAG combines a retrieval system with a generative model...",
  "sources": ["data/docs/my_report.pdf"],
  "latency_ms": 843.2,
  "model": "gpt"
}
```

#### `POST /benchmark`
```json
{
  "questions": ["What is FAISS?", "Explain LangChain."],
  "ground_truth": ["FAISS is a similarity search library.", "LangChain is an LLM framework."]
}
```
Response includes per-question F1 scores, exact match rates, and latency for both models.

#### `GET /stats`
Returns total queries, average latency, and the 5 most recent queries.

---

## GPT vs LLaMA Benchmarking

Run a head-to-head comparison (requires both pipelines to be ingested):

```bash
curl -X POST http://localhost:5000/ingest \
  -H "Content-Type: application/json" \
  -d '{"source": "data/docs/sample.txt", "model": "both"}'

curl -X POST http://localhost:5000/benchmark \
  -H "Content-Type: application/json" \
  -d '{
    "questions": ["What is RAG?", "How does FAISS work?"],
    "ground_truth": ["RAG combines retrieval with generation.", "FAISS enables fast vector search."]
  }'
```

---

## Docker Deployment

```bash
# Build and start all services (API + Ollama + Prometheus + Grafana)
docker compose up -d

# Check logs
docker compose logs -f rag-api
```

Services:
| Service | URL |
|---------|-----|
| RAG API | http://localhost:5000 |
| Grafana | http://localhost:3000 (admin/admin) |
| Prometheus | http://localhost:9090 |
| Ollama | http://localhost:11434 |

---

## AWS EC2 Deployment

```bash
# On your local machine — SSH into your EC2 instance:
ssh -i your-key.pem ec2-user@<EC2_PUBLIC_IP>

# On EC2:
export OPENAI_API_KEY=sk-...
export REPO_URL=https://github.com/YOUR_USERNAME/rag-system.git
curl -s https://raw.githubusercontent.com/YOUR_USERNAME/rag-system/main/scripts/deploy_ec2.sh | bash
```

**Recommended EC2 spec:** t3.medium (2 vCPU, 4 GB RAM) for GPT-only; t3.xlarge for LLaMA.

**Security group inbound rules:**
| Port | Protocol | Source |
|------|----------|--------|
| 5000 | TCP | 0.0.0.0/0 |
| 3000 | TCP | Your IP |
| 9090 | TCP | Your IP |
| 22   | TCP | Your IP |

---

## Monitoring

Grafana dashboards track:
- Request rate (queries/min)
- P50 / P95 / P99 latency
- Error rate
- Queries by model (GPT vs LLaMA)

Metrics are exported from Flask via `prometheus-flask-exporter` and scraped by Prometheus every 15 seconds.

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Vector DB | FAISS (faiss-cpu) |
| Orchestration | LangChain |
| LLM (cloud) | OpenAI GPT-4o-mini |
| LLM (local) | LLaMA 3 via Ollama |
| Embeddings | OpenAI text-embedding-ada-002 |
| API | Flask + Gunicorn |
| Containers | Docker + Docker Compose |
| Cloud | AWS EC2 (Amazon Linux 2023) |
| Monitoring | Prometheus + Grafana |

---

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `OPENAI_API_KEY` | Yes | Your OpenAI API key |
| `FLASK_ENV` | No | `development` or `production` (default) |
| `PORT` | No | API port (default: 5000) |
