"""
Flask REST API — wraps the RAG pipeline.

Endpoints:
  POST /ingest        — load & index documents
  POST /query         — ask a question
  POST /benchmark     — GPT vs LLaMA head-to-head
  GET  /health        — liveness check
  GET  /stats         — index + query statistics
"""

import os
import time
import json
import logging
from datetime import datetime
from functools import wraps
from flask import Flask, request, jsonify
from app.rag_pipeline import RAGPipeline
from benchmark.evaluator import BenchmarkEvaluator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config["JSON_SORT_KEYS"] = False

# ── Globals ───────────────────────────────────────────────────────────────────

gpt_pipeline = None
llama_pipeline = None
query_log: list = []          # in-memory log; swap for DB in production

START_TIME = time.time()

# ── Helpers ───────────────────────────────────────────────────────────────────

def require_json(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        if not request.is_json:
            return jsonify({"error": "Content-Type must be application/json"}), 415
        return f(*args, **kwargs)
    return wrapper

def log_query(endpoint, payload, response, latency_ms):
    query_log.append({
        "ts": datetime.utcnow().isoformat(),
        "endpoint": endpoint,
        "payload": payload,
        "response": response,
        "latency_ms": latency_ms,
    })

# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return jsonify({
        "status": "ok",
        "uptime_seconds": round(time.time() - START_TIME, 1),
        "gpt_ready": gpt_pipeline is not None and gpt_pipeline.qa_chain is not None,
        "llama_ready": llama_pipeline is not None and llama_pipeline.qa_chain is not None,
    })


@app.get("/stats")
def stats():
    latencies = [q["latency_ms"] for q in query_log if q.get("latency_ms")]
    return jsonify({
        "total_queries": len(query_log),
        "avg_latency_ms": round(sum(latencies) / len(latencies), 1) if latencies else 0,
        "max_latency_ms": max(latencies) if latencies else 0,
        "recent_queries": query_log[-5:],
    })


@app.post("/ingest")
@require_json
def ingest():
    """
    Body: { "source": "data/docs/myfile.pdf", "model": "gpt" }
    model is optional (defaults to gpt); pass "both" to index for both models.
    """
    global gpt_pipeline, llama_pipeline

    body = request.get_json()
    source = body.get("source")
    model = body.get("model", "gpt")

    if not source:
        return jsonify({"error": "Missing 'source' field"}), 400

    results = {}
    t0 = time.time()

    try:
        if model in ("gpt", "both"):
            api_key = os.getenv("OPENAI_API_KEY")
            gpt_pipeline = RAGPipeline(model="gpt", openai_api_key=api_key)
            results["gpt"] = gpt_pipeline.ingest(source)

        if model in ("llama", "both"):
            llama_pipeline = RAGPipeline(model="llama")
            results["llama"] = llama_pipeline.ingest(source)

        return jsonify({
            "status": "indexed",
            "source": source,
            "results": results,
            "total_elapsed_seconds": round(time.time() - t0, 2),
        })

    except Exception as e:
        logger.exception("Ingest failed")
        return jsonify({"error": str(e)}), 500


@app.post("/query")
@require_json
def query():
    """
    Body: { "question": "...", "model": "gpt" }
    model defaults to "gpt".
    """
    body = request.get_json()
    question = body.get("question")
    model = body.get("model", "gpt")

    if not question:
        return jsonify({"error": "Missing 'question' field"}), 400

    pipeline = gpt_pipeline if model == "gpt" else llama_pipeline
    if pipeline is None or pipeline.qa_chain is None:
        return jsonify({"error": f"{model} pipeline not ready. POST /ingest first."}), 503

    try:
        t0 = time.time()
        result = pipeline.query(question)
        latency_ms = round((time.time() - t0) * 1000, 1)

        log_query("/query", body, result, latency_ms)
        return jsonify(result)

    except Exception as e:
        logger.exception("Query failed")
        return jsonify({"error": str(e)}), 500


@app.post("/benchmark")
@require_json
def benchmark():
    """
    Run GPT vs LLaMA on a set of questions and return scored results.

    Body: {
      "questions": ["Q1", "Q2", ...],
      "ground_truth": ["A1", "A2", ...]   # optional, for accuracy scoring
    }
    """
    body = request.get_json()
    questions = body.get("questions", [])
    ground_truth = body.get("ground_truth", [])

    if not questions:
        return jsonify({"error": "Provide at least one question in 'questions'"}), 400

    if gpt_pipeline is None or llama_pipeline is None:
        return jsonify({
            "error": "Both pipelines must be ready. POST /ingest with model='both' first."
        }), 503

    try:
        evaluator = BenchmarkEvaluator(gpt_pipeline, llama_pipeline)
        report = evaluator.run(questions, ground_truth)
        return jsonify(report)
    except Exception as e:
        logger.exception("Benchmark failed")
        return jsonify({"error": str(e)}), 500


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    debug = os.getenv("FLASK_ENV") == "development"
    app.run(host="0.0.0.0", port=port, debug=debug)
