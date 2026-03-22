#!/usr/bin/env python3
"""
quickstart.py — Run the full pipeline locally in one script.

Usage:
    export OPENAI_API_KEY=sk-...
    python quickstart.py
"""

import os, sys, json

# Make sure we can import our modules
sys.path.insert(0, os.path.dirname(__file__))

from app.rag_pipeline import RAGPipeline

SAMPLE_DOC = "data/docs/sample.txt"

def write_sample_doc():
    """Write a tiny sample document so you can test without uploading a real PDF."""
    os.makedirs("data/docs", exist_ok=True)
    content = """
Artificial Intelligence Overview

Machine learning is a subset of artificial intelligence that gives systems the ability
to automatically learn and improve from experience without being explicitly programmed.

Deep learning is a subset of machine learning based on artificial neural networks with
multiple layers. It powers modern applications like image recognition, NLP, and more.

Retrieval-Augmented Generation (RAG) is a technique that combines a retrieval system
(fetching relevant documents) with a generative model (producing fluent answers).
This significantly reduces hallucinations compared to pure generation.

FAISS (Facebook AI Similarity Search) is a library for efficient similarity search
and clustering of dense vectors. It is widely used in production RAG pipelines.

LangChain is an open-source framework that simplifies building LLM-powered applications
by providing abstractions for chains, memory, agents, and document loaders.
    """.strip()

    with open(SAMPLE_DOC, "w") as f:
        f.write(content)
    print(f"✅ Sample document written → {SAMPLE_DOC}")


QUESTIONS = [
    "What is retrieval-augmented generation?",
    "How does FAISS work?",
    "What is the relationship between deep learning and machine learning?",
]

GROUND_TRUTH = [
    "RAG combines a retrieval system with a generative model to reduce hallucinations.",
    "FAISS is a library for efficient similarity search and clustering of dense vectors.",
    "Deep learning is a subset of machine learning based on artificial neural networks.",
]


def main():
    write_sample_doc()

    print("\n── Phase 1: Ingestion ────────────────────────────────────────")
    print("Using LLaMA 3 + nomic-embed-text via Ollama (100% free, runs locally)")
    pipeline = RAGPipeline(model="llama")
    result = pipeline.ingest(SAMPLE_DOC)
    print(json.dumps(result, indent=2))

    print("\n── Phase 2: Query ────────────────────────────────────────────")
    for q in QUESTIONS:
        res = pipeline.query(q)
        print(f"\nQ: {q}")
        print(f"A: {res['answer']}")
        print(f"   ⏱  {res['latency_ms']} ms  |  sources: {res['sources']}")

    print("\n✅  Quickstart complete. Now run: flask --app app.api run")


if __name__ == "__main__":
    main()
