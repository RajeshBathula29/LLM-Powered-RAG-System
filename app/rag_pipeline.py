"""
RAG Pipeline — FAISS + LangChain + LLaMA (via Ollama, 100% free)
"""

import os
import time
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path

from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    DirectoryLoader,
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

RAG_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""You are a precise, factual assistant. Use ONLY the context below to answer.
If the answer is not in the context, say "I don't have enough information to answer that."

Context:
{context}

Question: {question}

Answer:""",
)


class RAGPipeline:
    VECTORSTORE_PATH = "vectorstore/faiss_index"

    def __init__(self, model: str = "llama", openai_api_key: Optional[str] = None):
        self.model_type = model
        self.vectorstore: Optional[FAISS] = None
        self.qa_chain = None

        logger.info("Loading nomic-embed-text embeddings via Ollama ...")
        self.embeddings = OllamaEmbeddings(model="nomic-embed-text")
        self.llm = self._build_llm()

    def _build_llm(self):
        logger.info("Loading LLaMA 3.2 via Ollama ...")
        return Ollama(model="llama3.2", temperature=0)

    def ingest(self, source: str) -> Dict[str, Any]:
        t0 = time.time()
        source_path = Path(source)

        if source_path.is_dir():
            loader = DirectoryLoader(str(source_path), glob="**/*.{pdf,txt}", show_progress=True)
        elif source_path.suffix == ".pdf":
            loader = PyPDFLoader(str(source_path))
        else:
            loader = TextLoader(str(source_path))

        docs = loader.load()
        logger.info(f"Loaded {len(docs)} raw document(s).")

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=512, chunk_overlap=64,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        chunks = splitter.split_documents(docs)
        logger.info(f"Split into {len(chunks)} chunks.")

        self.vectorstore = FAISS.from_documents(chunks, self.embeddings)
        self.vectorstore.save_local(self.VECTORSTORE_PATH)
        logger.info(f"FAISS index saved -> {self.VECTORSTORE_PATH}")

        self._build_chain()
        return {"chunks_indexed": len(chunks), "elapsed_seconds": round(time.time() - t0, 2)}

    def load_existing_index(self):
        self.vectorstore = FAISS.load_local(
            self.VECTORSTORE_PATH, self.embeddings, allow_dangerous_deserialization=True,
        )
        self._build_chain()
        logger.info("Existing FAISS index loaded.")

    def _build_chain(self):
        retriever = self.vectorstore.as_retriever(
            search_type="similarity", search_kwargs={"k": 4},
        )
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm, chain_type="stuff", retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": RAG_PROMPT},
        )

    def query(self, question: str) -> Dict[str, Any]:
        if self.qa_chain is None:
            raise RuntimeError("No index loaded. Call ingest() or load_existing_index() first.")

        t0 = time.time()
        result = self.qa_chain.invoke({"query": question})
        latency_ms = round((time.time() - t0) * 1000, 1)

        sources = list({
            doc.metadata.get("source", "unknown")
            for doc in result.get("source_documents", [])
        })

        return {
            "answer": result["result"],
            "sources": sources,
            "latency_ms": latency_ms,
            "model": self.model_type,
        }