"""
BenchmarkEvaluator — GPT vs LLaMA head-to-head comparison.

Metrics tracked per question:
  - answer text
  - latency (ms)
  - token-level F1 score against ground truth (if provided)
  - exact match (if ground truth provided)

Summary statistics mirror what's described in the resume:
  "analyzed robustness and accuracy across multiple domains"
"""

import time
import re
import logging
from typing import List, Dict, Any, Optional
from collections import Counter

logger = logging.getLogger(__name__)


# ── Token-level scoring helpers ───────────────────────────────────────────────

def _tokenize(text: str) -> List[str]:
    """Lowercase, strip punctuation, split on whitespace."""
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    return text.split()


def _f1_score(prediction: str, ground_truth: str) -> float:
    """Compute token-overlap F1 between two strings."""
    pred_tokens = Counter(_tokenize(prediction))
    gt_tokens = Counter(_tokenize(ground_truth))

    common = sum((pred_tokens & gt_tokens).values())
    if common == 0:
        return 0.0

    precision = common / sum(pred_tokens.values())
    recall = common / sum(gt_tokens.values())
    return round(2 * precision * recall / (precision + recall), 4)


def _exact_match(prediction: str, ground_truth: str) -> bool:
    """Case-insensitive exact match after stripping whitespace."""
    return prediction.strip().lower() == ground_truth.strip().lower()


# ── Evaluator ─────────────────────────────────────────────────────────────────

class BenchmarkEvaluator:
    """
    Runs both pipelines on the same questions and produces a structured report.
    """

    def __init__(self, gpt_pipeline, llama_pipeline):
        self.gpt = gpt_pipeline
        self.llama = llama_pipeline

    def run(
        self,
        questions: List[str],
        ground_truth: List[str] = None,
    ) -> Dict[str, Any]:
        """
        Args:
            questions:     list of question strings
            ground_truth:  optional list of reference answers (same length)

        Returns:
            Full benchmark report dict.
        """
        gt = ground_truth or []
        results = []

        for i, question in enumerate(questions):
            ref = gt[i] if i < len(gt) else None
            row = self._eval_question(question, ref)
            results.append(row)
            logger.info(f"[{i+1}/{len(questions)}] '{question[:60]}' done.")

        report = self._summarize(results)
        report["per_question"] = results
        return report

    # ── Internal ──────────────────────────────────────────────────────────────

    def _eval_question(self, question: str, ground_truth: Optional[str]) -> Dict[str, Any]:
        row: Dict[str, Any] = {"question": question, "ground_truth": ground_truth}

        for name, pipeline in [("gpt", self.gpt), ("llama", self.llama)]:
            t0 = time.time()
            try:
                res = pipeline.query(question)
                answer = res["answer"]
                latency = round((time.time() - t0) * 1000, 1)
            except Exception as e:
                answer = f"ERROR: {e}"
                latency = None

            entry: Dict[str, Any] = {
                "answer": answer,
                "latency_ms": latency,
            }

            if ground_truth:
                entry["f1"] = _f1_score(answer, ground_truth)
                entry["exact_match"] = _exact_match(answer, ground_truth)

            row[name] = entry

        return row

    def _summarize(self, results: List[Dict]) -> Dict[str, Any]:
        """Compute aggregate stats for both models."""
        summary: Dict[str, Any] = {"total_questions": len(results)}

        for model in ("gpt", "llama"):
            latencies = [
                r[model]["latency_ms"]
                for r in results
                if r[model].get("latency_ms") is not None
            ]
            f1_scores = [
                r[model]["f1"]
                for r in results
                if "f1" in r[model]
            ]
            exact_matches = [
                r[model]["exact_match"]
                for r in results
                if "exact_match" in r[model]
            ]

            summary[model] = {
                "avg_latency_ms": round(sum(latencies) / len(latencies), 1) if latencies else None,
                "avg_f1": round(sum(f1_scores) / len(f1_scores), 4) if f1_scores else None,
                "exact_match_rate": round(sum(exact_matches) / len(exact_matches), 4) if exact_matches else None,
            }

        # Winner determination (by F1 if available, else latency)
        gpt_f1 = summary["gpt"].get("avg_f1")
        llama_f1 = summary["llama"].get("avg_f1")
        if gpt_f1 is not None and llama_f1 is not None:
            summary["winner_accuracy"] = "gpt" if gpt_f1 >= llama_f1 else "llama"
        gpt_lat = summary["gpt"].get("avg_latency_ms") or float("inf")
        llama_lat = summary["llama"].get("avg_latency_ms") or float("inf")
        summary["winner_speed"] = "gpt" if gpt_lat <= llama_lat else "llama"

        return summary
