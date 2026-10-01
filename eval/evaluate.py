"""Measure how well the assistant works, so the numbers on your resume are real.

Retrieval metrics (no API key needed):
  Hit@1 / Hit@3  - share of questions where a relevant passage is ranked 1st / in the top 3
  MRR            - mean reciprocal rank of the first relevant passage (1.0 = always first)
  Latency        - average retrieval time per question

With --llm (needs GEMINI_API_KEY):
  Citation rate  - share of in-scope answers that cite at least one passage
  Refusal rate   - share of out-of-scope questions correctly answered with "I couldn't find..."

A passage counts as relevant if it contains any of the question's expected_keywords
(case-insensitive). Edit eval/questions.json to match YOUR documents - check each
keyword really appears in your PDFs before trusting the scores.

Run:  python -m eval.evaluate            (retrieval only)
      python -m eval.evaluate --llm      (retrieval + answers)
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
import time
from pathlib import Path


def is_relevant(text: str, keywords: list[str]) -> bool:
    t = text.lower()
    return any(k.lower() in t for k in keywords)


def retrieval_metrics(index, questions: list[dict], k: int = 3) -> dict:
    in_scope = [q for q in questions if not q.get("out_of_scope")]
    hits1 = hitsk = 0
    rr, times = [], []
    for q in in_scope:
        t0 = time.perf_counter()
        results = index.search(q["question"], k=max(k, 10))
        times.append(time.perf_counter() - t0)
        ranks = [i for i, (c, _) in enumerate(results, 1) if is_relevant(c.text, q["expected_keywords"])]
        first = ranks[0] if ranks else None
        hits1 += first == 1
        hitsk += first is not None and first <= k
        rr.append(1 / first if first else 0.0)
    n = len(in_scope)
    return {"questions": n, "hit@1": hits1 / n, f"hit@{k}": hitsk / n,
            "mrr": sum(rr) / n, "avg_retrieval_ms": 1000 * statistics.mean(times)}


def llm_metrics(index, generator, questions: list[dict]) -> dict:
    from rag.generator import NOT_FOUND, answer
    cited = in_n = refused = out_n = 0
    for q in questions:
        res = answer(q["question"], index, generator)
        if q.get("out_of_scope"):
            out_n += 1
            refused += NOT_FOUND.lower().rstrip(".") in res["answer"].lower()
        else:
            in_n += 1
            cited += bool(re.search(r"\[\d+\]", res["answer"]))
    return {"citation_rate": cited / in_n if in_n else None,
            "refusal_rate_out_of_scope": refused / out_n if out_n else None}


if __name__ == "__main__":
    from dotenv import load_dotenv
    from rag.retriever import SentenceTransformerEmbedder, VectorIndex

    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", default="eval/questions.json")
    ap.add_argument("--index", default="index")
    ap.add_argument("--llm", action="store_true")
    args = ap.parse_args()
    load_dotenv()

    qs = json.loads(Path(args.questions).read_text(encoding="utf-8"))
    idx = VectorIndex.load(Path(args.index), SentenceTransformerEmbedder())
    report = retrieval_metrics(idx, qs)
    if args.llm:
        from rag.generator import GeminiGenerator
        report.update(llm_metrics(idx, GeminiGenerator(), qs))
    for key, val in report.items():
        print(f"{key:28s} {val:.3f}" if isinstance(val, float) else f"{key:28s} {val}")
    Path("eval/results.json").write_text(json.dumps(report, indent=2))
    print("\nSaved eval/results.json")
