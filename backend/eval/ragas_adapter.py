"""Adapter layer between repository RAG pipeline and RAGAS dataset format."""

from __future__ import annotations

from typing import Any

try:
    from prompts import generate_answer
    from retriever import retrieve
except ImportError:  # pragma: no cover
    from backend.prompts import generate_answer
    from backend.retriever import retrieve


def run_rag_pipeline_for_sample(sample: dict[str, Any]) -> dict[str, Any]:
    """Execute retrieval + generation for one evaluation sample."""
    question = str(sample["question"]).strip()
    collections = [str(c) for c in sample.get("collections") or []]
    if not question:
        raise ValueError("Sample question is empty")
    if not collections:
        raise ValueError("Sample collections must be non-empty")

    retrieved_contexts = retrieve(question, collections_to_search=collections)
    answer, usage = generate_answer(question, retrieved_contexts)

    context_texts = [str(ctx.get("content") or "") for ctx in retrieved_contexts if str(ctx.get("content") or "").strip()]
    context_sources = [str(ctx.get("source") or "Unknown") for ctx in retrieved_contexts]

    return {
        "id": sample.get("id"),
        "question": question,
        "ground_truth": str(sample.get("ground_truth") or "").strip(),
        "answer": answer,
        "contexts": context_texts,
        "context_sources": context_sources,
        "collections": collections,
        "metadata": sample.get("metadata") or {},
        "usage": usage,
    }


def to_ragas_rows(outputs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert pipeline outputs into RAGAS-evaluable row schema."""
    rows: list[dict[str, Any]] = []
    for output in outputs:
        rows.append(
            {
                "question": output["question"],
                "answer": output["answer"],
                "contexts": output.get("contexts") or [""],
                "ground_truth": output.get("ground_truth") or "",
                "id": output.get("id"),
                "collections": output.get("collections") or [],
                "context_sources": output.get("context_sources") or [],
                "metadata": output.get("metadata") or {},
            }
        )
    return rows
