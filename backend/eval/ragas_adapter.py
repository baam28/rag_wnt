"""Adapter layer between repository RAG pipeline and RAGAS dataset format.

Mirrors the full chat pipeline (supervisor → SQL agents → RAG retrieval → generate_answer)
so that drug_price questions route through the SQL agent instead of vector search.
"""

from __future__ import annotations

from typing import Any

try:
    from prompts import (
        generate_answer,
        SYSTEM_PROMPT,
        USER_PROMPT_TEMPLATE,
        DRUG_SYSTEM_PROMPT,
        DRUG_USER_PROMPT_TEMPLATE,
        ERP_SYSTEM_PROMPT,
        ERP_USER_PROMPT_TEMPLATE,
        COMBINED_SYSTEM_PROMPT,
        COMBINED_USER_PROMPT_TEMPLATE,
    )
    from retriever import retrieve
    from config import get_settings
except ImportError:  # pragma: no cover
    from backend.prompts import (
        generate_answer,
        SYSTEM_PROMPT,
        USER_PROMPT_TEMPLATE,
        DRUG_SYSTEM_PROMPT,
        DRUG_USER_PROMPT_TEMPLATE,
        ERP_SYSTEM_PROMPT,
        ERP_USER_PROMPT_TEMPLATE,
        COMBINED_SYSTEM_PROMPT,
        COMBINED_USER_PROMPT_TEMPLATE,
    )
    from backend.retriever import retrieve
    from backend.config import get_settings


def _select_prompts(
    collections: list[str],
    has_rag: bool,
    has_erp: bool,
) -> tuple[str, str]:
    """Pick system_prompt + user_template matching the chat router logic."""
    has_legal = "legal" in collections
    is_legal_only = collections == ["legal"]
    is_drug_only = not has_erp and collections == ["drug"]

    if has_rag and has_erp and has_legal:
        return COMBINED_SYSTEM_PROMPT, COMBINED_USER_PROMPT_TEMPLATE
    if has_erp:
        return ERP_SYSTEM_PROMPT, ERP_USER_PROMPT_TEMPLATE
    if is_legal_only:
        return SYSTEM_PROMPT, USER_PROMPT_TEMPLATE
    if is_drug_only:
        return DRUG_SYSTEM_PROMPT, DRUG_USER_PROMPT_TEMPLATE
    # mixed legal+drug without price
    return COMBINED_SYSTEM_PROMPT, COMBINED_USER_PROMPT_TEMPLATE


def run_rag_pipeline_for_sample(sample: dict[str, Any]) -> dict[str, Any]:
    """Execute the RAG pipeline for one evaluation sample.

    Uses the collection(s) specified in the sample directly — the supervisor,
    ERP agent, and legal routing are intentionally skipped so that eval results
    are stable and reproducible.  Only vector RAG + generate_answer run.
    """
    question = str(sample["question"]).strip()
    # collections from the dataset indicate which domain(s) to search
    sample_collections = [str(c) for c in sample.get("collections") or []]
    if not question:
        raise ValueError("Sample question is empty")
    if not sample_collections:
        raise ValueError("Sample collections must be non-empty")

    settings = get_settings()

    # Use sample collections directly — skip supervisor, ERP, and legal routing
    # for drug-only eval dataset
    collections = sample_collections

    # Vector RAG retrieval (drug collection only)
    physical_collections = []
    for c in collections:
        if c == "drug":
            physical_collections.append(settings.drug_collection_name)

    rag_docs: list[dict[str, Any]] = []
    if physical_collections:
        try:
            rag_docs = retrieve(question, collections_to_search=physical_collections)
        except Exception:
            rag_docs = []

    final_contexts = list(rag_docs)

    # Choose prompt template
    system_prompt, user_template = _select_prompts(
        collections=collections,
        has_rag=bool(rag_docs),
        has_erp=False,
    )

    # 6. Generate answer
    answer, usage = generate_answer(
        question,
        final_contexts,
        system_prompt=system_prompt,
        user_template=user_template,
    )

    context_texts = [
        str(ctx.get("content") or "")
        for ctx in final_contexts
        if str(ctx.get("content") or "").strip()
    ]
    context_sources = [str(ctx.get("source") or "Unknown") for ctx in final_contexts]

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
