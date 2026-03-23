"""CLI runner for RAGAS evaluation against the local RAG pipeline."""

from __future__ import annotations

import argparse
import datetime as dt
from typing import Any

from datasets import Dataset
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

try:
    from config import get_settings
    from eval.dataset_loader import load_eval_samples
    from eval.ragas_adapter import run_rag_pipeline_for_sample, to_ragas_rows
    from eval.reporting import ensure_output_dir, serialize_result, write_csv, write_json
except ImportError:  # pragma: no cover
    from backend.config import get_settings
    from backend.eval.dataset_loader import load_eval_samples
    from backend.eval.ragas_adapter import run_rag_pipeline_for_sample, to_ragas_rows
    from backend.eval.reporting import ensure_output_dir, serialize_result, write_csv, write_json


try:
    from ragas import evaluate
    from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness
except Exception as exc:  # pragma: no cover
    raise RuntimeError(
        "RAGAS is not installed. Install backend deps first: pip install -r backend/requirements.txt"
    ) from exc

try:
    from ragas.run_config import RunConfig
except Exception:  # pragma: no cover
    RunConfig = None


CORE_METRICS = {
    "faithfulness": faithfulness,
    "answer_relevancy": answer_relevancy,
    "context_precision": context_precision,
    "context_recall": context_recall,
}


def _build_eval_embeddings(settings):
    if settings.embedding_model.startswith("text-embedding"):
        return OpenAIEmbeddings(model=settings.embedding_model, api_key=settings.openai_api_key)
    emb = HuggingFaceEmbeddings(
        model_name=settings.embedding_model,
        model_kwargs={"trust_remote_code": True, "device": "cpu"},
    )
    return emb


def _select_metrics(metric_names: list[str]) -> list[Any]:
    metrics: list[Any] = []
    for name in metric_names:
        key = name.strip().lower()
        if not key:
            continue
        if key not in CORE_METRICS:
            raise ValueError(f"Unsupported metric: {name}. Supported: {', '.join(CORE_METRICS)}")
        metrics.append(CORE_METRICS[key])
    if not metrics:
        raise ValueError("No metrics selected")
    return metrics


def _parse_args() -> argparse.Namespace:
    settings = get_settings()
    parser = argparse.ArgumentParser(description="Run RAGAS evaluation for rag_wnt")
    parser.add_argument(
        "--dataset",
        default="backend/eval/datasets/curated.jsonl",
        help="Path to curated JSONL dataset",
    )
    parser.add_argument("--include-synthetic", action="store_true", help="Append synthetic auto-generated eval samples")
    parser.add_argument(
        "--synthetic-max-per-collection",
        type=int,
        default=40,
        help="Max synthetic samples generated per collection",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=settings.eval_max_samples,
        help="Max number of samples (0 = all)",
    )
    parser.add_argument(
        "--judge-model",
        default=settings.eval_judge_model,
        help="LLM model used by RAGAS judge",
    )
    parser.add_argument(
        "--metrics",
        default="faithfulness,answer_relevancy,context_precision,context_recall",
        help="Comma-separated metrics",
    )
    parser.add_argument(
        "--output-dir",
        default=str(settings.eval_output_dir),
        help="Directory for eval outputs",
    )
    parser.add_argument(
        "--run-name",
        default="",
        help="Optional run name; defaults to timestamp",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    settings = get_settings()

    metric_names = [m.strip() for m in args.metrics.split(",") if m.strip()]
    selected_metrics = _select_metrics(metric_names)

    samples = load_eval_samples(
        curated_path=args.dataset,
        include_synthetic=bool(args.include_synthetic),
        synthetic_max_per_collection=max(1, int(args.synthetic_max_per_collection)),
        max_samples=max(0, int(args.max_samples)),
    )
    if not samples:
        raise ValueError("No evaluation samples loaded")

    outputs: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for sample in samples:
        try:
            outputs.append(run_rag_pipeline_for_sample(sample))
        except Exception as exc:
            failures.append(
                {
                    "id": sample.get("id"),
                    "question": sample.get("question"),
                    "error": str(exc),
                }
            )

    if not outputs:
        raise RuntimeError("All eval samples failed during RAG pipeline execution")

    ragas_rows = to_ragas_rows(outputs)
    ragas_dataset = Dataset.from_list(ragas_rows)

    judge_llm = ChatOpenAI(model=args.judge_model, api_key=settings.openai_api_key, temperature=0)
    judge_embeddings = _build_eval_embeddings(settings)

    run_config = None
    if RunConfig is not None:
        run_config = RunConfig(
            max_workers=max(1, int(settings.eval_max_workers)),
            timeout=max(10, int(settings.eval_timeout_seconds)),
        )

    eval_kwargs: dict[str, Any] = {
        "dataset": ragas_dataset,
        "metrics": selected_metrics,
        "llm": judge_llm,
        "embeddings": judge_embeddings,
    }
    if run_config is not None:
        eval_kwargs["run_config"] = run_config

    try:
        result = evaluate(**eval_kwargs)
    except TypeError:
        eval_kwargs.pop("run_config", None)
        result = evaluate(**eval_kwargs)

    rows, metric_summary = serialize_result(result)

    output_dir = ensure_output_dir(args.output_dir)
    run_name = args.run_name.strip() or dt.datetime.utcnow().strftime("ragas_%Y%m%d_%H%M%S")

    run_payload = {
        "run_name": run_name,
        "created_at_utc": dt.datetime.utcnow().isoformat() + "Z",
        "judge_model": args.judge_model,
        "metrics": metric_names,
        "sample_count_total": len(samples),
        "sample_count_scored": len(outputs),
        "sample_failures": failures,
        "metric_summary": metric_summary,
    }

    write_json(output_dir / f"{run_name}.summary.json", run_payload)
    write_json(output_dir / f"{run_name}.raw.json", rows)
    write_json(output_dir / f"{run_name}.pipeline_outputs.json", outputs)
    write_csv(output_dir / f"{run_name}.raw.csv", rows)

    print(f"Evaluation complete: {run_name}")
    print(f"Scored samples: {len(outputs)}/{len(samples)}")
    print(f"Summary file: {output_dir / (run_name + '.summary.json')}")
    if metric_summary:
        for key, value in metric_summary.items():
            print(f"  {key}: {value:.4f}")


if __name__ == "__main__":
    main()
