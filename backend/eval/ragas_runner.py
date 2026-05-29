"""CLI runner for RAGAS evaluation against the local RAG pipeline."""

from __future__ import annotations

import argparse
import datetime as dt
import time
from typing import Any

from datasets import Dataset
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_voyageai import VoyageAIEmbeddings

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


def _percentile(vals: list[float], p: float) -> float | None:
    """Return the p-th percentile (0–1) of a sorted list using linear interpolation."""
    if not vals:
        return None
    if len(vals) == 1:
        return vals[0]
    rank = (len(vals) - 1) * p
    low = int(rank)
    high = min(low + 1, len(vals) - 1)
    return vals[low] + (vals[high] - vals[low]) * (rank - low)


def _build_eval_embeddings(settings):
    model = settings.embedding_model
    if model.startswith("voyage-"):
        if settings.voyage_api_key:
            return VoyageAIEmbeddings(model=model, voyage_api_key=settings.voyage_api_key)
        # No Voyage key — fall back to OpenAI for the RAGAS judge embeddings
        return OpenAIEmbeddings(model="text-embedding-3-small", api_key=settings.openai_api_key)
    if model.startswith("text-embedding"):
        return OpenAIEmbeddings(model=model, api_key=settings.openai_api_key)
    return HuggingFaceEmbeddings(
        model_name=model,
        model_kwargs={"trust_remote_code": True, "device": "cpu"},
    )


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
    run_started_at = time.perf_counter()
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
    sample_latencies_sec: list[float] = []
    total = len(samples)
    for i, sample in enumerate(samples):
        sample_id = sample.get("id", f"sample_{i}")
        print(f"[{i+1}/{total}] {sample_id} ...", flush=True)
        sample_started_at = time.perf_counter()
        try:
            output = run_rag_pipeline_for_sample(sample)
            outputs.append(output)
            sample_latencies_sec.append(time.perf_counter() - sample_started_at)
            answer_preview = (output["answer"] or "")[:80].replace("\n", " ")
            print(f"  ✓ {answer_preview}", flush=True)
        except Exception as exc:
            print(f"  ✗ ERROR: {exc}", flush=True)
            failures.append(
                {
                    "id": sample.get("id"),
                    "question": sample.get("question"),
                    "error": str(exc),
                }
            )

    if not outputs:
        raise RuntimeError("All eval samples failed during RAG pipeline execution")

    print(f"\nRAG pipeline done: {len(outputs)} scored, {len(failures)} failed.", flush=True)
    print("Running RAGAS judge evaluation...", flush=True)

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

    total_run_seconds = time.perf_counter() - run_started_at
    sorted_latencies = sorted(sample_latencies_sec)

    benchmark = {
        "total_run_seconds": round(total_run_seconds, 4),
        "scored_samples": len(sample_latencies_sec),
        "avg_latency_seconds": round(sum(sample_latencies_sec) / len(sample_latencies_sec), 4)
        if sample_latencies_sec
        else None,
        "p50_latency_seconds": round(_percentile(sorted_latencies, 0.50), 4)
        if sorted_latencies
        else None,
        "p95_latency_seconds": round(_percentile(sorted_latencies, 0.95), 4)
        if sorted_latencies
        else None,
    }

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
        "benchmark": benchmark,
        "metric_summary": metric_summary,
    }

    write_json(output_dir / f"{run_name}.summary.json", run_payload)
    write_json(output_dir / f"{run_name}.raw.json", rows)
    write_json(output_dir / f"{run_name}.pipeline_outputs.json", outputs)
    write_csv(output_dir / f"{run_name}.raw.csv", rows)

    print(f"Evaluation complete: {run_name}")
    print(f"Scored samples: {len(outputs)}/{len(samples)}")
    print(f"Summary file: {output_dir / (run_name + '.summary.json')}")
    print("Benchmark:")
    print(f"  total_run_seconds: {benchmark['total_run_seconds']:.4f}")
    if benchmark["avg_latency_seconds"] is not None:
        print(f"  avg_latency_seconds: {benchmark['avg_latency_seconds']:.4f}")
    if benchmark["p50_latency_seconds"] is not None:
        print(f"  p50_latency_seconds: {benchmark['p50_latency_seconds']:.4f}")
    if benchmark["p95_latency_seconds"] is not None:
        print(f"  p95_latency_seconds: {benchmark['p95_latency_seconds']:.4f}")
    if metric_summary:
        for key, value in metric_summary.items():
            print(f"  {key}: {value:.4f}")


if __name__ == "__main__":
    main()
