"""Ingest router: file upload, background jobs, job status polling."""

import asyncio
import threading
import uuid
from pathlib import Path
from typing import Any

from fastapi import APIRouter, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import JSONResponse

from config import get_settings
from deps import IngestResponse, IngestJobStatusResponse
from ingest import ingest_file

router = APIRouter(tags=["ingest"])


# ---------------------------------------------------------------------------
# In-memory job registry (thread-safe)
# ---------------------------------------------------------------------------

_ingest_jobs: dict[str, dict[str, Any]] = {}
_ingest_jobs_lock = threading.Lock()


def _get_job(job_id: str) -> dict[str, Any] | None:
    with _ingest_jobs_lock:
        return _ingest_jobs.get(job_id)


def _set_job(job_id: str, data: dict[str, Any]) -> None:
    with _ingest_jobs_lock:
        _ingest_jobs[job_id] = {**_ingest_jobs.get(job_id, {}), **data}


class IngestCancelled(Exception):
    """Raised when the user cancels an ingest job."""


def _run_ingest_job(job_id: str, target_path: Path, collection_name: str, skip_summary: bool = False) -> None:
    """Run ingest_file in a background thread and update job state for polling."""
    _set_job(job_id, {"status": "running", "phase": "start", "message": "Đang bắt đầu...", "current": 0, "total": 1})

    def progress_cb(step: str, msg: str, current: int, total: int) -> None:
        job = _get_job(job_id)
        if job and job.get("cancelled"):
            raise IngestCancelled()
        _set_job(job_id, {"phase": step, "message": msg, "current": current, "total": total or 1})

    try:
        result = ingest_file(target_path, collection_name=collection_name, on_progress=progress_cb, skip_summary=skip_summary)
        if "error" in result:
            _set_job(job_id, {"status": "error", "error": result["error"]})
            return
        _set_job(
            job_id,
            {
                "status": "done",
                "phase": "done",
                "message": "Hoàn thành",
                "result": IngestResponse(
                    file=result.get("file", str(target_path)),
                    collection_name=result.get("collection_name", collection_name),
                    num_parents=result.get("num_parents", 0),
                    num_children=result.get("num_children", 0),
                    total_chunks_in_db=result.get("total_chunks_in_db", 0),
                ),
            },
        )
    except IngestCancelled:
        _set_job(job_id, {"status": "cancelled", "error": "Cancelled by user"})
    except Exception as e:
        _set_job(job_id, {"status": "error", "error": str(e)})


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post("/ingest-file", response_model=IngestResponse)
async def ingest_file_endpoint(
    file: UploadFile = File(...),
    collection_name: str = Form("rag_chatbot"),
    skip_summary: str = Form("false"),
    async_mode: bool = Query(False, alias="async"),
):
    """Ingest a single uploaded file into the vector store.
    Use ?async=true for background processing and poll GET /ingest-jobs/{job_id}.
    """
    settings = get_settings()
    upload_dir = Path(settings.upload_dir)
    upload_dir.mkdir(parents=True, exist_ok=True)
    target_path = upload_dir / (file.filename or "document")

    try:
        contents = await file.read()
        with open(target_path, "wb") as f:
            f.write(contents)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save uploaded file: {e}")

    skip = skip_summary.strip().lower() in ("true", "1", "yes")

    if async_mode:
        job_id = str(uuid.uuid4())
        _set_job(job_id, {"status": "pending", "phase": "pending", "message": "Đang xếp hàng...", "current": 0, "total": 1})
        threading.Thread(
            target=_run_ingest_job,
            args=(job_id, target_path, collection_name),
            kwargs={"skip_summary": skip},
            daemon=True,
        ).start()
        return JSONResponse(content={"job_id": job_id}, status_code=202)

    try:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: ingest_file(target_path, collection_name=collection_name, skip_summary=skip),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])

    return IngestResponse(
        file=result.get("file", str(target_path)),
        collection_name=result.get("collection_name", collection_name),
        num_parents=result.get("num_parents", 0),
        num_children=result.get("num_children", 0),
        total_chunks_in_db=result.get("total_chunks_in_db", 0),
    )


@router.get("/ingest-jobs/{job_id}", response_model=IngestJobStatusResponse)
def get_ingest_job_status(job_id: str):
    """Poll ingest job progress."""
    job = _get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return IngestJobStatusResponse(
        job_id=job_id,
        status=job.get("status", "pending"),
        phase=job.get("phase"),
        current=job.get("current"),
        total=job.get("total"),
        message=job.get("message"),
        result=job.get("result"),
        error=job.get("error"),
    )


@router.post("/ingest-jobs/{job_id}/cancel")
def cancel_ingest_job(job_id: str):
    """Request cancellation of an ingest job."""
    job = _get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.get("status") in ("done", "error", "cancelled"):
        return {"message": "Job already finished", "status": job.get("status")}
    _set_job(job_id, {"cancelled": True})
    return {"message": "Cancellation requested", "job_id": job_id}
