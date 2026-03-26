#!/usr/bin/env bash
set -e
export EMBED_BATCH_SIZE=8
export EMBED_MAX_WORKERS=2
cd /Users/sonanhnguyen/Downloads/rag_wnt
exec .venv/bin/python backend/hf_legal_ingest.py \
  --collection legal \
  --filter health_legal \
  --batch-size 5 \
  --skip-summary true
