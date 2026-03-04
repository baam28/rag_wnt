# Changelog

## Unreleased

- Switch PDF ingestion to a Docling-based pipeline that produces structured Markdown (headings, lists, tables) for RAG, removing the old pdfplumber/PyMuPDF + vision-based path.
- Enable auto chunking strategy selection (`auto`) so the system chooses between paragraph-, sentence-, and semantic token-based chunking based on document statistics.

