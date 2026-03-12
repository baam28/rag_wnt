import asyncio
from ingest import _load_pdf_docling, count_tokens, semantic_parent_chunks
from langchain_core.documents import Document
from supervisor import get_intent_from_supervisor
from classifier import classify_intent

doc = Document(page_content="# Hướng dẫn\n\n## Tác dụng\nThuốc này có tác dụng giảm đau.\n\n## Cấp phép\nTheo thông tư 01, thuốc phải đăng ký.", metadata={"source": "test.md"})
chunks = semantic_parent_chunks([doc], 100, None)
print(f"Chunks produced: {len(chunks)}")
for i, c in enumerate(chunks):
    print(f"Chunk {i} meta: {c.metadata} content: {c.page_content[:30]}")

print("\n--- Testing Supervisor Intent ---")
intent = get_intent_from_supervisor("Paracetamol giá bao nhiêu và theo quy định thì thuốc này do cơ quan nào cấp phép?")
print(f"Supervisor intent: {intent}")

print("\n--- Testing Fallback Intent ---")
fallback = classify_intent("Paracetamol giá bao nhiêu và theo quy định thì thuốc này do cơ quan nào cấp phép?")
print(f"Fallback intent: {fallback}")
