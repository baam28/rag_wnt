"""Streamlit UI: document ingest and RAG chat."""
import sys
from pathlib import Path
import shutil

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import get_settings
from ingest import ingest_file
from retriever import retrieve
from app import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE, build_context_block
from langchain_openai import ChatOpenAI


st.set_page_config(
    page_title="RAG Chatbot | Semantic Q&A",
    page_icon="📚",
    layout="centered",
    initial_sidebar_state="expanded",
)

settings = get_settings()

# --- Sidebar: settings, ingest, database ---

with st.sidebar:
    st.header("Settings")
    if settings.openai_api_key:
        st.caption("OPENAI_API_KEY OK")
    else:
        st.warning("OPENAI_API_KEY not set.")

    st.divider()

    st.subheader("Ingest documents")
    uploaded_files = st.file_uploader(
        "Upload PDF or DOCX",
        type=["pdf", "docx", "doc"],
        accept_multiple_files=True,
        help="Add files to the knowledge base",
    )
    collection_name = st.text_input(
        "Collection name",
        value="rag_chatbot",
        help="Letters, numbers, underscore, hyphen. Spaces are converted to underscore.",
    )
    skip_metadata = st.checkbox("Skip LLM summary (faster ingest)", value=False)

    if st.button("Ingest selected files"):
        if not uploaded_files:
            st.warning("Select at least one file.")
        else:
            path = Path("uploads")
            path.mkdir(exist_ok=True)
            progress_placeholder = st.empty()

            def on_progress(step, msg, current, total):
                progress_placeholder.caption(f"**[{step}]** {msg}")

            for uploaded in uploaded_files:
                try:
                    file_path = path / uploaded.name
                    with open(file_path, "wb") as f:
                        f.write(uploaded.getvalue())
                    with st.spinner("Processing..."):
                        result = ingest_file(
                            file_path,
                            collection_name=collection_name,
                            skip_metadata_llm=skip_metadata,
                            on_progress=on_progress,
                        )
                    if "error" in result:
                        st.error(f"{uploaded.name}: {result['error']}")
                    else:
                        coll = result.get("collection_name", collection_name)
                        st.success(
                            f"{uploaded.name}: {result['num_parents']} parent chunks, "
                            f"{result['num_children']} child chunks → collection `{coll}`"
                        )
                except Exception as e:
                    st.error(f"{uploaded.name}: {e}")
            progress_placeholder.empty()

    st.divider()

    st.subheader("Database")
    if st.button("Clear entire DB"):
        try:
            db_path = settings.persist_dir
            if not isinstance(db_path, Path):
                db_path = Path(db_path)
            if db_path.exists():
                shutil.rmtree(db_path)
            db_path.mkdir(parents=True, exist_ok=True)
            st.success(f"Cleared database at `{db_path}`. You can ingest again.")
        except Exception as e:
            st.error(f"Error: {e}")

# --- Main: chat ---

st.title("Chatbot")

if not settings.openai_api_key:
    st.info("Set `OPENAI_API_KEY` in `.env` to get answers.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("Sources"):
                for s in msg["sources"]:
                    st.write(f"**[{s.get('rank')}]** {s.get('source', '')}")
                    st.caption(s.get("summary", "")[:200])

if prompt := st.chat_input("Ask a question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    context_list = []
    answer = ""
    sources = []
    with st.chat_message("assistant"):
        placeholder = st.empty()
        with st.spinner("Searching and answering..."):
            try:
                context_list = retrieve(prompt)

                if not context_list:
                    answer = "I don't have enough specific information to answer this. (No relevant context found.)"
                    placeholder.markdown(answer)
                else:
                    if not settings.openai_api_key:
                        placeholder.markdown("Error: OPENAI_API_KEY not set.")
                    else:
                        context_block = build_context_block(context_list)
                        user_msg = USER_PROMPT_TEMPLATE.format(context=context_block, question=prompt)
                        llm = ChatOpenAI(
                            model=settings.llm_model,
                            api_key=settings.openai_api_key,
                            temperature=0.2,
                            streaming=True,
                        )
                        for chunk in llm.stream(
                            [
                                {"role": "system", "content": SYSTEM_PROMPT},
                                {"role": "user", "content": user_msg},
                            ]
                        ):
                            token = getattr(chunk, "content", None) or ""
                            if not token:
                                continue
                            answer += token
                            placeholder.markdown(answer)

                sources = [
                    {
                        "rank": c.get("rank"),
                        "source": c.get("source"),
                        "summary": c.get("summary", "")[:200],
                        "collection_name": c.get("collection_name"),
                    }
                    for c in context_list
                ]
                if sources:
                    used_collection = sources[0].get("collection_name") or "unknown"
                    st.caption(f"From collection: `{used_collection}`")
                    with st.expander("Sources"):
                        for s in sources:
                            coll = s.get("collection_name") or used_collection
                            st.write(f"**[{s['rank']}]** {s['source']}  _(collection: {coll})_")
                            st.caption(s["summary"])
            except ValueError as e:
                st.error(str(e))
            except Exception as e:
                st.error(f"Error: {e}")

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources,
    })
