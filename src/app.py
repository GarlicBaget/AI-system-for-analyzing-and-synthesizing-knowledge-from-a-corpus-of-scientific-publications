import json
import os
from pathlib import Path
from typing import Dict, List

import streamlit as st
from dotenv import load_dotenv
from llama_index.core import Settings, StorageContext, load_index_from_storage
from llama_index.core.callbacks import CallbackManager
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llm_utils import SafeOllama

from rag_strategies import (
    build_agent,
    build_metadata_filters,
    build_multi_agent,
    build_query_engine,
    config_from_inputs,
    config_to_dict,
)

load_dotenv()

DEFAULT_EMBED_MODEL = "BAAI/bge-small-en-v1.5"
DEFAULT_OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:0.5b")


def _load_metadata(path: Path) -> Dict:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _init_settings() -> None:
    Settings.embed_model = HuggingFaceEmbedding(model_name=DEFAULT_EMBED_MODEL)
    Settings.callback_manager = CallbackManager([])
    Settings.llm = SafeOllama(model=DEFAULT_OLLAMA_MODEL, request_timeout=120.0)
    Settings.llm.callback_manager = Settings.callback_manager


def main() -> None:
    st.set_page_config(page_title="Open Science RAG Assistant", layout="wide")
    st.title("Open Science RAG Assistant")
    st.caption("Search and summarize open-access papers with citations.")

    base_dir = Path(__file__).resolve().parents[1]
    data_dir = base_dir / "data"
    storage_dir = data_dir / "storage"
    metadata_path = data_dir / "metadata.json"

    metadata = _load_metadata(metadata_path)
    _init_settings()

    title_lookup: Dict[str, str] = {}
    for file_name, info in metadata.items():
        title = info.get("title") or file_name
        title_lookup[title] = file_name

    st.sidebar.header("Retrieval settings")
    strategy = st.sidebar.selectbox(
        "RAG strategy",
        options=["baseline", "fusion", "rerank", "agent", "multi_agent"],
        format_func=lambda v: {
            "baseline": "Baseline vector",
            "fusion": "Query fusion",
            "rerank": "LLM rerank",
            "agent": "Agent mode",
            "multi_agent": "Multi-agent mode",
        }.get(v, v),
    )
    similarity_top_k = st.sidebar.slider("Top-K", min_value=1, max_value=10, value=3)
    fusion_queries = st.sidebar.slider(
        "Fusion queries", min_value=2, max_value=8, value=4
    )
    fusion_mode = st.sidebar.selectbox(
        "Fusion mode",
        options=["reciprocal_rerank", "relative_score", "simple"],
    )
    rerank_top_n = st.sidebar.slider(
        "Rerank Top-N", min_value=1, max_value=10, value=min(3, similarity_top_k)
    )
    rerank_top_n = min(rerank_top_n, similarity_top_k)

    selected_titles: List[str] = st.sidebar.multiselect(
        "Filter papers",
        options=sorted(title_lookup.keys()),
        default=[],
    )
    selected_files = [title_lookup[t] for t in selected_titles]

    if "index" not in st.session_state:
        if storage_dir.exists() and any(storage_dir.iterdir()):
            try:
                sc = StorageContext.from_defaults(persist_dir=str(storage_dir))
                st.session_state.index = load_index_from_storage(sc)
            except Exception:
                st.error("Index storage is missing or corrupted. Rebuild it with indexer.py.")
                st.stop()
        else:
            st.error("Index not found. Run indexer.py first.")
            st.stop()

    config = config_from_inputs(
        strategy=strategy,
        similarity_top_k=similarity_top_k,
        fusion_queries=fusion_queries,
        fusion_mode=fusion_mode,
        rerank_top_n=rerank_top_n,
        selected_files=selected_files,
    )
    config_dict = config_to_dict(config)

    if st.session_state.get("query_config") != config_dict:
        metadata_filters = build_metadata_filters(selected_files)
        if strategy == "agent":
            st.session_state.agent = build_agent(
                index=st.session_state.index,
                similarity_top_k=similarity_top_k,
                fusion_queries=fusion_queries,
                fusion_mode=fusion_mode,
                metadata_filters=metadata_filters,
            )
            st.session_state.query_engine = None
            st.session_state.multi_agent = None
        elif strategy == "multi_agent":
            st.session_state.multi_agent = build_multi_agent(
                index=st.session_state.index,
                similarity_top_k=similarity_top_k,
                fusion_queries=fusion_queries,
                fusion_mode=fusion_mode,
                metadata_filters=metadata_filters,
            )
            st.session_state.query_engine = None
            st.session_state.agent = None
        else:
            st.session_state.query_engine = build_query_engine(
                index=st.session_state.index,
                strategy=strategy,
                similarity_top_k=similarity_top_k,
                fusion_queries=fusion_queries,
                fusion_mode=fusion_mode,
                rerank_top_n=rerank_top_n,
                metadata_filters=metadata_filters,
            )
            st.session_state.agent = None
            st.session_state.multi_agent = None
        st.session_state.query_config = config_dict

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    prompt = st.chat_input("Ask a question about the indexed papers...")
    if not prompt:
        return

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching and synthesizing..."):
            try:
                if strategy == "agent":
                    response = st.session_state.agent.chat(prompt)
                elif strategy == "multi_agent":
                    response = st.session_state.multi_agent.chat(prompt)
                else:
                    response = st.session_state.query_engine.query(prompt)
            except Exception as exc:
                st.error(f"Query failed: {exc}")
                return
            answer = getattr(response, "response", None) or str(response)
            st.markdown(answer)

        with st.expander("Sources and context"):
            seen = set()
            source_nodes = getattr(response, "source_nodes", []) or []
            if not source_nodes:
                st.caption("No source nodes returned for this response.")
            for node in source_nodes:
                window = node.node.metadata.get("window") or ""
                if window:
                    st.write(f"Context window: {window}")

                file_name = node.metadata.get("file_name")
                if file_name and file_name in metadata and file_name not in seen:
                    info = metadata[file_name]
                    st.caption(f"Source: {info.get('title', 'unknown')}")
                    if info.get("year"):
                        st.caption(f"Year: {info.get('year')}")
                    if info.get("doi"):
                        st.caption(f"DOI: {info.get('doi')}")
                    if info.get("pdf_url"):
                        st.caption(f"PDF: {info.get('pdf_url')}")
                    if info.get("url"):
                        st.caption(f"URL: {info.get('url')}")
                    seen.add(file_name)
                st.divider()

    st.session_state.messages.append(
        {"role": "assistant", "content": answer}
    )


if __name__ == "__main__":
    main()
