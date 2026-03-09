import json
import os
from pathlib import Path
from typing import Dict

import streamlit as st
from dotenv import load_dotenv
from llama_index.core import Settings, StorageContext, load_index_from_storage
from llama_index.core.postprocessor import MetadataReplacementPostProcessor
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama

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
    Settings.llm = Ollama(model=DEFAULT_OLLAMA_MODEL, request_timeout=120.0)


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

    if "index" not in st.session_state:
        if storage_dir.exists():
            sc = StorageContext.from_defaults(persist_dir=str(storage_dir))
            st.session_state.index = load_index_from_storage(sc)
        else:
            st.error("Index not found. Run indexer.py first.")
            st.stop()

    if "query_engine" not in st.session_state:
        st.session_state.query_engine = st.session_state.index.as_query_engine(
            similarity_top_k=3,
            node_postprocessors=[
                MetadataReplacementPostProcessor(target_metadata_key="window")
            ],
        )

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
                response = st.session_state.query_engine.query(prompt)
            except Exception as exc:
                st.error(f"Query failed: {exc}")
                return
            st.markdown(response.response)

        with st.expander("Sources and context"):
            seen = set()
            for node in response.source_nodes:
                window = node.node.metadata.get("window") or ""
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
                    seen.add(file_name)
                st.divider()

    st.session_state.messages.append(
        {"role": "assistant", "content": response.response}
    )


if __name__ == "__main__":
    main()
