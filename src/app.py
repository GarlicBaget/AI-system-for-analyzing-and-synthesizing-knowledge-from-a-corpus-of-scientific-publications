import streamlit as st
import os
import json
from llama_index.core import StorageContext, load_index_from_storage, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.core.postprocessor import MetadataReplacementPostProcessor
from dotenv import load_dotenv

load_dotenv()

# 1. Настройка моделей (те же, что в evaluator.py)
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
Settings.llm = Ollama(model="qwen2.5:0.5b", request_timeout=120.0)

st.set_page_config(page_title="Advanced Sci-AI Analyst", layout="wide")
st.title("🔬 Advanced AI Analyst")
st.subheader("Sentence Window RAG + Local LLM")

# Пути к данным
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
storage_dir = os.path.join(base_dir, "data", "storage")
metadata_path = os.path.join(base_dir, "data", "metadata.json")

# Загружаем метаданные для ссылок
if os.path.exists(metadata_path):
    with open(metadata_path, "r", encoding='utf-8') as f:
        metadata = json.load(f)
else:
    metadata = {}

# Инициализация индекса (загружаем то, что создал indexer.py)
if "index" not in st.session_state:
    if os.path.exists(storage_dir):
        sc = StorageContext.from_defaults(persist_dir=storage_dir)
        st.session_state.index = load_index_from_storage(sc)
    else:
        st.error("База данных не найдена! Сначала запустите indexer.py")

# Инициализация "умного" движка
if "query_engine" not in st.session_state and "index" in st.session_state:
    st.session_state.query_engine = st.session_state.index.as_query_engine(
        similarity_top_k=3,
        # Нетривиальный RAG: заменяем найденное предложение на контекстное окно
        node_postprocessors=[
            MetadataReplacementPostProcessor(target_metadata_key="window")
        ]
    )

# Интерфейс чата
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Задайте вопрос по корпусу статей..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    if "query_engine" in st.session_state:
        with st.chat_message("assistant"):
            with st.spinner("ИИ анализирует статьи..."):
                response = st.session_state.query_engine.query(prompt)
                st.markdown(response.response)
            
            # Вывод источников
            with st.expander("Посмотреть источники и контекст"):
                seen_docs = set()
                for node in response.source_nodes:
                    # Показываем, какое именно окно контекста увидел ИИ
                    st.write(f"**Контекстное окно:** {node.node.metadata.get('window')}")
                    
                    # Пытаемся найти красивое название статьи в метаданных
                    file_name = node.metadata.get('file_name')
                    if file_name in metadata and file_name not in seen_docs:
                        info = metadata[file_name]
                        st.caption(f"Источник: {info['title']} ({info.get('year', '')})")
                        st.caption(f"DOI: {info.get('doi', '')}")
                        seen_docs.add(file_name)
                    st.divider()

        st.session_state.messages.append({"role": "assistant", "content": response.response})