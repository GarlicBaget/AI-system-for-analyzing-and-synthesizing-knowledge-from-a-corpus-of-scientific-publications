import os
from llama_index.core import VectorStoreIndex, StorageContext, Settings, SimpleDirectoryReader
from llama_index.core.node_parser import SentenceWindowNodeParser
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.readers.file import PyMuPDFReader

def build_advanced_index():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    pdf_dir = os.path.join(base_dir, "data", "raw_pdfs")
    storage_dir = os.path.join(base_dir, "data", "storage")

    # Бесплатная модель эмбеддингов
    Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
    
    # НЕТРИВИАЛЬНЫЙ ПАРСЕР: Разбиваем на предложения + окно контекста
    node_parser = SentenceWindowNodeParser.from_defaults(
        window_size=3,
        window_metadata_key="window",
        original_text_metadata_key="original_text",
    )

    loader = PyMuPDFReader()
    documents = SimpleDirectoryReader(pdf_dir, file_extractor={".pdf": loader}).load_data()
    
    # Создаем узлы с использованием продвинутого парсера
    nodes = node_parser.get_nodes_from_documents(documents)
    
    index = VectorStoreIndex(nodes)
    index.storage_context.persist(persist_dir=storage_dir)
    print("Продвинутый индекс создан.")

if __name__ == "__main__":
    build_advanced_index()