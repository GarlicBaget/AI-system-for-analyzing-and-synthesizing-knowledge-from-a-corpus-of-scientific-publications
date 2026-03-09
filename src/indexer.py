import argparse
from pathlib import Path

from llama_index.core import Settings, SimpleDirectoryReader, StorageContext, VectorStoreIndex
from llama_index.core.node_parser import SentenceWindowNodeParser
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.readers.file import PyMuPDFReader

DEFAULT_EMBED_MODEL = "BAAI/bge-small-en-v1.5"


def build_index(data_dir: Path) -> Path:
    pdf_dir = data_dir / "raw_pdfs"
    storage_dir = data_dir / "storage"
    storage_dir.mkdir(parents=True, exist_ok=True)

    if not pdf_dir.exists():
        raise FileNotFoundError(f"PDF folder not found: {pdf_dir}")

    Settings.embed_model = HuggingFaceEmbedding(model_name=DEFAULT_EMBED_MODEL)

    node_parser = SentenceWindowNodeParser.from_defaults(
        window_size=3,
        window_metadata_key="window",
        original_text_metadata_key="original_text",
    )

    loader = PyMuPDFReader()
    documents = SimpleDirectoryReader(
        str(pdf_dir),
        file_extractor={".pdf": loader},
    ).load_data()

    if not documents:
        raise RuntimeError("No documents loaded. Add PDFs to data/raw_pdfs first.")

    nodes = node_parser.get_nodes_from_documents(documents)
    index = VectorStoreIndex(nodes)
    index.storage_context.persist(persist_dir=str(storage_dir))
    return storage_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a Sentence-Window RAG index.")
    parser.add_argument("--data-dir", default="data")
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parents[1]
    data_dir = (base_dir / args.data_dir).resolve()
    data_dir.mkdir(parents=True, exist_ok=True)

    storage_dir = build_index(data_dir)
    print(f"Index saved to {storage_dir}")


if __name__ == "__main__":
    main()
