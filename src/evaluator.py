import argparse
from pathlib import Path

import numpy as np
from llama_index.core import Settings, StorageContext, load_index_from_storage
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama

DEFAULT_EMBED_MODEL = "BAAI/bge-small-en-v1.5"
DEFAULT_OLLAMA_MODEL = "qwen2.5:0.5b"


def cosine_similarity(v1, v2) -> float:
    return float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))


def try_bertscore(pred: str, ref: str) -> float:
    try:
        from bert_score import score as bert_score_compute
    except Exception:
        return float("nan")

    _, _, f1 = bert_score_compute([pred], [ref], lang="en", verbose=False)
    return float(f1.item())


def run_evaluation(question: str, ground_truth: str, data_dir: Path) -> None:
    Settings.embed_model = HuggingFaceEmbedding(model_name=DEFAULT_EMBED_MODEL)
    Settings.llm = Ollama(model=DEFAULT_OLLAMA_MODEL, request_timeout=120.0)

    storage_dir = data_dir / "storage"
    sc = StorageContext.from_defaults(persist_dir=str(storage_dir))
    index = load_index_from_storage(sc)
    query_engine = index.as_query_engine(similarity_top_k=2)

    response = query_engine.query(question)
    generated = response.response
    context = " ".join([node.node.get_content() for node in response.source_nodes])

    gt_emb = Settings.embed_model.get_text_embedding(ground_truth)
    ctx_emb = Settings.embed_model.get_text_embedding(context)
    semantic_recall = cosine_similarity(gt_emb, ctx_emb)

    bert_f1 = try_bertscore(generated, ground_truth)

    print("Evaluation results")
    print(f"Question: {question}")
    print(f"Answer preview: {generated[:120]}...")
    print(f"Semantic recall: {semantic_recall:.4f}")
    if np.isnan(bert_f1):
        print("BERTScore: not available (install bert-score)")
    else:
        print(f"BERTScore (F1): {bert_f1:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate RAG response quality.")
    parser.add_argument("--question", required=True)
    parser.add_argument("--ground-truth", required=True)
    parser.add_argument("--data-dir", default="data")
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parents[1]
    data_dir = (base_dir / args.data_dir).resolve()
    run_evaluation(args.question, args.ground_truth, data_dir)


if __name__ == "__main__":
    main()
