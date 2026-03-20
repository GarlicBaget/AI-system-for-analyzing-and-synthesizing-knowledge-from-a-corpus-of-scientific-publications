import argparse
from pathlib import Path
from typing import Iterable, List

import numpy as np
from llama_index.core import Settings, StorageContext, load_index_from_storage
from llama_index.core.callbacks import CallbackManager
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llm_utils import SafeOllama

from rag_strategies import build_metadata_filters, build_query_engine

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


def answer_relevancy(question: str, answer: str) -> float:
    q_emb = Settings.embed_model.get_text_embedding(question)
    a_emb = Settings.embed_model.get_text_embedding(answer)
    return cosine_similarity(q_emb, a_emb)


def answer_context_similarity(answer: str, context: str) -> float:
    a_emb = Settings.embed_model.get_text_embedding(answer)
    ctx_emb = Settings.embed_model.get_text_embedding(context)
    return cosine_similarity(a_emb, ctx_emb)


def run_evaluation(
    question: str,
    ground_truth: str,
    data_dir: Path,
    strategy: str,
    fusion_queries: int,
    fusion_mode: str,
    rerank_top_n: int,
    selected_files: Iterable[str],
) -> None:
    Settings.embed_model = HuggingFaceEmbedding(model_name=DEFAULT_EMBED_MODEL)
    Settings.callback_manager = CallbackManager([])
    Settings.llm = SafeOllama(model=DEFAULT_OLLAMA_MODEL, request_timeout=120.0)
    Settings.llm.callback_manager = Settings.callback_manager

    storage_dir = data_dir / "storage"
    sc = StorageContext.from_defaults(persist_dir=str(storage_dir))
    index = load_index_from_storage(sc)
    metadata_filters = build_metadata_filters(selected_files)
    query_engine = build_query_engine(
        index=index,
        strategy=strategy,
        similarity_top_k=2,
        fusion_queries=fusion_queries,
        fusion_mode=fusion_mode,
        rerank_top_n=rerank_top_n,
        metadata_filters=metadata_filters,
    )

    response = query_engine.query(question)
    generated = response.response
    context = " ".join([node.node.get_content() for node in response.source_nodes])

    gt_emb = Settings.embed_model.get_text_embedding(ground_truth)
    ctx_emb = Settings.embed_model.get_text_embedding(context)
    semantic_recall = cosine_similarity(gt_emb, ctx_emb)

    bert_f1 = try_bertscore(generated, ground_truth)
    relevancy = answer_relevancy(question, generated)
    groundedness = answer_context_similarity(generated, context)

    print("Evaluation results")
    print(f"Question: {question}")
    print(f"Answer preview: {generated[:120]}...")
    print(f"Semantic recall: {semantic_recall:.4f}")
    print(f"Answer relevancy: {relevancy:.4f}")
    print(f"Answer-context similarity: {groundedness:.4f}")
    if np.isnan(bert_f1):
        print("BERTScore: not available (install bert-score)")
    else:
        print(f"BERTScore (F1): {bert_f1:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate RAG response quality.")
    parser.add_argument("--question", required=True)
    parser.add_argument("--ground-truth", required=True)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--strategy", default="baseline")
    parser.add_argument("--fusion-queries", type=int, default=4)
    parser.add_argument(
        "--fusion-mode",
        default="reciprocal_rerank",
        choices=["reciprocal_rerank", "relative_score", "simple"],
    )
    parser.add_argument("--rerank-top-n", type=int, default=3)
    parser.add_argument("--files", nargs="*", default=[])
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parents[1]
    data_dir = (base_dir / args.data_dir).resolve()
    run_evaluation(
        question=args.question,
        ground_truth=args.ground_truth,
        data_dir=data_dir,
        strategy=args.strategy,
        fusion_queries=args.fusion_queries,
        fusion_mode=args.fusion_mode,
        rerank_top_n=args.rerank_top_n,
        selected_files=args.files,
    )


if __name__ == "__main__":
    main()
