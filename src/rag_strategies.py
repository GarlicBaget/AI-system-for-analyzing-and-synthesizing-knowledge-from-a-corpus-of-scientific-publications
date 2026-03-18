from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

from llama_index.core import Settings
from llama_index.core.agent import ReActAgent
from llama_index.core.indices.vector_store.retrievers.retriever import (
    VectorIndexRetriever,
)
from llama_index.core.postprocessor import LLMRerank, MetadataReplacementPostProcessor
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.core.retrievers.fusion_retriever import FUSION_MODES
from llama_index.core.tools import QueryEngineTool
from llama_index.core.vector_stores import FilterCondition, MetadataFilter, MetadataFilters


@dataclass(frozen=True)
class QueryConfig:
    strategy: str
    similarity_top_k: int
    fusion_queries: int
    fusion_mode: str
    rerank_top_n: int
    selected_files: tuple[str, ...]


def build_metadata_filters(selected_files: Iterable[str]) -> Optional[MetadataFilters]:
    files = [f for f in selected_files if f]
    if not files:
        return None
    filters = [MetadataFilter(key="file_name", value=f) for f in files]
    return MetadataFilters(filters=filters, condition=FilterCondition.OR)


def _base_postprocessors(enable_rerank: bool, rerank_top_n: int) -> List:
    postprocessors = [
        MetadataReplacementPostProcessor(target_metadata_key="window"),
    ]
    if enable_rerank:
        postprocessors.append(LLMRerank(top_n=rerank_top_n))
    return postprocessors


def build_query_engine(
    index,
    strategy: str,
    similarity_top_k: int,
    fusion_queries: int,
    fusion_mode: str,
    rerank_top_n: int,
    metadata_filters: Optional[MetadataFilters],
) -> RetrieverQueryEngine:
    vector_retriever = VectorIndexRetriever(
        index=index,
        similarity_top_k=similarity_top_k,
        filters=metadata_filters,
    )

    if strategy == "fusion":
        retriever = QueryFusionRetriever(
            retrievers=[vector_retriever],
            similarity_top_k=similarity_top_k,
            num_queries=fusion_queries,
            mode=FUSION_MODES(fusion_mode),
        )
        postprocessors = _base_postprocessors(enable_rerank=False, rerank_top_n=rerank_top_n)
    elif strategy == "rerank":
        retriever = vector_retriever
        postprocessors = _base_postprocessors(enable_rerank=True, rerank_top_n=rerank_top_n)
    else:
        retriever = vector_retriever
        postprocessors = _base_postprocessors(enable_rerank=False, rerank_top_n=rerank_top_n)

    return RetrieverQueryEngine.from_args(
        retriever=retriever,
        node_postprocessors=postprocessors,
    )


def build_agent(
    index,
    similarity_top_k: int,
    fusion_queries: int,
    fusion_mode: str,
    metadata_filters: Optional[MetadataFilters],
) -> ReActAgent:
    base_engine = build_query_engine(
        index=index,
        strategy="baseline",
        similarity_top_k=similarity_top_k,
        fusion_queries=fusion_queries,
        fusion_mode=fusion_mode,
        rerank_top_n=similarity_top_k,
        metadata_filters=metadata_filters,
    )
    fusion_engine = build_query_engine(
        index=index,
        strategy="fusion",
        similarity_top_k=similarity_top_k,
        fusion_queries=fusion_queries,
        fusion_mode=fusion_mode,
        rerank_top_n=similarity_top_k,
        metadata_filters=metadata_filters,
    )

    tools = [
        QueryEngineTool.from_defaults(
            name="vector_search",
            description="Vector search over the selected papers.",
            query_engine=base_engine,
        ),
        QueryEngineTool.from_defaults(
            name="fusion_search",
            description="Multi-query fusion retrieval for harder questions.",
            query_engine=fusion_engine,
        ),
    ]
    return ReActAgent.from_tools(tools=tools, llm=Settings.llm, verbose=False)


def config_from_inputs(
    strategy: str,
    similarity_top_k: int,
    fusion_queries: int,
    fusion_mode: str,
    rerank_top_n: int,
    selected_files: Iterable[str],
) -> QueryConfig:
    return QueryConfig(
        strategy=strategy,
        similarity_top_k=similarity_top_k,
        fusion_queries=fusion_queries,
        fusion_mode=fusion_mode,
        rerank_top_n=rerank_top_n,
        selected_files=tuple(sorted(selected_files)),
    )


def config_to_dict(config: QueryConfig) -> Dict:
    return {
        "strategy": config.strategy,
        "similarity_top_k": config.similarity_top_k,
        "fusion_queries": config.fusion_queries,
        "fusion_mode": config.fusion_mode,
        "rerank_top_n": config.rerank_top_n,
        "selected_files": list(config.selected_files),
    }
