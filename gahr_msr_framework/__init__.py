"""
GAHR-MSR Framework: Graph-Augmented Hybrid Retrieval and Multi-Stage Re-ranking

A standalone implementation of the GAHR-MSR framework for high-fidelity RAG systems.

This framework implements three key phases:
1. Graph-Aware Chunking and Indexing (uses RAGAnything + LightRAG)
2. High-Recall Hybrid Candidate Retrieval (dense + sparse + RRF)
3. High-Precision Cascaded Re-ranking (ColBERT)

Based on the research paper:
"Graph-Augmented Hybrid Retrieval and Multi-Stage Re-ranking for RAG Systems"

Example:
    >>> from gahr_msr_framework import GAHRMSRQuery, GAHRMSRConfig
    >>> from raganything import RAGAnything
    >>>
    >>> # Initialize RAGAnything
    >>> rag = RAGAnything(...)
    >>> await rag.process_document_complete("doc.pdf", "./output")
    >>>
    >>> # Create GAHR-MSR interface
    >>> config = GAHRMSRConfig(colbert_device="cuda", final_top_k=5)
    >>> gahr = GAHRMSRQuery(rag, config)
    >>>
    >>> # Query with multi-stage re-ranking
    >>> result = await gahr.query("What is the main contribution?", mode="hybrid")
"""

__version__ = "1.0.0"
__author__ = "GAHR-MSR Implementation"

from gahr_msr_framework.colbert_reranker import ColBERTReranker, ColBERTConfig
from gahr_msr_framework.hybrid_retrieval import (
    HybridRetriever,
    HybridRetrievalConfig,
    RRFFusion,
    RRFConfig,
    GraphAwareFilter
)
from gahr_msr_framework.gahr_msr import GAHRMSRQuery, GAHRMSRConfig

__all__ = [
    # Main Interface
    "GAHRMSRQuery",
    "GAHRMSRConfig",
    # ColBERT Re-ranker
    "ColBERTReranker",
    "ColBERTConfig",
    # Hybrid Retrieval
    "HybridRetriever",
    "HybridRetrievalConfig",
    "RRFFusion",
    "RRFConfig",
    "GraphAwareFilter",
]
