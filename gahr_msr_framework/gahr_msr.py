"""
GAHR-MSR Query Interface

Main interface for Graph-Augmented Hybrid Retrieval and Multi-Stage Re-ranking.

Integrates three key phases:
1. Graph-Aware Chunking and Indexing (pre-processing)
2. High-Recall Hybrid Candidate Retrieval (dense + sparse + RRF)
3. High-Precision Cascaded Re-ranking (ColBERT)

Based on the paper:
"Graph-Augmented Hybrid Retrieval and Multi-Stage Re-ranking for RAG Systems"
"""

from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass, field
import logging
from pathlib import Path

from gahr_msr_framework.colbert_reranker import ColBERTReranker, ColBERTConfig
from gahr_msr_framework.hybrid_retrieval import (
    HybridRetriever,
    HybridRetrievalConfig,
    GraphAwareFilter
)


logger = logging.getLogger(__name__)


@dataclass
class GAHRMSRConfig:
    """Configuration for GAHR-MSR framework"""

    # Phase 2: Hybrid Retrieval Configuration
    # ---
    dense_top_k: int = 100
    """Number of candidates from dense search"""

    sparse_top_k: int = 100
    """Number of candidates from sparse search"""

    hybrid_top_k: int = 100
    """Number of candidates after RRF fusion"""

    rrf_k: int = 60
    """RRF fusion constant (typically 60)"""

    # Phase 3: Re-ranking Configuration
    # ---
    enable_colbert_reranking: bool = True
    """Enable ColBERT re-ranking stage"""

    colbert_top_k: int = 20
    """Number of candidates to re-rank with ColBERT"""

    final_top_k: int = 5
    """Number of final results to return"""

    colbert_model: str = "colbert-ir/colbertv2.0"
    """ColBERT model name"""

    colbert_device: str = "cpu"
    """Device for ColBERT (cpu, cuda, mps)"""

    # Graph-Aware Filtering Configuration
    # ---
    enable_graph_filtering: bool = True
    """Enable graph-based pre-filtering"""

    graph_filter_mode: str = "any"
    """Entity matching mode: 'any' or 'all'"""

    # General Configuration
    # ---
    enable_intermediate_reranking: bool = False
    """Enable intermediate re-ranking before ColBERT (for very large candidate sets)"""

    intermediate_reranker_top_k: int = 50
    """Number of candidates for intermediate re-ranking"""


class GAHRMSRQuery:
    """
    GAHR-MSR Query Interface

    Main class for performing queries with the GAHR-MSR framework.
    Integrates hybrid retrieval with graph-aware filtering and
    multi-stage re-ranking.

    Example:
        >>> from raganything import RAGAnything
        >>> from gahr_msr_framework import GAHRMSRQuery, GAHRMSRConfig
        >>>
        >>> # Initialize RAGAnything (processes documents and builds graph)
        >>> rag = RAGAnything(config=config, llm_model_func=llm_func, ...)
        >>> await rag.process_document_complete("document.pdf", "./output")
        >>>
        >>> # Create GAHR-MSR query interface
        >>> gahr_config = GAHRMSRConfig(
        ...     colbert_device="cuda",
        ...     final_top_k=5
        ... )
        >>> gahr_query = GAHRMSRQuery(rag, gahr_config)
        >>>
        >>> # Perform query with re-ranking
        >>> result = await gahr_query.query(
        ...     "What is the late interaction mechanism in ColBERT?",
        ...     mode="hybrid"
        ... )
    """

    def __init__(
        self,
        raganything,
        config: Optional[GAHRMSRConfig] = None
    ):
        """
        Initialize GAHR-MSR query interface

        Args:
            raganything: RAGAnything instance (must be initialized with LightRAG)
            config: GAHR-MSR configuration
        """
        self.rag = raganything
        self.config = config or GAHRMSRConfig()

        # Ensure RAGAnything has LightRAG initialized
        if not hasattr(self.rag, 'lightrag') or self.rag.lightrag is None:
            raise ValueError(
                "RAGAnything must have LightRAG initialized. "
                "Process at least one document first."
            )

        # Initialize components
        self._init_hybrid_retriever()
        self._init_reranker()
        self._init_graph_filter()

        logger.info("GAHR-MSR Query interface initialized")
        logger.info(f"  Hybrid retrieval: dense={self.config.dense_top_k}, "
                   f"sparse={self.config.sparse_top_k}")
        logger.info(f"  ColBERT re-ranking: {self.config.enable_colbert_reranking}")
        logger.info(f"  Graph filtering: {self.config.enable_graph_filtering}")

    def _init_hybrid_retriever(self):
        """Initialize hybrid retriever"""
        hybrid_config = HybridRetrievalConfig(
            dense_top_k=self.config.dense_top_k,
            sparse_top_k=self.config.sparse_top_k,
            final_top_k=self.config.hybrid_top_k,
            rrf_k=self.config.rrf_k
        )

        self.hybrid_retriever = HybridRetriever(
            dense_search_func=self._dense_search_wrapper,
            sparse_search_func=self._sparse_search_wrapper,
            config=hybrid_config
        )

    def _init_reranker(self):
        """Initialize ColBERT re-ranker"""
        if self.config.enable_colbert_reranking:
            colbert_config = ColBERTConfig(
                model_name=self.config.colbert_model,
                device=self.config.colbert_device
            )
            self.colbert_reranker = ColBERTReranker(colbert_config)
        else:
            self.colbert_reranker = None

    def _init_graph_filter(self):
        """Initialize graph-aware filter"""
        if self.config.enable_graph_filtering:
            self.graph_filter = GraphAwareFilter(self.rag.lightrag)
        else:
            self.graph_filter = None

    async def _dense_search_wrapper(
        self,
        query: str,
        top_k: int,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[tuple]:
        """
        Wrapper for dense vector search using LightRAG

        Args:
            query: Query text
            top_k: Number of results
            filter: Optional filter

        Returns:
            List of (chunk_id, score) tuples
        """
        try:
            # Use LightRAG's embedding function to get query vector
            query_embedding = await self.rag.embedding_func([query])

            # Access LightRAG's vector storage for search
            # Note: This is a simplified example - actual implementation
            # depends on LightRAG's internal API
            if hasattr(self.rag.lightrag, 'chunk_entity_relation_graph'):
                # Perform vector search on chunks
                # This is a placeholder - actual implementation needs LightRAG API
                logger.warning("Dense search using LightRAG API - implementation specific")
                return []

            return []

        except Exception as e:
            logger.error(f"Dense search failed: {e}")
            return []

    async def _sparse_search_wrapper(
        self,
        query: str,
        top_k: int,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[tuple]:
        """
        Wrapper for sparse vector search

        Args:
            query: Query text
            top_k: Number of results
            filter: Optional filter

        Returns:
            List of (chunk_id, score) tuples
        """
        try:
            # Sparse search using BM25 or SPLADE
            # This is a placeholder - actual implementation needs sparse embedding model
            logger.warning("Sparse search not fully implemented - requires SPLADE model")
            return []

        except Exception as e:
            logger.error(f"Sparse search failed: {e}")
            return []

    async def _get_chunks_from_lightrag(
        self,
        query: str,
        mode: str = "hybrid",
        top_k: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Get chunks from LightRAG's retrieval

        Args:
            query: Query text
            mode: Retrieval mode
            top_k: Number of chunks

        Returns:
            List of chunk dictionaries with text and metadata
        """
        try:
            from lightrag import QueryParam

            # Use LightRAG's query with only_need_context=True to get chunks
            query_param = QueryParam(
                mode=mode,
                only_need_context=True,
                top_k=top_k
            )

            # Get context chunks
            context = await self.rag.lightrag.aquery(query, param=query_param)

            # Parse context into chunks
            # The exact format depends on LightRAG's output
            # This is a simplified parsing
            chunks = []

            if isinstance(context, str):
                # Split context by common delimiters
                import re
                # Look for chunk separators or paragraph breaks
                chunk_texts = re.split(r'\n\n+', context)

                for i, text in enumerate(chunk_texts):
                    if text.strip():
                        chunks.append({
                            'id': f'chunk_{i}',
                            'text': text.strip(),
                            'score': 1.0 / (i + 1)  # Simple ranking
                        })

            elif isinstance(context, list):
                # If context is already a list of chunks
                for i, chunk in enumerate(context):
                    if isinstance(chunk, dict):
                        chunks.append(chunk)
                    else:
                        chunks.append({
                            'id': f'chunk_{i}',
                            'text': str(chunk),
                            'score': 1.0 / (i + 1)
                        })

            logger.info(f"Retrieved {len(chunks)} chunks from LightRAG")
            return chunks

        except Exception as e:
            logger.error(f"Failed to get chunks from LightRAG: {e}")
            return []

    async def query(
        self,
        query: str,
        mode: str = "hybrid",
        use_graph_filter: Optional[bool] = None,
        use_reranking: Optional[bool] = None,
        return_context_only: bool = False
    ) -> str | Dict[str, Any]:
        """
        Perform GAHR-MSR query with multi-stage retrieval and re-ranking

        Args:
            query: Query text
            mode: LightRAG query mode ("local", "global", "hybrid", "naive", "mix")
            use_graph_filter: Enable graph filtering (None uses config default)
            use_reranking: Enable re-ranking (None uses config default)
            return_context_only: If True, return only re-ranked context without LLM generation

        Returns:
            Query result string or context dictionary if return_context_only=True
        """
        logger.info("="*60)
        logger.info("GAHR-MSR Query Pipeline Started")
        logger.info("="*60)
        logger.info(f"Query: {query}")
        logger.info(f"Mode: {mode}")

        # Determine if we should use graph filtering and re-ranking
        use_graph = use_graph_filter if use_graph_filter is not None else self.config.enable_graph_filtering
        use_rerank = use_reranking if use_reranking is not None else self.config.enable_colbert_reranking

        # Phase 1: Get initial candidates from LightRAG
        logger.info("\n--- Phase 1: Initial Retrieval from LightRAG ---")
        chunks = await self._get_chunks_from_lightrag(
            query,
            mode=mode,
            top_k=self.config.hybrid_top_k
        )

        if not chunks:
            logger.warning("No chunks retrieved. Falling back to standard query.")
            return await self.rag.aquery(query, mode=mode)

        # Phase 2: Apply graph-aware filtering (if enabled)
        if use_graph and self.graph_filter:
            logger.info("\n--- Phase 2: Graph-Aware Filtering ---")
            try:
                graph_filter_dict = await self.graph_filter.create_graph_filter_for_query(
                    query,
                    llm_func=self.rag.llm_model_func
                )

                if graph_filter_dict:
                    # Filter chunks based on graph metadata
                    # This is a placeholder - actual implementation depends on chunk metadata
                    logger.info(f"Graph filter created: {graph_filter_dict}")
            except Exception as e:
                logger.error(f"Graph filtering failed: {e}")

        # Phase 3: ColBERT Re-ranking
        if use_rerank and self.colbert_reranker:
            logger.info("\n--- Phase 3: ColBERT Re-ranking ---")
            try:
                # Extract texts for re-ranking
                chunk_texts = [chunk.get('text', '') for chunk in chunks]

                # Limit to top candidates for re-ranking
                colbert_candidates = min(self.config.colbert_top_k, len(chunks))
                candidate_texts = chunk_texts[:colbert_candidates]
                candidate_chunks = chunks[:colbert_candidates]

                # Re-rank with ColBERT
                ranked_indices = await self.colbert_reranker.rerank(
                    query=query,
                    documents=candidate_texts,
                    top_k=self.config.final_top_k,
                    return_scores=False
                )

                # Reorder chunks based on ColBERT ranking
                reranked_chunks = [candidate_chunks[idx] for idx in ranked_indices]

                logger.info(f"Re-ranked {len(candidate_texts)} chunks, selected top {len(reranked_chunks)}")

            except Exception as e:
                logger.error(f"ColBERT re-ranking failed: {e}")
                # Fall back to original ranking
                reranked_chunks = chunks[:self.config.final_top_k]
        else:
            # No re-ranking, just take top-k
            reranked_chunks = chunks[:self.config.final_top_k]

        # Prepare context from re-ranked chunks
        context_text = "\n\n".join([
            f"[Context {i+1}]\n{chunk.get('text', '')}"
            for i, chunk in enumerate(reranked_chunks)
        ])

        # Return context only if requested
        if return_context_only:
            return {
                'context': context_text,
                'chunks': reranked_chunks,
                'num_chunks': len(reranked_chunks)
            }

        # Phase 4: LLM Generation with re-ranked context
        logger.info("\n--- Phase 4: LLM Generation ---")

        try:
            # Build prompt with re-ranked context
            prompt = f"""Based on the following context, please answer the question.

Context:
{context_text}

Question: {query}

Answer:"""

            # Call LLM
            result = await self.rag.llm_model_func(
                prompt,
                system_prompt="You are a helpful assistant that answers questions based on the provided context."
            )

            logger.info("="*60)
            logger.info("GAHR-MSR Query Pipeline Completed")
            logger.info("="*60)

            return result

        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            return f"Error generating response: {e}"

    def get_config(self) -> GAHRMSRConfig:
        """Get current configuration"""
        return self.config

    def update_config(self, **kwargs):
        """
        Update configuration parameters

        Args:
            **kwargs: Configuration parameters to update
        """
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
                logger.info(f"Updated config: {key} = {value}")
            else:
                logger.warning(f"Unknown config parameter: {key}")

    def get_pipeline_info(self) -> Dict[str, Any]:
        """Get information about the pipeline configuration"""
        return {
            "framework": "GAHR-MSR",
            "version": "1.0",
            "phases": {
                "1_initial_retrieval": {
                    "method": "LightRAG",
                    "top_k": self.config.hybrid_top_k
                },
                "2_graph_filtering": {
                    "enabled": self.config.enable_graph_filtering,
                    "mode": self.config.graph_filter_mode
                },
                "3_colbert_reranking": {
                    "enabled": self.config.enable_colbert_reranking,
                    "model": self.config.colbert_model,
                    "top_k": self.config.colbert_top_k,
                    "device": self.config.colbert_device
                },
                "4_final_selection": {
                    "top_k": self.config.final_top_k
                }
            },
            "config": {
                "dense_top_k": self.config.dense_top_k,
                "sparse_top_k": self.config.sparse_top_k,
                "rrf_k": self.config.rrf_k,
            }
        }
