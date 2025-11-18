"""
Hybrid Retrieval with RRF Fusion

Implements hybrid search combining dense and sparse vectors with
Reciprocal Rank Fusion (RRF) for optimal recall.

Based on the paper section 3.2: High-Recall Hybrid Candidate Retrieval
"""

from typing import List, Dict, Any, Tuple, Optional, Callable
from dataclasses import dataclass, field
import logging
import numpy as np


logger = logging.getLogger(__name__)


@dataclass
class RRFConfig:
    """Configuration for Reciprocal Rank Fusion"""

    k: int = 60
    """Constant to diminish impact of lower-ranked documents (typically 60)"""

    weights: Dict[str, float] = field(default_factory=lambda: {"dense": 1.0, "sparse": 1.0})
    """Weights for different retrieval methods"""


class RRFFusion:
    """
    Reciprocal Rank Fusion (RRF) implementation

    Combines multiple ranked lists into a single unified ranking.
    Formula: Score_RRF(d) = Σ_i (1 / (k + rank_i(d)))

    where:
    - d is a document
    - i ranges over all result lists
    - rank_i(d) is the rank of document d in list i
    - k is a constant (typically 60)

    Example:
        >>> rrf = RRFFusion(RRFConfig(k=60))
        >>> dense_results = [(0, 0.95), (1, 0.85), (2, 0.75)]
        >>> sparse_results = [(0, 25.4), (3, 19.1), (1, 15.2)]
        >>> fused = rrf.fuse([dense_results, sparse_results])
    """

    def __init__(self, config: Optional[RRFConfig] = None):
        """
        Initialize RRF fusion

        Args:
            config: RRF configuration
        """
        self.config = config or RRFConfig()
        logger.info(f"RRF fusion initialized with k={self.config.k}")

    def fuse(
        self,
        ranked_lists: List[List[Tuple[Any, float]]],
        list_names: Optional[List[str]] = None
    ) -> List[Tuple[Any, float]]:
        """
        Fuse multiple ranked lists using RRF

        Args:
            ranked_lists: List of ranked result lists.
                         Each list contains (doc_id, score) tuples.
            list_names: Optional names for each list (for weighting)

        Returns:
            List of (doc_id, fused_score) tuples sorted by fused score
        """
        if not ranked_lists:
            return []

        # Use default list names if not provided
        if list_names is None:
            list_names = [f"list_{i}" for i in range(len(ranked_lists))]

        # Collect all unique document IDs
        doc_ids = set()
        for ranked_list in ranked_lists:
            for doc_id, _ in ranked_list:
                doc_ids.add(doc_id)

        # Calculate RRF scores
        rrf_scores = {}

        for doc_id in doc_ids:
            score = 0.0

            for i, ranked_list in enumerate(ranked_lists):
                # Find rank of document in this list (1-indexed)
                rank = None
                for pos, (d_id, _) in enumerate(ranked_list):
                    if d_id == doc_id:
                        rank = pos + 1  # 1-indexed
                        break

                # If document appears in this list, add to RRF score
                if rank is not None:
                    list_name = list_names[i]
                    weight = self.config.weights.get(list_name, 1.0)
                    score += weight / (self.config.k + rank)

            rrf_scores[doc_id] = score

        # Sort by RRF score (descending)
        fused_results = sorted(
            rrf_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )

        logger.debug(f"RRF fusion completed. Fused {len(doc_ids)} unique documents")
        return fused_results


@dataclass
class HybridRetrievalConfig:
    """Configuration for Hybrid Retrieval"""

    dense_top_k: int = 100
    """Number of candidates to retrieve from dense search"""

    sparse_top_k: int = 100
    """Number of candidates to retrieve from sparse search"""

    final_top_k: int = 100
    """Number of final candidates after fusion"""

    rrf_k: int = 60
    """RRF fusion constant"""

    rrf_weights: Dict[str, float] = field(
        default_factory=lambda: {"dense": 1.0, "sparse": 1.0}
    )
    """Weights for dense and sparse retrieval in RRF"""


class HybridRetriever:
    """
    Hybrid retriever combining dense and sparse search with RRF fusion

    Implements the high-recall retrieval stage of GAHR-MSR framework.
    Combines semantic search (dense vectors) with keyword search (sparse vectors).

    Example:
        >>> config = HybridRetrievalConfig(dense_top_k=100, sparse_top_k=100)
        >>> retriever = HybridRetriever(
        ...     dense_search_func=my_dense_search,
        ...     sparse_search_func=my_sparse_search,
        ...     config=config
        ... )
        >>> results = await retriever.retrieve("What is ColBERT?")
    """

    def __init__(
        self,
        dense_search_func: Optional[Callable] = None,
        sparse_search_func: Optional[Callable] = None,
        config: Optional[HybridRetrievalConfig] = None
    ):
        """
        Initialize hybrid retriever

        Args:
            dense_search_func: Function for dense vector search
                              Signature: async (query: str, top_k: int) -> List[Tuple[id, score]]
            sparse_search_func: Function for sparse vector search
                               Signature: async (query: str, top_k: int) -> List[Tuple[id, score]]
            config: Hybrid retrieval configuration
        """
        self.dense_search_func = dense_search_func
        self.sparse_search_func = sparse_search_func
        self.config = config or HybridRetrievalConfig()

        # Initialize RRF fusion
        rrf_config = RRFConfig(k=self.config.rrf_k, weights=self.config.rrf_weights)
        self.rrf = RRFFusion(rrf_config)

        logger.info("Hybrid retriever initialized")
        logger.info(f"  Dense top-k: {self.config.dense_top_k}")
        logger.info(f"  Sparse top-k: {self.config.sparse_top_k}")
        logger.info(f"  Final top-k: {self.config.final_top_k}")

    async def retrieve(
        self,
        query: str,
        graph_filter: Optional[Dict[str, Any]] = None,
        return_scores: bool = False
    ) -> List[Any] | List[Tuple[Any, float]]:
        """
        Perform hybrid retrieval with RRF fusion

        Args:
            query: Query text
            graph_filter: Optional graph-based filter for retrieval
            return_scores: If True, return (doc_id, score) tuples

        Returns:
            List of document IDs or (doc_id, score) tuples
        """
        logger.info(f"Hybrid retrieval for query: {query[:100]}...")

        # Parallel dense and sparse searches
        dense_results = []
        sparse_results = []

        # Perform dense search
        if self.dense_search_func:
            try:
                dense_results = await self.dense_search_func(
                    query,
                    top_k=self.config.dense_top_k,
                    filter=graph_filter
                )
                logger.info(f"Dense search returned {len(dense_results)} results")
            except Exception as e:
                logger.error(f"Dense search failed: {e}")

        # Perform sparse search
        if self.sparse_search_func:
            try:
                sparse_results = await self.sparse_search_func(
                    query,
                    top_k=self.config.sparse_top_k,
                    filter=graph_filter
                )
                logger.info(f"Sparse search returned {len(sparse_results)} results")
            except Exception as e:
                logger.error(f"Sparse search failed: {e}")

        # If neither search succeeded, return empty
        if not dense_results and not sparse_results:
            logger.warning("Both dense and sparse search failed or returned no results")
            return []

        # If only one search succeeded, return its results
        if not dense_results:
            logger.info("Using sparse results only")
            results = sparse_results[:self.config.final_top_k]
            return results if return_scores else [doc_id for doc_id, _ in results]

        if not sparse_results:
            logger.info("Using dense results only")
            results = dense_results[:self.config.final_top_k]
            return results if return_scores else [doc_id for doc_id, _ in results]

        # Fuse results using RRF
        logger.info("Fusing dense and sparse results with RRF")
        fused_results = self.rrf.fuse(
            [dense_results, sparse_results],
            list_names=["dense", "sparse"]
        )

        # Apply final top-k limit
        final_results = fused_results[:self.config.final_top_k]

        logger.info(f"Hybrid retrieval completed. Returned {len(final_results)} results")

        if return_scores:
            return final_results
        else:
            return [doc_id for doc_id, _ in final_results]

    def set_dense_search_func(self, func: Callable):
        """Set the dense search function"""
        self.dense_search_func = func
        logger.info("Dense search function updated")

    def set_sparse_search_func(self, func: Callable):
        """Set the sparse search function"""
        self.sparse_search_func = func
        logger.info("Sparse search function updated")


class GraphAwareFilter:
    """
    Graph-aware filtering for retrieval

    Uses entity and relationship metadata from knowledge graph
    to pre-filter candidates before vector search.
    """

    def __init__(self, lightrag=None):
        """
        Initialize graph-aware filter

        Args:
            lightrag: LightRAG instance for accessing knowledge graph
        """
        self.lightrag = lightrag
        logger.info("Graph-aware filter initialized")

    async def extract_entities_from_query(
        self,
        query: str,
        llm_func: Optional[Callable] = None
    ) -> List[str]:
        """
        Extract entities from query using LLM

        Args:
            query: Query text
            llm_func: Optional LLM function for entity extraction

        Returns:
            List of extracted entity names
        """
        if not llm_func and self.lightrag:
            llm_func = self.lightrag.llm_model_func

        if not llm_func:
            logger.warning("No LLM function available for entity extraction")
            return []

        try:
            prompt = f"""Extract the key entities (people, organizations, concepts, technical terms) from this query.
Return only the entity names, one per line, without explanations.

Query: {query}

Entities:"""

            response = await llm_func(prompt)

            # Parse entities from response
            entities = [
                line.strip()
                for line in response.split('\n')
                if line.strip() and not line.strip().startswith('#')
            ]

            logger.info(f"Extracted {len(entities)} entities from query")
            return entities

        except Exception as e:
            logger.error(f"Entity extraction failed: {e}")
            return []

    def create_entity_filter(
        self,
        entities: List[str],
        match_mode: str = "any"
    ) -> Dict[str, Any]:
        """
        Create filter condition for entity matching

        Args:
            entities: List of entity names to filter
            match_mode: "any" or "all" - how to match entities

        Returns:
            Filter dictionary compatible with vector DB
        """
        if not entities:
            return {}

        # This creates a generic filter structure
        # Specific implementation depends on vector DB backend
        filter_dict = {
            "entities": {
                "match_mode": match_mode,
                "values": entities
            }
        }

        logger.debug(f"Created entity filter: {match_mode} of {len(entities)} entities")
        return filter_dict

    async def create_graph_filter_for_query(
        self,
        query: str,
        llm_func: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """
        Create graph-based filter for a query

        Args:
            query: Query text
            llm_func: Optional LLM function

        Returns:
            Filter dictionary for vector search
        """
        # Extract entities from query
        entities = await self.extract_entities_from_query(query, llm_func)

        # Create filter
        if entities:
            return self.create_entity_filter(entities, match_mode="any")
        else:
            return {}
