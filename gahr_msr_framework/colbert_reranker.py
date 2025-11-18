"""
ColBERT Re-ranker Implementation

Implements the ColBERT (Contextualized Late Interaction over BERT) model
for high-precision re-ranking of retrieved documents.

Based on the paper:
- ColBERT uses late interaction mechanism where query and document token embeddings
  are computed independently and similarity is calculated using MaxSim operator
- Formula: Score_ColBERT(q,d) = Σ_i max_j (E_q^i · E_d^j^T)
  where E_q is query token embeddings and E_d is document token embeddings
"""

import torch
import torch.nn.functional as F
from typing import List, Dict, Any, Tuple, Optional
import logging
from dataclasses import dataclass


logger = logging.getLogger(__name__)


@dataclass
class ColBERTConfig:
    """Configuration for ColBERT re-ranker"""

    model_name: str = "colbert-ir/colbertv2.0"
    """ColBERT model name from HuggingFace"""

    max_query_length: int = 32
    """Maximum number of query tokens"""

    max_doc_length: int = 180
    """Maximum number of document tokens"""

    device: str = "cpu"
    """Device for computation (cpu, cuda, mps)"""

    batch_size: int = 32
    """Batch size for encoding documents"""

    normalize_embeddings: bool = True
    """Whether to normalize embeddings for cosine similarity"""


class ColBERTReranker:
    """
    ColBERT re-ranker for high-precision document ranking

    Uses late interaction mechanism for efficient and accurate re-ranking.
    Token-level embeddings are computed for both query and documents,
    and relevance is calculated using the MaxSim operator.

    Example:
        >>> config = ColBERTConfig(device="cuda")
        >>> reranker = ColBERTReranker(config)
        >>> scores = await reranker.rerank(
        ...     query="What is ColBERT?",
        ...     documents=["ColBERT is a ranking model...", "BERT is a language model..."]
        ... )
    """

    def __init__(self, config: Optional[ColBERTConfig] = None):
        """
        Initialize ColBERT re-ranker

        Args:
            config: ColBERT configuration, creates default if None
        """
        self.config = config or ColBERTConfig()
        self.model = None
        self.tokenizer = None
        self._initialized = False

        logger.info(f"ColBERT re-ranker created with config: {self.config}")

    def _lazy_init(self):
        """Lazy initialization of model and tokenizer"""
        if self._initialized:
            return

        try:
            from transformers import AutoTokenizer, AutoModel

            logger.info(f"Loading ColBERT model: {self.config.model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
            self.model = AutoModel.from_pretrained(self.config.model_name)

            # Move model to device
            self.model.to(self.config.device)
            self.model.eval()

            self._initialized = True
            logger.info(f"ColBERT model loaded successfully on {self.config.device}")

        except ImportError as e:
            raise ImportError(
                "transformers library is required for ColBERT. "
                "Install with: pip install transformers"
            ) from e
        except Exception as e:
            raise RuntimeError(f"Failed to load ColBERT model: {e}") from e

    def _encode_query(self, query: str) -> torch.Tensor:
        """
        Encode query to token-level embeddings

        Args:
            query: Query text

        Returns:
            torch.Tensor: Query token embeddings, shape (num_query_tokens, embedding_dim)
        """
        inputs = self.tokenizer(
            query,
            return_tensors="pt",
            max_length=self.config.max_query_length,
            truncation=True,
            padding=False
        ).to(self.config.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            # Use last hidden state as token embeddings
            embeddings = outputs.last_hidden_state.squeeze(0)  # (seq_len, hidden_dim)

        return embeddings

    def _encode_documents(self, documents: List[str]) -> List[torch.Tensor]:
        """
        Encode documents to token-level embeddings

        Args:
            documents: List of document texts

        Returns:
            List[torch.Tensor]: List of document token embeddings
        """
        doc_embeddings = []

        # Process documents in batches
        for i in range(0, len(documents), self.config.batch_size):
            batch_docs = documents[i:i + self.config.batch_size]

            inputs = self.tokenizer(
                batch_docs,
                return_tensors="pt",
                max_length=self.config.max_doc_length,
                truncation=True,
                padding=True
            ).to(self.config.device)

            with torch.no_grad():
                outputs = self.model(**inputs)
                batch_embeddings = outputs.last_hidden_state  # (batch, seq_len, hidden_dim)

            # Split batch back into individual documents
            for j in range(len(batch_docs)):
                # Get actual length (excluding padding)
                attention_mask = inputs['attention_mask'][j]
                actual_length = attention_mask.sum().item()
                doc_emb = batch_embeddings[j, :actual_length, :]
                doc_embeddings.append(doc_emb)

        return doc_embeddings

    def calculate_maxsim_score(
        self,
        query_embeddings: torch.Tensor,
        document_embeddings: torch.Tensor
    ) -> float:
        """
        Calculate ColBERT MaxSim score between query and document

        Implements the formula:
        Score_ColBERT(q,d) = Σ_i max_j (E_q^i · E_d^j^T)

        For each query token, find its maximum similarity with any document token,
        then sum all maximum similarities.

        Args:
            query_embeddings: Query token embeddings, shape (num_query_tokens, dim)
            document_embeddings: Document token embeddings, shape (num_doc_tokens, dim)

        Returns:
            float: ColBERT MaxSim score
        """
        # Normalize embeddings for cosine similarity
        if self.config.normalize_embeddings:
            query_embeddings = F.normalize(query_embeddings, p=2, dim=-1)
            document_embeddings = F.normalize(document_embeddings, p=2, dim=-1)

        # Calculate similarity matrix: (num_query_tokens, num_doc_tokens)
        similarity_matrix = torch.matmul(query_embeddings, document_embeddings.T)

        # MaxSim operation: for each query token, find max similarity with doc tokens
        max_sim_scores, _ = torch.max(similarity_matrix, dim=1)

        # Sum the max similarity scores
        final_score = torch.sum(max_sim_scores).item()

        return final_score

    async def rerank(
        self,
        query: str,
        documents: List[str],
        top_k: Optional[int] = None,
        return_scores: bool = False
    ) -> List[int] | List[Tuple[int, float]]:
        """
        Re-rank documents using ColBERT

        Args:
            query: Query text
            documents: List of document texts to re-rank
            top_k: Return only top-k results (None for all)
            return_scores: If True, return (index, score) tuples

        Returns:
            List of document indices sorted by relevance (descending)
            or List of (index, score) tuples if return_scores=True
        """
        # Lazy initialize model
        self._lazy_init()

        if not documents:
            return []

        logger.info(f"Re-ranking {len(documents)} documents with ColBERT")

        # Encode query
        query_embeddings = self._encode_query(query)

        # Encode documents
        doc_embeddings_list = self._encode_documents(documents)

        # Calculate scores for each document
        scores = []
        for idx, doc_emb in enumerate(doc_embeddings_list):
            score = self.calculate_maxsim_score(query_embeddings, doc_emb)
            scores.append((idx, score))

        # Sort by score (descending)
        ranked_results = sorted(scores, key=lambda x: x[1], reverse=True)

        # Apply top-k filtering
        if top_k is not None:
            ranked_results = ranked_results[:top_k]

        logger.info(f"Re-ranking completed. Top score: {ranked_results[0][1]:.4f}")

        # Return based on return_scores flag
        if return_scores:
            return ranked_results
        else:
            return [idx for idx, _ in ranked_results]

    def rerank_with_context(
        self,
        query: str,
        documents: List[str],
        contexts: List[Dict[str, Any]],
        top_k: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Re-rank documents and return with their associated contexts

        Args:
            query: Query text
            documents: List of document texts
            contexts: List of context dictionaries (same length as documents)
            top_k: Return only top-k results

        Returns:
            List of context dictionaries sorted by relevance
        """
        # Lazy initialize model
        self._lazy_init()

        if len(documents) != len(contexts):
            raise ValueError("documents and contexts must have same length")

        # Get ranked indices with scores
        import asyncio
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # Already in async context
            ranked_results = asyncio.create_task(
                self.rerank(query, documents, top_k=top_k, return_scores=True)
            )
            ranked_results = asyncio.run_coroutine_threadsafe(
                ranked_results, loop
            ).result()
        else:
            ranked_results = asyncio.run(
                self.rerank(query, documents, top_k=top_k, return_scores=True)
            )

        # Reconstruct contexts with scores
        reranked_contexts = []
        for idx, score in ranked_results:
            context = contexts[idx].copy()
            context['colbert_score'] = score
            context['original_rank'] = idx
            reranked_contexts.append(context)

        return reranked_contexts

    def batch_encode_documents(
        self,
        documents: List[str]
    ) -> List[torch.Tensor]:
        """
        Pre-compute and store document embeddings for later use

        This is useful for large document collections where you want to
        encode documents once and reuse the embeddings.

        Args:
            documents: List of document texts

        Returns:
            List of document token embeddings
        """
        self._lazy_init()
        logger.info(f"Batch encoding {len(documents)} documents")
        return self._encode_documents(documents)
