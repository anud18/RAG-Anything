# GAHR-MSR Framework

## Graph-Augmented Hybrid Retrieval and Multi-Stage Re-ranking

This module implements the GAHR-MSR framework for high-fidelity Retrieval-Augmented Generation (RAG) as described in the research paper:

> "Graph-Augmented Hybrid Retrieval and Multi-Stage Re-ranking for RAG Systems"

### Overview

The GAHR-MSR framework addresses two critical limitations in modern RAG systems:
1. **Context Fragmentation** - Standard chunking methods split coherent information across disconnected segments
2. **Relevance-Performance Trade-off** - Initial retrieval often captures topically related but not precisely relevant documents

### Architecture

The framework consists of three key phases:

#### Phase 1: Graph-Aware Chunking and Indexing
- **Knowledge Graph Construction**: Builds a knowledge graph from the document corpus using entity and relationship extraction
- **Semantic Chunking**: Intelligently splits documents at natural semantic boundaries
- **Chunk Enrichment**: Enriches each chunk with structured metadata from the knowledge graph

*Note: This phase is handled automatically by RAGAnything + LightRAG during document processing*

#### Phase 2: High-Recall Hybrid Candidate Retrieval
- **Dense Vector Search**: Captures semantic meaning using transformer-based embeddings
- **Sparse Vector Search**: Ensures keyword precision using BM25/SPLADE
- **RRF Fusion**: Combines results using Reciprocal Rank Fusion (RRF)
  ```
  Score_RRF(d) = Σ_i (1 / (k + rank_i(d)))
  ```
- **Graph-Aware Pre-filtering**: Uses knowledge graph metadata to eliminate irrelevant documents early

#### Phase 3: High-Precision Cascaded Re-ranking
- **ColBERT Re-ranking**: Uses late interaction mechanism for token-level relevance scoring
  ```
  Score_ColBERT(q,d) = Σ_i max_j (E_q^i · E_d^j^T)
  ```
- **Cascaded Architecture**: Applies expensive re-ranking only to top candidates from Phase 2
- **Final Selection**: Returns top-k most relevant chunks for LLM generation

### Components

#### 1. ColBERTReranker
High-precision re-ranker using the ColBERT late interaction model.

**Features:**
- Token-level embedding comparison
- Efficient MaxSim operator
- GPU/CPU support
- Batch processing

**Example:**
```python
from raganything.reranking import ColBERTReranker, ColBERTConfig

config = ColBERTConfig(
    model_name="colbert-ir/colbertv2.0",
    device="cuda",  # or "cpu"
    max_query_length=32,
    max_doc_length=180
)

reranker = ColBERTReranker(config)

# Re-rank documents
ranked_indices = await reranker.rerank(
    query="What is ColBERT?",
    documents=["ColBERT is...", "BERT is...", ...],
    top_k=5,
    return_scores=True
)
```

#### 2. HybridRetriever
Combines dense and sparse retrieval with RRF fusion.

**Features:**
- Parallel dense + sparse search
- Reciprocal Rank Fusion (RRF)
- Configurable weights
- Graph-aware filtering support

**Example:**
```python
from raganything.reranking import HybridRetriever, HybridRetrievalConfig

config = HybridRetrievalConfig(
    dense_top_k=100,
    sparse_top_k=100,
    final_top_k=100,
    rrf_k=60
)

retriever = HybridRetriever(
    dense_search_func=my_dense_search,
    sparse_search_func=my_sparse_search,
    config=config
)

results = await retriever.retrieve(
    query="What is hybrid search?",
    graph_filter={"entities": ["RAG", "ColBERT"]}
)
```

#### 3. GAHRMSRQuery
Main interface integrating all components.

**Features:**
- End-to-end query pipeline
- Automatic phase orchestration
- Configurable re-ranking
- Context-only retrieval mode

**Example:**
```python
from raganything import RAGAnything
from raganything.reranking import GAHRMSRQuery, GAHRMSRConfig

# Initialize RAGAnything
rag = RAGAnything(...)
await rag.process_document_complete("document.pdf", "./output")

# Create GAHR-MSR interface
config = GAHRMSRConfig(
    enable_colbert_reranking=True,
    colbert_device="cuda",
    final_top_k=5
)

gahr = GAHRMSRQuery(rag, config)

# Query with multi-stage re-ranking
result = await gahr.query(
    "What is the main contribution?",
    mode="hybrid"
)
```

### Installation

The GAHR-MSR framework requires additional dependencies:

```bash
# Install transformers for ColBERT
pip install transformers torch

# For GPU support (optional but recommended)
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

### Quick Start

See `examples/gahr_msr_example.py` for a complete working example:

```bash
# Basic usage
python examples/gahr_msr_example.py document.pdf

# With GPU acceleration
python examples/gahr_msr_example.py document.pdf --colbert_device cuda

# Custom configuration
python examples/gahr_msr_example.py document.pdf \
    --colbert_device cuda \
    --final_top_k 10 \
    --working_dir ./my_storage
```

### Configuration

#### GAHRMSRConfig

Main configuration class for the framework:

```python
@dataclass
class GAHRMSRConfig:
    # Phase 2: Hybrid Retrieval
    dense_top_k: int = 100          # Candidates from dense search
    sparse_top_k: int = 100         # Candidates from sparse search
    hybrid_top_k: int = 100         # After RRF fusion
    rrf_k: int = 60                 # RRF constant

    # Phase 3: Re-ranking
    enable_colbert_reranking: bool = True
    colbert_top_k: int = 20         # Candidates for re-ranking
    final_top_k: int = 5            # Final results
    colbert_model: str = "colbert-ir/colbertv2.0"
    colbert_device: str = "cpu"     # "cpu", "cuda", or "mps"

    # Graph Filtering
    enable_graph_filtering: bool = True
    graph_filter_mode: str = "any"  # "any" or "all"
```

### Performance Considerations

#### GPU Acceleration
For optimal performance, use GPU for ColBERT re-ranking:
```python
config = GAHRMSRConfig(colbert_device="cuda")
```

#### Batch Processing
When processing multiple queries, reuse the same GAHRMSRQuery instance to avoid model reloading.

#### Memory Usage
- ColBERT model: ~440MB on GPU/CPU
- Token embeddings: ~1KB per chunk
- Recommended: 8GB RAM minimum, 16GB for large documents

### Evaluation Metrics

The framework improves retrieval quality as measured by:
- **nDCG@10**: Normalized Discounted Cumulative Gain at 10
- **Recall@100**: Proportion of relevant documents in top 100
- **Latency**: Average query processing time

Expected improvements over baseline (from paper):
- nDCG@10: +0.174 (from 0.685 to 0.859)
- Recall@100: +0.113 (from 0.852 to 0.965)

### Architecture Diagram

```
┌─────────────────────────────────────────────────────────┐
│              GAHR-MSR Query Pipeline                    │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  Phase 1: Graph-Aware Chunking (Pre-processing)        │
│  ┌───────────────────────────────────────────┐         │
│  │ Document → Entities/Relations → Graph     │         │
│  │ Chunks + Graph Metadata → Indexed         │         │
│  └───────────────────────────────────────────┘         │
│                                                         │
│  Phase 2: High-Recall Hybrid Retrieval                 │
│  ┌───────────────────────────────────────────┐         │
│  │ Query → Dense Search (Semantic)           │         │
│  │      ↘                                     │         │
│  │        → RRF Fusion → Top 100 Candidates  │         │
│  │      ↗                                     │         │
│  │ Query → Sparse Search (Keywords)          │         │
│  │                                            │         │
│  │ Optional: Graph-Aware Pre-filtering       │         │
│  └───────────────────────────────────────────┘         │
│                                                         │
│  Phase 3: High-Precision Re-ranking                    │
│  ┌───────────────────────────────────────────┐         │
│  │ Top 100 → Select Top 20 for ColBERT       │         │
│  │                                            │         │
│  │ ColBERT MaxSim Re-ranking                 │         │
│  │   Query Tokens × Document Tokens          │         │
│  │   → Token-level Similarity Matrix         │         │
│  │   → MaxSim Scores                         │         │
│  │                                            │         │
│  │ Final Top-K Selection → Context           │         │
│  └───────────────────────────────────────────┘         │
│                                                         │
│  Phase 4: LLM Generation                               │
│  ┌───────────────────────────────────────────┐         │
│  │ Re-ranked Context + Query → LLM → Answer  │         │
│  └───────────────────────────────────────────┘         │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

### Citation

If you use this implementation in your research, please cite:

```bibtex
@article{gahr-msr-2024,
  title={Graph-Augmented Hybrid Retrieval and Multi-Stage Re-ranking for RAG Systems},
  author={[Authors]},
  journal={[Journal]},
  year={2024}
}
```

### References

1. **ColBERT**: Khattab, O., & Zaharia, M. (2020). ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT.
2. **Reciprocal Rank Fusion**: Cormack, G. V., Clarke, C. L., & Büttcher, S. (2009). Reciprocal rank fusion outperforms condorcet and individual rank learning methods.
3. **GraphRAG**: Microsoft Research. (2024). GraphRAG: Unlocking LLM discovery on narrative private data.

### License

This implementation is part of RAGAnything and follows the same license (MIT).

### Contributing

Contributions are welcome! Please submit issues and pull requests to the main RAGAnything repository.

### Support

For questions and support:
- GitHub Issues: https://github.com/[repo]/issues
- Documentation: See `examples/gahr_msr_example.py`
