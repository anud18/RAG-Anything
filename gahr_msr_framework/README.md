# GAHR-MSR Framework

## Graph-Augmented Hybrid Retrieval and Multi-Stage Re-ranking

An independent implementation of the GAHR-MSR framework for high-fidelity Retrieval-Augmented Generation (RAG) systems.

---

## 📖 Overview

This framework implements the methodology described in the research paper:

> **"Graph-Augmented Hybrid Retrieval and Multi-Stage Re-ranking for RAG Systems"**

The GAHR-MSR framework addresses two critical limitations in modern RAG systems:
1. **Context Fragmentation** - Standard chunking methods split coherent information across disconnected segments
2. **Relevance-Performance Trade-off** - Initial retrieval often captures topically related but not precisely relevant documents

---

## 🏗️ Architecture

### Three-Phase Pipeline

#### Phase 1: Graph-Aware Chunking and Indexing
- **Knowledge Graph Construction**: Builds knowledge graph from document corpus
- **Semantic Chunking**: Intelligently splits documents at natural boundaries
- **Chunk Enrichment**: Enriches chunks with structured graph metadata

*Note: This phase is handled by RAGAnything + LightRAG during document processing*

#### Phase 2: High-Recall Hybrid Candidate Retrieval
- **Dense Vector Search**: Semantic similarity using transformer embeddings
- **Sparse Vector Search**: Keyword precision using BM25/SPLADE
- **RRF Fusion**: Reciprocal Rank Fusion combines results
- **Graph-Aware Filtering**: Pre-filters using knowledge graph metadata

**RRF Formula:**
```
Score_RRF(d) = Σ_i (1 / (k + rank_i(d)))
```

#### Phase 3: High-Precision Cascaded Re-ranking
- **ColBERT Re-ranking**: Token-level late interaction mechanism
- **MaxSim Operator**: Fine-grained relevance scoring
- **Cascaded Architecture**: Focuses computation on top candidates

**ColBERT MaxSim Formula:**
```
Score_ColBERT(q,d) = Σ_i max_j (E_q^i · E_d^j^T)
```

---

## 📦 Installation

### Requirements

```bash
# Core dependencies (from RAGAnything)
pip install raganything[all]

# GAHR-MSR specific requirements
pip install torch>=2.0.0 transformers>=4.30.0

# For GPU support (recommended)
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

### Quick Install

```bash
# From RAG-Anything root directory
cd RAG-Anything
pip install -e .  # Install RAGAnything

# Install GAHR-MSR dependencies
pip install torch transformers
```

---

## 🚀 Quick Start

### Basic Usage

```python
import asyncio
from raganything import RAGAnything, RAGAnythingConfig
from gahr_msr_framework import GAHRMSRQuery, GAHRMSRConfig

async def main():
    # 1. Initialize RAGAnything
    rag = RAGAnything(
        config=RAGAnythingConfig(working_dir="./rag_storage"),
        llm_model_func=your_llm_function,
        embedding_func=your_embedding_function
    )

    # 2. Process documents (builds knowledge graph)
    await rag.process_document_complete("document.pdf", "./output")

    # 3. Create GAHR-MSR interface
    gahr_config = GAHRMSRConfig(
        colbert_device="cuda",  # Use GPU for 5-10x speedup
        final_top_k=5,
        enable_colbert_reranking=True
    )
    gahr = GAHRMSRQuery(rag, gahr_config)

    # 4. Query with multi-stage re-ranking
    result = await gahr.query(
        "What is the main contribution of this paper?",
        mode="hybrid"
    )

    print(result)

asyncio.run(main())
```

### Command Line

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

---

## ⚙️ Configuration

### GAHRMSRConfig Parameters

```python
from gahr_msr_framework import GAHRMSRConfig

config = GAHRMSRConfig(
    # Phase 2: Hybrid Retrieval
    dense_top_k=100,              # Dense search candidates
    sparse_top_k=100,             # Sparse search candidates
    hybrid_top_k=100,             # After RRF fusion
    rrf_k=60,                     # RRF constant

    # Phase 3: Re-ranking
    enable_colbert_reranking=True,  # Enable ColBERT
    colbert_top_k=20,               # Candidates to re-rank
    final_top_k=5,                  # Final results
    colbert_model="colbert-ir/colbertv2.0",
    colbert_device="cuda",          # "cpu", "cuda", or "mps"

    # Graph Filtering
    enable_graph_filtering=True,
    graph_filter_mode="any"         # "any" or "all"
)
```

### Device Selection

| Device | Use Case | Performance |
|--------|----------|-------------|
| `cpu` | Development, no GPU | Baseline |
| `cuda` | Production, NVIDIA GPU | 5-10x faster |
| `mps` | Apple M1/M2/M3 | 3-5x faster |

---

## 📊 Performance

Based on SciFact benchmark (from paper):

| Metric | Baseline | GAHR-MSR | Improvement |
|--------|----------|----------|-------------|
| **nDCG@10** | 0.685 | **0.859** | +25.4% |
| **Recall@100** | 0.852 | **0.965** | +13.3% |
| **Latency** | 245ms | 215ms | +12.2% faster |

---

## 💡 Usage Examples

### Example 1: Basic Query

```python
from gahr_msr_framework import GAHRMSRQuery, GAHRMSRConfig

# Initialize (assumes RAGAnything already setup)
gahr = GAHRMSRQuery(rag, GAHRMSRConfig())

# Query
result = await gahr.query("What is ColBERT?", mode="hybrid")
```

### Example 2: Context-Only Retrieval

```python
# Get re-ranked context without LLM generation
context = await gahr.query(
    "What are the main findings?",
    return_context_only=True
)

print(f"Retrieved {context['num_chunks']} chunks")
print(context['context'])
```

### Example 3: GPU-Accelerated

```python
# High-performance configuration
config = GAHRMSRConfig(
    colbert_device="cuda",
    colbert_top_k=50,
    final_top_k=10
)

gahr = GAHRMSRQuery(rag, config)
result = await gahr.query("Complex query", mode="hybrid")
```

### Example 4: Disable Re-ranking

```python
# Fast mode: hybrid retrieval only, no ColBERT
config = GAHRMSRConfig(
    enable_colbert_reranking=False,
    final_top_k=10
)

gahr = GAHRMSRQuery(rag, config)
result = await gahr.query("Fast query", mode="hybrid")
```

---

## 📚 Components

### 1. `ColBERTReranker`

High-precision re-ranker using ColBERT late interaction model.

```python
from gahr_msr_framework import ColBERTReranker, ColBERTConfig

config = ColBERTConfig(device="cuda")
reranker = ColBERTReranker(config)

ranked = await reranker.rerank(
    query="What is X?",
    documents=["Doc 1", "Doc 2", "Doc 3"],
    top_k=2
)
```

### 2. `HybridRetriever`

Combines dense and sparse retrieval with RRF fusion.

```python
from gahr_msr_framework import HybridRetriever, HybridRetrievalConfig

config = HybridRetrievalConfig(
    dense_top_k=100,
    sparse_top_k=100,
    rrf_k=60
)

retriever = HybridRetriever(config=config)
results = await retriever.retrieve("Query text")
```

### 3. `GAHRMSRQuery`

Main interface integrating all components.

```python
from gahr_msr_framework import GAHRMSRQuery, GAHRMSRConfig

gahr = GAHRMSRQuery(rag, GAHRMSRConfig())
result = await gahr.query("Question", mode="hybrid")
```

---

## 🛠️ Advanced Features

### Pipeline Information

```python
# Get detailed pipeline configuration
info = gahr.get_pipeline_info()
print(f"Framework: {info['framework']}")
print(f"Phases: {info['phases']}")
```

### Dynamic Configuration

```python
# Update configuration at runtime
gahr.update_config(
    final_top_k=10,
    colbert_device="cuda"
)
```

### Batch Processing

```python
# Reuse instance for multiple queries (avoids model reloading)
gahr = GAHRMSRQuery(rag, config)

for query in queries:
    result = await gahr.query(query)
    print(result)
```

---

## 🔧 Troubleshooting

### Out of Memory (GPU)

```python
# Use CPU or reduce candidates
config = GAHRMSRConfig(
    colbert_device="cpu",  # or reduce colbert_top_k
    colbert_top_k=10
)
```

### Model Download Fails

```python
# Pre-download model
from transformers import AutoModel
AutoModel.from_pretrained("colbert-ir/colbertv2.0")
```

### LightRAG Not Initialized

```python
# Process at least one document first
await rag.process_document_complete("doc.pdf", "./output")
gahr = GAHRMSRQuery(rag, config)  # Now works
```

---

## 📁 Project Structure

```
gahr_msr_framework/
├── __init__.py              # Package initialization
├── colbert_reranker.py      # ColBERT re-ranker (350 lines)
├── hybrid_retrieval.py      # Hybrid search + RRF (450 lines)
├── gahr_msr.py              # Main interface (500 lines)
├── README.md                # This file
└── requirements.txt         # Dependencies

examples/
└── gahr_msr_example.py      # Complete example (330 lines)
```

---

## 📖 Citation

If you use this implementation in your research, please cite:

```bibtex
@article{gahr-msr-2024,
  title={Graph-Augmented Hybrid Retrieval and Multi-Stage Re-ranking for RAG Systems},
  author={[Authors]},
  journal={[Journal]},
  year={2024}
}
```

---

## 📝 License

This implementation is part of RAG-Anything and follows the same license (MIT).

---

## 🤝 Contributing

Contributions are welcome! Please submit issues and pull requests to the main RAG-Anything repository.

---

## 🔗 Resources

- **Example Script**: `examples/gahr_msr_example.py`
- **RAGAnything**: Main repository for document processing
- **ColBERT Paper**: Khattab & Zaharia (2020)
- **RRF Paper**: Cormack et al. (2009)

---

## ⚡ Quick Reference

| Task | Command |
|------|---------|
| Install | `pip install torch transformers` |
| Basic Example | `python examples/gahr_msr_example.py doc.pdf` |
| GPU Mode | `python examples/gahr_msr_example.py doc.pdf --colbert_device cuda` |
| Context Only | `return_context_only=True` in query |
| Disable Re-rank | `enable_colbert_reranking=False` in config |

---

**Version**: 1.0.0
**Status**: Production Ready
**Maintained by**: GAHR-MSR Implementation Team
