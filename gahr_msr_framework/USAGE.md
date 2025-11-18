# GAHR-MSR Framework - Usage Guide

This guide provides step-by-step instructions for using the GAHR-MSR framework.

---

## Installation

### Step 1: Install RAGAnything

```bash
cd RAG-Anything
pip install -e .
```

### Step 2: Install GAHR-MSR Dependencies

```bash
# Install PyTorch and Transformers
pip install torch>=2.0.0 transformers>=4.30.0

# For GPU support (NVIDIA)
pip install torch --index-url https://download.pytorch.org/whl/cu118

# For Apple Silicon (M1/M2/M3)
# PyTorch 2.0+ has built-in MPS support, no special installation needed
```

### Step 3: Verify Installation

```python
from gahr_msr_framework import GAHRMSRQuery, ColBERTReranker
print("GAHR-MSR framework installed successfully!")
```

---

## Basic Workflow

### 1. Initialize RAGAnything

```python
from raganything import RAGAnything, RAGAnythingConfig
from lightrag.llm.openai import openai_complete_if_cache, openai_embed
from lightrag.utils import EmbeddingFunc

# Create configuration
config = RAGAnythingConfig(
    working_dir="./rag_storage",
    parser="mineru",
    parse_method="auto"
)

# Define LLM function
def llm_func(prompt, **kwargs):
    return openai_complete_if_cache(
        "gpt-4o-mini",
        prompt,
        api_key="your-api-key",
        **kwargs
    )

# Define embedding function
embedding_func = EmbeddingFunc(
    embedding_dim=3072,
    max_token_size=8192,
    func=lambda texts: openai_embed(
        texts,
        model="text-embedding-3-large",
        api_key="your-api-key"
    )
)

# Initialize RAGAnything
rag = RAGAnything(
    config=config,
    llm_model_func=llm_func,
    embedding_func=embedding_func
)
```

### 2. Process Documents

```python
import asyncio

async def process_document():
    # Process document (this builds the knowledge graph)
    await rag.process_document_complete(
        file_path="document.pdf",
        output_dir="./output"
    )

asyncio.run(process_document())
```

### 3. Create GAHR-MSR Interface

```python
from gahr_msr_framework import GAHRMSRQuery, GAHRMSRConfig

# Create configuration
gahr_config = GAHRMSRConfig(
    # Use GPU for 5-10x speedup
    colbert_device="cuda",  # or "cpu" or "mps"

    # Re-ranking settings
    enable_colbert_reranking=True,
    colbert_top_k=20,
    final_top_k=5,

    # Hybrid retrieval settings
    dense_top_k=100,
    sparse_top_k=100,
    rrf_k=60,

    # Graph filtering
    enable_graph_filtering=True
)

# Initialize GAHR-MSR
gahr = GAHRMSRQuery(rag, gahr_config)
```

### 4. Query

```python
async def query_example():
    # Perform query with multi-stage re-ranking
    result = await gahr.query(
        "What is the main contribution of this paper?",
        mode="hybrid"
    )

    print(result)

asyncio.run(query_example())
```

---

## Common Use Cases

### Case 1: High-Precision Research Questions

For academic papers, technical documentation:

```python
config = GAHRMSRConfig(
    colbert_device="cuda",
    enable_colbert_reranking=True,
    colbert_top_k=30,
    final_top_k=10,
    enable_graph_filtering=True
)

gahr = GAHRMSRQuery(rag, config)
result = await gahr.query(
    "What is the theoretical foundation of this approach?",
    mode="hybrid"
)
```

### Case 2: Fast Queries (No Re-ranking)

For quick lookups, simple questions:

```python
config = GAHRMSRConfig(
    enable_colbert_reranking=False,  # Skip re-ranking
    final_top_k=5,
    enable_graph_filtering=False
)

gahr = GAHRMSRQuery(rag, config)
result = await gahr.query("What is the publication date?", mode="hybrid")
```

### Case 3: Context Retrieval Only

To get re-ranked context without LLM generation:

```python
context = await gahr.query(
    "Explain the methodology",
    mode="hybrid",
    return_context_only=True
)

print(f"Retrieved {context['num_chunks']} chunks")
print(context['context'])

# Use context for custom processing
for i, chunk in enumerate(context['chunks']):
    print(f"Chunk {i+1}: {chunk['text'][:100]}...")
```

### Case 4: Batch Processing

Process multiple queries efficiently:

```python
queries = [
    "What is the main contribution?",
    "What are the limitations?",
    "What are future directions?"
]

gahr = GAHRMSRQuery(rag, GAHRMSRConfig(colbert_device="cuda"))

results = []
for query in queries:
    result = await gahr.query(query, mode="hybrid")
    results.append(result)
```

---

## Configuration Tuning

### For Maximum Quality

```python
config = GAHRMSRConfig(
    # Large candidate pool
    dense_top_k=200,
    sparse_top_k=200,
    hybrid_top_k=150,

    # Thorough re-ranking
    colbert_top_k=50,
    final_top_k=10,

    # GPU for speed
    colbert_device="cuda",

    # All features enabled
    enable_colbert_reranking=True,
    enable_graph_filtering=True
)
```

### For Maximum Speed

```python
config = GAHRMSRConfig(
    # Small candidate pool
    dense_top_k=50,
    sparse_top_k=50,
    hybrid_top_k=50,

    # Minimal or no re-ranking
    enable_colbert_reranking=False,
    final_top_k=5,

    # Skip graph filtering
    enable_graph_filtering=False
)
```

### For Balanced Performance

```python
config = GAHRMSRConfig(
    # Standard settings (defaults)
    dense_top_k=100,
    sparse_top_k=100,
    hybrid_top_k=100,
    rrf_k=60,

    colbert_top_k=20,
    final_top_k=5,
    colbert_device="cuda",

    enable_colbert_reranking=True,
    enable_graph_filtering=True
)
```

---

## Command Line Usage

### Basic Example

```bash
python examples/gahr_msr_example.py document.pdf
```

### With Options

```bash
# GPU acceleration
python examples/gahr_msr_example.py document.pdf \
    --colbert_device cuda

# Custom working directory
python examples/gahr_msr_example.py document.pdf \
    --working_dir ./my_rag_storage \
    --output ./my_output

# Disable re-ranking
python examples/gahr_msr_example.py document.pdf \
    --no_reranking

# Custom number of results
python examples/gahr_msr_example.py document.pdf \
    --final_top_k 10
```

### All Options

```bash
python examples/gahr_msr_example.py document.pdf \
    --working_dir ./rag_storage \
    --output ./output \
    --api-key YOUR_API_KEY \
    --base-url https://api.openai.com/v1 \
    --parser mineru \
    --colbert_device cuda \
    --final_top_k 5
```

---

## Monitoring and Debugging

### Enable Debug Logging

```python
import logging

# Enable debug logs for GAHR-MSR
logging.getLogger("gahr_msr_framework").setLevel(logging.DEBUG)

# Enable debug logs for LightRAG
logging.getLogger("lightrag").setLevel(logging.DEBUG)
```

### Get Pipeline Information

```python
# Get detailed pipeline configuration
info = gahr.get_pipeline_info()

print(f"Framework: {info['framework']} v{info['version']}")
print("\nPhases:")
for phase_name, phase_info in info['phases'].items():
    print(f"  {phase_name}: {phase_info}")
```

### Monitor Query Progress

The framework logs detailed progress:

```
=============================================================
GAHR-MSR Query Pipeline Started
=============================================================
Query: What is the main contribution?
Mode: hybrid

--- Phase 1: Initial Retrieval from LightRAG ---
Retrieved 100 chunks from LightRAG

--- Phase 2: Graph-Aware Filtering ---
Graph filter created: {'entities': {...}}

--- Phase 3: ColBERT Re-ranking ---
Re-ranking 20 documents with ColBERT
Re-ranked 20 chunks, selected top 5

--- Phase 4: LLM Generation ---
=============================================================
GAHR-MSR Query Pipeline Completed
=============================================================
```

---

## Performance Tips

### 1. Use GPU

```python
# 5-10x faster than CPU
config = GAHRMSRConfig(colbert_device="cuda")
```

### 2. Reuse GAHR Instance

```python
# Model loads once, reused for all queries
gahr = GAHRMSRQuery(rag, config)

for query in queries:
    result = await gahr.query(query)
```

### 3. Adjust Candidate Sizes

```python
# Fewer candidates = faster
config = GAHRMSRConfig(
    colbert_top_k=10,  # Instead of 20
    final_top_k=3      # Instead of 5
)
```

### 4. Pre-initialize Model

```python
# Force model loading at initialization
gahr.colbert_reranker._lazy_init()
```

---

## Troubleshooting

### Problem: Out of Memory (GPU)

**Solution:**
```python
# Use CPU or reduce batch size
config = GAHRMSRConfig(
    colbert_device="cpu"  # or reduce colbert_top_k
)
```

### Problem: Slow First Query

**Explanation:** Model loading happens on first query

**Solution:**
```python
# Pre-load model
gahr.colbert_reranker._lazy_init()
```

### Problem: LightRAG Not Initialized Error

**Solution:**
```python
# Process at least one document first
await rag.process_document_complete("doc.pdf", "./output")

# Then create GAHR-MSR interface
gahr = GAHRMSRQuery(rag, config)
```

### Problem: Import Error

**Solution:**
```bash
# Ensure you're in the RAG-Anything root directory
cd RAG-Anything

# Add to Python path if needed
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Or use absolute imports
import sys
sys.path.append('/path/to/RAG-Anything')
```

---

## Next Steps

1. **Read the Main README**: `gahr_msr_framework/README.md`
2. **Run the Example**: `examples/gahr_msr_example.py`
3. **Experiment with Configuration**: Try different settings
4. **Integrate into Your Project**: Use GAHR-MSR in your application

---

## Support

For issues and questions:
- Check the main README: `gahr_msr_framework/README.md`
- Run the example script: `examples/gahr_msr_example.py`
- Check RAGAnything documentation

---

**Happy querying with GAHR-MSR!** 🚀
