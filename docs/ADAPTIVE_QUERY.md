# Adaptive Multi-Mode Query 自適應多模式查詢

## 概述 Overview

RAGAnything 現在支持基於 Adaptive-RAG 研究論文的自適應查詢路由功能。系統可以根據查詢的複雜度自動選擇最佳的檢索模式，或嘗試多個模式並讓 LLM 選擇最佳結果。

RAGAnything now supports adaptive query routing based on the Adaptive-RAG research paper. The system can automatically select the best retrieval mode based on query complexity, or try multiple modes and let the LLM select the best result.

## 核心概念 Core Concepts

### 查詢複雜度分類 Query Complexity Classification

系統將查詢分為三個複雜度級別：

The system classifies queries into three complexity levels:

- **Simple（簡單）**: 直接的事實性問題
  - 示例: "什麼是機器學習？", "文檔的標題是什麼？"
  - 推薦模式: `naive` 或 `local`

- **Moderate（中等）**: 需要一些分析或資訊綜合的問題
  - 示例: "系統如何工作？", "X 和 Y 有什麼區別？"
  - 推薦模式: `hybrid`

- **Complex（複雜）**: 需要深度分析或多步推理的問題
  - 示例: "分析 X 對 Y 的影響", "從多個角度綜合關於 X 的資訊"
  - 推薦模式: `mix` 或 `global`

### 檢索模式 Retrieval Modes

- **naive**: 基礎向量搜索 (Basic vector search)
- **local**: 上下文相關檢索 (Context-dependent retrieval)
- **hybrid**: 混合方法 (Combined approaches)
- **global**: 全局知識結構 (Global knowledge structure)
- **mix**: 圖譜 + 向量檢索 (Knowledge graph + vector retrieval)

## 使用方法 Usage

### 1. 自適應查詢路由 Adaptive Query Routing

自動選擇最佳檢索模式（推薦用於大多數情況）

Automatically select the best retrieval mode (recommended for most cases):

```python
from raganything import RAGAnything

# 初始化 RAGAnything / Initialize RAGAnything
rag = RAGAnything(...)

# 簡單使用 - 只獲取答案 / Simple usage - just get the answer
result = await rag.aquery_adaptive("What is machine learning?")
print(result)

# 獲取路由分析詳情 / Get routing analysis details
response = await rag.aquery_adaptive(
    "What is machine learning?",
    return_analysis=True
)

print(f"使用的模式 Mode used: {response['analysis']['recommended_mode']}")
print(f"複雜度 Complexity: {response['analysis']['complexity']}")
print(f"查詢類型 Query type: {response['analysis']['query_type']}")
print(f"推理原因 Reasoning: {response['analysis']['reasoning']}")
print(f"答案 Answer: {response['result']}")
```

**優點 Advantages:**
- ✓ 快速 - 只執行一次檢索 (Fast - only one retrieval)
- ✓ 高效 - LLM 只用於分類 (Efficient - LLM only for classification)
- ✓ 適合生產環境 (Good for production)
- ✓ 推薦用於大多數場景 (Recommended for most cases)

### 2. 多模式查詢評估 Multi-Mode Query Evaluation

嘗試多個模式並讓 LLM 選擇最佳結果（適用於關鍵查詢）

Try multiple modes and let LLM select the best result (for critical queries):

```python
# 獲取最佳結果 / Get best result
result = await rag.aquery_multi_mode(
    "What are the key findings of the research?"
)
print(result)

# 比較所有模式的結果 / Compare all mode results
comparison = await rag.aquery_multi_mode(
    "What are the key findings of the research?",
    modes=["naive", "local", "hybrid", "global", "mix"],
    return_all_results=True
)

print(f"最佳模式 Best mode: {comparison['best_mode']}")
print(f"評估 Evaluation: {comparison['evaluation']}")

# 查看所有結果 / View all results
for mode, data in comparison['results'].items():
    print(f"\n[{mode}]: {data['result'][:100]}...")
```

**優點 Advantages:**
- ✓ 全面 - 嘗試所有模式 (Comprehensive - tries all modes)
- ✓ 最佳質量 - LLM 選擇最佳結果 (Best quality - LLM selects best)
- ✓ 適合關鍵查詢 (Good for critical queries)
- ✗ 成本較高 - 多次檢索 + 評估 (Higher cost - multiple retrievals + evaluation)

### 3. 同步版本 Synchronous Versions

```python
# 同步自適應查詢 / Synchronous adaptive query
result = rag.query_adaptive("What is the document about?")

# 同步多模式查詢 / Synchronous multi-mode query
result = rag.query_multi_mode(
    "What is the document about?",
    return_all_results=True
)
```

## 完整示例 Complete Example

查看 `examples/my_rag.py` 獲取完整的使用示例：

See `examples/my_rag.py` for a complete usage example:

```bash
# 運行示例 / Run example
python examples/my_rag.py your_document.pdf \
    --api-key YOUR_API_KEY \
    --working_dir ./rag_storage
```

## 自定義路由策略 Customize Routing Strategy

您可以自定義複雜度到模式的映射：

You can customize the complexity-to-mode mapping:

```python
from raganything.adaptive_router import AdaptiveQueryRouter, QueryComplexity

# 創建路由器 / Create router
router = AdaptiveQueryRouter(llm_model_func)

# 自定義映射 / Customize mapping
router.update_mode_mapping(
    complexity_map={
        QueryComplexity.SIMPLE: "local",     # 簡單查詢使用 local
        QueryComplexity.MODERATE: "mix",     # 中等查詢使用 mix
        QueryComplexity.COMPLEX: "global",   # 複雜查詢使用 global
    }
)

# 分析查詢 / Analyze query
analysis = await router.analyze_query("Your question here")
print(analysis)
```

## 性能比較 Performance Comparison

| 特性 Feature | Adaptive Routing | Multi-Mode |
|--------------|------------------|------------|
| 速度 Speed | ⚡⚡⚡ 快 Fast | ⚡ 慢 Slow |
| 成本 Cost | 💰 低 Low | 💰💰💰 高 High |
| 質量 Quality | ✅ 好 Good | ✅✅✅ 最佳 Best |
| 生產適用性 Production | ✅ 推薦 Recommended | ⚠️ 關鍵查詢 Critical only |

## API 參考 API Reference

### aquery_adaptive()

```python
async def aquery_adaptive(
    query: str,
    return_analysis: bool = False,
    **kwargs
) -> str | Dict[str, Any]
```

**參數 Parameters:**
- `query`: 查詢文本 (Query text)
- `return_analysis`: 是否返回路由分析詳情 (Return routing analysis details)
- `**kwargs`: 其他查詢參數 (Other query parameters)

**返回 Returns:**
- `str`: 查詢結果（當 return_analysis=False）
- `Dict`: 包含 'result' 和 'analysis'（當 return_analysis=True）

### aquery_multi_mode()

```python
async def aquery_multi_mode(
    query: str,
    modes: List[str] = None,
    return_all_results: bool = False,
    **kwargs
) -> str | Dict[str, Any]
```

**參數 Parameters:**
- `query`: 查詢文本 (Query text)
- `modes`: 要嘗試的模式列表 (List of modes to try)
- `return_all_results`: 是否返回所有結果以供比較 (Return all results for comparison)
- `**kwargs`: 其他查詢參數 (Other query parameters)

**返回 Returns:**
- `str`: 最佳查詢結果（當 return_all_results=False）
- `Dict`: 包含所有結果和評估（當 return_all_results=True）

## 相關研究 Related Research

本功能基於以下研究論文：

This feature is based on:

**Adaptive-RAG: Learning to Adapt Retrieval-Augmented Large Language Models through Question Complexity**

核心思想：通過分類器判斷問題複雜度，將簡單問題路由到向量搜索，將複雜問題路由到基於圖譜的多步推理方法。

Core idea: Use a classifier to determine query complexity, routing simple queries to vector search and complex queries to graph-based multi-hop reasoning methods.

## 最佳實踐 Best Practices

1. **生產環境 Production**: 使用 `aquery_adaptive` 以獲得速度和成本的最佳平衡
   Use `aquery_adaptive` for best balance of speed and cost

2. **關鍵查詢 Critical Queries**: 使用 `aquery_multi_mode` 以獲得最高質量
   Use `aquery_multi_mode` for highest quality

3. **調試 Debugging**: 使用 `return_analysis=True` 了解路由決策
   Use `return_analysis=True` to understand routing decisions

4. **自定義 Customization**: 根據您的領域調整複雜度映射
   Adjust complexity mapping based on your domain

5. **監控 Monitoring**: 追蹤路由決策以優化性能
   Track routing decisions to optimize performance
