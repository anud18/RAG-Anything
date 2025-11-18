# 多模式 RAG 檢索與智能評估

## 🎯 快速開始

### 選擇檢索策略

**Adaptive RAG (推薦)**：根據查詢複雜度智能選擇模式
```bash
export RETRIEVAL_STRATEGY=adaptive
python my_rag.py /path/to/documents --output ./output
```

**Comprehensive Mode**：執行所有 5 種模式並比較
```bash
export RETRIEVAL_STRATEGY=comprehensive
python my_rag.py /path/to/documents --output ./output
```

📖 **詳細說明**: 查看 [ADAPTIVE_RAG.md](./ADAPTIVE_RAG.md) 了解 Adaptive RAG 的完整文檔

---

## 功能概述

`my_rag.py` 已增強為支持多模式檢索並使用 LLM 智能評估，解決了以下三個關鍵問題：

### 🆕 0. Adaptive RAG - 智能模式選擇 (NEW!)

**問題**：不同複雜度的查詢使用相同的檢索策略，造成資源浪費

**解決方案**：
- 使用 LLM 分析查詢複雜度（simple/moderate/complex）
- 根據複雜度動態選擇最優檢索模式組合
- 簡單查詢僅用 2 個模式，複雜查詢用 3 個模式

**效益**：
- ✅ 減少 40-60% 的 API 調用
- ✅ 降低 40-60% 的執行時間
- ✅ 保持相同或更好的答案質量

**查詢分類示例**：
```
簡單查詢 → ["naive", "local"]      (60% 節省)
中等查詢 → ["hybrid", "local"]     (60% 節省)
複雜查詢 → ["mix", "global", "hybrid"] (40% 節省)
```

### 1. 增加答案可靠性與可追溯性 ✅

**問題**：原始答案缺乏來源引用，難以驗證信息的準確性

**解決方案**：
- 使用 `QueryParam(only_need_prompt=True)` 獲取包含檢索上下文的完整 prompt
- 要求 LLM 在生成答案時明確標註信息來源
- 使用 `[Source: ...]` 格式標記每個關鍵點的出處

**實現細節**：
```python
# 獲取原始檢索上下文
query_param = QueryParam(mode=mode, only_need_prompt=True)
raw_prompt = await rag.lightrag.aquery(query_text, query_param)

# 增強 prompt 要求引用來源
citation_prompt = f"""{raw_prompt}

IMPORTANT: When answering, please cite the specific sources from the context above.
For each key point in your answer, indicate which part of the retrieved context it comes from.
Use this format: [Source: brief description of the source section]
"""
```

### 2. 解決長上下文評估問題 ✅

**問題**：5 個模式的完整答案組合可能超過 LLM 上下文限制

**解決方案**：兩階段評估策略

#### Stage 1: 個別評估與評分
- 對每個模式的答案進行獨立評估
- 使用 5 個評分維度（每項 0-10 分）：
  - **Completeness** (完整性)
  - **Accuracy** (準確性)
  - **Relevance** (相關性)
  - **Clarity** (清晰度)
  - **Source Citation** (來源引用)
- 生成簡短摘要（2-3 句話）

#### Stage 2: 比較分析與最終推薦
- 基於 Stage 1 的評分摘要進行比較（而非完整答案）
- 顯著減少上下文長度
- 識別最佳模式並提供綜合答案
- 可選擇性地融合多個模式的優點

## 檔案輸出

### 1. `mode_results_with_sources.txt`
包含所有模式的完整答案及其檢索上下文：
```
【LOCAL MODE】
Answer: [帶有來源引用的答案]

--- Retrieved Context ---
[檢索到的上下文內容預覽（前2000字符）]
```

### 2. `evaluations_detailed.txt`
包含兩階段評估的完整結果：

**Stage 1 輸出示例**：
```
LOCAL MODE:
Completeness: 8/10
Accuracy: 9/10
Relevance: 10/10
Clarity: 8/10
Source Citation: 9/10
Total Score: 44/50
Brief Summary: The answer provides comprehensive information with good source citations...
```

**Stage 2 輸出示例**：
```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
EVALUATION SUMMARY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Best Mode: hybrid
Runner-up: mix

Reasoning: Hybrid mode achieved the highest overall score...

Key Strengths:
- Excellent source citations
- Comprehensive coverage
- Well-structured response

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RECOMMENDED FINAL ANSWER
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

[融合多模式優點的最佳答案，包含完整來源引用]
```

## 執行流程

```
Query: "給我PCD急救人員名單"

┌─────────────────────────────────────────┐
│ Step 1: 多模式檢索（帶來源引用）           │
├─────────────────────────────────────────┤
│ ✓ Local mode   - 獲取上下文 + 生成答案   │
│ ✓ Global mode  - 獲取上下文 + 生成答案   │
│ ✓ Hybrid mode  - 獲取上下文 + 生成答案   │
│ ✓ Naive mode   - 獲取上下文 + 生成答案   │
│ ✓ Mix mode     - 獲取上下文 + 生成答案   │
└─────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────┐
│ Step 2: 兩階段 LLM 評估                  │
├─────────────────────────────────────────┤
│ Stage 1: 個別評估                        │
│   ├─ Local:  44/50 分                   │
│   ├─ Global: 38/50 分                   │
│   ├─ Hybrid: 47/50 分 ← 最高分          │
│   ├─ Naive:  35/50 分                   │
│   └─ Mix:    45/50 分                   │
│                                         │
│ Stage 2: 比較分析                        │
│   └─ 選擇 Hybrid 模式                   │
│   └─ 融合 Mix 模式的優點                │
└─────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────┐
│ Step 3: 輸出最終推薦答案                 │
└─────────────────────────────────────────┘
```

## 技術優勢

### 1. 可追溯性
- ✅ 每個答案都標註信息來源
- ✅ 保存完整的檢索上下文供審查
- ✅ 提高答案的可信度和可驗證性

### 2. 處理長上下文
- ✅ 兩階段評估避免上下文溢出
- ✅ 先評分摘要，再比較分析
- ✅ 支持更多檢索模式的同時評估

### 3. 智能決策
- ✅ 結構化評分系統（5 個維度）
- ✅ 客觀量化比較
- ✅ 自動識別最佳模式
- ✅ 可選融合多模式優點

### 4. 可擴展性
- ✅ 易於添加新的評估維度
- ✅ 可調整評分權重
- ✅ 支持自定義評估標準

## 使用示例

```bash
python my_rag.py /path/to/documents \
  --working_dir ./rag_storage \
  --output ./output \
  --api-key YOUR_API_KEY
```

## 檢索模式說明

| 模式 | 說明 | 適用場景 |
|------|------|----------|
| **local** | 關注上下文相關信息 | 需要局部細節時 |
| **global** | 使用全局知識 | 需要整體概覽時 |
| **hybrid** | 結合 local 和 global | 平衡細節與整體 |
| **naive** | 基本向量搜索 | 快速簡單查詢 |
| **mix** | 知識圖譜 + 向量檢索 | 複雜關係查詢 |

## 注意事項

1. **LLM 調用次數**：每個查詢會調用 LLM 約 7 次（5次生成答案 + 5次個別評估 + 1次最終評估）
2. **執行時間**：比單模式查詢慢，但提供更可靠的結果
3. **成本考量**：API 調用成本會增加，建議用於重要查詢
4. **上下文管理**：檢索上下文僅保存前 2000 字符到文件中以控制文件大小

## 未來改進方向

- [ ] 支持並行評估以加快速度
- [ ] 添加可配置的評分權重
- [ ] 支持用戶自定義評估標準
- [ ] 實現評估結果的可視化dashboard
- [ ] 添加歷史評估數據分析功能
