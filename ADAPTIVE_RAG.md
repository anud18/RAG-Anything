# Adaptive RAG 實現文檔

## 概述

基於論文《Adaptive-RAG: Learning to Adapt Retrieval-Augmented Large Language Models through Question Complexity》的核心概念，本實現提供了智能查詢分類器，根據查詢複雜度動態選擇最優檢索策略。

## 核心原理

### 問題背景

傳統 RAG 系統對所有查詢使用相同的檢索策略，但不同複雜度的查詢需要不同的處理方式：

- **簡單查詢**（Simple）：直接事實查詢 → 使用簡單向量搜索即可
- **中等查詢**（Moderate）：需要組合多個事實 → 使用混合檢索
- **複雜查詢**（Complex）：需要多跳推理 → 使用知識圖譜+向量搜索

### Adaptive RAG 優勢

1. **提高效率**：避免對簡單查詢使用複雜（且慢）的檢索方法
2. **降低成本**：減少不必要的 API 調用（最多節省 60% 的調用次數）
3. **保持質量**：為複雜查詢提供足夠的檢索深度
4. **智能路由**：自動選擇最適合的檢索模式組合

## 架構設計

```
┌─────────────────────────────────────────────────────────────┐
│                    User Query                               │
│              "給我PCD急救人員名單"                            │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│              Step 0: Query Complexity Classifier            │
│                    (LLM-based Router)                       │
├─────────────────────────────────────────────────────────────┤
│  Analyzes query based on:                                   │
│  - Question type (who, what, where, analyze, compare)       │
│  - Number of entities involved                              │
│  - Reasoning depth required                                 │
│  - Abstract vs concrete concepts                            │
├─────────────────────────────────────────────────────────────┤
│  Output:                                                    │
│  ├─ Complexity: simple                                      │
│  ├─ Confidence: 0.95                                        │
│  ├─ Reasoning: "Direct factual lookup of a single list"    │
│  └─ Recommended Modes: naive, local                         │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│            Step 1: Mode Selection Strategy                  │
├─────────────────────────────────────────────────────────────┤
│  SIMPLE    → [naive, local]         (2 modes)              │
│  MODERATE  → [hybrid, local]        (2 modes)              │
│  COMPLEX   → [mix, global, hybrid]  (3 modes)              │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│        Step 2: Execute Selected Modes with Citations       │
│                   (only 2-3 modes vs 5)                     │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│          Step 3: Two-Stage Evaluation & Selection           │
│            (Evaluate only selected modes)                   │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                  Final Answer + Metrics                     │
│  Efficiency Gain: ~60% reduction in API calls               │
└─────────────────────────────────────────────────────────────┘
```

## 查詢複雜度分類標準

### 🟢 SIMPLE (簡單查詢)

**特徵**：
- 直接的事實性問題（Who, What, When, Where）
- 單一實體查詢
- 信息檢索直接明確
- 無需推理或綜合

**示例**：
```
✓ "給我PCD急救人員名單"
✓ "What is the capital of France?"
✓ "張三的電話號碼是多少？"
✓ "List all employees in the marketing department"
```

**推薦模式**：`naive`, `local`
- **naive**: 基本向量搜索，速度最快
- **local**: 關注局部上下文相關信息

**預期性能**：
- 查詢時間: ~2-5 秒
- API 調用: ~3 次（1次分類 + 2次檢索）
- 成本效率: 最高

---

### 🟡 MODERATE (中等查詢)

**特徵**：
- 需要組合多個事實
- 比較或分析 2-3 個實體
- 需要一定程度的推理
- 跨文檔信息整合

**示例**：
```
✓ "比較A部門和B部門的人數差異"
✓ "How does temperature affect plant growth?"
✓ "樹林廠和新竹廠的安全記錄哪個更好？"
✓ "What are the main differences between X and Y?"
```

**推薦模式**：`hybrid`, `local`
- **hybrid**: 結合局部和全局檢索方法
- **local**: 補充詳細的上下文信息

**預期性能**：
- 查詢時間: ~5-10 秒
- API 調用: ~5 次（1次分類 + 2次檢索 + 2次評估）
- 成本效率: 中等

---

### 🔴 COMPLEX (複雜查詢)

**特徵**：
- 多跳推理需求
- 需要綜合多個來源的信息
- 抽象概念或關係分析
- 分析、總結、趨勢識別

**示例**：
```
✓ "分析過去三年的安全事故趨勢並提出改進建議"
✓ "What are the long-term implications of climate change on agriculture?"
✓ "綜合所有部門的數據，找出組織效率的瓶頸"
✓ "Explain the relationship between economic policy and social mobility"
```

**推薦模式**：`mix`, `global`, `hybrid`
- **mix**: 知識圖譜 + 向量檢索，處理複雜關係
- **global**: 利用全局知識進行綜合分析
- **hybrid**: 平衡細節與整體視角

**預期性能**：
- 查詢時間: ~10-20 秒
- API 調用: ~9 次（1次分類 + 3次檢索 + 5次評估）
- 成本效率: 較低（但必要）

## 使用方式

### 方法 1: 環境變數配置

```bash
# 使用 Adaptive RAG（推薦）
export RETRIEVAL_STRATEGY=adaptive

# 使用全面檢索（所有模式）
export RETRIEVAL_STRATEGY=comprehensive

python my_rag.py /path/to/documents --output ./output
```

### 方法 2: 代碼內修改

在 `my_rag.py` 中修改：

```python
# Line ~577
retrieval_strategy = "adaptive"  # or "comprehensive"
```

## 輸出文件

### 1. `adaptive_classifications.txt`

記錄每個查詢的複雜度分類結果：

```
================================================================================
Query: 給我PCD急救人員名單，你可以使用工具計算
================================================================================
Complexity: simple
Confidence: 0.95
Reasoning: This is a direct factual question asking for a specific list...
Selected Modes: naive, local

Full Classification:
Complexity: simple
Confidence: 0.95
Reasoning: This is a direct factual question asking for a specific list...
Recommended Modes: naive, local
```

### 2. `mode_results_with_sources.txt`

僅包含選定模式的結果（而非所有 5 個模式）：

```
【NAIVE MODE】
Answer: [帶來源引用的答案]
--- Retrieved Context ---
[上下文預覽]

【LOCAL MODE】
Answer: [帶來源引用的答案]
--- Retrieved Context ---
[上下文預覽]
```

### 3. 日誌輸出示例

```
================================================================================
[Text Query]: 給我PCD急救人員名單，你可以使用工具計算
================================================================================

[Step 0/4] 🧠 Adaptive RAG: Classifying query complexity...
  ├─ Complexity: SIMPLE
  ├─ Confidence: 0.95
  ├─ Reasoning: Direct factual question asking for a specific list
  └─ Selected Modes: NAIVE, LOCAL

[Step 1/4] Querying with selected modes (naive, local) (with source citations)...
  Querying with mode: naive
  Querying with mode: local

[Step 2/4] Two-stage LLM evaluation...
  Stage 1: Individual evaluation and scoring...
    ✓ Evaluated naive mode
    ✓ Evaluated local mode
  Stage 2: Comparative analysis and final selection...

[Step 3/4] Evaluation Complete!

[Step 4/4] 📊 Adaptive RAG Summary:
  ├─ Query Complexity: SIMPLE
  ├─ Modes Used: NAIVE, LOCAL
  ├─ Modes Saved: 3 mode(s) skipped
  └─ Efficiency Gain: ~60% reduction in API calls
```

## 性能對比

### Comprehensive Mode (所有模式)
```
查詢次數: 5 modes × 每個查詢
API 調用: ~11 次
- 5次檢索（每個模式）
- 5次評估
- 1次最終評估

估計成本: $$$$$ (基準)
估計時間: ~15-25秒
```

### Adaptive Mode (智能選擇)
```
簡單查詢: 2 modes
API 調用: ~5 次
- 1次分類
- 2次檢索
- 2次評估
- 1次最終評估

估計成本: $$ (節省 60%)
估計時間: ~5-10秒

中等查詢: 2 modes
估計成本: $$ (節省 60%)
估計時間: ~7-12秒

複雜查詢: 3 modes
估計成本: $$$ (節省 40%)
估計時間: ~12-20秒
```

## 配置與調優

### 自定義複雜度策略

如需調整模式選擇策略，修改 `select_modes_by_complexity` 函數：

```python
mode_strategy = {
    "simple": ["naive"],                    # 最快，僅向量搜索
    "moderate": ["hybrid"],                 # 平衡
    "complex": ["mix", "global"]            # 深度檢索
}
```

### 調整分類標準

修改 `classify_query_complexity` 函數中的 prompt 來調整分類邏輯。

## 最佳實踐

### ✅ 推薦使用場景

1. **生產環境**：降低成本且保持質量
2. **高頻查詢**：大量簡單查詢的場景
3. **混合工作負載**：同時有簡單和複雜查詢
4. **成本敏感**：需要控制 API 調用成本

### ⚠️ 注意事項

1. **分類器開銷**：每次查詢額外增加 1 次 LLM 調用
   - 對於極簡單的查詢，可能抵消部分節省
   - 建議：批量查詢時效益更明顯

2. **分類準確性**：取決於 LLM 的判斷
   - 使用 `confidence` 分數評估可靠性
   - 低信心分數時可回退到 moderate 策略

3. **邊界情況**：某些查詢可能難以分類
   - 系統默認使用 moderate 作為安全選項

### 🔧 故障排除

**分類結果不理想？**
- 檢查 `adaptive_classifications.txt` 查看分類推理
- 調整分類 prompt 以符合你的用例
- 考慮為特定領域添加示例

**性能未如預期？**
- 確認查詢分布（簡單/中等/複雜比例）
- 只有當大量簡單查詢時效益最明顯
- 考慮使用緩存減少重複查詢

## 理論基礎

本實現基於以下研究：

**Adaptive-RAG: Learning to Adapt Retrieval-Augmented Large Language Models through Question Complexity**

核心思想：
1. **查詢路由器**（Query Router）：使用 LLM 作為分類器判斷查詢複雜度
2. **策略選擇**（Strategy Selection）：根據複雜度選擇檢索策略
3. **動態適應**（Dynamic Adaptation）：不同查詢使用不同的計算資源

與原論文的差異：
- 原論文訓練專門的分類器模型
- 本實現使用 LLM 進行零樣本分類（更靈活，無需訓練）
- 本實現增加了來源引用和兩階段評估

## 未來改進方向

- [ ] 收集查詢分類數據，訓練專用分類器
- [ ] 基於歷史性能自動調整策略
- [ ] 支持用戶反饋來優化分類
- [ ] 添加查詢改寫（Query Rewriting）功能
- [ ] 實現多查詢批處理優化
- [ ] 添加 A/B 測試框架比較策略效果

## 總結

Adaptive RAG 通過智能查詢分類，在保持答案質量的同時：

✅ **減少 40-60% 的 API 調用**
✅ **降低 40-60% 的執行時間**
✅ **保持相同或更好的答案質量**
✅ **自動適應不同複雜度的查詢**

推薦作為默認策略使用！
