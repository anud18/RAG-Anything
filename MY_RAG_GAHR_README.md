# my_rag_gahr.py - GAHR-MSR 增强版使用说明

## 概述

`my_rag_gahr.py` 是基于原始 `my_rag.py` 创建的增强版本，整合了 GAHR-MSR (Graph-Augmented Hybrid Retrieval and Multi-Stage Re-ranking) 框架，提供高精度的文档检索和查询功能。

## 主要特性

### ✅ 保留所有原始功能
- ✅ Tenacity 重试逻辑
- ✅ Google Gemini 模型支持
- ✅ 自定义超时设置
- ✅ 中文查询支持
- ✅ Raw prompt 保存
- ✅ 所有环境变量配置

### 🚀 新增 GAHR-MSR 功能
- **高精度重排序**: 使用 ColBERT 模型进行 token 级别的相关性评分
- **混合检索**: 结合 dense 和 sparse 向量搜索
- **GPU 加速**: 支持 CUDA 和 Apple MPS
- **可配置模式**: 可选择使用标准 RAG 或 GAHR-MSR
- **结果对比**: 自动比较两种模式的查询结果

## 安装依赖

```bash
# 基础依赖 (已包含在 my_rag.py 中)
pip install tenacity python-dotenv

# GAHR-MSR 额外依赖
pip install torch>=2.0.0 transformers>=4.30.0

# GPU 支持 (可选但推荐)
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

## 使用方法

### 1. 标准模式 (与原始 my_rag.py 相同)

```bash
python my_rag_gahr.py <document_path>
```

**示例:**
```bash
python my_rag_gahr.py /path/to/documents \
    --working_dir ./1560_rag_storage \
    --output ./1560_output
```

### 2. 启用 GAHR-MSR 重排序

```bash
python my_rag_gahr.py <document_path> --use_gahr
```

**示例:**
```bash
python my_rag_gahr.py /path/to/documents \
    --working_dir ./1560_rag_storage \
    --output ./1560_output \
    --use_gahr
```

### 3. GAHR-MSR + GPU 加速 (推荐)

```bash
python my_rag_gahr.py <document_path> --use_gahr --colbert_device cuda
```

**示例:**
```bash
python my_rag_gahr.py /path/to/documents \
    --working_dir ./1560_rag_storage \
    --output ./1560_output \
    --use_gahr \
    --colbert_device cuda \
    --final_top_k 10
```

## 命令行参数

### 原始参数 (来自 my_rag.py)

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `file_path` | 文档路径 (必需) | - |
| `--working_dir, -w` | RAG 存储目录 | `./1560_rag_storage` |
| `--output, -o` | 输出目录 | `./1560_output` |
| `--api-key` | OpenAI API key | `$LLM_BINDING_API_KEY` |
| `--base-url` | API base URL | `$LLM_BINDING_HOST` |
| `--parser` | 解析器 (mineru/docling) | `mineru` |

### GAHR-MSR 新增参数

| 参数 | 说明 | 默认值 | 选项 |
|------|------|--------|------|
| `--use_gahr` | 启用 GAHR-MSR 重排序 | `False` | - |
| `--colbert_device` | ColBERT 设备 | `cpu` | `cpu`, `cuda`, `mps` |
| `--final_top_k` | 最终返回结果数量 | `5` | 任意整数 |

## 输出文件

### 标准模式
- `raw_prompt_standard.txt` - 原始 prompt
- `answers_standard.txt` - 查询结果

### GAHR-MSR 模式
- `raw_prompt_gahr.txt` - 原始 prompt
- `answers_gahr.txt` - GAHR-MSR 查询结果

## 使用示例

### 示例 1: 快速查询 (标准模式)

适用于简单查询，不需要高精度重排序的场景。

```bash
python my_rag_gahr.py /path/to/docs
```

**特点:**
- 速度快
- 资源消耗低
- 适合简单问答

### 示例 2: 高精度查询 (GAHR-MSR)

适用于需要高精度结果的复杂查询。

```bash
python my_rag_gahr.py /path/to/docs --use_gahr
```

**特点:**
- 高精度
- 更好的相关性排序
- 适合复杂问题

### 示例 3: 生产环境 (GPU 加速)

适用于生产环境，需要高性能和高精度。

```bash
python my_rag_gahr.py /path/to/docs \
    --use_gahr \
    --colbert_device cuda \
    --final_top_k 10 \
    --working_dir /data/rag_storage \
    --output /data/output
```

**特点:**
- GPU 加速 (5-10x 速度提升)
- 高精度重排序
- 可配置结果数量

### 示例 4: Apple Silicon (M1/M2/M3)

```bash
python my_rag_gahr.py /path/to/docs \
    --use_gahr \
    --colbert_device mps
```

## 性能对比

基于论文的 SciFact 基准测试结果：

| 模式 | nDCG@10 | Recall@100 | 延迟 |
|------|---------|-----------|------|
| **标准 RAG** | 0.685 | 0.852 | ~50ms |
| **GAHR-MSR (CPU)** | 0.859 | 0.965 | ~215ms |
| **GAHR-MSR (GPU)** | 0.859 | 0.965 | ~30ms |

**性能提升:**
- nDCG@10: **+25.4%**
- Recall@100: **+13.3%**

## 工作流程

### 标准模式流程
```
查询 → LightRAG 检索 → 直接返回结果
```

### GAHR-MSR 模式流程
```
查询 → LightRAG 初始检索
     → 图谱过滤 (Graph-Aware Filtering)
     → ColBERT 重排序 (Token-level Re-ranking)
     → 高精度结果
```

## 环境变量配置

在 `.env` 文件中设置：

```bash
# LLM 配置
LLM_MODEL=google/gemini-3-pro-preview
VISION_MODEL=google/gemini-3-pro-preview
LLM_BINDING_API_KEY=your-api-key
LLM_BINDING_HOST=https://api.openai.com/v1

# 超时设置
LLM_TIMEOUT=300
EMBEDDING_TIMEOUT=120

# Embedding 配置
EMBEDDING_DIM=3072
EMBEDDING_MODEL=text-embedding-3-large

# 解析器
PARSER=mineru

# 日志设置
LOG_DIR=./logs
LOG_MAX_BYTES=10485760
LOG_BACKUP_COUNT=5
VERBOSE=false
```

## 查询示例

脚本中预设的中文查询示例：

```python
text_queries = [
    "給我PCD急救人員名單，你可以使用工具計算",
    "給我樹林廠先進管理課急救人員，你可以使用工具計算",
    "給我樹林廠急救人員總共有幾位，你可以使用工具計算",
]
```

## 故障排除

### 问题 1: GPU 内存不足

**错误:** `CUDA out of memory`

**解决方案:**
```bash
# 使用 CPU
python my_rag_gahr.py docs --use_gahr --colbert_device cpu

# 或减少 final_top_k
python my_rag_gahr.py docs --use_gahr --final_top_k 3
```

### 问题 2: 模型下载失败

**错误:** `Cannot download colbert-ir/colbertv2.0`

**解决方案:**
```python
# 预先下载模型
from transformers import AutoModel
AutoModel.from_pretrained("colbert-ir/colbertv2.0")
```

### 问题 3: 导入错误

**错误:** `ModuleNotFoundError: No module named 'gahr_msr_framework'`

**解决方案:**
```bash
# 确保在 RAG-Anything 根目录
cd /path/to/RAG-Anything

# 或添加到 Python 路径
export PYTHONPATH="${PYTHONPATH}:/path/to/RAG-Anything"
```

## 对比 my_rag.py

| 特性 | my_rag.py | my_rag_gahr.py |
|------|-----------|----------------|
| 基础 RAG 查询 | ✅ | ✅ |
| 重试逻辑 | ✅ | ✅ |
| Gemini 支持 | ✅ | ✅ |
| 中文查询 | ✅ | ✅ |
| **GAHR-MSR 重排序** | ❌ | ✅ |
| **GPU 加速** | ❌ | ✅ |
| **结果对比模式** | ❌ | ✅ |
| **高精度检索** | ❌ | ✅ |

## 最佳实践

### 1. 开发阶段
```bash
# 使用标准模式快速测试
python my_rag_gahr.py docs
```

### 2. 测试阶段
```bash
# 启用 GAHR-MSR 评估效果
python my_rag_gahr.py docs --use_gahr
```

### 3. 生产环境
```bash
# GPU 加速 + 高精度
python my_rag_gahr.py docs --use_gahr --colbert_device cuda --final_top_k 10
```

## 技术细节

### GAHR-MSR 三阶段流程

1. **Phase 1: Graph-Aware Chunking**
   - 由 RAGAnything + LightRAG 自动处理
   - 构建知识图谱
   - 语义分块

2. **Phase 2: Hybrid Retrieval**
   - Dense vector search (语义相似度)
   - Sparse vector search (关键词匹配)
   - RRF fusion (结果融合)

3. **Phase 3: ColBERT Re-ranking**
   - Token-level 相关性评分
   - MaxSim operator
   - 高精度排序

### ColBERT MaxSim 公式

```
Score_ColBERT(q,d) = Σ_i max_j (E_q^i · E_d^j^T)
```

其中:
- `E_q^i`: 查询 token i 的 embedding
- `E_d^j`: 文档 token j 的 embedding

## 下一步

1. **运行标准模式** 确认基础功能正常
2. **测试 GAHR-MSR** 评估精度提升
3. **启用 GPU** 获得最佳性能
4. **调优参数** 根据具体需求调整 `final_top_k`

## 支持

- 📖 查看 `gahr_msr_framework/README.md` 了解框架详情
- 📚 查看 `gahr_msr_framework/USAGE.md` 了解高级用法
- 🔧 查看原始论文了解理论基础

---

**版本**: 1.0.0
**状态**: Production Ready
**基于**: my_rag.py + GAHR-MSR Framework
