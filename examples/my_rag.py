#!/usr/bin/env python
"""
Adaptive Multi-Mode RAG Query - 完整独立实现

这个脚本包含完整的自适应查询路由功能，可以直接复制粘贴使用。
包含：
1. AdaptiveQueryRouter - 查询复杂度分析和路由
2. 扩展的 RAG 类 - 添加 adaptive 和 multi_mode 查询方法
3. 完整的使用示例

基于 Adaptive-RAG 论文的方法。
"""

import os
import argparse
import asyncio
import logging
import logging.config
import json
from pathlib import Path
from typing import Dict, Any, List, Callable, Optional
from enum import Enum

# Add project root directory to Python path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from lightrag.llm.openai import openai_complete_if_cache, openai_embed
from lightrag.utils import EmbeddingFunc, logger, set_verbose_debug
from raganything import RAGAnything, RAGAnythingConfig
from dotenv import load_dotenv

load_dotenv(dotenv_path=".env", override=False)


# ============================================================================
# Adaptive Query Router 实现
# ============================================================================

class QueryComplexity(Enum):
    """查询复杂度级别"""
    SIMPLE = "simple"           # 简单事实性问题
    MODERATE = "moderate"       # 中等复杂度
    COMPLEX = "complex"         # 复杂分析问题


class QueryType(Enum):
    """查询类型分类"""
    FACTUAL = "factual"                     # 事实查找
    ANALYTICAL = "analytical"               # 分析推理
    SUMMARIZATION = "summarization"         # 总结概括
    COMPARISON = "comparison"               # 比较对比
    MULTI_HOP = "multi_hop"                # 多步推理


class AdaptiveQueryRouter:
    """
    自适应查询路由器 - 分析查询并推荐最佳检索模式

    实现 Adaptive-RAG 的路由策略：根据查询复杂度分类并路由到不同的检索策略
    """

    def __init__(self, llm_model_func: Callable):
        """
        初始化路由器

        Args:
            llm_model_func: LLM 函数用于查询分析
        """
        self.llm_model_func = llm_model_func

        # 基于复杂度的模式映射
        self.complexity_mode_map = {
            QueryComplexity.SIMPLE: "naive",      # 简单查询 → naive/向量搜索
            QueryComplexity.MODERATE: "hybrid",   # 中等查询 → hybrid 混合方法
            QueryComplexity.COMPLEX: "mix",       # 复杂查询 → mix 图+向量
        }

        # 特定查询类型的模式覆盖
        self.type_mode_override = {
            QueryType.FACTUAL: "local",           # 事实查询 → local 上下文
            QueryType.SUMMARIZATION: "global",    # 总结查询 → global 全局知识
            QueryType.MULTI_HOP: "mix",          # 多步推理 → mix 图方法
        }

    async def analyze_query(self, query: str) -> Dict[str, Any]:
        """
        分析查询并推荐检索模式

        Args:
            query: 用户查询字符串

        Returns:
            Dict 包含: complexity, query_type, recommended_mode, reasoning, confidence
        """
        try:
            # 准备分析提示词
            analysis_prompt = f"""分析以下用户查询并进行分类：

用户查询: "{query}"

请基于以下标准对查询进行分类：

1. **复杂度级别:**
   - SIMPLE: 直接的事实性问题，可以用文档中的具体事实回答
     例如: "什么是X?", "Y何时发生?", "谁是Z?"

   - MODERATE: 需要一些分析或信息综合的问题
     例如: "X如何工作?", "X和Y有什么区别?", "解释X和Y之间的关系"

   - COMPLEX: 需要深度分析、多步推理或全面理解的问题
     例如: "分析X对Y的影响", "X的含义是什么?", "从多个角度综合关于X的信息"

2. **查询类型:**
   - FACTUAL: 查找具体事实或定义
   - ANALYTICAL: 需要分析或推理信息
   - SUMMARIZATION: 需要总结或压缩信息
   - COMPARISON: 比较多个实体或概念
   - MULTI_HOP: 需要通过推理连接多条信息

3. **推荐的检索模式:**
   - naive: 基础向量搜索（适合简单事实查询）
   - local: 上下文相关检索（适合特定主题的问题）
   - hybrid: 混合多种方法（适合中等复杂度）
   - global: 使用全局知识结构（适合总结和概览）
   - mix: 整合图和向量检索（适合复杂多步推理）

请用以下 JSON 格式提供你的分析：
{{
    "complexity": "simple|moderate|complex",
    "query_type": "factual|analytical|summarization|comparison|multi_hop",
    "recommended_mode": "naive|local|hybrid|global|mix",
    "reasoning": "简要解释你选择此分类的原因（1-2句话）",
    "confidence": 0.0-1.0
}}

请简洁准确地进行分析。"""

            system_prompt = "你是一个查询分析专家。你的任务是分析用户查询并基于复杂度和类型进行分类，以帮助路由到合适的检索策略。"

            # 调用 LLM 进行分析
            response = await self.llm_model_func(
                analysis_prompt,
                system_prompt=system_prompt
            )

            # 解析 JSON 响应
            response_text = response.strip()
            if "```json" in response_text:
                start = response_text.find("```json") + 7
                end = response_text.find("```", start)
                response_text = response_text[start:end].strip()
            elif "```" in response_text:
                start = response_text.find("```") + 3
                end = response_text.find("```", start)
                response_text = response_text[start:end].strip()

            analysis = json.loads(response_text)

            # 验证并规范化响应
            complexity_str = analysis.get("complexity", "moderate").lower()
            query_type_str = analysis.get("query_type", "factual").lower()

            # 映射到枚举
            complexity = QueryComplexity(complexity_str)
            query_type = QueryType(query_type_str)

            # 获取推荐模式
            recommended_mode = analysis.get("recommended_mode",
                                           self._get_mode_for_query(complexity, query_type))

            return {
                "complexity": complexity,
                "query_type": query_type,
                "recommended_mode": recommended_mode,
                "reasoning": analysis.get("reasoning", "未提供推理"),
                "confidence": float(analysis.get("confidence", 0.8)),
                "raw_analysis": analysis
            }

        except Exception as e:
            # 如果分析失败，回退到默认值
            return {
                "complexity": QueryComplexity.MODERATE,
                "query_type": QueryType.FACTUAL,
                "recommended_mode": "hybrid",
                "reasoning": f"查询分析失败，使用默认模式。错误: {str(e)}",
                "confidence": 0.5,
                "error": str(e)
            }

    def _get_mode_for_query(self, complexity: QueryComplexity, query_type: QueryType) -> str:
        """
        基于复杂度和查询类型获取检索模式

        Args:
            complexity: 查询复杂度级别
            query_type: 查询类型分类

        Returns:
            str: 推荐的检索模式
        """
        # 检查查询类型是否有特定覆盖
        if query_type in self.type_mode_override:
            return self.type_mode_override[query_type]

        # 否则使用基于复杂度的映射
        return self.complexity_mode_map.get(complexity, "hybrid")


# ============================================================================
# 扩展的 RAG 类 - 添加自适应查询方法
# ============================================================================

class AdaptiveRAG(RAGAnything):
    """
    扩展 RAGAnything 类，添加自适应查询功能
    """

    async def aquery_adaptive(
        self, query: str, return_analysis: bool = False, **kwargs
    ) -> str | Dict[str, Any]:
        """
        自适应查询 - 基于查询复杂度自动选择最佳检索模式

        实现 Adaptive-RAG 方法：
        1. 分析查询以确定其复杂度和类型
        2. 路由到最合适的检索模式
        3. 使用选定的模式执行查询

        Args:
            query: 查询文本
            return_analysis: 如果为 True，返回包含结果和路由分析的字典
            **kwargs: 其他查询参数（将传递给 QueryParam）

        Returns:
            str: 查询结果（当 return_analysis=False）
            Dict: 包含 'result' 和 'analysis'（当 return_analysis=True）

        示例:
            # 简单使用
            result = await rag.aquery_adaptive("什么是机器学习?")

            # 获取分析详情
            response = await rag.aquery_adaptive(
                "什么是机器学习?",
                return_analysis=True
            )
            print(f"使用的模式: {response['analysis']['recommended_mode']}")
            print(f"结果: {response['result']}")
        """
        # 确保 LightRAG 已初始化
        await self._ensure_lightrag_initialized()

        # 确保有 LLM 函数可用
        if not hasattr(self, "llm_model_func") or self.llm_model_func is None:
            if hasattr(self, "lightrag") and hasattr(self.lightrag, "llm_model_func"):
                llm_func = self.lightrag.llm_model_func
            else:
                raise ValueError("自适应查询路由需要 LLM 模型函数")
        else:
            llm_func = self.llm_model_func

        self.logger.info(f"分析查询以进行自适应路由: {query[:100]}...")

        # 创建路由器并分析查询
        router = AdaptiveQueryRouter(llm_func)
        analysis = await router.analyze_query(query)

        # 记录路由决策
        self.logger.info(
            f"查询被分类为 {analysis['complexity'].value} "
            f"({analysis['query_type'].value})"
        )
        self.logger.info(f"路由到模式: {analysis['recommended_mode']}")
        self.logger.info(f"推理: {analysis['reasoning']}")

        # 使用推荐的模式执行查询
        result = await self.aquery(query, mode=analysis["recommended_mode"], **kwargs)

        if return_analysis:
            return {
                "result": result,
                "analysis": {
                    "complexity": analysis["complexity"].value,
                    "query_type": analysis["query_type"].value,
                    "recommended_mode": analysis["recommended_mode"],
                    "reasoning": analysis["reasoning"],
                    "confidence": analysis["confidence"],
                },
            }
        else:
            return result

    async def aquery_multi_mode(
        self,
        query: str,
        modes: List[str] = None,
        return_all_results: bool = False,
        **kwargs,
    ) -> str | Dict[str, Any]:
        """
        多模式查询 - 尝试多个检索模式并使用 LLM 选择最佳结果

        此方法：
        1. 使用多个检索模式执行查询
        2. 使用 LLM 评估并选择最佳结果
        3. 返回最佳结果（或所有结果用于比较）

        Args:
            query: 查询文本
            modes: 要尝试的模式列表（默认: ["naive", "local", "hybrid", "global", "mix"]）
            return_all_results: 如果为 True，返回所有结果用于比较
            **kwargs: 其他查询参数（将传递给 QueryParam）

        Returns:
            str: 最佳查询结果（当 return_all_results=False）
            Dict: 所有结果及评估（当 return_all_results=True）

        示例:
            # 获取最佳结果
            result = await rag.aquery_multi_mode("什么是机器学习?")

            # 比较所有结果
            comparison = await rag.aquery_multi_mode(
                "什么是机器学习?",
                return_all_results=True
            )
            for mode, data in comparison['results'].items():
                print(f"{mode}: {data['result'][:100]}...")
        """
        # 确保 LightRAG 已初始化
        await self._ensure_lightrag_initialized()

        # 默认要尝试的模式
        if modes is None:
            modes = ["naive", "local", "hybrid", "global", "mix"]

        self.logger.info(f"使用模式执行多模式查询: {modes}")
        self.logger.info(f"查询: {query[:100]}...")

        # 使用所有模式执行查询
        results = {}
        for mode in modes:
            try:
                self.logger.info(f"尝试模式: {mode}")
                result = await self.aquery(query, mode=mode, **kwargs)
                results[mode] = {"result": result, "success": True, "error": None}
            except Exception as e:
                self.logger.error(f"模式 {mode} 出错: {str(e)}")
                results[mode] = {"result": None, "success": False, "error": str(e)}

        # 过滤成功的结果
        successful_results = {
            mode: data for mode, data in results.items() if data["success"]
        }

        if not successful_results:
            raise ValueError("所有检索模式都失败了")

        self.logger.info(
            f"成功从 {len(successful_results)} 个模式检索到结果"
        )

        # 使用 LLM 评估并选择最佳结果
        best_mode, evaluation = await self._evaluate_multi_mode_results(
            query, successful_results
        )

        self.logger.info(f"选择的最佳模式: {best_mode}")
        self.logger.info(f"评估: {evaluation}")

        if return_all_results:
            return {
                "best_result": successful_results[best_mode]["result"],
                "best_mode": best_mode,
                "evaluation": evaluation,
                "results": successful_results,
            }
        else:
            return successful_results[best_mode]["result"]

    async def _evaluate_multi_mode_results(
        self, query: str, results: Dict[str, Dict[str, Any]]
    ) -> tuple[str, str]:
        """
        使用 LLM 评估多个结果并选择最佳的一个

        Args:
            query: 原始查询
            results: 模式 -> 结果数据的字典

        Returns:
            tuple: (best_mode, evaluation_reasoning)
        """
        # 确保有 LLM 函数
        if not hasattr(self, "llm_model_func") or self.llm_model_func is None:
            if hasattr(self, "lightrag") and hasattr(self.lightrag, "llm_model_func"):
                llm_func = self.lightrag.llm_model_func
            else:
                # 如果没有 LLM，回退到第一个模式
                return list(results.keys())[0], "没有可用于评估的 LLM"
        else:
            llm_func = self.llm_model_func

        # 准备评估提示词
        results_text = ""
        for i, (mode, data) in enumerate(results.items(), 1):
            results_text += f"\n--- 结果 {i} (模式: {mode}) ---\n"
            results_text += data["result"]
            results_text += "\n"

        evaluation_prompt = f"""你正在评估同一查询的多个检索结果以选择最佳的一个。

用户查询: "{query}"

从不同模式检索到的结果:
{results_text}

请基于以下标准评估每个结果：
1. 与查询的相关性
2. 信息的完整性
3. 准确性和正确性
4. 清晰度和连贯性

选择最佳结果并用以下 JSON 格式响应：
{{
    "best_mode": "最佳结果的模式名称",
    "reasoning": "简要解释为什么这个结果最好（2-3句话）",
    "ranking": ["mode1", "mode2", ...] (所有模式从最好到最差的排名)
}}"""

        evaluation_system = "你是评估信息检索结果和选择最相关、最全面答案的专家。"

        try:
            # 获取 LLM 评估
            response = await llm_func(
                evaluation_prompt, system_prompt=evaluation_system
            )

            # 解析响应
            response_text = response.strip()
            if "```json" in response_text:
                start = response_text.find("```json") + 7
                end = response_text.find("```", start)
                response_text = response_text[start:end].strip()
            elif "```" in response_text:
                start = response_text.find("```") + 3
                end = response_text.find("```", start)
                response_text = response_text[start:end].strip()

            evaluation = json.loads(response_text)

            best_mode = evaluation.get("best_mode")
            reasoning = evaluation.get("reasoning", "未提供推理")

            # 验证 best_mode 在结果中
            if best_mode not in results:
                self.logger.warning(
                    f"LLM 选择了无效的模式 {best_mode}，使用第一个模式"
                )
                best_mode = list(results.keys())[0]

            return best_mode, reasoning

        except Exception as e:
            self.logger.error(f"评估结果时出错: {str(e)}")
            # 回退到第一个模式
            return list(results.keys())[0], f"评估失败: {str(e)}"

    # 同步版本
    def query_adaptive(
        self, query: str, return_analysis: bool = False, **kwargs
    ) -> str | Dict[str, Any]:
        """同步版本的自适应查询"""
        import asyncio
        from lightrag.utils import always_get_an_event_loop
        loop = always_get_an_event_loop()
        return loop.run_until_complete(
            self.aquery_adaptive(query, return_analysis=return_analysis, **kwargs)
        )

    def query_multi_mode(
        self,
        query: str,
        modes: List[str] = None,
        return_all_results: bool = False,
        **kwargs,
    ) -> str | Dict[str, Any]:
        """同步版本的多模式查询"""
        import asyncio
        from lightrag.utils import always_get_an_event_loop
        loop = always_get_an_event_loop()
        return loop.run_until_complete(
            self.aquery_multi_mode(
                query, modes=modes, return_all_results=return_all_results, **kwargs
            )
        )


# ============================================================================
# 日志配置
# ============================================================================

def configure_logging():
    """配置应用程序日志"""
    log_dir = os.getenv("LOG_DIR", os.getcwd())
    log_file_path = os.path.abspath(os.path.join(log_dir, "my_rag_example.log"))

    print(f"\n日志文件: {log_file_path}\n")
    os.makedirs(os.path.dirname(log_dir), exist_ok=True)

    log_max_bytes = int(os.getenv("LOG_MAX_BYTES", 10485760))
    log_backup_count = int(os.getenv("LOG_BACKUP_COUNT", 5))

    logging.config.dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "default": {
                    "format": "%(levelname)s: %(message)s",
                },
                "detailed": {
                    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                },
            },
            "handlers": {
                "console": {
                    "formatter": "default",
                    "class": "logging.StreamHandler",
                    "stream": "ext://sys.stderr",
                },
                "file": {
                    "formatter": "detailed",
                    "class": "logging.handlers.RotatingFileHandler",
                    "filename": log_file_path,
                    "maxBytes": log_max_bytes,
                    "backupCount": log_backup_count,
                    "encoding": "utf-8",
                },
            },
            "loggers": {
                "lightrag": {
                    "handlers": ["console", "file"],
                    "level": "INFO",
                    "propagate": False,
                },
            },
        }
    )

    logger.setLevel(logging.INFO)
    set_verbose_debug(os.getenv("VERBOSE", "false").lower() == "true")


# ============================================================================
# 演示函数
# ============================================================================

async def demo_adaptive_queries(
    file_path: str,
    output_dir: str,
    api_key: str,
    base_url: str = None,
    working_dir: str = None,
    parser: str = None,
):
    """
    演示自适应和多模式查询功能
    """
    try:
        # 创建 RAGAnything 配置
        config = RAGAnythingConfig(
            working_dir=working_dir or "./rag_storage",
            parser=parser,
            parse_method="auto",
            enable_image_processing=True,
            enable_table_processing=True,
            enable_equation_processing=True,
        )

        # 定义 LLM 模型函数
        def llm_model_func(prompt, system_prompt=None, history_messages=[], **kwargs):
            return openai_complete_if_cache(
                "gpt-4o-mini",
                prompt,
                system_prompt=system_prompt,
                history_messages=history_messages,
                api_key=api_key,
                base_url=base_url,
                **kwargs,
            )

        # 定义视觉模型函数
        def vision_model_func(
            prompt,
            system_prompt=None,
            history_messages=[],
            image_data=None,
            messages=None,
            **kwargs,
        ):
            if messages:
                return openai_complete_if_cache(
                    "gpt-4o",
                    "",
                    system_prompt=None,
                    history_messages=[],
                    messages=messages,
                    api_key=api_key,
                    base_url=base_url,
                    **kwargs,
                )
            elif image_data:
                return openai_complete_if_cache(
                    "gpt-4o",
                    "",
                    system_prompt=None,
                    history_messages=[],
                    messages=[
                        {"role": "system", "content": system_prompt}
                        if system_prompt
                        else None,
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/jpeg;base64,{image_data}"
                                    },
                                },
                            ],
                        }
                        if image_data
                        else {"role": "user", "content": prompt},
                    ],
                    api_key=api_key,
                    base_url=base_url,
                    **kwargs,
                )
            else:
                return llm_model_func(prompt, system_prompt, history_messages, **kwargs)

        # 定义嵌入函数
        embedding_dim = int(os.getenv("EMBEDDING_DIM", "3072"))
        embedding_model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-large")

        embedding_func = EmbeddingFunc(
            embedding_dim=embedding_dim,
            max_token_size=8192,
            func=lambda texts: openai_embed(
                texts,
                model=embedding_model,
                api_key=api_key,
                base_url=base_url,
            ),
        )

        # 初始化 AdaptiveRAG（使用扩展的类）
        rag = AdaptiveRAG(
            config=config,
            llm_model_func=llm_model_func,
            vision_model_func=vision_model_func,
            embedding_func=embedding_func,
        )

        # 处理文档
        logger.info(f"处理文档: {file_path}")
        await rag.process_document_complete(
            file_path=file_path, output_dir=output_dir, parse_method="auto"
        )

        print("\n" + "=" * 80)
        print("自适应查询路由演示")
        print("=" * 80)

        # 测试不同复杂度级别的查询
        test_queries = [
            {
                "query": "文档的标题是什么？",
                "expected_complexity": "simple",
                "description": "简单事实性问题",
            },
            {
                "query": "系统如何工作，其主要组件是什么？",
                "expected_complexity": "moderate",
                "description": "需要一些综合的中等问题",
            },
            {
                "query": "分析整体方法论并讨论其对该领域的影响，比较提到的不同方法。",
                "expected_complexity": "complex",
                "description": "复杂分析问题",
            },
        ]

        print("\n--- 1. 自适应查询路由 ---")
        print("系统分析每个查询并自动选择最佳检索模式。\n")

        for i, test in enumerate(test_queries, 1):
            print(f"\n[查询 {i}] {test['description']}")
            print(f"问题: {test['query']}")
            print(f"预期复杂度: {test['expected_complexity']}")

            # 使用带分析的自适应查询
            response = await rag.aquery_adaptive(
                test["query"], return_analysis=True
            )

            print(f"\n路由决策:")
            print(f"  - 检测到的复杂度: {response['analysis']['complexity']}")
            print(f"  - 查询类型: {response['analysis']['query_type']}")
            print(f"  - 选择的模式: {response['analysis']['recommended_mode']}")
            print(f"  - 置信度: {response['analysis']['confidence']:.2f}")
            print(f"  - 推理: {response['analysis']['reasoning']}")
            print(f"\n答案预览: {response['result'][:200]}...")
            print("-" * 80)

        print("\n--- 2. 多模式查询与 LLM 评估 ---")
        print("系统尝试多个检索模式并使用 LLM 选择最佳结果。\n")

        # 示例：为中等复杂度问题尝试多个模式
        example_query = "本文档的关键发现和结论是什么？"

        print(f"查询: {example_query}\n")
        print("尝试模式: naive, local, hybrid, global, mix\n")

        # 从所有模式获取结果并进行比较
        multi_mode_response = await rag.aquery_multi_mode(
            example_query,
            modes=["naive", "local", "hybrid", "global", "mix"],
            return_all_results=True,
        )

        print(f"选择的最佳模式: {multi_mode_response['best_mode']}")
        print(f"评估: {multi_mode_response['evaluation']}\n")

        print("所有模式的结果:")
        for mode, data in multi_mode_response["results"].items():
            result_preview = data["result"][:150].replace("\n", " ")
            print(f"\n  [{mode.upper()}]")
            print(f"  预览: {result_preview}...")

        print(f"\n\n最终答案 (使用 {multi_mode_response['best_mode']} 模式):")
        print(multi_mode_response["best_result"])

        print("\n" + "=" * 80)
        print("比较: 自适应 vs 多模式")
        print("=" * 80)
        print("""
自适应查询路由:
  ✓ 快速 - 只需一次检索调用
  ✓ 高效 - LLM 仅用于分类
  ✓ 适合有许多查询的生产环境
  ✓ 推荐用于大多数用例

多模式查询:
  ✓ 全面 - 尝试所有模式
  ✓ 最佳质量 - LLM 选择最佳结果
  ✓ 适合关键查询
  ✓ 更高成本（多次检索 + 评估）
  ✓ 推荐用于质量 > 速度/成本的场景
        """)

        print("\n--- 3. 实际使用示例 ---\n")

        # 显示不带分析详情的简单使用
        print("简单使用（只获取答案）:")
        simple_result = await rag.aquery_adaptive(
            "文档的主题是什么？"
        )
        print(f"答案: {simple_result[:200]}...\n")

        print("=" * 80)
        print("\n演示成功完成!")

    except Exception as e:
        logger.error(f"演示出错: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())


# ============================================================================
# 主函数
# ============================================================================

def main():
    """运行示例的主函数"""
    parser = argparse.ArgumentParser(
        description="自适应多模式 RAG 查询示例"
    )
    parser.add_argument("file_path", help="要处理的文档路径")
    parser.add_argument(
        "--working_dir", "-w", default="./rag_storage", help="工作目录路径"
    )
    parser.add_argument(
        "--output", "-o", default="./output", help="输出目录路径"
    )
    parser.add_argument(
        "--api-key",
        default=os.getenv("LLM_BINDING_API_KEY"),
        help="OpenAI API 密钥（默认使用 LLM_BINDING_API_KEY 环境变量）",
    )
    parser.add_argument(
        "--base-url",
        default=os.getenv("LLM_BINDING_HOST"),
        help="API 的可选基础 URL",
    )
    parser.add_argument(
        "--parser",
        default=os.getenv("PARSER", "mineru"),
        help="要使用的解析器（mineru 或 docling）",
    )

    args = parser.parse_args()

    # 检查是否提供了 API 密钥
    if not args.api_key:
        logger.error("错误: 需要 OpenAI API 密钥")
        logger.error("设置 api 密钥环境变量或使用 --api-key 选项")
        return

    # 如果指定，创建输出目录
    if args.output:
        os.makedirs(args.output, exist_ok=True)

    # 运行演示
    asyncio.run(
        demo_adaptive_queries(
            args.file_path,
            args.output,
            args.api_key,
            args.base_url,
            args.working_dir,
            args.parser,
        )
    )


if __name__ == "__main__":
    # 首先配置日志
    configure_logging()

    print("\n" + "=" * 80)
    print("自适应多模式 RAG 查询演示")
    print("=" * 80)
    print("\n此演示展示了两个高级查询功能:")
    print("1. 自适应查询路由 - 基于复杂度自动选择最佳模式")
    print("2. 多模式查询 - 尝试多个模式，LLM 选择最佳结果")
    print("\n基于 Adaptive-RAG 研究论文方法")
    print("=" * 80 + "\n")

    main()
