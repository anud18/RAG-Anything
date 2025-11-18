#!/usr/bin/env python
"""
Adaptive RAG Example using QueryRouter

This example demonstrates how to use the QueryRouter module for intelligent
query classification and mode selection in a RAG pipeline.

Features:
- Uses QueryRouter for automatic complexity classification
- Dynamically selects optimal retrieval modes
- Includes source citation and two-stage evaluation
- Configurable retrieval strategy (adaptive vs comprehensive)
"""

import os
import argparse
import asyncio
import logging
import logging.config
from pathlib import Path
from tenacity import retry, stop_after_attempt, wait_exponential

# Add project root directory to Python path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from lightrag.llm.openai import openai_complete_if_cache, openai_embed
from lightrag.utils import EmbeddingFunc, logger, set_verbose_debug
from lightrag.kg.shared_storage import initialize_pipeline_status
from lightrag import LightRAG, QueryParam
from raganything import RAGAnything, RAGAnythingConfig

from dotenv import load_dotenv
from query_router import QueryRouter, RouterConfig

load_dotenv(dotenv_path=".env", override=True)


def configure_logging():
    """Configure logging for the application"""
    log_dir = os.getenv("LOG_DIR", os.getcwd())
    log_file_path = os.path.abspath(os.path.join(log_dir, "adaptive_rag_example.log"))

    print(f"\nAdaptive RAG example log file: {log_file_path}\n")
    os.makedirs(os.path.dirname(log_dir), exist_ok=True)

    log_max_bytes = int(os.getenv("LOG_MAX_BYTES", 10485760))
    log_backup_count = int(os.getenv("LOG_BACKUP_COUNT", 5))

    logging.config.dictConfig({
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "default": {"format": "%(levelname)s: %(message)s"},
            "detailed": {"format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"},
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
    })

    logger.setLevel(logging.INFO)
    set_verbose_debug(os.getenv("VERBOSE", "false").lower() == "true")


async def process_with_adaptive_rag(
    file_path: str,
    output_dir: str,
    api_key: str,
    base_url: str = None,
    working_dir: str = None,
    parser: str = None,
):
    """
    Process document with Adaptive RAG using QueryRouter

    Args:
        file_path: Path to the document
        output_dir: Output directory for results
        api_key: OpenAI API key
        base_url: Optional base URL for API
        working_dir: Working directory for RAG storage
        parser: Parser to use (mineru or docling)
    """
    try:
        # Create RAGAnything configuration
        config = RAGAnythingConfig(
            working_dir=working_dir or "./rag_storage",
            parser=parser,
            parse_method="auto",
            enable_image_processing=True,
            enable_table_processing=True,
            enable_equation_processing=True,
        )

        llm_model = os.getenv("LLM_MODEL", "google/gemini-3-pro-preview")
        vision_model = os.getenv("VISION_MODEL", "google/gemini-3-pro-preview")
        llm_timeout = int(os.getenv("LLM_TIMEOUT", "300"))
        embedding_timeout = int(os.getenv("EMBEDDING_TIMEOUT", "120"))

        # Define LLM model function
        async def llm_model_func_with_retry(prompt, system_prompt=None, history_messages=[], **kwargs):
            kwargs.setdefault('timeout', llm_timeout)
            return await openai_complete_if_cache(
                llm_model,
                prompt,
                system_prompt=system_prompt,
                history_messages=history_messages,
                api_key=api_key,
                base_url=base_url,
                **kwargs,
            )

        def llm_model_func(prompt, system_prompt=None, history_messages=[], **kwargs):
            return asyncio.create_task(
                llm_model_func_with_retry(prompt, system_prompt, history_messages, **kwargs)
            )

        # Define vision model function
        @retry(
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=1, min=4, max=10),
            reraise=True
        )
        async def vision_model_func_with_retry(
            prompt,
            system_prompt=None,
            history_messages=[],
            image_data=None,
            messages=None,
            **kwargs,
        ):
            kwargs.setdefault('timeout', llm_timeout)
            if messages:
                return await openai_complete_if_cache(
                    vision_model, "", system_prompt=None, history_messages=[],
                    messages=messages, api_key=api_key, base_url=base_url, **kwargs,
                )
            elif image_data:
                return await openai_complete_if_cache(
                    vision_model, "", system_prompt=None, history_messages=[],
                    messages=[
                        {"role": "system", "content": system_prompt} if system_prompt else None,
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}},
                            ],
                        } if image_data else {"role": "user", "content": prompt},
                    ],
                    api_key=api_key, base_url=base_url, **kwargs,
                )
            else:
                return await llm_model_func_with_retry(prompt, system_prompt, history_messages, **kwargs)

        def vision_model_func(prompt, system_prompt=None, history_messages=[], image_data=None, messages=None, **kwargs):
            return asyncio.create_task(
                vision_model_func_with_retry(prompt, system_prompt, history_messages, image_data, messages, **kwargs)
            )

        # Define embedding function
        embedding_dim = int(os.getenv("EMBEDDING_DIM", "3072"))
        embedding_model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-large")

        embedding_func = EmbeddingFunc(
            embedding_dim=embedding_dim,
            max_token_size=8192,
            func=lambda texts: openai_embed(
                texts, model=embedding_model, api_key=api_key, base_url=base_url,
                client_configs={"timeout": 30},
            ),
        )

        # Initialize RAGAnything
        rag = RAGAnything(
            config=config,
            llm_model_func=llm_model_func,
            vision_model_func=vision_model_func,
            embedding_func=embedding_func,
        )

        try:
            if rag.lightrag is None:
                rag.lightrag = LightRAG(
                    working_dir=working_dir or "./rag_storage",
                    llm_model_func=llm_model_func,
                    embedding_func=embedding_func,
                    max_entity_tokens=6000,
                    max_relation_tokens=8000,
                    chunk_token_size=2000,
                    chunk_overlap_token_size=500,
                )
        except Exception as e:
            logger.error(f"Error initializing LightRAG: {str(e)}")
            raise e
        finally:
            await rag.lightrag.initialize_storages()
            await initialize_pipeline_status()

        # ========================================================================
        # Initialize QueryRouter for Adaptive RAG
        # ========================================================================

        # Optional: Create custom configuration
        router_config = RouterConfig(
            simple_modes=["naive", "local"],
            moderate_modes=["hybrid", "local"],
            complex_modes=["mix", "global", "hybrid"],
            confidence_threshold=0.3
        )

        query_router = QueryRouter(
            llm_func=llm_model_func_with_retry,
            config=router_config,
            logger=logger
        )

        logger.info("✓ QueryRouter initialized")
        logger.info(f"  - Simple queries    → {router_config.simple_modes}")
        logger.info(f"  - Moderate queries  → {router_config.moderate_modes}")
        logger.info(f"  - Complex queries   → {router_config.complex_modes}")

        # ========================================================================
        # Helper Functions
        # ========================================================================

        async def query_with_modes(query_text: str, modes: list) -> dict:
            """Query using specified modes and return results with source context"""
            results = {}

            for mode in modes:
                try:
                    logger.info(f"  Querying with mode: {mode}")

                    # Get the raw prompt with retrieved context
                    query_param = QueryParam(mode=mode, only_need_prompt=True)
                    raw_prompt = await rag.lightrag.aquery(query_text, query_param)

                    # Create enhanced prompt asking LLM to cite sources
                    citation_prompt = f"""{raw_prompt}

IMPORTANT: When answering, please cite the specific sources from the context above.
For each key point in your answer, indicate which part of the retrieved context it comes from.
Use this format: [Source: brief description of the source section]

Please provide your answer with source citations."""

                    # Get answer with citations
                    answer_with_sources = await llm_model_func_with_retry(
                        citation_prompt,
                        system_prompt="You are a helpful assistant that provides answers with clear source citations from the given context."
                    )

                    results[mode] = {
                        "answer": answer_with_sources,
                        "raw_context": raw_prompt,
                        "mode": mode
                    }

                except Exception as e:
                    logger.warning(f"  Error in mode {mode}: {str(e)}")
                    results[mode] = {
                        "answer": f"Error: {str(e)}",
                        "raw_context": "",
                        "mode": mode
                    }

            return results

        async def evaluate_results(query_text: str, mode_results: dict):
            """Two-stage evaluation of results"""

            # Stage 1: Individual evaluation
            logger.info("  Stage 1: Individual evaluation and scoring...")
            mode_evaluations = {}

            for mode, result_data in mode_results.items():
                answer = result_data["answer"]

                eval_prompt = f"""Evaluate this answer to the user's query.

User Query: {query_text}

Answer from {mode.upper()} mode:
{answer}

Please evaluate based on these criteria and provide a score (0-10) for each:
1. Completeness - Does it fully answer the question?
2. Accuracy - Is the information correct and well-sourced?
3. Relevance - Does it stay focused on the query?
4. Clarity - Is it well-structured and easy to understand?
5. Source Citation - Does it properly cite sources from retrieved context?

Provide your evaluation in this EXACT format:
Completeness: [score]/10
Accuracy: [score]/10
Relevance: [score]/10
Clarity: [score]/10
Source Citation: [score]/10
Total Score: [sum of scores]/50
Brief Summary: [2-3 sentences summarizing the answer's strengths and weaknesses]
"""

                try:
                    evaluation = await llm_model_func_with_retry(
                        eval_prompt,
                        system_prompt="You are an expert evaluator. Be objective and precise in your scoring."
                    )
                    mode_evaluations[mode] = evaluation
                    logger.info(f"    ✓ Evaluated {mode} mode")
                except Exception as e:
                    logger.warning(f"    ✗ Error evaluating {mode}: {str(e)}")
                    mode_evaluations[mode] = f"Evaluation error: {str(e)}"

            # Stage 2: Comparative analysis
            logger.info("  Stage 2: Comparative analysis and final selection...")

            comparison_prompt = f"""Based on the individual evaluations below, determine which retrieval mode performed best for this query.

User Query: {query_text}

Individual Mode Evaluations:
"""

            for mode, evaluation in mode_evaluations.items():
                comparison_prompt += f"\n{'='*60}\n{mode.upper()} MODE:\n{evaluation}\n"

            comparison_prompt += f"""
{'='*60}

Based on these evaluations, please:
1. Identify the best performing mode(s)
2. Explain why this mode performed best
3. If beneficial, suggest how to combine insights from multiple modes
4. Provide a final recommended answer that incorporates the best elements

Provide your analysis in this format:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
EVALUATION SUMMARY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Best Mode: [mode name]
Runner-up: [mode name if applicable]

Reasoning: [Detailed explanation of why the best mode performed better]

Key Strengths:
- [List specific strengths of the best answer]

Potential Improvements:
- [Any suggestions for improvement]

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RECOMMENDED FINAL ANSWER
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

[Provide the best answer, optionally enhanced by combining insights from multiple modes. Include source citations.]
"""

            try:
                final_evaluation = await llm_model_func_with_retry(
                    comparison_prompt,
                    system_prompt="You are an expert evaluator making final recommendations. Be thorough and objective."
                )

                return {
                    "individual_evaluations": mode_evaluations,
                    "final_evaluation": final_evaluation
                }
            except Exception as e:
                logger.error(f"  Error in final evaluation: {str(e)}")
                hybrid_result = mode_results.get("hybrid", mode_results.get("mix", {}))
                fallback_answer = hybrid_result.get("answer", "N/A") if isinstance(hybrid_result, dict) else str(hybrid_result)
                return {
                    "individual_evaluations": mode_evaluations,
                    "final_evaluation": f"Final evaluation failed: {str(e)}\n\nDefaulting to HYBRID/MIX mode result:\n{fallback_answer}"
                }

        # ========================================================================
        # Query Processing with Adaptive RAG
        # ========================================================================

        logger.info("\nQuerying processed document with Adaptive RAG:")

        # Configuration: Set retrieval strategy
        retrieval_strategy = os.getenv("RETRIEVAL_STRATEGY", "adaptive")

        logger.info(f"\nRetrieval Strategy: {retrieval_strategy.upper()}")
        logger.info(f"  - adaptive: Smart mode selection via QueryRouter")
        logger.info(f"  - comprehensive: Query all 5 modes for comparison\n")

        # Example queries
        text_queries = [
            "給我PCD急救人員名單，你可以使用工具計算",
            "給我樹林廠先進管理課急救人員，你可以使用工具計算",
            "給我樹林廠急救人員總共有幾位，你可以使用工具計算",
        ]

        for query in text_queries:
            logger.info(f"\n{'='*80}")
            logger.info(f"[Text Query]: {query}")
            logger.info(f"{'='*80}")

            try:
                # ============================================================
                # Step 0: Adaptive RAG - Use QueryRouter
                # ============================================================

                selected_modes = None
                routing_result = None

                if retrieval_strategy == "adaptive":
                    logger.info("\n[Step 0/4] 🧠 Adaptive RAG: Using QueryRouter...")

                    # Use QueryRouter's complete routing pipeline
                    routing_result = await query_router.route_query(query)

                    classification = routing_result['classification']
                    selected_modes = routing_result['selected_modes']
                    efficiency_gain = routing_result['efficiency_gain']

                    logger.info(f"  ├─ Complexity: {classification.complexity.upper()}")
                    logger.info(f"  ├─ Confidence: {classification.confidence:.2f}")
                    logger.info(f"  ├─ Reasoning: {classification.reasoning}")
                    logger.info(f"  ├─ Selected Modes: {', '.join(m.upper() for m in selected_modes)}")
                    logger.info(f"  └─ Efficiency Gain: {efficiency_gain:.0f}%")

                    # Save classification to file
                    classification_file = os.path.join(output_dir, "adaptive_classifications.txt")
                    with open(classification_file, "a", encoding="utf-8") as f:
                        f.write(f"\n{'='*80}\n")
                        f.write(f"Query: {query}\n")
                        f.write(f"{'='*80}\n")
                        f.write(f"Complexity: {classification.complexity}\n")
                        f.write(f"Confidence: {classification.confidence:.2f}\n")
                        f.write(f"Reasoning: {classification.reasoning}\n")
                        f.write(f"Selected Modes: {', '.join(selected_modes)}\n")
                        f.write(f"Efficiency Gain: {efficiency_gain:.0f}%\n")
                        f.write(f"\nFull Classification:\n{classification.raw_classification}\n")

                # ============================================================
                # Step 1: Query with selected modes
                # ============================================================

                step_num = "1/4" if retrieval_strategy == "adaptive" else "1/3"
                mode_desc = f"selected modes ({', '.join(selected_modes)})" if selected_modes else "all modes"
                logger.info(f"\n[Step {step_num}] Querying with {mode_desc} (with source citations)...")

                if selected_modes is None:
                    selected_modes = ["local", "global", "hybrid", "naive", "mix"]

                mode_results = await query_with_modes(query, modes=selected_modes)

                # Save detailed results
                results_file = os.path.join(output_dir, "mode_results_with_sources.txt")
                with open(results_file, "a", encoding="utf-8") as f:
                    f.write(f"\n\n{'='*80}\n")
                    f.write(f"Query: {query}\n")
                    f.write(f"{'='*80}\n\n")
                    for mode, result_data in mode_results.items():
                        f.write(f"【{mode.upper()} MODE】\n")
                        f.write(f"Answer:\n{result_data['answer']}\n")
                        f.write(f"\n--- Retrieved Context ---\n")
                        context_preview = result_data['raw_context'][:2000]
                        if len(result_data['raw_context']) > 2000:
                            context_preview += f"\n... (truncated, total length: {len(result_data['raw_context'])} chars)"
                        f.write(f"{context_preview}\n")
                        f.write(f"{'-'*80}\n\n")

                # ============================================================
                # Step 2: Two-stage LLM evaluation
                # ============================================================

                step_num = "2/4" if retrieval_strategy == "adaptive" else "2/3"
                logger.info(f"\n[Step {step_num}] Two-stage LLM evaluation...")
                evaluation_results = await evaluate_results(query, mode_results)

                # ============================================================
                # Step 3: Log and save evaluation
                # ============================================================

                step_num = "3/4" if retrieval_strategy == "adaptive" else "3/3"
                logger.info(f"\n[Step {step_num}] Evaluation Complete!")
                logger.info(f"\n{evaluation_results['final_evaluation']}")

                # ============================================================
                # Step 4: Summary and insights (Adaptive RAG only)
                # ============================================================

                if retrieval_strategy == "adaptive" and routing_result:
                    logger.info(f"\n[Step 4/4] 📊 Adaptive RAG Summary:")
                    logger.info(f"  ├─ Query Complexity: {routing_result['classification'].complexity.upper()}")
                    logger.info(f"  ├─ Modes Used: {', '.join(m.upper() for m in selected_modes)}")
                    logger.info(f"  ├─ Modes Saved: {5 - len(selected_modes)} mode(s) skipped")
                    logger.info(f"  └─ Efficiency Gain: ~{routing_result['efficiency_gain']:.0f}% reduction in API calls")

                # Save comprehensive evaluation
                eval_file = os.path.join(output_dir, "evaluations_detailed.txt")
                with open(eval_file, "a", encoding="utf-8") as f:
                    f.write(f"\n\n{'='*80}\n")
                    f.write(f"Query: {query}\n")
                    f.write(f"{'='*80}\n\n")

                    # Save individual evaluations
                    f.write("STAGE 1: INDIVIDUAL MODE EVALUATIONS\n")
                    f.write("="*80 + "\n\n")
                    for mode, eval_text in evaluation_results['individual_evaluations'].items():
                        f.write(f"{mode.upper()} MODE:\n")
                        f.write(f"{eval_text}\n")
                        f.write(f"{'-'*60}\n\n")

                    # Save final evaluation
                    f.write("\n" + "="*80 + "\n")
                    f.write("STAGE 2: FINAL COMPARATIVE EVALUATION\n")
                    f.write("="*80 + "\n\n")
                    f.write(f"{evaluation_results['final_evaluation']}\n")

            except Exception as e:
                logger.warning(f"Error in query processing: {str(e)}")
                import traceback
                logger.error(traceback.format_exc())
            finally:
                await rag.lightrag.finalize_storages()

    except Exception as e:
        logger.error(f"Error processing with Adaptive RAG: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())


def main():
    """Main function to run the Adaptive RAG example"""
    parser = argparse.ArgumentParser(description="Adaptive RAG Example with QueryRouter")
    parser.add_argument("file_path", help="Path to the document to process")
    parser.add_argument("--working_dir", "-w", default="./rag_storage", help="Working directory path")
    parser.add_argument("--output", "-o", default="./output", help="Output directory path")
    parser.add_argument("--api-key", default=os.getenv("LLM_BINDING_API_KEY"), help="OpenAI API key")
    parser.add_argument("--base-url", default=os.getenv("LLM_BINDING_HOST"), help="Optional base URL for API")
    parser.add_argument("--parser", default=os.getenv("PARSER", "mineru"), help="Parser to use (mineru or docling)")

    args = parser.parse_args()

    # Check if API key is provided
    if not args.api_key or os.getenv("LLM_BINDING_API_KEY") is None:
        logger.error("Error: OpenAI API key is required")
        logger.error("Set LLM_BINDING_API_KEY environment variable or use --api-key option")
        return

    # Create output directory
    if args.output:
        os.makedirs(args.output, exist_ok=True)

    # Process with Adaptive RAG
    asyncio.run(
        process_with_adaptive_rag(
            args.file_path,
            args.output,
            args.api_key,
            args.base_url,
            args.working_dir,
            args.parser,
        )
    )


if __name__ == "__main__":
    # Configure logging first
    configure_logging()

    print("Adaptive RAG Example with QueryRouter")
    print("=" * 50)
    print("Intelligent query-driven mode selection")
    print("=" * 50)

    main()
