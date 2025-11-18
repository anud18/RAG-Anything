#!/usr/bin/env python
"""
Example script demonstrating the integration of MinerU parser with RAGAnything

This example shows how to:
1. Process documents with RAGAnything using MinerU parser
2. Perform pure text queries using aquery() method
3. Perform multimodal queries with specific multimodal content using aquery_with_multimodal() method
4. Handle different types of multimodal content (tables, equations) in queries
"""

import os
import argparse
import asyncio
import logging
import logging.config
from pathlib import Path
from tenacity import retry, stop_after_attempt, wait_exponential
from lightrag import LightRAG


# Add project root directory to Python path
import sys

sys.path.append(str(Path(__file__).parent.parent))
from lightrag.llm.openai import openai_complete_if_cache, openai_embed
from lightrag.utils import EmbeddingFunc, logger, set_verbose_debug
from lightrag.kg.shared_storage import initialize_pipeline_status
from lightrag import LightRAG
from raganything import RAGAnything, RAGAnythingConfig

from dotenv import load_dotenv
from lightrag import QueryParam  


load_dotenv(dotenv_path=".env", override=True)

def configure_logging():
    """Configure logging for the application"""
    # Get log directory path from environment variable or use current directory
    log_dir = os.getenv("LOG_DIR", os.getcwd())
    log_file_path = os.path.abspath(os.path.join(log_dir, "raganything_example.log"))

    print(f"\nRAGAnything example log file: {log_file_path}\n")
    os.makedirs(os.path.dirname(log_dir), exist_ok=True)

    # Get log file max size and backup count from environment variables
    log_max_bytes = int(os.getenv("LOG_MAX_BYTES", 10485760))  # Default 10MB
    log_backup_count = int(os.getenv("LOG_BACKUP_COUNT", 5))  # Default 5 backups

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

    # Set the logger level to INFO
    logger.setLevel(logging.INFO)
    # Enable verbose debug if needed
    set_verbose_debug(os.getenv("VERBOSE", "false").lower() == "true")


async def process_with_rag(
    file_path: str,
    output_dir: str,
    api_key: str,
    base_url: str = None,
    working_dir: str = None,
    parser: str = None,
):
    """
    Process document with RAGAnything

    Args:
        file_path: Path to the document
        output_dir: Output directory for RAG results
        api_key: OpenAI API key
        base_url: Optional base URL for API
        working_dir: Working directory for RAG storage
    """
    try:
        # Create RAGAnything configuration
        config = RAGAnythingConfig(
            working_dir=working_dir or "./rag_storage",
            parser=parser,  # Parser selection: mineru or docling
            parse_method="auto",  # Parse method: auto, ocr, or txt
            enable_image_processing=True,
            enable_table_processing=True,
            enable_equation_processing=True,
        )
        llm_model = os.getenv("LLM_MODEL", "google/gemini-3-pro-preview")
        vision_model = os.getenv("VISION_MODEL", "google/gemini-3-pro-preview")
        print(f"llm_model: {llm_model}")
        #llm_model = os.getenv("LLM_MODEL", "google/gemini-2.5-flash")
        #vision_model = os.getenv("VISION_MODEL", "google/gemini-2.5-flash")
        
        # Get timeout settings
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

        # Define vision model function for image processing with timeout
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
            # If messages format is provided (for multimodal VLM enhanced query), use it directly
            if messages:
                return await openai_complete_if_cache(
                    vision_model,
                    "",
                    system_prompt=None,
                    history_messages=[],
                    messages=messages,
                    api_key=api_key,
                    base_url=base_url,
                    **kwargs,
                )
            # Traditional single image format
            elif image_data:
                return await openai_complete_if_cache(
                    vision_model,
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
            # Pure text format
            else:
                return await llm_model_func_with_retry(prompt, system_prompt, history_messages, **kwargs)

        def vision_model_func(
            prompt,
            system_prompt=None,
            history_messages=[],
            image_data=None,
            messages=None,
            **kwargs,
        ):
            return asyncio.create_task(
                vision_model_func_with_retry(
                    prompt, system_prompt, history_messages, image_data, messages, **kwargs
                )
            )


        # Define embedding function - using environment variables for configuration
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
                client_configs={
                    "timeout": 30,  
                },
            ),
        )

        # Initialize RAGAnything with new dataclass structure
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


        # # Process document
        # try:
        #     # await rag.process_document_complete(
        #     #     file_path=file_path, output_dir=output_dir, parse_method="auto"
        #     # )
        #     await rag.process_folder_complete(
        #         folder_path=file_path, output_dir=output_dir, parse_method="auto",
        #         file_extensions=[".pdf", ".docx", ".pptx"],
        #         recursive=True,
        #         max_workers=4
        #     )

        # except Exception as e:
        #     logger.error(f"Error processing document: {str(e)}")
        # finally:
        #     await rag.finalize_storages()


        # Helper function to query with all modes and get source context
        async def query_all_modes_with_context(query_text: str):
            """Query using all retrieval modes and return results with source context"""
            modes = ["local", "global", "hybrid", "naive", "mix"]
            results = {}

            for mode in modes:
                try:
                    logger.info(f"  Querying with mode: {mode}")

                    # Step 1: Get the raw prompt with retrieved context
                    query_param = QueryParam(mode=mode, only_need_prompt=True)
                    raw_prompt = await rag.lightrag.aquery(query_text, query_param)

                    # Step 2: Create enhanced prompt asking LLM to cite sources
                    citation_prompt = f"""{raw_prompt}

IMPORTANT: When answering, please cite the specific sources from the context above.
For each key point in your answer, indicate which part of the retrieved context it comes from.
Use this format: [Source: brief description of the source section]

Please provide your answer with source citations."""

                    # Step 3: Get answer with citations
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

        # Helper function for two-stage LLM evaluation to handle long context
        async def evaluate_best_answer_two_stage(query_text: str, mode_results: dict):
            """Two-stage evaluation to handle long context effectively"""

            # Stage 1: Evaluate each mode individually with scoring
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

            # Stage 2: Compare evaluations and select the best
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
                # Fallback to hybrid mode
                hybrid_result = mode_results.get("hybrid", mode_results.get("mix", {}))
                fallback_answer = hybrid_result.get("answer", "N/A") if isinstance(hybrid_result, dict) else str(hybrid_result)
                return {
                    "individual_evaluations": mode_evaluations,
                    "final_evaluation": f"Final evaluation failed: {str(e)}\n\nDefaulting to HYBRID/MIX mode result:\n{fallback_answer}"
                }

        # Example queries - demonstrating different query approaches
        logger.info("\nQuerying processed document:")

        # 1. Pure text queries using aquery()
        text_queries = [
            "給我PCD急救人員名單，你可以使用工具計算",
            "給我樹林廠先進管理課急救人員，你可以使用工具計算",
            "給我樹林廠急救人員總共有幾位，你可以使用工具計算",
            # "給我新竹廠健檢流程",
            # "給我鶯歌廠安委會名單",
        ]

        for query in text_queries:
            logger.info(f"\n{'='*80}")
            logger.info(f"[Text Query]: {query}")
            logger.info(f"{'='*80}")

            try:
                # Step 1: Get results from all modes with source context
                logger.info("\n[Step 1/3] Querying with all retrieval modes (with source citations)...")
                mode_results = await query_all_modes_with_context(query)

                # Save detailed results with context
                results_file = os.path.join(output_dir, "mode_results_with_sources.txt")
                with open(results_file, "a", encoding="utf-8") as f:
                    f.write(f"\n\n{'='*80}\n")
                    f.write(f"Query: {query}\n")
                    f.write(f"{'='*80}\n\n")
                    for mode, result_data in mode_results.items():
                        f.write(f"【{mode.upper()} MODE】\n")
                        f.write(f"Answer:\n{result_data['answer']}\n")
                        f.write(f"\n--- Retrieved Context ---\n")
                        # Save first 2000 chars of context to avoid huge files
                        context_preview = result_data['raw_context'][:2000]
                        if len(result_data['raw_context']) > 2000:
                            context_preview += f"\n... (truncated, total length: {len(result_data['raw_context'])} chars)"
                        f.write(f"{context_preview}\n")
                        f.write(f"{'-'*80}\n\n")

                # Step 2: Two-stage LLM evaluation
                logger.info("\n[Step 2/3] Two-stage LLM evaluation...")
                evaluation_results = await evaluate_best_answer_two_stage(query, mode_results)

                # Step 3: Log and save evaluation
                logger.info("\n[Step 3/3] Evaluation Complete!")
                logger.info(f"\n{evaluation_results['final_evaluation']}")

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
        logger.error(f"Error processing with RAG: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())


def main():
    """Main function to run the example"""
    parser = argparse.ArgumentParser(description="MinerU RAG Example")
    parser.add_argument("file_path", help="Path to the document to process")
    parser.add_argument(
        "--working_dir", "-w", default="./1560_rag_storage", help="Working directory path"
    )
    parser.add_argument(
        "--output", "-o", default="./1560_output", help="Output directory path"
    )
    parser.add_argument(
        "--api-key",
        default=os.getenv("LLM_BINDING_API_KEY"),
        help="OpenAI API key (defaults to LLM_BINDING_API_KEY env var)",
    )
    parser.add_argument(
        "--base-url",
        default=os.getenv("LLM_BINDING_HOST"),
        help="Optional base URL for API",
    )
    parser.add_argument(
        "--parser",
        default=os.getenv("PARSER", "mineru"),
        help="Optional base URL for API",
    )

    args = parser.parse_args()

    # Check if API key is provided
    if not args.api_key or os.getenv("LLM_BINDING_API_KEY") is None:
        logger.error("Error: OpenAI API key is required")
        logger.error("Set api key environment variable or use --api-key option")
        return

    # Create output directory if specified
    if args.output:
        os.makedirs(args.output, exist_ok=True)

    # Process with RAG
    asyncio.run(
        process_with_rag(
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

    print("RAGAnything Example")
    print("=" * 30)
    print("Processing document with multimodal RAG pipeline")
    print("=" * 30)

    main()
