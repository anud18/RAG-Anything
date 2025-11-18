#!/usr/bin/env python
"""
GAHR-MSR Enhanced RAG Script

This script integrates the GAHR-MSR (Graph-Augmented Hybrid Retrieval and
Multi-Stage Re-ranking) framework with the standard RAGAnything pipeline.

Features:
1. Standard RAG queries (original my_rag.py functionality)
2. GAHR-MSR high-precision queries with ColBERT re-ranking
3. Configurable query modes (standard vs GAHR-MSR)
4. All original features preserved (retry logic, timeouts, etc.)

Usage:
    # Standard mode
    python my_rag_gahr.py <document_path>

    # Enable GAHR-MSR re-ranking
    python my_rag_gahr.py <document_path> --use_gahr

    # GAHR-MSR with GPU
    python my_rag_gahr.py <document_path> --use_gahr --colbert_device cuda
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
from lightrag import QueryParam
from raganything import RAGAnything, RAGAnythingConfig

# Import GAHR-MSR framework
from gahr_msr_framework import GAHRMSRQuery, GAHRMSRConfig

from dotenv import load_dotenv

load_dotenv(dotenv_path=".env", override=True)


def configure_logging():
    """Configure logging for the application"""
    # Get log directory path from environment variable or use current directory
    log_dir = os.getenv("LOG_DIR", os.getcwd())
    log_file_path = os.path.abspath(os.path.join(log_dir, "my_rag_gahr.log"))

    print(f"\nGAHR-MSR RAG log file: {log_file_path}\n")
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
                "gahr_msr_framework": {
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
    use_gahr: bool = False,
    colbert_device: str = "cpu",
    final_top_k: int = 5,
):
    """
    Process document with RAGAnything and optionally use GAHR-MSR for queries

    Args:
        file_path: Path to the document
        output_dir: Output directory for RAG results
        api_key: OpenAI API key
        base_url: Optional base URL for API
        working_dir: Working directory for RAG storage
        parser: Parser to use (mineru or docling)
        use_gahr: Whether to use GAHR-MSR re-ranking
        colbert_device: Device for ColBERT (cpu, cuda, mps)
        final_top_k: Number of final results for GAHR-MSR
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
        logger.info(f"LLM Model: {llm_model}")
        logger.info(f"Vision Model: {vision_model}")

        # Get timeout settings
        llm_timeout = int(os.getenv("LLM_TIMEOUT", "300"))
        embedding_timeout = int(os.getenv("EMBEDDING_TIMEOUT", "120"))

        # Define LLM model function with retry
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

        # Initialize GAHR-MSR if requested
        gahr_query = None
        if use_gahr:
            logger.info("\n" + "="*70)
            logger.info("Initializing GAHR-MSR Framework")
            logger.info("="*70)

            gahr_config = GAHRMSRConfig(
                # Hybrid retrieval settings
                dense_top_k=100,
                sparse_top_k=100,
                hybrid_top_k=100,
                rrf_k=60,
                # Re-ranking settings
                enable_colbert_reranking=True,
                colbert_top_k=20,
                final_top_k=final_top_k,
                colbert_model="colbert-ir/colbertv2.0",
                colbert_device=colbert_device,
                # Graph filtering settings
                enable_graph_filtering=True,
                graph_filter_mode="any",
            )

            gahr_query = GAHRMSRQuery(rag, gahr_config)

            pipeline_info = gahr_query.get_pipeline_info()
            logger.info(f"  Framework: {pipeline_info['framework']} v{pipeline_info['version']}")
            logger.info(f"  ColBERT Device: {colbert_device}")
            logger.info(f"  Final Top-K: {final_top_k}")
            logger.info("="*70 + "\n")

        # Example queries - demonstrating different query approaches
        logger.info("\nQuerying processed document:")
        logger.info(f"Query Mode: {'GAHR-MSR (High-Precision Re-ranking)' if use_gahr else 'Standard RAG'}")
        logger.info("="*70 + "\n")

        # 1. Pure text queries using aquery()
        text_queries = [
            "給我PCD急救人員名單，你可以使用工具計算",
            "給我樹林廠先進管理課急救人員，你可以使用工具計算",
            "給我樹林廠急救人員總共有幾位，你可以使用工具計算",
            # "給我新竹廠健檢流程",
            # "給我鶯歌廠安委會名單",
        ]

        for i, query in enumerate(text_queries, 1):
            logger.info(f"\n{'='*70}")
            logger.info(f"Query {i}/{len(text_queries)}: {query}")
            logger.info('='*70)

            try:
                # Save raw prompt for debugging
                query_param = QueryParam(mode="naive", only_need_prompt=True)
                raw_prompt = await rag.lightrag.aquery(query, query_param)

                prompt_file = os.path.join(output_dir, f"raw_prompt_{'gahr' if use_gahr else 'standard'}.txt")
                with open(prompt_file, "a", encoding="utf-8") as f:
                    f.write(f"\n\n[Query {i}]: {query}\n")
                    f.write(raw_prompt)

                # Execute query based on mode
                if use_gahr and gahr_query:
                    logger.info("Using GAHR-MSR high-precision query...")
                    answer = await gahr_query.query(query, mode="hybrid")
                else:
                    logger.info("Using standard RAG query...")
                    answer = await rag.aquery(query, mode="mix")

                logger.info(f"\nAnswer: {answer}\n")

                # Save answer
                answer_file = os.path.join(output_dir, f"answers_{'gahr' if use_gahr else 'standard'}.txt")
                with open(answer_file, "a", encoding="utf-8") as f:
                    f.write(f"\n\n[Query {i}]: {query}\n")
                    f.write(f"Answer: {answer}\n")
                    f.write("="*70 + "\n")

            except Exception as e:
                logger.warning(f"Error in query: {str(e)}")
                import traceback
                logger.error(traceback.format_exc())
            finally:
                await rag.lightrag.finalize_storages()

        # Comparison mode: if GAHR-MSR is enabled, also show standard results
        if use_gahr and len(text_queries) > 0:
            logger.info("\n" + "="*70)
            logger.info("Comparison: Standard RAG vs GAHR-MSR")
            logger.info("="*70)

            comparison_query = text_queries[0]
            logger.info(f"\nComparison Query: {comparison_query}\n")

            # Standard RAG
            logger.info("--- Standard RAG ---")
            try:
                standard_answer = await rag.aquery(comparison_query, mode="mix")
                logger.info(f"Answer: {standard_answer}\n")
            except Exception as e:
                logger.error(f"Standard query failed: {e}")

            # GAHR-MSR
            logger.info("--- GAHR-MSR (with re-ranking) ---")
            try:
                gahr_answer = await gahr_query.query(comparison_query, mode="hybrid")
                logger.info(f"Answer: {gahr_answer}\n")
            except Exception as e:
                logger.error(f"GAHR-MSR query failed: {e}")

            logger.info("="*70 + "\n")

        logger.info("\nAll queries completed successfully!")

    except Exception as e:
        logger.error(f"Error processing with RAG: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())


def main():
    """Main function to run the example"""
    parser = argparse.ArgumentParser(
        description="GAHR-MSR Enhanced RAG Example",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Standard mode
  python my_rag_gahr.py document.pdf

  # Enable GAHR-MSR re-ranking
  python my_rag_gahr.py document.pdf --use_gahr

  # GAHR-MSR with GPU acceleration
  python my_rag_gahr.py document.pdf --use_gahr --colbert_device cuda

  # Custom final top-k
  python my_rag_gahr.py document.pdf --use_gahr --final_top_k 10
        """
    )

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
        help="Parser to use (mineru or docling)",
    )

    # GAHR-MSR specific arguments
    parser.add_argument(
        "--use_gahr",
        action="store_true",
        help="Enable GAHR-MSR high-precision re-ranking",
    )
    parser.add_argument(
        "--colbert_device",
        default="cpu",
        choices=["cpu", "cuda", "mps"],
        help="Device for ColBERT model (cpu, cuda, mps)",
    )
    parser.add_argument(
        "--final_top_k",
        type=int,
        default=5,
        help="Number of final results for GAHR-MSR (default: 5)",
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

    # Display configuration
    print("\n" + "="*70)
    print("GAHR-MSR Enhanced RAG Pipeline")
    print("="*70)
    print(f"Mode: {'GAHR-MSR (High-Precision)' if args.use_gahr else 'Standard RAG'}")
    if args.use_gahr:
        print(f"ColBERT Device: {args.colbert_device}")
        print(f"Final Top-K: {args.final_top_k}")
    print("="*70 + "\n")

    # Process with RAG
    asyncio.run(
        process_with_rag(
            args.file_path,
            args.output,
            args.api_key,
            args.base_url,
            args.working_dir,
            args.parser,
            use_gahr=args.use_gahr,
            colbert_device=args.colbert_device,
            final_top_k=args.final_top_k,
        )
    )


if __name__ == "__main__":
    # Configure logging first
    configure_logging()

    main()
