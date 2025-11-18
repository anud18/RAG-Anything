#!/usr/bin/env python
"""
GAHR-MSR Framework Example

This example demonstrates the Graph-Augmented Hybrid Retrieval and
Multi-Stage Re-ranking (GAHR-MSR) framework for high-fidelity RAG.

The framework implements three key phases:
1. Graph-Aware Chunking and Indexing (handled by RAGAnything + LightRAG)
2. High-Recall Hybrid Candidate Retrieval (dense + sparse + RRF)
3. High-Precision Cascaded Re-ranking (ColBERT)

Based on the paper:
"Graph-Augmented Hybrid Retrieval and Multi-Stage Re-ranking for RAG Systems"

Usage:
    python examples/gahr_msr_example.py <document_path> [options]

Examples:
    # Basic usage
    python examples/gahr_msr_example.py document.pdf

    # With custom working directory
    python examples/gahr_msr_example.py document.pdf --working_dir ./my_rag_storage

    # With GPU acceleration for ColBERT
    python examples/gahr_msr_example.py document.pdf --colbert_device cuda

    # Disable re-ranking (use only hybrid retrieval)
    python examples/gahr_msr_example.py document.pdf --no_reranking
"""

import os
import argparse
import asyncio
import logging
import logging.config
from pathlib import Path
import sys

# Add project root directory to Python path
sys.path.append(str(Path(__file__).parent.parent))

from lightrag.llm.openai import openai_complete_if_cache, openai_embed
from lightrag.utils import EmbeddingFunc, logger, set_verbose_debug
from raganything import RAGAnything, RAGAnythingConfig

# Import GAHR-MSR framework
from raganything.reranking import GAHRMSRQuery, GAHRMSRConfig

from dotenv import load_dotenv

load_dotenv(dotenv_path=".env", override=False)


def configure_logging():
    """Configure logging for the application"""
    log_dir = os.getenv("LOG_DIR", os.getcwd())
    log_file_path = os.path.abspath(os.path.join(log_dir, "gahr_msr_example.log"))

    print(f"\nGAHR-MSR example log file: {log_file_path}\n")
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


async def process_with_gahr_msr(
    file_path: str,
    output_dir: str,
    api_key: str,
    base_url: str = None,
    working_dir: str = None,
    parser: str = None,
    colbert_device: str = "cpu",
    enable_reranking: bool = True,
    enable_graph_filter: bool = True,
    final_top_k: int = 5,
):
    """
    Process document and query with GAHR-MSR framework

    Args:
        file_path: Path to the document
        output_dir: Output directory for RAG results
        api_key: OpenAI API key
        base_url: Optional base URL for API
        working_dir: Working directory for RAG storage
        parser: Parser selection (mineru or docling)
        colbert_device: Device for ColBERT (cpu, cuda, mps)
        enable_reranking: Enable ColBERT re-ranking
        enable_graph_filter: Enable graph-aware filtering
        final_top_k: Number of final results to return
    """
    try:
        # Create RAGAnything configuration
        rag_config = RAGAnythingConfig(
            working_dir=working_dir or "./rag_storage",
            parser=parser or "mineru",
            parse_method="auto",
            enable_image_processing=True,
            enable_table_processing=True,
            enable_equation_processing=True,
        )

        # Define LLM model function
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

        # Define vision model function
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

        # Define embedding function
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

        # Initialize RAGAnything
        logger.info("Initializing RAGAnything...")
        rag = RAGAnything(
            config=rag_config,
            llm_model_func=llm_model_func,
            vision_model_func=vision_model_func,
            embedding_func=embedding_func,
        )

        # Process document
        logger.info(f"Processing document: {file_path}")
        await rag.process_document_complete(
            file_path=file_path, output_dir=output_dir, parse_method="auto"
        )

        logger.info("\n" + "="*70)
        logger.info("GAHR-MSR Framework Initialization")
        logger.info("="*70)

        # Create GAHR-MSR configuration
        gahr_config = GAHRMSRConfig(
            # Hybrid retrieval settings
            dense_top_k=100,
            sparse_top_k=100,
            hybrid_top_k=100,
            rrf_k=60,
            # Re-ranking settings
            enable_colbert_reranking=enable_reranking,
            colbert_top_k=20,
            final_top_k=final_top_k,
            colbert_model="colbert-ir/colbertv2.0",
            colbert_device=colbert_device,
            # Graph filtering settings
            enable_graph_filtering=enable_graph_filter,
            graph_filter_mode="any",
        )

        # Initialize GAHR-MSR query interface
        logger.info("Initializing GAHR-MSR query interface...")
        gahr_query = GAHRMSRQuery(rag, gahr_config)

        # Display pipeline info
        pipeline_info = gahr_query.get_pipeline_info()
        logger.info("\nPipeline Configuration:")
        logger.info(f"  Framework: {pipeline_info['framework']} v{pipeline_info['version']}")
        logger.info(f"  Graph Filtering: {pipeline_info['phases']['2_graph_filtering']['enabled']}")
        logger.info(f"  ColBERT Re-ranking: {pipeline_info['phases']['3_colbert_reranking']['enabled']}")
        logger.info(f"  Final Top-K: {pipeline_info['phases']['4_final_selection']['top_k']}")

        # Example queries
        logger.info("\n" + "="*70)
        logger.info("Running Example Queries")
        logger.info("="*70)

        queries = [
            "What is the main content of the document?",
            "What are the key topics discussed?",
            "Summarize the methodology or approach described.",
        ]

        for i, query in enumerate(queries, 1):
            logger.info(f"\n{'='*70}")
            logger.info(f"Query {i}: {query}")
            logger.info('='*70)

            # Query with GAHR-MSR
            result = await gahr_query.query(
                query=query,
                mode="hybrid",  # Use hybrid mode for best results
                use_graph_filter=enable_graph_filter,
                use_reranking=enable_reranking,
            )

            logger.info(f"\nResult:\n{result}\n")

        # Example: Get context only (without LLM generation)
        logger.info("\n" + "="*70)
        logger.info("Example: Retrieve Context Only (No LLM Generation)")
        logger.info("="*70)

        context_result = await gahr_query.query(
            query="What are the main findings?",
            mode="hybrid",
            return_context_only=True
        )

        logger.info(f"\nRetrieved {context_result['num_chunks']} chunks")
        logger.info(f"Context preview:\n{context_result['context'][:500]}...\n")

        # Comparison: Standard RAGAnything query vs GAHR-MSR
        logger.info("\n" + "="*70)
        logger.info("Comparison: Standard RAG vs GAHR-MSR")
        logger.info("="*70)

        comparison_query = "What is the significance of this work?"

        logger.info(f"\nQuery: {comparison_query}\n")

        # Standard RAGAnything query
        logger.info("--- Standard RAGAnything Query ---")
        standard_result = await rag.aquery(comparison_query, mode="hybrid")
        logger.info(f"Result: {standard_result}\n")

        # GAHR-MSR query
        logger.info("--- GAHR-MSR Query (with re-ranking) ---")
        gahr_result = await gahr_query.query(comparison_query, mode="hybrid")
        logger.info(f"Result: {gahr_result}\n")

        logger.info("="*70)
        logger.info("GAHR-MSR Example Completed Successfully")
        logger.info("="*70)

    except Exception as e:
        logger.error(f"Error in GAHR-MSR processing: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())


def main():
    """Main function to run the example"""
    parser = argparse.ArgumentParser(
        description="GAHR-MSR Framework Example",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python examples/gahr_msr_example.py document.pdf

  # With GPU acceleration
  python examples/gahr_msr_example.py document.pdf --colbert_device cuda

  # Disable re-ranking
  python examples/gahr_msr_example.py document.pdf --no_reranking

  # Custom number of final results
  python examples/gahr_msr_example.py document.pdf --final_top_k 10
        """
    )

    parser.add_argument("file_path", help="Path to the document to process")
    parser.add_argument(
        "--working_dir", "-w", default="./rag_storage", help="Working directory path"
    )
    parser.add_argument(
        "--output", "-o", default="./output", help="Output directory path"
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
        help="Parser selection: mineru or docling",
    )
    parser.add_argument(
        "--colbert_device",
        default="cpu",
        choices=["cpu", "cuda", "mps"],
        help="Device for ColBERT model (cpu, cuda, mps)",
    )
    parser.add_argument(
        "--no_reranking",
        action="store_true",
        help="Disable ColBERT re-ranking",
    )
    parser.add_argument(
        "--no_graph_filter",
        action="store_true",
        help="Disable graph-aware filtering",
    )
    parser.add_argument(
        "--final_top_k",
        type=int,
        default=5,
        help="Number of final results to return (default: 5)",
    )

    args = parser.parse_args()

    # Check if API key is provided
    if not args.api_key:
        logger.error("Error: OpenAI API key is required")
        logger.error("Set LLM_BINDING_API_KEY environment variable or use --api-key option")
        return

    # Create output directory if specified
    if args.output:
        os.makedirs(args.output, exist_ok=True)

    # Process with GAHR-MSR
    asyncio.run(
        process_with_gahr_msr(
            args.file_path,
            args.output,
            args.api_key,
            args.base_url,
            args.working_dir,
            args.parser,
            colbert_device=args.colbert_device,
            enable_reranking=not args.no_reranking,
            enable_graph_filter=not args.no_graph_filter,
            final_top_k=args.final_top_k,
        )
    )


if __name__ == "__main__":
    # Configure logging first
    configure_logging()

    print("\n" + "="*70)
    print("GAHR-MSR Framework Example")
    print("Graph-Augmented Hybrid Retrieval and Multi-Stage Re-ranking")
    print("="*70)
    print("This example demonstrates the GAHR-MSR framework for high-fidelity RAG")
    print("with multi-stage retrieval and ColBERT re-ranking.")
    print("="*70 + "\n")

    main()
