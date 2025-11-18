#!/usr/bin/env python
"""
Adaptive Multi-Mode RAG Query Example

This example demonstrates the new adaptive query features in RAGAnything:
1. Adaptive Query Routing - automatically selects the best retrieval mode based on query complexity
2. Multi-Mode Query - tries multiple modes and uses LLM to select the best result

Based on Adaptive-RAG approach: queries are classified by complexity and routed to appropriate strategies.
"""

import os
import argparse
import asyncio
import logging
import logging.config
from pathlib import Path

# Add project root directory to Python path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from lightrag.llm.openai import openai_complete_if_cache, openai_embed
from lightrag.utils import EmbeddingFunc, logger, set_verbose_debug
from raganything import RAGAnything, RAGAnythingConfig

from dotenv import load_dotenv

load_dotenv(dotenv_path=".env", override=False)


def configure_logging():
    """Configure logging for the application"""
    log_dir = os.getenv("LOG_DIR", os.getcwd())
    log_file_path = os.path.abspath(os.path.join(log_dir, "my_rag_example.log"))

    print(f"\nMy RAG example log file: {log_file_path}\n")
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


async def demo_adaptive_queries(
    file_path: str,
    output_dir: str,
    api_key: str,
    base_url: str = None,
    working_dir: str = None,
    parser: str = None,
):
    """
    Demonstrate adaptive and multi-mode query features

    Args:
        file_path: Path to the document
        output_dir: Output directory for RAG results
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
        rag = RAGAnything(
            config=config,
            llm_model_func=llm_model_func,
            vision_model_func=vision_model_func,
            embedding_func=embedding_func,
        )

        # Process document
        logger.info(f"Processing document: {file_path}")
        await rag.process_document_complete(
            file_path=file_path, output_dir=output_dir, parse_method="auto"
        )

        print("\n" + "=" * 80)
        print("ADAPTIVE QUERY ROUTING DEMONSTRATION")
        print("=" * 80)

        # Test queries with different complexity levels
        test_queries = [
            {
                "query": "What is the title of the document?",
                "expected_complexity": "simple",
                "description": "Simple factual question",
            },
            {
                "query": "How does the system work and what are its main components?",
                "expected_complexity": "moderate",
                "description": "Moderate question requiring some synthesis",
            },
            {
                "query": "Analyze the overall methodology and discuss its implications for the field, comparing different approaches mentioned.",
                "expected_complexity": "complex",
                "description": "Complex analytical question",
            },
        ]

        print("\n--- 1. ADAPTIVE QUERY ROUTING ---")
        print(
            "The system analyzes each query and automatically selects the best retrieval mode.\n"
        )

        for i, test in enumerate(test_queries, 1):
            print(f"\n[Query {i}] {test['description']}")
            print(f"Question: {test['query']}")
            print(f"Expected complexity: {test['expected_complexity']}")

            # Use adaptive query with analysis
            response = await rag.aquery_adaptive(
                test["query"], return_analysis=True
            )

            print(f"\nRouting Decision:")
            print(f"  - Detected complexity: {response['analysis']['complexity']}")
            print(f"  - Query type: {response['analysis']['query_type']}")
            print(f"  - Selected mode: {response['analysis']['recommended_mode']}")
            print(f"  - Confidence: {response['analysis']['confidence']:.2f}")
            print(f"  - Reasoning: {response['analysis']['reasoning']}")
            print(f"\nAnswer preview: {response['result'][:200]}...")
            print("-" * 80)

        print("\n--- 2. MULTI-MODE QUERY WITH LLM EVALUATION ---")
        print(
            "The system tries multiple retrieval modes and uses LLM to select the best result.\n"
        )

        # Example: Try multiple modes for a moderate complexity question
        example_query = "What are the key findings and conclusions of this document?"

        print(f"Query: {example_query}\n")
        print("Trying modes: naive, local, hybrid, global, mix\n")

        # Get results from all modes with comparison
        multi_mode_response = await rag.aquery_multi_mode(
            example_query,
            modes=["naive", "local", "hybrid", "global", "mix"],
            return_all_results=True,
        )

        print(f"Best Mode Selected: {multi_mode_response['best_mode']}")
        print(f"Evaluation: {multi_mode_response['evaluation']}\n")

        print("Results from all modes:")
        for mode, data in multi_mode_response["results"].items():
            result_preview = data["result"][:150].replace("\n", " ")
            print(f"\n  [{mode.upper()}]")
            print(f"  Preview: {result_preview}...")

        print(f"\n\nFinal Answer (using {multi_mode_response['best_mode']} mode):")
        print(multi_mode_response["best_result"])

        print("\n" + "=" * 80)
        print("COMPARISON: Adaptive vs Multi-Mode")
        print("=" * 80)
        print(
            """
Adaptive Query Routing:
  ✓ Fast - only one retrieval call
  ✓ Efficient - uses LLM only for classification
  ✓ Good for production with many queries
  ✓ Recommended for most use cases

Multi-Mode Query:
  ✓ Comprehensive - tries all modes
  ✓ Best quality - LLM selects best result
  ✓ Good for critical queries
  ✓ Higher cost (multiple retrievals + evaluation)
  ✓ Recommended when quality > speed/cost
        """
        )

        print("\n--- 3. PRACTICAL USAGE EXAMPLES ---\n")

        # Show simple usage without analysis details
        print("Simple usage (just get the answer):")
        simple_result = await rag.aquery_adaptive(
            "What is the main topic of the document?"
        )
        print(f"Answer: {simple_result[:200]}...\n")

        print("=" * 80)
        print("\nDemo completed successfully!")

    except Exception as e:
        logger.error(f"Error in demo: {str(e)}")
        import traceback

        logger.error(traceback.format_exc())


def main():
    """Main function to run the example"""
    parser = argparse.ArgumentParser(
        description="Adaptive Multi-Mode RAG Query Example"
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
        help="Parser to use (mineru or docling)",
    )

    args = parser.parse_args()

    # Check if API key is provided
    if not args.api_key:
        logger.error("Error: OpenAI API key is required")
        logger.error("Set api key environment variable or use --api-key option")
        return

    # Create output directory if specified
    if args.output:
        os.makedirs(args.output, exist_ok=True)

    # Run demo
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
    # Configure logging first
    configure_logging()

    print("\n" + "=" * 80)
    print("ADAPTIVE MULTI-MODE RAG QUERY DEMONSTRATION")
    print("=" * 80)
    print("\nThis demo showcases two advanced query features:")
    print("1. Adaptive Query Routing - Auto-selects best mode based on complexity")
    print("2. Multi-Mode Query - Tries multiple modes and LLM picks the best")
    print("\nBased on Adaptive-RAG research paper approach")
    print("=" * 80 + "\n")

    main()
