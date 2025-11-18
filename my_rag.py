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
            logger.info(f"\n[Text Query]: {query}")

            # Query with similarity scores
            try:
                # result = await rag.aquery(query, mode="hybrid")
                query_param = QueryParam(mode="naive", only_need_prompt=True)  
                raw_prompt = await rag.lightrag.aquery(query, query_param)
                with open(os.path.join(output_dir, "raw_prompt.txt"), "a", encoding="utf-8") as f:
                    f.write(f"\n\n[Query]: {query}\n")
                    f.write(raw_prompt)

                answer = await rag.aquery(query, mode="mix")
                logger.info(f"Answer: {answer}")
            except Exception as e:
                logger.warning(f"Error query: {str(e)}")
                # Fallback to regular query
                # result = await rag.aquery(query, mode="hybrid")
                # logger.info(f"Answer: {result}")
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
