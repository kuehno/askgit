#!/usr/bin/env python3
"""CLI tool for scraping GitHub repositories and indexing them."""

import argparse
import asyncio
import sys

from loguru import logger

from askgit.services.github_scraper import (
    MAX_DEFAULT_BYTES,
    GithubScraper,
)
from askgit.services.ingestion import IngestionService


async def scrape_and_index(
    repo_url: str,
    max_file_size: int,
    batch_size: int,
    chunk_lines: int,
    chunk_lines_overlap: int,
) -> int:
    """Scrape a GitHub repository and index its contents.

    Args:
        repo_url: URL of the GitHub repository to scrape.
        max_file_size: Maximum file size in bytes to include.
        batch_size: Batch size for document ingestion.
        chunk_lines: Maximum number of lines per chunk for code files.
        chunk_lines_overlap: Number of lines to overlap between chunks.
        max_chars: Maximum characters per chunk for code files.

    Returns:
        Number of documents indexed.

    Raises:
        RuntimeError: If scraping or indexing fails.
    """
    # Initialize services
    scraper = GithubScraper(
        max_file_size=max_file_size,
        chunk_lines=chunk_lines,
        chunk_lines_overlap=chunk_lines_overlap,
    )
    ingestion_service = IngestionService()

    try:
        # Scrape the repository
        logger.info(f"Starting scrape of repository: {repo_url}")
        documents, metadata = scraper.scrape_repository(repo_url)

        if not documents:
            logger.warning("No documents to index from repository")
            return 0

        # Log metadata
        logger.info(f"Repository metadata: {metadata}")

        # Ingest documents
        logger.info(
            f"Starting ingestion of {len(documents)} documents "
            f"in batches of {batch_size}"
        )
        document_ids = await ingestion_service.ingest_documents(
            documents, batch_size=batch_size
        )

        logger.info(
            f"Successfully indexed {len(document_ids)} documents from {repo_url}"
        )

        return len(document_ids)

    except Exception as e:
        logger.error(f"Failed to scrape and index repository: {e}")
        raise

    finally:
        await ingestion_service.close()


def main() -> int:
    """Run the GitHub scraper CLI."""
    parser = argparse.ArgumentParser(
        description="Scrape a GitHub repository and index its contents",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Scrape a repository with default settings
  %(prog)s https://github.com/owner/repo

  # Scrape with custom max file size (100KB)
  %(prog)s https://github.com/owner/repo --max-bytes 102400

  # Scrape with custom chunking parameters
  %(prog)s https://github.com/owner/repo --chunk-lines 50 --max-chars 2000
        """,
    )

    parser.add_argument(
        "repo_url",
        help="GitHub repository URL (e.g., https://github.com/owner/repo)",
    )

    parser.add_argument(
        "--max-bytes",
        type=int,
        default=MAX_DEFAULT_BYTES,
        help=(f"Maximum file size in bytes to include (default: {MAX_DEFAULT_BYTES})"),
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for document ingestion (default: 16)",
    )

    parser.add_argument(
        "--chunk-lines",
        type=int,
        default=40,
        help="Maximum number of lines per chunk for code files (default: 40)",
    )

    parser.add_argument(
        "--chunk-lines-overlap",
        type=int,
        default=15,
        help="Number of lines to overlap between chunks (default: 15)",
    )

    parser.add_argument(
        "--max-chars",
        type=int,
        default=1500,
        help="Maximum characters per chunk (default: 1500)",
    )

    args = parser.parse_args()

    # Validate repository URL
    if not args.repo_url.startswith(("https://github.com/", "git@github.com:")):
        logger.error(
            "Invalid repository URL. Must start with "
            "'https://github.com/' or 'git@github.com:'"
        )
        return 1

    try:
        # Run the scraper
        num_indexed = asyncio.run(
            scrape_and_index(
                args.repo_url,
                args.max_bytes,
                args.batch_size,
                args.chunk_lines,
                args.chunk_lines_overlap,
            )
        )

        if num_indexed > 0:
            logger.success(
                f"✓ Successfully scraped and indexed {num_indexed} documents"
            )
            return 0
        else:
            logger.warning("No documents were indexed")
            return 1

    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        return 130

    except Exception as e:
        logger.error(f"Error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
