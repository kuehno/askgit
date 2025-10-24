"""Ingestion service for efficient document processing and storage."""

import asyncio
from datetime import datetime
from typing import Any

import pytz
from loguru import logger
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, create_async_engine
from sqlmodel import delete, select

from askgit.core.config import settings
from askgit.core.utils import sanitize_text_for_postgres
from askgit.models import Github
from askgit.services.embeddings import EmbeddingService


class IngestionService:
    """Service for ingesting documents with embeddings into the knowledge base."""

    def __init__(
        self,
        engine: AsyncEngine | None = None,
        embedding_service: EmbeddingService | None = None,
    ) -> None:
        """Initialize the ingestion service.

        Args:
            engine: Async SQLAlchemy engine. If not provided, creates one.
            embedding_service: Embedding service instance. If not provided,
                creates one.
        """
        self.engine = engine or self._create_async_engine()
        self.embedding_service = embedding_service or EmbeddingService()
        self.tz = pytz.timezone("Europe/Berlin")
        logger.info("Initialized IngestionService")

    def _create_async_engine(self) -> AsyncEngine:
        """Create an async database engine."""
        database_url = str(settings.SQLALCHEMY_DATABASE_URI).replace(
            "postgresql+psycopg://", "postgresql+psycopg://"
        )
        if "+psycopg://" in database_url:
            database_url = database_url.replace("+psycopg://", "+psycopg_async://")

        return create_async_engine(
            database_url,
            echo=False,
            pool_size=20,
            max_overflow=40,
            pool_pre_ping=True,
        )

    async def check_repo_exists(self, repo_url: str) -> bool:
        """Check if a repository has already been ingested.

        Args:
            repo_url: URL of the repository to check.

        Returns:
            True if the repository exists in the database.
        """
        async with AsyncSession(self.engine) as session:
            statement = select(Github).where(Github.repo_url == repo_url).limit(1)
            result = await session.execute(statement)
            exists = result.first() is not None
            logger.info(f"Repository {repo_url} exists: {exists}")
            return exists

    async def delete_repo_data(self, repo_url: str) -> int:
        """Delete all data for a specific repository.

        Args:
            repo_url: URL of the repository to delete.

        Returns:
            Number of records deleted.
        """
        async with AsyncSession(self.engine) as session:
            statement = delete(Github).where(Github.repo_url == repo_url)
            result = await session.execute(statement)
            await session.commit()
            deleted_count = result.rowcount
            logger.info(f"Deleted {deleted_count} records for repository {repo_url}")
            return deleted_count

    async def ingest_documents(
        self,
        documents: list[dict[str, Any]],
        batch_size: int | None = None,
        repo_url: str | None = None,
        clear_existing: bool = False,
        concurrency_limit: int = 10,
    ) -> list[int]:
        """Ingest multiple documents with embeddings into the database.

        Processes documents in batches for optimal performance using async
        operations and parallel embedding generation.

        Args:
            documents: List of document dictionaries. Each dict should have:
                - content: str - The text content to embed
                - meta: dict - Optional metadata
                - id: int - Optional document ID
            batch_size: Size of batches for processing. Defaults to
                settings.EMBEDDING_BATCH_SIZE.
            repo_url: Optional repository URL to associate with all documents.
            clear_existing: If True and repo_url is provided, delete existing
                data for this repository before ingesting new data.
            concurrency_limit: Maximum number of batches to process
                concurrently. Defaults to 10.

        Returns:
            List of document IDs that were successfully ingested.

        Raises:
            ValueError: If documents list is empty or malformed.
            RuntimeError: If ingestion fails.
        """
        if not documents:
            raise ValueError("Cannot ingest empty documents list")

        # Check and clear existing repo data if requested
        if repo_url and clear_existing:
            exists = await self.check_repo_exists(repo_url)
            if exists:
                deleted = await self.delete_repo_data(repo_url)
                logger.info(
                    f"Cleared {deleted} existing records for repository {repo_url}"
                )

        batch_size = batch_size or settings.EMBEDDING_BATCH_SIZE
        logger.info(
            f"Starting ingestion of {len(documents)} documents "
            f"in batches of {batch_size}"
        )

        # Process documents in batches
        batches = [
            documents[i : i + batch_size] for i in range(0, len(documents), batch_size)
        ]

        # Create a semaphore to limit concurrent batch processing
        semaphore = asyncio.Semaphore(concurrency_limit)

        async def process_batch_with_semaphore(
            batch_idx: int, batch: list[dict[str, Any]]
        ) -> list[int]:
            """Process a batch with semaphore for concurrency control."""
            async with semaphore:
                try:
                    return await self._process_batch(
                        batch, batch_idx + 1, len(batches), repo_url
                    )
                except Exception as e:
                    logger.error(
                        f"Failed to process batch {batch_idx + 1}/{len(batches)}: {e}"
                    )
                    raise RuntimeError(f"Batch {batch_idx + 1} ingestion failed") from e

        # Process all batches concurrently with limited parallelism
        logger.info(
            f"Processing {len(batches)} batches with "
            f"concurrency limit of {concurrency_limit}"
        )
        results = await asyncio.gather(
            *[
                process_batch_with_semaphore(idx, batch)
                for idx, batch in enumerate(batches)
            ]
        )

        # Flatten results
        all_ids = [doc_id for batch_ids in results for doc_id in batch_ids]

        logger.info(
            f"Successfully ingested {len(all_ids)} documents out of {len(documents)}"
        )
        return all_ids

    async def _process_batch(
        self,
        batch: list[dict[str, Any]],
        batch_num: int,
        total_batches: int,
        repo_url: str | None = None,
    ) -> list[int]:
        """Process a single batch of documents.

        Args:
            batch: List of document dictionaries.
            batch_num: Current batch number (1-indexed).
            total_batches: Total number of batches.
            repo_url: Optional repository URL to associate with documents.

        Returns:
            List of document IDs from this batch.
        """
        logger.info(f"Processing batch {batch_num}/{total_batches}")

        # Extract content for embedding generation
        contents = [doc.get("content", "") for doc in batch]
        logger.info(f"Processing {len(contents)} content files")

        # Validate all contents
        for i, content in enumerate(contents):
            if not content or not isinstance(content, str):
                raise ValueError(f"Document at index {i} in batch has invalid content")

        # Generate embeddings in parallel batch
        embeddings = await self.embedding_service.generate_embeddings_batch(contents)

        # Create Github records
        now = datetime.now(self.tz)
        github_records = []

        for doc, embedding in zip(batch, embeddings, strict=False):
            # Sanitize content to remove NUL bytes that PostgreSQL cannot handle
            content = sanitize_text_for_postgres(doc.get("content"))

            record = Github(
                id=doc.get("id"),
                repo_url=repo_url,
                content=content,
                embedding=embedding,
                meta=doc.get("meta"),
                created_at=now,
                updated_at=now,
            )
            github_records.append(record)

        # Store in database
        async with AsyncSession(self.engine) as session:
            session.add_all(github_records)
            await session.commit()

            # Refresh to get IDs if they were auto-generated
            for record in github_records:
                await session.refresh(record)

        ids = [record.id for record in github_records]
        logger.info(
            f"Batch {batch_num}/{total_batches} complete: stored {len(ids)} records"
        )
        return ids

    async def ingest_document(
        self,
        content: str,
        meta: dict[str, Any] | None = None,
        doc_id: int | None = None,
    ) -> int:
        """Ingest a single document.

        Args:
            content: Text content to embed and store.
            meta: Optional metadata dictionary.
            doc_id: Optional document ID.

        Returns:
            The document ID.
        """
        document = {"content": content, "meta": meta}
        if doc_id is not None:
            document["id"] = doc_id

        ids = await self.ingest_documents([document])
        return ids[0]

    async def search_similar(
        self,
        query: str,
    ) -> list[Github]:
        """Search for similar documents using vector similarity.

        Args:
            query: Query text to search for.

        Returns:
            List of similar Github documents.
        """
        # Generate query embedding
        query_embedding = await self.embedding_service.generate_embedding(query)

        # Search database
        async with AsyncSession(self.engine) as session:
            # Use pgvector's cosine distance operator
            statement = (
                select(Github)
                .order_by(Github.embedding.cosine_distance(query_embedding))
                .limit(2)
            )

            result = await session.execute(statement)
            documents = result.scalars().all()

            logger.info(f"Found {len(documents)} similar documents for query")
            return list(documents)

    async def close(self) -> None:
        """Close the database engine and cleanup resources."""
        await self.engine.dispose()
        logger.info("IngestionService closed")
