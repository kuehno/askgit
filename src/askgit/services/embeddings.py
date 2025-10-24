"""Async embedding generation service using LiteLLM."""

from litellm import aembedding
from loguru import logger

from askgit.core.config import settings


class EmbeddingService:
    """Service for generating embeddings using LiteLLM with async batch processing."""

    def __init__(
        self,
        model_name: str | None = None,
        batch_size: int | None = None,
    ) -> None:
        """Initialize the embedding service.

        Args:
            model_name: Name of the embedding model to use.
                Defaults to settings.EMBEDDING_MODEL.
            batch_size: Maximum batch size for processing.
                Defaults to settings.EMBEDDING_BATCH_SIZE.
        """
        self.model_name = model_name or settings.EMBEDDING_MODEL
        self.batch_size = batch_size or settings.EMBEDDING_BATCH_SIZE
        logger.info(
            f"Initialized EmbeddingService with model={self.model_name}, "
            f"batch_size={self.batch_size}"
        )

    async def generate_embeddings_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts in batch.

        Args:
            texts: List of text strings to generate embeddings for.

        Returns:
            List of embedding vectors, one per input text.

        Raises:
            ValueError: If texts list is empty or contains invalid entries.
            RuntimeError: If embedding generation fails.
        """
        if not texts:
            raise ValueError("Cannot generate embeddings for empty text list")

        filtered_texts = []
        text_indices = []

        for i, text in enumerate(texts):
            if not text or not isinstance(text, str):
                raise ValueError(f"Invalid text at index {i}: must be non-empty string")

            stripped_text = text.strip()
            if not stripped_text:
                raise ValueError(f"Text at index {i} is empty or whitespace-only")

            filtered_texts.append(stripped_text)
            text_indices.append(i)

        try:
            response = await aembedding(
                self.model_name,
                input=filtered_texts,
                api_key=settings.API_KEY,
                api_base=settings.API_BASE,
            )

            embeddings = [item["embedding"] for item in response.data]
            logger.debug(
                f"Successfully generated {len(embeddings)} embeddings in batch"
            )
            return embeddings

        except Exception as e:
            logger.error(f"Batch embedding failed: {e}, falling back to sequential")
            embeddings = []
            for i, text in enumerate(filtered_texts):
                try:
                    response = await aembedding(
                        self.model_name,
                        input=[text],
                        api_key=settings.API_KEY,
                        api_base=settings.API_BASE,
                    )
                    embeddings.append(response.data[0]["embedding"])
                except Exception as inner_e:
                    logger.error(f"Individual embedding failed for text {i}: {inner_e}")
                    raise RuntimeError(
                        f"Failed to generate embedding for text at index {i}"
                    ) from inner_e

            return embeddings

    async def generate_embedding(self, text: str) -> list[float]:
        """Generate embedding for a single text.

        Args:
            text: Text string to generate embedding for.

        Returns:
            Embedding vector.

        Raises:
            ValueError: If text is empty or invalid.
            RuntimeError: If embedding generation fails.
        """
        embeddings = await self.generate_embeddings_batch([text])
        return embeddings[0]
