"""
Async embedding client with fallback support.

Uses OpenAI's text-embedding-3-large by default with automatic
fallback to text-embedding-3-small on failure.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import TypeVar

import numpy as np

try:
    import litellm
except ImportError:
    litellm = None  # type: ignore

logger = logging.getLogger(__name__)

T = TypeVar("T")


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class EmbeddingConfig:
    """Configuration for embedding client.

    Attributes:
        primary_model: Primary embedding model.
        fallback_model: Fallback if primary fails.
        batch_size: Max embeddings per API call.
        max_retries: Retry count on failure.
        cache_size: LRU cache size (0 to disable).
        dimensions: Embedding dimensions (model-dependent).
    """

    primary_model: str = "text-embedding-3-large"
    fallback_model: str = "text-embedding-3-small"
    batch_size: int = 100
    max_retries: int = 3
    cache_size: int = 10000
    dimensions: int = 3072  # text-embedding-3-large default


# =============================================================================
# Embedding Model Metadata
# =============================================================================


EMBEDDING_MODELS = {
    "text-embedding-3-large": {
        "dimensions": 3072,
        "max_input_tokens": 8191,
        "price_per_1k": 0.00013,
    },
    "text-embedding-3-small": {
        "dimensions": 1536,
        "max_input_tokens": 8191,
        "price_per_1k": 0.00002,
    },
    "text-embedding-ada-002": {
        "dimensions": 1536,
        "max_input_tokens": 8191,
        "price_per_1k": 0.0001,
    },
}


def get_embedding_dimensions(model: str) -> int:
    """Get embedding dimensions for a model.

    Args:
        model: Model name.

    Returns:
        Embedding dimensions.
    """
    if model in EMBEDDING_MODELS:
        return EMBEDDING_MODELS[model]["dimensions"]
    # Default for unknown models
    return 1536


# =============================================================================
# LRU Cache
# =============================================================================


class LRUCache:
    """Simple LRU cache for embeddings.

    Uses OrderedDict for O(1) operations with LRU eviction.
    """

    def __init__(self, max_size: int = 10000) -> None:
        """Initialize cache.

        Args:
            max_size: Maximum cache entries.
        """
        self._cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self._max_size = max_size
        self._hits = 0
        self._misses = 0

    def _hash_text(self, text: str) -> str:
        """Create hash key for text."""
        return hashlib.md5(text.encode()).hexdigest()

    def get(self, text: str) -> np.ndarray | None:
        """Get cached embedding.

        Args:
            text: Input text.

        Returns:
            Cached embedding or None.
        """
        key = self._hash_text(text)
        if key in self._cache:
            # Move to end (most recently used)
            self._cache.move_to_end(key)
            self._hits += 1
            return self._cache[key]
        self._misses += 1
        return None

    def set(self, text: str, embedding: np.ndarray) -> None:
        """Cache an embedding.

        Args:
            text: Input text.
            embedding: Embedding vector.
        """
        if self._max_size <= 0:
            return

        key = self._hash_text(text)

        if key in self._cache:
            self._cache.move_to_end(key)
        else:
            if len(self._cache) >= self._max_size:
                # Remove oldest
                self._cache.popitem(last=False)
            self._cache[key] = embedding

    def clear(self) -> None:
        """Clear the cache."""
        self._cache.clear()

    @property
    def size(self) -> int:
        """Current cache size."""
        return len(self._cache)

    @property
    def hit_rate(self) -> float:
        """Cache hit rate."""
        total = self._hits + self._misses
        if total == 0:
            return 0.0
        return self._hits / total


# =============================================================================
# Embedding Client
# =============================================================================


class EmbeddingClient:
    """Async embedding client with caching and fallback.

    Uses litellm for provider-agnostic embedding generation
    with automatic fallback to smaller models on failure.
    """

    def __init__(
        self,
        config: EmbeddingConfig | None = None,
        primary_model: str | None = None,
        fallback_model: str | None = None,
        cache_size: int | None = None,
    ) -> None:
        """Initialize embedding client.

        Args:
            config: Full configuration object.
            primary_model: Override primary model.
            fallback_model: Override fallback model.
            cache_size: Override cache size.
        """
        if litellm is None:
            raise ImportError("litellm is required for EmbeddingClient")

        self.config = config or EmbeddingConfig()

        if primary_model:
            self.config.primary_model = primary_model
        if fallback_model:
            self.config.fallback_model = fallback_model
        if cache_size is not None:
            self.config.cache_size = cache_size

        self._cache = LRUCache(self.config.cache_size)

        # Statistics
        self._total_tokens = 0
        self._total_cost = 0.0
        self._request_count = 0
        self._fallback_count = 0

    # -------------------------------------------------------------------------
    # Single Embedding
    # -------------------------------------------------------------------------

    async def embed(self, text: str) -> np.ndarray:
        """Get embedding for a single text.

        Args:
            text: Input text to embed.

        Returns:
            Embedding vector as numpy array.
        """
        # Check cache
        cached = self._cache.get(text)
        if cached is not None:
            return cached

        # Generate embedding
        embedding = await self._generate_embedding(text)
        self._cache.set(text, embedding)
        return embedding

    async def _generate_embedding(
        self,
        text: str,
        use_fallback: bool = False,
    ) -> np.ndarray:
        """Generate embedding via API.

        Args:
            text: Input text.
            use_fallback: Use fallback model.

        Returns:
            Embedding vector.
        """
        model = (
            self.config.fallback_model if use_fallback else self.config.primary_model
        )

        for attempt in range(self.config.max_retries):
            try:
                response = await litellm.aembedding(
                    model=model,
                    input=[text],
                )

                self._request_count += 1
                if use_fallback:
                    self._fallback_count += 1

                # Extract embedding
                embedding = np.array(response.data[0]["embedding"])

                # Track tokens (if available)
                if hasattr(response, "usage") and response.usage:
                    self._total_tokens += getattr(response.usage, "total_tokens", 0)

                return embedding

            except Exception as e:
                logger.warning(
                    f"Embedding attempt {attempt + 1} failed for {model}: {e}"
                )

                if attempt == self.config.max_retries - 1:
                    # Last attempt - try fallback if not already
                    if not use_fallback and self.config.fallback_model:
                        logger.info(f"Falling back to {self.config.fallback_model}")
                        return await self._generate_embedding(text, use_fallback=True)
                    raise

                await asyncio.sleep(2**attempt)  # Exponential backoff

        raise RuntimeError("Failed to generate embedding")

    # -------------------------------------------------------------------------
    # Batch Embedding
    # -------------------------------------------------------------------------

    async def embed_batch(
        self,
        texts: list[str],
        show_progress: bool = False,
    ) -> list[np.ndarray]:
        """Get embeddings for multiple texts.

        Uses batching for efficiency and caching to avoid
        redundant API calls.

        Args:
            texts: List of texts to embed.
            show_progress: Show progress bar.

        Returns:
            List of embedding vectors.
        """
        results: list[np.ndarray | None] = [None] * len(texts)
        to_embed: list[tuple[int, str]] = []

        # Check cache first
        for i, text in enumerate(texts):
            cached = self._cache.get(text)
            if cached is not None:
                results[i] = cached
            else:
                to_embed.append((i, text))

        if not to_embed:
            return results  # type: ignore

        # Batch embed uncached texts
        batch_size = self.config.batch_size
        for batch_start in range(0, len(to_embed), batch_size):
            batch = to_embed[batch_start : batch_start + batch_size]
            batch_texts = [text for _, text in batch]

            embeddings = await self._generate_batch(batch_texts)

            for (idx, text), emb in zip(batch, embeddings):
                results[idx] = emb
                self._cache.set(text, emb)

        return results  # type: ignore

    async def _generate_batch(
        self,
        texts: list[str],
        use_fallback: bool = False,
    ) -> list[np.ndarray]:
        """Generate embeddings for a batch.

        Args:
            texts: List of texts.
            use_fallback: Use fallback model.

        Returns:
            List of embedding vectors.
        """
        model = (
            self.config.fallback_model if use_fallback else self.config.primary_model
        )

        for attempt in range(self.config.max_retries):
            try:
                response = await litellm.aembedding(
                    model=model,
                    input=texts,
                )

                self._request_count += 1
                if use_fallback:
                    self._fallback_count += 1

                embeddings = [
                    np.array(item["embedding"]) for item in response.data
                ]

                if hasattr(response, "usage") and response.usage:
                    self._total_tokens += getattr(response.usage, "total_tokens", 0)

                return embeddings

            except Exception as e:
                logger.warning(
                    f"Batch embedding attempt {attempt + 1} failed: {e}"
                )

                if attempt == self.config.max_retries - 1:
                    if not use_fallback and self.config.fallback_model:
                        logger.info(f"Falling back to {self.config.fallback_model}")
                        return await self._generate_batch(texts, use_fallback=True)
                    raise

                await asyncio.sleep(2**attempt)

        raise RuntimeError("Failed to generate batch embeddings")

    # -------------------------------------------------------------------------
    # Similarity
    # -------------------------------------------------------------------------

    async def similarity(self, text1: str, text2: str) -> float:
        """Compute cosine similarity between two texts.

        Args:
            text1: First text.
            text2: Second text.

        Returns:
            Cosine similarity (0-1).
        """
        emb1, emb2 = await asyncio.gather(
            self.embed(text1),
            self.embed(text2),
        )
        return self._cosine_similarity(emb1, emb2)

    def _cosine_similarity(
        self,
        emb1: np.ndarray,
        emb2: np.ndarray,
    ) -> float:
        """Compute cosine similarity between embeddings.

        Args:
            emb1: First embedding.
            emb2: Second embedding.

        Returns:
            Cosine similarity.
        """
        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return float(np.dot(emb1, emb2) / (norm1 * norm2))

    # -------------------------------------------------------------------------
    # Statistics
    # -------------------------------------------------------------------------

    @property
    def cache_hit_rate(self) -> float:
        """Cache hit rate."""
        return self._cache.hit_rate

    @property
    def request_count(self) -> int:
        """Total API requests made."""
        return self._request_count

    @property
    def fallback_rate(self) -> float:
        """Rate of fallback model usage."""
        if self._request_count == 0:
            return 0.0
        return self._fallback_count / self._request_count

    def stats(self) -> dict[str, float]:
        """Get client statistics.

        Returns:
            Statistics dictionary.
        """
        return {
            "request_count": self._request_count,
            "total_tokens": self._total_tokens,
            "cache_size": self._cache.size,
            "cache_hit_rate": self.cache_hit_rate,
            "fallback_rate": self.fallback_rate,
        }

    def clear_cache(self) -> None:
        """Clear the embedding cache."""
        self._cache.clear()
