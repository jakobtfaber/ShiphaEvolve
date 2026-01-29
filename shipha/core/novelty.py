"""
Novelty judge for filtering redundant programs.

Ensures population diversity by rejecting programs that are too
similar to existing archive members. Uses both semantic (embedding)
and syntactic (edit distance) similarity measures.
"""

from __future__ import annotations

import difflib
import hashlib
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

import numpy as np

if TYPE_CHECKING:
    from shipha.database import Program
    from shipha.llm.embedding import EmbeddingClient

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class NoveltyConfig:
    """Configuration for novelty filtering.

    Attributes:
        enabled: Enable novelty filtering.
        similarity_threshold: Max similarity to pass filter (0-1).
        use_semantic: Use embedding-based similarity.
        use_syntactic: Use edit-distance similarity.
        syntactic_weight: Weight for syntactic similarity (0-1).
        cache_embeddings: Cache program embeddings.
    """

    enabled: bool = True
    similarity_threshold: float = 0.85
    use_semantic: bool = True
    use_syntactic: bool = True
    syntactic_weight: float = 0.3
    cache_embeddings: bool = True


# =============================================================================
# Similarity Functions
# =============================================================================


def syntactic_similarity(code1: str, code2: str) -> float:
    """Compute syntactic similarity between two code strings.

    Uses difflib SequenceMatcher ratio for edit-distance-based
    similarity.

    Args:
        code1: First code string.
        code2: Second code string.

    Returns:
        Similarity ratio (0-1).
    """
    if not code1 or not code2:
        return 0.0
    return difflib.SequenceMatcher(None, code1, code2).ratio()


def normalize_code(code: str) -> str:
    """Normalize code for comparison.

    Removes comments, empty lines, and normalizes whitespace.

    Args:
        code: Source code.

    Returns:
        Normalized code string.
    """
    lines = []
    for line in code.split("\n"):
        # Remove inline comments (simple version)
        line = line.split("#")[0] if "#" in line else line
        line = line.rstrip()
        if line:
            lines.append(line)
    return "\n".join(lines)


def code_hash(code: str) -> str:
    """Create hash of normalized code.

    Args:
        code: Source code.

    Returns:
        MD5 hash string.
    """
    normalized = normalize_code(code)
    return hashlib.md5(normalized.encode()).hexdigest()


# =============================================================================
# Novelty Judge
# =============================================================================


class NoveltyJudge:
    """Novelty filter for program population.

    Maintains an embedding index of archive programs and filters
    new candidates that are too similar to existing members.
    """

    def __init__(
        self,
        config: NoveltyConfig | None = None,
        embedding_client: EmbeddingClient | None = None,
    ) -> None:
        """Initialize novelty judge.

        Args:
            config: Novelty configuration.
            embedding_client: Client for semantic embeddings.
        """
        self.config = config or NoveltyConfig()
        self._embedding_client = embedding_client

        # Embedding cache: program_id -> embedding
        self._embeddings: dict[str, np.ndarray] = {}

        # Code hash set for exact duplicate detection
        self._code_hashes: set[str] = set()

        # Archive reference
        self._archive_codes: dict[str, str] = {}

        # Statistics
        self._checked_count = 0
        self._rejected_count = 0
        self._exact_duplicate_count = 0

    # -------------------------------------------------------------------------
    # Archive Management
    # -------------------------------------------------------------------------

    async def add_to_archive(self, program: Program) -> None:
        """Add a program to the novelty archive.

        Args:
            program: Program to add.
        """
        # Store code
        self._archive_codes[program.id] = program.code

        # Store hash for exact duplicate detection
        self._code_hashes.add(code_hash(program.code))

        # Compute and cache embedding if semantic enabled
        if (
            self.config.use_semantic
            and self._embedding_client
            and program.id not in self._embeddings
        ):
            try:
                embedding = await self._embedding_client.embed(program.code)
                self._embeddings[program.id] = embedding
            except Exception as e:
                logger.warning(f"Failed to embed program {program.id}: {e}")

    async def add_batch_to_archive(self, programs: list[Program]) -> None:
        """Add multiple programs to archive.

        Args:
            programs: List of programs.
        """
        for prog in programs:
            await self.add_to_archive(prog)

    def remove_from_archive(self, program_id: str) -> None:
        """Remove a program from the archive.

        Args:
            program_id: ID of program to remove.
        """
        if program_id in self._archive_codes:
            code = self._archive_codes.pop(program_id)
            h = code_hash(code)
            self._code_hashes.discard(h)

        self._embeddings.pop(program_id, None)

    def clear_archive(self) -> None:
        """Clear the novelty archive."""
        self._archive_codes.clear()
        self._code_hashes.clear()
        self._embeddings.clear()

    # -------------------------------------------------------------------------
    # Novelty Checking
    # -------------------------------------------------------------------------

    async def is_novel(self, program: Program) -> bool:
        """Check if a program is sufficiently novel.

        Args:
            program: Candidate program to check.

        Returns:
            True if novel (passes filter), False if too similar.
        """
        if not self.config.enabled:
            return True

        self._checked_count += 1

        # Check for exact duplicate first (fast path)
        if self._is_exact_duplicate(program.code):
            self._rejected_count += 1
            self._exact_duplicate_count += 1
            logger.debug(f"Program {program.id} rejected: exact duplicate")
            return False

        # Check similarity against archive
        max_similarity = await self._max_archive_similarity(program)

        if max_similarity > self.config.similarity_threshold:
            self._rejected_count += 1
            logger.debug(
                f"Program {program.id} rejected: similarity {max_similarity:.3f} "
                f"> threshold {self.config.similarity_threshold}"
            )
            return False

        return True

    def _is_exact_duplicate(self, code: str) -> bool:
        """Check if code is an exact duplicate.

        Args:
            code: Source code to check.

        Returns:
            True if exact duplicate exists.
        """
        return code_hash(code) in self._code_hashes

    async def _max_archive_similarity(self, program: Program) -> float:
        """Find maximum similarity to any archive member.

        Args:
            program: Candidate program.

        Returns:
            Maximum similarity score (0-1).
        """
        if not self._archive_codes:
            return 0.0

        max_sim = 0.0

        # Compute candidate embedding if using semantic
        candidate_embedding = None
        if self.config.use_semantic and self._embedding_client:
            try:
                candidate_embedding = await self._embedding_client.embed(program.code)
            except Exception as e:
                logger.warning(f"Failed to embed candidate: {e}")

        for prog_id, archive_code in self._archive_codes.items():
            sim = await self._compute_similarity(
                program.code,
                archive_code,
                candidate_embedding,
                self._embeddings.get(prog_id),
            )
            max_sim = max(max_sim, sim)

            # Early exit if already above threshold
            if max_sim > self.config.similarity_threshold:
                break

        return max_sim

    async def _compute_similarity(
        self,
        code1: str,
        code2: str,
        emb1: np.ndarray | None = None,
        emb2: np.ndarray | None = None,
    ) -> float:
        """Compute combined similarity between two programs.

        Args:
            code1: First code string.
            code2: Second code string.
            emb1: Pre-computed embedding for code1.
            emb2: Pre-computed embedding for code2.

        Returns:
            Combined similarity score (0-1).
        """
        similarities: list[tuple[float, float]] = []  # (similarity, weight)

        # Syntactic similarity
        if self.config.use_syntactic:
            syn_sim = syntactic_similarity(
                normalize_code(code1), normalize_code(code2)
            )
            similarities.append((syn_sim, self.config.syntactic_weight))

        # Semantic similarity
        if self.config.use_semantic and emb1 is not None and emb2 is not None:
            sem_sim = self._cosine_similarity(emb1, emb2)
            sem_weight = 1.0 - self.config.syntactic_weight
            similarities.append((sem_sim, sem_weight))

        if not similarities:
            return 0.0

        # Weighted average
        total_weight = sum(w for _, w in similarities)
        if total_weight == 0:
            return 0.0

        return sum(s * w for s, w in similarities) / total_weight

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
    # Batch Operations
    # -------------------------------------------------------------------------

    async def filter_novel(
        self,
        programs: list[Program],
    ) -> list[Program]:
        """Filter a list of programs for novelty.

        Args:
            programs: Candidate programs.

        Returns:
            List of novel programs that passed the filter.
        """
        novel_programs = []
        for prog in programs:
            if await self.is_novel(prog):
                novel_programs.append(prog)
        return novel_programs

    # -------------------------------------------------------------------------
    # Statistics
    # -------------------------------------------------------------------------

    @property
    def archive_size(self) -> int:
        """Number of programs in archive."""
        return len(self._archive_codes)

    @property
    def rejection_rate(self) -> float:
        """Rate of program rejection."""
        if self._checked_count == 0:
            return 0.0
        return self._rejected_count / self._checked_count

    def stats(self) -> dict[str, float]:
        """Get novelty judge statistics.

        Returns:
            Statistics dictionary.
        """
        return {
            "archive_size": self.archive_size,
            "checked_count": self._checked_count,
            "rejected_count": self._rejected_count,
            "exact_duplicate_count": self._exact_duplicate_count,
            "rejection_rate": self.rejection_rate,
            "embeddings_cached": len(self._embeddings),
        }

    def reset_stats(self) -> None:
        """Reset statistics counters."""
        self._checked_count = 0
        self._rejected_count = 0
        self._exact_duplicate_count = 0
