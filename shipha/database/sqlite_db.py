"""
Async SQLite Program Database for ShiphaEvolve

Provides persistent storage for evolved programs with async support.

Usage:
    from shipha.database import ProgramDatabase, Program

    async with ProgramDatabase("evolution.db") as db:
        program = Program(id="prog_001", code="def f(x): return x*2")
        await db.add(program)
        retrieved = await db.get("prog_001")

Features:
    - Async SQLite via aiosqlite
    - Program storage with metrics and metadata
    - Archive management for elite programs
    - Parent selection support
"""

from __future__ import annotations

import asyncio
import json
import logging
import sqlite3
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import aiosqlite

logger = logging.getLogger(__name__)


def generate_id() -> str:
    """Generate unique program ID."""
    return f"prog_{uuid.uuid4().hex[:12]}"


@dataclass
class DatabaseConfig:
    """Configuration for the program database."""

    db_path: str = "evolution_db.sqlite"
    """Path to SQLite database file. Use :memory: for in-memory DB."""

    archive_size: int = 100
    """Maximum number of programs to keep in the elite archive."""

    num_islands: int = 4
    """Number of islands for island-model evolution."""


@dataclass
class Program:
    """
    Represents a program in the evolutionary database.

    Core fields:
        id: Unique program identifier
        code: The program source code
        combined_score: Overall fitness score

    Evolution fields:
        parent_id: ID of parent program (if evolved)
        generation: Generation number in evolution
        code_diff: Diff from parent code

    Metrics fields:
        public_metrics: Scores visible to the LLM
        private_metrics: Hidden evaluation scores
        correct: Whether the program passes all tests
    """

    # Identification
    id: str = field(default_factory=generate_id)
    """Unique program identifier."""

    code: str = ""
    """The program source code."""

    language: str = "python"
    """Programming language of the code."""

    # Evolution lineage
    parent_id: str | None = None
    """ID of parent program."""

    inspiration_ids: list[str] = field(default_factory=list)
    """IDs of programs used as inspiration context."""

    generation: int = 0
    """Generation number in evolution."""

    timestamp: float = field(default_factory=time.time)
    """Unix timestamp when program was created."""

    code_diff: str | None = None
    """Diff showing changes from parent code."""

    # Performance metrics
    combined_score: float = 0.0
    """Overall fitness score (higher is better)."""

    public_metrics: dict[str, Any] = field(default_factory=dict)
    """Metrics visible to the LLM for feedback."""

    private_metrics: dict[str, Any] = field(default_factory=dict)
    """Hidden metrics for evaluation only."""

    text_feedback: str = ""
    """Human-readable feedback about the program."""

    correct: bool = False
    """Whether the program is functionally correct."""

    # Derived features
    complexity: float = 0.0
    """Code complexity score."""

    children_count: int = 0
    """Number of child programs evolved from this one."""

    # Evolution metadata
    island_idx: int | None = None
    """Island index for island-model evolution."""

    metadata: dict[str, Any] = field(default_factory=dict)
    """Additional metadata storage."""

    in_archive: bool = False
    """Whether this program is in the elite archive."""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Program:
        """Create Program from dictionary."""
        # Filter to known fields
        known_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in known_fields}

        # Ensure dict types
        for field_name in ["public_metrics", "private_metrics", "metadata"]:
            if field_name in filtered and not isinstance(filtered[field_name], dict):
                filtered[field_name] = {}

        # Ensure list types
        for field_name in ["inspiration_ids"]:
            if field_name in filtered and not isinstance(filtered[field_name], list):
                filtered[field_name] = []

        return cls(**filtered)


class ProgramDatabase:
    """
    Async SQLite-backed database for evolved programs.

    Supports:
        - CRUD operations for programs
        - Elite archive management
        - Parent selection queries
        - Generation tracking

    Usage:
        async with ProgramDatabase("evolution.db") as db:
            await db.add(program)
            best = await db.get_best()

    Or for manual lifecycle:
        db = ProgramDatabase("evolution.db")
        await db.connect()
        try:
            await db.add(program)
        finally:
            await db.close()
    """

    def __init__(self, config: DatabaseConfig | str = "evolution_db.sqlite") -> None:
        """
        Initialize the database.

        Args:
            config: DatabaseConfig or path string.
        """
        if isinstance(config, str):
            config = DatabaseConfig(db_path=config)

        self.config = config
        self.db_path = config.db_path
        self._conn: aiosqlite.Connection | None = None
        self._lock = asyncio.Lock()

        # Cached metadata
        self.last_iteration: int = 0
        self.best_program_id: str | None = None

    async def __aenter__(self) -> ProgramDatabase:
        """Async context manager entry."""
        await self.connect()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async context manager exit."""
        await self.close()

    async def connect(self) -> None:
        """Connect to the database and create tables."""
        if self._conn is not None:
            return

        # Ensure directory exists
        if self.db_path != ":memory:":
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)

        self._conn = await aiosqlite.connect(self.db_path, timeout=30.0)
        self._conn.row_factory = aiosqlite.Row

        await self._create_tables()
        await self._load_metadata()

        count = await self.count()
        logger.info(f"Database connected: {self.db_path} ({count} programs)")

    async def close(self) -> None:
        """Close the database connection."""
        if self._conn is not None:
            await self._conn.close()
            self._conn = None
            logger.info("Database connection closed")

    async def _create_tables(self) -> None:
        """Create database tables if they don't exist."""
        if self._conn is None:
            raise ConnectionError("Database not connected")

        # Configure SQLite for performance
        await self._conn.execute("PRAGMA journal_mode = WAL;")
        await self._conn.execute("PRAGMA synchronous = NORMAL;")
        await self._conn.execute("PRAGMA cache_size = -64000;")  # 64MB
        await self._conn.execute("PRAGMA temp_store = MEMORY;")

        # Programs table
        await self._conn.execute("""
            CREATE TABLE IF NOT EXISTS programs (
                id TEXT PRIMARY KEY,
                code TEXT NOT NULL,
                language TEXT NOT NULL DEFAULT 'python',
                parent_id TEXT,
                inspiration_ids TEXT,
                generation INTEGER NOT NULL DEFAULT 0,
                timestamp REAL NOT NULL,
                code_diff TEXT,
                combined_score REAL DEFAULT 0.0,
                public_metrics TEXT,
                private_metrics TEXT,
                text_feedback TEXT DEFAULT '',
                correct INTEGER DEFAULT 0,
                complexity REAL DEFAULT 0.0,
                children_count INTEGER DEFAULT 0,
                island_idx INTEGER,
                metadata TEXT
            )
        """)

        # Indices
        await self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_generation ON programs(generation)"
        )
        await self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_score ON programs(combined_score DESC)"
        )
        await self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_parent ON programs(parent_id)"
        )
        await self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_correct ON programs(correct)"
        )

        # Archive table
        await self._conn.execute("""
            CREATE TABLE IF NOT EXISTS archive (
                program_id TEXT PRIMARY KEY,
                FOREIGN KEY (program_id) REFERENCES programs(id) ON DELETE CASCADE
            )
        """)

        # Metadata table
        await self._conn.execute("""
            CREATE TABLE IF NOT EXISTS metadata (
                key TEXT PRIMARY KEY,
                value TEXT
            )
        """)

        await self._conn.commit()
        logger.debug("Database tables created")

    async def _load_metadata(self) -> None:
        """Load cached metadata from database."""
        if self._conn is None:
            return

        # Load last iteration
        cursor = await self._conn.execute(
            "SELECT value FROM metadata WHERE key = 'last_iteration'"
        )
        row = await cursor.fetchone()
        self.last_iteration = int(row["value"]) if row else 0

        # Load best program ID
        cursor = await self._conn.execute(
            "SELECT value FROM metadata WHERE key = 'best_program_id'"
        )
        row = await cursor.fetchone()
        self.best_program_id = row["value"] if row and row["value"] != "None" else None

    async def _save_metadata(self, key: str, value: str | None) -> None:
        """Save metadata to database."""
        if self._conn is None:
            return

        await self._conn.execute(
            "INSERT OR REPLACE INTO metadata (key, value) VALUES (?, ?)",
            (key, value),
        )
        await self._conn.commit()

    async def count(self) -> int:
        """Get total number of programs."""
        if self._conn is None:
            return 0

        cursor = await self._conn.execute("SELECT COUNT(*) as cnt FROM programs")
        row = await cursor.fetchone()
        return row["cnt"] if row else 0

    async def add(self, program: Program) -> str:
        """
        Add a program to the database.

        Args:
            program: Program to add.

        Returns:
            The program ID.
        """
        if self._conn is None:
            raise ConnectionError("Database not connected")

        async with self._lock:
            # Serialize JSON fields
            inspiration_ids_json = json.dumps(program.inspiration_ids)
            public_metrics_json = json.dumps(program.public_metrics)
            private_metrics_json = json.dumps(program.private_metrics)
            metadata_json = json.dumps(program.metadata)

            await self._conn.execute(
                """
                INSERT INTO programs (
                    id, code, language, parent_id, inspiration_ids,
                    generation, timestamp, code_diff, combined_score,
                    public_metrics, private_metrics, text_feedback,
                    correct, complexity, children_count, island_idx, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    program.id,
                    program.code,
                    program.language,
                    program.parent_id,
                    inspiration_ids_json,
                    program.generation,
                    program.timestamp,
                    program.code_diff,
                    program.combined_score,
                    public_metrics_json,
                    private_metrics_json,
                    program.text_feedback,
                    1 if program.correct else 0,
                    program.complexity,
                    program.children_count,
                    program.island_idx,
                    metadata_json,
                ),
            )

            # Increment parent's children count
            if program.parent_id:
                await self._conn.execute(
                    "UPDATE programs SET children_count = children_count + 1 WHERE id = ?",
                    (program.parent_id,),
                )

            await self._conn.commit()

            # Update archive
            await self._update_archive(program)

            # Update best program tracking
            await self._update_best_program(program)

            # Update generation
            if program.generation > self.last_iteration:
                self.last_iteration = program.generation
                await self._save_metadata("last_iteration", str(self.last_iteration))

            logger.debug(f"Added program {program.id} (score: {program.combined_score})")
            return program.id

    async def get(self, program_id: str) -> Program | None:
        """
        Get a program by ID.

        Args:
            program_id: The program ID to retrieve.

        Returns:
            The Program or None if not found.
        """
        if self._conn is None:
            return None

        cursor = await self._conn.execute(
            "SELECT * FROM programs WHERE id = ?", (program_id,)
        )
        row = await cursor.fetchone()

        if row is None:
            return None

        return self._row_to_program(row)

    async def get_best(self) -> Program | None:
        """Get the program with the highest score."""
        if self._conn is None:
            return None

        cursor = await self._conn.execute(
            "SELECT * FROM programs ORDER BY combined_score DESC LIMIT 1"
        )
        row = await cursor.fetchone()
        return self._row_to_program(row) if row else None

    async def get_by_generation(self, generation: int) -> list[Program]:
        """Get all programs from a specific generation."""
        if self._conn is None:
            return []

        cursor = await self._conn.execute(
            "SELECT * FROM programs WHERE generation = ?", (generation,)
        )
        rows = await cursor.fetchall()
        return [self._row_to_program(row) for row in rows]

    async def get_archive(self) -> list[Program]:
        """Get all programs in the elite archive."""
        if self._conn is None:
            return []

        cursor = await self._conn.execute("""
            SELECT p.* FROM programs p
            JOIN archive a ON p.id = a.program_id
            ORDER BY p.combined_score DESC
        """)
        rows = await cursor.fetchall()
        return [self._row_to_program(row) for row in rows]

    async def get_correct_programs(self, limit: int = 100) -> list[Program]:
        """Get programs that are marked as correct."""
        if self._conn is None:
            return []

        cursor = await self._conn.execute(
            "SELECT * FROM programs WHERE correct = 1 ORDER BY combined_score DESC LIMIT ?",
            (limit,),
        )
        rows = await cursor.fetchall()
        return [self._row_to_program(row) for row in rows]

    async def sample_parent(self) -> Program | None:
        """
        Sample a program to use as parent for evolution.

        Uses power-law selection favoring higher-scoring programs.
        """
        if self._conn is None:
            return None

        # Get archive programs sorted by score
        archive = await self.get_archive()
        if not archive:
            # Fallback to best overall
            return await self.get_best()

        # Power-law selection (higher scores more likely)
        n = len(archive)
        # Weights: [1, 1/2, 1/3, ..., 1/n]
        weights = [1.0 / (i + 1) for i in range(n)]
        total = sum(weights)
        weights = [w / total for w in weights]

        import random
        idx = random.choices(range(n), weights=weights, k=1)[0]
        return archive[idx]

    async def sample_inspirations(self, num: int = 3) -> list[Program]:
        """
        Sample programs to use as inspiration context.

        Args:
            num: Number of inspiration programs to sample.
        """
        if self._conn is None:
            return []

        archive = await self.get_archive()
        if len(archive) <= num:
            return archive

        import random
        return random.sample(archive, num)

    async def update(self, program: Program) -> None:
        """Update an existing program."""
        if self._conn is None:
            raise ConnectionError("Database not connected")

        async with self._lock:
            # Serialize JSON fields
            inspiration_ids_json = json.dumps(program.inspiration_ids)
            public_metrics_json = json.dumps(program.public_metrics)
            private_metrics_json = json.dumps(program.private_metrics)
            metadata_json = json.dumps(program.metadata)

            await self._conn.execute(
                """
                UPDATE programs SET
                    code = ?, language = ?, combined_score = ?,
                    public_metrics = ?, private_metrics = ?,
                    text_feedback = ?, correct = ?, complexity = ?,
                    inspiration_ids = ?, metadata = ?
                WHERE id = ?
                """,
                (
                    program.code,
                    program.language,
                    program.combined_score,
                    public_metrics_json,
                    private_metrics_json,
                    program.text_feedback,
                    1 if program.correct else 0,
                    program.complexity,
                    inspiration_ids_json,
                    metadata_json,
                    program.id,
                ),
            )
            await self._conn.commit()

            # Re-evaluate archive membership
            await self._update_archive(program)
            await self._update_best_program(program)

    async def _update_archive(self, program: Program) -> None:
        """Update archive membership after adding/updating a program."""
        if self._conn is None:
            return

        # Get current archive size
        cursor = await self._conn.execute("SELECT COUNT(*) as cnt FROM archive")
        row = await cursor.fetchone()
        archive_size = row["cnt"] if row else 0

        # Check if program qualifies for archive
        if program.correct:
            # Get lowest score in archive
            cursor = await self._conn.execute("""
                SELECT MIN(p.combined_score) as min_score
                FROM programs p JOIN archive a ON p.id = a.program_id
            """)
            row = await cursor.fetchone()
            min_score = row["min_score"] if row and row["min_score"] is not None else float("-inf")

            if archive_size < self.config.archive_size or program.combined_score > min_score:
                # Add to archive
                await self._conn.execute(
                    "INSERT OR IGNORE INTO archive (program_id) VALUES (?)",
                    (program.id,),
                )

                # Remove lowest if over capacity
                if archive_size >= self.config.archive_size:
                    cursor = await self._conn.execute("""
                        SELECT a.program_id
                        FROM archive a JOIN programs p ON a.program_id = p.id
                        ORDER BY p.combined_score ASC LIMIT 1
                    """)
                    row = await cursor.fetchone()
                    if row and row["program_id"] != program.id:
                        await self._conn.execute(
                            "DELETE FROM archive WHERE program_id = ?",
                            (row["program_id"],),
                        )

                await self._conn.commit()

    async def _update_best_program(self, program: Program) -> None:
        """Update best program tracking."""
        if self._conn is None:
            return

        if self.best_program_id is None:
            self.best_program_id = program.id
            await self._save_metadata("best_program_id", program.id)
        else:
            # Check if new program is better
            best = await self.get(self.best_program_id)
            if best is None or program.combined_score > best.combined_score:
                self.best_program_id = program.id
                await self._save_metadata("best_program_id", program.id)

    def _row_to_program(self, row: aiosqlite.Row) -> Program:
        """Convert a database row to a Program object."""
        data = dict(row)

        # Deserialize JSON fields
        for field_name in ["inspiration_ids"]:
            if data.get(field_name):
                try:
                    data[field_name] = json.loads(data[field_name])
                except json.JSONDecodeError:
                    data[field_name] = []

        for field_name in ["public_metrics", "private_metrics", "metadata"]:
            if data.get(field_name):
                try:
                    data[field_name] = json.loads(data[field_name])
                except json.JSONDecodeError:
                    data[field_name] = {}

        # Convert boolean
        data["correct"] = bool(data.get("correct", 0))

        return Program.from_dict(data)

    async def get_lineage(self, program_id: str) -> list[Program]:
        """
        Get the full lineage (ancestors) of a program.

        Returns list from oldest ancestor to the given program.
        """
        if self._conn is None:
            return []

        lineage: list[Program] = []
        current_id: str | None = program_id

        while current_id:
            program = await self.get(current_id)
            if program is None:
                break
            lineage.append(program)
            current_id = program.parent_id

        lineage.reverse()
        return lineage

    async def get_children(self, program_id: str) -> list[Program]:
        """Get all direct children of a program."""
        if self._conn is None:
            return []

        cursor = await self._conn.execute(
            "SELECT * FROM programs WHERE parent_id = ?", (program_id,)
        )
        rows = await cursor.fetchall()
        return [self._row_to_program(row) for row in rows]


# Sync wrapper for non-async contexts
class SyncProgramDatabase:
    """Synchronous wrapper around ProgramDatabase."""

    def __init__(self, config: DatabaseConfig | str = "evolution_db.sqlite") -> None:
        self._async_db = ProgramDatabase(config)

    def connect(self) -> None:
        asyncio.run(self._async_db.connect())

    def close(self) -> None:
        asyncio.run(self._async_db.close())

    def add(self, program: Program) -> str:
        return asyncio.run(self._async_db.add(program))

    def get(self, program_id: str) -> Program | None:
        return asyncio.run(self._async_db.get(program_id))

    def get_best(self) -> Program | None:
        return asyncio.run(self._async_db.get_best())

    def get_archive(self) -> list[Program]:
        return asyncio.run(self._async_db.get_archive())

    def sample_parent(self) -> Program | None:
        return asyncio.run(self._async_db.sample_parent())

    def sample_inspirations(self, num: int = 3) -> list[Program]:
        return asyncio.run(self._async_db.sample_inspirations(num))

    def count(self) -> int:
        return asyncio.run(self._async_db.count())

    @property
    def last_iteration(self) -> int:
        return self._async_db.last_iteration

    @property
    def best_program_id(self) -> str | None:
        return self._async_db.best_program_id
