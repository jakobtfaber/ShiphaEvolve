"""
Tests for async SQLite program database.

Tests cover:
    - Database connection and table creation
    - Program CRUD operations
    - Archive management
    - Parent selection and sampling
    - Lineage tracking
"""

from __future__ import annotations

import pytest

from shipha.database import DatabaseConfig, Program, ProgramDatabase, generate_id


class TestProgram:
    """Tests for the Program dataclass."""

    def test_generate_id(self) -> None:
        """Test that generate_id creates unique IDs."""
        ids = [generate_id() for _ in range(100)]
        assert len(set(ids)) == 100  # All unique
        assert all(id_.startswith("prog_") for id_ in ids)

    def test_program_defaults(self) -> None:
        """Test Program default values."""
        prog = Program(code="print('hello')")
        
        assert prog.id.startswith("prog_")
        assert prog.code == "print('hello')"
        assert prog.language == "python"
        assert prog.parent_id is None
        assert prog.generation == 0
        assert prog.combined_score == 0.0
        assert prog.correct is False
        assert prog.public_metrics == {}
        assert prog.private_metrics == {}

    def test_program_to_dict(self, sample_program: Program) -> None:
        """Test Program serialization to dict."""
        data = sample_program.to_dict()
        
        assert isinstance(data, dict)
        assert data["id"] == sample_program.id
        assert data["code"] == sample_program.code
        assert data["combined_score"] == sample_program.combined_score

    def test_program_from_dict(self, sample_program: Program) -> None:
        """Test Program deserialization from dict."""
        data = sample_program.to_dict()
        restored = Program.from_dict(data)
        
        assert restored.id == sample_program.id
        assert restored.code == sample_program.code
        assert restored.combined_score == sample_program.combined_score
        assert restored.correct == sample_program.correct

    def test_program_from_dict_filters_unknown_fields(self) -> None:
        """Test that from_dict ignores unknown fields."""
        data = {
            "id": "test",
            "code": "x",
            "unknown_field": "should be ignored",
            "another_unknown": 123,
        }
        prog = Program.from_dict(data)
        assert prog.id == "test"
        assert not hasattr(prog, "unknown_field")


class TestDatabaseConfig:
    """Tests for DatabaseConfig."""

    def test_default_config(self) -> None:
        """Test default configuration values."""
        config = DatabaseConfig()
        
        assert config.db_path == "evolution_db.sqlite"
        assert config.archive_size == 100
        assert config.num_islands == 4

    def test_custom_config(self) -> None:
        """Test custom configuration."""
        config = DatabaseConfig(
            db_path="/tmp/custom.db",
            archive_size=50,
            num_islands=8,
        )
        
        assert config.db_path == "/tmp/custom.db"
        assert config.archive_size == 50
        assert config.num_islands == 8


class TestProgramDatabaseConnection:
    """Tests for database connection lifecycle."""

    @pytest.mark.asyncio
    async def test_memory_db_connect(self) -> None:
        """Test connecting to in-memory database."""
        db = ProgramDatabase(":memory:")
        await db.connect()
        
        assert db._conn is not None
        
        await db.close()
        assert db._conn is None

    @pytest.mark.asyncio
    async def test_file_db_connect(self, temp_db_path: str) -> None:
        """Test connecting to file-based database."""
        db = ProgramDatabase(temp_db_path)
        await db.connect()
        
        assert db._conn is not None
        count = await db.count()
        assert count == 0
        
        await db.close()

    @pytest.mark.asyncio
    async def test_context_manager(self) -> None:
        """Test async context manager."""
        async with ProgramDatabase(":memory:") as db:
            assert db._conn is not None
            count = await db.count()
            assert count == 0


class TestProgramDatabaseCRUD:
    """Tests for CRUD operations."""

    @pytest.mark.asyncio
    async def test_add_program(
        self, memory_db: ProgramDatabase, sample_program: Program
    ) -> None:
        """Test adding a program."""
        program_id = await memory_db.add(sample_program)
        
        assert program_id == sample_program.id
        count = await memory_db.count()
        assert count == 1

    @pytest.mark.asyncio
    async def test_get_program(
        self, memory_db: ProgramDatabase, sample_program: Program
    ) -> None:
        """Test retrieving a program by ID."""
        await memory_db.add(sample_program)
        
        retrieved = await memory_db.get(sample_program.id)
        
        assert retrieved is not None
        assert retrieved.id == sample_program.id
        assert retrieved.code == sample_program.code
        assert retrieved.combined_score == sample_program.combined_score

    @pytest.mark.asyncio
    async def test_get_nonexistent(self, memory_db: ProgramDatabase) -> None:
        """Test retrieving nonexistent program returns None."""
        result = await memory_db.get("nonexistent_id")
        assert result is None

    @pytest.mark.asyncio
    async def test_get_best(
        self, memory_db: ProgramDatabase, sample_programs: list[Program]
    ) -> None:
        """Test getting the best program."""
        for prog in sample_programs:
            await memory_db.add(prog)
        
        best = await memory_db.get_best()
        
        assert best is not None
        # Last program has highest score (0.5 + 4*0.1 = 0.9)
        assert best.combined_score == max(p.combined_score for p in sample_programs)

    @pytest.mark.asyncio
    async def test_update_program(
        self, memory_db: ProgramDatabase, sample_program: Program
    ) -> None:
        """Test updating a program."""
        await memory_db.add(sample_program)
        
        sample_program.combined_score = 0.99
        sample_program.correct = True
        await memory_db.update(sample_program)
        
        retrieved = await memory_db.get(sample_program.id)
        assert retrieved is not None
        assert retrieved.combined_score == 0.99


class TestProgramDatabaseGeneration:
    """Tests for generation tracking."""

    @pytest.mark.asyncio
    async def test_get_by_generation(
        self, memory_db: ProgramDatabase, sample_programs: list[Program]
    ) -> None:
        """Test retrieving programs by generation."""
        for prog in sample_programs:
            await memory_db.add(prog)
        
        gen_2 = await memory_db.get_by_generation(2)
        assert len(gen_2) == 1
        assert gen_2[0].generation == 2

    @pytest.mark.asyncio
    async def test_last_iteration_tracking(
        self, memory_db: ProgramDatabase
    ) -> None:
        """Test that last_iteration is updated."""
        assert memory_db.last_iteration == 0
        
        prog = Program(id="p1", code="x", generation=5)
        await memory_db.add(prog)
        
        assert memory_db.last_iteration == 5


class TestProgramDatabaseArchive:
    """Tests for elite archive management."""

    @pytest.mark.asyncio
    async def test_correct_programs_added_to_archive(
        self, memory_db: ProgramDatabase
    ) -> None:
        """Test that correct programs are added to archive."""
        prog = Program(
            id="correct_prog",
            code="x",
            correct=True,
            combined_score=0.8,
        )
        await memory_db.add(prog)
        
        archive = await memory_db.get_archive()
        assert len(archive) == 1
        assert archive[0].id == "correct_prog"

    @pytest.mark.asyncio
    async def test_incorrect_programs_not_in_archive(
        self, memory_db: ProgramDatabase
    ) -> None:
        """Test that incorrect programs are not in archive."""
        prog = Program(
            id="wrong_prog",
            code="x",
            correct=False,
            combined_score=0.8,
        )
        await memory_db.add(prog)
        
        archive = await memory_db.get_archive()
        assert len(archive) == 0

    @pytest.mark.asyncio
    async def test_archive_size_limit(self) -> None:
        """Test that archive respects size limit."""
        config = DatabaseConfig(db_path=":memory:", archive_size=3)
        async with ProgramDatabase(config) as db:
            # Add 5 correct programs
            for i in range(5):
                prog = Program(
                    id=f"prog_{i}",
                    code=f"x{i}",
                    correct=True,
                    combined_score=0.1 * i,
                )
                await db.add(prog)
            
            archive = await db.get_archive()
            assert len(archive) == 3
            # Should keep highest scoring
            scores = [p.combined_score for p in archive]
            assert min(scores) >= 0.2  # Lowest in top 3

    @pytest.mark.asyncio
    async def test_get_correct_programs(
        self, memory_db: ProgramDatabase, sample_programs: list[Program]
    ) -> None:
        """Test getting correct programs."""
        for prog in sample_programs:
            await memory_db.add(prog)
        
        correct = await memory_db.get_correct_programs()
        assert all(p.correct for p in correct)


class TestProgramDatabaseParentSelection:
    """Tests for parent and inspiration selection."""

    @pytest.mark.asyncio
    async def test_sample_parent(self, memory_db: ProgramDatabase) -> None:
        """Test sampling a parent program."""
        # Add some correct programs
        for i in range(5):
            prog = Program(
                id=f"prog_{i}",
                code=f"x{i}",
                correct=True,
                combined_score=0.1 * i,
            )
            await memory_db.add(prog)
        
        parent = await memory_db.sample_parent()
        assert parent is not None
        assert parent.correct is True

    @pytest.mark.asyncio
    async def test_sample_parent_empty_db(
        self, memory_db: ProgramDatabase
    ) -> None:
        """Test sampling parent from empty database."""
        parent = await memory_db.sample_parent()
        assert parent is None

    @pytest.mark.asyncio
    async def test_sample_inspirations(
        self, memory_db: ProgramDatabase
    ) -> None:
        """Test sampling inspiration programs."""
        for i in range(10):
            prog = Program(
                id=f"prog_{i}",
                code=f"x{i}",
                correct=True,
                combined_score=0.1 * i,
            )
            await memory_db.add(prog)
        
        inspirations = await memory_db.sample_inspirations(num=3)
        assert len(inspirations) == 3
        assert len(set(p.id for p in inspirations)) == 3  # All unique


class TestProgramDatabaseLineage:
    """Tests for lineage tracking."""

    @pytest.mark.asyncio
    async def test_children_count(self, memory_db: ProgramDatabase) -> None:
        """Test that children count is updated."""
        parent = Program(id="parent", code="p")
        child1 = Program(id="child1", code="c1", parent_id="parent")
        child2 = Program(id="child2", code="c2", parent_id="parent")
        
        await memory_db.add(parent)
        await memory_db.add(child1)
        await memory_db.add(child2)
        
        retrieved_parent = await memory_db.get("parent")
        assert retrieved_parent is not None
        assert retrieved_parent.children_count == 2

    @pytest.mark.asyncio
    async def test_get_lineage(self, memory_db: ProgramDatabase) -> None:
        """Test getting program lineage."""
        grandparent = Program(id="gp", code="gp")
        parent = Program(id="p", code="p", parent_id="gp", generation=1)
        child = Program(id="c", code="c", parent_id="p", generation=2)
        
        await memory_db.add(grandparent)
        await memory_db.add(parent)
        await memory_db.add(child)
        
        lineage = await memory_db.get_lineage("c")
        
        assert len(lineage) == 3
        assert lineage[0].id == "gp"  # Oldest first
        assert lineage[1].id == "p"
        assert lineage[2].id == "c"  # Child last

    @pytest.mark.asyncio
    async def test_get_children(self, memory_db: ProgramDatabase) -> None:
        """Test getting direct children."""
        parent = Program(id="parent", code="p")
        child1 = Program(id="child1", code="c1", parent_id="parent")
        child2 = Program(id="child2", code="c2", parent_id="parent")
        grandchild = Program(id="gc", code="gc", parent_id="child1")
        
        await memory_db.add(parent)
        await memory_db.add(child1)
        await memory_db.add(child2)
        await memory_db.add(grandchild)
        
        children = await memory_db.get_children("parent")
        
        assert len(children) == 2
        child_ids = {c.id for c in children}
        assert child_ids == {"child1", "child2"}
        assert "gc" not in child_ids  # Grandchild excluded


class TestDatabasePersistence:
    """Tests for database persistence."""

    @pytest.mark.asyncio
    async def test_data_persists_across_connections(
        self, temp_db_path: str
    ) -> None:
        """Test that data persists when reconnecting."""
        # First connection: add data
        async with ProgramDatabase(temp_db_path) as db:
            prog = Program(id="persist_test", code="x", correct=True)
            await db.add(prog)
        
        # Second connection: verify data
        async with ProgramDatabase(temp_db_path) as db:
            count = await db.count()
            assert count == 1
            
            retrieved = await db.get("persist_test")
            assert retrieved is not None
            assert retrieved.code == "x"

    @pytest.mark.asyncio
    async def test_metadata_persists(self, temp_db_path: str) -> None:
        """Test that metadata persists."""
        async with ProgramDatabase(temp_db_path) as db:
            prog = Program(
                id="gen_test",
                code="x",
                generation=42,
                correct=True,
            )
            await db.add(prog)
        
        async with ProgramDatabase(temp_db_path) as db:
            assert db.last_iteration == 42
            assert db.best_program_id == "gen_test"
