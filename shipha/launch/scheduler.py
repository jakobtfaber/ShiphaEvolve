"""
Job scheduler for managing concurrent LLM queries and evaluations.

Provides:
- Async job queue with priority support
- Concurrent execution with configurable limits
- Progress tracking and cancellation
- Rate limiting and backoff
"""

from __future__ import annotations

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from collections.abc import Callable, Coroutine
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Generic, TypeVar

logger = logging.getLogger(__name__)


# =============================================================================
# Type Variables and Enums
# =============================================================================

T = TypeVar("T")  # Job result type


class JobStatus(str, Enum):
    """Status of a scheduled job."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class JobPriority(int, Enum):
    """Priority levels for jobs."""

    LOW = 0
    NORMAL = 1
    HIGH = 2
    CRITICAL = 3


# =============================================================================
# Configuration Dataclasses
# =============================================================================


@dataclass
class LocalJobConfig:
    """Configuration for local job execution.

    Attributes:
        max_concurrent_llm: Maximum concurrent LLM queries.
        max_concurrent_eval: Maximum concurrent evaluations.
        timeout_seconds: Default job timeout.
        retry_count: Number of retries on failure.
        retry_delay: Delay between retries in seconds.
    """

    max_concurrent_llm: int = 10
    max_concurrent_eval: int = 4
    timeout_seconds: float = 120.0
    retry_count: int = 3
    retry_delay: float = 1.0


@dataclass
class SlurmJobConfig:
    """Configuration for SLURM cluster execution.

    Attributes:
        partition: SLURM partition name.
        nodes: Number of nodes to request.
        gpus_per_node: GPUs per node.
        time_limit: Job time limit (HH:MM:SS).
        memory_gb: Memory per node in GB.
        account: SLURM account name.
    """

    partition: str = "gpu"
    nodes: int = 1
    gpus_per_node: int = 1
    time_limit: str = "01:00:00"
    memory_gb: int = 32
    account: str | None = None


# =============================================================================
# Job Representation
# =============================================================================


@dataclass
class Job(Generic[T]):
    """A scheduled job with metadata.

    Attributes:
        id: Unique job identifier.
        coro: The coroutine to execute.
        priority: Job priority level.
        status: Current job status.
        result: Job result when completed.
        error: Error message if failed.
        created_at: Creation timestamp.
        started_at: Execution start timestamp.
        completed_at: Completion timestamp.
        metadata: Additional job metadata.
    """

    id: str
    coro: Coroutine[Any, Any, T]
    priority: JobPriority = JobPriority.NORMAL
    status: JobStatus = JobStatus.PENDING
    result: T | None = None
    error: str | None = None
    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    completed_at: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __lt__(self, other: Job[Any]) -> bool:
        """Compare by priority (higher priority first)."""
        if self.priority != other.priority:
            return self.priority.value > other.priority.value
        return self.created_at < other.created_at

    @property
    def elapsed_ms(self) -> float | None:
        """Get elapsed execution time in milliseconds."""
        if self.started_at is None:
            return None
        end = self.completed_at or time.time()
        return (end - self.started_at) * 1000


# =============================================================================
# Job Scheduler
# =============================================================================


class JobScheduler:
    """Async job scheduler for concurrent task execution.

    Manages a priority queue of jobs with configurable concurrency
    limits for different job types (LLM queries, evaluations).
    """

    def __init__(
        self,
        config: LocalJobConfig | None = None,
        name: str = "default",
    ) -> None:
        """Initialize the scheduler.

        Args:
            config: Scheduler configuration.
            name: Scheduler instance name for logging.
        """
        self.config = config or LocalJobConfig()
        self.name = name

        # Job tracking
        self._jobs: dict[str, Job[Any]] = {}
        self._job_counter = 0

        # Priority queue (using list + heapq)
        self._pending: list[Job[Any]] = []

        # Semaphores for concurrency control
        self._llm_sem = asyncio.Semaphore(self.config.max_concurrent_llm)
        self._eval_sem = asyncio.Semaphore(self.config.max_concurrent_eval)

        # State
        self._running = False
        self._worker_task: asyncio.Task[None] | None = None

        # Statistics
        self._completed_count = 0
        self._failed_count = 0
        self._total_time_ms = 0.0

    # -------------------------------------------------------------------------
    # Job Submission
    # -------------------------------------------------------------------------

    def submit(
        self,
        coro: Coroutine[Any, Any, T],
        priority: JobPriority = JobPriority.NORMAL,
        job_type: str = "llm",
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Submit a job for execution.

        Args:
            coro: Async coroutine to execute.
            priority: Job priority.
            job_type: Type of job ("llm" or "eval").
            metadata: Additional metadata.

        Returns:
            Job ID for tracking.
        """
        import heapq

        self._job_counter += 1
        job_id = f"{self.name}_{self._job_counter:06d}"

        job = Job(
            id=job_id,
            coro=coro,
            priority=priority,
            metadata=metadata or {},
        )
        job.metadata["job_type"] = job_type

        self._jobs[job_id] = job
        heapq.heappush(self._pending, job)

        logger.debug(f"Submitted job {job_id} with priority {priority.name}")
        return job_id

    def submit_batch(
        self,
        coros: list[Coroutine[Any, Any, T]],
        priority: JobPriority = JobPriority.NORMAL,
        job_type: str = "llm",
    ) -> list[str]:
        """Submit multiple jobs.

        Args:
            coros: List of coroutines.
            priority: Priority for all jobs.
            job_type: Type of jobs.

        Returns:
            List of job IDs.
        """
        return [self.submit(c, priority, job_type) for c in coros]

    # -------------------------------------------------------------------------
    # Job Execution
    # -------------------------------------------------------------------------

    async def run_one(self, job: Job[T]) -> T:
        """Execute a single job with appropriate semaphore.

        Args:
            job: The job to execute.

        Returns:
            Job result.
        """
        job_type = job.metadata.get("job_type", "llm")
        sem = self._llm_sem if job_type == "llm" else self._eval_sem

        async with sem:
            job.status = JobStatus.RUNNING
            job.started_at = time.time()

            try:
                result = await asyncio.wait_for(
                    job.coro,
                    timeout=self.config.timeout_seconds,
                )
                job.result = result
                job.status = JobStatus.COMPLETED
                self._completed_count += 1
                return result

            except asyncio.TimeoutError:
                job.error = "Timeout"
                job.status = JobStatus.FAILED
                self._failed_count += 1
                raise

            except Exception as e:
                job.error = str(e)
                job.status = JobStatus.FAILED
                self._failed_count += 1
                raise

            finally:
                job.completed_at = time.time()
                if job.elapsed_ms:
                    self._total_time_ms += job.elapsed_ms

    async def run_all(self) -> list[Any]:
        """Execute all pending jobs concurrently.

        Returns:
            List of results (or exceptions).
        """
        if not self._pending:
            return []

        # Drain the queue
        jobs = []
        while self._pending:
            jobs.append(self._pending.pop(0))

        # Execute concurrently
        tasks = [self.run_one(job) for job in jobs]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        return list(results)

    async def run_until_complete(
        self,
        callback: Callable[[Job[Any]], None] | None = None,
    ) -> list[Any]:
        """Run jobs as they're submitted until queue is empty.

        Args:
            callback: Optional callback for each completed job.

        Returns:
            List of all results.
        """
        results: list[Any] = []

        while self._pending or any(
            j.status == JobStatus.RUNNING for j in self._jobs.values()
        ):
            if self._pending:
                job = self._pending.pop(0)
                try:
                    result = await self.run_one(job)
                    results.append(result)
                except Exception as e:
                    results.append(e)

                if callback:
                    callback(job)
            else:
                await asyncio.sleep(0.01)

        return results

    # -------------------------------------------------------------------------
    # Job Management
    # -------------------------------------------------------------------------

    def get_job(self, job_id: str) -> Job[Any] | None:
        """Get a job by ID."""
        return self._jobs.get(job_id)

    def cancel(self, job_id: str) -> bool:
        """Cancel a pending job.

        Args:
            job_id: Job ID to cancel.

        Returns:
            True if cancelled, False if not found or already running.
        """
        job = self._jobs.get(job_id)
        if not job or job.status != JobStatus.PENDING:
            return False

        job.status = JobStatus.CANCELLED
        self._pending = [j for j in self._pending if j.id != job_id]
        return True

    def cancel_all(self) -> int:
        """Cancel all pending jobs.

        Returns:
            Number of cancelled jobs.
        """
        count = 0
        for job in list(self._pending):
            job.status = JobStatus.CANCELLED
            count += 1
        self._pending.clear()
        return count

    # -------------------------------------------------------------------------
    # Statistics
    # -------------------------------------------------------------------------

    @property
    def pending_count(self) -> int:
        """Number of pending jobs."""
        return len(self._pending)

    @property
    def completed_count(self) -> int:
        """Number of completed jobs."""
        return self._completed_count

    @property
    def failed_count(self) -> int:
        """Number of failed jobs."""
        return self._failed_count

    @property
    def average_time_ms(self) -> float:
        """Average job execution time in milliseconds."""
        total = self._completed_count + self._failed_count
        if total == 0:
            return 0.0
        return self._total_time_ms / total

    def stats(self) -> dict[str, Any]:
        """Get scheduler statistics.

        Returns:
            Dictionary of statistics.
        """
        return {
            "name": self.name,
            "pending": self.pending_count,
            "completed": self._completed_count,
            "failed": self._failed_count,
            "average_time_ms": self.average_time_ms,
            "total_time_ms": self._total_time_ms,
        }

    def reset_stats(self) -> None:
        """Reset statistics counters."""
        self._completed_count = 0
        self._failed_count = 0
        self._total_time_ms = 0.0


# =============================================================================
# Convenience Functions
# =============================================================================


async def run_concurrent(
    coros: list[Coroutine[Any, Any, T]],
    max_concurrent: int = 10,
    timeout: float = 120.0,
) -> list[T]:
    """Run coroutines with concurrency limit.

    Simple wrapper without full scheduler overhead.

    Args:
        coros: List of coroutines to execute.
        max_concurrent: Maximum concurrent executions.
        timeout: Timeout per coroutine.

    Returns:
        List of results.
    """
    sem = asyncio.Semaphore(max_concurrent)

    async def limited(c: Coroutine[Any, Any, T]) -> T:
        async with sem:
            return await asyncio.wait_for(c, timeout=timeout)

    return list(await asyncio.gather(*[limited(c) for c in coros]))

