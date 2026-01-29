"""
ShiphaEvolve Evaluator with Tiered Trust Levels and Parallel Execution

This module provides:
- SANDBOX: Full Docker isolation (safest, slowest)
- RESTRICTED: subprocess with resource limits (medium)
- DIRECT: in-process execution (fastest, trusted code only)

Parallel execution via ThreadPoolExecutor for running multiple
evaluations concurrently.
"""

import asyncio
import json
import logging
import os
import tempfile
import time
import subprocess
import resource
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class TrustLevel(Enum):
    """Evaluation trust levels from most to least isolated."""

    SANDBOX = "sandbox"  # Docker container (safest)
    RESTRICTED = "restricted"  # Subprocess with limits
    DIRECT = "direct"  # In-process (fastest, trusted only)


@dataclass
class EvaluatorConfig:
    """Configuration for the evaluator."""

    trust_level: TrustLevel = TrustLevel.SANDBOX
    docker_image: str = "python:3.11-slim"
    memory_limit_mb: int = 512
    cpu_limit: float = 1.0
    timeout_seconds: float = 30.0
    network_disabled: bool = True
    max_workers: int = 4
    enable_parallel: bool = True


@dataclass
class TestCase:
    """A single test case."""

    inputs: Any  # Can be list (positional) or dict (keyword)
    expected_output: Optional[Any] = None
    validation_func: Optional[str] = None  # Python code defining validate(output) -> bool
    timeout_seconds: Optional[float] = None


@dataclass
class TestGroup:
    """Group of test cases at a specific level."""

    name: str
    level: int
    test_cases: List[TestCase]


@dataclass
class EvaluationResult:
    """Result of evaluating a single program."""

    program_id: str
    correctness: float = 0.0
    passed_tests: int = 0
    total_tests: int = 0
    average_runtime_ms: float = float("inf")
    errors: List[str] = field(default_factory=list)
    test_outputs: List[Dict[str, Any]] = field(default_factory=list)
    execution_time_seconds: float = 0.0


class ExecutionBackend(ABC):
    """Abstract base class for code execution backends."""

    @abstractmethod
    async def execute(
        self,
        code: str,
        function_name: str,
        test_cases: List[TestCase],
        timeout: float,
    ) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        """Execute code against test cases."""
        raise NotImplementedError


class DockerBackend(ExecutionBackend):
    """Execute code in Docker container (SANDBOX level)."""

    def __init__(self, config: EvaluatorConfig):
        self.config = config

    async def execute(
        self,
        code: str,
        function_name: str,
        test_cases: List[TestCase],
        timeout: float,
    ) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        """Execute code in isolated Docker container."""
        temp_dir = tempfile.mkdtemp()
        temp_file_path = os.path.join(temp_dir, "temp_script.py")

        try:
            # Serialize test cases
            test_cases_data = [
                {
                    "input": tc.inputs,
                    "output": tc.expected_output,
                    "validation_func": tc.validation_func,
                }
                for tc in test_cases
            ]
            test_cases_str = json.dumps(test_cases_data)
            test_cases_str = test_cases_str.replace('"Infinity"', 'float("inf")')
            test_cases_str = test_cases_str.replace('"-Infinity"', 'float("-inf")')
            test_cases_str = test_cases_str.replace('"NaN"', 'float("nan")')
            test_cases_str = (
                test_cases_str.replace("true", "True")
                .replace("false", "False")
                .replace("null", "None")
            )

            test_harness_code = f'''
import json
import time
import sys
import math
import numpy as np

# User's code
{code}

results = []
total_execution_time = 0
num_tests = 0

test_cases = {test_cases_str}
function_to_test_name = "{function_name}"

if function_to_test_name not in globals():
    for name, obj in list(globals().items()):
        if isinstance(obj, type):
            if hasattr(obj, function_to_test_name):
                method = getattr(obj, function_to_test_name)
                if callable(method):
                    globals()[function_to_test_name] = method
                    break
    else:
        print(json.dumps({{"error": f"Function '{{function_to_test_name}}' not found"}}))
        sys.exit(1)

function_to_test = globals()[function_to_test_name]

for i, test_case in enumerate(test_cases):
    input_args = test_case.get("input")
    start_time = time.perf_counter()
    try:
        if isinstance(input_args, list):
            actual_output = function_to_test(*input_args)
        elif isinstance(input_args, dict):
            actual_output = function_to_test(**input_args)
        elif input_args is None:
            actual_output = function_to_test()
        else:
            actual_output = function_to_test(input_args)

        end_time = time.perf_counter()
        execution_time_ms = (end_time - start_time) * 1000
        total_execution_time += execution_time_ms
        num_tests += 1
        results.append({{
            "test_case_id": i,
            "output": actual_output,
            "runtime_ms": execution_time_ms,
            "status": "success"
        }})
    except Exception as e:
        end_time = time.perf_counter()
        execution_time_ms = (end_time - start_time) * 1000
        results.append({{
            "test_case_id": i,
            "error": str(e),
            "error_type": type(e).__name__,
            "runtime_ms": execution_time_ms,
            "status": "error"
        }})

final_output = {{"test_outputs": results}}
if num_tests > 0:
    final_output["average_runtime_ms"] = total_execution_time / num_tests

def custom_json_serializer(obj):
    if hasattr(obj, "tolist"):
        return obj.tolist()
    if hasattr(obj, "item"):
        return obj.item()
    if isinstance(obj, float):
        if obj == float("inf"):
            return "Infinity"
        elif obj == float("-inf"):
            return "-Infinity"
        elif obj != obj:
            return "NaN"
    raise TypeError(f"Object of type {{type(obj).__name__}} is not JSON serializable")

print(json.dumps(final_output, default=custom_json_serializer))
'''
            with open(temp_file_path, "w") as f:
                f.write(test_harness_code)

            # Unique container name for parallel execution
            container_name = f"shipha-eval-{os.getpid()}-{time.time_ns()}"

            cmd: List[str] = [
                "docker",
                "run",
                "--rm",
                "--name",
                container_name,
                "-i",
                f"--memory={self.config.memory_limit_mb}m",
                f"--cpus={self.config.cpu_limit}",
            ]

            if self.config.network_disabled:
                cmd.extend(["--network", "none"])

            cmd.extend(
                [
                    "-v",
                    f"{os.path.abspath(temp_dir)}:/app/user_code",
                    "-w",
                    "/app/user_code",
                    self.config.docker_image,
                    "python",
                    "temp_script.py",
                ]
            )

            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            try:
                stdout, stderr = await asyncio.wait_for(
                    proc.communicate(), timeout=timeout
                )
            except asyncio.TimeoutError:
                # Kill the container
                stop_proc = await asyncio.create_subprocess_exec(
                    "docker", "stop", container_name
                )
                await asyncio.wait_for(stop_proc.wait(), timeout=10)
                return None, f"Execution timed out after {timeout}s"

            stdout_str = stdout.decode("utf-8", errors="replace").strip()
            stderr_str = stderr.decode("utf-8", errors="replace").strip()

            if proc.returncode != 0:
                if not stdout_str:
                    return None, f"Docker error: {stderr_str}"
                logger.warning(f"Non-zero exit but has stdout: {stderr_str}")

            if not stdout_str:
                return None, f"No output. Stderr: {stderr_str}"

            try:
                return json.loads(stdout_str), None
            except json.JSONDecodeError as e:
                return None, f"JSON decode error: {e}. Output: {stdout_str[:500]}"

        finally:
            # Cleanup
            try:
                if os.path.exists(temp_file_path):
                    os.remove(temp_file_path)
                if os.path.exists(temp_dir):
                    os.rmdir(temp_dir)
            except Exception as e:
                logger.warning(f"Cleanup error: {e}")


class SubprocessBackend(ExecutionBackend):
    """Execute code in restricted subprocess (RESTRICTED level)."""

    def __init__(self, config: EvaluatorConfig):
        self.config = config

    async def execute(
        self,
        code: str,
        function_name: str,
        test_cases: List[TestCase],
        timeout: float,
    ) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        """Execute code in subprocess with resource limits."""
        temp_dir = tempfile.mkdtemp()
        temp_file_path = os.path.join(temp_dir, "temp_script.py")

        try:
            # Same test harness as Docker but with subprocess
            test_cases_data = [
                {"input": tc.inputs, "output": tc.expected_output}
                for tc in test_cases
            ]
            test_cases_str = json.dumps(test_cases_data)

            # Wrapper that sets resource limits
            test_harness_code = f'''
import json
import time
import sys
import resource

# Set resource limits
resource.setrlimit(resource.RLIMIT_AS, ({self.config.memory_limit_mb * 1024 * 1024}, {self.config.memory_limit_mb * 1024 * 1024}))
resource.setrlimit(resource.RLIMIT_CPU, ({int(timeout)}, {int(timeout)}))

# User's code
{code}

test_cases = {test_cases_str}
function_to_test = globals().get("{function_name}")
if not function_to_test:
    print(json.dumps({{"error": "Function not found"}}))
    sys.exit(1)

results = []
total_time = 0
num_tests = 0

for i, tc in enumerate(test_cases):
    input_args = tc.get("input")
    start = time.perf_counter()
    try:
        if isinstance(input_args, list):
            output = function_to_test(*input_args)
        elif isinstance(input_args, dict):
            output = function_to_test(**input_args)
        else:
            output = function_to_test(input_args) if input_args is not None else function_to_test()
        end = time.perf_counter()
        runtime_ms = (end - start) * 1000
        total_time += runtime_ms
        num_tests += 1
        results.append({{"test_case_id": i, "output": output, "runtime_ms": runtime_ms, "status": "success"}})
    except Exception as e:
        end = time.perf_counter()
        results.append({{"test_case_id": i, "error": str(e), "runtime_ms": (end - start) * 1000, "status": "error"}})

final = {{"test_outputs": results}}
if num_tests > 0:
    final["average_runtime_ms"] = total_time / num_tests
print(json.dumps(final, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))
'''

            with open(temp_file_path, "w") as f:
                f.write(test_harness_code)

            proc = await asyncio.create_subprocess_exec(
                sys.executable,
                temp_file_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            try:
                stdout, stderr = await asyncio.wait_for(
                    proc.communicate(), timeout=timeout
                )
            except asyncio.TimeoutError:
                proc.kill()
                return None, f"Subprocess timed out after {timeout}s"

            stdout_str = stdout.decode("utf-8", errors="replace").strip()

            if not stdout_str:
                stderr_str = stderr.decode("utf-8", errors="replace").strip()
                return None, f"No output. Stderr: {stderr_str}"

            try:
                return json.loads(stdout_str), None
            except json.JSONDecodeError as e:
                return None, f"JSON decode error: {e}"

        finally:
            try:
                if os.path.exists(temp_file_path):
                    os.remove(temp_file_path)
                if os.path.exists(temp_dir):
                    os.rmdir(temp_dir)
            except Exception:
                pass


class DirectBackend(ExecutionBackend):
    """Execute code directly in-process (DIRECT level - trusted only)."""

    def __init__(self, config: EvaluatorConfig):
        self.config = config

    async def execute(
        self,
        code: str,
        function_name: str,
        test_cases: List[TestCase],
        timeout: float,
    ) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        """Execute code directly - ONLY for trusted code!"""
        # Create isolated namespace
        namespace: Dict[str, Any] = {}

        try:
            exec(code, namespace)
        except Exception as e:
            return None, f"Code execution error: {e}"

        func = namespace.get(function_name)
        if not func or not callable(func):
            return None, f"Function '{function_name}' not found"

        results = []
        total_time = 0.0
        num_tests = 0

        for i, tc in enumerate(test_cases):
            start = time.perf_counter()
            try:
                if isinstance(tc.inputs, list):
                    output = func(*tc.inputs)
                elif isinstance(tc.inputs, dict):
                    output = func(**tc.inputs)
                elif tc.inputs is None:
                    output = func()
                else:
                    output = func(tc.inputs)

                end = time.perf_counter()
                runtime_ms = (end - start) * 1000
                total_time += runtime_ms
                num_tests += 1
                results.append(
                    {
                        "test_case_id": i,
                        "output": output,
                        "runtime_ms": runtime_ms,
                        "status": "success",
                    }
                )
            except Exception as e:
                results.append(
                    {
                        "test_case_id": i,
                        "error": str(e),
                        "runtime_ms": (time.perf_counter() - start) * 1000,
                        "status": "error",
                    }
                )

        return {
            "test_outputs": results,
            "average_runtime_ms": total_time / num_tests if num_tests > 0 else 0,
        }, None


class ParallelEvaluator:
    """
    Evaluator with tiered trust levels and parallel execution.

    This class manages the evaluation of multiple programs concurrently,
    using the appropriate execution backend based on the configured trust level.
    """

    def __init__(self, config: EvaluatorConfig):
        self.config = config
        self._executor: Optional[ThreadPoolExecutor] = None

        # Select backend based on trust level
        if config.trust_level == TrustLevel.SANDBOX:
            self.backend = DockerBackend(config)
        elif config.trust_level == TrustLevel.RESTRICTED:
            self.backend = SubprocessBackend(config)
        else:
            self.backend = DirectBackend(config)

        logger.info(
            f"ParallelEvaluator initialized: trust_level={config.trust_level.value}, "
            f"max_workers={config.max_workers}, parallel={config.enable_parallel}"
        )

    @property
    def executor(self) -> ThreadPoolExecutor:
        """Lazy initialization of thread pool."""
        if self._executor is None:
            self._executor = ThreadPoolExecutor(
                max_workers=self.config.max_workers,
                thread_name_prefix="shipha-eval",
            )
        return self._executor

    def shutdown(self) -> None:
        """Shutdown the thread pool."""
        if self._executor:
            self._executor.shutdown(wait=True)
            self._executor = None

    async def evaluate_single(
        self,
        program_id: str,
        code: str,
        function_name: str,
        test_cases: List[TestCase],
    ) -> EvaluationResult:
        """Evaluate a single program."""
        start_time = time.monotonic()
        result = EvaluationResult(
            program_id=program_id,
            total_tests=len(test_cases),
        )

        # Syntax check
        try:
            compile(code + "\n", "program.py", "exec")
        except SyntaxError as e:
            result.errors.append(f"SyntaxError: {e.msg} at line {e.lineno}")
            result.execution_time_seconds = time.monotonic() - start_time
            return result

        # Execute
        execution_result, error = await self.backend.execute(
            code=code,
            function_name=function_name,
            test_cases=test_cases,
            timeout=self.config.timeout_seconds,
        )

        if error:
            result.errors.append(error)
            result.execution_time_seconds = time.monotonic() - start_time
            return result

        if execution_result is None:
            result.errors.append("No execution result returned")
            result.execution_time_seconds = time.monotonic() - start_time
            return result

        # Assess correctness
        test_outputs = execution_result.get("test_outputs", [])
        result.test_outputs = test_outputs
        result.average_runtime_ms = execution_result.get("average_runtime_ms", 0.0)

        passed = 0
        for i, tc in enumerate(test_cases):
            actual_detail = next(
                (t for t in test_outputs if t.get("test_case_id") == i), None
            )
            if actual_detail and actual_detail.get("status") == "success":
                actual = actual_detail.get("output")

                # Use validation function if provided
                if tc.validation_func:
                    try:
                        namespace: Dict[str, Any] = {}
                        exec(tc.validation_func, namespace)
                        validate = namespace.get("validate")
                        if validate and callable(validate) and validate(actual):
                            passed += 1
                    except Exception as e:
                        logger.warning(f"Validation function error: {e}")
                # Otherwise compare with expected
                elif tc.expected_output is not None:
                    if self._compare_outputs(actual, tc.expected_output):
                        passed += 1

        result.passed_tests = passed
        result.correctness = passed / len(test_cases) if test_cases else 0.0
        result.execution_time_seconds = time.monotonic() - start_time

        return result

    def _compare_outputs(self, actual: Any, expected: Any) -> bool:
        """Compare two outputs with tolerance for floats."""
        if isinstance(expected, float) and isinstance(actual, (int, float)):
            if expected != expected:  # NaN
                return actual != actual
            return abs(actual - expected) < 1e-6

        if isinstance(expected, list) and isinstance(actual, list):
            if len(expected) != len(actual):
                return False
            return all(self._compare_outputs(a, e) for a, e in zip(actual, expected))

        if isinstance(expected, dict) and isinstance(actual, dict):
            if expected.keys() != actual.keys():
                return False
            return all(
                self._compare_outputs(actual[k], expected[k]) for k in expected
            )

        return actual == expected

    async def evaluate_batch(
        self,
        programs: List[Tuple[str, str, str, List[TestCase]]],
    ) -> List[EvaluationResult]:
        """
        Evaluate multiple programs in parallel.

        Args:
            programs: List of (program_id, code, function_name, test_cases) tuples

        Returns:
            List of EvaluationResult objects in the same order as input
        """
        if not self.config.enable_parallel or len(programs) <= 1:
            # Sequential evaluation
            results = []
            for program_id, code, function_name, test_cases in programs:
                result = await self.evaluate_single(
                    program_id, code, function_name, test_cases
                )
                results.append(result)
            return results

        # Parallel evaluation
        loop = asyncio.get_event_loop()
        futures = {}

        for i, (program_id, code, function_name, test_cases) in enumerate(programs):
            future = loop.run_in_executor(
                self.executor,
                lambda pid=program_id, c=code, fn=function_name, tc=test_cases: (
                    asyncio.run(self.evaluate_single(pid, c, fn, tc))
                ),
            )
            futures[future] = i

        results = [None] * len(programs)  # type: ignore
        for future in asyncio.as_completed(futures.keys()):
            idx = futures[future]  # type: ignore
            try:
                results[idx] = await future
            except Exception as e:
                program_id = programs[idx][0]
                results[idx] = EvaluationResult(
                    program_id=program_id,
                    errors=[f"Parallel evaluation error: {e}"],
                )

        return results  # type: ignore


# Convenience function for simple usage
async def evaluate_programs(
    programs: List[Tuple[str, str, str, List[TestCase]]],
    trust_level: TrustLevel = TrustLevel.SANDBOX,
    max_workers: int = 4,
) -> List[EvaluationResult]:
    """
    Convenience function to evaluate multiple programs.

    Args:
        programs: List of (program_id, code, function_name, test_cases) tuples
        trust_level: Isolation level for execution
        max_workers: Number of parallel workers

    Returns:
        List of EvaluationResult objects
    """
    config = EvaluatorConfig(trust_level=trust_level, max_workers=max_workers)
    evaluator = ParallelEvaluator(config)
    try:
        return await evaluator.evaluate_batch(programs)
    finally:
        evaluator.shutdown()
