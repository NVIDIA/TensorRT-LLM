# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Unit tests for the HostProfiler module.

Tests cover:
1. Core API functionality (add_*, clear_targets, chaining)
2. Line profiler integration with report validation
3. TLLM_LINE_PROFILER_FUNCTIONS env var override semantics
4. E2E test with actual model inference
"""

import os
import re
import sys
import tempfile

import pytest

# Add path for test utilities
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from utils.llm_data import llm_models_root

from tensorrt_llm.tools.profiler.host_profile_tools.host_profiler import (
    LINE_PROFILER_FUNCTIONS_ENV_VAR,
    PROFILE_START_STOP_ENV_VAR,
    HostProfiler,
    ProfileTarget,
    host_profiler_context,
)


def _sample_function_to_profile(n: int) -> int:
    """A simple function with multiple lines to profile."""
    total = 0
    for i in range(n):
        total += i
    return total


def _another_function(x: int, y: int) -> int:
    """Another function for testing."""
    result = x + y
    result *= 2
    return result


class _SampleClassToProfile:
    """A sample class with methods to profile."""

    def instance_method(self, n: int) -> int:
        """Instance method to profile."""
        result = 0
        for i in range(n):
            result += i * 2
        return result


def _patch_mpi_pool_session_for_env(mocker, env_vars: dict):
    """Patch MpiPoolSession to propagate env vars to MPI workers."""
    from mpi4py.futures import MPIPoolExecutor

    from tensorrt_llm.llmapi.mpi_session import MpiPoolSession

    def patched_start_mpi_pool(self):
        assert not self.mpi_pool, "MPI session already started"
        self.mpi_pool = MPIPoolExecutor(max_workers=self.n_workers, path=sys.path, env=env_vars)

    mocker.patch.object(MpiPoolSession, "_start_mpi_pool", patched_start_mpi_pool)


def test_add_and_clear_targets():
    """Test add_*, clear_targets(), and chaining work correctly."""
    profiler = HostProfiler(use_defaults=False)

    # Test add_* methods with chaining
    result = (
        profiler.add_function("m1", "C1", "f1")
        .add_standalone_function("m2", "f2")
        .add_target(ProfileTarget("m3", "C3", "f3"))
    )

    assert result is profiler
    assert len(profiler.targets) == 3
    assert profiler.list_targets() == ["m1.C1.f1", "m2.f2", "m3.C3.f3"]

    # Test clear_targets with chaining
    result = profiler.clear_targets()
    assert result is profiler
    assert len(profiler.targets) == 0

    # Test add after clear
    profiler.add_function("os", None, "getcwd")
    assert len(profiler.targets) == 1


def test_defaults_and_clear():
    """Test that defaults are loaded and can be cleared."""
    profiler = HostProfiler(use_defaults=True)
    initial_count = len(profiler.targets)
    assert initial_count > 10, "Should have many default targets"

    # Clear all and add custom
    profiler.clear_targets().add_standalone_function("os", "getcwd").add_standalone_function(
        "os.path", "join"
    )

    assert len(profiler.targets) == 2
    assert "os.getcwd" in profiler.list_targets()
    assert "os.path.join" in profiler.list_targets()


def test_profiling_cycle_and_report_validation():
    """Test complete profiling cycle and validate report format.

    This is the main unit test that validates:
    1. Profiler starts and stops correctly
    2. Report contains Timer unit header
    3. Report contains column headers (Line, Hits, Time)
    4. Report contains profiled function name
    5. Report contains timing data
    6. Only profiled functions appear (not non-profiled ones)
    """
    try:
        import line_profiler  # noqa: F401
    except ImportError:
        pytest.skip("line_profiler not installed")

    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
        output_path = f.name

    try:
        # Use clear_targets to ensure only our test functions are profiled
        profiler = HostProfiler(output_path=output_path, use_defaults=True)
        profiler.clear_targets().add_standalone_function(
            __name__, "_sample_function_to_profile"
        ).add_function(__name__, "_SampleClassToProfile", "instance_method")

        assert len(profiler.targets) == 2

        # Start profiling
        assert profiler.start() is True
        assert profiler.enabled is True

        # Execute profiled functions
        sample_obj = _SampleClassToProfile()
        for _ in range(50):
            _sample_function_to_profile(10)
            sample_obj.instance_method(5)
            # Execute non-profiled function - should NOT appear in output
            _another_function(3, 4)

        # Stop and save
        assert profiler.stop() is True
        assert profiler.enabled is False

        # Validate output file exists
        assert os.path.exists(output_path)
        with open(output_path) as f:
            content = f.read()

        # --- Report Format Validation ---

        # 1. Timer unit header
        assert "Timer unit:" in content, "Report missing 'Timer unit:' header"

        # 2. Column headers
        assert "Line" in content, "Report missing 'Line' column header"
        assert "Hits" in content, "Report missing 'Hits' column header"
        assert "Time" in content, "Report missing 'Time' column header"

        # 3. Profiled functions appear
        assert "_sample_function_to_profile" in content, "Profiled function not in output"
        assert "instance_method" in content, "Profiled method not in output"

        # 4. Non-profiled function should NOT appear
        assert "_another_function" not in content, "Non-profiled function should NOT be in output"

        # 5. Should have timing data (lines with numbers)
        lines = content.split("\n")
        data_lines = [
            line for line in lines if line.strip() and any(char.isdigit() for char in line)
        ]
        assert len(data_lines) > 5, "Report should have multiple lines with timing data"

        print("\n=== Report Validation Passed ===")
        print(f"Output size: {len(content)} bytes")
        print(f"Data lines: {len(data_lines)}")

    finally:
        if os.path.exists(output_path):
            os.unlink(output_path)


# ---------------------------------------------------------------------------
# TLLM_LINE_PROFILER_FUNCTIONS env var override tests
# ---------------------------------------------------------------------------


def test_env_functions_replaces_defaults():
    """Test TLLM_LINE_PROFILER_FUNCTIONS replaces default targets.

    When set, host_profiler_context should disable defaults and only profile
    the env-var specified functions.
    """
    try:
        import line_profiler  # noqa: F401
    except ImportError:
        pytest.skip("line_profiler not installed")

    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
        output_path = f.name

    try:
        # Set TLLM_LINE_PROFILER_FUNCTIONS — this should replace defaults
        func_path = f"{__name__}::_sample_function_to_profile"
        with pytest.MonkeyPatch.context() as mp:
            mp.setenv(LINE_PROFILER_FUNCTIONS_ENV_VAR, func_path)
            mp.setenv("TLLM_LINE_PROFILER_PATH", output_path)

            with host_profiler_context() as profiler:
                assert profiler is not None
                # Only env-var targets in profiler.targets (no defaults)
                assert len(profiler.targets) == 1
                assert profiler.targets[0].method_name == "_sample_function_to_profile"

                for _ in range(50):
                    _sample_function_to_profile(10)

        with open(output_path) as f:
            content = f.read()

        assert "Timer unit:" in content
        # Env-var target appears
        assert "_sample_function_to_profile" in content
        # Default targets like _forward_step should NOT appear
        assert "_forward_step" not in content
    finally:
        if os.path.exists(output_path):
            os.unlink(output_path)


def test_iteration_aware_profiling():
    """Test iteration-aware profiling using TLLM_PROFILE_START_STOP.

    Covers both cases:
    1. With TLLM_PROFILE_START_STOP set: only specified iterations are profiled.
    2. Without TLLM_PROFILE_START_STOP: all iterations are profiled.
    """
    try:
        import line_profiler  # noqa: F401
    except ImportError:
        pytest.skip("line_profiler not installed")

    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
        range_output = f.name
    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
        full_output = f.name

    try:
        # --- Case 1: iteration range set (profile iterations 5-9 only) ---
        with pytest.MonkeyPatch.context() as mp:
            mp.setenv(PROFILE_START_STOP_ENV_VAR, "5-10")
            mp.setenv("TLLM_LINE_PROFILER_PATH", range_output)

            profiler = HostProfiler(output_path=range_output, use_defaults=False)
            profiler.add_standalone_function(__name__, "_sample_function_to_profile")
            assert profiler.start() is True
            assert profiler._iteration_aware is True
            assert profiler._tracing_active is False

            for i in range(20):
                profiler.notify_iteration(i)
                _sample_function_to_profile(10)

            assert profiler.stop() is True

        with open(range_output) as f:
            range_content = f.read()

        assert "Timer unit:" in range_content
        assert "_sample_function_to_profile" in range_content
        range_hits = re.findall(r"^\s+\d+\s+(\d+)\s+", range_content, re.MULTILINE)
        range_max = max(int(h) for h in range_hits if int(h) > 0)
        assert range_max <= 6, (
            f"Expected hits ~5 (iterations 5-9), got max {range_max}. "
            "Iteration filtering may not be working."
        )

        # --- Case 2: no iteration range (profiles everything) ---
        with pytest.MonkeyPatch.context() as mp:
            mp.delenv(PROFILE_START_STOP_ENV_VAR, raising=False)
            mp.setenv("TLLM_LINE_PROFILER_PATH", full_output)

            profiler = HostProfiler(output_path=full_output, use_defaults=False)
            profiler.add_standalone_function(__name__, "_sample_function_to_profile")
            assert profiler.start() is True
            assert profiler._iteration_aware is False
            assert profiler._tracing_active is True

            # notify_iteration is a no-op when not iteration-aware
            for i in range(20):
                profiler.notify_iteration(i)
                _sample_function_to_profile(10)

            assert profiler.stop() is True

        with open(full_output) as f:
            full_content = f.read()

        assert "_sample_function_to_profile" in full_content
        full_hits = re.findall(r"^\s+\d+\s+(\d+)\s+", full_content, re.MULTILINE)
        full_max = max(int(h) for h in full_hits if int(h) > 0)
        assert full_max >= 20, f"Expected hits >= 20 (all iterations), got {full_max}"

        # Sanity: iteration-ranged hits should be much less than full
        assert range_max < full_max
    finally:
        for p in (range_output, full_output):
            if os.path.exists(p):
                os.unlink(p)


# Core PyExecutor methods that are always executed during inference.
# These are stable, fundamental methods unlikely to be renamed.
E2E_PROFILE_TARGETS = [
    "tensorrt_llm._torch.pyexecutor.py_executor.PyExecutor._forward_step",
    "tensorrt_llm._torch.pyexecutor.py_executor.PyExecutor._schedule",
    "tensorrt_llm._torch.pyexecutor.sampler.TorchSampler.sample_async",
]


@pytest.fixture
def tinyllama_path():
    """Get TinyLlama model path."""
    model_path = llm_models_root() / "llama-models-v2" / "TinyLlama-1.1B-Chat-v1.0"
    if not model_path.exists():
        pytest.skip(f"TinyLlama model not found at {model_path}")
    return str(model_path)


def test_e2e_profiler_with_model(tinyllama_path, mocker):
    """E2E test: verify profiler works with actual model inference.

    Clears default profile targets and adds only specific targets,
    then verifies those targets appear in the report with non-zero timing.
    """
    from tensorrt_llm import LLM
    from tensorrt_llm.llmapi import KvCacheConfig, SamplingParams
    from tensorrt_llm.tools.profiler.host_profile_tools import host_profiler

    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = os.path.join(tmpdir, "profiler_output.txt")

        # Clear default targets and use only our specific targets
        mocker.patch.object(host_profiler, "DEFAULT_PROFILE_TARGETS", [])

        # Patch MPI to propagate env vars to workers
        _patch_mpi_pool_session_for_env(
            mocker,
            {
                "TLLM_LINE_PROFILER_PATH": output_path,
                "TLLM_LINE_PROFILER_FUNCTIONS": ", ".join(E2E_PROFILE_TARGETS),
            },
        )

        with LLM(
            model=tinyllama_path,
            kv_cache_config=KvCacheConfig(free_gpu_memory_fraction=0.3),
        ) as llm:
            # Generate enough tokens to ensure profiled methods are executed
            prompts = ["Hello, how are you?", "What is AI?"]
            outputs = llm.generate(prompts, SamplingParams(max_tokens=16, end_id=-1))
            assert len(outputs) == len(prompts)

        # Validate output file was created
        assert os.path.exists(output_path), f"Profiler output not created at {output_path}"

        with open(output_path) as f:
            content = f.read()

        # Validate report format
        assert "Timer unit:" in content, "Missing Timer unit header"

        # Verify all specified targets were profiled with actual timing data
        # Format: "Total time: X s" then "File: ..." then "Function: ClassName.method_name at line X"
        expected_methods = ["_forward_step", "_schedule", "sample_async"]

        for method in expected_methods:
            # Find block: "Total time: X s" followed by "File: ..." then "Function: ...method_name..."
            pattern = (
                rf"Total time:\s*([\d.e+-]+)\s*s\nFile:.*?\nFunction:.*\.{method}\s+at line \d+"
            )
            match = re.search(pattern, content)

            assert match, f"Method '{method}' not found in profiler output"

            total_time = float(match.group(1))
            assert total_time > 0, f"Method '{method}' has zero total time - not actually profiled"

        print("\n=== E2E Passed ===")
        print(f"Output: {len(content)} bytes")
        print(f"Verified methods with timing data: {expected_methods}")
