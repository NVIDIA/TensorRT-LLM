# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
3. E2E test with actual model inference
"""

import os
import re
import sys
import tempfile

import pytest

# Add path for test utilities
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from utils.llm_data import llm_models_root

from tensorrt_llm.tools.profiler.host_profile_tools.host_profiler import HostProfiler, ProfileTarget


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
