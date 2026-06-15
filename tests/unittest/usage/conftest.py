# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Conftest for usage tests.

Patches the tensorrt_llm top-level __init__.py import chain so that
the usage subpackage can be tested in isolation (without GPU libs,
nvtx, MPI, etc.).

This works by pre-populating sys.modules with a stub 'tensorrt_llm'
package that only contains the usage subpackage, before any test
imports trigger the full (heavy) __init__.py.
"""

import sys
import types
from pathlib import Path

import pytest

# Locate the repo root so we can import from tensorrt_llm/usage/ directly
_REPO_ROOT = Path(__file__).resolve().parents[3]
_TRTLLM_PKG = _REPO_ROOT / "tensorrt_llm"


def _create_stub_tensorrt_llm():
    """Create a minimal stub tensorrt_llm package in sys.modules.

    This prevents the real __init__.py (which imports torch, nvtx, etc.)
    from executing, while still allowing `from tensorrt_llm.usage import ...`
    to resolve correctly.
    """
    if "tensorrt_llm" in sys.modules:
        # Already loaded (e.g. in a full TRT-LLM environment) -- skip
        return

    # Create stub package module
    stub = types.ModuleType("tensorrt_llm")
    stub.__path__ = [str(_TRTLLM_PKG)]
    stub.__file__ = str(_TRTLLM_PKG / "__init__.py")
    stub.__package__ = "tensorrt_llm"
    stub.__version__ = "0.0.0-test"
    sys.modules["tensorrt_llm"] = stub


# Run before any test collection
_create_stub_tensorrt_llm()


def pytest_addoption(parser):
    """Register --run-staging CLI flag for live endpoint tests."""
    parser.addoption(
        "--run-staging",
        action="store_true",
        default=False,
        help="Run live tests against the GXT staging endpoint.",
    )


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "staging: live tests against GXT staging endpoint (require --run-staging)"
    )


@pytest.fixture
def enable_telemetry(monkeypatch):
    """Clear all opt-out env vars and force-enable for clean telemetry testing.

    Uses TRTLLM_USAGE_FORCE_ENABLED=1 to override CI/test auto-detection
    (PYTEST_CURRENT_TEST is set by pytest after fixtures run, so delenv
    is not sufficient).
    """
    monkeypatch.delenv("TRTLLM_NO_USAGE_STATS", raising=False)
    monkeypatch.delenv("DO_NOT_TRACK", raising=False)
    monkeypatch.delenv("TELEMETRY_DISABLED", raising=False)
    monkeypatch.setenv("TRTLLM_USAGE_FORCE_ENABLED", "1")
    monkeypatch.setattr(
        "tensorrt_llm.usage.usage_lib._OPT_OUT_FILE",
        Path("/nonexistent/path/do_not_track"),
    )
