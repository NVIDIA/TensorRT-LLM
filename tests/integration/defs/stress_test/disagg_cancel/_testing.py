# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Shared helpers for ``disagg_cancel`` thread-body unit tests.

Each thread (``log_scanner``, ``metrics``, ``injector``, ``canary``,
``load``) is tested in isolation against a harness wired with
placeholder ``WorkerLaunchSpec`` entries — no live disagg cluster.
Per-thread tests only need to override the spec fields relevant to
their thread, so this module exposes a single ``make_spec`` factory
that fills in the rest, plus a minimal valid marathon YAML for the
``DisaggCancellationStressHarness`` constructor.
"""

from __future__ import annotations

import textwrap
from pathlib import Path
from typing import Optional

from .harness import WorkerLaunchSpec

# Minimal valid marathon YAML for tests that only need the harness to
# construct. Includes ``log_scan`` so the log-scanner thread has
# patterns to compile; threads that don't read ``log_scan`` ignore it.
DUMMY_YAML = textwrap.dedent(
    """\
    hostname: localhost
    model: dummy
    backend: pytorch
    context_servers: {}
    generation_servers: {}
    stress_config:
      duration_min: 1
      kv_cache_manager: v1
      transceiver: cpp
      log_scan:
        hard_zero_patterns:
          - "Broken promise"
          - "Segfault"
          - "Poisoned .* cache transfer buffer"
    """
)


def make_spec(
    role: str,
    index: int,
    *,
    port: int = 18000,
    host: str = "localhost",
    log_path: Optional[Path] = None,
) -> WorkerLaunchSpec:
    """``WorkerLaunchSpec`` factory for unit tests.

    Callers override only the fields their thread reads (``log_path``
    for the log scanner, ``host`` / ``port`` for the metrics scraper,
    etc.); the rest are placeholders that ``setup_disagg_cluster``
    would normally populate.

    Args:
        role: ``"ctx"`` or ``"gen"``. Surfaces in failure reasons.
        index: Per-role index (``ctx_0``, ``gen_0``, ...).
        port: TCP port for HTTP scraping. Default ``18000``; tests
            that bind a real socket pass an OS-allocated port.
        host: HTTP hostname; default ``"localhost"``.
        log_path: File path the log scanner should tail, or ``None``
            to simulate a worker launched with ``save_log=False``.
    """
    return WorkerLaunchSpec(
        role=role,
        index=index,
        model_name="dummy",
        worker_config={},
        work_dir="/tmp",
        port=port,
        device="0",
        env={},
        log_path=str(log_path) if log_path is not None else None,
        host=host,
    )
