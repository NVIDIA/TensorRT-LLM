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
"""CPU coverage for benchmark startup-timeout delegation."""

from pathlib import Path

import pytest
import test_disaggregated as disaggregated_tests

pytestmark = pytest.mark.cpu_only

_CONFIG_FILE = Path(__file__).parent / "test_configs" / "disagg_config_ctxtp2pp2_gentp2pp2.yaml"


class _ClusterSetupReached(Exception):
    def __init__(self, startup_budget: int | None) -> None:
        super().__init__()
        self.startup_budget = startup_budget


def _observed_startup_budget(
    monkeypatch: pytest.MonkeyPatch, **timeout_override: int | None
) -> int | None:
    def capture_cluster_setup(
        config_file: str, *, server_start_timeout: int | None = 900, **setup_options: object
    ) -> None:
        raise _ClusterSetupReached(server_start_timeout)

    monkeypatch.setattr(disaggregated_tests, "setup_disagg_cluster", capture_cluster_setup)
    monkeypatch.setattr(disaggregated_tests, "get_ucx_tls", lambda: "all")

    with pytest.raises(_ClusterSetupReached) as startup:
        disaggregated_tests.run_disaggregated_benchmark(
            "example",
            str(_CONFIG_FILE),
            "benchmark",
            "model",
            "dataset",
            env={"STARTUP_TIMEOUT_TEST": "1"},
            **timeout_override,
        )

    return startup.value.startup_budget


def test_benchmark_preserves_cluster_startup_default(monkeypatch: pytest.MonkeyPatch) -> None:
    assert _observed_startup_budget(monkeypatch) == 900


def test_benchmark_forwards_explicit_startup_override(monkeypatch: pytest.MonkeyPatch) -> None:
    assert _observed_startup_budget(monkeypatch, server_start_timeout=1200) == 1200
