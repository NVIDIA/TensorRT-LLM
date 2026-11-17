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
"""Unit tests for the TRTLLM_MOONCAKE_* switches around chain mode.

Every switch is off by default, and only the `TRTLLM_MOONCAKE_*` names are
read. Switches read at import time are checked in a fresh interpreter. Runs
without a Mooncake installation and without a GPU.
"""

import json
import os
import socket
import subprocess  # nosec B404
import sys
from types import SimpleNamespace

import pytest

from tensorrt_llm._torch.pyexecutor import py_executor
from tensorrt_llm._torch.pyexecutor.connectors.mooncake_store import keys as keys_module
from tensorrt_llm._torch.pyexecutor.connectors.mooncake_store import master as master_module
from tensorrt_llm._torch.pyexecutor.connectors.mooncake_store import settings
from tensorrt_llm._torch.pyexecutor.connectors.mooncake_store.scheduler import _common_prefix_tokens
from tensorrt_llm.llmapi.llm_args import KvCacheConfig, MooncakeStoreConfig
from tensorrt_llm.runtime.kv_cache_manager_v2 import CacheTier

pytestmark = pytest.mark.cpu_only

GIB = 1 << 30

#: Switches read once, when their module is imported: (module, attribute, switch).
IMPORT_TIME_SWITCHES = (
    (
        "tensorrt_llm._torch.pyexecutor.connectors.mooncake_store.chain",
        "_KEYS_PY",
        "TRTLLM_MOONCAKE_KEYS_PY",
    ),
    (
        "tensorrt_llm._torch.pyexecutor.connectors.mooncake_store.scheduler",
        "_CARRY",
        "TRTLLM_MOONCAKE_CARRY",
    ),
    (
        "tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2",
        "_SKIP_UNUSED_CONNECTOR_GATHER",
        "TRTLLM_MOONCAKE_CARRY",
    ),
)
ALL_SWITCHES = (
    "TRTLLM_MOONCAKE_CHAIN",
    "TRTLLM_MOONCAKE_KEYS_PY",
    "TRTLLM_MOONCAKE_CARRY",
    "TRTLLM_MOONCAKE_CARRY_MAX",
    "TRTLLM_MOONCAKE_MASTER_EXTRA_ARGS",
)


@pytest.fixture(autouse=True)
def clean_switches(monkeypatch):
    for name in ALL_SWITCHES:
        monkeypatch.delenv(name, raising=False)


def import_time_values(env):
    """The import-time switch values a fresh interpreter sees under `env`."""
    script = (
        "import importlib, json, sys\n"
        "out = {}\n"
        "for module, attr in json.loads(sys.argv[1]):\n"
        "    out[module + '.' + attr] = getattr(importlib.import_module(module), attr)\n"
        "print('VALUES=' + json.dumps(out))\n"
    )
    targets = json.dumps([[module, attr] for module, attr, _ in IMPORT_TIME_SWITCHES])
    child_env = {
        name: value for name, value in os.environ.items() if not name.startswith("TRTLLM_MOONCAKE_")
    }
    child_env.update(env)
    result = subprocess.run(  # nosec B603
        [sys.executable, "-c", script, targets],
        env=child_env,
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )
    assert result.returncode == 0, result.stderr[-4000:]
    line = [ln for ln in result.stdout.splitlines() if ln.startswith("VALUES=")][-1]
    return json.loads(line[len("VALUES=") :])


def test_import_time_switches_are_off_by_default():
    assert set(import_time_values({}).values()) == {False}


def test_import_time_switches_read_their_env_vars():
    enabled = {switch: "1" for _, _, switch in IMPORT_TIME_SWITCHES}
    assert set(import_time_values(enabled).values()) == {True}


# ---- native host tier under the connector ----


@pytest.fixture
def executor_check(monkeypatch):
    """Let a test patch PyExecutor's tier check and have it put back afterwards."""
    original = py_executor.PyExecutor.__dict__["_reject_non_gpu_cache_tiers"]
    monkeypatch.setattr(py_executor.PyExecutor, "_reject_non_gpu_cache_tiers", original)
    monkeypatch.setattr(
        py_executor.PyExecutor, "_mooncake_chain_tiers_allowed", False, raising=False
    )
    monkeypatch.setattr(settings, "HOST_TIER_MANAGER", None)
    return py_executor.PyExecutor


def manager_with_host_tier():
    tiers = [SimpleNamespace(tier=CacheTier.GPU_MEM), SimpleNamespace(tier=CacheTier.HOST_MEM)]
    return SimpleNamespace(impl=SimpleNamespace(init_config=SimpleNamespace(cache_tiers=tiers)))


def test_without_chain_mode_the_native_tiers_go_off(executor_check):
    config = KvCacheConfig(host_cache_size=4 * GIB)
    settings.disable_native_kv_offload(config)
    assert (config.host_cache_size, config.disk_cache_size) == (0, 0)
    with pytest.raises(NotImplementedError, match="cache tiers below GPU"):
        executor_check._reject_non_gpu_cache_tiers(manager_with_host_tier())


@pytest.mark.parametrize("host_cache_size", [4 * GIB, None])
def test_chain_mode_keeps_the_host_tier_and_hands_the_manager_to_the_worker(
    monkeypatch, executor_check, host_cache_size
):
    monkeypatch.setenv("TRTLLM_MOONCAKE_CHAIN", "1")
    config = KvCacheConfig(host_cache_size=host_cache_size)
    settings.disable_native_kv_offload(config)
    assert config.host_cache_size == host_cache_size
    assert executor_check._mooncake_chain_tiers_allowed is True
    manager = manager_with_host_tier()
    executor_check._reject_non_gpu_cache_tiers(manager)  # no longer raises
    assert settings.HOST_TIER_MANAGER is manager


# ---- master flags ----


class _Launched(Exception):
    pass


@pytest.mark.parametrize(
    "extra, expected",
    [
        (None, []),
        (
            "--eviction_high_watermark_ratio=0.9  --default_kv_lease_ttl=2000",
            ["--eviction_high_watermark_ratio=0.9", "--default_kv_lease_ttl=2000"],
        ),
    ],
)
def test_master_extra_args_are_appended(monkeypatch, tmp_path, extra, expected):
    if extra is not None:
        monkeypatch.setenv("TRTLLM_MOONCAKE_MASTER_EXTRA_ARGS", extra)
    commands = []

    def popen(command, **kwargs):
        commands.append(list(command))
        raise _Launched()

    monkeypatch.setattr(master_module.shutil, "which", lambda binary: "/usr/bin/" + binary)
    monkeypatch.setattr(master_module, "local_address", lambda: "127.0.0.1")
    monkeypatch.setattr(master_module.subprocess, "Popen", popen)
    with pytest.raises(_Launched):
        master_module._launch_master(str(tmp_path))
    (command,) = commands
    assert command[0] == "/usr/bin/mooncake_master"
    if expected:
        assert command[-len(expected) :] == expected
    else:
        assert not any("watermark" in arg or "lease" in arg for arg in command)


# ---- run_dir ----


def test_run_dir_host_placeholder_is_this_nodes_short_hostname(monkeypatch):
    monkeypatch.setattr(socket, "gethostname", lambda: "node-17.cluster.example")
    pool = MooncakeStoreConfig(
        pool="file:///shared/pool.json", model_key="m", run_dir="/runs/{host}/pool"
    )
    assert pool.run_dir == "/runs/node-17/pool"
    plain = MooncakeStoreConfig(pool="file:///shared/pool.json", model_key="m", run_dir="/runs/x")
    assert plain.run_dir == "/runs/x"


# ---- hash-chain carry across turns ----


@pytest.mark.parametrize(
    "a, b, expected",
    [
        (list(range(12)), list(range(12)), 12),
        (list(range(12)), list(range(20)), 12),  # one extends the other: exact
        (list(range(20)), list(range(12)), 12),
        (list(range(12)), list(range(9)) + [99, 10, 11], 8),  # differs in block 2: rounded down
        ([5] + list(range(1, 12)), list(range(12)), 0),
        ([], list(range(4)), 0),
    ],
)
def test_common_prefix_tokens(a, b, expected):
    assert _common_prefix_tokens(a, b, 4) == expected


def test_common_prefix_tokens_across_a_long_prompt():
    a = list(range(4 * 64 * 3))
    b = list(a)
    b[4 * 64 * 2 + 9] = -1  # inside block 2 of the third span
    assert _common_prefix_tokens(a, b, 4) == 4 * 64 * 2 + 8


@pytest.mark.parametrize("lookahead", [0, 1, 3])
def test_a_carried_chain_hashes_like_a_fresh_one(lookahead):
    """A next turn that adopts its previous turn's hashes gets the chain it would have hashed.

    A block's hash also covers the lookahead tokens after it, so only blocks
    whose lookahead lies inside the common prefix are carried.
    """
    previous = list(range(1000, 1000 + 4 * 10))
    current = previous[:30] + list(range(5000, 5000 + 22))

    def chain():
        return keys_module.BlockHashChain(4, prompt_lookahead=lookahead)

    fresh = chain()
    fresh.extend(current)
    turn = chain()
    turn.extend(previous)
    assert turn.seed == fresh.seed

    common = _common_prefix_tokens(current, previous, 4)
    blocks = min(len(turn.hashes), max(0, common - lookahead) // 4)
    assert blocks > 0
    carried = chain()
    carried.adopt_prefix(turn.hashes[:blocks])
    carried.extend(current)
    assert list(carried.hashes) == list(fresh.hashes)
    with pytest.raises(ValueError):
        carried.adopt_prefix(turn.hashes[:1])
