# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib.util
import stat
import sys
from pathlib import Path

import pytest
import yaml

_REPO_ROOT = Path(__file__).resolve().parents[3]
_BENCHMARK_DIR = _REPO_ROOT / "examples" / "disaggregated" / "slurm" / "benchmark"


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


SUBMIT = _load_module("submit", _BENCHMARK_DIR / "submit.py")
SUBMIT_DWDP = _load_module("submit_dwdp", _BENCHMARK_DIR / "submit_dwdp.py")
sys.modules.pop("submit", None)
sys.modules.pop("submit_dwdp", None)

pytestmark = pytest.mark.cpu_only


def _launcher_config(**overrides):
    config = {
        "worker_config": {
            "ctx": {},
            "gen": {},
        }
    }
    config.update(overrides)
    return config


def test_launcher_generates_internal_request_auth_key(monkeypatch):
    monkeypatch.setattr(SUBMIT.secrets, "token_hex", lambda size: "generated-key")

    assert SUBMIT.resolve_internal_request_auth_key(_launcher_config()) == "generated-key"


@pytest.mark.parametrize(
    "location",
    [
        ("internal_request_auth_key",),
        ("server_config_extra", "internal_request_auth_key"),
        ("worker_config", "ctx", "internal_request_auth_key"),
        ("worker_config", "gen", "internal_request_auth_key"),
    ],
)
def test_launcher_preserves_configured_internal_request_auth_key(location):
    config = _launcher_config()
    target = config
    for part in location[:-1]:
        target = target.setdefault(part, {})
    target[location[-1]] = "configured-key"

    assert SUBMIT.resolve_internal_request_auth_key(config) == "configured-key"


def test_launcher_rejects_mismatched_internal_request_auth_keys():
    config = _launcher_config(internal_request_auth_key="proxy-key")
    config["worker_config"]["ctx"]["internal_request_auth_key"] = "worker-key"

    with pytest.raises(ValueError, match="must match"):
        SUBMIT.resolve_internal_request_auth_key(config)


@pytest.mark.parametrize("invalid_key", ["", 123, False])
def test_launcher_rejects_invalid_internal_request_auth_key(invalid_key):
    config = _launcher_config(internal_request_auth_key=invalid_key)

    with pytest.raises(ValueError, match="non-empty string"):
        SUBMIT.resolve_internal_request_auth_key(config)


def test_worker_config_receives_shared_key_without_mutating_input(tmp_path):
    worker_config = {"cache_transceiver_config": {"backend": "NIXL"}}
    output_path = tmp_path / "worker.yaml"

    SUBMIT.save_worker_config(worker_config, output_path, "shared-key")

    assert "internal_request_auth_key" not in worker_config
    assert yaml.safe_load(output_path.read_text())["internal_request_auth_key"] == "shared-key"
    assert stat.S_IMODE(output_path.stat().st_mode) == 0o600


def test_dwdp_config_receives_shared_key(tmp_path):
    output_path = tmp_path / "mpi-worker.yaml"
    allocations = {
        "CTX": {0: {"nodes": {"ctx-host": [0]}, "port": 8001}},
        "GEN": {0: {"nodes": {"gen-host": [0]}, "port": 8002}},
    }

    SUBMIT_DWDP.generate_mpi_worker_config(
        {"ctx": {}, "gen": {}},
        allocations,
        {"model_path": "test-model"},
        "proxy-host",
        8000,
        output_path,
        "shared-key",
    )

    generated = yaml.safe_load(output_path.read_text())
    assert generated["internal_request_auth_key"] == "shared-key"
    assert generated["context_servers"]["urls"] == ["ctx-host:8001"]
    assert generated["generation_servers"]["urls"] == ["gen-host:8002"]
    assert stat.S_IMODE(output_path.stat().st_mode) == 0o600
