# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib.util
import os
from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.cpu_only
_REPO = Path(__file__).resolve().parents[3]


@pytest.fixture
def launchers(monkeypatch):
    directory = _REPO / "examples/disaggregated/slurm/benchmark"
    modules = []
    for name in ("submit", "submit_dwdp"):
        spec = importlib.util.spec_from_file_location(name, directory / f"{name}.py")
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(__import__("sys").modules, name, module)
        spec.loader.exec_module(module)
        modules.append(module)
    return modules


def test_private_yaml_restricts_existing_file_before_writing(launchers, tmp_path, monkeypatch):
    submit, _ = launchers
    path = tmp_path / "worker.yaml"
    path.write_text("old contents" * 50)
    path.chmod(0o644)
    original_dump = submit.yaml.safe_dump

    def check_mode(config, stream, **kwargs):
        assert os.fstat(stream.fileno()).st_mode & 0o777 == 0o600
        return original_dump(config, stream, **kwargs)

    monkeypatch.setattr(submit.yaml, "safe_dump", check_mode)
    submit.write_private_yaml({"internal_request_auth_key": "secret"}, path)
    assert yaml.safe_load(path.read_text()) == {"internal_request_auth_key": "secret"}
    fresh = tmp_path / "fresh.yaml"
    submit.write_private_yaml({"internal_request_auth_key": "secret"}, fresh)
    assert fresh.stat().st_mode & 0o777 == 0o600


def test_conflicting_launcher_keys_are_rejected(launchers):
    submit, _ = launchers
    with pytest.raises(ValueError, match="must match"):
        submit.resolve_internal_request_auth_key(
            {
                "internal_request_auth_key": "one",
                "worker_config": {"gen": {"internal_request_auth_key": "two"}},
            }
        )
    assert (
        submit.resolve_internal_request_auth_key(
            {
                "worker_config": {"ctx": {"internal_request_auth_key": "shared"}},
            }
        )
        == "shared"
    )


def test_dwdp_embeds_the_same_key_in_all_sections(launchers, tmp_path):
    _, dwdp = launchers
    output = tmp_path / "mpi.yaml"
    worker_config = {"ctx": {"internal_request_auth_key": "old"}, "gen": {}}
    dwdp.generate_mpi_worker_config(
        worker_config,
        {"CTX": {}, "GEN": {}},
        {"model_path": "model"},
        "localhost",
        8000,
        str(output),
        "shared",
    )
    config = yaml.safe_load(output.read_text())
    assert config["internal_request_auth_key"] == "shared"
    for section in ("context_servers", "generation_servers"):
        assert config[section]["internal_request_auth_key"] == "shared"
    assert worker_config["ctx"]["internal_request_auth_key"] == "old"
    assert output.stat().st_mode & 0o777 == 0o600
