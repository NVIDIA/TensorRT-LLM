# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU tests of OpenEngine deployment provisioning and process cleanup."""

import importlib.util
import json
import os
import signal
import stat
import subprocess
import sys
import time
from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.cpu_only
_LAUNCHER = Path(__file__).resolve().parents[3] / "examples/disaggregated/openengine/launch.py"


def test_generated_keys_are_shared_private_and_fresh(tmp_path):
    spec = importlib.util.spec_from_file_location("openengine_launcher", _LAUNCHER)
    launcher = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(launcher)
    sources = [tmp_path / "ctx.yaml", tmp_path / "gen.yaml"]
    originals = [
        "internal_request_auth_key: old-context\nmax_batch_size: 4\n",
        "internal_request_auth_key: old-generation\nmax_batch_size: 8\n",
    ]
    for path, content in zip(sources, originals):
        path.write_text(content)
    keys = []
    for run in range(2):
        directory = tmp_path / str(run)
        directory.mkdir(mode=0o700)
        paths = launcher._write_configs(sources, directory)
        configs = [yaml.safe_load(path.read_text()) for path in paths]
        key = configs[0]["internal_request_auth_key"]
        assert len(bytes.fromhex(key)) == 32
        assert key == configs[1]["internal_request_auth_key"]
        assert [config["max_batch_size"] for config in configs] == [4, 8]
        assert all(stat.S_IMODE(path.stat().st_mode) == 0o600 for path in paths)
        keys.append(key)
    assert keys[0] != keys[1]
    assert [path.read_text() for path in sources] == originals


@pytest.mark.parametrize("stop_mode", ["server_failure", "sigterm"])
def test_launcher_provisions_children_and_cleans_up(tmp_path, stop_mode):
    executable = tmp_path / "trtllm-serve"
    executable.write_text(
        f"#!{sys.executable}\n"
        + """
import json, os, sys, time
from pathlib import Path
import yaml
config_path = Path(sys.argv[sys.argv.index("--config") + 1])
config = yaml.safe_load(config_path.read_text())
report = Path(os.environ["LAUNCHER_TEST_REPORTS"])
role = config_path.stem
temporary_report = report / (role + ".tmp")
temporary_report.write_text(json.dumps({
    "key": config["internal_request_auth_key"], "pid": os.getpid(),
    "config": str(config_path), "mode": config_path.stat().st_mode & 0o777,
    "directory_mode": config_path.parent.stat().st_mode & 0o777,
    "gpus": os.environ["CUDA_VISIBLE_DEVICES"], "args": sys.argv[1:],
}))
temporary_report.replace(report / (role + ".json"))
while True:
    if role == "generation" and (report / "fail").exists():
        sys.exit(7)
    time.sleep(0.05)
"""
    )
    executable.chmod(0o700)
    config = tmp_path / "worker.yaml"
    config.write_text("max_batch_size: 4\n")
    env = dict(
        os.environ, PATH=f"{tmp_path}:{os.environ['PATH']}", LAUNCHER_TEST_REPORTS=str(tmp_path)
    )
    process = subprocess.Popen(
        [
            sys.executable,
            str(_LAUNCHER),
            "test/model",
            "--context-config",
            str(config),
            "--generation-config",
            str(config),
            "--context-gpus",
            "2",
            "--generation-gpus",
            "3",
        ],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        reports = [tmp_path / f"{role}.json" for role in ("context", "generation")]
        deadline = time.monotonic() + 15
        while not all(path.exists() for path in reports):
            assert process.poll() is None, process.communicate()[0]
            assert time.monotonic() < deadline, "Workers did not start"
            time.sleep(0.05)
        records = [json.loads(path.read_text()) for path in reports]
        key = records[0]["key"]
        assert key == records[1]["key"]
        assert [record["gpus"] for record in records] == ["2", "3"]
        for record in records:
            assert record["mode"] == 0o600
            assert record["directory_mode"] == 0o700
            assert key not in " ".join(record["args"])
            assert record["args"][record["args"].index("--grpc-protocol") + 1] == "openengine"
        if stop_mode == "sigterm":
            process.send_signal(signal.SIGTERM)
        else:
            (tmp_path / "fail").touch()
        output, _ = process.communicate(timeout=15)
        assert process.returncode == (128 + signal.SIGTERM if stop_mode == "sigterm" else 7)
        assert key not in output
        for record in records:
            assert not Path(record["config"]).parent.exists()
            with pytest.raises(ProcessLookupError):
                os.kill(record["pid"], 0)
        assert config.read_text() == "max_batch_size: 4\n"
    finally:
        if process.poll() is None:
            process.terminate()
            process.communicate(timeout=40)
