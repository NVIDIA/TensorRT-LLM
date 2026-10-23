# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU coverage for the perf-sanity precheck's shared RankInfo key."""

import importlib.util
import stat
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

pytestmark = pytest.mark.cpu_only


@pytest.fixture
def precheck(monkeypatch):
    directory = (
        Path(__file__).resolve().parents[2] / "scripts/perf-sanity/cache_transceiver_precheck"
    )
    monkeypatch.syspath_prepend(str(directory))
    spec = importlib.util.spec_from_file_location(
        "ct_precheck_runner", directory / "run_precheck.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _runner(precheck, tmp_path, role="ctx", index=0):
    runner = precheck.PrecheckRunner.__new__(precheck.PrecheckRunner)
    runner.is_leader = True
    runner.is_ctx = role == "ctx"
    runner.server_idx = index
    runner.work_dir = str(tmp_path)
    runner.plan = {"setup_timeout_s": 1}
    runner.comm = Mock()
    runner.comm.bcast.side_effect = lambda result, root: result
    return runner


def test_precheck_key_shared_across_instances_and_mpi_ranks(precheck, tmp_path, monkeypatch):
    ctx0 = _runner(precheck, tmp_path)
    key = ctx0._provision_rank_info_auth_key()
    assert len(key) == 32
    key_file = tmp_path / "rendezvous/rank_info_auth.addr"
    assert stat.S_IMODE(key_file.stat().st_mode) == 0o600
    for role, index in [("ctx", 1), ("gen", 0), ("gen", 1)]:
        runner = _runner(precheck, tmp_path, role, index)
        assert runner._provision_rank_info_auth_key() == key
        runner.comm.bcast.assert_called_once_with((key, None), root=0)
    nonleader = _runner(precheck, tmp_path, "gen")
    nonleader.is_leader = False
    nonleader.comm.bcast.side_effect = None
    nonleader.comm.bcast.return_value = (key, None)
    with monkeypatch.context() as patch:
        read = Mock(side_effect=AssertionError("Nonleaders must use MPI"))
        patch.setattr(precheck, "wait_for_addr", read)
        assert nonleader._provision_rank_info_auth_key() == key
        nonleader.comm.bcast.assert_called_once_with(None, root=0)
        read.assert_not_called()
    assert ctx0._provision_rank_info_auth_key() != key


def test_precheck_provisioning_error_reaches_all_ranks(precheck, tmp_path, monkeypatch):
    runner = _runner(precheck, tmp_path, "gen")
    monkeypatch.setattr(precheck, "wait_for_addr", Mock(side_effect=precheck._Timeout("timed out")))
    with pytest.raises(RuntimeError, match="RankInfo key provisioning failed"):
        runner._provision_rank_info_auth_key()
    result = runner.comm.bcast.call_args.args[0]
    assert result[0] is None
    runner.is_leader = False
    runner.comm.bcast.side_effect = None
    runner.comm.bcast.return_value = result
    with pytest.raises(RuntimeError, match="RankInfo key provisioning failed"):
        runner._provision_rank_info_auth_key()


def test_precheck_sets_key_before_transceiver_creation(precheck, tmp_path, monkeypatch):
    runner = _runner(precheck, tmp_path)
    config = SimpleNamespace(transceiver_runtime="PYTHON")
    runner._resolve_transfer_config = Mock(return_value=config)
    runner._consensus_error = Mock(return_value=None)
    runner.transfer_contract = {}
    runner.comm.allgather.return_value = [{}]
    runner.rank = 0
    runner.plan["gpus_per_node"] = 1
    runner.side = {
        "parallel": {"world_size": 1, "tp": 1, "pp": 1, "cp": 1, "enable_attention_dp": False}
    }
    runner.csv_dir = str(tmp_path / "csv")
    runner.use_v2 = False
    api = Mock()

    def create_transceiver(*args):
        assert args[-1] is config
        assert len(config._rank_info_auth_key) == 32
        return Mock()

    api.create_kv_cache_transceiver.side_effect = create_transceiver
    monkeypatch.setattr(precheck, "load_internal_apis", lambda: api)
    monkeypatch.setattr(precheck, "build_kv_cache_manager", Mock())
    monkeypatch.setenv("TRTLLM_KVCACHE_TIME_OUTPUT_PATH", "")
    runner.setup({"is_mla": False}, max_req_len=16)
    api.create_kv_cache_transceiver.assert_called_once()
