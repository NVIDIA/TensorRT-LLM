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
"""Unit tests for the KV cache transceiver harness report.py helpers.

CPU-only — no GPU, MPI, or tensorrt_llm dependency.
"""

import csv
import json
import os
import sys

import pytest

sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(__file__),
        os.pardir,
        os.pardir,
        os.pardir,
        "examples",
        "disaggregated",
        "slurm",
        "cache_transceiver_test",
    ),
)

from report import (  # noqa: E402
    RID_COMBINATION_STRIDE,
    RID_REQLEN_STRIDE,
    _is_kv_data_header,
    _parse_cpp_recv_csvs,
    _parse_proto_info,
    _parse_proto_info_by_case,
    _parse_python_csvs,
    _rank_best_per_combination,
    _read_status,
    _TransportAcc,
    aggregate,
    build_cases,
    decode_rid,
    emit_launch_vars,
    emit_ucx_env,
)


@pytest.fixture
def sample_cfg():
    return {
        "slurm": {
            "partition": "batch",
            "account": "test",
            "job_time": "00:10:00",
            "job_name": "ctt",
        },
        "hardware": {"gpus_per_node": 2},
        "environment": {
            "container_image": "nvcr.io/test:latest",
            "container_mount": "/mnt:/mnt",
            "work_dir": "/tmp/ctt_test",
            "trtllm_repo": "",
            "trtllm_wheel_path": "",
            "build_wheel": False,
            "cuda_architectures": "",
        },
        "test_matrix": {
            "combinations": [
                {"backend": "UCX", "runtime": "CPP"},
                {"backend": "NIXL", "runtime": "CPP"},
                {"backend": "NIXL", "runtime": "PYTHON"},
            ],
            "cache_manager_versions": ["V1", "V2"],
            "request_lengths": [100, 1000],
            "num_requests_per_length": 4,
            "warmup_requests": 1,
        },
        "kv_cache": {
            "num_layers": 32,
            "num_kv_heads": 8,
            "head_dim": 128,
            "tokens_per_block": 32,
            "dtype": "HALF",
            "max_tokens_in_buffer": 16384,
        },
        "parallel": {"ctx_tp": 2, "ctx_pp": 1, "gen_tp": 2, "gen_pp": 1},
        "ucx_env_sweep": [
            {"name": "all", "env": {"UCX_TLS": "all"}},
            {"name": "tcp_only", "env": {"UCX_TLS": "tcp"}},
        ],
        "run": {
            "timeout_per_cell_s": 60,
            "max_sweep_s": 180,
            "capture_proto_info": True,
        },
    }


# ---------------------------------------------------------------------------
# build_cases
# ---------------------------------------------------------------------------
class TestBuildCases:
    def test_basic(self, sample_cfg):
        cases = build_cases(sample_cfg)
        labels = [c["label"] for c in cases]
        assert "UCX/CPP/V1" in labels
        assert "NIXL/CPP/V1" in labels
        assert "NIXL/PYTHON/V1" in labels
        assert "NIXL/PYTHON/V2" in labels
        # V2 + CPP should be skipped
        assert "UCX/CPP/V2" not in labels
        assert "NIXL/CPP/V2" not in labels

    def test_v1_only(self, sample_cfg):
        sample_cfg["test_matrix"]["cache_manager_versions"] = ["V1"]
        cases = build_cases(sample_cfg)
        assert all(c["cache_manager"] == "V1" for c in cases)

    def test_combos_backward_compat(self, sample_cfg):
        combos = sample_cfg["test_matrix"].pop("combinations")
        sample_cfg["test_matrix"]["combos"] = combos
        cases = build_cases(sample_cfg)
        assert len(cases) > 0
        assert cases[0]["backend"] == "UCX"

    def test_v2_python_only(self, sample_cfg):
        cases = build_cases(sample_cfg)
        v2_cases = [c for c in cases if c["cache_manager"] == "V2"]
        assert all(c["runtime"] == "PYTHON" for c in v2_cases)


# ---------------------------------------------------------------------------
# decode_rid
# ---------------------------------------------------------------------------
class TestDecodeRid:
    @pytest.mark.parametrize(
        "ci,li,r",
        [(0, 0, 0), (1, 2, 3), (5, 9, 99), (99, 99, 9999)],
    )
    def test_roundtrip(self, ci, li, r):
        rid = ci * RID_COMBINATION_STRIDE + li * RID_REQLEN_STRIDE + r
        assert decode_rid(rid) == (ci, li, r)

    def test_zero(self):
        assert decode_rid(0) == (0, 0, 0)


# ---------------------------------------------------------------------------
# emit_launch_vars / emit_ucx_env
# ---------------------------------------------------------------------------
class TestEmitHelpers:
    def test_emit_launch_vars(self, sample_cfg, capfd):
        emit_launch_vars(sample_cfg)
        out = capfd.readouterr().out
        assert "N=" in out
        assert "IMAGE=" in out
        assert "NUM_SWEEPS=" in out
        assert "PER_SWEEP_TIMEOUT=" in out

    def test_emit_ucx_env(self, sample_cfg, capfd):
        emit_ucx_env(sample_cfg, 0)
        out = capfd.readouterr().out
        assert "export CTT_SWEEP_NAME=" in out
        assert "export UCX_TLS=" in out

    def test_emit_ucx_env_second_sweep(self, sample_cfg, capfd):
        emit_ucx_env(sample_cfg, 1)
        out = capfd.readouterr().out
        assert "tcp_only" in out


# ---------------------------------------------------------------------------
# CSV parsing
# ---------------------------------------------------------------------------
class TestParseCppRecvCsvs:
    def test_basic(self, tmp_path):
        csv_path = tmp_path / "rank_0_recv.csv"
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["RequestID", "Bandwidth(Gbps)", "Bandwidth(Gbps)"])
            w.writerow([1000000, 80.0, 160.0])  # rid=1M -> ci=1,li=0,r=0
        result = _parse_cpp_recv_csvs(str(tmp_path))
        assert 1000000 in result
        # mean of (80/8, 160/8) = mean(10, 20) = 15 GB/s
        bws = result[1000000]
        assert len(bws) == 1
        assert abs(bws[0] - 15.0) < 0.01

    def test_renamed_pattern(self, tmp_path):
        csv_path = tmp_path / "rank_0_recv__c0.csv"
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["RequestID", "Bandwidth(Gbps)"])
            w.writerow([0, 40.0])
        result = _parse_cpp_recv_csvs(str(tmp_path))
        assert 0 in result
        assert abs(result[0][0] - 5.0) < 0.01  # 40/8

    def test_empty_dir(self, tmp_path):
        assert _parse_cpp_recv_csvs(str(tmp_path)) == {}


class TestParsePythonCsvs:
    def test_basic(self, tmp_path):
        csv_path = tmp_path / "py_abc_0.csv"
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(
                f,
                fieldnames=["unique_rid", "throughput_mbs", "task_type", "other"],
            )
            w.writeheader()
            w.writerow(
                {
                    "unique_rid": "0",
                    "throughput_mbs": "1048.576",
                    "task_type": "Send",
                    "other": "",
                }
            )
            # Recv task should be ignored
            w.writerow(
                {
                    "unique_rid": "0",
                    "throughput_mbs": "2000.0",
                    "task_type": "Recv",
                    "other": "",
                }
            )
        result = _parse_python_csvs(str(tmp_path))
        assert 0 in result
        # 1048.576 MiB/s * 1024^2 / 1e9 ≈ 1.0995 GB/s
        assert len(result[0]) == 1
        assert abs(result[0][0] - 1048.576 * 1024 * 1024 / 1e9) < 0.001

    def test_empty_dir(self, tmp_path):
        assert _parse_python_csvs(str(tmp_path)) == {}


# ---------------------------------------------------------------------------
# Transport parsing
# ---------------------------------------------------------------------------
class TestTransportAcc:
    def test_is_kv_data_header(self):
        assert _is_kv_data_header("cfg#0 ucp_put from 0 to 1 (cuda/cuda)")
        assert _is_kv_data_header("cfg#1 rendezvous remote memory read (cuda)")
        assert not _is_kv_data_header("cfg#2 tagged message (cuda)")
        assert not _is_kv_data_header("cfg#3 active message (host)")
        assert not _is_kv_data_header("cfg#4 ucp_put from 0 to 1 (host)")

    def test_basic_transport(self):
        acc = _TransportAcc()
        acc.feed("cfg#0 ucp_put (cuda/cuda)")
        acc.feed("  | rendezvous zero-copy | cuda_ipc/cuda |")
        ranked = acc.ranked()
        assert "cuda_ipc" in ranked

    def test_sw_emul(self):
        acc = _TransportAcc()
        acc.feed("cfg#0 ucp_put (cuda/cuda)")
        acc.feed("  | software emulation | tcp/eth0 |")
        ranked = acc.ranked()
        assert "tcp(sw-emul)" in ranked

    def test_empty(self):
        acc = _TransportAcc()
        assert acc.ranked() == []

    def test_kv_only_excludes_control_transport(self):
        acc = _TransportAcc()
        acc.feed("cfg#0 tagged message (cuda/cuda)")
        acc.feed("  | eager short | self/memory |")

        assert acc.ranked() == ["self"]
        assert acc.ranked(kv_only=True) == []


class TestParseProtoInfo:
    def _write_log(self, path, lines):
        with open(path, "w") as f:
            f.write("\n".join(lines) + "\n")

    def test_basic(self, tmp_path):
        self._write_log(
            tmp_path / "rank0.log",
            [
                "cfg#0 ucp_put (cuda/cuda)",
                "  | rendezvous zero-copy | rc_mlx5/mlx5_0:1 |",
            ],
        )
        result = _parse_proto_info(str(tmp_path / "rank*.log"))
        assert "rc_mlx5" in result

    def test_kv_only_ignores_control_traffic_and_env_echo(self, tmp_path):
        self._write_log(
            tmp_path / "rank0.log",
            [
                "export UCX_TLS=all,self,rc_mlx5",
                "cfg#0 tagged message (cuda/cuda)",
                "  | eager short | self/memory |",
            ],
        )

        result = _parse_proto_info(str(tmp_path / "rank*.log"), kv_only=True)

        assert result == []

    def test_kv_only_parses_weighted_multi_lane_transport(self, tmp_path):
        self._write_log(
            tmp_path / "rank0.log",
            [
                "cfg#0 ucp_put (cuda/cuda)",
                "  | rendezvous zero-copy | 50% on rc_mlx5/mlx5_0:1 and 50% on rc_mlx5/mlx5_1:1 |",
            ],
        )

        result = _parse_proto_info(str(tmp_path / "rank*.log"), kv_only=True)

        assert result == ["rc_mlx5"]

    def test_no_files(self, tmp_path):
        result = _parse_proto_info(str(tmp_path / "rank*.log"))
        assert result == []

    def test_by_case(self, tmp_path):
        self._write_log(
            tmp_path / "rank0.log",
            [
                "[CTT_CASE_BEGIN] ci=0 label=UCX/CPP/V1",
                "cfg#0 ucp_put (cuda/cuda)",
                "  | rendezvous | cuda_ipc/cuda |",
                "[CTT_CASE_BEGIN] ci=1 label=NIXL/PYTHON/V1",
                "cfg#0 ucp_put (cuda/cuda)",
                "  | rendezvous | rc_mlx5/cuda |",
            ],
        )
        result = _parse_proto_info_by_case(str(tmp_path / "rank*.log"))
        assert "cuda_ipc" in result[0]
        assert "rc_mlx5" in result[1]

    def test_by_case_kv_only_ignores_control_traffic(self, tmp_path):
        self._write_log(
            tmp_path / "rank0.log",
            [
                "[CTT_CASE_BEGIN] ci=0 label=NIXL/PYTHON/V1",
                "export UCX_TLS=all,self",
                "cfg#0 tagged message (cuda/cuda)",
                "  | eager short | self/memory |",
            ],
        )

        result = _parse_proto_info_by_case(str(tmp_path / "rank*.log"), kv_only=True)

        assert result == {0: []}


# ---------------------------------------------------------------------------
# Status merging
# ---------------------------------------------------------------------------
class TestReadStatus:
    def test_merge_worst_severity(self, tmp_path):
        status_dir = tmp_path / "status"
        status_dir.mkdir()
        with open(status_dir / "sweep0_ctx.jsonl", "w") as f:
            f.write(json.dumps({"combination_idx": 0, "reqlen_idx": 0, "status": "PASS"}) + "\n")
        with open(status_dir / "sweep0_gen.jsonl", "w") as f:
            f.write(
                json.dumps(
                    {
                        "combination_idx": 0,
                        "reqlen_idx": 0,
                        "status": "MISMATCH",
                        "reason": "data mismatch",
                    }
                )
                + "\n"
            )
        result = _read_status(str(tmp_path), 0)
        assert result[(0, 0)]["status"] == "MISMATCH"

    def test_missing_files(self, tmp_path):
        (tmp_path / "status").mkdir()
        result = _read_status(str(tmp_path), 0)
        assert result == {}


# ---------------------------------------------------------------------------
# Best per combination
# ---------------------------------------------------------------------------
class TestRankBest:
    def test_picks_highest_bw(self):
        by_combination = [
            {
                "combination": "NIXL/PYTHON/V1",
                "sweeps": [
                    {
                        "sweep": "all",
                        "status": "PASS",
                        "per_gpu_BW_GBps": 10.0,
                        "aggregate_BW_GBps": 20.0,
                        "env": {"UCX_TLS": "all"},
                        "selected_transport": "cuda_ipc",
                    },
                    {
                        "sweep": "tcp",
                        "status": "PASS",
                        "per_gpu_BW_GBps": 5.0,
                        "aggregate_BW_GBps": 10.0,
                        "env": {"UCX_TLS": "tcp"},
                        "selected_transport": "tcp",
                    },
                ],
            }
        ]
        best = _rank_best_per_combination(by_combination)
        assert len(best) == 1
        assert best[0]["best_sweep"] == "all"
        assert best[0]["per_gpu_BW_GBps"] == 10.0

    def test_skips_non_pass(self):
        by_combination = [
            {
                "combination": "UCX/CPP/V1",
                "sweeps": [
                    {
                        "sweep": "all",
                        "status": "TIMEOUT",
                        "per_gpu_BW_GBps": None,
                        "aggregate_BW_GBps": None,
                        "env": {"UCX_TLS": "all"},
                        "selected_transport": "",
                    }
                ],
            }
        ]
        best = _rank_best_per_combination(by_combination)
        assert best == []


# ---------------------------------------------------------------------------
# Full aggregate
# ---------------------------------------------------------------------------
class TestAggregate:
    def _setup_work_dir(self, tmp_path, sample_cfg):
        work = tmp_path / "work"
        sample_cfg["environment"]["work_dir"] = str(work)
        # Use only 1 sweep and 1 combination for simplicity
        sample_cfg["test_matrix"]["combinations"] = [{"backend": "NIXL", "runtime": "PYTHON"}]
        sample_cfg["test_matrix"]["cache_manager_versions"] = ["V1"]
        sample_cfg["ucx_env_sweep"] = [{"name": "default", "env": {"UCX_TLS": "all"}}]
        sample_cfg["test_matrix"]["request_lengths"] = [100, 1000]
        sample_cfg["test_matrix"]["warmup_requests"] = 1
        sample_cfg["test_matrix"]["num_requests_per_length"] = 2

        # Create CSV for gen side (C++ recv)
        gen_csv = work / "csv" / "0" / "gen"
        gen_csv.mkdir(parents=True)
        with open(gen_csv / "rank_0_recv.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["RequestID", "Bandwidth(Gbps)"])
            # warmup rid (r=0) should be excluded
            w.writerow([0 * RID_COMBINATION_STRIDE + 0 * RID_REQLEN_STRIDE + 0, 80.0])
            # timed rids (r=1,2) for reqlen_idx=1 (longest=1000)
            w.writerow([0 * RID_COMBINATION_STRIDE + 1 * RID_REQLEN_STRIDE + 1, 160.0])
            w.writerow([0 * RID_COMBINATION_STRIDE + 1 * RID_REQLEN_STRIDE + 2, 200.0])

        # Create empty ctx CSV dir for Python parsing
        ctx_csv = work / "csv" / "0" / "ctx"
        ctx_csv.mkdir(parents=True)

        # Create logs dir (empty)
        (work / "logs").mkdir(parents=True)

        # Status
        status = work / "status"
        status.mkdir(parents=True)
        with open(status / "sweep0_gen.jsonl", "w") as f:
            for li in range(2):
                f.write(
                    json.dumps(
                        {
                            "combination_idx": 0,
                            "reqlen_idx": li,
                            "status": "PASS",
                        }
                    )
                    + "\n"
                )

        return work

    def test_aggregate_produces_json(self, tmp_path, sample_cfg):
        work = self._setup_work_dir(tmp_path, sample_cfg)
        out_path = str(work / "results.json")
        aggregate(sample_cfg, out_path)

        assert os.path.exists(out_path)
        with open(out_path) as f:
            data = json.load(f)
        assert "by_combination" in data
        assert data["req_len"] == 1000
        assert len(data["by_combination"]) == 1
        assert data["by_combination"][0]["combination"] == "NIXL/PYTHON/V1"

        # Best file should also be written
        best_path = str(work / "results.best.json")
        assert os.path.exists(best_path)

    def test_aggregate_requires_kv_data_transport(self, tmp_path, sample_cfg):
        work = self._setup_work_dir(tmp_path, sample_cfg)
        with open(work / "logs" / "sweep0_gen_rank0.log", "w") as f:
            f.write(
                "[CTT_CASE_BEGIN] ci=0 label=NIXL/PYTHON/V1\n"
                "export UCX_TLS=all,self\n"
                "cfg#0 tagged message (cuda/cuda)\n"
                "  | eager short | self/memory |\n"
            )

        out_path = str(work / "results.json")
        results = aggregate(sample_cfg, out_path, require_kv_transport=True)

        sweep = results["by_combination"][0]["sweeps"][0]
        assert sweep["selected_transport"] == ""
