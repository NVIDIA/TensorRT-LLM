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

import builtins
import importlib.util
import json
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
SCRIPT_PATH = REPO_ROOT / "jenkins" / "scripts" / "infra_dry_run_benchmark.py"
SPEC = importlib.util.spec_from_file_location("infra_dry_run_benchmark", SCRIPT_PATH)
BENCHMARK = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(BENCHMARK)


class _Scalar:
    def __init__(self, value):
        self.value = value

    def item(self):
        return self.value


class _Output:
    def __init__(self, *, finite=True, numel=16):
        self.finite = finite
        self._numel = numel

    def numel(self):
        return self._numel

    def all(self):
        return _Scalar(self.finite)

    def float(self):
        return self

    def sum(self):
        return _Scalar(128.0)


class _Cuda:
    def __init__(self, available=True):
        self.available = available
        self.selected = None
        self.synchronized = None

    def is_available(self):
        return self.available

    def device_count(self):
        return 1

    def set_device(self, device):
        self.selected = device

    def manual_seed_all(self, _seed):
        return None

    def synchronize(self, device):
        self.synchronized = device


class _Probe:
    def __init__(self, values):
        self.values = values

    def cpu(self):
        return self

    def tolist(self):
        return self.values


class _Distributed:
    def __init__(self, remote):
        self.remote = remote
        self.initialized = False
        self.destroyed = False
        self.gather_calls = 0

    def is_available(self):
        return True

    def is_nccl_available(self):
        return True

    def init_process_group(self, **kwargs):
        assert kwargs["backend"] == "nccl"
        assert kwargs["init_method"] == "env://"
        self.initialized = True

    def is_initialized(self):
        return self.initialized

    def all_gather(self, gathered, local):
        self.gather_calls += 1
        gathered[0].values = local.values
        gathered[1].values = [
            float(self.remote["rank"]),
            float(self.remote["world_size"]),
            float(self.remote["status"] == "passed"),
            float(self.remote["checksum"]),
        ]

    def destroy_process_group(self):
        self.destroyed = True
        self.initialized = False


class _Torch:
    float16 = "float16"
    float32 = "float32"
    float64 = "float64"

    def __init__(self, *, cuda=True, finite=True, numel=16, remote=None):
        self.cuda = _Cuda(cuda)
        self.output = _Output(finite=finite, numel=numel)
        self.distributed = _Distributed(remote) if remote else SimpleNamespace()

    def manual_seed(self, _seed):
        return None

    def full(self, *_args, **_kwargs):
        return object()

    def matmul(self, _left, _right):
        return self.output

    def isfinite(self, output):
        return output

    def tensor(self, values, **_kwargs):
        return _Probe(values)

    def empty_like(self, probe):
        return _Probe([0.0] * len(probe.values))


TRTLLM = SimpleNamespace(__version__="1.2.3", __file__="/installed/tensorrt_llm/__init__.py")


def _run(output_dir, torch, **kwargs):
    with mock.patch.object(BENCHMARK, "_load_runtime_modules", return_value=(TRTLLM, torch)):
        return BENCHMARK.run_benchmark(output_dir, **kwargs)


def _read_outputs(output_dir):
    rank = json.loads((output_dir / "infra_dry_run_rank_0.json").read_text())
    manifest = json.loads((output_dir / "infra_dry_run_manifest.json").read_text())
    junit = ET.parse(output_dir / "results-infra_dry_run.xml")
    return rank, manifest, junit


def _result(rank, *, world_size=2, status="passed", checksum=128.0):
    return {
        "rank": rank,
        "world_size": world_size,
        "status": status,
        "checksum": checksum,
        "error": "" if status == "passed" else "CUDA work failed",
    }


class InfraDryRunBenchmarkTest(unittest.TestCase):
    def test_runtime_loader_performs_real_package_imports(self):
        imported = []
        real_import = builtins.__import__
        modules = {"tensorrt_llm": SimpleNamespace(), "torch": SimpleNamespace()}

        def record_import(name, *args, **kwargs):
            if name in modules:
                imported.append(name)
                return modules[name]
            return real_import(name, *args, **kwargs)

        with mock.patch.object(builtins, "__import__", side_effect=record_import):
            self.assertEqual(
                BENCHMARK._load_runtime_modules(),
                (modules["tensorrt_llm"], modules["torch"]),
            )
        self.assertCountEqual(imported, ["tensorrt_llm", "torch"])

    def test_single_rank_success_writes_rank_manifest_and_junit(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            torch = _Torch()
            result = _run(
                output_dir,
                torch,
                stage="Single-GPU",
                commit="deadbeef",
                environ={},
            )
            rank, manifest, junit = _read_outputs(output_dir)

        self.assertEqual(result["overall_status"], "passed")
        self.assertEqual(rank["tensorrt_llm_module"], TRTLLM.__file__)
        self.assertEqual(manifest["product_tests_executed"], 0)
        self.assertEqual((manifest["stage"], manifest["commit"]), ("Single-GPU", "deadbeef"))
        self.assertEqual(manifest["observed_ranks"], [0])
        self.assertEqual(
            junit.find(".//testcase").attrib["name"],
            "infra_dry_run_rank_0_cuda_matmul",
        )
        self.assertIsNone(junit.find(".//failure"))
        self.assertEqual((torch.cuda.selected, torch.cuda.synchronized), (0, 0))

    def test_cuda_failures_return_nonzero_and_write_failure_junit(self):
        scenarios = [
            (_Torch(cuda=False), "CUDA is required"),
            (_Torch(finite=False), "non-finite values"),
            (_Torch(numel=0), "empty tensor"),
        ]
        for torch, expected in scenarios:
            with self.subTest(expected=expected), tempfile.TemporaryDirectory() as temp_dir:
                output_dir = Path(temp_dir)
                with self.assertRaises(BENCHMARK.BenchmarkError):
                    _run(output_dir, torch, environ={})
                rank, manifest, junit = _read_outputs(output_dir)
                self.assertIn(expected, rank["error"])
                self.assertEqual(manifest["status"], "failed")
                self.assertIsNotNone(junit.find(".//failure"))

    def test_cpu_mode_runs_without_cuda_and_writes_cpu_metadata(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            result = _run(
                output_dir,
                _Torch(cuda=False),
                stage="CPU-Generic-x86-1",
                device_type="cpu",
                environ={},
            )
            rank, manifest, junit = _read_outputs(output_dir)

        self.assertEqual(result["overall_status"], "passed")
        self.assertEqual(rank["cpu"]["device"], "cpu")
        self.assertEqual(manifest["device_type"], "cpu")
        self.assertEqual(manifest["distributed_backend"], "none")
        self.assertEqual(manifest["product_tests_executed"], 0)
        self.assertIsNone(result.get("cuda"))
        self.assertEqual(
            junit.find(".//testcase").attrib["name"],
            "infra_dry_run_rank_0_cpu_matmul",
        )
        self.assertIsNone(junit.find(".//failure"))

    def test_import_failure_writes_failure_artifacts(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            with mock.patch.object(
                BENCHMARK,
                "_load_runtime_modules",
                side_effect=ImportError("tensorrt_llm is not installed"),
            ):
                with self.assertRaises(BENCHMARK.BenchmarkError):
                    BENCHMARK.run_benchmark(output_dir, environ={})
            rank, manifest, junit = _read_outputs(output_dir)

        self.assertIn("is not installed", rank["error"])
        self.assertEqual(manifest["status"], "failed")
        self.assertIsNotNone(junit.find(".//failure"))

    def test_multi_rank_uses_one_nccl_gather_and_cleans_up(self):
        remote = _result(1)
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            torch = _Torch(remote=remote)
            result = _run(
                output_dir,
                torch,
                stage="Multi-GPU",
                environ={"RANK": "0", "LOCAL_RANK": "0", "WORLD_SIZE": "2"},
            )
            _, manifest, junit = _read_outputs(output_dir)

        self.assertEqual(result["overall_status"], "passed")
        self.assertEqual(manifest["observed_ranks"], [0, 1])
        self.assertEqual(len(junit.findall(".//testcase")), 3)
        self.assertEqual(torch.distributed.gather_calls, 1)
        self.assertTrue(torch.distributed.destroyed)

    def test_remote_rank_failure_makes_process_fail_and_cleans_up(self):
        remote = _result(1, status="failed", checksum=0.0)
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            torch = _Torch(remote=remote)
            with self.assertRaises(BENCHMARK.BenchmarkError):
                _run(
                    output_dir,
                    torch,
                    environ={"RANK": "0", "LOCAL_RANK": "0", "WORLD_SIZE": "2"},
                )
            rank, manifest, junit = _read_outputs(output_dir)

        self.assertEqual(rank["overall_status"], "failed")
        self.assertEqual(manifest["status"], "failed")
        self.assertIsNotNone(junit.find(".//failure"))
        self.assertEqual(torch.distributed.gather_calls, 1)
        self.assertTrue(torch.distributed.destroyed)

    def test_manifest_rejects_missing_mismatched_and_inconsistent_ranks(self):
        scenarios = [
            ([_result(0)], "observed ranks [0] do not match expected [0, 1]"),
            (
                [_result(0), _result(1, world_size=3)],
                "rank results contain a world-size mismatch",
            ),
            (
                [_result(0), _result(1, checksum=256.0)],
                "rank results contain inconsistent CUDA checksums",
            ),
        ]
        for summaries, expected in scenarios:
            with self.subTest(expected=expected), tempfile.TemporaryDirectory() as temp_dir:
                output_dir = Path(temp_dir)
                result = {
                    "world_size": 2,
                    "stage": "Multi-GPU",
                    "commit": "abc123",
                }
                errors = BENCHMARK._validate(summaries, 2)
                manifest = BENCHMARK._write_reports(output_dir, result, summaries, errors)
                junit = ET.parse(output_dir / "results-infra_dry_run.xml")
                self.assertIn(expected, errors)
                self.assertEqual(manifest["status"], "failed")
                self.assertIsNotNone(junit.find(".//failure"))


if __name__ == "__main__":
    unittest.main()
