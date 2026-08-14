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

import importlib.util
import os
import subprocess
import sys
import tempfile
import textwrap
import types
import unittest
from pathlib import Path
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parents[3]
BENCHMARK_PATH = REPO_ROOT / "tests" / "integration" / "defs" / "test_infra_dry_run_benchmark.py"

_TORCH_IMPORT_STUB = types.ModuleType("torch")
with mock.patch.dict(sys.modules, {"torch": _TORCH_IMPORT_STUB}):
    _SPEC = importlib.util.spec_from_file_location("test_infra_dry_run_benchmark", BENCHMARK_PATH)
    assert _SPEC is not None and _SPEC.loader is not None
    BENCHMARK = importlib.util.module_from_spec(_SPEC)
    _SPEC.loader.exec_module(BENCHMARK)


class _Scalar:
    def __init__(self, value):
        self._value = value

    def item(self):
        return self._value


class _Tensor:
    def __init__(self, value, *, dtype, device):
        self.value = value
        self.dtype = dtype
        self.device = device

    def all(self):
        return _Scalar(True)


class _Cuda:
    def __init__(self, available=True, count=2):
        self.available = available
        self.count = count
        self.selected = []
        self.synchronized = []

    def is_available(self):
        return self.available

    def device_count(self):
        return self.count

    def set_device(self, device):
        self.selected.append(device.index)

    def synchronize(self, device):
        self.synchronized.append(device.index)


class _Torch:
    float16 = "float16"
    float32 = "float32"

    def __init__(self, *, cuda_available=True, cuda_count=2):
        self.cuda = _Cuda(cuda_available, cuda_count)
        self.devices = []

    def device(self, device_type, index=None):
        device = types.SimpleNamespace(type=device_type, index=index)
        self.devices.append(device)
        return device

    def full(self, _shape, value, *, dtype, device):
        return _Tensor(value, dtype=dtype, device=device)

    def matmul(self, _left, _right):
        device = self.devices[-1]
        dtype = self.float32 if device.type == "cpu" else self.float16
        return _Tensor(4.0, dtype=dtype, device=device)

    def full_like(self, tensor, value):
        return _Tensor(value, dtype=tensor.dtype, device=tensor.device)

    def isfinite(self, tensor):
        return tensor

    def equal(self, left, right):
        return left.value == right.value


class InfraDryRunBenchmarkTest(unittest.TestCase):
    def test_cpu_path_uses_fp32_cpu_matmul(self):
        torch_stub = _Torch()
        with mock.patch.object(BENCHMARK, "torch", torch_stub):
            BENCHMARK._run_cpu()
        self.assertEqual(
            [(device.type, device.index) for device in torch_stub.devices], [("cpu", None)]
        )

    def test_cuda_path_exercises_every_visible_device(self):
        torch_stub = _Torch(cuda_count=3)
        with mock.patch.object(BENCHMARK, "torch", torch_stub):
            BENCHMARK._run_cuda()
        self.assertEqual(torch_stub.cuda.selected, [0, 1, 2])
        self.assertEqual(torch_stub.cuda.synchronized, [0, 1, 2])

    def test_cuda_path_does_not_fall_back_to_cpu(self):
        with mock.patch.object(BENCHMARK, "torch", _Torch(cuda_available=False)):
            with self.assertRaisesRegex(AssertionError, "CUDA is required"):
                BENCHMARK._run_cuda()

    def test_standard_pytest_collection_selects_only_the_requested_context(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / BENCHMARK_PATH.name).write_text(BENCHMARK_PATH.read_text())
            (root / "test_product.py").write_text("def test_product(): pass\n")
            (root / "torch.py").write_text(
                textwrap.dedent(
                    """
                    float32 = "float32"
                    class Device:
                        def __init__(self, kind, index=None):
                            self.type, self.index = kind, index
                    class Scalar:
                        def item(self): return True
                    class Tensor:
                        def __init__(self, value, dtype, device):
                            self.value, self.dtype, self.device = value, dtype, device
                        def all(self): return Scalar()
                    def device(kind, index=None): return Device(kind, index)
                    def full(_shape, value, *, dtype, device): return Tensor(value, dtype, device)
                    def matmul(left, _right): return Tensor(4.0, left.dtype, left.device)
                    def full_like(tensor, value): return Tensor(value, tensor.dtype, tensor.device)
                    def isfinite(tensor): return tensor
                    def equal(left, right): return left.value == right.value
                    """
                )
            )
            (root / "conftest.py").write_text(
                textwrap.dedent(
                    """
                    def pytest_addoption(parser):
                        parser.addoption("--test-list")
                    def pytest_collection_modifyitems(config, items):
                        wanted = {
                            line.strip() for line in open(config.getoption("--test-list"))
                            if line.strip()
                        }
                        selected = [item for item in items if item.nodeid in wanted]
                        config.hook.pytest_deselected(
                            items=[item for item in items if item not in selected]
                        )
                        items[:] = selected
                    """
                )
            )

            dry_list = root / "dry.txt"
            dry_list.write_text(f"{BENCHMARK_PATH.name}::test_infra_dry_run_benchmark\n")
            normal_list = root / "normal.txt"
            normal_list.write_text("test_product.py::test_product\n")
            env = {**os.environ, "stageName": "CPU-Generic-x86-1"}
            for test_list, expected in (
                (dry_list, BENCHMARK_PATH.name),
                (normal_list, "test_product.py"),
            ):
                result = subprocess.run(
                    [sys.executable, "-m", "pytest", f"--test-list={test_list}", "-vv"],
                    cwd=root,
                    env=env,
                    check=True,
                    capture_output=True,
                    text=True,
                )
                self.assertIn(expected, result.stdout)
                self.assertIn("1 passed, 1 deselected", result.stdout)

    def test_positional_nodeid_does_not_import_unrelated_product_test(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / BENCHMARK_PATH.name).write_text(BENCHMARK_PATH.read_text())
            (root / "test_unrelated_product.py").write_text(
                'raise RuntimeError("unrelated product test imported")\n'
            )
            (root / "torch.py").write_text(
                textwrap.dedent(
                    """
                    float32 = "float32"
                    class Device:
                        def __init__(self, kind, index=None):
                            self.type, self.index = kind, index
                    class Scalar:
                        def item(self): return True
                    class Tensor:
                        def __init__(self, value, dtype, device):
                            self.value, self.dtype, self.device = value, dtype, device
                        def all(self): return Scalar()
                    def device(kind, index=None): return Device(kind, index)
                    def full(_shape, value, *, dtype, device): return Tensor(value, dtype, device)
                    def matmul(left, _right): return Tensor(4.0, left.dtype, left.device)
                    def full_like(tensor, value): return Tensor(value, tensor.dtype, tensor.device)
                    def isfinite(tensor): return tensor
                    def equal(left, right): return left.value == right.value
                    """
                )
            )
            (root / "conftest.py").write_text(
                textwrap.dedent(
                    """
                    def pytest_addoption(parser):
                        parser.addoption("--test-list")
                    """
                )
            )

            target = f"{BENCHMARK_PATH.name}::test_infra_dry_run_benchmark"
            dry_list = root / "dry.txt"
            dry_list.write_text(f"{target}\n")
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pytest",
                    "--collect-only",
                    f"--test-list={dry_list}",
                    target,
                    "-q",
                ],
                cwd=root,
                env={**os.environ, "stageName": "CPU-Generic-x86-1"},
                check=True,
                capture_output=True,
                text=True,
            )
            self.assertIn(target, result.stdout)
            self.assertNotIn("test_unrelated_product.py", result.stdout)


if __name__ == "__main__":
    unittest.main()
