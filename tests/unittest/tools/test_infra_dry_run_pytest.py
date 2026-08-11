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
import pickle
import signal
import subprocess
import sys
import tempfile
import textwrap
import threading
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
BENCHMARK_PATH = REPO_ROOT / "tests" / "integration" / "defs" / "infra_dry_run_benchmark.py"

_IMPORT_TORCH = types.ModuleType("torch")
with mock.patch.dict(sys.modules, {"torch": _IMPORT_TORCH}):
    # Pytest's default import mode treats defs/ as a package; exercise the
    # package-qualified controller name rather than the worker's top-level name.
    SPEC = importlib.util.spec_from_file_location("defs.infra_dry_run_benchmark", BENCHMARK_PATH)
    BENCHMARK = importlib.util.module_from_spec(SPEC)
    assert SPEC.loader is not None
    SPEC.loader.exec_module(BENCHMARK)


class _Scalar:
    def __init__(self, value):
        self.value = value

    def item(self):
        return self.value


class _Tensor:
    def __init__(self, values, *, device="cpu", dtype="float32"):
        self.values = values
        self.device = SimpleNamespace(type=device)
        self.dtype = dtype

    def all(self):
        return _Scalar(True)

    def cpu(self):
        return self

    def tolist(self):
        return list(self.values)

    def item(self):
        return self.values


class _CpuTorch:
    float32 = "float32"

    def __init__(self):
        self.seed = None
        self.matmul_calls = 0

    def manual_seed(self, seed):
        self.seed = seed

    def full(self, _shape, value, *, dtype, device):
        return _Tensor(value, device=device, dtype=dtype)

    def matmul(self, _left, _right):
        self.matmul_calls += 1
        return _Tensor(4.0)

    def full_like(self, _tensor, value):
        return _Tensor(value)

    def isfinite(self, tensor):
        return tensor

    def equal(self, left, right):
        return left.values == right.values


class InfraDryRunPytestTest(unittest.TestCase):
    def test_explicit_module_and_test_list_execute_but_normal_collection_ignores_it(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            benchmark = root / BENCHMARK_PATH.name
            benchmark.write_text(BENCHMARK_PATH.read_text())
            (root / "torch.py").write_text(
                textwrap.dedent(
                    """
                    float32 = "float32"

                    class Tensor:
                        def __init__(self, value, dtype="float32", device="cpu"):
                            self.value = value
                            self.dtype = dtype
                            self.device = type("Device", (), {"type": device})()
                        def all(self): return self
                        def item(self): return self.value

                    def manual_seed(_seed): pass
                    def full(_shape, value, *, dtype, device):
                        return Tensor(value, dtype, device)
                    def matmul(_left, _right): return Tensor(4.0)
                    def full_like(_tensor, value): return Tensor(value)
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
                        parser.addoption("--test-prefix")

                    def pytest_collection_modifyitems(config, items):
                        prefix = config.getoption("--test-prefix")
                        if prefix:
                            for item in items:
                                item._nodeid = f"{prefix}/{item.nodeid}"
                        test_list = config.getoption("--test-list")
                        if not test_list:
                            return
                        wanted = {
                            f"{prefix}/{line.strip()}" if prefix else line.strip()
                            for line in open(test_list)
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
            test_list = root / "infra_dry_run.txt"
            test_list.write_text("infra_dry_run_benchmark.py::test_infra_dry_run_benchmark\n")
            (root / "test_normal.py").write_text("def test_normal(): pass\n")
            validation_list = root / "all_l0.txt"
            validation_list.write_text(
                "infra_dry_run_benchmark.py::test_infra_dry_run_benchmark\n"
                "test_normal.py::test_normal\n"
            )
            env = {**os.environ, "stageName": "CPU-Validation"}
            explicit = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pytest",
                    "-q",
                    f"--test-list={test_list}",
                    "--test-prefix=CPU-Validation",
                    str(benchmark),
                ],
                cwd=root,
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                check=False,
            )
            validation = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pytest",
                    "-q",
                    "--collect-only",
                    "-o",
                    "python_files=test_*.py *_test.py infra_dry_run_benchmark.py",
                    f"--test-list={validation_list}",
                ],
                cwd=root,
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                check=False,
            )
            normal = subprocess.run(
                [sys.executable, "-m", "pytest", "--collect-only", "-q", str(root)],
                cwd=root,
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                check=False,
            )

        self.assertEqual(explicit.returncode, 0, explicit.stdout)
        self.assertIn("1 passed", explicit.stdout)
        self.assertEqual(validation.returncode, 0, validation.stdout)
        self.assertIn("infra_dry_run_benchmark.py::test_infra_dry_run_benchmark", validation.stdout)
        self.assertIn("test_normal.py::test_normal", validation.stdout)
        self.assertIn("2 tests collected", validation.stdout)
        self.assertEqual(normal.returncode, 0, normal.stdout)
        self.assertIn("test_normal.py::test_normal", normal.stdout)
        self.assertNotIn(BENCHMARK_PATH.name, normal.stdout)

    def test_bounded_wait_interrupts_and_restores_the_signal_handler(self):
        previous_handler = signal.getsignal(signal.SIGALRM)
        with self.assertRaisesRegex(TimeoutError, "exceeded 30 seconds"):
            with BENCHMARK._bounded_wait(30):
                signal.raise_signal(signal.SIGALRM)
        self.assertIs(signal.getsignal(signal.SIGALRM), previous_handler)

    def test_bounded_wait_rejects_non_main_threads_and_existing_timers(self):
        errors = []

        def enter_wait():
            try:
                with BENCHMARK._bounded_wait(30):
                    pass
            except BaseException as error:
                errors.append(error)

        thread = threading.Thread(target=enter_wait)
        thread.start()
        thread.join()
        self.assertEqual(len(errors), 1)
        self.assertRegex(str(errors[0]), "requires the main thread")

        with (
            mock.patch.object(BENCHMARK.signal, "getitimer", return_value=(1.0, 0.0)),
            self.assertRaisesRegex(RuntimeError, "cannot replace an active ITIMER_REAL"),
        ):
            with BENCHMARK._bounded_wait(30):
                pass

    def test_cpu_path_is_explicit_deterministic_fp32(self):
        torch_module = _CpuTorch()
        BENCHMARK._run_cpu(torch_module)
        self.assertEqual(torch_module.seed, 0)
        self.assertEqual(torch_module.matmul_calls, 1)

    def test_cpu_stage_routes_through_the_pytest_controller(self):
        with (
            mock.patch.dict(os.environ, {"stageName": "CPU-Generic-x86-1"}, clear=True),
            mock.patch.object(BENCHMARK, "_run_cpu") as run_cpu,
        ):
            BENCHMARK.test_infra_dry_run_benchmark()
        run_cpu.assert_called_once_with()

    def test_gpu_stage_never_falls_back_to_cpu(self):
        fake_torch = SimpleNamespace(
            cuda=SimpleNamespace(is_available=lambda: False, device_count=lambda: 0)
        )
        with (
            mock.patch.dict(os.environ, {"stageName": "A10-GPU"}, clear=True),
            mock.patch.object(BENCHMARK, "torch", fake_torch),
            mock.patch.object(BENCHMARK, "_run_cpu") as run_cpu,
            self.assertRaisesRegex(RuntimeError, "CUDA is required"),
        ):
            BENCHMARK.test_infra_dry_run_benchmark()
        run_cpu.assert_not_called()

    def test_rank_environment_must_be_complete_and_in_range(self):
        self.assertIsNone(BENCHMARK._external_rank_context({}))
        with self.assertRaisesRegex(RuntimeError, "missing"):
            BENCHMARK._external_rank_context({"RANK": "0"})
        with self.assertRaisesRegex(RuntimeError, "outside WORLD_SIZE"):
            BENCHMARK._external_rank_context({"RANK": "2", "LOCAL_RANK": "0", "WORLD_SIZE": "2"})
        self.assertEqual(
            BENCHMARK._external_rank_context({"RANK": "1", "LOCAL_RANK": "1", "WORLD_SIZE": "2"}),
            (1, 1, 2),
        )

    def test_rank_summary_validation_rejects_missing_or_inconsistent_ranks(self):
        BENCHMARK._validate_rank_summaries([[0.0, 2.0, 10.0], [1.0, 2.0, 10.0]], 2)
        with self.assertRaisesRegex(RuntimeError, "observed ranks"):
            BENCHMARK._validate_rank_summaries([[0.0, 2.0, 10.0]], 2)
        with self.assertRaisesRegex(RuntimeError, "world sizes"):
            BENCHMARK._validate_rank_summaries([[0.0, 2.0, 10.0], [1.0, 3.0, 10.0]], 2)
        with self.assertRaisesRegex(RuntimeError, "checksums"):
            BENCHMARK._validate_rank_summaries([[0.0, 2.0, 10.0], [1.0, 2.0, 11.0]], 2)

    def test_distributed_failure_always_destroys_the_process_group(self):
        class Distributed:
            def __init__(self):
                self.destroyed = False

            def is_available(self):
                return True

            def is_nccl_available(self):
                return True

            def is_initialized(self):
                return True

            def destroy_process_group(self):
                self.destroyed = True

        distributed = Distributed()
        torch_module = SimpleNamespace(distributed=distributed)
        with (
            mock.patch.object(
                BENCHMARK, "_run_cuda_matmul", side_effect=RuntimeError("CUDA failed")
            ),
            self.assertRaisesRegex(RuntimeError, "CUDA failed"),
        ):
            BENCHMARK._run_distributed_rank(0, 0, 2, torch_module=torch_module)
        self.assertTrue(distributed.destroyed)

    def test_multi_node_uses_the_existing_llmapi_session(self):
        class Session:
            def __init__(self):
                self.shutdown_called = False
                self.submission = None

            def submit_sync(self, task, timeout):
                self.submission = (task, timeout)
                return [[0.0, 2.0, 10.0], [1.0, 2.0, 10.0]]

            def shutdown(self):
                self.shutdown_called = True

        session = Session()
        with mock.patch.dict(sys.modules, {"torch": _IMPORT_TORCH}):
            BENCHMARK._run_with_existing_llmapi_launcher(
                2, timeout_seconds=30, session_factory=lambda world_size: session
            )
        task, timeout = session.submission
        self.assertEqual(task.__module__, "infra_dry_run_benchmark")
        self.assertEqual(Path(task.__code__.co_filename).resolve(), BENCHMARK_PATH)
        self.assertEqual(timeout, 30)
        self.assertTrue(session.shutdown_called)

    def test_llmapi_rank_task_pickle_is_importable_in_worker_directory(self):
        with mock.patch.dict(sys.modules, {"torch": _IMPORT_TORCH}):
            task = BENCHMARK._pickleable_llmapi_rank_task()
            payload = pickle.dumps(task)
        with tempfile.TemporaryDirectory() as temp_dir:
            Path(temp_dir, "torch.py").write_text("# worker import stub\n")
            env = os.environ.copy()
            env["PYTHONPATH"] = os.pathsep.join(
                value for value in (temp_dir, env.get("PYTHONPATH")) if value
            )
            result = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    "import pickle, sys; print(pickle.loads(sys.stdin.buffer.read()).__module__)",
                ],
                cwd=BENCHMARK_PATH.parent,
                env=env,
                input=payload,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
        self.assertEqual(result.returncode, 0, result.stderr.decode())
        self.assertEqual(result.stdout.decode().strip(), "infra_dry_run_benchmark")

    def test_top_level_worker_module_is_importable_by_a_real_spawn_child(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            (temp_path / "torch.py").write_text("# import-only torch stub\n")
            driver = temp_path / "spawn_driver.py"
            driver.write_text(
                textwrap.dedent(
                    f"""
                    import importlib.util
                    import multiprocessing
                    import sys
                    from pathlib import Path

                    benchmark_path = Path({str(BENCHMARK_PATH)!r})

                    def main():
                        spec = importlib.util.spec_from_file_location(
                            "defs.infra_dry_run_benchmark", benchmark_path
                        )
                        module = importlib.util.module_from_spec(spec)
                        sys.modules[spec.name] = module
                        spec.loader.exec_module(module)

                        module_dir = benchmark_path.parent.resolve()
                        sys.path[:] = [
                            entry
                            for entry in sys.path
                            if Path(entry or ".").resolve() != module_dir
                        ]
                        worker_module = module._worker_import_module()
                        process = multiprocessing.get_context("spawn").Process(
                            target=worker_module._required_int,
                            args=({{"VALUE": "7"}}, "VALUE"),
                        )
                        process.start()
                        process.join(30)
                        if process.is_alive():
                            process.terminate()
                            process.join()
                            raise RuntimeError("spawn child did not finish")
                        raise SystemExit(process.exitcode)

                    if __name__ == "__main__":
                        main()
                    """
                )
            )
            env = {**os.environ, "PYTHONPATH": temp_dir}
            result = subprocess.run(
                [sys.executable, str(driver)],
                cwd=temp_dir,
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                check=False,
            )

        self.assertEqual(result.returncode, 0, result.stdout)

    def test_llmapi_worker_binds_rank_environment_to_the_mpi_communicator(self):
        communicator = SimpleNamespace(Get_rank=lambda: 1, Get_size=lambda: 2)
        mpi4py = SimpleNamespace(MPI=SimpleNamespace(COMM_WORLD=communicator))
        environ = {"RANK": "1", "LOCAL_RANK": "0", "WORLD_SIZE": "2"}
        with (
            mock.patch.dict(os.environ, environ, clear=True),
            mock.patch.dict(sys.modules, {"mpi4py": mpi4py}),
            mock.patch.object(
                BENCHMARK, "_run_distributed_rank", return_value=[1.0, 2.0, 10.0]
            ) as run_rank,
        ):
            result = BENCHMARK._llmapi_rank_task(45)
        self.assertEqual(result, [1.0, 2.0, 10.0])
        run_rank.assert_called_once_with(1, 0, 2, 45)

    def test_external_multi_node_rank_reuses_existing_launcher(self):
        environ = {
            "stageName": "GB300-MultiNode",
            "RANK": "0",
            "LOCAL_RANK": "0",
            "WORLD_SIZE": "2",
            "MASTER_ADDR": "host0",
            "MASTER_PORT": "23456",
            "TLLM_SPAWN_PROXY_PROCESS": "1",
        }
        with (
            mock.patch.dict(os.environ, environ, clear=True),
            mock.patch.object(BENCHMARK, "_run_with_existing_llmapi_launcher") as run_launcher,
        ):
            BENCHMARK.test_infra_dry_run_benchmark()
        run_launcher.assert_called_once_with(2)

    def test_nonzero_proxy_rank_cannot_be_the_pytest_controller(self):
        environ = {
            "stageName": "GB300-MultiNode",
            "RANK": "1",
            "LOCAL_RANK": "1",
            "WORLD_SIZE": "2",
            "MASTER_ADDR": "host0",
            "MASTER_PORT": "23456",
            "TLLM_SPAWN_PROXY_PROCESS": "1",
        }
        with (
            mock.patch.dict(os.environ, environ, clear=True),
            mock.patch.object(BENCHMARK, "_run_with_existing_llmapi_launcher") as run_launcher,
            self.assertRaisesRegex(RuntimeError, "only LLMAPI rank 0"),
        ):
            BENCHMARK.test_infra_dry_run_benchmark()
        run_launcher.assert_not_called()

    def test_single_node_multi_gpu_spawns_one_worker_per_visible_gpu(self):
        cuda = SimpleNamespace(is_available=lambda: True, device_count=lambda: 4)
        multiprocessing = SimpleNamespace(spawn=mock.Mock())
        fake_torch = SimpleNamespace(cuda=cuda, multiprocessing=multiprocessing)
        with (
            mock.patch.dict(os.environ, {"stageName": "H100-Multi-GPU"}, clear=True),
            mock.patch.dict(sys.modules, {"torch": fake_torch}),
            mock.patch.object(BENCHMARK, "torch", fake_torch),
            mock.patch.object(BENCHMARK, "_reserve_local_port", return_value=23456),
        ):
            BENCHMARK.test_infra_dry_run_benchmark()

        multiprocessing.spawn.assert_called_once()
        worker = multiprocessing.spawn.call_args.args[0]
        self.assertEqual(worker.__module__, "infra_dry_run_benchmark")
        self.assertEqual(
            multiprocessing.spawn.call_args.kwargs,
            {
                "args": (4, 23456, BENCHMARK._DISTRIBUTED_TIMEOUT_SECONDS),
                "nprocs": 4,
                "join": True,
            },
        )


if __name__ == "__main__":
    unittest.main()
