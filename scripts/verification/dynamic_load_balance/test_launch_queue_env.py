# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Check process-start launch queue setup without importing the GPU runtime."""

from __future__ import annotations

import ast
import builtins
import os
import sys
import unittest
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

REPO = Path(__file__).resolve().parents[3]
QUEUE = "CUDA_SCALE_LAUNCH_QUEUES"
HELPER = REPO / "tensorrt_llm/llmapi/_load_balance_env.py"


def load_helper():
    # Importing through tensorrt_llm would load native libraries. Execute only
    # this production module and reject accidental GPU/runtime imports.
    real_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name.split(".")[0] in {"torch", "cuda", "cupy", "tensorrt_llm", "llm_args"}:
            raise AssertionError(f"Queue setup imported a GPU module: {name}")
        return real_import(name, *args, **kwargs)

    namespace = {"__name__": "_queue_setup_under_test"}
    with patch("builtins.__import__", guarded_import):
        exec(compile(HELPER.read_text(), str(HELPER), "exec"), namespace)
    return namespace["configure_moe_launch_queues"]


def config(enabled=True, slots=4):
    return SimpleNamespace(rebalance=SimpleNamespace(enabled=enabled, helper_slots_per_rank=slots))


def function_node(relative_path, class_name, method_name):
    tree = ast.parse((REPO / relative_path).read_text())
    scope = (
        tree
        if class_name is None
        else next(
            node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name
        )
    )
    return next(
        node
        for node in scope.body
        if isinstance(node, ast.FunctionDef) and node.name == method_name
    )


def named_calls(node, name):
    return [
        call
        for call in ast.walk(node)
        if isinstance(call, ast.Call)
        and (
            (isinstance(call.func, ast.Name) and call.func.id == name)
            or (isinstance(call.func, ast.Attribute) and call.func.attr == name)
        )
    ]


WORKER_ENTRIES = (
    ("tensorrt_llm/executor/worker.py", None, "worker_main"),
    ("tensorrt_llm/executor/base_worker.py", "BaseWorker", "__init__"),
    ("tensorrt_llm/executor/ray/gpu_worker.py", "RayWorkerWrapper", "__init__"),
)


class QueueEnvironmentTests(unittest.TestCase):
    def setUp(self):
        self.context = ExitStack()
        self.addCleanup(self.context.close)
        self.context.enter_context(
            patch.dict(os.environ, {"DLB_TEST_SENTINEL": "unchanged"}, clear=True)
        )
        # No other CUDA API exists on this fake. A new device probe fails the
        # tests rather than silently initializing CUDA on a development host.
        self.cuda = SimpleNamespace(is_initialized=Mock(return_value=False))
        self.context.enter_context(
            patch.dict(sys.modules, {"torch": SimpleNamespace(cuda=self.cuda)})
        )
        self.configure = load_helper()

    def test_module_import_does_not_import_gpu_runtime(self):
        self.assertTrue(callable(load_helper()))
        self.cuda.is_initialized.assert_not_called()
        self.assertNotIn(QUEUE, os.environ)

    def test_on_works_without_torch_being_imported(self):
        sys.modules.pop("torch")
        self.assertEqual(self.configure(config()), {QUEUE: "4x"})
        self.assertEqual(os.environ[QUEUE], "4x")
        self.assertNotIn("torch", sys.modules)

    def test_on_copies_overrides_and_preserves_unrelated_values(self):
        original = {"TLLM_TEST_SETTING": "keep", "CUSTOM_NUMBER": 7}
        result = self.configure(config(), original)
        self.assertIsNot(result, original)
        self.assertEqual(original, {"TLLM_TEST_SETTING": "keep", "CUSTOM_NUMBER": 7})
        self.assertEqual(result, {**original, QUEUE: "4x"})
        self.assertEqual(os.environ, {"DLB_TEST_SENTINEL": "unchanged", QUEUE: "4x"})
        self.cuda.is_initialized.assert_called_once_with()

    def test_on_replaces_conflicting_values_before_cuda_init(self):
        os.environ[QUEUE] = "2x"
        original = {QUEUE: "1x"}
        self.assertEqual(self.configure(config(), original), {QUEUE: "4x"})
        self.assertEqual(os.environ[QUEUE], "4x")
        self.assertEqual(original, {QUEUE: "1x"})

    def test_off_and_zero_slots_are_complete_noops(self):
        self.cuda.is_initialized.side_effect = AssertionError("OFF must not inspect CUDA state")
        for moe in (None, SimpleNamespace(rebalance=None), config(enabled=False), config(slots=0)):
            for original in (None, {}, {QUEUE: "2x", "KEEP": "value"}):
                for previous in (None, "2x", "4x"):
                    with self.subTest(moe=moe, original=original, previous=previous):
                        os.environ.pop(QUEUE, None)
                        if previous is not None:
                            os.environ[QUEUE] = previous
                        before = dict(os.environ)
                        self.assertIs(self.configure(moe, original), original)
                        self.assertEqual(dict(os.environ), before)
        self.cuda.is_initialized.assert_not_called()

    def test_late_init_rejects_before_any_partial_mutation(self):
        self.cuda.is_initialized.return_value = True
        for previous in (None, "1x", "2x"):
            for requested in (None, "1x", "4x"):
                with self.subTest(previous=previous, requested=requested):
                    os.environ.pop(QUEUE, None)
                    if previous is not None:
                        os.environ[QUEUE] = previous
                    original = {"KEEP": "value"}
                    if requested is not None:
                        original[QUEUE] = requested
                    saved = dict(original)
                    before = dict(os.environ)
                    with self.assertRaisesRegex(
                        RuntimeError, "before CUDA initialization.*Restart"
                    ):
                        self.configure(config(), original)
                    self.assertEqual(dict(os.environ), before)
                    self.assertEqual(original, saved)

    def test_already_initialized_with_4x_is_allowed(self):
        os.environ[QUEUE] = "4x"
        self.cuda.is_initialized.return_value = True
        self.assertEqual(self.configure(config(), {QUEUE: "1x"}), {QUEUE: "4x"})
        self.assertEqual(os.environ[QUEUE], "4x")

    def test_repeated_on_setup_is_idempotent(self):
        first = self.configure(config(), {"KEEP": "value"})
        second = self.configure(config(), first)
        self.assertEqual(first, second)
        self.assertEqual(os.environ[QUEUE], "4x")


class QueueStartupContractTests(unittest.TestCase):
    setUp = QueueEnvironmentTests.setUp

    def test_llm_sets_queue_before_env_replay_validation_and_gpu_probes(self):
        node = function_node("tensorrt_llm/llmapi/llm.py", "BaseLLM", "__init__")
        (setup,) = named_calls(node, "configure_moe_launch_queues")
        for name in (
            "_process_env_overrides",
            "llm_args_cls",
            "get_device_count",
            "MpiPoolSession",
        ):
            calls = named_calls(node, name)
            self.assertTrue(calls, f"Missing startup boundary: {name}")
            self.assertLess(setup.lineno, min(call.lineno for call in calls))

    def test_worker_setup_precedes_mpi_super_and_ray_cuda_probes(self):
        boundaries = (("barrier", "update"), ("__init__",), ("device_count", "set_device"))
        for entry, names in zip(WORKER_ENTRIES, boundaries):
            with self.subTest(entry=entry):
                node = function_node(*entry)
                (setup,) = named_calls(node, "configure_moe_launch_queues")
                for name in names:
                    calls = named_calls(node, name)
                    self.assertTrue(calls, f"Missing startup boundary: {name}")
                    self.assertLess(setup.lineno, min(call.lineno for call in calls))

    def execute_worker_prefix(self, entry, llm_args):
        # Execute production through queue setup and stop before runtime setup.
        # Order checks cover the later boundaries; this exercises actual gates.
        node = function_node(*entry)
        prefix = []
        for statement in node.body:
            prefix.append(statement)
            if named_calls(statement, "configure_moe_launch_queues"):
                break
        else:
            self.fail(f"Missing queue setup in {entry}")
        module = ast.Module(body=prefix, type_ignores=[])
        namespace = {
            "configure_moe_launch_queues": self.configure,
            "llm_args": llm_args,
            "worker_kwargs": {"llm_args": llm_args},
        }
        exec(compile(ast.fix_missing_locations(module), str(REPO / entry[0]), "exec"), namespace)

    def test_each_worker_entry_propagates_on_override(self):
        for entry in WORKER_ENTRIES:
            with self.subTest(entry=entry):
                os.environ.pop(QUEUE, None)
                original = {"KEEP": "value"}
                args = SimpleNamespace(
                    backend="pytorch", moe_config=config(), env_overrides=original
                )
                self.execute_worker_prefix(entry, args)
                self.assertEqual(os.environ[QUEUE], "4x")
                self.assertEqual(args.env_overrides, {"KEEP": "value", QUEUE: "4x"})
                self.assertEqual(original, {"KEEP": "value"})
                self.assertIsNot(args.env_overrides, original)

    def test_each_worker_entry_leaves_off_arguments_untouched(self):
        class ReadOnlyArgs:
            def __init__(self, backend, moe_config):
                self.backend = backend
                self.moe_config = moe_config
                self._overrides = {QUEUE: "2x", "KEEP": "value"}

            @property
            def env_overrides(self):
                return self._overrides

            @env_overrides.setter
            def env_overrides(self, value):
                raise AssertionError("OFF must not reassign env_overrides")

        self.cuda.is_initialized.side_effect = AssertionError("OFF must not inspect CUDA state")
        for entry in WORKER_ENTRIES:
            for backend, moe in (
                ("pytorch", None),
                ("pytorch", config(enabled=False)),
                ("pytorch", config(slots=0)),
                ("_autodeploy", config()),
            ):
                with self.subTest(entry=entry, backend=backend, moe=moe):
                    before = dict(os.environ)
                    args = ReadOnlyArgs(backend, moe)
                    self.execute_worker_prefix(entry, args)
                    self.assertEqual(dict(os.environ), before)
                    self.assertEqual(args.env_overrides, {QUEUE: "2x", "KEEP": "value"})
            self.execute_worker_prefix(entry, None)
        self.cuda.is_initialized.assert_not_called()

    def test_mpi_pool_passes_queue_in_explicit_child_environment(self):
        self.configure(config())
        os.environ.update(
            {"TRTLLM_TEST_SETTING": "parent", "UNRELATED_PARENT_SETTING": "not forwarded"}
        )
        node = function_node(
            "tensorrt_llm/llmapi/mpi_session.py", "MpiPoolSession", "_start_mpi_pool"
        )
        captured = {}

        def create_pool(**kwargs):
            captured.update(kwargs)
            return object()

        constant_names = {
            "_FLASHINFER_WORKSPACE_ROOT",
            "_FLASHINFER_WORKSPACE_ENV",
            "_FLASHINFER_WORKSPACE_MANAGED_ENV",
            "_FLASHINFER_WORKER_BOOTSTRAP",
        }
        source = ast.parse((REPO / "tensorrt_llm/llmapi/mpi_session.py").read_text())
        constants = {
            target.id: ast.literal_eval(statement.value)
            for statement in source.body
            if isinstance(statement, ast.Assign)
            for target in statement.targets
            if isinstance(target, ast.Name) and target.id in constant_names
        }
        self.assertEqual(set(constants), constant_names)
        namespace = {"os": os, "sys": sys, "MPIPoolExecutor": create_pool, **constants}
        module = ast.Module(body=[node], type_ignores=[])
        exec(compile(ast.fix_missing_locations(module), "mpi_pool_contract", "exec"), namespace)
        session = SimpleNamespace(
            mpi_pool=None,
            n_workers=2,
            _env_overrides={"TRTLLM_TEST_SETTING": "worker", "WORKER_ONLY_SETTING": "keep"},
        )
        namespace["_start_mpi_pool"](session)
        self.assertEqual(
            captured["env"],
            {QUEUE: "4x", "TRTLLM_TEST_SETTING": "worker", "WORKER_ONLY_SETTING": "keep"},
        )
        self.assertEqual(captured["max_workers"], 2)
        self.assertEqual(
            captured["python_args"],
            [
                "-c",
                constants["_FLASHINFER_WORKER_BOOTSTRAP"],
                constants["_FLASHINFER_WORKSPACE_ROOT"],
            ],
        )
        self.assertIsNotNone(session.mpi_pool)

    def test_ray_actor_runtime_environment_carries_parent_queue(self):
        self.configure(config())
        os.environ.update({"RAY_RAYLET_PID": "parent pid", "RAY_NODE_IP_ADDRESS": "parent address"})
        node = function_node(
            "tensorrt_llm/executor/ray/executor.py", "RayExecutor", "create_workers"
        )
        local_vars = next(
            statement.value
            for statement in node.body
            if isinstance(statement, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "_NODE_LOCAL_VARS"
                for target in statement.targets
            )
        )
        environment = next(
            statement.value
            for statement in node.body
            if isinstance(statement, ast.Assign)
            and any(
                isinstance(target, ast.Subscript)
                and isinstance(target.value, ast.Name)
                and target.value.id == "runtime_env"
                and isinstance(target.slice, ast.Constant)
                and target.slice.value == "env_vars"
                for target in statement.targets
            )
        )
        namespace = {"os": os, "_NODE_LOCAL_VARS": ast.literal_eval(local_vars)}
        expression = ast.Expression(body=environment)
        result = eval(
            compile(ast.fix_missing_locations(expression), "ray_environment_contract", "eval"),
            namespace,
        )
        self.assertEqual(result, {QUEUE: "4x", "DLB_TEST_SENTINEL": "unchanged"})


if __name__ == "__main__":
    unittest.main(verbosity=2)
