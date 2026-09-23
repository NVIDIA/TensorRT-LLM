# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU regressions for dense scheduler pins and reserved-SM runners.

Execute the production policy and runner methods with hardware capability
stand-ins. Override TRTLLM_DENSE_POLICY_TEST_SOURCE to check a separate source
checkout; by default this test uses the enclosing repository.
"""

from __future__ import annotations

import ast
import copy
import itertools
import os
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock, patch

SOURCE = Path(
    os.environ.get(
        "TRTLLM_DENSE_POLICY_TEST_SOURCE",
        str(
            Path(__file__).resolve().parents[3]
            / "tensorrt_llm/_torch/custom_ops/cute_dsl_custom_ops.py"
        ),
    )
)
ENV = "TRTLLM_CUTEDSL_DENSE_GEMM_SCHEDULER"


def load_production_methods() -> dict[str, Any]:
    """Keep real class inheritance, including the zero-argument super cells."""
    tree = ast.parse(SOURCE.read_text())
    helpers = {"_dense_gemm_scheduler_override", "_dense_gemm_scheduler_modes"}
    constants = {"_DENSE_GEMM_SCHEDULER_ENV", "_dense_gemm_scheduler_announced"}
    methods = {
        "__init__",
        "__hash__",
        "__eq__",
        "unique_id",
        "_max_active_clusters",
        "get_valid_tactics",
        "should_profile_tactic_in_subprocess",
        "forward",
    }
    body = [ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)]
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in helpers:
            body.append(node)
        elif isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id in constants for target in node.targets
        ):
            body.append(node)
    for name in ("CuteDSLBlockScaledRubinLinear", "CuteDSLMXFP8RubinLinear"):
        cls = copy.deepcopy(
            next(n for n in ast.walk(tree) if isinstance(n, ast.ClassDef) and n.name == name)
        )
        cls.body = [
            node
            for node in cls.body
            if (isinstance(node, ast.FunctionDef) and node.name in methods)
            or (
                isinstance(node, (ast.Assign, ast.AnnAssign))
                and not any(
                    isinstance(n, ast.Name) and n.id == "tuning_config" for n in ast.walk(node)
                )
            )
        ]
        body.append(cls)

    namespace = {
        "os": os,
        "itertools": itertools,
        "logger": Mock(),
        "TunableRunner": object,
        "torch": SimpleNamespace(
            bfloat16="bfloat16",
            float8_e4m3fn="float8_e4m3fn",
            cuda=SimpleNamespace(
                current_device=lambda: 0,
                get_device_properties=lambda _: SimpleNamespace(multi_processor_count=212),
            ),
        ),
        "cutlass": SimpleNamespace(BFloat16="bf16", Float8E4M3FN="e4m3", Float8E8M0FNU="e8m0"),
        "Sm107BlockScaledPersistentDenseGemmKernel": Mock(can_implement=Mock(return_value=True)),
        "Sm107BlockScaledPersistentDenseGemmMixedClustersKernel": Mock(
            can_implement=Mock(return_value=True)
        ),
        "get_sm_version": lambda: 107,
        "get_max_activate_clusters": lambda size: {1: 212, 2: 106, 4: 53, 8: 22}[size],
        "_get_cute_dsl_swap_ab_candidates": Mock(return_value=[False, True]),
    }
    module = ast.Module(body=body, type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(SOURCE), "exec"), namespace)
    return namespace


class ReachedTensorInputs(Exception):
    """Stop a real forward call after tactic validation and before device work."""


class InputBoundary:
    def __len__(self) -> int:
        raise ReachedTensorInputs


class DenseSchedulerPolicy(unittest.TestCase):
    def setUp(self) -> None:
        env_patch = patch.dict(os.environ)
        env_patch.start()
        self.addCleanup(env_patch.stop)
        os.environ.pop(ENV, None)
        self.ns = load_production_methods()
        self.runner_class = self.ns["CuteDSLMXFP8RubinLinear"]
        self.inputs = [
            SimpleNamespace(shape=(128, 128), dim=lambda: 2),
            SimpleNamespace(shape=(256, 128), dim=lambda: 2),
        ]

    def pin(self, value: str | None) -> None:
        if value is None:
            os.environ.pop(ENV, None)
        else:
            os.environ[ENV] = value

    def runner(self, reserved_sms: int = 0) -> Any:
        return self.runner_class(
            output_dtype="bfloat16", use_tvm_ffi=False, reserved_sms=reserved_sms
        )

    def tactics(self, runner: Any) -> list[tuple[object, ...]]:
        return runner.get_valid_tactics(self.inputs, None)

    def test_pin_intersects_supported_modes_and_normalizes(self) -> None:
        modes = self.ns["_dense_gemm_scheduler_modes"]
        for pin, expected in (
            (None, ("static", "clc_dynamic")),
            ("", ("static", "clc_dynamic")),
            ("static", ("static",)),
            (" CLC_DYNAMIC ", ("clc_dynamic",)),
        ):
            with self.subTest(pin=pin):
                self.pin(pin)
                self.assertEqual(modes(("static", "clc_dynamic")), expected)
                self.assertEqual(modes(("static",)), ("static",))
                self.assertEqual(modes(("clc_dynamic",)), ("clc_dynamic",))
                self.assertEqual(modes(()), ())
                expected_reserved = ("clc_dynamic",) if pin == " CLC_DYNAMIC " else ("static",)
                self.assertEqual(modes(("static",), ("static", "clc_dynamic")), expected_reserved)

    def test_enumerated_tactics_pass_profile_and_forward_validation(self) -> None:
        for reserved_sms, pin, expected_modes in (
            (0, None, {"static", "clc_dynamic"}),
            (0, "static", {"static"}),
            (0, "clc_dynamic", {"clc_dynamic"}),
            (8, None, {"static"}),
            (8, "static", {"static"}),
            (8, "clc_dynamic", {"clc_dynamic"}),
        ):
            with self.subTest(reserved_sms=reserved_sms, pin=pin):
                self.pin(pin)
                runner = self.runner(reserved_sms)
                tactics = self.tactics(runner)
                self.assertEqual({t[6] for t in tactics if t[0] == "base"}, expected_modes)
                base_tactics = [t for t in tactics if t[0] == "base"]
                self.assertTrue(all(len(t) == 9 for t in base_tactics))
                self.assertEqual({t[8] for t in base_tactics}, {1, 2, 4, 8})
                mixed_tactics = [t for t in tactics if t[0] == "mixed_clusters"]
                self.assertEqual(
                    {t[7] if len(t) == 9 else "static" for t in mixed_tactics}, expected_modes
                )
                self.assertTrue(
                    all(t[8] == "m" for t in mixed_tactics if len(t) == 9 and t[7] == "clc_dynamic")
                )
                for tactic in tactics:
                    self.assertTrue(
                        runner.should_profile_tactic_in_subprocess(
                            "test", self.inputs, tactic, None
                        )
                    )
                    with self.assertRaises(ReachedTensorInputs):
                        runner.forward(InputBoundary(), tactic)

    def test_off_cache_identity_keeps_target_behavior(self) -> None:
        runner = self.runner()
        base = ("bfloat16", False, False)
        for pin, expected in (
            (None, base),
            ("static", (*base, "static")),
            ("clc_dynamic", (*base, "clc_dynamic")),
        ):
            with self.subTest(pin=pin):
                self.pin(pin)
                self.assertEqual(runner.unique_id(), expected)

    def test_reserved_cache_separates_explicit_dynamic_pin(self) -> None:
        runner = self.runner(8)
        cache_id = runner.unique_id()
        tactics = self.tactics(runner)
        self.assertEqual(cache_id, ("bfloat16", False, False, "reserved_sms", 8, "static_grid_v1"))
        self.pin("clc_dynamic")
        self.assertEqual(
            runner.unique_id(),
            ("bfloat16", False, False, "clc_dynamic", "reserved_sms", 8, "clc_dynamic_grid_v1"),
        )
        self.assertNotEqual(self.tactics(runner), tactics)
        self.assertNotEqual(self.runner().unique_id(), runner.unique_id())
        self.assertEqual(runner.scheduler_mode_candidates, ("static",))
        self.assertEqual(runner.supported_scheduler_modes, ("static", "clc_dynamic"))
        self.assertEqual(self.runner().scheduler_mode_candidates, ("static", "clc_dynamic"))
        self.pin("static")
        self.assertEqual(
            runner.unique_id(),
            ("bfloat16", False, False, "static", "reserved_sms", 8, "static_grid_v1"),
        )

    def test_mixed_kernel_capability_still_limits_candidates(self) -> None:
        mixed_kernel = self.ns["Sm107BlockScaledPersistentDenseGemmMixedClustersKernel"]
        mixed_kernel.can_implement.return_value = False
        for reserved_sms in (0, 8):
            for pin in (None, "static", "clc_dynamic"):
                with self.subTest(reserved_sms=reserved_sms, pin=pin):
                    self.pin(pin)
                    tactics = self.tactics(self.runner(reserved_sms))
                    self.assertTrue(tactics)
                    self.assertTrue(all(t[0] == "base" for t in tactics))

    def test_profile_policy_and_runtime_support_are_distinct(self) -> None:
        mixed_tactic = next(t for t in self.tactics(self.runner()) if t[0] == "mixed_clusters")
        self.pin("clc_dynamic")
        dynamic_tactic = next(t for t in self.tactics(self.runner()) if t[0] == "base")
        for reserved_sms in (0, 8):
            for pin in (None, "static", "clc_dynamic"):
                with self.subTest(reserved_sms=reserved_sms, pin=pin):
                    self.pin(pin)
                    runner = self.runner(reserved_sms)
                    self.assertEqual(
                        runner.should_profile_tactic_in_subprocess(
                            "test", self.inputs, dynamic_tactic, None
                        ),
                        pin == "clc_dynamic" or (pin is None and reserved_sms == 0),
                    )
                    self.assertEqual(
                        runner.should_profile_tactic_in_subprocess(
                            "test", self.inputs, mixed_tactic, None
                        ),
                        pin != "clc_dynamic",
                    )
                    legacy_static_tactic = dynamic_tactic[:6]
                    self.assertEqual(
                        runner.should_profile_tactic_in_subprocess(
                            "test", self.inputs, legacy_static_tactic, None
                        ),
                        pin != "clc_dynamic",
                    )
                    # Forward checks support, while enumeration/profile enforce
                    # the current policy. Cached policies have distinct IDs.
                    with self.assertRaises(ReachedTensorInputs):
                        runner.forward(InputBoundary(), dynamic_tactic)
                    with self.assertRaises(ReachedTensorInputs):
                        runner.forward(InputBoundary(), legacy_static_tactic)
                    invalid = (*dynamic_tactic[:6], "unsupported", dynamic_tactic[7])
                    with self.assertRaisesRegex(
                        ValueError, "Unsupported CuteDSL SM107 scheduler mode"
                    ):
                        runner.forward(InputBoundary(), invalid)

    def test_invalid_split_k_is_rejected(self) -> None:
        for reserved_sms in (0, 8):
            for pin in (None, "static", "clc_dynamic"):
                self.pin(pin)
                runner = self.runner(reserved_sms)
                base_tactic = next(t for t in self.tactics(runner) if t[0] == "base")
                for split_k in (0, -1, True):
                    with self.subTest(reserved_sms=reserved_sms, pin=pin, split_k=split_k):
                        invalid = (*base_tactic[:-1], split_k)
                        self.assertFalse(
                            runner.should_profile_tactic_in_subprocess(
                                "test", self.inputs, invalid, None
                            )
                        )
                        with self.assertRaisesRegex(
                            ValueError, "split_k must be a positive integer"
                        ):
                            runner.forward(InputBoundary(), invalid)

    def test_unpinned_and_static_grids_keep_reserved_budget(self) -> None:
        for pin in (None, "static"):
            with self.subTest(pin=pin):
                self.pin(pin)
                self.assertEqual(
                    [self.runner(8)._max_active_clusters(c) for c in (1, 2, 4, 8)],
                    [204, 102, 51, 22],
                )
                self.assertEqual(
                    [self.runner()._max_active_clusters(c) for c in (1, 2, 4, 8)],
                    [212, 106, 53, 22],
                )
                with self.assertRaisesRegex(ValueError, "leaves no active GEMM cluster"):
                    self.runner(212)._max_active_clusters(2)

    def test_invalid_environment_is_rejected(self) -> None:
        for pin in ("dynamic", "both", "1"):
            with self.subTest(pin=pin):
                self.pin(pin)
                for reserved_sms in (0, 8):
                    runner = self.runner(reserved_sms)
                    for call in (runner.unique_id, lambda: self.tactics(runner)):
                        with self.assertRaisesRegex(ValueError, ENV):
                            call()

    def test_announcement_describes_pin_scope_once(self) -> None:
        self.pin("clc_dynamic")
        modes = self.ns["_dense_gemm_scheduler_modes"]
        modes(("static",), ("static", "clc_dynamic"))
        modes(("static", "clc_dynamic"))
        logger = self.ns["logger"]
        logger.info.assert_called_once()
        message = logger.info.call_args.args[0]
        self.assertIn("supporting", message)
        self.assertIn("unsupported", message)
        self.assertIn("reserved_sms does not cap CLC", message)


if __name__ == "__main__":
    unittest.main(verbosity=2)
