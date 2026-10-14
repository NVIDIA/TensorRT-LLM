# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU-only checks for the opt-in router gate, not GPU reproduction evidence."""

import ast
import asyncio
import importlib.util
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

_ROOT = Path(__file__).resolve().parents[3]
_HELPER = _ROOT / "tensorrt_llm/_disagg_late_receiver_diagnostic.py"
_SPEC = importlib.util.spec_from_file_location("late_receiver_diagnostic", _HELPER)
diagnostic = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(diagnostic)


class _DiagnosticFixture(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.environment = patch.dict(
            os.environ,
            {
                "TRTLLM_LATE_RECEIVER_DIAGNOSTIC_DIR": str(self.root),
                "TRTLLM_DISAGG_NO_RETRY": "1",
            },
        )
        self.environment.start()
        self.addCleanup(self.environment.stop)
        self.request_id = 421
        self.ctx_dp_rank = 2

    def claim(self, mode="after_retirement", wait_seconds=90):
        self.trial = diagnostic.arm(self.root, mode, wait_seconds)
        return diagnostic.claim_request(self.request_id, eligible=True, clients_no_retry=True)

    def send(self, **overrides):
        fields = dict(
            request_id=57,
            rank=2,
            ctx_dp_rank=self.ctx_dp_rank,
            pp_size=1,
            cp_size=1,
            attention_dp=True,
            bridge_enabled=False,
            pipeline_transfer=False,
            kv_transfer_timeout_ms=60000,
        )
        fields.update(overrides)
        diagnostic.observe("ctx_send_published", self.request_id, **fields)
        diagnostic.observe(
            "ctx_timeout_started",
            self.request_id,
            rank=2,
            timeout_start_monotonic=1.0,
            kv_transfer_timeout_ms=60000,
        )

    def retire(self, **overrides):
        diagnostic.observe("ctx_native_closed", self.request_id, rank=2)
        fields = dict(
            rank=2,
            ctx_dp_rank=self.ctx_dp_rank,
            timed_out=True,
            close_confirmed=True,
            session_absent=True,
            retired_flag=True,
        )
        fields.update(overrides)
        diagnostic.observe("ctx_retired", self.request_id, **fields)

    def events(self):
        return diagnostic.read_events(self.root, self.trial["trial_id"])

    async def gate(self, claim):
        await diagnostic.before_generation(
            claim, self.request_id, self.request_id, self.ctx_dp_rank
        )


class LateReceiverDiagnosticTest(_DiagnosticFixture):
    async def test_disabled_is_noop_without_filesystem_access(self):
        with (
            patch.dict(os.environ, {"TRTLLM_LATE_RECEIVER_DIAGNOSTIC_DIR": ""}),
            patch.object(diagnostic, "_read_json", side_effect=AssertionError("unexpected I/O")),
        ):
            diagnostic.validate_environment()
            self.assertFalse(diagnostic.is_enabled())
            self.assertIsNone(diagnostic.claim_request(1, eligible=False, clients_no_retry=False))
            diagnostic.observe("ignored", 1, request_id=2)
            await diagnostic.before_generation(None, 1, 2, 3)

    async def test_control_does_not_wait_for_retirement(self):
        claim = self.claim("control")
        self.send()
        with patch.object(diagnostic.asyncio, "sleep", new_callable=AsyncMock) as sleep:
            await self.gate(claim)
        sleep.assert_not_awaited()
        self.assertEqual(self.events()[-1]["reason"], "control")

    async def test_delayed_releases_only_after_observed_retirement(self):
        claim = self.claim()
        self.send()

        async def retire_after_yield(_seconds):
            self.assertNotIn("router_gate_released", [event["event"] for event in self.events()])
            self.retire()

        with patch.object(diagnostic.asyncio, "sleep", side_effect=retire_after_yield) as sleep:
            await self.gate(claim)
        sleep.assert_awaited_once()
        names = [event["event"] for event in self.events()]
        self.assertLess(names.index("ctx_native_closed"), names.index("ctx_retired"))
        self.assertLess(names.index("ctx_retired"), names.index("router_gate_released"))
        self.assertEqual(self.events()[-1]["reason"], "confirmed_retirement")

    async def test_missing_send_marker_waits_bounded_even_for_control(self):
        claim = self.claim("control")

        async def publish_after_yield(_seconds):
            self.send()

        with patch.object(diagnostic.asyncio, "sleep", side_effect=publish_after_yield) as sleep:
            await self.gate(claim)
        sleep.assert_awaited_once()

    async def test_timeout_warning_is_not_retirement(self):
        claim = self.claim(wait_seconds=0.01)
        self.send()
        diagnostic.observe("timeout_warning", self.request_id, rank=2)
        with patch.object(diagnostic.time, "monotonic", side_effect=[0, 1]):
            with self.assertRaisesRegex(diagnostic.DiagnosticError, "GEN not sent"):
                await self.gate(claim)
        self.assertNotIn("router_gate_released", [event["event"] for event in self.events()])

    async def test_wrong_owner_retirement_cannot_release(self):
        claim = self.claim(wait_seconds=0.01)
        self.send()
        self.retire(rank=3)
        with patch.object(diagnostic.time, "monotonic", side_effect=[0, 1]):
            with self.assertRaises(diagnostic.DiagnosticError):
                await self.gate(claim)
        self.assertNotIn("router_gate_released", [event["event"] for event in self.events()])

    async def test_failed_close_is_not_retirement(self):
        claim = self.claim()
        self.send()
        self.retire(close_confirmed=False)
        with self.assertRaisesRegex(diagnostic.DiagnosticError, "sender removal"):
            await self.gate(claim)

    async def test_unsupported_topology_is_rejected(self):
        claim = self.claim("control")
        self.send(attention_dp=False)
        with self.assertRaisesRegex(diagnostic.DiagnosticError, "PP1/CP1"):
            await self.gate(claim)

    async def test_changed_request_id_is_rejected(self):
        claim = self.claim()
        with self.assertRaisesRegex(diagnostic.DiagnosticError, "identity change"):
            await diagnostic.before_generation(claim, 422, 422, self.ctx_dp_rank)

    async def test_only_one_request_is_claimed_and_other_ids_are_not_logged(self):
        self.claim()
        self.assertIsNone(diagnostic.claim_request(422, eligible=True, clients_no_retry=True))
        diagnostic.observe("ctx_send_published", 422)
        self.assertEqual([event["event"] for event in self.events()], ["router_claimed"])

    async def test_disarm_requires_exact_trial_and_keeps_evidence(self):
        self.claim()
        with self.assertRaises(diagnostic.DiagnosticError):
            diagnostic.disarm(self.root, "0" * 32)
        with self.assertRaises(diagnostic.DiagnosticError):
            diagnostic.arm(self.root, "control")
        diagnostic.disarm(self.root, self.trial["trial_id"])
        self.assertEqual(len(self.events()), 1)
        second = diagnostic.arm(self.root, "control")
        self.assertNotEqual(second["trial_id"], self.trial["trial_id"])

    async def test_disarmed_gate_does_not_send_generation(self):
        claim = self.claim()
        diagnostic.disarm(self.root, self.trial["trial_id"])
        with self.assertRaisesRegex(diagnostic.DiagnosticError, "disarmed"):
            await self.gate(claim)

    async def test_no_retry_must_be_set_before_server_start_and_cached_in_clients(self):
        with patch.dict(os.environ, {"TRTLLM_DISAGG_NO_RETRY": "0"}):
            with self.assertRaisesRegex(diagnostic.DiagnosticError, "startup"):
                diagnostic.validate_environment()
        diagnostic.arm(self.root, "control")
        with self.assertRaisesRegex(diagnostic.DiagnosticError, "no retries"):
            diagnostic.claim_request(421, eligible=True, clients_no_retry=False)

    async def test_cross_host_events_are_invalid(self):
        claim = self.claim()
        self.send()
        directory = self.root / "trials" / self.trial["trial_id"] / "events"
        event = self.events()[0] | {"host": "different-node"}
        diagnostic._write_json(directory / "cross-host.json", event)
        with self.assertRaisesRegex(diagnostic.DiagnosticError, "one request, host"):
            await self.gate(claim)

    async def test_cancelled_gate_does_not_release_generation(self):
        claim = self.claim()
        self.send()
        with patch.object(diagnostic.asyncio, "sleep", side_effect=asyncio.CancelledError):
            with self.assertRaises(asyncio.CancelledError):
                await self.gate(claim)
        self.assertEqual(self.events()[-1]["event"], "router_cancelled")

    async def test_observation_io_failure_does_not_escape_into_product(self):
        self.claim()
        with patch.object(diagnostic, "_write_json", side_effect=OSError("unavailable")):
            with self.assertLogs(diagnostic.__name__, level="WARNING"):
                diagnostic.observe("ctx_retired", self.request_id)

    async def test_request_id_field_does_not_collide_with_trace_identity(self):
        self.claim()
        diagnostic.observe("identity", self.request_id, request_id=57)
        self.assertEqual(self.events()[-1]["request_id"], 57)
        self.assertEqual(self.events()[-1]["disagg_request_id"], self.request_id)


class DiagnosticHookWiringTest(unittest.TestCase):
    def test_disabled_hook_guards_do_not_access_runtime_attributes(self):
        paths = (
            "tensorrt_llm/_torch/disaggregation/transceiver.py",
            "tensorrt_llm/_torch/disaggregation/orchestration/coordinator.py",
            "tensorrt_llm/_torch/disaggregation/native/transfer.py",
        )
        count = 0
        with patch.dict(os.environ, {"TRTLLM_LATE_RECEIVER_DIAGNOSTIC_DIR": ""}):
            for path in paths:
                for node in ast.walk(ast.parse((_ROOT / path).read_text())):
                    if not isinstance(
                        node, ast.If
                    ) or "late_receiver_diagnostic.is_enabled()" not in ast.unparse(node.test):
                        continue
                    # Execute the actual guarded hook with deliberately empty
                    # runtime objects; disabled instrumentation must not inspect them.
                    code = compile(ast.Module(body=[node], type_ignores=[]), path, "exec")
                    exec(
                        code,
                        {
                            "late_receiver_diagnostic": diagnostic,
                            "self": object(),
                            "req": object(),
                            "request": object(),
                            "kind": "generation",
                        },
                    )
                    count += 1
        self.assertGreaterEqual(count, 9)

    def test_observation_calls_are_guarded_and_accept_their_actual_keywords(self):
        paths = (
            "tensorrt_llm/serve/openai_disagg_service.py",
            "tensorrt_llm/_torch/disaggregation/transceiver.py",
            "tensorrt_llm/_torch/disaggregation/orchestration/coordinator.py",
            "tensorrt_llm/_torch/disaggregation/native/transfer.py",
        )
        count = 0
        with patch.dict(os.environ, {"TRTLLM_LATE_RECEIVER_DIAGNOSTIC_DIR": ""}):
            for path in paths:
                tree = ast.parse((_ROOT / path).read_text())
                parents = {
                    child: parent
                    for parent in ast.walk(tree)
                    for child in ast.iter_child_nodes(parent)
                }
                for call in ast.walk(tree):
                    if not isinstance(call, ast.Call) or not isinstance(call.func, ast.Attribute):
                        continue
                    if (
                        not isinstance(call.func.value, ast.Name)
                        or call.func.value.id != "late_receiver_diagnostic"
                    ):
                        continue
                    if call.func.attr != "observe":
                        continue
                    count += 1
                    ancestors = []
                    node = call
                    while node in parents:
                        node = parents[node]
                        ancestors.append(node)
                    self.assertTrue(
                        any(
                            isinstance(node, ast.If)
                            and (
                                "late_receiver_diagnostic.is_enabled()" in ast.unparse(node.test)
                                or "diagnostic_claim is not None" in ast.unparse(node.test)
                            )
                            for node in ancestors
                        ),
                        path,
                    )
                    diagnostic.observe(
                        "hook", 1, **{keyword.arg: None for keyword in call.keywords}
                    )
        self.assertGreaterEqual(count, 8)


class DiagnosticTimeoutHookTest(_DiagnosticFixture):
    async def test_actual_timeout_decision_emits_once_after_deadline(self):
        self.claim()
        source = ast.parse(
            (_ROOT / "tensorrt_llm/_torch/disaggregation/orchestration/coordinator.py").read_text()
        )
        coordinator = next(
            node
            for node in source.body
            if isinstance(node, ast.ClassDef) and node.name == "DisaggTransferCoordinator"
        )
        method = next(
            node
            for node in coordinator.body
            if isinstance(node, ast.FunctionDef) and node.name == "check_transfer_timeouts"
        )
        method.decorator_list = []
        timer = SimpleNamespace(monotonic=lambda: 61.0)
        namespace = {
            "late_receiver_diagnostic": diagnostic,
            "get_unique_rid": lambda request: request.disagg_request_id,
            "LlmRequest": object,
            "time": timer,
            "logger": SimpleNamespace(warning=lambda message: None),
        }
        exec(
            compile(ast.Module(body=[method], type_ignores=[]), "coordinator.py", "exec"), namespace
        )
        request = SimpleNamespace(
            disagg_request_id=self.request_id,
            py_request_id=57,
            py_kv_transfer_start_time=1.0,
            py_kv_transfer_timed_out=False,
            is_disagg_generation_transmission_in_progress=True,
        )
        owner = SimpleNamespace(
            _transceiver=SimpleNamespace(kv_transfer_timeout_ms=60000),
            _transfers=SimpleNamespace(requests_in_transfer=lambda: {}),
            _registry=SimpleNamespace(active_requests=lambda: [request]),
            _dist=SimpleNamespace(rank=2),
            inflight_cancel_active=lambda: False,
        )
        check = namespace["check_transfer_timeouts"]
        check(owner)
        self.assertFalse(request.py_kv_transfer_timed_out)
        timer.monotonic = lambda: 61.001
        check(owner)
        check(owner)
        expired = [event for event in self.events() if event["event"] == "gen_timeout_expired"]
        self.assertEqual(len(expired), 1)
        self.assertEqual(expired[0]["timeout_start_monotonic"], 1.0)
        self.assertEqual(expired[0]["kv_transfer_timeout_ms"], 60000)
        self.assertAlmostEqual(expired[0]["observed_elapsed_ms"], 60001)

        # The same production timeout method needs no diagnostic-only rank
        # metadata when the diagnostic is disabled.
        del owner._dist
        request.py_kv_transfer_timed_out = False
        with patch.dict(os.environ, {"TRTLLM_LATE_RECEIVER_DIAGNOSTIC_DIR": ""}):
            check(owner)
        self.assertTrue(request.py_kv_transfer_timed_out)


if __name__ == "__main__":
    unittest.main()
