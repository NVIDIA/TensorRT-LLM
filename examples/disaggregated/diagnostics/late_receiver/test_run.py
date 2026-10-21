# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit checks for evidence classification, not TensorRT-LLM reproduction."""

import unittest

from run import classify


def _events(mode: str = "after_retirement") -> list[dict]:
    """Construct a single-node trace with distinct router/CTX/GEN processes."""
    common = {
        "trial_id": "a" * 32,
        "mode": mode,
        "disagg_request_id": 41,
        "host": "test-node",
        "no_retry": True,
    }
    ctx = dict(common, pid=200, rank=2)
    gen = dict(common, pid=300, rank=1, request_id=99)
    router = dict(common, pid=100, rank=2, ctx_dp_rank=2)
    release_time = 161 if mode == "after_retirement" else 101
    timer_start = float(release_time + 3)
    events = [
        dict(
            ctx,
            event="ctx_send_published",
            monotonic_ns=100_000_000_000,
            ctx_dp_rank=2,
            pp_size=1,
            cp_size=1,
            attention_dp=True,
            bridge_enabled=False,
            pipeline_transfer=False,
            kv_transfer_timeout_ms=60_000,
        ),
        dict(ctx, event="ctx_timeout_started", monotonic_ns=100_010_000_000),
        dict(
            router,
            event="router_gate_released",
            monotonic_ns=release_time * 1_000_000_000,
            reason="confirmed_retirement" if mode == "after_retirement" else "control",
        ),
        dict(gen, event="gen_request_data_sent", monotonic_ns=(release_time + 1) * 1_000_000_000),
        dict(gen, event="gen_receive_published", monotonic_ns=(release_time + 2) * 1_000_000_000),
        dict(
            gen,
            event="gen_timeout_started",
            monotonic_ns=int(timer_start * 1_000_000_000) + 1,
            timeout_start_monotonic=timer_start,
            kv_transfer_timeout_ms=60_000,
        ),
        dict(
            ctx,
            event="sender_request_data",
            monotonic_ns=(release_time + 1) * 1_000_000_000 + 500_000_000,
            peer_rank=1,
            session_present=mode == "control",
            fenced=False,
        ),
    ]
    if mode == "after_retirement":
        events.extend(
            [
                dict(ctx, event="ctx_native_closed", monotonic_ns=160_100_000_000),
                dict(
                    ctx,
                    event="ctx_retired",
                    monotonic_ns=160_200_000_000,
                    ctx_dp_rank=2,
                    timed_out=True,
                    close_confirmed=True,
                    session_absent=True,
                    retired_flag=True,
                ),
                dict(
                    gen,
                    event="gen_timeout_expired",
                    monotonic_ns=int((timer_start + 60) * 1_000_000_000) + 2_000_000,
                    timeout_start_monotonic=timer_start,
                    kv_transfer_timeout_ms=60_000,
                    observed_elapsed_ms=60_001.0,
                ),
            ]
        )
    return events


def _event(events: list[dict], name: str) -> dict:
    """Return the fixture's one event of the requested kind."""
    return next(event for event in events if event["event"] == name)


class ClassificationTests(unittest.TestCase):
    """Reject observations that do not establish the injected ordering."""

    def setUp(self) -> None:
        """Use the specific product failure, not an outer client timeout."""
        self.response = {"status": 400, "body": "Request timed out (KV transfer)"}

    def test_matched_retirement_and_second_timeout(self) -> None:
        """Matched CTX retirement and a full independent GEN deadline qualify."""
        self.assertEqual(
            classify("after_retirement", self.response, _events()), "MECHANISM_REPRODUCED"
        )

    def test_sender_receipt_can_precede_wire_observer(self) -> None:
        """Send-return observation may race receipt in another process."""
        events = _events()
        _event(events, "sender_request_data")["monotonic_ns"] = 161_500_000_000
        self.assertEqual(
            classify("after_retirement", self.response, events), "MECHANISM_REPRODUCED"
        )

    def test_missing_or_repeated_event_is_inconclusive(self) -> None:
        """Every necessary event is unique, with no ambiguous repeated attempt."""
        for omitted in _events():
            with self.subTest(event=omitted["event"]):
                events = [event for event in _events() if event["event"] != omitted["event"]]
                self.assertEqual(
                    classify("after_retirement", self.response, events), "INCONCLUSIVE"
                )
                self.assertEqual(
                    classify("after_retirement", self.response, [*_events(), omitted]),
                    "INCONCLUSIVE",
                )

    def test_invalid_retirement_is_inconclusive(self) -> None:
        """Require actual same-owner removal, not just a timeout observation."""
        for key, value in [
            ("ctx_dp_rank", 3),
            ("pid", 201),
            ("rank", 3),
            ("timed_out", False),
            ("close_confirmed", False),
            ("session_absent", False),
            ("retired_flag", False),
        ]:
            with self.subTest(field=key):
                events = _events()
                _event(events, "ctx_retired")[key] = value
                self.assertEqual(
                    classify("after_retirement", self.response, events), "INCONCLUSIVE"
                )

    def test_mixed_identity_and_retry_evidence_is_inconclusive(self) -> None:
        """Do not silently filter unrelated IDs, hosts, trials or retry modes."""
        for key, value in [
            ("disagg_request_id", 42),
            ("host", "another-node"),
            ("trial_id", "b" * 32),
            ("mode", "control"),
            ("no_retry", False),
        ]:
            with self.subTest(field=key):
                events = _events()
                events[-1][key] = value
                self.assertEqual(
                    classify("after_retirement", self.response, events), "INCONCLUSIVE"
                )

    def test_ambiguous_sender_message_is_inconclusive(self) -> None:
        """Require the expected sender and receiver to observe an absent session."""
        for key, value in [
            ("session_present", True),
            ("fenced", True),
            ("rank", 3),
            ("pid", 201),
            ("peer_rank", 2),
        ]:
            with self.subTest(field=key):
                events = _events()
                _event(events, "sender_request_data")[key] = value
                self.assertEqual(
                    classify("after_retirement", self.response, events), "INCONCLUSIVE"
                )

    def test_invalid_timing_is_inconclusive(self) -> None:
        """GEN publication follows the gate and expires strictly after 60 seconds."""
        for name, key, value in [
            ("gen_request_data_sent", "monotonic_ns", 160_000_000_000),
            ("gen_receive_published", "monotonic_ns", 161_500_000_000),
            ("gen_timeout_started", "monotonic_ns", 163_500_000_000),
            ("gen_timeout_started", "timeout_start_monotonic", 162.0),
            ("ctx_retired", "monotonic_ns", 162_000_000_000),
            ("ctx_native_closed", "monotonic_ns", 161_000_000_000),
            ("sender_request_data", "monotonic_ns", 160_000_000_000),
            ("sender_request_data", "monotonic_ns", 225_000_000_000),
            ("gen_timeout_expired", "monotonic_ns", 224_000_000_000),
            ("gen_timeout_expired", "observed_elapsed_ms", 60_000),
            ("gen_timeout_expired", "observed_elapsed_ms", float("nan")),
            ("gen_timeout_expired", "timeout_start_monotonic", 163.0),
            ("gen_timeout_expired", "kv_transfer_timeout_ms", 59_000),
            ("gen_timeout_expired", "request_id", 100),
        ]:
            with self.subTest(event=name, field=key, value=value):
                events = _events()
                _event(events, name)[key] = value
                self.assertEqual(
                    classify("after_retirement", self.response, events), "INCONCLUSIVE"
                )

    def test_gen_events_must_use_one_worker(self) -> None:
        """A timer in another worker cannot explain this receive publication."""
        for name in (
            "gen_request_data_sent",
            "gen_receive_published",
            "gen_timeout_started",
            "gen_timeout_expired",
        ):
            for key, value in (("pid", 301), ("rank", 0)):
                with self.subTest(event=name, field=key):
                    events = _events()
                    _event(events, name)[key] = value
                    self.assertEqual(
                        classify("after_retirement", self.response, events), "INCONCLUSIVE"
                    )

    def test_invalid_cancelled_or_terminal_trace_is_inconclusive(self) -> None:
        """A rejected gate or terminal sender reply contradicts this mechanism."""
        for name in (
            "gate_invalid",
            "gate_timeout",
            "router_cancelled",
            "router_no_generation",
            "sender_terminal_reply_attempt",
        ):
            with self.subTest(event=name):
                events = _events()
                events.append(dict(events[-1], event=name))
                self.assertEqual(
                    classify("after_retirement", self.response, events), "INCONCLUSIVE"
                )

    def test_unqualified_profile_is_inconclusive(self) -> None:
        """Do not extend single-owner conclusions to unsupported profiles."""
        for key, value in [
            ("pp_size", 2),
            ("cp_size", 2),
            ("attention_dp", False),
            ("pipeline_transfer", True),
            ("bridge_enabled", True),
            ("kv_transfer_timeout_ms", 120_000),
        ]:
            with self.subTest(field=key):
                events = _events()
                _event(events, "ctx_send_published")[key] = value
                self.assertEqual(
                    classify("after_retirement", self.response, events), "INCONCLUSIVE"
                )

    def test_client_timeout_or_unrelated_error_is_inconclusive(self) -> None:
        """An outer timeout or unrelated HTTP error is not a reproduction."""
        for response in [
            {"status": None, "body": "timed out"},
            {"status": 500, "body": "Request timed out (KV transfer)"},
            {"status": 400, "body": "invalid request"},
            {"status": 200, "body": "ok"},
            {"status": 400},
        ]:
            with self.subTest(response=response):
                self.assertEqual(classify("after_retirement", response, _events()), "INCONCLUSIVE")

    def test_early_receiver_control(self) -> None:
        """Control requires success and a receiver message finding a live session."""
        events = _events("control")
        self.assertEqual(classify("control", {"status": 200}, events), "CONTROL_OK")
        _event(events, "sender_request_data")["session_present"] = False
        self.assertEqual(classify("control", {"status": 200}, events), "INCONCLUSIVE")

    def test_control_rejects_timeout_or_invalid_gate(self) -> None:
        """Even HTTP success cannot make a contradictory control usable."""
        for name in ("ctx_retired", "gen_timeout_expired", "gate_invalid"):
            with self.subTest(event=name):
                events = _events("control")
                events.append(dict(events[-1], event=name))
                self.assertEqual(classify("control", {"status": 200}, events), "INCONCLUSIVE")

    def test_malformed_event_is_inconclusive(self) -> None:
        """Missing identity or invalid numeric values fail closed without crashing."""
        for key, value in [
            ("event", None),
            ("host", ""),
            ("pid", True),
            ("monotonic_ns", -1),
            ("disagg_request_id", True),
        ]:
            with self.subTest(field=key):
                events = _events()
                events[0][key] = value
                self.assertEqual(
                    classify("after_retirement", self.response, events), "INCONCLUSIVE"
                )
        self.assertEqual(classify("after_retirement", self.response, []), "INCONCLUSIVE")
        self.assertEqual(classify("after_retirement", self.response, [None]), "INCONCLUSIVE")


if __name__ == "__main__":
    unittest.main()
