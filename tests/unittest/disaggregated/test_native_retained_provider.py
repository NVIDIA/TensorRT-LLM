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
"""Actual-module CPU provider tests; run in the allocated candidate runtime."""

import json
import pickle
import threading
import time
import unittest
from dataclasses import FrozenInstanceError, asdict
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch

import pytest
from pydantic import TypeAdapter, ValidationError

from tensorrt_llm._torch.disaggregation.native.retained_coverage import VERSION
from tensorrt_llm._torch.pyexecutor.llm_request import LlmResult
from tensorrt_llm._torch.pyexecutor.native_retained_prefix import (
    NativeRetainedPrefix,
    _Command,
    _Export,
)
from tensorrt_llm._torch.pyexecutor.py_executor import PyExecutor
from tensorrt_llm.disaggregated_params import DisaggregatedParams
from tensorrt_llm.executor.result import GenerationResult
from tensorrt_llm.retained_prefix import RetainedPrefixParams, parse_retained_prefix
from tensorrt_llm.sampling_params import SamplingParams

pytestmark = pytest.mark.cpu_only


def message(action="claim", **changes):
    result = {
        "action": action,
        "protocol": VERSION,
        "capability": asdict(capability()),
        "identity": "3" * 64,
        "nonce": "a" * 32,
    }
    if action == "claim":
        result["destination_peer"] = "prefill"
    elif action == "admit":
        result["certificates"] = ["certificate"]
    return result | changes


def capability():
    return RetainedPrefixParams(1, "donor", "1" * 32, "2" * 64, 32, "3" * 64)


class WireContract(unittest.TestCase):
    def test_copy_immutability_and_nested_request_parse(self):
        wire = asdict(capability())
        params = DisaggregatedParams(retain_prefix=True, retained_prefix=wire)
        wire["identity"] = "4" * 64
        self.assertEqual(capability(), params.retained_prefix)
        with self.assertRaises(FrozenInstanceError):
            params.retained_prefix.prefix_tokens = 64

    def test_malformed_and_raw_coordinates_reject(self):
        for changes in (
            {"address": 12},
            {"endpoint": "tcp://evil"},
            {"version": True},
            {"prefix_tokens": True},
            {"prefix_tokens": 0},
            {"identity": "bad"},
            {"peer_id": "tcp://evil"},
        ):
            wire = asdict(capability()) | changes
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                parse_retained_prefix(wire)

    def test_actual_result_ipc_preserves_typed_export(self):
        result = LlmResult(b"", NS(), True)
        result.retained_prefix = capability()
        restored = pickle.loads(pickle.dumps(result))
        self.assertEqual(capability(), restored.retained_prefix)
        self.assertTrue(restored.is_final)

    def test_consumed_input_reference_is_not_echoed_as_an_export(self):
        params = DisaggregatedParams(retain_prefix=True, retained_prefix=capability())
        request = NS(
            id=7,
            sampling_params=SamplingParams(max_tokens=1),
            postproc_params=None,
            streaming=False,
            trace_headers=None,
        )
        result = GenerationResult(request, disaggregated_params=params)
        self.assertIsNone(result.disaggregated_params.retained_prefix)
        self.assertTrue(result.disaggregated_params.retain_prefix)
        self.assertEqual(capability(), params.retained_prefix)

    def test_pydantic_json_ingress_cannot_discard_raw_coordinates_or_coerce_bool(self):
        adapter = TypeAdapter(RetainedPrefixParams)
        self.assertEqual(capability(), adapter.validate_python(asdict(capability())))
        for changed in ({"endpoint": "tcp://evil"}, {"version": True}, {"prefix_tokens": True}):
            with self.subTest(changed=changed), self.assertRaises(ValidationError):
                adapter.validate_python(asdict(capability()) | changed)

    def test_existing_guard_uses_resolved_gpu_tiers_even_with_falsey_host_config(self):
        from tensorrt_llm.runtime.kv_cache_manager_v2 import CacheTier

        manager = NS(
            host_cache_size=None,
            disk_cache_size=0,
            impl=NS(init_config=NS(cache_tiers=[NS(tier=CacheTier.GPU_MEM)])),
        )
        self.assertIsNone(PyExecutor._reject_non_gpu_cache_tiers(manager))
        manager.impl.init_config.cache_tiers.append(NS(tier=CacheTier.HOST_MEM))
        with self.assertRaisesRegex(NotImplementedError, "cache tiers below GPU"):
            PyExecutor._reject_non_gpu_cache_tiers(manager)


class Ownership(unittest.TestCase):
    def provider(self):
        provider = object.__new__(NativeRetainedPrefix)
        provider.generation = "1" * 32
        provider.config = NS(model_identity="m", max_claims_per_export=1, max_exports=8)
        provider.layout = "layout"
        provider.rank = 0
        provider.world_size = 1
        provider.worker = Mock()
        provider.executor = NS(
            is_shutdown=False, async_transfer_manager=Mock(), _terminate_request=Mock()
        )
        provider.executor.async_transfer_manager.end_transfer.return_value = True
        provider.transceiver = NS(_context_info_endpoint="trusted-source-endpoint")
        provider.exports = {}
        provider.retired_exports = {}
        provider.pending_imports = {}
        provider.quarantined = []
        provider.lock = threading.Lock()
        provider.commands = __import__("queue").Queue()
        provider.stopping = threading.Event()
        return provider

    def test_expiry_never_releases_active_native_source(self):
        provider = self.provider()
        session = Mock()
        session.wait_complete.return_value = None
        session.resources_drained.return_value = False
        export = _Export(
            capability(), NS(), NS(), time.monotonic() - 1, claims={"nonce": (NS(), session)}
        )
        provider.exports[capability().export_id] = export
        provider.poll()
        session.close.assert_not_called()
        provider.executor.async_transfer_manager.end_transfer.assert_not_called()
        provider.executor._terminate_request.assert_not_called()
        self.assertTrue(provider.pending)

    def test_release_requires_successful_close_after_actual_drain(self):
        provider = self.provider()
        session = Mock()
        session.wait_complete.return_value = object()
        session.resources_drained.return_value = True
        session.close.return_value = False
        export = _Export(
            capability(), NS(), NS(), time.monotonic() - 1, claims={"nonce": (NS(), session)}
        )
        provider.exports[capability().export_id] = export
        provider.poll()
        provider.executor.async_transfer_manager.end_transfer.assert_not_called()
        session.close.return_value = True
        provider.poll()
        provider.executor.async_transfer_manager.end_transfer.assert_called_once_with(
            export.request
        )
        provider.executor._terminate_request.assert_called_once_with(export.request)

    def test_actual_executor_termination_respects_destination_quarantine(self):
        provider = self.provider()
        request = NS()
        provider.quarantined = [(request, NS(), NS(), NS())]
        executor = NS(native_retained_prefix=provider, _do_terminate_request=Mock())
        PyExecutor._terminate_request(executor, request)
        executor._do_terminate_request.assert_not_called()

    def test_expired_foreign_generation_and_forged_identity_never_contact_native(self):
        for supplied, identity in (
            (capability(), "4" * 64),
            (RetainedPrefixParams(1, "donor", "9" * 32, "2" * 64, 32, "3" * 64), "3" * 64),
        ):
            provider = self.provider()
            provider.exports[capability().export_id] = _Export(
                capability(), NS(), NS(), time.monotonic() + 20
            )
            with patch(
                "tensorrt_llm._torch.pyexecutor.native_retained_prefix.create_retained_tx_session"
            ) as create:
                with self.assertRaises(ValueError):
                    provider._claim(
                        message("admit", capability=asdict(supplied), identity=identity)
                    )
                create.assert_not_called()

    def test_claim_budget_and_replay_do_not_create_extra_native_sessions(self):
        provider = self.provider()
        export = _Export(capability(), NS(), NS(), time.monotonic() + 20, closed_claims={"a" * 32})
        provider.exports[capability().export_id] = export
        with patch(
            "tensorrt_llm._torch.pyexecutor.native_retained_prefix.create_retained_tx_session"
        ) as create:
            for nonce in ("a" * 32, "b" * 32):
                with self.assertRaises((ValueError, RuntimeError)):
                    provider._claim(message("admit", nonce=nonce))
            create.assert_not_called()


class ControlAdmission(Ownership):
    def test_protocol_and_coordinates_fail_before_native_construction(self):
        provider = self.provider()
        provider.exports[capability().export_id] = _Export(
            capability(), NS(), NS(), time.monotonic() + 10
        )
        for supplied in (
            message(protocol="old"),
            message(endpoint="tcp://injected"),
            message(nonce="bad"),
            message("admit", certificates=[]),
        ):
            with (
                self.subTest(supplied=supplied),
                patch(
                    "tensorrt_llm._torch.pyexecutor.native_retained_prefix.create_retained_tx_session"
                ) as native,
            ):
                with self.assertRaises(ValueError):
                    provider._claim(supplied)
                native.assert_not_called()

    def test_explicit_finish_closes_admission_but_preserves_inflight_source(self):
        provider = self.provider()
        session = Mock()
        session.wait_complete.return_value = None
        export = _Export(
            capability(),
            NS(),
            NS(),
            time.monotonic() + 10,
            claims={"a" * 32: (NS(), session)},
            admitted=True,
        )
        provider.exports[capability().export_id] = export
        self.assertFalse(provider._finish(message("finish"))["expired_used"])
        self.assertTrue(export.retired)
        with self.assertRaises(ValueError):
            provider._check_claim(message())
        provider.poll()
        session.close.assert_not_called()
        provider.executor.async_transfer_manager.end_transfer.assert_not_called()
        session.wait_complete.return_value = object()
        session.resources_drained.return_value = True
        session.close.return_value = True
        provider.poll()
        self.assertFalse(provider.pending)
        tombstone = provider.retired_exports[capability().export_id]
        self.assertEqual(frozenset({"a" * 32}), tombstone[1])
        self.assertFalse(tombstone[2]["a" * 32]["expired_used"])

    def test_sibling_finish_keeps_expiry_accounting_and_individual_receipts(self):
        provider = self.provider()
        provider.config.max_claims_per_export = 2
        export = _Export(
            capability(), NS(), NS(), 10, closed_claims={"a" * 32, "b" * 32}, admitted=True
        )
        provider.exports[capability().export_id] = export
        with patch.object(time, "monotonic", return_value=1):
            first = provider._finish(message("finish"))
            self.assertFalse(first["expired_used"])
            provider.poll()
            provider.executor.async_transfer_manager.end_transfer.assert_not_called()
        with patch.object(time, "monotonic", return_value=11):
            late = provider._finish(message("finish", nonce="b" * 32))
            self.assertTrue(late["expired_used"])
            self.assertEqual(first, provider._finish(message("finish")))
            provider.poll()
        receipts = provider.retired_exports[capability().export_id][2]
        self.assertFalse(receipts["a" * 32]["expired_used"])
        self.assertTrue(receipts["b" * 32]["expired_used"])

    def test_slow_preparation_cannot_hide_expiry_from_next_queued_finish(self):
        provider = self.provider()
        provider.config.max_claims_per_export = 2
        session = Mock()
        session.wait_complete.return_value = None
        export = _Export(
            capability(), NS(), NS(), 10, claims={"a" * 32: (NS(), session)}, admitted=True
        )
        provider.exports[capability().export_id] = export
        clock = [1]

        def prepare(*args):
            clock[0] = 11
            return {"prepared": True}

        provider._prepare_claim = Mock(side_effect=prepare)
        pending = [_Command(message(nonce="b" * 32)), _Command(message("finish"))]
        for command in pending:
            provider.commands.put(command)
        with patch.object(time, "monotonic", side_effect=lambda: clock[0]):
            provider.poll()
        self.assertTrue(pending[1].response["expired_used"])
        provider.executor.async_transfer_manager.end_transfer.assert_not_called()

    def test_partial_or_complete_local_prefix_falls_back_without_remote_write_or_cursor_change(
        self,
    ):
        for local_hits in (32, 64):
            with self.subTest(local_hits=local_hits):
                provider = self.provider()
                retained = RetainedPrefixParams(1, "donor", "1" * 32, "2" * 64, 64, "3" * 64)
                request = NS(
                    py_request_id=7,
                    py_disaggregated_params=NS(retained_prefix=retained),
                    context_current_position=local_hits,
                    context_chunk_size=96 - local_hits,
                    py_retained_loaded=False,
                )
                provider.executor.kv_cache_manager = NS(
                    kv_cache_map={7: NS(num_committed_tokens=local_hits)}
                )
                provider._call_source = Mock()
                self.assertFalse(provider.import_prefix(request))
                provider._call_source.assert_not_called()
                self.assertEqual(local_hits, request.context_current_position)
                self.assertEqual(96 - local_hits, request.context_chunk_size)
                self.assertTrue(request.py_retained_loaded)
                self.assertFalse(provider.quarantined)
                self.assertFalse(hasattr(request, "py_retained_import_receipt"))

    def test_unqualified_encoder_and_custom_auxiliary_state_reject_before_transport(self):
        for encoder, extra in ((True, {}), (False, {"custom_state": object()})):
            executor = NS(
                model_engine=NS(model=NS(model_config=NS(is_encoder_decoder=encoder))),
                llm_args=NS(extra_resource_managers=extra),
                dist=NS(rank=0, world_size=1),
            )
            with (
                self.subTest(encoder=encoder),
                self.assertRaisesRegex(ValueError, "retained all-rank admission failed"),
            ):
                NativeRetainedPrefix(executor, NS())

    def test_absent_native_session_is_not_logged_as_drained_and_closed(self):
        provider = self.provider()
        provider.exports[capability().export_id] = _Export(
            capability(),
            NS(),
            NS(),
            time.monotonic() - 1,
            claims={"a" * 32: (NS(disagg_request_id=71), None)},
        )
        captured = Mock()
        with patch.dict(NativeRetainedPrefix.poll.__globals__, {"logger": NS(info=captured)}):
            provider.poll()
        events = [
            json.loads(call.args[0].removeprefix("retained_prefix "))
            for call in captured.call_args_list
        ]
        receipt = next(event for event in events if event["event"] == "source_claim_closed")
        self.assertFalse(receipt["native_session_created"])
        self.assertTrue(receipt["native_session_not_started"])
        self.assertFalse(receipt["physical_resources_drained"])
        self.assertFalse(receipt["native_session_closed"])
        self.assertIsNone(receipt["native_outcome"])

    def test_provider_calls_resolved_tier_guard_before_retirement_or_identity(self):
        provider = self.provider()
        mapping = NS(
            rank=0,
            world_size=1,
            tp_size=1,
            pp_size=1,
            cp_size=1,
            dp_size=1,
            enable_attention_dp=False,
        )
        manager = NS(mapping=mapping, enable_block_reuse=True)
        guard = Mock(side_effect=NotImplementedError("resolved HOST tier"))
        executor = NS(
            model_engine=NS(model=NS(model_config=NS(is_encoder_decoder=False))),
            llm_args=NS(extra_resource_managers={}),
            kv_cache_transceiver=NS(_transfer_worker=object()),
            kv_cache_manager=manager,
            _reject_non_gpu_cache_tiers=guard,
        )
        with self.assertRaisesRegex(NotImplementedError, "resolved HOST tier"):
            provider._initialize_local_prerequisites(executor, NS())
        guard.assert_called_once_with(manager)

    def test_completed_unadmitted_export_cannot_be_finished(self):
        provider = self.provider()
        provider.exports[capability().export_id] = _Export(
            capability(), NS(), NS(), time.monotonic() + 10
        )
        with self.assertRaises(ValueError):
            provider._finish(message("finish"))
        self.assertFalse(provider.exports[capability().export_id].retired)


class _Collectives:
    """Two rank bounded rendezvous; mismatched collective order fails loudly."""

    def __init__(self):
        self.barrier = threading.Barrier(2, timeout=3)
        self.values = [None, None]
        self.kinds = [None, None]
        self.trace = [[], []]

    def call(self, rank, kind, value):
        self.values[rank] = value
        self.kinds[rank] = kind
        self.trace[rank].append(kind)
        self.barrier.wait()
        if self.kinds[0] != self.kinds[1]:
            raise AssertionError("different all-rank collective ordering")
        answer = list(self.values) if kind == "gather" else self.values[0]
        self.barrier.wait()
        return answer

    def dist(self, rank):
        return NS(
            rank=rank,
            world_size=2,
            allgather=lambda value: self.call(rank, "gather", value),
            broadcast=lambda value, root: self.call(rank, "broadcast", value),
        )


class AllRankLifetime(Ownership):
    def pair(self):
        rendezvous = _Collectives()
        providers = [self.provider(), self.provider()]
        for rank, provider in enumerate(providers):
            provider.rank = rank
            provider.world_size = 2
            provider.executor.dist = rendezvous.dist(rank)
        return rendezvous, providers

    def run_pair(self, providers, method):
        failures = []

        def run(provider):
            try:
                method(provider)
            except BaseException as error:
                failures.append(error)

        threads = [
            threading.Thread(target=run, args=(provider,), daemon=True) for provider in providers
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(5)
            self.assertFalse(thread.is_alive(), "all-rank frame deadlocked")
        return failures

    def test_release_waits_for_every_rank_and_uses_only_leader_expiry(self):
        rendezvous, providers = self.pair()
        for rank, provider in enumerate(providers):
            session = Mock()
            session.wait_complete.return_value = object()
            session.resources_drained.return_value = True
            session.close.return_value = rank == 0
            # Rank one's future local clock must not override leader expiry.
            provider.exports[capability().export_id] = _Export(
                capability(),
                NS(),
                NS(),
                time.monotonic() + (-10 if rank == 0 else 10000),
                claims={"a" * 32: (NS(), session)},
            )
        self.assertEqual([], self.run_pair(providers, lambda provider: provider.poll()))
        for provider in providers:
            provider.executor.async_transfer_manager.end_transfer.assert_not_called()
            self.assertTrue(provider.exports[capability().export_id].expired)
        providers[1].exports[capability().export_id].claims["a" * 32][1].close.return_value = True
        self.assertEqual([], self.run_pair(providers, lambda provider: provider.poll()))
        for provider in providers:
            provider.executor.async_transfer_manager.end_transfer.assert_called_once()
            self.assertFalse(provider.pending)
        self.assertEqual(rendezvous.trace[0], rendezvous.trace[1])

    def test_rank_local_startup_prerequisites_vote_before_identity_or_transport(self):
        rendezvous = _Collectives()
        executors = []
        for rank in range(2):
            mapping = NS(
                rank=rank,
                world_size=2,
                tp_size=2,
                pp_size=1,
                cp_size=1,
                dp_size=1,
                enable_attention_dp=False,
            )
            executors.append(
                NS(
                    dist=rendezvous.dist(rank),
                    model_engine=NS(model=NS(model_config=NS(is_encoder_decoder=rank == 0))),
                    llm_args=NS(extra_resource_managers={}),
                    kv_cache_transceiver=NS(_transfer_worker=object()),
                    kv_cache_manager=NS(mapping=mapping, enable_block_reuse=False),
                )
            )
        errors = self.run_pair(executors, lambda executor: NativeRetainedPrefix(executor, NS()))
        self.assertEqual(2, len(errors))
        for error in errors:
            self.assertIn("encoder/cross-attention", str(error))
            self.assertIn("restored-KV correctness restriction", str(error))
        self.assertEqual([["gather"], ["gather"]], rendezvous.trace)

    def test_rank_local_retirement_and_allocator_errors_fail_stop_in_same_order(self):
        for operation in (
            "wait_complete",
            "resources_drained",
            "close",
            "end_transfer",
            "terminate",
        ):
            with self.subTest(operation=operation):
                rendezvous, providers = self.pair()
                for rank, provider in enumerate(providers):

                    def fail(message, executor=provider.executor):
                        executor._kv_connector_failed = True
                        raise RuntimeError(message)

                    provider.executor._kv_connector_fail = fail
                    session = Mock()
                    session.wait_complete.return_value = object()
                    session.resources_drained.return_value = True
                    session.close.return_value = True
                    provider.exports[capability().export_id] = _Export(
                        capability(),
                        NS(),
                        NS(),
                        time.monotonic() - 1,
                        claims={"a" * 32: (NS(), session)},
                    )
                    if rank == 1:
                        target = (
                            provider.executor.async_transfer_manager.end_transfer
                            if operation == "end_transfer"
                            else provider.executor._terminate_request
                            if operation == "terminate"
                            else getattr(session, operation)
                        )
                        target.side_effect = RuntimeError("injected rank1 " + operation)
                errors = self.run_pair(providers, lambda provider: provider.poll())
                self.assertEqual(2, len(errors))
                self.assertTrue(
                    all("injected rank1 " + operation in str(error) for error in errors)
                )
                for provider in providers:
                    self.assertTrue(provider.executor._kv_connector_failed)
                    self.assertIn(capability().export_id, provider.exports)
                    if operation in ("wait_complete", "resources_drained", "close"):
                        provider.executor.async_transfer_manager.end_transfer.assert_not_called()
                self.assertEqual(rendezvous.trace[0], rendezvous.trace[1])

    def test_rank_divergent_admission_metadata_fails_before_nested_factory_collectives(self):
        rendezvous, providers = self.pair()
        for rank, provider in enumerate(providers):

            def fail(message, executor=provider.executor):
                executor._kv_connector_failed = True
                raise RuntimeError(message)

            provider.executor._kv_connector_fail = fail
            export = _Export(capability(), NS(), NS(), time.monotonic() + 10)
            if rank == 0:
                export.prepared["a" * 32] = {"certificates": ["certificate"]}
            provider.exports[capability().export_id] = export
        errors = self.run_pair(providers, lambda provider: provider._claim(message("admit")))
        self.assertEqual(2, len(errors))
        self.assertTrue(all("admission metadata diverged" in str(error) for error in errors))
        self.assertEqual(rendezvous.trace[0], rendezvous.trace[1])

    def test_two_rank_sibling_finishes_preserve_per_claim_ttl_receipts(self):
        rendezvous, providers = self.pair()
        for provider in providers:
            provider.config.max_claims_per_export = 2
            provider.exports[capability().export_id] = _Export(
                capability(), NS(), NS(), 10, closed_claims={"a" * 32, "b" * 32}, admitted=True
            )
        first, late, replay = {}, {}, {}
        with patch.object(time, "monotonic", return_value=1):
            self.assertEqual(
                [],
                self.run_pair(
                    providers,
                    lambda provider: first.update(
                        {provider.rank: provider._finish(message("finish"))}
                    ),
                ),
            )
            self.assertEqual([], self.run_pair(providers, lambda provider: provider.poll()))
            for provider in providers:
                provider.executor.async_transfer_manager.end_transfer.assert_not_called()
        with patch.object(time, "monotonic", return_value=11):
            self.assertEqual(
                [],
                self.run_pair(
                    providers,
                    lambda provider: late.update(
                        {provider.rank: provider._finish(message("finish", nonce="b" * 32))}
                    ),
                ),
            )
            self.assertEqual(
                [],
                self.run_pair(
                    providers,
                    lambda provider: replay.update(
                        {provider.rank: provider._finish(message("finish"))}
                    ),
                ),
            )
            self.assertEqual([], self.run_pair(providers, lambda provider: provider.poll()))
        for rank in range(2):
            self.assertFalse(first[rank]["expired_used"])
            self.assertTrue(late[rank]["expired_used"])
            self.assertEqual(first[rank], replay[rank])
        self.assertEqual(rendezvous.trace[0], rendezvous.trace[1])

    def test_two_rank_slow_prepare_refreshes_before_queued_finish(self):
        rendezvous, providers = self.pair()
        clock = [1]
        for provider in providers:
            provider.config.max_claims_per_export = 2
            session = Mock()
            session.wait_complete.return_value = None
            provider.exports[capability().export_id] = _Export(
                capability(), NS(), NS(), 10, claims={"a" * 32: (NS(), session)}, admitted=True
            )

            def prepare(*args):
                clock[0] = 11
                return {"prepared": True}

            provider._prepare_claim = Mock(side_effect=prepare)
        commands = [_Command(message(nonce="b" * 32)), _Command(message("finish"))]
        for command in commands:
            providers[0].commands.put(command)
        with patch.object(time, "monotonic", side_effect=lambda: clock[0]):
            self.assertEqual([], self.run_pair(providers, lambda provider: provider.poll()))
        self.assertTrue(commands[1].response["expired_used"])
        self.assertEqual(rendezvous.trace[0], rendezvous.trace[1])

    def test_empty_and_different_completion_frames_are_all_rank(self):
        rendezvous, providers = self.pair()
        self.assertEqual(
            [],
            self.run_pair(
                providers, lambda provider: provider.prepare_exports(NS(generation_requests=[]))
            ),
        )
        self.assertEqual([["gather"], ["gather"]], rendezvous.trace)
        request = NS(
            py_request_id=8,
            py_disaggregated_params=NS(retain_prefix=True),
            state=object(),
            py_retained_export=None,
        )
        errors = self.run_pair(
            providers,
            lambda provider: provider.prepare_exports(
                NS(generation_requests=[request] if provider.rank == 0 else [])
            ),
        )
        self.assertEqual(2, len(errors))
        self.assertTrue(all("completion frame differs" in str(error) for error in errors))


if __name__ == "__main__":
    unittest.main(verbosity=2)
