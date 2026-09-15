# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared harness for ``DisaggTransferCoordinator`` behavior tests.

The coordinator runs against the contract fake transceiver, a real
``AsyncTransferManager`` and stateful fakes of the executor interfaces, so
tests assert on request state, transfer ownership and what the executor was
asked to do. Multi-rank tests build one harness per rank and share only
``dist``; everything else is rank-local.
"""

from types import SimpleNamespace
from unittest.mock import Mock

from fake_executor_effects import FakeExecutorEffects, FakeRequestRegistry
from fake_kv_cache_transceiver import FakeKvCacheTransceiver

from tensorrt_llm._torch.disaggregation.orchestration.coordinator import DisaggTransferCoordinator
from tensorrt_llm._torch.disaggregation.orchestration.transfer_manager import AsyncTransferManager
from tensorrt_llm._torch.pyexecutor.resource_manager import ResourceManagerType
from tensorrt_llm.bindings import LlmRequestState


class TransferRequest(SimpleNamespace):
    """Request stub with the attributes the transfer paths read."""

    def __init__(self, rid: int, **overrides) -> None:
        defaults = dict(
            py_request_id=rid,
            request_id=rid,
            parent_request_id=None,
            is_child=False,
            state=LlmRequestState.CONTEXT_INIT,
            is_context_only_request=True,
            is_context_finished=True,
            is_finished_due_to_length=False,
            is_finished_due_to_cancellation=False,
            is_disagg_generation_init_state=False,
            is_disagg_generation_transmission_in_progress=False,
            py_kv_transfer_start_time=None,
            py_kv_transfer_timed_out=False,
            py_disaggregated_params=None,
            cached_tokens=0,
            response=None,
        )
        defaults.update(overrides)
        super().__init__(**defaults)
        self.state_at_response_creation = None

    def create_response(self, _use_fast_logits, _rank):
        self.state_at_response_creation = self.state
        return self.response

    @property
    def is_generation_only_request(self) -> bool:
        return not self.is_context_only_request


class CoordinatorHarness:
    """One coordinator with its rank-local collaborators.

    ``dist`` defaults to a ``Mock`` carrying ``rank`` / ``tp_size`` /
    ``world_size`` for single-rank tests; multi-rank tests pass one FakeDist
    rank per harness instead, and the rank object then defines those sizes.
    """

    def __init__(
        self,
        *,
        kv_transfer_timeout_ms=None,
        supports_inflight_cancellation=False,
        enable_attention_dp=False,
        world_size=1,
        tp_size=1,
        force_terminate_ctx_for_partial_reuse=False,
        draft_kv_cache_manager=None,
        dist=None,
    ) -> None:
        self.transceiver = FakeKvCacheTransceiver(
            kv_transfer_timeout_ms=kv_transfer_timeout_ms,
            supports_inflight_cancellation=supports_inflight_cancellation,
        )
        self.transceiver.has_retired_send_session = lambda req: False
        self.kv_cache_manager = Mock(spec=["store_blocks_for_reuse", "unpin_blocks_by_id"])
        self.kv_cache_manager.store_blocks_for_reuse.side_effect = lambda req, _: req.py_request_id
        resource_manager = SimpleNamespace(
            resource_managers={ResourceManagerType.KV_CACHE_MANAGER: self.kv_cache_manager}
        )
        self.transfers = AsyncTransferManager(resource_manager)
        self.active = []
        self.registry = FakeRequestRegistry(self.active)
        self.effects = FakeExecutorEffects()
        self.dist = (
            dist if dist is not None else Mock(rank=0, tp_size=tp_size, world_size=world_size)
        )
        self.delegates = Mock()
        self.coordinator = DisaggTransferCoordinator(
            transceiver=self.transceiver,
            transfer_manager=self.transfers,
            kv_cache_manager=self.kv_cache_manager,
            dist=self.dist,
            effects=self.effects,
            registry=self.registry,
            enable_attention_dp=enable_attention_dp,
            force_terminate_ctx_for_partial_reuse=force_terminate_ctx_for_partial_reuse,
            delegates=self.delegates,
            draft_kv_cache_manager=draft_kv_cache_manager,
        )

    def send(self, *requests: TransferRequest) -> None:
        self.coordinator.send_completed_context(list(requests))

    def in_transfer(self, req: TransferRequest) -> bool:
        return req.py_request_id in self.transfers.requests_in_transfer()
