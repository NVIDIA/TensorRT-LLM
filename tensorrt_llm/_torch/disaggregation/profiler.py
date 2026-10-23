# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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
"""Request-scoped, observational KV transfer correlation, independent of transport."""

import weakref
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

from tensorrt_llm import _profiler
from tensorrt_llm._torch.disaggregation.base.transfer import get_unique_rid
from tensorrt_llm._torch.disaggregation.kv_cache_transceiver import (
    BindKvCacheTransceiver,
    CtxTransferStatus,
    GenTransferStatus,
    KvCacheTransceiver,
    SendTerminalOutcome,
)
from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequest
from tensorrt_llm.bindings import LlmRequestState

_Direction = Literal["send", "receive"]


@dataclass
class _KvTransferRecord:
    request: LlmRequest | weakref.ReferenceType[LlmRequest]
    key: tuple[int, _Direction]
    request_id: int
    transfer_id: int
    status_id: int
    role: Literal["context", "generation"]
    direction: _Direction
    state_at_submission: LlmRequestState
    metadata: str | None = None
    admission_uncertain: bool = False

    def get_request(self) -> LlmRequest | None:
        return self.request() if isinstance(self.request, weakref.ReferenceType) else self.request


class _ProfiledKvCacheTransceiver(KvCacheTransceiver):
    """Delegate the current ABC while recording submit and terminal observations.

    Args:
        transceiver: The factory-selected native or Python resource owner.
        backend: Its resolved transport name, used only in metadata.
        rank: Its instance rank, used only in metadata.

    No method adds polling or collectives. Unknown admission exceptions retain
    weak request observations only; request expiry drops observation, not native
    resources, and is not recorded as safe cancellation.
    """

    def __init__(self, transceiver: KvCacheTransceiver, backend: str, rank: int) -> None:
        self._transceiver = transceiver
        self._backend = backend
        self._rank = rank
        self._records: dict[tuple[int, _Direction], _KvTransferRecord] = {}

    @property
    def kv_transfer_timeout_ms(self) -> int | None:
        return self._transceiver.kv_transfer_timeout_ms

    @property
    def pipeline_transfer_enabled(self) -> bool:
        return self._transceiver.pipeline_transfer_enabled

    @property
    def consumes_transfer_buffer(self) -> bool:
        return self._transceiver.consumes_transfer_buffer

    def has_inflight_transfer(self, req: LlmRequest) -> bool:
        return self._transceiver.has_inflight_transfer(req)

    def has_retired_send_session(self, req: LlmRequest) -> bool:
        return self._transceiver.has_retired_send_session(req)

    def transfer_resource_ownership(self, req: LlmRequest) -> bool | None:
        return self._transceiver.transfer_resource_ownership(req)

    def observe_send_terminal_outcome(self, req: LlmRequest, enabled: bool = True) -> None:
        self._transceiver.observe_send_terminal_outcome(req, enabled)

    def get_send_terminal_outcome(self, req: LlmRequest) -> SendTerminalOutcome | None:
        return self._transceiver.get_send_terminal_outcome(req)

    def supports_inflight_request_cancellation(self) -> bool:
        return self._transceiver.supports_inflight_request_cancellation()

    def has_poisoned_transfer_buffer(self) -> bool:
        return self._transceiver.has_poisoned_transfer_buffer()

    def _register(self, req: LlmRequest, direction: _Direction) -> _KvTransferRecord:
        request_id = req.request_id
        params = req.py_disaggregated_params
        transfer_id = request_id
        if params is not None:
            if params.disagg_request_id is not None:
                transfer_id = params.disagg_request_id
            elif params.ctx_request_id is not None:
                transfer_id = params.ctx_request_id
        # Native status uses mRequestId; V2 uses its disaggregated local handle.
        # Peer ctx_request_id is metadata, never a substitute for that handle.
        status_id = (
            request_id
            if isinstance(self._transceiver, BindKvCacheTransceiver)
            else get_unique_rid(req)
        )
        if status_id is None:
            status_id = request_id
        record = _KvTransferRecord(
            req,
            (id(req), direction),
            request_id,
            transfer_id,
            status_id,
            "context" if direction == "send" else "generation",
            direction,
            req.state,
        )
        for old in list(self._records.values()):
            if (
                old.admission_uncertain
                and old.direction == direction
                and old.status_id == status_id
            ):
                # Numeric status cannot distinguish the uncertain older generation.
                # Discard only its observation, without terminal or release claims.
                self._discard(old)
        if direction == "send":
            self._transceiver.observe_send_terminal_outcome(req)
        self._records[record.key] = record
        return record

    def _metadata(self, record: _KvTransferRecord) -> str:
        if record.metadata is None:
            req = record.get_request()
            if req is None:
                raise RuntimeError("KV observation request expired")
            record.metadata = _profiler._encode_metadata(
                "kv",
                {
                    "request_id": record.request_id,
                    "transfer_id": record.transfer_id,
                    "aliases": {
                        "request": record.request_id,
                        "transfer": record.transfer_id,
                        "status": record.status_id,
                    },
                    "role": record.role,
                    "direction": record.direction,
                    "backend": self._backend,
                    "rank": self._rank,
                    "prompt_tokens": req.prompt_len,
                },
            )
        return record.metadata

    def _discard(self, record: _KvTransferRecord) -> None:
        if self._records.get(record.key) is not record:
            return
        self._records.pop(record.key)
        req = record.get_request()
        if record.direction == "send" and req is not None:
            try:
                self._transceiver.observe_send_terminal_outcome(req, enabled=False)
            except Exception:
                # Only observational cleanup is discarded; native ownership is unchanged.
                pass

    def _finish(self, record: _KvTransferRecord, outcome: SendTerminalOutcome) -> None:
        if self._records.get(record.key) is not record:
            return
        try:
            with _profiler.record_function_metadata_scope(
                f"trtllm::kv_transfer_{outcome}", "kv", lambda: self._metadata(record)
            ):
                pass
        finally:
            self._discard(record)

    def _submission_failed(self, record: _KvTransferRecord) -> None:
        req = record.get_request()
        if req is None:
            self._discard(record)
            return
        try:
            ownership = self._transceiver.transfer_resource_ownership(req)
        except Exception:
            # Failed observation cannot turn unknown admission into safe rejection.
            ownership = None
        if ownership is False:
            self._finish(record, "failed")
        elif ownership is None:
            record.admission_uncertain = True
            # Keep no strong failed-request owner when native admission is unknown.
            # Native owners/callers still determine release and request lifetime.
            try:
                record.request = weakref.ref(req, lambda reference: self._discard(record))
            except TypeError:
                # A non-weak-referenceable request cannot support bounded observation.
                self._discard(record)

    def _submit(
        self,
        req: LlmRequest,
        direction: _Direction,
        submit: Callable[[LlmRequest], None],
        *,
        synchronous: bool = False,
    ) -> None:
        existing = self._records.get((id(req), direction))
        if existing is not None:
            return submit(req)
        try:
            record = self._register(req, direction)
        except Exception:
            # Identity/accessor setup is observational; always perform admission.
            return submit(req)
        name = "trtllm::kv_transfer_sync" if synchronous else "trtllm::kv_transfer_submit"
        try:
            with _profiler.record_function_metadata_scope(
                name, "kv", lambda: self._metadata(record)
            ):
                result = submit(req)
        except BaseException:
            # Preserve the admission exception, even after partial resource admission.
            self._submission_failed(record)
            raise
        if synchronous or req.state == LlmRequestState.DISAGG_TRANS_ERROR:
            self._observe_request_terminal(record)
        return result

    def respond_and_send_async(self, req: LlmRequest) -> None:
        return self._submit(req, "send", self._transceiver.respond_and_send_async)

    def request_and_receive_async(self, req: LlmRequest) -> None:
        return self._submit(req, "receive", self._transceiver.request_and_receive_async)

    def request_and_receive_sync(self, req: LlmRequest) -> None:
        return self._submit(
            req, "receive", self._transceiver.request_and_receive_sync, synchronous=True
        )

    def _observe_request_terminal(self, record: _KvTransferRecord) -> None:
        req = record.get_request()
        if req is None:
            self._discard(record)
        elif req.state in (
            LlmRequestState.DISAGG_GENERATION_TRANS_COMPLETE,
            LlmRequestState.DISAGG_TRANS_ERROR,
        ):
            if record.admission_uncertain and req.state == record.state_at_submission:
                return
            # V2 terminal state can precede a refused close; its owner decides release.
            try:
                owned = self._transceiver.transfer_resource_ownership(req)
            except Exception:
                return
            if owned is not True:
                outcome: SendTerminalOutcome = (
                    "complete"
                    if req.state == LlmRequestState.DISAGG_GENERATION_TRANS_COMPLETE
                    else "failed"
                )
                self._finish(record, outcome)

    def check_context_transfer_status(
        self,
        at_least_request_num: int | None,
        mark_complete: bool = False,
    ) -> CtxTransferStatus:
        result = self._transceiver.check_context_transfer_status(
            at_least_request_num, mark_complete
        )
        completed = set(result.completed_request_ids)
        failed = set(result.error_request_ids)
        for record in list(self._records.values()):
            if record.direction != "send":
                continue
            req = record.get_request()
            if req is None:
                self._discard(record)
                continue
            try:
                outcome = self._transceiver.get_send_terminal_outcome(req)
            except Exception:
                # Status/result preservation takes priority over observational extraction.
                continue
            if outcome is not None:
                self._finish(record, outcome)
            elif record.status_id in completed:
                self._finish(record, "complete")
            elif record.status_id in failed:
                self._finish(record, "failed")
        return result

    def check_gen_transfer_status(self, at_least_request_num: int | None) -> GenTransferStatus:
        result = self._transceiver.check_gen_transfer_status(at_least_request_num)
        cancelled = {id(req) for req in result.cancelled_requests}
        completed = set(result.completed_request_ids)
        failed = set(result.error_request_ids)
        for record in list(self._records.values()):
            if record.direction != "receive":
                continue
            if record.key[0] in cancelled:
                self._finish(record, "cancelled")
            elif record.status_id in completed:
                self._finish(record, "complete")
            elif record.status_id in failed:
                self._finish(record, "failed")
            else:
                self._observe_request_terminal(record)
        return result

    def check_gen_transfer_complete(self) -> bool:
        return self._transceiver.check_gen_transfer_complete()

    def cancel_request(self, req: LlmRequest) -> bool:
        cancelled = self._transceiver.cancel_request(req)
        if cancelled:
            directions: tuple[_Direction, ...] = ("send", "receive")
            for direction in directions:
                record = self._records.get((id(req), direction))
                if record is not None:
                    self._finish(record, "cancelled")
        return cancelled

    def prepare_context_requests(self, requests: list[LlmRequest]) -> None:
        return self._transceiver.prepare_context_requests(requests)

    def get_disaggregated_params(self) -> dict[str, object]:
        return self._transceiver.get_disaggregated_params()

    def commit_blocks_for_reuse(self, req: LlmRequest) -> None:
        return self._transceiver.commit_blocks_for_reuse(req)

    def get_data_transceiver_state(self) -> bytes:
        return self._transceiver.get_data_transceiver_state()

    def get_status_dump(self) -> str:
        return self._transceiver.get_status_dump()

    def shutdown(self) -> None:
        result = self._transceiver.shutdown()
        for record in list(self._records.values()):
            self._finish(record, "cancelled")
        return result
