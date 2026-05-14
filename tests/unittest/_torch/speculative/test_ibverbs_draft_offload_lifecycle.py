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
"""Unit tests for ``IbverbsDraftOffloadLayer`` (no hardware required).

The transport is faked with a tiny in-process backend that has a
separate outbound queue (target → fake-draft) and inbound queue
(fake-draft → target).  This avoids the trap that ``InMemoryEndpointBackend``
falls into here — its single deque means a sender would read its own
packets if both ends were sharing one backend.
"""

import threading
import time

import pytest

torch = pytest.importorskip("torch")

# Imports below the importorskip are intentionally late so that the entire
# module is skipped on torch-less hosts; ruff E402 doesn't understand the
# pytest importorskip pattern.
from collections import deque  # noqa: E402

from tensorrt_llm._torch.speculative.draft_api_protocol import DraftApiProtocol  # noqa: E402
from tensorrt_llm._torch.speculative.ibverbs_draft_offload import (  # noqa: E402
    LLAMA3_TERMINAL_TOKENS,
    IbverbsDraftOffloadConfig,
    IbverbsDraftOffloadLayer,
)
from tensorrt_llm._torch.speculative.spec_decode_channel import (  # noqa: E402
    EndpointPacket,
    EndpointStatus,
    SpecDecodeChannel,
)


class _LoopbackTargetBackend:
    """Endpoint backend with separate outbound/inbound queues.

    ``send`` pushes to ``_outbox`` (consumed by the bridge thread).
    ``recv`` pops from ``_inbox`` (filled by the bridge thread).
    """

    def __init__(self, max_queue_depth=32):
        self._max = max_queue_depth
        self._started = False
        self._recv_credits = 0
        self._outbox = deque()
        self._inbox = deque()
        self._lock = threading.Lock()

    # EndpointBackend protocol -------------------------------------------------

    def start(self):
        self._started = True
        return EndpointStatus.kOk

    def stop(self):
        self._started = False
        with self._lock:
            self._recv_credits = 0
            self._outbox.clear()
            self._inbox.clear()
        return EndpointStatus.kOk

    def prime_recv(self, count):
        if not self._started:
            return EndpointStatus.kNotStarted
        if count <= 0:
            return EndpointStatus.kInvalidPayload
        with self._lock:
            if self._recv_credits + count > self._max:
                return EndpointStatus.kQueueFull
            self._recv_credits += count
        return EndpointStatus.kOk

    def poll_once(self):
        if not self._started:
            return EndpointStatus.kNotStarted
        with self._lock:
            return EndpointStatus.kOk if self._inbox else EndpointStatus.kEmpty

    def send(self, packet):
        if not self._started:
            return EndpointStatus.kNotStarted
        with self._lock:
            self._outbox.append(
                EndpointPacket(
                    imm_data=int(packet.imm_data),
                    payload=bytes(packet.payload),
                )
            )
        return EndpointStatus.kOk

    def recv(self):
        if not self._started:
            return EndpointStatus.kNotStarted, None
        with self._lock:
            if not self._inbox:
                return EndpointStatus.kEmpty, None
            self._recv_credits = max(0, self._recv_credits - 1)
            return EndpointStatus.kOk, self._inbox.popleft()

    # Bridge helpers used by FakeDraft -----------------------------------------

    def fetch_outbound(self):
        with self._lock:
            if not self._outbox:
                return None
            return self._outbox.popleft()

    def deliver_inbound(self, packet):
        with self._lock:
            self._inbox.append(
                EndpointPacket(
                    imm_data=int(packet.imm_data),
                    payload=bytes(packet.payload),
                )
            )


class _FakeDraftBridge:
    """Reads target outbound, generates a draft-to-target response."""

    def __init__(self, backend, max_draft_len, response_fn=None):
        self._backend = backend
        self._max_draft_len = max_draft_len
        self._response_fn = response_fn or (
            lambda last_token, position, round_seq: list(range(1, max_draft_len + 1))
        )
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self._t.start()

    def stop(self, timeout=2.0):
        self._stop.set()
        self._t.join(timeout)

    def _run(self):
        while not self._stop.is_set():
            packet = self._backend.fetch_outbound()
            if packet is None:
                time.sleep(0.001)
                continue
            ims, imm = SpecDecodeChannel.unpack_imm_data(packet.imm_data)
            ds, msg = DraftApiProtocol.decode(packet.payload)
            assert ims == SpecDecodeChannel.Status.kOk
            assert ds == DraftApiProtocol.Status.kOk
            tokens = self._response_fn(msg.tokens[0], msg.position, msg.round_seq_num)
            tokens = list(tokens) + [0] * (DraftApiProtocol.kMaxTokens - len(tokens))
            response = DraftApiProtocol.Message(
                message_type=DraftApiProtocol.MessageType.kDraftToTarget,
                round_seq_num=int(msg.round_seq_num),
                position=0,
                num_tokens=self._max_draft_len,
                tokens=tokens,
            )
            ris, rimm = SpecDecodeChannel.pack_imm_data(
                SpecDecodeChannel.ImmData(
                    msg_type=int(DraftApiProtocol.MessageType.kDraftToTarget),
                    stream_id=imm.stream_id,
                    slot=imm.slot,
                )
            )
            rs, rpayload = DraftApiProtocol.encode(response)
            assert rs == DraftApiProtocol.Status.kOk
            self._backend.deliver_inbound(
                EndpointPacket(
                    imm_data=rimm,
                    payload=rpayload,
                )
            )


@pytest.fixture
def layer_with_loopback():
    cfg = IbverbsDraftOffloadConfig(max_num_requests=8, max_draft_len=5)
    backend = _LoopbackTargetBackend(max_queue_depth=32)
    layer = IbverbsDraftOffloadLayer(cfg, endpoint=backend)
    layer._ensure_started()
    bridge = _FakeDraftBridge(backend, max_draft_len=5)
    bridge.start()
    yield layer, backend, bridge
    bridge.stop()


def _make_input_tensors(batch_size=1, max_draft_len=5, last_token=42, position=0):
    accepted_tokens = torch.tensor([[last_token] + [0] * max_draft_len], dtype=torch.int32)
    num_accepted_tokens = torch.tensor([1], dtype=torch.int32)
    position_ids = torch.tensor([position], dtype=torch.int64)
    input_ids = torch.tensor([0], dtype=torch.int64)
    return input_ids, position_ids, accepted_tokens, num_accepted_tokens


def test_forward_with_loopback(layer_with_loopback):
    layer, _, _ = layer_with_loopback
    input_ids, position_ids, accepted, num_acc = _make_input_tensors(last_token=99)
    out = layer.forward(
        input_ids=input_ids,
        position_ids=position_ids,
        accepted_tokens=accepted,
        num_accepted_tokens=num_acc,
        batch_size=1,
        num_contexts=0,
        request_ids=[7],
    )
    assert out.shape == (1, 5)
    assert list(out[0].tolist()) == [1, 2, 3, 4, 5]
    assert layer.last_response_token_counts is not None
    assert int(layer.last_response_token_counts[0].item()) == 5


def test_round_seq_increments(layer_with_loopback):
    layer, _, _ = layer_with_loopback
    for round_idx in range(3):
        input_ids, position_ids, accepted, num_acc = _make_input_tensors(
            last_token=10 + round_idx, position=round_idx
        )
        layer.forward(
            input_ids=input_ids,
            position_ids=position_ids,
            accepted_tokens=accepted,
            num_accepted_tokens=num_acc,
            batch_size=1,
            num_contexts=0,
            request_ids=[7],
        )
    assert layer._last_sent_round_seq_by_request[7] == 3
    assert layer._last_recv_round_seq_by_request[7] == 3
    assert 7 not in layer._pending_round_seq_by_request


def test_round_seq_validation_strict():
    cfg = IbverbsDraftOffloadConfig(max_num_requests=4, max_draft_len=5)
    backend = _LoopbackTargetBackend(max_queue_depth=16)
    layer = IbverbsDraftOffloadLayer(cfg, endpoint=backend)
    layer._ensure_started()
    bridge = _FakeDraftBridge(
        backend,
        max_draft_len=5,
        # Bridge sends a wrong round_seq, then we let _FakeDraftBridge.run()
        # use round_seq+99 — emulate by overriding response.
    )

    # Replace the bridge's run with a liar.
    def liar_run():
        while not bridge._stop.is_set():
            packet = backend.fetch_outbound()
            if packet is None:
                time.sleep(0.001)
                continue
            ims, imm = SpecDecodeChannel.unpack_imm_data(packet.imm_data)
            ds, msg = DraftApiProtocol.decode(packet.payload)
            response = DraftApiProtocol.Message(
                message_type=DraftApiProtocol.MessageType.kDraftToTarget,
                round_seq_num=int(msg.round_seq_num) + 99,  # WRONG round_seq
                position=0,
                num_tokens=5,
                tokens=[1, 2, 3, 4, 5] + [0] * 15,
            )
            ris, rimm = SpecDecodeChannel.pack_imm_data(
                SpecDecodeChannel.ImmData(
                    msg_type=int(DraftApiProtocol.MessageType.kDraftToTarget),
                    stream_id=imm.stream_id,
                    slot=imm.slot,
                )
            )
            rs, rpayload = DraftApiProtocol.encode(response)
            backend.deliver_inbound(EndpointPacket(imm_data=rimm, payload=rpayload))

    bridge._t = threading.Thread(target=liar_run, daemon=True)
    bridge.start()
    try:
        input_ids, position_ids, accepted, num_acc = _make_input_tensors()
        with pytest.raises(RuntimeError, match="Round sequence mismatch"):
            layer.forward(
                input_ids=input_ids,
                position_ids=position_ids,
                accepted_tokens=accepted,
                num_accepted_tokens=num_acc,
                batch_size=1,
                num_contexts=0,
                request_ids=[7],
            )
    finally:
        bridge.stop()


def test_terminal_token_short_circuit(layer_with_loopback):
    layer, _, _ = layer_with_loopback
    terminal = next(iter(LLAMA3_TERMINAL_TOKENS))
    input_ids, position_ids, accepted, num_acc = _make_input_tensors(last_token=terminal)
    out = layer.forward(
        input_ids=input_ids,
        position_ids=position_ids,
        accepted_tokens=accepted,
        num_accepted_tokens=num_acc,
        batch_size=1,
        num_contexts=0,
        request_ids=[7],
    )
    assert out.shape == (1, 5)
    assert int(out.sum().item()) == 0
    assert 7 not in layer._bound_requests


def test_bind_unbind_idempotent():
    cfg = IbverbsDraftOffloadConfig(max_num_requests=4, max_draft_len=5)
    backend = _LoopbackTargetBackend(max_queue_depth=16)
    layer = IbverbsDraftOffloadLayer(cfg, endpoint=backend)
    layer.bind_requests([1, 2, 3])
    assert layer._bound_requests == {1, 2, 3}
    layer.bind_requests([1, 2, 3])  # no-op
    assert layer._bound_requests == {1, 2, 3}
    layer.unbind_requests([2])
    assert layer._bound_requests == {1, 3}


def test_retire_keeps_request_skipped(layer_with_loopback):
    layer, _, _ = layer_with_loopback
    layer.retire_requests([7])
    input_ids, position_ids, accepted, num_acc = _make_input_tensors()
    out = layer.forward(
        input_ids=input_ids,
        position_ids=position_ids,
        accepted_tokens=accepted,
        num_accepted_tokens=num_acc,
        batch_size=1,
        num_contexts=0,
        request_ids=[7],
    )
    assert int(out.sum().item()) == 0
    assert layer.last_response_token_counts is not None
    assert int(layer.last_response_token_counts[0].item()) == 0
