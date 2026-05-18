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
"""Robustness tests for IbverbsDraftOffloadLayer + ibverbs endpoint.

Covers:
- multi-request with ``_serialize_multi_request=False``
- response timeout when the fake draft never replies
- multiple concurrent slots routed correctly through one backend
"""

import threading
import time
from collections import deque

import pytest

torch = pytest.importorskip("torch")

# E402: imports below importorskip are intentional.
from tensorrt_llm._torch.speculative.draft_api_protocol import DraftApiProtocol  # noqa: E402
from tensorrt_llm._torch.speculative.ibverbs_draft_offload import (  # noqa: E402
    IbverbsDraftOffloadConfig,
    IbverbsDraftOffloadLayer,
)
from tensorrt_llm._torch.speculative.spec_decode_channel import (  # noqa: E402
    EndpointPacket,
    EndpointStatus,
    SpecDecodeChannel,
)


# Reuse the loopback backend from the lifecycle tests.
class _LoopbackTargetBackend:
    def __init__(self, max_queue_depth=64):
        self._max = max_queue_depth
        self._started = False
        self._recv_credits = 0
        self._outbox = deque()
        self._inbox = deque()
        self._lock = threading.Lock()

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

    def fetch_outbound(self):
        with self._lock:
            return self._outbox.popleft() if self._outbox else None

    def deliver_inbound(self, packet):
        with self._lock:
            self._inbox.append(
                EndpointPacket(
                    imm_data=int(packet.imm_data),
                    payload=bytes(packet.payload),
                )
            )


def _start_bridge(backend, response_fn, max_draft_len=5):
    stop = threading.Event()

    def _run():
        while not stop.is_set():
            packet = backend.fetch_outbound()
            if packet is None:
                time.sleep(0.001)
                continue
            ims, imm = SpecDecodeChannel.unpack_imm_data(packet.imm_data)
            ds, msg = DraftApiProtocol.decode(packet.payload)
            tokens = response_fn(msg, imm)
            tokens = list(tokens) + [0] * (DraftApiProtocol.kMaxTokens - len(tokens))
            response = DraftApiProtocol.Message(
                message_type=DraftApiProtocol.MessageType.kDraftToTarget,
                round_seq_num=int(msg.round_seq_num),
                position=0,
                num_tokens=max_draft_len,
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
            backend.deliver_inbound(EndpointPacket(imm_data=rimm, payload=rpayload))

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    return stop, t


def test_response_timeout_raises(monkeypatch):
    """Bridge never replies → forward must raise within timeout."""
    monkeypatch.setenv("TLLM_RDMA_DRAFT_OFFLOAD_RESPONSE_TIMEOUT_S", "0.5")
    cfg = IbverbsDraftOffloadConfig(max_num_requests=4, max_draft_len=5)
    backend = _LoopbackTargetBackend()
    layer = IbverbsDraftOffloadLayer(cfg, endpoint=backend)
    layer._ensure_started()
    # Bridge that just discards everything.
    stop = threading.Event()
    t = threading.Thread(
        target=lambda: (
            time.sleep(0.5),
            stop.set(),  # noqa: E731
        ),
        daemon=True,
    )
    t.start()
    accepted = torch.tensor([[42, 0, 0, 0, 0, 0]], dtype=torch.int32)
    num_acc = torch.tensor([1], dtype=torch.int32)
    pos = torch.tensor([0], dtype=torch.int64)
    inp = torch.tensor([0], dtype=torch.int64)
    start = time.monotonic()
    with pytest.raises(RuntimeError, match="Timed out waiting"):
        layer.forward(
            input_ids=inp,
            position_ids=pos,
            accepted_tokens=accepted,
            num_accepted_tokens=num_acc,
            batch_size=1,
            num_contexts=0,
            request_ids=[7],
        )
    elapsed = time.monotonic() - start
    # Within ~1s — generous bound.
    assert elapsed < 2.0, "timeout took too long: %s" % elapsed


def test_multi_request_parallel_no_serialize(monkeypatch):
    """With _serialize_multi_request=False, layer batches all reqs in one round."""
    monkeypatch.setenv("TLLM_RDMA_DRAFT_OFFLOAD_SERIALIZE_MULTI_REQUEST", "0")
    cfg = IbverbsDraftOffloadConfig(max_num_requests=4, max_draft_len=5)
    backend = _LoopbackTargetBackend()
    layer = IbverbsDraftOffloadLayer(cfg, endpoint=backend)
    layer._ensure_started()
    # Bridge generates per-slot tokens so we can verify routing.
    stop, t = _start_bridge(
        backend,
        response_fn=lambda msg, imm: [int(imm.slot) * 1000 + i for i in range(1, 6)],
        max_draft_len=5,
    )
    try:
        # batch_size=3, three distinct request_ids — should run as one round.
        accepted = torch.tensor(
            [
                [10, 0, 0, 0, 0, 0],
                [20, 0, 0, 0, 0, 0],
                [30, 0, 0, 0, 0, 0],
            ],
            dtype=torch.int32,
        )
        num_acc = torch.tensor([1, 1, 1], dtype=torch.int32)
        pos = torch.tensor([0, 0, 0], dtype=torch.int64)
        inp = torch.tensor([0], dtype=torch.int64)
        out = layer.forward(
            input_ids=inp,
            position_ids=pos,
            accepted_tokens=accepted,
            num_accepted_tokens=num_acc,
            batch_size=3,
            num_contexts=0,
            request_ids=[101, 102, 103],
        )
        assert out.shape == (3, 5)
        # Each row should reflect the slot bound for that request.
        # The layer probes slots starting at request_id & 0x0FFF, so
        # 101 → slot 101, 102 → slot 102, 103 → slot 103.
        for row, req_id in enumerate([101, 102, 103]):
            slot = req_id & 0x0FFF
            expected = [slot * 1000 + i for i in range(1, 6)]
            assert list(out[row].tolist()) == expected, (req_id, out[row].tolist())
    finally:
        stop.set()


def test_round_seq_monotonic_across_rounds(monkeypatch):
    """If a request is served, then served again, round_seq must keep growing."""
    monkeypatch.setenv("TLLM_RDMA_DRAFT_OFFLOAD_RESPONSE_TIMEOUT_S", "5")
    cfg = IbverbsDraftOffloadConfig(max_num_requests=4, max_draft_len=5)
    backend = _LoopbackTargetBackend()
    layer = IbverbsDraftOffloadLayer(cfg, endpoint=backend)
    layer._ensure_started()
    stop, t = _start_bridge(
        backend,
        response_fn=lambda msg, imm: [1, 2, 3, 4, 5],
        max_draft_len=5,
    )
    try:
        for r in range(5):
            accepted = torch.tensor([[100 + r, 0, 0, 0, 0, 0]], dtype=torch.int32)
            num_acc = torch.tensor([1], dtype=torch.int32)
            pos = torch.tensor([r], dtype=torch.int64)
            inp = torch.tensor([0], dtype=torch.int64)
            layer.forward(
                input_ids=inp,
                position_ids=pos,
                accepted_tokens=accepted,
                num_accepted_tokens=num_acc,
                batch_size=1,
                num_contexts=0,
                request_ids=[7],
            )
            assert layer._last_sent_round_seq_by_request[7] == r + 1
            assert layer._last_recv_round_seq_by_request[7] == r + 1
    finally:
        stop.set()
