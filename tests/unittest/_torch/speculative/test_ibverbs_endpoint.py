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
"""Hardware tests for ``_IbverbsRdmaBackend``.

Skips when no RDMA NIC is present.  When present, exercises both
single-NIC and cross-NIC loopback at 96-byte WRITE_WITH_IMM granularity.
"""

import os
import socket
import threading
import time

import pytest

from tensorrt_llm._torch.speculative.draft_api_protocol import DraftApiProtocol
from tensorrt_llm._torch.speculative.ibverbs_endpoint import (
    IbverbsEndpointConfig,
    _IbverbsRdmaBackend,
)
from tensorrt_llm._torch.speculative.spec_decode_channel import SpecDecodeChannel


def _have_rdma():
    return os.path.isdir("/dev/infiniband") and os.access("/dev/infiniband", os.R_OK)


pytestmark = pytest.mark.skipif(not _have_rdma(), reason="needs /dev/infiniband")


def _free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def _start_pair(server_nic, client_nic, max_num_requests=8):
    port = _free_port()
    server_cfg = IbverbsEndpointConfig(
        nic_name=server_nic,
        is_server=True,
        local_port=port,
        remote_host="127.0.0.1",
        remote_port=port,
        max_num_requests=max_num_requests,
        recv_queue_depth=max_num_requests,
    )
    client_cfg = IbverbsEndpointConfig(
        nic_name=client_nic,
        is_server=False,
        remote_host="127.0.0.1",
        remote_port=port,
        max_num_requests=max_num_requests,
        recv_queue_depth=max_num_requests,
    )
    sb = _IbverbsRdmaBackend(server_cfg)
    cb = _IbverbsRdmaBackend(client_cfg)
    sc = SpecDecodeChannel(sb)
    cc = SpecDecodeChannel(cb)

    server_err = []

    def server_main():
        try:
            assert sc.start() == SpecDecodeChannel.Status.kOk
        except Exception as exc:
            server_err.append(exc)

    t = threading.Thread(target=server_main, daemon=True)
    t.start()
    time.sleep(0.05)
    assert cc.start() == SpecDecodeChannel.Status.kOk
    t.join(5.0)
    if server_err:
        raise server_err[0]
    return sc, cc, sb, cb


def _make_t2d(token, round_seq=1, position=0):
    return DraftApiProtocol.Message(
        message_type=DraftApiProtocol.MessageType.kTargetToDraft,
        round_seq_num=round_seq,
        position=position,
        num_tokens=1,
        tokens=[token] + [0] * 19,
    )


def _make_d2t(tokens, round_seq=1, position=0):
    pad = list(tokens) + [0] * (20 - len(tokens))
    return DraftApiProtocol.Message(
        message_type=DraftApiProtocol.MessageType.kDraftToTarget,
        round_seq_num=round_seq,
        position=position,
        num_tokens=len(tokens),
        tokens=pad,
    )


def _wait_pump(channel, timeout_s=5.0):
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        ps, rid, rcv = channel.pump_once_for_bound_request()
        if ps == SpecDecodeChannel.Status.kOk and rcv is not None:
            return rid, rcv
        time.sleep(0.001)
    raise TimeoutError("pump timed out")


@pytest.mark.parametrize("server_nic,client_nic", [("mlx5_0", "mlx5_0"), ("mlx5_0", "mlx5_1")])
def test_endpoint_pair_smoke(server_nic, client_nic):
    sc, cc, sb, cb = _start_pair(server_nic, client_nic)
    try:
        assert sc.prime_recv(8) == SpecDecodeChannel.Status.kOk
        assert cc.prime_recv(8) == SpecDecodeChannel.Status.kOk
        assert (
            sc.bind_request_route(7, SpecDecodeChannel.Route(0, 0)) == SpecDecodeChannel.Status.kOk
        )
        assert (
            cc.bind_request_route(7, SpecDecodeChannel.Route(0, 0)) == SpecDecodeChannel.Status.kOk
        )

        # Client → Server
        cc.send_for_request(
            7, int(DraftApiProtocol.MessageType.kTargetToDraft), _make_t2d(token=42)
        )
        rid, rcv = _wait_pump(sc)
        assert rid == 7
        assert rcv.message.tokens[0] == 42

        # Server → Client
        sc.send_for_request(
            7,
            int(DraftApiProtocol.MessageType.kDraftToTarget),
            _make_d2t([100, 200, 300, 400, 500]),
        )
        rid, rcv = _wait_pump(cc)
        assert rid == 7
        assert list(rcv.message.tokens[:5]) == [100, 200, 300, 400, 500]
    finally:
        sc.stop()
        cc.stop()


def test_endpoint_pump_after_recv_keeps_credits_stable():
    """Repeated send/pump cycles shouldn't drain recv credits to zero."""
    sc, cc, sb, cb = _start_pair("mlx5_0", "mlx5_1", max_num_requests=4)
    try:
        sc.prime_recv(4)
        cc.prime_recv(4)
        sc.bind_request_route(7, SpecDecodeChannel.Route(0, 0))
        cc.bind_request_route(7, SpecDecodeChannel.Route(0, 0))
        for r in range(8):
            cc.send_for_request(
                7,
                int(DraftApiProtocol.MessageType.kTargetToDraft),
                _make_t2d(token=5000 + r, round_seq=r),
            )
            rid, rcv = _wait_pump(sc)
            assert rid == 7
            assert rcv.message.tokens[0] == 5000 + r
            # Credits should be stable.
            assert sb._recv_credits >= 3
    finally:
        sc.stop()
        cc.stop()


def test_endpoint_multi_slot_routing():
    """Different slot ↔ different request_id; routing must stay correct."""
    sc, cc, sb, cb = _start_pair("mlx5_0", "mlx5_1", max_num_requests=8)
    try:
        sc.prime_recv(8)
        cc.prime_recv(8)
        for req_id, slot in [(101, 0), (102, 1), (103, 2)]:
            sc.bind_request_route(req_id, SpecDecodeChannel.Route(0, slot))
            cc.bind_request_route(req_id, SpecDecodeChannel.Route(0, slot))

        # Client sends to all three, then drain.
        for req_id in [101, 102, 103]:
            cc.send_for_request(
                req_id,
                int(DraftApiProtocol.MessageType.kTargetToDraft),
                _make_t2d(token=req_id),
            )
        seen = {}
        for _ in range(3):
            rid, rcv = _wait_pump(sc)
            assert rcv.message.tokens[0] == rid, (rid, rcv.message.tokens[0])
            seen[rid] = rcv.message.tokens[0]
        assert sorted(seen) == [101, 102, 103]
    finally:
        sc.stop()
        cc.stop()
