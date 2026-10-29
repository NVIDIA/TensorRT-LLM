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
"""GPU round trip of a partial KV bounce: two fan-in writers coalesce only their drafter fragments.

Uses real fabric bounce regions and the gather/scatter kernels; the NIXL write is emulated by
copying each descriptor of the request the transport builds.
"""

import threading
from types import SimpleNamespace

import numpy as np
import pytest
import torch

btr = pytest.importorskip("tensorrt_llm._torch.disaggregation.native.bounce.impl")

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")

_MIB = 1024 * 1024
FRAG = 8192  # one writer's heads of one (layer, K/V) plane of a drafter page
FRAGS_PER_PAGE = 4  # 2 layers x K/V
WRITERS = 2
PAGES = 6
DRAFT_BLOCK = WRITERS * FRAGS_PER_PAGE * FRAG  # receiver page: every writer's heads
MLA_BLOCK = 32768


def _copy(dst: int, src: int, nbytes: int) -> None:
    btr.CUASSERT(
        btr.cudart.cudaMemcpy(
            int(dst), int(src), int(nbytes), btr.cudart.cudaMemcpyKind.cudaMemcpyDeviceToDevice
        )
    )


def test_partial_bounce_round_trip(monkeypatch):
    torch.cuda.set_device(0)
    agent = SimpleNamespace(register_memory=lambda d: None, deregister_memory=lambda d: None)
    try:
        transport = btr.VmmBounceTransport(
            agent,
            device_id=0,
            capacity_bytes=64 * _MIB,
            phys_chunk_size=32 * _MIB,
            block_bytes_per_group=[MLA_BLOCK, DRAFT_BLOCK],
            min_bytes=1,
            min_blocks=1,
        )
    except Exception as e:  # fabric VMM unavailable on this node
        pytest.skip(f"bounce regions unavailable: {e}")

    # Capture the descriptor lists the transport builds instead of handing them to NIXL.
    monkeypatch.setattr(
        btr,
        "MemoryDescs",
        SimpleNamespace(
            from_arrays_uniform_device=lambda mem_type, ptrs, sizes, device: (ptrs.copy(), sizes)
        ),
    )
    monkeypatch.setattr(btr, "TransferRequest", lambda op, src, dst, name, msg: (src, dst))

    recv_draft = torch.zeros(PAGES * DRAFT_BLOCK, dtype=torch.uint8, device="cuda")
    recv_mla = torch.zeros(PAGES * MLA_BLOCK, dtype=torch.uint8, device="cuda")
    send_draft = [
        torch.randint(0, 256, (PAGES * FRAGS_PER_PAGE * FRAG,), dtype=torch.uint8, device="cuda")
        for _ in range(WRITERS)
    ]
    send_mla = torch.randint(0, 256, (PAGES * MLA_BLOCK,), dtype=torch.uint8, device="cuda")

    req = SimpleNamespace(
        block_ids_per_layer_groups=[np.arange(PAGES), np.arange(PAGES)],
        unique_rid=5,
        slice_id=0,
        bounce_dst_base=None,
    )
    assert transport.reserve(req, num_writers=WRITERS, layer_groups=[1])
    assert req.bounce_writer_bytes == PAGES * DRAFT_BLOCK // WRITERS
    key = (req.unique_rid, req.slice_id)

    done = threading.Event()
    outcome = []

    def on_done(success):
        outcome.append(success)
        done.set()

    page = np.arange(PAGES, dtype=np.int64)[:, None]
    frag = np.arange(FRAGS_PER_PAGE, dtype=np.int64)[None, :]
    for w in range(WRITERS):
        # A writer's page is contiguous; on the receiver its heads interleave with the other's.
        src = (send_draft[w].data_ptr() + page * FRAGS_PER_PAGE * FRAG + frag * FRAG).ravel()
        dst = (
            recv_draft.data_ptr() + page * DRAFT_BLOCK + frag * (WRITERS * FRAG) + w * FRAG
        ).ravel()
        sizes = np.full(src.size, FRAG, dtype=np.int64)
        num_bounced = src.size
        if w == 0:  # one writer also sends the target pages, in place
            mla = page.ravel() * MLA_BLOCK
            src = np.r_[src, send_mla.data_ptr() + mla]
            dst = np.r_[dst, recv_mla.data_ptr() + mla]
            sizes = np.r_[sizes, np.full(PAGES, MLA_BLOCK, dtype=np.int64)]
        write_meta = SimpleNamespace(
            src_ptrs=src,
            dst_ptrs=dst,
            sizes=sizes,
            bounce_dst_base=transport.writer_base(key, w),
            dst_device_id=0,
            peer_name="gen0",
            bounce_num_frags=num_bounced,
            bounce_writer_bytes=req.bounce_writer_bytes,
        )
        ((src_descs, sizes_out), (dst_descs, _)), slot_id = transport.build_request(write_meta)
        assert src_descs.size == 1 + (PAGES if w == 0 else 0)
        for s, d, n in zip(src_descs, dst_descs, sizes_out):  # what NIXL would write
            _copy(d, s, n)
        transport.release_send(slot_id)
        dst_ptrs, frag_sizes, src_base = btr.decode_result_tail(
            [b"KV_AGENT_RESULT", b"prefix"] + btr.encode_result_tail(write_meta)
        )
        assert frag_sizes.size == num_bounced
        transport.record_result(key, w, dst_ptrs, frag_sizes, src_base, on_done)

    assert done.wait(timeout=30), "the scatter never completed"
    assert outcome == [True]
    torch.cuda.synchronize()
    expected = torch.stack(
        [s.view(PAGES, FRAGS_PER_PAGE, FRAG) for s in send_draft], dim=2
    )  # [page, plane, writer, bytes]
    assert torch.equal(recv_draft.view(PAGES, FRAGS_PER_PAGE, WRITERS, FRAG), expected)
    assert torch.equal(recv_mla, send_mla)
    assert not transport.is_bounced(key)
    transport.close()
