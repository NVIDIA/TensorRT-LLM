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
"""Unit tests for the block/page-granular HiSparse swap-in op.

``trtllm::hisparse_swap_in_blocks`` is a TensorRT-LLM port of SGLang HiSparse's
``load_cache_to_device_buffer_kernel`` reinterpreted at KV-page granularity (one
MiniMax-M3 sparse block == one KV page) with separate K and V paged pools. These
tests reuse SGLang's exact hit / newest / LRU / miss scenarios (their integer
vectors are granularity-agnostic) and additionally assert that a missed page's
raw K and V bytes are copied host->device.
"""

import pytest
import torch

import tensorrt_llm  # noqa: F401  (loads libth_common.so and registers trtllm ops)

DEVICE = "cuda"
ITEM_BYTES = 32  # bytes per KV page item; multiple of 16 for the v2.b64 copy path
HOT_BUFFER_SIZE = 4
PADDED_BUFFER_SIZE = HOT_BUFFER_SIZE + 1  # +1 reserved "newest" slot
HOST_CACHE_PAGES = 16
DEVICE_CACHE_PAGES = 16
CUDA_BLOCK_SIZE = 128

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or not hasattr(torch.ops.trtllm, "hisparse_swap_in_blocks"),
    reason="HiSparse swap-in op requires a CUDA build of libth_common.so.",
)


def _page_pool(pages: int, salt: int) -> torch.Tensor:
    """Pinned host page pool with distinct, deterministic bytes per page."""
    base = torch.arange(ITEM_BYTES, dtype=torch.int64).view(1, ITEM_BYTES)
    page = torch.arange(pages, dtype=torch.int64).view(pages, 1)
    data = ((page * 7 + base + salt) % 256).to(torch.uint8)
    out = torch.empty((pages, ITEM_BYTES),
                      dtype=torch.uint8,
                      device="cpu",
                      pin_memory=True)
    out.copy_(data)
    return out


def _run_op(
    *,
    top_k_blocks: torch.Tensor,
    device_buffer_blocks: torch.Tensor,
    host_block_locs: torch.Tensor,
    device_buffer_locs: torch.Tensor,
    host_cache_k: torch.Tensor,
    host_cache_v: torch.Tensor,
    device_buffer_k: torch.Tensor,
    device_buffer_v: torch.Tensor,
    lru_slots: torch.Tensor,
    seq_len_blocks: int,
    seq_lens_dtype: torch.dtype = torch.int32,
    req_pool_indices: torch.Tensor = None,
    num_real_reqs: int = None,
    output_fill_value: int = -1,
) -> torch.Tensor:
    batch = top_k_blocks.shape[0]
    if req_pool_indices is None:
        req_pool_indices = torch.arange(batch, dtype=torch.int64, device=DEVICE)
    seq_lens = torch.full((batch, ),
                          seq_len_blocks,
                          dtype=seq_lens_dtype,
                          device=DEVICE)
    if num_real_reqs is None:
        num_real_reqs = batch
    out = torch.full_like(top_k_blocks, output_fill_value)
    torch.ops.trtllm.hisparse_swap_in_blocks(
        top_k_blocks,
        device_buffer_blocks,
        host_block_locs,
        device_buffer_locs,
        host_cache_k,
        host_cache_v,
        device_buffer_k,
        device_buffer_v,
        out,
        req_pool_indices,
        seq_lens,
        lru_slots,
        torch.tensor([num_real_reqs], dtype=torch.int32, device=DEVICE),
        top_k_blocks.shape[1],  # num_top_k
        HOT_BUFFER_SIZE,
        ITEM_BYTES,
        CUDA_BLOCK_SIZE,
    )
    torch.cuda.synchronize()
    return out


def _make_state(device_buffer_locs_rows, device_buffer_tokens_rows,
                newest_tokens):
    """Build a one-layer swap-in state, populating resident K/V pages.

    Mirrors SGLang test's ``_make_state``: slots 0..HOT_BUFFER_SIZE-1 participate
    in the LRU; slot HOT_BUFFER_SIZE is the reserved newest slot.
    """
    host_cache_k = _page_pool(HOST_CACHE_PAGES, salt=0)
    host_cache_v = _page_pool(HOST_CACHE_PAGES, salt=100)
    device_buffer_k = torch.zeros((DEVICE_CACHE_PAGES, ITEM_BYTES),
                                  dtype=torch.uint8,
                                  device=DEVICE)
    device_buffer_v = torch.zeros((DEVICE_CACHE_PAGES, ITEM_BYTES),
                                  dtype=torch.uint8,
                                  device=DEVICE)
    device_buffer_locs = torch.tensor(device_buffer_locs_rows,
                                      dtype=torch.int32,
                                      device=DEVICE)
    device_buffer_blocks = torch.tensor(device_buffer_tokens_rows,
                                        dtype=torch.int32,
                                        device=DEVICE)
    batch = device_buffer_locs.shape[0]
    lru_slots = (torch.arange(HOT_BUFFER_SIZE, dtype=torch.int16,
                              device=DEVICE).view(1, -1).repeat(batch,
                                                                1).contiguous())
    host_block_locs = (torch.arange(
        HOST_CACHE_PAGES, dtype=torch.int64,
        device=DEVICE).view(1, -1).repeat(batch, 1).contiguous())

    for rid, newest_token in enumerate(newest_tokens):
        for slot, token in enumerate(
                device_buffer_tokens_rows[rid][:HOT_BUFFER_SIZE]):
            if token >= 0:
                loc = int(device_buffer_locs[rid, slot])
                device_buffer_k[loc].copy_(host_cache_k[token].to(DEVICE))
                device_buffer_v[loc].copy_(host_cache_v[token].to(DEVICE))
        newest_loc = int(device_buffer_locs[rid, HOT_BUFFER_SIZE])
        device_buffer_k[newest_loc].copy_(host_cache_k[newest_token].to(DEVICE))
        device_buffer_v[newest_loc].copy_(host_cache_v[newest_token].to(DEVICE))
    torch.cuda.synchronize()

    return {
        "host_cache_k": host_cache_k,
        "host_cache_v": host_cache_v,
        "device_buffer_k": device_buffer_k,
        "device_buffer_v": device_buffer_v,
        "device_buffer_locs": device_buffer_locs,
        "device_buffer_blocks": device_buffer_blocks,
        "lru_slots": lru_slots,
        "host_block_locs": host_block_locs,
    }


def _long_case():
    # req 0 LRU slots      : [0, 1, 2, 3]
    # req 0 cached blocks  : slot0->1, slot1->4, slot2->2, slot3->5
    # req 0 physical locs  : slot0->9, slot1->7, slot2->3, slot3->5
    # req 0 newest slot    : slot4/newest -> block 7 at physical loc 11
    return _make_state([[9, 7, 3, 5, 11]], [[1, 4, 2, 5, -1]], [7])


@pytest.mark.parametrize("seq_lens_dtype", [torch.int32, torch.int64])
def test_fast_path(seq_lens_dtype):
    """seq_len <= hot_buffer_size: direct index into device_buffer_locs, no IO."""
    host_cache_k = _page_pool(HOST_CACHE_PAGES, salt=0)
    host_cache_v = _page_pool(HOST_CACHE_PAGES, salt=100)
    device_buffer_k = torch.arange(DEVICE_CACHE_PAGES * ITEM_BYTES,
                                   dtype=torch.uint8,
                                   device=DEVICE).view(DEVICE_CACHE_PAGES,
                                                       ITEM_BYTES)
    device_buffer_v = device_buffer_k.clone()
    k_before, v_before = device_buffer_k.clone(), device_buffer_v.clone()

    device_buffer_locs = torch.tensor([[13, 9, 5, 1, 15]],
                                      dtype=torch.int32,
                                      device=DEVICE)
    device_buffer_blocks = torch.tensor([[10, 11, 12, 13, -1]],
                                        dtype=torch.int32,
                                        device=DEVICE)
    blocks_before = device_buffer_blocks.clone()
    lru_slots = torch.tensor([[0, 1, 2, 3]], dtype=torch.int16, device=DEVICE)
    lru_before = lru_slots.clone()

    out = _run_op(
        top_k_blocks=torch.tensor([[2, 0, 1]],
                                  dtype=torch.int32,
                                  device=DEVICE),
        device_buffer_blocks=device_buffer_blocks,
        host_block_locs=torch.arange(HOST_CACHE_PAGES,
                                     dtype=torch.int64,
                                     device=DEVICE).view(1, -1),
        device_buffer_locs=device_buffer_locs,
        host_cache_k=host_cache_k,
        host_cache_v=host_cache_v,
        device_buffer_k=device_buffer_k,
        device_buffer_v=device_buffer_v,
        lru_slots=lru_slots,
        seq_len_blocks=3,
        seq_lens_dtype=seq_lens_dtype,
    )

    assert torch.equal(out.cpu(), torch.tensor([[5, 13, 9]],
                                               dtype=torch.int32))
    assert torch.equal(device_buffer_blocks.cpu(), blocks_before.cpu())
    assert torch.equal(lru_slots.cpu(), lru_before.cpu())
    assert torch.equal(device_buffer_k.cpu(), k_before.cpu())
    assert torch.equal(device_buffer_v.cpu(), v_before.cpu())


def test_fast_path_overwrites_stale_output():
    state = _make_state([[9, 7, 3, 5, 11]], [[0, 1, 2, 3, -1]], [4])
    out = _run_op(
        top_k_blocks=torch.tensor([[1, -1, 0, 0]],
                                  dtype=torch.int32,
                                  device=DEVICE),
        seq_len_blocks=2,
        output_fill_value=123456,
        **state,
    )
    assert torch.equal(out.cpu(),
                       torch.tensor([[7, -1, -1, -1]], dtype=torch.int32))


def test_hits_newest_and_updates_lru():
    state = _make_state([[9, 7, 3, 5, 11]], [[1, 4, 2, 5, -1]], [7])
    k_before = state["device_buffer_k"].clone()
    v_before = state["device_buffer_v"].clone()

    # Query [4, 2, 7]: 4 hits slot1->loc7, 2 hits slot2->loc3, 7 is newest->loc11.
    # Hits move to MRU tail, so LRU [0,1,2,3] becomes [0,3,1,2]. No misses => no IO.
    out = _run_op(
        top_k_blocks=torch.tensor([[4, 2, 7]],
                                  dtype=torch.int32,
                                  device=DEVICE),
        seq_len_blocks=8,
        **state,
    )

    assert torch.equal(out.cpu(), torch.tensor([[7, 3, 11]],
                                               dtype=torch.int32))
    assert torch.equal(state["device_buffer_blocks"].cpu(),
                       torch.tensor([[1, 4, 2, 5, -1]], dtype=torch.int32))
    assert torch.equal(state["lru_slots"].cpu(),
                       torch.tensor([[0, 3, 1, 2]], dtype=torch.int16))
    assert torch.equal(state["device_buffer_k"].cpu(), k_before.cpu())
    assert torch.equal(state["device_buffer_v"].cpu(), v_before.cpu())


def test_miss_copies_page_and_updates_lru():
    state = _make_state([[9, 7, 3, 5, 11]], [[1, 4, 2, 5, -1]], [7])

    # Step 1: touch blocks [4, 2] -> LRU becomes [0, 3, 1, 2].
    _run_op(
        top_k_blocks=torch.tensor([[4, 2]], dtype=torch.int32, device=DEVICE),
        seq_len_blocks=8,
        **state,
    )
    # Step 2: query block 6 -> miss. Reuses new LRU head slot0 (physical loc 9).
    out = _run_op(
        top_k_blocks=torch.tensor([[6]], dtype=torch.int32, device=DEVICE),
        seq_len_blocks=8,
        **state,
    )

    assert torch.equal(out.cpu(), torch.tensor([[9]], dtype=torch.int32))
    assert torch.equal(state["device_buffer_blocks"].cpu(),
                       torch.tensor([[6, 4, 2, 5, -1]], dtype=torch.int32))
    assert torch.equal(state["lru_slots"].cpu(),
                       torch.tensor([[3, 1, 2, 0]], dtype=torch.int16))

    # The missed page (block 6, host page 6) must be copied host->device at loc 9,
    # independently for K and V (distinct salts guard against K/V mix-ups).
    assert torch.equal(state["device_buffer_k"][9].cpu(),
                       state["host_cache_k"][6])
    assert torch.equal(state["device_buffer_v"][9].cpu(),
                       state["host_cache_v"][6])


def test_padded_request_returns_invalid():
    """CUDA-graph padding: rows >= num_real_reqs must early-return all -1."""
    state = _make_state([[9, 7, 3, 5, 11], [9, 7, 3, 5, 11]],
                        [[1, 4, 2, 5, -1], [1, 4, 2, 5, -1]], [7, 7])
    out = _run_op(
        top_k_blocks=torch.tensor([[4, 2], [4, 2]],
                                  dtype=torch.int32,
                                  device=DEVICE),
        seq_len_blocks=8,
        num_real_reqs=1,  # second request is padding
        **state,
    )
    # Real request resolves its hits; padded request is all -1.
    assert torch.equal(out[0].cpu(), torch.tensor([7, 3], dtype=torch.int32))
    assert torch.equal(out[1].cpu(), torch.tensor([-1, -1], dtype=torch.int32))
