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
"""trtllm-gen decode for MiniMax-M3's dense attention layers.

Two things are under test and they fail differently. The pool geometry
(``get_kv_subpage_pool``) is pure addressing against a real cache manager: if
the flat sub-page view disagrees with ``get_buffers``, the kernel silently
reads another layer's cache. The kernel call itself is checked against a
PyTorch oracle through a stub manager, so the flashinfer argument conventions
are exercised without standing up a model.
"""

from __future__ import annotations

import math
from typing import List, Optional

import pytest
import torch

from tensorrt_llm._torch.attention_backend.sparse.minimax_m3.msa_utils import (
    msa_ported_decode_active,
)
from tensorrt_llm._torch.attention_backend.sparse.minimax_m3.trtllm_gen_dense_decode import (
    _subpage_block_table,
    dense_decode_sm_scale,
    dense_decode_supported,
    minimax_m3_trtllm_gen_dense_decode,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")

PAGE_SIZE = 32
HEAD_DIM = 128


def _is_sm100f() -> bool:
    major, minor = torch.cuda.get_device_capability()
    return major == 10 and minor in (0, 3)


# --------------------------------------------------------------------------
# Pool geometry against a real MiniMaxM3KVCacheManagerV2
# --------------------------------------------------------------------------


def _create_manager(tp_size: int, sparse_layers: List[int], num_layers: int = 4):
    from tensorrt_llm import Mapping
    from tensorrt_llm._torch.attention_backend.sparse.minimax_m3 import MiniMaxM3KVCacheManagerV2
    from tensorrt_llm.bindings import DataType
    from tensorrt_llm.bindings.internal.batch_manager import CacheType as CacheTypeCpp
    from tensorrt_llm.llmapi.llm_args import KvCacheConfig

    max_num_tokens = 2048
    return MiniMaxM3KVCacheManagerV2(
        kv_cache_config=KvCacheConfig(
            enable_block_reuse=False,
            max_tokens=max_num_tokens,
            event_buffer_max_size=0,
            dtype="auto",
        ),
        kv_cache_type=CacheTypeCpp.SELF,
        num_layers=num_layers,
        num_kv_heads=2,
        head_dim=HEAD_DIM,
        tokens_per_block=PAGE_SIZE,
        max_seq_len=512,
        max_batch_size=4,
        mapping=Mapping(world_size=tp_size, rank=0, tp_size=tp_size, pp_size=1),
        dtype=DataType.BF16,
        vocab_size=1024,
        max_num_tokens=max_num_tokens,
        sparse_layer_ids=list(sparse_layers),
        disable_index_value_layer_ids=list(sparse_layers),
        sparse_index_dim=HEAD_DIM,
    )


@pytest.mark.parametrize(
    "tp_size,sparse_layers",
    [
        # TP=2 makes K == V == INDEX_KEY bytes per block, so V2 coalesces
        # index-K into the K/V pool and the per-layer stride goes non-uniform.
        # That is the layout build_trtllm_gen_kv_cache_metadata cannot express.
        pytest.param(2, [1, 3], id="coalesced-index-k"),
        pytest.param(1, [1, 3], id="separate-index-pool"),
        pytest.param(2, [], id="all-dense"),
    ],
)
@pytest.mark.parametrize("kv_layout", ["HND", "NHD"])
def test_subpage_pool_addresses_match_get_buffers(tp_size, sparse_layers, kv_layout):
    """flat[s * scale + {0,1}] must be exactly this layer's K and V at slot s."""
    manager = _create_manager(tp_size, sparse_layers)
    try:
        for layer_idx in range(4):
            kv = manager.get_buffers(layer_idx, kv_layout=kv_layout)
            flat, scale = manager.get_kv_subpage_pool(layer_idx, kv_layout)

            assert flat.is_contiguous()
            assert list(flat.shape[1:]) == list(kv.shape[2:])
            assert flat.dtype == kv.dtype
            num_slots = int(kv.shape[0])
            assert int(flat.shape[0]) == (num_slots - 1) * scale + 2

            for slot in (0, 1, num_slots // 2, num_slots - 1):
                assert flat[slot * scale].data_ptr() == kv[slot, 0].data_ptr()
                assert flat[slot * scale + 1].data_ptr() == kv[slot, 1].data_ptr()
    finally:
        manager.shutdown()


def test_subpage_pool_stops_at_the_last_slots_v():
    """The tail bound matters: every layer but the first starts mid-slot, so a
    naive num_slots * scale view would run off the end of the pool."""
    manager = _create_manager(2, [1, 3])
    try:
        for layer_idx in range(4):
            flat, scale = manager.get_kv_subpage_pool(layer_idx, "HND")
            kv = manager.get_buffers(layer_idx, kv_layout="HND")
            last_v = kv[int(kv.shape[0]) - 1, 1]
            flat_end = flat.data_ptr() + flat.numel() * flat.element_size()
            v_end = last_v.data_ptr() + last_v.numel() * last_v.element_size()
            assert flat_end == v_end
    finally:
        manager.shutdown()


# --------------------------------------------------------------------------
# Block-table expansion
# --------------------------------------------------------------------------


def test_subpage_block_table_splits_k_and_v_rows():
    slots = torch.tensor([[0, 3, 7], [2, 5, 11]], device="cuda", dtype=torch.int32)
    table = _subpage_block_table(slots, subpages_per_slot=9, reserve=False)

    assert table.shape == (2, 2, 3)
    assert table.dtype == torch.int32
    assert table[:, 0].tolist() == [[0, 27, 63], [18, 45, 99]]
    assert table[:, 1].tolist() == [[1, 28, 64], [19, 46, 100]]


def test_subpage_block_table_reuses_one_buffer():
    """All dense layers share the arena block, so a later call must not alias
    a live earlier one within a step; they are written before every use."""
    slots_a = torch.zeros((2, 4), device="cuda", dtype=torch.int32)
    slots_b = torch.full((2, 4), 5, device="cuda", dtype=torch.int32)
    first = _subpage_block_table(slots_a, 4, reserve=False)
    second = _subpage_block_table(slots_b, 4, reserve=False)

    assert first.data_ptr() == second.data_ptr()
    assert second[:, 0].tolist() == [[20] * 4] * 2


# --------------------------------------------------------------------------
# Kernel parity against a PyTorch oracle
# --------------------------------------------------------------------------


class _StubManager:
    """Presents a flat sub-page pool the way MiniMaxM3KVCacheManagerV2 does."""

    def __init__(self, pool: torch.Tensor, subpages_per_slot: int):
        self._pool = pool
        self._scale = subpages_per_slot

    def get_kv_subpage_pool(self, layer_idx: int, kv_layout: str = "HND"):
        assert kv_layout == "HND"
        return self._pool, self._scale


def _reference_dense_decode(
    q: torch.Tensor,  # [total_q, num_heads, head_dim]
    pool: torch.Tensor,  # [num_subpages, num_kv_heads, page, head_dim]
    subpages_per_slot: int,
    block_table: torch.Tensor,  # [batch, max_blocks]
    seq_lens: torch.Tensor,  # [batch]
    decode_query_len: int,
    sm_scale: float,
) -> torch.Tensor:
    pool_f32 = pool.float()
    num_heads = q.shape[1]
    num_kv_heads = pool.shape[1]
    group = num_heads // num_kv_heads
    out = torch.zeros_like(q, dtype=torch.float32)
    positions = torch.arange(PAGE_SIZE, device=q.device)

    for req in range(block_table.shape[0]):
        kv_len_full = int(seq_lens[req])
        num_pages = (kv_len_full + PAGE_SIZE - 1) // PAGE_SIZE
        slots = [int(block_table[req, p]) for p in range(num_pages)]
        for intra in range(decode_query_len):
            token = req * decode_query_len + intra
            kv_len = kv_len_full - decode_query_len + intra + 1
            for kv_head in range(num_kv_heads):
                keys = torch.cat([pool_f32[s * subpages_per_slot, kv_head] for s in slots])
                values = torch.cat([pool_f32[s * subpages_per_slot + 1, kv_head] for s in slots])
                valid = torch.cat([p * PAGE_SIZE + positions < kv_len for p in range(num_pages)])
                q_rows = q[token, kv_head * group : (kv_head + 1) * group].float()
                logits = (q_rows @ keys.T) * sm_scale
                logits = logits.masked_fill(~valid[None, :], -float("inf"))
                probs = torch.softmax(logits, dim=-1)
                out[token, kv_head * group : (kv_head + 1) * group] = probs @ values
    return out


def _make_pool_inputs(
    seq_lens: List[int],
    num_heads: int,
    num_kv_heads: int,
    decode_query_len: int,
    subpages_per_slot: int,
    dtype: torch.dtype,
    seed: int = 0,
):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    batch = len(seq_lens)
    max_blocks = max((s + PAGE_SIZE - 1) // PAGE_SIZE for s in seq_lens)
    num_slots = batch * max_blocks + 3

    pool = torch.randn(
        (num_slots * subpages_per_slot, num_kv_heads, PAGE_SIZE, HEAD_DIM),
        device="cuda",
        generator=generator,
        dtype=torch.float32,
    ).to(dtype)
    # Slots are handed out non-contiguously, exactly as the block manager does.
    perm = torch.randperm(num_slots, generator=generator, device="cuda")[: batch * max_blocks]
    block_table = perm.reshape(batch, max_blocks).to(torch.int32)

    total_q = batch * decode_query_len
    q = torch.randn(
        (total_q, num_heads, HEAD_DIM), device="cuda", generator=generator, dtype=torch.float32
    ).to(torch.bfloat16)
    return q, pool, block_table, torch.tensor(seq_lens, device="cuda", dtype=torch.int32)


def _run_dense(q, pool, subpages_per_slot, block_table, seq_lens, decode_query_len):
    out = torch.zeros_like(q, dtype=torch.bfloat16)
    sm_scale = dense_decode_sm_scale(HEAD_DIM, 1.0)
    minimax_m3_trtllm_gen_dense_decode(
        q,
        _StubManager(pool, subpages_per_slot),
        0,
        block_table,
        seq_lens,
        sm_scale=sm_scale,
        output=out,
        decode_query_len=decode_query_len,
        max_seq_len=int(seq_lens.max()),
        max_num_requests=int(seq_lens.shape[0]),
    )
    return out, sm_scale


@pytest.mark.skipif(not _is_sm100f(), reason="trtllm-gen decode kernels are SM100/SM103 only")
@pytest.mark.parametrize("kv_dtype", [torch.bfloat16, torch.float8_e4m3fn])
@pytest.mark.parametrize("num_heads,num_kv_heads", [(8, 1), (16, 2)], ids=["gqa8x1", "gqa16x2"])
@pytest.mark.parametrize("decode_query_len", [1, 2])
@pytest.mark.parametrize("subpages_per_slot", [2, 9])
def test_matches_reference(kv_dtype, num_heads, num_kv_heads, decode_query_len, subpages_per_slot):
    """subpages_per_slot=9 is the M3 coalesced case; 2 is the uniform one."""
    seq_lens = [PAGE_SIZE * 3, PAGE_SIZE + 5, PAGE_SIZE * 2 - 1]
    q, pool, block_table, seq_lens_t = _make_pool_inputs(
        seq_lens, num_heads, num_kv_heads, decode_query_len, subpages_per_slot, kv_dtype
    )
    out, sm_scale = _run_dense(
        q, pool, subpages_per_slot, block_table, seq_lens_t, decode_query_len
    )
    # The kernel runs Q in the KV dtype, so the oracle gets the same rounding.
    q_ref = q.to(kv_dtype).to(torch.float32) if kv_dtype == torch.float8_e4m3fn else q
    ref = _reference_dense_decode(
        q_ref, pool, subpages_per_slot, block_table, seq_lens_t, decode_query_len, sm_scale
    )
    tol = 6e-2 if kv_dtype == torch.float8_e4m3fn else 2e-2
    torch.testing.assert_close(out.float(), ref, rtol=tol, atol=tol)


@pytest.mark.skipif(not _is_sm100f(), reason="trtllm-gen decode kernels are SM100/SM103 only")
def test_cuda_graph_replay_tracks_inputs():
    seq_lens = [PAGE_SIZE * 2, PAGE_SIZE * 3]
    q, pool, block_table, seq_lens_t = _make_pool_inputs(
        seq_lens, 8, 1, 1, 9, torch.bfloat16, seed=7
    )
    stub = _StubManager(pool, 9)
    out = torch.zeros_like(q, dtype=torch.bfloat16)
    sm_scale = dense_decode_sm_scale(HEAD_DIM, 1.0)

    def run():
        minimax_m3_trtllm_gen_dense_decode(
            q,
            stub,
            0,
            block_table,
            seq_lens_t,
            sm_scale=sm_scale,
            output=out,
            decode_query_len=1,
            max_seq_len=PAGE_SIZE * 3,
            max_num_requests=2,
        )

    run()  # warm the arena and the counter buffer before capture
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        with torch.cuda.graph(graph):
            run()
    torch.cuda.current_stream().wait_stream(stream)

    q.copy_(torch.randn_like(q, dtype=torch.float32).to(torch.bfloat16))
    graph.replay()
    torch.cuda.synchronize()
    replayed = out.clone()

    out.zero_()
    run()
    torch.cuda.synchronize()
    torch.testing.assert_close(replayed.float(), out.float(), rtol=2e-2, atol=2e-2)


# --------------------------------------------------------------------------
# Gating
# --------------------------------------------------------------------------


class _FakeMeta:
    def __init__(self, decode_query_len: Optional[int]):
        self.msa_decode_query_len = decode_query_len
        self.msa_block_table = object() if decode_query_len is not None else None
        self.msa_seq_lens_cuda = object() if decode_query_len is not None else None


@pytest.mark.parametrize(
    "decode_query_len,expected",
    [
        # A resolved span, single-token or speculative: the kernel owns it
        # either way, and no switch can hand it back to fmha_sm100.
        (1, True),
        (2, True),
        # No span, so this is a pure prefill and fmha_sm100 runs every row.
        (None, False),
    ],
)
def test_gating(decode_query_len, expected):
    assert msa_ported_decode_active(_FakeMeta(decode_query_len)) is expected


def test_declines_unsupported_geometry():
    q = torch.zeros((1, 8, 64), device="cuda", dtype=torch.bfloat16)
    assert "head_dim 64" in dense_decode_supported(_StubManager(q, 2), q)

    class _NoPool:
        pass

    q128 = torch.zeros((1, 8, HEAD_DIM), device="cuda", dtype=torch.bfloat16)
    assert "flat sub-page pool" in dense_decode_supported(_NoPool(), q128)


def test_sm_scale_matches_flashinfer_convention():
    assert dense_decode_sm_scale(HEAD_DIM, 1.0) == pytest.approx(1.0 / math.sqrt(HEAD_DIM))
    assert dense_decode_sm_scale(HEAD_DIM, 2.0) == pytest.approx(1.0 / (2.0 * math.sqrt(HEAD_DIM)))
