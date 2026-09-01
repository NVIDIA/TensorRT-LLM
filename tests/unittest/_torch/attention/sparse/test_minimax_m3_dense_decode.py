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

The kernel reads its pages from a flat sub-page view of the K/V pool, so both
halves of that addressing are covered here: the block-table expansion that
names this layer's K and V sub-pages, and the kernel call itself against a
PyTorch oracle. A stub cache manager stands in for the pool, so the flashinfer
argument conventions are exercised without standing up a model.
"""

from __future__ import annotations

from typing import Dict, List

import pytest
import torch

from tensorrt_llm._torch.attention_backend.sparse.minimax_m3_kernels.trtllm_gen_dense_decode import (
    dense_decode_unsupported_reason,
    minimax_m3_trtllm_gen_dense_decode,
    subpage_block_table,
    uniform_subpages_per_slot,
    write_subpage_block_table,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")

PAGE_SIZE = 32
HEAD_DIM = 128
# The bmm1 scale, spelled as run_msa_prefill_gqa spells it at q_scaling 1.
SM_SCALE = HEAD_DIM**-0.5


def _is_sm100f() -> bool:
    major, minor = torch.cuda.get_device_capability()
    return major == 10 and minor in (0, 3)


# --------------------------------------------------------------------------
# Block-table expansion
# --------------------------------------------------------------------------


def test_subpage_block_table_splits_k_and_v_rows():
    """Slot s lands at s * subpages_per_slot for K and one sub-page later for V."""
    slots = torch.tensor([[0, 3, 7], [2, 5, 11]], device="cuda", dtype=torch.int32)
    table = subpage_block_table(slots, subpages_per_slot=9)

    assert table.shape == (2, 2, 3)
    assert table.dtype == torch.int32
    assert table[:, 0].tolist() == [[0, 27, 63], [18, 45, 99]]
    assert table[:, 1].tolist() == [[1, 28, 64], [19, 46, 100]]


def test_subpage_block_table_reuses_one_buffer():
    """Every caller of a given shape gets the same arena block back.

    The table is therefore only valid until the next call, which is why it is
    rewritten in full each time rather than held across layers.
    """
    slots_a = torch.zeros((2, 4), device="cuda", dtype=torch.int32)
    slots_b = torch.full((2, 4), 5, device="cuda", dtype=torch.int32)
    first = subpage_block_table(slots_a, 4)
    second = subpage_block_table(slots_b, 4)

    assert first.data_ptr() == second.data_ptr()
    assert second[:, 0].tolist() == [[20] * 4] * 2


def test_write_subpage_block_table_fills_a_caller_owned_buffer():
    """prepare() stages the expansion into its own graph-stable buffer, so the
    in-place writer has to agree with the arena-backed helper."""
    slots = torch.tensor([[0, 3, 7], [2, 5, 11]], device="cuda", dtype=torch.int32)
    out = torch.empty((2, 2, 3), device="cuda", dtype=torch.int32)

    write_subpage_block_table(slots, 9, out)

    assert torch.equal(out, subpage_block_table(slots, 9))


class _PerLayerFactors:
    """The two attributes uniform_subpages_per_slot reads off a cache manager."""

    def __init__(self, factors: Dict[int, int]):
        self.layer_offsets = dict.fromkeys(factors, 0)
        self._factors = factors

    def get_kv_subpage_pool(self, layer_idx: int, kv_layout: str = "HND"):
        return None, self._factors[layer_idx]


@pytest.mark.parametrize(
    "factors,expected",
    [
        pytest.param({0: 2, 1: 2, 2: 2, 3: 2}, 2, id="agreeing"),
        # A pool that coalesces the sparse layers' index-K into their K/V blocks
        # gives those layers a different factor from the dense ones.
        pytest.param({0: 2, 1: 9, 2: 2, 3: 9}, 0, id="disagreeing"),
    ],
)
def test_uniform_subpages_per_slot(factors, expected):
    """prepare() expands the block table without naming a layer, so the factor
    it stages has to be one every layer agrees on, and 0 where they do not."""
    assert uniform_subpages_per_slot(_PerLayerFactors(factors)) == expected


def test_uniform_subpages_per_slot_reports_zero_without_a_pool():
    """A manager with no sub-page pool leaves each dense layer to expand its
    own table, rather than being staged against a guessed factor."""

    class _NoPool:
        layer_offsets = {0: 0}

    assert uniform_subpages_per_slot(_NoPool()) == 0


# --------------------------------------------------------------------------
# Kernel parity against a PyTorch oracle
# --------------------------------------------------------------------------


class _StubManager:
    """The one method the kernel asks of a KV cache manager."""

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


def _run_dense(q, pool, subpages_per_slot, block_table, seq_lens, decode_query_len, **kwargs):
    out = torch.zeros_like(q, dtype=torch.bfloat16)
    minimax_m3_trtllm_gen_dense_decode(
        q,
        _StubManager(pool, subpages_per_slot),
        0,
        block_table,
        seq_lens,
        sm_scale=SM_SCALE,
        output=out,
        decode_query_len=decode_query_len,
        max_seq_len=int(seq_lens.max()),
        max_num_requests=int(seq_lens.shape[0]),
        **kwargs,
    )
    return out


@pytest.mark.skipif(not _is_sm100f(), reason="trtllm-gen decode kernels are SM100/SM103 only")
@pytest.mark.parametrize("kv_dtype", [torch.bfloat16, torch.float8_e4m3fn])
@pytest.mark.parametrize("num_heads,num_kv_heads", [(8, 1), (16, 2)], ids=["gqa8x1", "gqa16x2"])
@pytest.mark.parametrize("decode_query_len", [1, 2])
@pytest.mark.parametrize("subpages_per_slot", [2, 9])
def test_matches_reference(kv_dtype, num_heads, num_kv_heads, decode_query_len, subpages_per_slot):
    """Full-context decode against an fp32 oracle over the same shuffled pages.

    subpages_per_slot is 9 for a pool that coalesces index-K into the sparse
    layers' blocks and 2 for one that does not.
    """
    seq_lens = [PAGE_SIZE * 3, PAGE_SIZE + 5, PAGE_SIZE * 2 - 1]
    q, pool, block_table, seq_lens_t = _make_pool_inputs(
        seq_lens, num_heads, num_kv_heads, decode_query_len, subpages_per_slot, kv_dtype
    )
    out = _run_dense(q, pool, subpages_per_slot, block_table, seq_lens_t, decode_query_len)
    # The kernel runs Q in the KV dtype, so the oracle gets the same rounding.
    q_ref = q.to(kv_dtype).to(torch.float32) if kv_dtype == torch.float8_e4m3fn else q
    ref = _reference_dense_decode(
        q_ref, pool, subpages_per_slot, block_table, seq_lens_t, decode_query_len, SM_SCALE
    )
    tol = 6e-2 if kv_dtype == torch.float8_e4m3fn else 2e-2
    torch.testing.assert_close(out.float(), ref, rtol=tol, atol=tol)


@pytest.mark.skipif(not _is_sm100f(), reason="trtllm-gen decode kernels are SM100/SM103 only")
def test_staged_subpage_table_is_used_only_when_the_factor_matches():
    """prepare() stages one expansion for every dense layer, so the kernel must
    take it when the factor agrees and expand its own when it does not; either
    way the answer is the same."""
    subpages_per_slot = 9
    seq_lens = [PAGE_SIZE * 3, PAGE_SIZE + 5]
    q, pool, block_table, seq_lens_t = _make_pool_inputs(
        seq_lens, 8, 1, 1, subpages_per_slot, torch.bfloat16, seed=11
    )
    staged = torch.empty(
        (block_table.shape[0], 2, block_table.shape[1]), device="cuda", dtype=torch.int32
    )
    write_subpage_block_table(block_table, subpages_per_slot, staged)

    args = (q, pool, subpages_per_slot, block_table, seq_lens_t, 1)
    expanded_here = _run_dense(*args)
    from_staged = _run_dense(*args, staged_subpage_table=staged, staged_subpages_per_slot=9)
    # A stale factor must be ignored rather than trusted: this table addresses
    # the wrong sub-pages entirely.
    wrong = torch.zeros_like(staged)
    ignored = _run_dense(*args, staged_subpage_table=wrong, staged_subpages_per_slot=2)

    torch.testing.assert_close(from_staged.float(), expanded_here.float(), rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(ignored.float(), expanded_here.float(), rtol=1e-3, atol=1e-3)


@pytest.mark.skipif(not _is_sm100f(), reason="trtllm-gen decode kernels are SM100/SM103 only")
def test_cuda_graph_replay_tracks_inputs():
    """Replay must recompute from the live buffers, not reuse captured values."""
    seq_lens = [PAGE_SIZE * 2, PAGE_SIZE * 3]
    q, pool, block_table, seq_lens_t = _make_pool_inputs(
        seq_lens, 8, 1, 1, 9, torch.bfloat16, seed=7
    )
    stub = _StubManager(pool, 9)
    out = torch.zeros_like(q, dtype=torch.bfloat16)

    def run():
        minimax_m3_trtllm_gen_dense_decode(
            q,
            stub,
            0,
            block_table,
            seq_lens_t,
            sm_scale=SM_SCALE,
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


def test_declines_unsupported_geometry():
    """Each unsupported geometry has to be caught before the kernel is reached.

    The verdict is a string because it is logged, so the assertions look for the
    distinguishing part of it rather than the whole message.
    """
    pool = torch.zeros((1, 1, PAGE_SIZE, 64), device="cuda", dtype=torch.bfloat16)
    assert "head_dim 64" in dense_decode_unsupported_reason(_StubManager(pool, 2), 64)

    class _NoPool:
        pass

    assert "flat sub-page pool" in dense_decode_unsupported_reason(_NoPool(), HEAD_DIM)


def test_accepts_the_m3_geometry():
    pytest.importorskip("flashinfer")
    pool = torch.zeros((1, 1, PAGE_SIZE, HEAD_DIM), device="cuda", dtype=torch.bfloat16)
    assert dense_decode_unsupported_reason(_StubManager(pool, 2), HEAD_DIM) is None
