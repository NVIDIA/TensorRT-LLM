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
"""
Differential tests for the fused DSA decode-metadata Triton kernel.

The kernel in fused_metadata.py collapses the eager metadata chain in
DSAtrtllmAttentionMetadata.on_update_kv_lens() into one launch. Its whole
contract is that the integer metadata it produces is BIT-IDENTICAL to that eager
chain -- a wrong index silently corrupts sparse attention. These tests assert
bit-exactness against the real eager helpers (build_req_idx_per_token +
_compute_slot_mappings + the two int64 generation cumsums) across the decode
regime the kernel must cover: batch sizes 1 / non-power-of-two / 256, next_n
1..8, FP8 and FP4 byte layouts, both ends of the block clamp, block ids whose
byte offsets exceed 2**31, and the stale negative-position case the eager cuda
path defends against.
"""

import pytest
import torch

# The kernel is Triton + CUDA only.
pytest.importorskip("triton")
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="fused DSA metadata kernel requires CUDA"
)

from tensorrt_llm._torch.attention_backend.sparse.dsa import (  # noqa: E402
    _compute_slot_mappings,
    build_req_idx_per_token,
)
from tensorrt_llm._torch.attention_backend.sparse.dsa.fused_metadata import (  # noqa: E402
    fused_dsa_decode_metadata,
)
from tensorrt_llm._torch.attention_backend.sparse.dsa.metadata import (  # noqa: E402
    _fused_dsa_meta_enabled,
)

DEVICE = "cuda"
INDEX_HEAD_DIM = 128
QUANT_BLOCK_SIZE = 128
TOKENS_PER_BLOCK = 64


def _eager_reference(
    seq_lens: torch.Tensor,
    kv_lens: torch.Tensor,
    block_offsets: torch.Tensor,
    num_tokens: int,
    index_head_dim: int,
    tokens_per_block: int,
    quant_block_size: int,
    data_bytes_per_token: int,
):
    """Reproduce the exact eager chain from on_update_kv_lens (cuda path)."""
    num_seqs = seq_lens.shape[0]
    start_positions = kv_lens - seq_lens

    req_idx = build_req_idx_per_token(seq_lens, num_tokens).to(torch.int32)
    req_indices = req_idx.to(torch.int64)
    seq_starts = torch.cumsum(seq_lens, dim=0, dtype=torch.int64) - seq_lens.to(torch.int64)
    token_offsets = (
        torch.arange(num_tokens, device=seq_lens.device, dtype=torch.int64)
        - seq_starts[req_indices]
    )
    global_positions = start_positions[req_indices] + token_offsets

    fp8, scale = _compute_slot_mappings(
        global_positions,
        block_offsets,
        req_indices,
        index_head_dim,
        tokens_per_block,
        quant_block_size,
        data_bytes_per_token=data_bytes_per_token,
    )

    gen_kv_indptr = torch.zeros(num_seqs + 1, dtype=torch.int64, device=seq_lens.device)
    gen_kv_indptr[1:] = torch.cumsum(kv_lens, dim=0, dtype=torch.int64)
    gen_cached = torch.zeros(num_seqs + 1, dtype=torch.int64, device=seq_lens.device)
    gen_cached[1:] = torch.cumsum(kv_lens - seq_lens, dim=0, dtype=torch.int64)
    return req_idx, fp8, scale, gen_kv_indptr, gen_cached


def _run_fused(
    seq_lens: torch.Tensor,
    kv_lens: torch.Tensor,
    block_offsets: torch.Tensor,
    num_tokens: int,
    max_query_len: int,
    index_head_dim: int,
    tokens_per_block: int,
    quant_block_size: int,
    data_bytes_per_token: int,
):
    """Allocate output buffers and invoke the fused kernel."""
    num_seqs = seq_lens.shape[0]
    req_idx = torch.full((num_tokens,), -1, dtype=torch.int32, device=DEVICE)
    slot_fp8 = torch.full((num_tokens,), -1, dtype=torch.int64, device=DEVICE)
    slot_scale = torch.full((num_tokens,), -1, dtype=torch.int64, device=DEVICE)
    gen_kv_indptr = torch.full((num_seqs + 1,), -1, dtype=torch.int64, device=DEVICE)
    gen_cached = torch.full((num_seqs + 1,), -1, dtype=torch.int64, device=DEVICE)

    fused_dsa_decode_metadata(
        seq_lens,
        kv_lens,
        block_offsets,
        req_idx,
        slot_fp8,
        slot_scale,
        gen_kv_indptr,
        gen_cached,
        num_tokens=num_tokens,
        max_query_len=max_query_len,
        tokens_per_block=tokens_per_block,
        index_head_dim=index_head_dim,
        quant_block_size=quant_block_size,
        data_bytes_per_token=data_bytes_per_token,
    )
    return req_idx, slot_fp8, slot_scale, gen_kv_indptr, gen_cached


def _assert_bit_identical(fused, eager):
    names = [
        "req_idx_per_token",
        "slot_mapping_fp8",
        "slot_mapping_scale",
        "gen_kv_indptr",
        "gen_cached_token_indptr",
    ]
    for name, f, e in zip(names, fused, eager):
        assert torch.equal(f, e), (
            f"{name} mismatch: {(f != e).sum().item()} / {e.numel()} elements differ"
        )


def _build_inputs(
    num_seqs: int,
    next_n: int,
    max_blocks: int,
    *,
    block_id_hi: int = 4096,
    start_block_lo: int = 0,
    start_block_hi=None,
    seed: int = 0,
):
    """Build (seq_lens, kv_lens, block_offsets) for a pure-decode step.

    Every generation request carries exactly next_n query tokens (the decode
    contract), so sum(seq_lens) == num_tokens and each seq_len == next_n.
    start position (cached length) is a random multiple-free offset in blocks.
    """
    gen = torch.Generator(device="cpu").manual_seed(seed)
    seq_lens = torch.full((num_seqs,), next_n, dtype=torch.int32)
    if start_block_hi is None:
        start_block_hi = max(1, max_blocks - (next_n // TOKENS_PER_BLOCK) - 2)
    # cached KV length per request; keep gpos < max_blocks * tpb for the no-clamp
    # cases by bounding the starting block.
    start_blocks = torch.randint(
        start_block_lo, max(start_block_lo + 1, start_block_hi), (num_seqs,), generator=gen
    )
    start_positions = start_blocks.to(torch.int32) * TOKENS_PER_BLOCK
    kv_lens = start_positions + seq_lens
    block_offsets = torch.randint(
        0, block_id_hi, (num_seqs, max_blocks), dtype=torch.int32, generator=gen
    )
    return seq_lens.to(DEVICE), kv_lens.to(DEVICE), block_offsets.to(DEVICE)


@pytest.mark.parametrize("num_seqs", [1, 3, 7, 256], ids=lambda v: f"bs{v}")
@pytest.mark.parametrize("next_n", [1, 2, 5, 8], ids=lambda v: f"nextn{v}")
@pytest.mark.parametrize("use_fp4", [False, True], ids=["fp8", "fp4"])
def test_fused_matches_eager(num_seqs, next_n, use_fp4):
    """Core differential check across the decode batch/next_n/quant matrix."""
    max_blocks = 32
    data_bytes_per_token = INDEX_HEAD_DIM // 2 if use_fp4 else INDEX_HEAD_DIM
    seq_lens, kv_lens, block_offsets = _build_inputs(
        num_seqs, next_n, max_blocks, seed=num_seqs * 100 + next_n
    )
    num_tokens = int(seq_lens.sum())

    fused = _run_fused(
        seq_lens,
        kv_lens,
        block_offsets,
        num_tokens,
        next_n,
        INDEX_HEAD_DIM,
        TOKENS_PER_BLOCK,
        QUANT_BLOCK_SIZE,
        data_bytes_per_token,
    )
    eager = _eager_reference(
        seq_lens,
        kv_lens,
        block_offsets,
        num_tokens,
        INDEX_HEAD_DIM,
        TOKENS_PER_BLOCK,
        QUANT_BLOCK_SIZE,
        data_bytes_per_token,
    )
    _assert_bit_identical(fused, eager)


def test_fused_clamp_active_matches_eager():
    """Stale positions past the block table exercise the upper clamp; both the
    fused kernel and the eager cuda path clamp block index to max_blocks - 1."""
    num_seqs, next_n, max_blocks = 4, 5, 8
    seq_lens = torch.full((num_seqs,), next_n, dtype=torch.int32, device=DEVICE)
    # start beyond the table so gpos // tpb >= max_blocks for every token.
    start_positions = torch.full((num_seqs,), max_blocks * TOKENS_PER_BLOCK + 3, dtype=torch.int32)
    kv_lens = (start_positions.to(DEVICE) + seq_lens).to(torch.int32)
    gen = torch.Generator(device="cpu").manual_seed(7)
    block_offsets = torch.randint(
        0, 4096, (num_seqs, max_blocks), dtype=torch.int32, generator=gen
    ).to(DEVICE)
    num_tokens = int(seq_lens.sum())

    for use_fp4 in (False, True):
        dbt = INDEX_HEAD_DIM // 2 if use_fp4 else INDEX_HEAD_DIM
        fused = _run_fused(
            seq_lens,
            kv_lens,
            block_offsets,
            num_tokens,
            next_n,
            INDEX_HEAD_DIM,
            TOKENS_PER_BLOCK,
            QUANT_BLOCK_SIZE,
            dbt,
        )
        eager = _eager_reference(
            seq_lens,
            kv_lens,
            block_offsets,
            num_tokens,
            INDEX_HEAD_DIM,
            TOKENS_PER_BLOCK,
            QUANT_BLOCK_SIZE,
            dbt,
        )
        _assert_bit_identical(fused, eager)


def test_fused_large_block_offset_int64():
    """Block ids large enough that byte offsets exceed 2**31 must not wrap: the
    load-bearing fp8/scale offsets are int64."""
    num_seqs, next_n, max_blocks = 2, 4, 16
    seq_lens = torch.full((num_seqs,), next_n, dtype=torch.int32, device=DEVICE)
    start_positions = torch.tensor([0, TOKENS_PER_BLOCK], dtype=torch.int32)
    kv_lens = (start_positions.to(DEVICE) + seq_lens).to(torch.int32)
    # block_stride = tpb * (data + scale) = 64 * (128 + 4) = 8448; ~300000 ids
    # push fp8 offsets well past 2**31.
    block_offsets = torch.full((num_seqs, max_blocks), 300000, dtype=torch.int32, device=DEVICE)
    num_tokens = int(seq_lens.sum())

    fused = _run_fused(
        seq_lens,
        kv_lens,
        block_offsets,
        num_tokens,
        next_n,
        INDEX_HEAD_DIM,
        TOKENS_PER_BLOCK,
        QUANT_BLOCK_SIZE,
        INDEX_HEAD_DIM,
    )
    eager = _eager_reference(
        seq_lens,
        kv_lens,
        block_offsets,
        num_tokens,
        INDEX_HEAD_DIM,
        TOKENS_PER_BLOCK,
        QUANT_BLOCK_SIZE,
        INDEX_HEAD_DIM,
    )
    assert fused[1].max().item() > 2**31  # slot_mapping_fp8 actually crossed the boundary
    _assert_bit_identical(fused, eager)


def test_fused_negative_gpos_matches_eager():
    """Stale kv_lens < seq_lens makes global positions negative. Triton lowers
    signed //,% to truncate-toward-zero, so the kernel normalizes the remainder
    to match torch's floor semantics (the eager cuda path clamps the quotient)."""
    num_seqs, next_n, max_blocks = 3, 4, 16
    seq_lens = torch.full((num_seqs,), next_n, dtype=torch.int32, device=DEVICE)
    # kv_lens < seq_lens -> start_positions < 0 -> some negative gpos.
    kv_lens = torch.tensor([1, 2, 0], dtype=torch.int32, device=DEVICE)
    gen = torch.Generator(device="cpu").manual_seed(11)
    block_offsets = torch.randint(
        0, 4096, (num_seqs, max_blocks), dtype=torch.int32, generator=gen
    ).to(DEVICE)
    num_tokens = int(seq_lens.sum())

    fused = _run_fused(
        seq_lens,
        kv_lens,
        block_offsets,
        num_tokens,
        next_n,
        INDEX_HEAD_DIM,
        TOKENS_PER_BLOCK,
        QUANT_BLOCK_SIZE,
        INDEX_HEAD_DIM,
    )
    eager = _eager_reference(
        seq_lens,
        kv_lens,
        block_offsets,
        num_tokens,
        INDEX_HEAD_DIM,
        TOKENS_PER_BLOCK,
        QUANT_BLOCK_SIZE,
        INDEX_HEAD_DIM,
    )
    _assert_bit_identical(fused, eager)


@pytest.mark.parametrize(
    "value,expected",
    [("0", False), ("1", True), ("garbage", False)],
)
def test_meta_gate_values(monkeypatch, value, expected):
    """The env gate is on only for exactly "1"; anything else keeps eager."""
    monkeypatch.setenv("TRTLLM_FUSED_DSA_METADATA", value)
    _fused_dsa_meta_enabled.cache_clear()
    try:
        assert _fused_dsa_meta_enabled() is expected
    finally:
        _fused_dsa_meta_enabled.cache_clear()


def test_meta_gate_unset(monkeypatch):
    """With the env var unset, the gate is off (eager path)."""
    monkeypatch.delenv("TRTLLM_FUSED_DSA_METADATA", raising=False)
    _fused_dsa_meta_enabled.cache_clear()
    try:
        assert _fused_dsa_meta_enabled() is False
    finally:
        _fused_dsa_meta_enabled.cache_clear()


def test_meta_gate_cached(monkeypatch):
    """The gate is read once and pinned; a mid-run env flip must not change it."""
    monkeypatch.setenv("TRTLLM_FUSED_DSA_METADATA", "1")
    _fused_dsa_meta_enabled.cache_clear()
    try:
        assert _fused_dsa_meta_enabled() is True
        monkeypatch.setenv("TRTLLM_FUSED_DSA_METADATA", "0")
        assert _fused_dsa_meta_enabled() is True  # cached, not re-read
    finally:
        _fused_dsa_meta_enabled.cache_clear()
