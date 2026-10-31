# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Dense decode against the fp32 reference and between valid sequence-length bounds."""

import zlib

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_nvfp4__sm_103__tp4.ops.attention.m3_dense_decode import (  # noqa: E501
    HEAD_DIM,
    HEADS,
    KV_HEADS,
    MODEL_MAX,
    SHORT_BOUND,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_nvfp4__sm_103__tp4.ops.attention.m3_dense_decode import (  # noqa: E501
    m3_dense_decode as op,
)

__extra_import_path__ = ["../../../.."]
from _validating import assert_within_ulp, certifying, validating  # noqa: E402, F401

assert torch.cuda.is_available(), "m3_dense_decode requires a CUDA device"

PAGE = 32
SUBPAGES = 2
SM_SCALE = HEAD_DIM**-0.5
# The FP8-KV band of the in-tree dense decode test (test_minimax_m3_dense_decode.test_matches_reference).
FP8_TOL = 6e-2


class _Pool:
    """The one method the kernel asks of a KV cache manager."""

    def __init__(self, pool: torch.Tensor):
        self.pool = pool

    def get_kv_subpage_pool(self, layer_idx: int, kv_layout: str = "HND"):
        assert kv_layout == "HND"
        return self.pool, SUBPAGES


def _inputs(kv_lens, qlen: int, seed: int):
    g = torch.Generator(device="cuda").manual_seed(seed)
    batch = len(kv_lens)
    pages = max((n + PAGE - 1) // PAGE for n in kv_lens)
    slots = batch * pages + 3
    pool = torch.randn(
        (slots * SUBPAGES, KV_HEADS, PAGE, HEAD_DIM), device="cuda", generator=g
    ).clamp(-4, 4)
    # Slots handed out non-contiguously, as the block manager does.
    perm = torch.randperm(slots, generator=g, device="cuda")[: batch * pages]
    block_table = perm.reshape(batch, pages).to(torch.int32)
    # The last third of every request's values is shifted by +1: a third of each row's softmax mass sits there, far
    # outside the band, so a decode that missed the tail of a context fails.
    for b, n in enumerate(kv_lens):
        start = 2 * n // 3
        first_full = (start + PAGE - 1) // PAGE
        last = (n + PAGE - 1) // PAGE
        if first_full < last:
            v_idx = block_table[b, first_full:last].long() * SUBPAGES + 1
            pool[v_idx] += 1.0
        if start % PAGE:
            v0 = int(block_table[b, start // PAGE]) * SUBPAGES + 1
            pool[v0, :, start % PAGE :] += 1.0
    q = torch.randn((batch * qlen, HEADS, HEAD_DIM), device="cuda", generator=g).to(torch.bfloat16)
    return (
        q,
        _Pool(pool.to(torch.float8_e4m3fn)),
        block_table,
        torch.tensor(kv_lens, device="cuda", dtype=torch.int32),
    )


def _args(q, manager, block_table, seq_lens, qlen, max_seq_len):
    out = torch.zeros(q.shape, dtype=torch.bfloat16, device="cuda")
    return (
        q,
        manager,
        0,
        block_table,
        seq_lens,
        SM_SCALE,
        out,
        qlen,
        max_seq_len,
        seq_lens.shape[0],
    )


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell, certifying) -> None:  # noqa: F811 — pytest resolves the fixture by this name
    spec = cell.spec
    q, manager, block_table, seq_lens = _inputs(
        spec["kv_lens"], spec["qlen"], zlib.crc32(cell.why.encode()) % 2**31
    )
    args = _args(q, manager, block_table, seq_lens, spec["qlen"], spec["max_seq_len"])
    certifying(op)
    ref = op.reference(*args)
    out = op(*args).clone()
    torch.testing.assert_close(out.float(), ref, rtol=FP8_TOL, atol=FP8_TOL)
    if spec["tight_bound"]:
        model_max_output = op(
            *_args(q, manager, block_table, seq_lens, spec["qlen"], MODEL_MAX)
        ).clone()
        # The split reorders only the merge of the CTAs' fp32 partial sums before the one BF16 rounding: every element
        # is within 4 of its own BF16 spacing (2^(e - 8) for |x| = m * 2^e, m in [0.5, 1)).
        ref_bf16 = model_max_output.float()
        _, exp = torch.frexp(ref_bf16.abs())
        spacing = torch.where(
            ref_bf16 == 0,
            torch.full_like(ref_bf16, 2.0**-133),
            torch.ldexp(torch.ones_like(ref_bf16), exp - 8),
        )
        assert bool(((out.float() - ref_bf16).abs() <= 4 * spacing).all())
        assert_within_ulp(out, model_max_output, element_ulp=4.0, rms_ulp=1.0)


def test_guard_refuses_more_requests_than_the_counters() -> None:
    q, manager, block_table, seq_lens = _inputs((3000, 3000), 1, 3)
    args = list(_args(q, manager, block_table, seq_lens, 1, SHORT_BOUND))
    args[9] = 1
    with pytest.raises(AssertionError, match="counters cover"):
        with validating(op):
            op(*args)


def test_guard_refuses_an_understated_sequence_length_bound() -> None:
    q, manager, block_table, seq_lens = _inputs((20000,), 1, 7)
    args = _args(q, manager, block_table, seq_lens, 1, SHORT_BOUND)
    with pytest.raises(AssertionError, match="sequence length exceeds its bound"):
        with validating(op):
            op(*args)
