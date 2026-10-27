# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test: the short-sequence identity selection (runtime policy ``short_sequence_identity``) is the stock table.

On a step whose every query token has at most 16 valid blocks, this target skips the index scorer and the select and
stages the table (i < n_valid ? i : -1). Stock runs the in-tree CuTe DSL scorer and ``select_blocks_from_maxscore``,
which returns ascending ids and drops a valid block only when its max score is -inf. The scorer's scores are FP8
products summed in FP32 (|score| <= 448^2 * 128), so -inf comes only from the causal mask, which n_valid excludes.
This test drives the real scorer with extreme FP8 operands at the TP8 geometry (1 index head per rank) and checks
both claims.
"""

import pytest
import torch

import tensorrt_llm._torch.custom_ops  # noqa: F401 — registers torch.ops.trtllm.*
from tensorrt_llm._torch.attention.backends.sparse.minimax_m3.kernels.msa_utils import (
    select_blocks_from_maxscore,
)
from tensorrt_llm._torch.custom_ops import (
    cute_dsl_custom_ops,  # noqa: F401 — registers the scorer op
)

PAGE, DIM, TOPK, HEADS, QLEN = 128, 128, 16, 1, 4
E4M3_MAX = 448.0

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 7),
    reason="the sm_107 TP8 target runs on SM107 only",
)


def _fill(kind, shape, gen):
    if kind == "random":
        x = torch.randn(shape, generator=gen, device="cuda") * 64
    elif kind == "max":
        x = torch.full(shape, E4M3_MAX, device="cuda")
    elif kind == "negmax":
        x = torch.full(shape, -E4M3_MAX, device="cuda")
    elif kind == "alternating":
        x = torch.full(shape, E4M3_MAX, device="cuda")
        x[..., 1::2] = -E4M3_MAX
    elif kind == "zero":
        x = torch.zeros(shape, device="cuda")
    else:  # the smallest E4M3 subnormal
        x = torch.full(shape, 2.0**-9, device="cuda")
    return x.clamp(-E4M3_MAX, E4M3_MAX).to(torch.float8_e4m3fn)


_KINDS = ("random", "max", "negmax", "alternating", "zero", "tiny")


@pytest.mark.parametrize(
    "seq_lens", [(4,), (129,), (2048,), (700, 2048), (5, 1500, 2047, 300), (1, 128, 256, 1024)]
)
@pytest.mark.parametrize("q_kind", _KINDS)
@pytest.mark.parametrize("k_kind", _KINDS)
def test_stock_selection_of_a_short_step_is_the_identity_table(seq_lens, q_kind, k_kind) -> None:
    gen = torch.Generator(device="cuda").manual_seed(0)
    reqs, total_q = len(seq_lens), len(seq_lens) * QLEN
    nblk = [(s + PAGE - 1) // PAGE for s in seq_lens]
    pages = sum(nblk) + 4
    perm = torch.randperm(pages, generator=gen, device="cuda").int()
    block_table = torch.zeros((reqs, max(nblk)), dtype=torch.int32, device="cuda")
    start = 0
    for r, n in enumerate(nblk):
        block_table[r, :n] = perm[start : start + n]
        start += n
    n_valid = torch.tensor(
        [(max(s - QLEN + o + 1, 0) + PAGE - 1) // PAGE for s in seq_lens for o in range(QLEN)],
        dtype=torch.int32,
        device="cuda",
    )
    idx_q = _fill(q_kind, (total_q, HEADS, DIM), gen)
    idx_k = _fill(k_kind, (pages, PAGE, DIM), gen)
    max_score = torch.full((HEADS, max(nblk), total_q), float("-inf"), device="cuda")
    torch.ops.trtllm.cute_dsl_minimax_m3_index_decode_score(
        idx_q,
        idx_k,
        block_table,
        torch.tensor(seq_lens, dtype=torch.int32, device="cuda"),
        max_score.transpose(1, 2),
        QLEN,
    )
    for tok, nv in enumerate(n_valid.tolist()):
        assert bool(torch.isfinite(max_score[:, :nv, tok]).all()), (
            f"token {tok}: a causally valid block scored non-finite"
        )
    table = select_blocks_from_maxscore(
        max_score,
        topk=TOPK,
        n_valid_blocks=n_valid,
        init_blocks=1,
        local_blocks=4,
        head_major_output=True,
    ).to(torch.int32)
    # The selector returns [tokens, kv_heads, topk]; the decode reads it as [kv_heads, tokens, topk] (layers.py).
    table = table.permute(1, 0, 2)
    i = torch.arange(TOPK, dtype=torch.int32, device="cuda")
    identity = torch.where(
        i[None, :] < n_valid[:, None], i[None, :], torch.full_like(i[None, :], -1)
    )
    assert torch.equal(table, identity[None].expand(HEADS, total_q, TOPK))
