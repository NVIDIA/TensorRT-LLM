# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the attention/m31_dense_decode entry of the MiniMax-M3.1-Flash sm_107 targets: the dense layers'
trtllm-gen decode at the KV split each target sizes per request count.

Every cell runs against the fp32 reference, and a capped cell also against the same call at the model maximum: the
two differ only by the order the CTAs' partial softmax sums are merged in, held to a few BF16 ulps.
"""

import zlib

import pytest
import torch

from tensorrt_llm._torch.modeling_v2.models.minimax_m3.ops.attention.m31_dense_decode import (  # noqa: E501
    HEAD_DIM,
    HEADS,
    KV_HEADS,
    MODEL_MAX,
)
from tensorrt_llm._torch.modeling_v2.models.minimax_m3.ops.attention.m31_dense_decode import (  # noqa: E501
    m3_dense_decode as op,
)

__extra_import_path__ = ["../../../.."]
from _validating import assert_within_ulp, certifying  # noqa: E402, F401

if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 7):
    pytest.skip("the sm_107 targets run on SM107 only", allow_module_level=True)

PAGE = 32
SUBPAGES = 2
SM_SCALE = HEAD_DIM**-0.5
# The FP8-KV band of the in-tree dense decode test (test_minimax_m3_dense_decode.test_matches_reference).
FP8_TOL = 6e-2
# The targets share most cells; each distinct one runs once.
CELLS = list({repr(c): c for c in op.CELLS}.values())


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
    perm = torch.randperm(slots, generator=g, device="cuda")[: batch * pages]
    block_table = perm.reshape(batch, pages).to(torch.int32)
    # The last third of every request's values is shifted by +1, so a split that missed the tail of a context longer
    # than its max_seq_len fails.
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


@pytest.mark.parametrize("cell", CELLS, ids=[c.why[:44] for c in CELLS])
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
    if spec["capped"]:
        uncapped = op(*_args(q, manager, block_table, seq_lens, spec["qlen"], MODEL_MAX)).clone()
        # Every element within 4 of its own BF16 spacing (2^(e - 8) for |x| = m * 2^e, m in [0.5, 1)).
        ref_bf16 = uncapped.float()
        _, exp = torch.frexp(ref_bf16.abs())
        spacing = torch.where(
            ref_bf16 == 0,
            torch.full_like(ref_bf16, 2.0**-133),
            torch.ldexp(torch.ones_like(ref_bf16), exp - 8),
        )
        assert bool(((out.float() - ref_bf16).abs() <= 4 * spacing).all())
        assert_within_ulp(out, uncapped, element_ulp=4.0, rms_ulp=1.0)
