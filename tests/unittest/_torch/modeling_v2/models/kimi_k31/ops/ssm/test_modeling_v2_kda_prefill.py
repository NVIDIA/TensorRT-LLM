# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the Kimi K3.1 kda_prefill entry.

Each cell is a packed ragged prefill as the KDA layer issues it: raw q/k/v and gate logits, beta as the strided
column view of the merged [f_a | b | pad] projection, a padded-slot V-first state pool whose fresh rows the caller
zeroed, mode "bt16" and the metadata's planned seq_order / cu_chunks. Output and the selected state rows are checked
against an fp32 token-by-token recurrence; every other pool row must be untouched.
"""

import math

import pytest
import torch

from tensorrt_llm._torch.modeling_v2.models.kimi_k31.ops.ssm.kda_prefill import kda_prefill

from ._kda_ops_util import (
    DTYPES,
    HEAD_DIM,
    LOWER_BOUND,
    SCALE,
    assert_rows,
    assert_scaled_close,
    clone_strided,
    generator,
    padded_slots,
    requires_sm103,
    step_schedule,
)

__extra_import_path__ = ["../../../..", "../.."]
from _k31_utils import sample_cells  # noqa: E402
from _validating import certifying  # noqa: E402,F401 — path above; fixture by name

pytestmark = requires_sm103

#: A boundary and a typical cell of ``kda_prefill.CELLS``.
_CELLS = sample_cells(kda_prefill.CELLS, size_key="heads")[:2]


# Chunked bf16 tensor-core evaluation against an fp32 recurrence (the engine prefill tests' bands).
TOL_OUT, TOL_STATE = 2e-2, 1e-2
POOL_SLOTS = 12


def _inputs(spec, gen):
    heads, lengths = spec["heads"], list(spec["seq_lens"])
    total = sum(lengths)
    shape = (1, total, heads, HEAD_DIM)

    def rnd(*s, scale=1.0):
        return (torch.randn(*s, generator=gen, device="cuda") * scale).to(torch.bfloat16)

    width = (HEAD_DIM + heads + 7) // 8 * 8  # merged [f_a | b | pad] projection
    merged = rnd(1, total, width)
    # 64 bytes of slot padding: the compiled prefill takes slot strides that are multiples of 8 elements.
    pool = padded_slots(
        POOL_SLOTS, (heads, HEAD_DIM, HEAD_DIM), DTYPES[spec["state_dtype"]], pad_bytes=64
    )
    pool.copy_(torch.randn(pool.shape, generator=gen, device="cuda") * 0.05)
    for slot, fresh in zip(spec["slots"], spec["fresh"]):
        if fresh:
            pool[slot].zero_()
    offsets = [0]
    for length in lengths:
        offsets.append(offsets[-1] + length)
    seq_order, cu_chunks = step_schedule(lengths, "cuda")
    return dict(
        q=rnd(*shape),
        k=rnd(*shape),
        v=rnd(*shape, scale=0.5),
        g=rnd(*shape),
        beta=merged[..., HEAD_DIM : HEAD_DIM + heads],
        state_pool=pool,
        state_indices=torch.tensor(spec["slots"], dtype=torch.int32, device="cuda"),
        cu_seqlens=torch.tensor(offsets, dtype=torch.int64, device="cuda"),
        A_log=torch.log(torch.empty(heads, device="cuda").uniform_(1, 16, generator=gen)),
        dt_bias=torch.empty(heads * HEAD_DIM, device="cuda").uniform_(
            math.log(1e-3), math.log(1e-1), generator=gen
        ),
        scale=SCALE,
        lower_bound=LOWER_BOUND,
        mode="bt16",
        max_seqlen=max(lengths),
        seq_order=seq_order,
        cu_chunks=cu_chunks,
    )


@pytest.mark.parametrize("cell", _CELLS, ids=[c.why[:60] for c in _CELLS])
@torch.inference_mode()
def test_certified_cells(cell, certifying) -> None:  # noqa: F811 — pytest resolves the fixture by this name
    certifying(kda_prefill)
    args = _inputs(cell.spec, generator(cell))
    pool = args["state_pool"]
    ref_args = dict(args, state_pool=clone_strided(pool))
    out = kda_prefill(**args)
    expected = kda_prefill.reference(**ref_args)
    assert_scaled_close(out, expected, TOL_OUT, "output")
    assert_rows(pool, ref_args["state_pool"], set(cell.spec["slots"]), TOL_STATE)
