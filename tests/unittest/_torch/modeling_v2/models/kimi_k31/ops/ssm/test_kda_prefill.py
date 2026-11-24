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
from _kda_ops_util import (
    DTYPES,
    HEAD_DIM,
    LOWER_BOUND,
    SCALE,
    assert_scaled_close,
    bits,
    clone_strided,
    generator,
    padded_slots,
    requires_sm103,
)

from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.ssm.kda_prefill import (
    kda_prefill,
)

__extra_import_path__ = ["../../../.."]
from _validating import certifying, validating  # noqa: E402,F401 — path above; fixture by name

pytestmark = requires_sm103

# Chunked bf16 tensor-core evaluation against an fp32 recurrence (the engine prefill tests' bands).
TOL_OUT, TOL_STATE = 2e-2, 1e-2
POOL_SLOTS = 12


def _planned_metadata(lengths):
    order = sorted(range(len(lengths)), key=lambda i: (-lengths[i], i))
    chunks = [0]
    for length in lengths:
        chunks.append(chunks[-1] + (length + 15) // 16)
    as_int32 = lambda xs: torch.tensor(xs, dtype=torch.int32, device="cuda")  # noqa: E731
    return as_int32(order), as_int32(chunks)


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
    seq_order, cu_chunks = _planned_metadata(lengths)
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


@pytest.mark.parametrize("cell", kda_prefill.CELLS, ids=[c.why[:60] for c in kda_prefill.CELLS])
@torch.inference_mode()
def test_certified_cells(cell, certifying) -> None:  # noqa: F811 — pytest resolves the fixture by this name
    certifying(kda_prefill)
    args = _inputs(cell.spec, generator(cell))
    pool = args["state_pool"]
    ref_args = dict(args, state_pool=clone_strided(pool))
    out = kda_prefill(**args)
    expected = kda_prefill.reference(**ref_args)
    assert out.shape == args["v"].shape and out.dtype == torch.bfloat16
    assert_scaled_close(out, expected, TOL_OUT, "output")
    selected = set(cell.spec["slots"])
    for slot in range(POOL_SLOTS):
        got, ref = pool[slot], ref_args["state_pool"][slot]
        if slot in selected:
            assert_scaled_close(got, ref, TOL_STATE, f"state slot {slot}")
        else:
            assert torch.equal(bits(got), bits(ref)), f"unselected slot {slot} changed"


@torch.inference_mode()
def test_state_row_beyond_int32_offset() -> None:
    """A pool row whose element offset exceeds INT32_MAX updates exactly like row 0.

    The pool is a strided view of KV-cache memory, so a resident row can lie past 2**31 elements of its base.
    """
    spec = dict(heads=96, seq_lens=(256,), slots=(0,), fresh=(False,), state_dtype="float32")
    heads = spec["heads"]
    row_shape = (heads, HEAD_DIM, HEAD_DIM)
    pad_bytes = 64
    row_elems = math.prod(row_shape) + pad_bytes // 4
    far_slot = 2**31 // row_elems + 1
    free_bytes, _ = torch.cuda.mem_get_info()
    if free_bytes < (far_slot + 1) * row_elems * 4 + (1 << 30):
        pytest.skip("needs a state pool beyond 8 GiB")
    args = _inputs(spec, torch.Generator("cuda").manual_seed(7))
    small_pool = args["state_pool"]
    ref_pool = clone_strided(small_pool)
    expected = kda_prefill.reference(**dict(args, state_pool=ref_pool))

    pool = padded_slots(far_slot + 1, row_shape, torch.float32, pad_bytes=pad_bytes)
    assert far_slot * pool.stride(0) > 2**31 - 1
    results = []
    for slot in (0, far_slot):
        pool[slot].copy_(small_pool[0])
        slot_index = torch.tensor([slot], dtype=torch.int32, device="cuda")
        out = kda_prefill(**dict(args, state_pool=pool, state_indices=slot_index))
        results.append((out, pool[slot].clone()))
    (near_out, near_state), (far_out, far_state) = results
    assert_scaled_close(far_out, expected, TOL_OUT, "output")
    assert_scaled_close(far_state, ref_pool[0], TOL_STATE, "state")
    assert torch.equal(bits(far_out), bits(near_out))
    assert torch.equal(bits(far_state), bits(near_state))


@torch.inference_mode()
def test_guard_refuses_a_stale_seq_order() -> None:
    spec = dict(heads=6, seq_lens=(16, 48), slots=(1, 2), fresh=(True, True), state_dtype="float32")
    args = _inputs(spec, torch.Generator("cuda").manual_seed(5))
    args["seq_order"] = torch.tensor([0, 1], dtype=torch.int32, device="cuda")
    with pytest.raises(AssertionError, match="seq_order"):
        with validating(kda_prefill):
            kda_prefill(**args)


@torch.inference_mode()
def test_guard_refuses_repeated_slots() -> None:
    spec = dict(heads=6, seq_lens=(16, 48), slots=(3, 3), fresh=(True, True), state_dtype="float32")
    args = _inputs(spec, torch.Generator("cuda").manual_seed(6))
    with pytest.raises(AssertionError, match="unique"):
        with validating(kda_prefill):
            kda_prefill(**args)
