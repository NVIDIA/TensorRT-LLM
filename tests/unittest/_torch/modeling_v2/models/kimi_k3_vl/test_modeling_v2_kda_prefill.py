# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the kda_prefill catalog entry."""

import math
import zlib
from typing import List

import _kda_reference as kr
import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.ops.ssm.kda_prefill import (
    kda_prefill,
)

__extra_import_path__ = ["../.."]
from _validating import validating

assert torch.cuda.is_available(), "kda_prefill requires a CUDA device"
pytestmark = pytest.mark.skipif(
    torch.cuda.get_device_capability() not in ((10, 0), (10, 3)),
    reason="certified on sm_100 and sm_103",
)

HEAD_DIM, LOWER_BOUND, SCALE, CHUNK = kr.HEAD_DIM, kr.LOWER_BOUND, kr.SCALE, kr.CHUNK
NUM_SLOTS = 12
SLOT_PAD = 4096  # floats past each slot: the cache manager's slot stride is wider than one state


class _Inputs:
    """One prefill batch at Kimi K3's KDA shapes, as its layer hands them to the op."""

    def __init__(
        self,
        heads: int,
        seq_lens: List[int],
        seed: int,
        state_dtype: torch.dtype = torch.float32,
    ) -> None:
        gen = torch.Generator(device="cuda").manual_seed(seed)
        self.heads = heads
        self.seq_lens = seq_lens
        tokens = sum(seq_lens)
        shape = (1, tokens, heads, HEAD_DIM)
        for name in ("q", "k", "v", "g"):
            setattr(self, name, torch.randn(*shape, generator=gen, device="cuda").bfloat16())
        self.beta = torch.randn(1, tokens, heads, generator=gen, device="cuda").bfloat16()
        self.A_log = torch.log(torch.empty(heads, device="cuda").uniform_(1, 16, generator=gen))
        self.dt_bias = torch.empty(heads * HEAD_DIM, device="cuda").uniform_(
            math.log(1e-3), math.log(1e-1), generator=gen
        )
        slot = heads * HEAD_DIM * HEAD_DIM
        storage = (torch.randn(NUM_SLOTS, slot + SLOT_PAD, generator=gen, device="cuda") * 0.1).to(
            state_dtype
        )
        self.storage = storage
        self.pool = storage[:, :slot].view(NUM_SLOTS, heads, HEAD_DIM, HEAD_DIM)
        self.cu_seqlens = kr.cu_seqlens_of(seq_lens)

    def call(self, slots: List[int], fresh: List[bool], reference=False, pool=None, **options):
        """Zero the fresh sequences' rows (the layer's job) and run the op as Kimi K3's layer does; return o."""
        state_pool = self.pool if pool is None else pool
        for s, is_fresh in zip(slots, fresh):
            if is_fresh:
                state_pool[s].zero_()
        state_indices = torch.tensor(slots, dtype=torch.int32, device="cuda")
        call = kda_prefill.reference if reference else kda_prefill
        return call(
            self.q,
            self.k,
            self.v,
            self.g,
            self.beta,
            state_pool,
            state_indices,
            self.cu_seqlens,
            self.A_log,
            self.dt_bias,
            SCALE,
            LOWER_BOUND,
            **options,
        )


def _assert_state_close(got: torch.Tensor, ref: torch.Tensor) -> None:
    """The fp32 state after a chunked bf16 evaluation: within 1e-2 of the state's largest entry."""
    kr.assert_scaled_close(got, ref, atol=1e-2)


@pytest.mark.parametrize("cell", kda_prefill.CELLS, ids=[c.why[:40] for c in kda_prefill.CELLS])
def test_kimi_k3_varlen_prefill(cell) -> None:
    # Kimi K3's TP16 and TP4 packed raw inputs, fresh and continuing resident state,
    # ragged and empty sequences, scattered slots and both supported state dtypes.
    spec = cell.spec
    inp = _Inputs(
        spec["heads"],
        spec["seq_lens"],
        zlib.crc32(cell.why.encode()) % 2**31,
        state_dtype=spec["state_dtype"],
    )
    slots, fresh = spec["slots"], spec["fresh"]
    before = inp.storage.clone()
    ref_pool = inp.pool.clone()
    ref_o = inp.call(slots, fresh, reference=True, pool=ref_pool)
    with validating(kda_prefill):
        o = inp.call(slots, fresh)
    assert o.shape == inp.v.shape and o.dtype == inp.v.dtype
    start = 0
    for seq, (length, slot) in enumerate(zip(inp.seq_lens, slots)):
        if length:
            where = (0, slice(start, start + length))
            kr.assert_scaled_close(o[where], ref_o[where], atol=2e-2)
        _assert_state_close(inp.pool[slot], ref_pool[slot])
        start += length
    untouched = [s for s in range(NUM_SLOTS) if s not in slots]
    assert torch.equal(inp.storage[untouched], before[untouched])
    slot_floats = spec["heads"] * HEAD_DIM * HEAD_DIM
    assert torch.equal(inp.storage[:, slot_floats:], before[:, slot_floats:]), (
        "slot padding written"
    )


def test_continuation_equals_one_call() -> None:
    # A 512-token sequence prefilled as 256 + 256, the second call continuing from the state the
    # first left at the slot, matches one 512-token call; swapping the halves gives other numbers.
    heads = 6
    whole = _Inputs(heads, [256, 256], seed=2)
    one = _Inputs(heads, [512], seed=2)
    one.q, one.k, one.v, one.g, one.beta = whole.q, whole.k, whole.v, whole.g, whole.beta
    one.A_log, one.dt_bias = whole.A_log, whole.dt_bias
    o_one = one.call([3], [True])
    state_one = one.pool[3].clone()
    first = _Inputs(heads, [256], seed=2)
    second = _Inputs(heads, [256], seed=2)
    for part, sl in ((first, slice(0, 256)), (second, slice(256, 512))):
        part.q, part.k, part.v = (
            whole.q[:, sl].contiguous(),
            whole.k[:, sl].contiguous(),
            whole.v[:, sl].contiguous(),
        )
        part.g, part.beta = whole.g[:, sl].contiguous(), whole.beta[:, sl].contiguous()
        part.A_log, part.dt_bias = whole.A_log, whole.dt_bias
    second.storage, second.pool = first.storage, first.pool
    o_first = first.call([3], [True])
    o_second = second.call([3], [False])
    # Each chunk boundary falls on a 64-token chunk boundary, so the halves give the one call's
    # outputs and state up to the evaluation order of the fp32 state hand-off.
    kr.assert_scaled_close(torch.cat([o_first, o_second], dim=1), o_one, atol=2e-3)
    _assert_state_close(first.pool[3], state_one)
    # Negative control: the second half first, then the first half, raises nothing.
    swapped_first = _Inputs(heads, [256], seed=2)
    swapped_second = _Inputs(heads, [256], seed=2)
    for part, src in ((swapped_first, second), (swapped_second, first)):
        part.q, part.k, part.v, part.g, part.beta = src.q, src.k, src.v, src.g, src.beta
        part.A_log, part.dt_bias = whole.A_log, whole.dt_bias
    swapped_second.storage, swapped_second.pool = swapped_first.storage, swapped_first.pool
    swapped_first.call([3], [True])
    swapped_second.call([3], [False])
    assert (swapped_first.pool[3] - state_one).abs().max() > 1e-2


def test_equal_length_batch() -> None:
    # Two equal-length sequences share one packed token dimension and explicit offsets.
    heads = 6
    inp = _Inputs(heads, [256, 256], seed=3)
    slots = [1, 9]
    ref_pool = inp.pool.clone()
    ref_o = inp.call(slots, [True, True], reference=True, pool=ref_pool)
    with validating(kda_prefill):
        o = inp.call(slots, [True, True])
    assert o.shape == inp.v.shape
    for row, slot in enumerate(slots):
        sl = slice(row * 256, (row + 1) * 256)
        kr.assert_scaled_close(o[0, sl], ref_o[0, sl], atol=2e-2)
        _assert_state_close(inp.pool[slot], ref_pool[slot])


def test_output_is_a_new_tensor() -> None:
    # Every call returns a new o: later calls, with the same shapes or others, share no storage with
    # it and leave it unchanged.
    heads = 6
    a = _Inputs(heads, [512], seed=5)
    o_a = a.call([3], [True])
    kept = o_a.clone()
    for lens, seed in (([256], 6), ([512], 7)):
        o_b = _Inputs(heads, lens, seed=seed).call([1], [True])
        assert o_b.untyped_storage().data_ptr() != o_a.untyped_storage().data_ptr()
    assert torch.equal(o_a, kept)


def test_rejects_out_of_contract() -> None:
    inp = _Inputs(6, [256], seed=4)
    good_indices = torch.tensor([0], dtype=torch.int32, device="cuda")
    narrow = torch.zeros(4, 6 * HEAD_DIM * HEAD_DIM + 1, device="cuda")[:, 1:].view(
        4, 6, HEAD_DIM, HEAD_DIM
    )
    args = dict(
        q=inp.q,
        k=inp.k,
        v=inp.v,
        g=inp.g,
        beta=inp.beta,
        state_pool=inp.pool,
        state_indices=good_indices,
        cu_seqlens=inp.cu_seqlens,
        A_log=inp.A_log,
        dt_bias=inp.dt_bias,
        scale=SCALE,
        lower_bound=LOWER_BOUND,
    )
    bad = (
        dict(state_pool=inp.pool[:, :5]),
        dict(state_pool=narrow),
        dict(state_indices=torch.tensor([0, 1], dtype=torch.int32, device="cuda")),
        dict(state_indices=good_indices.float()),
        dict(cu_seqlens=inp.cu_seqlens.int()),
        dict(beta=inp.beta.float()),
        dict(q=inp.q.view(2, 128, 6, HEAD_DIM)),
        dict(seq_order=good_indices),
    )
    for overrides in bad:
        with pytest.raises(ValueError):
            kda_prefill(**(args | overrides))


@pytest.mark.parametrize("bad", ("offsets", "slots", "slot_range", "max_seqlen", "order", "chunks"))
def test_guard_refuses_silent_wrong_inputs(bad) -> None:
    inp = _Inputs(6, [37, 100], seed=9)
    slots = [0, 1]
    options = {}
    if bad == "offsets":
        inp.cu_seqlens = kr.cu_seqlens_of([37, 99])
        match = "partition the packed tokens"
    elif bad == "slots":
        slots = [0, 0]
        match = "unique slot"
    elif bad == "slot_range":
        slots = [0, NUM_SLOTS]
        match = "out of range"
    elif bad == "max_seqlen":
        options["max_seqlen"] = 99
        match = "below a sequence length"
    else:
        options["seq_order"] = torch.tensor(
            [0, 1] if bad == "order" else [1, 0], dtype=torch.int32, device="cuda"
        )
        options["cu_chunks"] = torch.tensor(
            [0, 3, 9 if bad == "chunks" else 10], dtype=torch.int32, device="cuda"
        )
        match = "descending" if bad == "order" else "prefix-sum"
    with pytest.raises(AssertionError, match=match), validating(kda_prefill):
        inp.call(slots, [False, False], **options)


def test_preplanned_metadata_matches_automatic() -> None:
    inp = _Inputs(6, [37, 100, 64], seed=10)
    slots = [7, 2, 9]
    planned_pool = inp.pool.clone()
    automatic = inp.call(slots, [False] * 3, mode="bt16")
    with validating(kda_prefill):
        planned = inp.call(
            slots,
            [False] * 3,
            pool=planned_pool,
            mode="bt16",
            max_seqlen=100,
            seq_order=torch.tensor([1, 2, 0], dtype=torch.int32, device="cuda"),
            cu_chunks=torch.tensor([0, 3, 10, 14], dtype=torch.int32, device="cuda"),
        )
    kr.assert_scaled_close(planned, automatic, atol=2e-2)
    _assert_state_close(planned_pool, inp.pool)
