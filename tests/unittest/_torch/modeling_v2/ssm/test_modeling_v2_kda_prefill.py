# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the kda_prefill catalog entry."""

import zlib
from typing import List

import _kda_reference as kr
import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.ops.ssm.kda_prefill import (
    kda_prefill,
)
from tensorrt_llm._torch.modules.fla.index import prepare_chunk_indices

__extra_import_path__ = [".."]
from _validating import validating

assert torch.cuda.is_available(), "kda_prefill requires a CUDA device"
pytestmark = pytest.mark.skipif(
    torch.cuda.get_device_capability() != (10, 0), reason="certified on sm_100 only"
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
        batch_rows: int = 1,
        cu_dtype: torch.dtype = torch.int64,
    ) -> None:
        gen = torch.Generator(device="cuda").manual_seed(seed)
        self.heads = heads
        self.seq_lens = seq_lens
        tokens = sum(seq_lens) if batch_rows == 1 else seq_lens[0]
        for name, t in kr.kimi_k3_inputs(heads, tokens, gen, batch_rows).items():
            setattr(self, name, t)
        # V-first fp32 state rows in a pool with a padded slot stride.
        slot = heads * HEAD_DIM * HEAD_DIM
        storage = torch.randn(NUM_SLOTS, slot + SLOT_PAD, generator=gen, device="cuda") * 0.1
        self.storage = storage
        self.pool = storage[:, :slot].view(NUM_SLOTS, heads, HEAD_DIM, HEAD_DIM)
        self.cu_seqlens = kr.cu_seqlens_of(seq_lens, cu_dtype) if batch_rows == 1 else None

    def call(self, slots: List[int], fresh: List[bool], reference=False, pool=None):
        """Zero the fresh sequences' rows (the layer's job) and run the op as Kimi K3's layer does; return o."""
        state_pool = self.pool if pool is None else pool
        for s, is_fresh in zip(slots, fresh):
            if is_fresh:
                state_pool[s].zero_()
        state_indices = torch.tensor(slots, dtype=torch.int32, device="cuda")
        varlen = self.cu_seqlens is not None
        call = kda_prefill.reference if reference else kda_prefill
        return call(
            self.q,
            self.k,
            self.v,
            self.g,
            self.beta,
            state_pool,
            state_indices,
            SCALE,
            cu_seqlens=self.cu_seqlens,
            chunk_indices=prepare_chunk_indices(self.cu_seqlens, CHUNK) if varlen else None,
            chunk_size=CHUNK,
            safe_gate=True,
            lower_bound=LOWER_BOUND,
            use_gate_in_kernel=True,
            use_beta_sigmoid_in_kernel=True,
            A_log=self.A_log,
            dt_bias=self.dt_bias,
            # The host metadata Kimi K3's layer prepares once per batch.
            varlen_is_aligned=all(n % CHUNK == 0 for n in self.seq_lens) if varlen else None,
            single_sequence_length=self.seq_lens[0] if varlen and len(self.seq_lens) == 1 else None,
        )


def _assert_state_close(got: torch.Tensor, ref: torch.Tensor) -> None:
    """The fp32 state after a chunked bf16 evaluation: within 1e-2 of the state's largest entry."""
    kr.assert_scaled_close(got, ref, atol=1e-2)


@pytest.mark.parametrize("cell", kda_prefill.CELLS, ids=[c.why[:40] for c in kda_prefill.CELLS])
def test_kimi_k3_varlen_prefill(cell) -> None:
    # Kimi K3's prefill call at its TP16 rank slice (6 heads) and TP4 (24): a varlen batch of fresh
    # sequences and continuations (chunked prefill / a reused prefix, which start from the slot's
    # state), sequence lengths not chunk multiples, scattered slots of a padded pool. The layer's
    # cu_seqlens are int64; the op takes int32 as well.
    spec = cell.spec
    inp = _Inputs(
        spec["heads"],
        spec["seq_lens"],
        zlib.crc32(cell.why.encode()) % 2**31,
        batch_rows=spec["batch_rows"],
        cu_dtype=spec["cu_dtype"],
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
        where = (0, slice(start, start + length)) if spec["batch_rows"] == 1 else (seq,)
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
    # Two equal-length sequences without cu_seqlens: q / k / v are [2, T, H, K].
    heads = 6
    inp = _Inputs(heads, [256], seed=3, batch_rows=2)
    slots = [1, 9]
    ref_pool = inp.pool.clone()
    ref_o = inp.call(slots, [True, True], reference=True, pool=ref_pool)
    with validating(kda_prefill):
        o = inp.call(slots, [True, True])
    assert o.shape == inp.v.shape
    for row, slot in enumerate(slots):
        kr.assert_scaled_close(o[row], ref_o[row], atol=2e-2)
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
    bf16_pool = inp.pool.to(torch.bfloat16)
    narrow = torch.zeros(4, 6 * HEAD_DIM * HEAD_DIM + 1, device="cuda")[:, 1:].view(
        4, 6, HEAD_DIM, HEAD_DIM
    )
    bad = {
        "bf16 pool": (bf16_pool, good_indices),
        "wrong head count": (inp.pool[:, :5], good_indices),
        "misaligned pool": (narrow, good_indices),
        "two indices for one sequence": (
            inp.pool,
            torch.tensor([0, 1], dtype=torch.int32, device="cuda"),
        ),
        "float indices": (inp.pool, good_indices.float()),
    }
    for name, (pool, indices) in bad.items():
        try:
            kda_prefill(
                inp.q,
                inp.k,
                inp.v,
                inp.g,
                inp.beta,
                pool,
                indices,
                SCALE,
                cu_seqlens=inp.cu_seqlens,
                chunk_indices=prepare_chunk_indices(inp.cu_seqlens, CHUNK),
                safe_gate=True,
                lower_bound=LOWER_BOUND,
                use_gate_in_kernel=True,
                A_log=inp.A_log,
                dt_bias=inp.dt_bias,
            )
        except ValueError:
            continue
        raise AssertionError(f"{name}: the op accepted an out-of-contract call")


@pytest.mark.parametrize("bad", ("head_dim", "device", "padding"))
def test_guard_refuses_silent_wrong_inputs(bad) -> None:
    inp = _Inputs(6, [256], seed=9)
    if bad == "head_dim":
        inp.q = inp.q[..., :64].contiguous()
        match = "K == V == 128"
    elif bad == "device":
        inp.g = inp.g.cpu()
        match = "share q's device"
    else:
        inp.cu_seqlens = kr.cu_seqlens_of([255])
        inp.seq_lens = [255]
        match = "padding of q/k/v/beta must be zero"
    with pytest.raises(AssertionError, match=match), validating(kda_prefill):
        inp.call([0], [True])
