# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the kda_prefill catalog entry."""

from typing import List

import _kda_reference as kr
import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog.ssm.kda_prefill import kda_prefill
from tensorrt_llm._torch.modules.fla.index import prepare_chunk_indices

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

    def call(self, slots: List[int], fresh: List[bool]):
        """Zero the fresh sequences' rows (the layer's job) and run the op as Kimi K3's layer does; return o."""
        for s, is_fresh in zip(slots, fresh):
            if is_fresh:
                self.pool[s].zero_()
        state_indices = torch.tensor(slots, dtype=torch.int32, device="cuda")
        varlen = self.cu_seqlens is not None
        return kda_prefill(
            self.q,
            self.k,
            self.v,
            self.g,
            self.beta,
            self.pool,
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


def _check_sequences(inp: _Inputs, o: torch.Tensor, slots, initial_states) -> None:
    assert o.shape == inp.v.shape and o.dtype == inp.v.dtype
    start = 0
    for length, slot, state0 in zip(inp.seq_lens, slots, initial_states):
        sl = slice(start, start + length)
        ref_o, ref_s = kr.recurrent_reference(
            inp.q[0, sl],
            inp.k[0, sl],
            inp.v[0, sl],
            inp.g[0, sl],
            inp.beta[0, sl],
            state0,
            inp.A_log,
            inp.dt_bias,
        )
        # Chunked evaluation with bf16 operands against an fp64 recurrence: within 2e-2 of each
        # sequence's largest output, 1e-2 of its state's largest entry.
        kr.assert_scaled_close(o[0, sl], ref_o, atol=2e-2)
        _assert_state_close(inp.pool[slot], ref_s)
        start += length


def test_kimi_k3_varlen_prefill() -> None:
    # Kimi K3's prefill call at its TP16 rank slice (6 heads) and TP4 (24): a varlen batch of fresh
    # sequences and continuations (chunked prefill / a reused prefix, which start from the slot's
    # state), sequence lengths not chunk multiples, scattered slots of a padded pool. The layer's
    # cu_seqlens are int64; the op takes int32 as well.
    for heads, seed, cu_dtype in ((6, 0, torch.int64), (24, 1, torch.int64), (6, 8, torch.int32)):
        inp = _Inputs(heads, [100, 64, 300, 37], seed, cu_dtype=cu_dtype)
        slots, fresh = [7, 2, 10, 4], [True, False, True, False]
        before = inp.storage.clone()
        initial = [
            torch.zeros_like(inp.pool[s]) if f else inp.pool[s].clone()
            for s, f in zip(slots, fresh)
        ]
        o = inp.call(slots, fresh)
        _check_sequences(inp, o, slots, initial)
        untouched = [s for s in range(NUM_SLOTS) if s not in slots]
        assert torch.equal(inp.storage[untouched], before[untouched])
        slot_floats = heads * HEAD_DIM * HEAD_DIM
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
    o_one = one.call([3], [True]).clone()  # o is runner scratch: copy it before the next call
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
    o_first = first.call([3], [True]).clone()
    o_second = second.call([3], [False]).clone()
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
    initial = [torch.zeros_like(inp.pool[s]) for s in slots]
    o = inp.call(slots, [True, True])
    assert o.shape == inp.v.shape
    for row, (slot, state0) in enumerate(zip(slots, initial)):
        ref_o, ref_s = kr.recurrent_reference(
            inp.q[row],
            inp.k[row],
            inp.v[row],
            inp.g[row],
            inp.beta[row],
            state0,
            inp.A_log,
            inp.dt_bias,
        )
        kr.assert_scaled_close(o[row], ref_o, atol=2e-2)
        _assert_state_close(inp.pool[slot], ref_s)


def test_output_is_runner_scratch() -> None:
    # The returned o is a view of the runner's scratch for that batch shape: the next call with the
    # same shapes writes its own o into the same storage, so a caller consumes (or copies) o before
    # its next call. A call with other shapes leaves it alone.
    heads = 6
    a = _Inputs(heads, [512], seed=5)
    o_a = a.call([3], [True])
    kept = o_a.clone()
    other = _Inputs(heads, [256], seed=6).call([1], [True])
    assert other.untyped_storage().data_ptr() != o_a.untyped_storage().data_ptr()
    assert torch.equal(o_a, kept)
    same = _Inputs(heads, [512], seed=7).call([1], [True])
    assert same.untyped_storage().data_ptr() == o_a.untyped_storage().data_ptr()
    assert not torch.equal(o_a, kept)


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
