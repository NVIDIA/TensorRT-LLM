# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for flashinfer_chunk_gdn_inforward_split (the in-forward Mamba snapshot split of
the Qwen3.8 GDN prefill).

A context chunk that reaches the prompt end is cut at the snapshot point P inside one
forward. The test holds what the cache manager then relies on:

- each split request's slot holds the state at P and its staging row the state at the
  prompt end, an unsplit request's slot the state at its chunk end, and the outputs are
  those of the two context iterations the split replaces (fp64 reference, the op's band);
- no other slot and no other staging row is written;
- a single split request's tail read and written as a row slice is bitwise the gathered
  tail;
- `Mamba2Metadata` names the tail's row range only when exactly one request is split.
"""

import types

import pytest
import torch

from tensorrt_llm._torch.modeling_v2.models.qwen4_exp.ops.ssm.flashinfer_chunk_gated_delta_rule import (  # noqa: E501
    INPLACE_MAX_TOKENS,
    RANK4_GEOMETRY,
    flashinfer_chunk_gdn_inforward_split,
)
from tensorrt_llm._torch.modules.mamba.mamba2_metadata import Mamba2Metadata
from tensorrt_llm._torch.pyexecutor.kv_cache.mamba_cache_manager import InforwardSnapshotEntry

assert torch.cuda.is_available(), "flashinfer_chunk_gdn_inforward_split requires a CUDA device"

DEV = "cuda"
#: The band of flashinfer_chunk_gated_delta_rule against fp64 (its test's constants).
RTOL, ATOL_OUTPUT, ATOL_STATE = 1.6e-2, 2e-3, 2e-2
NUM_SLOTS, STAGE_ROWS = 16, 4
#: A stand-in for the conv window width: the split's `conv_win` is not read by this op.
_CONV_WIDTH = 3

# (request id, slot, chunk length, carries a state, tokens past P or None)
#: Three requests split (the gathered tail), two not; one of the split ones is fresh.
_MANY = (
    (11, 5, 40, True, None),
    (12, 2, 70, True, 2),
    (13, 7, 100, False, 2),
    (14, 9, 33, False, None),
    (15, 0, 130, True, 2),
)
#: Exactly one request split (the sliced tail), not the last row of the batch.
_ONE = (
    (21, 3, 90, True, None),
    (22, 11, 200, True, 2),
    (23, 6, 17, False, None),
)
#: A longer tail than the two tokens of `additional_snapshot_offsets_from_end: [2]`.
_LONG_TAIL = (
    (31, 4, 150, True, 7),
    (32, 1, 64, True, None),
)


def _split(batch):
    """`Mamba2Metadata.inforward_split` for `batch`, as the executor plans it."""
    plan, row = {}, 0
    for rid, _, length, _, after in batch:
        if after is not None:
            plan[rid] = InforwardSnapshotEntry(
                row=row, point=length - after, num_before=length - after, num_after=after
            )
            row += 1
    slots = {rid: slot for rid, slot, _, _, _ in batch}
    manager = types.SimpleNamespace(
        conv_state_shape=[8, _CONV_WIDTH],
        inforward_snapshot_rows=STAGE_ROWS,
        inforward_snapshot_max_tail=16,
        inforward_snapshot_plan=plan,
        state_index=lambda rid: slots[rid],
    )
    attn = types.SimpleNamespace(
        request_ids=[rid for rid, _, _, _, _ in batch],
        seq_lens=torch.tensor([length for _, _, length, _, _ in batch], dtype=torch.int32),
    )
    metadata = Mamba2Metadata(max_batch_size=8, chunk_size=128)
    metadata._prepare_inforward_split(attn, manager, plan, len(batch), len(batch))
    return metadata.inforward_split


def _inputs(batch, seed):
    """Post-conv GDN operands (unit-norm q/k, the linear gate, beta a sigmoid), a bf16 pool
    and staging rows of random bits, so a row the op should not write is seen if it is."""
    gen = torch.Generator(device=DEV).manual_seed(seed)
    num_k_heads, num_v_heads, head_k, head_v = RANK4_GEOMETRY
    total = sum(length for _, _, length, _, _ in batch)
    q = torch.randn(1, total, num_k_heads, head_k, generator=gen, device=DEV)
    k = torch.randn(1, total, num_k_heads, head_k, generator=gen, device=DEV)
    q = (q / torch.sqrt(q.pow(2).sum(-1, keepdim=True) + 1e-6)).bfloat16()
    k = (k / torch.sqrt(k.pow(2).sum(-1, keepdim=True) + 1e-6)).bfloat16()
    v = torch.randn(1, total, num_v_heads, head_v, generator=gen, device=DEV).bfloat16()
    a_log = torch.log(torch.rand(num_v_heads, generator=gen, device=DEV) * 15 + 1)
    x = torch.randn(1, total, num_v_heads, generator=gen, device=DEV)
    alpha = torch.exp(-torch.exp(a_log) * torch.nn.functional.softplus(x)).float()
    beta = torch.rand(1, total, num_v_heads, generator=gen, device=DEV).sigmoid().float()

    def states(rows):
        return (
            torch.randn(rows, num_v_heads, head_v, head_k, generator=gen, device=DEV) * 0.1
        ).bfloat16()

    return (q, k, v, alpha, beta), states(NUM_SLOTS), states(STAGE_ROWS)


def _run(batch, operands, pool, stage, sliced=True):
    """The op as `PrefillTarget._gdn_core` calls it, on copies of `pool` and `stage`.

    With `sliced` false it names no tail row range, so a single split request's tail is
    gathered like several requests' tails."""
    split = _split(batch)
    q, k, v, alpha, beta = operands
    pool, stage = pool.clone(), stage.clone()
    index = torch.int32
    carried = torch.zeros(split.gather.shape[0], dtype=torch.bool, device=DEV)
    carried.index_copy_(
        0,
        split.main_rows,
        torch.tensor([carries for _, _, _, carries, _ in batch], dtype=torch.bool, device=DEV),
    )
    output = torch.full(
        (1, q.shape[1], v.shape[2], v.shape[3]), -7.0, dtype=torch.bfloat16, device=DEV
    )
    lo, hi = (split.tail_lo, split.tail_hi) if sliced else (-1, -1)
    flashinfer_chunk_gdn_inforward_split(
        q,
        k,
        v,
        alpha,
        beta,
        pool,
        cu_seqlens=split.cu,
        gather=split.gather.to(index),
        has_initial_state=carried,
        main_rows=split.main_rows,
        live=split.live.to(index),
        tail_tok=split.tail_tok,
        cu_tail=split.cu_tail,
        tail_slots=split.tail_slots,
        stage=stage,
        stage_rows=split.stage_rows,
        stage_idx=split.stage_rows.to(index),
        output=output,
        tail_lo=lo,
        tail_hi=hi,
    )
    torch.cuda.synchronize()
    return output, pool, stage


def _scan(operands, state, start, end):
    """fp64 token-at-a-time recurrence over rows `[start, end)` from `state`:
    `(outputs, final state)`."""
    q, k, v, alpha, beta = operands
    dt = torch.float64
    repeats = v.shape[2] // q.shape[2]
    scale = q.shape[3] ** -0.5
    outs = []
    for t in range(start, end):
        q_t = q[0, t].to(dt).repeat_interleave(repeats, dim=0) * scale
        k_t = k[0, t].to(dt).repeat_interleave(repeats, dim=0)
        state = state * alpha[0, t].to(dt)[:, None, None]
        u = (v[0, t].to(dt) - (state * k_t[:, None, :]).sum(-1)) * beta[0, t].to(dt)[:, None]
        state = state + u[:, :, None] * k_t[:, None, :]
        outs.append((state * q_t[:, None, :]).sum(-1))
    return torch.stack(outs), state


def _two_iterations(batch, operands, pool, stage):
    """What two context iterations leave: the chunk up to P into the slot (rounded to the
    pool's dtype, as the first iteration's scatter does), then the tail from that state."""
    out = torch.empty(
        1, operands[0].shape[1], operands[2].shape[2], operands[2].shape[3], dtype=torch.float64
    ).to(DEV)
    pool, stage = pool.clone(), stage.clone()
    start, row = 0, 0
    for _, slot, length, carries, after in batch:
        before = length if after is None else length - after
        state = pool[slot].to(torch.float64)
        if not carries:
            state = torch.zeros_like(state)
        out[0, start : start + before], state = _scan(operands, state, start, start + before)
        pool[slot] = state.to(pool.dtype)
        if after is not None:
            out[0, start + before : start + length], state = _scan(
                operands, pool[slot].to(torch.float64), start + before, start + length
            )
            stage[row] = state.to(stage.dtype)
            row += 1
        start += length
    return out.to(torch.bfloat16), pool, stage


@pytest.mark.parametrize(
    "batch", [_MANY, _ONE, _LONG_TAIL], ids=["three_split", "one_split", "long_tail"]
)
def test_split_is_the_two_context_iterations(batch) -> None:
    operands, pool, stage = _inputs(batch, seed=1)
    out, got_pool, got_stage = _run(batch, operands, pool, stage)
    want_out, want_pool, want_stage = _two_iterations(batch, operands, pool, stage)

    torch.testing.assert_close(out, want_out, rtol=RTOL, atol=ATOL_OUTPUT)
    torch.testing.assert_close(got_pool, want_pool, rtol=RTOL, atol=ATOL_STATE)
    torch.testing.assert_close(got_stage, want_stage, rtol=RTOL, atol=ATOL_STATE)
    # The slots of other requests and the staging rows no request was given are not written.
    named = {slot for _, slot, _, _, _ in batch}
    others = [slot for slot in range(NUM_SLOTS) if slot not in named]
    assert torch.equal(got_pool[others], pool[others])
    num_split = sum(after is not None for _, _, _, _, after in batch)
    assert torch.equal(got_stage[num_split:], stage[num_split:])
    # The state at P is not the state at the prompt end: a split that left the wrong one in
    # the slot is outside the band.
    split_slots = [slot for _, slot, _, _, after in batch if after is not None]
    assert not torch.allclose(
        got_pool[split_slots].float(),
        want_stage[:num_split].float(),
        rtol=RTOL,
        atol=ATOL_STATE,
    )


@pytest.mark.parametrize("batch", [_ONE, _LONG_TAIL], ids=["two_token_tail", "long_tail"])
def test_sliced_tail_is_bitwise_the_gathered_tail(batch) -> None:
    operands, pool, stage = _inputs(batch, seed=2)
    gathered = _run(batch, operands, pool, stage, sliced=False)
    sliced = _run(batch, operands, pool, stage, sliced=True)
    for name, got, want in zip(("output", "pool", "stage"), sliced, gathered):
        assert torch.equal(got, want), name


def test_tail_row_range_is_named_for_a_single_split_request_only() -> None:
    one = _split(_ONE)
    assert one.k == 1
    # Request 22's last two rows: it follows a 90-token request and is 200 tokens long.
    assert (one.tail_lo, one.tail_hi) == (90 + 198, 90 + 200)
    assert one.tail_tok.tolist() == list(range(one.tail_lo, one.tail_hi))

    many = _split(_MANY)
    assert many.k == 3
    assert (many.tail_lo, many.tail_hi) == (-1, -1)

    long_tail = _split(_LONG_TAIL)
    assert (long_tail.tail_lo, long_tail.tail_hi) == (150 - 7, 150)
    assert long_tail.num_tail_tokens == 7 <= INPLACE_MAX_TOKENS
