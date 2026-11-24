# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``rubin_kda_prefill`` as two launches (``phase="prep"`` then ``phase="chain"``) against one call.

The cep hand-off computes a lead's chunk factors (the prefill's first kernel, which reads q / k /
gate / beta and no state) before its hand-off round and runs only the state recursion in the
round. On one sm_107 GPU, per case: the ``"prep"`` half, then the state the round would hand over
written into the state row, then the ``"chain"`` half, give the output, the end state and the
snapshot rows of one ``"both"`` call from that state bit for bit. Cases: the hand-off's lead at
cep8 (16,384 rows, 96 heads, FP32 state), a lead with a block-reuse snapshot inside it, a short
lead, and two sequences at 96 heads (the engine route: the ``"prep"`` half issues nothing).
"""

import pytest
import torch
from _rubin_kda_prefill_inputs import HEAD_DIM, LOWER_BOUND, SCALE, coalesced_pool, step_schedule

from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.ssm.rubin_kda_prefill import (
    rubin_kda_prefill,
    rubin_kda_prefill_workspace,
)

_SM107 = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 7),
    reason="the cep hand-off runs on sm_107",
)

HEADS = 96

# (sequence lengths, snapshot chunks of 64 tokens per sequence (0: none))
_CASES = {
    "cep8_lead_16384": ((16384,), (0,)),
    "cep8_lead_snapshot": ((16384,), (129,)),
    "short_lead_700": ((700,), (0,)),
    "two_sequences_engine": ((4000, 2384), (0, 0)),
}


def _inputs(seq_lens, snap_chunks, seed: int) -> dict:
    gen = torch.Generator(device="cuda").manual_seed(seed)
    tokens, n = sum(seq_lens), len(seq_lens)
    shape = (1, tokens, HEADS, HEAD_DIM)

    def rnd(*size, scale=1.0):
        return torch.randn(*size, generator=gen, device="cuda") * scale

    pool = coalesced_pool(2 * n + 2, (HEADS, HEAD_DIM, HEAD_DIM), torch.float32, "cuda")
    pool.copy_(rnd(*pool.shape, scale=0.05))
    order, cu_chunks = step_schedule(seq_lens, "cuda")
    starts = [0]
    for length in seq_lens:
        starts.append(starts[-1] + length)
    args = dict(
        q=rnd(*shape).to(torch.bfloat16),
        k=rnd(*shape).to(torch.bfloat16),
        v=rnd(*shape, scale=0.5).to(torch.bfloat16),
        g=rnd(*shape).to(torch.bfloat16),
        beta=rnd(1, tokens, HEADS).to(torch.bfloat16),
        state_pool=pool,
        state_indices=torch.arange(n, dtype=torch.int32, device="cuda"),
        cu_seqlens=torch.tensor(starts, dtype=torch.int64, device="cuda"),
        A_log=rnd(HEADS, scale=0.5),
        dt_bias=rnd(HEADS * HEAD_DIM, scale=0.5),
        scale=SCALE,
        lower_bound=LOWER_BOUND,
        mode="bt16",
        seq_order=order,
        cu_chunks=cu_chunks,
    )
    if any(snap_chunks):
        args["snapshot_rows"] = torch.tensor(
            [n + b if c else -1 for b, c in enumerate(snap_chunks)],
            dtype=torch.int32,
            device="cuda",
        )
        args["snapshot_chunks"] = torch.tensor(snap_chunks, dtype=torch.int32, device="cuda")
    # The state the hand-off round writes into the lead's row before the recursion.
    handed = rnd(n, HEADS, HEAD_DIM, HEAD_DIM, scale=0.05)
    return args, handed


def _same_bits(a: torch.Tensor, b: torch.Tensor) -> bool:
    return torch.equal(a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8))


@_SM107
@pytest.mark.parametrize("case", sorted(_CASES))
@torch.inference_mode()
def test_prep_then_chain_is_the_call_bit_for_bit(case: str) -> None:
    seq_lens, snap_chunks = _CASES[case]
    args, handed = _inputs(seq_lens, snap_chunks, seed=len(case))
    rows = args["state_indices"].long()
    pool = args["state_pool"]
    pool0 = pool.clone()

    # One call from the handed-over state.
    pool[rows] = handed
    out_both = torch.empty_like(args["q"])
    rubin_kda_prefill(**args, out=out_both)
    pool_both = pool.clone()

    # The two halves: the factors while the state row still holds something else, then the state
    # handed over, then the recursion.
    pool.copy_(pool0)
    out_split = torch.full_like(args["q"], float("nan"))
    ws = rubin_kda_prefill_workspace(args["q"], pool, args["state_indices"], LOWER_BOUND)
    if ws is not None:
        ws.fill_(0xFF)  # NaN bytes: the chain half must read only what the prep half wrote
    rubin_kda_prefill(**args, out=out_split, phase="prep", workspace=ws)
    pool[rows] = handed
    rubin_kda_prefill(**args, out=out_split, phase="chain", workspace=ws)

    torch.cuda.synchronize()
    assert _same_bits(out_split, out_both), case
    assert _same_bits(pool, pool_both), case
    engine = 2 * len(seq_lens) * HEADS > torch.cuda.get_device_properties(0).multi_processor_count
    assert (ws is None) == engine, (case, ws is None, engine)


@_SM107
@torch.inference_mode()
def test_a_phase_needs_its_workspace_and_the_planned_schedule() -> None:
    args, _ = _inputs((700,), (0,), seed=3)
    with pytest.raises(ValueError, match="workspace"):
        rubin_kda_prefill(**args, phase="chain", workspace=None)
    unplanned = dict(args, seq_order=None, cu_chunks=None)
    ws = rubin_kda_prefill_workspace(args["q"], args["state_pool"], args["state_indices"], -5.0)
    with pytest.raises(ValueError, match="planned"):
        rubin_kda_prefill(**unplanned, phase="prep", workspace=ws)
    with pytest.raises(ValueError, match="phase"):
        rubin_kda_prefill(**args, phase="half", workspace=ws)
