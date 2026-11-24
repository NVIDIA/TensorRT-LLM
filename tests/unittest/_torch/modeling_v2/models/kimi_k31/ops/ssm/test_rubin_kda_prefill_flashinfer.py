# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the rubin_kda_prefill_conv entry and the context KDA chain on sm_107.

The cells run the fused convolution through its guard against the entry's reference (outputs
within its band, the conv pool bit for bit). The chain tests run convolution then BT16 prefill
the way a context step does, eagerly and under CUDA graph replay with refreshed packed lengths.
"""

import zlib

import pytest
import torch
from _rubin_kda_prefill_inputs import (
    HEAD_DIM,
    LOWER_BOUND,
    SCALE,
    assert_scaled_close,
    clone_written,
    conv_args,
    cu_seqlens_of,
    padded_pool,
    step_schedule,
)

from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.ssm.rubin_kda_prefill import (
    rubin_kda_prefill,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.ssm.rubin_kda_prefill_conv import (
    rubin_kda_prefill_conv,
)

__extra_import_path__ = ["../../../.."]
from _validating import certifying, validating  # noqa: E402,F401 — path above; fixture by name

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 7),
    reason="Rubin entry: certified on sm_107",
)

# fp32 accumulation and an exact SiLU, one BF16 rounding per output (the entry's note): at most one
# BF16 ulp (2**-7 of the value) from the fp32 reference, plus accumulation-order noise.
TOL_CONV = 8e-3
# The chain: the conv band carried through the prefill band (2e-2), with margin.
CHAIN_TOL = 5e-2


def _bits(t: torch.Tensor) -> torch.Tensor:
    return t.view({1: torch.uint8, 2: torch.int16, 4: torch.int32}[t.element_size()])


@pytest.mark.parametrize(
    "cell", rubin_kda_prefill_conv.CELLS, ids=[c.why[:60] for c in rubin_kda_prefill_conv.CELLS]
)
@torch.inference_mode()
def test_certified_cells(cell, certifying) -> None:  # noqa: F811 — pytest resolves the fixture by this name
    certifying(rubin_kda_prefill_conv)
    args = conv_args(cell.spec, zlib.crc32(cell.why.encode()) % 2**31)
    ref_args = clone_written(args, ("conv_state",))
    out = rubin_kda_prefill_conv(**args)
    ref = rubin_kda_prefill_conv.reference(**ref_args)
    torch.cuda.synchronize()
    for name, got, want in zip("qkv", out, ref):
        assert got.shape == want.shape and got.is_contiguous(), name
        assert_scaled_close(name, got, want, TOL_CONV)
    # The window update copies raw inputs: every slot bit for bit, selected or not.
    assert torch.equal(_bits(args["conv_state"]), _bits(ref_args["conv_state"]))


def _chain_inputs(heads: int, seq_lens, seed: int) -> dict:
    """One layer's context inputs: the fused [q|k|v|g] projection, the [f_a|b|pad] projection, the
    padded conv and fp32 state pools, the step metadata."""
    gen = torch.Generator(device="cuda").manual_seed(seed)
    dim = heads * HEAD_DIM
    tokens = sum(seq_lens)
    n = len(seq_lens)

    def rnd(*shape, scale=1.0):
        return torch.randn(*shape, generator=gen, device="cuda") * scale

    conv_pool = padded_pool(n + 2, (3 * dim, 3), torch.bfloat16, "cuda")
    conv_pool.copy_(rnd(*conv_pool.shape, scale=0.5))
    state = padded_pool(n + 2, (heads, HEAD_DIM, HEAD_DIM), torch.float32, "cuda")
    state.copy_(rnd(*state.shape, scale=0.05))
    return dict(
        qkvg=rnd(tokens, 4 * dim).to(torch.bfloat16),
        fab=rnd(tokens, (HEAD_DIM + heads + 7) // 8 * 8).to(torch.bfloat16),
        gate=rnd(tokens, dim).to(torch.bfloat16),
        weight=(rnd(3 * dim, 4, scale=0.3)).to(torch.bfloat16),
        conv_pool=conv_pool,
        state=state,
        slots=torch.arange(n, dtype=torch.int32, device="cuda"),
        has_init=torch.tensor([i % 2 == 0 for i in range(n)], device="cuda"),
        seq_lens=seq_lens,
        A_log=rnd(heads, scale=0.5),
        dt_bias=rnd(dim, scale=0.5),
        heads=heads,
    )


def _chain(inp: dict, conv_op, prefill_op) -> torch.Tensor:
    """Convolution, then prefill on the state rows (fresh rows zeroed first, as the cache does)."""
    heads, seq_lens = inp["heads"], inp["seq_lens"]
    dim = heads * HEAD_DIM
    tokens = sum(seq_lens)
    for i, init in enumerate(inp["has_init"].tolist()):
        if not init:
            inp["state"][int(inp["slots"][i])].zero_()
    q, k, v = conv_op(
        inp["qkvg"][:, : 3 * dim],
        inp["weight"],
        inp["conv_pool"],
        cu_seqlens_of(seq_lens, "cuda", torch.int32),
        inp["slots"],
        inp["has_init"],
        max(seq_lens),
    )
    seq_order, cu_chunks = step_schedule(seq_lens, "cuda")
    return prefill_op(
        q,
        k,
        v,
        inp["gate"].view(1, tokens, heads, HEAD_DIM),
        inp["fab"][:, HEAD_DIM : HEAD_DIM + heads].unsqueeze(0),
        inp["state"],
        inp["slots"],
        cu_seqlens_of(seq_lens, "cuda"),
        inp["A_log"],
        inp["dt_bias"],
        SCALE,
        LOWER_BOUND,
        "bt16",
        seq_order,
        cu_chunks,
    )


@pytest.mark.parametrize("heads", [96, 12])
@torch.inference_mode()
def test_conv_then_prefill_matches_the_reference_chain(heads) -> None:
    """The two kernels chained like a context step equal the two references chained."""
    seq_lens = (100, 64, 300, 37)
    inp = _chain_inputs(heads, seq_lens, seed=heads)
    ref = dict(inp, conv_pool=inp["conv_pool"].clone(), state=inp["state"].clone())
    with validating(rubin_kda_prefill_conv, rubin_kda_prefill):
        out = _chain(inp, rubin_kda_prefill_conv, rubin_kda_prefill)
    want = _chain(ref, rubin_kda_prefill_conv.reference, rubin_kda_prefill.reference)
    torch.cuda.synchronize()
    # The conv's BF16 rounding feeds the chunked recurrence's approximations.
    assert_scaled_close("output", out, want, CHAIN_TOL)
    for i in range(len(seq_lens)):
        assert_scaled_close(f"state {i}", inp["state"][i], ref["state"][i], CHAIN_TOL)
    assert torch.equal(_bits(inp["conv_pool"]), _bits(ref["conv_pool"]))


@torch.inference_mode()
def test_graph_replay_follows_refreshed_lengths() -> None:
    """A captured conv + prefill replays new packed lengths written into its metadata tensors."""
    heads = 12
    first, second = (64, 32, 32), (16, 80, 32)
    inp = _chain_inputs(heads, first, seed=3)
    tokens = sum(first)
    starts = cu_seqlens_of(first, "cuda", torch.int32)
    cu = cu_seqlens_of(first, "cuda")
    seq_order, cu_chunks = step_schedule(first, "cuda")
    beta = inp["fab"][:, HEAD_DIM : HEAD_DIM + heads].unsqueeze(0)
    gate = inp["gate"].view(1, tokens, heads, HEAD_DIM)

    def step():
        q, k, v = rubin_kda_prefill_conv(
            inp["qkvg"][:, : 3 * heads * HEAD_DIM],
            inp["weight"],
            inp["conv_pool"],
            starts,
            inp["slots"],
            inp["has_init"],
            max(max(first), max(second)),
        )
        return rubin_kda_prefill(
            q,
            k,
            v,
            gate,
            beta,
            inp["state"],
            inp["slots"],
            cu,
            inp["A_log"],
            inp["dt_bias"],
            SCALE,
            LOWER_BOUND,
            "bt16",
            seq_order,
            cu_chunks,
        )

    for i, init in enumerate(inp["has_init"].tolist()):
        if not init:
            inp["state"][i].zero_()  # the cache zeroes fresh rows before the step
    pools = (inp["conv_pool"].clone(), inp["state"].clone())
    step()  # compile both kernels outside the capture
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = step()
    # Replay with the second lengths, from the original pools.
    inp["conv_pool"].copy_(pools[0])
    inp["state"].copy_(pools[1])
    starts.copy_(cu_seqlens_of(second, "cuda", torch.int32))
    cu.copy_(cu_seqlens_of(second, "cuda"))
    order, chunks = step_schedule(second, "cuda")
    seq_order.copy_(order)
    cu_chunks.copy_(chunks)
    graph.replay()
    torch.cuda.synchronize()

    ref = dict(inp, conv_pool=pools[0].clone(), state=pools[1].clone(), seq_lens=second)
    want = _chain(ref, rubin_kda_prefill_conv.reference, rubin_kda_prefill.reference)
    assert_scaled_close("replayed output", out, want, CHAIN_TOL)
    assert torch.equal(_bits(inp["conv_pool"]), _bits(ref["conv_pool"]))
