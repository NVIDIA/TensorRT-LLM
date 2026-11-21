# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU tests for the inkling2_short_conv and inkling2_dynamic_short_conv entries."""

import zlib

import pytest
import torch

from tensorrt_llm._torch.modeling_v2.models.inkling2.ops.conv.inkling2_short_conv import (
    inkling2_dynamic_short_conv,
    inkling2_short_conv,
)

__extra_import_path__ = ["../../../.."]
from _validating import assert_within_ulp, certifying, validating  # noqa: E402,F401

assert torch.cuda.is_available(), "the Inkling 2 short convolutions require a CUDA device"

NUM_SLOTS = 12


def _seed(text: str) -> int:
    return zlib.crc32(text.encode()) % 2**31


def _build(spec, seed, dynamic):
    """The step of `spec["requests"]` ((new, cached) pairs) on random inputs and windows."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    channels = spec["channels"]
    lengths = [n for n, _ in spec["requests"]]
    num_tokens = sum(lengths)
    x = torch.randn(num_tokens, channels, generator=g, device="cuda").to(torch.bfloat16)
    taps = 0.5 * torch.randn(4, channels, generator=g, device="cuda")
    # Every slot holds a window, including the ones of fresh requests: has_state 0 must ignore it.
    state = torch.randn(NUM_SLOTS, 3, channels, generator=g, device="cuda").to(torch.bfloat16)
    perm = torch.randperm(NUM_SLOTS, generator=torch.Generator().manual_seed(seed))
    slots = perm[: len(lengths)].to(torch.int32).cuda()
    bounds = [0]
    for n in lengths:
        bounds.append(bounds[-1] + n)
    query_start = torch.tensor(bounds, dtype=torch.int32, device="cuda")
    has_state = torch.tensor(
        [int(c > 0) for _, c in spec["requests"]], dtype=torch.int32, device="cuda"
    )
    token_request = torch.repeat_interleave(
        torch.arange(len(lengths), dtype=torch.int32), torch.tensor(lengths)
    ).cuda()
    if not dynamic:
        return x, taps, state, slots, query_start, has_state, token_request
    heads = spec["heads"]
    dynamic_taps = (0.3 * torch.randn(num_tokens, heads * 4, generator=g, device="cuda")).to(
        torch.bfloat16
    )
    return (
        x,
        taps,
        dynamic_taps,
        state,
        slots,
        query_start,
        has_state,
        token_request,
        channels // heads,
    )


def _check_cell(op, args, state_index):
    op_args, ref_args = list(args), list(args)
    op_args[state_index] = args[state_index].clone()
    ref_args[state_index] = args[state_index].clone()
    out = op(*op_args)
    ref = op.reference(*ref_args)
    assert out.shape == ref.shape and out.dtype == ref.dtype
    # The op and the reference sum the same four fp32 products in the same order; FMA
    # contraction moves the fp32 sum by an ulp, which a bf16 rounding boundary can turn into one
    # bf16 ulp of the output.
    assert_within_ulp(out, ref, element_ulp=1.0, rms_ulp=0.05)
    # The window is a copy of inputs: exact.
    torch.testing.assert_close(op_args[state_index], ref_args[state_index], rtol=0, atol=0)


@pytest.mark.parametrize(
    "cell", inkling2_short_conv.CELLS, ids=[c.why[:48] for c in inkling2_short_conv.CELLS]
)
def test_static_certified_cells(cell, certifying) -> None:  # noqa: F811
    certifying(inkling2_short_conv)
    _check_cell(
        inkling2_short_conv, _build(cell.spec, _seed(cell.why), dynamic=False), state_index=2
    )


@pytest.mark.parametrize(
    "cell",
    inkling2_dynamic_short_conv.CELLS,
    ids=[c.why[:48] for c in inkling2_dynamic_short_conv.CELLS],
)
def test_dynamic_certified_cells(cell, certifying) -> None:  # noqa: F811
    certifying(inkling2_dynamic_short_conv)
    _check_cell(
        inkling2_dynamic_short_conv, _build(cell.spec, _seed(cell.why), dynamic=True), state_index=3
    )


@pytest.mark.parametrize("first", [1, 2, 21])
@pytest.mark.parametrize("dynamic", [False, True], ids=["static", "dynamic"])
def test_chunked_steps_equal_one_prefill(dynamic, first) -> None:
    """A fresh prefill of `first` tokens over a stale window, then chunks and decodes, equals one
    29-token prefill: the window carries exactly the inputs a step needs from the steps before it,
    and a fresh request shorter than the window starts it with zeros, not the stale inputs."""
    channels, heads = 256, 2
    g = torch.Generator(device="cuda").manual_seed(5)
    total = 29
    x = torch.randn(total, channels, generator=g, device="cuda").to(torch.bfloat16)
    taps = 0.5 * torch.randn(4, channels, generator=g, device="cuda")
    dyn = (0.3 * torch.randn(total, heads * 4, generator=g, device="cuda")).to(torch.bfloat16)

    def run(start, end, state, has_state):
        n = end - start
        args = [x[start:end], taps]
        if dynamic:
            args.append(dyn[start:end])
        args += [
            state,
            torch.tensor([1], dtype=torch.int32, device="cuda"),
            torch.tensor([0, n], dtype=torch.int32, device="cuda"),
            torch.tensor([has_state], dtype=torch.int32, device="cuda"),
            torch.zeros(n, dtype=torch.int32, device="cuda"),
        ]
        if dynamic:
            args.append(channels // heads)
        op = inkling2_dynamic_short_conv if dynamic else inkling2_short_conv
        return op(*args)

    whole = run(0, total, torch.zeros(3, 3, channels, dtype=torch.bfloat16, device="cuda"), 0)
    state = torch.randn(3, 3, channels, device="cuda").to(torch.bfloat16)
    pieces, start = [], 0
    for end in sorted({first, *(e for e in (21, 23, 24, 25, 26, 27, 28, 29) if e > first)}):
        pieces.append(run(start, end, state, int(start > 0)))
        if start == 0 and end < 3:
            zeros = torch.zeros(3 - end, channels, dtype=torch.bfloat16, device="cuda")
            torch.testing.assert_close(state[1], torch.cat([zeros, x[:end]]), rtol=0, atol=0)
        start = end
    torch.testing.assert_close(torch.cat(pieces), whole, rtol=0, atol=0)
    torch.testing.assert_close(state[1], x[-3:], rtol=0, atol=0)


def test_dynamic_taps_are_per_head() -> None:
    """Swapping the two heads' dynamic taps changes the output: one head's taps never reach the
    other head's channels."""
    spec = dict(channels=256, heads=2, requests=((7, 0),))
    args = list(_build(spec, 3, dynamic=True))
    out = inkling2_dynamic_short_conv(*args[:3], args[3].clone(), *args[4:])
    swapped = args[2].view(-1, 2, 4).flip(1).reshape(-1, 8).contiguous()
    out_swapped = inkling2_dynamic_short_conv(args[0], args[1], swapped, args[3].clone(), *args[4:])
    assert not torch.equal(out[:, :128], out_swapped[:, :128])


@pytest.mark.parametrize(
    "case,match",
    [
        ("duplicate_slot", "own slot"),
        ("bf16_taps", "lag-major fp32"),
        ("bad_token_request", "token_request"),
        ("strided_slots", "slots must be contiguous"),
    ],
)
def test_guard_refuses_inputs_the_op_answers_wrongly(case, match) -> None:
    spec = dict(channels=256, requests=((3, 0), (1, 4)))
    x, taps, state, slots, query_start, has_state, token_request = _build(spec, 9, dynamic=False)
    if case == "duplicate_slot":
        slots = slots[:1].repeat(2)
    elif case == "bf16_taps":
        taps = taps.to(torch.bfloat16)
    elif case == "strided_slots":
        # The same slot values, as a view the kernels would index at unit stride.
        slots = torch.stack([slots, slots + NUM_SLOTS], dim=1)[:, 0]
    else:
        token_request = torch.zeros_like(token_request)
    with pytest.raises(AssertionError, match=match):
        with validating(inkling2_short_conv):
            inkling2_short_conv(x, taps, state, slots, query_start, has_state, token_request)
