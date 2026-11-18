# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests of the opt-in pool GEMM pin (``pool_gemm_pin``).

Host: the pinned row set of the ``ROWS`` ranges. GPU (sm_107): the pinned algorithm's output is
the built-in ``F.linear``'s bit for bit at every pinned row count, eagerly and replayed from a
CUDA graph, on several input / weight distributions; the pool ``Linear`` re-classed by
``install`` runs the pin only for the calls it takes and the built-in path for every other call.
"""

import pytest
import torch
import torch.nn.functional as F

from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_107__tep.kernels.pool_gemm_pin import (  # noqa: E501
    pin as pin_path,
)

ON_SM107 = torch.cuda.is_available() and tuple(torch.cuda.get_device_capability()) == (10, 7)
needs_sm107 = pytest.mark.skipif(not ON_SM107, reason="the pin is validated on sm_107 only")

DEV = "cuda"
PINNED = tuple(range(744, 769, 8))


# ------------------------------------------------------------------------------------------ host
def test_pinned_rows_are_the_multiples_of_8_in_the_ranges():
    assert pin_path.pinned_rows(((744, 768),)) == frozenset({744, 752, 760, 768})
    assert pin_path.pinned_rows(((648, 672), (745, 760))) == frozenset(
        {648, 656, 664, 672, 752, 760}
    )
    with pytest.raises(ValueError, match="520..768"):
        pin_path.pinned_rows(((512, 768),))
    with pytest.raises(ValueError, match="no multiple of 8"):
        pin_path.pinned_rows(((745, 751),))


# ------------------------------------------------------------------------------------------ GPU
def _inputs(rows: int, seed: int, kind: str):
    g = torch.Generator(device=DEV).manual_seed(seed)
    w = (torch.randn(pin_path.POOLED, pin_path.HIDDEN, generator=g, device=DEV) * 0.02).to(
        torch.bfloat16
    )
    x = torch.randn(rows, pin_path.HIDDEN, generator=g, device=DEV)
    if kind == "heavy":
        # Heavy tails, a few massive channels and every 10th row zero (graph padding rows).
        x = x * torch.exp2(torch.randint(-6, 7, x.shape, generator=g, device=DEV).float())
        x[:, :6] *= 80.0
        x[9::10] = 0.0
    elif kind == "large":
        x = x * 64.0
    return x.to(torch.bfloat16).contiguous(), w.contiguous()


@pytest.fixture(scope="module")
def pin():
    return pin_path.PinnedPoolGemm(PINNED)


@needs_sm107
@pytest.mark.parametrize("rows", PINNED)
@pytest.mark.parametrize("kind", ("normal", "heavy", "large"))
def test_the_pin_writes_the_bits_of_f_linear(pin, rows, kind):
    for seed in range(3):
        x, w = _inputs(rows, seed + 31 * rows, kind)
        want = F.linear(x, w)
        got = pin(x, w)
        torch.cuda.synchronize()
        assert torch.equal(got.view(torch.int16), want.view(torch.int16)), (rows, kind, seed)


@needs_sm107
def test_the_pin_replays_from_a_cuda_graph(pin):
    inputs = [_inputs(rows, 7 + rows, "heavy") for rows in PINNED]
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for x, w in inputs:
            pin(x, w)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        outs = [pin(x, w) for x, w in inputs]
    for x, w in inputs:  # new values in the captured buffers
        x.copy_(torch.randn_like(x, dtype=torch.float32).to(torch.bfloat16))
    graph.replay()
    torch.cuda.synchronize()
    for (x, w), out in zip(inputs, outs):
        assert torch.equal(out.view(torch.int16), F.linear(x, w).view(torch.int16))


@needs_sm107
def test_a_reclassed_pool_linear_pins_only_the_calls_it_takes(pin):
    from tensorrt_llm._torch.modules.linear import Linear

    linear = Linear(pin_path.HIDDEN, pin_path.POOLED, bias=False, dtype=torch.bfloat16)
    linear.weight.data = _inputs(8, 5, "normal")[1]
    linear.to(DEV)
    base = type(linear)
    linear.__class__ = pin_path._pinned_class(base)
    linear._pool_gemm_pin = pin
    pin._reported.clear()
    for rows in (520, 744, 750, 760, 768):
        x = _inputs(rows, rows, "normal")[0]
        got = linear(x)
        want = F.linear(x, linear.weight)
        assert torch.equal(got.view(torch.int16), want.view(torch.int16)), rows
    # The pin reports each row count it ran (520 and 750 are not pinned).
    assert pin._reported == {744, 760, 768}
    # A non-contiguous input of a pinned row count keeps the built-in path.
    pin._reported.clear()
    wide = torch.zeros(752, 2 * pin_path.HIDDEN, device=DEV, dtype=torch.bfloat16)
    got = linear(wide[:, : pin_path.HIDDEN])
    torch.cuda.synchronize()
    assert pin._reported == set() and got.shape == (752, pin_path.POOLED)
