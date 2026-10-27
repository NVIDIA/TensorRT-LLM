# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""TP2CTX_SWIGLU_MXFP8 (#5): the swigluoai activation and down_proj's MXFP8 quantize as one C++ kernel.

trtllm::minimax_m31_swiglu_oai_mxfp8 must be bitwise the stock pair, the Triton silu_and_mul (limit 7, alpha 1.702,
beta 1) followed by mxfp8_quantize(h, True): the E4M3 values, the swizzled UE8M0 scales (padding rows zeroed) and the
optional bf16 activation. The tp2ctx target swaps it into the swigluoai MLPs whose down_proj is a TP2CTX_NORMQ
consumer (kill switch TRTLLM_M31_DISABLE_TP2CTX_SWIGLU_MXFP8). GPU tests need an sm_10x build carrying the op.
"""

from pathlib import Path

import pytest
import torch

import tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3 as _family

_TARGET = Path(_family.__file__).resolve().parent / "minimax_m31_nvfp4__sm_107__tp2ctx"


def test_tp2ctx_swaps_only_swigluoai_normq_mlps_with_a_kill_switch():
    src = (_TARGET / "modeling.py").read_text()
    assert '_SWIGLU_MXFP8 = not m31_disabled("TP2CTX_SWIGLU_MXFP8")' in src
    assert "module.__class__ = _SwigluMxfp8GatedMLP" in src
    flat = "".join(src.split())  # the condition may wrap across lines
    assert "(module.swiglu_alpha,module.swiglu_beta,module.swiglu_limit)==(1.702,1.0,7.0)" in flat
    # The swap runs after down_proj's quant method became the TP2CTX_NORMQ consumer.
    assert src.index("module.quant_method.__class__ = _NormqMXFP8LinearMethod") < src.index(
        "module.__class__ = _SwigluMxfp8GatedMLP"
    )


def _op():
    import tensorrt_llm  # noqa: F401  (loads the trtllm op library)

    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("needs an SM100-family GPU")
    op = getattr(torch.ops.trtllm, "minimax_m31_swiglu_oai_mxfp8", None)
    if op is None:
        pytest.skip("this build lacks minimax_m31_swiglu_oai_mxfp8")
    return op


def _inputs(rows, d, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(rows, 2 * d, generator=g, device="cuda") * 3.0
    if rows >= 4:
        x[1] = 0.0  # an all-zero row: zero scales and zero values
        x[2, :d] = torch.linspace(
            -12.0, 12.0, d, device="cuda"
        )  # gates across both sides of the clamp
        x[2, d:] = torch.linspace(-12.0, 12.0, d, device="cuda")
        x[3] *= 1e-30  # subnormal-range activations
    return x.to(torch.bfloat16).contiguous()


def _reference(x):
    h = torch.ops.trtllm.silu_and_mul(x, swiglu_limit=7.0, swiglu_alpha=1.702, swiglu_beta=1.0)
    q, sf = torch.ops.trtllm.mxfp8_quantize(h, True)
    return h, q, sf.view(torch.uint8).flatten()


@pytest.mark.parametrize("d", [256, 1536, 3072])
@pytest.mark.parametrize("rows", [1, 127, 1031, 4096, 16384])
def test_bitwise_vs_silu_and_mul_then_mxfp8_quantize(rows, d):
    op = _op()
    x = _inputs(rows, d, seed=rows * 7 + d)
    h_ref, q_ref, sf_ref = _reference(x)
    padded = (rows + 127) // 128 * 128
    h = torch.full((rows, d), float("nan"), dtype=torch.bfloat16, device="cuda")
    q = torch.empty((rows, d), dtype=torch.float8_e4m3fn, device="cuda")
    sf = torch.full(
        (padded * d // 32,), 0xAB, dtype=torch.uint8, device="cuda"
    )  # padding must be rewritten to 0
    op(x, q, sf, h)
    torch.cuda.synchronize()
    assert sf_ref.numel() == sf.numel()
    assert torch.equal(h.view(torch.int16), h_ref.view(torch.int16))
    assert torch.equal(q.view(torch.uint8), q_ref.view(torch.uint8))
    assert torch.equal(sf, sf_ref)


def test_without_the_bf16_output_and_under_graph_replay():
    op = _op()
    rows, d = 4096, 1536
    x = _inputs(rows, d, seed=11)
    _, q_ref, sf_ref = _reference(x)
    q = torch.empty((rows, d), dtype=torch.float8_e4m3fn, device="cuda")
    sf = torch.empty((rows * d // 32,), dtype=torch.uint8, device="cuda")
    op(x, q, sf, None)
    torch.cuda.synchronize()
    assert torch.equal(q.view(torch.uint8), q_ref.view(torch.uint8)) and torch.equal(sf, sf_ref)
    q.zero_()
    sf.zero_()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        op(x, q, sf, None)
    torch.cuda.current_stream().wait_stream(s)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        op(x, q, sf, None)
    q.zero_()
    sf.zero_()
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(q.view(torch.uint8), q_ref.view(torch.uint8)) and torch.equal(sf, sf_ref)


def test_rejects_unsupported_widths():
    op = _op()
    x = torch.zeros(8, 2 * 160, dtype=torch.bfloat16, device="cuda")
    with pytest.raises(RuntimeError, match="multiple of 256"):
        op(x, torch.empty(8, 160, dtype=torch.float8_e4m3fn, device="cuda"),
           torch.empty(128 * 5, dtype=torch.uint8, device="cuda"), None)  # fmt: skip


def test_the_mlp_publishes_the_pair_for_its_down_proj():
    """The swapped MLP's down_proj takes the published (q, sf) for its own bf16 input, so its GEMM reads the operands
    the stock quantize would have produced; a different input is not taken."""
    op = _op()
    from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m31_nvfp4__sm_107__tp2ctx import (
        modeling,
    )

    rows, d = modeling._NORMQ_MIN_ROWS, 1536
    x = _inputs(rows, d, seed=5)
    h = torch.empty((rows, d), dtype=torch.bfloat16, device="cuda")
    q = torch.empty((rows, d), dtype=torch.float8_e4m3fn, device="cuda")
    sf = torch.empty((rows * d // 32,), dtype=torch.uint8, device="cuda")
    op(x, q, sf, h)
    modeling._NORMQ_SLOT.publish(h, q, sf)
    assert modeling._NORMQ_SLOT.take(h.clone()) is None
    modeling._NORMQ_SLOT.publish(h, q, sf)
    taken = modeling._NORMQ_SLOT.take(h)
    assert taken is not None and taken[0] is q and taken[1] is sf
