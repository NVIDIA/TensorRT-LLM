# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the rubin_mxfp8_gemm entry: the dense MXFP8 GEMM on sm_107.

Every cell runs the entry against its reference, and a sample of the runner's tactics (swapped
operands, split-K in FP32 planes, mixed clusters) against the same reference. The scale layout
is checked against the engine's ``block_scale_interleave``.
"""

import pytest
import torch
from _rubin_gemm_util import activation, generator, mxfp8_weight, requires_sm107

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Cell
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.gemm.rubin_mxfp8_gemm import (  # noqa: E501
    mxfp8_scales_to_r128c4,
    quantize_mxfp8_reference,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.gemm.rubin_mxfp8_gemm import (  # noqa: E501
    rubin_mxfp8_gemm as op,
)

__extra_import_path__ = ["../../../.."]
from _cells import certified_cells  # noqa: E402 — needs the path declared above
from _validating import (  # noqa: E402,F401 — path above; fixture by name
    assert_within_ulp,
    certifying,
    validating,
)

pytestmark = requires_sm107

TARGET_CELLS: dict[str, tuple[Cell, ...]] = {}
CELLS = certified_cells(op, TARGET_CELLS)


def _operands(spec, gen):
    a, a_scale = quantize_mxfp8_reference(activation(spec["m"], spec["k"], gen))
    b, b_scale = mxfp8_weight(spec["n"], spec["k"], gen)
    return a, b, mxfp8_scales_to_r128c4(a_scale), b_scale


def _check(out, ref) -> None:
    # E4M3 x E4M3 products and the power-of-two scales are exact in FP32, so both sides sum the
    # same exact terms in FP32 in another order and round once to BF16: the band of one BF16
    # rounding, as for a BF16 GEMM (2 units of 2^-8 of the row's largest magnitude per element,
    # half a unit in RMS).
    assert_within_ulp(out, ref, element_ulp=2.0, rms_ulp=0.5)


@pytest.mark.parametrize("cell", CELLS, ids=[c.why[:44] for c in CELLS])
def test_certified_cells(cell, certifying) -> None:  # noqa: F811 — pytest resolves the fixture by this name
    certifying(op)
    args = _operands(cell.spec, generator(cell.why))
    out = op(*args)
    _check(out, op.reference(*args))


def test_tactics_agree_with_the_reference() -> None:
    """The autotuner picks among these in production; outside a tuning pass the cells run the
    fallback tile only. One tactic of each kind offered at the 96-row verify shape (split-K in FP32
    planes is the one that sums in another order); mixed clusters only where the shape takes them."""
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.gemm.rubin_mxfp8_gemm import (  # noqa: E501
        _runner,
    )

    _, runner = _runner()
    args = _operands(dict(m=96, n=6144, k=7168), generator("tactics"))
    inputs = [*args, torch.ones((), dtype=torch.float32, device="cuda")]
    kinds = {
        "swapped": lambda t: t[0] == "base" and t[4] and t[-1] == 1,
        "split-K 2": lambda t: t[0] == "base" and t[-1] == 2,
        "split-K 4": lambda t: t[0] == "base" and t[-1] == 4,
        "mixed clusters": lambda t: t[0] == "mixed_clusters",
    }
    picked = {}
    for tactic in runner.get_valid_tactics(inputs, None):
        for kind, match in kinds.items():
            if kind not in picked and match(tactic):
                picked[kind] = tactic
    assert {"swapped", "split-K 2"} <= set(picked), f"offered: {sorted(picked)}"
    ref = op.reference(*args)
    for kind, tactic in picked.items():
        out = runner(inputs, tactic=tactic)
        torch.cuda.synchronize()
        assert torch.isfinite(out.float()).all(), kind
        _check(out, ref)


def test_a_replayed_graph_matches_eager() -> None:
    args = _operands(dict(m=8, n=7168, k=1536), generator("graph"))
    eager = op(*args)  # compiles and makes the FP32 one outside the capture
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = op(*args)
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(out, eager)


def test_scale_layout_is_the_engines_block_scale_interleave() -> None:
    """The bytes ``mxfp8_quantize`` and the engine's MXFP8 weights hand this kernel."""
    import tensorrt_llm._torch.custom_ops  # noqa: F401 — registers torch.ops.trtllm.*

    gen = generator("layout")
    for rows, cols in ((1, 4), (96, 224), (130, 48), (6144, 224), (7168, 384)):
        scales = torch.randint(0, 255, (rows, cols), generator=gen, device="cuda").to(torch.uint8)
        engine = torch.ops.trtllm.block_scale_interleave(scales)
        assert torch.equal(mxfp8_scales_to_r128c4(scales), engine.reshape(-1)), (rows, cols)


def test_reference_quantization_is_mxfp8_quantize() -> None:
    """``quantize_mxfp8_reference`` gives the kernel's bytes, row-major and swizzled."""
    import tensorrt_llm._torch.custom_ops  # noqa: F401 — registers torch.ops.trtllm.*

    gen = generator("quantize")
    for rows, k in ((1, 7168), (96, 1536), (300, 6144)):
        x = torch.randn(rows, k, generator=gen, device="cuda")
        x[:, ::7] *= 30.0
        x = x.to(torch.bfloat16)
        values, scales = quantize_mxfp8_reference(x)
        kernel_values, kernel_scales = torch.ops.trtllm.mxfp8_quantize(x, False, 32)
        assert torch.equal(values.view(torch.uint8), kernel_values.view(torch.uint8))
        assert torch.equal(scales.reshape(-1), kernel_scales)
        _, swizzled = torch.ops.trtllm.mxfp8_quantize(x, True, 32)
        assert torch.equal(mxfp8_scales_to_r128c4(scales), swizzled)


def test_guard_refuses_row_major_scales() -> None:
    """A [128, 4] row-major scale matrix has exactly the bytes R128c4 expects for N 128, K 128:
    the runner would read it, in the wrong order."""
    gen = generator("guard")
    a, a_scale = mxfp8_weight(4, 128, gen)
    b, _ = mxfp8_weight(128, 128, gen)
    _, row_major = quantize_mxfp8_reference(torch.randn(128, 128, device="cuda"))
    assert row_major.numel() == 128 * 4
    with pytest.raises(AssertionError, match="row-major"):
        with validating(op):
            op(a, b, a_scale, row_major)


def test_guard_refuses_bf16_operands() -> None:
    """The runner reads the operands as E4M3 whatever their dtype."""
    gen = generator("guard dtype")
    _, a_scale = mxfp8_weight(4, 128, gen)
    b, b_scale = mxfp8_weight(128, 128, gen)
    a = torch.randn(4, 128, device="cuda").to(torch.bfloat16)
    with pytest.raises(AssertionError, match="E4M3"):
        with validating(op):
            op(a, b, a_scale, b_scale)
