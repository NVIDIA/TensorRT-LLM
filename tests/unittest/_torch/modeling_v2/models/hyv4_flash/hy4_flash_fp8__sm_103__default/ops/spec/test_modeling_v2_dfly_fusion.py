# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for this target's DFly2 fusion and hidden-correction entries.

Drives every cell both entries certify -- 1, 6, 64 and 384 rows at BF16 -- through
the entry's guard and against its own fp32 reference, plus the band's negative
controls (a wrong formula keeps every shape, so the band is the only thing that
can reject it) and each metadata refusal.

The weights are built with the element statistics of this checkpoint's
`model.dfly_layers.*` tensors, measured on shard 148 of
`hf-003107f_orig`; the activations use the measured scale of the tensors the
drafter feeds them.
"""

import zlib

import pytest
import torch

from tensorrt_llm._torch.modeling_v2.models.hyv4_flash.hy4_flash_fp8__sm_103__default.ops.spec.dfly_fusion import (  # noqa: E501
    DEFAULT_EPS,
)
from tensorrt_llm._torch.modeling_v2.models.hyv4_flash.hy4_flash_fp8__sm_103__default.ops.spec.dfly_fusion import (  # noqa: E501
    dfly_fusion as fuse,
)
from tensorrt_llm._torch.modeling_v2.models.hyv4_flash.hy4_flash_fp8__sm_103__default.ops.spec.dfly_fusion import (  # noqa: E501
    dfly_hidden_correction as correct,
)

__extra_import_path__ = ["~/tests/unittest/_torch/modeling_v2"]
from _validating import (  # noqa: E402,F401 — needs the path declared above
    ULP,
    assert_within_ulp,
    certifying,
    validating,
)

assert torch.cuda.is_available(), "dfly_fusion requires a CUDA device"

FUSION_CELLS = fuse.CELLS
CORRECTION_CELLS = correct.CELLS

DEV = "cuda"
HIDDEN = 3072

# Element statistics of this checkpoint's DFly tensors (`model.dfly_layers.*`,
# shard 148, BF16), measured with rms and absmax over each tensor.
CKPT_ENORM_RMS = 0.8557
CKPT_HNORM_RMS = 1.1506
CKPT_EH_PROJ_RMS = 0.02212
CKPT_GATE_RMS = 0.16264
CKPT_GATE_ABSMAX = 0.23633
CKPT_HIDDEN_NORM_RMS = 0.65689
CKPT_EMBED_NORM_RMS = 0.72357
CKPT_GATE_PROJ_RMS = 0.02465
CKPT_UP_PROJ_RMS = 0.02598
CKPT_DOWN_PROJ_RMS = 0.02050
CKPT_CONF_PROJ_RMS = 0.007263
CKPT_CONF_BIAS = 0.008423

# The activations: `anchors` is a token embedding (`model.embed_tokens.weight`
# rms 0.1436), `target_hidden` the target's post-norm hidden (`model.norm.weight`
# rms 1.1854), `hidden` the drafter's post-`final_norm` hidden (that norm's rms
# 1.1187), and `previous_embed` another token embedding.
CKPT_ANCHORS_RMS = 0.14360
CKPT_TARGET_HIDDEN_RMS = 1.18538
CKPT_DRAFT_HIDDEN_RMS = 1.11868


def _normal(g, shape, scale, dtype):
    return (torch.randn(*shape, generator=g, device=DEV) * scale).to(dtype)


def _fusion_args(spec, seed):
    """One fusion cell: activations, the four weights, eps."""
    g = torch.Generator(device=DEV).manual_seed(seed)
    rows, dtype = spec["rows"], spec["dtype"]
    anchors = _normal(g, (rows, HIDDEN), CKPT_ANCHORS_RMS, dtype)
    target_hidden = _normal(g, (rows, HIDDEN), CKPT_TARGET_HIDDEN_RMS, dtype)
    enorm = _normal(g, (HIDDEN,), CKPT_ENORM_RMS, dtype)
    hnorm = _normal(g, (HIDDEN,), CKPT_HNORM_RMS, dtype)
    eh_proj = _normal(g, (HIDDEN, 2 * HIDDEN), CKPT_EH_PROJ_RMS, dtype)
    gate = _normal(g, (HIDDEN,), CKPT_GATE_RMS, dtype)
    gate[0] = CKPT_GATE_ABSMAX
    gate[HIDDEN // 2] = -CKPT_GATE_ABSMAX
    return anchors, target_hidden, enorm, hnorm, eh_proj, gate, DEFAULT_EPS


def _correction_args(spec, seed):
    """One correction cell: activations, the seven weights, eps."""
    g = torch.Generator(device=DEV).manual_seed(seed)
    rows, dtype = spec["rows"], spec["dtype"]
    hidden = _normal(g, (rows, HIDDEN), CKPT_DRAFT_HIDDEN_RMS, dtype)
    previous_embed = _normal(g, (rows, HIDDEN), CKPT_ANCHORS_RMS, dtype)
    hidden_norm = _normal(g, (HIDDEN,), CKPT_HIDDEN_NORM_RMS, dtype)
    embed_norm = _normal(g, (HIDDEN,), CKPT_EMBED_NORM_RMS, dtype)
    gate_proj = _normal(g, (HIDDEN, 2 * HIDDEN), CKPT_GATE_PROJ_RMS, dtype)
    up_proj = _normal(g, (HIDDEN, 2 * HIDDEN), CKPT_UP_PROJ_RMS, dtype)
    down_proj = _normal(g, (HIDDEN, HIDDEN), CKPT_DOWN_PROJ_RMS, dtype)
    confidence_proj = _normal(g, (1, HIDDEN), CKPT_CONF_PROJ_RMS, dtype)
    confidence_bias = torch.full((1,), CKPT_CONF_BIAS, device=DEV, dtype=dtype)
    return (
        hidden,
        previous_embed,
        hidden_norm,
        embed_norm,
        gate_proj,
        up_proj,
        down_proj,
        confidence_proj,
        confidence_bias,
        DEFAULT_EPS,
    )


def _compare(out, ref, element_ulp: float, rms_ulp: float, label: str = "") -> None:
    """A chain of bf16-rounded stages against the same formula in fp32.

    The kernel path rounds after each norm, after each projection and after
    silu/mul; the reference rounds once, so the band is wider than one GEMM's
    and is stated in bf16 ulp of the row's largest magnitude (the same
    construction `_validating.assert_within_ulp` documents). The achieved
    margin is printed so the certification log carries the measurement the
    band was set from.
    """
    ulp = ULP[torch.bfloat16]
    flat_out, flat_ref = (
        out.float().reshape(-1, out.shape[-1]),
        ref.float().reshape(-1, ref.shape[-1]),
    )
    row_scale = flat_ref.abs().amax(dim=1, keepdim=True).clamp_min(1e-9)
    worst = ((flat_out - flat_ref).abs() / row_scale).max().item() / ulp
    rel_rms = (
        (flat_out - flat_ref).pow(2).mean().sqrt() / flat_ref.pow(2).mean().sqrt().clamp_min(1e-9)
    ).item() / ulp
    print(
        f"MEASURE {label or out.shape}: worst element {worst:.2f} ulp "
        f"(band {element_ulp}), rel rms {rel_rms:.2f} ulp (band {rms_ulp})"
    )
    assert_within_ulp(out, ref, element_ulp=element_ulp, rms_ulp=rms_ulp, ulp=ulp)


@pytest.mark.parametrize("cell", FUSION_CELLS, ids=[c.why[:44] for c in FUSION_CELLS])
def test_fusion_certified_cells(cell, certifying) -> None:  # noqa: F811 — fixture by name
    certifying(fuse)
    args = _fusion_args(cell.spec, seed=zlib.crc32(cell.why.encode()) % 2**31)
    out = fuse(*args)
    assert out.shape == args[0].shape and out.dtype is args[0].dtype
    _compare(out, fuse.reference(*args), element_ulp=4.0, rms_ulp=2.0, label=cell.why)


@pytest.mark.parametrize("cell", CORRECTION_CELLS, ids=[c.why[:44] for c in CORRECTION_CELLS])
def test_correction_certified_cells(cell, certifying) -> None:  # noqa: F811 — fixture by name
    certifying(correct)
    args = _correction_args(cell.spec, seed=zlib.crc32(cell.why.encode()) % 2**31)
    corrected, confidence = correct(*args)
    assert corrected.shape == args[0].shape and corrected.dtype is args[0].dtype
    assert confidence.shape == (args[0].shape[0],) and confidence.dtype is args[0].dtype
    ref_corrected, ref_confidence = correct.reference(*args)
    _compare(corrected, ref_corrected, element_ulp=8.0, rms_ulp=4.0, label=f"{cell.why} corrected")
    # The confidence head is a 3072->1 dot product, so its absolute error is
    # set by the whole row's scale, not by each logit's own magnitude: compared
    # as one row, a logit that lands near zero is not held to a relative gate
    # no bf16 dot product could meet.
    _compare(
        confidence.reshape(1, -1),
        ref_confidence.reshape(1, -1),
        element_ulp=4.0,
        rms_ulp=2.0,
        label=f"{cell.why} confidence",
    )


def test_fusion_band_rejects_a_different_gate() -> None:
    """The band separates a wrong formula, not just a different accumulation."""
    args = _fusion_args(dict(rows=6, dtype=torch.bfloat16), seed=11)
    out = fuse(*args)
    *head, gate, eps = args
    other_gate = gate.flip(0)
    with pytest.raises(AssertionError):
        _compare(
            out,
            fuse.reference(*head, other_gate, eps),
            element_ulp=4.0,
            rms_ulp=2.0,
            label="control: a different gate",
        )


def test_fusion_band_rejects_a_dropped_residual() -> None:
    args = _fusion_args(dict(rows=6, dtype=torch.bfloat16), seed=12)
    out = fuse(*args)
    anchors, target_hidden, enorm, hnorm, eh_proj, gate, eps = args
    zero = torch.zeros_like(anchors)
    with pytest.raises(AssertionError):
        _compare(
            out,
            fuse.reference(zero, target_hidden, enorm, hnorm, eh_proj, gate, eps),
            element_ulp=4.0,
            rms_ulp=2.0,
            label="control: dropped residual",
        )


def test_fusion_band_rejects_the_swapped_norms() -> None:
    """`enorm` on the target hidden and `hnorm` on the anchor is a mis-read
    that keeps every shape."""
    args = _fusion_args(dict(rows=6, dtype=torch.bfloat16), seed=13)
    out = fuse(*args)
    anchors, target_hidden, enorm, hnorm, eh_proj, gate, eps = args
    with pytest.raises(AssertionError):
        _compare(
            out,
            fuse.reference(anchors, target_hidden, hnorm, enorm, eh_proj, gate, eps),
            element_ulp=4.0,
            rms_ulp=2.0,
            label="control: swapped norms",
        )


def test_correction_band_rejects_swapped_gate_and_up() -> None:
    """`gate_proj` and `up_proj` have the same shape; only the band sees a swap."""
    args = _correction_args(dict(rows=6, dtype=torch.bfloat16), seed=21)
    corrected, _ = correct(*args)
    hidden, previous_embed, hn, en, gate_proj, up_proj, down_proj, cp, cb, eps = args
    ref, _ = correct.reference(
        hidden, previous_embed, hn, en, up_proj, gate_proj, down_proj, cp, cb, eps
    )
    with pytest.raises(AssertionError):
        _compare(corrected, ref, element_ulp=8.0, rms_ulp=4.0, label="control: swapped gate/up")


def test_correction_band_rejects_a_dropped_residual() -> None:
    args = _correction_args(dict(rows=6, dtype=torch.bfloat16), seed=22)
    corrected, _ = correct(*args)
    hidden, previous_embed, hn, en, gate_proj, up_proj, down_proj, cp, cb, eps = args
    zero = torch.zeros_like(hidden)
    ref, _ = correct.reference(
        zero, previous_embed, hn, en, gate_proj, up_proj, down_proj, cp, cb, eps
    )
    with pytest.raises(AssertionError):
        _compare(corrected, ref, element_ulp=8.0, rms_ulp=4.0, label="control: dropped residual")


def test_correction_band_rejects_confidence_on_the_uncorrected_hidden() -> None:
    """AngelSpec feeds the confidence head the corrected hidden; the
    pre-correction one keeps the shape and moves the logit."""
    args = _correction_args(dict(rows=6, dtype=torch.bfloat16), seed=23)
    _, confidence = correct(*args)
    hidden, _, _, _, _, _, _, cp, cb, _ = args
    wrong = (hidden.float() @ cp.float().t() + cb.float()).squeeze(-1).to(hidden.dtype)
    with pytest.raises(AssertionError):
        _compare(
            confidence.reshape(1, -1),
            wrong.reshape(1, -1),
            element_ulp=4.0,
            rms_ulp=2.0,
            label="control: confidence on uncorrected hidden",
        )


def test_fusion_guard_refuses_a_strided_last_dim() -> None:
    """`rmsnorm` reads `stride(-1) == 1` and answers with the wrong elements."""
    args = list(_fusion_args(dict(rows=6, dtype=torch.bfloat16), seed=31))
    args[0] = torch.randn(6, 2 * HIDDEN, device=DEV, dtype=torch.bfloat16)[:, ::2]
    assert args[0].stride(-1) == 2
    with pytest.raises(AssertionError, match="contiguous"):
        with validating(fuse):
            fuse(*args)


def test_fusion_guard_refuses_a_broadcast_gate() -> None:
    """A [rows, 1] gate broadcasts into a per-row scale and answers plausibly."""
    args = list(_fusion_args(dict(rows=6, dtype=torch.bfloat16), seed=32))
    rows = args[0].shape[0]
    args[5] = torch.randn(rows, 1, device=DEV, dtype=torch.bfloat16)
    unguarded = fuse(*args)
    assert unguarded.shape == args[0].shape, "the hazard this guard covers is gone"
    with pytest.raises(AssertionError, match="per-channel"):
        with validating(fuse):
            fuse(*args)


def test_fusion_guard_refuses_a_padded_eh_proj() -> None:
    """A weight sliced from a wider buffer keeps the right shape and a
    column-major transpose the op accepts, but `cublas_mm` reads `mat_b` with
    the constant leading dimension `k`, not the slice's own stride: it reads
    the wrong elements and answers plausibly.

    A *transposed* weight is not the hazard here -- `mat_b.strides()[0] == 1`
    is checked by the op and raises.
    """
    args = list(_fusion_args(dict(rows=6, dtype=torch.bfloat16), seed=33))
    wide = torch.randn(HIDDEN, 4 * HIDDEN, device=DEV, dtype=torch.bfloat16)
    args[4] = wide[:, : 2 * HIDDEN]
    assert args[4].shape == (HIDDEN, 2 * HIDDEN) and not args[4].is_contiguous()
    assert args[4].t().stride(0) == 1, "the op's own layout check would reject this"
    unguarded = fuse(*args)
    assert unguarded.shape == args[0].shape, "the hazard this guard covers is gone"
    ref = fuse.reference(*args[:4], args[4].contiguous(), *args[5:])
    with pytest.raises(AssertionError):
        _compare(unguarded, ref, element_ulp=4.0, rms_ulp=2.0, label="hazard: padded eh_proj")
    with pytest.raises(AssertionError, match="dense row-major"):
        with validating(fuse):
            fuse(*args)


def test_correction_guard_refuses_a_padded_down_proj() -> None:
    """`down_proj` is square, so a padded slice keeps the shape, passes the
    op's `mat_b.strides()[0] == 1`, and is read at the wrong leading
    dimension: a plausible wrong product, not an error."""
    args = list(_correction_args(dict(rows=6, dtype=torch.bfloat16), seed=41))
    wide = torch.randn(HIDDEN, 2 * HIDDEN, device=DEV, dtype=torch.bfloat16)
    args[6] = wide[:, :HIDDEN]
    assert args[6].shape == (HIDDEN, HIDDEN) and not args[6].is_contiguous()
    assert args[6].t().stride(0) == 1, "the op's own layout check would reject this"
    unguarded, _ = correct(*args)
    assert unguarded.shape == args[0].shape, "the hazard this guard covers is gone"
    ref, _ = correct.reference(*args[:6], args[6].contiguous(), *args[7:])
    with pytest.raises(AssertionError):
        _compare(unguarded, ref, element_ulp=8.0, rms_ulp=4.0, label="hazard: padded down_proj")
    with pytest.raises(AssertionError, match="dense row-major"):
        with validating(correct):
            correct(*args)


def test_correction_guard_refuses_a_wrong_confidence_bias() -> None:
    args = list(_correction_args(dict(rows=6, dtype=torch.bfloat16), seed=42))
    args[8] = torch.zeros(1, 1, device=DEV, dtype=torch.bfloat16)
    with pytest.raises(AssertionError, match="confidence_bias must be"):
        with validating(correct):
            correct(*args)


def test_fusion_is_cuda_graph_capturable() -> None:
    """The drafter runs under CUDA graphs; capture and replay must be exact."""
    args = _fusion_args(dict(rows=6, dtype=torch.bfloat16), seed=51)
    warmup = torch.cuda.Stream()
    warmup.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(warmup):
        for _ in range(3):
            fuse(*args)
    torch.cuda.current_stream().wait_stream(warmup)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_out = fuse(*args)

    fresh = _fusion_args(dict(rows=6, dtype=torch.bfloat16), seed=52)
    args[0].copy_(fresh[0])
    graph.replay()
    torch.cuda.synchronize()
    eager = fuse(*args)
    assert torch.equal(graph_out, eager)


def test_correction_is_cuda_graph_capturable() -> None:
    args = _correction_args(dict(rows=6, dtype=torch.bfloat16), seed=61)
    warmup = torch.cuda.Stream()
    warmup.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(warmup):
        for _ in range(3):
            correct(*args)
    torch.cuda.current_stream().wait_stream(warmup)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_corrected, graph_confidence = correct(*args)

    fresh = _correction_args(dict(rows=6, dtype=torch.bfloat16), seed=62)
    args[0].copy_(fresh[0])
    graph.replay()
    torch.cuda.synchronize()
    eager_corrected, eager_confidence = correct(*args)
    assert torch.equal(graph_corrected, eager_corrected)
    assert torch.equal(graph_confidence, eager_confidence)
