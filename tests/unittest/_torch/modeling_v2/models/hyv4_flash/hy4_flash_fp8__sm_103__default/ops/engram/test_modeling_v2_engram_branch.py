# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the engram_branch entry, at the checkpoint's widest (31) and dimmest (1) layers."""

import zlib

import pytest
import torch

from tensorrt_llm._torch.modeling_v2.models.hyv4_flash.hy4_flash_fp8__sm_103__default.ops.engram.engram_branch import (  # noqa: E501
    engram_branch as op,
)

__extra_import_path__ = ["~/tests/unittest/_torch/modeling_v2"]
from _validating import assert_within_ulp, certifying  # noqa: E402,F401

assert torch.cuda.is_available(), "engram_branch requires a CUDA device"

HIDDEN = 3072
STREAMS = 4
ENGRAM_SIZE = 1536
#: The gate is invariant to the streams' scale; 8.0 makes a dropped query norm visible.
STREAM_SIGMA = 8.0

# HF's chain rounds to bf16 at every step; the float64 reference rounds once.
ELEMENT_ULP = 64.0
RMS_ULP = 4.0

# Checkpoint weight statistics per layer: (mean, std) per stream, or rms for zero-mean weights.
# `norm_query` tracks `norm_key` to three decimals, so one `norm_gain` serves both.
_LAYER_STATS = {
    31: dict(
        pre_gamma=(1.8811, 0.0641463),
        hc_scales=(
            (-1.10345, 0.034509),
            (-0.185462, 0.00299721),
            (-0.27072, 0.00349725),
            (-0.126602, 0.00269659),
        ),
        key_weight_rms=(0.0232301, 0.0233326, 0.0230245, 0.0230648),
        key_bias=(
            (-0.0109894, 1.10055),
            (0.011381, 1.38945),
            (-0.00249593, 1.18116),
            (-0.00679149, 1.43005),
        ),
        norm_gain=(
            (1.22996, 0.275402),
            (0.709922, 0.117214),
            (0.282928, 0.0495669),
            (0.662535, 0.110089),
        ),
        value_weight_rms=0.0234388,
        value_bias=(-0.00514729, 0.228995),
    ),
    1: dict(
        pre_gamma=(0.529057, 0.0708961),
        hc_scales=(
            (-0.00696826, 0.0074922),
            (-0.0136814, 0.00634353),
            (0.0051046, 0.00895684),
            (-0.0255126, 0.00969647),
        ),
        key_weight_rms=(0.0222615, 0.0184399, 0.0202152, 0.0200779),
        key_bias=(
            (0.0195444, 1.24935),
            (0.00612009, 1.03129),
            (0.0414796, 1.3986),
            (0.0149754, 0.84112),
        ),
        norm_gain=(
            (0.73442, 0.312102),
            (0.488775, 0.236551),
            (0.295818, 0.172645),
            (0.603789, 0.243781),
        ),
        value_weight_rms=0.0205679,
        value_bias=(0.0132625, 0.452777),
    ),
}

CELLS = op.CELLS


def _build(spec, seed):
    """Unit-rms embeddings (per-head-normed table rows); rows 3, 10, 17, ... masked."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    rows, s = spec["rows"], _LAYER_STATS[spec["layer"]]
    hidden = torch.randn((rows, STREAMS, HIDDEN), generator=g, device="cuda") * STREAM_SIGMA
    embeddings = torch.randn((rows, ENGRAM_SIZE), generator=g, device="cuda")
    key_weights = torch.stack(
        [
            torch.randn((HIDDEN, ENGRAM_SIZE), generator=g, device="cuda") * rms
            for rms in s["key_weight_rms"]
        ]
    )
    key_biases = torch.stack(
        [
            torch.randn(HIDDEN, generator=g, device="cuda") * std + mean
            for mean, std in s["key_bias"]
        ]
    )
    gains = torch.stack(
        [
            torch.randn(HIDDEN, generator=g, device="cuda") * std + mean
            for mean, std in s["norm_gain"]
        ]
    )
    hc_scales = torch.stack(
        [
            torch.randn(HIDDEN, generator=g, device="cuda") * std + mean
            for mean, std in s["hc_scales"]
        ]
    )
    pre_gamma = (
        torch.randn(HIDDEN, generator=g, device="cuda") * s["pre_gamma"][1] + s["pre_gamma"][0]
    )
    value_weight = (
        torch.randn((HIDDEN, ENGRAM_SIZE), generator=g, device="cuda") * s["value_weight_rms"]
    )
    value_bias = (
        torch.randn(HIDDEN, generator=g, device="cuda") * s["value_bias"][1] + s["value_bias"][0]
    )
    mask = torch.ones(rows, dtype=torch.bool, device="cuda")
    if rows >= 8:
        mask[3::7] = False
    bf16 = torch.bfloat16
    return (
        hidden.to(bf16),
        embeddings.to(bf16),
        key_weights.to(bf16),
        key_biases.to(bf16),
        gains.to(bf16),  # norm_key.{i}.weight
        gains.to(bf16),  # norm_query.{i}.weight
        value_weight.to(bf16),
        value_bias.to(bf16),
        pre_gamma.to(bf16),
        hc_scales.to(bf16),
        mask,
    )


@pytest.mark.parametrize("cell", CELLS, ids=[c.why[:44] for c in CELLS])
def test_certified_cells(cell, certifying) -> None:  # noqa: F811 — pytest resolves the fixture by this name
    certifying(op)
    args = _build(cell.spec, seed=zlib.crc32(cell.why.encode()) % 2**31)
    out = op(*args)
    assert out.shape == args[0].shape and out.dtype == args[0].dtype
    assert_within_ulp(out, op.reference(*args), element_ulp=ELEMENT_ULP, rms_ulp=RMS_ULP)


def _hf_forward(args):
    """``HYV4FlashEngramLayer.forward`` transcribed with the ``nn.Linear``/``nn.RMSNorm`` modules HF builds."""
    (
        hidden_states,
        embeddings,
        key_weights,
        key_biases,
        key_norm_weights,
        query_norm_weights,
        value_weight,
        value_bias,
        pre_gamma,
        hc_scales,
        engram_mask,
    ) = args
    streams = hidden_states.shape[-2]
    device, dtype = hidden_states.device, hidden_states.dtype
    key_projs, norm_key, norm_query = [], [], []
    for hc in range(streams):
        projection = torch.nn.Linear(
            embeddings.shape[-1], hidden_states.shape[-1], bias=True, device=device, dtype=dtype
        )
        projection.weight = torch.nn.Parameter(key_weights[hc].clone(), requires_grad=False)
        projection.bias = torch.nn.Parameter(key_biases[hc].clone(), requires_grad=False)
        key_projs.append(projection)
        for weights, norms in ((key_norm_weights, norm_key), (query_norm_weights, norm_query)):
            norm = torch.nn.RMSNorm(hidden_states.shape[-1], device=device, dtype=dtype)
            norm.weight = torch.nn.Parameter(weights[hc].clone(), requires_grad=False)
            norms.append(norm)
    value_proj = torch.nn.Linear(
        embeddings.shape[-1], hidden_states.shape[-1], bias=True, device=device, dtype=dtype
    )
    value_proj.weight = torch.nn.Parameter(value_weight.clone(), requires_grad=False)
    value_proj.bias = torch.nn.Parameter(value_bias.clone(), requires_grad=False)
    unified = torch.nn.RMSNorm(
        hidden_states.shape[-1], elementwise_affine=False, device=device, dtype=dtype
    )

    gates = []
    for hc_index in range(streams):
        key = norm_key[hc_index](key_projs[hc_index](embeddings))
        query = norm_query[hc_index](hidden_states[..., hc_index, :])
        gate = (key * query).sum(dim=-1) * hidden_states.shape[-1] ** -0.5
        gate = gate.abs().clamp_min(1e-6).sqrt() * gate.sign()
        gates.append(gate.sigmoid())
    gates = torch.stack(gates, dim=-1).unsqueeze(-1)

    main_signal = pre_gamma * unified(value_proj(embeddings))
    branch = torch.stack([scale * main_signal for scale in hc_scales], dim=-2)
    engram_output = gates * main_signal.unsqueeze(-2) + branch
    return engram_output.masked_fill(~engram_mask[..., None, None], 0.0)


def test_the_entry_is_the_hf_module() -> None:
    """Bit for bit against HF's modules: pins the rounding points and the norm eps, which the
    reference shares with the entry and so cannot check."""
    for layer, rows in ((31, 1), (31, 512), (1, 64)):
        args = _build(dict(rows=rows, layer=layer), seed=layer * 1000 + rows)
        got = op(*args)
        want = _hf_forward(args)
        assert torch.equal(got, want), (
            f"layer {layer}, {rows} rows: the entry is not HF's module bit for bit "
            f"(maxdiff {(got.float() - want.float()).abs().max().item():.4e})"
        )
