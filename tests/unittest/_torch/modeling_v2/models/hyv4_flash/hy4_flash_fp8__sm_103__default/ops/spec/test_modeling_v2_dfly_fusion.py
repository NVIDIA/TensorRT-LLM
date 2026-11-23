# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for this target's DFly2 fusion and hidden-correction entries.

Inputs use the element scales of the checkpoint's `model.dfly_layers.*` tensors.
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
)

assert torch.cuda.is_available(), "dfly_fusion requires a CUDA device"

FUSION_CELLS = fuse.CELLS
CORRECTION_CELLS = correct.CELLS

DEV = "cuda"
HIDDEN = 3072

# Checkpoint tensor rms (and the fusion gate's absmax).
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
# Activation scales: token embedding, target post-norm hidden, draft post-norm hidden.
CKPT_ANCHORS_RMS = 0.14360
CKPT_TARGET_HIDDEN_RMS = 1.18538
CKPT_DRAFT_HIDDEN_RMS = 1.11868


def _normal(g, shape, scale, dtype):
    return (torch.randn(*shape, generator=g, device=DEV) * scale).to(dtype)


def _fusion_args(spec, seed):
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


def _compare(out, ref, element_ulp: float, rms_ulp: float) -> None:
    # The kernel path rounds to bf16 after every stage, the reference once; the band is in bf16 ulp.
    assert_within_ulp(out, ref, element_ulp=element_ulp, rms_ulp=rms_ulp, ulp=ULP[torch.bfloat16])


@pytest.mark.parametrize("cell", FUSION_CELLS, ids=[c.why[:44] for c in FUSION_CELLS])
def test_fusion_certified_cells(cell, certifying) -> None:  # noqa: F811 — fixture by name
    certifying(fuse)
    args = _fusion_args(cell.spec, seed=zlib.crc32(cell.why.encode()) % 2**31)
    out = fuse(*args)
    assert out.shape == args[0].shape and out.dtype is args[0].dtype
    _compare(out, fuse.reference(*args), element_ulp=4.0, rms_ulp=2.0)


@pytest.mark.parametrize("cell", CORRECTION_CELLS, ids=[c.why[:44] for c in CORRECTION_CELLS])
def test_correction_certified_cells(cell, certifying) -> None:  # noqa: F811 — fixture by name
    certifying(correct)
    args = _correction_args(cell.spec, seed=zlib.crc32(cell.why.encode()) % 2**31)
    corrected, confidence = correct(*args)
    assert corrected.shape == args[0].shape and corrected.dtype is args[0].dtype
    assert confidence.shape == (args[0].shape[0],) and confidence.dtype is args[0].dtype
    ref_corrected, ref_confidence = correct.reference(*args)
    _compare(corrected, ref_corrected, element_ulp=8.0, rms_ulp=4.0)
    # A 3072->1 dot product: scored as one row, so near-zero logits are held to the batch's scale.
    _compare(confidence.reshape(1, -1), ref_confidence.reshape(1, -1), element_ulp=4.0, rms_ulp=2.0)
