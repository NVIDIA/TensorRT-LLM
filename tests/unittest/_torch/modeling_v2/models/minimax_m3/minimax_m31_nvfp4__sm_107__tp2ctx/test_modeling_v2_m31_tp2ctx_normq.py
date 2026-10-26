# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""TP2CTX_NORMQ (#20629 NORM_MXFP8, N1): the prefill boundary's MXFP8 activation reaches the next MXFP8 linear.

The boundary op's MXFP8 pair is mxfp8_quantize of its own norm bit for bit, and an MXFP8 linear whose input is the
published normed rows returns exactly what the stock MXFP8LinearMethod returns for that input; any other input runs
the stock path and drops the entry. The all-reduce side is checked in-model (TRTLLM_M31_ENABLE_TP2CTX_CHECK_NORMQ=1 on
two ranks). GPU (an sm_10x build carrying minimax_m31_ctx_residual_rmsnorm_mxfp8 and the MXFP8 GEMM).
"""

from pathlib import Path

import pytest
import torch

import tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3 as _family
from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m31_nvfp4__sm_107__tp2ctx import (
    modeling as m,
)
from tensorrt_llm._torch.modules.linear import Linear, MXFP8LinearMethod
from tensorrt_llm._torch.modules.mxfp8_utils import quant_bf16_to_mxfp8
from tensorrt_llm.models.modeling_utils import QuantConfig
from tensorrt_llm.quantization.mode import QuantAlgo

_TARGET = Path(_family.__file__).resolve().parent / "minimax_m31_nvfp4__sm_107__tp2ctx"
H = 6144
EPS = 1e-6

assert torch.cuda.is_available(), "TP2CTX_NORMQ requires a CUDA device"
_needs_op = pytest.mark.skipif(
    m._normq_op() is None, reason="minimax_m31_ctx_residual_rmsnorm_mxfp8 is not built"
)


def test_boundaries_take_normq_with_a_kill_switch():
    src = (_TARGET / "modeling.py").read_text()
    assert '_NORMQ = not m31_disabled("TP2CTX_NORMQ")' in src
    assert "_NORMQ_MIN_ROWS = 1024" in src
    # Both fused boundaries try it first; the final norm (lm_head) does not.
    assert src.count("fused = _normq_boundary(") == 2
    assert "layer._normq_feeds_mxfp8 = idx != num_layers - 1" in src
    assert "module.quant_method.__class__ = _NormqMXFP8LinearMethod" in src


def test_the_copy_engine_allreduce_is_opt_in():
    """TP2CTX_CEAR (N2): opt-in (LEAD 10-23: not faster than NCCL standalone; the pipelined N3a carries its value)."""
    src = (_TARGET / "modeling.py").read_text()
    assert '_CEAR_ON = m31_enabled("TP2CTX_CEAR")' in src
    assert "    if not _CEAR_ON:\n        return None\n" in src


def test_short_boundaries_keep_the_fused_allreduce():
    """Decided before any collective: the all-reduce argument is never touched."""
    x = torch.zeros(m._NORMQ_MIN_ROWS - 1, H, dtype=torch.bfloat16, device="cuda")
    assert m._normq_boundary(None, x, x, None) is None


def _operands(rows, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    inter = torch.randn(rows, H, generator=g, device="cuda").to(torch.bfloat16)
    residual = (2 * torch.randn(rows, H, generator=g, device="cuda")).to(torch.bfloat16)
    weight = (1.0 + 0.1 * torch.randn(H, generator=g, device="cuda")).to(torch.bfloat16)
    return inter, residual, weight


def _mxfp8_linear(out_features, seed):
    torch.manual_seed(seed)
    w_e4m3, scale = quant_bf16_to_mxfp8(torch.randn(out_features, H, dtype=torch.bfloat16), 32)
    lin = Linear(
        in_features=H,
        out_features=out_features,
        bias=False,
        dtype=torch.bfloat16,
        quant_config=QuantConfig(quant_algo=QuantAlgo.MXFP8, group_size=32),
    ).cuda()
    lin.load_weights([{"weight": w_e4m3, "weight_scale_inv": scale}])
    assert type(lin.quant_method) is MXFP8LinearMethod
    return lin


@_needs_op
@pytest.mark.parametrize("rows", [1024, 1160, 8192])
def test_published_activation_gives_the_stock_linear_output(rows):
    inter, residual, weight = _operands(rows, 20261024 + rows)
    normed, xq, xsf = m._normq_op()(inter, residual, weight, EPS, True)[:3]
    q_ref, sf_ref = torch.ops.trtllm.mxfp8_quantize(normed, True)
    assert torch.equal(xq.view(torch.uint8), q_ref.view(torch.uint8))
    assert torch.equal(xsf.view(torch.uint8)[: sf_ref.numel()], sf_ref.view(torch.uint8).flatten())

    lin = _mxfp8_linear(1536, rows)
    stock = lin(normed)
    lin.quant_method.__class__ = m._NormqMXFP8LinearMethod
    m._NORMQ_SLOT.publish(normed, xq, xsf)
    taken = lin(normed)
    assert m._NORMQ_SLOT.pending is None
    assert torch.equal(taken.view(torch.int16), stock.view(torch.int16))


@_needs_op
def test_other_inputs_run_the_stock_path_and_drop_the_entry():
    inter, residual, weight = _operands(2048, 7)
    normed, xq, xsf = m._normq_op()(inter, residual, weight, EPS, True)[:3]
    lin = _mxfp8_linear(512, 7)
    other = normed.clone()
    stock = lin(other)
    lin.quant_method.__class__ = m._NormqMXFP8LinearMethod
    m._NORMQ_SLOT.publish(normed, xq, xsf)
    got = lin(other)
    assert m._NORMQ_SLOT.pending is None  # a mismatch still clears the one-entry slot
    assert torch.equal(got.view(torch.int16), stock.view(torch.int16))
    # The same storage but another shape is not the published tensor either.
    m._NORMQ_SLOT.publish(normed, xq, xsf)
    assert m._NORMQ_SLOT.take(normed[:1024]) is None


def test_slot_holds_the_source_alive():
    """The entry keeps the normed tensor referenced, so its storage can't be handed to another tensor that a later
    linear could read under the same data pointer."""
    x = torch.ones(1024, 64, dtype=torch.bfloat16, device="cuda")
    q = torch.empty(1024, 64, dtype=torch.uint8, device="cuda")
    m._NORMQ_SLOT.publish(x, q, q)
    assert m._NORMQ_SLOT.pending[0] is x
    m._NORMQ_SLOT.pending = None
