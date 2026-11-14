# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""DEP_SHARED_MXFP8 on an SM107 GPU.

``swiglu_mxfp8`` is ``torch.ops.trtllm.silu_and_mul`` followed by both quantize tactics of the decode-graph
``mxfp8_quantize_gemm_autotuned``, bit for bit; ``add_rmsnorm_replicated_mxfp8`` is FlashInfer's fused add + RMSNorm,
``torch.cat`` and both quantize tactics, bit for bit; and the M3.1 shared expert with both hand-offs installed returns
the stock shared expert's output bit for bit, eagerly and under graph replay.
"""

import types

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_1_flash_nvfp4__sm_107__dep import (  # noqa: E501
    gemm_provider,
    router_operand,
    shared_mxfp8,
)


def _sm107() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability() == (10, 7)


pytestmark = pytest.mark.skipif(not _sm107(), reason="needs an SM107 GPU")

ROWS = (448, 256, 129, 88, 44, 8, 1)
HIDDEN = 6144
WIDTH = 3072


def _scale_bytes(rows, width):
    return (rows + 127) // 128 * 128 * ((width // 32 + 3) // 4 * 4)


def _quantizers():
    """Both quantize tactics of MXFP8QuantizeRunner: trtllm's, and FlashInfer's CuTe DSL one (FlashInfer is a
    TensorRT-LLM dependency: a missing import fails the test rather than dropping the tactic)."""
    from flashinfer import mxfp8_quantize

    return {
        "trtllm": lambda x: torch.ops.trtllm.mxfp8_quantize(x, True),
        "cute_dsl": lambda x: mxfp8_quantize(
            x, is_sf_swizzled_layout=True, alignment=32, enable_pdl=None, backend="cute-dsl"
        ),
    }


def _assert_mxfp8(q, sf, ref_q, ref_sf, rows, width, what):
    need = _scale_bytes(rows, width)
    assert torch.equal(
        q.view(torch.uint8).flatten(), ref_q.view(torch.uint8).flatten()[: rows * width]
    ), what
    assert torch.equal(sf.flatten()[:need], ref_sf.view(torch.uint8).flatten()[:need]), what


def _gate_up_rows(rows, std, gen):
    x = torch.randn((rows, 2 * WIDTH), device="cuda", generator=gen) * std
    flat = x.view(-1)
    picks = torch.randint(
        0, flat.numel(), (max(flat.numel() // 50, 16),), device="cuda", generator=gen
    )
    edges = torch.tensor(
        [
            7.0,
            -7.0,
            7.0001,
            -7.0001,
            20.0,
            -20.0,
            100.0,
            -100.0,
            0.0,
            -0.0,
            3e-3,
            -3e-3,
            60.0,
            -60.0,
        ],
        device="cuda",
    )
    flat[picks] = edges[torch.arange(picks.numel(), device="cuda") % edges.numel()]
    return x.to(torch.bfloat16)


@pytest.mark.parametrize(
    "alpha, beta, limit", [(1.702, 1.0, 7.0), (1.0, 0.0, None)], ids=["swigluoai", "silu"]
)
def test_swiglu_mxfp8_is_silu_and_mul_and_both_quantize_tactics(alpha, beta, limit):
    import tensorrt_llm._torch.custom_ops  # noqa: F401
    from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.kernels.megamoe.swiglu_mxfp8 import (
        swiglu_mxfp8,
    )

    gen = torch.Generator(device="cuda").manual_seed(17)
    tactics = _quantizers()
    for rows in ROWS:
        for std in (1.0, 4.0):
            x = _gate_up_rows(rows, std, gen)
            ref_h = torch.ops.trtllm.silu_and_mul(
                x, swiglu_limit=limit, swiglu_alpha=alpha, swiglu_beta=beta
            )
            h = torch.empty((rows, WIDTH), dtype=torch.bfloat16, device="cuda")
            q = torch.empty((rows, WIDTH), dtype=torch.float8_e4m3fn, device="cuda")
            sf = torch.full((_scale_bytes(rows, WIDTH),), 0xA5, dtype=torch.uint8, device="cuda")
            swiglu_mxfp8(x, h, q, sf, alpha=alpha, beta=beta, limit=limit, enable_pdl=True)
            assert torch.equal(h.view(torch.int16), ref_h.view(torch.int16)), (rows, std)
            for name, quantize in tactics.items():
                ref_q, ref_sf = quantize(ref_h)
                _assert_mxfp8(q, sf, ref_q, ref_sf, rows, WIDTH, (rows, std, name))


def test_add_rmsnorm_replicated_mxfp8_is_the_stock_norm_cat_and_quantize():
    import tensorrt_llm._torch.custom_ops  # noqa: F401
    from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.kernels.megamoe.add_rmsnorm_mxfp8 import (
        add_rmsnorm_replicated_mxfp8,
    )
    from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.kernels.megamoe.megamoe_topk_reduce_tail import (  # noqa: E501
        megamoe_topk_reduce_add_rmsnorm_matches_flashinfer,
    )

    device = torch.device("cuda", torch.cuda.current_device())
    if not megamoe_topk_reduce_add_rmsnorm_matches_flashinfer(device, HIDDEN):
        pytest.skip("FlashInfer does not run FusedAddRMSNormKernel with one CTA per row here")
    gen = torch.Generator(device="cuda").manual_seed(19)
    weight = (1 + 0.1 * torch.randn(HIDDEN, device="cuda", generator=gen)).to(torch.bfloat16)
    tactics = _quantizers()
    for rows in ROWS:
        x = torch.randn((rows, HIDDEN), device="cuda", generator=gen).to(torch.bfloat16)
        residual = (torch.randn((rows, HIDDEN), device="cuda", generator=gen) * 4).to(
            torch.bfloat16
        )
        ref_x, ref_residual = x.clone(), residual.clone()
        torch.ops.trtllm.flashinfer_fused_add_rmsnorm(ref_x, ref_residual, weight, 1e-6)
        replicas = torch.empty((rows, 3 * HIDDEN), dtype=torch.bfloat16, device="cuda")
        q = torch.empty((rows, HIDDEN), dtype=torch.float8_e4m3fn, device="cuda")
        sf = torch.full((_scale_bytes(rows, HIDDEN),), 0xA5, dtype=torch.uint8, device="cuda")
        add_rmsnorm_replicated_mxfp8(x, residual, weight, 1e-6, replicas, q, sf, enable_pdl=True)
        assert torch.equal(x.view(torch.int16), ref_x.view(torch.int16)), rows
        assert torch.equal(residual.view(torch.int16), ref_residual.view(torch.int16)), rows
        assert torch.equal(
            replicas.view(torch.int16), torch.cat([ref_x] * 3, dim=-1).view(torch.int16)
        ), rows
        for name, quantize in tactics.items():
            ref_q, ref_sf = quantize(ref_x)
            _assert_mxfp8(q, sf, ref_q, ref_sf, rows, HIDDEN, (rows, name))


def _site(monkeypatch):
    """The M3.1 MXFP8 shared expert (random E4M3 weights) behind a post-attention norm at both sites, installed."""
    from transformers.configuration_utils import PretrainedConfig

    import tensorrt_llm._torch.custom_ops  # noqa: F401
    from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.kernels.megamoe.megamoe_topk_reduce_tail import (  # noqa: E501
        megamoe_topk_reduce_add_rmsnorm_matches_flashinfer,
    )
    from tensorrt_llm._torch.model_config import ModelConfig
    from tensorrt_llm._torch.models.modeling_minimaxm3 import _build_swiglu_oai_dense_mlp
    from tensorrt_llm._torch.modules.rms_norm import RMSNorm
    from tensorrt_llm.mapping import Mapping
    from tensorrt_llm.models.modeling_utils import QuantAlgo, QuantConfig

    device = torch.device("cuda", torch.cuda.current_device())
    if not megamoe_topk_reduce_add_rmsnorm_matches_flashinfer(device, HIDDEN):
        pytest.skip("FlashInfer does not run FusedAddRMSNormKernel with one CTA per row here")
    for module, values in (
        (router_operand, {"DEP_ROUTER_OPERAND": True, "_BUFFERS": {}}),
        (
            shared_mxfp8,
            {
                "DEP_SHARED_MXFP8": True,
                "DEP_SHARED_GU_MXFP8": True,
                "DEP_SHARED_DOWN_MXFP8": True,
                "_BUFFERS": {},
                "_WIDTHS": set(),
            },
        ),
    ):
        for name, value in values.items():
            monkeypatch.setattr(module, name, value)
    for slot in (router_operand.SLOT, shared_mxfp8.GATE_UP_SLOT, shared_mxfp8.DOWN_SLOT):
        slot.clear()
    config = PretrainedConfig()
    config.hidden_size, config.intermediate_size = HIDDEN, WIDTH
    config.swiglu_alpha, config.swiglu_limit = 1.702, 7.0
    config.torch_dtype = torch.bfloat16
    model_config = ModelConfig(
        pretrained_config=config,
        mapping=Mapping(world_size=1, tp_size=1),
        quant_config=QuantConfig(quant_algo=QuantAlgo.MXFP8),
    )
    mlp = _build_swiglu_oai_dense_mlp(
        model_config=model_config, intermediate_size=WIDTH, layer_idx=0, is_shared_expert=True
    ).cuda()
    gen = torch.Generator(device="cuda").manual_seed(23)
    with torch.no_grad():
        for linear in (mlp.gate_up_proj, mlp.down_proj):
            out_f, in_f = linear.weight.shape
            w = torch.randn((out_f, in_f), generator=gen, device="cuda") * in_f**-0.5 * 64.0
            linear.weight.copy_(w.to(torch.float8_e4m3fn))
            linear.weight_scale.fill_(121)
    norm = RMSNorm(hidden_size=HIDDEN, eps=1e-6, dtype=torch.bfloat16, device=device)
    with torch.no_grad():
        norm.weight.copy_(1 + 0.1 * torch.randn(HIDDEN, device="cuda", generator=gen))
    norm.forward = types.MethodType(router_operand._norm_forward, norm)
    layer = types.SimpleNamespace(
        post_attention_layernorm=norm,
        block_sparse_moe=types.SimpleNamespace(shared_experts=mlp),
        pre_feed_forward_fusion=False,
    )
    assert shared_mxfp8.install([layer], lambda layer: True) == (1, 1)
    router_operand.reserve(device, HIDDEN)
    shared_mxfp8.reserve(device)
    return norm, mlp, gen


def _stock(norm, mlp, x, residual):
    from tensorrt_llm._torch.modules.gated_mlp import GatedMLP

    ref_x, ref_residual = x.clone(), residual.clone()
    torch.ops.trtllm.flashinfer_fused_add_rmsnorm(ref_x, ref_residual, norm.weight, 1e-6)
    return ref_x, ref_residual, GatedMLP.forward(mlp, ref_x)


def test_the_shared_expert_with_both_hand_offs_is_the_stock_shared_expert(monkeypatch):
    """From the widest step down: the norm takes the hand-off form, both projections take their hand-offs, and the
    output is the stock norm + shared expert's (stock quantizes, SwiGLU and GEMMs) bit for bit."""
    # DEP_GEMM_PROVIDER would run the gate_up GEMM at split-K 2 for 65..128 rows (not bitwise; its own test covers it).
    monkeypatch.setattr(gemm_provider, "DEP_GEMM_PROVIDER", False)
    norm, mlp, gen = _site(monkeypatch)
    for rows in ROWS:
        x = torch.randn((rows, HIDDEN), device="cuda", generator=gen).to(torch.bfloat16)
        residual = (torch.randn((rows, HIDDEN), device="cuda", generator=gen) * 4).to(
            torch.bfloat16
        )
        ref_x, ref_residual, stock = _stock(norm, mlp, x, residual)
        out, res = norm(x, residual)
        router_operand.SLOT.clear()
        assert shared_mxfp8.GATE_UP_SLOT.pending is not None, rows
        ours = mlp(out)
        assert shared_mxfp8.GATE_UP_SLOT.pending is None and shared_mxfp8.DOWN_SLOT.pending is None
        assert torch.equal(out.view(torch.int16), ref_x.view(torch.int16)), rows
        assert torch.equal(res.view(torch.int16), ref_residual.view(torch.int16)), rows
        assert torch.equal(ours.view(torch.int16), stock.view(torch.int16)), rows


def test_the_shared_expert_hand_offs_under_graph_replay_are_the_stock_shared_expert(monkeypatch):
    """The norm and the shared expert captured together replay the stock output on new rows."""
    norm, mlp, gen = _site(monkeypatch)
    rows = 44
    x = torch.randn((rows, HIDDEN), device="cuda", generator=gen).to(torch.bfloat16)
    residual = (torch.randn((rows, HIDDEN), device="cuda", generator=gen) * 4).to(torch.bfloat16)
    x_in, residual_in = x.clone(), residual.clone()
    result = {}

    def run():
        x.copy_(x_in)
        residual.copy_(residual_in)
        out, _ = norm(x, residual)
        router_operand.SLOT.clear()
        result["out"] = mlp(out)

    run()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    for _ in range(2):
        new_x = torch.randn((rows, HIDDEN), device="cuda", generator=gen).to(torch.bfloat16)
        new_residual = (torch.randn((rows, HIDDEN), device="cuda", generator=gen) * 4).to(
            torch.bfloat16
        )
        x_in.copy_(new_x)
        residual_in.copy_(new_residual)
        graph.replay()
        torch.cuda.synchronize()
        _, _, stock = _stock(norm, mlp, new_x, new_residual)
        assert torch.equal(result["out"].view(torch.int16), stock.view(torch.int16))
