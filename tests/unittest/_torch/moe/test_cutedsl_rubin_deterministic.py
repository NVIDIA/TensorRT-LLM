# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Rule 1b gate a for the deterministic CuteDSL NVFP4 MoE on SM107 (FC2 per permuted row + fixed-order
moe_unpermute; SwiGLU-OAI in the FC1 act epilogue), at the MiniMax-M3 EP2 rank shape (64 local experts, hidden 6144,
intermediate 3072, top-4, SwiGLU-OAI alpha 1.702 / beta 1 / limit 7, NVFP4):
- run-to-run determinism: 3 calls bitwise, at two token counts;
- batch independence (fixed permutation / finalize order): a token block's output is bitwise the same whether it runs
  alone or inside a larger batch with other tokens;
- band: max abs / max rel error vs the quantize util's reference module (dequantized math) within 1.25x of the stock
  TRTLLM-gen backend's error on the same weights and inputs.
The kill switch (TLLM_CUTEDSL_MOE_DETERMINISTIC=0) restores the fused scatter-add FC2."""

import pytest
import torch
from _torch.moe.moe_test_utils import MoeBackendType
from _torch.moe.quantize_utils import get_test_quant_params
from _torch.moe.test_moe_backend import create_test_backend, run_backend_moe

import tensorrt_llm._torch.moe.fused_moe.fused_moe_cute_dsl as cute_moe
from tensorrt_llm._torch.autotuner import AutoTuner
from tensorrt_llm._torch.moe.fused_moe import RenormalizeMoeRoutingMethod
from tensorrt_llm._torch.utils import ActivationType
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models.modeling_utils import QuantAlgo

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 7),
    reason="the deterministic CuteDSL FC2 path is SM107 only",
)

H, INTER, E, TOP_K = 6144, 3072, 64, 4


def _build(backend_type, x):
    dtype = torch.bfloat16
    qcls, qcfg, qkw = get_test_quant_params(QuantAlgo.NVFP4, x, backend_type)
    qu = qcls(
        num_experts=E, dtype=dtype, intermediate_size=INTER, hidden_size=H, quant_config=qcfg, bias=False,
        swiglu_gptoss_style=True, swiglu_alpha=1.702, swiglu_beta=1.0, swiglu_limit=7.0,
        activation_type=ActivationType.Swiglu,
    )  # fmt: skip
    sw = qu._swiglu_tensors
    routing = RenormalizeMoeRoutingMethod(top_k=TOP_K)
    backend = create_test_backend(
        backend_type=backend_type, routing_method=routing, num_experts=E, hidden_size=H, intermediate_size=INTER,
        dtype=dtype, quant_config=qcfg, mapping=Mapping(), swiglu_alpha=sw["swiglu_alpha"],
        swiglu_beta=sw["swiglu_beta"], swiglu_limit=sw["swiglu_limit"],
    )  # fmt: skip
    return qu, qkw, routing, backend


def _run(backend, backend_type, routing, x, logits):
    with torch.inference_mode():
        sel, scl = routing.apply(logits)
        xq, xsf = backend.quantize_input(x, post_quant_comm=False)
        return run_backend_moe(
            backend, backend_type, xq, xsf, sel, scl, torch.bfloat16, logits
        ).clone()


def _err(y, ref):
    d = (y.float() - ref.float()).abs()
    return float(d.max()), float(
        (d / ref.float().abs().amax(dim=1, keepdim=True).clamp_min(1e-30)).max()
    )


def test_deterministic_cutedsl_gate_a() -> None:
    assert cute_moe._RUBIN_DETERMINISTIC_FC2, "the kill switch is off in this environment"
    with torch.device("cuda"):
        torch.manual_seed(0)
        AutoTuner.get().setup_distributed_state(Mapping())
        big = 8192
        x = torch.randn((big, H), dtype=torch.bfloat16)
        logits = torch.randn((big, E), dtype=torch.bfloat16)
        qu, qkw, routing, cute = _build(MoeBackendType.CUTEDSL, x)
        ref_cls = qkw.pop("ref_cls", None)
        weights = qu.create_weights(**qkw)
        cute.load_weights([weights])
        cute.post_load_weights()
        cute.cuda()
        # run-to-run determinism at two token counts
        for t in (2048, big):
            outs = [
                _run(cute, MoeBackendType.CUTEDSL, routing, x[:t], logits[:t]) for _ in range(3)
            ]
            for o in outs[1:]:
                assert torch.equal(outs[0].view(torch.int16), o.view(torch.int16)), (
                    f"not deterministic at {t} tokens"
                )
        # batch independence: rows [0, 1024) alone vs inside the full batch
        alone = _run(cute, MoeBackendType.CUTEDSL, routing, x[:1024], logits[:1024])
        full = _run(cute, MoeBackendType.CUTEDSL, routing, x, logits)
        assert torch.equal(alone.view(torch.int16), full[:1024].view(torch.int16)), (
            "output depends on the batch"
        )
        # band vs the reference module, next to stock TRTLLM-gen on the same weights
        ref_mod = (
            qu.create_ref_module(routing, ref_cls=ref_cls)
            if ref_cls
            else qu.create_ref_module(routing)
        )
        ref_mod.load_weights([weights])
        ref_mod.cuda()
        with torch.inference_mode():
            ref = ref_mod.forward(x, logits)
        _, _, _, stock = _build(MoeBackendType.TRTLLM, x)
        stock.load_weights([weights])
        stock.post_load_weights()
        stock.cuda()
        e_cute = _err(full, ref)
        e_stock = _err(_run(stock, MoeBackendType.TRTLLM, routing, x, logits), ref)
        print(
            f"\nGATEA cutedsl det: err vs ref abs {e_cute[0]:.3e} rel {e_cute[1]:.3e} | stock abs {e_stock[0]:.3e} "
            f"rel {e_stock[1]:.3e} | ratio abs {e_cute[0] / e_stock[0]:.2f} rel {e_cute[1] / e_stock[1]:.2f}",
            flush=True,
        )
        assert e_cute[0] <= 1.25 * e_stock[0] and e_cute[1] <= 1.25 * e_stock[1]


@pytest.mark.parametrize("tokens", [8192, 16384])
def test_shared_fold_bitwise(tokens) -> None:
    """The shared-expert output offered to the deterministic finalize (TLLM_M3_MOE_FOLD_SHARED) is bitwise
    ``shared.add_(routed)``."""
    with torch.device("cuda"):
        torch.manual_seed(1)
        AutoTuner.get().setup_distributed_state(Mapping())
        x = torch.randn((tokens, H), dtype=torch.bfloat16)
        logits = torch.randn((tokens, E), dtype=torch.bfloat16)
        shared = (torch.randn((tokens, H)) * 0.5).to(torch.bfloat16)
        qu, qkw, routing, cute = _build(MoeBackendType.CUTEDSL, x)
        qkw.pop("ref_cls", None)
        cute.load_weights([qu.create_weights(**qkw)])
        cute.post_load_weights()
        cute.cuda()

        def plain():
            return shared.clone().add_(_run(cute, MoeBackendType.CUTEDSL, routing, x, logits))

        def folded():
            holder = cute_moe.offer_finalize_residual(shared.clone())
            try:
                out = _run(cute, MoeBackendType.CUTEDSL, routing, x, logits)
            finally:
                cute_moe.clear_finalize_residual()
            assert holder.consumed, "the deterministic finalize did not take the residual"
            return out

        ref, got = plain(), folded()
        assert torch.equal(ref.view(torch.int16), got.view(torch.int16)), (
            "fold is not bitwise the separate add"
        )


@pytest.mark.parametrize("tokens", [8192, 16384])
def test_shared_fold_bitwise_cuda_graph(tokens) -> None:
    """The folded deterministic finalize captured in a CUDA graph and replayed is bitwise the eager
    ``shared.add_(routed)`` (the fold's arithmetic does not depend on the launch mode)."""
    with torch.device("cuda"):
        torch.manual_seed(1)
        AutoTuner.get().setup_distributed_state(Mapping())
        x = torch.randn((tokens, H), dtype=torch.bfloat16)
        logits = torch.randn((tokens, E), dtype=torch.bfloat16)
        shared = (torch.randn((tokens, H)) * 0.5).to(torch.bfloat16)
        qu, qkw, routing, cute = _build(MoeBackendType.CUTEDSL, x)
        qkw.pop("ref_cls", None)
        cute.load_weights([qu.create_weights(**qkw)])
        cute.post_load_weights()
        cute.cuda()
        ref = shared.clone().add_(_run(cute, MoeBackendType.CUTEDSL, routing, x, logits))
        residual = shared.clone()

        def folded():
            holder = cute_moe.offer_finalize_residual(residual)
            try:
                return _run(cute, MoeBackendType.CUTEDSL, routing, x, logits), holder
            finally:
                cute_moe.clear_finalize_residual()

        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(2):  # warm-up outside capture (autotuning, allocations)
                residual.copy_(shared)
                folded()
        torch.cuda.current_stream().wait_stream(side)
        graph = torch.cuda.CUDAGraph()
        residual.copy_(shared)
        with torch.cuda.graph(graph):
            out, holder = folded()
        assert holder.consumed, "the captured deterministic finalize did not take the residual"
        residual.copy_(shared)
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(ref.view(torch.int16), out.view(torch.int16)), (
            "captured fold is not bitwise the separate add"
        )
