# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Gate a for the M3 CTX shared-add fold (TLLM_M3_MOE_FOLD_SHARED): through the real dispatch (create_moe with the
TRTLLM backend, MiniMaxM3MoeRoutingMethod with a correction bias and routed scaling 2.0, SwigluBiasActivation),
fold_shared_into_finalize(experts(..., do_finalize=False), shared) must equal experts(...) then shared.add_ bit for
bit, with the experts input as bf16 (CTX's NVFP4 slot killed) and as the slot's unswizzled Fp4QuantizedTensor (slot
engaged), at CTX chunk sizes; and it must be run-to-run deterministic. With routing outside the kernel the triple's
expert_weights is never written, so the fold must be refused there."""

from types import SimpleNamespace

import pytest
import torch
from _torch.moe.moe_test_utils import MoeBackendType
from _torch.moe.quantize_utils import get_test_quant_params
from transformers.configuration_utils import PretrainedConfig

import tensorrt_llm._torch.models.modeling_minimaxm3 as modeling_minimaxm3
import tensorrt_llm._torch.moe.fused_moe.moe_scheduler as moe_scheduler
import tensorrt_llm._torch.moe.fused_moe.trtllm_gen.base as trtllm_gen_base
from tensorrt_llm._torch.autotuner import AutoTuner
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.models.modeling_minimaxm3 import MiniMaxM3MoE, fold_shared_into_finalize
from tensorrt_llm._torch.moe.fused_moe.activation import SwigluBiasActivation
from tensorrt_llm._torch.moe.fused_moe.create_moe import create_moe
from tensorrt_llm._torch.moe.fused_moe.routing import MiniMaxM3MoeRoutingMethod
from tensorrt_llm._torch.utils import ActivationType, Fp4QuantizedTensor
from tensorrt_llm._utils import get_sm_version
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models.modeling_utils import QuantAlgo

# TP2 / EP2 CTX rank shape is 64 local experts; one GPU here: 64 experts, top-4, the M3 widths.
H, INTER, E, TOP_K = 6144, 3072, 64, 4
ACT = SwigluBiasActivation(gate_sigmoid_scale=1.702, linear_offset=1.0, clamp=7.0)

pytestmark = pytest.mark.skipif(
    get_sm_version() not in (100, 103, 107), reason="TRTLLM-gen NVFP4 MoE"
)


def _build(tokens):
    AutoTuner.get().setup_distributed_state(Mapping())
    g = torch.Generator(device="cuda").manual_seed(0)
    x = torch.randn((tokens, H), device="cuda", generator=g).to(torch.bfloat16)
    bias = (0.05 * torch.randn(E, device="cuda", generator=g)).float()
    logits = (
        x.float() @ (torch.randn((E, H), device="cuda", generator=g) / H**0.5).t()
    ).contiguous()
    shared = (0.5 * torch.randn((tokens, H), device="cuda", generator=g)).to(torch.bfloat16)
    routing = MiniMaxM3MoeRoutingMethod(top_k=TOP_K, num_experts=E, callable_e_score_correction_bias=lambda: bias,
                                        routed_scaling_factor=2.0)  # fmt: skip
    torch.manual_seed(0)
    qcls, qcfg, qkw = get_test_quant_params(QuantAlgo.NVFP4, x, MoeBackendType.TRTLLM)
    qu = qcls(num_experts=E, dtype=torch.bfloat16, intermediate_size=INTER, hidden_size=H, quant_config=qcfg,
              bias=False, swiglu_gptoss_style=True, swiglu_alpha=1.702, swiglu_beta=1.0, swiglu_limit=7.0,
              activation_type=ActivationType.Swiglu)  # fmt: skip
    qkw.pop("ref_cls", None)
    weights = qu.create_weights(**qkw)
    pc = PretrainedConfig()
    pc.torch_dtype = torch.bfloat16
    mc = ModelConfig(pretrained_config=pc, mapping=Mapping(world_size=1, rank=0, tp_size=1), moe_backend="TRTLLM",
                     quant_config=qcfg, moe_max_num_tokens=32768)  # fmt: skip
    moe = create_moe(routing_method=routing, num_experts=E, hidden_size=H, intermediate_size=INTER,
                     dtype=torch.bfloat16, model_config=mc, activation=ACT, allow_backend_degradation=False,
                     layer_idx=3)  # fmt: skip
    moe.load_weights([weights])
    moe.post_load_weights()
    moe.cuda()
    return moe, x, logits, shared


@pytest.mark.parametrize("slot", ["killed", "engaged"])
@pytest.mark.parametrize("tokens", [4096, 16384])
def test_fold_shared_bitwise(tokens, slot) -> None:
    with torch.device("cuda"), torch.inference_mode():
        moe, x, logits, shared = _build(tokens)
        backend = moe.backend
        assert type(backend).__name__ == "TrtllmTrtllmGenNvfp4Impl", type(backend)
        assert backend.supports_unfinalized_output() and moe.comm is None
        if slot == "engaged":  # CTX's fused pre-MoE norm hand-off (TRTLLM_CTX_NORM_FP4=1)
            fp4, sf = torch.ops.trtllm.fp4_quantize(x, backend.fc31_input_scale, 16, False, False)
            inp = Fp4QuantizedTensor(fp4, sf, is_sf_swizzled=False, unquantized_hidden_states=x)
            kw = {"output_dtype": torch.bfloat16}
        else:
            inp, kw = x, {}
        stock = shared.clone().add_(moe(inp, logits, **kw))
        outs = []
        for _ in range(3):
            unfinalized = moe(inp, logits, do_finalize=False, **kw)
            assert isinstance(unfinalized, (list, tuple)) and len(unfinalized) == 3
            outs.append(fold_shared_into_finalize(unfinalized, shared, TOP_K).clone())
        for o in outs:
            assert torch.equal(o.view(torch.int16), stock.view(torch.int16)), (
                "fold != finalize + add_"
            )
        print(f"\nFOLDGATE T={tokens} slot={slot}: bitwise vs finalize + add_ x3 PASS", flush=True)


def _fold_ok(moe, x) -> bool:
    block = SimpleNamespace(
        shared_experts=object(), experts=moe, routed_output_is_global=False, hidden_dim=H
    )
    return MiniMaxM3MoE.fold_shared_ok(block, x, None)


def test_fold_shared_refused_with_separated_routing(monkeypatch) -> None:
    tokens = 512
    monkeypatch.setattr(modeling_minimaxm3, "_FOLD_SHARED_INTO_FINALIZE", True)
    # The fold engages on SM107 only; pin it so the routing checks below are exercised on every SM10x.
    monkeypatch.setattr(modeling_minimaxm3, "get_sm_version", lambda: 107)
    with torch.device("cuda"), torch.inference_mode():
        moe, x, logits, shared = _build(tokens)
        assert _fold_ok(moe, x), "fold must stay enabled when the kernel routes from the logits"
        for sm in (100, 103):
            monkeypatch.setattr(modeling_minimaxm3, "get_sm_version", lambda sm=sm: sm)
            assert not _fold_ok(moe, x), f"fold must stay off on SM{sm}"
        monkeypatch.setattr(modeling_minimaxm3, "get_sm_version", lambda: 107)

        monkeypatch.setattr(trtllm_gen_base, "FORCE_SEPARATED_ROUTING", True)
        monkeypatch.setattr(moe_scheduler, "FORCE_SEPARATED_ROUTING", True)
        assert moe.backend._routes_outside_the_kernel()
        stock = shared.clone().add_(moe(x, logits))
        # The runner finalizes with the precomputed top-k weights and returns an expert_weights tensor it never
        # writes: poison the allocator's free blocks so the unwritten tensor cannot alias a copy of those weights.
        poison = [
            torch.full((tokens, TOP_K), float("nan"), dtype=torch.bfloat16) for _ in range(64)
        ]
        del poison
        unfinalized = moe(x, logits, do_finalize=False)
        folded = fold_shared_into_finalize(unfinalized, shared, TOP_K)
        assert not torch.equal(folded.view(torch.int16), stock.view(torch.int16)), (
            "expected the unwritten expert_weights to corrupt the fold"
        )
        assert not _fold_ok(moe, x), (
            "fold must be refused when top-k is precomputed outside the kernel"
        )
