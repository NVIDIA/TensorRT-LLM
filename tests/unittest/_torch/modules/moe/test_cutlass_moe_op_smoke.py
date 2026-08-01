# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Non-LoRA smoke test for the CutlassMoEOp op path.

CutlassMoEOp.run_moe calls the TorchBind C++ method fused_moe_runner.run_moe.
That method takes trailing routed-expert LoRA arguments, and TorchBind schemas
do not honor C++ default-argument values, so every positional argument must be
supplied by the caller. This op is what the WideEP and ConfigurableMoE paths
select via MoEOpSelector, so a missing or extra argument here is a silent
regression for a feature unrelated to LoRA.

This test exercises that callsite on a single GPU (tp/ep/cluster = 1, the
degenerate WideEP configuration) with unquantized bf16 weights and no LoRA,
confirming the call succeeds and produces finite output.
"""

import pytest
import torch

_TRTLLM_AVAILABLE = hasattr(torch.ops, "trtllm") and hasattr(torch.ops.trtllm, "fused_moe")

requires_cuda_and_op = pytest.mark.skipif(
    not torch.cuda.is_available() or not _TRTLLM_AVAILABLE,
    reason="Requires CUDA and built TensorRT-LLM C++ extension (torch.ops.trtllm.fused_moe).",
)


class _RoutingStub:
    def __init__(self, top_k):
        self.experts_per_token = top_k


class _ModuleStub:
    """Minimal stand-in for an MoE module exposing only the attributes that
    CutlassMoEOp.finalize_tactic and compute_moe read."""

    def __init__(self, w3_w1_weight, w2_weight, top_k, hidden_size, activation_type):
        self.w3_w1_weight = w3_w1_weight
        self.w2_weight = w2_weight
        self.routing_method = _RoutingStub(top_k)

        # Parallelism: single rank (degenerate WideEP).
        self.tp_size = 1
        self.tp_rank = 0
        self.ep_size = 1
        self.ep_rank = 0
        self.cluster_size = 1
        self.cluster_rank = 0

        # SwiGLU params, unused here.
        self.swiglu_alpha = None
        self.swiglu_beta = None
        self.swiglu_limit = None

        # Quantization and layout flags, all off for unquantized bf16.
        self.has_w4afp8 = False
        self.has_w4a16_mxfp4 = False
        self.has_deepseek_fp8_block_scales = False
        self.has_int8_woq_per_channel = False
        self.has_mxfp8_act_scaling = False
        self.has_nvfp4 = False
        self.force_dynamic_quantization = False

        self.activation_type = activation_type
        self.tune_max_num_tokens = 8192
        self.unpadded_hidden_size = hidden_size


@requires_cuda_and_op
def test_cutlass_moe_op_run_moe_no_lora_smoke():
    from tensorrt_llm._torch.modules.fused_moe.ops.moe_op_cutlass import CutlassMoEOp
    from tensorrt_llm._torch.utils import ActivationType

    device = torch.device("cuda")
    dtype = torch.bfloat16
    num_tokens, hidden_size, inter_size = 8, 128, 256
    num_experts, top_k = 4, 2

    torch.manual_seed(0)
    x = torch.randn(num_tokens, hidden_size, dtype=dtype, device=device)
    # FC1 packs [up(w3); gate(w1)] -> [E, 2 * inter, hidden].
    w3_w1_weight = (
        torch.randn(num_experts, 2 * inter_size, hidden_size, dtype=dtype, device=device) * 0.02
    )
    w2_weight = torch.randn(num_experts, hidden_size, inter_size, dtype=dtype, device=device) * 0.02

    logits = torch.randn(num_tokens, num_experts, dtype=torch.float32, device=device)
    topk_scores, topk_ids = torch.topk(logits, k=top_k, dim=-1)
    topk_scores = torch.softmax(topk_scores, dim=-1)

    module = _ModuleStub(
        w3_w1_weight=w3_w1_weight,
        w2_weight=w2_weight,
        top_k=top_k,
        hidden_size=hidden_size,
        activation_type=int(ActivationType.Swiglu),
    )

    op = CutlassMoEOp()
    # No LoRA arguments: guards the run_moe callsite in moe_op_cutlass.py.
    out = op.run_moe(
        module=module,
        input=x,
        token_selected_slots=topk_ids.to(torch.int32),
        token_final_scales=topk_scores.to(torch.float32),
        w3_w1_weight=w3_w1_weight,
        w3_w1_bias=None,
        w2_weight=w2_weight,
        w2_bias=None,
        output_dtype=dtype,
        quant_scales=[],
        use_all_to_all=False,
    )

    # Non-min-latency run_moe returns a single-element list.
    assert isinstance(out, list) and len(out) == 1
    result = out[0]
    assert result.shape == (num_tokens, hidden_size)
    assert torch.isfinite(result).all()


@requires_cuda_and_op
def test_cutlass_moe_op_run_moe_no_lora_matches_fused_moe_op():
    """The CutlassMoEOp path and the direct torch.ops.trtllm.fused_moe path
    should produce the same result for an unquantized bf16 MoE with no LoRA,
    since both ultimately call the same C++ run_moe."""
    from tensorrt_llm._torch.modules.fused_moe.ops.moe_op_cutlass import CutlassMoEOp
    from tensorrt_llm._torch.utils import ActivationType

    device = torch.device("cuda")
    dtype = torch.bfloat16
    num_tokens, hidden_size, inter_size = 8, 128, 256
    num_experts, top_k = 4, 2

    torch.manual_seed(1)
    x = torch.randn(num_tokens, hidden_size, dtype=dtype, device=device)
    w3_w1_weight = (
        torch.randn(num_experts, 2 * inter_size, hidden_size, dtype=dtype, device=device) * 0.02
    )
    w2_weight = torch.randn(num_experts, hidden_size, inter_size, dtype=dtype, device=device) * 0.02
    logits = torch.randn(num_tokens, num_experts, dtype=torch.float32, device=device)
    topk_scores, topk_ids = torch.topk(logits, k=top_k, dim=-1)
    topk_scores = torch.softmax(topk_scores, dim=-1).to(torch.float32)
    topk_ids = topk_ids.to(torch.int32)

    # Direct op reference.
    ref = torch.ops.trtllm.fused_moe(
        input=x,
        token_selected_experts=topk_ids,
        token_final_scales=topk_scores,
        fc1_expert_weights=w3_w1_weight,
        fc1_expert_biases=None,
        fc2_expert_weights=w2_weight,
        fc2_expert_biases=None,
        output_dtype=dtype,
        quant_scales=[],
    )[0]

    module = _ModuleStub(
        w3_w1_weight=w3_w1_weight,
        w2_weight=w2_weight,
        top_k=top_k,
        hidden_size=hidden_size,
        activation_type=int(ActivationType.Swiglu),
    )
    out = CutlassMoEOp().run_moe(
        module=module,
        input=x,
        token_selected_slots=topk_ids,
        token_final_scales=topk_scores,
        w3_w1_weight=w3_w1_weight,
        w3_w1_bias=None,
        w2_weight=w2_weight,
        w2_bias=None,
        output_dtype=dtype,
        quant_scales=[],
        use_all_to_all=False,
    )[0]

    torch.testing.assert_close(out, ref, rtol=5e-2, atol=1e-2)
