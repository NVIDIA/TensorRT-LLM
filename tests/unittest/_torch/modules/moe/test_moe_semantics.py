# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Executable semantic reference for ``create_moe`` — the factory that builds every
MoE layer in TensorRT-LLM (``tensorrt_llm/_torch/modules/fused_moe/create_moe.py``).

Each test class documents one parameter (or parameter group) of ``create_moe``
by verifying the optimized implementation against a pure-PyTorch reference.
Because these tests run in CI, the reference implementations are guaranteed to
stay correct — unlike standalone docs that can silently go stale.

Function signature::

    def create_moe(
        routing_method: BaseMoeRoutingMethod,          # §1
        num_experts: Optional[int] = None,             # §2
        hidden_size: Optional[int] = None,             # §3
        intermediate_size: Optional[int] = None,       # §4
        dtype: Optional[torch.dtype] = None,           # §5
        reduce_results: bool = False,                  # §6
        model_config: ModelConfig = ModelConfig(),      # §7
        override_quant_config: ... = None,             # §8
        aux_stream_dict: ... = None,                   # §9
        weight_loading_mode: MoEWeightLoadingMode = VANILLA,  # §10
        bias: bool = False,                            # §11
        apply_router_weight_on_input: bool = False,    # §12
        layer_idx: Optional[int] = None,               # §13
        swiglu_alpha: Optional[Tensor] = None,         # §14
        swiglu_beta: Optional[Tensor] = None,          # §14
        swiglu_limit: Optional[Tensor] = None,         # §14
        activation_type: ActivationType = Swiglu,      # §15
    ) -> MoE

Input / output shapes::

    x             : [num_tokens, hidden_size]    — hidden states
    router_logits : [num_tokens, num_experts]    — raw router outputs
    output        : [num_tokens, hidden_size]    — MoE layer output

Parameter quick-reference (§ = section in this file)::

    §  | Parameter                   | Default    | Math?  | Key effect
    ---+-----------------------------+------------+--------+-----------
    1  | routing_method              | (required) | Yes    | Token→expert assignment & scales
    2  | num_experts                 | None       | Yes    | dim-0 of weight tensors
    3  | hidden_size                 | None       | Yes    | Input/output dim
    4  | intermediate_size           | None       | Yes    | FC1→FC2 intermediate dim
    5  | dtype                       | None       | Prec   | Weight dtype
    6  | reduce_results              | False      | Comm   | All-reduce across TP ranks
    7  | model_config                | default    | Indir  | Backend, TP/EP sharding
    8  | override_quant_config       | None       | Indir  | Override quant config
    9  | aux_stream_dict             | None       | No     | CUDA stream scheduling
    10 | weight_loading_mode         | VANILLA    | No     | W1/W3 memory layout
    11 | bias                        | False      | Yes    | Bias in FC1/FC2 (kernel-only)
    12 | apply_router_weight_on_input| False      | Yes    | Scale on input vs output
    13 | layer_idx                   | None       | No     | Profiling / load-balancer
    14 | swiglu_alpha/beta/limit     | None       | Yes    | GPT-OSS SwiGLU variant
    15 | activation_type             | Swiglu     | Yes    | Activation function

    Parameters §6–§9, §11, §13 are not tested here because they have no
    pure-PyTorch reference implementation (they control communication,
    backend selection, kernel-level bias, or profiling).

Backend note:

    All MoE backends (Cutlass, Triton, DeepGemm, CuteDsl, TRTLLMGen, WideEP)
    share the same computation semantics — they all call
    ``routing_method.apply()`` for routing, then execute the same
    gate-up → activation → down projection per expert.  The backends differ
    only in kernel implementation (fused GEMM, quantization support, comm
    overlap, etc.), not in mathematical behaviour.

    This file therefore uses VanillaMoE — the simplest, pure-PyTorch backend
    — as the reference implementation for all semantic tests.
"""

import pytest
import torch
import torch.nn.functional as F

# ──────────────────────────────────────────────────────────────────────────────
# §2-§5  Weight Shape Tests
# ──────────────────────────────────────────────────────────────────────────────


class TestWeightShapes:
    """§2–§5: Weight tensor shapes (num_experts, hidden_size, intermediate_size, dtype).

    These four parameters determine the shape of every expert weight tensor::

        Gated (Swiglu/Geglu):
            W1 (gate) : [num_experts, intermediate_size, hidden_size]
            W3 (up)   : [num_experts, intermediate_size, hidden_size]
            W2 (down) : [num_experts, hidden_size, intermediate_size]
            — or fused as gate_up_proj: [2*intermediate_size, hidden_size] per expert

        Non-gated (Relu2):
            W1 (up)   : [num_experts, intermediate_size, hidden_size]
            W2 (down) : [num_experts, hidden_size, intermediate_size]
            — no W3

    VanillaMoE stores experts as a ModuleList: each element is a GatedMLP
    (for gated activations) or MLP (for non-gated).
    """

    def test_gated_activation_shapes(self):
        """Swiglu uses W1 (gate), W3 (up), W2 (down)."""
        from tensorrt_llm._torch.model_config import ModelConfig
        from tensorrt_llm._torch.modules.fused_moe import DefaultMoeRoutingMethod, VanillaMoE
        from tensorrt_llm._torch.utils import ActivationType

        num_experts = 4
        hidden_size = 32
        intermediate_size = 64

        routing = DefaultMoeRoutingMethod(top_k=2, force_enable_pytorch_op=True)
        model_config = ModelConfig()
        moe = VanillaMoE(
            routing_method=routing,
            num_experts=num_experts,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            dtype=torch.float32,
            model_config=model_config,
            activation_type=ActivationType.Swiglu,
        ).cuda()

        # Each expert is a GatedMLP with gate_up_proj and down_proj
        expert = moe[0]
        # gate_up_proj: [2*intermediate_size, hidden_size]
        assert expert.gate_up_proj.weight.shape == (2 * intermediate_size, hidden_size)
        # down_proj: [hidden_size, intermediate_size]
        assert expert.down_proj.weight.shape == (hidden_size, intermediate_size)

    def test_nongated_activation_shapes(self):
        """Relu2 uses W1 (up) and W2 (down), no W3."""
        from tensorrt_llm._torch.model_config import ModelConfig
        from tensorrt_llm._torch.modules.fused_moe import DefaultMoeRoutingMethod, VanillaMoE
        from tensorrt_llm._torch.utils import ActivationType

        num_experts = 4
        hidden_size = 32
        intermediate_size = 64

        routing = DefaultMoeRoutingMethod(top_k=2, force_enable_pytorch_op=True)
        model_config = ModelConfig()
        moe = VanillaMoE(
            routing_method=routing,
            num_experts=num_experts,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            dtype=torch.float32,
            model_config=model_config,
            activation_type=ActivationType.Relu2,
        ).cuda()

        # Each expert is an MLP with up_proj and down_proj (no gate_up_proj)
        expert = moe[0]
        # up_proj: [intermediate_size, hidden_size]
        assert expert.up_proj.weight.shape == (intermediate_size, hidden_size)
        # down_proj: [hidden_size, intermediate_size]
        assert expert.down_proj.weight.shape == (hidden_size, intermediate_size)
        assert not hasattr(expert, "gate_up_proj")


# ──────────────────────────────────────────────────────────────────────────────
# VanillaMoE Forward Pass Tests (§ Complete Forward)
# ──────────────────────────────────────────────────────────────────────────────


class TestVanillaMoeForward:
    """Complete MoE forward pass — VanillaMoE vs pure-PyTorch reference.

    The reference implementation ``_reference_moe_forward`` below IS the
    authoritative documentation of what a single-rank MoE forward does::

        1. Routing  — routing_method.apply(router_logits) → (experts, scales)
        2. Dispatch — group tokens by expert, sort by expert index
        3. Compute  — run each expert's MLP on its assigned tokens
        4. Accumulate — weighted sum back into output: out[t] += expert(x[t]) * scale

    Based on fused_moe_vanilla.py: VanillaMoE.forward and
    VanillaMoE.run_experts.
    """

    @staticmethod
    def _reference_moe_forward(x, router_logits, routing_method, experts, expert_start, expert_end):
        """Pure PyTorch reference MoE forward from moe_semantics.md."""
        token_selected_experts, token_final_scales = routing_method.apply(router_logits)

        expert_masks = (token_selected_experts >= expert_start) & (
            token_selected_experts < expert_end
        )
        batch_indices, nth_experts = torch.where(expert_masks)

        local_selected_experts = token_selected_experts[expert_masks]
        sort_indices = torch.argsort(local_selected_experts)
        sorted_experts = local_selected_experts[sort_indices]
        batch_indices = batch_indices[sort_indices]
        nth_experts = nth_experts[sort_indices]

        expanded_inputs = x[batch_indices]
        expanded_scales = token_final_scales[batch_indices, nth_experts, None]

        final_hidden_states = torch.zeros_like(x)
        for expert_idx in range(expert_start, expert_end):
            expert_mask = sorted_experts == expert_idx
            if not torch.any(expert_mask):
                continue
            expanded_input = expanded_inputs[expert_mask]
            batch_idx = batch_indices[expert_mask]
            expanded_scale = expanded_scales[expert_mask]

            output = experts[expert_idx](expanded_input)
            final_hidden_states[batch_idx] += output * expanded_scale
        return final_hidden_states

    @pytest.mark.parametrize("activation_type_name", ["Swiglu", "Relu2"])
    @pytest.mark.parametrize("top_k", [1, 2])
    def test_forward_matches_reference(self, activation_type_name, top_k):
        from tensorrt_llm._torch.model_config import ModelConfig
        from tensorrt_llm._torch.modules.fused_moe import DefaultMoeRoutingMethod, VanillaMoE
        from tensorrt_llm._torch.utils import ActivationType

        activation_type = ActivationType[activation_type_name]
        num_experts = 4
        hidden_size = 32
        intermediate_size = 64
        num_tokens = 8

        torch.manual_seed(42)
        x = torch.randn(num_tokens, hidden_size, device="cuda")
        router_logits = torch.randn(num_tokens, num_experts, device="cuda")

        routing = DefaultMoeRoutingMethod(top_k=top_k, force_enable_pytorch_op=True)

        model_config = ModelConfig()
        vanilla_moe = VanillaMoE(
            routing_method=routing,
            num_experts=num_experts,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            dtype=torch.float32,
            model_config=model_config,
            activation_type=activation_type,
        ).cuda()

        # Initialize weights with random values (default is torch.empty / uninitialized)
        with torch.no_grad():
            for param in vanilla_moe.parameters():
                param.normal_(std=0.02)

        # Forward
        result = vanilla_moe(x, router_logits)

        # Reference
        ref = self._reference_moe_forward(x, router_logits, routing, vanilla_moe, 0, num_experts)

        assert torch.allclose(result, ref, atol=1e-5), (
            f"Max diff: {(result - ref).abs().max().item()}"
        )


# ──────────────────────────────────────────────────────────────────────────────
# §1  Routing Method Tests
# ──────────────────────────────────────────────────────────────────────────────


class TestDefaultMoeRouting:
    """§1a: DefaultMoeRoutingMethod — softmax then topk.

    Reference — routing.py: DefaultMoeRoutingMethod.apply_pytorch::

        scores = torch.softmax(router_logits.float(), dim=-1)
        topk_values, topk_indices = torch.topk(scores, k=top_k, dim=-1)
    """

    @pytest.mark.parametrize("top_k", [1, 2, 4])
    @pytest.mark.parametrize("num_tokens", [1, 8])
    @pytest.mark.parametrize("num_experts", [4, 16])
    def test_matches_reference(self, top_k, num_tokens, num_experts):
        from tensorrt_llm._torch.modules.fused_moe import DefaultMoeRoutingMethod

        if top_k > num_experts:
            pytest.skip("top_k > num_experts")

        torch.manual_seed(42)
        router_logits = torch.randn(num_tokens, num_experts, device="cuda")

        routing = DefaultMoeRoutingMethod(top_k=top_k, force_enable_pytorch_op=True)
        indices, scales = routing.apply(router_logits)

        # Reference implementation from moe_semantics.md §1a
        scores = torch.softmax(router_logits.float(), dim=-1)
        ref_values, ref_indices = torch.topk(scores, k=top_k, dim=-1)

        assert indices.dtype == torch.int32
        assert scales.dtype == torch.float32
        assert torch.equal(indices, ref_indices.to(torch.int32))
        assert torch.allclose(scales, ref_values, atol=1e-6)


class TestRenormalizeMoeRouting:
    """§1b: RenormalizeMoeRoutingMethod — topk then softmax.

    Reference — routing.py: RenormalizeMoeRoutingMethod.apply_pytorch::

        topk_values, topk_indices = torch.topk(router_logits, k=top_k, dim=-1)
        token_final_scales = torch.softmax(topk_values.float(), dim=-1)
    """

    @pytest.mark.parametrize("top_k", [1, 2, 4])
    @pytest.mark.parametrize("num_tokens", [1, 8])
    @pytest.mark.parametrize("num_experts", [4, 16])
    def test_matches_reference(self, top_k, num_tokens, num_experts):
        from tensorrt_llm._torch.modules.fused_moe import RenormalizeMoeRoutingMethod

        if top_k > num_experts:
            pytest.skip("top_k > num_experts")

        torch.manual_seed(42)
        router_logits = torch.randn(num_tokens, num_experts, device="cuda")

        routing = RenormalizeMoeRoutingMethod(top_k=top_k, force_enable_pytorch_op=True)
        indices, scales = routing.apply(router_logits)

        # Reference implementation from moe_semantics.md §1b
        ref_values, ref_indices = torch.topk(router_logits, k=top_k, dim=-1)
        ref_scales = torch.softmax(ref_values.float(), dim=-1)

        assert indices.dtype == torch.int32
        assert scales.dtype == torch.float32
        assert torch.equal(indices, ref_indices.to(torch.int32))
        assert torch.allclose(scales, ref_scales, atol=1e-6)


class TestRenormalizeNaiveMoeRouting:
    """§1c: RenormalizeNaiveMoeRoutingMethod — equivalent to §1b.

    Uses a different code path (softmax-then-topk-then-renormalize) but
    produces identical results because softmax of the top-k raw logits
    equals re-normalizing the softmax probabilities of the top-k entries.
    """

    @pytest.mark.parametrize("top_k", [1, 2, 4])
    @pytest.mark.parametrize("num_tokens", [1, 8])
    @pytest.mark.parametrize("num_experts", [4, 16])
    def test_equivalent_to_renormalize(self, top_k, num_tokens, num_experts):
        from tensorrt_llm._torch.modules.fused_moe import (
            RenormalizeMoeRoutingMethod,
            RenormalizeNaiveMoeRoutingMethod,
        )

        if top_k > num_experts:
            pytest.skip("top_k > num_experts")

        torch.manual_seed(42)
        # Use unique values to avoid ties
        router_logits = torch.randn(num_tokens, num_experts, device="cuda")

        renorm = RenormalizeMoeRoutingMethod(top_k=top_k, force_enable_pytorch_op=True)
        naive = RenormalizeNaiveMoeRoutingMethod(top_k=top_k)

        idx_renorm, scales_renorm = renorm.apply(router_logits)
        idx_naive, scales_naive = naive.apply(router_logits)

        assert torch.equal(idx_renorm, idx_naive)
        assert torch.allclose(scales_renorm, scales_naive, atol=1e-5)


class TestLlama4MoeRouting:
    """§1e: Llama4RenormalizeMoeRoutingMethod — topk then sigmoid.

    Reference — routing.py: Llama4RenormalizeMoeRoutingMethod.apply_pytorch::

        topk_values, topk_indices = torch.topk(router_logits, k=top_k, dim=-1)
        token_final_scales = torch.sigmoid(topk_values.float())

    Note: uses sigmoid (not softmax), so scales are independent per expert.
    """

    @pytest.mark.parametrize("top_k", [1, 2])
    @pytest.mark.parametrize("num_tokens", [1, 8])
    @pytest.mark.parametrize("num_experts", [4, 16])
    def test_matches_reference(self, top_k, num_tokens, num_experts):
        from tensorrt_llm._torch.modules.fused_moe import Llama4RenormalizeMoeRoutingMethod

        if top_k > num_experts:
            pytest.skip("top_k > num_experts")

        torch.manual_seed(42)
        router_logits = torch.randn(num_tokens, num_experts, device="cuda")

        routing = Llama4RenormalizeMoeRoutingMethod(top_k=top_k)
        indices, scales = routing.apply(router_logits)

        # Reference implementation from moe_semantics.md §1e
        ref_values, ref_indices = torch.topk(router_logits, k=top_k, dim=-1)
        ref_scales = torch.sigmoid(ref_values.float())

        assert indices.dtype == torch.int32
        assert scales.dtype == torch.float32
        assert torch.equal(indices, ref_indices.to(torch.int32))
        assert torch.allclose(scales, ref_scales, atol=1e-6)


class TestDeepSeekV3MoeRouting:
    """§1d: DeepSeekV3MoeRoutingMethod — sigmoid + group topk + expert topk.

    Three-stage routing — routing.py: DeepSeekV3MoeRoutingMethod.apply_pytorch::

        1. scores = sigmoid(logits) + e_score_correction_bias
        2. Group-level topk: score each of n_group groups by sum of its
           top-2 expert scores; keep only topk_group groups.
        3. Expert-level topk: from surviving experts pick top_k.
        4. Normalize: selected sigmoid scores (without bias) are divided
           by their sum and multiplied by routed_scaling_factor.
    """

    @pytest.mark.parametrize(
        "top_k,n_group,topk_group,num_experts",
        [
            (6, 4, 2, 64),
            (8, 8, 4, 256),
        ],
    )
    def test_matches_reference(self, top_k, n_group, topk_group, num_experts):
        from tensorrt_llm._torch.modules.fused_moe import DeepSeekV3MoeRoutingMethod

        num_tokens = 4
        routed_scaling_factor = 2.5
        torch.manual_seed(42)

        router_logits = torch.randn(num_tokens, num_experts, device="cuda", dtype=torch.float32)
        e_score_correction_bias = torch.randn(num_experts, device="cuda", dtype=torch.float32) * 0.1

        routing = DeepSeekV3MoeRoutingMethod(
            top_k=top_k,
            n_group=n_group,
            topk_group=topk_group,
            routed_scaling_factor=routed_scaling_factor,
            callable_e_score_correction_bias=lambda: e_score_correction_bias,
            is_fused=False,
        )
        indices, scales = routing.apply(router_logits)

        # Reference implementation from moe_semantics.md §1d
        scores = torch.sigmoid(router_logits)
        scores_with_bias = scores + e_score_correction_bias

        # Group-level TopK
        grouped = scores_with_bias.view(num_tokens, n_group, num_experts // n_group)
        group_scores = torch.topk(grouped, k=2, dim=-1).values.sum(dim=-1)
        _, group_idx = torch.topk(group_scores, k=topk_group, dim=-1)

        group_mask = torch.zeros_like(group_scores)
        group_mask.scatter_(-1, group_idx, 1)
        score_mask = group_mask.unsqueeze(-1).expand_as(grouped).reshape_as(scores_with_bias)
        scores_with_bias = torch.where(
            score_mask.bool(), scores_with_bias, torch.tensor(float("-inf"))
        )

        # Expert-level TopK
        _, topk_idx = torch.topk(scores_with_bias, k=top_k, dim=-1)

        new_mask = torch.zeros_like(scores)
        new_mask.scatter_(-1, topk_idx, 1)
        scores = scores * new_mask

        score_sum = scores.sum(dim=-1, keepdim=True) + 1e-20
        scores = scores / score_sum * routed_scaling_factor

        ref_values, ref_indices = torch.topk(scores, k=top_k, dim=-1)

        assert indices.dtype == torch.int32
        assert scales.dtype == torch.float32
        assert torch.equal(indices, ref_indices.to(torch.int32))
        assert torch.allclose(scales, ref_values.float(), atol=1e-5)


class TestMiniMaxM2MoeRouting:
    """§1f: MiniMaxM2MoeRoutingMethod — sigmoid + bias topk + normalize.

    Reference — routing.py: MiniMaxM2MoeRoutingMethod.apply_pytorch::

        scores = sigmoid(logits)
        scores_with_bias = scores + e_score_correction_bias
        _, topk_idx = topk(scores_with_bias, k=top_k, sorted=False)
        top_k_weights = scores.gather(1, topk_idx)  # no bias
        top_k_weights /= top_k_weights.sum(dim=-1) + 1e-20  # normalize
    """

    @pytest.mark.parametrize("top_k", [2, 4])
    @pytest.mark.parametrize("num_experts", [8, 32])
    def test_matches_reference(self, top_k, num_experts):
        from tensorrt_llm._torch.modules.fused_moe import MiniMaxM2MoeRoutingMethod

        num_tokens = 8
        torch.manual_seed(42)

        router_logits = torch.randn(num_tokens, num_experts, device="cuda", dtype=torch.float32)
        e_score_correction_bias = torch.randn(num_experts, device="cuda", dtype=torch.float32) * 0.1

        routing = MiniMaxM2MoeRoutingMethod(
            top_k=top_k,
            num_experts=num_experts,
            callable_e_score_correction_bias=lambda: e_score_correction_bias,
        )
        indices, scales = routing.apply(router_logits)

        # Reference implementation from moe_semantics.md §1f
        ref_scores = torch.sigmoid(router_logits)
        ref_scores_with_bias = ref_scores + e_score_correction_bias

        _, ref_topk_idx = torch.topk(ref_scores_with_bias, k=top_k, dim=-1, sorted=False)
        ref_top_k_weights = ref_scores.gather(1, ref_topk_idx)
        ref_top_k_weights = ref_top_k_weights / (
            ref_top_k_weights.sum(dim=-1, keepdim=True) + 1e-20
        )

        assert indices.dtype == torch.int32
        assert scales.dtype == torch.float32
        # Indices may be in different order (sorted=False), so compare sets
        for t in range(num_tokens):
            actual_set = set(indices[t].cpu().tolist())
            ref_set = set(ref_topk_idx[t].cpu().tolist())
            assert actual_set == ref_set, f"Token {t}: experts mismatch {actual_set} vs {ref_set}"
        # Scales should sum to 1.0
        assert torch.allclose(
            scales.sum(dim=-1),
            torch.ones(num_tokens, device="cuda", dtype=torch.float32),
            atol=1e-5,
        )


# ──────────────────────────────────────────────────────────────────────────────
# §14  SwiGLU Variant Tests (swiglu_torch)
# ──────────────────────────────────────────────────────────────────────────────


class TestSwigluTorch:
    """§14: swiglu_alpha / swiglu_beta / swiglu_limit — GPT-OSS SwiGLU variant.

    When these three tensors are provided to ``create_moe``, the activation
    switches from standard SwiGLU to the GPT-OSS variant defined in
    ``fused_moe_triton.py: swiglu_torch``::

        gate = fc1_out[..., ::2]  # even columns
        up = fc1_out[..., 1::2]  # odd columns
        gate = gate.clamp(max=limit)  # clamp BEFORE activation
        up = up.clamp(min=-limit, max=limit)
        output = (gate * sigmoid(alpha * gate)) * (up + beta)

    Special cases:
    - alpha=1, beta=0, limit=None  =>  standard SwiGLU: silu(gate) * up
    - limit=None                   =>  no clamping
    """

    def test_matches_reference(self):
        from tensorrt_llm._torch.modules.fused_moe.fused_moe_triton import swiglu_torch

        torch.manual_seed(42)
        # Input shape: interleaved [gate, up, gate, up, ...] of size 2*intermediate
        intermediate_size = 64
        num_tokens = 8
        a = torch.randn(num_tokens, intermediate_size * 2, device="cuda")

        alpha = 1.702
        beta = 1.0
        limit = 7.0

        result = swiglu_torch(a, alpha, beta, limit)

        # Reference from moe_semantics.md §14
        a_glu = a[..., ::2]  # gate (even columns)
        a_glu = a_glu.clamp(max=limit)
        a_linear = a[..., 1::2]  # up (odd columns)
        a_linear = a_linear.clamp(min=-limit, max=limit)

        out_glu = a_glu * torch.sigmoid(alpha * a_glu)
        ref_result = out_glu * (a_linear + beta)

        assert torch.allclose(result, ref_result, atol=1e-6)

    def test_no_limit(self):
        from tensorrt_llm._torch.modules.fused_moe.fused_moe_triton import swiglu_torch

        torch.manual_seed(42)
        intermediate_size = 64
        num_tokens = 8
        a = torch.randn(num_tokens, intermediate_size * 2, device="cuda")

        alpha = 1.702
        beta = 1.0

        result = swiglu_torch(a, alpha, beta, limit=None)

        # Without limit, no clamping
        a_glu = a[..., ::2]
        a_linear = a[..., 1::2]
        out_glu = a_glu * torch.sigmoid(alpha * a_glu)
        ref_result = out_glu * (a_linear + beta)

        assert torch.allclose(result, ref_result, atol=1e-6)

    def test_standard_swiglu_equiv(self):
        """When alpha=1, beta=0, limit=None: should be equivalent to standard SwiGLU."""
        from tensorrt_llm._torch.modules.fused_moe.fused_moe_triton import swiglu_torch

        torch.manual_seed(42)
        intermediate_size = 64
        num_tokens = 8
        a = torch.randn(num_tokens, intermediate_size * 2, device="cuda")

        result = swiglu_torch(a, alpha=1.0, beta=0.0, limit=None)

        # Standard SwiGLU: silu(gate) * up
        gate = a[..., ::2]
        up = a[..., 1::2]
        ref_result = F.silu(gate) * up

        assert torch.allclose(result, ref_result, atol=1e-6)


# ──────────────────────────────────────────────────────────────────────────────
# §15  Activation Type Tests
# ──────────────────────────────────────────────────────────────────────────────


class TestActivationTypes:
    """§15: activation_type — Swiglu, Geglu, Relu2.

    Controls the activation between FC1 and FC2 in each expert::

        Swiglu (gated, default):
            W1=gate, W3=up → silu(gate) * up           (swiglu.py: silu_and_mul_kernel)
            VanillaMoE uses GatedMLP per expert.

        Geglu (gated):
            W1=gate, W3=up → gelu(gate) * up            (gated_mlp.py: GatedMLP._apply_activation)
            VanillaMoE uses GatedMLP per expert.

        Relu2 (non-gated):
            W1=up only     → relu(up)²                  (utils.py: relu2)
            VanillaMoE uses MLP per expert.

    ``is_gated_activation()`` returns True for Swiglu/Geglu, False for Relu2.
    Gated activations have intermediate_size_expand_ratio=2 (fused FC1 weight
    has 2*intermediate_size rows); non-gated have ratio=1.
    """

    def test_relu2(self):
        """§15c: relu2(x) = relu(x)^2."""
        from tensorrt_llm._torch.utils import relu2

        torch.manual_seed(42)
        x = torch.randn(8, 64, device="cuda")

        result = relu2(x)
        ref = torch.square(F.relu(x))

        assert torch.allclose(result, ref, atol=1e-6)

    def test_swiglu_kernel(self):
        """§15a: silu_and_mul — silu(gate) * up."""
        torch.manual_seed(42)
        intermediate_size = 64
        x = torch.randn(8, intermediate_size * 2, device="cuda")

        result = torch.ops.trtllm.silu_and_mul(x)

        # Reference: silu(first_half) * second_half
        gate = x[..., :intermediate_size]
        up = x[..., intermediate_size:]
        ref = F.silu(gate) * up

        assert result.shape == (8, intermediate_size)
        assert torch.allclose(result, ref, atol=1e-5)

    def test_is_gated_activation(self):
        """Gated vs non-gated classification."""
        from tensorrt_llm._torch.utils import ActivationType, is_gated_activation

        assert is_gated_activation(ActivationType.Swiglu) is True
        assert is_gated_activation(ActivationType.Geglu) is True
        assert is_gated_activation(ActivationType.Relu2) is False


# ──────────────────────────────────────────────────────────────────────────────
# §12  apply_router_weight_on_input Tests
# ──────────────────────────────────────────────────────────────────────────────


class TestApplyRouterWeightOnInput:
    """§12: apply_router_weight_on_input — scale position.

    Controls where the routing scale factor is multiplied::

        False (default):  output += expert_forward(x) * scale
        True:             output += expert_forward(x * scale)

    For a linear expert these are equivalent (s*(x@W) == (s*x)@W).
    For a nonlinear expert (e.g. SwiGLU) they differ because
    silu(s*x) != s*silu(x).

    Currently restricted to top_k==1 (asserted in CutlassFusedMoE.__init__).
    Used by Llama 4 (modeling_llama.py: LlamaMoE).
    """

    def test_scale_equivalence_top1(self):
        """For a single linear expert, scale*input*W should equal scale*(input*W)."""
        torch.manual_seed(42)
        hidden_size = 32
        num_tokens = 4

        x = torch.randn(num_tokens, hidden_size, device="cuda")
        W = torch.randn(hidden_size, hidden_size, device="cuda")
        scales = torch.rand(num_tokens, 1, device="cuda")

        # apply_router_weight_on_input=True: scale on input
        out_on_input = (x * scales) @ W

        # apply_router_weight_on_input=False: scale on output
        out_on_output = (x @ W) * scales

        # For linear operations, these are mathematically equivalent
        # (relaxed tolerance due to float32 matmul reordering)
        assert torch.allclose(out_on_input, out_on_output, atol=5e-3)

    def test_scale_not_equivalent_nonlinear(self):
        """For nonlinear ops (SwiGLU), scale-before != scale-after."""
        torch.manual_seed(42)
        hidden_size = 32
        intermediate_size = 64
        num_tokens = 4

        x = torch.randn(num_tokens, hidden_size, device="cuda")
        W1 = torch.randn(intermediate_size, hidden_size, device="cuda")
        W3 = torch.randn(intermediate_size, hidden_size, device="cuda")
        W2 = torch.randn(hidden_size, intermediate_size, device="cuda")
        scales = torch.rand(num_tokens, 1, device="cuda") * 0.5 + 0.5

        # Scale on input
        x_scaled = x * scales
        gate1 = x_scaled @ W1.T
        up1 = x_scaled @ W3.T
        mid1 = F.silu(gate1) * up1
        out_on_input = mid1 @ W2.T

        # Scale on output
        gate2 = x @ W1.T
        up2 = x @ W3.T
        mid2 = F.silu(gate2) * up2
        out_on_output = (mid2 @ W2.T) * scales

        # These should NOT be equal due to nonlinear activation
        assert not torch.allclose(out_on_input, out_on_output, atol=1e-3)


# ──────────────────────────────────────────────────────────────────────────────
# §10  Weight Loading Mode Tests
# ──────────────────────────────────────────────────────────────────────────────


class TestWeightLoadingMode:
    """§10: weight_loading_mode — VANILLA vs FUSED_GATE_UP_PROJ.

    Controls how W1 (gate) and W3 (up) are stored in memory.
    The logical computation is identical::

        VANILLA:
            W1: [num_experts, intermediate_size, hidden_size]  (gate)
            W3: [num_experts, intermediate_size, hidden_size]  (up)
            gate = x @ W1.T;  up = x @ W3.T

        FUSED_GATE_UP_PROJ:
            W3_W1: [num_experts, 2*intermediate_size, hidden_size]
              first  intermediate_size rows = W3 (up)
              second intermediate_size rows = W1 (gate)
            gate_up = x @ W3_W1.T;  up, gate = gate_up.chunk(2, dim=-1)
    """

    def test_fused_gate_up_equivalence(self):
        """Fused gate+up matmul then split equals separate matmuls."""
        torch.manual_seed(42)
        hidden_size = 32
        intermediate_size = 64
        num_tokens = 4

        x = torch.randn(num_tokens, hidden_size, device="cuda")
        W1 = torch.randn(intermediate_size, hidden_size, device="cuda")
        W3 = torch.randn(intermediate_size, hidden_size, device="cuda")

        # VANILLA: separate matmuls
        gate = x @ W1.T
        up = x @ W3.T

        # FUSED_GATE_UP_PROJ: single matmul then split
        # Layout: first intermediate_size rows = W3 (up), next = W1 (gate)
        W3_W1 = torch.cat([W3, W1], dim=0)
        gate_up = x @ W3_W1.T
        up_fused, gate_fused = gate_up.chunk(2, dim=-1)

        assert torch.allclose(gate, gate_fused, atol=1e-5)
        assert torch.allclose(up, up_fused, atol=1e-5)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
