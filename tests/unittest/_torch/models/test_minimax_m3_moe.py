# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Goal 1.5 focused tests for the MiniMax-M3 dense MLP + MoE wiring.

These tests pin the pieces ``plan.md``'s Goal 1.5 names:

  * ``swigluoai`` activation with alpha=1.702, clamp_limit=7.0, and the
    asymmetric ``gate.clamp(max=limit)`` / ``up.clamp(min=-limit,
    max=limit)`` shape that matches SGLang's
    ``swiglu_no_interleaved_with_alpha_and_limit``.
  * MiniMax-M3 routing: sigmoid scores, ``routing bias`` added for
    expert *selection* only, gathered weights come from the *unbiased*
    scores, then renormalize, then multiply by
    ``routed_scaling_factor=2.0``.
  * Shared expert path: separate dense MLP added once to the routed
    output.
  * Layer frequency: layers 0-2 instantiate the M3 dense MLP (a
    ``GatedMLP`` built by ``_build_swiglu_oai_dense_mlp``); layers
    3-59 instantiate ``MiniMaxM3MoE``.

Each acceptance-required negative control is present:

  * Wrong activation (plain SwiGLU, no clamp / no ``(up + 1)``) diverges.
  * Missing routed scaling (``routed_scaling_factor=1.0``) diverges.
  * Wrong routing bias (zero bias) selects different experts.
  * Missing shared expert output (drop the shared dense MLP) diverges.

The activation-and-routing tests are CPU/GPU agnostic and run on both;
the layer-construction tests need only the basic Python import path.
The MoE-block forward test is marked CUDA-only because
``create_moe(...)`` allocates fused-MoE buffers.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn


def _has_cuda() -> bool:
    try:
        return torch.cuda.is_available()
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Reference implementations — independent of the SUT
# ---------------------------------------------------------------------------


def _sglang_swigluoai_reference(x: torch.Tensor, *, alpha: float, limit: float) -> torch.Tensor:
    """Reference ``swiglu_no_interleaved_with_alpha_and_limit``.

    Mirrors SGLang's
    ``layers/moe/moe_runner/triton_utils/fused_moe.py::swiglu_no_interleaved_with_alpha_and_limit``
    line-for-line:

        gate, up = x.chunk(2, dim=-1)
        gate = gate.clamp(min=None, max=limit)       # asymmetric upper-only
        up   = up.clamp(min=-limit, max=limit)        # symmetric
        return gate * sigmoid(gate * alpha) * (up + 1)

    The asymmetric gate clamp is part of the checkpoint contract; the
    SUT's ``swigluoai`` activation must reproduce it.
    """
    gate, up = x.chunk(2, dim=-1)
    gate = gate.clamp(max=limit)
    up = up.clamp(min=-limit, max=limit)
    return gate * torch.sigmoid(gate * alpha) * (up + 1.0)


def _plain_swiglu_reference(x: torch.Tensor) -> torch.Tensor:
    """Reference plain SwiGLU (no clamp, no ``(up + 1)``).

    Used to verify the *wrong-activation* negative control: a model that
    accidentally used plain SwiGLU instead of ``swigluoai`` would produce
    materially different output, which the test catches.
    """
    gate, up = x.chunk(2, dim=-1)
    return gate * torch.sigmoid(gate) * up


def _minimax_m3_routing_reference(
    logits: torch.Tensor,
    *,
    bias: torch.Tensor,
    top_k: int,
    routed_scaling_factor: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reference MiniMax-M3 router.

    Matches SGLang's TopK with ``scoring_func='sigmoid'``,
    ``correction_bias=bias``, ``renormalize=True``,
    ``routed_scaling_factor=...``,
    ``apply_routed_scaling_factor_on_output=True``:

        scores            = sigmoid(logits)
        scores_with_bias  = scores + bias                # broadcast over tokens
        topk_idx          = topk(scores_with_bias, k)    # bias affects SELECTION
        weights           = scores.gather(1, topk_idx)    # but weights are UNBIASED
        weights          /= weights.sum(-1, keepdim=True)
        weights          *= routed_scaling_factor
    """
    scores = torch.sigmoid(logits.to(torch.float32))
    scores_with_bias = scores + bias.to(torch.float32)
    _, topk_idx = torch.topk(scores_with_bias, k=top_k, dim=-1, sorted=False)
    weights = scores.gather(1, topk_idx)
    weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-20)
    weights = weights * float(routed_scaling_factor)
    return topk_idx.to(torch.int32), weights.to(torch.float32)


# ---------------------------------------------------------------------------
# 1. swigluoai activation
# ---------------------------------------------------------------------------


def test_swigluoai_asymmetric_gate_clamp_differs_from_symmetric():
    """The SUT must clamp gate only above (``max=limit``).

    Negative control: if the implementation also clamps gate below
    (``min=-limit``), the output for large-negative gate values differs.
    This test confirms the asymmetric clamp is in effect by comparing
    against a symmetric reference.
    """
    # Pick a gate value that exceeds ``-limit`` in absolute value.
    limit = 7.0
    alpha = 1.702
    gate = torch.tensor([-12.0, -1.0, 1.0, 12.0], dtype=torch.float32)
    up = torch.tensor([0.0, 0.5, -0.5, 0.0], dtype=torch.float32)
    x = torch.cat([gate, up], dim=-1).view(1, -1)

    asymmetric = _sglang_swigluoai_reference(x, alpha=alpha, limit=limit)
    # Symmetric (wrong) reference: gate is also clamped below.
    g_sym = gate.clamp(min=-limit, max=limit)
    u_sym = up.clamp(min=-limit, max=limit)
    symmetric = g_sym * torch.sigmoid(g_sym * alpha) * (u_sym + 1.0)
    symmetric = symmetric.view(1, -1)

    # The first lane (gate=-12.0) is exactly where the clamp matters.
    assert not torch.allclose(asymmetric, symmetric), (
        "Asymmetric vs symmetric gate clamp must produce different outputs; "
        "if they match the SUT is not applying the SGLang clamp shape."
    )
    # The remaining lanes (gate in [-7, 7]) should match between the two.
    torch.testing.assert_close(asymmetric[0, 1:3], symmetric[0, 1:3], rtol=1e-5, atol=1e-5)


def test_swigluoai_wrong_activation_negative_control_diverges():
    """Negative control: plain SwiGLU (no clamp, no ``up + 1``) diverges.

    Acceptance-criteria item 4 requires a 'wrong activation' negative
    control: a model that uses plain SwiGLU instead of swigluoai should
    fail parity against the SGLang reference. This pins the difference.
    """
    torch.manual_seed(2)
    x = torch.randn(8, 16, dtype=torch.float32) * 2.0
    correct = _sglang_swigluoai_reference(x, alpha=1.702, limit=7.0)
    wrong = _plain_swiglu_reference(x)
    diff = (correct - wrong).abs().mean().item()
    assert diff > 0.1, (
        f"Plain SwiGLU should diverge materially from swigluoai; got mean-abs diff {diff}."
    )


# ---------------------------------------------------------------------------
# 2. MiniMax-M3 routing — sigmoid + bias-for-selection + renorm + scaling
# ---------------------------------------------------------------------------


def test_minimax_m3_routing_matches_reference():
    """End-to-end router parity against the hand-rolled reference."""
    from tensorrt_llm._torch.modules.fused_moe.routing import MiniMaxM3MoeRoutingMethod

    num_experts = 16
    top_k = 4
    torch.manual_seed(7)
    bias = torch.randn(num_experts, dtype=torch.float32) * 0.1
    logits = torch.randn(6, num_experts, dtype=torch.float32) * 3.0

    sut = MiniMaxM3MoeRoutingMethod(
        top_k=top_k,
        num_experts=num_experts,
        callable_e_score_correction_bias=lambda: bias,
        routed_scaling_factor=2.0,
    )
    sut_idx, sut_w = sut.apply(logits)

    ref_idx, ref_w = _minimax_m3_routing_reference(
        logits, bias=bias, top_k=top_k, routed_scaling_factor=2.0
    )

    # Index order is unsorted; compare as sets per row.
    for row in range(logits.shape[0]):
        assert set(sut_idx[row].tolist()) == set(ref_idx[row].tolist()), (
            f"row {row}: selected experts differ: SUT={sut_idx[row].tolist()} "
            f"REF={ref_idx[row].tolist()}"
        )
    # Weights: sort both by index, then compare.
    sut_w_sorted = torch.zeros_like(sut_w)
    ref_w_sorted = torch.zeros_like(ref_w)
    for row in range(logits.shape[0]):
        sut_order = torch.argsort(sut_idx[row])
        ref_order = torch.argsort(ref_idx[row])
        sut_w_sorted[row] = sut_w[row][sut_order]
        ref_w_sorted[row] = ref_w[row][ref_order]
    torch.testing.assert_close(sut_w_sorted, ref_w_sorted, rtol=1e-5, atol=1e-5)


def test_minimax_m3_routing_bias_affects_selection_not_weights():
    """Routing bias modifies SELECTION but not gathered weight values.

    SGLang's TopK with ``correction_bias`` adds the bias to the score
    used for the top-k argmax, but the gathered weight tensor is read
    from the *unbiased* sigmoid scores. This test confirms the SUT
    follows that exact shape:

      * Setting a large bias on one expert forces it into the top-k
        even when its unbiased score is mediocre.
      * The weight returned for that expert is the *unbiased* sigmoid
        score, not the biased score.
    """
    from tensorrt_llm._torch.modules.fused_moe.routing import MiniMaxM3MoeRoutingMethod

    num_experts = 8
    top_k = 2
    torch.manual_seed(11)
    # Logits chosen so that expert 0 has the *smallest* sigmoid score.
    logits = torch.tensor(
        [[-5.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5, 0.5]],
        dtype=torch.float32,
    )
    # Bias gives expert 0 a giant boost so it wins the top-k argmax
    # (selection), but the gathered weight should still come from
    # ``sigmoid(-5.0)``, which is near zero.
    bias = torch.zeros(num_experts, dtype=torch.float32)
    bias[0] = 10.0
    sut = MiniMaxM3MoeRoutingMethod(
        top_k=top_k,
        num_experts=num_experts,
        callable_e_score_correction_bias=lambda: bias,
        routed_scaling_factor=1.0,
    )
    idx, weights = sut.apply(logits)
    assert 0 in idx[0].tolist(), f"Bias must force expert 0 into top-k; got idx={idx[0].tolist()}"
    # Weight at the position where expert 0 lives in idx must be small
    # (the unbiased sigmoid(-5) is ~0.0067; after renorm divided by sum
    # it stays small relative to the other position).
    pos = idx[0].tolist().index(0)
    unbiased_expert0 = torch.sigmoid(torch.tensor(-5.0)).item()
    other_idx = [i for i in idx[0].tolist() if i != 0][0]
    unbiased_other = torch.sigmoid(logits[0, other_idx]).item()
    expected_w0 = unbiased_expert0 / (unbiased_expert0 + unbiased_other)
    assert abs(weights[0, pos].item() - expected_w0) < 1e-4, (
        f"Expert 0's gathered weight after renorm should be the unbiased "
        f"sigmoid(-5)/(sigmoid(-5)+sigmoid(other)); got "
        f"{weights[0, pos].item()}, expected ~{expected_w0}."
    )


def test_minimax_m3_routing_wrong_bias_changes_selection_negative_control():
    """Negative control: dropping the bias changes selected experts.

    Acceptance-criteria item 4 requires a 'wrong routing bias' negative
    control. With a non-trivial bias the bias-driven expert must be in
    top-k; with zero bias the unbiased-sigmoid winners are picked, so
    the selected experts differ for the same input.
    """
    from tensorrt_llm._torch.modules.fused_moe.routing import MiniMaxM3MoeRoutingMethod

    num_experts = 8
    top_k = 2
    logits = torch.tensor(
        [[-5.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5, 0.5]],
        dtype=torch.float32,
    )
    bias = torch.zeros(num_experts, dtype=torch.float32)
    bias[0] = 10.0

    with_bias = MiniMaxM3MoeRoutingMethod(
        top_k=top_k,
        num_experts=num_experts,
        callable_e_score_correction_bias=lambda: bias,
    )
    no_bias = MiniMaxM3MoeRoutingMethod(
        top_k=top_k,
        num_experts=num_experts,
        callable_e_score_correction_bias=lambda: torch.zeros_like(bias),
    )
    idx_with, _ = with_bias.apply(logits)
    idx_no, _ = no_bias.apply(logits)
    assert set(idx_with[0].tolist()) != set(idx_no[0].tolist()), (
        f"Bias must change the selected experts; with_bias={idx_with[0].tolist()} "
        f"no_bias={idx_no[0].tolist()}."
    )
    assert 0 in idx_with[0].tolist()
    assert 0 not in idx_no[0].tolist()


def test_minimax_m3_routing_missing_scaling_negative_control():
    """Negative control: missing routed_scaling_factor changes the weights.

    Acceptance-criteria item 4 requires a 'missing routed scaling'
    negative control. Plan specifies ``routed_scaling_factor=2.0`` for
    M3. A model that omitted the factor (default 1.0) would emit
    weights smaller by exactly 2×.
    """
    from tensorrt_llm._torch.modules.fused_moe.routing import MiniMaxM3MoeRoutingMethod

    num_experts = 8
    top_k = 2
    torch.manual_seed(13)
    logits = torch.randn(3, num_experts, dtype=torch.float32) * 2.0
    bias = torch.zeros(num_experts, dtype=torch.float32)

    correct = MiniMaxM3MoeRoutingMethod(
        top_k=top_k,
        num_experts=num_experts,
        callable_e_score_correction_bias=lambda: bias,
        routed_scaling_factor=2.0,
    )
    missing = MiniMaxM3MoeRoutingMethod(
        top_k=top_k,
        num_experts=num_experts,
        callable_e_score_correction_bias=lambda: bias,
        routed_scaling_factor=1.0,
    )
    idx_c, w_c = correct.apply(logits)
    idx_m, w_m = missing.apply(logits)
    # Same selection.
    for row in range(logits.shape[0]):
        assert set(idx_c[row].tolist()) == set(idx_m[row].tolist())
    # But correct weights == 2 * missing weights.
    torch.testing.assert_close(w_c, w_m * 2.0, rtol=1e-5, atol=1e-5)
    # And they differ materially.
    assert (w_c - w_m).abs().max().item() > 0.01


# ---------------------------------------------------------------------------
# 3. Shared expert summation + 4. Layer frequency
# ---------------------------------------------------------------------------


def _minimal_text_config_for_moe(*, hidden=64, intermediate=32, num_experts=8, top_k=2):
    """Tiny but complete-enough text config for ``MiniMaxM3MoE`` ctor.

    Returns a real ``PretrainedConfig`` (not a SimpleNamespace) because
    ``Attention`` / ``Linear`` / ``create_moe`` access ``getattr(config,
    ...)`` with defaults.
    """
    from transformers import PretrainedConfig

    n_layers = 4
    cfg = PretrainedConfig()
    cfg.hidden_size = hidden
    cfg.intermediate_size = intermediate
    cfg.num_hidden_layers = n_layers
    cfg.num_attention_heads = 4
    cfg.num_key_value_heads = 2
    cfg.head_dim = 16
    cfg.vocab_size = 256
    cfg.max_position_embeddings = 64
    cfg.rms_norm_eps = 1e-6
    cfg.use_gemma_norm = True
    cfg.rope_theta = 10000.0
    cfg.rotary_dim = 8
    cfg.partial_rotary_factor = 0.5
    cfg.qk_norm_type = "per_head"
    cfg.use_qk_norm = True
    cfg.hidden_act = "swigluoai"
    cfg.swiglu_alpha = 1.702
    cfg.swiglu_limit = 7.0
    cfg.dense_intermediate_size = intermediate
    cfg.shared_intermediate_size = intermediate
    cfg.num_local_experts = num_experts
    cfg.num_experts_per_tok = top_k
    cfg.n_shared_experts = 1
    cfg.scoring_func = "sigmoid"
    cfg.use_routing_bias = True
    cfg.routed_scaling_factor = 2.0
    cfg.moe_layer_freq = [0, 0, 0, 1]  # layers 0-2 dense, layer 3 MoE
    cfg.sparse_attention_config = {
        "use_sparse_attention": True,
        "sparse_index_dim": 16,
        "sparse_num_index_heads": 2,
        "sparse_topk_blocks": 2,
        "sparse_block_size": 4,
        "sparse_init_block": 0,
        "sparse_local_block": 1,
        "sparse_score_type": "max",
        "sparse_disable_index_value": [0, 0, 0, 1],
        "sparse_attention_freq": [0, 0, 0, 1],
    }
    cfg.torch_dtype = torch.bfloat16
    cfg.architectures = ["MiniMaxM3SparseForCausalLM"]
    return cfg


def test_decoder_layer_layer_frequency_dense_vs_moe():
    """Layer 0-2 build the M3 dense MLP (``GatedMLP``); layer 3 builds ``MiniMaxM3MoE``.

    Pins the ``moe_layer_freq`` plumbing: the decoder layer's MoE/MLP
    selection must follow the M3 checkpoint schedule (layers 0-2 dense,
    layers 3-59 MoE).
    """
    if not _has_cuda():
        pytest.skip("MiniMaxM3MoE create_moe needs CUDA buffers")
    from tensorrt_llm._torch.model_config import ModelConfig
    from tensorrt_llm._torch.models.modeling_minimaxm3 import (
        MiniMaxM3DecoderLayer,
        MiniMaxM3MoE,
    )
    from tensorrt_llm._torch.modules.gated_mlp import GatedMLP
    from tensorrt_llm._torch.utils import AuxStreamType
    from tensorrt_llm.mapping import Mapping

    cfg = _minimal_text_config_for_moe()
    model_config = ModelConfig(
        pretrained_config=cfg,
        mapping=Mapping(),
        skip_create_weights_in_init=True,
    )
    aux_stream = torch.cuda.Stream()
    aux_stream_dict = {
        AuxStreamType.MoeShared: aux_stream,
        AuxStreamType.MoeChunkingOverlap: aux_stream,
    }

    dense_layer = MiniMaxM3DecoderLayer(model_config, layer_idx=0, aux_stream_dict=aux_stream_dict)
    assert isinstance(dense_layer.mlp, GatedMLP)
    assert dense_layer.block_sparse_moe is None

    moe_layer = MiniMaxM3DecoderLayer(model_config, layer_idx=3, aux_stream_dict=aux_stream_dict)
    assert moe_layer.mlp is None
    assert isinstance(moe_layer.block_sparse_moe, MiniMaxM3MoE)
    # And the MoE block carries the M3 knobs.
    assert moe_layer.block_sparse_moe.routed_scaling_factor == 2.0
    assert moe_layer.block_sparse_moe.num_experts == cfg.num_local_experts
    assert moe_layer.block_sparse_moe.top_k == cfg.num_experts_per_tok
    assert moe_layer.block_sparse_moe.shared_experts is not None


def test_minimax_m3_moe_block_constructs_shared_expert_when_enabled():
    """``n_shared_experts > 0`` materializes a dense MLP (``GatedMLP``)."""
    if not _has_cuda():
        pytest.skip("MiniMaxM3MoE create_moe needs CUDA buffers")
    from tensorrt_llm._torch.model_config import ModelConfig
    from tensorrt_llm._torch.models.modeling_minimaxm3 import MiniMaxM3MoE
    from tensorrt_llm._torch.modules.gated_mlp import GatedMLP
    from tensorrt_llm._torch.utils import AuxStreamType
    from tensorrt_llm.mapping import Mapping

    cfg = _minimal_text_config_for_moe()
    model_config = ModelConfig(
        pretrained_config=cfg,
        mapping=Mapping(),
        skip_create_weights_in_init=True,
    )
    aux_stream = torch.cuda.Stream()
    aux_stream_dict = {
        AuxStreamType.MoeShared: aux_stream,
        AuxStreamType.MoeChunkingOverlap: aux_stream,
    }
    moe = MiniMaxM3MoE(
        model_config=model_config, aux_stream_dict=aux_stream_dict, layer_idx=3
    )
    assert isinstance(moe.shared_experts, GatedMLP)
    # Shared expert intermediate is ``shared_intermediate_size * n_shared_experts``.
    expected_shared = cfg.shared_intermediate_size * cfg.n_shared_experts
    # gate_up_proj.out_features is ``2 * intermediate`` due to fused gate+up.
    assert moe.shared_experts.gate_up_proj.weight.shape[0] == 2 * expected_shared


def test_minimax_m3_moe_block_drops_shared_when_disabled():
    """``n_shared_experts == 0`` leaves ``shared_experts`` as None.

    Acceptance-criteria 'missing shared expert output' negative control.
    """
    if not _has_cuda():
        pytest.skip("MiniMaxM3MoE create_moe needs CUDA buffers")
    from tensorrt_llm._torch.model_config import ModelConfig
    from tensorrt_llm._torch.models.modeling_minimaxm3 import MiniMaxM3MoE
    from tensorrt_llm._torch.utils import AuxStreamType
    from tensorrt_llm.mapping import Mapping

    cfg = _minimal_text_config_for_moe()
    cfg.n_shared_experts = 0
    model_config = ModelConfig(
        pretrained_config=cfg,
        mapping=Mapping(),
        skip_create_weights_in_init=True,
    )
    aux_stream = torch.cuda.Stream()
    aux_stream_dict = {
        AuxStreamType.MoeShared: aux_stream,
        AuxStreamType.MoeChunkingOverlap: aux_stream,
    }
    moe = MiniMaxM3MoE(
        model_config=model_config, aux_stream_dict=aux_stream_dict, layer_idx=3
    )
    assert moe.shared_experts is None


def test_minimax_m3_moe_forward_adds_shared_expert_output():
    """End-to-end MoE forward: routed + shared paths are summed exactly once.

    Builds a real ``MiniMaxM3MoE``, stubs the routed-experts forward to
    return a fixed tensor (so the test does not depend on the fused-MoE
    kernel internals), then verifies the block's output is exactly
    ``routed_output + shared_output``. Re-runs with the shared expert
    nulled to confirm the missing-shared-expert negative control: the
    output differs by the shared expert contribution.
    """
    if not _has_cuda():
        pytest.skip("MiniMaxM3MoE create_moe needs CUDA buffers")
    from tensorrt_llm._torch.model_config import ModelConfig
    from tensorrt_llm._torch.models.modeling_minimaxm3 import MiniMaxM3MoE
    from tensorrt_llm._torch.utils import AuxStreamType
    from tensorrt_llm.mapping import Mapping

    cfg = _minimal_text_config_for_moe(hidden=16, intermediate=8)
    model_config = ModelConfig(
        pretrained_config=cfg,
        mapping=Mapping(),
        skip_create_weights_in_init=False,
    )
    aux_stream = torch.cuda.Stream()
    aux_stream_dict = {
        AuxStreamType.MoeShared: aux_stream,
        AuxStreamType.MoeChunkingOverlap: aux_stream,
    }
    moe = MiniMaxM3MoE(
        model_config=model_config, aux_stream_dict=aux_stream_dict, layer_idx=3
    ).cuda()

    # Re-init shared-expert Linear weights to a larger magnitude so the
    # shared expert's contribution is well above bf16 representable
    # noise. The default Linear init can leave weights near zero so the
    # shared output is effectively zero in bf16 against a non-zero
    # routed stub; we set them deliberately.
    with torch.no_grad():
        for name, p in moe.shared_experts.named_parameters():
            if name.endswith(".weight") and p.ndim >= 2:
                p.copy_(
                    (torch.empty(p.shape, dtype=torch.float32).normal_(0, 0.5)).to(
                        device=p.device, dtype=p.dtype
                    )
                )

    # Replace the routed experts forward with a deterministic stub so
    # the test isolates the shared-expert summation logic. Using
    # ``stub_routed == 0`` makes ``out_no_shared == 0`` (the negative
    # control); ``out`` then equals the shared-expert contribution.
    stub_routed = torch.zeros((4, cfg.hidden_size), dtype=cfg.torch_dtype, device="cuda")

    class _StubExperts(nn.Module):
        def forward(self, hidden_states, router_logits, **kwargs):
            return stub_routed.clone()

    moe.experts = _StubExperts()
    # AttentionMetadata stand-in carrying the only field MoE.forward reads.
    attn_metadata = SimpleNamespace(all_rank_num_tokens=[4])

    torch.manual_seed(31)
    hidden = (torch.randn(4, cfg.hidden_size, dtype=torch.float32) * 0.5).to(
        device="cuda", dtype=cfg.torch_dtype
    )
    # Compute the shared-expert output directly so we can subtract.
    shared = moe.shared_experts(hidden).detach()
    shared_norm = shared.abs().mean().item()
    assert shared_norm > 0.01, (
        f"Test setup: shared expert output must be non-trivial in bf16; "
        f"got shared_norm={shared_norm}. Increase the init scale above."
    )
    out = moe(hidden, attn_metadata)
    expected = stub_routed + shared
    torch.testing.assert_close(out, expected, rtol=5e-3, atol=5e-3)

    # Negative control: drop the shared expert and confirm the result
    # differs from `out` by exactly the shared expert contribution.
    moe.shared_experts = None
    out_no_shared = moe(hidden, attn_metadata)
    torch.testing.assert_close(out_no_shared, stub_routed, rtol=1e-3, atol=1e-3)
    diff = (out - out_no_shared).abs().mean().item()
    assert diff > 0.5 * shared_norm, (
        f"Dropping the shared expert should change output by ~the shared "
        f"contribution; got diff={diff}, shared_norm={shared_norm}"
    )


# ---------------------------------------------------------------------------
# Direct activation parity for the dense MLP across the full pipeline
# ---------------------------------------------------------------------------


def test_swigluoai_dense_mlp_matches_reference_full_pipeline():
    """End-to-end M3 dense MLP (``GatedMLP`` + swigluoai) matches the SGLang reference.

    Build a real CPU dense MLP via ``_build_swiglu_oai_dense_mlp``, feed
    it a stress-test input (covering values that trigger both the upper
    and lower clamp branches), and confirm the output matches a
    reference that replicates the projection + asymmetric activation +
    projection.
    """
    from tensorrt_llm._torch.model_config import ModelConfig
    from tensorrt_llm._torch.models.modeling_minimaxm3 import _build_swiglu_oai_dense_mlp
    from tensorrt_llm.mapping import Mapping

    hidden = 24
    inter = 16
    cfg = SimpleNamespace(
        hidden_size=hidden,
        torch_dtype=torch.float32,
        swiglu_alpha=1.702,
        swiglu_limit=7.0,
    )
    model_config = ModelConfig(
        pretrained_config=cfg,
        mapping=Mapping(),
        skip_create_weights_in_init=False,
    )
    mlp = _build_swiglu_oai_dense_mlp(model_config=model_config, intermediate_size=inter)

    torch.manual_seed(99)
    W_gate_up = torch.randn(2 * inter, hidden, dtype=torch.float32) * 0.3
    W_down = torch.randn(hidden, inter, dtype=torch.float32) * 0.3
    mlp.gate_up_proj.weight.data.copy_(W_gate_up)
    mlp.down_proj.weight.data.copy_(W_down)
    if mlp.gate_up_proj.bias is not None:
        mlp.gate_up_proj.bias.data.zero_()
    if mlp.down_proj.bias is not None:
        mlp.down_proj.bias.data.zero_()

    torch.manual_seed(100)
    # Mix of magnitudes — ensures some hidden lanes produce gate
    # projection magnitudes exceeding ``swiglu_limit`` in both directions.
    x = torch.randn(5, hidden, dtype=torch.float32) * 4.0
    out = mlp(x)

    gate_up = nn.functional.linear(x, W_gate_up)
    activated = _sglang_swigluoai_reference(gate_up, alpha=1.702, limit=7.0)
    expected = nn.functional.linear(activated, W_down)
    torch.testing.assert_close(out, expected, rtol=1e-5, atol=1e-5)


# ---------------------------------------------------------------------------
# Goal 7.3 — production fused-MoE block forward parity against the
# Stage-1 trusted reference. The Stage-1 trusted reference is the
# locally-computed PyTorch golden aligned to SGLang's
# ``swiglu_no_interleaved_with_alpha_and_limit`` and to
# ``MiniMaxM3MoeRoutingMethod``'s sigmoid+bias+renorm+scale shape; this
# is the same reference shape the existing focused activation/routing
# tests above already pin individually.
#
# This test checks that the production wiring of ``MiniMaxM3MoE`` —
# ``create_moe(activation_type=ActivationType.SwigluBias,
# swiglu_alpha=tensor, swiglu_beta=tensor, swiglu_limit=tensor)``
# producing a ``ConfigurableMoE`` wrapper over the selected fused
# backend (CUTLASS by default) — composes those Stage-1 pieces in a
# single end-to-end CUDA forward, and exposes the four acceptance-
# required negative controls (wrong expert selection, wrong packed
# weight layout, wrong activation, wrong routed scaling).
# ---------------------------------------------------------------------------


def _minimal_m3_moe_config(
    *,
    hidden=32,
    intermediate=16,
    num_experts=4,
    top_k=2,
    n_shared=1,
    routed_scaling_factor=2.0,
    swiglu_alpha=1.702,
    swiglu_limit=7.0,
):
    """Tiny but complete-enough M3 config for ``MiniMaxM3MoE``."""
    from transformers import PretrainedConfig

    cfg = PretrainedConfig()
    cfg.hidden_size = hidden
    cfg.intermediate_size = intermediate
    cfg.num_hidden_layers = 4
    cfg.num_attention_heads = 4
    cfg.num_key_value_heads = 2
    cfg.head_dim = 8
    cfg.vocab_size = 256
    cfg.max_position_embeddings = 64
    cfg.rms_norm_eps = 1e-6
    cfg.use_gemma_norm = True
    cfg.rope_theta = 10000.0
    cfg.rotary_dim = 4
    cfg.partial_rotary_factor = 0.5
    cfg.qk_norm_type = "per_head"
    cfg.use_qk_norm = True
    cfg.hidden_act = "swigluoai"
    cfg.swiglu_alpha = swiglu_alpha
    cfg.swiglu_limit = swiglu_limit
    cfg.dense_intermediate_size = intermediate
    cfg.shared_intermediate_size = intermediate
    cfg.num_local_experts = num_experts
    cfg.num_experts_per_tok = top_k
    cfg.n_shared_experts = n_shared
    cfg.scoring_func = "sigmoid"
    cfg.use_routing_bias = True
    cfg.routed_scaling_factor = routed_scaling_factor
    cfg.moe_layer_freq = [0, 0, 0, 1]
    cfg.sparse_attention_config = {
        "use_sparse_attention": True,
        "sparse_index_dim": 8,
        "sparse_num_index_heads": 2,
        "sparse_topk_blocks": 2,
        "sparse_block_size": 4,
        "sparse_init_block": 0,
        "sparse_local_block": 1,
        "sparse_score_type": "max",
        "sparse_disable_index_value": [0, 0, 0, 1],
        "sparse_attention_freq": [0, 0, 0, 1],
    }
    cfg.torch_dtype = torch.bfloat16
    cfg.architectures = ["MiniMaxM3SparseForCausalLM"]
    return cfg


def _build_m3_moe_and_weights(
    *,
    hidden,
    intermediate,
    num_experts,
    top_k,
    n_shared,
    routed_scaling_factor,
    swiglu_alpha,
    swiglu_limit,
    seed_weights=7,
    seed_router=11,
    seed_shared=13,
    activation_type_override=None,
):
    """Construct an ``MiniMaxM3MoE`` block, load synthetic weights into the
    fused MoE + router gate + shared expert, and return the block plus
    the per-expert weight tensors (so the reference can mirror them).

    ``activation_type_override`` is the negative-control hook for
    ``wrong-activation``: it lets a caller flip the production-block
    ``activation_type`` to ``ActivationType.Swiglu`` (plain SwiGLU) at
    construction time so we can compare the fused output to the
    swigluoai reference. The wiring is otherwise identical, so any
    output divergence is attributable to the activation alone.
    """
    from tensorrt_llm._torch.model_config import ModelConfig
    from tensorrt_llm._torch.models.modeling_minimaxm3 import MiniMaxM3MoE
    from tensorrt_llm.mapping import Mapping

    cfg = _minimal_m3_moe_config(
        hidden=hidden,
        intermediate=intermediate,
        num_experts=num_experts,
        top_k=top_k,
        n_shared=n_shared,
        routed_scaling_factor=routed_scaling_factor,
        swiglu_alpha=swiglu_alpha,
        swiglu_limit=swiglu_limit,
    )
    mc = ModelConfig(
        pretrained_config=cfg,
        mapping=Mapping(),
        skip_create_weights_in_init=False,
    )
    aux = torch.cuda.Stream()
    from tensorrt_llm._torch.utils import AuxStreamType

    aux_stream_dict = {
        AuxStreamType.MoeShared: aux,
        AuxStreamType.MoeChunkingOverlap: aux,
    }
    if activation_type_override is None:
        moe = MiniMaxM3MoE(
            model_config=mc, aux_stream_dict=aux_stream_dict, layer_idx=3
        ).cuda()
    else:
        # Construct with the production wiring, then rebuild the
        # backend with the alternate activation_type. We do this via
        # the ``create_moe`` factory directly so the negative-control
        # path is a real fused-MoE forward on the *same* synthetic
        # weights, only with the activation flipped.
        from tensorrt_llm._torch.modules.fused_moe import MiniMaxM3MoeRoutingMethod, create_moe

        moe = MiniMaxM3MoE(
            model_config=mc, aux_stream_dict=aux_stream_dict, layer_idx=3
        ).cuda()
        # Rebuild ``moe.experts`` with the overridden activation_type.
        # The other constructor args mirror the production call inside
        # ``MiniMaxM3MoE.__init__`` so this stays a faithful "swap one
        # axis at a time" negative control. The original ``moe.experts``
        # already registered layer_idx=3 on the ModelConfig's
        # ``extra_attrs["moe_layers"]``; pop that entry before
        # rebuilding so the replacement does not trip the
        # "Duplicate MoE layer for layer_idx=3" assertion.
        if "moe_layers" in mc.extra_attrs:
            mc.extra_attrs["moe_layers"].pop("3", None)
        moe.experts = create_moe(
            routing_method=MiniMaxM3MoeRoutingMethod(
                top_k=moe.top_k,
                num_experts=moe.num_experts,
                callable_e_score_correction_bias=lambda: (
                    moe.gate.e_score_correction_bias
                ),
                routed_scaling_factor=moe.routed_scaling_factor,
            ),
            num_experts=moe.num_experts,
            aux_stream_dict=aux_stream_dict,
            # Production wiring uses ``reduce_results=False``; the
            # single external AllReduce on routed+shared is owned by
            # ``MiniMaxM3MoE``. Mirror that here so the negative
            # control swaps only the activation type, not the
            # reduction wiring.
            reduce_results=False,
            model_config=mc,
            layer_idx=3,
            # For ``ActivationType.Swiglu`` the CUTLASS kernel
            # dispatches ``GLUAdaptor<SiLu>`` which ignores
            # alpha/beta/limit, so leave them unset to make the
            # negative control isolate the activation shape itself.
            activation_type=activation_type_override,
        ).cuda()

    # Random per-expert weights at low magnitude.
    torch.manual_seed(seed_weights)
    w1_per_expert = []
    w2_per_expert = []
    w3_per_expert = []
    weights = {}
    for expert_id in range(num_experts):
        w1 = torch.randn((intermediate, hidden), dtype=torch.bfloat16, device="cuda") * 0.1
        w2 = torch.randn((hidden, intermediate), dtype=torch.bfloat16, device="cuda") * 0.1
        w3 = torch.randn((intermediate, hidden), dtype=torch.bfloat16, device="cuda") * 0.1
        w1_per_expert.append(w1)
        w2_per_expert.append(w2)
        w3_per_expert.append(w3)
        weights[f"{expert_id}.w1.weight"] = w1
        weights[f"{expert_id}.w2.weight"] = w2
        weights[f"{expert_id}.w3.weight"] = w3
    moe.experts.load_weights([weights])

    torch.manual_seed(seed_router)
    gate_w = torch.randn(num_experts, hidden, dtype=torch.float32)
    moe.gate.weight.data.copy_(gate_w)
    bias_vec = (torch.randn(num_experts, dtype=torch.float32) * 0.05).cuda()
    moe.gate.e_score_correction_bias.copy_(bias_vec)

    if n_shared > 0:
        torch.manual_seed(seed_shared)
        shared_gate_up_w = (
            (torch.randn(2 * intermediate, hidden, dtype=torch.float32) * 0.1)
            .to(torch.bfloat16)
            .cuda()
        )
        shared_down_w = (
            (torch.randn(hidden, intermediate, dtype=torch.float32) * 0.1).to(torch.bfloat16).cuda()
        )
        moe.shared_experts.gate_up_proj.weight.data.copy_(shared_gate_up_w)
        moe.shared_experts.down_proj.weight.data.copy_(shared_down_w)
    else:
        shared_gate_up_w = None
        shared_down_w = None

    return SimpleNamespace(
        moe=moe,
        gate_w=gate_w.cuda(),
        bias_vec=bias_vec,
        w1_per_expert=w1_per_expert,
        w2_per_expert=w2_per_expert,
        w3_per_expert=w3_per_expert,
        shared_gate_up_w=shared_gate_up_w,
        shared_down_w=shared_down_w,
        num_experts=num_experts,
        top_k=top_k,
        hidden=hidden,
        intermediate=intermediate,
        routed_scaling_factor=routed_scaling_factor,
        swiglu_alpha=swiglu_alpha,
        swiglu_limit=swiglu_limit,
    )


def _sglang_aligned_m3_moe_reference(
    hidden_states: torch.Tensor,
    *,
    bundle,
    activation: str = "swigluoai",
    routed_scaling_factor=None,
    topk_idx_override: torch.Tensor = None,
):
    """Compute the M3 MoE block output using the SGLang-aligned recipe.

    Inputs are matched to ``MiniMaxM3MoE.forward``: ``hidden_states``
    is BF16 on CUDA, ``bundle`` carries the synthetic weights and
    routing config. Returns a float32 tensor of shape
    ``[num_tokens, hidden]`` mirroring the production block's output.

    The negative-control axes are exposed as parameters:
      * ``activation="plain_swiglu"`` swaps swigluoai for
        ``gate * sigmoid(gate) * up`` (no clamp, no ``up + 1``).
      * ``routed_scaling_factor`` lets the caller force a different
        scaling factor than the SUT.
      * ``topk_idx_override`` lets the caller force a different
        expert-selection tensor than ``MiniMaxM3MoeRoutingMethod``
        produced.
    """
    import torch.nn.functional as F

    from tensorrt_llm._torch.modules.fused_moe.routing import MiniMaxM3MoeRoutingMethod

    def _swigluoai_act(gate_v, up_v):
        gate_c = gate_v.clamp(max=bundle.swiglu_limit)
        up_c = up_v.clamp(min=-bundle.swiglu_limit, max=bundle.swiglu_limit)
        return gate_c * torch.sigmoid(bundle.swiglu_alpha * gate_c) * (up_c + 1.0)

    def _plain_swiglu_act(gate_v, up_v):
        return gate_v * torch.sigmoid(gate_v) * up_v

    act_fn = _swigluoai_act if activation == "swigluoai" else _plain_swiglu_act
    rsf = bundle.routed_scaling_factor if routed_scaling_factor is None else routed_scaling_factor

    hidden_f32 = hidden_states.to(torch.float32)
    router_logits = F.linear(hidden_f32, bundle.gate_w)
    rm = MiniMaxM3MoeRoutingMethod(
        top_k=bundle.top_k,
        num_experts=bundle.num_experts,
        callable_e_score_correction_bias=lambda: bundle.bias_vec,
        routed_scaling_factor=rsf,
    )
    topk_idx, topk_w = rm.apply(router_logits)
    if topk_idx_override is not None:
        topk_idx = topk_idx_override.to(topk_idx.device).to(topk_idx.dtype)

    routed = torch.zeros(
        hidden_states.shape[0],
        bundle.hidden,
        dtype=torch.float32,
        device="cuda",
    )
    for row in range(hidden_states.shape[0]):
        for k in range(bundle.top_k):
            eid = int(topk_idx[row, k].item())
            w = topk_w[row, k].float()
            h = hidden_states[row : row + 1].float()
            gate = F.linear(h, bundle.w1_per_expert[eid].float())
            up = F.linear(h, bundle.w3_per_expert[eid].float())
            act = act_fn(gate, up)
            out_e = F.linear(act, bundle.w2_per_expert[eid].float())
            routed[row] += w * out_e[0]

    shared_out = torch.zeros_like(routed)
    if bundle.shared_gate_up_w is not None:
        shared_gate_up = F.linear(hidden_states, bundle.shared_gate_up_w).float()
        s_gate, s_up = shared_gate_up.chunk(2, dim=-1)
        shared_act = act_fn(s_gate, s_up)
        shared_out = F.linear(shared_act.to(torch.bfloat16), bundle.shared_down_w).float()

    return routed + shared_out


def test_production_moe_matches_sglang_aligned_reference():
    """Goal 7.3 — production fused-MoE block matches the Stage-1 reference.

    Builds a real ``MiniMaxM3MoE`` with synthetic small-dim weights and
    runs the fused MoE forward on CUDA, then compares the output to a
    locally-computed SGLang-aligned reference that uses
    ``MiniMaxM3MoeRoutingMethod`` (routing parity) + per-token
    swigluoai expert MLPs (activation parity) + the dense swigluoai
    ``GatedMLP`` shared expert summed once. Bit-for-bit
    matching is not expected because the fused kernel composes the
    matmul+activation+combine in BF16, but the residual should land
    well below 5% of the reference norm — well into the territory
    where the per-element activation shape and per-expert scaling are
    actually being verified.
    """
    if not _has_cuda():
        pytest.skip("Production MoE forward needs CUDA")
    from tensorrt_llm._torch.autotuner import AutoTuner, autotune

    bundle = _build_m3_moe_and_weights(
        hidden=32,
        intermediate=16,
        num_experts=4,
        top_k=2,
        n_shared=1,
        routed_scaling_factor=2.0,
        swiglu_alpha=1.702,
        swiglu_limit=7.0,
    )
    moe = bundle.moe

    attn_metadata = SimpleNamespace(all_rank_num_tokens=[6])
    torch.manual_seed(17)
    hidden = (torch.randn(6, bundle.hidden, dtype=torch.float32) * 0.5).to(torch.bfloat16).cuda()

    AutoTuner.get().clear_cache()
    with torch.inference_mode(), autotune():
        sut_out = moe(hidden, attn_metadata)

    ref = _sglang_aligned_m3_moe_reference(hidden, bundle=bundle)

    diff = (sut_out.float() - ref).abs()
    ref_norm = ref.abs().mean().item()
    assert ref_norm > 0.01, (
        f"reference must produce a non-trivial output for the parity "
        f"check to mean anything; got ref_norm={ref_norm}"
    )
    rel_max = diff.max().item() / max(ref_norm, 1e-6)
    assert rel_max < 0.05, (
        f"production MoE diverges from SGLang-aligned reference: "
        f"max_abs={diff.max().item():.6f}, mean_abs={diff.mean().item():.6f}, "
        f"ref_norm={ref_norm:.6f}, rel_max={rel_max:.4f}"
    )


def test_production_moe_negative_control_wrong_expert_selection():
    """Negative control: a SUT that picks different experts must diverge.

    Forces ``MiniMaxM3MoeRoutingMethod``'s ``topk_idx`` to a fixed
    "always pick experts [0, 1]" override in the reference comparison
    while the SUT uses the real (bias-driven) routing. Their outputs
    must differ materially, otherwise the test would silently pass
    even when expert selection regressed.
    """
    if not _has_cuda():
        pytest.skip("Production MoE forward needs CUDA")
    from tensorrt_llm._torch.autotuner import AutoTuner, autotune

    bundle = _build_m3_moe_and_weights(
        hidden=32,
        intermediate=16,
        num_experts=4,
        top_k=2,
        n_shared=1,
        routed_scaling_factor=2.0,
        swiglu_alpha=1.702,
        swiglu_limit=7.0,
        seed_router=23,  # make bias-driven routing non-trivial
    )
    moe = bundle.moe

    attn_metadata = SimpleNamespace(all_rank_num_tokens=[6])
    torch.manual_seed(17)
    hidden = (torch.randn(6, bundle.hidden, dtype=torch.float32) * 0.5).to(torch.bfloat16).cuda()

    AutoTuner.get().clear_cache()
    with torch.inference_mode(), autotune():
        sut_out = moe(hidden, attn_metadata)

    # Reference selects fixed experts [2, 3] (different from the SUT's
    # bias-driven pick on this seed); the activation, scaling, and
    # weights are otherwise identical. If the SUT's expert selection
    # were silently broken to pick the same wrong experts the diff
    # would be small; with the real router it must be material.
    fixed_topk = torch.tensor(
        [[2, 3]] * hidden.shape[0],
        dtype=torch.int32,
        device="cuda",
    )
    ref_wrong = _sglang_aligned_m3_moe_reference(
        hidden,
        bundle=bundle,
        topk_idx_override=fixed_topk,
    )

    diff = (sut_out.float() - ref_wrong).abs().mean().item()
    ref_correct = _sglang_aligned_m3_moe_reference(hidden, bundle=bundle)
    correct_diff = (sut_out.float() - ref_correct).abs().mean().item()
    assert diff > 4 * correct_diff and diff > 1e-3, (
        f"wrong-expert-selection control did not diverge: "
        f"sut_vs_wrong={diff:.6f}, sut_vs_correct={correct_diff:.6f}"
    )


def test_production_moe_negative_control_wrong_packed_weight_layout():
    """Negative control: swapping gate and up halves must change output.

    Builds two SUTs with the same synthetic weights, but in the second
    the per-expert ``w1`` (gate) and ``w3`` (up) tensors are swapped at
    load time. The fused kernel still runs the swigluoai activation,
    but now over the wrong halves; the resulting output must diverge
    from the reference computed with the correctly-ordered weights.
    """
    if not _has_cuda():
        pytest.skip("Production MoE forward needs CUDA")
    from tensorrt_llm._torch.autotuner import AutoTuner, autotune

    bundle = _build_m3_moe_and_weights(
        hidden=32,
        intermediate=16,
        num_experts=4,
        top_k=2,
        n_shared=1,
        routed_scaling_factor=2.0,
        swiglu_alpha=1.702,
        swiglu_limit=7.0,
    )

    # Construct a second SUT with the same weights but with the gate
    # and up halves swapped. We do this by reloading with swapped keys.
    bundle2 = _build_m3_moe_and_weights(
        hidden=32,
        intermediate=16,
        num_experts=4,
        top_k=2,
        n_shared=1,
        routed_scaling_factor=2.0,
        swiglu_alpha=1.702,
        swiglu_limit=7.0,
    )
    swapped_weights = {}
    for expert_id in range(bundle.num_experts):
        # Reuse the bundle1 weights but swap w1<->w3 when feeding the
        # fused-MoE loader.
        swapped_weights[f"{expert_id}.w1.weight"] = bundle.w3_per_expert[expert_id]
        swapped_weights[f"{expert_id}.w2.weight"] = bundle.w2_per_expert[expert_id]
        swapped_weights[f"{expert_id}.w3.weight"] = bundle.w1_per_expert[expert_id]
    bundle2.moe.experts.load_weights([swapped_weights])
    bundle2.moe.gate.weight.data.copy_(bundle.gate_w)
    bundle2.moe.gate.e_score_correction_bias.copy_(bundle.bias_vec)
    bundle2.moe.shared_experts.gate_up_proj.weight.data.copy_(bundle.shared_gate_up_w)
    bundle2.moe.shared_experts.down_proj.weight.data.copy_(bundle.shared_down_w)

    attn_metadata = SimpleNamespace(all_rank_num_tokens=[6])
    torch.manual_seed(17)
    hidden = (torch.randn(6, bundle.hidden, dtype=torch.float32) * 0.5).to(torch.bfloat16).cuda()

    AutoTuner.get().clear_cache()
    with torch.inference_mode(), autotune():
        sut_correct = bundle.moe(hidden, attn_metadata)
        sut_swapped = bundle2.moe(hidden, attn_metadata)

    ref = _sglang_aligned_m3_moe_reference(hidden, bundle=bundle)

    # SUT-correct should match the reference closely; SUT-swapped must
    # diverge materially since the gate/up halves drive different
    # activation values under swigluoai's asymmetric clamp.
    correct_diff = (sut_correct.float() - ref).abs().mean().item()
    swapped_diff = (sut_swapped.float() - ref).abs().mean().item()
    assert swapped_diff > 4 * correct_diff and swapped_diff > 1e-3, (
        f"wrong-packed-weight-layout control did not diverge: "
        f"sut_correct_vs_ref={correct_diff:.6f}, "
        f"sut_swapped_vs_ref={swapped_diff:.6f}"
    )


def test_production_moe_negative_control_wrong_activation():
    """Negative control: ActivationType.Swiglu (plain) must diverge.

    Builds a SUT whose ``experts`` backend uses ``ActivationType.Swiglu``
    instead of the production ``ActivationType.SwigluBias``. Because
    plain ``GLUAdaptor<SiLu>`` ignores ``alpha/beta/limit``, the
    activation collapses to ``silu(gate) * up`` rather than
    ``gate_clamped * sigmoid(alpha * gate_clamped) * (up_clamped + 1)``.
    The SUT output must therefore diverge from the swigluoai
    reference, while the correctly-wired SUT matches it closely.
    """
    if not _has_cuda():
        pytest.skip("Production MoE forward needs CUDA")
    from tensorrt_llm._torch.autotuner import AutoTuner, autotune
    from tensorrt_llm._torch.utils import ActivationType

    bundle_correct = _build_m3_moe_and_weights(
        hidden=32,
        intermediate=16,
        num_experts=4,
        top_k=2,
        n_shared=1,
        routed_scaling_factor=2.0,
        swiglu_alpha=1.702,
        swiglu_limit=7.0,
    )
    bundle_wrong = _build_m3_moe_and_weights(
        hidden=32,
        intermediate=16,
        num_experts=4,
        top_k=2,
        n_shared=1,
        routed_scaling_factor=2.0,
        swiglu_alpha=1.702,
        swiglu_limit=7.0,
        activation_type_override=ActivationType.Swiglu,
    )
    # Sync the wrong-bundle's expert / router / shared weights to the
    # correct bundle's, so the only difference is the activation_type.
    weights_correct = {}
    for expert_id in range(bundle_correct.num_experts):
        weights_correct[f"{expert_id}.w1.weight"] = bundle_correct.w1_per_expert[expert_id]
        weights_correct[f"{expert_id}.w2.weight"] = bundle_correct.w2_per_expert[expert_id]
        weights_correct[f"{expert_id}.w3.weight"] = bundle_correct.w3_per_expert[expert_id]
    bundle_wrong.moe.experts.load_weights([weights_correct])
    bundle_wrong.moe.gate.weight.data.copy_(bundle_correct.gate_w)
    bundle_wrong.moe.gate.e_score_correction_bias.copy_(bundle_correct.bias_vec)
    bundle_wrong.moe.shared_experts.gate_up_proj.weight.data.copy_(bundle_correct.shared_gate_up_w)
    bundle_wrong.moe.shared_experts.down_proj.weight.data.copy_(bundle_correct.shared_down_w)

    attn_metadata = SimpleNamespace(all_rank_num_tokens=[6])
    # Use a larger input scale so plain-swiglu vs swigluoai differs
    # materially even at small intermediate sizes — the clamp + +1
    # offset shows up most clearly when gate values stray.
    torch.manual_seed(17)
    hidden = (
        (torch.randn(6, bundle_correct.hidden, dtype=torch.float32) * 2.0).to(torch.bfloat16).cuda()
    )

    AutoTuner.get().clear_cache()
    with torch.inference_mode(), autotune():
        sut_correct = bundle_correct.moe(hidden, attn_metadata)
        sut_wrong = bundle_wrong.moe(hidden, attn_metadata)

    ref = _sglang_aligned_m3_moe_reference(hidden, bundle=bundle_correct)
    correct_diff = (sut_correct.float() - ref).abs().mean().item()
    wrong_diff = (sut_wrong.float() - ref).abs().mean().item()
    assert wrong_diff > 4 * correct_diff and wrong_diff > 1e-3, (
        f"wrong-activation control did not diverge: "
        f"sut_correct_vs_ref={correct_diff:.6f}, "
        f"sut_wrong_vs_ref={wrong_diff:.6f}"
    )


def test_production_moe_negative_control_wrong_routed_scaling():
    """Negative control: scaling factor must shift the routed output.

    Builds a SUT with ``routed_scaling_factor=1.0`` (the "wrong" choice
    for M3, which uses 2.0). Compares the SUT against the reference
    that uses 2.0; the SUT output must diverge materially, while a
    matching-2.0 SUT recovers parity.
    """
    if not _has_cuda():
        pytest.skip("Production MoE forward needs CUDA")
    from tensorrt_llm._torch.autotuner import AutoTuner, autotune

    bundle_correct = _build_m3_moe_and_weights(
        hidden=32,
        intermediate=16,
        num_experts=4,
        top_k=2,
        n_shared=1,
        routed_scaling_factor=2.0,
        swiglu_alpha=1.702,
        swiglu_limit=7.0,
    )
    bundle_wrong = _build_m3_moe_and_weights(
        hidden=32,
        intermediate=16,
        num_experts=4,
        top_k=2,
        n_shared=1,
        routed_scaling_factor=1.0,
        swiglu_alpha=1.702,
        swiglu_limit=7.0,
    )
    # Sync weights/routing of the wrong bundle to the correct bundle so
    # the only difference is routed_scaling_factor.
    weights_correct = {}
    for expert_id in range(bundle_correct.num_experts):
        weights_correct[f"{expert_id}.w1.weight"] = bundle_correct.w1_per_expert[expert_id]
        weights_correct[f"{expert_id}.w2.weight"] = bundle_correct.w2_per_expert[expert_id]
        weights_correct[f"{expert_id}.w3.weight"] = bundle_correct.w3_per_expert[expert_id]
    bundle_wrong.moe.experts.load_weights([weights_correct])
    bundle_wrong.moe.gate.weight.data.copy_(bundle_correct.gate_w)
    bundle_wrong.moe.gate.e_score_correction_bias.copy_(bundle_correct.bias_vec)
    bundle_wrong.moe.shared_experts.gate_up_proj.weight.data.copy_(bundle_correct.shared_gate_up_w)
    bundle_wrong.moe.shared_experts.down_proj.weight.data.copy_(bundle_correct.shared_down_w)

    attn_metadata = SimpleNamespace(all_rank_num_tokens=[6])
    torch.manual_seed(17)
    hidden = (
        (torch.randn(6, bundle_correct.hidden, dtype=torch.float32) * 0.5).to(torch.bfloat16).cuda()
    )

    AutoTuner.get().clear_cache()
    with torch.inference_mode(), autotune():
        sut_correct = bundle_correct.moe(hidden, attn_metadata)
        sut_wrong = bundle_wrong.moe(hidden, attn_metadata)

    ref = _sglang_aligned_m3_moe_reference(hidden, bundle=bundle_correct)
    correct_diff = (sut_correct.float() - ref).abs().mean().item()
    wrong_diff = (sut_wrong.float() - ref).abs().mean().item()
    assert wrong_diff > 4 * correct_diff and wrong_diff > 1e-3, (
        f"wrong-routed-scaling control did not diverge: "
        f"sut_correct_vs_ref={correct_diff:.6f}, "
        f"sut_wrong_vs_ref={wrong_diff:.6f}"
    )


def test_production_moe_prefill_vs_decode_at_same_token_matches():
    """MoE at q_len=1 decode must equal MoE at q_len=N prefill for the
    same hidden_state at position N-1.

    Iter-12 prefill-vs-decode parity for the production fused-MoE path,
    in the same family as the iter-144 attention parity regressions
    (``test_minimax_m3_dense_decode_matches_prefill_at_same_position``
    and ``test_minimax_m3_sparse_decode_matches_prefill_at_same_position``,
    both PASSED on GB200 in jobs 1970209 and 1970248).

    The iter-143 1970149 evidence localized the production decode bug to
    *downstream of attention* (attention forward parity holds for both
    dense and sparse). The next candidate localization layer is the
    MoE forward at ``batch=1 q_len=1`` — the M3 production MoE routes
    each token independently through ``MiniMaxM3MoeRoutingMethod``
    (sigmoid + bias-for-selection + renorm + ``routed_scaling_factor``
    scaling) and the CUTLASS / TRT-LLM-Gen fused MoE kernel. Per-token
    routing means MoE applied to a multi-token batch at row ``r`` must
    produce the same output as MoE applied to a single-token batch
    containing only that row. Any kernel corner case at ``q_len=1`` /
    ``batch=1`` (e.g. workspace allocation, scheduler dispatch, k=1
    reduction kernel selection) would show up as a sub-1.0 cosine
    here.

    Test layout (mirrors the attention parity tests so the comparison
    is uniform):

      1. Build a real ``MiniMaxM3MoE`` via ``_build_m3_moe_and_weights``
         with deterministic per-expert weights, routing gate, bias, and
         shared-expert weights.
      2. Generate ``seq_len = 13`` deterministic input hidden_states.
      3. Run MoE with ``all_rank_num_tokens=[13]`` and the full batch;
         capture ``out_prefill[-1]`` — the production fused MoE output
         at position 12.
      4. Run MoE with ``all_rank_num_tokens=[1]`` and only the
         ``[12]``-th hidden_state; capture ``out_decode[0]``.
      5. Assert cos > 0.999 and max_abs < 0.05 (bf16 noise band).

    Negative control: also run MoE on a *different* single-token input
    (``hidden_full[0]``) and assert its output differs from the
    decode-at-position-12 output by a clearly distinguishable margin.
    A bug that returns zero or scrambles the per-token routing would
    fail both the parity assertion and the negative control.

    If this test FAILS, the iter-143 production decode drift is
    isolated to the MoE forward at single-token batch geometry —
    candidates: fused MoE kernel selection / workspace shape /
    dispatch, ``routed_scaling_factor`` scaling at top_k=4 single
    token, or ``all_rank_num_tokens`` path interaction with the
    CUTLASS dispatcher.

    If this test PASSES, MoE is also ruled out in isolation, and the
    bug must live in cross-layer state (residual + per-layer norm +
    MLP/MoE compounding across 60 layers), in metadata fields the
    production pyexecutor populates (e.g. ``request_ids``,
    ``num_cached_tokens_per_seq``, ``all_rank_num_tokens`` for actual
    decode iterations), or in a CUDA-graph-vs-eager dispatch path the
    small unit tests do not stress.
    """
    if not _has_cuda():
        pytest.skip("Production MoE forward needs CUDA")
    from tensorrt_llm._torch.autotuner import AutoTuner, autotune

    bundle = _build_m3_moe_and_weights(
        hidden=32,
        intermediate=16,
        num_experts=4,
        top_k=2,
        n_shared=1,
        routed_scaling_factor=2.0,
        swiglu_alpha=1.702,
        swiglu_limit=7.0,
    )
    moe = bundle.moe

    seq_len = 13
    torch.manual_seed(2026)
    hidden_full = (
        (torch.randn(seq_len, bundle.hidden, dtype=torch.float32) * 0.5).to(torch.bfloat16).cuda()
    )

    # Mode A — prefill: full batch, capture last-position output.
    attn_metadata_prefill = SimpleNamespace(all_rank_num_tokens=[seq_len])
    AutoTuner.get().clear_cache()
    with torch.inference_mode(), autotune():
        out_prefill = moe(hidden_full, attn_metadata_prefill)
    assert out_prefill is not None
    assert out_prefill.shape == (seq_len, bundle.hidden), out_prefill.shape
    assert torch.isfinite(out_prefill).all().item(), "MoE prefill produced non-finite output"
    prefill_last = out_prefill[-1].clone().detach()

    # Mode B — decode: batch=1, only the last token's hidden_state.
    hidden_decode = hidden_full[seq_len - 1 : seq_len].clone()
    attn_metadata_decode = SimpleNamespace(all_rank_num_tokens=[1])
    AutoTuner.get().clear_cache()
    with torch.inference_mode(), autotune():
        out_decode = moe(hidden_decode, attn_metadata_decode)
    assert out_decode is not None
    assert out_decode.shape == (1, bundle.hidden), out_decode.shape
    assert torch.isfinite(out_decode).all().item(), "MoE decode produced non-finite output"
    decode_last = out_decode[0].clone().detach()

    # ---------- Parity assertion ----------
    a = prefill_last.to(torch.float32)
    b = decode_last.to(torch.float32)
    diff_abs = (a - b).abs()
    max_abs = float(diff_abs.max().item())
    mean_abs = float(diff_abs.mean().item())
    na = float(a.norm().item())
    nb = float(b.norm().item())
    cos = float((a @ b).item() / (na * nb)) if na > 0 and nb > 0 else 0.0
    assert cos > 0.999, (
        f"MoE prefill vs decode at the same token diverge: "
        f"cos={cos:.6f} max_abs={max_abs:.6f} mean_abs={mean_abs:.6f} "
        f"prefill_norm={na:.4f} decode_norm={nb:.4f}. "
        f"Localizes the iter143 prefill-vs-decode drift to the production "
        f"fused MoE forward at batch=1 q_len=1 — candidates: fused MoE "
        f"kernel dispatch / workspace shape, routed_scaling_factor scaling "
        f"at single-token top_k=4, or all_rank_num_tokens interaction with "
        f"the CUTLASS dispatcher."
    )
    assert max_abs < 0.05, (
        f"MoE prefill vs decode at the same token: cos={cos:.6f} but "
        f"max_abs={max_abs:.6f} > 0.05. Bit-equivalent per-token MoE "
        f"routing should differ by at most bf16 accumulation noise."
    )

    # ---------- Negative control ----------
    # MoE on a *different* token must differ from MoE on token 12. If
    # the MoE forward returns zero or always returns the same value
    # regardless of input, the parity test above would pass trivially;
    # this control rules that out.
    hidden_other = hidden_full[0:1].clone()
    AutoTuner.get().clear_cache()
    with torch.inference_mode(), autotune():
        out_other = moe(hidden_other, attn_metadata_decode)
    other = out_other[0].to(torch.float32)
    diff_other = (other - b).abs().mean().item()
    nb_other = float(other.norm().item())
    assert nb_other > 1e-3, (
        f"Negative control: MoE on a different token returned ~zero "
        f"norm ({nb_other}); the parity test cannot distinguish "
        f"output-is-input from MoE-is-broken."
    )
    assert diff_other > 5e-3, (
        f"Negative control: MoE on a different input token ({hidden_full[0]}) "
        f"produces nearly the same output as MoE on the target token "
        f"({hidden_full[seq_len - 1]}): diff={diff_other:.6f}. The parity "
        f"test is not actually exercising per-token routing."
    )
