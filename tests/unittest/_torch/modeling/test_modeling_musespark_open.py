# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Op-contract tests for the MuseSparkOpen decoder.

Each test pins one contract the source model defines, against a golden
transcribed from the source formula rather than from TRT-LLM's own modules, so
a shared helper cannot make both sides agree on the wrong answer. Shapes are
small; correctness at checkpoint scale is the integration file's job.
"""

import math
from unittest.mock import patch

import pytest
import torch

from tensorrt_llm._torch.configs.musespark_open import (
    FULL_ATTENTION,
    SLIDING_ATTENTION,
    MuseSparkOpenConfig,
    layer_uses_rope,
    layer_window,
)
from tensorrt_llm._torch.models.checkpoints.hf.musespark_open_weight_mapper import (
    classify_keys,
    missing_in_range_keys,
)
from tensorrt_llm._torch.models.modeling_musespark_open import (
    MuseSparkOpenForCausalLM,
    MuseSparkOpenGatedResidual,
    MuseSparkOpenLogitsProcessor,
    MuseSparkOpenRMSNormWeightFirst,
    MuseSparkOpenRoutingMethod,
    _routing_weights,
    _routing_weights_kernel,
    assert_supported_spec_config,
    head_interleaved_q_gate,
    normalize_exclude_modules,
    post_expert_norm_combine,
    post_expert_norm_rows,
)
from tensorrt_llm.models.modeling_utils import QuantConfig

DEVICE = "cuda"


def make_config(num_hidden_layers=62, **overrides):
    kwargs = dict(
        num_hidden_layers=num_hidden_layers,
        moe={
            "topk": 8,
            "num_experts": 256,
            "override_routed_hidden_dim": 4096,
            "compression_factor": 2,
            "router_score_division_eps": 1e-15,
        },
        metap={"base_width": 256.0, "m_emb": 1.0},
    )
    kwargs.update(overrides)
    return MuseSparkOpenConfig(**kwargs)


# --------------------------------------------------------------------------
# Attention schedule
# --------------------------------------------------------------------------
def test_layer_schedule_matches_source_formula():
    """NoPE layers are exactly the full-attention layers, at the backward-
    counted positions the source derives from num_hidden_layers."""
    config = make_config(62)
    nope = [i for i, r in enumerate(config.use_rope_per_layer) if not r]
    assert nope == list(range(1, 62, 4))
    assert len(nope) == 16
    for idx, layer_type in enumerate(config.layer_types):
        expected = FULL_ATTENTION if idx in nope else SLIDING_ATTENTION
        assert layer_type == expected, idx
    assert config.sliding_window == 2048


def test_layer_schedule_recomputes_for_a_truncated_config():
    """The schedule is a function of num_hidden_layers, so a 4-layer config
    puts its single NoPE layer at index 3, not at index 1. Both the source
    plugin and TRT-LLM derive it the same way from the same config, so a
    truncated-checkpoint comparison stays apples to apples."""
    config = make_config(4)
    assert [i for i, r in enumerate(config.use_rope_per_layer) if not r] == [3]
    assert config.layer_types == [SLIDING_ATTENTION] * 3 + [FULL_ATTENTION]


@pytest.mark.parametrize("num_layers", [4, 8, 62, 63])
def test_layer_helpers_agree_with_the_config(num_layers):
    config = make_config(num_layers)
    for idx in range(num_layers):
        assert layer_uses_rope(idx, num_layers, 4) == config.use_rope_per_layer[idx]
        window = layer_window(idx, num_layers, 4, [2048, 2048, 2048, 0])
        assert (window is None) == (config.layer_types[idx] == FULL_ATTENTION)


def test_query_pre_scale_constant():
    config = make_config()
    assert config.scale_query_by == pytest.approx(43.7840518911 / math.sqrt(64))
    # 43.784... = 3.87 * sqrt(128); the point of the constant is that the max
    # logit is head_dim-independent.
    assert config.scale_query_by == pytest.approx(5.4730064, rel=1e-6)


def test_metap_multiplier():
    config = make_config()
    assert config.metap_multiplier == pytest.approx(32.0**-0.5)


def test_mismatched_window_sizes_are_rejected():
    with pytest.raises(ValueError, match="one sliding-window size"):
        make_config(62, global_attn_cfg=[2048, 1024, 2048, 0])


# --------------------------------------------------------------------------
# Norms and residual
# --------------------------------------------------------------------------
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_weight_first_rmsnorm_matches_the_source_form(dtype):
    """post_expert_norm / post_ffn_norm scale *before* normalizing and apply
    no output gain, which is a different function from (1+w)*rmsnorm(x)."""
    torch.manual_seed(0)
    hidden, eps = 128, 1e-8
    module = MuseSparkOpenRMSNormWeightFirst(hidden, eps, dtype).to(DEVICE)
    module.weight.data.normal_(0, 0.5)
    x = torch.randn(7, hidden, dtype=dtype, device=DEVICE)

    scaled = ((module.weight + 1) * x).to(torch.float32)
    golden = (scaled * torch.rsqrt(scaled.pow(2).mean(-1, keepdim=True) + eps)).to(dtype)
    torch.testing.assert_close(module(x), golden, atol=0, rtol=0)

    x32 = x.to(torch.float32)
    gain_after = (
        x32
        * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + eps)
        * (module.weight.to(torch.float32) + 1)
    )
    assert not torch.allclose(golden.float(), gain_after, atol=1e-2)


def test_gated_residual_is_norm_preserving():
    torch.manual_seed(0)
    hidden = 64
    module = MuseSparkOpenGatedResidual(hidden, torch.bfloat16).to(DEVICE)
    module.gate.data.normal_(0, 1.0)
    alpha, beta = module.coefficients()

    scaled = module.gate.to(torch.float32) / 0.3
    expected_beta = torch.sigmoid(scaled)
    expected_alpha = torch.sqrt(torch.clamp(torch.sigmoid(-scaled) * (1 + expected_beta), min=1e-3))
    torch.testing.assert_close(beta, expected_beta)
    torch.testing.assert_close(alpha, expected_alpha)
    # alpha^2 + beta^2 == 1 away from the clamp.
    unclamped = torch.sigmoid(-scaled) * (1 + expected_beta) > 1e-3
    torch.testing.assert_close(
        (alpha[unclamped] ** 2 + beta[unclamped] ** 2),
        torch.ones_like(alpha[unclamped]),
        atol=1e-5,
        rtol=1e-5,
    )

    h = torch.randn(3, hidden, dtype=torch.float32, device=DEVICE)
    y = torch.randn(3, hidden, dtype=torch.bfloat16, device=DEVICE)
    out = module(h, y)
    assert out.dtype == torch.float32
    torch.testing.assert_close(out, alpha * h + beta * y.float())
    # Not the ungated form.
    assert not torch.allclose(out, h + y.float(), atol=1e-3)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_frozen_constant_cache_is_bit_exact(dtype):
    """Caching the frozen-gate constants must not move a single bit.

    ``coefficients()`` and ``(weight + 1)`` read only ``requires_grad=False``
    parameters, so they are recomputed identically on every call -- 124 and 62
    times per decode step. ``cache_derived_state()`` hoists them out of the
    hot path, and the whole justification for doing so is that it changes *when*
    the ops run and not *what* they compute. This asserts that at zero
    tolerance, on the module outputs and not only on the cached vectors, so a
    regression shows up here in seconds instead of in a parity gate hours later.
    """
    torch.manual_seed(0)
    hidden = 128

    gated = MuseSparkOpenGatedResidual(hidden, dtype).to(DEVICE)
    gated.gate.data.normal_(0, 1.0)
    h = torch.randn(5, hidden, dtype=torch.float32, device=DEVICE)
    y = torch.randn(5, hidden, dtype=dtype, device=DEVICE)
    uncached_alpha, uncached_beta = gated.coefficients()
    uncached_out = gated(h, y)

    gated.cache_derived_state()
    assert gated._alpha is not None and gated._beta is not None
    cached_alpha, cached_beta = gated.coefficients()
    torch.testing.assert_close(cached_alpha, uncached_alpha, atol=0, rtol=0)
    torch.testing.assert_close(cached_beta, uncached_beta, atol=0, rtol=0)
    torch.testing.assert_close(gated(h, y), uncached_out, atol=0, rtol=0)

    norm = MuseSparkOpenRMSNormWeightFirst(hidden, 1e-8, dtype).to(DEVICE)
    norm.weight.data.normal_(0, 0.5)
    x = torch.randn(5, hidden, dtype=dtype, device=DEVICE)
    uncached_norm = norm(x)
    norm.cache_derived_state()
    assert norm._weight_plus_one is not None
    torch.testing.assert_close(norm(x), uncached_norm, atol=0, rtol=0)

    # Derived, so it must never reach a checkpoint, and it lives on the same
    # device as the parameter it came from. What happens to it across a module
    # move is NOT tested here -- an earlier version of this comment claimed the
    # cache "must follow the module across a device/dtype move" and asserted
    # nothing of the kind, while the module performed no move at all. It must in
    # fact be dropped on a move; test_frozen_constant_cache_is_dropped_by_a_dtype_move
    # performs the move and checks that.
    assert not any(k.startswith("_alpha") for k in gated.state_dict())
    assert not any(k.startswith("_weight_plus_one") for k in norm.state_dict())
    assert dict(gated.named_buffers())["_alpha"].device == gated.gate.device

    # A gate rewritten after the cache was filled must not be silently ignored:
    # the refresh hook is what makes a second load_weights correct.
    gated.gate.data.normal_(0, 2.0)
    gated.cache_derived_state()
    torch.testing.assert_close(
        gated.coefficients()[0], gated._compute_coefficients()[0], atol=0, rtol=0
    )


def _tiny_cached_model(hidden=32):
    class TinyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.norm = MuseSparkOpenRMSNormWeightFirst(hidden, 1e-8, torch.float32)
            self.gated = MuseSparkOpenGatedResidual(hidden, torch.float32)

    model = TinyModel().to(DEVICE)
    model.norm.weight.data.normal_(0, 0.5)
    model.gated.gate.data.normal_(0, 1.0)
    assert model.norm._weight_plus_one is None and model.gated._alpha is None
    return model


@pytest.mark.parametrize("stage", ["post_load_weights", "cache_derived_state"])
def test_frozen_constant_cache_fills_from_either_loader_walk(stage):
    """Both loader walks must reach both cache owners.

    ``ModelLoader`` has two post-load walks and they look up *different* names
    (``pyexecutor/model_loader.py``):

    * ``_walk_full_post_load`` looks up ``post_load_weights``. This is the
      branch the ordinary load takes.
    * ``_walk_cache_state`` looks up ``cache_derived_state``. This is the branch
      the GPU-memory-service read-only reader and the staged-receiver path take;
      an RO rank never calls the MuseSparkOpen weight mapper at all, it zero-copy
      imports weights another rank committed and then runs exactly this stage.

    Both skip any module lacking the name they look up, so implementing only one
    of the two leaves the cache empty on the other path -- silently, since the
    forward just falls back to recomputing per call. That is the whole failure
    mode this cache exists to remove, reappearing on some ranks and not others.

    Parametrized over the two stage names, walking by ``getattr`` exactly as the
    loader does, so dropping either hook fails here rather than in a profile.
    """
    torch.manual_seed(0)
    model = _tiny_cached_model()

    visited = 0
    for module in model.modules():
        hook = getattr(module, stage, None)
        if hook is not None and not getattr(module, "_weights_removed", False):
            hook()
            visited += 1
    assert visited == 2, f"{stage} walk reached {visited} cache owners, expected 2"
    assert model.norm._weight_plus_one is not None
    assert model.gated._alpha is not None and model.gated._beta is not None

    # And the stage must be idempotent, as its contract requires.
    alpha_before = model.gated._alpha.clone()
    gain_before = model.norm._weight_plus_one.clone()
    getattr(model.gated, stage)()
    getattr(model.norm, stage)()
    torch.testing.assert_close(model.gated._alpha, alpha_before, atol=0, rtol=0)
    torch.testing.assert_close(model.norm._weight_plus_one, gain_before, atol=0, rtol=0)


@pytest.mark.parametrize("stage", ["post_load_weights", "cache_derived_state"])
def test_frozen_constant_cache_skips_meta_parameters(stage):
    """Neither owner may cache anything while its parameter is still meta.

    Both walks can reach a module before its weights are real -- the RO path
    materializes between ``setup_aliases()`` and the cache stage. Computing from
    a meta tensor does not raise; it yields meta tensors, which would be cached
    and then read by the forward as if they held values. The guard has to be on
    *both* owners, so this is parametrized over both stage names and asserts the
    cache is still empty rather than merely that nothing raised.
    """
    hidden = 16
    with torch.device("meta"):
        norm = MuseSparkOpenRMSNormWeightFirst(hidden, 1e-8, torch.float32)
        gated = MuseSparkOpenGatedResidual(hidden, torch.float32)
    assert norm.weight.is_meta and gated.gate.is_meta

    getattr(norm, stage)()
    getattr(gated, stage)()

    assert norm._weight_plus_one is None
    assert gated._alpha is None and gated._beta is None


@pytest.mark.parametrize("moved_dtype", [torch.float32, torch.bfloat16])
def test_frozen_constant_cache_is_dropped_by_a_dtype_move(moved_dtype):
    """A ``.to()`` after the cache is filled must not leave a stale cache.

    nn.Module carries buffers through ``.to()``/``.cuda()``/``.half()``, so
    without the ``_apply`` hook a dtype move would silently rewrite the cached
    constants. That is wrong in two different ways: casting ``w + 1`` is not
    ``cast(w) + 1``, and alpha/beta are FP32 by construction whatever dtype the
    module is in, so a move to bf16 would hand back something the uncached path
    can never produce. Nothing in the serving path moves a module after weight
    load, which is exactly why this would go unnoticed.
    """
    torch.manual_seed(0)
    hidden = 64

    gated = MuseSparkOpenGatedResidual(hidden, torch.bfloat16).to(DEVICE)
    gated.gate.data.normal_(0, 1.0)
    gated.cache_derived_state()
    assert gated._alpha is not None
    gated.to(moved_dtype)
    assert gated._alpha is None and gated._beta is None

    alpha, beta = gated.coefficients()
    exp_alpha, exp_beta = gated._compute_coefficients()
    torch.testing.assert_close(alpha, exp_alpha, atol=0, rtol=0)
    torch.testing.assert_close(beta, exp_beta, atol=0, rtol=0)
    assert alpha.dtype == torch.float32 and beta.dtype == torch.float32
    # A refresh after the move refills it with the same values.
    gated.cache_derived_state()
    torch.testing.assert_close(gated.coefficients()[0], exp_alpha, atol=0, rtol=0)
    torch.testing.assert_close(gated.coefficients()[1], exp_beta, atol=0, rtol=0)

    norm = MuseSparkOpenRMSNormWeightFirst(hidden, 1e-8, torch.bfloat16).to(DEVICE)
    norm.weight.data.normal_(0, 0.5)
    norm.cache_derived_state()
    assert norm._weight_plus_one is not None
    norm.to(moved_dtype)
    assert norm._weight_plus_one is None

    # After the move the output must match a module that never had a cache.
    fresh = MuseSparkOpenRMSNormWeightFirst(hidden, 1e-8, moved_dtype).to(DEVICE)
    fresh.weight.data.copy_(norm.weight.data)
    x = torch.randn(3, hidden, dtype=moved_dtype, device=DEVICE)
    torch.testing.assert_close(norm(x), fresh(x), atol=0, rtol=0)


def test_logit_soft_cap_and_multiplier():
    torch.manual_seed(0)

    class _Head(torch.nn.Module):
        def __init__(self, weight):
            super().__init__()
            self.weight = weight

        def forward(self, x):
            return x.to(self.weight.dtype) @ self.weight.T

    hidden, vocab = 16, 32
    weight = torch.randn(vocab, hidden, dtype=torch.bfloat16, device=DEVICE)
    head = _Head(weight)
    processor = MuseSparkOpenLogitsProcessor(logit_multiplier=32.0**-0.5, soft_cap=20.0).to(DEVICE)
    x = torch.randn(5, hidden, dtype=torch.bfloat16, device=DEVICE) * 20

    out = processor(x, head, None, return_context_logits=True)
    raw = (x @ weight.T).float() * (32.0**-0.5)
    torch.testing.assert_close(out, 20.0 * torch.tanh(raw / 20.0))
    assert out.abs().max() < 20.0


# --------------------------------------------------------------------------
# Fused Q / output-gate projection layout
# --------------------------------------------------------------------------
def test_q_gate_rows_are_interleaved_per_head():
    """Attention.preprocess_qkv reads [tokens, num_heads, 2*head_dim] and
    splits the last dim, so each head's query rows must be followed by that
    head's gate rows. Row sentinels catch a block-concat layout."""
    num_heads, head_dim, hidden = 4, 8, 3
    q = torch.arange(num_heads * head_dim, dtype=torch.float32).reshape(-1, 1).repeat(1, hidden)
    gate = q + 1000.0
    fused = head_interleaved_q_gate(q, gate, num_heads, head_dim)
    assert fused.shape == (num_heads * 2 * head_dim, hidden)

    # Replay the module's split on the fused rows.
    split = fused[:, 0].reshape(num_heads, 2 * head_dim)
    q_part, gate_part = split.chunk(2, dim=-1)
    torch.testing.assert_close(q_part.reshape(-1), q[:, 0])
    torch.testing.assert_close(gate_part.reshape(-1), gate[:, 0])

    # A plain concat would fail the same replay.
    wrong = torch.cat([q, gate], dim=0)[:, 0].reshape(num_heads, 2 * head_dim)
    assert not torch.allclose(wrong.chunk(2, dim=-1)[0].reshape(-1), q[:, 0])


def test_q_gate_interleave_shards_heads_contiguously():
    """A column-parallel split of the fused rows must give each rank whole
    heads with their own gates."""
    num_heads, head_dim, hidden, tp = 8, 4, 2, 4
    q = torch.arange(num_heads * head_dim, dtype=torch.float32).reshape(-1, 1).repeat(1, hidden)
    gate = q + 1000.0
    fused = head_interleaved_q_gate(q, gate, num_heads, head_dim)
    rank_rows = fused.shape[0] // tp
    for rank in range(tp):
        shard = fused[rank * rank_rows : (rank + 1) * rank_rows, 0]
        heads = num_heads // tp
        q_part, gate_part = shard.reshape(heads, 2 * head_dim).chunk(2, dim=-1)
        first_head = rank * heads
        expected_q = q[first_head * head_dim : (first_head + heads) * head_dim, 0]
        torch.testing.assert_close(q_part.reshape(-1), expected_q)
        torch.testing.assert_close(gate_part.reshape(-1), expected_q + 1000.0)


# --------------------------------------------------------------------------
# Routing
# --------------------------------------------------------------------------
@pytest.mark.parametrize("tokens", [1, 3, 256, 8192])
@pytest.mark.parametrize("case", ["random", "tiny", "zero", "ties"])
def test_routing_topk8_fused_weights(tokens: int, case: str) -> None:
    """Exercise the production kernel and preserve the unfused FP32 result."""
    torch.manual_seed(0)
    num_experts, top_k, eps = 256, 8, 1e-15
    logits = torch.randn(tokens, num_experts, dtype=torch.float32, device=DEVICE)
    if case == "tiny":
        logits.fill_(-60.0)
    elif case == "ties":
        logits.zero_()
    scores = torch.sigmoid(logits)
    if case == "zero":
        scores.zero_()
    bias = torch.linspace(-0.2, 0.2, num_experts, device=DEVICE)
    if case == "ties":
        bias.zero_()
    _, indices = torch.topk(scores + bias, k=top_k, dim=-1, sorted=False)
    expected_weights = scores.gather(1, indices)
    expected_weights = expected_weights / (expected_weights.sum(-1, keepdim=True) + eps)

    # A fallback-only implementation must not silently satisfy this test.
    with patch.object(_routing_weights_kernel, "run", wraps=_routing_weights_kernel.run) as run:
        selected, weights = _routing_weights(scores, indices, eps, torch.float32)
        assert run.call_count == 1

    assert selected.dtype == torch.int32
    assert weights.dtype == torch.float32
    assert selected.shape == weights.shape == (tokens, top_k)
    assert selected.is_contiguous() and weights.is_contiguous()
    torch.testing.assert_close(selected, indices.to(torch.int32), rtol=0, atol=0)
    torch.testing.assert_close(weights, expected_weights, rtol=0, atol=0)


def test_routing_bias_steers_selection_but_not_weights():
    torch.manual_seed(0)
    num_experts, top_k, tokens = 16, 4, 6
    bias = torch.zeros(num_experts, dtype=torch.float32, device=DEVICE)
    # Push one otherwise-unselected expert into the top-k.
    logits = torch.randn(tokens, num_experts, dtype=torch.float32, device=DEVICE)
    bias[3] = 10.0
    method = MuseSparkOpenRoutingMethod(
        top_k=top_k,
        num_experts=num_experts,
        callable_e_score_correction_bias=lambda: bias,
        renorm_eps=1e-15,
    ).to(DEVICE)
    ids, weights = method.apply(logits)

    assert method.requires_separated_routing is True
    assert (ids == 3).any(dim=-1).all(), "biased expert must always be picked"
    scores = torch.sigmoid(logits)
    expected = scores.gather(1, ids.long())
    expected = expected / (expected.sum(-1, keepdim=True) + 1e-15)
    torch.testing.assert_close(weights, expected)
    torch.testing.assert_close(
        weights.sum(-1), torch.ones(tokens, device=DEVICE), atol=1e-5, rtol=1e-5
    )
    # Weights must come from the unbiased scores: with a +10 bias, a biased
    # weight would be ~1.0 for that slot.
    biased_slot = (ids == 3).float().argmax(dim=-1)
    picked = weights.gather(1, biased_slot.unsqueeze(-1))
    assert picked.max() < 0.99


def test_routing_epsilon_is_the_source_value():
    num_experts, top_k = 8, 2
    bias = torch.zeros(num_experts, dtype=torch.float32, device=DEVICE)
    method = MuseSparkOpenRoutingMethod(
        top_k=top_k,
        num_experts=num_experts,
        callable_e_score_correction_bias=lambda: bias,
        renorm_eps=1e-15,
    ).to(DEVICE)
    assert method.renorm_eps == 1e-15
    # A hugely negative logit row makes the sum underflow; the divisor epsilon
    # then decides the result, so it must be the source's 1e-15.
    logits = torch.full((1, num_experts), -60.0, device=DEVICE)
    _, weights = method.apply(logits)
    scores = torch.sigmoid(logits.double())
    top = scores.topk(top_k, dim=-1).values
    expected = (top / (top.sum(-1, keepdim=True) + 1e-15)).float()
    torch.testing.assert_close(
        weights.sort(-1, descending=True).values,
        expected.sort(-1, descending=True).values,
        atol=1e-6,
        rtol=1e-6,
    )


def test_routing_memo_is_bit_exact_and_correctly_scoped():
    """The duplicate-routing memo must return the same answer, and only ever
    for the tensor it was computed from.

    MuseSparkOpenMoE routes the same logits twice per layer -- once inside the MoE
    backend and once in the model, because the backend's expert_weights output
    is uninitialized when topk_weights are supplied to it. The memo removes the
    second pass. Its whole justification is that the second call recomputed
    exactly what the first produced, so a hit must be bit-identical; and it must
    MISS whenever the tensor is not provably the same one, because on the padded
    and chunked paths a different tensor with different contents arrives.
    """
    torch.manual_seed(0)
    num_experts, top_k, tokens = 16, 4, 5
    # Deterministic and small: non-uniform enough that a memo which silently
    # dropped the bias would change the selection, but bounded well inside the
    # +-30 logit swing used below, so that swing decides the top-k outright.
    bias = torch.linspace(-0.2, 0.2, num_experts, dtype=torch.float32, device=DEVICE)

    def make_method():
        return MuseSparkOpenRoutingMethod(
            top_k=top_k,
            num_experts=num_experts,
            callable_e_score_correction_bias=lambda: bias,
            renorm_eps=1e-15,
        ).to(DEVICE)

    method = make_method()
    logits = torch.randn(tokens, num_experts, dtype=torch.float32, device=DEVICE)

    cold_idx, cold_w = method.apply(logits)
    cold_idx, cold_w = cold_idx.clone(), cold_w.clone()

    # A hit returns the identical tensors, not merely equal ones.
    hot_idx, hot_w = method.apply(logits)
    assert hot_idx is method._routing_memo[2] and hot_w is method._routing_memo[3]
    torch.testing.assert_close(hot_idx, cold_idx, atol=0, rtol=0)
    torch.testing.assert_close(hot_w, cold_w, atol=0, rtol=0)

    # A different tensor object carrying the same values must MISS and
    # recompute; that is the safe-degradation path the padded and chunked cases
    # take, and it must still give the same answer.
    same_values = logits.clone()
    miss_idx, miss_w = method.apply(same_values)
    assert method._routing_memo[0] is same_values
    torch.testing.assert_close(miss_idx, cold_idx, atol=0, rtol=0)
    torch.testing.assert_close(miss_w, cold_w, atol=0, rtol=0)

    # An in-place write to the memoized tensor bumps _version and must
    # invalidate it. Returning the stale routing here would be a silent wrong
    # answer rather than a slow one, which is the only way this memo can hurt.
    method.clear_routing_memo()
    stale_idx, stale_w = method.apply(logits)
    stale_version = method._routing_memo[1]
    with torch.no_grad():
        # Force the top-k to be exactly the first top_k experts: sigmoid(30) is
        # 1 and sigmoid(-30) is ~1e-13, and the bias above is bounded by 0.2, so
        # the selection is decided outright rather than probably.
        logits.fill_(-30.0)
        logits[:, :top_k] = 30.0
    assert logits._version != stale_version, "the _version guard's input never moved"

    changed_idx, changed_w = method.apply(logits)
    # The memo MISSED: fresh objects, not the memoized ones. Checking identity
    # rather than values is what actually pins the mechanism -- an earlier
    # version of this test mutated one logit by +25 and asserted the answer
    # changed, which sigmoid saturation made false, so a stale hit would have
    # passed unnoticed.
    assert changed_idx is not stale_idx and changed_w is not stale_w
    exp_idx, exp_w = make_method().apply(logits)
    torch.testing.assert_close(changed_idx, exp_idx, atol=0, rtol=0)
    torch.testing.assert_close(changed_w, exp_w, atol=0, rtol=0)
    # ...and the answer really did move, so a stale hit would also have been
    # visible in the values.
    assert sorted(changed_idx[0].tolist()) == list(range(top_k))
    assert not torch.equal(changed_idx, stale_idx)

    method.clear_routing_memo()
    assert method._routing_memo is None


def test_routing_memo_works_under_inference_mode():
    """The mode the engine actually runs in, which the test above is not in.

    An inference tensor carries no version counter, and merely touching
    ``_version`` on one raises ``RuntimeError: Inference tensors do not track
    version counter``. The forward runs under ``torch.inference_mode()`` in
    serving, so an unguarded read there is not an edge case: it killed all four
    ranks at engine warmup on the first routing call of the first layer, while
    every test above passed, because pytest does not run under inference mode.

    Asserting only that the call does not raise would be too weak -- an
    implementation that gave up and recomputed would also not raise -- so this
    checks the memo still HITS and still returns the identical objects.
    """
    torch.manual_seed(0)
    num_experts, top_k, tokens = 16, 4, 5
    bias = torch.linspace(-0.2, 0.2, num_experts, dtype=torch.float32, device=DEVICE)
    method = MuseSparkOpenRoutingMethod(
        top_k=top_k,
        num_experts=num_experts,
        callable_e_score_correction_bias=lambda: bias,
        renorm_eps=1e-15,
    ).to(DEVICE)

    with torch.inference_mode():
        logits = torch.randn(tokens, num_experts, dtype=torch.float32, device=DEVICE)
        assert logits.is_inference(), "pointless unless the tensor really is an inference one"

        cold_idx, cold_w = method.apply(logits)
        hot_idx, hot_w = method.apply(logits)

        # A hit, not a recompute: the very objects the first call memoized.
        assert hot_idx is cold_idx and hot_w is cold_w
        assert method._routing_memo[0] is logits
        # No counter exists for an inference tensor, so the memo stores None and
        # leans on object identity -- which is the guard that carries it anyway.
        assert method._routing_memo[1] is None

        # A different object still misses and still gives the same answer.
        other = logits.clone()
        miss_idx, miss_w = method.apply(other)
        assert miss_idx is not cold_idx
        assert torch.equal(miss_idx, cold_idx) and torch.equal(miss_w, cold_w)

    method.clear_routing_memo()
    assert method._routing_memo is None


def test_the_whole_decoder_stack_of_levers_runs_under_inference_mode():
    """A blanket guard for the class of bug, not just the one instance.

    The routing memo's version read was one autograd-only API on the serving
    path; the same mistake in any other lever would fail the same way and
    equally invisibly. This drives EVERY lever entry point on the decode path
    under ``torch.inference_mode()`` and requires the answers to match the
    ordinary-mode ones exactly.

    The routing method is in the list deliberately. When this test was first
    written it left the routing method out, and the negative control showed the
    consequence: it PASSED on the commit whose routing memo crashed every rank.
    A blanket that does not cover the one thing that broke is not a blanket.
    """
    torch.manual_seed(0)
    hidden, tokens, top_k, num_experts = 32, 3, 2, 8

    norm = MuseSparkOpenRMSNormWeightFirst(hidden, 1e-6, torch.float32).to(DEVICE)
    gate = MuseSparkOpenGatedResidual(hidden, torch.float32).to(DEVICE)
    with torch.no_grad():
        norm.weight.normal_(0, 0.1)
        gate.gate.normal_(0, 0.5)
    norm.cache_derived_state()
    gate.cache_derived_state()

    bias = torch.linspace(-0.2, 0.2, num_experts, dtype=torch.float32, device=DEVICE)
    routing = MuseSparkOpenRoutingMethod(
        top_k=top_k,
        num_experts=num_experts,
        callable_e_score_correction_bias=lambda: bias,
        renorm_eps=1e-15,
    ).to(DEVICE)

    branch = torch.randn(tokens, hidden, dtype=torch.float32, device=DEVICE)
    residual = torch.randn(tokens, hidden, dtype=torch.float32, device=DEVICE)
    rows = torch.randn(tokens * top_k, hidden, dtype=torch.float32, device=DEVICE)
    index = torch.arange(tokens * top_k, dtype=torch.int32, device=DEVICE)
    weights = torch.rand(tokens, top_k, dtype=torch.float32, device=DEVICE)
    logits = torch.randn(tokens, num_experts, dtype=torch.float32, device=DEVICE)

    def run():
        routing.clear_routing_memo()
        selected, scales = routing.apply(logits)
        return (
            norm(branch),
            gate(residual, branch),
            gate.add_after_rmsnorm(residual, branch, 1e-6),
            gate.add_after_rmsnorm(residual, branch, 1e-6, norm.gain()),
            selected,
            scales,
            post_expert_norm_combine(
                rows,
                index,
                weights,
                norm.weight,
                1e-6,
                num_tokens=tokens,
                top_k=top_k,
                hidden_size=hidden,
            ),
        )

    ordinary = run()
    with torch.inference_mode():
        inferred = run()

    for a, b in zip(ordinary, inferred):
        assert torch.equal(a, b)


# --------------------------------------------------------------------------
# Post-expert norm and combine
# --------------------------------------------------------------------------
def test_post_expert_norm_runs_before_the_combine():
    torch.manual_seed(0)
    tokens, top_k, hidden, eps = 3, 2, 32, 1e-8
    rows = torch.randn(tokens * top_k, hidden, dtype=torch.bfloat16, device=DEVICE)
    index = torch.arange(tokens * top_k, dtype=torch.int32, device=DEVICE)
    weights = torch.rand(tokens, top_k, dtype=torch.float32, device=DEVICE)
    norm_weight = torch.randn(hidden, dtype=torch.bfloat16, device=DEVICE) * 0.3

    out = post_expert_norm_combine(
        rows, index, weights, norm_weight, eps, num_tokens=tokens, top_k=top_k, hidden_size=hidden
    )

    def weight_first_norm(x):
        z = x.float() * (norm_weight.float() + 1.0)
        return z * torch.rsqrt(z.pow(2).mean(-1, keepdim=True) + eps)

    grouped = rows.reshape(tokens, top_k, hidden)
    golden = (weight_first_norm(grouped) * weights.unsqueeze(-1)).sum(dim=1)
    torch.testing.assert_close(out, golden, atol=1e-5, rtol=1e-5)

    # Norm-after-sum is a different function; the test must be able to tell.
    summed = (grouped.float() * weights.unsqueeze(-1)).sum(dim=1)
    norm_after = weight_first_norm(summed)
    assert not torch.allclose(out, norm_after, atol=1e-2)


def test_invalid_expert_slots_are_selected_out_not_masked():
    """Rows the kernel never wrote may hold NaN, so a multiply-by-zero mask
    would poison the whole token."""
    tokens, top_k, hidden, eps = 2, 4, 16, 1e-8
    rows = torch.randn(tokens * top_k, hidden, dtype=torch.bfloat16, device=DEVICE)
    index = (
        torch.arange(tokens * top_k, dtype=torch.int32, device=DEVICE)
        .reshape(tokens, top_k)
        .clone()
    )
    index[0, 1] = -1  # this slot is not local to this rank
    index[1, 1] = -1
    # Rows 1 and 5 are now referenced by no valid slot, so the kernel never
    # writes them and they may hold anything.
    rows[1].fill_(float("nan"))
    rows[5].fill_(float("inf"))
    weights = torch.rand(tokens, top_k, dtype=torch.float32, device=DEVICE)
    norm_weight = torch.zeros(hidden, dtype=torch.bfloat16, device=DEVICE)

    out = post_expert_norm_combine(
        rows, index, weights, norm_weight, eps, num_tokens=tokens, top_k=top_k, hidden_size=hidden
    )
    assert torch.isfinite(out).all()

    def weight_first_norm(x):
        z = x.float()
        return z * torch.rsqrt(z.pow(2).mean(-1, keepdim=True) + eps)

    golden = torch.zeros(tokens, hidden, dtype=torch.float32, device=DEVICE)
    for t in range(tokens):
        for k in range(top_k):
            row = int(index[t, k])
            if row < 0:
                continue
            golden[t] += weight_first_norm(rows[row].unsqueeze(0))[0] * weights[t, k]
    torch.testing.assert_close(out, golden, atol=1e-5, rtol=1e-5)


def test_post_expert_norm_rows_are_what_the_combine_sums():
    """The pre-combination rows the parity replay compares are what production sums.

    `post_expert_norm_rows` exists so the per-expert output can be compared
    against the source before it is weighted and reduced -- the source's fused
    finalize kernel destroys that boundary. If the two ever drifted apart, the
    gate would be checking a formula the model does not run.
    """
    torch.manual_seed(1)
    tokens, top_k, hidden, eps = 4, 3, 32, 1e-8
    rows = torch.randn(tokens * top_k, hidden, dtype=torch.bfloat16, device=DEVICE)
    index = (
        torch.arange(tokens * top_k, dtype=torch.int32, device=DEVICE)
        .reshape(tokens, top_k)
        .clone()
    )
    index[2, 0] = -1  # an expert this rank does not own
    weights = torch.rand(tokens, top_k, dtype=torch.float32, device=DEVICE)
    norm_weight = torch.randn(hidden, dtype=torch.bfloat16, device=DEVICE) * 0.3

    normed, valid = post_expert_norm_rows(
        rows, index, norm_weight, eps, num_tokens=tokens, top_k=top_k, hidden_size=hidden
    )
    assert normed.shape == (tokens, top_k, hidden)
    assert valid.shape == (tokens, top_k)
    assert not bool(valid[2, 0]) and int(valid.sum()) == tokens * top_k - 1
    assert torch.count_nonzero(normed[2, 0]) == 0

    combined = post_expert_norm_combine(
        rows, index, weights, norm_weight, eps, num_tokens=tokens, top_k=top_k, hidden_size=hidden
    )
    torch.testing.assert_close(
        (normed * weights.unsqueeze(-1)).sum(dim=1), combined, atol=1e-6, rtol=1e-6
    )

    # Same arithmetic the source's fused kernel applies to its own inputs:
    # scale by the pre-added weight first, then normalize by the RMS of the
    # scaled row. This is what the reference-parity replay reconstructs on the source side.
    scaled = rows.float() * (norm_weight.float() + 1.0)
    kernel_rows = scaled * torch.rsqrt(scaled.pow(2).mean(-1, keepdim=True) + eps)
    kernel_rows = kernel_rows.reshape(tokens, top_k, hidden)
    torch.testing.assert_close(normed[valid], kernel_rows[valid], atol=1e-5, rtol=1e-5)

    # A normalize-after-combine implementation would not reproduce these rows,
    # so the comparison can actually tell the two apart.
    after = combined * torch.rsqrt(combined.pow(2).mean(-1, keepdim=True) + eps)
    assert not torch.allclose(after, (normed * weights.unsqueeze(-1)).sum(dim=1), atol=1e-2)


# --------------------------------------------------------------------------
# Quantization: gate/up global scales
# --------------------------------------------------------------------------
def test_trtllm_gen_keeps_split_gate_up_weight_scale_2():
    """trtllm-gen takes the gate and up global scales separately, so unequal
    columns must survive; a single-alpha backend still reconciles."""
    from tensorrt_llm._torch.moe.fused_moe.quantization import (
        NVFP4CutlassFusedMoEMethod,
        NVFP4TRTLLMGenFusedMoEBaseMethod,
    )

    w1 = torch.tensor(1.272e-05)
    w3 = torch.tensor(1.689e-05)

    gen = NVFP4TRTLLMGenFusedMoEBaseMethod.__new__(NVFP4TRTLLMGenFusedMoEBaseMethod)
    assert gen.supports_split_gate_up_weight_scale_2 is True
    gate, up = gen._resolve_gate_up_weight_scale_2(w1, w3)
    assert gate.item() == pytest.approx(w1.item())
    assert up.item() == pytest.approx(w3.item())

    cutlass = NVFP4CutlassFusedMoEMethod.__new__(NVFP4CutlassFusedMoEMethod)
    assert cutlass.supports_split_gate_up_weight_scale_2 is False
    gate, up = cutlass._resolve_gate_up_weight_scale_2(w1, w3)
    assert gate.item() == pytest.approx(w3.item())
    assert up.item() == pytest.approx(w3.item())

    # Equal columns are untouched on either backend.
    same = torch.tensor(2.0)
    for method in (gen, cutlass):
        gate, up = method._resolve_gate_up_weight_scale_2(same, same.clone())
        assert gate.item() == 2.0 and up.item() == 2.0


def test_exclude_module_patterns_are_anchored():
    """The checkpoint writes bare substrings, which fnmatch never matches."""
    raw = ["self_attn", "lm_head", "layers.0.", "layers.61."]
    patterns = normalize_exclude_modules(raw)
    quant = QuantConfig(exclude_modules=patterns)

    assert quant.is_module_excluded_from_quantization("model.layers.0.feed_forward.experts")
    assert quant.is_module_excluded_from_quantization("model.layers.61.feed_forward.experts")
    assert quant.is_module_excluded_from_quantization("model.layers.30.self_attn.qkv_proj")
    assert quant.is_module_excluded_from_quantization("lm_head")
    # Quantized layers stay quantized, including the ones whose index shares a
    # prefix with an excluded one.
    for name in (
        "model.layers.1.feed_forward.experts",
        "model.layers.10.feed_forward.experts",
        "model.layers.6.feed_forward.experts",
        "model.layers.60.feed_forward.experts",
    ):
        assert not quant.is_module_excluded_from_quantization(name), name

    # The raw patterns match nothing, which is the bug this normalization fixes.
    raw_quant = QuantConfig(exclude_modules=raw)
    assert not raw_quant.is_module_excluded_from_quantization("model.layers.0.feed_forward.experts")


def test_unquantized_layers_from_exclude_patterns():
    config = make_config(62)
    assert config.unquantized_layers(["self_attn", "lm_head", "layers.0.", "layers.61."]) == [0, 61]
    assert config.unquantized_layers(normalize_exclude_modules(["layers.0.", "layers.61."])) == [
        0,
        61,
    ]
    assert config.unquantized_layers(None) == []


# --------------------------------------------------------------------------
# Weight-key accounting
# --------------------------------------------------------------------------
def test_out_of_range_and_perception_keys_are_ignored():
    keys = [
        "model.embed_tokens.weight",
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.3.self_attn.q_proj.weight",
        "model.layers.4.self_attn.q_proj.weight",
        "model.layers.61.feed_forward.experts.w13_weight",
        "vision_model.encoder.layers.0.attn.q_proj.weight",
        "modalities.soft_audio_perception.interaction_encoder.norm.weight",
        "lm_head.weight",
    ]
    accounting = classify_keys(keys, num_hidden_layers=4)
    assert accounting.ignored_out_of_range == {
        "model.layers.4.self_attn.q_proj.weight",
        "model.layers.61.feed_forward.experts.w13_weight",
    }
    assert len(accounting.ignored_perception) == 2
    assert accounting.consumed == set()

    # With the full depth, nothing is out of range.
    full = classify_keys(keys, num_hidden_layers=62)
    assert full.ignored_out_of_range == set()


def test_missing_in_range_keys_reports_a_dropped_tensor():
    present = []
    for layer in range(2):
        prefix = f"model.layers.{layer}"
        present += [
            f"{prefix}.self_attn.pre_qkv_proj_norm.weight",
            f"{prefix}.self_attn.q_proj.weight",
            f"{prefix}.self_attn.k_proj.weight",
            f"{prefix}.self_attn.v_proj.weight",
            f"{prefix}.self_attn.o_gate_proj.weight",
            f"{prefix}.self_attn.o_proj.weight",
            f"{prefix}.add_attn.gate",
            f"{prefix}.add_ffn.gate",
            f"{prefix}.pre_ffn_norm.weight",
            f"{prefix}.post_ffn_norm.weight",
            f"{prefix}.feed_forward.router.weight",
            f"{prefix}.feed_forward.expert_bias",
            f"{prefix}.feed_forward.pooler.weight",
            f"{prefix}.feed_forward.unpooler.weight",
            f"{prefix}.feed_forward.pre_expert_norm.weight",
            f"{prefix}.feed_forward.post_expert_norm.weight",
            f"{prefix}.feed_forward.experts.w13_weight",
            f"{prefix}.feed_forward.experts.w2_weight",
        ]
    for name in (
        "w13_weight_scale",
        "w2_weight_scale",
        "w13_weight_scale_2",
        "w2_weight_scale_2",
        "w13_input_scale",
        "w2_input_scale",
    ):
        present.append(f"model.layers.1.feed_forward.experts.{name}")
    present += ["model.embed_tokens.weight", "model.norm.weight", "lm_head.weight"]

    missing = missing_in_range_keys(
        present, num_hidden_layers=2, num_experts=256, bf16_expert_layers=[0]
    )
    assert missing == []

    missing = missing_in_range_keys(
        [k for k in present if not k.endswith("o_gate_proj.weight")],
        num_hidden_layers=2,
        num_experts=256,
        bf16_expert_layers=[0],
    )
    assert len(missing) == 2
    assert all("o_gate_proj" in name for name in missing)


# --------------------------------------------------------------------------
# The fused residual join (lever 2b)
#
# A decoder layer joins each branch as `gated_add(residual, rmsnorm(branch))`,
# 124 times per forward over the 8192-wide FP32 residual stream. These pin the
# fused kernel against the composition it replaces -- including, for the
# attention side, TRT-LLM's own RMSNorm module rather than a local golden, since
# that module is what the layer actually stopped calling.
# --------------------------------------------------------------------------
def _residual_join_case(tokens, hidden, dtype, seed=0):
    gen = torch.Generator(device=DEVICE).manual_seed(seed)
    residual = torch.randn(tokens, hidden, generator=gen, dtype=torch.float32, device=DEVICE)
    branch = torch.randn(tokens, hidden, generator=gen, dtype=torch.float32, device=DEVICE).to(
        dtype
    )
    gate = MuseSparkOpenGatedResidual(hidden, dtype).to(DEVICE)
    with torch.no_grad():
        gate.gate.normal_(0, 0.5, generator=gen)
    gate.cache_derived_state()
    return residual, branch, gate


@pytest.mark.parametrize("tokens", [1, 5, 64])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_fused_residual_join_matches_the_weightless_norm_path(tokens, dtype):
    """Attention side: `add_attn(h, post_attn_norm(y))` with post_attn_norm the
    real `RMSNorm(has_weights=False)` the layer used to call."""
    from tensorrt_llm._torch.modules.rms_norm import RMSNorm

    hidden, eps = 256, 1e-6
    residual, branch, gate = _residual_join_case(tokens, hidden, dtype)
    reference_norm = RMSNorm(hidden_size=hidden, eps=eps, dtype=dtype, has_weights=False).to(DEVICE)

    fused = gate.add_after_rmsnorm(residual, branch, eps)
    reference = gate(residual, reference_norm(branch))

    assert fused.dtype is torch.float32
    max_abs = (fused - reference).abs().max().item()
    scale = reference.abs().max().item()
    print(f"weightless tokens={tokens} {dtype} max_abs={max_abs:.3e} rel={max_abs / scale:.3e}")
    assert max_abs < 1e-4 * scale
    torch.testing.assert_close(fused, reference, rtol=2e-5, atol=2e-5)


@pytest.mark.parametrize("tokens", [1, 5, 64])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_fused_residual_join_matches_the_weight_first_norm_path(tokens, dtype):
    """FFN side: `add_ffn(h, post_ffn_norm(y))` with the weight-first norm.

    The gain is passed as the cached `1 + w`, which is how the decoder layer
    calls it, so this also pins that the cache and the kernel agree.
    """
    hidden, eps = 256, 1e-6
    residual, branch, gate = _residual_join_case(tokens, hidden, dtype)
    norm = MuseSparkOpenRMSNormWeightFirst(hidden, eps, dtype).to(DEVICE)
    with torch.no_grad():
        norm.weight.normal_(0, 0.3)
    norm.cache_derived_state()

    fused = gate.add_after_rmsnorm(residual, branch, eps, norm.gain())
    reference = gate(residual, norm(branch))

    max_abs = (fused - reference).abs().max().item()
    scale = reference.abs().max().item()
    print(f"weight-first tokens={tokens} {dtype} max_abs={max_abs:.3e} rel={max_abs / scale:.3e}")
    assert max_abs < 1e-4 * scale
    torch.testing.assert_close(fused, reference, rtol=2e-5, atol=2e-5)

    # The gain really is applied: dropping it is a different function, so a
    # kernel that ignored `gain` could not pass both this and the test above.
    assert not torch.allclose(fused, gate.add_after_rmsnorm(residual, branch, eps), atol=1e-2)


def test_fused_residual_join_declines_a_gain_of_a_different_dtype():
    """A gain whose dtype differs from the activation must take the composition.

    The kernel rounds the gain product back to `branch.dtype`, which is what the
    composition does only while the two share a dtype -- the case this model
    always builds, since `gain` is `weight + 1` over a parameter created in the
    module dtype. Give it an FP32 gain against an FP16 branch and torch promotes
    instead, skipping that round, and the fused and unfused paths stop being the
    same function. Measured at 5.1e-04 relative before the guard clause existed.

    So the contract is that the two paths agree for ANY gain dtype, and the way
    the guard keeps it is by declining rather than by matching torch's promotion
    rules inside the kernel.
    """
    hidden, eps, tokens = 128, 1e-6, 4
    residual, branch, gate = _residual_join_case(tokens, hidden, torch.float16)
    gain = torch.empty(hidden, dtype=torch.float32, device=DEVICE).normal_(0, 0.3) + 1.0
    assert gain.dtype is not branch.dtype

    got = gate.add_after_rmsnorm(residual, branch, eps, gain)

    scaled = (gain * branch).to(torch.float32)
    normed = (scaled * torch.rsqrt(scaled.pow(2).mean(-1, keepdim=True) + eps)).to(branch.dtype)
    reference = gate(residual, normed)

    # Bit-exact equality with the composition IS the proof the guard fired: the
    # kernel reduces the variance in a different order, so it cannot land on the
    # torch result to the last bit.
    torch.testing.assert_close(got, reference, atol=0, rtol=0)


def test_fused_residual_join_uses_the_cache_but_does_not_depend_on_it():
    """`gain()` falls back to a recompute when no cache has been refreshed, and
    the fused kernel must give the same answer either way -- the same contract
    lever 1a's caches carry, now on a second consumer."""
    hidden, eps, tokens = 128, 1e-6, 4
    residual, branch, gate = _residual_join_case(tokens, hidden, torch.bfloat16)
    norm = MuseSparkOpenRMSNormWeightFirst(hidden, eps, torch.bfloat16).to(DEVICE)
    with torch.no_grad():
        norm.weight.normal_(0, 0.3)

    assert norm._weight_plus_one is None
    uncached = gate.add_after_rmsnorm(residual, branch, eps, norm.gain())
    norm.cache_derived_state()
    assert norm._weight_plus_one is not None
    cached = gate.add_after_rmsnorm(residual, branch, eps, norm.gain())

    torch.testing.assert_close(cached, uncached, atol=0, rtol=0)


def test_fused_residual_join_replays_from_a_cuda_graph():
    """The serving path captures the decode step, and this kernel now runs twice
    per layer inside it."""
    hidden, eps, tokens = 256, 1e-6, 8
    residual, branch, gate = _residual_join_case(tokens, hidden, torch.bfloat16)
    norm = MuseSparkOpenRMSNormWeightFirst(hidden, eps, torch.bfloat16).to(DEVICE)
    with torch.no_grad():
        norm.weight.normal_(0, 0.3)
    norm.cache_derived_state()

    gate.add_after_rmsnorm(residual, branch, eps, norm.gain())
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        static_out = gate.add_after_rmsnorm(residual, branch, eps, norm.gain())

    fresh_residual, fresh_branch, _ = _residual_join_case(tokens, hidden, torch.bfloat16, seed=7)
    residual.copy_(fresh_residual)
    branch.copy_(fresh_branch)
    graph.replay()
    torch.cuda.synchronize()

    expected = gate(residual, norm(branch))
    assert torch.isfinite(static_out).all()
    torch.testing.assert_close(static_out, expected, rtol=2e-5, atol=2e-5)


@pytest.mark.parametrize("tokens", [1, 256])
def test_fused_residual_join_at_the_checkpoint_width(tokens):
    """The width the model actually runs at, which the cases above are not.

    The other residual-join tests use hidden=256, and correctness here turns out
    to be width- and warp-dependent: the sizing sweep found the same kernel at
    8192 wide and 4 warps disagreeing with the reference by 1.5e-3 relative,
    four orders worse than every other configuration it measured. The shipped configuration is a
    different warp count and is fine, but "fine at 256 wide" was never evidence
    for that, so the checkpoint's 8192 is pinned here directly.
    """
    hidden, eps = 8192, 1e-6
    residual, branch, gate = _residual_join_case(tokens, hidden, torch.bfloat16)
    norm = MuseSparkOpenRMSNormWeightFirst(hidden, eps, torch.bfloat16).to(DEVICE)
    with torch.no_grad():
        norm.weight.normal_(0, 0.3)
    norm.cache_derived_state()

    weightless = gate.add_after_rmsnorm(residual, branch, eps)
    weight_first = gate.add_after_rmsnorm(residual, branch, eps, norm.gain())

    from tensorrt_llm._torch.modules.rms_norm import RMSNorm

    reference_norm = RMSNorm(
        hidden_size=hidden, eps=eps, dtype=torch.bfloat16, has_weights=False
    ).to(DEVICE)
    for fused, reference in (
        (weightless, gate(residual, reference_norm(branch))),
        (weight_first, gate(residual, norm(branch))),
    ):
        max_abs = (fused - reference).abs().max().item()
        scale = reference.abs().max().item()
        print(f"width {hidden} tokens={tokens} max_abs={max_abs:.3e} rel={max_abs / scale:.3e}")
        assert torch.isfinite(fused).all()
        assert max_abs < 1e-4 * scale
        torch.testing.assert_close(fused, reference, rtol=2e-5, atol=2e-5)


@pytest.mark.parametrize("value", ["0", "false", "off", "No"])
def test_fused_residual_switch_off_takes_the_composition(monkeypatch, value):
    """MUSESPARK_OPEN_FUSED_RESIDUAL=<falsy> must route both sides to the chain.

    This is the lever's control leg, and it exists so the A/B runs on ONE image:
    without it the only way to time a layer without this fusion is a second
    build, and two images differ by more than the lever under test.

    Bit-exact equality with the composition IS the proof the kernel did not run,
    for the same reason the mismatched-gain test relies on it: the kernel
    reduces the variance in a different order and so cannot land on the torch
    result to the last bit. Asserting only "close" would pass just as well with
    the switch ignored entirely.
    """
    hidden, eps, tokens = 256, 1e-6, 8
    monkeypatch.setenv("MUSESPARK_OPEN_FUSED_RESIDUAL", value)
    residual, branch, _ = _residual_join_case(tokens, hidden, torch.bfloat16)
    gen = torch.Generator(device=DEVICE).manual_seed(3)
    off = MuseSparkOpenGatedResidual(hidden, torch.bfloat16).to(DEVICE)
    with torch.no_grad():
        off.gate.normal_(0, 0.5, generator=gen)
    off.cache_derived_state()
    assert off._fused_enabled is False

    norm = MuseSparkOpenRMSNormWeightFirst(hidden, eps, torch.bfloat16).to(DEVICE)
    with torch.no_grad():
        norm.weight.normal_(0, 0.3)
    norm.cache_derived_state()

    # Both sides of the layer: weight-first (FFN) and weightless (attention).
    from tensorrt_llm._torch.modules.rms_norm import RMSNorm

    weightless_norm = RMSNorm(
        hidden_size=hidden, eps=eps, dtype=torch.bfloat16, has_weights=False
    ).to(DEVICE)
    for gain, reference_norm in ((norm.gain(), norm), (None, weightless_norm)):
        got = off.add_after_rmsnorm(residual, branch, eps, gain)
        torch.testing.assert_close(got, off(residual, reference_norm(branch)), atol=0, rtol=0)


# The next three run on BOTH cpu and DEVICE. The property they assert -- that a
# switch read once in __init__ selects the arm it names, and that the two arms of
# a change declared BIT-EXACT agree to the last bit -- has no device-dependent
# part in principle, and an earlier revision ran them on cpu alone for that
# reason. It is still the wrong choice here: these are the pass evidence for two
# bit-exact claims on a path that only ever executes on a GPU, and a claim about
# GPU arithmetic proved on cpu is proved somewhere else. Reduction order is
# exactly what differs between the two backends, so the cpu leg cannot see a
# regression that appears only in a cuda kernel. Keeping the cpu leg as well
# costs milliseconds and keeps the switch itself checkable on a machine with no
# GPU.


@pytest.mark.parametrize("device", ["cpu", DEVICE])
@pytest.mark.parametrize("value", ["0", "false", "off", "No"])
def test_frozen_gate_cache_switch_off_recomputes_and_matches_the_cache(monkeypatch, value, device):
    """MUSESPARK_OPEN_FROZEN_GATE_CACHE=<falsy> must recompute, identically.

    Sub-item 1a hoists constants derived from frozen parameters out of the
    forward. It is declared BIT-EXACT, and bit-exactness can only be shown by
    running both arms and comparing -- which needs both arms reachable from ONE
    process, because this build is not greedy-deterministic across server boots
    and two images therefore cannot answer the question.

    Equality is asserted at atol=0, rtol=0 on purpose. "Close" would pass with
    the switch ignored entirely, which is the failure this test exists to catch.
    """
    monkeypatch.setenv("MUSESPARK_OPEN_FROZEN_GATE_CACHE", value)
    hidden = 128

    norm = MuseSparkOpenRMSNormWeightFirst(hidden, 1e-6, torch.float32).to(device)
    with torch.no_grad():
        norm.weight.normal_(0, 0.3)
    norm.cache_derived_state()
    assert norm._cache_enabled is False
    assert norm._weight_plus_one is not None, "the cache is filled; the switch must bypass it"
    torch.testing.assert_close(norm.gain(), norm.weight + 1, atol=0, rtol=0)

    gated = MuseSparkOpenGatedResidual(hidden, torch.float32).to(device)
    with torch.no_grad():
        gated.gate.normal_(0, 0.5)
    gated.cache_derived_state()
    assert gated._cache_enabled is False
    assert gated._alpha is not None
    alpha, beta = gated.coefficients()
    recomputed_alpha, recomputed_beta = gated._compute_coefficients()
    torch.testing.assert_close(alpha, recomputed_alpha, atol=0, rtol=0)
    torch.testing.assert_close(beta, recomputed_beta, atol=0, rtol=0)


@pytest.mark.parametrize("device", ["cpu", DEVICE])
def test_frozen_gate_cache_default_on_uses_the_cached_tensors(device):
    """Unset means cached, and cached means the buffer itself, not a copy.

    Identity rather than equality: the point of the hoist is that the forward
    stops launching the sigmoid/sigmoid/multiply/clamp/sqrt chain, and a value
    that merely equals the cache would still have launched it.
    """
    hidden = 128
    norm = MuseSparkOpenRMSNormWeightFirst(hidden, 1e-6, torch.float32).to(device)
    with torch.no_grad():
        norm.weight.normal_(0, 0.3)
    norm.cache_derived_state()
    assert norm._cache_enabled is True
    assert norm.gain() is norm._weight_plus_one

    gated = MuseSparkOpenGatedResidual(hidden, torch.float32).to(device)
    with torch.no_grad():
        gated.gate.normal_(0, 0.5)
    gated.cache_derived_state()
    alpha, beta = gated.coefficients()
    assert alpha is gated._alpha and beta is gated._beta


@pytest.mark.parametrize("device", ["cpu", DEVICE])
@pytest.mark.parametrize("value,expected_passes", [(None, 1), ("1", 1), ("0", 2), ("off", 2)])
def test_routing_memo_switch_controls_the_duplicate_pass(
    monkeypatch, value, expected_passes, device
):
    """MUSESPARK_OPEN_ROUTING_MEMO=<falsy> must route the second call to a recompute.

    Sub-item 1b removes the duplicated routing pass by memoizing the first
    result on the identity of the logits tensor. Also declared BIT-EXACT, and
    the assertion that makes that claim testable is here: with the memo OFF the
    second pass really recomputes, and it lands on the same tensors to the last
    bit. Counting the recomputes is what separates "the switch works" from "the
    switch is ignored and both calls happened to agree".
    """
    if value is None:
        monkeypatch.delenv("MUSESPARK_OPEN_ROUTING_MEMO", raising=False)
    else:
        monkeypatch.setenv("MUSESPARK_OPEN_ROUTING_MEMO", value)

    calls = []

    class CountingRouting(MuseSparkOpenRoutingMethod):
        def get_scores(self, router_logits, bias):
            calls.append(1)
            scores = torch.softmax(router_logits.float(), dim=-1)
            return scores, scores

    router = CountingRouting(
        top_k=2,
        num_experts=8,
        renorm_eps=1e-6,
        callable_e_score_correction_bias=lambda: None,
    )
    router.output_dtype = torch.bfloat16
    # Seeded on cpu and then moved, so the cuda leg routes exactly the logits the
    # cpu leg routed -- a device-side generator would give the two legs different
    # inputs and turn a bit-exactness check into two unrelated runs.
    logits = torch.randn(4, 8, generator=torch.Generator().manual_seed(11)).to(device)

    first_selected, first_weights = router.apply(logits)
    second_selected, second_weights = router.apply(logits)

    assert len(calls) == expected_passes
    torch.testing.assert_close(second_selected, first_selected, atol=0, rtol=0)
    torch.testing.assert_close(second_weights, first_weights, atol=0, rtol=0)


def test_fused_residual_switch_default_on_agrees_with_the_switch_off():
    """Unset means fused, and the two arms differ only by the reduction order.

    The bound is the lever's declared contract, not a shrug: the kernel
    reproduces both of the chain's rounds to the activation dtype -- the gain
    product before the variance and the norm result before the gated add -- so
    the variance reduction order is the only arithmetic left that can differ.
    The sizing bench measured that at 8.5e-08 relative at the checkpoint's
    width; 1e-6 here is that with margin, and a kernel that dropped either round
    would be MORE accurate than the chain and would blow through it.
    """
    hidden, eps, tokens = 8192, 1e-6, 4
    residual, branch, _ = _residual_join_case(tokens, hidden, torch.bfloat16)
    norm = MuseSparkOpenRMSNormWeightFirst(hidden, eps, torch.bfloat16).to(DEVICE)
    with torch.no_grad():
        norm.weight.normal_(0, 0.3)
    norm.cache_derived_state()

    def build(env_value):
        gen = torch.Generator(device=DEVICE).manual_seed(3)
        with pytest.MonkeyPatch.context() as patch:
            if env_value is None:
                patch.delenv("MUSESPARK_OPEN_FUSED_RESIDUAL", raising=False)
            else:
                patch.setenv("MUSESPARK_OPEN_FUSED_RESIDUAL", env_value)
            gate = MuseSparkOpenGatedResidual(hidden, torch.bfloat16).to(DEVICE)
        with torch.no_grad():
            gate.gate.normal_(0, 0.5, generator=gen)
        gate.cache_derived_state()
        return gate

    on, off = build(None), build("0")
    assert (on._fused_enabled, off._fused_enabled) == (True, False)

    for gain in (norm.gain(), None):
        fused = on.add_after_rmsnorm(residual, branch, eps, gain)
        unfused = off.add_after_rmsnorm(residual, branch, eps, gain)
        max_abs = (fused - unfused).abs().max().item()
        scale = unfused.abs().max().item()
        print(f"switch gain={gain is not None} max_abs={max_abs:.3e} rel={max_abs / scale:.3e}")
        assert torch.isfinite(fused).all()
        assert max_abs < 1e-6 * scale


def test_fused_residual_switch_is_read_once_at_construction():
    """The engine's choice is fixed when the module is built, not per call.

    Per call it would be an environment lookup twice per layer per step, and --
    the reason that matters rather than just being slow -- a value that could
    change under a captured CUDA graph, where the arm recorded at capture is the
    one that replays whatever the variable says afterwards. A test that only
    checked the OFF path would not notice the difference.
    """
    hidden = 128
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("MUSESPARK_OPEN_FUSED_RESIDUAL", "0")
        gate = MuseSparkOpenGatedResidual(hidden, torch.bfloat16).to(DEVICE)
    assert gate._fused_enabled is False

    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("MUSESPARK_OPEN_FUSED_RESIDUAL", "1")
        assert gate._fused_enabled is False
        assert MuseSparkOpenGatedResidual(hidden, torch.bfloat16)._fused_enabled is True


def test_causal_lm_is_a_one_engine_speculative_model():
    """The decoder derives from the one-engine base so the SA worker runs in its forward.

    The omni wrapper forwards ``spec_metadata`` to this class, and the executor
    never reads speculative attributes on the wrapper, so this is the only place
    the plumbing has to live. The decoder's own overrides must survive the base change.
    """
    from tensorrt_llm._torch.models.modeling_speculative import SpecDecOneEngineForCausalLM

    assert issubclass(MuseSparkOpenForCausalLM, SpecDecOneEngineForCausalLM)
    assert MuseSparkOpenForCausalLM.load_weights is not SpecDecOneEngineForCausalLM.load_weights
    assert MuseSparkOpenForCausalLM.get_preferred_kv_cache_manager_version() == "V2"
    assert MuseSparkOpenForCausalLM.requires_kv_cache_manager_v2() is True
    assert "infer_max_seq_len" in vars(MuseSparkOpenForCausalLM)


def test_spec_admission_accepts_no_speculation_external_drafters_sa_and_dflash():
    from tensorrt_llm.llmapi import DFlashDecodingConfig, NGramDecodingConfig, SADecodingConfig

    assert_supported_spec_config(None)
    assert_supported_spec_config(NGramDecodingConfig(max_draft_len=4, max_matching_ngram_size=2))
    assert_supported_spec_config(SADecodingConfig(max_draft_len=4))
    # DFlash is one-engine, but its drafter weights come from the separate
    # speculative_model checkpoint, so admission must not gate on it.
    assert_supported_spec_config(
        DFlashDecodingConfig(max_draft_len=7, speculative_model="unused-drafter-path")
    )


def test_spec_admission_refuses_modes_that_need_draft_weights():
    from tensorrt_llm.llmapi import MTPDecodingConfig

    with pytest.raises(AssertionError, match="draft weights"):
        assert_supported_spec_config(MTPDecodingConfig(num_nextn_predict_layers=1))
