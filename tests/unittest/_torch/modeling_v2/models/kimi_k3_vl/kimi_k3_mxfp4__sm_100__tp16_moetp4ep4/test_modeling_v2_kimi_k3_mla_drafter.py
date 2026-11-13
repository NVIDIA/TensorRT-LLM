# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The Kimi K3 target's MLA DSpark drafter (``drafter_mla.K3MLADSparkDrafter`` of
``kimi_k3_mxfp4__sm_100__tp16_moetp4ep4``) on one GPU against the built-in ``MLADSparkForCausalLM`` (Modeling V1), both
loaded from the same draft checkpoint at one TP16 rank's shapes of Inferact-Kimi-K3-DSpark (4 heads, q_lora 1536,
kv_lora 512 + rope 64, nope 128, v 128, MLP 896, YaRN factor 32 / theta 5e4), two layers, CUTEDSL block decode over a
64-row paged latent context.

Bit for bit (the same kernels in the same order): every loaded tensor, the YaRN table, the context latents
(``precompute_context_kv``), ``project_target_hidden``, the block forward and the block latents it leaves in the
pages, on every split tested; a captured block replays the eager result. The new drafter has every field the worker
reads.

On ``attention_backend`` VANILLA (the built-in eager torch decode over the gathered pages and the block), bit for bit
against the built-in drafter on the same backend: the block forward on every split, the pages left as they were (the
eager decode stores nothing), a captured block's replay; and the private-arena context (no page table), which both
backends decode eagerly, as the built-in drafter does.

On ``attention_backend`` TRTLLM (the built-in flashinfer TRTLLM-Gen decode, causal over the block, plus the block's
upper triangle attended eagerly and merged by log-sum-exp), bit for bit against the built-in drafter on the same
backend: the block forward on every split, the block latents left in the pages, a captured block's replay.
"""

import math
from types import SimpleNamespace

import pytest
import torch
from transformers import PretrainedConfig

from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.kimi_k3_mxfp4__sm_100__tp16_moetp4ep4 import (  # noqa: E501
    drafter as target_drafter,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.kimi_k3_mxfp4__sm_100__tp16_moetp4ep4 import (  # noqa: E501
    drafter_mla as target,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.kimi_k3_mxfp4__sm_100__tp16_moetp4ep4.ops.attention import (  # noqa: E501
    k3_dspark_mla,
)
from tensorrt_llm._torch.model_config import ModelConfig

if not torch.cuda.is_available() or torch.cuda.get_device_capability() not in ((10, 0), (10, 3)):
    pytest.skip(
        "the MLA DSpark drafter's kernels run on sm_100 and sm_103", allow_module_level=True
    )
if k3_dspark_mla.mla_decode_unavailability_reason() is not None:
    pytest.skip(k3_dspark_mla.mla_decode_unavailability_reason(), allow_module_level=True)

from tensorrt_llm._torch.models.modeling_dspark import MLADSparkForCausalLM  # noqa: E402

HIDDEN, INTER, LAYERS, PAGE, VOCAB, RANK = 7168, 896, 2, 64, 1024, 16
HEADS, Q_LORA, KV_LORA, ROPE, NOPE, V_DIM = 4, 1536, 512, 64, 128, 128
CTX = (37, 64, 130, 0, 5, 200, 63, 1)


def _config():
    return PretrainedConfig(
        architectures=["K3DSparkModel"],
        model_type="k3_dspark",
        hidden_size=HIDDEN,
        intermediate_size=INTER,
        num_hidden_layers=LAYERS,
        num_attention_heads=HEADS,
        num_key_value_heads=HEADS,
        q_lora_rank=Q_LORA,
        kv_lora_rank=KV_LORA,
        qk_nope_head_dim=NOPE,
        qk_rope_head_dim=ROPE,
        v_head_dim=V_DIM,
        vocab_size=VOCAB,
        rms_norm_eps=1e-5,
        max_position_embeddings=4096,
        rope_theta=50000.0,
        rope_parameters={
            "rope_type": "yarn",
            "factor": 32.0,
            "original_max_position_embeddings": 32768,
            "rope_theta": 50000.0,
            "beta_fast": 32,
            "beta_slow": 1,
            "mscale": 1.0,
            "mscale_all_dim": 1.0,
        },
        mask_token_id=VOCAB - 3,
        markov_rank=RANK,
        markov_head_type="vanilla",
        enable_confidence_head=True,
        tie_word_embeddings=False,
        torch_dtype=torch.bfloat16,
    )


def _weights(seed=5):
    g = torch.Generator().manual_seed(seed)

    def rnd(*shape, scale=0.02):
        return (torch.randn(*shape, generator=g) * scale).to(torch.bfloat16)

    w = {
        "context_proj.weight": rnd(HIDDEN, 2 * HIDDEN),
        "context_norm.weight": rnd(HIDDEN, scale=0.05) + 1.0,
        "final_norm.weight": rnd(HIDDEN, scale=0.05) + 1.0,
        "embed_tokens.weight": rnd(VOCAB, HIDDEN),
        "markov_head.markov_w1.weight": rnd(VOCAB, RANK),
        "markov_head.markov_w2.weight": rnd(VOCAB, RANK),
        "confidence_head.proj.weight": rnd(1, HIDDEN + RANK),
        "confidence_head.proj.bias": rnd(1),
    }
    for i in range(LAYERS):
        p = f"layers.{i}."
        w[p + "self_attn.q_a_proj.weight"] = rnd(Q_LORA, HIDDEN)
        w[p + "self_attn.q_a_layernorm.weight"] = rnd(Q_LORA, scale=0.05) + 1.0
        w[p + "self_attn.q_b_proj.weight"] = rnd(HEADS * (NOPE + ROPE), Q_LORA)
        w[p + "self_attn.kv_a_proj_with_mqa.weight"] = rnd(KV_LORA + ROPE, HIDDEN)
        w[p + "self_attn.kv_a_layernorm.weight"] = rnd(KV_LORA, scale=0.05) + 1.0
        w[p + "self_attn.kv_b_proj.weight"] = rnd(HEADS * (NOPE + V_DIM), KV_LORA)
        w[p + "self_attn.o_proj.weight"] = rnd(HIDDEN, HEADS * V_DIM)
        w[p + "input_layernorm.weight"] = rnd(HIDDEN, scale=0.05) + 1.0
        w[p + "post_attention_layernorm.weight"] = rnd(HIDDEN, scale=0.05) + 1.0
        w[p + "mlp.gate_proj.weight"] = rnd(INTER, HIDDEN)
        w[p + "mlp.up_proj.weight"] = rnd(INTER, HIDDEN)
        w[p + "mlp.down_proj.weight"] = rnd(HIDDEN, INTER)
    return w


def _load(cls, backend="CUTEDSL"):
    module = cls(
        ModelConfig(pretrained_config=_config(), attn_backend="TRTLLM"),
        dflash_attention_backend=backend,
    ).to("cuda")
    module.load_weights(dict(_weights()))
    return module


@pytest.fixture(scope="module")
def pair():
    return SimpleNamespace(new=_load(target.K3MLADSparkDrafter), old=_load(MLADSparkForCausalLM))


def _assert_bitwise(new, old, what):
    assert new.shape == old.shape and new.dtype == old.dtype, f"{what}: {new.shape} vs {old.shape}"
    if not torch.equal(new, old):
        diff = (new.float() - old.float()).abs()
        raise AssertionError(
            f"{what}: {int((new != old).sum())} of {new.numel()} differ, max |diff| {float(diff.max())}"
        )


def _block(batch, block, seed):
    """Noise rows, positions after each request's context, a paged latent context per layer
    ([pages, 1, 1, 64, 576], the requests' pages scattered) and its page table."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    ctx = torch.tensor([CTX[i % len(CTX)] for i in range(batch)], dtype=torch.int32, device="cuda")
    width = math.ceil((int(ctx.max()) + block) / PAGE) + 1
    pages = batch * width + 3
    order = torch.randperm(pages, generator=g, device="cuda")[: batch * width]
    caches = [
        torch.randn(pages, 1, 1, PAGE, KV_LORA + ROPE, generator=g, device="cuda").to(
            torch.bfloat16
        )
        for _ in range(LAYERS)
    ]
    return dict(
        noise_embedding=torch.randn(batch, block, HIDDEN, generator=g, device="cuda").to(
            torch.bfloat16
        ),
        query_positions=(ctx.long()[:, None] + torch.arange(block, device="cuda")),
        num_ctx_per_req=ctx,
        ctx_k_cache=None,
        ctx_v_cache=None,
        ctx_cache_batch_idx=torch.arange(batch, dtype=torch.int32, device="cuda"),
        ctx_kv_cache=caches,
        ctx_page_table=order.view(batch, width).to(torch.int32),
    )


def _copy(inputs):
    out = dict(inputs)
    out["ctx_kv_cache"] = [c.clone() for c in inputs["ctx_kv_cache"]]
    return out


_PARAMS = (
    "self_attn.q_a_proj.weight",
    "self_attn.q_a_layernorm.weight",
    "self_attn.q_b_proj.weight",
    "self_attn.kv_a_proj_with_mqa.weight",
    "self_attn.kv_a_layernorm.weight",
    "self_attn.kv_b_proj.weight",
    "self_attn.o_proj.weight",
    "input_layernorm.weight",
    "post_attention_layernorm.weight",
    "mlp.gate_up_proj.weight",
    "mlp.down_proj.weight",
)


def test_weight_load_matches_main(pair):
    old = dict(pair.old.named_parameters(remove_duplicate=False))
    new = dict(pair.new.named_parameters(remove_duplicate=False))
    for i in range(LAYERS):
        for name in _PARAMS:
            key = f"model.layers.{i}.{name}"
            _assert_bitwise(new[key], old[key], key)
    for key in (
        "model.norm.weight",
        "model.embed_tokens.weight",
        "fc.weight",
        "hidden_norm.weight",
    ):
        _assert_bitwise(new[key], old[key], key)
    for attr in ("markov_w1", "markov_w2", "confidence_proj_weight", "confidence_proj_bias"):
        _assert_bitwise(getattr(pair.new, attr), getattr(pair.old, attr), attr)
    assert pair.new.hidden_norm.eps == pair.old.hidden_norm.eps
    for field in (
        "mask_token_id",
        "target_layer_ids",
        "block_size",
        "_dspark_shift_label",
        "softmax_scale",
        "dflash_attention_backend",
        "_num_attn_layers",
        "_num_heads",
        "_num_kv_heads",
        "_head_dim",
        "_kv_factor",
        "_paged_ctx_cache",
        "_uses_worker_attention_backend",
        "_tp_split",
    ):
        assert getattr(pair.new, field) == getattr(pair.old, field), field
    assert pair.new.has_markov_head and pair.old.has_markov_head


def test_worker_fields(pair):
    for name in target_drafter.WORKER_FIELDS:
        assert hasattr(pair.new, name), name


def test_embedding_matches_main(pair):
    ids = torch.tensor([0, 5, VOCAB - 3, VOCAB - 1, 17], device="cuda")
    new = pair.new.draft_model_full.model.embed_tokens
    old = pair.old.draft_model_full.model.embed_tokens
    _assert_bitwise(new(ids), old(ids), "embedding lookup")


def test_yarn_table_and_context_latents_match_main(pair):
    pair.old._build_fused_kv_buffers()
    pair.new._build_fused_kv_buffers()
    _assert_bitwise(pair.new._mla_freqs_cis("cuda"), pair.old._mla_freqs_cis("cuda"), "YaRN table")
    _assert_bitwise(pair.new._fused_kv_weight, pair.old._fused_kv_weight, "fused latent weight")
    _assert_bitwise(pair.new._kv_a_norm_stacked, pair.old._kv_a_norm_stacked, "stacked kv_a norm")
    for i, (a, b) in enumerate(zip(pair.new.model.layers, pair.old.model.layers)):
        _assert_bitwise(a.self_attn._k_b_proj, b.self_attn._k_b_proj, f"k_b {i}")
        _assert_bitwise(a.self_attn._v_b_proj, b.self_attn._v_b_proj, f"v_b {i}")
    g = torch.Generator(device="cuda").manual_seed(2)
    for rows in (1, 7, 300):
        projected = torch.randn(rows, HIDDEN, generator=g, device="cuda").to(torch.bfloat16)
        positions = torch.randint(0, 4000, (rows,), generator=g, device="cuda")
        new_latent, new_v = pair.new.precompute_context_kv(projected, positions)
        old_latent, old_v = pair.old.precompute_context_kv(projected, positions)
        assert new_v is None and old_v is None
        _assert_bitwise(new_latent, old_latent, f"context latents, {rows} rows")


@pytest.mark.parametrize("rows", [1, 56, 300])
def test_projection_matches_main(pair, rows):
    g = torch.Generator(device="cuda").manual_seed(rows)
    features = torch.randn(rows, 2 * HIDDEN, generator=g, device="cuda").to(torch.bfloat16)
    _assert_bitwise(
        pair.new.project_target_hidden(features),
        pair.old.project_target_hidden(features),
        "projection",
    )


@pytest.mark.parametrize("split", [(1, 7), (2, 7), (8, 7), (3, 8), (4, 4), (16, 7)], ids=str)
def test_block_matches_main(pair, split):
    """The block forward and the block latents it leaves in the context pages."""
    batch, block = split
    inputs = _block(batch, block, seed=batch * 10 + block)
    old_inputs = _copy(inputs)
    out = pair.new.dflash_forward(**inputs)
    ref = pair.old.dflash_forward(**old_inputs)
    torch.cuda.synchronize()
    _assert_bitwise(out, ref, f"block {split}")
    for layer, (c_new, c_old) in enumerate(zip(inputs["ctx_kv_cache"], old_inputs["ctx_kv_cache"])):
        _assert_bitwise(c_new, c_old, f"pages of layer {layer} after {split}")


def test_block_replays_under_capture(pair):
    inputs = _block(2, 7, seed=91)
    eager = pair.new.dflash_forward(**_copy(inputs))
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = pair.new.dflash_forward(**inputs)
    graph.replay()
    torch.cuda.synchronize()
    _assert_bitwise(captured, eager, "replayed block")


def test_negative_control_a_changed_weight_fails(pair):
    inputs = _block(1, 7, seed=81)
    old_inputs = _copy(inputs)
    weight = pair.new.model.layers[0].self_attn.o_proj.weight
    saved = weight.detach().clone()
    try:
        with torch.no_grad():
            weight.add_(torch.randn_like(weight) * 0.02)
        out = pair.new.dflash_forward(**inputs)
    finally:
        with torch.no_grad():
            weight.copy_(saved)
    ref = pair.old.dflash_forward(**old_inputs)
    assert not torch.equal(out, ref)


# ----------------------------------------------------------------------------------------------------------------------
# attention_backend VANILLA: the built-in eager decode.
# ----------------------------------------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def vanilla_pair():
    return SimpleNamespace(
        new=_load(target.K3MLADSparkDrafter, "VANILLA"), old=_load(MLADSparkForCausalLM, "VANILLA")
    )


def test_vanilla_resolves_like_main(vanilla_pair):
    assert (
        vanilla_pair.new.dflash_attention_backend
        == vanilla_pair.old.dflash_attention_backend
        == "VANILLA"
    )
    assert vanilla_pair.new._paged_ctx_cache and vanilla_pair.old._paged_ctx_cache
    assert vanilla_pair.new._block_decode_variant(True) == "eager"
    assert vanilla_pair.old._mla_block_decode_variant(True) == "eager"


@pytest.mark.parametrize("split", [(1, 7), (2, 7), (8, 7), (3, 8), (4, 4), (16, 7)], ids=str)
def test_vanilla_block_matches_main(vanilla_pair, split):
    """The eager decode over the gathered pages: the output bit for bit, and the pages untouched on both sides."""
    batch, block = split
    inputs = _block(batch, block, seed=batch * 10 + block + 3)
    old_inputs = _copy(inputs)
    before = [c.clone() for c in inputs["ctx_kv_cache"]]
    out = vanilla_pair.new.dflash_forward(**inputs)
    ref = vanilla_pair.old.dflash_forward(**old_inputs)
    torch.cuda.synchronize()
    _assert_bitwise(out, ref, f"VANILLA block {split}")
    for layer, (c_new, c_old, c_before) in enumerate(
        zip(inputs["ctx_kv_cache"], old_inputs["ctx_kv_cache"], before)
    ):
        _assert_bitwise(c_new, c_before, f"VANILLA pages of layer {layer} (new)")
        _assert_bitwise(c_old, c_before, f"VANILLA pages of layer {layer} (built-in)")


def test_vanilla_block_replays_under_capture(vanilla_pair):
    inputs = _block(2, 7, seed=95)
    eager = vanilla_pair.new.dflash_forward(**_copy(inputs))
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = vanilla_pair.new.dflash_forward(**inputs)
    graph.replay()
    torch.cuda.synchronize()
    _assert_bitwise(captured, eager, "replayed VANILLA block")


def _arena(batch, block, seed):
    """The block's inputs on a private arena ([slots, layers, columns, 1, 576], no page table)."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    ctx = torch.tensor([CTX[i % len(CTX)] for i in range(batch)], dtype=torch.int32, device="cuda")
    slots, columns = batch + 3, int(ctx.max()) + block + 5
    return dict(
        noise_embedding=torch.randn(batch, block, HIDDEN, generator=g, device="cuda").to(
            torch.bfloat16
        ),
        query_positions=(ctx.long()[:, None] + torch.arange(block, device="cuda")),
        num_ctx_per_req=ctx,
        ctx_k_cache=torch.randn(
            slots, LAYERS, columns, 1, KV_LORA + ROPE, generator=g, device="cuda"
        ).to(torch.bfloat16),
        ctx_v_cache=None,
        ctx_cache_batch_idx=torch.randperm(slots, generator=g, device="cuda")[:batch].to(
            torch.int32
        ),
        ctx_kv_cache=None,
        ctx_page_table=None,
    )


@pytest.mark.parametrize("backend", ["CUTEDSL", "VANILLA", "TRTLLM"])
@pytest.mark.parametrize("split", [(1, 7), (3, 8)], ids=str)
def test_arena_block_matches_main(pair, vanilla_pair, request, backend, split):
    """Without a page table every backend decodes eagerly over the arena, as the built-in drafter does."""
    if backend == "TRTLLM" and _TRTLLM_REASON is not None:
        pytest.skip(f"TRTLLM needs flashinfer's MLA decode: {_TRTLLM_REASON}")
    drafters = {"CUTEDSL": pair, "VANILLA": vanilla_pair}.get(backend) or request.getfixturevalue(
        "trtllm_pair"
    )
    assert drafters.new._block_decode_variant(False) == "eager"
    inputs = _arena(*split, seed=split[0] * 10 + split[1] + 11)
    old_inputs = dict(inputs, ctx_k_cache=inputs["ctx_k_cache"].clone())
    out = drafters.new.dflash_forward(**inputs)
    ref = drafters.old.dflash_forward(**old_inputs)
    torch.cuda.synchronize()
    _assert_bitwise(out, ref, f"{backend} arena block {split}")


# ----------------------------------------------------------------------------------------------------------------------
# attention_backend TRTLLM: the built-in flashinfer TRTLLM-Gen decode with the upper-triangle fixup.
# ----------------------------------------------------------------------------------------------------------------------

_TRTLLM_REASON = k3_dspark_mla.mla_paged_decode_unavailability_reason()
_needs_trtllm = pytest.mark.skipif(
    _TRTLLM_REASON is not None, reason=f"TRTLLM needs flashinfer's MLA decode: {_TRTLLM_REASON}"
)


@pytest.fixture(scope="module")
def trtllm_pair():
    return SimpleNamespace(
        new=_load(target.K3MLADSparkDrafter, "TRTLLM"), old=_load(MLADSparkForCausalLM, "TRTLLM")
    )


@_needs_trtllm
def test_trtllm_resolves_like_main(trtllm_pair):
    assert (
        trtllm_pair.new.dflash_attention_backend
        == trtllm_pair.old.dflash_attention_backend
        == "TRTLLM"
    )
    assert trtllm_pair.new._block_decode_variant(True) == "trtllm_gen"
    assert trtllm_pair.old._mla_block_decode_variant(True) == "trtllm_gen"


@_needs_trtllm
@pytest.mark.parametrize("split", [(1, 7), (2, 7), (8, 7), (3, 8), (4, 4), (16, 7)], ids=str)
def test_trtllm_block_matches_main(trtllm_pair, split):
    """The block forward and the block latents it leaves in the context pages, bit for bit."""
    batch, block = split
    inputs = _block(batch, block, seed=batch * 10 + block + 5)
    old_inputs = _copy(inputs)
    out = trtllm_pair.new.dflash_forward(**inputs)
    ref = trtllm_pair.old.dflash_forward(**old_inputs)
    torch.cuda.synchronize()
    _assert_bitwise(out, ref, f"TRTLLM block {split}")
    for layer, (c_new, c_old) in enumerate(zip(inputs["ctx_kv_cache"], old_inputs["ctx_kv_cache"])):
        _assert_bitwise(c_new, c_old, f"TRTLLM pages of layer {layer} after {split}")


@_needs_trtllm
def test_trtllm_block_replays_under_capture(trtllm_pair):
    inputs = _block(2, 7, seed=97)
    eager = trtllm_pair.new.dflash_forward(**_copy(inputs))
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = trtllm_pair.new.dflash_forward(**inputs)
    graph.replay()
    torch.cuda.synchronize()
    _assert_bitwise(captured, eager, "replayed TRTLLM block")
