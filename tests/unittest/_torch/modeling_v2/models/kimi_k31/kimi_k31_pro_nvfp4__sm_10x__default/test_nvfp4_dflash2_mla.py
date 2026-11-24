# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The DFlash2 MLA drafter: the paged absorbed block decode (window, convs, norms, YaRN RoPE, softmax
scale) against an un-absorbed dense torch reference at block widths 5 and 8, the RoPE pair rotation, and
the window-attention kernel choice by SM."""

import math
from unittest.mock import patch

import pytest
import torch
from transformers import PretrainedConfig

from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.kimi_k31_pro_nvfp4__sm_10x__default import (
    modeling as k31_modeling,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.kimi_k31_pro_nvfp4__sm_10x__default.modeling import (  # noqa: E501
    MLADFlash2ForCausalLM,
    mla_rope_pairs,
)
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.models.modeling_dspark import _yarn_get_mscale
from tensorrt_llm.mapping import Mapping

# The checkpoint's rope_parameters.
_ROPE = dict(
    rope_type="yarn",
    rope_theta=50000.0,
    factor=32.0,
    original_max_position_embeddings=32768,
    beta_fast=32,
    beta_slow=1,
    mscale=1.0,
    mscale_all_dim=1.0,
)


def _yarn_reference(
    dim, base, factor, orig, beta_fast, beta_slow, mscale, mscale_all_dim, positions
):
    """DeepSeek YaRN cos / sin (the SGLang DeepseekScalingRotaryEmbedding formulas), FP64."""

    def corr_dim(rot):
        return dim * math.log(orig / (rot * 2 * math.pi)) / (2 * math.log(base))

    low = max(math.floor(corr_dim(beta_fast)), 0)
    high = min(math.ceil(corr_dim(beta_slow)), dim - 1)
    if low == high:
        high += 0.001
    ramp = ((torch.arange(dim // 2, dtype=torch.float64) - low) / (high - low)).clamp(0, 1)
    pos_freqs = base ** (torch.arange(0, dim, 2, dtype=torch.float64) / dim)
    extra, inter = 1.0 / pos_freqs, 1.0 / (factor * pos_freqs)
    mask = 1 - ramp
    inv_freq = inter * (1 - mask) + extra * mask

    def get_mscale(s, m):
        return 1.0 if s <= 1 else 0.1 * m * math.log(s) + 1.0

    scale = get_mscale(factor, mscale) / get_mscale(factor, mscale_all_dim)
    freqs = torch.outer(positions.double(), inv_freq)
    return freqs.cos() * scale, freqs.sin() * scale


def test_rope_pairs_is_interleaved_rotation():
    """Adjacent pairs (2i, 2i+1) rotate by angle i: the HF ``rope_interleave`` form, whose
    de-interleave permutation cancels in q . k."""
    torch.manual_seed(0)
    dim = 16
    x = torch.randn(3, dim)
    angles = torch.randn(3, dim // 2)
    freqs = torch.polar(torch.ones_like(angles), angles)
    got = mla_rope_pairs(x, freqs)
    ref = torch.empty_like(x)
    ref[:, 0::2] = x[:, 0::2] * angles.cos() - x[:, 1::2] * angles.sin()
    ref[:, 1::2] = x[:, 0::2] * angles.sin() + x[:, 1::2] * angles.cos()
    torch.testing.assert_close(got, ref)


# A small drafter of the checkpoint's structure.
_H, _NH, _QL, _KVL, _NOPE, _ROPE_D, _V, _I, _VOCAB, _L = 128, 6, 32, 512, 16, 64, 16, 256, 512, 2
# The window is a multiple of 4: the TRTLLM-Gen window attention reads index lists of 4k rows.
_WINDOW, _GROUP, _TAPS = 8, 16, 2


def _draft_config(block_size=5):
    cfg = PretrainedConfig(
        architectures=["DFlash2DraftModel"],
        hidden_size=_H,
        num_attention_heads=_NH,
        num_key_value_heads=_NH,
        q_lora_rank=_QL,
        kv_lora_rank=_KVL,
        qk_nope_head_dim=_NOPE,
        qk_rope_head_dim=_ROPE_D,
        v_head_dim=_V,
        intermediate_size=_I,
        hidden_act="silu",
        num_hidden_layers=_L,
        rms_norm_eps=1e-5,
        vocab_size=_VOCAB,
        max_position_embeddings=4096,
        rope_parameters=dict(_ROPE),
        rope_theta=50000.0,
        sliding_window=_WINDOW,
        use_sliding_window=True,
        layer_types=["sliding_attention"] * _L,
        is_causal=True,
        dflash_config=dict(
            attention_mode="mla",
            block_size=block_size,
            conv_kernel_size=_TAPS,
            conv_group_size=_GROUP,
            selector_rank=8,
            selector_top_k=4,
            mask_token_id=_VOCAB - 1,
            target_layer_ids=[0, 1],
        ),
    )
    cfg.model_type = "qwen3"
    cfg.torch_dtype = torch.bfloat16
    return ModelConfig(pretrained_config=cfg, mapping=Mapping(world_size=1, tp_size=1, rank=0))


def _weights(seed=0):
    g = torch.Generator().manual_seed(seed)

    def rnd(*shape, scale=None):
        scale = scale if scale is not None else 1.0 / math.sqrt(shape[-1])
        return (torch.randn(*shape, generator=g) * scale).to(torch.bfloat16)

    def norm_w(n):
        return (1.0 + 0.1 * torch.randn(n, generator=g)).to(torch.bfloat16)

    w = {}
    for i in range(_L):
        p = f"layers.{i}."
        w[p + "input_layernorm.weight"] = norm_w(_H)
        w[p + "post_attention_layernorm.weight"] = norm_w(_H)
        w[p + "self_attn.q_a_proj.weight"] = rnd(_QL, _H)
        w[p + "self_attn.q_a_layernorm.weight"] = norm_w(_QL)
        w[p + "self_attn.q_b_proj.weight"] = rnd(_NH * (_NOPE + _ROPE_D), _QL)
        w[p + "self_attn.kv_a_proj_with_mqa.weight"] = rnd(_KVL + _ROPE_D, _H)
        w[p + "self_attn.kv_a_layernorm.weight"] = norm_w(_KVL)
        w[p + "self_attn.kv_b_proj.weight"] = rnd(_NH * (_NOPE + _V), _KVL)
        w[p + "self_attn.o_proj.weight"] = rnd(_H, _NH * _V)
        w[p + "mlp.gate_proj.weight"] = rnd(_I, _H)
        w[p + "mlp.up_proj.weight"] = rnd(_I, _H)
        w[p + "mlp.down_proj.weight"] = rnd(_H, _I)
        for conv in ("attention_conv", "mlp_conv"):
            base = 0.2 * torch.randn(2, _TAPS, _H, generator=g)
            base[:, 0] += 1.0
            w[p + conv + ".base_kernel"] = base.to(torch.bfloat16)
            w[p + conv + ".kernel_projection.weight"] = rnd(
                2 * _TAPS * _H // _GROUP, _H, scale=0.02
            )
    w["fc.weight"] = rnd(_H, 2 * _H)
    w["hidden_norm.weight"] = norm_w(_H)
    w["norm.weight"] = norm_w(_H)
    w["candidate_selector.predecessor_codebook"] = rnd(_VOCAB, 8)
    w["candidate_selector.successor_codebook"] = rnd(_VOCAB, 8)
    w["candidate_selector.hidden_projection.weight"] = rnd(8, _H)
    return w


def _rms(x, w, eps=1e-5):
    x = x.float()
    return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + eps) * w.float()


def _conv(x, coef, base, block_size):
    """One block of the reference convolution (rows = block positions)."""
    out = torch.zeros_like(x)
    for t in range(x.shape[0]):
        for tau in range(min(_TAPS, t + 1)):
            k = base[tau].float() + coef[t, tau].float().repeat_interleave(_GROUP)
            out[t] += k * x[t - tau]
    return out


def _rope(x, cos, sin):
    out = torch.empty_like(x)
    out[..., 0::2] = x[..., 0::2] * cos - x[..., 1::2] * sin
    out[..., 1::2] = x[..., 0::2] * sin + x[..., 1::2] * cos
    return out


def _reference_block(w, noise, ctx_proj, ctx_len, block_size, scale, history="auto"):
    """Un-absorbed DFlash2 MLA drafter on one request, FP32: per-head k = [c W_UK | k_pe],
    v = c W_UV, causal window [p - W + 1, p] over context rows 0..L-1 and block rows L..p."""

    # FP8 rounding is discontinuous at bin boundaries, so its input must include
    # the BF16 activation rounding that precedes cache storage.
    def activation(x):
        return x.bfloat16().float() if history == "fp8" else x

    def linear(x, weight):
        return activation(x @ weight.T)

    def norm(x, weight):
        return activation(_rms(x, weight))

    L = ctx_len
    positions = torch.arange(L + block_size)
    cos, sin = _yarn_reference(_ROPE_D, 50000.0, 32.0, 32768, 32, 1, 1.0, 1.0, positions)
    cos, sin = cos.float().to(noise.device), sin.float().to(noise.device)
    x = noise.float()
    res = None
    for i in range(_L):
        p = f"layers.{i}."
        f = {k[len(p) :]: v.float().to(noise.device) for k, v in w.items() if k.startswith(p)}
        res = x if res is None else activation(x + res)
        h = norm(res, f["input_layernorm.weight"])
        coef = linear(h, f["attention_conv.kernel_projection.weight"]).view(
            block_size, 2, _TAPS, -1
        )
        h = activation(_conv(h, coef[:, 0], f["attention_conv.base_kernel"][0], block_size))
        q = norm(linear(h, f["self_attn.q_a_proj.weight"]), f["self_attn.q_a_layernorm.weight"])
        q = linear(q, f["self_attn.q_b_proj.weight"]).view(block_size, _NH, _NOPE + _ROPE_D)
        q_nope = q[..., :_NOPE]
        q_pe = _rope(q[..., _NOPE:], cos[L:, None], sin[L:, None])
        lat_ctx = linear(ctx_proj.float(), f["self_attn.kv_a_proj_with_mqa.weight"])
        lat_blk = linear(h, f["self_attn.kv_a_proj_with_mqa.weight"])
        lat = torch.cat([lat_ctx, lat_blk])  # [L + block, KVL + rope]
        c = _rms(lat[:, :_KVL], f["self_attn.kv_a_layernorm.weight"])
        k_pe = _rope(lat[:, _KVL:], cos, sin)
        if history == "fp8":
            # Both context and draft-block latents cross the unit-scale E4M3 cache.
            c = c.bfloat16().to(torch.float8_e4m3fn).float()
            k_pe = k_pe.bfloat16().to(torch.float8_e4m3fn).float()
        kv = (c @ f["self_attn.kv_b_proj.weight"].T).view(-1, _NH, _NOPE + _V)
        k_nope, v = kv[..., :_NOPE], kv[..., _NOPE:]
        s = torch.einsum("qhd,thd->hqt", q_nope, k_nope) + torch.einsum("qhd,td->hqt", q_pe, k_pe)
        qpos = torch.arange(L, L + block_size).view(-1, 1)
        t = torch.arange(L + block_size).view(1, -1)
        mask = (t <= qpos) & (t >= qpos - _WINDOW + 1)
        s = (s * scale).masked_fill(~mask.to(s.device), float("-inf"))
        o = torch.einsum("hqt,thv->qhv", torch.softmax(s, -1), v).reshape(block_size, -1)
        a = linear(activation(o), f["self_attn.o_proj.weight"])
        a = activation(_conv(a, coef[:, 1], f["attention_conv.base_kernel"][1], block_size))
        res = activation(a + res)
        h2 = norm(res, f["post_attention_layernorm.weight"])
        coef = linear(h2, f["mlp_conv.kernel_projection.weight"]).view(block_size, 2, _TAPS, -1)
        h2 = activation(_conv(h2, coef[:, 0], f["mlp_conv.base_kernel"][0], block_size))
        m = activation(
            torch.nn.functional.silu(linear(h2, f["mlp.gate_proj.weight"]))
            * linear(h2, f["mlp.up_proj.weight"])
        )
        m = linear(m, f["mlp.down_proj.weight"])
        x = activation(_conv(m, coef[:, 1], f["mlp_conv.base_kernel"][1], block_size))
    return norm(activation(x + res), w["norm.weight"].to(noise.device))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
@pytest.mark.parametrize("block_size", [5, 8])
@pytest.mark.parametrize("history", ["auto", "fp8"])
def test_block_decode_matches_unabsorbed_reference(block_size, history):
    """Paged absorbed decode == dense reference; the block width overrides the checkpoint's 5."""
    torch.manual_seed(0)
    device = torch.device("cuda")
    draft = MLADFlash2ForCausalLM(
        _draft_config(block_size=5), block_size=block_size, context_kv_dtype=history
    )
    assert draft.block_size == block_size
    w = _weights()
    draft.load_weights({k: v.clone() for k, v in w.items()})
    draft = draft.to(device)
    expected_scale = _yarn_get_mscale(32.0, 1.0) ** 2 / math.sqrt(_NOPE + _ROPE_D)
    assert abs(draft.softmax_scale - expected_scale) < 1e-9

    page_size = 32  # the TRTLLM-Gen window attention reads 32- or 64-token pages
    ctx_lens = [0, 3, 35]  # the last window crosses a page boundary
    batch = len(ctx_lens)
    max_pages = (max(ctx_lens) + block_size + page_size - 1) // page_size + 1
    table = torch.randperm(batch * max_pages, device=device).view(batch, max_pages)
    dim = _KVL + _ROPE_D
    pool_dtype = torch.float8_e4m3fn if history == "fp8" else torch.bfloat16
    pools = [
        torch.zeros(batch * max_pages, 1, 1, page_size, dim, dtype=pool_dtype, device=device)
        for _ in range(_L)
    ]
    ctx_projs = []
    for b, n in enumerate(ctx_lens):
        proj = torch.randn(n, _H, device=device).to(torch.bfloat16)
        ctx_projs.append(proj)
        if n == 0:
            continue
        lat, v = draft.precompute_context_kv(proj, torch.arange(n, device=device))
        assert v is None and lat.shape == (n, _L, 1, dim)
        pos = torch.arange(n, device=device)
        pages, offs = table[b, pos // page_size], pos % page_size
        for layer in range(_L):
            values = lat[:, layer, 0].to(pool_dtype).contiguous()
            storage = pools[layer]
            if history == "fp8":
                values, storage = values.view(torch.uint8), storage.view(torch.uint8)
            storage[:, 0, 0][pages, offs] = values
    # Stale garbage past every request's context, where the block latents land.
    noise = torch.randn(batch, block_size, _H, device=device).to(torch.bfloat16)
    ctx_len = torch.tensor(ctx_lens, device=device)
    query_positions = ctx_len.view(-1, 1) + torch.arange(block_size, device=device)
    out = draft.dflash_forward(
        noise_embedding=noise,
        query_positions=query_positions,
        num_ctx_per_req=ctx_len,
        ctx_k_cache=None,
        ctx_v_cache=None,
        ctx_cache_batch_idx=torch.arange(batch, device=device),
        ctx_kv_cache=pools,
        ctx_page_table=table.to(torch.int32),
    ).view(batch, block_size, _H)
    for b, n in enumerate(ctx_lens):
        ref = _reference_block(w, noise[b], ctx_projs[b], n, block_size, expected_scale, history)
        torch.testing.assert_close(out[b].float(), ref, atol=6e-2, rtol=5e-2)


@pytest.mark.parametrize("sm_version, trtllm_gen", [(103, True), (100, False), (107, False)])
def test_window_attention_kernel_is_chosen_once_by_sm(sm_version, trtllm_gen):
    """The drafter picks its window attention at construction: the TRTLLM-Gen op on SM 10.3, where
    it is certified, and its exact FP32 reference (sparse_mla_rows_eager) on every other SM."""
    with patch.object(k31_modeling, "get_sm_version", return_value=sm_version):
        draft = MLADFlash2ForCausalLM(_draft_config(block_size=5), block_size=5)
    assert draft._window_attn_trtllm_gen is trtllm_gen

    dim = _KVL + _ROPE_D
    query = torch.zeros(2, _NH, dim, dtype=torch.bfloat16)
    pool = torch.zeros(3, 32, dim, dtype=torch.bfloat16)
    rows = torch.zeros(2, _WINDOW, dtype=torch.int32)
    lengths = torch.ones(2, dtype=torch.int32)
    with (
        patch.object(k31_modeling, "k31_dflash2_window_attention") as trtllm_gen_op,
        patch.object(k31_modeling, "sparse_mla_rows_eager") as eager,
        patch.object(k31_modeling, "dflash2_window_attention_buffers", return_value=(None, None)),
        # The step does not look at the device again.
        patch.object(k31_modeling, "get_sm_version", side_effect=AssertionError("per-step query")),
    ):
        draft._block_attention(query, pool, rows, lengths)
    assert trtllm_gen_op.called is trtllm_gen
    assert eager.called is not trtllm_gen


@pytest.mark.parametrize("block_size", [5, 8])
@pytest.mark.parametrize("history", ["auto", "fp8"])
@torch.inference_mode()
def test_helix_draft_uses_transferred_shards(block_size, history):
    """Two-rank decode consumes CTX-owned pages, including empty shards and page boundaries."""
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.kimi_k31_pro_nvfp4__sm_10x__default.runtime.draft_attention import (  # noqa: E501
        HelixDraftShard,
    )
    from tensorrt_llm._torch.nccl_window_tensor_scope import nccl_window_tensor_scope
    from tensorrt_llm._utils import mpi_rank, mpi_world_size

    world, rank = mpi_world_size(), mpi_rank()
    if world != 2:
        pytest.skip("Run with two MPI GPU ranks")
    torch.cuda.set_device(rank)
    config = _draft_config(block_size)
    config._frozen = False
    config.mapping = Mapping(
        world_size=2,
        rank=rank,
        tp_size=1,
        cp_size=2,
        cp_config={"cp_type": "HELIX"},
        gpus_per_node=4,
    )
    draft = MLADFlash2ForCausalLM(config, block_size=block_size, context_kv_dtype=history).cuda()
    weights = _weights()
    draft.load_weights({k: v.clone() for k, v in weights.items()})
    source = MLADFlash2ForCausalLM(_draft_config(block_size), block_size=block_size).cuda()
    source.load_weights({k: v.clone() for k, v in weights.items()})
    torch.manual_seed(123)
    lengths = [0, 1, 31, 32, 65, 95]
    page_size = 32
    local_pages = 2
    tables = torch.arange(len(lengths) * local_pages, device="cuda").view(len(lengths), local_pages)
    pool_dtype = torch.float8_e4m3fn if history == "fp8" else torch.bfloat16
    pools = [
        torch.full(
            (len(lengths) * local_pages, 1, 1, page_size, _KVL + _ROPE_D),
            float("nan"),
            device="cuda",
            dtype=torch.bfloat16,
        ).to(pool_dtype)
        for _ in range(_L)
    ]
    noise = torch.randn(len(lengths), block_size, _H, device="cuda").bfloat16()
    expected = []
    shard = HelixDraftShard(world, rank, page_size)
    for row, length in enumerate(lengths):
        projected = torch.randn(length, _H, device="cuda").bfloat16()
        expected.append(
            _reference_block(
                weights, noise[row], projected, length, block_size, draft.softmax_scale, history
            )
        )
        if length:
            positions = torch.arange(length, device="cuda")
            latents, _ = source.precompute_context_kv(projected, positions)
            slots, owned = shard.local_slots(positions)
            slots = slots[owned]
            for layer, pool in enumerate(pools):
                values = latents[owned, layer, 0].to(pool_dtype).contiguous()
                if history == "fp8":
                    values, pool = values.view(torch.uint8), pool.view(torch.uint8)
                pool[tables[row, slots // page_size], 0, 0, slots % page_size] = values
    counts = torch.tensor(lengths, device="cuda")
    kwargs = dict(
        query_positions=counts[:, None] + torch.arange(block_size, device="cuda"),
        num_ctx_per_req=counts,
        ctx_k_cache=None,
        ctx_v_cache=None,
        ctx_cache_batch_idx=torch.arange(len(lengths), device="cuda"),
        ctx_kv_cache=pools,
        ctx_page_table=tables,
    )
    with nccl_window_tensor_scope(noise):
        actual = draft.dflash_forward(noise.clone(), **kwargs)
        torch.testing.assert_close(actual.float(), torch.cat(expected), atol=0.06, rtol=0.05)
        for pool in pools:
            pool.zero_()
        missing = draft.dflash_forward(noise.clone(), **kwargs)
        assert not torch.allclose(
            actual[2 * block_size :], missing[2 * block_size :], atol=0.01, rtol=0.01
        )
