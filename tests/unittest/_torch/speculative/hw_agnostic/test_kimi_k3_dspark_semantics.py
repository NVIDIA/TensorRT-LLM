# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kimi K3 DSpark drafter-forward semantics (MR A).

Validates the weights-independent dspark math against a port of the
DeepSpec reference (github.com/deepseek-ai/DeepSpec):
  - vanilla Markov intra-block logit bias (deepspec/modeling/dspark/
    markov_head.py VanillaMarkov, greedy chain of sample_block_tokens),
  - the shift_label output convention (block slot j predicts draft token
    j+1; deepspec/eval/dspark/draft_ops.py build_dspark_proposal),
  - sliding-window attention on 'sliding_attention' draft layers
    (HF flash translation: window_size = (w-1, w-1), non-causal),
and the no-regression property: a dflash_config WITHOUT dspark fields
resolves to the exact pre-dspark behavior (slots 1..K, no window, no
Markov bias). CPU-only where possible; the tiny end-to-end block-decode
parity test needs CUDA + flash_attn and is skip-guarded.

Confidence-scheduled verification is MR B: here we only check that
confidence_proj weights load without being used.
"""

import pytest
import torch
import torch.nn.functional as F

from tensorrt_llm._torch.models.modeling_dflash import DFlashForCausalLM, dspark_layer_window_size
from tensorrt_llm._torch.models.modeling_dspark import Qwen3DSparkForCausalLM
from tensorrt_llm._torch.models.modeling_speculative import (
    dspark_markov_chain_logits,
    dspark_markov_step_bias,
)
from tensorrt_llm._torch.speculative.dflash import dflash_draft_slot_ids

# ---------------------------------------------------------------------------
# Reference oracle: line-for-line port of DeepSpec VanillaMarkov
# (deepspec/modeling/dspark/markov_head.py) at temperature 0.
# ---------------------------------------------------------------------------


class _RefVanillaMarkov:
    def __init__(self, markov_w1: torch.Tensor, markov_w2: torch.Tensor):
        # markov_w1: nn.Embedding(vocab, rank).weight  -> [vocab, rank]
        # markov_w2: nn.Linear(rank, vocab, bias=False).weight -> [vocab, rank]
        self.w1 = markov_w1
        self.w2 = markov_w2

    def compute_step_bias(self, token_ids: torch.Tensor) -> torch.Tensor:
        # markov_w2(markov_w1(ids))
        return F.linear(F.embedding(token_ids.long(), self.w1), self.w2)

    def sample_block_tokens(self, base_logits: torch.Tensor, first_prev_token_ids: torch.Tensor):
        """Greedy (temperature 0) reference chain."""
        sampled, corrected = [], []
        prev = first_prev_token_ids.long()
        for step in range(base_logits.shape[1]):
            step_logits = base_logits[:, step, :] + self.compute_step_bias(prev)
            corrected.append(step_logits.unsqueeze(1))
            nxt = torch.argmax(step_logits, dim=-1)
            sampled.append(nxt)
            prev = nxt
        return torch.stack(sampled, dim=1), torch.cat(corrected, dim=1)


VOCAB, RANK, B, K = 512, 16, 3, 7


def _random_markov(seed=1234, dtype=torch.float32):
    g = torch.Generator().manual_seed(seed)
    w1 = torch.randn(VOCAB, RANK, generator=g, dtype=dtype)
    w2 = torch.randn(VOCAB, RANK, generator=g, dtype=dtype)
    base = torch.randn(B, K, VOCAB, generator=g, dtype=dtype)
    anchor = torch.randint(0, VOCAB, (B,), generator=g)
    return w1, w2, base, anchor


def test_markov_step_bias_formula():
    """bias over vocab = markov_w1[prev] @ markov_w2.T (both [vocab, rank])."""
    w1, w2, _, anchor = _random_markov()
    bias = dspark_markov_step_bias(anchor, w1, w2)
    expected = w1[anchor] @ w2.T
    torch.testing.assert_close(bias, expected)


def test_markov_chain_matches_deepspec_reference():
    """Corrected block logits and the greedy token chain match the ported
    DeepSpec VanillaMarkov.sample_block_tokens bitwise (same dtype/ops)."""
    w1, w2, base, anchor = _random_markov()
    ref_tokens, ref_logits = _RefVanillaMarkov(w1, w2).sample_block_tokens(base, anchor)

    out = dspark_markov_chain_logits(base, anchor, w1, w2)
    assert torch.equal(out, ref_logits)
    # Greedy per-position argmax of the corrected logits reproduces the
    # reference sequentially-sampled chain (what sample_draft_tokens does).
    assert torch.equal(torch.argmax(out, dim=-1), ref_tokens)


def test_markov_chain_empty_block_is_noop():
    w1, w2, base, anchor = _random_markov()
    empty = base[:, :0, :]
    assert dspark_markov_chain_logits(empty, anchor, w1, w2) is empty


def test_markov_chain_sharded_matches_full_vocab():
    """The DFlashWorker TP path — every rank runs the chain on its
    contiguous markov_w2/logits vocab shard, chained through a global
    argmax over all shards — reassembles to the full-vocab chain."""
    w1, w2, base, anchor = _random_markov()
    full = dspark_markov_chain_logits(base, anchor, w1, w2)

    tp = 4
    shard_w = VOCAB // tp
    shards = [slice(r * shard_w, (r + 1) * shard_w) for r in range(tp)]

    # Lockstep emulation: per-rank shard bias, "TP gather" = global argmax
    # across the concatenated shards (what greedy_sample_draft_with_tp_gather
    # computes), returning full-vocab ids for the next markov_w1 lookup.
    prev = anchor.long()
    rank_outputs = [[] for _ in range(tp)]
    for i in range(K):
        step_shards = []
        for r, sl in enumerate(shards):
            bias = dspark_markov_step_bias(prev, w1, w2[sl])
            step = base[:, i, sl] + bias
            rank_outputs[r].append(step)
            step_shards.append(step)
        prev = torch.argmax(torch.cat(step_shards, dim=-1), dim=-1)

    reassembled = torch.cat([torch.stack(rank_outputs[r], dim=1) for r in range(tp)], dim=-1)
    torch.testing.assert_close(reassembled, full)


# ---------------------------------------------------------------------------
# shift_label slot convention
# ---------------------------------------------------------------------------


def test_slot_ids_plain_dflash_matches_old_formula():
    """No-regression: shift_label off reproduces the previous inline
    formula (mask slots 1..K)."""
    num_gens, block, k = 3, 8, 7
    ids = dflash_draft_slot_ids(num_gens, block, k, False, device="cpu")
    bases = torch.arange(num_gens, dtype=torch.long) * block
    offs = torch.arange(k, dtype=torch.long)
    old = (bases.unsqueeze(1) + 1 + offs.unsqueeze(0)).flatten()
    assert torch.equal(ids, old)


def test_slot_ids_shift_label_uses_anchor_slot():
    """DSpark shift_label: slots 0..K-1; slot 0 (anchor token slot)
    predicts the first draft token (DeepSpec build_dspark_proposal reads
    block_hidden[:, :block_size])."""
    ids = dflash_draft_slot_ids(2, 8, 8, True, device="cpu")
    assert ids.tolist() == list(range(8)) + [8 + j for j in range(8)]
    # With shift_label, K == block_size stays in range (plain would not).
    assert ids.max().item() == 2 * 8 - 1


# ---------------------------------------------------------------------------
# SWA window convention
# ---------------------------------------------------------------------------


def test_swa_window_conventions():
    sliding = ["sliding_attention", "full_attention"]
    # HF flash translation: window_size = (w-1, w-1) on sliding layers.
    assert dspark_layer_window_size(True, 1024, sliding, 0) == (1023, 1023)
    assert dspark_layer_window_size(True, 1024, sliding, 1) == (-1, -1)
    # use_swa off -> flash-attn default regardless of layer_types.
    assert dspark_layer_window_size(False, 1024, sliding, 0) == (-1, -1)
    # No layer_types declared + use_swa -> window on every layer.
    assert dspark_layer_window_size(True, 8, None, 0) == (7, 7)


# ---------------------------------------------------------------------------
# Tiny end-to-end drafter: config parsing, weight loading, block-decode
# parity vs an fp32 eager oracle (needs CUDA + flash_attn).
# ---------------------------------------------------------------------------

TINY = dict(
    architectures=["DFlashDraftModel"],
    model_type="qwen3",
    block_size=4,
    hidden_size=64,
    num_hidden_layers=2,
    num_attention_heads=4,
    num_key_value_heads=2,
    # 128 = the real K3 drafter head_dim; small head dims are rejected by
    # the fusedQKNormRope kernel the bf16 block decode uses.
    head_dim=128,
    intermediate_size=128,
    hidden_act="silu",
    rms_norm_eps=1e-6,
    vocab_size=VOCAB,
    max_position_embeddings=2048,
    rope_theta=10000.0,
    rope_scaling=None,
    attention_bias=False,
    torch_dtype="bfloat16",
    num_target_layers=4,
    tie_word_embeddings=False,
)

SWA_WINDOW = 8
CTX_LEN = 24  # > SWA_WINDOW so the window binds
NUM_CAPTURE = 2


def _tiny_config(dspark: bool):
    from transformers import Qwen3Config

    cfg = dict(TINY)
    dflash = {"mask_token_id": VOCAB - 2, "target_layer_ids": [0, 1]}
    if dspark:
        dflash.update(
            projector_type="dspark",
            causal=False,
            use_swa=True,
            swa_window_size=SWA_WINDOW,
            shift_label=True,
            markov_rank=RANK,
            markov_head_type="vanilla",
            use_confidence_head=True,
        )
        cfg["layer_types"] = ["sliding_attention"] * cfg["num_hidden_layers"]
        cfg["sliding_window"] = SWA_WINDOW
    cfg["dflash_config"] = dflash
    return Qwen3Config.from_dict(cfg)


def _tiny_weights(seed=7):
    g = torch.Generator().manual_seed(seed)

    def rnd(*shape):
        return (torch.randn(*shape, generator=g) * 0.05).to(torch.bfloat16)

    h, inter = TINY["hidden_size"], TINY["intermediate_size"]
    nh, nkv, hd = (TINY["num_attention_heads"], TINY["num_key_value_heads"], TINY["head_dim"])
    w = {
        "fc.weight": rnd(h, h * NUM_CAPTURE),
        "hidden_norm.weight": rnd(h) + 1.0,
        "norm.weight": rnd(h) + 1.0,
        "markov_w1.weight": rnd(VOCAB, RANK),
        "markov_w2.weight": rnd(VOCAB, RANK),
        "confidence_proj.weight": rnd(1, h + RANK),
        "confidence_proj.bias": rnd(1),
    }
    for i in range(TINY["num_hidden_layers"]):
        p = f"layers.{i}."
        w[p + "self_attn.q_proj.weight"] = rnd(nh * hd, h)
        w[p + "self_attn.k_proj.weight"] = rnd(nkv * hd, h)
        w[p + "self_attn.v_proj.weight"] = rnd(nkv * hd, h)
        w[p + "self_attn.o_proj.weight"] = rnd(h, nh * hd)
        w[p + "self_attn.q_norm.weight"] = rnd(hd) + 1.0
        w[p + "self_attn.k_norm.weight"] = rnd(hd) + 1.0
        w[p + "input_layernorm.weight"] = rnd(h) + 1.0
        w[p + "post_attention_layernorm.weight"] = rnd(h) + 1.0
        w[p + "mlp.gate_proj.weight"] = rnd(inter, h)
        w[p + "mlp.up_proj.weight"] = rnd(inter, h)
        w[p + "mlp.down_proj.weight"] = rnd(h, inter)
    return w


def _build_drafter(dspark: bool, weights):
    from tensorrt_llm._torch.model_config import ModelConfig

    model_config = ModelConfig(pretrained_config=_tiny_config(dspark), attn_backend="TRTLLM")
    # The DSpark head set lives in the DSpark drafter, not in the DFlash base.
    drafter_cls = Qwen3DSparkForCausalLM if dspark else DFlashForCausalLM
    drafter = drafter_cls(model_config).to("cuda")
    # Drop dspark head tensors for the plain drafter (schema without them).
    if not dspark:
        weights = {k: v for k, v in weights.items() if not k.startswith(("markov_", "confidence_"))}
    drafter.load_weights(dict(weights))
    return drafter


def _rms(x, w, eps=1e-6):
    xf = x.float()
    return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)) * w.float()


def _rope(x, positions, theta=10000.0):
    # NeoX half-split convention, matching RotaryEmbedding(is_neox=True).
    # x: [T, heads, hd]; positions: [T]
    hd = x.shape[-1]
    inv = 1.0 / theta ** (torch.arange(0, hd, 2, dtype=torch.float64) / hd)
    ang = positions.double().unsqueeze(-1) * inv  # [T, hd/2]
    cos = ang.cos().float().unsqueeze(1)
    sin = ang.sin().float().unsqueeze(1)
    x1, x2 = x[..., : hd // 2], x[..., hd // 2 :]
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)


def _oracle_block_decode(weights, captured, noise_embed, use_swa):
    """fp32 eager port of the DeepSpec dspark block decode
    (Qwen3DSparkDecoderLayer stack over [context ; draft block])."""
    w = {k: v.float() for k, v in weights.items()}
    nh, nkv, hd = (TINY["num_attention_heads"], TINY["num_key_value_heads"], TINY["head_dim"])
    ctx = captured.shape[0]
    blk = noise_embed.shape[0]
    ctx_pos = torch.arange(ctx, dtype=torch.long)
    q_pos = torch.arange(ctx, ctx + blk, dtype=torch.long)
    all_pos = torch.cat([ctx_pos, q_pos])

    # Target feature projection: hidden_norm(fc(captured)); constant across
    # layers, no input_layernorm on the context path (generic DFlash).
    ctx_feat = _rms(captured.float() @ w["fc.weight"].T, w["hidden_norm.weight"])

    hs = noise_embed.float()
    for i in range(TINY["num_hidden_layers"]):
        p = f"layers.{i}."
        h = _rms(hs, w[p + "input_layernorm.weight"])
        q = (h @ w[p + "self_attn.q_proj.weight"].T).view(blk, nh, hd)
        k_ctx = (ctx_feat @ w[p + "self_attn.k_proj.weight"].T).view(ctx, nkv, hd)
        k_noise = (h @ w[p + "self_attn.k_proj.weight"].T).view(blk, nkv, hd)
        v_ctx = (ctx_feat @ w[p + "self_attn.v_proj.weight"].T).view(ctx, nkv, hd)
        v_noise = (h @ w[p + "self_attn.v_proj.weight"].T).view(blk, nkv, hd)
        k = torch.cat([k_ctx, k_noise], dim=0)
        v = torch.cat([v_ctx, v_noise], dim=0)
        q = _rms(q, w[p + "self_attn.q_norm.weight"])
        k = _rms(k, w[p + "self_attn.k_norm.weight"])
        q = _rope(q, q_pos)
        k = _rope(k, all_pos)
        # GQA expand
        rep = nh // nkv
        k = k.repeat_interleave(rep, dim=1)
        v = v.repeat_interleave(rep, dim=1)
        scores = torch.einsum("qhd,khd->hqk", q, k) / hd**0.5
        if use_swa:
            dist = (q_pos.unsqueeze(1) - all_pos.unsqueeze(0)).abs()
            scores = scores.masked_fill(dist.unsqueeze(0) > SWA_WINDOW - 1, float("-inf"))
        attn = torch.softmax(scores, dim=-1)
        o = torch.einsum("hqk,khd->qhd", attn, v).reshape(blk, nh * hd)
        hs = hs + o @ w[p + "self_attn.o_proj.weight"].T
        h2 = _rms(hs, w[p + "post_attention_layernorm.weight"])
        gate = h2 @ w[p + "mlp.gate_proj.weight"].T
        up = h2 @ w[p + "mlp.up_proj.weight"].T
        hs = hs + (F.silu(gate) * up) @ w[p + "mlp.down_proj.weight"].T
    return _rms(hs, w["norm.weight"])


def _has_flash_attn():
    try:
        import flash_attn  # noqa: F401

        return True
    except ImportError:
        return False


needs_gpu = pytest.mark.skipif(
    not torch.cuda.is_available() or not _has_flash_attn(),
    reason="tiny block-decode parity needs CUDA + flash_attn",
)


@needs_gpu
def test_dspark_drafter_loads_head_weights_and_parses_config():
    weights = _tiny_weights()
    drafter = _build_drafter(True, weights)
    assert drafter._dspark_shift_label and drafter._use_swa
    assert drafter._swa_window == SWA_WINDOW
    assert drafter._layer_windows == [(SWA_WINDOW - 1, SWA_WINDOW - 1)] * 2
    assert drafter.has_markov_head
    torch.testing.assert_close(drafter.markov_w1.cpu(), weights["markov_w1.weight"])
    torch.testing.assert_close(drafter.markov_w2.cpu(), weights["markov_w2.weight"])
    # Confidence weights loaded for MR B, but never consumed here.
    torch.testing.assert_close(
        drafter.confidence_proj_weight.cpu(), weights["confidence_proj.weight"]
    )
    assert drafter.confidence_proj_bias is not None


@needs_gpu
def test_plain_dflash_drafter_keeps_old_gates():
    """No-regression: a config WITHOUT dspark fields resolves to the exact
    old code path (no window, no markov, mask slots 1..K).

    The DFlash base no longer carries the DSpark head set at all, so the
    assertions are that those attributes are absent rather than inert.
    """
    drafter = _build_drafter(False, _tiny_weights())
    assert not drafter._use_swa
    assert drafter._layer_windows == [(-1, -1)] * 2
    for absent in (
        "_dspark_shift_label",
        "has_markov_head",
        "markov_w1",
        "confidence_proj_weight",
        "apply_markov_chain_logits",
    ):
        assert not hasattr(drafter, absent), f"DFlash base still carries {absent}"


@needs_gpu
def test_legacy_causal_dflash_config_constructs():
    """No-regression: legacy DFlash drafter configs (e.g. Laguna) declare
    causal=true without any dspark fields; their causality is handled by
    the legacy decode path, so construction must not raise."""
    from tensorrt_llm._torch.model_config import ModelConfig

    cfg = _tiny_config(False)
    cfg.dflash_config = dict(cfg.dflash_config, causal=True)
    drafter = DFlashForCausalLM(ModelConfig(pretrained_config=cfg, attn_backend="TRTLLM"))
    assert drafter._layer_windows == [(-1, -1)] * 2
    assert not hasattr(drafter, "has_markov_head")


@needs_gpu
def test_dspark_causal_config_rejected():
    """The dspark block decode only supports the non-causal convention."""
    from tensorrt_llm._torch.model_config import ModelConfig

    cfg = _tiny_config(True)
    cfg.dflash_config = dict(cfg.dflash_config, causal=True)
    with pytest.raises(ValueError, match="non-causal DSpark convention"):
        Qwen3DSparkForCausalLM(ModelConfig(pretrained_config=cfg, attn_backend="TRTLLM"))


@needs_gpu
def test_dspark_projector_type_alone_rejects_causal():
    """projector_type='dspark' marks the dspark convention even when no
    dspark feature flag is enabled; causal=true must still be rejected."""
    from tensorrt_llm._torch.model_config import ModelConfig

    cfg = _tiny_config(False)
    cfg.dflash_config = dict(cfg.dflash_config, projector_type="dspark", causal=True)
    with pytest.raises(ValueError, match="non-causal DSpark convention"):
        Qwen3DSparkForCausalLM(ModelConfig(pretrained_config=cfg, attn_backend="TRTLLM"))


def _run_block_decode(drafter, weights, captured, noise_embed):
    dev = "cuda"
    blk = TINY["block_size"]
    proj = drafter.project_target_hidden(captured.to(dev, torch.bfloat16))
    ctx_pos = torch.arange(CTX_LEN, device=dev)
    k, v = drafter.precompute_context_kv(proj, ctx_pos)
    L = drafter._num_attn_layers
    nkv, hd = drafter._num_kv_heads, drafter._head_dim
    pool_k = torch.zeros(1, L, CTX_LEN + blk, nkv, hd, dtype=torch.bfloat16, device=dev)
    pool_v = torch.zeros_like(pool_k)
    pool_k[0, :, :CTX_LEN] = k.permute(1, 0, 2, 3)
    pool_v[0, :, :CTX_LEN] = v.permute(1, 0, 2, 3)
    q_pos = torch.arange(CTX_LEN, CTX_LEN + blk, device=dev).unsqueeze(0)
    out = drafter.dflash_forward(
        noise_embedding=noise_embed.to(dev, torch.bfloat16).unsqueeze(0),
        query_positions=q_pos,
        num_ctx_per_req=torch.tensor([CTX_LEN], device=dev),
        ctx_k_cache=pool_k,
        ctx_v_cache=pool_v,
        ctx_cache_batch_idx=torch.tensor([0], device=dev),
    )
    return out.float().cpu()


@needs_gpu
def test_dspark_block_decode_matches_reference_oracle():
    """The full drafter block decode (fc/hidden_norm projection, per-layer
    QKV + q/k-norm + RoPE, non-causal SWA flash attention over
    [context ; block], MLP, final norm) matches the fp32 eager oracle; the
    no-window oracle does NOT match (the window demonstrably binds)."""
    torch.manual_seed(0)
    weights = _tiny_weights()
    drafter = _build_drafter(True, weights)

    g = torch.Generator().manual_seed(42)
    captured = torch.randn(CTX_LEN, TINY["hidden_size"] * NUM_CAPTURE, generator=g) * 0.5
    noise_embed = torch.randn(TINY["block_size"], TINY["hidden_size"], generator=g) * 0.5

    out = _run_block_decode(drafter, weights, captured, noise_embed)

    # Oracle consumes the same bf16-quantized inputs the drafter sees.
    captured_q = captured.to(torch.bfloat16)
    noise_q = noise_embed.to(torch.bfloat16)
    oracle_swa = _oracle_block_decode(weights, captured_q, noise_q, True)
    oracle_full = _oracle_block_decode(weights, captured_q, noise_q, False)

    diff_swa = (out - oracle_swa).abs().max().item()
    diff_full = (out - oracle_full).abs().max().item()
    # bf16 forward vs fp32 oracle: tolerance well below the SWA-vs-full gap.
    assert diff_swa < 0.02, f"SWA parity failed: max abs diff {diff_swa}"
    assert diff_full > 4 * max(diff_swa, 1e-4), (
        f"negative control failed: no-window oracle too close "
        f"({diff_full} vs swa {diff_swa}) — window may not be applied"
    )


@needs_gpu
def test_plain_dflash_block_decode_matches_full_attention_oracle():
    """No-regression numeric check: the plain-DFlash drafter (no dspark
    fields) still runs full non-causal attention over the whole context."""
    weights = _tiny_weights()
    drafter = _build_drafter(False, weights)
    g = torch.Generator().manual_seed(43)
    captured = torch.randn(CTX_LEN, TINY["hidden_size"] * NUM_CAPTURE, generator=g) * 0.5
    noise_embed = torch.randn(TINY["block_size"], TINY["hidden_size"], generator=g) * 0.5
    out = _run_block_decode(drafter, weights, captured, noise_embed)
    oracle = _oracle_block_decode(
        weights, captured.to(torch.bfloat16), noise_embed.to(torch.bfloat16), False
    )
    diff = (out - oracle).abs().max().item()
    assert diff < 0.02, f"plain DFlash parity failed: max abs diff {diff}"
