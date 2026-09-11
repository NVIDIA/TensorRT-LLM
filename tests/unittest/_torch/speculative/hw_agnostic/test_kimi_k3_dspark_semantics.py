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

import math
import re
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from tensorrt_llm._torch.models.modeling_dflash import DFlashForCausalLM, dspark_layer_window_size
from tensorrt_llm._torch.models.modeling_dspark import GQADSparkForCausalLM
from tensorrt_llm._torch.models.modeling_speculative import (
    dspark_markov_chain_logits,
    dspark_markov_step_bias,
)
from tensorrt_llm._torch.speculative.dflash import dflash_draft_slot_ids

needs_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="drafter module construction needs CUDA"
)


def _has_absorbed_mla_kernel() -> bool:
    """SM100-family only: trtllm-gen has no absorbed-MLA decode below it.

    This directory is mapped to the CPU and H100 stages, so a plain
    torch.cuda.is_available() gate sends the paged variant to sm90, where the
    kernel does not exist -- it either raises inside flashinfer or falls back
    to eager and trips the branch assertion. Either way the failure is about
    the stage, not the code.
    """
    if not torch.cuda.is_available():
        return False
    from tensorrt_llm._utils import is_sm_100f

    return is_sm_100f()


needs_absorbed_mla = pytest.mark.skipif(
    not _has_absorbed_mla_kernel(),
    reason="paged absorbed-MLA decode needs an SM100-family GPU",
)

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


def _tiny_config(dspark: bool, *, published_spelling: bool = False):
    """Tiny drafter config.

    ``published_spelling`` reproduces the two public K3 DSpark checkpoints
    (RadixArk, Inferact): the head switches sit at the TOP level, the
    confidence flag is ``enable_confidence_head``, and neither ``shift_label``
    nor ``projector_type`` is declared at all -- shift_label rides on the
    DSpark default. Reading only ``dflash_config`` resolves markov_rank to 0
    there, which drops the heads without raising.
    """
    from transformers import Qwen3Config

    cfg = dict(TINY)
    dflash = {"mask_token_id": VOCAB - 2, "target_layer_ids": [0, 1]}
    if dspark and published_spelling:
        cfg.update(
            markov_rank=RANK,
            markov_head_type="vanilla",
            enable_confidence_head=True,
            confidence_head_with_markov=True,
        )
        cfg["dflash_config"] = dflash
        return Qwen3Config.from_dict(cfg)
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


def _tiny_weights(seed=7, *, published_head_keys=False):
    """Tiny drafter weights.

    ``published_head_keys`` names the head tensors the way both public
    checkpoints ship them -- after the submodules that own them -- instead of
    the bare spellings the DSv4 stage weights use.
    """
    g = torch.Generator().manual_seed(seed)

    def rnd(*shape):
        return (torch.randn(*shape, generator=g) * 0.05).to(torch.bfloat16)

    h, inter = TINY["hidden_size"], TINY["intermediate_size"]
    nh, nkv, hd = (TINY["num_attention_heads"], TINY["num_key_value_heads"], TINY["head_dim"])
    head = {
        "markov_w1.weight": rnd(VOCAB, RANK),
        "markov_w2.weight": rnd(VOCAB, RANK),
        "confidence_proj.weight": rnd(1, h + RANK),
        "confidence_proj.bias": rnd(1),
    }
    if published_head_keys:
        head = {
            "markov_head.markov_w1.weight": head["markov_w1.weight"],
            "markov_head.markov_w2.weight": head["markov_w2.weight"],
            "confidence_head.proj.weight": head["confidence_proj.weight"],
            "confidence_head.proj.bias": head["confidence_proj.bias"],
        }
    w = {
        "fc.weight": rnd(h, h * NUM_CAPTURE),
        "hidden_norm.weight": rnd(h) + 1.0,
        "norm.weight": rnd(h) + 1.0,
        **head,
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


def _build_drafter(
    dspark: bool,
    weights,
    *,
    published_spelling: bool = False,
    dflash_attention_backend: str = "VANILLA",
):
    from tensorrt_llm._torch.model_config import ModelConfig

    model_config = ModelConfig(
        pretrained_config=_tiny_config(dspark, published_spelling=published_spelling),
        attn_backend="TRTLLM",
    )
    # The DSpark head set lives in the DSpark drafter, not in the DFlash base.
    drafter_cls = GQADSparkForCausalLM if dspark else DFlashForCausalLM
    drafter = drafter_cls(model_config, dflash_attention_backend=dflash_attention_backend).to(
        "cuda"
    )
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
def test_published_drafter_spelling_activates_the_heads():
    """Both public K3 DSpark checkpoints load with their heads live.

    They declare the switches at the top level and name the head tensors after
    the owning submodules. Reading only ``dflash_config`` and the bare tensor
    names resolves markov_rank to 0 and drops markov_w1/w2 on the floor:
    correct output, lower acceptance, nothing raised.
    """
    weights = _tiny_weights(published_head_keys=True)
    drafter = _build_drafter(True, weights, published_spelling=True)

    assert drafter.has_markov_head, "markov weights dropped despite being in the checkpoint"
    assert drafter._dspark_use_confidence_head, "enable_confidence_head spelling not resolved"
    # Declared nowhere in the published config, so it rides on the DSpark
    # default. False would run slots 1..K on a block_size-K drafter and read
    # the next request's anchor slot.
    assert drafter._dspark_shift_label
    torch.testing.assert_close(drafter.markov_w1.cpu(), weights["markov_head.markov_w1.weight"])
    assert drafter.confidence_proj_bias is not None


@needs_gpu
@needs_cuda
def test_head_weights_without_a_resolvable_rank_raise():
    """The inverse of the missing-weights check.

    A checkpoint that ships markov_w1/w2 while the rank resolves to 0 means the
    switches were spelled somewhere this build cannot read. Loading it anyway
    would silently cost acceptance, so it is an error.
    """
    from tensorrt_llm._torch.model_config import ModelConfig

    # dspark head weights, but a config that declares no head switches at all.
    model_config = ModelConfig(pretrained_config=_tiny_config(False), attn_backend="TRTLLM")
    drafter = GQADSparkForCausalLM(model_config).to("cuda")

    with pytest.raises(ValueError, match="markov_rank resolved to 0"):
        drafter.load_weights(_tiny_weights(published_head_keys=True))


def test_dflash_refuses_a_drafter_that_declares_the_dspark_heads():
    """``decoding_type: DFlash`` must reject a DSpark drafter, not degrade it.

    DFlash does not implement the Markov / confidence / shift_label semantics,
    so serving one here would lower the acceptance rate with no error and no
    way to attribute it back.
    """
    from tensorrt_llm._torch.models import modeling_dflash

    model_config = SimpleNamespace(
        spec_config=SimpleNamespace(attention_backend="TRTLLM", speculative_model="/nonexistent")
    )
    draft_config = SimpleNamespace(pretrained_config=_tiny_config(True))

    with pytest.raises(ValueError, match="DSpark"):
        modeling_dflash._build_dflash_draft(model_config, draft_config, None, None)


@pytest.mark.parametrize(
    "layers,expected",
    [
        # MLA-shaped: no per-head q/k/v projection to split at all.
        ([SimpleNamespace(self_attn=SimpleNamespace())], "qkv_proj"),
        # GQA-shaped but heterogeneous: the cross-layer K/V fusion needs one
        # uniform num_key_value_heads.
        (
            [
                SimpleNamespace(self_attn=SimpleNamespace(qkv_proj=object(), num_key_value_heads=n))
                for n in (8, 8, 4)
            ],
            "[2]",
        ),
    ],
    ids=["no_fused_qkv", "mismatched_kv_heads"],
)
def test_block_decode_rejects_a_backbone_it_cannot_express(layers, expected):
    """The GQA precondition replaced the old per-model_type whitelist.

    Without it the registry happily builds an unsupported backbone and the
    failure surfaces much later inside _build_fused_kv_buffers, or as silently
    mis-sliced weights.
    """
    drafter = DFlashForCausalLM.__new__(DFlashForCausalLM)
    drafter.model = SimpleNamespace(layers=layers)
    drafter.config = SimpleNamespace()

    with pytest.raises(ValueError, match=re.escape(expected)):
        drafter._validate_gqa_shape()


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
        GQADSparkForCausalLM(ModelConfig(pretrained_config=cfg, attn_backend="TRTLLM"))


@needs_gpu
def test_dspark_projector_type_alone_rejects_causal():
    """projector_type='dspark' marks the dspark convention even when no
    dspark feature flag is enabled; causal=true must still be rejected."""
    from tensorrt_llm._torch.model_config import ModelConfig

    cfg = _tiny_config(False)
    cfg.dflash_config = dict(cfg.dflash_config, projector_type="dspark", causal=True)
    with pytest.raises(ValueError, match="non-causal DSpark convention"):
        GQADSparkForCausalLM(ModelConfig(pretrained_config=cfg, attn_backend="TRTLLM"))


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


# ---------------------------------------------------------------------------
# MLA-backboned DSpark drafter (Inferact/Kimi-K3-DSpark shape).
#
# The GQA block decode cannot express it: there is no per-head K/V to fuse, and
# the cache holds one MLA latent per token instead of a K and a V half. What
# can break silently here is the absorption and the YaRN rope the drafter was
# distilled under, so both are checked against references written in the
# un-absorbed / HF form rather than against the same math again.
# ---------------------------------------------------------------------------

MLA_HIDDEN = 64
MLA_INTERMEDIATE = 32
MLA_HEADS = 4
MLA_Q_LORA = 16
# Real MLA head geometry, even though everything else here is tiny. Shrinking
# these silently disables both kernel paths: flashinfer's MLA templates are
# specialised on (kv_lora, rope) = (512, 64) and fail to compile at 16/4, and
# the fused RMSNorm/RoPE op requires rope_dim/2 % 32 == 0. With the old toy
# dims every "kernel" variant fell back to eager and the test proved nothing.
MLA_KV_LORA = 512
MLA_NOPE = 128
MLA_ROPE = 64
MLA_V_DIM = 128
MLA_LAYERS = 2
MLA_BLOCK = 3
MLA_LATENT = MLA_KV_LORA + MLA_ROPE
MLA_THETA = 50000.0
MLA_FACTOR = 4.0
MLA_ORIG_MAX = 64


def _tiny_mla_config():
    """Tiny MLA drafter config in the published (TorchSpec) spelling.

    Mirrors Inferact/Kimi-K3-DSpark: an architecture label no registry knows,
    ``model_type`` "k3_dspark", head switches and ``target_layer_ids`` at the
    top level, and YaRN under the transformers-v5 ``rope_parameters`` key.
    """
    from transformers import PretrainedConfig

    cfg = PretrainedConfig(
        architectures=["K3DSparkModel"],
        model_type="k3_dspark",
        hidden_size=MLA_HIDDEN,
        intermediate_size=MLA_INTERMEDIATE,
        num_hidden_layers=MLA_LAYERS,
        num_attention_heads=MLA_HEADS,
        num_key_value_heads=MLA_HEADS,
        q_lora_rank=MLA_Q_LORA,
        kv_lora_rank=MLA_KV_LORA,
        qk_nope_head_dim=MLA_NOPE,
        qk_rope_head_dim=MLA_ROPE,
        v_head_dim=MLA_V_DIM,
        vocab_size=VOCAB,
        rms_norm_eps=1e-6,
        max_position_embeddings=256,
        rope_theta=MLA_THETA,
        block_size=MLA_BLOCK,
        mask_token_id=VOCAB - 3,
        target_layer_ids=[0, 1],
        markov_rank=RANK,
        markov_head_type="vanilla",
        enable_confidence_head=True,
        confidence_head_with_markov=True,
        tie_word_embeddings=False,
        rope_parameters={
            "rope_type": "yarn",
            "factor": MLA_FACTOR,
            "original_max_position_embeddings": MLA_ORIG_MAX,
            "rope_theta": MLA_THETA,
            "beta_fast": 32,
            "beta_slow": 1,
            "mscale": 1.0,
            "mscale_all_dim": 1.0,
        },
    )
    cfg.dflash_config = {"mask_token_id": VOCAB - 3, "target_layer_ids": [0, 1]}
    cfg.torch_dtype = torch.bfloat16
    return cfg


def _tiny_mla_weights(seed=11):
    g = torch.Generator().manual_seed(seed)

    def rnd(*shape):
        return (torch.randn(*shape, generator=g) * 0.05).to(torch.bfloat16)

    qk_head_dim = MLA_NOPE + MLA_ROPE
    w = {
        # TorchSpec spellings, deliberately not SpecForge's fc/hidden_norm/norm.
        "context_proj.weight": rnd(MLA_HIDDEN, MLA_HIDDEN * NUM_CAPTURE),
        "context_norm.weight": rnd(MLA_HIDDEN).abs() + 1.0,
        "final_norm.weight": rnd(MLA_HIDDEN).abs() + 1.0,
        "embed_tokens.weight": rnd(VOCAB, MLA_HIDDEN),
        "markov_head.markov_w1.weight": rnd(VOCAB, RANK),
        "markov_head.markov_w2.weight": rnd(VOCAB, RANK),
        "confidence_head.proj.weight": rnd(1, MLA_HIDDEN + RANK),
        "confidence_head.proj.bias": rnd(1),
    }
    for i in range(MLA_LAYERS):
        p = f"layers.{i}."
        w[p + "input_layernorm.weight"] = rnd(MLA_HIDDEN).abs() + 1.0
        w[p + "post_attention_layernorm.weight"] = rnd(MLA_HIDDEN).abs() + 1.0
        w[p + "self_attn.q_a_proj.weight"] = rnd(MLA_Q_LORA, MLA_HIDDEN)
        w[p + "self_attn.q_a_layernorm.weight"] = rnd(MLA_Q_LORA).abs() + 1.0
        w[p + "self_attn.q_b_proj.weight"] = rnd(MLA_HEADS * qk_head_dim, MLA_Q_LORA)
        w[p + "self_attn.kv_a_proj_with_mqa.weight"] = rnd(MLA_LATENT, MLA_HIDDEN)
        w[p + "self_attn.kv_a_layernorm.weight"] = rnd(MLA_KV_LORA).abs() + 1.0
        w[p + "self_attn.kv_b_proj.weight"] = rnd(MLA_HEADS * (MLA_NOPE + MLA_V_DIM), MLA_KV_LORA)
        w[p + "self_attn.o_proj.weight"] = rnd(MLA_HIDDEN, MLA_HEADS * MLA_V_DIM)
        w[p + "mlp.gate_proj.weight"] = rnd(MLA_INTERMEDIATE, MLA_HIDDEN)
        w[p + "mlp.up_proj.weight"] = rnd(MLA_INTERMEDIATE, MLA_HIDDEN)
        w[p + "mlp.down_proj.weight"] = rnd(MLA_HIDDEN, MLA_INTERMEDIATE)
    return w


def _build_mla_drafter(weights, *, dflash_attention_backend="VANILLA"):
    from tensorrt_llm._torch.model_config import ModelConfig
    from tensorrt_llm._torch.models.modeling_dspark import MLADSparkForCausalLM

    model_config = ModelConfig(pretrained_config=_tiny_mla_config(), attn_backend="TRTLLM")
    drafter = MLADSparkForCausalLM(
        model_config, dflash_attention_backend=dflash_attention_backend
    ).to("cuda")
    drafter.load_weights(dict(weights))
    return drafter


def _hf_yarn_reference(dim, base, factor, original_max, max_pos, beta_fast=32.0, beta_slow=1.0):
    """Line-for-line port of HF ``DeepseekV3YarnRotaryEmbedding`` (mscale 1)."""

    def find_dim(num_rotations):
        return (dim * math.log(original_max / (num_rotations * 2 * math.pi))) / (2 * math.log(base))

    low = max(math.floor(find_dim(beta_fast)), 0)
    high = min(math.ceil(find_dim(beta_slow)), dim - 1)
    if low == high:
        high += 0.001
    freq_extra = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    freq_inter = 1.0 / (factor * base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    ramp = torch.clamp((torch.arange(dim // 2, dtype=torch.float32) - low) / (high - low), 0, 1)
    inv_freq = freq_inter * ramp + freq_extra * (1 - ramp)
    freqs = torch.outer(torch.arange(max_pos, dtype=torch.float32), inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos(), emb.sin()


def _hf_apply_rope(x, cos, sin):
    """HF ``DeepseekV3`` rope: de-interleave, then the half-split rotation."""
    d = x.shape[-1]
    x = x.reshape(*x.shape[:-1], d // 2, 2).transpose(-1, -2).reshape(*x.shape[:-1], d)
    x1, x2 = x[..., : d // 2], x[..., d // 2 :]
    return x * cos + torch.cat((-x2, x1), dim=-1) * sin


def test_mla_dspark_yarn_rope_matches_hf_reference():
    """The drafter's YaRN table and rotation are HF DeepSeek's.

    Both published MLA drafters are distilled under HF/vLLM numerics. Routing
    through ``RopeParams`` instead picks up the fused MLA kernel's
    ``duplicate_data`` / GPT-J packing convention, which rotates differently and
    costs acceptance with nothing raised.
    """
    from tensorrt_llm._torch.models.modeling_dspark import (
        apply_dspark_mla_rope,
        build_dspark_mla_yarn_rope,
    )

    dim, max_pos = 8, 32
    cos, sin = build_dspark_mla_yarn_rope(
        dim=dim,
        base=MLA_THETA,
        scaling_factor=MLA_FACTOR,
        original_max_position_embeddings=MLA_ORIG_MAX,
        max_position_embeddings=max_pos,
        beta_fast=32.0,
        beta_slow=1.0,
        mscale=1.0,
        mscale_all_dim=1.0,
        device="cpu",
    )
    ref_cos, ref_sin = _hf_yarn_reference(dim, MLA_THETA, MLA_FACTOR, MLA_ORIG_MAX, max_pos)
    torch.testing.assert_close(cos, ref_cos)
    torch.testing.assert_close(sin, ref_sin)

    g = torch.Generator().manual_seed(3)
    x = torch.randn(5, 3, dim, generator=g)
    pos = torch.tensor([0, 7, 31])
    torch.testing.assert_close(
        apply_dspark_mla_rope(x, cos[pos], sin[pos]),
        _hf_apply_rope(x, ref_cos[pos], ref_sin[pos]),
    )


def _oracle_mla_block_decode(weights, captured, noise_embed, *, swap_kv_b=False):
    """fp32 eager MLA block decode, written in the UN-ABSORBED form.

    The drafter attends the stored latent directly (``q_nope`` folded through
    ``kv_b_proj``'s K half); this expands the latent into per-head K/V and runs
    plain attention instead, so agreement is evidence about the absorption
    rather than a restatement of it. ``swap_kv_b`` is the negative control:
    exchanging the K and V halves must break parity.
    """
    from tensorrt_llm._torch.models.modeling_dspark import apply_dspark_mla_rope

    w = {k: v.float() for k, v in weights.items()}
    ctx, blk = captured.shape[0], noise_embed.shape[0]
    ctx_pos = torch.arange(ctx, dtype=torch.long)
    q_pos = torch.arange(ctx, ctx + blk, dtype=torch.long)
    cos, sin = _hf_yarn_reference(
        MLA_ROPE, MLA_THETA, MLA_FACTOR, MLA_ORIG_MAX, MLA_ORIG_MAX * int(MLA_FACTOR)
    )
    scale = _yarn_mscale_sq() / (MLA_NOPE + MLA_ROPE) ** 0.5

    # Context features: context_norm(context_proj(captured)); constant across
    # layers and NOT passed through input_layernorm (the DFlash contract).
    ctx_feat = _rms(captured.float() @ w["context_proj.weight"].T, w["context_norm.weight"])

    hs = noise_embed.float()
    for i in range(MLA_LAYERS):
        p = f"layers.{i}."
        h = _rms(hs, w[p + "input_layernorm.weight"])

        q = _rms(h @ w[p + "self_attn.q_a_proj.weight"].T, w[p + "self_attn.q_a_layernorm.weight"])
        q = (q @ w[p + "self_attn.q_b_proj.weight"].T).view(blk, MLA_HEADS, MLA_NOPE + MLA_ROPE)
        q_nope, q_rope = q.split([MLA_NOPE, MLA_ROPE], dim=-1)
        q_rope = apply_dspark_mla_rope(q_rope, cos[q_pos].unsqueeze(1), sin[q_pos].unsqueeze(1))

        def latent_of(x, pos):
            lat = x @ w[p + "self_attn.kv_a_proj_with_mqa.weight"].T
            return torch.cat(
                [
                    _rms(lat[..., :MLA_KV_LORA], w[p + "self_attn.kv_a_layernorm.weight"]),
                    apply_dspark_mla_rope(lat[..., MLA_KV_LORA:], cos[pos], sin[pos]),
                ],
                dim=-1,
            )

        kv = torch.cat([latent_of(ctx_feat, ctx_pos), latent_of(h, q_pos)], dim=0)
        kv_b = w[p + "self_attn.kv_b_proj.weight"].view(
            MLA_HEADS, MLA_NOPE + MLA_V_DIM, MLA_KV_LORA
        )
        k_b, v_b = kv_b[:, :MLA_NOPE], kv_b[:, MLA_NOPE:]
        if swap_kv_b:
            k_b, v_b = v_b, k_b
        c_kv, k_pe = kv[:, :MLA_KV_LORA], kv[:, MLA_KV_LORA:]
        k = torch.cat(
            [
                torch.einsum("tc,hdc->thd", c_kv, k_b),
                k_pe.unsqueeze(1).expand(-1, MLA_HEADS, -1),
            ],
            dim=-1,
        )
        v = torch.einsum("tc,hvc->thv", c_kv, v_b)

        scores = torch.einsum("qhd,khd->hqk", torch.cat([q_nope, q_rope], -1), k) * scale
        attn = torch.softmax(scores, dim=-1)
        o = torch.einsum("hqk,khv->qhv", attn, v).reshape(blk, MLA_HEADS * MLA_V_DIM)

        hs = hs + o @ w[p + "self_attn.o_proj.weight"].T
        h2 = _rms(hs, w[p + "post_attention_layernorm.weight"])
        gate = h2 @ w[p + "mlp.gate_proj.weight"].T
        up = h2 @ w[p + "mlp.up_proj.weight"].T
        hs = hs + (F.silu(gate) * up) @ w[p + "mlp.down_proj.weight"].T
    return _rms(hs, w["final_norm.weight"])


def _yarn_mscale_sq():
    m = 0.1 * 1.0 * math.log(MLA_FACTOR) + 1.0
    return m * m


# trtllm-gen accepts only 32 or 64 (production uses 64); the wrapper path is
# looser, so a smaller value silently tests one variant and not the other.
MLA_PAGE = 32


def _run_mla_block_decode(drafter, captured, noise_embed, paged=False):
    """Drive one block decode, optionally through the paged cache.

    ``paged=False`` keeps the dense arena the eager reference path uses.
    ``paged=True`` hands ``dflash_forward`` a page table, which is what selects
    the kernel paths -- without it ``use_kernel`` is False and every kernel
    variant silently falls back to the same eager branch.
    """
    dev = "cuda"
    proj = drafter.project_target_hidden(captured.to(dev, torch.bfloat16))
    ctx_pos = torch.arange(CTX_LEN, device=dev)
    latent, v = drafter.precompute_context_kv(proj, ctx_pos)
    assert v is None, "an MLA drafter stores no V half"
    assert latent.shape == (CTX_LEN, MLA_LAYERS, 1, MLA_LATENT)

    kwargs = dict(
        noise_embedding=noise_embed.to(dev, torch.bfloat16).unsqueeze(0),
        query_positions=torch.arange(CTX_LEN, CTX_LEN + MLA_BLOCK, device=dev).unsqueeze(0),
        num_ctx_per_req=torch.tensor([CTX_LEN], device=dev),
        ctx_v_cache=None,
        ctx_cache_batch_idx=torch.tensor([0], device=dev),
    )
    if not paged:
        pool = torch.zeros(
            1, MLA_LAYERS, CTX_LEN + MLA_BLOCK, 1, MLA_LATENT, dtype=torch.bfloat16, device=dev
        )
        pool[0, :, :CTX_LEN] = latent.permute(1, 0, 2, 3)
        out = drafter.dflash_forward(ctx_k_cache=pool, **kwargs)
        return out.float().cpu()

    # [pages, kv_factor=1, nkv=1, page_size, head_dim] per layer, plus a block's
    # worth of slack: the manager reserves it, and the non-causal path writes
    # the draft block into it.
    npages = -(-(CTX_LEN + MLA_BLOCK) // MLA_PAGE)
    pool = [
        torch.zeros(npages, 1, 1, MLA_PAGE, MLA_LATENT, dtype=torch.bfloat16, device=dev)
        for _ in range(MLA_LAYERS)
    ]
    rows, cols = ctx_pos // MLA_PAGE, ctx_pos % MLA_PAGE
    for layer_idx in range(MLA_LAYERS):
        pool[layer_idx][rows, 0, 0, cols] = latent[:, layer_idx, 0]
    page_table = torch.arange(npages, device=dev, dtype=torch.int32).unsqueeze(0)
    out = drafter.dflash_forward(
        ctx_k_cache=pool[0], ctx_kv_cache=pool, ctx_page_table=page_table, **kwargs
    )
    return out.float().cpu()


@needs_cuda
@pytest.mark.parametrize(
    "paged",
    [False, pytest.param(True, marks=needs_absorbed_mla)],
    ids=["eager", "kernel_fixup"],
)
def test_mla_dspark_block_decode_matches_unabsorbed_oracle(monkeypatch, paged):
    """Absorbed block decode == un-absorbed eager MLA, and the halves are not
    interchangeable (the negative control keeps this from being a tautology).

    The paged variant writes the draft block into the pool and reads it back
    through the kernel, so a bug in that write shows up here as a parity
    failure rather than as lower acceptance length.
    """
    import tensorrt_llm._torch.models.modeling_dspark as md

    # Count the builder instead of trusting the parametrisation: the paged
    # variant has to enter the kernel branch, and a fixture that quietly falls
    # back to eager is exactly how this test used to pass without testing
    # anything (no page table, then MLA dims and a page size both kernels
    # reject).
    calls = {"fixup": 0}
    _original = md._build_mla_block_fixup

    def _counted(*a, **kw):
        calls["fixup"] += 1
        return _original(*a, **kw)

    monkeypatch.setattr(md, "_build_mla_block_fixup", _counted)
    torch.manual_seed(0)
    weights = _tiny_mla_weights()
    drafter = _build_mla_drafter(weights)

    g = torch.Generator().manual_seed(42)
    captured = torch.randn(CTX_LEN, MLA_HIDDEN * NUM_CAPTURE, generator=g) * 0.5
    noise_embed = torch.randn(MLA_BLOCK, MLA_HIDDEN, generator=g) * 0.5

    out = _run_mla_block_decode(drafter, captured, noise_embed, paged=paged)
    captured_q, noise_q = captured.to(torch.bfloat16), noise_embed.to(torch.bfloat16)
    oracle = _oracle_mla_block_decode(weights, captured_q, noise_q)
    swapped = _oracle_mla_block_decode(weights, captured_q, noise_q, swap_kv_b=True)

    assert calls == {"fixup": 1 if paged else 0}, f"paged={paged} took the wrong branch: {calls}"

    diff = (out - oracle).abs().max().item()
    diff_swapped = (out - swapped).abs().max().item()
    assert diff < 0.02, f"MLA parity failed: max abs diff {diff}"
    assert diff_swapped > 4 * max(diff, 1e-4), (
        f"negative control failed: kv_b K/V halves swapped is too close "
        f"({diff_swapped} vs {diff}) -- the absorption may not be exercised"
    )


@needs_cuda
def test_mla_dspark_loads_the_torchspec_spelling_and_keeps_its_embedding():
    """Loads unconverted, and does not adopt the target's embedding.

    Two silent failures guarded here: the TorchSpec capture projection is
    ``context_proj`` / ``context_norm`` / ``final_norm`` where the DFlash base
    expects ``fc`` / ``hidden_norm`` / ``norm``, and the drafter ships a trained
    ``embed_tokens`` whose values differ from the target's, so taking the
    target's would feed the block decode embeddings it was not distilled on.
    """
    weights = _tiny_mla_weights()
    drafter = _build_mla_drafter(weights)

    assert drafter._dspark_shift_label
    assert drafter.has_markov_head
    assert drafter._kv_factor == 1, "the MLA cache stores one latent, not K and V"
    assert (drafter._num_kv_heads, drafter._head_dim) == (1, MLA_LATENT)
    torch.testing.assert_close(drafter.markov_w1.cpu(), weights["markov_head.markov_w1.weight"])
    torch.testing.assert_close(
        drafter.fc.weight.cpu().float(), weights["context_proj.weight"].float()
    )

    target = SimpleNamespace(
        model=SimpleNamespace(embed_tokens=torch.nn.Embedding(VOCAB, MLA_HIDDEN)),
        lm_head=torch.nn.Linear(MLA_HIDDEN, VOCAB, bias=False),
    )
    drafter.load_weights_from_target_model(target)
    assert drafter.lm_head is target.lm_head
    torch.testing.assert_close(
        drafter.model.embed_tokens.weight.cpu().float(), weights["embed_tokens.weight"].float()
    )


@needs_cuda
@pytest.mark.parametrize("backend", ["VANILLA", "TRTLLM"])
def test_mla_dspark_ignores_the_worker_attention_backend(backend):
    """The MLA drafter runs _mla_paged_attention, so the field selects nothing.

    Both values must construct, and neither worker op set may be loaded --
    otherwise a drafter that never calls them drags in an optional dependency,
    and the worker's per-backend shape checks (which the absorbed 64:1 /
    head_dim 576 shape fails) would bind on a path that does not use them.
    """
    from tensorrt_llm._torch.model_config import ModelConfig
    from tensorrt_llm._torch.models.modeling_dspark import MLADSparkForCausalLM

    model_config = ModelConfig(pretrained_config=_tiny_mla_config(), attn_backend=backend)
    drafter = MLADSparkForCausalLM(model_config, dflash_attention_backend=backend)
    assert drafter._uses_worker_attention_backend is False
    assert drafter.dflash_attention_backend == backend
    assert drafter._dflash_flash_attention is None
    assert drafter._dflash_trtllm_gen_ops is None


@needs_cuda
def test_mla_dspark_still_rejects_an_unknown_backend():
    """Dropping the VANILLA-only guard must not drop the typo check."""
    from tensorrt_llm._torch.model_config import ModelConfig
    from tensorrt_llm._torch.models.modeling_dspark import MLADSparkForCausalLM

    model_config = ModelConfig(pretrained_config=_tiny_mla_config(), attn_backend="VANILLA")
    with pytest.raises(ValueError, match="must be VANILLA or TRTLLM"):
        MLADSparkForCausalLM(model_config, dflash_attention_backend="FLASHINFER")


def test_mla_dspark_rope_conventions_agree_on_scores():
    """The drafter's adjacent-pair RoPE must be the HF one up to a lane swap.

    The block decode moved off the HF layout onto the layout the fused
    RMSNorm/RoPE kernel consumes. That is only sound because the permutation
    cancels inside q_rope . k_rope -- and only if EVERY producer switches. The
    negative control is the half-migrated state, which is silent in accuracy and
    shows up solely as lower acceptance length.
    """
    from tensorrt_llm._torch.models.modeling_dspark import (
        apply_dspark_rotary_batched,
        build_dspark_mla_yarn_freqs_cis,
        build_dspark_mla_yarn_rope,
    )

    torch.manual_seed(0)
    dim, batch, blk, heads = 64, 3, 8, 4
    params = dict(
        dim=dim,
        base=50000.0,
        scaling_factor=32.0,
        original_max_position_embeddings=32768,
        max_position_embeddings=256,
        beta_fast=32.0,
        beta_slow=1.0,
        mscale=1.0,
        mscale_all_dim=1.0,
        device="cpu",
    )
    cos, sin = build_dspark_mla_yarn_rope(**params)
    freqs = build_dspark_mla_yarn_freqs_cis(**params)

    pos = torch.arange(blk)
    q = torch.randn(batch, blk, heads, dim)
    k = torch.randn(batch, blk, dim)
    fc = freqs[pos].unsqueeze(0).expand(batch, blk, dim // 2)

    hf_q = _hf_apply_rope(q, cos[pos].view(1, blk, 1, dim), sin[pos].view(1, blk, 1, dim))
    hf_k = _hf_apply_rope(k, cos[pos].view(1, blk, dim), sin[pos].view(1, blk, dim))
    ap_q = apply_dspark_rotary_batched(q, fc)
    ap_k = apply_dspark_rotary_batched(k, fc)

    perm = torch.empty(dim, dtype=torch.long)
    perm[: dim // 2] = torch.arange(0, dim, 2)
    perm[dim // 2 :] = torch.arange(1, dim, 2)
    torch.testing.assert_close(hf_q, ap_q[..., perm], atol=1e-5, rtol=1e-5)

    hf_score = torch.einsum("bshd,bsd->bsh", hf_q, hf_k)
    ap_score = torch.einsum("bshd,bsd->bsh", ap_q, ap_k)
    torch.testing.assert_close(hf_score, ap_score, atol=1e-4, rtol=1e-4)

    # Negative control: only q migrated.
    mixed = torch.einsum("bshd,bsd->bsh", ap_q, hf_k)
    assert (mixed - hf_score).abs().max() > 1e-2, (
        "mixing the two RoPE conventions must change the scores; if it does not, "
        "this test cannot catch a half-finished migration"
    )


def test_mla_block_fixup_stays_inside_the_allocation():
    """A context that fills its allocation must still leave the block room.

    The bound comes from dflash.py, so it is called here rather than restated:
    a test that computed `allocated - block_size` itself would still pass
    against a production path truncating to `allocated`. The MLA writes the
    block's own latents at ctx_len..ctx_len+block_size, so that regression puts
    the first of them on the first unallocated page, whose block-table entry
    _refresh_ctx_block_tables clamped from a negative placeholder to 0 --
    another request's block. Silent cross-request corruption, not a fault.
    """
    from tensorrt_llm._torch.models.modeling_dspark import _build_mla_block_fixup
    from tensorrt_llm._torch.speculative.dflash import dflash_allocated_ctx_limit

    page_size, block_size, allocated_pages = 8, 3, 2
    allocated = allocated_pages * page_size
    # Entries past the allocation are the clamped placeholders: physical 0.
    page_tables = torch.tensor([[41, 42, 0, 0]])

    ctx_len = dflash_allocated_ctx_limit(torch.tensor([allocated_pages]), page_size, block_size)
    fixup = _build_mla_block_fixup(ctx_len, page_tables, block_size, page_size)

    # Exactly the last allocated page, not merely a subset of the allocation:
    # `<= {41, 42}` also passes for a block that landed a page early.
    assert set(fixup.pages.flatten().tolist()) == {42}
    assert int(fixup.seq_lens_i32[0]) <= allocated

    # Negative control: the value the production path yields if the block room
    # is dropped. It must reach the unallocated page, or this bound is untested.
    overrun = _build_mla_block_fixup(torch.tensor([allocated]), page_tables, block_size, page_size)
    assert 0 in overrun.pages.flatten().tolist()


def test_mla_rope_table_follows_runtime_max_seq_len():
    """The position table is sized by what is served, not what is advertised.

    K3 declares max_position_embeddings = 1,048,576; at complex64 that table is
    ~256 MiB per rank, built through full-size fp32 cos/sin. The served
    max_seq_len is what the context cache is bounded by, so the table follows
    it. Only the length moves -- YaRN's correction range is computed from
    original_max_position_embeddings.
    """
    from tensorrt_llm._torch.models.modeling_dspark import _resolve_dspark_mla_rope_params

    cfg = _tiny_mla_config()
    cfg.max_position_embeddings = 1 << 20

    advertised = _resolve_dspark_mla_rope_params(cfg)
    served = _resolve_dspark_mla_rope_params(cfg, SimpleNamespace(max_seq_len=4096))

    assert advertised["max_positions"] == 1 << 20
    assert served["max_positions"] == 4096
    # the YaRN inputs that decide the rotation must not move with it
    for key in ("theta", "scaling_factor", "original_max_positions", "beta_fast", "beta_slow"):
        assert advertised[key] == served[key]


@pytest.mark.parametrize(
    "max_ctx,checkpoint_block_size,max_draft_len",
    [
        (8205, None, 7),  # the shipped recipe: no block_size, fallback K+1 = 8
        (8205, 7, 7),  # a checkpoint declaring a window NARROWER than the verify group
        (4104, 16, 3),  # ... and one declaring a WIDER window
        (1026, 2, 1),
    ],
)
def test_dflash_position_ceiling_covers_both_index_paths(
    max_ctx, checkpoint_block_size, max_draft_len
):
    """The published ceiling must exceed every index the forward can produce.

    Two independent position sets read the same table, and they do not have the
    same reach: the block decode runs to ctx_len + block_size - 1 while the
    context path runs to ctx_len + max_draft_len. Sizing for either one alone is
    short whenever the other is wider, which is why the ceiling takes a max.

    max_ctx here is the RUNTIME value (attn_metadata.max_seq_len, what ctx_len
    is clamped to), not model_config.max_seq_len -- 8205 is the measured value
    for a configured 8192 with max_draft_len 7.
    """
    from tensorrt_llm._torch.speculative.dflash import dflash_position_ceiling

    block_size = checkpoint_block_size or (max_draft_len + 1)
    ceiling = dflash_position_ceiling(max_ctx, block_size, max_draft_len)

    # dflash.py: query_position_ids = clamp(ctx_len + num_accepted) + j
    worst_block = max_ctx + block_size - 1
    # dflash.py: ctx_position_ids = ctx_len + [0, max_draft_len]
    worst_ctx = max_ctx + max_draft_len

    assert ceiling > worst_block, (
        f"ceiling {ceiling} does not cover block decode index {worst_block} "
        f"(max_ctx {max_ctx}, block_size {block_size})"
    )
    assert ceiling > worst_ctx, (
        f"ceiling {ceiling} does not cover context index {worst_ctx} "
        f"(max_ctx {max_ctx}, max_draft_len {max_draft_len})"
    )
    # and no more than one position of slack, so it cannot quietly grow
    assert ceiling == max(worst_block, worst_ctx) + 1


@needs_cuda
def test_mla_rope_table_prefers_the_runtime_ceiling_over_the_config_cap():
    """A worker-published ceiling replaces the config-derived cap outright.

    Not max() of the two: with max_seq_len unset the config cap is the
    checkpoint's advertised max_position_embeddings (1,048,576 for K3), and
    taking the larger would rebuild exactly the ~256 MiB table the cap exists to
    avoid. Exercises the real _mla_freqs_cis, so it also pins that the table is
    still lazy enough to see an attribute set after construction.
    """
    drafter = _build_mla_drafter(_tiny_mla_weights())
    # construction-time cap: the plain config value, no lookahead baked in
    assert drafter._mla_rope_params["max_positions"] == _tiny_mla_config().max_position_embeddings

    drafter._runtime_position_ceiling = 8213
    table = drafter._mla_freqs_cis(torch.device("cuda"))
    assert table.shape[0] == 8213, (
        f"table has {table.shape[0]} positions, expected the published ceiling 8213"
    )


@needs_cuda
def test_mla_drafter_rejects_a_checkpoint_missing_backbone_weights():
    """Only lm_head may be absent; a truncated backbone must not load quietly.

    The generic loader skips a module whose whole subtree filters to nothing,
    so allow_partial_loading=False would not catch this either -- the check has
    to come from the drafter's own module tree.
    """
    weights = _tiny_mla_weights()
    drafter = _build_mla_drafter(weights)

    # A whole module, not one tensor: the check is module-granular on purpose,
    # since a fused module (gate_up_proj) is stored unfused and dropping half of
    # it is a parameter-level gap the loader's own naming cannot distinguish.
    # Derive the prefix from the fixture rather than spelling it: DFlash
    # checkpoints name layers without the `model.` prefix, so a hardcoded key
    # silently matches nothing and the test passes by not truncating anything.
    victim = next(k for k in weights if k.endswith("self_attn.o_proj.weight"))
    truncated = {k: v for k, v in weights.items() if k != victim}
    assert len(truncated) == len(weights) - 1

    with pytest.raises(ValueError, match="WEIGHTS_SHARED_WITH_TARGET"):
        drafter.load_weights(truncated)


@needs_cuda
@pytest.mark.parametrize("victim", ["context_proj.weight", "context_norm.weight"])
def test_mla_drafter_requires_the_wrapper_owned_tensors(victim):
    """fc / hidden_norm are built FROM the checkpoint, so no module walk sees them.

    They live on the wrapper, not on draft_model_full, and the extraction is
    `if <name> in remapped`. Without an explicit requirement a checkpoint that
    omits them loads clean and the drafter runs with no capture projection:
    has_target_features stays False, _ctx_len never advances, and it drafts
    from an empty context for the life of the process.
    """
    weights = _tiny_mla_weights()
    drafter = _build_mla_drafter(weights)

    assert victim in weights, "fixture no longer ships the TorchSpec spelling"
    truncated = {k: v for k, v in weights.items() if k != victim}

    with pytest.raises(ValueError, match="wrapper owns"):
        drafter.load_weights(truncated)


@needs_cuda
def test_mla_drafter_rejects_a_partial_fused_component_set():
    """One of q/k/v present is not the module being present.

    The fused load path accepts a subset under allow_partial_loading and
    leaves the absent shards at torch.empty -- uninitialised device memory,
    not zeros -- so the check has to require every component, not any.
    """
    weights = _tiny_mla_weights()
    drafter = _build_mla_drafter(weights)

    # gate_proj/up_proj, NOT an unfused module: the checkpoint stores them
    # separately and the module tree names them once as mlp.gate_up_proj, so
    # dropping one is the only way to reach the fused branch of _supplied.
    # (kv_b_proj would exercise the plain _has path and duplicate the
    # missing-backbone test above.)
    victim = next(k for k in weights if k.endswith("mlp.gate_proj.weight"))
    assert any(k.endswith("mlp.up_proj.weight") for k in weights), (
        "the surviving half of the fused pair must be present, or this is not "
        "testing the fused branch"
    )
    truncated = {k: v for k, v in weights.items() if k != victim}

    with pytest.raises(ValueError, match="WEIGHTS_SHARED_WITH_TARGET"):
        drafter.load_weights(truncated)
