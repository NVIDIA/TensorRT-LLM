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
from typing import Optional

import pytest
import torch
import torch.nn.functional as F
from _torch.helpers import dspark_markov_chain_logits_reference

from tensorrt_llm._torch.models.modeling_dflash import DFlashForCausalLM, dspark_layer_window_size
from tensorrt_llm._torch.models.modeling_dspark import GQADSparkForCausalLM
from tensorrt_llm._torch.models.modeling_speculative import dspark_markov_step_bias
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

# The TRTLLM-Gen paged attention the DFlash drafter uses ships for the same
# SM100 family.
needs_trtllm_gen = pytest.mark.skipif(
    not _has_absorbed_mla_kernel(),
    reason="TRTLLM-Gen paged block decode needs an SM100-family GPU",
)

# ---------------------------------------------------------------------------
# Reference: line-for-line port of DeepSpec VanillaMarkov
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

    out = dspark_markov_chain_logits_reference(base, anchor, w1, w2)
    assert torch.equal(out, ref_logits)
    # Greedy per-position argmax of the corrected logits reproduces the
    # reference sequentially-sampled chain (what sample_draft_tokens does).
    assert torch.equal(torch.argmax(out, dim=-1), ref_tokens)


def test_markov_chain_empty_block_is_noop():
    w1, w2, base, anchor = _random_markov()
    empty = base[:, :0, :]
    assert dspark_markov_chain_logits_reference(empty, anchor, w1, w2) is empty


def test_markov_chain_sharded_matches_full_vocab():
    """The DFlashWorker TP path — every rank runs the chain on its
    contiguous markov_w2/logits vocab shard, chained through a global
    argmax over all shards — reassembles to the full-vocab chain."""
    w1, w2, base, anchor = _random_markov()
    full = dspark_markov_chain_logits_reference(base, anchor, w1, w2)

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


def test_swa_window_all_full_attention_layer_types():
    """use_swa with no sliding layer declared windows every layer.

    Some windowed DFlash 2 drafters list every layer as `full_attention` and
    carry the window in `dflash_config`; reading those layer types literally
    would serve them with full attention and lose acceptance silently.
    """
    full = ["full_attention"] * 5
    for layer_idx in range(len(full)):
        assert dspark_layer_window_size(True, 2048, full, layer_idx) == (2047, 2047)

    # Without use_swa the same layer types keep flash-attn's default.
    for layer_idx in range(len(full)):
        assert dspark_layer_window_size(False, 2048, full, layer_idx) == (-1, -1)
    assert dspark_layer_window_size(False, 0, None, 0) == (-1, -1)

    # A checkpoint that tells sliding layers from full ones keeps the per-layer windows.
    mixed = ["sliding_attention", "full_attention", "sliding_attention"]
    assert dspark_layer_window_size(True, 512, mixed, 0) == (511, 511)
    assert dspark_layer_window_size(True, 512, mixed, 1) == (-1, -1)
    assert dspark_layer_window_size(True, 512, mixed, 2) == (511, 511)

    # Layer types built in Python can be named objects; they are classified the
    # way the attention mask classifies them.
    named = [SimpleNamespace(name="SLIDING_ATTENTION"), "full_attention"]
    assert dspark_layer_window_size(True, 512, named, 0) == (511, 511)
    assert dspark_layer_window_size(True, 512, named, 1) == (-1, -1)


# ---------------------------------------------------------------------------
# Tiny end-to-end drafter: config parsing, weight loading, block-decode
# parity vs an eager bf16 reference (needs CUDA + flash_attn).
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


def _tiny_config(
    dspark: bool, *, published_spelling: bool = False, mask_token_id: Optional[int] = VOCAB - 2
):
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
    dflash = {"target_layer_ids": [0, 1]}
    if mask_token_id is not None:
        dflash["mask_token_id"] = mask_token_id
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
    mask_token_id: Optional[int] = VOCAB - 2,
):
    from tensorrt_llm._torch.model_config import ModelConfig

    model_config = ModelConfig(
        pretrained_config=_tiny_config(
            dspark, published_spelling=published_spelling, mask_token_id=mask_token_id
        ),
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
    # fp32 accumulation, x's dtype out -- the same contract as trtllm's RMSNorm.
    # Returning fp32 unconditionally would silently upcast a bf16 reference at
    # the first norm and make every downstream matmul fp32.
    xf = x.float()
    return ((xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)) * w.float()).to(x.dtype)


def _rope(x, positions, theta=10000.0):
    # NeoX half-split convention, matching RotaryEmbedding(is_neox=True).
    # x: [T, heads, hd]; positions: [T]
    hd = x.shape[-1]
    inv = 1.0 / theta ** (torch.arange(0, hd, 2, dtype=torch.float64) / hd)
    ang = positions.double().unsqueeze(-1) * inv  # [T, hd/2]
    cos = ang.cos().to(x.dtype).unsqueeze(1)
    sin = ang.sin().to(x.dtype).unsqueeze(1)
    x1, x2 = x[..., : hd // 2], x[..., hd // 2 :]
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)


def _reference_block_decode(weights, captured, noise_embed, use_swa):
    """Eager bf16 port of the DeepSpec dspark block decode
    (Qwen3DSparkDecoderLayer stack over [context ; draft block]).

    bf16 matches the drafter, so the comparison measures the algorithm rather
    than the dtype; torch keeps fp32 accumulation inside its bf16 matmuls.
    """
    bf16 = torch.bfloat16
    w = {k: v.to(bf16) for k, v in weights.items()}
    nh, nkv, hd = (TINY["num_attention_heads"], TINY["num_key_value_heads"], TINY["head_dim"])
    ctx = captured.shape[0]
    blk = noise_embed.shape[0]
    ctx_pos = torch.arange(ctx, dtype=torch.long)
    q_pos = torch.arange(ctx, ctx + blk, dtype=torch.long)
    all_pos = torch.cat([ctx_pos, q_pos])

    # Target feature projection: hidden_norm(fc(captured)); constant across
    # layers, no input_layernorm on the context path (generic DFlash).
    ctx_feat = _rms(captured.to(bf16) @ w["fc.weight"].T, w["hidden_norm.weight"])

    hs = noise_embed.to(bf16)
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
    assert drafter.shift_label and drafter._use_swa
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
def test_gqa_dspark_keeps_its_trained_mask_row_over_the_target_embedding():
    """The GQA drafter takes the target's embedding (load_weights_from_target_model), but a checkpoint that ships its
    own embedding trained the mask token's row, which the target's embedding does not have. The drafter keeps that row
    as ``mask_token_embedding``, and the worker's noise block puts it in every masked slot; slot 0 stays the bonus
    token's embedding from the shared lookup.
    """
    h = TINY["hidden_size"]
    weights = _tiny_weights()
    g = torch.Generator().manual_seed(23)
    weights["embed_tokens.weight"] = (torch.randn(VOCAB, h, generator=g) * 0.05).to(torch.bfloat16)
    drafter = _build_drafter(True, weights)
    target_embed = torch.nn.Embedding(VOCAB, h).to("cuda", torch.bfloat16)
    target = SimpleNamespace(
        model=SimpleNamespace(embed_tokens=target_embed),
        lm_head=torch.nn.Linear(h, VOCAB, bias=False),
    )
    drafter.load_weights_from_target_model(target)

    mask_id = drafter.mask_token_id
    trained = weights["embed_tokens.weight"][mask_id]
    assert not torch.equal(target_embed.weight[mask_id].detach().cpu(), trained)
    torch.testing.assert_close(drafter.mask_token_embedding.cpu(), trained, rtol=0, atol=0)

    from tensorrt_llm._torch.speculative.dflash import dflash_noise_block_embedding

    bonus = torch.tensor([3, 5], dtype=torch.long, device="cuda")
    block = 4
    with torch.no_grad():
        noise = dflash_noise_block_embedding(
            drafter.draft_model_full.model.embed_tokens,
            bonus,
            mask_id,
            block,
            drafter.mask_token_embedding,
        )
    assert tuple(noise.shape) == (2, block, h)
    torch.testing.assert_close(
        noise[:, 0].cpu(), target_embed.weight[bonus].detach().cpu(), rtol=0, atol=0
    )
    for j in range(1, block):
        torch.testing.assert_close(noise[:, j].cpu(), trained.expand(2, -1), rtol=0, atol=0)


@needs_gpu
def test_gqa_dspark_mask_row_takes_the_drafter_dtype_and_device_at_load():
    """The kept row is cast once, at load, to the drafter's dtype (bf16 here, from an fp32 checkpoint) and placed on
    the device of the drafter's own parameters, so the worker's noise block uses it as is."""
    weights = _tiny_weights()
    g = torch.Generator().manual_seed(29)
    weights["embed_tokens.weight"] = torch.randn(VOCAB, TINY["hidden_size"], generator=g) * 0.05
    drafter = _build_drafter(True, weights)

    row = drafter.mask_token_embedding
    assert row.dtype == torch.bfloat16
    assert row.device == drafter.fc.weight.device
    trained = weights["embed_tokens.weight"][drafter.mask_token_id].to(torch.bfloat16)
    torch.testing.assert_close(row.cpu(), trained, rtol=0, atol=0)


@needs_gpu
@pytest.mark.parametrize(
    "width, mask_token_id, message",
    [
        (TINY["hidden_size"], VOCAB + 1, f"mask_token_id {VOCAB + 1} is not a row"),
        (TINY["hidden_size"], -1, "mask_token_id -1 is not a row"),
        (
            TINY["hidden_size"] + 8,
            VOCAB - 2,
            rf"expected \[rows, hidden_size={TINY['hidden_size']}\]",
        ),
    ],
    ids=["mask_id_past_the_rows", "negative_mask_id", "wrong_width"],
)
def test_gqa_dspark_rejects_a_mask_row_the_shipped_embedding_does_not_have(
    width, mask_token_id, message
):
    """A mask id that is not a row of the shipped embedding, or rows that are not hidden_size wide, fail at load,
    before any weight is copied, with an error that names the problem."""
    weights = _tiny_weights()
    weights["embed_tokens.weight"] = torch.zeros(VOCAB, width, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match=message):
        _build_drafter(True, weights, mask_token_id=mask_token_id)


@needs_gpu
def test_gqa_dspark_keeps_no_mask_row_when_its_config_names_no_mask_token():
    """With no mask token named by the drafter config or a speculative config, the drafter keeps DFlash's default id
    (vocab_size), one past the shipped rows: it keeps no row and loads."""
    weights = _tiny_weights()
    weights["embed_tokens.weight"] = torch.zeros(VOCAB, TINY["hidden_size"], dtype=torch.bfloat16)
    drafter = _build_drafter(True, weights, mask_token_id=None)
    assert drafter.mask_token_id == VOCAB
    assert getattr(drafter, "mask_token_embedding", None) is None


@needs_gpu
def test_gqa_dspark_warns_when_its_shipped_embedding_keeps_no_mask_row():
    """Keeping no row for DFlash's default id is logged as a warning, since the masked slots then use the shared
    embedding's row instead of a trained one."""
    from unittest.mock import patch

    from tensorrt_llm._torch.models import modeling_dspark

    weights = _tiny_weights()
    weights["embed_tokens.weight"] = torch.zeros(VOCAB, TINY["hidden_size"], dtype=torch.bfloat16)
    with patch.object(modeling_dspark.logger, "warning") as warning:
        drafter = _build_drafter(True, weights, mask_token_id=None)
    assert getattr(drafter, "mask_token_embedding", None) is None
    messages = [" ".join(map(str, call.args)) for call in warning.call_args_list]
    assert any("no trained mask row is kept" in m for m in messages), messages


@needs_gpu
def test_gqa_dspark_keeps_the_trained_row_of_the_speculative_configs_mask_id():
    """A checkpoint that names its mask token only as ``dspark_noise_token_id``: the speculative config resolves that
    key and the worker embeds its id, while the drafter config resolution falls back to DFlash's default. The builder
    hands the drafter the speculative config's id, so the drafter keeps that id's trained row and its mask id is the
    worker's."""
    from tensorrt_llm._torch.model_config import ModelConfig
    from tensorrt_llm._torch.models.modeling_dspark import _build_dspark_draft

    mask_id = VOCAB - 2
    pretrained = _tiny_config(True, mask_token_id=None)
    pretrained.dspark_noise_token_id = mask_id
    spec = SimpleNamespace(
        mask_token_id=mask_id,
        attention_backend="VANILLA",
        draft_is_embedded_in_target=False,
        context_kv_only=False,
        context_kv_dtype="auto",
    )
    drafter = _build_dspark_draft(
        SimpleNamespace(spec_config=spec),
        ModelConfig(pretrained_config=pretrained, attn_backend="TRTLLM"),
        None,
        None,
    ).to("cuda")
    weights = _tiny_weights()
    g = torch.Generator().manual_seed(31)
    trained = (torch.randn(VOCAB, TINY["hidden_size"], generator=g) * 0.05).to(torch.bfloat16)
    weights["embed_tokens.weight"] = trained
    drafter.load_weights(dict(weights))

    assert isinstance(drafter, GQADSparkForCausalLM)
    assert drafter.mask_token_id == spec.mask_token_id
    torch.testing.assert_close(drafter.mask_token_embedding.cpu(), trained[mask_id], rtol=0, atol=0)


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
    assert drafter.shift_label
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
    assert drafter.shift_label is False
    for absent in (
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
def test_dspark_block_decode_matches_reference():
    """The full drafter block decode (fc/hidden_norm projection, per-layer
    QKV + q/k-norm + RoPE, non-causal SWA flash attention over
    [context ; block], MLP, final norm) matches the eager bf16 reference; the
    no-window reference does NOT match (the window demonstrably binds)."""
    torch.manual_seed(0)
    weights = _tiny_weights()
    drafter = _build_drafter(True, weights)

    g = torch.Generator().manual_seed(42)
    captured = torch.randn(CTX_LEN, TINY["hidden_size"] * NUM_CAPTURE, generator=g) * 0.5
    noise_embed = torch.randn(TINY["block_size"], TINY["hidden_size"], generator=g) * 0.5

    out = _run_block_decode(drafter, weights, captured, noise_embed)

    # The reference consumes the same bf16 inputs the drafter sees.
    captured_q = captured.to(torch.bfloat16)
    noise_q = noise_embed.to(torch.bfloat16)
    expected_swa = _reference_block_decode(weights, captured_q, noise_q, True)
    expected_full = _reference_block_decode(weights, captured_q, noise_q, False)

    diff_swa = (out - expected_swa).abs().max().item()
    diff_full = (out - expected_full).abs().max().item()
    # Tolerance well below the SWA-vs-full gap; see the measured values below.
    assert diff_swa < 0.02, f"SWA parity failed: max abs diff {diff_swa}"
    assert diff_full > 4 * max(diff_swa, 1e-4), (
        f"negative control failed: no-window reference too close "
        f"({diff_full} vs swa {diff_swa}) — window may not be applied"
    )


@needs_gpu
def test_plain_dflash_block_decode_matches_full_attention_reference():
    """No-regression numeric check: the plain-DFlash drafter (no dspark
    fields) still runs full non-causal attention over the whole context."""
    weights = _tiny_weights()
    drafter = _build_drafter(False, weights)
    g = torch.Generator().manual_seed(43)
    captured = torch.randn(CTX_LEN, TINY["hidden_size"] * NUM_CAPTURE, generator=g) * 0.5
    noise_embed = torch.randn(TINY["block_size"], TINY["hidden_size"], generator=g) * 0.5
    out = _run_block_decode(drafter, weights, captured, noise_embed)
    expected = _reference_block_decode(
        weights, captured.to(torch.bfloat16), noise_embed.to(torch.bfloat16), False
    )
    diff = (out - expected).abs().max().item()
    assert diff < 0.02, f"plain DFlash parity failed: max abs diff {diff}"


def _run_paged_block_decode(drafter, captured, noise_embed, cache_dtype, page_size=32):
    """Block decode through the TRTLLM paged cache, one request per history.

    Pages are shuffled across requests, so only the block tables decide which
    history each request reads.
    """
    dev = "cuda"
    blk = TINY["block_size"]
    lengths = [ctx.shape[0] for ctx in captured]
    history = []
    for ctx in captured:
        proj = drafter.project_target_hidden(ctx.to(dev, torch.bfloat16))
        history.append(drafter.precompute_context_kv(proj, torch.arange(ctx.shape[0], device=dev)))
    # [N, layers, kv heads, head dim]; the drafter sizes its fused K/V lazily.
    num_layers, nkv, hd = history[0][0].shape[1:]
    pages_per_req = max((n + blk + page_size - 1) // page_size for n in lengths)
    num_pages = len(lengths) * pages_per_req
    order = torch.randperm(num_pages, generator=torch.Generator().manual_seed(3))
    page_table = order.view(len(lengths), pages_per_req).to(dev, torch.int32)
    caches = [
        torch.zeros(num_pages, 2, nkv, page_size, hd, dtype=torch.bfloat16, device=dev)
        for _ in range(num_layers)
    ]
    for row, (k, v) in enumerate(history):
        positions = torch.arange(lengths[row], device=dev)
        pages = page_table[row, positions // page_size].long()
        offsets = positions % page_size
        for layer, cache in enumerate(caches):
            cache[pages, 0, :, offsets] = k[:, layer]
            cache[pages, 1, :, offsets] = v[:, layer]
    caches = [cache.to(cache_dtype) for cache in caches]
    out = drafter.dflash_forward(
        noise_embedding=torch.stack(noise_embed).to(dev, torch.bfloat16),
        query_positions=torch.stack([torch.arange(n, n + blk) for n in lengths]).to(dev),
        num_ctx_per_req=torch.tensor(lengths, device=dev),
        ctx_k_cache=None,
        ctx_v_cache=None,
        ctx_cache_batch_idx=torch.arange(len(lengths), device=dev),
        ctx_kv_cache=caches,
        ctx_page_table=page_table,
    )
    return out.float().cpu().view(len(lengths), blk, -1)


@needs_trtllm_gen
@pytest.mark.parametrize("cache_dtype,tolerance", [(torch.bfloat16, 0.02)], ids=["bf16"])
def test_trtllm_paged_block_decode_matches_full_attention_reference(cache_dtype, tolerance):
    """The TRTLLM non-causal block attention over a paged BF16 cache matches the
    eager reference for requests with different history lengths.

    Each block query runs as its own one-token decode request over its
    parent's pages, so a wrong table, length or block write shows up here.
    """
    weights = _tiny_weights()
    drafter = _build_drafter(False, weights, dflash_attention_backend="TRTLLM")
    g = torch.Generator().manual_seed(44)
    lengths = (CTX_LEN, 70)
    captured = [
        torch.randn(n, TINY["hidden_size"] * NUM_CAPTURE, generator=g) * 0.5 for n in lengths
    ]
    noise_embed = [
        torch.randn(TINY["block_size"], TINY["hidden_size"], generator=g) * 0.5 for _ in lengths
    ]

    out = _run_paged_block_decode(drafter, captured, noise_embed, cache_dtype)

    for row in range(len(lengths)):
        expected = _reference_block_decode(
            weights, captured[row].to(torch.bfloat16), noise_embed[row].to(torch.bfloat16), False
        )
        expected = expected.float()
        rel_err = ((out[row] - expected).norm() / expected.norm()).item()
        assert rel_err < tolerance, f"request {row}: relative error {rel_err}"


@needs_trtllm_gen
def test_trtllm_paged_block_decode_rejects_a_quantized_cache():
    """The plain DFlash paged block decode writes the block's K/V in the draft
    activation dtype; an FP8 cache is rejected, not written lossily."""
    weights = _tiny_weights()
    drafter = _build_drafter(False, weights, dflash_attention_backend="TRTLLM")
    g = torch.Generator().manual_seed(44)
    captured = [torch.randn(CTX_LEN, TINY["hidden_size"] * NUM_CAPTURE, generator=g) * 0.5]
    noise_embed = [torch.randn(TINY["block_size"], TINY["hidden_size"], generator=g) * 0.5]

    with pytest.raises(ValueError, match="quantized context-cache writes are not supported"):
        _run_paged_block_decode(drafter, captured, noise_embed, torch.float8_e4m3fn)


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


def _build_mla_drafter(
    weights, *, dflash_attention_backend="TRTLLM", context_kv_only=False, context_kv_dtype="auto"
):
    from tensorrt_llm._torch.model_config import ModelConfig
    from tensorrt_llm._torch.models.modeling_dspark import MLADSparkForCausalLM

    model_config = ModelConfig(pretrained_config=_tiny_mla_config(), attn_backend="TRTLLM")
    drafter = MLADSparkForCausalLM(
        model_config,
        dflash_attention_backend=dflash_attention_backend,
        context_kv_only=context_kv_only,
        context_kv_dtype=context_kv_dtype,
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


def _reference_mla_block_decode(weights, captured, noise_embed, *, swap_kv_b=False):
    """Eager bf16 MLA block decode, written in the UN-ABSORBED form.

    The drafter attends the stored latent directly (``q_nope`` folded through
    ``kv_b_proj``'s K half); this expands the latent into per-head K/V and runs
    plain attention instead, so agreement is evidence about the absorption
    rather than a restatement of it. ``swap_kv_b`` is the negative control:
    exchanging the K and V halves must break parity.

    bf16, matching the drafter, so the comparison measures the ALGORITHM and
    not the dtype. torch keeps fp32 accumulation inside its bf16 matmuls, which
    is also what the drafter's GEMMs and its fp32 attention accumulator do.
    """
    from tensorrt_llm._torch.models.modeling_dspark import apply_dspark_mla_rope

    bf16 = torch.bfloat16
    w = {k: v.to(bf16) for k, v in weights.items()}
    ctx, blk = captured.shape[0], noise_embed.shape[0]
    ctx_pos = torch.arange(ctx, dtype=torch.long)
    q_pos = torch.arange(ctx, ctx + blk, dtype=torch.long)
    cos, sin = _hf_yarn_reference(
        MLA_ROPE, MLA_THETA, MLA_FACTOR, MLA_ORIG_MAX, MLA_ORIG_MAX * int(MLA_FACTOR)
    )
    cos, sin = cos.to(bf16), sin.to(bf16)
    scale = _yarn_mscale_sq() / (MLA_NOPE + MLA_ROPE) ** 0.5

    # Context features: context_norm(context_proj(captured)); constant across
    # layers and NOT passed through input_layernorm (the DFlash contract).
    ctx_feat = _rms(captured.to(bf16) @ w["context_proj.weight"].T, w["context_norm.weight"])

    hs = noise_embed.to(bf16)
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

# Two pages so the batch crosses a page boundary -- a single page never
# exercises this addressing. No longer a hard requirement since
# cute_dsl_custom_ops.py pre-marks the degenerate (1, 1) table.
MLA_CTX_LEN = 40  # ceil((40 + MLA_BLOCK) / MLA_PAGE) = 2 pages
MLA_SHORT_CTX_LEN = 20  # one page, and the block stays inside it


def _run_mla_block_decode(drafter, requests, paged=False, fp8=False):
    """Drive one block decode over a BATCH, optionally through the paged cache.

    ``requests`` is a list of ``(captured, noise_embed)`` pairs with deliberately
    unequal context lengths. B >= 2 with unequal lengths is what makes the paged
    path testable: at B == 1 the page table's stride is 1 either way, so ``.t()``
    and ``.t().contiguous()`` are indistinguishable, and ``kv_bounds`` built with
    ``repeat`` instead of ``repeat_interleave`` gives the same vector.

    ``paged=False`` keeps the dense arena the eager reference path uses.
    ``paged=True`` hands ``dflash_forward`` a page table, which is what selects
    the kernel paths -- without it ``use_kernel`` is False and every kernel
    variant silently falls back to the same eager branch. ``fp8`` stores the
    paged context as ``to_draft_fp8`` (the CuTe DSL decode then reads FP8).
    """
    dev = "cuda"
    batch = len(requests)
    ctx_lens = [c.shape[0] for c, _ in requests]
    max_ctx = max(ctx_lens)

    latents = []
    for captured, _ in requests:
        proj = drafter.project_target_hidden(captured.to(dev, torch.bfloat16))
        latent, v = drafter.precompute_context_kv(proj, torch.arange(captured.shape[0], device=dev))
        assert v is None, "an MLA drafter stores no V half"
        assert latent.shape == (captured.shape[0], MLA_LAYERS, 1, MLA_LATENT)
        latents.append(latent)

    kwargs = dict(
        noise_embedding=torch.stack([n.to(dev, torch.bfloat16) for _, n in requests]),
        query_positions=torch.stack([torch.arange(L, L + MLA_BLOCK, device=dev) for L in ctx_lens]),
        num_ctx_per_req=torch.tensor(ctx_lens, device=dev),
        ctx_v_cache=None,
        ctx_cache_batch_idx=torch.arange(batch, device=dev),
    )
    if not paged:
        pool = torch.zeros(
            batch,
            MLA_LAYERS,
            max_ctx + MLA_BLOCK,
            1,
            MLA_LATENT,
            dtype=torch.bfloat16,
            device=dev,
        )
        for b, latent in enumerate(latents):
            pool[b, :, : ctx_lens[b]] = latent.permute(1, 0, 2, 3)
        out = drafter.dflash_forward(ctx_k_cache=pool, **kwargs)
        # dflash_forward returns [B * block, hidden]; give the caller the batch
        # axis back. At B == 1 the flat form happened to match the reference's
        # [block, hidden], which is why this never surfaced before.
        return out.reshape(batch, MLA_BLOCK, -1).float().cpu()

    # [pages, kv_factor=1, nkv=1, page_size, head_dim] per layer, plus a block's
    # worth of slack: the manager reserves it, and the non-causal path writes
    # the draft block into it. Each request owns a DISJOINT page range, so a
    # transposed or mis-ordered page table reads another request's context
    # rather than silently reading the same one back.
    per_req = -(-(max_ctx + MLA_BLOCK) // MLA_PAGE)
    pool = [
        torch.zeros(batch * per_req, 1, 1, MLA_PAGE, MLA_LATENT, dtype=torch.bfloat16, device=dev)
        for _ in range(MLA_LAYERS)
    ]
    for b, latent in enumerate(latents):
        pos = torch.arange(ctx_lens[b], device=dev)
        rows = b * per_req + pos // MLA_PAGE
        cols = pos % MLA_PAGE
        for layer_idx in range(MLA_LAYERS):
            pool[layer_idx][rows, 0, 0, cols] = latent[:, layer_idx, 0]
    if fp8:
        from tensorrt_llm._torch.speculative.dflash_attention import to_draft_fp8

        pool = [to_draft_fp8(p) for p in pool]
    page_table = torch.arange(batch * per_req, device=dev, dtype=torch.int32).view(batch, per_req)
    out = drafter.dflash_forward(
        ctx_k_cache=pool[0], ctx_kv_cache=pool, ctx_page_table=page_table, **kwargs
    )
    return out.reshape(batch, MLA_BLOCK, -1).float().cpu()


@needs_cuda
@pytest.mark.parametrize(
    "paged,backend",
    [
        (False, "VANILLA"),
        pytest.param(True, "TRTLLM", marks=needs_absorbed_mla),
        pytest.param(True, "CUTEDSL", marks=needs_absorbed_mla),
    ],
    ids=["eager", "kernel_fixup", "cute_dsl"],
)
def test_mla_dspark_block_decode_matches_unabsorbed_reference(monkeypatch, paged, backend):
    """Absorbed block decode == un-absorbed eager MLA, and the halves are not
    interchangeable (the negative control keeps this from being a tautology).

    The paged variants write the draft block into the pool and read it back
    through a kernel, so a bug in that write shows up here as a parity failure
    rather than as lower acceptance length. Both kernels meet the same reference;
    they differ in whether the strict upper triangle comes from a separate fixup
    (TRTLLM) or from kv_bounds in one pass (CUTEDSL).

    Two facts say the batch is wired right rather than the tolerance being
    generous: ctx 20 is IDENTICAL across the eager and paged paths (a wrong page
    range or kv_bounds for the short request would not be), and ctx 40 is
    bit-identical to the same request run alone.
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
    if backend == "CUTEDSL":
        reason = md.cute_dsl_mla_decode_unavailability_reason()
        if reason is not None:
            pytest.skip(f"cute-dsl MLA decode unavailable: {reason}")
    torch.manual_seed(0)
    weights = _tiny_mla_weights()
    drafter = _build_mla_drafter(weights, dflash_attention_backend=backend)

    g = torch.Generator().manual_seed(42)
    # Two requests with UNEQUAL context lengths. MLA_CTX_LEN spans two pages and
    # puts the draft block in the second; MLA_SHORT_CTX_LEN sits inside the
    # first, so the two requests differ in page count, in intra-page offset and
    # in kv_bounds. At B == 1 none of those can disagree.
    requests = [
        (
            torch.randn(L, MLA_HIDDEN * NUM_CAPTURE, generator=g) * 0.5,
            torch.randn(MLA_BLOCK, MLA_HIDDEN, generator=g) * 0.5,
        )
        for L in (MLA_CTX_LEN, MLA_SHORT_CTX_LEN)
    ]

    out = _run_mla_block_decode(drafter, requests, paged=paged)
    expected = torch.stack(
        [
            _reference_mla_block_decode(weights, c.to(torch.bfloat16), n.to(torch.bfloat16))
            for c, n in requests
        ]
    )
    swapped = torch.stack(
        [
            _reference_mla_block_decode(
                weights, c.to(torch.bfloat16), n.to(torch.bfloat16), swap_kv_b=True
            )
            for c, n in requests
        ]
    )

    assert calls == {"fixup": 1 if paged else 0}, f"paged={paged} took the wrong branch: {calls}"
    assert out.shape[0] == len(requests)

    diff = (out - expected).abs().max().item()
    diff_swapped = (out - swapped).abs().max().item()
    # Job 3049452, per request vs the bf16 reference: ctx 40 is 0.023438 eager /
    # 0.015625 paged, ctx 20 is 0.015625 in both. Every value is an exact
    # multiple of 2^-6 -- one bf16 ULP here -- so 0.03 leaves ~1.5 ULP.
    assert diff < 0.03, f"MLA parity failed: max abs diff {diff}"
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

    assert drafter.shift_label
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
@pytest.mark.parametrize(
    "backend,variant",
    [("VANILLA", "eager"), ("TRTLLM", "trtllm_gen"), ("CUTEDSL", "cute_dsl")],
)
def test_mla_dspark_backend_selects_its_own_block_decode(backend, variant):
    """The field picks among this class's implementations, not the worker's.

    Neither worker op set may be loaded -- otherwise a drafter that never calls
    them drags in an optional dependency, and the worker's per-backend shape
    checks (which the absorbed 64:1 / head_dim 576 shape fails) bind on a path
    that does not use them. What the field DOES drive is _mla_block_decode_variant,
    and getting that wrong is silent: every variant returns the same shape.
    """
    from tensorrt_llm._torch.model_config import ModelConfig
    from tensorrt_llm._torch.models.modeling_dspark import (
        MLADSparkForCausalLM,
        cute_dsl_mla_decode_unavailability_reason,
    )

    if backend == "CUTEDSL":
        reason = cute_dsl_mla_decode_unavailability_reason()
        if reason is not None:
            pytest.skip(f"cute-dsl MLA decode unavailable: {reason}")

    model_config = ModelConfig(pretrained_config=_tiny_mla_config(), attn_backend="VANILLA")
    drafter = MLADSparkForCausalLM(model_config, dflash_attention_backend=backend)
    assert drafter._uses_worker_attention_backend is False
    assert drafter.dflash_attention_backend == backend
    assert drafter._dflash_flash_attention is None
    assert drafter._dflash_trtllm_gen_ops is None
    assert drafter._mla_block_decode_variant(paged=True) == variant
    # No page tables, no kernel: the private arena is eager whatever was asked.
    assert drafter._mla_block_decode_variant(paged=False) == "eager"


@needs_cuda
@needs_absorbed_mla
def test_mla_dspark_cute_dsl_takes_a_single_page_batch_one():
    """One page and one request -- the page table the runtime used to reject.

    (max_blocks, B) == (1, 1) has stride (1, 1) and no dimension with size > 1,
    so CuTe DSL's layout deduction refused to pick a leading dimension even though
    with both extents 1 the choice cannot change an address. Reachable whenever
    the draft pool is one page deep and a single request is resident -- a
    first-draft-step shape, and since AUTO resolves to CUTEDSL, the default path.

    The parity bound is the same one the batched test uses; a wrongly deduced
    axis would show up as a magnitude error, not a last-bit one.
    """
    import tensorrt_llm._torch.models.modeling_dspark as md

    reason = md.cute_dsl_mla_decode_unavailability_reason()
    if reason is not None:
        pytest.skip(f"cute-dsl MLA decode unavailable: {reason}")

    torch.manual_seed(0)
    weights = _tiny_mla_weights()
    drafter = _build_mla_drafter(weights, dflash_attention_backend="CUTEDSL")
    g = torch.Generator().manual_seed(42)
    captured = torch.randn(MLA_SHORT_CTX_LEN, MLA_HIDDEN * NUM_CAPTURE, generator=g) * 0.5
    noise = torch.randn(MLA_BLOCK, MLA_HIDDEN, generator=g) * 0.5

    out = _run_mla_block_decode(drafter, [(captured, noise)], paged=True)
    expected = _reference_mla_block_decode(
        weights, captured.to(torch.bfloat16), noise.to(torch.bfloat16)
    )
    diff = (out[0] - expected).abs().max().item()
    assert diff < 0.03, f"single-page parity failed: max abs diff {diff}"


@needs_cuda
@needs_absorbed_mla
@pytest.mark.parametrize("context_kv_fp8", [False, True])
def test_mla_dspark_fused_rope_is_bitwise(context_kv_fp8):
    """The fused MLA RoPE kernels change launches, not values.

    The per-layer block kernel (query rope half + block latent norm/rope +
    pool store, through scratch rows for an FP8 pool) and the one-launch
    context latents must reproduce the separate launches bit for bit: two
    unequal requests through the paged cute-dsl block decode, and the
    layer-major context latents against the per-layer ones.
    """
    import tensorrt_llm._torch.models.modeling_dspark as md

    reason = md.cute_dsl_mla_decode_unavailability_reason()
    if reason is not None:
        pytest.skip(f"cute-dsl MLA decode unavailable: {reason}")

    torch.manual_seed(0)
    weights = _tiny_mla_weights()
    # An FP8 pool is bound only to a drafter built for it (context_kv_dtype="fp8").
    drafter = _build_mla_drafter(
        weights,
        dflash_attention_backend="CUTEDSL",
        context_kv_dtype="fp8" if context_kv_fp8 else "auto",
    )
    if not drafter._fused_mla_rope:
        pytest.skip("the fused MLA RoPE kernels do not fit here")
    g = torch.Generator().manual_seed(42)
    requests = [
        (
            torch.randn(L, MLA_HIDDEN * NUM_CAPTURE, generator=g) * 0.5,
            torch.randn(MLA_BLOCK, MLA_HIDDEN, generator=g) * 0.5,
        )
        for L in (MLA_CTX_LEN, MLA_SHORT_CTX_LEN)
    ]
    captured = requests[0][0].to("cuda", torch.bfloat16)
    projected = drafter.project_target_hidden(captured)
    positions = torch.arange(37, 37 + captured.shape[0], device="cuda")

    drafter._fused_mla_rope = False
    base = _run_mla_block_decode(drafter, requests, paged=True, fp8=context_kv_fp8)
    ctx_base, _ = drafter.precompute_context_kv(projected, positions)
    drafter._fused_mla_rope = True
    # The folded query head (FP8 pool) rounds differently; it has its own test.
    drafter._qb_folded = None
    fused = _run_mla_block_decode(drafter, requests, paged=True, fp8=context_kv_fp8)
    ctx_fused, _ = drafter.precompute_context_kv(projected, positions)

    assert torch.equal(fused.view(torch.int32), base.view(torch.int32))
    assert torch.equal(ctx_fused.view(torch.int16), ctx_base.view(torch.int16))


@needs_cuda
@needs_absorbed_mla
def test_mla_dspark_fp8_query_head_agrees_to_drafter_rounding():
    """Over an FP8 pool, q_a_layernorm folded into q_b plus draft_qhead (query
    RoPE, absorption, block latent store, FP8 query in one launch) gives the
    block decode of the separate launches up to drafter rounding."""
    import tensorrt_llm._torch.models.modeling_dspark as md

    reason = md.cute_dsl_mla_decode_unavailability_reason()
    if reason is not None:
        pytest.skip(f"cute-dsl MLA decode unavailable: {reason}")

    torch.manual_seed(0)
    weights = _tiny_mla_weights()
    drafter = _build_mla_drafter(
        weights, dflash_attention_backend="CUTEDSL", context_kv_dtype="fp8"
    )
    if not drafter._fused_mla_qhead:
        pytest.skip("the folded MLA query head does not fit here")
    g = torch.Generator().manual_seed(7)
    requests = [
        (
            torch.randn(L, MLA_HIDDEN * NUM_CAPTURE, generator=g) * 0.5,
            torch.randn(MLA_BLOCK, MLA_HIDDEN, generator=g) * 0.5,
        )
        for L in (MLA_CTX_LEN, MLA_SHORT_CTX_LEN)
    ]
    qhead = _run_mla_block_decode(drafter, requests, paged=True, fp8=True)
    assert drafter._qb_folded is not None
    folded, drafter._qb_folded = drafter._qb_folded, None
    separate = _run_mla_block_decode(drafter, requests, paged=True, fp8=True)
    drafter._qb_folded = folded

    rel_l2 = ((qhead - separate).norm() / separate.norm()).item()
    assert rel_l2 < 2e-2, rel_l2


@needs_cuda
def test_mla_dspark_mlp_split_at_the_down_projection_is_exact():
    """The pushed down_proj site runs the MLP up to its down projection
    (GatedMLP.forward_activation) and the projection itself at the all-reduce:
    together bitwise the module's forward. Without the TP fusions the sites are off."""
    torch.manual_seed(3)
    drafter = _build_mla_drafter(_tiny_mla_weights())
    assert not drafter._ar_push
    mlp = drafter.model.layers[0].mlp
    x = torch.randn(7, MLA_HIDDEN, device="cuda").bfloat16()
    with torch.inference_mode():
        h2 = mlp.forward_activation(x)
        assert h2 is not None
        split = mlp.down_proj(h2)
        whole = mlp(x)
    assert torch.equal(split.view(torch.int16), whole.view(torch.int16))


@needs_cuda
def test_mla_dspark_auto_backend_resolves_to_cutedsl():
    """AUTO is the one-pass kernel, and it must reach the dispatch as such.

    Asserting the dispatch and not just the field: every variant returns the
    same shape, so a default that resolved but never reached
    _mla_block_decode_variant would be silent.
    """
    from tensorrt_llm._torch.model_config import ModelConfig
    from tensorrt_llm._torch.models.modeling_dspark import (
        MLADSparkForCausalLM,
        cute_dsl_mla_decode_unavailability_reason,
    )

    reason = cute_dsl_mla_decode_unavailability_reason()
    if reason is not None:
        pytest.skip(f"cute-dsl MLA decode unavailable: {reason}")
    model_config = ModelConfig(pretrained_config=_tiny_mla_config(), attn_backend="VANILLA")
    drafter = MLADSparkForCausalLM(model_config, dflash_attention_backend="AUTO")
    assert drafter.dflash_attention_backend == "CUTEDSL"
    assert drafter._mla_block_decode_variant(paged=True) == "cute_dsl"


@needs_cuda
def test_gqa_dspark_auto_backend_prefers_trtllm_and_degrades():
    """AUTO on the GQA drafter follows the op set's own availability probe.

    VANILLA needs the optional flash-attn package and TRTLLM needs flashinfer
    on SM100/SM103, so AUTO must consult the probe rather than assume either.
    """
    from tensorrt_llm._torch.model_config import ModelConfig
    from tensorrt_llm._torch.models.modeling_dspark import GQADSparkForCausalLM
    from tensorrt_llm._torch.speculative.dflash_attention import (
        dflash_trtllm_gen_unavailability_reason,
    )

    model_config = ModelConfig(pretrained_config=_tiny_config(True), attn_backend="VANILLA")
    drafter = GQADSparkForCausalLM(model_config, dflash_attention_backend="AUTO")
    expected = "VANILLA" if dflash_trtllm_gen_unavailability_reason() else "TRTLLM"
    assert drafter.dflash_attention_backend == expected


def test_mla_dspark_still_rejects_an_unknown_backend():
    """Dropping the VANILLA-only guard must not drop the typo check."""
    from tensorrt_llm._torch.model_config import ModelConfig
    from tensorrt_llm._torch.models.modeling_dspark import MLADSparkForCausalLM

    model_config = ModelConfig(pretrained_config=_tiny_mla_config(), attn_backend="VANILLA")
    with pytest.raises(ValueError, match="attention backend must be one of"):
        MLADSparkForCausalLM(model_config, dflash_attention_backend="FLASHINFER")


def test_gqa_dspark_rejects_the_mla_only_backend():
    """CUTEDSL is MLA-only; the GQA drafter must not accept it silently."""
    from tensorrt_llm._torch.model_config import ModelConfig
    from tensorrt_llm._torch.models.modeling_dspark import GQADSparkForCausalLM

    model_config = ModelConfig(pretrained_config=_tiny_config(True), attn_backend="VANILLA")
    with pytest.raises(ValueError, match="attention backend must be one of"):
        GQADSparkForCausalLM(model_config, dflash_attention_backend="CUTEDSL")


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

    The bound comes from dflash.py, so it is called here rather than restated: a
    test computing `allocated - block_size` itself would still pass against a path
    truncating to `allocated`. The MLA writes the block's latents at
    ctx_len..ctx_len+block_size, so that regression puts the first on the first
    unallocated page, clamped from a negative placeholder to 0 -- another request's.
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

    # Module-granular on purpose: a fused module (gate_up_proj) is stored
    # unfused, and dropping half is a parameter-level gap the loader's naming
    # cannot see. Derive the prefix -- DFlash checkpoints omit `model.`.
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


def test_mla_dspark_raises_when_the_requested_backend_cannot_run(monkeypatch):
    """An explicit backend the build cannot serve must not degrade silently.

    The old behaviour fell through ``_mla_block_decode_variant`` to the eager
    reference: identical output, orders slower, nothing raised. Only a log line
    distinguished it, and that line named the GQA op set this drafter never
    loads.
    """
    from tensorrt_llm._torch.model_config import ModelConfig
    from tensorrt_llm._torch.models.modeling_dspark import MLADSparkForCausalLM

    monkeypatch.setattr(MLADSparkForCausalLM, "_mla_decode_op", staticmethod(lambda: None))
    model_config = ModelConfig(pretrained_config=_tiny_mla_config(), attn_backend="VANILLA")
    with pytest.raises(ValueError, match="trtllm_batch_decode_with_kv_cache_mla"):
        MLADSparkForCausalLM(model_config, dflash_attention_backend="TRTLLM")


def test_mla_dspark_auto_does_not_degrade_to_the_eager_reference(monkeypatch):
    """AUTO on this family propagates the reason instead of falling back.

    The GQA drafter degrades because VANILLA/FlashAttention is a real path on a
    smaller part. This drafter needs a 288 GB Blackwell part to hold the target
    at all, so the eager reference is never the answer -- see
    ``_auto_fallback_attention_backend = None``.
    """
    from tensorrt_llm._torch.model_config import ModelConfig
    from tensorrt_llm._torch.models import modeling_dspark
    from tensorrt_llm._torch.models.modeling_dspark import MLADSparkForCausalLM

    # AUTO's pick is CUTEDSL; make it unavailable, and TRTLLM too, so a
    # degrade to either kernel backend would also be caught.
    monkeypatch.setattr(
        modeling_dspark, "cute_dsl_mla_decode_unavailability_reason", lambda: "probe says no"
    )
    monkeypatch.setattr(MLADSparkForCausalLM, "_mla_decode_op", staticmethod(lambda: None))
    model_config = ModelConfig(pretrained_config=_tiny_mla_config(), attn_backend="VANILLA")
    with pytest.raises(ValueError, match="does not degrade"):
        MLADSparkForCausalLM(model_config, dflash_attention_backend="AUTO")


def test_gqa_dspark_auto_still_degrades(monkeypatch):
    """The base policy is unchanged: a GQA drafter falls back rather than raise."""
    from tensorrt_llm._torch.model_config import ModelConfig
    from tensorrt_llm._torch.models import modeling_dflash
    from tensorrt_llm._torch.models.modeling_dspark import GQADSparkForCausalLM

    monkeypatch.setattr(
        modeling_dflash, "dflash_trtllm_gen_unavailability_reason", lambda: "probe says no"
    )
    model_config = ModelConfig(pretrained_config=_tiny_config(True), attn_backend="VANILLA")
    drafter = GQADSparkForCausalLM(model_config, dflash_attention_backend="AUTO")
    assert drafter.dflash_attention_backend == "VANILLA"


# ---------------------------------------------------------------------------
# Context-KV-only drafter (disaggregated context server).
# ---------------------------------------------------------------------------


@needs_cuda
def test_context_kv_only_drafter_keeps_only_the_context_kv_path():
    weights = _tiny_mla_weights()
    full = _build_mla_drafter(weights)
    ctx = _build_mla_drafter(weights, context_kv_only=True)

    backbone = ctx.draft_model_full.model
    assert backbone.embed_tokens is None and backbone.norm is None
    assert ctx.lm_head is None and ctx.markov_w1 is None
    for layer in backbone.layers:
        attn = layer.self_attn
        assert layer.mlp is None
        assert attn.q_b_proj is None and attn.kv_b_proj is None and attn.o_proj is None
        assert attn.kv_a_proj_with_mqa is not None and attn.kv_a_layernorm is not None

    def param_bytes(module):
        return sum(p.numel() * p.element_size() for p in module.parameters())

    kept = (MLA_HIDDEN * MLA_HIDDEN * NUM_CAPTURE + MLA_HIDDEN) * 2 + MLA_LAYERS * (
        MLA_LATENT * MLA_HIDDEN + MLA_KV_LORA
    ) * 2
    assert param_bytes(ctx) == kept
    assert param_bytes(ctx) < param_bytes(full) / 4

    torch.manual_seed(0)
    captured = torch.randn(37, MLA_HIDDEN * NUM_CAPTURE, device="cuda", dtype=torch.bfloat16)
    positions = torch.arange(5, 42, device="cuda")

    def latents(drafter):
        latent, v = drafter.precompute_context_kv(
            drafter.project_target_hidden(captured), positions
        )
        assert v is None
        return latent

    assert torch.equal(latents(ctx), latents(full))


def test_context_kv_only_is_rejected_for_a_drafter_that_drafts_elsewhere():
    from tensorrt_llm._torch.models.modeling_dspark import _build_dspark_draft

    for embedded, kv_lora_rank in ((False, None), (True, MLA_KV_LORA)):
        spec = SimpleNamespace(
            context_kv_only=True,
            context_kv_dtype="auto",
            attention_backend="AUTO",
            draft_is_embedded_in_target=embedded,
        )
        model_config = SimpleNamespace(spec_config=spec)
        pretrained = SimpleNamespace(kv_lora_rank=kv_lora_rank, qk_rope_head_dim=MLA_ROPE)
        draft_config = SimpleNamespace(pretrained_config=pretrained)
        with pytest.raises(ValueError, match="standalone MLA DSpark"):
            _build_dspark_draft(model_config, draft_config, None, None)


def test_context_kv_only_requires_disaggregated_serving():
    from tensorrt_llm._torch.pyexecutor._util import KvCacheCreator
    from tensorrt_llm.llmapi.llm_args import DSparkDecodingConfig

    creator = object.__new__(KvCacheCreator)
    creator._speculative_config = DSparkDecodingConfig(max_draft_len=4, context_kv_only=True)
    creator._is_disagg = True
    creator._validate_context_kv_only()
    creator._is_disagg = False
    with pytest.raises(ValueError, match="context_kv_only"):
        creator._validate_context_kv_only()
    creator._speculative_config = None
    creator._validate_context_kv_only()


@needs_cuda
def test_context_server_takes_the_drafter_out_of_the_kv_budget_not_the_headroom():
    """The KV budget gives back (1 - fraction) of what the drafter keeps resident,
    so a context-KV-only drafter does not shrink PyTorch's headroom."""
    from tensorrt_llm._torch.pyexecutor._util import KvCacheCreator
    from tensorrt_llm.llmapi.llm_args import DSparkDecodingConfig

    head = torch.nn.Linear(16, 8, bias=False, device="cuda")
    draft = torch.nn.Module()
    draft.fc = torch.nn.Linear(32, 16, bias=False, device="cuda")
    draft.lm_head = head  # shared with the target: not the drafter's
    draft.table = torch.zeros(100, device="cuda")  # derived, not registered
    draft.blocks = draft.fc.weight[:, :16]  # a view: counted once
    model = torch.nn.Module()
    model.model = torch.nn.Linear(16, 16, bias=False, device="cuda")
    model.lm_head = head
    model.draft_model = draft
    metadata = SimpleNamespace(captures=torch.zeros(8, 16, device="cuda"), host_rows=torch.zeros(8))

    creator = object.__new__(KvCacheCreator)
    creator._model_engine = SimpleNamespace(model=model, spec_metadata=metadata)
    creator._speculative_config = DSparkDecodingConfig(max_draft_len=4, context_kv_only=True)
    resident = draft.fc.weight.nbytes + draft.table.nbytes + metadata.captures.nbytes
    assert creator._get_context_server_spec_reserve(0.7) == int((1.0 - 0.7) * resident)
    creator._speculative_config.context_kv_only = False
    assert creator._get_context_server_spec_reserve(0.7) == 0


# ---------------------------------------------------------------------------
# FP8 storage of the drafter context KV (speculative_config.context_kv_dtype).
# ---------------------------------------------------------------------------


def test_context_kv_dtype_defaults_to_the_drafter_dtype():
    from tensorrt_llm.llmapi.llm_args import DSparkDecodingConfig

    assert DSparkDecodingConfig(max_draft_len=2).context_kv_dtype == "auto"
    assert DSparkDecodingConfig(max_draft_len=2, context_kv_dtype="fp8").context_kv_dtype == "fp8"
    with pytest.raises(ValueError):
        DSparkDecodingConfig(max_draft_len=2, context_kv_dtype="nvfp4")


def test_fp8_context_kv_is_rejected_for_a_drafter_without_an_mla_latent_cache():
    from tensorrt_llm._torch.models.modeling_dspark import _build_dspark_draft

    for embedded, kv_lora_rank in ((False, None), (True, MLA_KV_LORA)):
        spec = SimpleNamespace(
            context_kv_only=False,
            context_kv_dtype="fp8",
            attention_backend="AUTO",
            draft_is_embedded_in_target=embedded,
        )
        pretrained = SimpleNamespace(kv_lora_rank=kv_lora_rank, qk_rope_head_dim=MLA_ROPE)
        with pytest.raises(ValueError, match="context_kv_dtype is implemented for standalone MLA"):
            _build_dspark_draft(
                SimpleNamespace(spec_config=spec),
                SimpleNamespace(pretrained_config=pretrained),
                None,
                None,
            )


@needs_cuda
def test_fp8_context_kv_needs_the_cutedsl_block_decode():
    from tensorrt_llm._torch.model_config import ModelConfig
    from tensorrt_llm._torch.models.modeling_dspark import MLADSparkForCausalLM

    def build(**kwargs):
        model_config = ModelConfig(pretrained_config=_tiny_mla_config(), attn_backend="TRTLLM")
        return MLADSparkForCausalLM(model_config, **kwargs)

    with pytest.raises(ValueError, match="CUTEDSL"):
        build(dflash_attention_backend="VANILLA", context_kv_dtype="fp8")
    # A context server never runs the block decode, so any backend stores it.
    ctx = build(dflash_attention_backend="VANILLA", context_kv_only=True, context_kv_dtype="fp8")
    assert ctx.context_kv_fp8
    assert not build(dflash_attention_backend="VANILLA").context_kv_fp8


@needs_cuda
def test_fp8_paged_context_store_writes_saturated_fp8():
    from tensorrt_llm._torch.speculative.dflash import DFlashWorker
    from tensorrt_llm._torch.speculative.dflash_attention import to_draft_fp8

    page, num_pages, layers, num_tokens = 32, 6, 2, 11
    worker = object.__new__(DFlashWorker)
    worker._ctx_helix = None
    worker._ctx_store_stream = None
    worker._ctx_store_operands = ()
    worker._ctx_page_size = page
    worker._ctx_block_tables = torch.tensor([[4, 1, 5], [0, 2, 3]], device="cuda")
    worker._ctx_kv_buf = [
        torch.zeros(num_pages, 1, 1, page, MLA_LATENT, device="cuda").to(torch.float8_e4m3fn)
        for _ in range(layers)
    ]
    gen = torch.Generator(device="cuda").manual_seed(3)
    k = (torch.randn(num_tokens, layers, 1, MLA_LATENT, device="cuda", generator=gen) * 200).to(
        torch.bfloat16
    )
    rows = torch.tensor([0] * 6 + [1] * 5, device="cuda")
    positions = torch.tensor([0, 1, 31, 32, 33, 70, 5, 6, 40, 64, 95], device="cuda")
    worker._store_context_kv_paged(k, None, rows, positions)

    pages = worker._ctx_block_tables[rows, positions // page].long()
    offsets = positions % page
    for layer in range(layers):
        stored = worker._ctx_kv_buf[layer][pages, 0, 0, offsets]
        assert torch.equal(stored.view(torch.uint8), to_draft_fp8(k[:, layer, 0]).view(torch.uint8))
    assert not worker._ctx_kv_buf[0].float().isnan().any()


def _k3_block_decoder():
    """What ``MLADSparkForCausalLM._cute_dsl_block_decode`` reads, K3 DSpark geometry."""
    import types

    from tensorrt_llm._torch.models.modeling_dspark import (
        MLADSparkForCausalLM,
        _mla_fp8_decode_scales,
        _yarn_get_mscale,
    )

    cls = MLADSparkForCausalLM
    stub = SimpleNamespace(
        kv_lora_rank=512,
        qk_rope_head_dim=64,
        _num_heads=64,
        _cute_dsl_checked_shape=None,
        _mla_workspace=None,
    )
    # K3 DSpark YaRN: factor 32, mscale_all_dim 1; qk_nope 128 + rope 64.
    mscale = _yarn_get_mscale(32.0, 1.0)
    stub.softmax_scale = mscale * mscale / math.sqrt(128 + 64)
    stub._mla_fp8_scales = _mla_fp8_decode_scales(stub.softmax_scale)
    for name in (
        "_assert_cute_dsl_can_implement",
        "_mla_decode_workspace",
        "_cute_dsl_block_decode",
    ):
        setattr(stub, name, types.MethodType(getattr(cls, name), stub))
    return stub


@needs_absorbed_mla
@pytest.mark.parametrize("block_size", [5, 7])
def test_fp8_block_decode_follows_bf16_at_k3_shapes(block_size):
    """The FP8 decode reads to_draft_fp8 of the BF16 pool and query: against an
    FP32 reference on its own FP8 operands it is as close as the BF16 decode is
    to one on BF16 operands; against the BF16 decode only the E4M3 rounding of
    q and K remains."""
    from tensorrt_llm._torch.models.modeling_dspark import _build_mla_block_fixup
    from tensorrt_llm._torch.speculative.dflash_attention import to_draft_fp8

    heads, page, kv_lora, latent_dim = 64, 64, 512, 576
    stub = _k3_block_decoder()
    gen = torch.Generator(device="cuda").manual_seed(block_size)
    ctx = torch.tensor([300, 1700, 6400, 13700], device="cuda")
    pages_needed = (ctx + block_size - 1) // page + 1
    total = int(pages_needed.sum())
    perm = torch.randperm(total, device="cuda", generator=gen)
    table = torch.zeros(len(ctx), int(pages_needed.max()), dtype=torch.long, device="cuda")
    start = 0
    for b, n in enumerate(pages_needed.tolist()):
        table[b, :n] = perm[start : start + n]
        start += n
    values = torch.randn(total, 1, 1, page, latent_dim, device="cuda", generator=gen)
    values[..., :kv_lora] *= 0.5 + torch.rand(kv_lora, device="cuda", generator=gen)
    values[..., kv_lora:] *= 2.0
    pool = values.bfloat16()
    query = torch.randn(
        len(ctx), block_size, heads, latent_dim, device="cuda", generator=gen
    ).bfloat16()
    fixup = _build_mla_block_fixup(ctx, table, block_size, page)

    def reference(q, kv_pool):
        flat = kv_pool[:, 0, 0].reshape(-1, latent_dim)
        outs = []
        for b in range(q.shape[0]):
            pos = torch.arange(int(fixup.seq_lens_i32[b]), device="cuda")
            rows = fixup.page_tables_i32[b].long()[pos // page] * page + pos % page
            kv = flat[rows].float()
            scores = torch.einsum("shc,tc->sht", q[b].float(), kv) * stub.softmax_scale
            outs.append(torch.einsum("sht,tc->shc", scores.softmax(-1), kv[:, :kv_lora]))
        return torch.stack(outs)

    def rel_l2(a, e):
        return ((a.float() - e.float()).norm() / e.float().norm()).item()

    out_bf16 = stub._cute_dsl_block_decode(query, pool, fixup, block_size)
    pool_fp8 = to_draft_fp8(pool)
    out_fp8 = stub._cute_dsl_block_decode(query, pool_fp8, fixup, block_size)
    assert out_fp8.dtype == torch.bfloat16
    assert rel_l2(out_bf16, reference(query, pool)) < 0.005
    fp8_operands = reference(to_draft_fp8(query).bfloat16(), pool_fp8.bfloat16())
    assert rel_l2(out_fp8, fp8_operands) < 0.01
    cosine = F.cosine_similarity(out_fp8.float().reshape(-1), out_bf16.float().reshape(-1), dim=0)
    assert cosine.item() > 0.993 and rel_l2(out_fp8, out_bf16) < 0.12


@needs_absorbed_mla
@pytest.mark.parametrize("block_size", [5, 7])
def test_block_decode_split_plan_follows_the_planner_and_the_uniform_split(block_size):
    """The block decode plans its per-request split-KV counts with the target's
    planner (host mirror plan_mla_split_kv) from the step's lengths; another
    split changes only the reduction order."""
    import types

    from tensorrt_llm._torch.attention.backends.fmha.cute_dsl_mla import (
        _SPLIT_PLAN_MAX_REQUESTS,
        mla_split_kv_plan_geometry,
        plan_mla_split_kv,
    )
    from tensorrt_llm._torch.models.modeling_dspark import (
        MLADSparkForCausalLM,
        _build_mla_block_fixup,
    )

    heads, page, latent_dim = 64, 64, 576
    stub = _k3_block_decoder()
    stub._mla_split_plan = torch.zeros(_SPLIT_PLAN_MAX_REQUESTS, dtype=torch.int32, device="cuda")
    stub._mla_split_kv_plan = types.MethodType(MLADSparkForCausalLM._mla_split_kv_plan, stub)
    gen = torch.Generator(device="cuda").manual_seed(block_size)
    ctx = torch.tensor([300, 1700, 6400, 13700, 90], device="cuda")
    pages = (ctx + block_size + page - 1) // page
    table = torch.zeros(len(ctx), int(pages.max()), dtype=torch.long, device="cuda")
    start = 0
    for b, n in enumerate(pages.tolist()):
        table[b, :n] = torch.arange(start, start + n, device="cuda")
        start += n
    pool = torch.randn(start, 1, 1, page, latent_dim, device="cuda", generator=gen).bfloat16()
    query = torch.randn(
        len(ctx), block_size, heads, latent_dim, device="cuda", generator=gen
    ).bfloat16()
    fixup = _build_mla_block_fixup(ctx, table, block_size, page)

    plan = stub._mla_split_kv_plan(fixup.seq_lens_i32, heads, block_size)

    rows, waves = mla_split_kv_plan_geometry(heads, block_size)
    expected = plan_mla_split_kv(
        fixup.seq_lens_i32.tolist(), rows_per_request=rows, wave_clusters=waves
    )
    assert plan.tolist() == expected
    planned = stub._cute_dsl_block_decode(query, pool, fixup._replace(split_kvs=plan), block_size)
    uniform = stub._cute_dsl_block_decode(query, pool, fixup, block_size)
    rel_l2 = ((planned.float() - uniform.float()).norm() / uniform.float().norm()).item()
    assert rel_l2 < 5e-3


# ---------------------------------------------------------------------------
# Drafter context KV sharded over helix CP.
# ---------------------------------------------------------------------------


def _helix_rank_pool(values, seq, shard, page):
    """Rank ``shard.cp_rank``'s local pages of every request's global latents."""
    latent_dim = values.shape[-1]
    span = page * shard.cp_size
    local_pages = [(int(n) + span - 1) // span for n in seq.tolist()]
    table = torch.zeros(len(local_pages), max(local_pages), dtype=torch.long, device="cuda")
    pool = torch.zeros(sum(local_pages), 1, 1, page, latent_dim, device="cuda")
    start = 0
    for b, n in enumerate(local_pages):
        table[b, :n] = torch.arange(start, start + n, device="cuda")
        for k in range(n):
            first = (k * shard.cp_size + shard.cp_rank) * page
            rows = values[b, first : min(first + page, int(seq[b]))]
            pool[start + k, 0, 0, : rows.shape[0]] = rows
        start += n
    return pool, table


@needs_absorbed_mla
@pytest.mark.parametrize("fp8", [False, True], ids=["bf16", "fp8"])
@pytest.mark.parametrize("cp_size", [2, 4, 8])
def test_helix_sharded_block_decode_matches_the_full_context(cp_size, fp8):
    """Each helix rank attends with every head over its own pages (rank-local
    page table, lengths and kv_bounds, softmax stats out); merging the ranks'
    partials as _helix_post_process does equals the decode over the whole
    context. Ranks that hold none of a short request's pages contribute nothing."""
    from tensorrt_llm._torch.attention.attention import _helix_sanitize_empty_kv
    from tensorrt_llm._torch.models.modeling_dspark import _build_mla_block_fixup
    from tensorrt_llm._torch.speculative.dflash_attention import HelixDraftShard, to_draft_fp8

    heads, page, kv_lora, latent_dim, block_size = 64, 64, 512, 576, 7
    stub = _k3_block_decoder()
    gen = torch.Generator(device="cuda").manual_seed(cp_size)
    ctx = torch.tensor([40, 300, 1700, 6400], device="cuda")
    seq = ctx + block_size
    values = torch.randn(len(ctx), int(seq.max()), latent_dim, device="cuda", generator=gen)
    values[..., :kv_lora] *= 0.5
    query = torch.randn(
        len(ctx), block_size, heads, latent_dim, device="cuda", generator=gen
    ).bfloat16()

    def to_pool(pool):
        return to_draft_fp8(pool) if fp8 else pool.bfloat16()

    full_pool, full_table = _helix_rank_pool(values, seq, HelixDraftShard(1, 0, page), page)
    full_fixup = _build_mla_block_fixup(ctx, full_table, block_size, page)
    expected = stub._cute_dsl_block_decode(query, to_pool(full_pool), full_fixup, block_size)

    tokens = len(ctx) * block_size
    partials, stats = [], []
    for cp_rank in range(cp_size):
        shard = HelixDraftShard(cp_size, cp_rank, page)
        pool, table = _helix_rank_pool(values, seq, shard, page)
        fixup = _build_mla_block_fixup(ctx, table, block_size, page, shard)
        assert torch.equal(fixup.seq_lens_i32.long(), shard.local_lengths(seq))
        out, stat = stub._cute_dsl_block_decode(query, to_pool(pool), fixup, block_size)
        # [tokens, heads * kv_lora]: the layout _helix_post_process takes.
        out, stat = _helix_sanitize_empty_kv(
            out.view(tokens, heads * kv_lora), stat, fixup.kv_bounds_i32 == 0
        )
        partials.append(out)
        stats.append(stat)
    merged = torch.ops.trtllm.helix_post_process(
        torch.stack(partials).contiguous(), torch.stack(stats).contiguous(), 1.0
    ).view(tokens, heads, kv_lora)

    expected = expected.view(tokens, heads, kv_lora).float()
    rel_l2 = ((merged.float() - expected).norm() / expected.norm()).item()
    # Another KV split changes only the reduction order and rounding.
    assert rel_l2 < (0.02 if fp8 else 0.01)


@needs_cuda
def test_helix_context_store_writes_only_owned_positions():
    """A helix rank stores the positions of its own pages at their local slots;
    another rank's position leaves the slot it maps to unchanged."""
    from tensorrt_llm._torch.speculative.dflash import DFlashWorker
    from tensorrt_llm._torch.speculative.dflash_attention import HelixDraftShard

    page, layers = 4, 2
    worker = object.__new__(DFlashWorker)
    worker._ctx_helix = None
    worker._ctx_store_stream = None
    worker._ctx_store_operands = ()
    worker._ctx_page_size = page
    # Rank 1 of 2 owns global pages 1, 3, ... as its local pages 0, 1, ...
    worker._ctx_helix = HelixDraftShard(2, 1, page)
    worker._ctx_block_tables = torch.tensor([[3, 0, 2]], device="cuda")
    sentinel = -7.0
    worker._ctx_kv_buf = [
        torch.full((4, 1, 1, page, MLA_LATENT), sentinel, device="cuda").bfloat16()
        for _ in range(layers)
    ]
    positions = torch.tensor([6, 7, 8, 9], device="cuda")
    k = positions.view(-1, 1, 1, 1).expand(-1, layers, 1, MLA_LATENT).bfloat16().contiguous()
    worker._store_context_kv_paged(k, None, torch.zeros_like(positions), positions)

    for layer in range(layers):
        pool = worker._ctx_kv_buf[layer][:, 0, 0, :, 0]
        # Positions 6, 7 (global page 1) land in local page 0 = physical page 3.
        assert pool[3].tolist() == [sentinel, sentinel, 6.0, 7.0]
        # Positions 8, 9 (global page 2, rank 0's) map to local page 1 =
        # physical page 0 and leave it unchanged.
        assert pool[0].tolist() == [sentinel] * page


@needs_cuda
def test_helix_context_store_owned_position_wins_a_shared_slot():
    """A window wider than a page maps another rank's position and an owned one to
    the same local slot; the owned value is stored, whatever the launch order."""
    from tensorrt_llm._torch.speculative.dflash import DFlashWorker
    from tensorrt_llm._torch.speculative.dflash_attention import HelixDraftShard

    page = 4
    worker = object.__new__(DFlashWorker)
    worker._ctx_helix = None
    worker._ctx_store_stream = None
    worker._ctx_store_operands = ()
    worker._ctx_page_size = page
    # Rank 1 of 2: global page 0 (rank 0's) and page 1 (its own) both map to local page 0.
    worker._ctx_helix = HelixDraftShard(2, 1, page)
    worker._ctx_block_tables = torch.tensor([[2, 0]], device="cuda")
    worker._ctx_kv_buf = [torch.full((3, 1, 1, page, MLA_LATENT), -7.0, device="cuda").bfloat16()]
    positions = torch.arange(2 * page, device="cuda")
    k = positions.view(-1, 1, 1, 1).expand(-1, 1, 1, MLA_LATENT).bfloat16().contiguous()
    worker._store_context_kv_paged(k, None, torch.zeros_like(positions), positions)

    assert worker._ctx_kv_buf[0][2, 0, 0, :, 0].tolist() == [4.0, 5.0, 6.0, 7.0]
