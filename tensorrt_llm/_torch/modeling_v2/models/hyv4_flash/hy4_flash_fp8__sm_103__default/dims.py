# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Every per-rank width of HY4-Flash's text core and DFly2 drafter, from the config and Mapping.

The MLA latent is replicated across attention ranks (the KV cache is not split by heads); only
`q_b_proj`, `kv_b_proj`, `linear_gate` and `o_proj` are cut by head.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import NamedTuple

from tensorrt_llm._torch.pyexecutor.config_utils import (
    get_hyv4_flash_layer_masks,
    unwrap_hyv4_flash_text_config,
)

#: RoPE bases used when the config publishes no per-layer-type block.
THETA_SPARSE = 10_000_000.0
THETA_SWA = 10_000.0


class LayerGeom(NamedTuple):
    """What changes from one decoder layer to the next."""

    heads: int
    heads_local: int
    #: Sparse MLA over the indexer's top-k (no window).
    sparse: bool
    theta: float
    #: Attention window in tokens, or None.
    window: int | None


def _theta(text, layer_type: str, fallback: float) -> float:
    """This layer type's RoPE base from its own `rope_parameters` block (there is no global one)."""
    blocks = getattr(text, "rope_parameters", None) or {}
    block = blocks.get(layer_type)
    if isinstance(block, dict) and block.get("rope_theta") is not None:
        return float(block["rope_theta"])
    return float(fallback)


def _dfly_layers(cfg, text, name: str) -> tuple[int, ...]:
    """A DFly2 layer list: the composite config first, then the text config."""
    value = getattr(cfg, name, None)
    if value is None:
        value = getattr(text, name, None)
    return tuple(int(i) for i in (value or ()))


def _vision_v(cfg) -> SimpleNamespace | None:
    """The vision tower's geometry (replicated on every rank), or None without `vision_config`.

    The non-shape asserts pin the checkpoint to the entries the tower calls.
    """
    v = getattr(cfg, "vision_config", None)
    if v is None:
        return None

    def field(name):
        value = getattr(v, name, None)
        assert value is not None, f"vision_config publishes no {name}"
        return value

    hidden = int(field("hidden_size"))
    heads = int(field("num_attention_heads"))
    patch = int(field("patch_size"))
    merge = int(field("spatial_merge_size"))
    channels = int(field("num_channels"))
    assert hidden % heads == 0, (hidden, heads)
    assert int(field("num_key_value_heads")) == heads, "the tower's attention is MHA"
    assert str(field("hidden_act")) == "gelu", (
        f"the tower's GELU entry is the erf form; vision_config says {field('hidden_act')!r}"
    )
    assert str(field("interpolate_mode")) == "bilinear", (
        f"the position table resamples bilinearly; vision_config says {field('interpolate_mode')!r}"
    )
    assert int(field("temporal_patch_size")) == 1, "a pixel row is (channels, patch, patch)"
    # `(max_image_size / patch)**2` patch rows plus the cls row the reference skips.
    position_edge = int(field("max_image_size")) // patch
    declared = int(getattr(v, "max_vit_seq_len", position_edge**2))
    assert declared == position_edge**2, (
        f"vision_config's max_vit_seq_len {declared} is not the position grid "
        f"{position_edge}x{position_edge}"
    )
    return SimpleNamespace(
        depth=int(field("num_hidden_layers")),
        hidden=hidden,
        heads=heads,
        head_dim=hidden // heads,
        inter=int(field("intermediate_size")),
        merge=merge,
        patch=patch,
        channels=channels,
        out_hidden=int(field("out_hidden_size")),
        position_edge=position_edge,
        pixel_width=channels * patch * patch,
        # The reference's `layer_norm_eps` maps onto `rms_norm_eps`.
        eps=float(field("rms_norm_eps")),
        merge_dim=hidden * merge * merge,
    )


def dims(core) -> SimpleNamespace:
    """Every per-rank width, from the config and the engine's Mapping."""
    model_config = core.model_config
    cfg = model_config.pretrained_config
    text = unwrap_hyv4_flash_text_config(cfg)
    m = model_config.mapping

    attention_dp = bool(m.enable_attention_dp)
    attn_tp = 1 if attention_dp else m.tp_size
    attn_tp_rank = 0 if attention_dp else m.tp_rank

    heads_sparse = int(text.num_attention_heads)
    heads_swa = int(text.swa_num_attention_heads)
    assert heads_sparse % attn_tp == 0, (heads_sparse, attn_tp)
    assert heads_swa % attn_tp == 0, (heads_swa, attn_tp)

    sparse_mask, _ = get_hyv4_flash_layer_masks(cfg)
    sparse_layers = tuple(i for i, is_sparse in enumerate(sparse_mask) if is_sparse)
    num_layers = int(text.num_hidden_layers)
    assert len(sparse_mask) == num_layers, (len(sparse_mask), num_layers)
    sparse_set = set(sparse_layers)

    nope = int(text.qk_nope_head_dim)
    rope_dim = int(text.qk_rope_head_dim)
    v_dim = int(text.v_head_dim)
    kv_lora = int(text.kv_lora_rank)
    q_lora = int(text.q_lora_rank)
    assert int(getattr(text, "swa_qk_nope_head_dim", nope)) == nope, (
        "the windowed layers' no-PE width differs from the sparse layers'"
    )
    assert int(getattr(text, "swa_v_head_dim", v_dim)) == v_dim
    assert int(getattr(text, "swa_kv_lora_rank", kv_lora)) == kv_lora

    window = int(text.sliding_window)
    theta_sparse = _theta(text, "deepseek_sparse_attention", THETA_SPARSE)
    theta_swa = _theta(text, "sliding_attention", THETA_SWA)
    layers = tuple(
        LayerGeom(
            heads=heads_sparse if i in sparse_set else heads_swa,
            heads_local=(heads_sparse if i in sparse_set else heads_swa) // attn_tp,
            sparse=i in sparse_set,
            theta=theta_sparse if i in sparse_set else theta_swa,
            window=None if i in sparse_set else window,
        )
        for i in range(num_layers)
    )

    # The dense layer and the shared expert split over the attention-TP group.
    mlp_tp, mlp_tp_rank = attn_tp, attn_tp_rank
    dense_inter = int(text.intermediate_size)
    shared_inter = int(text.moe_intermediate_size)
    for name, full in (("dense", dense_inter), ("shared", shared_inter)):
        assert full % mlp_tp == 0, f"{name} intermediate {full} does not split {mlp_tp} ways"

    num_experts = int(text.n_routed_experts)
    assert num_experts % m.moe_ep_size == 0, (num_experts, m.moe_ep_size)
    local_experts = num_experts // m.moe_ep_size
    moe_inter = int(text.moe_intermediate_size)
    assert moe_inter % m.moe_tp_size == 0, (moe_inter, m.moe_tp_size)

    # The config's Engram ids are one-based decoder layers.
    engram_ids = tuple(int(i) for i in (text.engram_layer_ids or ()))
    engram_layers = tuple(i - 1 for i in engram_ids)
    head_vocab_sizes = {
        i: [int(size) for order in text.engram_layer_head_vocab_sizes[str(i)] for size in order]
        for i in engram_ids
    }
    n_head_per_ngram = int(text.engram_n_head_per_ngram)
    n_embed_per_ngram = int(text.engram_n_embed_per_ngram)
    for i, sizes in head_vocab_sizes.items():
        assert len(sizes) == (int(text.engram_max_ngram_size) - 1) * n_head_per_ngram, (
            f"engram layer {i} publishes {len(sizes)} head vocabularies"
        )

    index_n_heads = int(text.index_n_heads)
    index_head_dim = int(text.index_head_dim)

    # DFly2 draft layers reuse a sparse layer's KV and its picks, at the indexer's own top-k.
    draft_source_layers = _dfly_layers(cfg, text, "dfly_target_kv_layers")
    draft_index_layers = _dfly_layers(cfg, text, "dfly_target_index_layer")
    if draft_source_layers and draft_index_layers:
        assert draft_source_layers == draft_index_layers, (
            f"dfly_target_kv_layers={list(draft_source_layers)} and "
            f"dfly_target_index_layer={list(draft_index_layers)} differ"
        )
    for i in draft_source_layers:
        assert i in sparse_set, f"DFly2 drafts over layer {i}, which has no indexer"
    draft_topk = getattr(cfg, "dfly_target_index_topk", None)
    if draft_topk is None:
        draft_topk = getattr(text, "dfly_target_index_topk", None)
    if draft_source_layers and int(draft_topk or 0) != int(text.index_topk):
        raise ValueError(
            f"the picks state holds index_topk={int(text.index_topk)} ids; the checkpoint asks "
            f"the draft layers for dfly_target_index_topk={draft_topk}"
        )

    return SimpleNamespace(
        num_layers=num_layers,
        hidden=int(text.hidden_size),
        vocab=int(text.vocab_size),
        dtype=model_config.torch_dtype,
        vision=_vision_v(cfg),
        # attention
        attention_dp=attention_dp,
        attn_tp=attn_tp,
        attn_tp_rank=attn_tp_rank,
        world=m.world_size,
        rank=m.rank,
        layers=layers,
        sparse_layers=sparse_layers,
        draft_source_layers=draft_source_layers,
        heads_sparse=heads_sparse,
        nope=nope,
        rope_dim=rope_dim,
        v_dim=v_dim,
        kv_lora=kv_lora,
        q_lora=q_lora,
        qk_dim=nope + rope_dim,
        lat_dim=kv_lora + rope_dim,
        # The FMHA scales by q_scaling * sqrt(qk_nope + qk_rope) / sqrt(kv_lora + rope) on the
        # sparse branch, so 1.0 is the checkpoint's 1/sqrt(256).
        q_scaling=1.0,
        # compressed indexer
        index_topk=int(text.index_topk),
        index_kpool=int(getattr(text, "index_kpool", 4)),
        index_n_heads=index_n_heads,
        index_head_dim=index_head_dim,
        index_scale=float(index_n_heads**-0.5 * index_head_dim**-0.5),
        # MLP
        mlp_tp=mlp_tp,
        mlp_tp_rank=mlp_tp_rank,
        dense_inter=dense_inter // mlp_tp,
        shared_inter=shared_inter // mlp_tp,
        first_dense=int(text.first_k_dense_replace),
        # MoE
        num_experts=num_experts,
        top_k=int(text.num_experts_per_tok),
        n_group=int(text.n_group),
        topk_group=int(text.topk_group),
        routed_scale=float(text.routed_scaling_factor),
        swiglu_limit=float(text.swiglu_limit),
        moe_tp=m.moe_tp_size,
        moe_tp_rank=m.moe_tp_rank,
        moe_inter=moe_inter // m.moe_tp_size,
        #: The experts' input width, whole on every rank: MoE TP splits the intermediate.
        latent=moe_inter,
        moe_ep=m.moe_ep_size,
        moe_ep_rank=m.moe_ep_rank,
        local_experts=local_experts,
        expert_offset=local_experts * m.moe_ep_rank,
        # hyper-connections
        hc_mult=int(text.hc_mult),
        hc_eps=float(text.hc_eps),
        hc_magnitude=float(text.hc_magnitude),
        # engram; keyed by zero-based layer except `engram_head_vocab_sizes` (config ids)
        engram_layers=engram_layers,
        engram_ids=engram_ids,
        engram_head_vocab_sizes=head_vocab_sizes,
        #: One disjoint row block per head, in head order.
        engram_table_rows={i - 1: sum(sizes) for i, sizes in head_vocab_sizes.items()},
        engram_head_offsets={
            i - 1: [sum(sizes[:k]) for k in range(len(sizes))]
            for i, sizes in head_vocab_sizes.items()
        },
        engram_num_heads=(int(text.engram_max_ngram_size) - 1) * n_head_per_ngram,
        engram_head_dim=n_embed_per_ngram // n_head_per_ngram,
        engram_size=(int(text.engram_max_ngram_size) - 1) * n_embed_per_ngram,
        engram_max_ngram=int(text.engram_max_ngram_size),
        engram_seed=int(getattr(text, "engram_seed", 0)),
        engram_pad_id=int(text.engram_pad_id),
        engram_table_size=int(getattr(text, "engram_lookup_table_size", None) or 120818),
        engram_fp8=bool(text.engram_fp8_embedding),
        # engine
        rms_eps=float(text.rms_norm_eps),
        #: The engine's per-step token budget.
        max_num_tokens=int(model_config.max_num_tokens),
        # The PyTorch flow carries no `tokens_per_block` on ModelConfig; the DSA pool uses 32.
        tokens_per_block=int(getattr(model_config, "tokens_per_block", 0) or 32),
        max_seq_len=int(getattr(model_config, "max_seq_len", 0) or 0),
        max_batch=int(getattr(model_config, "max_batch_size", 0) or 0),
        max_context_len=int(
            getattr(model_config, "max_context_length", 0)
            or getattr(model_config, "max_seq_len", 0)
            or 0
        ),
    )


#: The draft FFN and hidden-correction widths as multiples of the hidden size; the checkpoint
#: config publishes neither.
DRAFT_MLP_MULTIPLE = 3
DRAFT_CORRECTION_MULTIPLE = 1

#: Alignment of the drafter's per-row picked-index buffer.
DRAFT_INDEX_ALIGN = 128

#: Five drafts and the anchor, when the spec config does not say.
DEFAULT_BLOCK_WIDTH = 6


def round_up(value: int, multiple: int) -> int:
    return -(-value // multiple) * multiple


def _dfly(text, cfg, key, default=None):
    for block in (text, cfg):
        value = getattr(block, key, None)
        if value is not None:
            return value
    return default


def draft_dims(core) -> SimpleNamespace:
    """The DFly2 drafter's widths: five layers on the target's sparse-layer MLA geometry."""
    d = dims(core)
    cfg = core.model_config.pretrained_config
    text = unwrap_hyv4_flash_text_config(cfg)

    kv_layers = tuple(int(i) for i in _dfly(text, cfg, "dfly_target_kv_layers", ()))
    index_layers = tuple(int(i) for i in _dfly(text, cfg, "dfly_target_index_layer", kv_layers))
    assert kv_layers and kv_layers == index_layers, (
        f"DFly2 kv layers {kv_layers} must equal index layers {index_layers}"
    )
    num_layers = int(_dfly(text, cfg, "dfly_num_hidden_layers", 0))
    assert num_layers == len(kv_layers), (
        f"{num_layers} draft layers for {len(kv_layers)} target layers"
    )
    for layer in kv_layers:
        assert 0 <= layer < d.num_layers, (layer, d.num_layers)
        assert d.layers[layer].sparse, f"target layer {layer} is not a sparse layer"

    spec_config = getattr(core.model_config, "spec_config", None)
    max_draft_len = getattr(spec_config, "max_draft_len", None)
    block_width = int(max_draft_len + 1) if max_draft_len is not None else DEFAULT_BLOCK_WIDTH

    index_topk = int(_dfly(text, cfg, "dfly_target_index_topk", d.index_topk))
    local_window = int(_dfly(text, cfg, "dfly_context_local_window", None) or 0)

    return SimpleNamespace(
        num_layers=num_layers,
        target_kv_layers=kv_layers,
        index_topk=index_topk,
        #: Rows per request: the anchor in column 0, then the drafts.
        block_width=block_width,
        #: Per-row index buffer width: picks + local window + the block's own slots, aligned.
        index_width=round_up(index_topk + local_window + block_width, DRAFT_INDEX_ALIGN),
        hidden=d.hidden,
        heads_local=d.heads_sparse // d.attn_tp,
        nope=d.nope,
        rope_dim=d.rope_dim,
        v_dim=d.v_dim,
        kv_lora=d.kv_lora,
        q_lora=d.q_lora,
        qk_dim=d.qk_dim,
        lat_dim=d.lat_dim,
        theta=THETA_SPARSE,
        q_scaling=d.q_scaling,
        rms_eps=d.rms_eps,
        attn_tp=d.attn_tp,
        attn_tp_rank=d.attn_tp_rank,
        mlp_inter=DRAFT_MLP_MULTIPLE * d.hidden,
        correction_inter=DRAFT_CORRECTION_MULTIPLE * d.hidden,
        fusion_in=2 * d.hidden,
        mlp_tp=d.mlp_tp,
        mlp_tp_rank=d.mlp_tp_rank,
        dtype=d.dtype,
    )
