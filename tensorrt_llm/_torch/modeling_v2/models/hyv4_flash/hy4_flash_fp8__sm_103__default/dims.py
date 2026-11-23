# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Every per-rank width of HY4-Flash's text core, from the checkpoint and the engine's Mapping.

One derivation, read by the weight manifest and the forward alike. The checkpoint
publishes its geometry in ``text_config``; the per-layer differences (head count,
RoPE base, attention window, Engram presence, indexer presence) come from its
``layer_types`` and ``index_topk_pattern`` through the same helpers the router
uses, so a config whose layers move cannot disagree with the code that sharded
the weights.

The five layouts this target serves, and what each does to a rank's share:

* **tp4** -- attention TP 4 (8 sparse / 16 SWA heads), MoE TP 4 with every expert
  (``moe_ep_size`` 1): each rank holds all 512 experts over a quarter of their
  intermediate width.
* **tep4** -- attention TP 4, MoE EP 4: each rank holds a contiguous window of 128
  expert ids at full intermediate width.
* **dep4** -- attention DP with MoE EP 4: attention and the dense/shared MLPs are
  whole on every rank, the experts are split as in tep4.
* **tep8** -- attention TP 8 (4 sparse / 8 SWA heads), MoE EP 8 over two nodes.
* **tep2** -- attention TP 2 (16 sparse / 32 SWA heads), MoE EP 2 with
  ``moe_tp_size`` 1: each rank holds a contiguous window of 256 expert ids. The
  disaggregated deployment's context engine runs this width.

The MLA latent (``kv_lora 512 | rope 64``) is one head wide and replicated across
the attention ranks: every rank writes and reads the same latent rows, so the KV
cache is not split by heads. Only ``q_b_proj``, ``kv_b_proj``, ``linear_gate`` and
``o_proj`` are cut by head.

Two engine conventions are fixed here rather than in the forward:

* ``q_scaling`` is **1.0**. The native MLA path takes V's width from the latent
  rank and divides the softmax by ``sqrt(kv_lora + rope)`` times ``q_scaling``,
  then corrects the sparse branch by ``sqrt(qk_nope + qk_rope)/sqrt(kv_lora +
  rope)``; at 1.0 the effective scale is exactly ``1/sqrt(256)``, the
  checkpoint's own.
* ``tokens_per_block`` is the engine's KV page, read from the model config, not
  from the workload.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import NamedTuple

from tensorrt_llm._torch.pyexecutor.config_utils import (
    get_hyv4_flash_layer_masks,
    unwrap_hyv4_flash_text_config,
)

#: HY4-Flash's two RoPE bases, used when the config publishes no per-layer block.
THETA_SPARSE = 10_000_000.0
THETA_SWA = 10_000.0

#: The interleaved mRoPE table's section split (config ``mrope_section``).
MROPE_SECTION = (12, 10, 10)


class LayerGeom(NamedTuple):
    """What changes from one decoder layer to the next."""

    #: Query heads before the attention-TP cut: 32 on a sparse layer, 64 on a windowed one.
    heads: int
    #: Query heads this rank runs.
    heads_local: int
    #: Whether the layer runs sparse MLA over the indexer's top-k (no window).
    sparse: bool
    #: RoPE base: 1e7 on a sparse layer, 1e4 on a windowed one.
    theta: float
    #: Attention window in tokens, or None for a layer that attends the whole context.
    window: int | None


def _theta(text, layer_type: str, fallback: float) -> float:
    """This layer type's RoPE base, from its own ``rope_parameters`` block.

    The checkpoint nests one block per layer type and publishes no top-level
    ``rope_theta``; a single global base would collapse 1e7 and 1e4 into one.
    """
    blocks = getattr(text, "rope_parameters", None) or {}
    block = blocks.get(layer_type)
    if isinstance(block, dict) and block.get("rope_theta") is not None:
        return float(block["rope_theta"])
    return float(fallback)


def _dfly_layers(cfg, text, name: str) -> tuple[int, ...]:
    """One of the checkpoint's DFly2 layer lists, as zero-based decoder layers.

    The ``dfly_*`` keys sit on the composite config (``hy_v4_flash``, whose
    ``text_config`` is the text model) and on the flattened text config when a
    checkpoint publishes no wrapper; the composite form wins, the order the
    engine's own DFly2 resolver reads them in.
    """
    value = getattr(cfg, name, None)
    if value is None:
        value = getattr(text, name, None)
    return tuple(int(i) for i in (value or ()))


def _vision_v(cfg) -> SimpleNamespace | None:
    """The vision tower's geometry, from the composite config's ``vision_config``.

    ``None`` when the config publishes no tower (the text-only stand-ins some tests
    build), in which case the manifest carries no ``vis_*`` weight and the checkpoint's
    ``vit.*`` tensors stay unconsumed.

    Nothing here depends on the engine's ``Mapping``: every rank holds the whole tower
    and runs it on the whole batch's media. The three asserts that are not shape
    derivation pin the checkpoint to the entries the tower calls -- a bilinearly
    resampled position table, the erf GELU at both activations, and one patch per row.
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
    assert int(field("num_key_value_heads")) == heads, (
        "the tower's attention is MHA; this vision_config splits its value heads"
    )
    assert str(field("hidden_act")) == "gelu", (
        "the tower's GELU entry is the erf form, which is what vision_config's "
        f"hidden_act resolves to; it says {field('hidden_act')!r}"
    )
    assert str(field("interpolate_mode")) == "bilinear", (
        "the position table entry resamples bilinearly with the reference's +0.1 scale; "
        f"vision_config says {field('interpolate_mode')!r}"
    )
    assert int(field("temporal_patch_size")) == 1, (
        "a patch row is (channels, patch, patch): the reference folds no temporal axis "
        "into it, so this target's pixel rows assume temporal_patch_size 1"
    )
    # The position table's grid: `(max_image_size / patch)**2` patch rows plus the cls
    # row the reference skips. A frame's patch count is bounded by it, and every other
    # frame is packed into a call of at most that many rows.
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
        #: One flat pixel row per patch: the image processor's own `[patches, 3*16*16]`.
        pixel_width=channels * patch * patch,
        #: The tower's two LayerNorms' eps (the reference's `layer_norm_eps` maps onto
        #: `rms_norm_eps`) and the merger's RMSNorm eps.
        eps=float(field("rms_norm_eps")),
        merge_dim=hidden * merge * merge,
        img_max_token_num=int(getattr(v, "img_max_token_num", 0) or 0),
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
        "the windowed layers' no-PE width differs from the sparse layers'; this target "
        "runs one MLA geometry on both"
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

    # The MLPs split over the attention-TP group: the dense layer and the shared
    # expert are partial sums a TP allreduce completes, like the attention output.
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

    # DFly2's draft layers attend over the target sparse layer's own picks, so
    # the checkpoint names the layers whose selection must be published
    # (`cache_manager.DraftPicks`) for the drafter to read. The two config lists
    # are the same layers: a draft layer reusing one layer's KV and another's
    # picks would read a row's keys off the wrong index. The picks width is this
    # target's indexer's top-k, which the checkpoint's `dfly_target_index_topk`
    # must match -- the state holds the selection the indexer computes, not a
    # prefix of it.
    draft_source_layers = _dfly_layers(cfg, text, "dfly_target_kv_layers")
    draft_index_layers = _dfly_layers(cfg, text, "dfly_target_index_layer")
    if draft_source_layers and draft_index_layers:
        assert draft_source_layers == draft_index_layers, (
            f"the checkpoint's DFly2 draft layers reuse their target layer's MLA KV and index "
            f"picks, but dfly_target_kv_layers={list(draft_source_layers)} and "
            f"dfly_target_index_layer={list(draft_index_layers)} differ"
        )
    for i in draft_source_layers:
        assert i in sparse_set, (
            f"DFly2 drafts over sparse layer {i}'s picks, but the checkpoint's index_topk_pattern "
            f"gives that layer no indexer (sparse layers: {list(sparse_layers)})"
        )
    draft_topk = getattr(cfg, "dfly_target_index_topk", None)
    if draft_topk is None:
        draft_topk = getattr(text, "dfly_target_index_topk", None)
    if draft_source_layers and int(draft_topk or 0) != int(text.index_topk):
        raise ValueError(
            f"this target's picks state holds the indexer's selection, which is "
            f"index_topk={int(text.index_topk)} wide; the checkpoint asks the draft layers for "
            f"dfly_target_index_topk={draft_topk}"
        )

    return SimpleNamespace(
        num_layers=num_layers,
        hidden=int(text.hidden_size),
        vocab=int(text.vocab_size),
        dtype=model_config.torch_dtype,
        #: The vision tower, whole on every rank, or None for a config without one.
        vision=_vision_v(cfg),
        # attention
        attention_dp=attention_dp,
        attn_tp=attn_tp,
        attn_tp_rank=attn_tp_rank,
        world=m.world_size,
        rank=m.rank,
        layers=layers,
        sparse_layers=sparse_layers,
        #: The sparse layers the checkpoint's DFly2 draft layers read the picks
        #: of (`dfly_target_kv_layers`), in the draft layers' own order; empty
        #: on a checkpoint that ships no drafter.
        draft_source_layers=draft_source_layers,
        heads_sparse=heads_sparse,
        heads_swa=heads_swa,
        nope=nope,
        rope_dim=rope_dim,
        v_dim=v_dim,
        kv_lora=kv_lora,
        q_lora=q_lora,
        qk_dim=nope + rope_dim,
        lat_dim=kv_lora + rope_dim,
        window=window,
        # The engine's FMHA divides by sqrt(headSize) and multiplies by
        # q_scaling*sqrt(qk_nope + qk_rope)/sqrt(kv_lora + rope) on the sparse
        # branch, so 1.0 is the checkpoint's 1/sqrt(256) exactly.
        q_scaling=1.0,
        mrope_section=MROPE_SECTION,
        # compressed indexer (sparse layers only)
        index_topk=int(text.index_topk),
        index_kpool=int(getattr(text, "index_kpool", 4)),
        index_n_heads=index_n_heads,
        index_head_dim=index_head_dim,
        index_group_topk=int(getattr(text, "compressed_indexer_group_topk", 512)),
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
        norm_topk_prob=bool(getattr(text, "norm_topk_prob", True)),
        moe_tp=m.moe_tp_size,
        moe_tp_rank=m.moe_tp_rank,
        moe_inter=moe_inter // m.moe_tp_size,
        #: The routed experts' input width (the latent), whole on every rank: the
        #: MoE TP split is along the intermediate, never along K.
        latent=moe_inter,
        moe_ep=m.moe_ep_size,
        moe_ep_rank=m.moe_ep_rank,
        local_experts=local_experts,
        expert_offset=local_experts * m.moe_ep_rank,
        # hyper-connections
        hc_mult=int(text.hc_mult),
        hc_eps=float(text.hc_eps),
        hc_magnitude=float(text.hc_magnitude),
        # engram
        engram_layers=engram_layers,
        engram_ids=engram_ids,
        engram_head_vocab_sizes=head_vocab_sizes,
        #: Each layer's table holds one disjoint row block per head, in head order:
        #: its row count is the sum of that layer's own sixteen moduli (the
        #: checkpoint's `multi_head_embedding.embedding.weight` row count).
        # Keyed by the zero-based decoder layer, the index the manifest and the
        # forward use; `head_vocab_sizes` keeps the config's own one-based ids.
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
        #: The engine's per-step token budget, the MoE runner's autotuning range.
        max_num_tokens=int(model_config.max_num_tokens),
        # The KV page is the executor's, not the model config's: the PyTorch flow
        # does not carry `tokens_per_block` on `ModelConfig`, and the DSA pool
        # requires the engine's page size (32), which is also the value the
        # attention entry's cells are certified at. The other three are limits the
        # engine passes down to the model; they are read where they exist and
        # defaulted here, and the forward prefers the metadata's own values.
        tokens_per_block=int(getattr(model_config, "tokens_per_block", 0) or 32),
        max_seq_len=int(getattr(model_config, "max_seq_len", 0) or 0),
        max_batch=int(getattr(model_config, "max_batch_size", 0) or 0),
        max_context_len=int(
            getattr(model_config, "max_context_length", 0)
            or getattr(model_config, "max_seq_len", 0)
            or 0
        ),
    )


#: The DFly2 drafter's FFN width as a multiple of the hidden size. The five
#: draft layers' `mlp.gate_proj`/`up_proj` are `[9216, 3072]` and `down_proj`
#: `[3072, 9216]`; the checkpoint publishes no width for them (the reference
#: implementations stream it off the first tensor they read), so it is derived
#: here and `test_draft.py` holds it against the checkpoint's own headers.
DRAFT_MLP_MULTIPLE = 3

#: The hidden-correction MLP's width: `gate_proj`/`up_proj` are `[3072, 6144]`
#: over the concatenated hidden and embedding, `down_proj` `[3072, 3072]`.
DRAFT_CORRECTION_MULTIPLE = 1

#: The width the drafter's per-row picked-index buffer is aligned to. A row
#: attends `dfly_target_index_topk` context picks plus its own block's slots;
#: the reference implementation rounds that sum up to this multiple, which is
#: what the draft attention entry is then driven at.
DRAFT_INDEX_ALIGN = 128

#: The DFly2 block's width when the spec config does not say: five drafts and
#: the anchor.
DEFAULT_BLOCK_WIDTH = 6


def round_up(value: int, multiple: int) -> int:
    """`value` rounded up to the next multiple."""
    return -(-value // multiple) * multiple


def _dfly(text, cfg, key, default=None):
    """A DFly2 field: the checkpoint publishes them on the config's top level."""
    for block in (text, cfg):
        value = getattr(block, key, None)
        if value is not None:
            return value
    return default


def draft_dims(core) -> SimpleNamespace:
    """The DFly2 drafter's every width, from the checkpoint and the deployment.

    The drafter is five decoder layers that run on the target's own sparse-layer
    geometry: 32 query heads, the same MLA latent (``kv_lora 512 | rope 64``) and
    the same no-PE/value widths, so the core's per-layer numbers carry over. What
    is the drafter's own is the FFN width, the hidden-correction MLP, and the
    block it drafts -- the deployment's ``max_draft_len + 1`` rows per request.

    ``target_kv_layers`` is the pair of config fields that must agree; the plan
    reads it as the run's own contract, so a checkpoint that splits them would
    give a drafter reading one layer's latent pool with another's picks.
    """
    d = dims(core)
    cfg = core.model_config.pretrained_config
    text = unwrap_hyv4_flash_text_config(cfg)

    kv_layers = tuple(int(i) for i in _dfly(text, cfg, "dfly_target_kv_layers", ()))
    index_layers = tuple(int(i) for i in _dfly(text, cfg, "dfly_target_index_layer", kv_layers))
    assert kv_layers and kv_layers == index_layers, (
        "the DFly2 drafter reuses one target layer's latent pool and its picks: "
        f"kv layers {kv_layers} must equal index layers {index_layers}"
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
    local_window = _dfly(text, cfg, "dfly_context_local_window", None)
    local_window = int(local_window or 0)

    # Every draft layer is a sparse layer of the target, so its attention copies
    # the target's sparse geometry and its head split.
    heads = d.heads_sparse
    heads_local = heads // d.attn_tp
    mlp_inter = DRAFT_MLP_MULTIPLE * d.hidden
    correction_inter = DRAFT_CORRECTION_MULTIPLE * d.hidden

    return SimpleNamespace(
        # --- the target layers this drafter reads and writes ---
        num_layers=num_layers,
        target_kv_layers=kv_layers,
        target_index_layers=index_layers,
        index_topk=index_topk,
        context_local_window=local_window or None,
        # --- the block the deployment drafts ---
        #: Rows per request: the drafts, then the anchor in column 0.
        block_width=block_width,
        #: `dfly_target_index_topk` picks + this row's own block slots.
        picked_rows=index_topk + local_window + block_width,
        #: The same, aligned: the width the draft attention's index buffer is built at.
        index_width=round_up(index_topk + local_window + block_width, DRAFT_INDEX_ALIGN),
        mask_token_id=int(_dfly(text, cfg, "dfly_mask_token_id", 0)),
        mtp_fusion=str(_dfly(text, cfg, "dfly_mtp_fusion", "")),
        hidden_correction=bool(_dfly(text, cfg, "dfly_enable_hidden_correction", False)),
        # --- the drafter's own attention: the target's sparse-layer MLA ---
        hidden=d.hidden,
        heads=heads,
        heads_local=heads_local,
        nope=d.nope,
        rope_dim=d.rope_dim,
        v_dim=d.v_dim,
        kv_lora=d.kv_lora,
        q_lora=d.q_lora,
        qk_dim=d.qk_dim,
        lat_dim=d.lat_dim,
        window=None,
        theta=THETA_SPARSE,
        q_scaling=d.q_scaling,
        rms_eps=d.rms_eps,
        # --- the same head split as the core's attention TP group ---
        attn_tp=d.attn_tp,
        attn_tp_rank=d.attn_tp_rank,
        # --- the draft FFN and the correction MLP ---
        mlp_inter=mlp_inter,
        correction_inter=correction_inter,
        fusion_in=2 * d.hidden,
        mlp_tp=d.mlp_tp,
        mlp_tp_rank=d.mlp_tp_rank,
        dtype=d.dtype,
        hidden_dtype=d.dtype,
    )
