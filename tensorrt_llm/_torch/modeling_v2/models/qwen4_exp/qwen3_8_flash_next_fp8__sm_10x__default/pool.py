# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The KV cache manager's pools as one rank of this target sees them, and their contract.

Every rank builds the stock ``Qwen4ExpHybridCacheManager``, a KV cache manager V2 (which QSA
requires). `pool_geometry` derives the per-rank shapes it allocates:

===================  ============================================  =============================
pool                 shape after the slot dim                      split over ranks by
===================  ============================================  =============================
paged K/V (HND)      ``[2, H_kv, tokens_per_block, kv_width]``     attention TP (a)
NVFP4 block scales   ``[2, H_kv, tokens_per_block, head_dim/16]``  attention TP (a)
QSA index plane      ``[tokens_per_block, 1, index_head_dim]``     replicated (b)
QSA position plane   ``[tokens_per_block, 3]`` int32               replicated (b)
GDN conv state       ``[2 * H_k * D_k + H_v * D_v, kernel - 1]``   recurrent-state TP (c)
GDN recurrent state  ``[H_v, D_v, D_k]``                           recurrent-state TP (c)
PLE conv state       ``[hc_count * hidden, (ple_kernel-1) * n]``   replicated (d)
PLE n-gram history   ``[n - 1]`` int64                             replicated (d)
===================  ============================================  =============================

``H_kv`` is ``num_key_value_heads``, ``H_k``/``D_k`` and ``H_v``/``D_v`` the GDN key and value
heads, ``kernel`` ``linear_conv_kernel_dim``, ``ple_kernel`` ``ple_conv_kernel_size`` and ``n``
``ngram_size``; head counts are the rank's share.

(a) Over ``tp_size`` (1 under attention DP), rounding up: with more ranks than KV heads, rank
    ``r`` holds global KV head ``r * H / tp``, the order ``duplicate_kv_weight`` repeats them in.
(b) No mapping divides the indexer. When an index or position page could equal a K/V or
    block-scale page, which V2 would coalesce into that pool, both are packed into one per-slot
    ``QSA_SIDE_PLANES`` page (`PoolGeometry.qsa_side_planes`), as strided views of these shapes.
(c) Over ``mamba_effective_tp_size`` (1 under attention DP). Conv channels are ``[q | k | v]`` of
    the rank's own heads; rank ``r`` holds key heads from ``r * H_k / tp`` and value heads from
    ``r * H_v / tp``, the contiguous ranges a disaggregated transfer between layouts moves. The
    recurrent state is served in bf16 only (`SSM_STATE_DTYPES`), not the checkpoint's fp32.
(d) The PLE layer reads the whole hyper-connection stream on every rank.

With one-model MTP the draft layer (global layer ``num_hidden_layers``) is one more QSA attention
layer at the trunk's per-rank shapes (`draft_kv_cache_is_separate`).

The ``check_*`` functions read only ``.shape``, ``.dtype`` and ``.stride()``, so they also run on
meta tensors. Each rejects a pool its consuming kernel would accept and mis-address.
"""

from __future__ import annotations

from typing import NamedTuple

import torch

from tensorrt_llm._torch.pyexecutor.config_utils import mamba_effective_tp_size
from tensorrt_llm._torch.speculative.interface import should_use_separate_draft_kv_cache
from tensorrt_llm.quantization.mode import QuantAlgo

#: Coordinates per token in the QSA position plane (the mRoPE layout's three rows).
QSA_POSITION_AXES = 3

#: Values per e4m3 block scale of an NVFP4 K/V pool.
NVFP4_BLOCK = 16

#: Recurrent-state pool dtypes served (`kv_cache_config.mamba_ssm_cache_dtype`).
SSM_STATE_DTYPES = (torch.bfloat16,)

#: The GDN cached replay's history length the replay entry is certified at.
GDN_REPLAY_HISTORY = 16

#: Most query rows per generation request in a verify step (`max_draft_len + 1`): the replay
#: kernel asserts `T <= 8` and `thop_attention`'s `predicted_tokens_per_seq` is certified to it.
MAX_VERIFY_WIDTH = 8

#: CUDA-graph padding dummy slots: one per draft length, and routing admits one fixed length.
CUDA_GRAPH_PADDING_DUMMY_SLOTS = 1

#: Bytes per element of the QSA index (bf16) and position (int32) planes.
_INDEX_ELEMENT_BYTES = 2
_POSITION_ELEMENT_BYTES = 4


class PoolGeometry(NamedTuple):
    """The pool shapes one rank binds. Head counts and channel widths are this rank's share."""

    tokens_per_block: int
    heads_kv: int  # KV heads in this rank's K/V pages
    kv_head_start: int  # the global KV head the rank's first page head holds
    head_dim: int
    index_kv_heads: int
    index_head_dim: int
    conv_dim: int  # this rank's GDN conv channels, [q | k | v] of its own heads
    conv_kernel: int
    num_k_heads: int  # this rank's GDN key heads
    k_head_start: int  # the global key head the rank's first one is
    num_v_heads: int  # this rank's GDN value heads
    v_head_start: int  # the global value head the rank's first one is
    head_v_dim: int
    head_k_dim: int
    stream: int  # the hyper-connection stream width, the PLE conv channels
    ple_state_len: int
    ngram_context_len: int
    dtype: torch.dtype  # the activation dtype: GDN conv, QSA index and PLE conv pools
    kv_dtype: torch.dtype  # the paged K/V pool's dtype: bf16, e4m3 (FP8) or uint8 (NVFP4)
    kv_width: int  # its innermost extent: head_dim, or head_dim // 2 for NVFP4's e2m1 pairs
    qsa_side_planes: bool  # the index and position planes share one packed page per slot

    @property
    def conv_sections(self) -> tuple[int, int, int]:
        """The q, k and v channel widths of this rank's conv state, in that order."""
        key = self.num_k_heads * self.head_k_dim
        return key, key, self.num_v_heads * self.head_v_dim

    @property
    def kv_block_scales(self) -> bool:
        """Whether the K/V pool is NVFP4, whose e4m3 block scales sit in a second pool."""
        return self.kv_dtype == torch.uint8

    @property
    def index_page_bytes(self) -> int:
        per_token = self.index_kv_heads * self.index_head_dim * _INDEX_ELEMENT_BYTES
        return self.tokens_per_block * per_token

    @property
    def position_page_bytes(self) -> int:
        return self.tokens_per_block * QSA_POSITION_AXES * _POSITION_ELEMENT_BYTES


def attention_tp_size(mapping) -> int:
    """The ranks the KV heads are split over (`KVCacheManagerV2.__init__`)."""
    return 1 if mapping.enable_attention_dp else mapping.tp_size


def reserved_dummy_slots(mapping) -> int:
    """State slots beyond one per request: the graph padding dummy and the attention-DP dummy."""
    return CUDA_GRAPH_PADDING_DUMMY_SLOTS + int(mapping.enable_attention_dp)


def draft_kv_cache_is_separate(mapping, spec_config=None) -> bool:
    """Whether the one-model MTP draft layer has a KV cache manager of its own.

    Only under attention TP, unless `spec_config` disallows it (disaggregated serving does)."""
    if mapping.enable_attention_dp:
        return False
    return spec_config is None or should_use_separate_draft_kv_cache(spec_config)


def _share(total: int, ranks: int, rank: int) -> tuple[int, int]:
    """(heads on `rank`, its first global head): an even split, or each head on several ranks."""
    if total % ranks and ranks % total:
        raise ValueError(f"{total} heads do not split over {ranks} ranks")
    return -(-total // ranks), rank * total // ranks


#: K/V pool storage per `kv_cache_quant_algo`: (dtype or None for activation, values per element).
_KV_STORAGE = {
    None: (None, 1),
    QuantAlgo.FP8: (torch.float8_e4m3fn, 1),
    QuantAlgo.NVFP4: (torch.uint8, 2),
}


def _side_planes_packed(
    num_kv_heads: int,
    kv_page_bytes_per_head: int,
    scale_page_bytes_per_head: int,
    index_page: int,
    position_page: int,
) -> bool:
    """Whether `_extra_buffers_per_layer` packs the QSA index and position pages into one page."""
    pages = [p for p in (kv_page_bytes_per_head, scale_page_bytes_per_head) if p]
    main = {page * heads for page in pages for heads in range(1, num_kv_heads + 1)}
    return index_page in main or position_page in main


def pool_geometry(
    cfg, mapping, *, tokens_per_block: int, kv_cache_quant_algo, dtype: torch.dtype
) -> PoolGeometry:
    """The pools rank `mapping.rank` binds, from the engine-normalized `Qwen4ExpTextConfig`."""
    if kv_cache_quant_algo not in _KV_STORAGE:
        raise ValueError(
            f"kv_cache_quant_algo {kv_cache_quant_algo!r}; the K/V pool stores the checkpoint's "
            "bf16, FP8 or NVFP4"
        )
    if (mapping.pp_size, mapping.cp_size) != (1, 1):
        raise ValueError(
            f"pp_size={mapping.pp_size}, cp_size={mapping.cp_size}: the members split heads over "
            "the TP group alone, and the manager shards pipeline and helix layouts otherwise"
        )
    attn_tp = attention_tp_size(mapping)
    heads_kv, kv_head_start = _share(
        cfg.num_key_value_heads, attn_tp, mapping.tp_rank if attn_tp > 1 else 0
    )
    state_tp = mamba_effective_tp_size(mapping)
    state_rank = mapping.tp_rank if state_tp > 1 else 0
    num_k_heads, k_head_start = _share(cfg.linear_num_key_heads, state_tp, state_rank)
    num_v_heads, v_head_start = _share(cfg.linear_num_value_heads, state_tp, state_rank)
    kv_dtype, per_element = _KV_STORAGE[kv_cache_quant_algo]
    kv_dtype = kv_dtype or dtype
    kv_width = cfg.head_dim // per_element
    block_scales = kv_dtype == torch.uint8
    index_page = (
        tokens_per_block * cfg.indexer_kv_heads * cfg.indexer_head_dim * _INDEX_ELEMENT_BYTES
    )
    position_page = tokens_per_block * QSA_POSITION_AXES * _POSITION_ELEMENT_BYTES
    side_planes = _side_planes_packed(
        cfg.num_key_value_heads,
        tokens_per_block * kv_width * kv_dtype.itemsize,
        tokens_per_block * (cfg.head_dim // NVFP4_BLOCK) if block_scales else 0,
        index_page,
        position_page,
    )
    return PoolGeometry(
        tokens_per_block=tokens_per_block,
        heads_kv=heads_kv,
        kv_head_start=kv_head_start,
        head_dim=cfg.head_dim,
        index_kv_heads=cfg.indexer_kv_heads,
        index_head_dim=cfg.indexer_head_dim,
        conv_dim=2 * num_k_heads * cfg.linear_key_head_dim
        + num_v_heads * cfg.linear_value_head_dim,
        conv_kernel=cfg.linear_conv_kernel_dim,
        num_k_heads=num_k_heads,
        k_head_start=k_head_start,
        num_v_heads=num_v_heads,
        v_head_start=v_head_start,
        head_v_dim=cfg.linear_value_head_dim,
        head_k_dim=cfg.linear_key_head_dim,
        stream=cfg.hc_count * cfg.hidden_size,
        ple_state_len=(cfg.ple_conv_kernel_size - 1) * cfg.ngram_size,
        ngram_context_len=cfg.ngram_size - 1,
        dtype=dtype,
        kv_dtype=kv_dtype,
        kv_width=kv_width,
        qsa_side_planes=side_planes,
    )


def require(cm, names, what: str) -> None:
    for name in names:
        assert hasattr(cm, name), (
            f"the KV cache manager {type(cm).__name__} does not expose {name}; this target {what}"
        )


def _check_pool(t, what, lead, tail, dtype=None) -> None:
    """`t` has `lead` leading dims of any size, then exactly `tail`, at `dtype`."""
    assert t is not None, f"{what}: the runtime did not allocate it"
    expected = ["*"] * lead + [str(n) for n in tail]
    assert t.dim() == lead + len(tail) and tuple(t.shape[lead:]) == tuple(tail), (
        f"{what} is {tuple(t.shape)}, expected [{', '.join(expected)}]"
    )
    assert dtype is None or t.dtype == dtype, f"{what} is {t.dtype}, expected {dtype}"


def check_position_plane(g: PoolGeometry, position) -> None:
    shape = (g.tokens_per_block, QSA_POSITION_AXES)
    _check_pool(position, "QSA position plane", 1, shape, torch.int32)


def check_attention_pools(g: PoolGeometry, layer: int, index_cache, pool) -> None:
    index_shape = (g.tokens_per_block, g.index_kv_heads, g.index_head_dim)
    _check_pool(index_cache, f"layer {layer}: QSA index plane", 1, index_shape, g.dtype)
    # The sparse path needs adjacent HND K/V role planes: [pages, 2, H, block, D].
    kv_shape = (g.heads_kv, g.tokens_per_block, g.kv_width)
    _check_pool(pool, f"layer {layer}: paged K/V", 2, kv_shape, g.kv_dtype)
    assert pool.shape[1] == 2, f"layer {layer}: paged K/V has {pool.shape[1]} role planes, not 2"


def check_kv_scale_planes(g: PoolGeometry, layer: int, pool, scales) -> None:
    """NVFP4 block scales: `[slots, 2, H, block, D / 16]`, as many pages per slot as the data."""
    scale_shape = (g.heads_kv, g.tokens_per_block, g.head_dim // NVFP4_BLOCK)
    _check_pool(scales, f"layer {layer}: NVFP4 block scales", 2, scale_shape, torch.uint8)
    assert scales.shape[1] == 2 and scales.shape[0] == pool.shape[0], (
        f"layer {layer}: the block scales are {tuple(scales.shape)} against paged K/V "
        f"{tuple(pool.shape)}; they must hold the same slots, K then V"
    )
    pages = []
    for what, t in (("paged K/V", pool), ("block scales", scales)):
        page = t[0, 0].numel()
        assert t.stride(0) % page == 0, (
            f"layer {layer}: the {what} slot stride {t.stride(0)} is not a whole number of "
            f"{page}-element pages"
        )
        pages.append(t.stride(0) // page)
    assert pages[0] == pages[1], (
        f"layer {layer}: the paged K/V pool holds {pages[0]} pages per slot and the block-scale "
        f"pool {pages[1]}; the attention op would address the scales of other slots"
    )


def check_side_planes(g: PoolGeometry, index_caches, position) -> None:
    """Packed QSA planes (`g.qsa_side_planes`) all advance by one slot stride covering them."""
    if not g.qsa_side_planes:
        return
    stride = position.stride(0) * position.element_size()
    for layer, index in index_caches.items():
        index_stride = index.stride(0) * index.element_size()
        assert index_stride == stride, (
            f"layer {layer}: the QSA index plane advances {index_stride} bytes a slot and the "
            f"position plane {stride}; packed side planes share one per-slot page"
        )
    need = len(index_caches) * g.index_page_bytes + g.position_page_bytes
    assert stride >= need, (
        f"the packed QSA side page is {stride} bytes a slot, short of the {len(index_caches)} "
        f"index pages and the position page it holds ({need} bytes)"
    )


def check_recurrent_pools(g: PoolGeometry, conv, ssm) -> None:
    _check_pool(conv, "conv state pool", 1, (g.conv_dim, g.conv_kernel - 1), g.dtype)
    # Key dim innermost.
    _check_pool(ssm, "recurrent state pool", 1, (g.num_v_heads, g.head_v_dim, g.head_k_dim))
    assert ssm.dtype in SSM_STATE_DTYPES, (
        f"the recurrent state pool is {ssm.dtype}; this target serves "
        f"{' or '.join(str(d) for d in SSM_STATE_DTYPES)} (kv_cache_config.mamba_ssm_cache_dtype)"
    )


def check_conv_layout(g: PoolGeometry, section_dims) -> None:
    assert tuple(section_dims) == g.conv_sections, (
        f"the manager lays a rank's conv state out as sections {tuple(section_dims)}; this target "
        f"writes [q | k | v] = {g.conv_sections} (conv_state_layout 'q_k_v')"
    )


def check_ple_pools(g: PoolGeometry, ple_conv, ple_ngram, conv) -> None:
    _check_pool(ple_conv, "PLE conv state", 1, (g.stream, g.ple_state_len), g.dtype)
    # The id-history entry writes int64 into it; an int32 pool raises.
    _check_pool(ple_ngram, "PLE n-gram context pool", 1, (g.ngram_context_len,), torch.int64)
    # The forward addresses all three with one scheduler slot vector.
    assert ple_conv.shape[0] == conv.shape[0] == ple_ngram.shape[0], (
        "the PLE and Gated-DeltaNet pools must share one slot space"
    )


def check_state_slots(mapping, max_batch_size: int, slots: int) -> None:
    need = max_batch_size + reserved_dummy_slots(mapping)
    assert slots >= need, (
        f"the recurrent-state pools hold {slots} slots; max_batch_size {max_batch_size} plus "
        f"{reserved_dummy_slots(mapping)} reserved dummies (attention DP "
        f"{mapping.enable_attention_dp}) need {need}"
    )


def check_step_buffers(g: PoolGeometry, int_conv, conv, int_idx) -> None:
    """MTP verify buffers: per-step conv windows at the pool dtype (promotion does not convert)."""
    assert int_conv is not None, (
        "the runtime allocated no per-step conv buffer; a partially accepted draft "
        "cannot be rolled back"
    )
    _check_pool(int_conv, "per-step conv buffer", 2, (g.conv_dim, g.conv_kernel - 1), conv.dtype)
    assert int_idx is not None and int_idx.dtype == torch.int32, (
        "the cache manager exposes no int32 intermediate_state_indices"
    )


def check_replay_state(history, width, prev_accepted, cache_buf_idx, ssm, replay_slots) -> None:
    """Cached replay, manager-wide: the certified geometry and the per-slot bookkeeping, which
    must cover every replay slot. The manager may address the history by replay slots of its
    own, which then differ from the recurrent pool's."""
    assert ssm.dtype in SSM_STATE_DTYPES, (
        f"the recurrent pool is {ssm.dtype}; this target runs the replay over a bf16 checkpoint "
        "state (kv_cache_config.mamba_ssm_cache_dtype: bfloat16)"
    )
    assert history == GDN_REPLAY_HISTORY, (
        f"replay_history_size {history}; the replay entry is certified at {GDN_REPLAY_HISTORY}"
    )
    assert 2 <= width <= min(MAX_VERIFY_WIDTH, history), (
        f"replay_step_width {width}; the replay entry is certified at 2..{MAX_VERIFY_WIDTH} "
        f"verify rows and never above the history ({history})"
    )
    for name, t in (("prev_num_accepted_tokens", prev_accepted), ("cache_buf_idx", cache_buf_idx)):
        assert t.dim() == 1 and t.shape[0] >= replay_slots and t.dtype == torch.int32, (
            f"{name} is {tuple(t.shape)} {t.dtype}; the kernel reads one int32 per replay slot "
            f"({replay_slots})"
        )


def check_replay_history(
    g: PoolGeometry, layer: int, history, replay_slots, old_x, old_B, old_dt
) -> None:
    """One GDN layer's replay history: `[replay slots, 2 halves, ...]` over this rank's heads,
    addressed by `history` and the head geometry, never by the buffer's own shape."""
    for name, t in (("old_x", old_x), ("old_B", old_B), ("old_dt", old_dt)):
        assert t is not None, f"layer {layer}: the manager's layer cache carries no {name}"
    for name, t in (("old_x", old_x), ("old_B", old_B), ("old_dt", old_dt)):
        assert t.shape[0] >= replay_slots, (
            f"layer {layer}: the replay history {name} has {t.shape[0]} slots for "
            f"{replay_slots} replay slots"
        )
    bf16, fp32 = torch.bfloat16, torch.float32
    x_tail, b_tail = (history, g.num_v_heads, g.head_v_dim), (history, g.num_k_heads, g.head_k_dim)
    _check_pool(old_x, f"layer {layer}: old_x", 2, x_tail, bf16)
    _check_pool(old_B, f"layer {layer}: old_B", 2, b_tail, bf16)
    _check_pool(old_dt, f"layer {layer}: old_dt", 2, (g.num_v_heads, history), fp32)
    assert old_x.shape[1] == old_B.shape[1] == old_dt.shape[1] == 2, (
        f"layer {layer}: the replay history is not double-buffered"
    )


def check_verify_width(rows: int) -> None:
    assert 1 <= rows <= MAX_VERIFY_WIDTH, (
        f"predicted_tokens_per_seq {rows} is outside the 1..{MAX_VERIFY_WIDTH} range "
        "attention/thop_attention is certified at"
    )


def check_replay_width(replay, rows: int) -> None:
    """A cached-replay step carries exactly `replay.width` rows per sequence.

    The pool row is a checkpoint whose pending history only the replay kernel applies."""
    assert replay is None or rows == replay.width, (
        f"the batch carries {rows} query rows per generation sequence and the GDN cached "
        f"replay runs at step width {replay.width}; a runtime_draft_len of {rows - 1} (a "
        "draft-length schedule, or skipped drafting) cannot run under it. Export "
        "TRTLLM_USE_GDN_REPLAY=0 for variable draft lengths"
    )


def check_shared_selection(shared, slots, layer: int, block_topk: int) -> None:
    assert shared is not None, "IndexShare is on but the QSA metadata allocated no shared buffers"
    assert layer in slots and slots[layer] < shared.shape[0], (
        f"the KV cache manager gives layer {layer} no row of the {shared.shape[0]}-row shared "
        f"selection buffers (qsa_shared_topk_slots={slots})"
    )
    assert shared.shape[2] == block_topk, (
        f"the shared selection rows are {shared.shape[2]} wide and this target selects "
        f"{block_topk} blocks per row"
    )


class KvPlanes(NamedTuple):
    """One layer's paged K/V views, and under an NVFP4 pool its e4m3 block-scale views:
    `[slots, H_kv, tokens_per_block, kv_width]` and `[slots, H_kv, tokens_per_block, D / 16]`."""

    k: torch.Tensor
    v: torch.Tensor
    k_scale: torch.Tensor | None = None
    v_scale: torch.Tensor | None = None


class RankPools(NamedTuple):
    """The trunk's pool views on one rank, from one manager. A view is a `TensorWrapper` over a
    raw device address that keeps no memory alive, so it is valid only for that manager."""

    pool_idx: dict  # attention layer -> pool offset (what thop_attention addresses)
    index_cache: dict  # attention layer -> QSA index plane
    position_cache: torch.Tensor  # the shared QSA position plane
    kv: dict  # attention layer -> `KvPlanes`
    conv: dict  # GDN layer -> conv state pool
    ssm: dict  # GDN layer -> recurrent state pool
    ple_state: tuple  # (PLE conv state, PLE n-gram id history)


class DraftPools(NamedTuple):
    """The MTP draft layer's views in the manager its draft step is handed."""

    pool_idx: int  # the draft layer's pool offset in that manager
    index_cache: torch.Tensor  # its QSA index plane
    position_cache: torch.Tensor  # that manager's position plane
    kv: KvPlanes


def kv_planes(cm, g: PoolGeometry, layer: int, index_cache) -> KvPlanes:
    """One layer's checked paged K/V views from `cm`, plus the block-scale views under NVFP4."""
    pool = cm.get_buffers(layer, kv_layout="HND")
    check_attention_pools(g, layer, index_cache, pool)
    if not g.kv_block_scales:
        return KvPlanes(pool[:, 0], pool[:, 1])
    require(cm, ("get_block_scale_buffers",), "reads an NVFP4 pool's e4m3 block-scale planes")
    scales = cm.get_block_scale_buffers(layer, kv_layout="HND")
    check_kv_scale_planes(g, layer, pool, scales)
    return KvPlanes(pool[:, 0], pool[:, 1], scales[:, 0], scales[:, 1])


def take_pools(cm, g: PoolGeometry, attn_layers, gdn_layers, ple_layer: int) -> RankPools:
    """Every trunk pool view of `cm` this rank binds, each checked against `g`.

    Step-dependent state (pool ids, windows, the QSA selection width) is the binder's to check."""
    require(
        cm,
        ("get_index_k_buffer", "get_qsa_position_buffer", "get_buffers", "layer_offsets"),
        "reads the QSA index plane, the shared position plane and the paged K/V",
    )
    # Only the V2 hybrid manager has `ple_layer_cache`; under V1 the PLE state is silently absent.
    require(
        cm,
        ("get_conv_states", "get_ssm_states", "conv_section_dims", "ple_layer_cache"),
        "reads the recurrent, conv and PLE pools",
    )
    require(cm, ("max_batch_size",), "sizes the recurrent-state slots it addresses")
    # thop_attention addresses a layer by its pool offset, not its ordinal among attention layers.
    pool_idx = {i: cm.layer_offsets[i] for i in attn_layers}
    position_cache = cm.get_qsa_position_buffer()
    check_position_plane(g, position_cache)
    index_cache, kv = {}, {}
    for i in attn_layers:
        index_cache[i] = cm.get_index_k_buffer(i)
        kv[i] = kv_planes(cm, g, i, index_cache[i])
    check_side_planes(g, index_cache, position_cache)

    check_conv_layout(g, cm.conv_section_dims)
    conv = {i: cm.get_conv_states(i) for i in gdn_layers}
    ssm = {i: cm.get_ssm_states(i) for i in gdn_layers}
    for i in gdn_layers:
        check_recurrent_pools(g, conv[i], ssm[i])

    ple_cache = cm.ple_layer_cache(ple_layer)
    assert ple_cache is not None, (
        f"no PLE lifecycle state for decoder layer {ple_layer}: the runtime allocated neither "
        "the dilated conv state nor the n-gram id history"
    )
    ple_conv, ple_ngram = ple_cache
    check_ple_pools(g, ple_conv, ple_ngram, conv[gdn_layers[0]])
    check_state_slots(cm.mapping, cm.max_batch_size, ple_conv.shape[0])
    return RankPools(pool_idx, index_cache, position_cache, kv, conv, ssm, (ple_conv, ple_ngram))


def take_draft_pools(cm, g: PoolGeometry, layer: int, *, separate: bool) -> DraftPools:
    """The draft layer's checked views in `cm`, the manager its draft step is handed."""
    require(
        cm,
        ("get_index_k_buffer", "get_qsa_position_buffer", "get_buffers", "layer_offsets"),
        "reads the draft layer's QSA index plane, position plane and paged K/V",
    )
    layers = set(cm.layer_offsets)
    assert layer in layers, (
        f"the MTP module's attention is global layer {layer}, which the cache manager "
        f"{type(cm).__name__} does not map: the engine allocated no KV layer for the draft module"
    )
    if separate:
        assert layers == {layer}, (
            f"the separate draft KV cache maps layers {sorted(layers)}; it holds the draft layer "
            f"{layer} alone"
        )
    else:
        assert len(layers) > 1, (
            f"without a separate draft KV cache the draft layer {layer} is a layer of the trunk's "
            "manager, which maps the trunk's layers too"
        )
    position_cache = cm.get_qsa_position_buffer()
    check_position_plane(g, position_cache)
    index_cache = cm.get_index_k_buffer(layer)
    kv = kv_planes(cm, g, layer, index_cache)
    if separate:
        check_side_planes(g, {layer: index_cache}, position_cache)
    return DraftPools(cm.layer_offsets[layer], index_cache, position_cache, kv)


__all__ = [
    "CUDA_GRAPH_PADDING_DUMMY_SLOTS",
    "GDN_REPLAY_HISTORY",
    "MAX_VERIFY_WIDTH",
    "NVFP4_BLOCK",
    "QSA_POSITION_AXES",
    "SSM_STATE_DTYPES",
    "DraftPools",
    "KvPlanes",
    "PoolGeometry",
    "RankPools",
    "attention_tp_size",
    "check_attention_pools",
    "check_conv_layout",
    "check_kv_scale_planes",
    "check_ple_pools",
    "check_position_plane",
    "check_recurrent_pools",
    "check_replay_history",
    "check_replay_state",
    "check_replay_width",
    "check_shared_selection",
    "check_side_planes",
    "check_state_slots",
    "check_step_buffers",
    "check_verify_width",
    "draft_kv_cache_is_separate",
    "kv_planes",
    "pool_geometry",
    "require",
    "reserved_dummy_slots",
    "take_draft_pools",
    "take_pools",
]
