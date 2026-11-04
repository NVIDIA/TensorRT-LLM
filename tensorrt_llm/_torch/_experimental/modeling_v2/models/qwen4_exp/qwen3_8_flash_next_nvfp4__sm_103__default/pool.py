# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The KV cache manager's pools as one rank of this target sees them, and their contract.

Every rank builds the stock ``Qwen4ExpHybridCacheManager`` (``modules/qwen4_exp/cache_manager.py``,
a ``MambaHybridCacheManagerV2``): QSA is this architecture's default sparse attention, so
``get_kv_cache_manager_cls`` resolves the manager through ``get_sparse_attn_kv_cache_manager``,
which requires KV cache manager V2. The manager sizes each pool from the checkpoint's text config
and the rank's ``Mapping``; `pool_geometry` derives the same per-rank shapes the same way:

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

``H_kv`` is ``num_key_value_heads``; ``H_k``/``D_k`` and ``H_v``/``D_v`` are the GDN key and value
heads (``linear_num_key_heads``, ``linear_key_head_dim``, ...); ``kernel`` is
``linear_conv_kernel_dim``, ``ple_kernel`` ``ple_conv_kernel_size`` and ``n`` ``ngram_size``. Head
counts are the rank's share.

(a) ``KVCacheManagerV2.__init__`` splits ``num_key_value_heads`` over ``tp_size``, or over 1 under
    attention DP, rounding up: with more TP ranks than KV heads each head is repeated, and rank
    ``r`` holds global KV head ``r * H / tp`` (the order ``duplicate_kv_weight`` repeats them in).
(b) ``Qwen4ExpHybridCacheManager._extra_buffers_per_layer`` sizes them from ``QSASparseParams``
    (``index_kv_heads``, ``index_head_dim``), which no mapping divides: the indexer is replicated.
    Under an FP8 or NVFP4 pool both are packed into one ``QSA_SIDE_PLANES`` buffer; the views
    keep these shapes either way.
(c) ``MambaHybridCacheManagerV2.__init__`` divides the GDN value heads, key heads (``n_groups``)
    and ``conv_dim`` by ``mamba_effective_tp_size``: ``tp_size``, or 1 under attention DP. The
    conv channels of a rank are ``[q | k | v]`` of its own heads (``conv_state_layout`` "q_k_v",
    the manager's ``conv_section_dims``), and rank ``r`` holds key heads ``[r * H_k / tp, ...)``
    and value heads ``[r * H_v / tp, ...)``: the contiguous head ranges a disaggregated transfer
    between layouts moves (``get_disagg_role_mapper_kinds``: conv ``SECTIONED``, recurrent state
    by head).
(d) ``extract_qwen4_exp_ple_cache_params``: the PLE layer reads the whole hyper-connection
    stream on every rank.

At tp4 and tep4 a rank therefore holds one of the two KV heads, four key and twelve value GDN
heads; at dep4 it holds all of them, and the recurrent pools carry one more slot (the
attention-DP dummy request's). With MTP, attention TP puts the draft layer in a separate draft
KV cache (``should_use_separate_draft_kv_cache``); under attention DP the draft layer is a layer
of the trunk's manager. Its pools have the trunk's per-rank shapes in either.

The ``check_*`` functions read only ``.shape``, ``.dtype`` and ``.stride()``, so they run on meta
tensors as well as on a manager's views. Each one is a wrong pool its consuming kernel would
accept and mis-address.
"""

from __future__ import annotations

from typing import NamedTuple

import torch

from tensorrt_llm._torch.pyexecutor.config_utils import mamba_effective_tp_size
from tensorrt_llm.quantization.mode import QuantAlgo

#: Coordinates per token in the QSA position plane (the mRoPE layout's three rows).
QSA_POSITION_AXES = 3

#: Values per e4m3 block scale of an NVFP4 K/V pool.
NVFP4_BLOCK = 16

#: The recurrent-state pool dtypes the target serves (`kv_cache_config.mamba_ssm_cache_dtype`): the
#: checkpoint's fp32, or bf16.
SSM_STATE_DTYPES = (torch.float32, torch.bfloat16)

#: The GDN cached replay's history length the replay entry is certified at.
GDN_REPLAY_HISTORY = 16

#: The most query rows a verify step carries per generation request, `max_draft_len + 1`: the
#: replay kernel asserts `T <= 8` and `thop_attention`'s `predicted_tokens_per_seq` is certified
#: up to it.
MAX_VERIFY_WIDTH = 8

#: The same bound on the legacy verify path (`TRTLLM_USE_GDN_REPLAY=0`), whose entries are
#: certified at up to 4 rows per request.
MAX_LEGACY_VERIFY_WIDTH = 4


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

    @property
    def conv_sections(self) -> tuple[int, int, int]:
        """The q, k and v channel widths of this rank's conv state, in that order."""
        key = self.num_k_heads * self.head_k_dim
        return key, key, self.num_v_heads * self.head_v_dim

    @property
    def kv_block_scales(self) -> bool:
        """Whether the K/V pool is NVFP4, whose e4m3 block scales sit in a second pool."""
        return self.kv_dtype == torch.uint8


def attention_tp_size(mapping) -> int:
    """The ranks the KV heads are split over (`KVCacheManagerV2.__init__`)."""
    return 1 if mapping.enable_attention_dp else mapping.tp_size


def _share(total: int, ranks: int, rank: int) -> tuple[int, int]:
    """(heads on `rank`, its first global head) when `total` heads are split over `ranks`: an even
    split, or each head repeated on `ranks // total` neighbouring ranks when there are more
    ranks."""
    return -(-total // ranks), rank * total // ranks


#: The paged K/V pool's storage per `kv_cache_quant_algo`: (dtype or None for the activation
#: dtype, values per stored element).
_KV_STORAGE = {
    None: (None, 1),
    QuantAlgo.FP8: (torch.float8_e4m3fn, 1),
    QuantAlgo.NVFP4: (torch.uint8, 2),
}


def pool_geometry(
    cfg, mapping, *, tokens_per_block: int, kv_cache_quant_algo, dtype: torch.dtype
) -> PoolGeometry:
    """The pools rank `mapping.rank` binds.

    Args:
        cfg: the engine-normalized text config (`Qwen4ExpTextConfig`) the manager is sized from.
        mapping: this rank's `Mapping`.
        tokens_per_block: the manager's page size (`kv_cache_config.tokens_per_block`).
        kv_cache_quant_algo: the KV-cache algorithm the engine resolved: None (the checkpoint's
            bf16), `QuantAlgo.FP8` or `QuantAlgo.NVFP4`.
        dtype: the activation dtype.
    """
    if kv_cache_quant_algo not in _KV_STORAGE:
        raise ValueError(
            f"kv_cache_quant_algo {kv_cache_quant_algo!r}; the K/V pool stores the checkpoint's "
            "bf16, FP8 or NVFP4"
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
        kv_dtype=kv_dtype or dtype,
        kv_width=cfg.head_dim // per_element,
    )


# ------------------------------------------------------------------ contract ---


def require(cm, names, what: str) -> None:
    """`cm` exposes every attribute in `names`, which this target reads to `what`."""
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
    """QSA index plane and paged K/V for one attention layer."""
    index_shape = (g.tokens_per_block, g.index_kv_heads, g.index_head_dim)
    _check_pool(index_cache, f"layer {layer}: QSA index plane", 1, index_shape, g.dtype)
    # The sparse path needs adjacent HND K/V role planes: [pages, 2, H, block, D].
    kv_shape = (g.heads_kv, g.tokens_per_block, g.kv_width)
    _check_pool(pool, f"layer {layer}: paged K/V", 2, kv_shape, g.kv_dtype)
    assert pool.shape[1] == 2, f"layer {layer}: paged K/V has {pool.shape[1]} role planes, not 2"


def check_kv_scale_planes(g: PoolGeometry, layer: int, pool, scales) -> None:
    """An NVFP4 layer's e4m3 block scales: one uint8 per `NVFP4_BLOCK` values, laid out like the
    data pages (`[slots, 2, H, block, D / 16]`), and holding as many pages per slot as the data
    pool does.

    The attention op addresses a layer's data pages and its scale pages with one block-offset
    table, so a scale pool whose pages-per-slot differs from the data pool's reads and writes other
    slots' scales without any error. Both counts are read off the views' slot strides."""
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


def check_recurrent_pools(g: PoolGeometry, conv, ssm) -> None:
    _check_pool(conv, "conv state pool", 1, (g.conv_dim, g.conv_kernel - 1), g.dtype)
    # Key dim innermost.
    _check_pool(ssm, "recurrent state pool", 1, (g.num_v_heads, g.head_v_dim, g.head_k_dim))
    assert ssm.dtype in SSM_STATE_DTYPES, (
        f"the recurrent state pool is {ssm.dtype}; this target serves "
        f"{' or '.join(str(d) for d in SSM_STATE_DTYPES)} (kv_cache_config.mamba_ssm_cache_dtype)"
    )


def check_conv_layout(g: PoolGeometry, section_dims) -> None:
    """The manager splits a rank's conv channels as `[q | k | v]` of the rank's own heads, the
    order this target's input projection writes them in. Another order (`x_b_c`) or another split
    has the same total width, so the conv kernel would read every section from the wrong place."""
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


def check_step_buffers(g: PoolGeometry, int_conv, conv, int_idx) -> None:
    """MTP verify buffers both recurrent modes use: the per-step conv windows
    `[slots, steps, ...]` at the pool's dtype (the promotion copies between them without
    converting) and the rows `update_mamba_states` reads its sources from."""
    assert int_conv is not None, (
        "the runtime allocated no per-step conv buffer; a partially accepted draft "
        "cannot be rolled back"
    )
    _check_pool(int_conv, "per-step conv buffer", 2, (g.conv_dim, g.conv_kernel - 1), conv.dtype)
    assert int_idx is not None and int_idx.dtype == torch.int32, (
        "the cache manager exposes no int32 intermediate_state_indices"
    )


def check_legacy_state(g: PoolGeometry, int_ssm, ssm) -> None:
    """Legacy verify: per-step recurrent buffer `[slots, steps, ...]` at the pool's dtype."""
    assert int_ssm is not None, (
        "the cache manager runs the legacy verify path but allocated no per-step recurrent "
        "buffer; a partially accepted draft cannot be rolled back"
    )
    ssm_shape = (g.num_v_heads, g.head_v_dim, g.head_k_dim)
    _check_pool(int_ssm, "per-step recurrent buffer", 2, ssm_shape, ssm.dtype)
    assert int_ssm.shape[1] <= MAX_LEGACY_VERIFY_WIDTH, (
        f"the legacy verify path runs {int_ssm.shape[1]} rows per generation request; its entries "
        f"are certified at up to {MAX_LEGACY_VERIFY_WIDTH} (max_draft_len "
        f"{MAX_LEGACY_VERIFY_WIDTH - 1}). Leave TRTLLM_USE_GDN_REPLAY unset (cached replay) for "
        "a longer draft"
    )


def check_replay_state(history, width, prev_accepted, cache_buf_idx, ssm) -> None:
    """Cached replay, manager-wide: the geometry the replay entry is certified at, and the
    per-slot bookkeeping, which must cover every pool slot."""
    assert ssm.dtype in SSM_STATE_DTYPES, (
        f"the recurrent pool is {ssm.dtype}; the replay entry is certified at an fp32 or a "
        "bf16 checkpoint"
    )
    assert history == GDN_REPLAY_HISTORY, (
        f"replay_history_size {history}; the replay entry is certified at {GDN_REPLAY_HISTORY}"
    )
    assert 2 <= width <= min(MAX_VERIFY_WIDTH, history), (
        f"replay_step_width {width}; the replay entry is certified at 2..{MAX_VERIFY_WIDTH} "
        f"verify rows and never above the history ({history})"
    )
    for name, t in (("prev_num_accepted_tokens", prev_accepted), ("cache_buf_idx", cache_buf_idx)):
        assert t.dim() == 1 and t.shape[0] >= ssm.shape[0] and t.dtype == torch.int32, (
            f"{name} is {tuple(t.shape)} {t.dtype}; the kernel reads one int32 per pool slot "
            f"({ssm.shape[0]})"
        )


def check_replay_history(
    g: PoolGeometry, layer: int, history, pool_slots, old_x, old_B, old_dt
) -> None:
    """One GDN layer's replay history: `[slots, 2 halves, ...]` over this rank's heads, addressed
    by `history` and the head geometry, never by the buffer's own shape."""
    for name, t in (("old_x", old_x), ("old_B", old_B), ("old_dt", old_dt)):
        assert t is not None, f"layer {layer}: the manager's layer cache carries no {name}"
    assert old_x.shape[0] >= pool_slots, (
        f"layer {layer}: the replay history has {old_x.shape[0]} slots for a {pool_slots}-slot pool"
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
    """The rows per generation request one step carries: `max_draft_len + 1` (`runtime_draft_len
    + 1` at run time), a `thop_attention` `predicted_tokens_per_seq` the entry is certified at."""
    assert 1 <= rows <= MAX_VERIFY_WIDTH, (
        f"predicted_tokens_per_seq {rows} is outside the 1..{MAX_VERIFY_WIDTH} range "
        "attention/thop_attention is certified at"
    )


def check_replay_width(replay, rows: int) -> None:
    """Under the cached replay a single-row generation step cannot take the ordinary
    recurrence: the pool row is a checkpoint whose pending history only the replay kernel
    applies. The manager's step width (`replay.width`) is the one row count per sequence a step
    may carry."""
    assert replay is None or rows == replay.width, (
        f"the batch carries {rows} query rows per generation sequence and the GDN cached "
        f"replay runs at step width {replay.width}; a runtime_draft_len of {rows - 1} (a "
        "draft-length schedule, or skipped drafting) cannot run under it. Export "
        "TRTLLM_USE_GDN_REPLAY=0 for variable draft lengths"
    )


def check_shared_selection(shared, slots, layer: int, block_topk: int) -> None:
    """IndexShare: the metadata's shared selection buffers have a row for `layer` and are
    `block_topk` wide."""
    assert shared is not None, "IndexShare is on but the QSA metadata allocated no shared buffers"
    assert layer in slots and slots[layer] < shared.shape[0], (
        f"the KV cache manager gives layer {layer} no row of the {shared.shape[0]}-row shared "
        f"selection buffers (qsa_shared_topk_slots={slots})"
    )
    assert shared.shape[2] == block_topk, (
        f"the shared selection rows are {shared.shape[2]} wide and this target selects "
        f"{block_topk} blocks per row"
    )


# ------------------------------------------------------------- the views ---


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


def kv_planes(cm, g: PoolGeometry, layer: int, index_cache) -> KvPlanes:
    """One layer's paged K/V views from `cm`, checked against the contract: the data planes, and
    under NVFP4 the block-scale planes that go with them."""
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

    `attn_layers` and `gdn_layers` are the full-attention and Gated-DeltaNet decoder layers, and
    `ple_layer` the zero-based PLE layer. Reads no attention metadata: what depends on the step
    (pool ids, windows, the QSA selection width) is the binder's to check."""
    require(
        cm,
        ("get_index_k_buffer", "get_qsa_position_buffer", "get_buffers", "layer_offsets"),
        "reads the QSA index plane, the shared position plane and the paged K/V",
    )
    # `ple_layer_cache` exists only on the V2 hybrid manager; under V1 the PLE state silently
    # does not exist.
    require(
        cm,
        ("get_conv_states", "get_ssm_states", "conv_section_dims", "ple_layer_cache"),
        "reads the recurrent, conv and PLE pools",
    )
    # thop_attention addresses the pool by the layer's pool offset, not its ordinal among
    # attention layers.
    pool_idx = {i: cm.layer_offsets[i] for i in attn_layers}
    position_cache = cm.get_qsa_position_buffer()
    check_position_plane(g, position_cache)
    index_cache, kv = {}, {}
    for i in attn_layers:
        index_cache[i] = cm.get_index_k_buffer(i)
        kv[i] = kv_planes(cm, g, i, index_cache[i])

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
    return RankPools(pool_idx, index_cache, position_cache, kv, conv, ssm, (ple_conv, ple_ngram))


__all__ = [
    "GDN_REPLAY_HISTORY",
    "MAX_LEGACY_VERIFY_WIDTH",
    "MAX_VERIFY_WIDTH",
    "NVFP4_BLOCK",
    "QSA_POSITION_AXES",
    "SSM_STATE_DTYPES",
    "KvPlanes",
    "PoolGeometry",
    "RankPools",
    "attention_tp_size",
    "check_attention_pools",
    "check_conv_layout",
    "check_kv_scale_planes",
    "check_legacy_state",
    "check_ple_pools",
    "check_position_plane",
    "check_recurrent_pools",
    "check_replay_history",
    "check_replay_state",
    "check_replay_width",
    "check_shared_selection",
    "check_step_buffers",
    "check_verify_width",
    "kv_planes",
    "pool_geometry",
    "require",
    "take_pools",
]
