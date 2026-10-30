# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Beam six-block small model / sm_103 / tp1: token ids to logits, one request step per call.

A component, not a routed target. There is no ``routing.py``, no router-index entry and
no registered architecture: nothing in the engine builds this class, and it serves no
request on its own. A caller constructs it from a six-block configuration of the
checkpoint -- the checkpoint's own fields with ``num_hidden_layers = 6`` and the first six
entries of its two layer schedules -- loads its 2394 tensors and runs one step of one BF16
request through it, with the attention batch state prepared on a six-layer
``KVCacheManagerV2``: ``prefill_context`` builds a fresh one for a single context, and the
stateful generator in ``generation.py`` keeps one across requests for cached decoding and
prefix reuse. It does not call the built-in Beam modeling.

The model, all blocks with their own parameters and logical index::

    x = embed_tokens[input_ids]                         [1, T, 5120] bf16, no scale or norm
    x = block_0(x); x = block_1(x); x = block_2(x)      dense, sliding attention + RoPE
    x = block_3(x); x = block_4(x)                      sparse, sliding attention + RoPE
    x = block_5(x)                                      sparse, full attention, no RoPE
    h = rmsnorm(x, norm)                                fp32 statistics and affine, one rounding
    head = h @ lm_head^T                                [T, 151936] bf16, untied, no bias
    logits = tanh(head.float() / 30) * 30               [1, T, 151936] fp32

The blocks are the dense and sparse block components, built for logical layers 0-2 and
3-5; each block's residual coefficients, attention kind and router are its own layer's.

Steps. A call is one step of one request: a context step computes the T tokens of the
request's prompt that are not yet in its cache, after the ``past`` tokens that are (none
for a fresh context; a prefix reused from another request or computed by an earlier step
otherwise), and a generation step appends one token. The tokens sit at absolute positions
``past .. past + T - 1``; the returned rows are those tokens' logits.

Cache. One call appends every layer's keys and values to one ``KVCacheManagerV2`` that
holds exactly logical layers 0-5, each in its own local slot: ``layer_offsets`` maps
logical layer i to slot i, and slot i's rows in the pool tables belong to it alone. The
five sliding layers attend through a 512-token window and layer 5 to the whole history;
the manager states this as its per-layer window vector ``[512] * 5 + [None]`` (``None`` is
the full ``max_seq_len``), and the attention op of each layer is called with that layer's
window. Before the first layer runs, ``check_cache`` verifies the manager, its slots and
the host pool tables the attention op reads -- that they address each slot's own key and
value pages as the manager allocated them -- and ``check_request`` verifies the request:
that the manager holds an active cache for it whose history ends exactly where the step
begins (a step never rewrites a cached token) and with the capacity the step needs, that
every page the step reads or writes is a valid page of its pool, and that the block-offset
table the op reads holds exactly the manager's addresses of those pages. Earlier tokens' keys
and values are read from the pool on a cached context step (paged-context FMHA) and on
every generation step.

Host synchronization per call: one for the model's token-id range, position and
block-offset check, one for each block's position check, and each sparse block's MoE
dispatch (two, plus two per selected expert). A call is therefore not capturable in a CUDA
graph.
"""

from __future__ import annotations

import contextlib
import weakref
from collections import OrderedDict
from typing import Iterator, Mapping

import numpy as np
import torch
from torch import nn

from tensorrt_llm._torch._experimental.modeling_v2.catalog.activation.beam_logit_softcap import (
    beam_logit_softcap,
)
from tensorrt_llm._torch._experimental.modeling_v2.catalog.gemm.cublas_mm import cublas_mm
from tensorrt_llm._torch._experimental.modeling_v2.catalog.norm.flashinfer_rmsnorm import (
    flashinfer_rmsnorm,
)
from tensorrt_llm._torch.attention.backends.trtllm import TrtllmAttentionMetadata
from tensorrt_llm._torch.metadata import KVCacheParams
from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import KVCacheManagerV2, Role
from tensorrt_llm._torch.pyexecutor.resource_manager import CacheTypeCpp
from tensorrt_llm.bindings import DataType
from tensorrt_llm.llmapi.llm_args import KvCacheConfig
from tensorrt_llm.mapping import Mapping as ParallelMapping
from tensorrt_llm.runtime.kv_cache_manager_v2 import BAD_PAGE_INDEX, LayerId, PageIndexMode

from ..dense_block_bf16__sm_103__tp1 import block as dense_block
from ..full_model_nvfp4__sm_103__tp4 import _attention
from ..moe_bf16__sm_103__tp1.moe import check_weights
from ..moe_bf16__sm_103__tp1.moe import weight_shapes as moe_weight_shapes
from ..sparse_block_bf16__sm_103__tp1 import block as sparse_block

NUM_LAYERS = 6
DENSE_LAYERS = (0, 1, 2)
SPARSE_LAYERS = (3, 4, 5)
#: The six-block schedules: the first six entries of the checkpoint's.
LAYER_TYPES = ("sliding_attention",) * 5 + ("full_attention",)
MLP_LAYER_TYPES = ("dense",) * 3 + ("sparse",) * 3

#: The configuration this component computes, beyond what its blocks check themselves.
#: Anything else is rejected at construction rather than computed approximately.
SUPPORTED_CONFIG = {
    "num_hidden_layers": NUM_LAYERS,
    "hidden_size": 5120,
    "vocab_size": 151936,
    "rms_norm_eps": 1e-6,
    "tie_word_embeddings": False,
    "final_logit_softcapping": 30.0,
}

#: The KV cache ``prefill_context`` builds: bf16, 32-token pages, a 1024-token sequence
#: capacity and one request. A window equal to ``max_seq_len`` is the whole context.
KV_CACHE = {"tokens_per_block": 32, "max_seq_len": 1024, "max_tokens": 1024}
ATTENTION_WINDOWS = (512,) * 5 + (KV_CACHE["max_seq_len"],)

#: The most tokens one call may compute: the sequence capacity. A request's whole KV
#: length is bounded by its cache's ``max_seq_len``.
MAX_TOKENS = KV_CACHE["max_seq_len"]

_SM = (10, 3)

#: Checkpoint names of the model's own tensors, beside its blocks'.
EMBED = "model.embed_tokens.weight"
NORM = "model.norm.weight"
HEAD = "lm_head.weight"


def block_prefix(layer_idx: int) -> str:
    return f"model.layers.{layer_idx}."


def check_config(config) -> None:
    """Raise ValueError naming the first field outside what this component computes."""
    for field, expected in SUPPORTED_CONFIG.items():
        actual = getattr(config, field, None)
        if actual != expected or type(actual) is not type(expected):
            raise ValueError(f"unsupported {field}={actual!r}; this model computes {expected!r}")
    for field, expected in (("layer_types", LAYER_TYPES), ("mlp_layer_types", MLP_LAYER_TYPES)):
        actual = getattr(config, field, None)
        if actual is None or tuple(actual) != expected:
            raise ValueError(
                f"unsupported {field}={actual!r}; this model computes {list(expected)}"
            )
    for layer_idx in DENSE_LAYERS:
        dense_block.check_config(config, layer_idx)
    for layer_idx in SPARSE_LAYERS:
        sparse_block.check_config(config, layer_idx)


def weight_schema(config) -> "OrderedDict[str, tuple[tuple[int, ...], torch.dtype]]":
    """Every tensor by checkpoint name, in checkpoint order, with its shape and dtype.

    2394 tensors: the embedding, each block's (15 per dense block, 782 per sparse block:
    12 attention and norm tensors and the MoE's FP32 router and 256 BF16 experts), the
    final norm and the head. Matrices are [out, in]; the embedding is [vocab, hidden].
    """
    vocab, hidden = config.vocab_size, config.hidden_size
    schema = OrderedDict([(EMBED, ((vocab, hidden), torch.bfloat16))])
    for layer_idx in range(NUM_LAYERS):
        prefix = block_prefix(layer_idx)
        if layer_idx in DENSE_LAYERS:
            names = dense_block.weight_shapes(config).items()
        else:
            names = sparse_block.weight_shapes(config).items()
        for name, shape in names:
            schema[prefix + name] = (shape, torch.bfloat16)
        if layer_idx in SPARSE_LAYERS:
            for name, shape_dtype in moe_weight_shapes(config).items():
                schema[prefix + sparse_block.MLP_PREFIX + name] = shape_dtype
    schema[NORM] = ((hidden,), torch.bfloat16)
    schema[HEAD] = ((vocab, hidden), torch.bfloat16)
    return schema


def check_cache(md: TrtllmAttentionMetadata, kv_len: int) -> None:
    """Raise ValueError unless the call's cache holds the six layers in distinct valid slots.

    Required, before any layer writes: a ``KVCacheManagerV2`` holding exactly logical
    layers 0-5, mapped one-to-one onto its six local slots; 8 bf16 kv heads of 128 in
    every slot; a 512-token window on the sliding layers and the whole context on layer
    5, the full window holding the request's ``kv_len`` tokens; keys and values in one
    shared page table per pool; host pool tables that address, for every slot, exactly
    the key and value pages the manager allocated to that slot, so that no two slots
    share storage (``_check_pool_tables``); and an int32 CUDA block-offset table of the
    shape the attention op indexes. Reads host-side manager state and tensor metadata only.
    """
    mgr = getattr(md, "kv_cache_manager", None)
    if not isinstance(mgr, KVCacheManagerV2):
        raise ValueError(f"the KV cache must be a KVCacheManagerV2, got {type(mgr).__name__}")
    offsets = dict(getattr(mgr, "layer_offsets", None) or {})
    if set(offsets) != set(range(NUM_LAYERS)) or mgr.num_local_layers != NUM_LAYERS:
        raise ValueError(
            f"the KV cache must hold exactly logical layers 0-{NUM_LAYERS - 1}; it holds "
            f"{sorted(offsets)} in {mgr.num_local_layers} local slots"
        )
    slots = [offsets[layer_idx] for layer_idx in range(NUM_LAYERS)]
    if any(type(slot) is not int for slot in slots) or sorted(slots) != list(range(NUM_LAYERS)):
        raise ValueError(f"the six layers must map one-to-one onto slots 0-5, got {slots}")
    heads = list(getattr(mgr, "num_kv_heads_per_layer", None) or [])
    head_dims = list(getattr(mgr, "head_dim_per_layer", None) or [])
    if (
        mgr.dtype != DataType.BF16
        or heads != [8] * NUM_LAYERS
        or (head_dims and head_dims != [128] * NUM_LAYERS)
        or mgr.head_dim != 128
    ):
        raise ValueError(
            f"every slot must hold 8 bf16 kv heads of 128; got {mgr.dtype}, {heads}, "
            f"head_dim {mgr.head_dim}"
        )
    windows = list(mgr.max_attention_window_vec)
    for layer_idx, slot in enumerate(slots):
        window = windows[slot]
        if LAYER_TYPES[layer_idx] == "sliding_attention":
            retained = mgr.max_seq_len if window is None else window
            if retained != ATTENTION_WINDOWS[layer_idx]:
                raise ValueError(
                    f"sliding layer {layer_idx} must keep a {ATTENTION_WINDOWS[layer_idx]}-token "
                    f"window in slot {slot}; the manager keeps {retained}"
                )
        elif window is not None:
            raise ValueError(
                f"full-attention layer {layer_idx} must keep the whole context in slot {slot}; "
                f"the manager keeps a {window}-token window"
            )
    if mgr.max_seq_len < kv_len or md.max_seq_len != mgr.max_seq_len:
        raise ValueError(
            f"the cache holds {mgr.max_seq_len} tokens per sequence and the metadata states "
            f"{md.max_seq_len}; the request reaches {kv_len}"
        )
    _check_pool_tables(md, mgr, slots)
    offsets = md.kv_cache_block_offsets
    if (
        not isinstance(offsets, torch.Tensor)
        or offsets.device.type != "cuda"
        or offsets.dtype != torch.int32
        or offsets.dim() != 4
        or tuple(offsets.shape[i] for i in (0, 2, 3))
        != (mgr.num_attention_op_pools, 2, mgr.max_blocks_per_seq)
        or offsets.shape[1] < 1
    ):
        raise ValueError(
            "the block-offset table must be a CUDA torch.int32 [pools, sequences, 2, "
            f"{mgr.max_blocks_per_seq}] tensor with {mgr.num_attention_op_pools} pools"
        )


def _pool_windows(mgr: KVCacheManagerV2) -> dict:
    """Each pool's attention window, from the windows of the slots it holds (``None``: full)."""
    windows: dict = {}
    for slot in range(NUM_LAYERS):
        pool = int(mgr.impl.get_layer_group_id(LayerId(slot)))
        window = mgr.max_attention_window_vec[slot]
        if windows.setdefault(pool, window) != window:
            raise ValueError(f"pool {pool} holds slots with windows {windows[pool]} and {window}")
    return windows


def check_request(md: TrtllmAttentionMetadata, num_tokens: int, kv_len: int):
    """Raise ValueError unless the call continues its request's cache on valid pages.

    The call reads keys and values of earlier tokens and writes its own ``num_tokens``
    after ``kv_len - num_tokens`` cached ones. Required, from the manager's own state: one
    request id, whose cache in the manager is active, holds exactly those ``kv_len -
    num_tokens`` tokens (its ``history_length``: the call appends right after them, so it
    never rewrites a token the cache holds -- committed or reused prefix pages included,
    which the manager keeps within the history -- and leaves no unwritten gap) and has
    capacity for ``kv_len`` tokens; and for every pool, a valid page of that pool -- not
    ``BAD_PAGE_INDEX``, inside the pool -- at every block ordinal a token of the call can
    read or write: on the full layer's pool every ordinal below ``kv_len``, on the sliding
    layers' pool those from the oldest key the call's first query attends to. Ordinals
    older than that may carry the manager's out-of-window sentinel.

    Returns ``(expected, live)``: host int32 ``[pools, 2, n]`` block-offset entries the op
    must read for those pages (the manager's page-to-offset conversion of each page) and
    a bool ``[pools, n]`` mask of the ordinals checked, ``n = ceil(kv_len / tokens_per_block)``.
    The caller compares them with the CUDA table. Reads host-side manager state only.
    """
    mgr = md.kv_cache_manager
    request_ids = list(md.request_ids or ())
    if len(request_ids) != 1:
        raise ValueError(f"one request per call is supported; metadata names {request_ids}")
    (request_id,) = request_ids
    kv_cache = mgr.kv_cache_map.get(request_id)
    if kv_cache is None or not kv_cache.is_active:
        raise ValueError(f"request {request_id} holds no active cache in this manager")
    history = int(kv_cache.history_length)
    if history != kv_len - num_tokens:
        raise ValueError(
            f"request {request_id}'s cache holds {history} tokens; the call writes tokens "
            f"{kv_len - num_tokens}..{kv_len - 1}, but a step must append right after the tokens "
            "the cache holds"
        )
    if kv_cache.capacity < kv_len:
        raise ValueError(
            f"request {request_id} has capacity for {kv_cache.capacity} tokens; the call reaches "
            f"{kv_len}"
        )
    tokens_per_block = mgr.tokens_per_block
    num_blocks = -(-kv_len // tokens_per_block)
    held = int(kv_cache.num_blocks)
    if held < num_blocks:
        raise ValueError(f"request {request_id} holds {held} blocks; the call needs {num_blocks}")
    first_query = kv_len - num_tokens
    pools = _pool_windows(mgr)
    num_pools = len(mgr.impl.layer_grouping)
    if sorted(pools) != list(range(num_pools)):
        raise ValueError(f"the six slots occupy pools {sorted(pools)} of {num_pools}")
    expected = torch.zeros((num_pools, 2, num_blocks), dtype=torch.int32)
    live = torch.zeros((num_pools, num_blocks), dtype=torch.bool)
    for pool, window in sorted(pools.items()):
        oldest = 0 if window is None else max(0, first_query - window + 1)
        pages = np.frombuffer(kv_cache.get_base_page_indices(pool), dtype=np.int32, count=held)
        layer = next(
            LayerId(slot)
            for slot in range(NUM_LAYERS)
            if int(mgr.impl.get_layer_group_id(LayerId(slot))) == pool
        )
        scale, value_offset = int(mgr.index_scales[pool]), int(mgr.kv_offset[pool])
        bound = int(mgr.impl.get_page_index_upper_bound(layer, Role.KEY))
        for ordinal in range(oldest // tokens_per_block, num_blocks):
            page = int(pages[ordinal])
            if page == BAD_PAGE_INDEX or page < 0 or scale * page + value_offset >= bound:
                raise ValueError(
                    f"request {request_id} has page {page} at block {ordinal} of pool {pool}; "
                    f"the call reads or writes that block and the pool holds pages below "
                    f"{bound // scale}"
                )
            expected[pool, 0, ordinal] = scale * page
            expected[pool, 1, ordinal] = scale * page + value_offset
            live[pool, ordinal] = True
    return expected, live


def _overlap(spans) -> bool:
    spans = sorted(spans)
    return any(end > start for (_, end), (start, _) in zip(spans, spans[1:]))


def _check_pool_tables(md: TrtllmAttentionMetadata, mgr: KVCacheManagerV2, slots) -> None:
    """Raise ValueError unless the host pool tables address each slot's own cache storage.

    The attention op reads slot s's row of ``host_kv_cache_pool_mapping`` as int32
    ``(pool, offset)`` and finds the slot's keys at ``host_kv_cache_pool_pointers[pool, 0] +
    offset * 2 * page``, a page holding one slot's keys of ``tokens_per_block`` tokens and
    the next page their values, with the pool's page indices from its block offsets. The
    tables are host tensors a caller can replace, so each row is checked against the
    manager's allocation itself: the pool is the slot's layer group, the offset keeps the
    slot's key and value pages inside one slot of that pool, the address is where the
    manager keeps the slot's keys, its values follow one page later, and no two slots'
    pages, and no two pools, overlap. The five sliding layers and layer 5 are layer groups
    of different slot sizes, which the manager keeps in separate pools; a cache whose layer
    groups shared one pool is rejected rather than reasoned about.
    """
    impl = mgr.impl
    num_pools = len(impl.layer_grouping)
    mapping = md.host_kv_cache_pool_mapping
    pointers = md.host_kv_cache_pool_pointers
    if (
        not isinstance(mapping, torch.Tensor)
        or mapping.device.type != "cpu"
        or mapping.dtype != torch.int32
        or tuple(mapping.shape) != (NUM_LAYERS, 2)
    ):
        raise ValueError(f"the pool mapping must be a host torch.int32 [{NUM_LAYERS}, 2] table")
    if (
        not isinstance(pointers, torch.Tensor)
        or pointers.device.type != "cpu"
        or pointers.dtype != torch.int64
        or tuple(pointers.shape) != (num_pools, 2)
    ):
        raise ValueError(
            f"the pool pointers must be a host torch.int64 [{num_pools}, 2] table, one row per pool"
        )
    if (
        mgr.kv_cache_type != CacheTypeCpp.SELF
        or mgr.enable_swa_scratch_reuse
        or mgr.num_attention_op_pools != num_pools
    ):
        raise ValueError(
            "the cache must hold keys and values with one shared page table per pool, "
            "without sliding-window scratch reuse"
        )
    page = mgr.tokens_per_block * 8 * 128 * torch.bfloat16.itemsize
    pairs, pools = [], {}
    for layer_idx, slot in enumerate(slots):
        layer = LayerId(slot)
        pool, offset = mapping[slot].tolist()
        group = int(impl.get_layer_group_id(layer))
        scale = impl.get_page_index_scale(layer, Role.KEY)
        if pool != group or not 0 <= offset < scale // 2:
            raise ValueError(
                f"layer {layer_idx} has pool row ({pool}, {offset}); the manager keeps slot {slot} "
                f"in pool {group}, whose slots hold {scale // 2} key and value page pairs"
            )
        start = impl.get_mem_pool_base_address(layer, Role.KEY, PageIndexMode.PER_LAYER)
        key = impl.get_mem_pool_base_address(layer, Role.KEY, PageIndexMode.SHARED)
        value = impl.get_mem_pool_base_address(layer, Role.VALUE, PageIndexMode.SHARED)
        addressed = int(pointers[pool, 0]) + offset * 2 * page
        if (
            addressed != key
            or int(pointers[pool, 1]) != 0
            or value != key + page
            or impl.get_page_stride(layer, Role.KEY) != page
            or impl.get_page_stride(layer, Role.VALUE) != page
            or not start <= key <= key + 2 * page <= start + scale * page
        ):
            raise ValueError(
                f"layer {layer_idx}'s pool row ({pool}, {offset}) with pool pointers "
                f"{pointers[pool].tolist()} addresses {addressed:#x}; the manager keeps slot "
                f"{slot}'s keys at {key:#x} and values at {value:#x} in pool {pool} from "
                f"{start:#x}, in {page}-byte pages"
            )
        extent = (start, key + impl.get_page_index_upper_bound(layer, Role.KEY) * page)
        if pools.setdefault(pool, extent) != extent:
            raise ValueError(
                f"pool {pool} spans {pools[pool]} for one slot and {extent} for another"
            )
        pairs.append((key, key + 2 * page))
    if _overlap(pairs) or _overlap(pools.values()):
        raise ValueError(
            f"cache slots share storage: key and value pages {pairs}, pools {sorted(pools.values())}"
        )


@contextlib.contextmanager
def prefill_context(num_tokens: int, request_id: int = 0) -> Iterator[TrtllmAttentionMetadata]:
    """A fresh six-layer ``KVCacheManagerV2`` and prepared metadata for one new request.

    The manager holds logical layers 0-5 in slots 0-5 (bf16, 8 kv heads of 128,
    ``KV_CACHE`` pages and capacity, windows ``ATTENTION_WINDOWS``) with block reuse,
    partial reuse, sliding-window scratch reuse and the overlap scheduler off; the
    metadata holds one context request of ``num_tokens`` tokens starting at an empty
    cache. The manager is shut down on exit, also after an error.
    """
    mgr = KVCacheManagerV2(
        KvCacheConfig(
            max_tokens=KV_CACHE["max_tokens"],
            max_attention_window=list(ATTENTION_WINDOWS),
            enable_block_reuse=False,
            enable_partial_reuse=False,
            enable_swa_scratch_reuse=False,
        ),
        CacheTypeCpp.SELF,
        num_layers=NUM_LAYERS,
        num_kv_heads=8,
        head_dim=128,
        tokens_per_block=KV_CACHE["tokens_per_block"],
        max_seq_len=KV_CACHE["max_seq_len"],
        max_batch_size=1,
        mapping=ParallelMapping(world_size=1, tp_size=1, rank=0),
        dtype=DataType.BF16,
        disable_overlap_scheduler=True,
    )
    try:
        mgr.add_dummy_requests([request_id], token_nums=[num_tokens])
        md = TrtllmAttentionMetadata(
            max_num_requests=1,
            max_num_tokens=MAX_TOKENS,
            kv_cache_manager=mgr,
            num_contexts=1,
            kv_cache_params=KVCacheParams(use_cache=True, num_cached_tokens_per_seq=[0]),
            seq_lens=torch.tensor([num_tokens], dtype=torch.int),
            request_ids=[request_id],
            prompt_lens=[num_tokens],
        )
        md.prepare()
        yield md
    finally:
        mgr.shutdown()


class BeamSmallModelPrefillBf16Sm103Tp1(nn.Module):
    """Beam six-block model: token ids in, every token's FP32 vocabulary logits out.

    State: ``embed_tokens`` ``[151936, 5120]`` and ``lm_head`` ``[151936, 5120]``, two
    separate BF16 tensors; ``layers``, dense blocks 0-2 and sparse blocks 3-5, each with
    its own tensors, router and 256 experts; ``norm`` ``[5120]`` BF16. ``load_weights``
    loads all 2394 tensors by checkpoint name; the model runs only after that succeeded.

    ``forward(input_ids [1, T] int32/int64, position_ids [1, T], attn_metadata)`` returns
    ``[1, T, 151936]`` ``torch.float32`` on the model's device: row t, entry v is the
    softcapped logit for vocabulary id v of the t-th token of the call. Accepted:
    1 <= T <= ``MAX_TOKENS`` token ids in ``[0, 151936)``, contiguous on the model's device;
    the prepared ``TrtllmAttentionMetadata`` of one step of one request -- a context step
    of T tokens after ``past`` cached ones (paged-context FMHA when ``past > 0``) or a
    one-token generation step -- on the six-layer cache ``check_cache`` describes, its
    request's pages valid as ``check_request`` describes and its KV length at most the
    cache's ``max_seq_len``; and integer positions equal to ``arange(past, past + T)``.
    ``prefill_context(T)`` yields such metadata for a fresh context; the generator in
    ``generation.py`` for every step of its requests. Anything else is rejected with
    ValueError before the embedding lookup or any other kernel runs, including an id
    outside the vocabulary. Inputs and parameters are never modified. Batches of several
    requests and padding are not computed here.

    Weights cannot change under a cache built from them: ``load_weights`` raises while a
    generator holding this model's cache is open (``close`` it first).
    """

    def __init__(self, config, device: str | torch.device = "cuda"):
        super().__init__()
        check_config(config)
        device = torch.device(device)
        if device.type != "cuda" or torch.cuda.get_device_capability(device) != _SM:
            raise ValueError(
                f"this model is built for sm_{_SM[0]}{_SM[1]} CUDA devices, got {device}"
            )
        self.hidden_size = config.hidden_size
        self.vocab_size = config.vocab_size
        self.eps = config.rms_norm_eps
        self.logit_softcap = config.final_logit_softcapping
        self._schema = weight_schema(config)

        def empty(name: str) -> nn.Parameter:
            shape, dtype = self._schema[name]
            return nn.Parameter(torch.empty(shape, dtype=dtype, device=device), requires_grad=False)

        self.embed_tokens = empty(EMBED)
        self.layers = nn.ModuleList(
            dense_block.BeamDenseBlockBf16Sm103Tp1(config, layer_idx, device=device)
            if layer_idx in DENSE_LAYERS
            else sparse_block.BeamSparseBlockBf16Sm103Tp1(config, layer_idx, device=device)
            for layer_idx in range(NUM_LAYERS)
        )
        self.norm = empty(NORM)
        self.lm_head = empty(HEAD)
        self._loaded = False
        #: Generators whose KV cache holds keys and values computed with these weights.
        self._cache_holders: "weakref.WeakSet" = weakref.WeakSet()

    @property
    def device(self) -> torch.device:
        return self.embed_tokens.device

    def checkpoint_tensors(self) -> "OrderedDict[str, torch.Tensor]":
        """All 2394 tensors by checkpoint name, in checkpoint order."""
        tensors = OrderedDict([(EMBED, self.embed_tokens)])
        for layer_idx, block in enumerate(self.layers):
            for name, tensor in block.checkpoint_tensors().items():
                tensors[block_prefix(layer_idx) + name] = tensor
        tensors[NORM] = self.norm
        tensors[HEAD] = self.lm_head
        return tensors

    def check_weights(self, weights: Mapping[str, torch.Tensor]) -> None:
        """Raise unless ``weights`` holds exactly the 2394 names with their dtypes and shapes.

        Reads tensor metadata only, never values, and copies nothing.
        """
        check_weights(self._schema, weights, "model")

    @torch.no_grad()
    def load_weights(self, weights: Mapping[str, torch.Tensor]) -> None:
        """Copy exactly the model's 2394 tensors, by checkpoint name, [out, in].

        Names are ``model.embed_tokens.weight``, ``model.layers.<i>.<block name>``,
        ``model.norm.weight`` and ``lm_head.weight``; dtypes are BF16 except each sparse
        block's FP32 ``mlp.gate.weight`` and ``mlp.gate.e_score_correction_bias``; host or
        CUDA tensors are copied as given, without transposing or converting. Every tensor
        is checked before any parameter changes, so a rejected load leaves the model as it
        was. The model runs only after a load completed: a failure while copying leaves it
        unloaded until a later load succeeds. Copies proceed one tensor (one expert, one
        block's attention tensors) at a time; no copy of a whole bank is made.

        Raises RuntimeError, before checking or copying anything, while an open generator
        holds a KV cache computed with the current weights: its cached and reusable keys
        and values would no longer match them.
        """
        holders = [holder for holder in self._cache_holders if not holder.closed]
        if holders:
            raise RuntimeError(
                f"{len(holders)} open generator(s) hold KV caches computed with these weights; "
                "close them before loading new weights"
            )
        self.check_weights(weights)
        self._loaded = False
        for name, param in ((EMBED, self.embed_tokens), (NORM, self.norm), (HEAD, self.lm_head)):
            param.copy_(weights[name])
        for layer_idx, block in enumerate(self.layers):
            prefix = block_prefix(layer_idx)
            block.load_weights(
                {name[len(prefix) :]: w for name, w in weights.items() if name.startswith(prefix)}
            )
        self._loaded = True

    def register_cache_holder(self, holder) -> None:
        """Record a generator whose KV cache holds keys and values computed with these weights.

        ``holder`` must expose a ``closed`` attribute; while it is False ``load_weights``
        raises. Holders are referenced weakly.
        """
        self._cache_holders.add(holder)

    def _check_call(
        self, input_ids: torch.Tensor, position_ids: torch.Tensor, md: TrtllmAttentionMetadata
    ) -> int:
        """Raise ValueError on anything the model would otherwise compute wrongly; return T."""
        if not self._loaded:
            raise ValueError("load_weights must complete before the model runs")
        if (
            not isinstance(input_ids, torch.Tensor)
            or input_ids.dtype not in (torch.int32, torch.int64)
            or input_ids.layout != torch.strided
        ):
            raise ValueError("input_ids must be a dense torch.int32 or torch.int64 tensor")
        if input_ids.dim() != 2 or input_ids.shape[0] != 1:
            raise ValueError(f"input_ids must be [1, T], got {tuple(input_ids.shape)}")
        num_tokens = input_ids.shape[1]
        if not 1 <= num_tokens <= MAX_TOKENS:
            raise ValueError(
                f"supported token counts per call are 1..{MAX_TOKENS}, got {num_tokens}"
            )
        if input_ids.device != self.device or not input_ids.is_contiguous():
            raise ValueError(f"input_ids must be contiguous on the model's device {self.device}")
        if (
            not isinstance(position_ids, torch.Tensor)
            or position_ids.dtype not in (torch.int32, torch.int64)
            or tuple(position_ids.shape) != (1, num_tokens)
            or position_ids.device != self.device
            or not position_ids.is_contiguous()
        ):
            raise ValueError(
                f"position_ids must be a contiguous integer [1, {num_tokens}] tensor on {self.device}"
            )
        if not isinstance(md, TrtllmAttentionMetadata):
            raise ValueError(
                f"attn_metadata must be TrtllmAttentionMetadata, got {type(md).__name__}"
            )
        mgr = getattr(md, "kv_cache_manager", None)
        if not isinstance(mgr, KVCacheManagerV2):
            raise ValueError(f"the KV cache must be a KVCacheManagerV2, got {type(mgr).__name__}")
        pasts = {
            _attention.check_metadata(
                md, num_tokens, layer_idx=layer_idx, num_kv_heads=8, head_dim=128
            )
            for layer_idx in range(NUM_LAYERS)
        }
        (past,) = pasts
        kv_len = past + num_tokens
        check_cache(md, kv_len)
        expected_table, live = check_request(md, num_tokens, kv_len)
        table = md.kv_cache_block_offsets[:, 0, :, : expected_table.shape[-1]]
        table_ok = (
            (table == expected_table.to(table.device, non_blocking=True))
            | ~live.to(table.device, non_blocking=True)[:, None, :]
        ).all()
        expected = torch.arange(
            past, past + num_tokens, dtype=position_ids.dtype, device=self.device
        )
        low, high = torch.aminmax(input_ids)
        positions_ok, ids_ok, table_ok = torch.stack(
            (
                (position_ids[0] == expected).all(),
                (low >= 0) & (high < self.vocab_size),
                table_ok,
            )
        ).tolist()
        if not positions_ok:
            raise ValueError(
                f"the call appends tokens {past}..{kv_len - 1} after {past} cached tokens; its "
                "positions must be exactly those"
            )
        if not table_ok:
            raise ValueError(
                "the block-offset table does not address the pages the manager allocated to "
                "the request"
            )
        if not ids_ok:
            raise ValueError(
                f"token ids must be in [0, {self.vocab_size}); got values in "
                f"[{int(low)}, {int(high)}]"
            )
        return num_tokens

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        attn_metadata: TrtllmAttentionMetadata,
    ) -> torch.Tensor:
        num_tokens = self._check_call(input_ids, position_ids, attn_metadata)
        x = nn.functional.embedding(input_ids, self.embed_tokens)
        for block in self.layers:
            x = block(x, position_ids, attn_metadata)
        h = flashinfer_rmsnorm(x.view(num_tokens, self.hidden_size), self.norm, self.eps)
        head = cublas_mm(h, self.lm_head.t())
        logits = beam_logit_softcap(head, self.logit_softcap)
        return logits.view(1, num_tokens, self.vocab_size)
