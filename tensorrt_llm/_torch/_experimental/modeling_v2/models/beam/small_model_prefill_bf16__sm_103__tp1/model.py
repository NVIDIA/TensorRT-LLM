# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Beam six-block small model / sm_103 / tp1: token ids to prefill logits, constructed directly.

A component, not a routed target. There is no ``routing.py``, no router-index entry and
no registered architecture: nothing in the engine builds this class, and it serves no
request. A caller constructs it from a six-block configuration of the checkpoint -- the
checkpoint's own fields with ``num_hidden_layers = 6`` and the first six entries of its
two layer schedules -- loads its 2394 tensors and runs one fresh BF16 context through it,
with the attention batch state prepared on a six-layer ``KVCacheManagerV2``
(``prefill_context`` builds one). It does not call the built-in Beam modeling.

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

Cache. One call appends every layer's keys and values to one ``KVCacheManagerV2`` that
holds exactly logical layers 0-5, each in its own local slot: ``layer_offsets`` maps
logical layer i to slot i, and slot i's rows in the pool tables belong to it alone. The
five sliding layers keep a 512-token window and layer 5 keeps the whole context; the
manager states this as its per-layer window vector ``[512] * 5 + [None]`` (``None`` is
the full ``max_seq_len``), and the attention op of each layer is called with that layer's
window. ``check_cache`` verifies all of it before the first layer runs, including that
the host pool tables the attention op reads address each slot's own key and value pages
as the manager allocated them. Cache contents are written as the backend requires; they
are not read back, reused or shared between calls.

Host synchronization per call: one for the model's token-id range and position check,
one for each block's position check, and each sparse block's MoE dispatch (two, plus two
per selected expert). A call is therefore not capturable in a CUDA graph.
"""

from __future__ import annotations

import contextlib
from collections import OrderedDict
from typing import Iterator, Mapping

import torch
from torch import nn

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
from tensorrt_llm.runtime.kv_cache_manager_v2 import LayerId, PageIndexMode

from .. import _attention
from ..dense_block_bf16__sm_103__tp1 import block as dense_block
from ..moe_bf16__sm_103__tp1.moe import check_weights
from ..moe_bf16__sm_103__tp1.moe import weight_shapes as moe_weight_shapes
from ..sparse_block_bf16__sm_103__tp1 import block as sparse_block
from .ops.activation.beam_logit_softcap import beam_logit_softcap

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

#: The longest fresh context the model was measured at; longer ones are rejected.
MAX_TOKENS = 129

#: The KV cache ``prefill_context`` builds: bf16, 32-token pages, a 1024-token sequence
#: capacity and one request. A window equal to ``max_seq_len`` is the whole context.
KV_CACHE = {"tokens_per_block": 32, "max_seq_len": 1024, "max_tokens": 1024}
ATTENTION_WINDOWS = (512,) * 5 + (KV_CACHE["max_seq_len"],)

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


def check_cache(md: TrtllmAttentionMetadata, num_tokens: int) -> None:
    """Raise ValueError unless the call's cache holds the six layers in distinct valid slots.

    Required, before any layer writes: a ``KVCacheManagerV2`` holding exactly logical
    layers 0-5, mapped one-to-one onto its six local slots; 8 bf16 kv heads of 128 in
    every slot; a 512-token window on the sliding layers and the whole context on layer
    5, the full window holding the call's tokens; keys and values in one shared page
    table per pool; and host pool tables that address, for every slot, exactly the key
    and value pages the manager allocated to that slot, so that no two slots share
    storage (``_check_pool_tables``). Reads host-side manager state and host tensors only.
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
    if mgr.max_seq_len < num_tokens or md.max_seq_len != mgr.max_seq_len:
        raise ValueError(
            f"the cache holds {mgr.max_seq_len} tokens per sequence and the metadata states "
            f"{md.max_seq_len}; the call has {num_tokens}"
        )
    _check_pool_tables(md, mgr, slots)


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
    ``[1, T, 151936]`` ``torch.float32`` on the model's device: row t, entry v is token t's
    softcapped logit for vocabulary id v. Accepted: 1 <= T <= 129 token ids in
    ``[0, 151936)``, contiguous on the model's device; integer positions equal to
    ``arange(T)``; and the prepared ``TrtllmAttentionMetadata`` of one fresh context
    request of T tokens on the six-layer cache ``check_cache`` describes, such as
    ``prefill_context(T)`` yields. Anything else is rejected with ValueError before the
    embedding lookup or any other kernel runs, including an id outside the vocabulary.
    Inputs and parameters are never modified. Decode, cached prefixes, chunked prefill,
    batches, padding and contexts longer than 129 tokens are not computed here.
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
        """
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
            raise ValueError(f"supported context lengths are 1..{MAX_TOKENS}, got {num_tokens}")
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
        check_cache(md, num_tokens)
        for layer_idx in range(NUM_LAYERS):
            _attention.check_metadata(
                md, num_tokens, layer_idx=layer_idx, num_kv_heads=8, head_dim=128
            )
        expected = torch.arange(num_tokens, dtype=position_ids.dtype, device=self.device)
        low, high = torch.aminmax(input_ids)
        positions_ok, ids_ok = torch.stack(
            ((position_ids[0] == expected).all(), (low >= 0) & (high < self.vocab_size))
        ).tolist()
        if not positions_ok:
            raise ValueError("a fresh context must have positions 0..T-1")
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
