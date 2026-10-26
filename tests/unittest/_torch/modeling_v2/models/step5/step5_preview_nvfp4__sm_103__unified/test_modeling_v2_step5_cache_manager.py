# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Step-5's KV cache, on real managers built from the checkpoint's own config.

The engine selects ``Step5CacheManager`` for a Step-5 config and builds it the way
``_create_kv_cache_manager`` does; every test here goes through those two engine
functions, at one KV head per rank (attention TP4) and at four (attention DP4):

* the per-layer windows come from ``layer_types``: 512 tokens on the 69 sliding
  layers, the whole ``max_seq_len`` on the 23 full-attention layers;
* the full-attention layers hold the sparse indexer's state -- bf16 ``[k | z]`` per
  token and an fp8 summary per 8-token region, one page per block -- in the slot of
  their K/V, served by slot-major views at the addresses the pools hold;
* block reuse covers both kinds: a request sharing a prefix reuses its blocks
  (``iterReusedBlocks`` moves in both windows) and, resuming mid-region and inside the
  window, sees exactly what a cold prefill computes, then finishes exactly as one;
* the main K/V is bf16, FP8 or NVFP4 (with its block scales), at heads zero-padded
  from 192 to 256 (the width the attention kernels run at), while the indexer state
  keeps its own dtypes;
* sizing at ``kv_cache_fraction`` 0.7, ``max_batch_size`` 128, ``max_seq_len`` 9216 and
  ``max_num_tokens`` 8192 charges every pool, in the static cost model and in the pools
  V2 allocates; the split between the two life cycles comes from those limits: the
  sliding pool holds 128 windows plus one step's 8192 context tokens and no more, and
  the full-attention pool (K/V and indexer pages) the rest, unless the KV cache config
  sets a ``pool_ratio`` or ``avg_seq_len``.

A "forward" here is a stand-in that writes what a prefill writes, as a function of the
token's prefix: K/V (and block-scale) bytes per layer, the indexer's ``[k | z]``, and
each region's summary once its last token is written, computed from the ``[k | z]``
rows in the pool (reused ones included) with the reference's compression. A cold
prefill is the same function evaluated from scratch, so a reused block is correct
exactly when it equals the cold value.

Needs the checkpoint's ``config.json``: ``STEP5_PREVIEW_CHECKPOINT``, else
``$LLM_MODELS_ROOT/Step-5-Preview-NVFP4``; skipped without one.
"""

from __future__ import annotations

import os
import random
from pathlib import Path

import pytest
import torch
from utils.llm_data import llm_models_root

from tensorrt_llm._torch.attention.backends.sparse.step5 import Step5CacheManager, step5_text_config
from tensorrt_llm._torch.attention.backends.sparse.step5.params import (
    step5_sliding_pool_blocks,
    step5_window_blocks,
)
from tensorrt_llm._torch.disaggregation.resource.page import MapperKind
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.pyexecutor import _util, config_utils
from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import (
    KVCacheManagerV2,
    Role,
    _gpu_alloc_granularity,
)
from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequest, LlmRequestState, SamplingConfig
from tensorrt_llm._torch.pyexecutor.scheduler import ScheduledRequests
from tensorrt_llm.bindings import DataType
from tensorrt_llm.llmapi.llm_args import KvCacheConfig
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models.modeling_utils import QuantConfig
from tensorrt_llm.quantization.mode import QuantAlgo
from tensorrt_llm.runtime.kv_cache_manager_v2 import GPU_LEVEL, PageIndexMode

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="the managers allocate their pools on the GPU"
)

# The serving envelope every gate runs.
MAX_SEQ_LEN = 9216
MAX_BATCH_SIZE = 128
MAX_NUM_TOKENS = 8192
KV_CACHE_FRACTION = 0.7
TOKENS_PER_BLOCK = 32
# The `chunked` variant's prefill chunk.
CHUNK = 512

# What the checkpoint's config.json declares, restated so the test checks the
# derivation rather than repeating it.
DECODER_LAYERS = 92
SLIDING_WINDOW = 512
FULL_LAYERS = tuple(range(3, DECODER_LAYERS, 4))
KV_HEADS, HEAD_DIM = 4, 192
# The attention kernels run on heads zero-padded to 256, so the K/V pools hold 256.
POOL_HEAD_DIM = 256
PROXY_DIM, REGION = 256, 8
# One indexer page per block: bf16 [k | z] per token, then an fp8 summary per region.
KZ_PAGE = TOKENS_PER_BLOCK * 2 * PROXY_DIM * 2
SUMMARY_PAGE = TOKENS_PER_BLOCK // REGION * PROXY_DIM
INDEX_PAGE = KZ_PAGE + SUMMARY_PAGE

# (tp_size, enable_attention_dp) on a four-rank world; this process is rank 0.
LAYOUTS = {"attn_tp4": (4, False), "attn_dp4": (4, True)}
KV_DTYPES = {
    "bf16": (None, DataType.BF16),
    "fp8": (QuantAlgo.FP8, DataType.FP8),
    "nvfp4": (QuantAlgo.NVFP4, DataType.NVFP4),
}


def _checkpoint_dir() -> Path | None:
    explicit = os.environ.get("STEP5_PREVIEW_CHECKPOINT")
    if explicit:
        return Path(explicit)
    root = llm_models_root()
    return None if root is None else root / "Step-5-Preview-NVFP4"


@pytest.fixture(scope="module")
def checkpoint() -> Path:
    path = _checkpoint_dir()
    if path is None or not (path / "config.json").is_file():
        pytest.skip(
            "no Step-5-Preview-NVFP4 checkpoint: set STEP5_PREVIEW_CHECKPOINT or stage "
            "Step-5-Preview-NVFP4 under LLM_MODELS_ROOT"
        )
    return path


@pytest.fixture(scope="module")
def pretrained(checkpoint: Path):
    """The config the engine loads: the published composite, as is."""
    return config_utils.load_pretrained_config(str(checkpoint))


def _mapping(layout: str) -> Mapping:
    tp_size, attention_dp = LAYOUTS[layout]
    return Mapping(
        world_size=tp_size, rank=0, tp_size=tp_size, pp_size=1, enable_attention_dp=attention_dp
    )


def _heads_per_rank(layout: str) -> int:
    tp_size, attention_dp = LAYOUTS[layout]
    return KV_HEADS if attention_dp else KV_HEADS // tp_size


def _model_config(pretrained, layout: str, kv_dtype: str) -> ModelConfig:
    kv_algo, _ = KV_DTYPES[kv_dtype]
    return ModelConfig(
        pretrained_config=pretrained,
        mapping=_mapping(layout),
        quant_config=QuantConfig(quant_algo=QuantAlgo.NVFP4, kv_cache_quant_algo=kv_algo),
    )


def _build(model_config: ModelConfig, kv_cache_config: KvCacheConfig) -> Step5CacheManager:
    """Select and build the manager through the engine's own two functions."""
    cls = _util.get_kv_cache_manager_cls(model_config, kv_cache_config)
    assert cls is Step5CacheManager
    return _util._create_kv_cache_manager(
        None,
        cls,
        model_config.mapping,
        kv_cache_config,
        tokens_per_block=TOKENS_PER_BLOCK,
        max_seq_len=MAX_SEQ_LEN,
        max_batch_size=MAX_BATCH_SIZE,
        spec_config=None,
        sparse_attention_config=None,
        max_num_tokens=MAX_NUM_TOKENS,
        max_beam_width=1,
        kv_connector_manager=None,
        enable_kv_cache_stats=True,
        model_config=model_config,
        dtype=torch.bfloat16,
    )


def _small_cache_config(**overrides) -> KvCacheConfig:
    kwargs = dict(max_tokens=64 * 1024, enable_block_reuse=True, host_cache_size=0)
    kwargs.update(overrides)
    return KvCacheConfig(**kwargs)


def _kv_page_bytes(kv_dtype: str, heads: int) -> tuple[int, int]:
    """(K or V page, K or V block-scale page) in bytes, per layer."""
    _, dtype = KV_DTYPES[kv_dtype]
    bits = {DataType.BF16: 16, DataType.FP8: 8, DataType.NVFP4: 4}[dtype]
    data = TOKENS_PER_BLOCK * heads * POOL_HEAD_DIM * bits // 8
    scales = TOKENS_PER_BLOCK * heads * POOL_HEAD_DIM // 16 if dtype == DataType.NVFP4 else 0
    return data, scales


def _sliding_layers() -> list[int]:
    return [layer for layer in range(DECODER_LAYERS) if layer not in FULL_LAYERS]


# ----------------------------------------------------------------------------
# (1) windows
# ----------------------------------------------------------------------------


@pytest.mark.parametrize("layout", LAYOUTS)
def test_windows_come_from_layer_types(pretrained, layout: str) -> None:
    """The 69 sliding layers keep a 512-token window and the 23 full-attention
    layers the whole sequence, in two V2 life cycles; the MTP entries of
    ``layer_types`` (92-94) are not cache layers."""
    model_config = _model_config(pretrained, layout, "bf16")
    derived = _util._derive_v2_layer_type_attention_windows(
        KvCacheConfig(), Step5CacheManager, model_config, MAX_SEQ_LEN
    )
    expected = [MAX_SEQ_LEN if layer in FULL_LAYERS else SLIDING_WINDOW for layer in range(92)]
    assert derived == expected
    assert step5_text_config(pretrained).sliding_window == SLIDING_WINDOW

    cm = _build(model_config, _small_cache_config())
    try:
        assert cm.num_local_layers == DECODER_LAYERS
        assert cm.max_attention_window_vec == [
            None if layer in FULL_LAYERS else SLIDING_WINDOW for layer in range(DECODER_LAYERS)
        ]
        assert cm.sparse_layer_ids == FULL_LAYERS
        assert cm.local_sparse_layer_ids == list(FULL_LAYERS)
        sliding_pools = {
            cm.layer_to_pool_mapping_dict[cm.layer_offsets[i]] for i in _sliding_layers()
        }
        full_pools = {cm.layer_to_pool_mapping_dict[cm.layer_offsets[i]] for i in FULL_LAYERS}
        assert len(sliding_pools) == len(full_pools) == 1 and sliding_pools != full_pools
        assert cm.num_kv_heads_per_layer == [_heads_per_rank(layout)] * DECODER_LAYERS
        assert step5_text_config(pretrained).head_dim == HEAD_DIM
        assert cm.head_dim_per_layer == [POOL_HEAD_DIM] * DECODER_LAYERS
    finally:
        cm.shutdown()


# ----------------------------------------------------------------------------
# (2) + (4) indexer pools, their accessors, and the main K/V dtypes
# ----------------------------------------------------------------------------


def _thop_addresses(cm, layer: int, slot: int, page_bytes: int, plane: int) -> tuple[int, int]:
    """Byte addresses of a layer's K and V page for ``slot`` as the attention op
    forms them from the pool pointers, the pool mapping and the page-index scale
    (``plane`` 0 = data, 1 = NVFP4 block scales)."""
    local = cm.layer_offsets[layer]
    pool, offset = cm.kv_cache_pool_mapping[local].tolist()
    pointers = cm.kv_cache_pool_pointers
    base = int(pointers[pool, 0, plane]) if pointers.dim() == 3 else int(pointers[pool, 0])
    key_entry = int(cm.index_scales[pool]) * slot
    value_entry = key_entry + int(cm.kv_offset[pool])
    shift = offset * cm.kv_factor * page_bytes
    return base + key_entry * page_bytes + shift, base + value_entry * page_bytes + shift


def _probe_slots(num_slots: int) -> list[int]:
    return sorted({0, 1, num_slots // 2, num_slots - 1})


@pytest.mark.parametrize("kv_dtype", KV_DTYPES)
@pytest.mark.parametrize("layout", LAYOUTS)
def test_pools_and_views(pretrained, layout: str, kv_dtype: str) -> None:
    """Every full-attention layer owns one indexer page per block in its K/V's
    slot -- bf16 ``[k | z]`` rows, then fp8 region summaries -- whatever the K/V
    dtype; the sliding layers own none. Each view starts where the pools hold its
    pages, the K/V views where the attention op addresses them, and no two views
    overlap."""
    heads = _heads_per_rank(layout)
    data_page, scale_page = _kv_page_bytes(kv_dtype, heads)
    _, binding_dtype = KV_DTYPES[kv_dtype]
    cm = _build(_model_config(pretrained, layout, kv_dtype), _small_cache_config())
    try:
        assert cm.dtype == binding_dtype
        assert cm.get_disagg_role_mapper_kinds()[Role.INDEX_KEY] == MapperKind.REPLICATED

        slot_counts = set()
        views = []
        for layer in range(DECODER_LAYERS):
            local = cm.layer_offsets[layer]
            kv = cm.get_kv_slot_buffers(layer, kv_layout="HND")
            if binding_dtype == DataType.NVFP4:
                assert kv.dtype == torch.uint8 and kv.shape[1:] == (
                    2,
                    heads,
                    32,
                    POOL_HEAD_DIM // 2,
                )
                scales = cm.get_kv_block_scale_slot_buffers(layer, kv_layout="HND")
                assert scales.dtype == torch.uint8
                assert scales.shape == (kv.shape[0], 2, heads, 32, POOL_HEAD_DIM // 16)
            else:
                torch_dtype = (
                    torch.bfloat16 if binding_dtype == DataType.BF16 else torch.float8_e4m3fn
                )
                assert kv.dtype == torch_dtype and kv.shape[1:] == (2, heads, 32, POOL_HEAD_DIM)
                scales = None
            for slot in _probe_slots(kv.shape[0]):
                key, value = _thop_addresses(cm, layer, slot, data_page, plane=0)
                assert (kv[slot, 0].data_ptr(), kv[slot, 1].data_ptr()) == (key, value), layer
                if scales is not None:
                    key, value = _thop_addresses(cm, layer, slot, scale_page, plane=1)
                    assert (scales[slot, 0].data_ptr(), scales[slot, 1].data_ptr()) == (key, value)
            views += [kv[:, 0], kv[:, 1]]
            if scales is not None:
                views += [scales[:, 0], scales[:, 1]]

            kz = cm.get_index_kz_buffer(layer)
            summary = cm.get_index_summary_buffer(layer)
            if layer not in FULL_LAYERS:
                assert kz is None and summary is None
                continue
            assert kz.dtype == torch.bfloat16 and kz.shape[1:] == (32, 2, PROXY_DIM)
            assert summary.dtype == torch.float8_e4m3fn and summary.shape[1:] == (4, PROXY_DIM)
            assert kz.shape[0] == summary.shape[0] == kv.shape[0]
            slot_counts.add(kz.shape[0])
            assert cm.impl.get_page_stride(local, Role.INDEX_KEY) == INDEX_PAGE
            assert cm.impl.get_page_index_scale(local, Role.INDEX_KEY) == len(FULL_LAYERS)
            # Slot s of this layer is page `scale * s + layer_offset` of the indexer
            # pool, [k | z] first and the summaries after them.
            first = cm.layer_offsets[FULL_LAYERS[0]]
            pool_base = cm.impl.get_mem_pool_base_address(
                first, Role.INDEX_KEY, PageIndexMode.SHARED
            )
            pool_base -= (
                int(cm.impl.get_page_index_converter(first, Role.INDEX_KEY).layer_offset)
                * INDEX_PAGE
            )
            layer_offset = int(cm.impl.get_page_index_converter(local, Role.INDEX_KEY).layer_offset)
            for slot in _probe_slots(kz.shape[0]):
                page = pool_base + (len(FULL_LAYERS) * slot + layer_offset) * INDEX_PAGE
                assert kz[slot].data_ptr() == page, (layer, slot)
                assert summary[slot].data_ptr() == page + KZ_PAGE, (layer, slot)
            views += [kz, summary]
        assert len(slot_counts) == 1

        # One slot written through every view reads back unchanged through each.
        raws = [view[view.shape[0] - 1].view(torch.uint8) for view in views]
        for i, raw in enumerate(raws):
            raw.fill_(i % 251 + 1)
        torch.cuda.synchronize()
        for i, raw in enumerate(raws):
            assert bool((raw == i % 251 + 1).all()), f"view {i} was overwritten by another view"
    finally:
        cm.shutdown()


# ----------------------------------------------------------------------------
# (3) block reuse across both kinds
# ----------------------------------------------------------------------------

_HASH_MOD = 2_147_483_647


def _prefix_hashes(tokens: list[int]) -> torch.Tensor:
    """One int per token, a function of the token and everything before it: the
    only thing a forward's per-token output can depend on."""
    value, hashes = 0, []
    for token in tokens:
        value = (value * 1_000_003 + token + 1) % _HASH_MOD
        hashes.append(value)
    return torch.tensor(hashes, dtype=torch.int64, device="cuda")


def _row_bytes(hashes: torch.Tensor, salt: int, nbytes: int) -> torch.Tensor:
    """uint8 ``[tokens, nbytes]``: one token's row of a K/V (or block-scale) page."""
    lanes = torch.arange(nbytes, device=hashes.device, dtype=torch.int64)
    return ((hashes[:, None] * (2 * salt + 1) + lanes[None, :] * 29 + salt) % 251).to(torch.uint8)


def _kz_rows(hashes: torch.Tensor, layer: int) -> torch.Tensor:
    """bf16 ``[tokens, 2, PROXY_DIM]``: each token's indexer k and z."""
    lanes = torch.arange(PROXY_DIM, device=hashes.device, dtype=torch.int64)
    planes = torch.arange(2, device=hashes.device, dtype=torch.int64) * 7919
    code = (hashes[:, None, None] * (layer + 1) + planes[None, :, None] + lanes * 13) % 1009
    return (code.to(torch.float32) / 1009 * 4 - 2).to(torch.bfloat16)


def _compress(kz: torch.Tensor) -> torch.Tensor:
    """fp8 ``[regions, PROXY_DIM]`` from bf16 ``[regions, REGION, 2, PROXY_DIM]``: the
    reference's region compression -- a per-dimension softmax over the region's z with
    one shift shared by the whole region, rounded through bf16 to e4m3."""
    k, z = kz[:, :, 0].float(), kz[:, :, 1].float()
    weights = torch.exp(z - z.amax(dim=(1, 2), keepdim=True))
    summary = (weights * k).sum(1) / weights.sum(1)
    return summary.to(torch.bfloat16).to(torch.float8_e4m3fn)


class _Forward:
    """Writes what a prefill writes into the manager's pools, and reads it back."""

    def __init__(self, cm: Step5CacheManager) -> None:
        self.cm = cm
        self.tpb = cm.tokens_per_block
        self.kv = {
            layer: self._rows(cm.get_kv_slot_buffers(layer)) for layer in range(DECODER_LAYERS)
        }
        self.scales = {}
        if cm.dtype == DataType.NVFP4:
            self.scales = {
                layer: self._rows(cm.get_kv_block_scale_slot_buffers(layer))
                for layer in range(DECODER_LAYERS)
            }
        self.kz = {layer: cm.get_index_kz_buffer(layer) for layer in FULL_LAYERS}
        self.summary = {layer: cm.get_index_summary_buffer(layer) for layer in FULL_LAYERS}

    @staticmethod
    def _rows(view: torch.Tensor) -> torch.Tensor:
        """``[slots, 2, tokens_per_block, row_bytes]`` uint8 over an NHD K/V view."""
        return view.view(torch.uint8).flatten(-2)

    def _slots(self, request_id: int, layer: int, positions: torch.Tensor) -> torch.Tensor:
        table = torch.tensor(self.cm.get_slot_table([request_id], layer)[0], dtype=torch.int64)
        slots = table[(positions // self.tpb).cpu()]
        assert bool((slots >= 0).all()), f"layer {layer} holds no page for some of {positions}"
        return slots.to(positions.device)

    def check_device_block_table(self, request_id: int) -> None:
        """The engine's device block offsets give the forward the same slots: the
        K lane holds ``slot * scale`` (``get_block_offset_slot_scale``)."""
        cm = self.cm
        offsets = torch.zeros(
            cm.num_pools, 1, 2, cm.max_blocks_per_seq, dtype=torch.int32, device="cuda"
        )
        cm.copy_batch_block_offsets(offsets, [request_id], 1, 1, 1)
        torch.cuda.synchronize()
        for layer in (FULL_LAYERS[0], 0):
            pool, scale = cm.get_block_offset_slot_scale(layer)
            host = torch.tensor(cm.get_slot_table([request_id], layer)[0], dtype=torch.int64)
            live = host >= 0
            device = offsets[pool, 0, 0, : host.numel()].cpu().to(torch.int64)
            assert torch.equal(device[live] // scale, host[live]), layer
            assert torch.equal(device[live] % scale, torch.zeros_like(host[live])), layer

    def _layer_rows(self, layer: int, hashes: torch.Tensor):
        """(K/V rows, block-scale rows or None) a cold prefill writes for these tokens."""
        width = self.kv[layer].shape[-1]
        data = [_row_bytes(hashes, 4 * layer + plane, width) for plane in (0, 1)]
        scales = None
        if self.scales:
            width = self.scales[layer].shape[-1]
            scales = [_row_bytes(hashes, 4 * layer + 2 + plane, width) for plane in (0, 1)]
        return data, scales

    def write(self, request_id: int, hashes: torch.Tensor, start: int, end: int) -> None:
        positions = torch.arange(start, end, device="cuda")
        within = positions % self.tpb
        for layer in range(DECODER_LAYERS):
            slots = self._slots(request_id, layer, positions)
            data, scales = self._layer_rows(layer, hashes[start:end])
            for plane in (0, 1):
                self.kv[layer][slots, plane, within] = data[plane]
                if scales is not None:
                    self.scales[layer][slots, plane, within] = scales[plane]
            if layer not in FULL_LAYERS:
                continue
            self.kz[layer][slots, within] = _kz_rows(hashes[start:end], layer)
            # A region's summary is written by the step that writes its last token,
            # from the [k | z] rows in the pool -- reused ones included.
            regions = [
                r
                for r in range(start // REGION, end // REGION + 1)
                if start <= REGION * r + REGION - 1 < end
            ]
            if not regions:
                continue
            members = (
                torch.tensor(regions, device="cuda")[:, None] * REGION
                + torch.arange(REGION, device="cuda")[None, :]
            )
            member_slots = self._slots(request_id, layer, members.flatten()).view(members.shape)
            rows = self.kz[layer][member_slots, members % self.tpb]
            self.summary[layer][member_slots[:, 0], (members[:, 0] % self.tpb) // REGION] = (
                _compress(rows)
            )

    def check(self, request_id: int, hashes: torch.Tensor, end: int, label: str) -> None:
        """Every page this request can read holds what a cold prefill of its first
        ``end`` tokens writes: all of them on the full-attention layers (K/V, ``[k | z]``
        and the summary of every completed region), the window before ``end`` on the
        sliding ones."""
        for layer in range(DECODER_LAYERS):
            first = 0 if layer in FULL_LAYERS else max(0, end - SLIDING_WINDOW)
            positions = torch.arange(first, end, device="cuda")
            slots = self._slots(request_id, layer, positions)
            within = positions % self.tpb
            data, scales = self._layer_rows(layer, hashes[first:end])
            for plane in (0, 1):
                assert torch.equal(self.kv[layer][slots, plane, within], data[plane]), (
                    f"{label}: layer {layer} K/V plane {plane}"
                )
                if scales is not None:
                    assert torch.equal(self.scales[layer][slots, plane, within], scales[plane]), (
                        f"{label}: layer {layer} block scales plane {plane}"
                    )
            if layer not in FULL_LAYERS:
                continue
            expected_kz = _kz_rows(hashes[:end], layer)
            assert torch.equal(self.kz[layer][slots, within], expected_kz), (
                f"{label}: layer {layer} [k|z]"
            )
            complete = end // REGION
            if complete == 0:
                continue
            region_starts = positions[: complete * REGION : REGION]
            got = self.summary[layer][
                slots[: complete * REGION : REGION], within[: complete * REGION : REGION] // REGION
            ]
            expected = _compress(
                expected_kz[: complete * REGION].view(complete, REGION, 2, PROXY_DIM)
            )
            assert torch.equal(got.view(torch.uint8), expected.view(torch.uint8)), (
                f"{label}: layer {layer} summaries of regions {region_starts.numel()}"
            )


def _request(request_id: int, tokens: list[int]) -> LlmRequest:
    return LlmRequest(
        request_id=request_id,
        max_new_tokens=8,
        input_tokens=list(tokens),
        sampling_config=SamplingConfig(1),
        is_streaming=False,
    )


def _schedule(cm: Step5CacheManager, request: LlmRequest, chunk: int) -> ScheduledRequests:
    """One scheduling pass and the batch it hands the forward, as the executor runs them."""
    assert cm.prepare_context(request)
    request.context_chunk_size = min(chunk, request.context_remaining_length)
    assert cm.resize_context(request, request.context_chunk_size)
    batch = ScheduledRequests()
    batch.append_context_request(request)
    cm.prepare_resources(batch)
    cm.commit_scheduled_kv_cache_stats(batch)
    return batch


def _prefill(cm, forward: _Forward, request: LlmRequest, hashes, chunk: int, on_resume=None):
    """Prefill in ``chunk``-token steps; ``on_resume`` sees the cache before the first write."""
    first = True
    while request.context_remaining_length > 0:
        batch = _schedule(cm, request, chunk)
        forward.check_device_block_table(request.py_request_id)
        if first and on_resume is not None:
            on_resume(request)
        first = False
        start = request.context_current_position
        forward.write(request.py_request_id, hashes, start, start + request.context_chunk_size)
        request.move_to_next_context_chunk()
        cm.update_context_resources(batch)


def _finish(cm: Step5CacheManager, request: LlmRequest) -> None:
    request.state = LlmRequestState.GENERATION_COMPLETE
    cm.free_resources(request)


def _reused_blocks(cm: Step5CacheManager) -> dict[int, tuple[int, int]]:
    """``{window: (iterFullReusedBlocks, iterPartialReusedBlocks)}`` since the last call."""
    report = cm.get_iteration_stats()
    return {
        window: (stats.iter_full_reused_blocks, stats.iter_partial_reused_blocks)
        for window, stats in report.by_window_size.items()
    }


@pytest.mark.parametrize("kv_dtype", KV_DTYPES)
@pytest.mark.parametrize("layout", LAYOUTS)
def test_block_reuse_across_both_kinds(pretrained, layout: str, kv_dtype: str) -> None:
    """A second request sharing a prefix reuses its blocks in both windows, and
    what it resumes on -- mid-region, 803 tokens in, so inside the window of the
    sliding layers -- is what a cold prefill computes; so is everything after."""
    cm = _build(_model_config(pretrained, layout, kv_dtype), _small_cache_config())
    try:
        assert cm.enable_partial_reuse
        forward = _Forward(cm)
        rng = random.Random(0)
        prompt_a = [rng.randrange(100, 120_000) for _ in range(1000)]
        shared = 803
        prompt_b = prompt_a[:shared] + [rng.randrange(100, 120_000) for _ in range(397)]
        prompt_c = prompt_a + [rng.randrange(100, 120_000) for _ in range(200)]

        # A: a cold prefill in the chunked variant's 512-token chunks.
        a, hashes_a = _request(1, prompt_a), _prefix_hashes(prompt_a)
        _prefill(cm, forward, a, hashes_a, CHUNK)
        forward.check(1, hashes_a, len(prompt_a), "A after prefill")
        _finish(cm, a)
        assert _reused_blocks(cm) == {SLIDING_WINDOW: (0, 0), MAX_SEQ_LEN: (0, 0)}

        # B shares A's first 803 tokens: a partial block, mid-region, past the window.
        b, hashes_b = _request(2, prompt_b), _prefix_hashes(prompt_b)
        resumed = {}

        def b_resumes(request: LlmRequest) -> None:
            resumed["at"] = request.context_current_position
            forward.check(2, hashes_b, request.context_current_position, "B on resume")

        _prefill(cm, forward, b, hashes_b, MAX_NUM_TOKENS, on_resume=b_resumes)
        assert resumed["at"] == shared
        reused = _reused_blocks(cm)
        # Full-attention layers: blocks 0-24 whole and block 25 in part; sliding
        # layers: only the blocks of the window before token 803.
        assert reused[MAX_SEQ_LEN] == (shared // TOKENS_PER_BLOCK, 1), reused
        first_window_block = (shared - SLIDING_WINDOW) // TOKENS_PER_BLOCK
        assert reused[SLIDING_WINDOW] == (shared // TOKENS_PER_BLOCK - first_window_block, 1), (
            reused
        )
        forward.check(2, hashes_b, len(prompt_b), "B after prefill")
        _finish(cm, b)

        # C extends all of A: it resumes at A's end, mid-block on a region boundary.
        c, hashes_c = _request(3, prompt_c), _prefix_hashes(prompt_c)

        def c_resumes(request: LlmRequest) -> None:
            resumed["at"] = request.context_current_position
            forward.check(3, hashes_c, request.context_current_position, "C on resume")

        _prefill(cm, forward, c, hashes_c, CHUNK, on_resume=c_resumes)
        assert resumed["at"] == len(prompt_a)
        reused = _reused_blocks(cm)
        assert all(full > 0 for full, _ in reused.values()), reused
        forward.check(3, hashes_c, len(prompt_c), "C after prefill")
        _finish(cm, c)
    finally:
        cm.shutdown()


# ----------------------------------------------------------------------------
# (5) sizing
# ----------------------------------------------------------------------------


# K/V bytes per token of one layer on one rank, at heads padded to 256 (K and V,
# NVFP4 with its e4m3 block scales).
KV_BYTES_PER_TOKEN = {
    ("attn_dp4", "bf16"): 4096,
    ("attn_dp4", "fp8"): 2048,
    ("attn_dp4", "nvfp4"): 1152,
    ("attn_tp4", "bf16"): 1024,
    ("attn_tp4", "fp8"): 512,
    ("attn_tp4", "nvfp4"): 288,
}


# Slots the sliding pool needs at the engine's limits: each of max_batch_size
# requests holds its 512-token window (16 blocks, one more at an unaligned
# offset), and one step's max_num_tokens context tokens add their own blocks.
WINDOW_BLOCKS = 17
SLIDING_NEED = MAX_BATCH_SIZE * WINDOW_BLOCKS + MAX_NUM_TOKENS // TOKENS_PER_BLOCK


def _grains(slot_sizes: list[int], slots: int, grain: int) -> int:
    """Allocator grains ``slots`` slots take: each pool rounds up on its own."""
    return sum(-(-slots * size // grain) for size in slot_sizes)


def _slots_in(slot_sizes: list[int], grains: int, grain: int) -> int:
    """The most slots ``grains`` grains hold."""
    slots = min(grains * grain // size for size in slot_sizes)
    while _grains(slot_sizes, slots + 1, grain) <= grains:
        slots += 1
    while _grains(slot_sizes, slots, grain) > grains:
        slots -= 1
    return slots


def test_sliding_need_is_derived_from_the_limits() -> None:
    """The sliding pool's need is a function of max_batch_size, the window,
    max_num_tokens and tokens_per_block, and of nothing the gates served."""
    assert step5_window_blocks(SLIDING_WINDOW, TOKENS_PER_BLOCK, 1) == WINDOW_BLOCKS
    assert step5_window_blocks(SLIDING_WINDOW, 64, 1) == 9
    assert step5_window_blocks(500, TOKENS_PER_BLOCK, 1) == 17
    need = step5_sliding_pool_blocks(
        max_batch_size=MAX_BATCH_SIZE,
        window=SLIDING_WINDOW,
        chunk_tokens=MAX_NUM_TOKENS,
        tokens_per_block=TOKENS_PER_BLOCK,
        headroom=1,
    )
    assert need == SLIDING_NEED == 2432
    # A prefill chunk behind a full window, beside max_batch_size - 1 decoders.
    behind_window = step5_window_blocks(SLIDING_WINDOW + MAX_NUM_TOKENS, TOKENS_PER_BLOCK, 1)
    assert (MAX_BATCH_SIZE - 1) * WINDOW_BLOCKS + behind_window == need
    assert (
        step5_sliding_pool_blocks(
            max_batch_size=8,
            window=SLIDING_WINDOW,
            chunk_tokens=CHUNK,
            tokens_per_block=TOKENS_PER_BLOCK,
            headroom=1,
        )
        == 8 * WINDOW_BLOCKS + 16
    )


def _weights_per_rank(checkpoint: Path, world_size: int) -> int:
    """The checkpoint's bytes over the ranks: what loading leaves each GPU short of."""
    return sum(f.stat().st_size for f in checkpoint.glob("*.safetensors")) // world_size


@pytest.mark.parametrize("kv_dtype", KV_DTYPES)
@pytest.mark.parametrize("layout", LAYOUTS)
def test_sizing_charges_every_pool(checkpoint, pretrained, layout: str, kv_dtype: str) -> None:
    """At ``kv_cache_fraction`` 0.7, ``max_batch_size`` 128, ``max_seq_len`` 9216 and
    ``max_num_tokens`` 8192: the static cost the creator splits the budget with
    charges the indexer state of every full-attention layer; the pools V2 allocates
    hold it in the K/V's slot and fit the budget; the split is the one the limits
    derive -- the sliding pool holds the 2432 slots of 128 windows (17 blocks each)
    plus one 8192-token step (256 blocks), rounded to the allocator's grains, and the
    full-attention pool every other grain; and the quota-to-token conversion the
    scheduler admits by promises the full-attention pool's tokens.

    Measured on one GB300 with a 128.80 GiB budget (0.7 of the free memory less a
    quarter of the checkpoint), the sliding pool holding 2433 slots (19.01 per
    request) everywhere:

    ============  =======  ======================  ==========================
    layout        KV       full-attention tokens   ratio to 128 x 9216 tokens
    ============  =======  ======================  ==========================
    attn_tp4      bf16     2775456                 2.353
    attn_tp4      fp8      3758336                 3.186
    attn_tp4      nvfp4    4422240                 3.749
    attn_dp4      bf16     981248                  0.832
    attn_dp4      fp8      1782976                 1.511
    attn_dp4      nvfp4    2600064                 2.204
    ============  =======  ======================  ==========================

    At attention DP with a bf16 KV cache the budget binds: the full-attention pool
    gets 108.29 GiB of the 130.18 GiB that 128 requests of 9216 tokens need there,
    beside the 20.49 GiB sliding pool.
    """
    model_config = _model_config(pretrained, layout, kv_dtype)
    mapping = model_config.mapping
    heads = _heads_per_rank(layout)
    data_page, scale_page = _kv_page_bytes(kv_dtype, heads)
    kv_per_token = 2 * (data_page + scale_page) // TOKENS_PER_BLOCK
    assert kv_per_token == KV_BYTES_PER_TOKEN[layout, kv_dtype]
    indexer_per_token = INDEX_PAGE // TOKENS_PER_BLOCK
    num_full, num_sliding = len(FULL_LAYERS), DECODER_LAYERS - len(FULL_LAYERS)

    windows = _util._derive_v2_layer_type_attention_windows(
        KvCacheConfig(), Step5CacheManager, model_config, MAX_SEQ_LEN
    )
    static_kwargs = dict(
        tokens_per_block=TOKENS_PER_BLOCK,
        kv_cache_config=KvCacheConfig(enable_block_reuse=True, max_attention_window=windows),
        max_seq_len=MAX_SEQ_LEN,
        max_batch_size=MAX_BATCH_SIZE,
        max_num_tokens=MAX_NUM_TOKENS,
    )
    slope, fixed = Step5CacheManager.get_cache_size_per_token(
        model_config, mapping, **static_kwargs
    )
    text_model_config = Step5CacheManager._text_model_config(model_config)
    base_slope, base_fixed = KVCacheManagerV2.get_cache_size_per_token(
        text_model_config, mapping, **static_kwargs
    )
    # Full-attention layers grow with the sequence: K/V plus the indexer state.
    assert slope == num_full * (kv_per_token + indexer_per_token)
    assert slope - base_slope == num_full * indexer_per_token
    # Sliding layers are a per-request intercept: their window for every request.
    assert fixed == base_fixed > 0

    free, total = torch.cuda.mem_get_info()
    peak = (total - free) + _weights_per_rank(checkpoint, mapping.world_size)
    budget = int((total - peak) * KV_CACHE_FRACTION)
    kv_cache_config = KvCacheConfig(
        free_gpu_memory_fraction=KV_CACHE_FRACTION,
        max_gpu_total_bytes=budget,
        enable_block_reuse=True,
        host_cache_size=0,
    )
    cm = _build(model_config, kv_cache_config)
    try:
        assert cm.get_cache_bytes_per_token() == (
            num_full * (kv_per_token + indexer_per_token) + num_sliding * kv_per_token
        )
        quota = cm.impl.get_quota(GPU_LEVEL)
        assert cm.get_kv_cache_stats().allocated_bytes == quota

        groups = {window: group for group, window, _ in cm._stats_life_cycle_metadata().values()}
        stats = cm._get_storage_statistics(GPU_LEVEL)
        nvfp4 = KV_DTYPES[kv_dtype][1] == DataType.NVFP4
        full_slot = [num_full * INDEX_PAGE, 2 * num_full * data_page]
        sliding_slot = [2 * num_sliding * data_page]
        if nvfp4:
            full_slot.append(2 * num_full * scale_page)
            sliding_slot.append(2 * num_sliding * scale_page)
        full_stats, sliding_stats = stats[groups[MAX_SEQ_LEN]], stats[groups[SLIDING_WINDOW]]
        assert sorted(cm._stats_slot_sizes(full_stats)) == sorted(full_slot)
        assert sorted(cm._stats_slot_sizes(sliding_stats)) == sorted(sliding_slot)
        # Every pool is carved from the budget: each is a whole number of the GPU
        # allocator's chunks, so the total is the budget to within a chunk per pool.
        num_pools = sum(len(cm._stats_slot_sizes(stat)) for stat in stats)
        assert abs(quota - budget) <= num_pools * _gpu_alloc_granularity(budget)
        pool_bytes = sum(stat.total * sum(cm._stats_slot_sizes(stat)) for stat in stats)
        assert pool_bytes <= quota

        # The quota/token conversion the scheduler and the cross-rank capacity sync
        # use charges a full-attention token its K/V and its indexer state.
        grown = cm._get_quota_from_max_tokens(MAX_NUM_TOKENS + 1000)
        assert grown - cm._get_quota_from_max_tokens(MAX_NUM_TOKENS) == 1000 * slope

        # The split comes from the engine's limits, not V2's modelled average
        # sequence: the sliding pool holds max_batch_size windows plus one step's
        # max_num_tokens context tokens, in the allocator's grains, and the
        # full-attention pool every other grain of the quota.
        split = cm.step5_pool_split
        assert split is not None
        assert split.sliding_blocks == {SLIDING_WINDOW: SLIDING_NEED}
        grain = _gpu_alloc_granularity(budget)
        assert split.grain_bytes == grain and quota % grain == 0
        sliding_grains = _grains(sliding_slot, SLIDING_NEED, grain)
        assert split.sliding_grains == sliding_grains
        # The full-attention pool gets every grain of the quota the sliding pool
        # leaves (V2 allocates the grains its pools fill, at most one short per pool).
        assert 0 <= split.sliding_grains + split.full_grains - quota // grain < num_pools
        assert sliding_stats.total == _slots_in(sliding_slot, sliding_grains, grain)
        assert SLIDING_NEED <= sliding_stats.total
        assert sliding_stats.total < SLIDING_NEED + MAX_BATCH_SIZE
        assert full_stats.total == _slots_in(full_slot, split.full_grains, grain)

        # Full-attention capacity: every token of the pool is a token the
        # scheduler's quota-to-token conversion promises, to within the grains
        # the two pools round to.
        full_tokens = full_stats.total * TOKENS_PER_BLOCK
        promised = cm._get_max_tokens_from_quota(quota)
        rounding = num_pools * grain // sum(full_slot) * TOKENS_PER_BLOCK + TOKENS_PER_BLOCK
        assert abs(full_tokens - promised) <= rounding, (full_tokens, promised, rounding)
        envelope = MAX_BATCH_SIZE * MAX_SEQ_LEN
        # The bytes the envelope needs beside the sliding pool, against the
        # full-attention pool's.
        full_need = MAX_BATCH_SIZE * -(-MAX_SEQ_LEN // TOKENS_PER_BLOCK) * sum(full_slot)
        print(
            f"\n[{layout} {kv_dtype}] budget {budget / 2**30:.2f} GiB, quota {quota / 2**30:.2f} GiB, "
            f"pools {pool_bytes / 2**30:.2f} GiB; sliding slots {sliding_stats.total} "
            f"({sliding_stats.total / MAX_BATCH_SIZE:.2f} per request at batch {MAX_BATCH_SIZE}, "
            f"need {SLIDING_NEED}), {sliding_stats.total * sum(sliding_slot) / 2**30:.2f} GiB; "
            f"full-attention slots {full_stats.total} ({full_tokens} tokens, "
            f"{full_tokens / envelope:.3f}x of {MAX_BATCH_SIZE}x{MAX_SEQ_LEN}), "
            f"{full_stats.total * sum(full_slot) / 2**30:.2f} GiB of the "
            f"{full_need / 2**30:.2f} GiB the envelope needs; quota->tokens {promised:.0f}; "
            f"static slope {slope} B/token, intercept {fixed} B"
        )
    finally:
        cm.shutdown()


def _sliding_slots(cm: Step5CacheManager) -> int:
    groups = {window: group for group, window, _ in cm._stats_life_cycle_metadata().values()}
    return cm._get_storage_statistics(GPU_LEVEL)[groups[SLIDING_WINDOW]].total


@pytest.mark.parametrize("layout", LAYOUTS)
def test_configured_split_wins(pretrained, layout: str) -> None:
    """Without a ``pool_ratio`` or ``avg_seq_len`` the manager pins the split the
    limits derive; either one, set in the KV cache config, hands the split back
    to V2."""
    model_config = _model_config(pretrained, layout, "bf16")
    data_page, _ = _kv_page_bytes("bf16", _heads_per_rank(layout))
    sliding_slot = [2 * (DECODER_LAYERS - len(FULL_LAYERS)) * data_page]

    cm = _build(model_config, _small_cache_config())
    try:
        split = cm.step5_pool_split
        assert split is not None and split.sliding_blocks == {SLIDING_WINDOW: SLIDING_NEED}
        grain = split.grain_bytes
        derived = _sliding_slots(cm)
        assert derived == _slots_in(sliding_slot, _grains(sliding_slot, SLIDING_NEED, grain), grain)
    finally:
        cm.shutdown()

    cm = _build(model_config, _small_cache_config(avg_seq_len=2048))
    try:
        assert cm.step5_pool_split is None
        assert _sliding_slots(cm) != derived
    finally:
        cm.shutdown()

    cm = _build(model_config, _small_cache_config(pool_ratio=[0.5, 0.5]))
    try:
        assert cm.step5_pool_split is None
        quota = cm.impl.get_quota(GPU_LEVEL)
        sliding_bytes = _sliding_slots(cm) * sum(sliding_slot)
        assert abs(sliding_bytes - quota // 2) <= _gpu_alloc_granularity(quota) + sum(sliding_slot)
    finally:
        cm.shutdown()
