# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Step-5's KV cache across the disaggregated hop, on real managers and real transceivers.

A context instance and a generation instance each build ``Step5CacheManager`` from the
checkpoint's own config through the engine's two functions (as ``test_modeling_v2_step5_
cache_manager.py`` does), with ``is_disagg`` set, and one ``KvCacheTransceiverV2`` per rank
(the Python NIXL transceiver, threads standing in for ranks, from the shared
``tests/unittest/disaggregated/kv_transfer_harness.py``). The context side prefills each
request in chunks through the manager's own scheduling calls, so the sliding layers drop the
blocks that leave their 512-token window exactly as in serving, then writes into every page the
request holds what a prefill writes there: a pattern of the layer, the role, the global KV head
and the request's tokens up to the block's end. The generation side admits the request as the
engine does (``prepare_disagg_gen_init``) and receives it.

What the generation forward then reads must be exactly what the context forward wrote, in every
pool ``Step5CacheManager`` allocates, through the views the target reads:

* the K/V of the 69 sliding layers, for the blocks of the window the first decode step attends
  to (the ones before it are not held on either side);
* the K/V of the 23 full-attention layers, for every prompt block;
* the sparse indexer's page of every full-attention block: the bf16 ``[k | z]`` rows and the fp8
  region summaries;
* with an NVFP4 cache, the e4m3 block scales of both kinds of layer;

with the heads zero-padded to 256 in the pools. The prompts run from 131 to 6200 tokens, the
lengths the equivalence witness asks for (6200 is past the 4104 tokens where the indexer starts
dropping regions), and every (context rank, generation rank) pair of the attention-DP layout carries one.
One more request shares a 1500-token prefix with the one before it and crosses the same ranks:
the context instance resumes on the reused prefix, mid-block, and the generation instance, which
committed what it received for the earlier request, admits it with whatever prefix V2 restores
there (the receiver then skips those blocks); either way it must read what the context wrote.

The gated layout is DEP2 on both sides; DEP4 on both sides, TEP4 into DEP4 (a context instance
whose ranks each hold one KV head) and DEP2 into DEP4 keep the transfer free of anything
specific to DEP2.

Needs the checkpoint's ``config.json`` (``STEP5_PREVIEW_CHECKPOINT``, else
``$LLM_MODELS_ROOT/Step-5-Preview-NVFP4``) and a GPU; one device holds both instances.
"""

from __future__ import annotations

import importlib.util
import os
import random
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pytest
import torch
from utils.llm_data import llm_models_root

from tensorrt_llm import DisaggregatedParams
from tensorrt_llm._torch.attention.backends.sparse.step5 import Step5CacheManager
from tensorrt_llm._torch.disaggregation.resource.kv_extractor import build_page_table_from_manager
from tensorrt_llm._torch.disaggregation.resource.page import (
    AttentionLayerGroup,
    MapperKind,
    PhysicalPool,
)
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.pyexecutor import _util, config_utils
from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import Role
from tensorrt_llm._torch.pyexecutor.llm_request import (
    LlmRequest,
    LlmRequestState,
    LlmRequestType,
    SamplingConfig,
)
from tensorrt_llm._torch.pyexecutor.scheduler import ScheduledRequests
from tensorrt_llm._utils import TensorWrapper, convert_to_torch_tensor
from tensorrt_llm.bindings import DataType
from tensorrt_llm.llmapi.llm_args import CacheTransceiverConfig, KvCacheConfig
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models.modeling_utils import QuantConfig
from tensorrt_llm.quantization.mode import QuantAlgo

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="the managers allocate their pools on the GPU"
)

# What the checkpoint's config.json declares, restated so the test checks the
# derivation rather than repeating it.
DECODER_LAYERS = 92
SLIDING_WINDOW = 512
FULL_LAYERS = tuple(range(3, DECODER_LAYERS, 4))
SLIDING_LAYERS = tuple(layer for layer in range(DECODER_LAYERS) if layer not in FULL_LAYERS)
KV_HEADS = 4
POOL_HEAD_DIM = 256
PROXY_DIM, REGION = 256, 8

# The engine's block size, and limits small enough for four instances' worth
# of managers on one device; the windows and the index geometry do not depend
# on them.
TOKENS_PER_BLOCK = 32
MAX_SEQ_LEN = 8192
MAX_BATCH_SIZE = 4
MAX_NUM_TOKENS = 2048
MAX_TOKENS = 32 * 1024

# The prompts of the equivalence witness, by length; 6200 is past the 4104
# tokens at which top-512 over 8-token regions starts dropping regions.
PROMPT_LENGTHS = (131, 700, 2050, 6200)
# The request that shares a prefix with prompt REUSE_DONOR on both sides: the
# prefix ends mid-block and mid-region, past the window.
REUSE_DONOR = 2
REUSE_INDEX = REUSE_DONOR + 1
REUSE_SHARED = 1500
REUSE_LENGTH = 2100

TRANSCEIVER = CacheTransceiverConfig(
    backend="NIXL", transceiver_runtime="PYTHON", max_tokens_in_buffer=MAX_SEQ_LEN
)

KV_DTYPES = {
    "bf16": (None, DataType.BF16),
    "fp8": (QuantAlgo.FP8, DataType.FP8),
    "nvfp4": (QuantAlgo.NVFP4, DataType.NVFP4),
}

# Role tags of the pattern each page holds.
_K, _V, _K_SCALE, _V_SCALE, _KZ, _SUMMARY = range(6)


@dataclass(frozen=True)
class Layout:
    """One instance's attention layout: ``tp_size`` ranks, attention DP or TP."""

    tp_size: int
    attention_dp: bool

    def mapping(self, rank: int) -> Mapping:
        return Mapping(
            world_size=self.tp_size,
            rank=rank,
            tp_size=self.tp_size,
            pp_size=1,
            enable_attention_dp=self.attention_dp,
        )

    @property
    def heads_per_rank(self) -> int:
        return KV_HEADS if self.attention_dp else KV_HEADS // self.tp_size

    def first_head(self, rank: int) -> int:
        """Global id of the first KV head ``rank`` holds."""
        return 0 if self.attention_dp else rank * self.heads_per_rank

    def ranks_for(self, dp_rank: Optional[int]) -> list[int]:
        """The ranks that hold a request served by ``dp_rank``: one under DP, all under TP."""
        return [dp_rank] if self.attention_dp else list(range(self.tp_size))


DEP2, DEP4, TEP4 = Layout(2, True), Layout(4, True), Layout(4, False)
TOPOLOGIES = {
    "dep2_to_dep2": (DEP2, DEP2),
    "dep4_to_dep4": (DEP4, DEP4),
    "tep4_to_dep4": (TEP4, DEP4),
    "dep2_to_dep4": (DEP2, DEP4),
}
CASES = [
    ("dep2_to_dep2", "bf16"),
    ("dep2_to_dep2", "fp8"),
    ("dep2_to_dep2", "nvfp4"),
    ("dep4_to_dep4", "bf16"),
    ("tep4_to_dep4", "bf16"),
    ("dep2_to_dep4", "bf16"),
]


def _checkpoint_dir() -> Path | None:
    explicit = os.environ.get("STEP5_PREVIEW_CHECKPOINT")
    if explicit:
        return Path(explicit)
    root = llm_models_root()
    return None if root is None else root / "Step-5-Preview-NVFP4"


@pytest.fixture(scope="module")
def pretrained():
    """The config the engine loads: the published composite, as is."""
    path = _checkpoint_dir()
    if path is None or not (path / "config.json").is_file():
        pytest.skip(
            "no Step-5-Preview-NVFP4 checkpoint: set STEP5_PREVIEW_CHECKPOINT or stage "
            "Step-5-Preview-NVFP4 under LLM_MODELS_ROOT"
        )
    return config_utils.load_pretrained_config(str(path))


@pytest.fixture(scope="module")
def harness():
    """The shared threaded NIXL harness: per-rank transceivers in one process."""
    path = Path(__file__).resolve().parents[5] / "disaggregated" / "kv_transfer_harness.py"
    if not path.is_file():
        pytest.skip(f"the shared transfer harness is not checked out at {path}")
    spec = importlib.util.spec_from_file_location("step5_kv_transfer_harness", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _model_config(pretrained, mapping: Mapping, kv_dtype: str) -> ModelConfig:
    kv_algo, _ = KV_DTYPES[kv_dtype]
    return ModelConfig(
        pretrained_config=pretrained,
        mapping=mapping,
        quant_config=QuantConfig(quant_algo=QuantAlgo.NVFP4, kv_cache_quant_algo=kv_algo),
    )


def _cache_config() -> KvCacheConfig:
    return KvCacheConfig(max_tokens=MAX_TOKENS, enable_block_reuse=True, host_cache_size=0)


def _build(pretrained, layout: Layout, rank: int, kv_dtype: str) -> Step5CacheManager:
    """One rank's manager, selected and built through the engine's own two functions."""
    model_config = _model_config(pretrained, layout.mapping(rank), kv_dtype)
    kv_cache_config = _cache_config()
    cls = _util.get_kv_cache_manager_cls(
        model_config, kv_cache_config, is_disagg=True, cache_transceiver_config=TRANSCEIVER
    )
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
        model_config=model_config,
        dtype=torch.bfloat16,
        is_disagg=True,
    )


def _physical_pools(cm: Step5CacheManager) -> list[PhysicalPool]:
    page_table = build_page_table_from_manager(cm)
    return [pool for group in page_table.pool_groups for pool in group.pools]


def _fill_pools(cm: Step5CacheManager, value: Optional[int]) -> None:
    """Every byte of every pool: ``value``, or noise when None, so a page nobody
    wrote never reads as the pattern."""
    seen = set()
    for pool in _physical_pools(cm):
        if pool.base_address in seen:
            continue
        seen.add(pool.base_address)
        extent = (pool.num_slots - 1) * pool.slot_stride_bytes + pool.slot_bytes
        raw = convert_to_torch_tensor(TensorWrapper(pool.base_address, torch.uint8, [extent]))
        if value is None:
            raw.random_()
        else:
            raw.fill_(value)
    torch.cuda.synchronize()


# ----------------------------------------------------------------------------
# What a prefill writes: a pattern of (layer, role, head, the tokens up to the
# block's end), so a page landing at the wrong block, layer, head or role, or a
# block reused from a different prefix, cannot match.
# ----------------------------------------------------------------------------

_PRIME = 251
_HASH_MOD = (1 << 61) - 1


def _block_keys(tokens: list[int]) -> list[int]:
    """A hash of the prompt up to the end of each block (the last one partial)."""
    keys, h = [], 0
    for i, token in enumerate(tokens):
        h = (h * 1_000_003 + token + 1) % _HASH_MOD
        if (i + 1) % TOKENS_PER_BLOCK == 0 or i + 1 == len(tokens):
            keys.append(h)
    return keys


def _pattern(keys: list[int], layer: int, role: int, heads: list[int], nbytes: int) -> torch.Tensor:
    """uint8 ``[blocks, heads, nbytes]``: an affine byte sequence mod 251 whose
    start and step come from the block's key, the layer, the role and the head."""
    key = torch.tensor(keys, dtype=torch.int64, device="cuda")[:, None, None]
    head = torch.tensor(heads, dtype=torch.int64, device="cuda")[None, :, None]
    salt = layer * 1009 + role * 101 + head * 13
    start = (key + salt) % _PRIME
    step = 1 + (key // _PRIME + salt) % (_PRIME - 1)
    offsets = torch.arange(nbytes, dtype=torch.int64, device="cuda")[None, None, :]
    return ((start + step * offsets) % _PRIME).to(torch.uint8)


def _kv_pages(cm: Step5CacheManager, layer: int) -> torch.Tensor:
    """uint8 ``[slots, 2, heads, page bytes per head]`` of a layer's K/V, HND as
    the attention op writes it."""
    kv = cm.get_kv_slot_buffers(layer, kv_layout="HND").view(torch.uint8)
    return kv.view(*kv.shape[:-2], -1)


def _scale_pages(cm: Step5CacheManager, layer: int) -> Optional[torch.Tensor]:
    if cm.dtype != DataType.NVFP4:
        return None
    scales = cm.get_kv_block_scale_slot_buffers(layer, kv_layout="HND")
    return scales.view(*scales.shape[:-2], -1)


def _index_pages(cm: Step5CacheManager, layer: int) -> Optional[tuple[torch.Tensor, torch.Tensor]]:
    """uint8 ``[slots, bytes]`` of a full-attention layer's ``[k | z]`` rows and summaries."""
    kz = cm.get_index_kz_buffer(layer)
    if kz is None:
        return None
    summary = cm.get_index_summary_buffer(layer).view(torch.uint8)
    kz = kz.view(torch.uint8)
    # .view, not .flatten: a copy would take the writes below away from the pool.
    return kz.view(kz.shape[0], -1), summary.view(summary.shape[0], -1)


def _views(cm: Step5CacheManager, layer: int, first_head: int):
    """``(role, view [slots, heads, bytes], global heads)`` for every page a layer
    holds per block."""
    kv = _kv_pages(cm, layer)
    heads = list(range(first_head, first_head + kv.shape[2]))
    views = [(_K, kv[:, 0], heads), (_V, kv[:, 1], heads)]
    scales = _scale_pages(cm, layer)
    if scales is not None:
        views += [(_K_SCALE, scales[:, 0], heads), (_V_SCALE, scales[:, 1], heads)]
    index = _index_pages(cm, layer)
    if index is not None:
        views += [(_KZ, index[0][:, None], [0]), (_SUMMARY, index[1][:, None], [0])]
    return views


def _slots(cm: Step5CacheManager, request_id: int, layer: int, blocks: int) -> list[int]:
    """The request's slot per block ordinal of ``layer``'s life cycle; -1 where it holds none."""
    table = cm.get_slot_table([request_id], layer)[0][:blocks]
    return list(table) + [-1] * (blocks - len(table))


def _needed(layer: int, prompt_len: int) -> range:
    """Block ordinals the first decode step reads: every prompt block on a
    full-attention layer, the window's on a sliding one."""
    blocks = -(-prompt_len // TOKENS_PER_BLOCK)
    if layer in FULL_LAYERS:
        return range(blocks)
    return range(max(0, prompt_len + 1 - SLIDING_WINDOW) // TOKENS_PER_BLOCK, blocks)


def _write_prefill(cm: Step5CacheManager, first_head: int, request_id: int, tokens: list[int]):
    """Write the pattern into every prompt block the request holds, in every layer."""
    keys = _block_keys(tokens)
    for layer in range(DECODER_LAYERS):
        slots = _slots(cm, request_id, layer, len(keys))
        held = [ordinal for ordinal, slot in enumerate(slots) if slot >= 0]
        assert set(_needed(layer, len(tokens))) <= set(held), (
            f"the context instance lost blocks the decode reads: layer {layer}"
        )
        index = torch.tensor([slots[o] for o in held], dtype=torch.int64, device="cuda")
        for role, view, heads in _views(cm, layer, first_head):
            view[index] = _pattern([keys[o] for o in held], layer, role, heads, view.shape[-1])
    torch.cuda.synchronize()


def _check_received(
    cm: Step5CacheManager, first_head: int, request_id: int, tokens: list[int], label: str
) -> None:
    """Every page the first decode step reads holds what the context prefill wrote."""
    keys = _block_keys(tokens)
    for layer in range(DECODER_LAYERS):
        slots = _slots(cm, request_id, layer, len(keys))
        needed = list(_needed(layer, len(tokens)))
        missing = [o for o in needed if slots[o] < 0]
        assert not missing, f"{label}: layer {layer} holds no block at ordinals {missing}"
        index = torch.tensor([slots[o] for o in needed], dtype=torch.int64, device="cuda")
        for role, view, heads in _views(cm, layer, first_head):
            expected = _pattern([keys[o] for o in needed], layer, role, heads, view.shape[-1])
            got = view[index]
            if not torch.equal(got, expected):
                bad = (got != expected).flatten(1).any(dim=1).nonzero().flatten().tolist()
                raise AssertionError(
                    f"{label}: layer {layer} role {role}: blocks "
                    f"{[needed[i] for i in bad][:8]} differ from the context instance's"
                )


# ----------------------------------------------------------------------------
# The page table the transceiver builds from the manager
# ----------------------------------------------------------------------------


@pytest.mark.parametrize("kv_dtype", KV_DTYPES)
@pytest.mark.parametrize("layout", ["dep2", "tep4"])
def test_page_table_names_every_pool(pretrained, layout: str, kv_dtype: str) -> None:
    """The transceiver sees two life cycles: the 69 sliding layers, windowed at
    512, whose K/V (and NVFP4 scales) move by KV head; and the 23 full-attention
    layers, whose K/V (and scales) move by KV head and whose indexer pages, one
    per block of each layer at their own page size, move whole as a replicated
    view."""
    shape = {"dep2": DEP2, "tep4": TEP4}[layout]
    cm = _build(pretrained, shape, 0, kv_dtype)
    try:
        page_table = build_page_table_from_manager(cm)
        groups = page_table.layer_groups
        assert len(groups) == 2 and all(isinstance(g, AttentionLayerGroup) for g in groups)
        by_layers = {
            tuple(sorted(layer.global_layer_id for layer in group.local_layers)): group
            for group in groups
        }
        assert set(by_layers) == {SLIDING_LAYERS, FULL_LAYERS}
        sliding, full = by_layers[SLIDING_LAYERS], by_layers[FULL_LAYERS]
        assert sliding.sliding_window_size == SLIDING_WINDOW
        assert full.sliding_window_size is None or full.sliding_window_size >= MAX_SEQ_LEN
        for group in groups:
            assert group.kv_head_num_per_rank == shape.heads_per_rank

        index_page = TOKENS_PER_BLOCK * (2 * PROXY_DIM * 2 + PROXY_DIM // REGION)
        sliding_roles = {frozenset(v.pool_role): v for v in sliding.pool_views}
        full_roles = {frozenset(v.pool_role): v for v in full.pool_views}
        kv_roles = frozenset({str(Role.KEY), str(Role.VALUE)})
        scale_roles = frozenset({str(Role.KEY_BLOCK_SCALE), str(Role.VALUE_BLOCK_SCALE)})
        expected = {kv_roles} | ({scale_roles} if cm.dtype == DataType.NVFP4 else set())
        assert set(sliding_roles) == expected
        assert set(full_roles) == expected | {frozenset({str(Role.INDEX_KEY)})}
        for views in (sliding_roles, full_roles):
            for roles in expected:
                assert views[roles].mapper_kind == MapperKind.INDEXED
        index_view = full_roles[frozenset({str(Role.INDEX_KEY)})]
        assert index_view.mapper_kind == MapperKind.REPLICATED
        assert index_view.bytes_per_layer == index_page
        assert len(index_view.buffer_entries) == len(FULL_LAYERS)
        kv_page = TOKENS_PER_BLOCK * shape.heads_per_rank * POOL_HEAD_DIM
        kv_page = kv_page * {DataType.BF16: 2, DataType.FP8: 1, DataType.NVFP4: 1}[cm.dtype]
        kv_page = kv_page // (2 if cm.dtype == DataType.NVFP4 else 1)
        assert sliding_roles[kv_roles].bytes_per_layer == 2 * kv_page
        assert full_roles[kv_roles].bytes_per_layer == 2 * kv_page
    finally:
        cm.shutdown()


@pytest.mark.parametrize(
    "transceiver",
    [
        CacheTransceiverConfig(backend="UCX", transceiver_runtime="PYTHON"),
        CacheTransceiverConfig(backend="NIXL", transceiver_runtime="CPP"),
    ],
    ids=["ucx", "cpp"],
)
def test_other_transceiver_routes_are_refused(pretrained, transceiver) -> None:
    """Only the Python NIXL transceiver moves V2 pools, so a disaggregated Step-5
    engine on any other route is refused before its pools are allocated."""
    model_config = _model_config(pretrained, DEP2.mapping(0), "bf16")
    with pytest.raises(ValueError, match="Step-5 disaggregated serving requires"):
        _util.get_kv_cache_manager_cls(
            model_config, _cache_config(), is_disagg=True, cache_transceiver_config=transceiver
        )


# ----------------------------------------------------------------------------
# The hop
# ----------------------------------------------------------------------------


@dataclass
class _Hop:
    """One request across the hop: its prompt and the attention-DP ranks it crosses."""

    index: int
    tokens: list[int]
    ctx_rank: Optional[int]
    gen_rank: Optional[int]
    uid: int

    @property
    def ctx_id(self) -> int:
        return 1000 + self.index

    @property
    def gen_id(self) -> int:
        return 2000 + self.index


def _dp_rank(layout: Layout, rank: int) -> Optional[int]:
    return rank if layout.attention_dp else None


def _plan(ctx: Layout, gen: Layout) -> list[_Hop]:
    """One hop per prompt; the attention-DP ranks cycle so that every pair is
    crossed where the layouts have two ranks. The reuse request follows its donor
    across the donor's pair, before a longer prompt can evict the donor's blocks."""
    rng = random.Random(0)
    prompts = [[rng.randrange(100, 120_000) for _ in range(n)] for n in PROMPT_LENGTHS]
    shared = prompts[REUSE_DONOR][:REUSE_SHARED] + [
        rng.randrange(100, 120_000) for _ in range(REUSE_LENGTH - REUSE_SHARED)
    ]
    pairs = [
        (_dp_rank(ctx, i % ctx.tp_size), _dp_rank(gen, (i + i // 2) % gen.tp_size))
        for i in range(len(prompts))
    ]
    prompts.insert(REUSE_INDEX, shared)
    pairs.insert(REUSE_INDEX, pairs[REUSE_DONOR])
    return [
        _Hop(i, tokens, ctx_rank, gen_rank, uuid.uuid4().int & 0x7FFFFFFFFFFFFFFF)
        for i, (tokens, (ctx_rank, gen_rank)) in enumerate(zip(prompts, pairs))
    ]


def _context_request(hop: _Hop) -> LlmRequest:
    request = LlmRequest(
        request_id=hop.ctx_id,
        max_new_tokens=1,
        input_tokens=list(hop.tokens),
        sampling_config=SamplingConfig(1),
        is_streaming=False,
        llm_request_type=LlmRequestType.LLMREQUEST_TYPE_CONTEXT_ONLY,
    )
    request.py_disaggregated_params = DisaggregatedParams(disagg_request_id=hop.uid)
    return request


def _generation_request(hop: _Hop, endpoint: Optional[str]) -> LlmRequest:
    request = LlmRequest(
        request_id=hop.gen_id,
        max_new_tokens=8,
        input_tokens=list(hop.tokens),
        sampling_config=SamplingConfig(1),
        is_streaming=False,
        llm_request_type=LlmRequestType.LLMREQUEST_TYPE_GENERATION_ONLY,
    )
    request.py_disaggregated_params = DisaggregatedParams(
        ctx_request_id=hop.ctx_id,
        ctx_dp_rank=hop.ctx_rank if hop.ctx_rank is not None else 0,
        ctx_info_endpoint=endpoint,
        disagg_request_id=hop.uid,
    )
    return request


def _prefill(cm: Step5CacheManager, request: LlmRequest) -> int:
    """Prefill in ``max_num_tokens`` chunks, scheduled as the executor schedules
    them; returns where the request resumed (its reused prefix)."""
    resumed = None
    while request.context_remaining_length > 0:
        assert cm.prepare_context(request)
        if resumed is None:
            resumed = request.context_current_position
        request.context_chunk_size = min(MAX_NUM_TOKENS, request.context_remaining_length)
        assert cm.resize_context(request, request.context_chunk_size)
        batch = ScheduledRequests()
        batch.append_context_request(request)
        cm.prepare_resources(batch)
        start = request.context_current_position
        request.py_last_context_chunk = (start, start + request.context_chunk_size)
        request.move_to_next_context_chunk()
        cm.update_context_resources(batch)
    return resumed


def _cross(harness, hop, ctx: Layout, gen: Layout, sides, endpoint) -> dict:
    """Prefill ``hop`` on the context instance, write it, and move it to the
    generation instance; returns what each side reports."""
    ctx_cms, gen_cms, ctx_tcs, gen_tcs = sides
    report = {"ctx_resumed": {}, "gen_cached": {}}

    ctx_requests = {rank: _context_request(hop) for rank in ctx.ranks_for(hop.ctx_rank)}
    for rank, request in ctx_requests.items():
        report["ctx_resumed"][rank] = _prefill(ctx_cms[rank], request)
        _write_prefill(ctx_cms[rank], ctx.first_head(rank), hop.ctx_id, hop.tokens)
        request.add_new_token(len(hop.tokens), 0)

    gen_requests = {
        rank: _generation_request(hop, endpoint) for rank in gen.ranks_for(hop.gen_rank)
    }
    for rank, request in gen_requests.items():
        assert gen_cms[rank].prepare_disagg_gen_init(request)
        report["gen_cached"][rank] = gen_cms[rank].kv_cache_map[hop.gen_id].num_committed_tokens
        request.context_current_position = request.prompt_len
        request.add_new_token(request.prompt_len, 0)

    for rank, request in gen_requests.items():
        gen_tcs[rank].request_and_receive_async(request)
    for rank, request in ctx_requests.items():
        ctx_tcs[rank].respond_and_send_async(request)
    harness.run_concurrent(
        ctx_tcs, lambda tc: tc.check_context_transfer_status(None, mark_complete=True)
    )
    harness.run_concurrent(gen_tcs, lambda tc: tc.check_gen_transfer_status(None))

    for rank, request in gen_requests.items():
        assert request.state == LlmRequestState.DISAGG_GENERATION_TRANS_COMPLETE, request.state
        assert request.py_kv_transfer_verified, f"gen rank {rank}: the received range is unverified"
        _check_received(
            gen_cms[rank],
            gen.first_head(rank),
            hop.gen_id,
            hop.tokens,
            f"prompt #{hop.index} ({len(hop.tokens)} tokens) ctx {hop.ctx_rank} -> gen rank {rank}",
        )

    # Hand the request to generation and retire both sides, as the executor
    # does: the generation instance commits what it received for reuse.
    for rank, request in gen_requests.items():
        request.state = LlmRequestState.GENERATION_IN_PROGRESS
        gen_tcs[rank].commit_blocks_for_reuse(request)
        request.state = LlmRequestState.GENERATION_COMPLETE
        gen_cms[rank].free_resources(request)
    for rank, request in ctx_requests.items():
        request.state = LlmRequestState.GENERATION_COMPLETE
        ctx_cms[rank].free_resources(request)
    return report


@pytest.mark.timeout(900)
@pytest.mark.parametrize(
    "topology,kv_dtype", CASES, ids=[f"{topology}-{kv}" for topology, kv in CASES]
)
def test_generation_reads_what_the_context_wrote(harness, pretrained, topology, kv_dtype) -> None:
    ctx, gen = TOPOLOGIES[topology]
    ctx_cms = [_build(pretrained, ctx, rank, kv_dtype) for rank in range(ctx.tp_size)]
    gen_cms = [_build(pretrained, gen, rank, kv_dtype) for rank in range(gen.tp_size)]
    ctx_tcs, gen_tcs = [], []
    try:
        for cm in ctx_cms:
            _fill_pools(cm, None)
        for cm in gen_cms:
            _fill_pools(cm, 0)
        ctx_tcs = harness.create_instance_transceivers(
            ctx.tp_size, 1, ctx.attention_dp, ctx_cms, TRANSCEIVER
        )
        gen_tcs = harness.create_instance_transceivers(
            gen.tp_size, 1, gen.attention_dp, gen_cms, TRANSCEIVER
        )
        endpoint = harness.get_ctx_info_endpoint(ctx_tcs[0])
        sides = (ctx_cms, gen_cms, ctx_tcs, gen_tcs)

        reports = [_cross(harness, hop, ctx, gen, sides, endpoint) for hop in _plan(ctx, gen)]

        # The other prompts share nothing, so nothing was reused for them.
        for index, report in enumerate(reports):
            if index != REUSE_INDEX:
                assert set(report["ctx_resumed"].values()) == {0}, report
                assert set(report["gen_cached"].values()) == {0}, report
        # The reuse request resumed on the context instance inside the shared
        # prefix; the generation instance restored no more than that prefix.
        reuse = reports[REUSE_INDEX]
        assert all(0 < at <= REUSE_SHARED for at in reuse["ctx_resumed"].values()), reuse
        assert all(0 <= cached <= REUSE_SHARED for cached in reuse["gen_cached"].values()), reuse
    finally:
        for tc in ctx_tcs + gen_tcs:
            tc.shutdown()
        for cm in ctx_cms + gen_cms:
            cm.shutdown()
