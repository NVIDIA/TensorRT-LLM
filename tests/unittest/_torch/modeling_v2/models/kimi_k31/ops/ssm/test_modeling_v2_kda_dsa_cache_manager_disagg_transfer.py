# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Disaggregated KV transfer of the Kimi K3.1 hybrid cache manager (``KimiK3DSAHybridCacheManager``) through the
V2 Python transceiver, from a TEP context server (attention TP, KDA state head-sharded) to a DEP generation server
(attention DP, each request on one rank with the whole KDA state).

Every rank of both servers lives in this process, as in the engine's hybrid-cache transfer tests: one cache manager
and one ``KvCacheTransceiverV2`` per rank, the transceivers created on threads over a barrier-based distributed
stand-in and moving bytes through real NIXL agents on one GPU (``tests/unittest/disaggregated/kv_transfer_harness.py``).
The ranks build the same transfer plans and issue the same NIXL writes as separate processes would.

The cache: the MLA latent pages in FP8 (``kv_cache_config.dtype`` ``fp8``), the DSA indexer's
FP8 keys and scales (one ``INDEX_KEY`` page per MLA layer and block) and the KDA conv / recurrent state. Each case
checks the transfer plan the context ranks build (``Sender._build_kv_write_meta``) and the receiver memory after
the writes:

* every write goes to the generation rank that owns the request, and its destinations cover the request's latent
  and indexer pages of the prompt blocks and its KDA state slot exactly once (no byte twice, none left out, none
  elsewhere); every context rank sends its KDA head shard;
* the owned bytes hold what the context ranks hold (the whole KDA state assembled from the shards) as the family's
  own views read them (``get_unique_primary_pool``, ``get_indexer_k_cache_buffers``, the KDA state views), and every
  other byte of each generation rank's pools keeps the sentinel it held before the transfer.
"""

import math
import uuid
from types import SimpleNamespace
from typing import NamedTuple

import pytest
import torch

import tensorrt_llm
from tensorrt_llm import DisaggregatedParams, SamplingParams
from tensorrt_llm._torch.disaggregation.native.transfer import Sender
from tensorrt_llm._torch.disaggregation.resource.kv_extractor import KVRegionExtractorV1
from tensorrt_llm._torch.disaggregation.resource.utils import get_pool_bytes
from tensorrt_llm._torch.modeling_v2.models.kimi_k31.ops.ssm.kda_dsa_cache_manager import (  # noqa: E501
    KimiK3DSAHybridCacheManager,
)
from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequest, LlmRequestType
from tensorrt_llm._torch.pyexecutor.resource_manager import CacheTypeCpp, DataType
from tensorrt_llm._torch.pyexecutor.scheduler import ScheduledRequests
from tensorrt_llm._utils import TensorWrapper, convert_to_torch_tensor
from tensorrt_llm.llmapi.llm_args import (
    CacheTransceiverConfig,
    DeepSeekSparseAttentionConfig,
    KvCacheConfig,
)
from tensorrt_llm.mapping import Mapping

__extra_import_path__ = ["~/tests/unittest/disaggregated"]

_TOKENS_PER_BLOCK = 32
# Layers 0 and 2 are KDA, layers 1 and 3 MLA with a DSA indexer (1-indexed 2 and 4).
_KDA_MASK = [True, False, True, False]
_MLA_MASK = [False, True, False, True]
#: MLA latent width (kv_lora_rank + qk_rope_head_dim).
_LATENT_DIM = 64
#: KDA heads, head width (= state width) and convolution width.
_KDA_HEADS, _KDA_HEAD_DIM, _KDA_CONV = 4, 8, 4
#: Prompt lengths: part of a block, exactly two blocks, part of a third, part of one.
_PROMPT_LENGTHS = [33, 64, 95, 17]
_DRAFT_LAYERS = 2
_DRAFT_WINDOW = 32
_SENTINEL = 0xA5
_LATENT_DTYPES = {"fp8": DataType.FP8, "auto": DataType.BF16}


def _sparse_config() -> DeepSeekSparseAttentionConfig:
    return DeepSeekSparseAttentionConfig(
        index_n_heads=4, index_head_dim=128, index_topk=64, indexer_k_dtype="fp8"
    )


def _text_config() -> SimpleNamespace:
    return SimpleNamespace(
        architectures=["KimiLinearForCausalLM"],
        model_type="kimi_linear",
        num_hidden_layers=4,
        kv_lora_rank=32,
        qk_rope_head_dim=32,
        linear_attn_config={
            "head_dim": _KDA_HEAD_DIM,
            "num_heads": _KDA_HEADS,
            "short_conv_kernel_size": _KDA_CONV,
            "kda_layers": [1, 3],
            "full_attn_layers": [2, 4],
        },
        dsa_config={"dsa_layers": [2, 4], "index_topk_freq": 1},
        index_topk_freq=1,
        index_skip_topk_offset=1,
        dtype=torch.bfloat16,
    )


def _manager(tp: int, rank: int, attention_dp: bool, kv_dtype: str) -> KimiK3DSAHybridCacheManager:
    return KimiK3DSAHybridCacheManager(
        mamba_d_state=_KDA_HEAD_DIM,
        mamba_d_conv=_KDA_CONV,
        mamba_num_heads=_KDA_HEADS,
        mamba_n_groups=_KDA_HEADS,
        mamba_head_dim=_KDA_HEAD_DIM,
        mamba_num_layers=2,
        mamba_layer_mask=_KDA_MASK,
        mamba_cache_dtype=torch.bfloat16,
        mamba_ssm_cache_dtype=torch.float32,
        kv_cache_config=KvCacheConfig(
            max_tokens=512,
            enable_block_reuse=False,
            event_buffer_max_size=0,
            use_kv_cache_manager_v2=True,
            dtype=kv_dtype,
        ),
        kv_cache_type=CacheTypeCpp.SELFKONLY,
        num_layers=2,
        num_kv_heads=1,
        head_dim=_LATENT_DIM,
        tokens_per_block=_TOKENS_PER_BLOCK,
        max_seq_len=256,
        max_batch_size=4,
        mapping=Mapping(
            world_size=tp, rank=rank, tp_size=tp, pp_size=1, enable_attention_dp=attention_dp
        ),
        dtype=_LATENT_DTYPES[kv_dtype],
        layer_mask=_MLA_MASK,
        sparse_attention_config=_sparse_config(),
        pretrained_config=_text_config(),
        draft_num_layers=_DRAFT_LAYERS,
        draft_window_size=_DRAFT_WINDOW,
        draft_num_kv_heads=1,
        draft_total_num_kv_heads=1,
        draft_kv_factor=1,
        draft_head_dim=64,
        is_disagg=True,
    )


# --- the transfer -----------------------------------------------------------------------------------------------


class _Request(NamedTuple):
    idx: int
    unique_rid: int
    ctx: LlmRequest
    gen: LlmRequest
    #: The generation rank that holds the request (attention DP).
    owner: int

    @property
    def num_blocks(self) -> int:
        return math.ceil(self.ctx.prompt_len / _TOKENS_PER_BLOCK)


class _Write(NamedTuple):
    sender: int
    receiver: int
    unique_rid: int
    dst: list
    sizes: list


class _Region(NamedTuple):
    """One cache region of a request: its view and the key of its content in the ground truth."""

    view: torch.Tensor
    key: tuple


def _bytes(view: torch.Tensor) -> torch.Tensor:
    assert view.is_contiguous()
    return view.reshape(-1).view(torch.uint8)


def _span(view: torch.Tensor) -> tuple[int, int]:
    assert view.is_contiguous()
    return view.data_ptr(), view.data_ptr() + view.nbytes


def _merge(spans) -> list[tuple[int, int]]:
    merged = []
    for start, end in sorted(spans):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def _mla_layers(manager) -> list[int]:
    return [manager.pp_layers[local] for local in manager._attention_local_layers]


def _indexer_layers(manager) -> list[int]:
    return [
        layer
        for layer, has_indexer in zip(manager.pp_layers, manager.indexer_k_cache_local_layer_mask)
        if has_indexer
    ]


def _regions(manager, request_id: int, idx: int, num_blocks: int) -> list[_Region]:
    """The request's regions the transfer moves: the latent and indexer pages of its prompt blocks (the family's
    DSA views at the block's pool slot) and its KDA conv / recurrent state slot of each KDA layer."""
    slots = manager.get_pool_block_indices(1, request_ids=[request_id])[0, :num_blocks].tolist()
    latent = manager.get_unique_primary_pool()
    regions = []
    for ordinal, layer in enumerate(_mla_layers(manager)):
        for block, slot in enumerate(slots):
            regions.append(_Region(latent[slot, ordinal], (idx, "latent", layer, block)))
    for layer in _indexer_layers(manager):
        pages = manager.get_indexer_k_cache_buffers(layer)
        for block, slot in enumerate(slots):
            page = pages[slot * manager.indexer_k_cache_page_scale]
            regions.append(_Region(page, (idx, "index_k", layer, block)))
    draft_slots = manager.get_draft_block_indices([request_id])[0]
    for layer in range(_DRAFT_LAYERS):
        pages = manager.get_draft_buffers(layer)
        for block, slot in enumerate(draft_slots[:num_blocks]):
            if slot >= 0:
                regions.append(_Region(pages[slot], (idx, "draft", layer, block)))
    state = manager._request_id_to_state_index[request_id]
    for layer in manager.mamba_layer_offsets:
        regions.append(_Region(manager.get_conv_states(layer)[state], (idx, "conv", layer)))
        regions.append(_Region(manager.get_ssm_states(layer)[state], (idx, "ssm", layer)))
    return regions


def _shard_conv(full: torch.Tensor, tp: int, rank: int) -> torch.Tensor:
    """Rank ``rank``'s rows of the ``[q | k | v]`` convolution state: each section split over ``tp`` ranks."""
    section = full.shape[0] // 3
    rows = section // tp
    return torch.cat(
        [full[s * section + rank * rows : s * section + (rank + 1) * rows] for s in range(3)]
    )


def _shard_ssm(full: torch.Tensor, tp: int, rank: int) -> torch.Tensor:
    """Rank ``rank``'s heads of the ``[heads, head_dim, state]`` recurrent state."""
    heads = full.shape[0] // tp
    return full[rank * heads : (rank + 1) * heads]


def _ground_truth(requests, manager) -> dict:
    """Random content of every region key, on the GPU: page bytes for the latent and indexer pages, the whole
    (unsharded) KDA states."""
    gen = torch.Generator().manual_seed(20261110)
    latent_bytes = manager.get_unique_primary_pool()[0, 0].nbytes
    index_bytes = manager.get_indexer_k_cache_buffers(_indexer_layers(manager)[0])[0].nbytes
    conv_rows = 3 * _KDA_HEADS * _KDA_HEAD_DIM
    truth = {}
    for request in requests:
        for block in range(request.num_blocks):
            for layer in _mla_layers(manager):
                key = (request.idx, "latent", layer, block)
                truth[key] = torch.randint(
                    0, 256, (latent_bytes,), generator=gen, dtype=torch.uint8
                )
            for layer in range(_DRAFT_LAYERS):
                key = (request.idx, "draft", layer, block)
                size = manager.get_draft_buffers(layer)[0].nbytes
                truth[key] = torch.randint(1, 256, (size,), generator=gen, dtype=torch.uint8)
            for layer in _indexer_layers(manager):
                key = (request.idx, "index_k", layer, block)
                truth[key] = torch.randint(0, 256, (index_bytes,), generator=gen, dtype=torch.uint8)
        for layer in manager.mamba_layer_offsets:
            truth[(request.idx, "conv", layer)] = torch.randn(
                conv_rows, _KDA_CONV - 1, generator=gen
            ).to(torch.bfloat16)
            truth[(request.idx, "ssm", layer)] = torch.randn(
                _KDA_HEADS, _KDA_HEAD_DIM, _KDA_HEAD_DIM, generator=gen
            )
    return {key: value.cuda() for key, value in truth.items()}


def _fill_context(manager, request: _Request, tp: int, rank: int, truth: dict) -> None:
    """The context rank's copy of the request: the replicated pages, and its head shard of the KDA state."""
    for region in _regions(manager, request.ctx.py_request_id, request.idx, request.num_blocks):
        kind = region.key[1]
        if kind == "conv":
            region.view.copy_(_shard_conv(truth[region.key], tp, rank))
        elif kind == "ssm":
            region.view.copy_(_shard_ssm(truth[region.key], tp, rank))
        else:
            _bytes(region.view).copy_(truth[region.key])


def _expected_bytes(region: _Region, truth: dict) -> torch.Tensor:
    expected = truth[region.key]
    return expected.reshape(-1).view(torch.uint8) if expected.dtype != torch.uint8 else expected


def _pool_spans(manager) -> list[tuple[int, int]]:
    page_table = KVRegionExtractorV1(manager).page_table
    return _merge(
        (pool.base_address, pool.base_address + get_pool_bytes(pool))
        for pool_group in page_table.pool_groups
        for pool in pool_group.pools
    )


def _pool_bytes(start: int, end: int) -> torch.Tensor:
    return convert_to_torch_tensor(TensorWrapper(start, torch.uint8, [end - start]))


@pytest.fixture
def transfer_harness(monkeypatch):
    """The shared threaded NIXL harness, with its transport environment set for this test only."""
    monkeypatch.setenv("TRTLLM_NIXL_NUM_THREADS", "1")
    monkeypatch.setenv("UCX_TLS", "^ib,gdr_copy")
    import kv_transfer_harness

    return kv_transfer_harness


def _llm_request(request_id: int, length: int, request_type) -> LlmRequest:
    return LlmRequest(
        request_id=request_id,
        max_new_tokens=1,
        input_tokens=list(range(length)),
        sampling_config=tensorrt_llm.bindings.SamplingConfig(
            SamplingParams()._get_sampling_config()
        ),
        is_streaming=False,
        llm_request_type=request_type,
    )


def _requests(ctx_endpoint: str, gen_tp: int) -> list[_Request]:
    requests = []
    for idx, length in enumerate(_PROMPT_LENGTHS):
        unique_rid = uuid.uuid4().int & 0x7FFFFFFFFFFFFFFF
        ctx_rid, gen_rid = 100 + 2 * idx, 101 + 2 * idx
        ctx = _llm_request(ctx_rid, length, LlmRequestType.LLMREQUEST_TYPE_CONTEXT_ONLY)
        ctx.py_disaggregated_params = DisaggregatedParams(disagg_request_id=unique_rid)
        gen = _llm_request(gen_rid, length, LlmRequestType.LLMREQUEST_TYPE_GENERATION_ONLY)
        # The context server runs attention TP: one DP rank.
        gen.py_disaggregated_params = DisaggregatedParams(
            ctx_request_id=ctx_rid,
            ctx_dp_rank=0,
            ctx_info_endpoint=ctx_endpoint,
            disagg_request_id=unique_rid,
        )
        requests.append(_Request(idx, unique_rid, ctx, gen, owner=idx % gen_tp))
    return requests


def _batch(requests: list[LlmRequest]) -> ScheduledRequests:
    batch = ScheduledRequests()
    batch.context_requests_last_chunk = requests
    return batch


def _allocate(managers, batches: list[list[LlmRequest]], generation: bool) -> None:
    """Allocate each rank's requests the way the executor does: prefill (context) or disaggregated generation
    init, then the step's resources (the KDA state slots)."""
    for manager, requests in zip(managers, batches):
        if not requests:
            continue
        for request in requests:
            if generation:
                assert manager.prepare_disagg_gen_init(request)
            else:
                assert manager.prepare_context(request)
                assert manager.resize_context(request, request.context_chunk_size)
        manager.prepare_resources(_batch(requests))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
@pytest.mark.timeout(240)
@pytest.mark.parametrize(
    "ctx_tp,gen_tp,kv_dtype",
    [(2, 4, "fp8")],
    ids=["tep2_to_dep4-fp8_latent"],
)
def test_tep_to_dep_transfer(transfer_harness, monkeypatch, ctx_tp, gen_tp, kv_dtype):
    harness = transfer_harness
    ctx_managers = [_manager(ctx_tp, rank, False, kv_dtype) for rank in range(ctx_tp)]
    gen_managers = [_manager(gen_tp, rank, True, kv_dtype) for rank in range(gen_tp)]
    ctx_tcs, gen_tcs = [], []
    try:
        config = CacheTransceiverConfig(
            backend="NIXL", transceiver_runtime="PYTHON", max_tokens_in_buffer=512
        )
        ctx_tcs = harness.create_instance_transceivers(ctx_tp, 1, False, ctx_managers, config)
        gen_tcs = harness.create_instance_transceivers(gen_tp, 1, True, gen_managers, config)
        requests = _requests(harness.get_ctx_info_endpoint(ctx_tcs[0]), gen_tp)
        # Every context rank holds every request (attention TP); a generation rank holds the ones it owns.
        ctx_batches = [[r.ctx for r in requests]] * ctx_tp
        gen_batches = [[r.gen for r in requests if r.owner == rank] for rank in range(gen_tp)]
        _allocate(ctx_managers, ctx_batches, generation=False)
        _allocate(gen_managers, gen_batches, generation=True)
        for request in requests:
            for req in (request.ctx, request.gen):
                req.context_current_position = req.prompt_len
                req.add_new_token(req.prompt_len, 0)
        for manager, reqs in zip(ctx_managers + gen_managers, ctx_batches + gen_batches):
            if reqs:
                manager.update_resources(_batch(reqs))
        torch.cuda.synchronize()

        truth = _ground_truth(requests, gen_managers[0])
        for rank, manager in enumerate(ctx_managers):
            for request in requests:
                _fill_context(manager, request, ctx_tp, rank, truth)
        gen_regions = {
            r.idx: _regions(gen_managers[r.owner], r.gen.py_request_id, r.idx, r.num_blocks)
            for r in requests
        }
        for r in requests:
            first_tail_block = max(0, r.ctx.prompt_len - _DRAFT_WINDOW) // _TOKENS_PER_BLOCK
            expected = {
                (r.idx, "draft", layer, block)
                for layer in range(_DRAFT_LAYERS)
                for block in range(first_tail_block, r.num_blocks)
            }
            assert expected <= {region.key for region in gen_regions[r.idx]}
        gen_pools = [_pool_spans(manager) for manager in gen_managers]
        for spans in gen_pools:
            for start, end in spans:
                _pool_bytes(start, end).fill_(_SENTINEL)
        torch.cuda.synchronize()

        writes = []
        build_write_meta = Sender._build_kv_write_meta

        def record_write_meta(sender, task, req_info):
            meta = build_write_meta(sender, task, req_info)
            writes.append(
                _Write(
                    sender._registrar.self_rank_info.instance_rank,
                    meta.peer_rank,
                    meta.unique_rid,
                    meta.dst_ptrs.tolist(),
                    meta.sizes.tolist(),
                )
            )
            return meta

        monkeypatch.setattr(Sender, "_build_kv_write_meta", record_write_meta)
        for request in requests:
            gen_tcs[request.owner].request_and_receive_async(request.gen)
        for request in requests:
            for tc in ctx_tcs:
                tc.respond_and_send_async(request.ctx)
        harness.run_concurrent(
            ctx_tcs, lambda tc: tc.check_context_transfer_status(None, mark_complete=True)
        )
        harness.run_concurrent(gen_tcs, lambda tc: tc.check_gen_transfer_status(None))
        torch.cuda.synchronize()

        # The plan: each request's writes go to its owner and cover its regions exactly once.
        by_rid = {r.unique_rid: r for r in requests}
        assert {w.unique_rid for w in writes} == set(by_rid)
        for request in requests:
            mine = [w for w in writes if w.unique_rid == request.unique_rid]
            assert {w.receiver for w in mine} == {request.owner}
            assert {w.sender for w in mine} <= set(range(ctx_tp))
            intervals = sorted(
                (dst, dst + size, w.sender)
                for w in mine
                for dst, size in zip(w.dst, w.sizes)
                if size > 0
            )
            for (_, end, _), (start, _, _) in zip(intervals, intervals[1:]):
                assert end <= start, f"request {request.idx}: a receiver byte is written twice"
            spans = [_span(region.view) for region in gen_regions[request.idx]]
            assert sum(end - start for start, end in spans) == sum(
                end - start for start, end in _merge(spans)
            )
            assert _merge((start, end) for start, end, _ in intervals) == _merge(spans), (
                f"request {request.idx}: the writes do not cover exactly its latent and indexer pages and KDA "
                "state slot"
            )
            state_spans = [
                _span(region.view)
                for region in gen_regions[request.idx]
                if region.key[1] in ("conv", "ssm")
            ]
            state_senders = {
                sender
                for start, end, sender in intervals
                if any(lo <= start and end <= hi for lo, hi in state_spans)
            }
            assert state_senders == set(range(ctx_tp)), (
                "every context rank sends its KDA head shard"
            )

        # The bytes: the owned regions hold the context's content, nothing else changed.
        for rank, manager in enumerate(gen_managers):
            regions = [region for r in requests if r.owner == rank for region in gen_regions[r.idx]]
            for region in regions:
                assert torch.equal(_bytes(region.view), _expected_bytes(region, truth)), (
                    f"gen rank {rank}: {region.key} differs from the context's"
                )
            for start, end in gen_pools[rank]:
                pool = _pool_bytes(start, end)
                untouched = torch.ones(end - start, dtype=torch.bool, device=pool.device)
                for region in regions:
                    lo, hi = _span(region.view)
                    if start <= lo and hi <= end:
                        untouched[lo - start : hi - start] = False
                assert bool((pool[untouched] == _SENTINEL).all()), (
                    f"gen rank {rank}: bytes outside the owned regions changed"
                )
            for region in regions:
                lo, hi = _span(region.view)
                assert any(start <= lo and hi <= end for start, end in gen_pools[rank])
    finally:
        for tc in ctx_tcs + gen_tcs:
            tc.shutdown()
        for manager in ctx_managers + gen_managers:
            manager.shutdown()
