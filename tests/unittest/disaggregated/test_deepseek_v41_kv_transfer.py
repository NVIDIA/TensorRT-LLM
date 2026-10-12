# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Disaggregated KV transfer of the DeepSeek-V4.1 layout of DeepseekV4CacheManager.

The context and generation managers carry the checkpoint's real 40-layer role
table. The context side runs without speculation and the generation side with a
DSpark draft depth, which is how a DSpark disaggregated deployment places the
drafter. Pools are compared per position, not per page: for every request the
generation side must hold, byte for byte, what the context side wrote at every
position the first generation step reads:

- SWA: the last ``sliding_window - 1`` prompt positions, on all 40 layers;
- compressor pending pair: the positions of the incomplete ratio-2 group;
- Engram token history: the last ``engram_max_ngram_size - 1`` prompt positions;
- main KV and index K: every complete compressed entry of every KV source.

The reference bytes are taken from the context pools right after allocation,
before the context side advances its history (which releases out-of-window
pages) and before the transfer.
"""

from types import SimpleNamespace
from typing import Dict, List, Optional, Sequence, Tuple

import pytest
import torch
from kv_transfer_harness import (
    TOKENS_PER_BLOCK,
    VOCAB_SIZE,
    create_instance_transceivers,
    get_ctx_info_endpoint,
    run_concurrent,
)
from utils.util import skip_pre_blackwell

import tensorrt_llm
import tensorrt_llm.bindings
from tensorrt_llm import DisaggregatedParams, Mapping, SamplingParams
from tensorrt_llm._torch.attention.backends.sparse.deepseek_v4 import DeepseekV4CacheManager
from tensorrt_llm._torch.attention.backends.sparse.deepseek_v4.params import DeepseekV4AttentionType
from tensorrt_llm._torch.attention.backends.sparse.deepseek_v4.v41_layout import (
    DEEPSEEK_V41_ATTENTION,
    DEEPSEEK_V41_ENCODER_REPLAY_ENV,
    DEEPSEEK_V41_KV_STORAGE_ENV,
    DeepseekV41Layout,
)
from tensorrt_llm._torch.disaggregation.resource.kv_extractor import KVRegionExtractorV1
from tensorrt_llm._torch.disaggregation.resource.utils import get_physical_pool, get_pool_bytes
from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequest, LlmRequestType
from tensorrt_llm._torch.pyexecutor.scheduler import ScheduledRequests
from tensorrt_llm._utils import TensorWrapper, convert_to_torch_tensor
from tensorrt_llm.bindings import DataType
from tensorrt_llm.bindings.internal.batch_manager import CacheType as CacheTypeCpp
from tensorrt_llm.llmapi.llm_args import (
    CacheTransceiverConfig,
    DeepSeekV4SparseAttentionConfig,
    DSparkDecodingConfig,
    KvCacheConfig,
)

T = DeepseekV4AttentionType
WINDOW = 128
HEAD_DIM = 512
INDEX_HEAD_DIM = 128
MAX_SEQ_LEN = 4096
MAX_BATCH_SIZE = 16
# The DSpark block size of the checkpoint; the generation engine's draft depth.
GEN_DRAFT_LEN = 5

# The checkpoint's text_config fields the layout reads (config.json of
# DeepSeek-V4.1-Flash-NVFP4), 40 backbone layers plus 3 DSpark stages.
V41_TEXT_FIELDS = dict(
    model_type="deepseek_v41_text",
    num_hidden_layers=40,
    num_nextn_predict_layers=3,
    compress_ratios=[0, 0] + [2] * 18 + [1] * 20 + [0, 0, 0],
    kv_source_layer_ids=[2, 8, 14, 20],
    index_source_layer_ids=[2, 8, 14, 20, 24, 28, 32, 36],
    candidate_source_layer_id=20,
    engram_layer_ids=[1, 14],
    engram_max_ngram_size=4,
    sliding_window=WINDOW,
    head_dim=HEAD_DIM,
    index_head_dim=INDEX_HEAD_DIM,
    index_topk=512,
    kv_lora_rank=448,
    qk_rope_head_dim=64,
)


@pytest.fixture(autouse=True)
def _no_encoder_replay(monkeypatch):
    """Encoder SWA bounded replay refuses disaggregated serving (and dep4): pin it off."""
    monkeypatch.setenv(DEEPSEEK_V41_ENCODER_REPLAY_ENV, "0")


# Prompt lengths: 1-4 (no longer than the Engram history), odd lengths (a
# ratio-2 pending token), block-aligned lengths, and lengths whose Engram
# history P-3 .. P-1 straddles a page boundary (P % 128 in {1, 2, 3}).
REQUEST_LENGTHS = [1, 2, 3, 4, 129, 130, 131, 256, 258, 383, 2049, 2050]


def make_v41_config() -> SimpleNamespace:
    top = {k: v for k, v in V41_TEXT_FIELDS.items() if k != "model_type"}
    return SimpleNamespace(
        architectures=["DeepseekV41ForCausalLM"],
        model_type="deepseek_v41",
        text_config=SimpleNamespace(**V41_TEXT_FIELDS),
        **top,
    )


def create_v41_manager(mapping: Mapping, draft_len: int = 0) -> DeepseekV4CacheManager:
    config = make_v41_config()
    sparse_config = DeepSeekV4SparseAttentionConfig(
        index_head_dim=INDEX_HEAD_DIM,
        window_size=WINDOW,
        compress_ratios=[1] * len(config.compress_ratios),
    )
    kv_cache_config = KvCacheConfig(
        enable_block_reuse=False,
        max_tokens=MAX_SEQ_LEN * MAX_BATCH_SIZE,
        event_buffer_max_size=0,
    )
    spec_config = DSparkDecodingConfig(max_draft_len=draft_len) if draft_len > 0 else None
    return DeepseekV4CacheManager(
        kv_cache_config=kv_cache_config,
        kv_cache_type=CacheTypeCpp.SELFKONLY,
        num_layers=config.num_hidden_layers,
        num_kv_heads=1,
        head_dim=HEAD_DIM,
        tokens_per_block=TOKENS_PER_BLOCK,
        max_seq_len=MAX_SEQ_LEN,
        max_batch_size=MAX_BATCH_SIZE,
        max_input_len=MAX_SEQ_LEN,
        mapping=mapping,
        dtype=DataType.BF16,
        compressor_dtype=DataType.FLOAT,
        vocab_size=VOCAB_SIZE,
        max_num_tokens=8192,
        sparse_attn_config=sparse_config,
        pretrained_config=config,
        spec_config=spec_config,
    )


def create_instance_managers(tp: int, enable_dp: bool, draft_len: int) -> List:
    return [
        create_v41_manager(
            Mapping(world_size=tp, rank=rank, tp_size=tp, pp_size=1, enable_attention_dp=enable_dp),
            draft_len=draft_len,
        )
        for rank in range(tp)
    ]


def fill_pools(managers: Sequence[DeepseekV4CacheManager], seed: Optional[int]) -> None:
    """Random bytes (the same on every TP rank: MLA state is replicated), or zeros."""
    for mgr in managers:
        page_table = KVRegionExtractorV1(mgr).page_table
        pools: Dict[int, int] = {}
        for lg_idx, lg in enumerate(page_table.layer_groups):
            for pv in lg.pool_views:
                pool = get_physical_pool(page_table, lg_idx, pv.pool_idx)
                pools[pool.base_address] = max(
                    pools.get(pool.base_address, 0), get_pool_bytes(pool)
                )
        for index, (base, size) in enumerate(sorted(pools.items())):
            view = convert_to_torch_tensor(TensorWrapper(base, DataType.INT8, [size]))
            if seed is None:
                view.zero_()
                continue
            gen = torch.Generator(device=view.device).manual_seed(seed + index)
            view.copy_(
                torch.randint(
                    -128, 128, view.shape, dtype=torch.int8, device=view.device, generator=gen
                )
            )
    torch.cuda.synchronize()


# ---------------------------------------------------------------------------
# What the first generation step reads, per owned (layer, role).
# ---------------------------------------------------------------------------


def needed_slots(layout: DeepseekV41Layout, layer: int, attn_type, prompt_len: int) -> List[int]:
    """The slots the generation side reads before it writes anything of its own.

    Positions for the windowed roles, compressed entries for the persistent roles.
    """
    if attn_type == T.SWA:
        return list(range(max(0, prompt_len - (layout.window_size - 1)), prompt_len))
    if attn_type in (T.COMPRESSOR_KV, T.COMPRESSOR_SCORE):
        ratio = layout.compress_ratios[layer]
        return list(range((prompt_len // ratio) * ratio, prompt_len))
    if attn_type == T.ENGRAM_TOKEN:
        return list(range(max(0, prompt_len - layout.engram_history_len), prompt_len))
    if attn_type in (T.COMPRESS, T.INDEXER_COMPRESS):
        return list(range(prompt_len // layout.compress_ratios[layer]))
    raise AssertionError(attn_type)


def owned_roles(mgr: DeepseekV4CacheManager) -> List[Tuple[int, DeepseekV4AttentionType]]:
    return sorted(mgr._layer_attn_to_layer_id, key=lambda key: (key[0], key[1].value))


def read_slots(mgr, request_id: int, layer: int, attn_type, slots: Sequence[int]):
    """(slot -> raw bytes) for every slot, or (slot -> None) where no page is mapped.

    Index K is compared page by page: its DeepGEMM page layout keeps an entry's
    code bytes and scale bytes apart, and the transfer is page granular anyway.
    """
    buffer = mgr.get_buffers(layer, attn_type)
    per_page = buffer.shape[1]
    pages = mgr.get_cache_indices(request_id, layer, attn_type)
    out = {}
    if attn_type == T.INDEXER_COMPRESS:
        flat = buffer.flatten(1)
        for ordinal in sorted({slot // per_page for slot in slots}):
            page = pages[ordinal] if ordinal < len(pages) else -1
            out[("page", ordinal)] = None if page < 0 else flat[page].clone()
        return out
    for slot in slots:
        ordinal, row = divmod(slot, per_page)
        page = pages[ordinal] if ordinal < len(pages) else -1
        out[slot] = None if page < 0 else buffer[page, row].contiguous().view(torch.uint8).clone()
    return out


def snapshot(mgr, request_id: int, prompt_len: int):
    layout = mgr.v41_layout
    return {
        (layer, attn_type): read_slots(
            mgr, request_id, layer, attn_type, needed_slots(layout, layer, attn_type, prompt_len)
        )
        for layer, attn_type in owned_roles(mgr)
    }


def make_request(request_id: int, prompt_len: int, request_type) -> LlmRequest:
    return LlmRequest(
        request_id=request_id,
        max_new_tokens=8,
        input_tokens=list(range(prompt_len)),
        sampling_config=tensorrt_llm.bindings.SamplingConfig(
            SamplingParams()._get_sampling_config()
        ),
        is_streaming=False,
        llm_request_type=request_type,
    )


# ---------------------------------------------------------------------------
# Manager-level checks (no transfer).
# ---------------------------------------------------------------------------


@skip_pre_blackwell
@pytest.mark.parametrize("draft_len", [0, GEN_DRAFT_LEN], ids=["no_spec", "dspark5"])
def test_context_history_keeps_what_generation_reads(draft_len):
    """After prefill commits its history, every position the next step reads is still mapped.

    This is the manager's windowed life cycle on its own: a context engine
    advances ``history_length`` to the prompt length before it hands the pages
    to the transceiver (and an aggregated engine does the same before its first
    decode step), so a position outside the live window is unmapped and can be
    neither transferred nor read.
    """
    mgr = create_v41_manager(Mapping(world_size=1, rank=0, tp_size=1, pp_size=1), draft_len)
    try:
        requests = []
        batch = ScheduledRequests()
        for rid, prompt_len in enumerate(REQUEST_LENGTHS):
            req = make_request(
                rid, prompt_len, LlmRequestType.LLMREQUEST_TYPE_CONTEXT_AND_GENERATION
            )
            assert mgr.prepare_context(req)
            assert mgr.resize_context(req, req.context_chunk_size)
            requests.append(req)
            batch.context_requests_last_chunk.append(req)
        for req in requests:
            req.context_current_position = req.prompt_len
        mgr.update_context_resources(batch)
        unmapped = []
        for req in requests:
            for (layer, attn_type), slots in snapshot(
                mgr, req.py_request_id, req.prompt_len
            ).items():
                missing = sorted(slot for slot, data in slots.items() if data is None)
                if missing:
                    unmapped.append((req.prompt_len, layer, attn_type.name, missing))
        assert not unmapped, f"unmapped after history=prompt_len: {unmapped}"
    finally:
        mgr.shutdown()


@skip_pre_blackwell
def test_disagg_global_layer_ids_are_owner_only():
    """The transfer's synthetic layer ids name owners only, and both peers agree on them."""
    ctx = create_v41_manager(Mapping(world_size=1, rank=0, tp_size=1, pp_size=1), 0)
    gen = create_v41_manager(Mapping(world_size=1, rank=0, tp_size=1, pp_size=1), GEN_DRAFT_LEN)
    try:
        layout = ctx.v41_layout
        num_types = ctx.disagg_num_attn_types
        assert num_types == gen.disagg_num_attn_types == max(t.value for t in T) + 1
        per_side = []
        for mgr in (ctx, gen):
            assert mgr.get_disagg_ignored_roles() == frozenset()
            page_table = KVRegionExtractorV1(mgr).page_table
            ids = {}
            for lg in page_table.layer_groups:
                local_to_global = {ll.local_layer_id: ll.global_layer_id for ll in lg.local_layers}
                for pv in lg.pool_views:
                    for entry in pv.buffer_entries:
                        gid = local_to_global[int(entry["local_layer_id"])]
                        ids.setdefault(gid, set()).update(str(r) for r in pv.pool_role)
            for gid in ids:
                layer, value = divmod(gid, num_types)
                attn_type = T(value)
                assert (layer, attn_type) in mgr._layer_attn_to_layer_id, (gid, layer, attn_type)
                assert layout.has_attention(layer, attn_type), (gid, layer, attn_type)
                if layout.compress_ratios[layer] > 0 and not layout.is_kv_source(layer):
                    # A reader layer: its only pool is its own SWA window.
                    assert attn_type == T.SWA, (gid, layer, attn_type)
            roles = set().union(*ids.values())
            for attn_type in DEEPSEEK_V41_ATTENTION:
                assert str(attn_type.role) in roles, (attn_type, roles)
            # One id per manager layer: SWA x 40, main KV + index K x 4,
            # compressor pair x 3, Engram history x 1.
            assert len(ids) == 40 + 4 + 3 + 1
            per_side.append(ids)
        assert per_side[0] == per_side[1]
    finally:
        ctx.shutdown()
        gen.shutdown()


# ---------------------------------------------------------------------------
# Transfer round trip.
# ---------------------------------------------------------------------------


def run_v41_transfer(ctx_tp: int, gen_tp: int, ctx_dp: bool, gen_dp: bool, update_before: bool):
    import uuid

    try:
        ctx_managers = create_instance_managers(ctx_tp, ctx_dp, draft_len=0)
    except NotImplementedError as error:
        pytest.skip(f"storage not implemented in this tree: {error}")
    gen_managers = create_instance_managers(gen_tp, gen_dp, draft_len=GEN_DRAFT_LEN)
    fill_pools(ctx_managers, seed=1000)
    fill_pools(gen_managers, seed=None)

    config = CacheTransceiverConfig(
        backend="NIXL", transceiver_runtime="PYTHON", max_tokens_in_buffer=4096
    )
    ctx_tcs = create_instance_transceivers(ctx_tp, 1, ctx_dp, ctx_managers, config)
    gen_tcs = create_instance_transceivers(gen_tp, 1, gen_dp, gen_managers, config)
    try:
        endpoint = get_ctx_info_endpoint(ctx_tcs[0])
        ctx_handle: Dict[int, List[Tuple[int, LlmRequest]]] = {r: [] for r in range(ctx_tp)}
        gen_handle: Dict[int, List[Tuple[int, LlmRequest]]] = {r: [] for r in range(gen_tp)}
        for idx, prompt_len in enumerate(REQUEST_LENGTHS):
            unique_rid = uuid.uuid4().int & 0x7FFFFFFFFFFFFFFF
            ctx_req = make_request(2 * idx, prompt_len, LlmRequestType.LLMREQUEST_TYPE_CONTEXT_ONLY)
            ctx_req.py_disaggregated_params = DisaggregatedParams(disagg_request_id=unique_rid)
            gen_req = make_request(
                2 * idx + 1, prompt_len, LlmRequestType.LLMREQUEST_TYPE_GENERATION_ONLY
            )
            gen_req.py_disaggregated_params = DisaggregatedParams(
                ctx_request_id=2 * idx,
                ctx_dp_rank=idx % ctx_tp if ctx_dp else 0,
                ctx_info_endpoint=endpoint,
                disagg_request_id=unique_rid,
            )
            for rank in range(ctx_tp):
                if not ctx_dp or idx % ctx_tp == rank:
                    ctx_handle[rank].append((idx, ctx_req))
            for rank in range(gen_tp):
                if not gen_dp or idx % gen_tp == rank:
                    gen_handle[rank].append((idx, gen_req))

        for rank, items in gen_handle.items():
            for _, req in items:
                assert gen_managers[rank].prepare_disagg_gen_init(req)
        ctx_batches = {}
        for rank, items in ctx_handle.items():
            batch = ScheduledRequests()
            for _, req in items:
                assert ctx_managers[rank].prepare_context(req)
                assert ctx_managers[rank].resize_context(req, req.context_chunk_size)
                batch.context_requests_last_chunk.append(req)
            ctx_batches[rank] = batch
        torch.cuda.synchronize()

        # What prefill left in the context pools, read before anything is released.
        truth = {
            (rank, idx): snapshot(ctx_managers[rank], req.py_request_id, req.prompt_len)
            for rank, items in ctx_handle.items()
            for idx, req in items
        }

        for handle in (ctx_handle, gen_handle):
            seen = set()
            for items in handle.values():
                for _, req in items:
                    if req.py_request_id not in seen:
                        req.context_current_position = req.prompt_len
                        req.add_new_token(req.prompt_len, 0)
                        seen.add(req.py_request_id)

        # Production order (py_executor): commit the context history, then send.
        if update_before:
            for rank, batch in ctx_batches.items():
                ctx_managers[rank].update_context_resources(batch)
        for rank, items in gen_handle.items():
            for _, req in items:
                gen_tcs[rank].request_and_receive_async(req)
        for rank, items in ctx_handle.items():
            for _, req in items:
                ctx_tcs[rank].respond_and_send_async(req)
        run_concurrent(
            ctx_tcs, lambda tc: tc.check_context_transfer_status(None, mark_complete=True)
        )
        run_concurrent(gen_tcs, lambda tc: tc.check_gen_transfer_status(None))
        if not update_before:
            for rank, batch in ctx_batches.items():
                ctx_managers[rank].update_context_resources(batch)
        torch.cuda.synchronize()

        failures = []
        compared = 0
        for gen_rank, items in gen_handle.items():
            for idx, gen_req in items:
                ctx_rank = idx % ctx_tp if ctx_dp else gen_rank % ctx_tp
                expected = truth[(ctx_rank, idx)]
                got = snapshot(gen_managers[gen_rank], gen_req.py_request_id, gen_req.prompt_len)
                for key, want_slots in expected.items():
                    layer, attn_type = key
                    for slot, want in want_slots.items():
                        have = got[key][slot]
                        if want is None:
                            failures.append(
                                (gen_req.prompt_len, layer, attn_type.name, slot, "ctx unmapped")
                            )
                        elif have is None:
                            failures.append(
                                (gen_req.prompt_len, layer, attn_type.name, slot, "gen unmapped")
                            )
                        elif not torch.equal(have, want):
                            failures.append(
                                (gen_req.prompt_len, layer, attn_type.name, slot, "bytes differ")
                            )
                        else:
                            compared += 1
        assert compared > 0
        assert not failures, (
            f"{len(failures)} slot(s) did not cross ({compared} did): {failures[:20]}"
        )
        print(f"compared {compared} slots byte-exact")
    finally:
        for tc in ctx_tcs + gen_tcs:
            try:
                tc.shutdown()
            except Exception:
                pass
        for mgr in ctx_managers + gen_managers:
            try:
                mgr.shutdown()
            except Exception:
                pass


@skip_pre_blackwell
@pytest.mark.timeout(300)
@pytest.mark.parametrize(
    "ctx_tp,gen_tp,ctx_dp,gen_dp,update_before,kv_storage",
    [
        (1, 1, False, False, True, "bf16"),
        (1, 1, False, False, False, "bf16"),
        (4, 4, False, False, True, "bf16"),
        (4, 4, True, True, True, "bf16"),
        (4, 4, False, False, True, "native"),
    ],
    ids=["tp1", "tp1_update_after", "tp4", "dep4", "tp4_native"],
)
def test_deepseek_v41_kv_transfer(
    ctx_tp, gen_tp, ctx_dp, gen_dp, update_before, kv_storage, monkeypatch
):
    monkeypatch.setenv(DEEPSEEK_V41_KV_STORAGE_ENV, kv_storage)
    run_v41_transfer(ctx_tp, gen_tp, ctx_dp, gen_dp, update_before)
