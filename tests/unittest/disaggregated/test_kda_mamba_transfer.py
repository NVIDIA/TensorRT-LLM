# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Kimi K3 KDA-shaped recurrent-state transfer through the Python V2 transceiver.

Synthetic loopback coverage of the KDA recurrent-state transfer path:

* KDA slot layout, mapped onto the Mamba cache-manager parametrization the
  way ``_util.py`` does for ``kimi_linear``:
    - short-conv slot  ``[3*H*hd, W]``  **bf16** (qwen3_next ``[Q|K|V]``
      3-section layout, all sections equal width),
    - delta-rule slot  ``[H, hd, hd]``  **fp32**  (``state_size == head_dim``),
    - EP pre-scale: ``num_heads``/``n_groups`` multiplied by ``tp_size`` when
      attention-DP is off, so the per-rank state is full-size (replicated).

* ``test_kda_layer_group_descriptors`` checks that the V2 page table
  describes BOTH slots with exact byte sizes and that the matched-TP
  ``MambaPolicy`` descriptors tile each layer's slot bytes exactly.

* ``test_kda_transfer`` performs a real single-node NIXL loopback transfer
  and bitwise-compares both dtypes on the gen side.

* ``test_kda_hetero_tp_rejected`` asserts the peer-registration
  guard: with replicated (pre-scaled) state, heterogeneous
  ctx/gen TP (attention-DP off) is rejected instead of producing fragment
  pointers outside the slot; supported layouts still validate.
"""

import uuid
from typing import Dict, List

import pytest
import torch
from test_mamba_transfer import _create_transceivers, _run_concurrent

import tensorrt_llm
import tensorrt_llm.bindings
import tensorrt_llm.tensorrt_llm_transfer_agent_binding  # noqa: F401
from tensorrt_llm import DisaggregatedParams, Mapping, SamplingParams
from tensorrt_llm._torch.disaggregation.native.mixers.ssm.peer import MambaPolicy
from tensorrt_llm._torch.disaggregation.native.peer import PeerRegistrar
from tensorrt_llm._torch.disaggregation.native.rank_info import RankInfo
from tensorrt_llm._torch.disaggregation.resource.kv_extractor import (
    KVRegionExtractorV1,
    build_page_table_from_manager,
)
from tensorrt_llm._torch.disaggregation.resource.page import MambaLayerGroup
from tensorrt_llm._torch.pyexecutor.kv_cache.mamba_cache_manager import MixedMambaHybridCacheManager
from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequest, LlmRequestType
from tensorrt_llm._torch.pyexecutor.scheduler import ScheduledRequests
from tensorrt_llm.bindings import DataType
from tensorrt_llm.bindings.internal.batch_manager import CacheType as CacheTypeCpp
from tensorrt_llm.llmapi.llm_args import CacheTransceiverConfig, KvCacheConfig

# ---------------------------------------------------------------------------
# KDA parameters (small stand-ins for Kimi K3's per-layer KDA state)
# ---------------------------------------------------------------------------
NUM_KDA_LAYERS = 4
KDA_NUM_HEADS = 4  # H
KDA_HEAD_DIM = 32  # hd; KDA state_size == head_dim
KDA_W = 3  # short_conv_kernel_size; manager d_conv = W + 1
CONV_DTYPE = torch.bfloat16
SSM_DTYPE = torch.float32
MAX_BATCH_SIZE = 4
REQUEST_LENGTHS = [16, 32]

# Internal: layer 0 is a dummy attention layer required by page table infra;
# layers 1..NUM_KDA_LAYERS are KDA (under test).
_NUM_TOTAL_LAYERS = NUM_KDA_LAYERS + 1
_KDA_MASK = [False] + [True] * NUM_KDA_LAYERS
_ATTN_MASK = [True] + [False] * NUM_KDA_LAYERS

# Full per-layer slot byte sizes (replicated on every rank for K3).
CONV_SLOT_BYTES = 3 * KDA_NUM_HEADS * KDA_HEAD_DIM * KDA_W * CONV_DTYPE.itemsize
SSM_SLOT_BYTES = KDA_NUM_HEADS * KDA_HEAD_DIM * KDA_HEAD_DIM * SSM_DTYPE.itemsize


def _create_kda_managers(
    tp: int, enable_attention_dp: bool = False, max_batch_size: int = MAX_BATCH_SIZE
):
    """Create MixedMambaHybridCacheManagers with K3-style KDA slots.

    Mirrors the ``is_kimi_linear`` route in ``_util.py:1855-1915``: KDA is
    mapped onto (state_size=hd, conv_kernel=W+1, num_heads=H, n_groups=H,
    head_dim=hd, model_type='qwen3_next'), and ``num_heads``/``n_groups``
    are pre-scaled by tp_size when attention-DP is off so the per-rank
    state stays full-size (EP-only parallelism, replicated KDA state).
    """
    state_tp = tp if not enable_attention_dp else 1
    managers = []
    for rank in range(tp):
        mapping = Mapping(
            world_size=tp,
            rank=rank,
            tp_size=tp,
            pp_size=1,
            enable_attention_dp=enable_attention_dp,
        )
        mgr = MixedMambaHybridCacheManager(
            mamba_d_state=KDA_HEAD_DIM,
            mamba_d_conv=KDA_W + 1,
            mamba_num_heads=KDA_NUM_HEADS * state_tp,
            mamba_n_groups=KDA_NUM_HEADS * state_tp,
            mamba_head_dim=KDA_HEAD_DIM,
            mamba_num_layers=NUM_KDA_LAYERS,
            mamba_layer_mask=_KDA_MASK,
            mamba_cache_dtype=CONV_DTYPE,
            mamba_ssm_cache_dtype=SSM_DTYPE,
            # dummy attention layer (page table scaffolding)
            kv_cache_config=KvCacheConfig(
                max_tokens=256 * max_batch_size,
                enable_block_reuse=False,
                event_buffer_max_size=0,
            ),
            kv_cache_type=CacheTypeCpp.SELF,
            num_layers=1,
            layer_mask=_ATTN_MASK,
            num_kv_heads=4,
            head_dim=64,
            tokens_per_block=8,
            max_seq_len=256,
            max_batch_size=max_batch_size,
            mapping=mapping,
            dtype=DataType.FLOAT,
            model_type="qwen3_next",
        )
        managers.append(mgr)
    return managers


def _get_mamba_layer_group(page_table) -> MambaLayerGroup:
    mlgs = [lg for lg in page_table.layer_groups if isinstance(lg, MambaLayerGroup)]
    assert len(mlgs) == 1, f"expected exactly one MambaLayerGroup, got {len(mlgs)}"
    return mlgs[0]


def _layer_slot_start(pool, local_layer_idx: int, slot: int) -> int:
    return pool.base_address + (local_layer_idx * pool.num_slots + slot) * pool.slot_bytes


def _assert_frags_tile_slots(frags, mlg: MambaLayerGroup, slot: int):
    """Assert (ptr, size) frags exactly tile every layer's slot bytes."""
    ptrs, sizes = frags
    covered: Dict[int, List[tuple]] = {}
    for ptr, size in zip(ptrs, sizes):
        pool = None
        for cand in (mlg.conv_states, mlg.ssm_states):
            lo = cand.base_address
            hi = lo + len(mlg.mamba_layer_offsets) * cand.num_slots * cand.slot_bytes
            if lo <= ptr < hi:
                pool = cand
                break
        assert pool is not None, f"frag ptr {ptr} outside both mamba pools"
        rel = ptr - pool.base_address
        layer = rel // (pool.num_slots * pool.slot_bytes)
        in_layer = rel - layer * pool.num_slots * pool.slot_bytes
        frag_slot = in_layer // pool.slot_bytes
        off = in_layer - frag_slot * pool.slot_bytes
        assert frag_slot == slot, f"frag targets slot {frag_slot}, expected {slot}"
        assert off + size <= pool.slot_bytes, (
            f"frag [{off}, {off + size}) exceeds slot_bytes {pool.slot_bytes}"
        )
        covered.setdefault((id(pool), int(layer)), []).append((off, off + size))

    # Exact tiling per (pool, layer): sorted intervals must be contiguous
    # from 0 to slot_bytes with no overlap.
    n_layers = len(mlg.mamba_layer_offsets)
    for pool in (mlg.conv_states, mlg.ssm_states):
        for layer in range(n_layers):
            intervals = sorted(covered.get((id(pool), layer), []))
            assert intervals, f"layer {layer} of pool not covered"
            pos = 0
            for lo, hi in intervals:
                assert lo == pos, f"gap/overlap at byte {pos} (next frag at {lo})"
                pos = hi
            assert pos == pool.slot_bytes, (
                f"layer {layer}: covered {pos} of {pool.slot_bytes} slot bytes"
            )


# ---------------------------------------------------------------------------
# Descriptor-level tests (no NIXL transfer)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("enable_attention_dp", [True, False], ids=["adp_on", "adp_off_tp1"])
def test_kda_layer_group_descriptors(enable_attention_dp):
    """V2 page table must describe BOTH KDA slots with exact byte extents."""
    mgr = _create_kda_managers(1, enable_attention_dp=enable_attention_dp)[0]
    try:
        pt = build_page_table_from_manager(mgr)
        mlg = _get_mamba_layer_group(pt)

        conv = mgr._impl.mamba_cache.conv
        ssm = mgr._impl.mamba_cache.temporal

        # Shapes/dtypes of the backing tensors are KDA-shaped.
        assert conv.dtype == CONV_DTYPE
        assert ssm.dtype == SSM_DTYPE
        assert tuple(conv.shape[2:]) == (3 * KDA_NUM_HEADS * KDA_HEAD_DIM, KDA_W)
        assert tuple(ssm.shape[2:]) == (KDA_NUM_HEADS, KDA_HEAD_DIM, KDA_HEAD_DIM)

        # Both pools present, byte-exact (bf16 conv, fp32 delta).
        assert mlg.conv_states is not None and mlg.ssm_states is not None
        assert mlg.conv_states.base_address == conv.data_ptr()
        assert mlg.ssm_states.base_address == ssm.data_ptr()
        assert mlg.conv_states.slot_bytes == CONV_SLOT_BYTES
        assert mlg.ssm_states.slot_bytes == SSM_SLOT_BYTES
        assert mlg.conv_states.num_slots == conv.shape[1]
        assert mlg.ssm_states.num_slots == ssm.shape[1]

        # qwen3_next 3-sectioning: equal sections summing to the conv slot.
        assert mlg.conv_section_bytes == [CONV_SLOT_BYTES // 3] * 3
        assert sum(mlg.conv_section_bytes) == mlg.conv_states.slot_bytes
        assert mlg.ssm_bytes_per_head == KDA_HEAD_DIM * KDA_HEAD_DIM * SSM_DTYPE.itemsize
        assert mlg.ssm_states.slot_bytes // mlg.ssm_bytes_per_head == KDA_NUM_HEADS

        # Layer offsets cover exactly the KDA layers.
        assert sorted(mlg.mamba_layer_offsets.keys()) == [i for i, m in enumerate(_KDA_MASK) if m]

        # Matched-parallelism frags: 2 pools x layers, full-slot, exact tiling.
        ri = RankInfo.from_kv_cache_manager("kda_test", mgr, device_id=0)
        src_slot, dst_slot = 0, 1
        src_frags, dst_frags, sizes = MambaPolicy.build_mamba_frags(
            mlg, mlg, src_slot, dst_slot, ri, ri
        )
        assert len(src_frags) == len(dst_frags) == len(sizes) == 2 * NUM_KDA_LAYERS
        _assert_frags_tile_slots((src_frags, sizes), mlg, src_slot)
        _assert_frags_tile_slots((dst_frags, sizes), mlg, dst_slot)

        # Full-slot copies at the expected per-layer addresses.
        for glid, lid in sorted(mlg.mamba_layer_offsets.items()):
            assert _layer_slot_start(mlg.conv_states, lid, src_slot) in src_frags
            assert _layer_slot_start(mlg.ssm_states, lid, src_slot) in src_frags
    finally:
        mgr.shutdown()


def test_kda_hetero_tp_rejected():
    """Replicated KDA state + heterogeneous TP (ADP off) must be rejected.

    Hetero ctx/gen TP would silently corrupt replicated state: the TP-mismatch
    mappers assume sharded state and would compute shard offsets past the
    end of K3's replicated (pre-scaled) slots. The peer-registration gate
    (``MambaPolicy.validate_peer_compatible``) must reject this loudly.
    """
    ctx_mgrs = _create_kda_managers(2, enable_attention_dp=False)
    gen_mgrs = _create_kda_managers(4, enable_attention_dp=False)
    ctx_mgr, gen_mgr = ctx_mgrs[0], gen_mgrs[1]
    try:
        ctx_pt = build_page_table_from_manager(ctx_mgr)
        gen_pt = build_page_table_from_manager(gen_mgr)
        ctx_ri = RankInfo.from_kv_cache_manager("kda_ctx", ctx_mgr, device_id=0)
        gen_ri = RankInfo.from_kv_cache_manager("kda_gen", gen_mgr, device_id=0)

        with pytest.raises(ValueError, match="TP-aggregated"):
            MambaPolicy.validate_peer_compatible(ctx_ri, gen_ri, ctx_pt, gen_pt)

        # And via the registrar entry point used at runtime.
        registrar = PeerRegistrar(ctx_ri, KVRegionExtractorV1(ctx_pt))
        with pytest.raises(ValueError, match="TP-aggregated"):
            registrar.register("kda_gen", 1, gen_ri)
    finally:
        for mgr in ctx_mgrs + gen_mgrs:
            mgr.shutdown()


@pytest.mark.parametrize(
    "ctx_cfg,gen_cfg",
    [
        ((2, False), (2, False)),  # matched TP, ADP off (EP pre-scaled)
        ((2, True), (4, True)),  # heterogeneous DEP with ADP on both sides
    ],
    ids=["matched_tp2_adp_off", "hetero_dep_adp_on"],
)
def test_kda_peer_validation_accepts_supported_shapes(ctx_cfg, gen_cfg):
    """Matched-TP and ADP-on-both-sides layouts must pass peer validation."""
    ctx_tp, ctx_adp = ctx_cfg
    gen_tp, gen_adp = gen_cfg
    ctx_mgrs = _create_kda_managers(ctx_tp, enable_attention_dp=ctx_adp)
    gen_mgrs = _create_kda_managers(gen_tp, enable_attention_dp=gen_adp)
    ctx_mgr, gen_mgr = ctx_mgrs[0], gen_mgrs[-1]
    try:
        ctx_pt = build_page_table_from_manager(ctx_mgr)
        gen_pt = build_page_table_from_manager(gen_mgr)
        ctx_ri = RankInfo.from_kv_cache_manager("kda_ctx", ctx_mgr, device_id=0)
        gen_ri = RankInfo.from_kv_cache_manager("kda_gen", gen_mgr, device_id=0)
        MambaPolicy.validate_peer_compatible(ctx_ri, gen_ri, ctx_pt, gen_pt)
    finally:
        for mgr in ctx_mgrs + gen_mgrs:
            mgr.shutdown()


def _synthetic_kda_page_table(ssm_slot_bytes: int, conv_slot_bytes: int, layer_ids=None):
    """MambaLayerGroup-only page table with fake addresses (no CUDA needed)."""
    from tensorrt_llm._torch.disaggregation.resource.page import KVCachePageTable, PhysicalPool

    if layer_ids is None:
        layer_ids = range(1, NUM_KDA_LAYERS + 1)
    mlg = MambaLayerGroup(
        pool_group_idx=0,
        mamba_layer_offsets={glid: i for i, glid in enumerate(layer_ids)},
        conv_states=PhysicalPool(base_address=0x1000, slot_bytes=conv_slot_bytes, num_slots=8),
        ssm_states=PhysicalPool(base_address=0x2000000, slot_bytes=ssm_slot_bytes, num_slots=8),
        conv_section_bytes=[conv_slot_bytes // 3] * 3,
        ssm_bytes_per_head=KDA_HEAD_DIM * KDA_HEAD_DIM * SSM_DTYPE.itemsize,
    )
    return KVCachePageTable(tokens_per_block=8, layer_groups=[mlg], pool_groups=[])


def _synthetic_rank_info(tp: int, adp: bool):
    from tensorrt_llm._torch.disaggregation.native.mixers.attention.spec import AttentionInfo

    return RankInfo(
        instance_name="syn",
        instance_rank=0,
        tp_size=tp,
        tp_rank=0,
        pp_size=1,
        pp_rank=0,
        layer_num_per_pp=[_NUM_TOTAL_LAYERS],
        sender_endpoints=[],
        self_endpoint="",
        transfer_engine_info=b"",
        attention=AttentionInfo(
            kv_heads_per_rank=4,
            tokens_per_block=8,
            dims_per_head=64,
            element_bytes=2,
            enable_attention_dp=adp,
            is_mla=False,
        ),
    )


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "ctx,gen,ok",
    [
        # (tp, adp, slot_scale_denominator): replicated K3 state = full slots.
        ((2, False, 1), (4, False, 1), False),  # hetero TP, ADP off: would corrupt state
        ((4, False, 1), (2, False, 1), False),  # ...both directions
        ((2, True, 1), (2, False, 1), False),  # mixed ADP, replicated
        ((2, False, 1), (2, False, 1), True),  # matched TP
        ((2, True, 1), (4, True, 1), True),  # hetero DEP, ADP on both
        ((2, False, 2), (4, False, 4), True),  # sharded state, hetero TP
    ],
    ids=[
        "reject_hetero_tp_adp_off",
        "reject_hetero_tp_adp_off_rev",
        "reject_mixed_adp_replicated",
        "accept_matched_tp",
        "accept_hetero_dep_adp_on",
        "accept_sharded_hetero_tp",
    ],
)
def test_kda_peer_validation_synthetic_cpu(ctx, gen, ok):
    """CPU-only reject/accept matrix for the gap-F1 guard (no CUDA manager)."""
    full_ssm = KDA_NUM_HEADS * KDA_HEAD_DIM * KDA_HEAD_DIM * SSM_DTYPE.itemsize
    full_conv = 3 * KDA_NUM_HEADS * KDA_HEAD_DIM * KDA_W * CONV_DTYPE.itemsize

    def build(cfg):
        tp, adp, denom = cfg
        return (
            _synthetic_rank_info(tp, adp),
            _synthetic_kda_page_table(full_ssm // denom, full_conv // denom),
        )

    ctx_ri, ctx_pt = build(ctx)
    gen_ri, gen_pt = build(gen)
    if ok:
        MambaPolicy.validate_peer_compatible(ctx_ri, gen_ri, ctx_pt, gen_pt)
    else:
        with pytest.raises(ValueError, match="TP-aggregated"):
            MambaPolicy.validate_peer_compatible(ctx_ri, gen_ri, ctx_pt, gen_pt)


@pytest.mark.cpu_only
def test_kda_peer_validation_allows_pipeline_parallel_layer_split():
    """Peer validation must not require identical layer sets.

    Under pipeline parallelism each rank publishes only its own stage's
    layers, and the transfer path takes the intersection of the two layer
    sets. Peers with partially overlapping, disjoint, or one-sided
    recurrent-layer sets are all legitimate stage pairings and must pass
    validation as long as the per-slot size invariants hold.
    """
    full_ssm = KDA_NUM_HEADS * KDA_HEAD_DIM * KDA_HEAD_DIM * SSM_DTYPE.itemsize
    full_conv = 3 * KDA_NUM_HEADS * KDA_HEAD_DIM * KDA_W * CONV_DTYPE.itemsize
    ctx_ri = _synthetic_rank_info(2, False)
    gen_ri = _synthetic_rank_info(2, False)

    # Partially overlapping stages (ctx PP split differs from gen's).
    ctx_pt = _synthetic_kda_page_table(full_ssm, full_conv, layer_ids=[1, 2, 3])
    gen_pt = _synthetic_kda_page_table(full_ssm, full_conv, layer_ids=[3, 4, 5])
    MambaPolicy.validate_peer_compatible(ctx_ri, gen_ri, ctx_pt, gen_pt)

    # Disjoint stages: this pair simply moves no recurrent state.
    gen_pt = _synthetic_kda_page_table(full_ssm, full_conv, layer_ids=[7, 8])
    MambaPolicy.validate_peer_compatible(ctx_ri, gen_ri, ctx_pt, gen_pt)

    # One side holds a recurrent-layer-free stage of the hybrid model
    # (no MambaLayerGroup at all): nothing to validate.
    from tensorrt_llm._torch.disaggregation.resource.page import KVCachePageTable

    empty_pt = KVCachePageTable(tokens_per_block=8, layer_groups=[], pool_groups=[])
    MambaPolicy.validate_peer_compatible(ctx_ri, gen_ri, ctx_pt, empty_pt)
    MambaPolicy.validate_peer_compatible(ctx_ri, gen_ri, empty_pt, gen_pt)

    # Size invariants still apply on the overlap: mismatched global state
    # sizes are rejected even when the layer sets differ.
    bad_gen_pt = _synthetic_kda_page_table(full_ssm // 2, full_conv // 2, layer_ids=[3, 4, 5])
    with pytest.raises(ValueError, match="TP-aggregated"):
        MambaPolicy.validate_peer_compatible(ctx_ri, gen_ri, ctx_pt, bad_gen_pt)


# ---------------------------------------------------------------------------
# Loopback transfer over a real NIXL agent (single node)
# ---------------------------------------------------------------------------
def _generate_ground_truth(num_requests: int, seed: int = 20260722):
    """Full replicated KDA states per (request, kda_layer), two dtypes."""
    gen = torch.Generator(device="cpu").manual_seed(seed)
    results = []
    for _ in range(num_requests):
        layers = {}
        for i, is_kda in enumerate(_KDA_MASK):
            if not is_kda:
                continue
            layers[i] = {
                "conv": torch.rand(3 * KDA_NUM_HEADS * KDA_HEAD_DIM, KDA_W, generator=gen).to(
                    CONV_DTYPE
                ),
                "ssm": torch.rand(
                    KDA_NUM_HEADS,
                    KDA_HEAD_DIM,
                    KDA_HEAD_DIM,
                    generator=gen,
                    dtype=SSM_DTYPE,
                ),
            }
        results.append(layers)
    return results


def run_kda_transfer_test(ctx_tp: int, gen_tp: int, enable_attention_dp: bool = False):
    """Loopback: matched ctx/gen TP, replicated full-size KDA state per rank."""
    ctx_mgrs = _create_kda_managers(ctx_tp, enable_attention_dp=enable_attention_dp)
    gen_mgrs = _create_kda_managers(gen_tp, enable_attention_dp=enable_attention_dp)
    ctx_tcs, gen_tcs = [], []
    try:
        for mgr in ctx_mgrs + gen_mgrs:
            mgr._impl.mamba_cache.conv.zero_()
            mgr._impl.mamba_cache.temporal.zero_()

        config = CacheTransceiverConfig(
            backend="NIXL",
            transceiver_runtime="PYTHON",
            max_tokens_in_buffer=512,
        )
        ctx_tcs = _create_transceivers(ctx_tp, ctx_mgrs, config)
        gen_tcs = _create_transceivers(gen_tp, gen_mgrs, config)
        ctx_endpoint = ctx_tcs[0]._context_info_endpoint

        sampling_params = SamplingParams()
        ctx_rids, gen_rids, ctx_reqs, gen_reqs = [], [], [], []
        for req_idx, req_len in enumerate(REQUEST_LENGTHS):
            unique_rid = uuid.uuid4().int & 0x7FFFFFFFFFFFFFFF
            ctx_rid, gen_rid = req_idx * 2, req_idx * 2 + 1
            ctx_rids.append(ctx_rid)
            gen_rids.append(gen_rid)
            sc = tensorrt_llm.bindings.SamplingConfig(sampling_params._get_sampling_config())
            ctx_req = LlmRequest(
                request_id=ctx_rid,
                max_new_tokens=1,
                input_tokens=list(range(req_len)),
                sampling_config=sc,
                is_streaming=False,
                llm_request_type=LlmRequestType.LLMREQUEST_TYPE_CONTEXT_ONLY,
            )
            ctx_req.py_disaggregated_params = DisaggregatedParams(disagg_request_id=unique_rid)
            gen_req = LlmRequest(
                request_id=gen_rid,
                max_new_tokens=1,
                input_tokens=list(range(req_len)),
                sampling_config=sc,
                is_streaming=False,
                llm_request_type=LlmRequestType.LLMREQUEST_TYPE_GENERATION_ONLY,
            )
            gen_req.py_disaggregated_params = DisaggregatedParams(
                ctx_request_id=ctx_rid,
                ctx_dp_rank=0,
                ctx_info_endpoint=ctx_endpoint,
                disagg_request_id=unique_rid,
            )
            ctx_reqs.append(ctx_req)
            gen_reqs.append(gen_req)

        ctx_batch = ScheduledRequests()
        ctx_batch.reset_context_requests(ctx_reqs)
        for mgr in ctx_mgrs:
            mgr.prepare_resources(ctx_batch)
        gen_batch = ScheduledRequests()
        gen_batch.reset_context_requests(gen_reqs)
        for mgr in gen_mgrs:
            mgr.prepare_resources(gen_batch)
        for req in ctx_reqs + gen_reqs:
            req.context_current_position = req.prompt_len
            req.add_new_token(req.prompt_len, 0)
        for mgr in ctx_mgrs:
            mgr.update_resources(ctx_batch)
        for mgr in gen_mgrs:
            mgr.update_resources(gen_batch)

        # Ground truth: identical full state on every ctx rank (replicated).
        ground_truth = _generate_ground_truth(len(REQUEST_LENGTHS))
        for mgr in ctx_mgrs:
            for req_idx, rid in enumerate(ctx_rids):
                slot = mgr.mamba_cache_index[rid]
                for layer_idx in mgr._impl.mamba_layer_offsets:
                    full = ground_truth[req_idx][layer_idx]
                    mgr.get_conv_states(layer_idx)[slot] = full["conv"]
                    mgr.get_ssm_states(layer_idx)[slot] = full["ssm"]

        for rank in range(gen_tp):
            for req in gen_reqs:
                gen_tcs[rank].request_and_receive_async(req)
        for rank in range(ctx_tp):
            for req in ctx_reqs:
                ctx_tcs[rank].respond_and_send_async(req)
        _run_concurrent(
            ctx_tcs, lambda tc: tc.check_context_transfer_status(None, mark_complete=True)
        )
        _run_concurrent(gen_tcs, lambda tc: tc.check_gen_transfer_status(None))

        # Transfer-size metric must include the fixed-size KDA state — the actual
        # transferred bytes must cover the computed payload size: per rank,
        # num_layers * (conv + ssm) slot bytes on top of any KV bytes.
        kda_bytes_per_rank = NUM_KDA_LAYERS * (CONV_SLOT_BYTES + SSM_SLOT_BYTES)
        rank_factor = 1 if enable_attention_dp else gen_tp
        for req in gen_reqs:
            assert req.py_kv_cache_xfer_bytes >= kda_bytes_per_rank * rank_factor, (
                f"kv_cache_xfer_bytes={req.py_kv_cache_xfer_bytes} misses the KDA "
                f"state payload ({kda_bytes_per_rank} bytes/rank x {rank_factor})"
            )

        # Bitwise comparison on every gen rank (state is replicated).
        for gen_rank, mgr in enumerate(gen_mgrs):
            for req_idx, rid in enumerate(gen_rids):
                slot = mgr.mamba_cache_index[rid]
                for layer_idx in mgr._impl.mamba_layer_offsets:
                    full = ground_truth[req_idx][layer_idx]
                    for name, getter in (
                        ("conv", mgr.get_conv_states),
                        ("ssm", mgr.get_ssm_states),
                    ):
                        torch.testing.assert_close(
                            getter(layer_idx)[slot].cpu(),
                            full[name],
                            rtol=0,
                            atol=0,
                            msg=lambda m, n=name, r=gen_rank, ri=req_idx, li=layer_idx: (
                                f"{n} mismatch: gen_rank={r} req={ri} layer={li} "
                                f"ctx_tp={ctx_tp} gen_tp={gen_tp}: {m}"
                            ),
                        )

    finally:
        for tc in ctx_tcs + gen_tcs:
            tc.shutdown()
        for mgr in ctx_mgrs + gen_mgrs:
            mgr.shutdown()


@pytest.mark.timeout(180)
@pytest.mark.parametrize(
    "ctx_tp,gen_tp",
    [(1, 1), (2, 2)],
    ids=["tp1_tp1", "tp2_tp2_ep_prescaled"],
)
def test_kda_transfer(ctx_tp, gen_tp):
    """KDA two-dtype state transfer, matched parallelism, real NIXL loopback."""
    run_kda_transfer_test(ctx_tp, gen_tp)
