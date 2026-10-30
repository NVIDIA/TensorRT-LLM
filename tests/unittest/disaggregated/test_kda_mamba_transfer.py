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
  way ``_util.py`` does for ``kimi_linear`` (params come from
  ``extract_mamba_kv_cache_params`` UNSCALED — global head counts):
    - short-conv slot  ``[3*H*hd, W-1]``  **bf16** (qwen3_next ``[Q|K|V]``
      3-section layout, all sections equal width),
    - delta-rule slot  ``[H, hd, hd]``  **fp32**  (``state_size == head_dim``),
    - TP semantics: the manager itself gates on the mapping
      (``tp_size = 1`` under attention-DP, else ``mapping.tp_size``) and
      divides ``num_heads``/``n_groups``/``conv_dim`` by it. So the per-rank
      state is a head shard when attention-DP is off (matching the model's
      head-sharded KDA compute in ``KimiKDARuntime``) and a full-size
      replica under attention-DP.

* ``test_kda_layer_group_descriptors`` checks that the V2 page table
  describes BOTH slots with exact byte sizes and that a matched-TP
  receiver receives each layer's slot bytes exactly.

* ``test_kda_transfer`` performs a real single-node NIXL loopback transfer
  and bitwise-compares both dtypes on the gen side: matched and
  heterogeneous ctx/gen TP, attention-DP on and off.

* ``test_kda_hetero_tp_sharded_accepted`` asserts that peer registration
  accepts heterogeneous ctx/gen TP with attention-DP off for what
  production actually builds (head-sharded per-rank state, so the global
  TP-aggregated size invariant holds).
"""

import uuid
from types import SimpleNamespace
from typing import List

import pytest
import torch
from test_mamba_transfer import _create_transceivers, _run_concurrent

import tensorrt_llm
import tensorrt_llm.bindings
import tensorrt_llm.tensorrt_llm_transfer_agent_binding  # noqa: F401
from tensorrt_llm import DisaggregatedParams, Mapping, SamplingParams
from tensorrt_llm._torch.disaggregation.native.peer import PeerRegistrar
from tensorrt_llm._torch.disaggregation.native.rank_info import RankInfo
from tensorrt_llm._torch.pyexecutor.config_utils import extract_mamba_kv_cache_params
from tensorrt_llm._torch.pyexecutor.kv_cache.layout_mapping import check_compatible, sent_bytes
from tensorrt_llm._torch.pyexecutor.kv_cache.mamba_cache_manager import (
    MambaHybridCacheManagerV2,
    MambaRole,
)
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
KDA_W = 3  # short_conv_kernel_size
KDA_CONV_STATE_WIDTH = KDA_W - 1
CONV_DTYPE = torch.bfloat16
SSM_DTYPE = torch.float32
MAX_BATCH_SIZE = 4
REQUEST_LENGTHS = [16, 32]

# Internal: layer 0 is a dummy attention layer required by page table infra;
# layers 1..NUM_KDA_LAYERS are KDA (under test).
_NUM_TOTAL_LAYERS = NUM_KDA_LAYERS + 1
_KDA_MASK = [False] + [True] * NUM_KDA_LAYERS
_ATTN_MASK = [True] + [False] * NUM_KDA_LAYERS

# Global (TP-aggregated) per-layer slot byte sizes. Per-rank slot bytes are
# these divided by tp_size when attention-DP is off (head-sharded state) and
# equal to these under attention-DP (replicated state).
CONV_SLOT_BYTES = 3 * KDA_NUM_HEADS * KDA_HEAD_DIM * KDA_CONV_STATE_WIDTH * CONV_DTYPE.itemsize
SSM_SLOT_BYTES = KDA_NUM_HEADS * KDA_HEAD_DIM * KDA_HEAD_DIM * SSM_DTYPE.itemsize


def _kimi_linear_config():
    """Minimal ``kimi_linear`` HF-config stand-in for the production mapping.

    Just enough attributes for ``extract_mamba_kv_cache_params`` (the
    ``is_kimi_linear`` route): KDA layers are 1..NUM_KDA_LAYERS+1 (1-indexed),
    layer 1 is the dummy full-attention layer.
    """
    kda_layers = [i + 1 for i, m in enumerate(_KDA_MASK) if m]
    full_attn_layers = [i + 1 for i, m in enumerate(_KDA_MASK) if not m]
    return SimpleNamespace(
        model_type="kimi_linear",
        linear_attn_config={
            "head_dim": KDA_HEAD_DIM,
            "short_conv_kernel_size": KDA_W,
            "num_heads": KDA_NUM_HEADS,
            "kda_layers": kda_layers,
            "full_attn_layers": full_attn_layers,
        },
        num_hidden_layers=_NUM_TOTAL_LAYERS,
        torch_dtype=CONV_DTYPE,
    )


def _create_kda_managers(
    tp: int, enable_attention_dp: bool = False, max_batch_size: int = MAX_BATCH_SIZE
):
    """Create MambaHybridCacheManagerV2s with K3-style KDA slots.

    Mirrors the ``is_kimi_linear`` route in ``_util.py``: the mamba params
    come from ``extract_mamba_kv_cache_params`` UNSCALED (global
    ``num_heads``/``n_groups``); the manager applies its own TP gate
    (``tp_size = 1`` under attention-DP, else ``mapping.tp_size``) and
    divides ``num_heads``/``n_groups``/``conv_dim`` by it — the same
    head-shard semantics the model runtime uses (``KimiKDARuntime``
    constructs the mixer with ``num_heads // tp_size``).
    """
    params = extract_mamba_kv_cache_params(_kimi_linear_config())
    assert params.mamba_layer_mask == _KDA_MASK
    assert params.target_full_attention_layer_mask == _ATTN_MASK
    assert params.dtype == CONV_DTYPE
    assert params.mamba_ssm_cache_dtype == SSM_DTYPE  # kimi_linear forces fp32
    managers = []
    for rank in range(tp):
        mapping = Mapping(
            world_size=tp,
            rank=rank,
            tp_size=tp,
            pp_size=1,
            enable_attention_dp=enable_attention_dp,
        )
        mgr = MambaHybridCacheManagerV2(
            mamba_d_state=params.state_size,
            mamba_d_conv=params.conv_kernel,
            mamba_num_heads=params.num_heads,
            mamba_n_groups=params.n_groups,
            mamba_head_dim=params.head_dim,
            mamba_num_layers=params.num_mamba_layers,
            mamba_layer_mask=params.mamba_layer_mask,
            mamba_cache_dtype=params.dtype,
            mamba_ssm_cache_dtype=params.mamba_ssm_cache_dtype,
            # dummy attention layer (page table scaffolding)
            kv_cache_config=KvCacheConfig(
                max_tokens=256 * max_batch_size,
                enable_block_reuse=False,
                event_buffer_max_size=0,
                use_kv_cache_manager_v2=True,
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
            is_disagg=True,
            conv_state_layout="q_k_v",
        )
        managers.append(mgr)
    return managers


def _state_group(page_table) -> int:
    """The index of the page table's one recurrent-state layer group."""
    groups = [i for i, g in enumerate(page_table.page_layout.layer_groups) if g.kind == "state"]
    assert len(groups) == 1, f"expected exactly one state layer group, got {len(groups)}"
    return groups[0]


def _state_buffer(page_table, role):
    """The page layout's ``role`` buffer of the first recurrent layer."""
    return next(b for b in page_table.page_layout.buffers if b.role == str(role))


# ---------------------------------------------------------------------------
# Descriptor-level tests (no NIXL transfer)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("enable_attention_dp", [True, False], ids=["adp_on", "adp_off_tp1"])
def test_kda_layer_group_descriptors(enable_attention_dp):
    """V2 page table must describe BOTH KDA slots with exact byte extents."""
    mgr = _create_kda_managers(1, enable_attention_dp=enable_attention_dp)[0]
    try:
        pt = mgr.get_page_table()
        state = _state_group(pt)

        layer_id = next(iter(mgr.mamba_layer_offsets))
        conv = mgr.get_conv_states(layer_id)
        ssm = mgr.get_ssm_states(layer_id)

        # Shapes/dtypes of the backing tensors are KDA-shaped.
        assert conv.dtype == CONV_DTYPE
        assert ssm.dtype == SSM_DTYPE
        assert tuple(conv.shape[1:]) == (
            3 * KDA_NUM_HEADS * KDA_HEAD_DIM,
            KDA_CONV_STATE_WIDTH,
        )
        assert tuple(ssm.shape[1:]) == (KDA_NUM_HEADS, KDA_HEAD_DIM, KDA_HEAD_DIM)

        # Both state buffers present with their byte sizes.
        conv_buffer = _state_buffer(pt, MambaRole.CONV_STATE)
        ssm_buffer = _state_buffer(pt, MambaRole.SSM_STATE)
        assert conv_buffer.size == CONV_SLOT_BYTES
        assert ssm_buffer.size == SSM_SLOT_BYTES

        # qwen3_next 3-sectioning: equal sections summing to the conv slot.
        sections = [t.axes[0].held for t in conv_buffer.tensors]
        assert sections == [CONV_SLOT_BYTES // 3] * 3
        assert sum(sections) == conv_buffer.size
        (head,) = (a for a in ssm_buffer.tensors[0].axes if a.name == "head")
        assert ssm_buffer.size // head.held == KDA_HEAD_DIM * KDA_HEAD_DIM * SSM_DTYPE.itemsize
        assert head.held == KDA_NUM_HEADS

        # The state group holds exactly the KDA layers (by global layer id).
        assert sorted(pt.page_layout.layer_groups[state].layers) == [
            i for i, m in enumerate(_KDA_MASK) if m
        ]

        # Matched-parallelism payload: a receiver receives
        # NUM_KDA_LAYERS * (conv + ssm) of recurrent state.
        payload = sent_bytes(pt.page_layout, pt.page_layout)[state]
        assert payload == NUM_KDA_LAYERS * (CONV_SLOT_BYTES + SSM_SLOT_BYTES)
    finally:
        mgr.shutdown()


def test_kda_hetero_tp_sharded_accepted():
    """Heterogeneous ctx/gen TP with ADP off passes peer validation.

    With attention-DP off, production builds a head-sharded per-rank KDA
    state (the manager divides num_heads/n_groups/conv_dim by tp_size,
    matching ``KimiKDARuntime``'s head-sharded compute), so both sides
    describe the same global state and the page layouts correspond across
    hetero ctx/gen TP.
    """
    ctx_mgrs = _create_kda_managers(2, enable_attention_dp=False)
    gen_mgrs = _create_kda_managers(4, enable_attention_dp=False)
    ctx_mgr, gen_mgr = ctx_mgrs[0], gen_mgrs[1]
    try:
        ctx_pt = ctx_mgr.get_page_table()
        gen_pt = gen_mgr.get_page_table()

        # Production per-rank slots are the global sizes divided by tp.
        ctx_conv_bpl = _state_buffer(ctx_pt, MambaRole.CONV_STATE).size
        ctx_ssm_bpl = _state_buffer(ctx_pt, MambaRole.SSM_STATE).size
        gen_conv_bpl = _state_buffer(gen_pt, MambaRole.CONV_STATE).size
        gen_ssm_bpl = _state_buffer(gen_pt, MambaRole.SSM_STATE).size
        assert ctx_conv_bpl == CONV_SLOT_BYTES // 2
        assert ctx_ssm_bpl == SSM_SLOT_BYTES // 2
        assert gen_conv_bpl == CONV_SLOT_BYTES // 4
        assert gen_ssm_bpl == SSM_SLOT_BYTES // 4

        ctx_ri = RankInfo.from_kv_cache_manager("kda_ctx", ctx_mgr, device_id=0)
        gen_ri = RankInfo.from_kv_cache_manager("kda_gen", gen_mgr, device_id=0)
        check_compatible(ctx_pt.page_layout, gen_pt.page_layout)

        # And via the registrar entry point used at runtime.
        registrar = PeerRegistrar(ctx_ri)
        registrar.register("kda_gen", 1, gen_ri)
    finally:
        for mgr in ctx_mgrs + gen_mgrs:
            mgr.shutdown()


@pytest.mark.parametrize(
    "ctx_cfg,gen_cfg",
    [
        ((2, False), (2, False)),  # matched TP, ADP off (head-sharded)
        ((2, True), (4, True)),  # heterogeneous DEP with ADP on both sides
    ],
    ids=["matched_tp2_adp_off", "hetero_dep_adp_on"],
)
def test_kda_peer_validation_accepts_supported_shapes(ctx_cfg, gen_cfg):
    """Matched-TP and ADP-on-both-sides layouts must pass peer validation.

    Hetero TP with ADP off (also supported for K3's sharded state) is
    covered by ``test_kda_hetero_tp_sharded_accepted``.
    """
    ctx_tp, ctx_adp = ctx_cfg
    gen_tp, gen_adp = gen_cfg
    ctx_mgrs = _create_kda_managers(ctx_tp, enable_attention_dp=ctx_adp)
    gen_mgrs = _create_kda_managers(gen_tp, enable_attention_dp=gen_adp)
    ctx_mgr, gen_mgr = ctx_mgrs[0], gen_mgrs[-1]
    try:
        ctx_pt = ctx_mgr.get_page_table()
        gen_pt = gen_mgr.get_page_table()
        check_compatible(ctx_pt.page_layout, gen_pt.page_layout)
    finally:
        for mgr in ctx_mgrs + gen_mgrs:
            mgr.shutdown()


def _synthetic_kda_layout(tp: int, adp: bool, denom: int, layer_ids=None):
    """A KDA layer group's page layout as a rank of TP ``tp`` (attention DP ``adp``) declares it.

    Each per-rank slot is ``1 / denom`` of the full one (no CUDA needed).
    """
    from tensorrt_llm._torch.pyexecutor.kv_cache.page_layout import (
        BufferDesc,
        LayerGroupDesc,
        PageLayout,
        RankTopology,
        head_major,
        sectioned,
    )

    if layer_ids is None:
        layer_ids = range(1, NUM_KDA_LAYERS + 1)
    layer_ids = tuple(layer_ids)
    ranks = 1 if adp else tp
    head = KDA_HEAD_DIM * KDA_HEAD_DIM * SSM_DTYPE.itemsize
    conv = CONV_SLOT_BYTES // denom
    ssm = KDA_NUM_HEADS * head // denom
    buffers = []
    for i, layer in enumerate(layer_ids):
        buffers.append(
            BufferDesc(layer, "conv_state", 0, 0, i * conv, conv, sectioned([conv // 3] * 3, ranks))
        )
        buffers.append(
            BufferDesc(
                layer,
                "ssm_state",
                0,
                1,
                i * ssm,
                ssm,
                head_major(ssm, ssm // head, ssm // head * ranks, ("tp", "cp")),
            )
        )
    return PageLayout(
        tokens_per_page=8,
        pool_slot_bytes={0: (max(len(layer_ids), 1) * conv, max(len(layer_ids), 1) * ssm)},
        layer_groups=(LayerGroupDesc("state", None, 0, 0, layer_ids),) if layer_ids else (),
        buffers=tuple(buffers),
        topology=RankTopology(tp_size=tp, dp_size=tp if adp else 1, attention_dp=adp),
    )


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "ctx,gen,ok",
    [
        # (tp, adp, slot_scale_denominator). denom == tp models K3's actual
        # head-sharded per-rank slots (slot_bytes = full // tp when ADP off);
        # denom == 1 with tp > 1 and ADP off models a hypothetical model that
        # keeps a replicated full-size per-rank state while reporting
        # mamba_tp > 1 — the layout the global-size gate must reject under
        # hetero TP (shard offsets would land past the end of the slot).
        ((2, False, 1), (4, False, 1), False),  # replicated slots, hetero TP: reject
        ((4, False, 1), (2, False, 1), False),  # ...both directions
        ((2, True, 1), (2, False, 1), False),  # ADP on vs sharded-claim, full slots both
        ((2, False, 1), (2, False, 1), True),  # matched TP: sizes agree either way
        ((2, True, 1), (4, True, 1), True),  # hetero DEP, ADP on both (replicated)
        ((2, False, 2), (4, False, 4), True),  # K3 layout: sharded slots, hetero TP
    ],
    ids=[
        "reject_hetero_tp_replicated_slots",
        "reject_hetero_tp_replicated_slots_rev",
        "reject_mixed_adp_full_slots",
        "accept_matched_tp",
        "accept_hetero_dep_adp_on",
        "accept_sharded_hetero_tp",
    ],
)
def test_kda_peer_validation_synthetic_cpu(ctx, gen, ok):
    """CPU-only reject/accept matrix: the two sides' state must be the same global tensors."""
    ctx_layout = _synthetic_kda_layout(*ctx)
    gen_layout = _synthetic_kda_layout(*gen)
    if ok:
        check_compatible(ctx_layout, gen_layout)
    else:
        with pytest.raises(ValueError, match="do not correspond"):
            check_compatible(ctx_layout, gen_layout)


@pytest.mark.cpu_only
def test_kda_peer_validation_allows_pipeline_parallel_layer_split():
    """Peer validation must not require identical layer sets.

    Under pipeline parallelism each rank publishes only its own stage's
    layers, and the transfer path takes the intersection of the two layer
    sets. Peers with partially overlapping, disjoint, or one-sided
    recurrent-layer sets are all legitimate stage pairings and must pass
    validation as long as the per-slot size invariants hold.
    """
    # Partially overlapping stages (ctx PP split differs from gen's).
    ctx_layout = _synthetic_kda_layout(2, False, 2, layer_ids=[1, 2, 3])
    gen_layout = _synthetic_kda_layout(2, False, 2, layer_ids=[3, 4, 5])
    check_compatible(ctx_layout, gen_layout)

    # Disjoint stages: this pair simply moves no recurrent state.
    check_compatible(ctx_layout, _synthetic_kda_layout(2, False, 2, layer_ids=[7, 8]))

    # One side holds a recurrent-layer-free stage of the hybrid model: nothing to compare.
    empty = _synthetic_kda_layout(2, False, 2, layer_ids=[])
    check_compatible(ctx_layout, empty)
    check_compatible(empty, gen_layout)

    # The state must still be the same on the overlap: a different global size is refused even
    # when the layer sets differ.
    bad_gen = _synthetic_kda_layout(2, False, 4, layer_ids=[3, 4, 5])
    with pytest.raises(ValueError, match="do not correspond"):
        check_compatible(ctx_layout, bad_gen)


# ---------------------------------------------------------------------------
# Loopback transfer over a real NIXL agent (single node)
# ---------------------------------------------------------------------------
def _shard_kda_ssm(full_ssm: torch.Tensor, tp: int, rank: int) -> torch.Tensor:
    """Head shard of the [H, hd, hd] delta-rule state (identity for tp=1)."""
    n = KDA_NUM_HEADS // tp
    return full_ssm[rank * n : (rank + 1) * n].clone()


def _shard_kda_conv(full_conv: torch.Tensor, tp: int, rank: int) -> torch.Tensor:
    """Per-section shard of the [3*H*hd, W-1] conv state (identity for tp=1).

    qwen3_next [Q | K | V] sectioning with three equal H*hd sections; each
    section is sharded independently across tp, like the manager's
    conv_section_dims.
    """
    sec = KDA_NUM_HEADS * KDA_HEAD_DIM
    n = sec // tp
    parts = [full_conv[s * sec + rank * n : s * sec + (rank + 1) * n] for s in range(3)]
    return torch.cat(parts, dim=0).clone()


def _generate_ground_truth(num_requests: int, seed: int = 20260722):
    """Full (global, unsharded) KDA states per (request, kda_layer), two dtypes."""
    gen = torch.Generator(device="cpu").manual_seed(seed)
    results = []
    for _ in range(num_requests):
        layers = {}
        for i, is_kda in enumerate(_KDA_MASK):
            if not is_kda:
                continue
            layers[i] = {
                "conv": torch.rand(
                    3 * KDA_NUM_HEADS * KDA_HEAD_DIM,
                    KDA_CONV_STATE_WIDTH,
                    generator=gen,
                ).to(CONV_DTYPE),
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
    """Loopback KDA transfer, matched or heterogeneous ctx/gen TP.

    ADP off: per-rank state is a head shard (state_tp = tp) and every TP rank
    participates in each request; the TP-mismatch mappers re-tile shards
    between the two TP layouts. ADP on: per-rank state is a full-size replica
    (state_tp = 1) and each request belongs to ONE dp rank per side (the
    production scheduler routes it), so only that rank moves and verifies it
    (same routing as ``test_kv_transfer.add_and_verify_request``).
    """
    ctx_state_tp = 1 if enable_attention_dp else ctx_tp
    gen_state_tp = 1 if enable_attention_dp else gen_tp

    def _ctx_ranks(req_idx: int) -> List[int]:
        return [req_idx % ctx_tp] if enable_attention_dp else list(range(ctx_tp))

    def _gen_ranks(req_idx: int) -> List[int]:
        return [req_idx % gen_tp] if enable_attention_dp else list(range(gen_tp))

    ctx_mgrs = _create_kda_managers(ctx_tp, enable_attention_dp=enable_attention_dp)
    gen_mgrs = _create_kda_managers(gen_tp, enable_attention_dp=enable_attention_dp)
    ctx_tcs, gen_tcs = [], []
    try:
        for mgr in ctx_mgrs + gen_mgrs:
            for layer_id in mgr.mamba_layer_offsets:
                mgr.get_conv_states(layer_id).zero_()
                mgr.get_ssm_states(layer_id).zero_()

        config = CacheTransceiverConfig(
            backend="NIXL",
            transceiver_runtime="PYTHON",
            max_tokens_in_buffer=512,
        )
        ctx_tcs = _create_transceivers(
            ctx_tp, ctx_mgrs, config, enable_attention_dp=enable_attention_dp
        )
        gen_tcs = _create_transceivers(
            gen_tp, gen_mgrs, config, enable_attention_dp=enable_attention_dp
        )
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
                ctx_dp_rank=_ctx_ranks(req_idx)[0] if enable_attention_dp else 0,
                ctx_info_endpoint=ctx_endpoint,
                disagg_request_id=unique_rid,
            )
            ctx_reqs.append(ctx_req)
            gen_reqs.append(gen_req)

        ctx_batch = ScheduledRequests()
        ctx_batch.context_requests_last_chunk = ctx_reqs
        for mgr in ctx_mgrs:
            for req in ctx_reqs:
                assert mgr.prepare_context(req)
                assert mgr.resize_context(req, req.context_chunk_size)
            mgr.prepare_resources(ctx_batch)
        gen_batch = ScheduledRequests()
        gen_batch.context_requests_last_chunk = gen_reqs
        for mgr in gen_mgrs:
            for req in gen_reqs:
                assert mgr.prepare_disagg_gen_init(req)
            mgr.prepare_resources(gen_batch)
        for req in ctx_reqs + gen_reqs:
            req.context_current_position = req.prompt_len
            req.add_new_token(req.prompt_len, 0)
        for mgr in ctx_mgrs:
            mgr.update_resources(ctx_batch)
        for mgr in gen_mgrs:
            mgr.update_resources(gen_batch)

        # Ground truth: each ctx rank gets its shard of the global state
        # (the full state on every rank when replicated under ADP).
        ground_truth = _generate_ground_truth(len(REQUEST_LENGTHS))
        for rank, mgr in enumerate(ctx_mgrs):
            # Under ADP state_tp is 1 and every rank holds shard 0 (the full
            # state); with ADP off rank r holds shard r.
            shard_rank = rank % ctx_state_tp
            for req_idx, rid in enumerate(ctx_rids):
                slot = mgr._request_id_to_state_index[rid]
                for layer_idx in mgr.mamba_layer_offsets:
                    full = ground_truth[req_idx][layer_idx]
                    mgr.get_conv_states(layer_idx)[slot] = _shard_kda_conv(
                        full["conv"], ctx_state_tp, shard_rank
                    )
                    mgr.get_ssm_states(layer_idx)[slot] = _shard_kda_ssm(
                        full["ssm"], ctx_state_tp, shard_rank
                    )

        for req_idx, req in enumerate(gen_reqs):
            for rank in _gen_ranks(req_idx):
                gen_tcs[rank].request_and_receive_async(req)
        for req_idx, req in enumerate(ctx_reqs):
            for rank in _ctx_ranks(req_idx):
                ctx_tcs[rank].respond_and_send_async(req)
        _run_concurrent(
            ctx_tcs, lambda tc: tc.check_context_transfer_status(None, mark_complete=True)
        )
        _run_concurrent(gen_tcs, lambda tc: tc.check_gen_transfer_status(None))

        # Transfer-size metric must include the fixed-size KDA state — the actual
        # transferred bytes must cover the computed payload size: per rank,
        # num_layers * (conv + ssm) per-rank slot bytes on top of any KV bytes.
        # The transceiver's _kv_size_rank_factor is 1 under attention-DP (the
        # local slot already holds the full replicated state) and gen_tp
        # otherwise (the request total is the sum of the per-rank shards).
        rank_factor = 1 if enable_attention_dp else gen_tp
        kda_bytes_per_rank = NUM_KDA_LAYERS * (CONV_SLOT_BYTES + SSM_SLOT_BYTES) // gen_state_tp
        for req in gen_reqs:
            assert req.py_kv_cache_xfer_bytes >= kda_bytes_per_rank * rank_factor, (
                f"kv_cache_xfer_bytes={req.py_kv_cache_xfer_bytes} misses the KDA "
                f"state payload ({kda_bytes_per_rank} bytes/rank x {rank_factor})"
            )

        # Bitwise comparison of every participating gen rank's shard (the
        # single owning dp rank under ADP; all TP ranks otherwise).
        for gen_rank, mgr in enumerate(gen_mgrs):
            shard_rank = gen_rank % gen_state_tp
            for req_idx, rid in enumerate(gen_rids):
                if gen_rank not in _gen_ranks(req_idx):
                    continue
                slot = mgr._request_id_to_state_index[rid]
                for layer_idx in mgr.mamba_layer_offsets:
                    full = ground_truth[req_idx][layer_idx]
                    expected = {
                        "conv": _shard_kda_conv(full["conv"], gen_state_tp, shard_rank),
                        "ssm": _shard_kda_ssm(full["ssm"], gen_state_tp, shard_rank),
                    }
                    for name, getter in (
                        ("conv", mgr.get_conv_states),
                        ("ssm", mgr.get_ssm_states),
                    ):
                        torch.testing.assert_close(
                            getter(layer_idx)[slot].cpu(),
                            expected[name],
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
    "ctx_tp,gen_tp,enable_attention_dp",
    [
        (1, 1, False),
        (2, 2, False),
        (2, 2, True),
        (2, 4, True),
        (2, 4, False),
        (4, 2, False),
    ],
    ids=[
        "tp1_tp1",
        "matched_tp2_adp_off_sharded",
        "matched_tp2_adp_on_replicated",
        "hetero_dep_adp_on",
        "hetero_ctx_tp2_gen_tp4_adp_off_sharded",
        "hetero_ctx_tp4_gen_tp2_adp_off_sharded",
    ],
)
def test_kda_transfer(ctx_tp, gen_tp, enable_attention_dp):
    """KDA two-dtype state transfer over real NIXL loopback.

    Covers matched TP with ADP on/off, hetero DEP with ADP on, and hetero
    ctx/gen TP with ADP off (head-sharded state re-tiled by the TP-mismatch
    mappers, both expand and contract directions).
    """
    run_kda_transfer_test(ctx_tp, gen_tp, enable_attention_dp=enable_attention_dp)
