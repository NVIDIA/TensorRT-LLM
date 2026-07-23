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

from contextlib import contextmanager
from types import SimpleNamespace
from unittest import mock

import torch


def encode_block_offsets(page_ids: torch.Tensor) -> torch.Tensor:
    """Build the native V2 [pool, request, K/V, block] layout.

    Accepts ``[request, block]`` page ids (one pool) or ``[pool, request,
    block]``; K offsets encode as ``2*page`` and V as ``2*page + 1``.
    """
    if page_ids.ndim == 2:
        page_ids = page_ids.unsqueeze(0)
    encoded = torch.empty(
        page_ids.shape[0],
        page_ids.shape[1],
        2,
        page_ids.shape[2],
        dtype=torch.int32,
        device=page_ids.device,
    )
    encoded[:, :, 0] = page_ids.to(torch.int32) * 2
    encoded[:, :, 1] = encoded[:, :, 0] + 1
    return encoded


def compaction_family(compaction, name):
    """Return one cache family dict ("dense", "swa", or "draft") or None."""
    for family in compaction["families"]:
        if family["name"] == name:
            return family
    return None


def _write_move_offsets(compaction, offsets, moves_per_request):
    cumulative = [0]
    for count in moves_per_request:
        cumulative.append(cumulative[-1] + count)
    # Rows past the cohort are padding and contribute no moves.
    cumulative.extend(cumulative[-1:] * (compaction["request_count"] - len(moves_per_request)))
    offsets.copy_(torch.tensor(cumulative, dtype=torch.int32), non_blocking=True)


def set_protected_tails(compaction, tail_lengths, draft_tail_lengths=None):
    """Load a cohort's per-request protected tails into the move offsets.

    Production stages these rows through the single round-metadata upload;
    tests drive the fixed buffers directly through this helper.
    """
    if len(tail_lengths) > compaction["request_count"]:
        raise ValueError("the cohort exceeds the compaction request capacity")
    if any(tail < 0 or tail > compaction["protected_tail_capacity"] for tail in tail_lengths):
        raise ValueError("a protected tail exceeds the configured capacity")
    _write_move_offsets(
        compaction,
        compaction_family(compaction, "dense")["offsets"],
        [compaction["decode_keep_count"] + int(tail) for tail in tail_lengths],
    )
    swa_family = compaction_family(compaction, "swa")
    if swa_family is not None:
        _write_move_offsets(
            compaction,
            swa_family["offsets"],
            [compaction["swa_window"] + int(tail) for tail in tail_lengths],
        )
    draft_family = compaction_family(compaction, "draft")
    if draft_family is not None:
        if draft_tail_lengths is None:
            draft_tail_lengths = [0] * len(tail_lengths)
        if len(draft_tail_lengths) != len(tail_lengths):
            raise ValueError("draft protected tails must match the cohort")
        if any(
            tail < 0 or tail > compaction["draft_protected_tail_capacity"]
            for tail in draft_tail_lengths
        ):
            raise ValueError("a draft protected tail exceeds the configured capacity")
        _write_move_offsets(
            compaction,
            draft_family["offsets"],
            [compaction["decode_keep_count"] + int(tail) for tail in draft_tail_lengths],
        )


def make_ramp_pools(
    count,
    *,
    num_kv_heads=2,
    pages=6,
    tokens_per_block=32,
    head_dim=64,
    layer_stride=37,
    base=0,
    device=None,
):
    """bf16 pools carrying a shifted ``arange % 251`` ramp payload.

    Every value is exact in bf16 and any wrong page/plane/head/token move
    lands on a different byte pattern, so pool-equality checks in the
    compaction tests stay conclusive. Geometry defaults to the compact op's
    supported production shape (32-token pages, head_dim 64).
    """
    return [
        (
            (
                torch.arange(
                    pages * 2 * num_kv_heads * tokens_per_block * head_dim,
                    dtype=torch.int32,
                    device=device,
                )
                + base
                + layer * layer_stride
            )
            % 251
        )
        .view(pages, 2, num_kv_heads, tokens_per_block, head_dim)
        .to(torch.bfloat16)
        for layer in range(count)
    ]


def build_compaction(**overrides):
    """``init_compaction_buffers`` with the suite's default 2-layer geometry."""
    from tensorrt_llm._torch.kv_cache_compression.compaction import init_compaction_buffers

    args = dict(
        eviction_mode="union",
        dense_layers=[0, 1],
        swa_layers=[],
        layer_group_representative={0: 0, 1: 1},
        layer_pool_keys=[("dense", 0), ("dense", 0)],
        page_table_slots={0: 0, 1: 0},
        request_count=2,
        decode_keep_count=4,
        swa_window=None,
    )
    args.update(overrides)
    return init_compaction_buffers(**args)


def launch_family_pack(compaction, name):
    """Standalone HAS_SETTLE=False pack launch for one family of a bundle.

    Production packs dense/SWA inside the fused settle launch and fires the
    draft pack inline in ``run_eviction_round``; standalone compaction tests
    pack here from the bundle's own geometry so the C++ moves read
    initialized indices. The settle-side pointer arguments are compiled away
    (any well-formed tensor stands in).
    """
    from tensorrt_llm._torch.kv_cache_compression.triattention.triattention_kernels import (
        SETTLE_PACK_BLOCK,
        SETTLE_PACK_NUM_WARPS,
        _settle_ties_and_pack_compaction_sources_kernel,
    )

    family = compaction_family(compaction, name)
    kept = compaction["kept_token_ordinals"]
    valid = compaction["valid_sequence_lengths"]
    keep_count = compaction["decode_keep_count"]
    if name == "draft":
        draft = compaction["draft_pack"]
        rows = 1
        shape = dict(
            DENSE_TOTAL=draft["dense_total"],
            SWA_TOTAL=0,
            MOVE_CAPACITY=draft["move_capacity"],
            NUM_KV_HEADS=draft["num_kv_heads"],
            SWA_WINDOW=0,
            UNION=True,
            PER_LAYER=False,
            HAS_SWA=False,
        )
        swa_offsets, swa_indices = None, None
    else:
        rows = compaction["selection_rows"]
        shape = compaction["settle_pack_shape"]
        swa_family = compaction_family(compaction, "swa")
        swa_offsets = swa_family["offsets"] if swa_family else None
        swa_indices = swa_family["source"] if swa_family else None
    _settle_ties_and_pack_compaction_sources_kernel[(compaction["request_count"], rows)](
        None,
        None,
        None,
        None,
        kept,
        valid,
        family["offsets"],
        family["source"],
        swa_offsets,
        swa_indices,
        WIDTH=keep_count,
        KEEP_COUNT=keep_count,
        SELECTION_ROWS=rows,
        **shape,
        HAS_SETTLE=False,
        BLOCK=SETTLE_PACK_BLOCK,
        num_warps=SETTLE_PACK_NUM_WARPS,
    )


def run_compaction(compaction, pack=("dense", "draft")):
    """Test-side replica of ``run_eviction_round``'s compact stage.

    Optional standalone family packs, the SWA destination rebase, then every
    family's C++ moves -- the same sequence production fires inline.
    """
    if compaction["swa_destination_bases"] is not None:
        torch.add(
            compaction["prompt_offsets"],
            compaction["swa_rebase_delta"],
            out=compaction["swa_destination_bases"],
        )
    for family in compaction["families"]:
        if family["name"] in pack and family["name"] != "swa":
            launch_family_pack(compaction, family["name"])
        for group in family["groups"]:
            torch.ops.trtllm.sparse_kv_cache_compact_layers(
                group["pools"],
                group["pool_pointers"],
                group["page_table"],
                family["source"],
                family["offsets"],
                family["destination_bases"],
                group["source_layer_indices"],
            )


def make_bare_staging(device, *, max_requests, copy_block_count, page_count=None):
    """A bare buffer namespace for the bulk page-table copy tests."""
    staging = SimpleNamespace()
    staging.device = device
    staging.max_requests = max_requests
    staging.copy_block_count = copy_block_count
    if page_count is not None:
        staging.page_count = page_count
    staging.bulk_copy_done = torch.cuda.Event()
    staging.bulk_consume_done = torch.cuda.Event()
    staging.page_tables_active = False
    staging.copy_done = torch.cuda.Event()
    staging.copy_pending = False
    staging._bulk_offsets_src = torch.empty(
        1, max_requests, 2, copy_block_count, dtype=torch.int32, device="cpu", pin_memory=True
    )
    staging._bulk_copy_idx_src = torch.arange(
        max_requests, dtype=torch.int32, device="cpu", pin_memory=True
    )
    staging.block_offsets_device = torch.empty(
        1, max_requests, 2, copy_block_count, dtype=torch.int32, device=device
    )
    return staging


def make_staging_manager(host_table, gather, manager_stream, *, num_slots=1):
    """The manager surface ``_stage_page_tables_bulk``/``stage`` consume."""
    return SimpleNamespace(
        host_kv_cache_block_offsets=host_table,
        kv_factor=2,
        index_mapper=SimpleNamespace(gather_k_block_offsets=gather),
        index_scales=torch.full((num_slots,), 2, dtype=torch.int32, pin_memory=True),
        kv_offset=torch.ones(num_slots, dtype=torch.int32, pin_memory=True),
        _stream=manager_stream,
    )


def make_buffer_stubs(manager, *, decode_width=260):
    """Stub the calibration/layout surfaces around ``_buffers_for``."""
    manager._H = 2
    manager._F = 2
    manager._freq_scale_sq = torch.ones(2)
    manager._offsets = torch.ones(2)
    manager._phase = {"rows": 8}
    manager.calibration = {"omega": torch.ones(2)}
    manager._local_score_calibration = mock.Mock(return_value=(torch.ones(2, 2, 2),) * 3)
    manager._page_table_pool_keys = mock.Mock(return_value=[("pool", 0)])
    pool = torch.empty(8, 2, 1, 4, 4)
    layout = dict(
        manager=SimpleNamespace(num_pools=1),
        num_layers=2,
        global_layers=[0, 1],
        layer_pools=[pool, pool],
        dense_layers=[0, 1],
        swa_layers=[],
        swa_window=None,
        storage_groups={0: [0, 1]},
        layer_group_representative={0: 0, 1: 0},
        layer_pool_keys=(("pool", 0), ("pool", 0)),
        pool_view_fingerprint=(("fixed",),),
    )
    buffers = SimpleNamespace(
        decode_width=decode_width,
        page_table_token_capacity=65537,
        max_requests=8,
        token_starts_device=torch.zeros(8, dtype=torch.int32),
        valid_widths=torch.empty(8, dtype=torch.int32),
    )
    return layout, buffers


def make_fake_v2(enable_block_reuse=False, *, is_draft=False):
    """Build an unallocated V2 double with TriAttention's production contract."""
    from tensorrt_llm._torch.pyexecutor.kv_cache_manager_v2 import KVCacheManagerV2

    fake_v2 = KVCacheManagerV2.__new__(KVCacheManagerV2)
    fake_v2.enable_block_reuse = enable_block_reuse
    fake_v2.enable_swa_scratch_reuse = False
    fake_v2.is_draft = is_draft
    fake_v2.kv_compression_manages_history = False
    fake_v2.kv_factor = 2
    fake_v2.mapping = SimpleNamespace(enable_attention_dp=False)
    fake_v2.is_disagg = False
    fake_v2.max_beam_width = 1
    fake_v2.max_batch_size = 8
    fake_v2.num_extra_kv_tokens = 0
    fake_v2.max_draft_len = 0
    fake_v2.max_total_draft_tokens = 0
    fake_v2._kv_reserve_draft_tokens = 0
    fake_v2.max_seq_len = 65536
    fake_v2.tokens_per_block = 64
    fake_v2.max_blocks_per_seq = 1028
    fake_v2.get_num_available_tokens = lambda *, token_num_upper_bound, **_: token_num_upper_bound
    fake_v2.max_attention_window_vec = []
    fake_v2.kv_cache_manager_py_config = SimpleNamespace(layers=[])
    fake_v2.impl = object()
    fake_v2.kv_cache_map = {}
    fake_v2.host_kv_cache_block_offsets = torch.zeros(1, 8, 2, 8, dtype=torch.int32)
    fake_v2.pp_layers = []
    fake_v2.layer_offsets = {}
    fake_v2.layer_to_pool_mapping_dict = {}
    return fake_v2


def make_triattention(**overrides):
    """Construct a fully initialized manager for method-level unit tests."""
    from tensorrt_llm._torch.kv_cache_compression.triattention.triattention import TriAttention

    options = {"budget": 8, "model_path": "/models/test"}
    options.update(overrides)
    return TriAttention(make_fake_v2(), **options)


def make_request(request_id, **overrides):
    """Build the explicit request fields consumed by TriAttention."""
    from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequestState

    fields = {
        "py_request_id": request_id,
        "py_prompt_len": 0,
        "py_max_new_tokens": 65536,
        "py_draft_tokens": [],
        "py_num_accepted_draft_tokens": 0,
        "py_num_compressed_tokens": 0,
        "is_dummy": False,
        "state": LlmRequestState.GENERATION_IN_PROGRESS,
    }
    fields.update(overrides)
    return SimpleNamespace(**fields)


@contextmanager
def mocked_eviction_internals(manager):
    """Run the real ``_evict_requests`` body around mocked GPU launches."""
    from tensorrt_llm._torch.kv_cache_compression.triattention import triattention as module

    buffers = SimpleNamespace(max_requests=8)
    layout = dict(swa_layers=[], swa_window=None)
    with (
        mock.patch.object(manager, "_runtime_kv_layout", return_value=layout),
        mock.patch.object(manager, "_buffers_for", return_value=buffers),
        mock.patch.object(module, "stage_eviction_cohort") as stage,
        mock.patch.object(module, "run_eviction_round") as run_round,
        mock.patch.object(module, "mark_page_tables_consumed") as consumed,
    ):
        yield SimpleNamespace(
            buffers=buffers,
            stage=stage,
            run_round=run_round,
            consumed=consumed,
        )


def torch_tri_score_oracle(
    layer_pools,
    page_ids,
    seq_lens,
    round_starts,
    q_real,
    q_imag,
    mlr_coef,
    freq_scale_sq,
    omega,
    offsets,
    layer_indices,
):
    """Independent Torch implementation of the paged TriAttention mean score.

    Covers GQA head mapping via ``head // group_size`` and the
    position-independent MLR term. Mean aggregation only: it is the single
    production aggregation (max was removed with the C++ score stack).
    """
    scores = []
    num_q_heads = int(q_real.shape[1])
    for request, seq_len in enumerate(seq_lens):
        phase = (round_starts[request] + offsets[:, None]) * omega[None, :]
        mean_cos = torch.cos(phase).mean(dim=0)
        mean_sin = torch.sin(phase).mean(dim=0)
        for layer in layer_indices:
            pool = layer_pools[layer]
            request_page_ids = (
                page_ids[layer][request] if isinstance(page_ids, dict) else page_ids[request]
            )
            keys = (
                pool.index_select(0, request_page_ids)[:, 0]
                .permute(1, 0, 2, 3)
                .reshape(pool.shape[2], -1, pool.shape[4])[:, :seq_len]
                .float()
            )
            num_kv_heads = int(keys.shape[0])
            group_size = num_q_heads // num_kv_heads
            head_scores = []
            for head in range(num_q_heads):
                key = keys[head // group_size]
                num_freqs = int(key.shape[-1]) // 2
                key_real = key[:, :num_freqs]
                key_imag = key[:, num_freqs:]
                product_real = q_real[layer, head] * key_real + q_imag[layer, head] * key_imag
                product_imag = q_imag[layer, head] * key_real - q_real[layer, head] * key_imag
                position = (
                    freq_scale_sq * (product_real * mean_cos - product_imag * mean_sin)
                ).sum(dim=-1)
                mlr = (
                    torch.sqrt(key_real.square() + key_imag.square())
                    * mlr_coef[layer, head]
                    * freq_scale_sq
                ).sum(dim=-1)
                head_scores.append(position + mlr)
            scores.append(torch.stack(head_scores))
    return scores
