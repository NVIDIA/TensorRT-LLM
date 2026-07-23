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
    """Native V2 [pool, request, K/V, block] layout: K = 2*page, V = K+1."""
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


def _write_move_offsets(compaction, offsets, moves_per_request):
    cumulative = [0]
    for count in moves_per_request:
        cumulative.append(cumulative[-1] + count)
    # Rows past the cohort are padding and contribute no moves.
    cumulative.extend(cumulative[-1:] * (compaction["request_count"] - len(moves_per_request)))
    offsets.copy_(torch.tensor(cumulative, dtype=torch.int32), non_blocking=True)


def set_protected_tails(compaction, tail_lengths, draft_tail_lengths=None):
    """Load per-request protected tails into the caller-owned move offsets."""
    if len(tail_lengths) > compaction["request_count"]:
        raise ValueError("the cohort exceeds the compaction request capacity")
    if any(tail < 0 or tail > compaction["protected_tail_capacity"] for tail in tail_lengths):
        raise ValueError("a protected tail exceeds the configured capacity")
    _write_move_offsets(
        compaction,
        compaction["dense_move_offsets"],
        [compaction["decode_keep_count"] + int(tail) for tail in tail_lengths],
    )
    if compaction["has_swa"]:
        _write_move_offsets(
            compaction,
            compaction["swa_move_offsets"],
            [compaction["swa_window"] + int(tail) for tail in tail_lengths],
        )
    if compaction["draft_move_offsets"] is not None:
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
            compaction["draft_move_offsets"],
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
    """bf16 pools with a shifted ``arange % 251`` ramp: every wrong move
    lands on a different byte pattern (supported geometry defaults)."""
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
    """``init_compaction_buffers`` with the suite's 2-layer defaults:
    translates ``eviction_mode`` into ``per_layer_sources``/``decision_rows``,
    allocates the caller-owned move-offset rows (capacity cumsum) and SWA
    destination bases, and hands the test's pre-settled
    ``kept_token_ordinals`` in as the decision rows. Returns the opaque
    ``plans`` plus a test-side mirror of the caller-owned inputs."""
    from tensorrt_llm._torch.kv_cache_compression.compaction import init_compaction_buffers

    args = dict(
        eviction_mode="union",
        dense_layers=[0, 1],
        swa_layers=[],
        layer_group_representative={0: 0, 1: 1},
        layer_pool_ids=[0, 0],
        request_count=2,
        decode_keep_count=4,
        swa_window=None,
    )
    args.update(overrides)
    mode = args.pop("eviction_mode")
    union = mode == "union"
    per_layer = mode == "per_layer_perhead"
    kept = args.pop("kept_token_ordinals")
    request_count = args["request_count"]
    keep_count = args["decode_keep_count"]
    tail = int(args.get("protected_tail_capacity", 0))
    draft_tail = int(args.get("draft_protected_tail_capacity") or 0)
    has_draft = bool(args.get("draft_layers"))
    has_swa = bool(args["swa_layers"])
    device = args["layer_pools"][args["dense_layers"][0]].device
    swa_window = int(args["swa_window"] or 0) if has_swa else 0
    swa_destination_bases = torch.empty_like(args["prompt_offsets"]) if has_swa else None

    def capacity_offsets(count):
        return torch.arange(0, (request_count + 1) * count, count, dtype=torch.int32, device=device)

    args.setdefault("dense_move_offsets", capacity_offsets(keep_count + tail))
    args.setdefault("swa_move_offsets", capacity_offsets(swa_window + tail) if has_swa else None)
    if has_draft:
        args.setdefault("draft_move_offsets", capacity_offsets(keep_count + draft_tail))
    num_kv_heads = int(args["layer_pools"][args["dense_layers"][0]].shape[2])
    selection_rows = (
        1 if union else (len(args["dense_layers"]) * num_kv_heads if per_layer else num_kv_heads)
    )
    draft = None
    if has_draft:
        draft = dict(
            layer_pools=args["draft_layer_pools"],
            dense_layers=args["draft_layers"],
            layer_group_representative=args["draft_layer_group_representative"],
            layer_pool_ids=args["draft_layer_pool_ids"],
            kv_block_offsets=args["draft_kv_block_offsets"],
            dense_move_offsets=args["draft_move_offsets"],
            protected_tail_capacity=draft_tail,
        )
    plans = init_compaction_buffers(
        target=dict(
            layer_pools=args["layer_pools"],
            dense_layers=args["dense_layers"],
            swa_layers=args["swa_layers"],
            swa_window=args["swa_window"],
            layer_group_representative=args["layer_group_representative"],
            layer_pool_ids=args["layer_pool_ids"],
            kv_block_offsets=args["kv_block_offsets"],
            token_starts=args["prompt_offsets"],
            swa_destination_bases=swa_destination_bases,
            dense_move_offsets=args["dense_move_offsets"],
            swa_move_offsets=args["swa_move_offsets"],
            per_layer_sources=per_layer,
            kept_ordinal_rows=kept.reshape(-1, keep_count),
            decision_rows=selection_rows,
            valid_seq_lens=args["valid_sequence_lengths"],
        ),
        capacities=dict(
            max_requests=request_count,
            keep_count=keep_count,
            protected_tail_capacity=tail,
        ),
        draft=draft,
    )
    # Opaque plans plus a test-side mirror of the caller-owned construction
    # inputs (production binds the same values on its buffer namespace); the
    # standalone helpers here need the move-offset rows and SWA staging back.
    return dict(
        plans=plans,
        prompt_offsets=args["prompt_offsets"],
        request_count=request_count,
        decode_keep_count=keep_count,
        protected_tail_capacity=tail,
        draft_protected_tail_capacity=draft_tail if has_draft else 0,
        dense_move_offsets=args["dense_move_offsets"],
        swa_move_offsets=args["swa_move_offsets"],
        draft_move_offsets=args["draft_move_offsets"] if has_draft else None,
        has_swa=has_swa,
        swa_window=swa_window,
        swa_destination_bases=swa_destination_bases,
        swa_rebase_delta=keep_count - swa_window,
    )


def run_compaction(compaction):
    """Replica of the round's move stage in production order: SWA
    destination rebase, then ``compact`` loops the opaque plans (each packs
    its decision rows into move sources and fires its native moves)."""
    from tensorrt_llm._torch.kv_cache_compression.compaction import compact

    if compaction["swa_destination_bases"] is not None:
        torch.add(
            compaction["prompt_offsets"],
            compaction["swa_rebase_delta"],
            out=compaction["swa_destination_bases"],
        )
    compact(compaction["plans"], compaction["request_count"])


def make_bare_staging(device, *, max_requests, staged_blocks_per_seq):
    """A bare buffer namespace for the bulk page-table copy tests."""
    staging = SimpleNamespace()
    staging.max_requests = max_requests
    staging.keep_count = 4
    staging.swa_window = None
    staging.draft_protected_tail_capacity = None
    staging.block_offsets_ready_event = torch.cuda.Event()
    staging.compaction_done_event = torch.cuda.Event()
    staging.staging_reuse_event = torch.cuda.Event()
    staging.copy_pending = False
    staging.block_offsets_host = torch.empty(
        1, max_requests, 2, staged_blocks_per_seq, dtype=torch.int32, device="cpu", pin_memory=True
    )
    staging.identity_copy_indices_host = torch.arange(
        max_requests, dtype=torch.int32, device="cpu", pin_memory=True
    )
    staging.block_offsets_device = torch.empty(
        1, max_requests, 2, staged_blocks_per_seq, dtype=torch.int32, device=device
    )
    return staging


def make_staging_manager(host_table, gather, manager_stream, *, num_slots=1):
    """The manager surface ``_stage_block_offsets`` consumes."""
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
    manager._freq_scale_sq = torch.ones(2)
    manager._phase = {"rows": 8}
    manager.calibration = {"omega": torch.ones(2)}
    manager._local_score_calibration = mock.Mock(return_value=(torch.ones(2, 2, 2),) * 3)
    pool = torch.empty(8, 2, 1, 4, 4)
    layout = dict(
        manager=SimpleNamespace(num_pools=1),
        global_layers=[0, 1],
        layer_pools=[pool, pool],
        dense_layers=[0, 1],
        swa_layers=[],
        swa_window=None,
        storage_groups={0: [0, 1]},
        layer_group_representative={0: 0, 1: 0},
        layer_pool_ids=(0, 0),
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


def make_tri_config(**overrides):
    """A real TriAttentionKvCacheCompressionConfig with test calibration inputs
    (the config validator requires both ``model_path`` and ``calibration_path``)."""
    from tensorrt_llm.llmapi.llm_args import TriAttentionKvCacheCompressionConfig

    options = {
        "budget": 8,
        "model_path": "/models/test",
        "calibration_path": "/calib/test.pt",
    }
    options.update(overrides)
    return TriAttentionKvCacheCompressionConfig(**options)


def make_triattention(**overrides):
    """Construct a fully initialized manager for method-level unit tests."""
    from tensorrt_llm._torch.kv_cache_compression.triattention.triattention import TriAttention

    return TriAttention(make_fake_v2(), make_tri_config(**overrides))


def make_prepared_item(
    request=None,
    *,
    request_id=0,
    seq_len,
    round_start=None,
    prompt_len=0,
    expected_keep_count=0,
    protected_tail=0,
    kv_cache=None,
    draft_kv_cache=None,
):
    """One prepared-cohort item shaped exactly like ``_periodic_evict`` builds."""
    return {
        "request": request,
        "request_id": request_id,
        "kv_cache": kv_cache,
        "draft_kv_cache": draft_kv_cache,
        "seq_len": int(seq_len),
        "round_start": int(seq_len if round_start is None else round_start),
        "prompt_len": int(prompt_len),
        "expected_keep_count": int(expected_keep_count),
        "protected_tail": int(protected_tail),
    }


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
    """Run the real ``_evict_requests`` body around a mocked round executor."""
    from tensorrt_llm._torch.kv_cache_compression.triattention import triattention as module

    buffers = SimpleNamespace(max_requests=8)
    with (
        mock.patch.object(manager, "_runtime_kv_layout", return_value={}),
        mock.patch.object(manager, "_buffers_for", return_value=buffers),
        mock.patch.object(module, "execute_eviction_round") as execute,
    ):
        yield SimpleNamespace(buffers=buffers, execute=execute)


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
    """Independent Torch oracle of the paged mean score (GQA mapping via
    ``head // group_size`` plus the position-independent MLR term)."""
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


def make_phase_table(offsets, omega, initial_rows):
    """Build the mean-phase table dict exactly like the product's inlined
    form and grow it to cover positions ``[0, initial_rows)``."""
    from tensorrt_llm._torch.kv_cache_compression.triattention.triattention import (
        grow_mean_phase_table,
    )

    phase = {
        "omega": omega.to(dtype=torch.float32).contiguous(),
        "offset_values": offsets.tolist(),
        "cos": None,
        "sin": None,
        "rows": 0,
    }
    grow_mean_phase_table(phase, max(int(initial_rows), 1))
    return phase


def make_cute_buffers(
    *,
    eviction_mode,
    layer_pools,
    max_requests,
    seq_len,
    num_q_heads,
    q_real,
    q_imag,
    mlr_coef,
    freq_scale_sq,
    omega,
    offsets,
    decode_width=None,
    keep_count=1,
    page_table_token_capacity=None,
    protected_tail_capacity=0,
    storage_groups=None,
    layer_pool_ids=None,
    normalize_scores=True,
):
    """Real eviction buffers over the one-shared-slot default layout; split
    reference legs use ``eviction_mode="per_head"`` over the same pools.
    ``storage_groups``/``layer_pool_ids`` override the page-table grouping
    (``layer_pool_ids`` is the canonical per-layer V2 pool id list)."""
    from tensorrt_llm._torch.kv_cache_compression.triattention.triattention import (
        init_eviction_buffers,
    )

    num_layers = len(layer_pools)
    assert int(q_real.shape[1]) == num_q_heads
    # The constructor takes every capacity explicitly (no test-only
    # None-derive path); the widest window defaults keep old call sites.
    if decode_width is None:
        decode_width = seq_len
    if storage_groups is None:
        storage_groups = {0: list(range(num_layers))}
    if layer_pool_ids is None:
        layer_pool_ids = [0] * num_layers
    layout = dict(
        manager=SimpleNamespace(num_pools=max(layer_pool_ids) + 1),
        layer_pools=layer_pools,
        dense_layers=list(range(num_layers)),
        swa_layers=[],
        swa_window=None,
        storage_groups=storage_groups,
        layer_group_representative={
            layer: layers[0] for layers in storage_groups.values() for layer in layers
        },
        layer_pool_ids=layer_pool_ids,
    )
    return init_eviction_buffers(
        eviction_mode=eviction_mode,
        layout=layout,
        calibration=dict(
            q_real=q_real,
            q_imag=q_imag,
            mlr_coef=mlr_coef,
            freq_scale_sq=freq_scale_sq,
        ),
        phase=make_phase_table(offsets, omega, seq_len),
        capacities=dict(
            max_requests=max_requests,
            bucket_seq_len=seq_len,
            decode_width=decode_width,
            page_table_token_capacity=(
                seq_len if page_table_token_capacity is None else page_table_token_capacity
            ),
            keep_count=keep_count,
            protected_tail_capacity=protected_tail_capacity,
        ),
        normalize_scores=normalize_scores,
    )


def write_block_offsets(bufs, encoded):
    """Load a test page table into the staged block-offset plane."""
    bufs.block_offsets_device.zero_()
    bufs.block_offsets_device[:, : encoded.shape[1], :, : encoded.shape[-1]].copy_(encoded)


def stage_score_metadata(bufs, request_count, valid_seq_lens, valid_widths, token_starts):
    """Stage the per-round score metadata exactly like production (the
    compiled runner reads the staged rows via pointer capture)."""
    torch.sub(
        valid_seq_lens[:request_count],
        token_starts[:request_count],
        out=valid_widths[:request_count],
    )
    bufs.valid_seq_lens_device[:request_count].copy_(valid_seq_lens[:request_count])
    bufs.token_starts_device[:request_count].copy_(token_starts[:request_count])


def launch_split_scores(
    bufs, request_count, valid_seq_lens, valid_widths, token_starts, mean_cos, mean_sin
):
    """The production score-only leg plus the decode-window gather
    (``execute_eviction_round``'s per-head sequence, parameterized by count).
    Test mean phases load into the runner's ctor-bound buffers, exactly like
    the production in-place gather refresh."""
    stage_score_metadata(bufs, request_count, valid_seq_lens, valid_widths, token_starts)
    bufs.mean_cos[:request_count].copy_(mean_cos[:request_count])
    bufs.mean_sin[:request_count].copy_(mean_sin[:request_count])
    assert request_count in bufs.runner._compiled
    bufs.runner.launch(request_count)
    num_segments = request_count * bufs.num_layers
    group_size = bufs.num_q_heads // bufs.num_kv_heads
    source = (
        bufs.score_scratch[: bufs.num_kv_heads * 8 * num_segments * bufs.bucket_seq_len]
        .view(bufs.num_kv_heads, 8, request_count, bufs.num_layers, bufs.bucket_seq_len)[
            :, :group_size
        ]
        .permute(2, 3, 0, 1, 4)
    )
    columns = (
        token_starts[:request_count].to(torch.int64).view(-1, 1, 1, 1, 1) + bufs.gather_columns
    )
    columns = columns.clamp_(max=bufs.bucket_seq_len - 1).expand(
        request_count, bufs.num_layers, bufs.num_kv_heads, group_size, bufs.decode_width
    )
    output = torch.full(
        (request_count, bufs.num_layers, bufs.num_q_heads, bufs.decode_width),
        float("nan"),
        dtype=torch.float32,
        device=bufs.score_scratch.device,
    )
    torch.gather(
        source,
        4,
        columns,
        out=output.view(
            request_count, bufs.num_layers, bufs.num_kv_heads, group_size, bufs.decode_width
        ),
    )
    return output
