# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared harness for KV-cache compression tests."""

import json
import os
import tempfile
from contextlib import contextmanager
from types import SimpleNamespace
from typing import Optional
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
    """``build_compaction_params`` with the suite's 2-layer defaults:
    allocates the caller-owned move-offset rows (capacity cumsum) and SWA
    destination bases, and hands the test's pre-settled
    ``kept_token_ordinals`` in as the decision rows. Returns the opaque
    ``params`` plus a test-side mirror of the caller-owned inputs."""
    from tensorrt_llm._torch.kv_cache_compression.compaction import build_compaction_params

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
    args.pop("eviction_mode")
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
    params_list = [
        build_compaction_params(
            dict(
                layer_pools=args["layer_pools"],
                dense_layers=args["dense_layers"],
                swa_layers=args["swa_layers"],
                swa_window=args["swa_window"],
                layer_pool_ids=args["layer_pool_ids"],
            ),
            block_offsets=args["kv_block_offsets"],
            kept_ordinals=kept.reshape(-1, keep_count),
            source_lengths=args["valid_sequence_lengths"],
            dense_destination_bases=args["prompt_offsets"],
            dense_move_offsets=args["dense_move_offsets"],
            protected_tail_capacity=tail,
            swa_move_offsets=args["swa_move_offsets"],
            swa_destination_bases=swa_destination_bases,
        )
    ]
    if has_draft:
        params_list.append(
            build_compaction_params(
                dict(
                    layer_pools=args["draft_layer_pools"],
                    dense_layers=args["draft_layers"],
                    swa_layers=[],
                    layer_pool_ids=args["draft_layer_pool_ids"],
                ),
                block_offsets=args["draft_kv_block_offsets"],
                kept_ordinals=kept.reshape(-1, keep_count),
                source_lengths=args["valid_sequence_lengths"],
                dense_destination_bases=args["prompt_offsets"],
                dense_move_offsets=args["draft_move_offsets"],
                protected_tail_capacity=draft_tail,
            )
        )
    params = tuple(params_list)
    # Opaque plans plus a test-side mirror of the caller-owned construction
    # inputs (production binds the same values as manager attributes); the
    # standalone helpers here need the move-offset rows and SWA staging back.
    return dict(
        params=params,
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
    destination rebase, then ``compact`` loops the opaque params (each packs
    its decision rows into move sources and fires its native moves)."""
    from tensorrt_llm._torch.kv_cache_compression.compaction import compact

    if compaction["swa_destination_bases"] is not None:
        torch.add(
            compaction["prompt_offsets"],
            compaction["swa_rebase_delta"],
            out=compaction["swa_destination_bases"],
        )
    compact(compaction["params"], compaction["request_count"])


def make_bare_staging(device, *, max_requests, staged_blocks_per_seq):
    """A bare manager carrying only the page-table staging attributes."""
    from tensorrt_llm._torch.kv_cache_compression.triattention.triattention import (
        TriAttentionCompressionManager,
    )

    staging = TriAttentionCompressionManager.__new__(TriAttentionCompressionManager)
    staging.kv_cache_manager = None
    staging.draft_kv_cache_manager = None
    staging._request_capacity = max_requests
    staging.budget = 4
    staging._swa_window = None
    staging._draft_protected_tail_capacity = 0
    staging._compaction_done_event = torch.cuda.Event()
    staging._staging_reuse_event = torch.cuda.Event()
    staging._block_offsets_host = torch.empty(
        1, max_requests, 2, staged_blocks_per_seq, dtype=torch.int32, device="cpu", pin_memory=True
    )
    staging._identity_copy_indices_host = torch.arange(
        max_requests, dtype=torch.int32, device="cpu", pin_memory=True
    )
    staging._block_offsets_device = torch.empty(
        1, max_requests, 2, staged_blocks_per_seq, dtype=torch.int32, device=device
    )
    return staging


def make_staging_manager(host_table, gather, manager_stream, *, num_slots=1):
    """The manager surface ``_stage_block_offset_snapshot`` consumes."""
    return SimpleNamespace(
        host_kv_cache_block_offsets=host_table,
        kv_factor=2,
        index_mapper=SimpleNamespace(gather_k_block_offsets=gather),
        index_scales=torch.full((num_slots,), 2, dtype=torch.int32, pin_memory=True),
        kv_offset=torch.ones(num_slots, dtype=torch.int32, pin_memory=True),
        uses_device_page_table=False,
        _stream=manager_stream,
    )


def make_fake_v2(enable_block_reuse=False, *, is_draft=False):
    """Build an unallocated V2 double with TriAttention's production contract."""
    from tensorrt_llm._torch.pyexecutor.kv_cache_manager_v2 import KVCacheManagerV2

    fake_v2 = KVCacheManagerV2.__new__(KVCacheManagerV2)
    fake_v2.enable_block_reuse = enable_block_reuse
    fake_v2.is_draft = is_draft
    fake_v2.kv_compression_manages_history = False
    fake_v2.kv_factor = 2
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
    fake_v2._page_table_materializer = SimpleNamespace(uses_device_expansion=False)
    fake_v2.pp_layers = []
    fake_v2.layer_offsets = {}
    fake_v2.layer_to_pool_mapping_dict = {}
    return fake_v2


_TEST_MODEL_DIR: Optional[str] = None


def make_test_model_dir() -> str:
    """Create a real dense-model config for production layer partitioning."""
    global _TEST_MODEL_DIR
    if _TEST_MODEL_DIR is None:
        _TEST_MODEL_DIR = tempfile.mkdtemp(prefix="triattention_test_model_")
        config = {
            "architectures": ["LlamaForCausalLM"],
            "model_type": "llama",
            "num_hidden_layers": 2,
            "hidden_size": 64,
            "num_attention_heads": 4,
        }
        with open(os.path.join(_TEST_MODEL_DIR, "config.json"), "w") as handle:
            json.dump(config, handle)
    return _TEST_MODEL_DIR


def make_test_calibration_pt() -> str:
    """A real on-disk flat calibration file: construction loads it for real."""
    path = os.path.join(make_test_model_dir(), "calibration.pt")
    if not os.path.exists(path):
        num_layers, num_heads, freq_count = 2, 2, 4
        torch.save(
            {
                "E_q": torch.zeros(num_layers, num_heads, freq_count, dtype=torch.complex64),
                "E_q_norm": torch.ones(num_layers, num_heads, freq_count),
                "omega": torch.ones(freq_count),
                "freq_scale_sq": torch.ones(freq_count),
            },
            path,
        )
    return path


def make_tri_config(**overrides):
    """Build a real TriAttention config with test calibration inputs."""
    from tensorrt_llm.llmapi.llm_args import TriAttentionKvCacheCompressionConfig

    options = {
        "budget": 8,
        "calibration_path": make_test_calibration_pt(),
    }
    options.update(overrides)
    return TriAttentionKvCacheCompressionConfig(**options)


def make_test_pretrained_config():
    """The test model's config, as the executor would hand it to the factory."""
    from transformers import AutoConfig

    return AutoConfig.from_pretrained(make_test_model_dir())


def make_triattention(**overrides):
    """Construct a manager while isolating GPU-owned persistent state."""
    from tensorrt_llm._torch.kv_cache_compression.triattention.triattention import (
        TriAttentionCompressionManager,
    )

    with mock.patch.object(TriAttentionCompressionManager, "_initialize_eviction_state"):
        return TriAttentionCompressionManager(
            make_tri_config(**overrides),
            make_fake_v2(),
            pretrained_config=make_test_pretrained_config(),
        )


def make_eviction_request(
    request=None,
    *,
    request_id=0,
    source_length,
    target_tail_length=0,
    target_cache=None,
    draft_cache=None,
):
    """One due request shaped exactly like ``_evict_due_requests`` builds."""
    from tensorrt_llm._torch.kv_cache_compression.triattention.triattention import _EvictionRequest

    if request is None:
        request = SimpleNamespace(
            py_request_id=request_id,
            py_prompt_len=0,
            py_num_compressed_tokens=0,
        )
    return _EvictionRequest(
        request=request,
        target_cache=target_cache,
        draft_cache=draft_cache,
        source_length=int(source_length),
        target_tail_length=int(target_tail_length),
    )


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
    """Run the real ``_evict_due_requests`` transaction around a mocked round executor."""
    with mock.patch.object(manager, "_execute_eviction_round") as execute:
        yield SimpleNamespace(execute=execute)


def torch_tri_score_oracle(
    layer_pools,
    page_ids,
    seq_lens,
    logical_source_lengths,
    q_real,
    q_imag,
    mlr_coef,
    freq_scale_sq,
    omega,
    offsets,
    layer_indices,
):
    """Compute paged mean scores independently with Torch."""
    scores = []
    num_q_heads = int(q_real.shape[1])
    for request, seq_len in enumerate(seq_lens):
        phase = (logical_source_lengths[request] + offsets[:, None]) * omega[None, :]
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
    """Build the semantic phase-table surface consumed by an eviction round."""
    omega = omega.to(dtype=torch.float32).contiguous()
    positions = torch.arange(max(int(initial_rows), 1), dtype=torch.float32, device=omega.device)
    angles = (positions[:, None, None] + offsets[None, :, None]) * omega[None, None, :]
    num_freqs = int(omega.numel())
    return SimpleNamespace(
        cos=torch.cos(angles).mean(dim=1).contiguous(),
        sin=torch.sin(angles).mean(dim=1).contiguous(),
        num_freqs=num_freqs,
    )


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
    protected_tail_capacity=0,
    layer_pool_ids=None,
    normalize_scores=True,
):
    """Build a bare manager with a real score pipeline over test pools."""
    from tensorrt_llm._torch.kv_cache_compression.triattention.triattention import (
        TriAttentionCompressionManager,
    )

    num_layers = len(layer_pools)
    assert int(q_real.shape[1]) == num_q_heads
    if decode_width is None:
        decode_width = seq_len
    if layer_pool_ids is None:
        layer_pool_ids = [0] * num_layers
    requested_tokens = seq_len + protected_tail_capacity
    tokens_per_block = int(layer_pools[0].shape[3])
    source_blocks = -(-int(requested_tokens) // tokens_per_block)
    source_blocks = (source_blocks + 3) // 4 * 4
    layout = dict(
        layer_pools=layer_pools,
        dense_layers=list(range(num_layers)),
        swa_layers=[],
        swa_window=None,
        layer_pool_ids=layer_pool_ids,
    )
    manager = TriAttentionCompressionManager.__new__(TriAttentionCompressionManager)
    manager.kv_cache_manager = SimpleNamespace(
        num_pools=max(layer_pool_ids) + 1,
        tokens_per_block=tokens_per_block,
        max_blocks_per_seq=source_blocks,
        host_kv_cache_block_offsets=torch.empty(1, 1, 2, source_blocks, dtype=torch.int32),
        mapping=SimpleNamespace(tp_size=1, tp_rank=0, enable_attention_dp=False),
    )
    manager.draft_kv_cache_manager = None
    manager._draft_protected_tail_capacity = 0
    manager.eviction_mode = eviction_mode
    manager.normalize_scores = normalize_scores
    manager._request_capacity = max_requests
    manager._selection_width_capacity = decode_width
    manager._phase = make_phase_table(offsets, omega, seq_len)
    manager.budget = keep_count
    manager._protected_tail_capacity = protected_tail_capacity
    manager._freq_scale_sq = freq_scale_sq
    manager._score_q_real = q_real
    manager._score_q_imag = q_imag
    manager._score_mlr_coef = mlr_coef
    manager._target_layout = layout
    manager._draft_layout = None
    manager._num_layers = num_layers
    manager._num_q_heads = num_q_heads
    manager._num_kv_heads = int(layer_pools[0].shape[2])
    manager._union_tp_mapping = None
    manager._swa_window = None
    manager._allocate_metadata_buffers(
        layer_pools[0].device,
        num_freqs=int(q_real.shape[2]),
    )
    manager._allocate_selection_buffers(layer_pools[0].device, tp_size=1)
    manager._score_scratch = None
    manager._score_token_capacity = 0
    manager._launch_score = None
    manager._compaction_params = ()
    manager._staging_reuse_event = torch.cuda.Event()
    manager._staging_reuse_event.record(torch.cuda.current_stream(layer_pools[0].device))
    manager._compaction_done_event = torch.cuda.Event()
    manager._compaction_done_event.record(torch.cuda.current_stream(layer_pools[0].device))
    manager._build_score_runtime(score_token_capacity=seq_len)
    return manager


def write_block_offsets(manager, encoded):
    """Load a test page table into the staged block-offset plane."""
    manager._block_offsets_device.zero_()
    manager._block_offsets_device[:, : encoded.shape[1], :, : encoded.shape[-1]].copy_(encoded)


def rect_to_score_scratch(scores, num_kv_heads, padded_head_columns=8):
    """Scatter rectangular scores into the fused scorer's scratch layout."""
    request_count, num_layers, num_q_heads, width = scores.shape
    group = num_q_heads // num_kv_heads
    scratch = torch.zeros(
        num_kv_heads * padded_head_columns * request_count * num_layers * width,
        dtype=torch.float32,
        device=scores.device,
    )
    view = scratch.view(num_kv_heads, padded_head_columns, request_count, num_layers, width)
    view[:, :group] = scores.view(request_count, num_layers, num_kv_heads, group, width).permute(
        2, 3, 0, 1, 4
    )
    prompt_lengths = torch.zeros(request_count, dtype=torch.int32, device=scores.device)
    return scratch, prompt_lengths


def stage_score_metadata(manager, request_count, source_lengths, decode_lengths, prompt_lengths):
    """Stage per-round score metadata exactly as production does."""
    torch.sub(
        source_lengths[:request_count],
        prompt_lengths[:request_count],
        out=decode_lengths[:request_count],
    )
    manager._source_lengths_device[:request_count].copy_(source_lengths[:request_count])
    manager._prompt_lengths_device[:request_count].copy_(prompt_lengths[:request_count])


def launch_split_scores(
    manager, request_count, source_lengths, decode_lengths, prompt_lengths, mean_cos, mean_sin
):
    """Run the score pipeline and gather its per-head decode-window rectangle."""
    stage_score_metadata(manager, request_count, source_lengths, decode_lengths, prompt_lengths)
    manager._mean_cos[:request_count].copy_(mean_cos[:request_count])
    manager._mean_sin[:request_count].copy_(mean_sin[:request_count])
    manager._launch_score(request_count)
    score_scratch = manager._score_scratch
    score_token_capacity = manager._score_token_capacity
    num_segments = request_count * manager._num_layers
    group_size = manager._num_q_heads // manager._num_kv_heads
    source = (
        score_scratch[: manager._num_kv_heads * 8 * num_segments * score_token_capacity]
        .view(
            manager._num_kv_heads,
            8,
            request_count,
            manager._num_layers,
            score_token_capacity,
        )[:, :group_size]
        .permute(2, 3, 0, 1, 4)
    )
    columns = prompt_lengths[:request_count].to(torch.int64).view(-1, 1, 1, 1, 1) + torch.arange(
        manager._selection_width_capacity,
        dtype=torch.int64,
        device=score_scratch.device,
    ).view(1, 1, 1, 1, -1)
    columns = columns.clamp_(max=score_token_capacity - 1).expand(
        request_count,
        manager._num_layers,
        manager._num_kv_heads,
        group_size,
        manager._selection_width_capacity,
    )
    output = torch.full(
        (
            request_count,
            manager._num_layers,
            manager._num_q_heads,
            manager._selection_width_capacity,
        ),
        float("nan"),
        dtype=torch.float32,
        device=score_scratch.device,
    )
    torch.gather(
        source,
        4,
        columns,
        out=output.view(
            request_count,
            manager._num_layers,
            manager._num_kv_heads,
            group_size,
            manager._selection_width_capacity,
        ),
    )
    return output
