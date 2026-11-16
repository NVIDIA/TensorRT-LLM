# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The SM100 TriAttention CuTe scorer (the only score path) vs oracles.

The launch matrix drives the named production geometries against the
pure-PyTorch oracle; the contract test pins the no-fallback loud raise.
"""

import pytest
import torch
from conftest import encode_block_offsets as _encode_block_offsets
from conftest import launch_split_scores as _launch_split_scores
from conftest import make_cute_buffers as _make_cute_buffers
from conftest import torch_tri_score_oracle as _torch_tri_score_oracle
from conftest import write_block_offsets as _write_block_offsets

requires_sm100 = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 0),
    reason="TriAttention score requires SM100",
)


def _build_case(
    *,
    max_requests: int,
    num_layers: int,
    page_count: int,
    tokens_per_block: int,
    head_dim: int,
    num_q_heads: int,
    num_kv_heads: int,
    prompt_len: int,
    seed: int,
    offsets: tuple = (1.0, 2.0, 4.0),
):
    device = torch.device("cuda", torch.cuda.current_device())
    torch.manual_seed(seed)
    num_freqs = head_dim // 2
    # 0.125 scaling keeps the kernel-vs-oracle tolerance tight.
    pools = [
        (
            0.125
            * torch.randn(
                max_requests * page_count,
                2,
                num_kv_heads,
                tokens_per_block,
                head_dim,
                device=device,
            )
        ).to(torch.bfloat16)
        for _ in range(num_layers)
    ]
    page_ids = torch.randperm(max_requests * page_count).view(max_requests, page_count).to(device)
    q_real = 0.125 * torch.randn(num_layers, num_q_heads, num_freqs, device=device)
    q_imag = 0.125 * torch.randn(num_layers, num_q_heads, num_freqs, device=device)
    mlr_coef = 0.125 * torch.randn(num_layers, num_q_heads, num_freqs, device=device)
    freq_scale_sq = torch.rand(num_freqs, device=device) + 0.5
    omega = torch.rand(num_freqs, device=device) * 0.05
    offsets_t = torch.tensor(offsets, dtype=torch.float32, device=device)
    capacity = page_count * tokens_per_block
    tri = _make_cute_buffers(
        eviction_mode="per_head",
        layer_pools=pools,
        max_requests=max_requests,
        seq_len=capacity,
        num_q_heads=num_q_heads,
        q_real=q_real,
        q_imag=q_imag,
        mlr_coef=mlr_coef,
        freq_scale_sq=freq_scale_sq,
        omega=omega,
        offsets=offsets_t,
        decode_width=capacity - prompt_len,
    )
    _write_block_offsets(tri, _encode_block_offsets(page_ids))
    logical_source_lengths = (
        torch.arange(max_requests, dtype=torch.int32, device=device) + 9
    ).contiguous()
    prompt_lengths = torch.full((max_requests,), prompt_len, dtype=torch.int32, device=device)
    # Mid-page/mid-tile tails; 58 leaves a fully-invalid trailing fragment.
    tail_cuts = (0, 58, 3, 33)
    seq_lens = [capacity - tail_cuts[request % len(tail_cuts)] for request in range(max_requests)]
    source_lengths = torch.tensor(seq_lens, dtype=torch.int32, device=device)
    phase = (logical_source_lengths.float()[:, None, None] + offsets_t[None, :, None]) * omega[
        None, None, :
    ]
    mean_cos = torch.cos(phase).mean(dim=1).contiguous()
    mean_sin = torch.sin(phase).mean(dim=1).contiguous()
    oracle_inputs = dict(
        page_ids=page_ids,
        q_real=q_real,
        q_imag=q_imag,
        mlr_coef=mlr_coef,
        freq_scale_sq=freq_scale_sq,
        omega=omega,
        offsets=offsets_t,
    )
    return (
        tri,
        pools,
        prompt_lengths,
        source_lengths,
        seq_lens,
        mean_cos,
        mean_sin,
        oracle_inputs,
    )


def _geometry(max_requests, num_layers, page_count, tokens_per_block, head_dim, num_q, num_kv):
    return dict(
        max_requests=max_requests,
        num_layers=num_layers,
        page_count=page_count,
        tokens_per_block=tokens_per_block,
        head_dim=head_dim,
        num_q_heads=num_q,
        num_kv_heads=num_kv,
    )


# One entry per supported production geometry.
_CASES = [
    pytest.param(_geometry(4, 2, 4, 32, 128, 8, 2), id="qwen3_f64_group4_tpb32"),
    pytest.param(_geometry(2, 3, 4, 32, 64, 8, 1), id="gptoss_f32_group8_tpb32"),
    pytest.param(_geometry(2, 2, 2, 128, 64, 8, 1), id="original_f32_group8_tpb128"),
]


@requires_sm100
@pytest.mark.parametrize("case", _CASES)
def test_cute_kernel_matches_torch_oracle(case):
    pytest.importorskip("cutlass")
    case = dict(case)  # parametrize reuses the dict across reruns
    prompt_len = 5
    max_requests = case["max_requests"]
    num_layers = case["num_layers"]
    (
        tri,
        pools,
        prompt_lengths,
        source_lengths,
        seq_lens,
        mean_cos,
        mean_sin,
        oracle_inputs,
    ) = _build_case(prompt_len=prompt_len, seed=20260719, **case)
    device = tri._score_scratch.device

    oracle = _torch_tri_score_oracle(
        pools,
        oracle_inputs["page_ids"],
        seq_lens,
        [int(start) for start in range(9, 9 + max_requests)],
        oracle_inputs["q_real"],
        oracle_inputs["q_imag"],
        oracle_inputs["mlr_coef"],
        oracle_inputs["freq_scale_sq"],
        oracle_inputs["omega"],
        oracle_inputs["offsets"],
        list(range(num_layers)),
    )

    # Every request count used by the runtime dispatches through the launcher.
    for request_count in dict.fromkeys((1, max_requests - 1, max_requests)):
        decode_lengths = torch.full((max_requests,), -1, dtype=torch.int32, device=device)
        scores = _launch_split_scores(
            tri,
            request_count,
            source_lengths,
            decode_lengths,
            prompt_lengths,
            mean_cos,
            mean_sin,
        )
        assert scores.shape == (
            request_count,
            num_layers,
            case["num_q_heads"],
            tri._selection_width_capacity,
        )
        # The score leg owns the per-request decode widths the selection
        # reduce kernels consume.
        assert decode_lengths[:request_count].tolist() == [
            seq_lens[request] - prompt_len for request in range(request_count)
        ]
        for request in range(request_count):
            width = seq_lens[request] - prompt_len
            for layer in range(num_layers):
                torch.testing.assert_close(
                    scores[request, layer, :, :width],
                    oracle[request * num_layers + layer][:, prompt_len : prompt_len + width],
                    rtol=5e-3,
                    atol=5e-3,
                )


def test_unsupported_geometry_raises_at_buffer_construction():
    pytest.importorskip("cutlass")
    device = torch.device("cuda", torch.cuda.current_device())
    torch.manual_seed(20260722)
    num_layers, max_requests, page_count, tokens_per_block, head_dim = 2, 2, 2, 4, 8
    num_freqs = head_dim // 2
    # fp32, 4-token pages, 4 freqs: outside the contract on every device.
    pools = [
        torch.randn(max_requests * page_count, 2, 1, tokens_per_block, head_dim, device=device)
        for _ in range(num_layers)
    ]
    calib = torch.randn(num_layers, 2, num_freqs, device=device)
    # No rewrap: the buffer build's own contract error surfaces directly
    # (the fp32 pools trip the BF16 gate first, at TMA descriptor encoding).
    with pytest.raises(TypeError, match="BF16"):
        _make_cute_buffers(
            eviction_mode="per_head",
            layer_pools=pools,
            max_requests=max_requests,
            seq_len=page_count * tokens_per_block,
            num_q_heads=2,
            q_real=calib,
            q_imag=calib.clone(),
            mlr_coef=calib.clone(),
            freq_scale_sq=torch.rand(num_freqs, device=device) + 0.5,
            omega=torch.rand(num_freqs, device=device) * 0.05,
            offsets=torch.tensor([1.0, 2.0], dtype=torch.float32, device=device),
            decode_width=page_count * tokens_per_block - 1,
        )


@requires_sm100
def test_compact_workspace_matches_absolute_scores_and_bounded_phase():
    """Long mixed prompts use decode-sized scratch without changing scores."""
    from unittest import mock

    from tensorrt_llm._torch.kv_cache_compression.triattention.triattention_cute_score_fused import (
        build_score_pipeline,
    )
    from tensorrt_llm._torch.kv_cache_compression.triattention.triattention_kernels import (
        compute_mean_phase,
        reduce_per_head_scores,
    )

    tri, _, prompts, lengths, _, _, _, _ = _build_case(
        max_requests=2,
        num_layers=2,
        page_count=64,
        tokens_per_block=128,
        head_dim=128,
        num_q_heads=8,
        num_kv_heads=1,
        prompt_len=7000,
        seed=19,
    )
    prompts.copy_(torch.tensor([7003, 4097], device="cuda", dtype=torch.int32))
    lengths.copy_(prompts + 193)
    tri._prompt_lengths_device.copy_(prompts)
    tri._source_lengths_device.copy_(lengths)
    tri._logical_source_lengths_device.copy_(
        torch.tensor([8192, 1048575], device="cuda", dtype=torch.int32)
    )
    compute_mean_phase(
        tri._logical_source_lengths_device,
        tri._phase_omega,
        lengths,
        prompts,
        tri._mean_cos,
        tri._mean_sin,
        tri._decode_lengths_device,
        None,
        request_count=2,
        swa_rebase_delta=0,
    )
    cosine = torch.zeros_like(tri._mean_cos)
    sine = torch.zeros_like(tri._mean_sin)
    for exponent in range(17):
        angle = (
            tri._logical_source_lengths_device.float()[:, None] + float(1 << exponent)
        ) * tri._phase_omega
        cosine += angle.cos()
        sine += angle.sin()
    torch.testing.assert_close(tri._mean_cos, cosine * (1.0 / 17), rtol=0, atol=0)
    torch.testing.assert_close(tri._mean_sin, sine * (1.0 / 17), rtol=0, atol=0)

    outputs = []
    scratch_sizes = []
    for compact, capacity in ((False, 8192), (True, 384)):
        scratch, launch = build_score_pipeline(
            tri._target_layout,
            block_offsets=tri._block_offsets_device,
            source_lengths=lengths,
            prompt_lengths=prompts,
            mean_cos=tri._mean_cos,
            mean_sin=tri._mean_sin,
            q_real=tri._score_q_real,
            q_imag=tri._score_q_imag,
            mlr_coef=tri._score_mlr_coef,
            freq_scale_sq=tri._freq_scale_sq,
            score_token_capacity=capacity,
            compact_scores=compact,
            resident_requests=1 if compact else None,
        )
        resident_scratch = scratch
        if compact:
            scratch = torch.empty(resident_scratch.numel() * 2, device="cuda")
            launch(2, scratch)
            for count in (0, 3):
                with pytest.raises(ValueError, match="request count"):
                    launch(count, scratch)
            invalid_workspaces = (
                resident_scratch,
                scratch.to(torch.float16),
                torch.empty(scratch.numel(), device="cpu"),
                torch.empty(scratch.numel() * 2, device="cuda")[::2],
            )
            for workspace in invalid_workspaces:
                with pytest.raises(ValueError, match="score workspace"):
                    launch(2, workspace)
            del invalid_workspaces, workspace
        else:
            launch(2)
        reduce_per_head_scores(
            scratch,
            tri._decode_lengths_device,
            prompts,
            tri._row_mean,
            tri._row_inv_std,
            tri._selection_scores_rows,
            tri._selection_row_lengths,
            request_count=2,
            padded_head_columns=8,
            score_token_capacity=capacity,
            per_layer=False,
            normalize_scores=True,
            compact_scores=compact,
        )
        outputs.append(tri._selection_scores_rows.clone())
        scratch_sizes.append(scratch.numel())
    torch.testing.assert_close(outputs[0], outputs[1], rtol=0, atol=0)
    assert scratch_sizes[1] * 20 < scratch_sizes[0]
    # Driver headroom must not reject an allocation the caching allocator can serve.
    with mock.patch.object(torch.cuda, "mem_get_info", side_effect=AssertionError("driver query")):
        for _ in range(2):
            optional = tri._try_score_workspace(2)
            assert optional is not None
            del optional
    assert tri._batch_workspace_attempts == 2
    assert tri._batch_workspace_ooms == 0

    with mock.patch.object(torch.cuda, "mem_get_info") as query:
        assert tri._try_score_workspace(1) is None
        with mock.patch.object(tri, "_union_tp_mapping", object()):
            assert tri._try_score_workspace(2) is None
        with pytest.raises(ValueError, match="batch capacity"):
            tri._try_score_workspace(tri._request_capacity + 1)
        query.assert_not_called()

    # Exercise allocation failure through the scheduling entry point, including
    # publication and resize, on the first eviction after initialization.
    from types import SimpleNamespace

    from conftest import make_eviction_request, make_request

    original_pools = [pool.clone() for pool in tri._target_layout["layer_pools"]]
    original_offsets = tri._block_offsets_device.clone()
    requests = [make_request(i, py_prompt_len=int(prompts[i])) for i in range(2)]
    batch = SimpleNamespace(generation_requests=requests)
    tri.kv_cache_manager._stream = torch.cuda.current_stream()
    tri._prepared_generation_batch = None
    tri._num_extra_kv_tokens = 0
    tri._overlap_tail_length = 1
    tri.beta = 64
    tri._global_layers = list(range(2))
    tri._calibration_q_real = tri._score_q_real
    tri._calibration_q_imag = tri._score_q_imag
    tri._calibration_mlr_coef = tri._score_mlr_coef
    tri._omega = tri._phase_omega
    layout = tri._target_layout

    def stage(manager, request_ids, host, device):
        for row, request_id in enumerate(request_ids):
            device[:, row].copy_(original_offsets[:, request_id])

    with mock.patch.object(tri, "_stage_block_offset_snapshot", side_effect=stage):
        for mode in ("union", "per_head", "per_layer_perhead"):
            tri.eviction_mode = mode
            tri.budget = 32
            tri._selection_width_capacity = 256
            for pool, original in zip(layout["layer_pools"], original_pools):
                pool.copy_(original)
            with mock.patch.object(tri, "_create_kv_layout", return_value=layout):
                tri._initialize_eviction_state()
            # Initialization launches the fallback kernels with no KV reads or
            # moves: the entire nonzero pool must remain unchanged.
            for pool, original in zip(layout["layer_pools"], original_pools):
                torch.testing.assert_close(pool, original, rtol=0, atol=0)
            tri.kv_cache_manager.kv_cache_map = {
                i: SimpleNamespace(
                    is_active=True,
                    capacity=int(lengths[i]),
                    history_length=int(lengths[i]),
                    resize=mock.Mock(return_value=True),
                )
                for i in range(2)
            }
            for request in requests:
                request.py_num_compressed_tokens = 0
            torch.cuda.synchronize()
            allocated = torch.cuda.memory_allocated()
            torch.cuda.reset_peak_memory_stats()
            with (
                mock.patch.object(
                    torch.cuda, "mem_get_info", side_effect=AssertionError("driver query")
                ),
                mock.patch.object(torch, "empty", side_effect=torch.OutOfMemoryError) as allocate,
                mock.patch.object(
                    tri, "_execute_eviction_round", wraps=tri._execute_eviction_round
                ) as execute,
            ):
                tri._evict_due_requests(batch)
            torch.cuda.synchronize()
            allocate.assert_called_once()
            assert (
                tri._batch_workspace_ooms
                == ("union", "per_head", "per_layer_perhead").index(mode) + 1
            )
            assert execute.call_count == 2
            assert [call.args[0][0].request for call in execute.call_args_list] == requests
            assert all(
                len(call.args[0]) == 1 and not call.kwargs for call in execute.call_args_list
            )
            assert torch.cuda.max_memory_allocated() == allocated
            for i, request in enumerate(requests):
                assert request.py_num_compressed_tokens == 193 - tri.budget
                tri.kv_cache_manager.kv_cache_map[i].resize.assert_called_once_with(
                    int(prompts[i]) + tri.budget, None
                )
            fallback_pools = [pool.clone() for pool in layout["layer_pools"]]

            # The batch reference needs only a small explicit allocation; it
            # must not depend on the shared GPU's free-memory admission check.
            for request in requests:
                request.py_num_compressed_tokens = 0
            for pool, original in zip(layout["layer_pools"], original_pools):
                pool.copy_(original)
            optional = torch.empty(tri._score_scratch.numel() * 2, device="cuda")
            eviction_requests = [
                make_eviction_request(request=request, source_length=int(lengths[i]))
                for i, request in enumerate(requests)
            ]
            tri._execute_eviction_round(eviction_requests, score_workspace=optional)
            for pool, expected in zip(layout["layer_pools"], fallback_pools):
                torch.testing.assert_close(pool, expected, rtol=0, atol=0)
            del optional

    assert tri._score_scratch.numel() == 8 * 2 * (tri._selection_width_capacity + 128)
    assert tri._score_token_capacity < int(prompts.min())
    ptr = tri._score_scratch.data_ptr()
    with mock.patch.object(torch, "empty", side_effect=AssertionError("admission allocated")):
        tri.on_request_init(make_request(0, py_prompt_len=7003, py_max_new_tokens=100000))
    assert tri._score_scratch.data_ptr() == ptr


@requires_sm100
def test_resident_selection_at_maximum_budget():
    """The resident selector compiles, settles ties, and reuses storage at k=16384."""
    from tensorrt_llm._torch.kv_cache_compression.triattention.triattention import (
        TriAttentionCompressionManager,
    )

    manager = TriAttentionCompressionManager.__new__(TriAttentionCompressionManager)
    manager.budget = 16384
    manager._selection_rows_per_request = 1
    # budget + beta + one token, rounded to the production 256-token alignment.
    width = 16640
    manager._selection_scores_rows = torch.zeros((1, width), device="cuda")
    manager._selection_row_lengths = torch.full((1,), width, device="cuda", dtype=torch.int32)
    manager._prompt_lengths_device = torch.tensor([17], device="cuda", dtype=torch.int32)
    manager._provisional_rows = torch.empty((1, manager.budget), device="cuda", dtype=torch.int32)
    manager._kept_ordinal_rows = torch.empty_like(manager._provisional_rows)
    # Exercise the same all-equal selection input used by initialization warmup.
    manager._select_kept_ordinals(1)
    expected = torch.arange(17, 17 + manager.budget, device="cuda", dtype=torch.int32)[None]
    torch.testing.assert_close(manager._kept_ordinal_rows, expected, rtol=0, atol=0)
    scores = torch.randperm(width, device="cuda").float()[None]
    manager._selection_scores_rows.copy_(scores)
    expected = torch.topk(scores, manager.budget).indices.sort(dim=-1).values.int() + 17
    torch.cuda.synchronize()
    allocated = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    manager._select_kept_ordinals(1)
    torch.cuda.synchronize()
    assert torch.cuda.max_memory_allocated() == allocated
    torch.testing.assert_close(manager._kept_ordinal_rows, expected, rtol=0, atol=0)
