# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exercise prior-state ownership through the actual indexer forward path."""

from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

from tensorrt_llm._torch.attention.backends.sparse.dsa.indexer import Indexer
from tensorrt_llm._torch.modules.top_k import TopK, TopKImplementation


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA"),
        ),
    ],
)
def test_update_gvr_prior_from_prefill_uses_device_lengths(device: str) -> None:
    """Seed the last prefill row using lengths on the selections' device."""
    top_k = TopK(2, decode_implementation=TopKImplementation.CUTE_DSL_GVR, gvr_self_sampling=False)
    prefill_indices = torch.tensor([[0, 1], [2, 3], [4, 5]], dtype=torch.int32, device=device)
    prior_indices = torch.zeros(3, 2, dtype=torch.int32, device=device)

    # Production passes the device seq_lens twin so the row gather stays async.
    top_k.update_gvr_prior_from_prefill(
        prefill_indices,
        torch.tensor([2, 1], dtype=torch.int32, device=device),
        prior_indices,
        request_offset=1,
    )

    assert prior_indices.tolist() == [[0, 0], [2, 3], [4, 5]]
    assert top_k.needs_gvr_prior


@pytest.mark.parametrize(
    "implementation,self_sampling",
    [
        (TopKImplementation.CUDA_RADIX, False),
        (TopKImplementation.CUTE_DSL_RADIX, False),
        (TopKImplementation.CUTE_DSL_GVR, True),
        (TopKImplementation.CUTE_DSL_GVR, False),
    ],
)
@pytest.mark.parametrize("phase", ["prefill", "decode", "mixed", "split_prefill", "split_decode"])
@pytest.mark.parametrize("next_n", [1, 4])
def test_indexer_forward_uses_prior_only_for_temporal_gvr(
    implementation: TopKImplementation, self_sampling: bool, phase: str, next_n: int
) -> None:
    """Preserve prior ownership and pass the device lengths through each phase."""
    topk = 2
    top_k = TopK(topk, decode_implementation=implementation, gvr_self_sampling=self_sampling)
    num_contexts = 0 if phase == "decode" else 2
    num_generations = 0 if phase == "prefill" else 2
    num_ctx_tokens = 5 if num_contexts else 0
    num_gen_tokens = num_generations * next_n
    total_tokens = num_ctx_tokens + num_gen_tokens
    is_generation = {"split_prefill": False, "split_decode": True}.get(phase)
    has_prefill = num_contexts > 0 and is_generation is not True
    has_decode = num_generations > 0 and is_generation is not False
    input_tokens = (
        total_tokens
        if is_generation is None
        else (num_gen_tokens if is_generation else num_ctx_tokens)
    )
    cache_manager = SimpleNamespace(
        quant_block_size=128,
        layer_offsets={7: 1},
        get_indexer_k_cache_buffers=Mock(return_value=torch.empty(0)),
    )
    metadata = SimpleNamespace(
        kv_cache_manager=cache_manager,
        num_contexts=num_contexts,
        num_generations=num_generations,
        num_ctx_tokens=num_ctx_tokens,
        num_tokens=total_tokens,
        seq_lens=torch.tensor(([2, 3] if num_contexts else []) + [next_n] * num_generations),
        cuda_graph_buffers={},
        is_cuda_graph=False,
        get_empty=lambda buffers, shape, **kwargs: torch.full(shape, -1, dtype=torch.int32),
        skip_indexer_for_ctx_reqs=False,
        skip_indexer_for_gen_reqs=False,
        indexer_prefill_chunks=None,
        num_ctx_kv_tokens=16,
        cu_seqlen_ks=torch.zeros(num_ctx_tokens, dtype=torch.int32),
        cu_seqlen_ke=torch.full((num_ctx_tokens,), 16, dtype=torch.int32),
        use_expanded_buffers_for_mtp=False,
        kv_lens_cuda_2d=torch.full((num_generations, next_n), 16, dtype=torch.int32),
        indexer_k_cache_block_offsets=torch.empty((num_contexts + num_generations, 1)),
        scheduler_metadata_buffer=None,
        scheduler_metadata_buffer_full_next_n=None,
        get_indexer_max_seq_len=lambda: 16,
        kv_lens_cuda_runtime=torch.full((num_contexts + num_generations,), 16),
        gen_indexer_kv_lens_cuda_runtime=torch.full((num_generations,), 16),
        kv_lens_row_reorder=None,
    )
    # Keep distinct storage so the caller test catches use of the host twin.
    metadata.seq_lens_cuda = metadata.seq_lens.clone()
    # Radix fallback and self-sampling deliberately have no prior attribute.
    # The raw heuristic option remains True, as in the SM107 failure.
    if top_k.needs_gvr_prior:
        metadata.gvr_prior_indices = torch.full(
            (2, num_contexts + num_generations, topk), -99, dtype=torch.int32
        )

    def logits(q: torch.Tensor, *args: object, **kwargs: object) -> torch.Tensor:
        return torch.ones((q.shape[0], 16))

    def selections(rows: int) -> torch.Tensor:
        return torch.arange(rows * topk, dtype=torch.int32).reshape(rows, topk)

    def decode(
        scores: torch.Tensor,
        sequence_lengths: torch.Tensor,
        scan_lengths: torch.Tensor,
        output: torch.Tensor,
        n: int,
        max_seq_len: int,
        extra: dict[str, torch.Tensor | None],
    ) -> torch.Tensor:
        prior = extra["gvr_prior_indices"]
        if top_k.needs_gvr_prior:
            assert prior.shape == (num_generations, topk)
            assert prior.data_ptr() == metadata.gvr_prior_indices[1].data_ptr()
            assert torch.all(prior == -99)
        else:
            assert prior is None
        assert n == next_n
        output.copy_(selections(num_gen_tokens))
        return output

    def prefill(
        scores: torch.Tensor,
        starts: torch.Tensor,
        ends: torch.Tensor,
        output: torch.Tensor,
        k: int,
    ) -> None:
        output.copy_(selections(num_ctx_tokens))

    indexer = SimpleNamespace(
        top_k=top_k,
        index_topk=topk,
        layer_idx=7,
        _enable_heuristic_topk=True,
        mtp_index_share=False,
        use_fp4=False,
        use_cute_dsl_paged_mqa_logits=False,
        _call_mqa_logits=logits,
        _call_paged_mqa_logits=Mock(return_value=torch.ones((num_gen_tokens, 16))),
        aux_stream=None,
    )
    with (
        patch.object(
            top_k, "update_gvr_prior_from_prefill", wraps=top_k.update_gvr_prior_from_prefill
        ) as seed_call,
        patch.object(top_k, "_forward_decode", side_effect=decode) as decode_call,
        patch.object(torch.ops.trtllm, "indexer_topk_prefill", side_effect=prefill),
    ):
        result = Indexer.sparse_attn_indexer(
            indexer,
            metadata,
            torch.empty((input_tokens, 2)),
            torch.empty((input_tokens, 1, 2)),
            torch.empty((16, 2)),
            torch.empty(16),
            torch.empty(input_tokens),
            is_generation=is_generation,
        )
    assert decode_call.call_count == int(has_decode)
    assert seed_call.call_count == int(top_k.needs_gvr_prior and has_prefill)
    if seed_call.called:
        assert seed_call.call_args.args[1].data_ptr() == metadata.seq_lens_cuda.data_ptr()
    token_offset = num_ctx_tokens if is_generation is None else 0
    if has_prefill:
        torch.testing.assert_close(result[:num_ctx_tokens], selections(num_ctx_tokens))
    if has_decode:
        torch.testing.assert_close(result[token_offset:], selections(num_gen_tokens))
    if top_k.needs_gvr_prior:
        expected = torch.full_like(metadata.gvr_prior_indices, -99)
        if has_prefill:
            expected[1, num_generations:] = selections(num_ctx_tokens)[[1, 4]]
        if has_decode:
            expected[1, :num_generations] = selections(num_gen_tokens)[next_n - 1 :: next_n]
        torch.testing.assert_close(metadata.gvr_prior_indices, expected)


@pytest.mark.parametrize("batch_size", [1, 8])
@pytest.mark.parametrize("next_n", [1, 4])
@pytest.mark.parametrize("score_width", [8192, 262144])
def test_self_sampling_gpu_exactness_and_graph_replay(
    batch_size: int, next_n: int, score_width: int
) -> None:
    from tensorrt_llm._torch.cute_dsl_utils import IS_CUTLASS_DSL_AVAILABLE
    from tensorrt_llm._utils import get_sm_version

    if (
        not torch.cuda.is_available()
        or not IS_CUTLASS_DSL_AVAILABLE
        or get_sm_version() not in (100, 103, 107)
    ):
        pytest.skip("Self-sampling GVR requires datacenter Blackwell or Rubin and CuTe DSL")

    topk = 512
    compress_ratio = 4
    rows = batch_size * next_n
    generator = torch.Generator(device="cuda").manual_seed(42)
    scores = torch.randn((rows, score_width), generator=generator, device="cuda")
    lengths = score_width * compress_ratio - 68 * torch.arange(
        batch_size, dtype=torch.int32, device="cuda"
    )
    row_ids = torch.arange(rows, device="cuda")
    valid_columns = (lengths[row_ids // next_n] - next_n + row_ids % next_n + 1) // compress_ratio
    valid = torch.arange(score_width, device="cuda")[None, :] < valid_columns[:, None]
    # Invalid tail values must never enter the result.
    scores.masked_fill_(~valid, 1.0e6)
    output = torch.empty((rows, topk), dtype=torch.int32, device="cuda")
    top_k = TopK(
        topk,
        decode_implementation=TopKImplementation.CUTE_DSL_GVR,
        compress_ratio=compress_ratio,
        gvr_self_sampling=True,
    )
    assert not top_k.needs_gvr_prior

    def run() -> None:
        top_k(
            scores,
            output,
            is_prefill=False,
            sequence_lengths=lengths,
            scan_lengths=lengths // compress_ratio,
            next_n=next_n,
            max_seq_len=score_width,
        )

    def check_result() -> None:
        assert torch.all((output >= 0) & (output < valid_columns[:, None]))
        sorted_indices = output.sort(dim=-1).values
        assert torch.all(sorted_indices[:, 1:] != sorted_indices[:, :-1])
        actual = scores.gather(1, output.long()).sort(dim=-1, descending=True).values
        expected = scores.masked_fill(~valid, float("-inf")).topk(topk, dim=-1).values
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    run()
    torch.cuda.synchronize()
    check_result()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    output.fill_(-1)
    graph.replay()
    torch.cuda.synchronize()
    check_result()
