# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Distributed PCG cache regressions without loading a full M3 checkpoint."""

import pickle
import sys
from types import SimpleNamespace

import cloudpickle
import pytest
import torch
from mpi4py import MPI
from mpi4py.futures import MPIPoolExecutor

import tensorrt_llm
from tensorrt_llm._torch.attention.backends.sparse.minimax_m3 import MiniMaxM3MsaSparseAttention
from tensorrt_llm._torch.compilation.piecewise_optimizer import PiecewiseRunner
from tensorrt_llm._torch.compilation.utils import capture_piecewise_cuda_graph
from tensorrt_llm._torch.distributed import Distributed
from tensorrt_llm._torch.pyexecutor.engine.runners.common import get_padding_params
from tensorrt_llm._torch.utils import (
    get_per_request_prefill_cuda_graph_flag,
    model_extra_attrs,
    piecewise_cuda_graph,
    set_per_request_prefill_cuda_graph_flag,
)
from tensorrt_llm.llmapi.llm_args import PrefillCudaGraphBackend
from tensorrt_llm.mapping import Mapping

cloudpickle.register_pickle_by_value(sys.modules[__name__])
MPI.pickle.__init__(cloudpickle.dumps, cloudpickle.loads, pickle.HIGHEST_PROTOCOL)

# Match the other MPIPoolExecutor tests: its worker thread outlives the test.
pytestmark = pytest.mark.threadleak(enabled=False)


@torch.inference_mode()
def _run_empty_adp_rank(world_size: int) -> int:
    rank = tensorrt_llm.mpi_rank()
    torch.cuda.set_device(rank)
    torch.manual_seed(42 + rank)
    mapping = Mapping(
        world_size=world_size, rank=rank, tp_size=world_size, enable_attention_dp=True
    )
    dist = Distributed.get(mapping)
    bucket = 4
    num_heads_q, num_kv_heads, num_index_heads = 8, 2, 2
    width = (num_heads_q + 2 * num_kv_heads + num_index_heads + 1) * 128
    packed = torch.randn(bucket, width, dtype=torch.bfloat16, device="cuda")
    weights = [torch.ones(128, dtype=torch.bfloat16, device="cuda") for _ in range(4)]
    positions = torch.arange(bucket, dtype=torch.int32, device="cuda")
    frequency = torch.outer(
        positions.float(),
        5_000_000.0 ** (-torch.arange(0, 64, 2, dtype=torch.float32, device="cuda") / 64),
    )
    rope_cache = torch.stack((frequency.cos(), frequency.sin()), dim=1).contiguous()
    # HND views with interleaved pages, as in the producer's kernel tests.
    main_backing = torch.full(
        (6, 2, num_kv_heads, 128, 128), 1.0, dtype=torch.float8_e4m3fn, device="cuda"
    )
    index_backing = torch.full((10, 1, 128, 128), 2.0, dtype=torch.float8_e4m3fn, device="cuda")
    main_cache, index_cache = main_backing[::3], index_backing[::5]

    # Reuse the metadata fixture pattern from test_msa_backend. Only the cache
    # allocator is stubbed; slot construction/clearing and GPU writes are real.
    metadata_cls = MiniMaxM3MsaSparseAttention.Metadata
    metadata = metadata_cls.__new__(metadata_cls)
    metadata._msa_buffers_ready = True
    metadata.kv_cache_manager = SimpleNamespace(
        tokens_per_block=128,
        get_buffers=lambda layer_idx: main_cache,
        get_block_ids_per_seq=lambda request_ids: torch.tensor([[1]], dtype=torch.int32),
    )
    metadata.msa_out_cache_loc = torch.full((bucket,), -1, dtype=torch.int32, device="cuda")
    metadata.msa_block_table = torch.zeros((1, 1), dtype=torch.int32, device="cuda")
    metadata.msa_seq_lens_cuda = torch.zeros(1, dtype=torch.int32, device="cuda")
    metadata.msa_subpage_block_table = None
    metadata._msa_runs_no_fmha = lambda: True
    metadata._msa_kv_lens_may_change = lambda: False

    def producer(
        projection: torch.Tensor,
        kv: torch.Tensor,
        index_k: torch.Tensor,
        slots: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.ops.trtllm.minimax_m3_fp8_qkv_indexer_norm_rope_kv_insert(
            projection,
            kv,
            index_k,
            slots,
            num_heads_q,
            num_kv_heads,
            num_index_heads,
            128,
            64,
            1e-5,
            *weights,
            rope_cache,
            positions,
        )

    # Exercise the production PCG runner, including its actual capture/replay,
    # rather than replacing the producer or its cache mutations with mocks.
    graph = torch.fx.symbolic_trace(producer)
    runner = PiecewiseRunner(
        graph=graph,
        name="empty_adp_rank_producer",
        compile_time_num_tokens=bucket,
        runtime_num_tokens_idx=None,
        capture_num_tokens=[bucket],
        graph_pool_handle=torch.cuda.graph_pool_handle(),
        default_callable=graph.forward,
        enable_inductor=False,
        is_first_runner=True,
        is_last_runner=True,
    )
    previous_prefill_flag = get_per_request_prefill_cuda_graph_flag()
    try:
        with model_extra_attrs({}), piecewise_cuda_graph(True):
            # First capture with both ranks live, then alternate which rank is
            # empty. Each empty replay therefore has stale live slots to clear.
            for empty_rank in (None, 0, 1):
                live_tokens = 0 if rank == empty_rank else 3
                local_contexts = int(live_tokens > 0)
                all_tokens = dist.tp_allgather_int64([live_tokens])[:, 0].tolist()
                padded, eligible, padded_all_tokens = get_padding_params(
                    live_tokens,
                    local_contexts,
                    all_tokens,
                    dist=dist,
                    enable_attention_dp=True,
                    prefill_cuda_graph_backend=PrefillCudaGraphBackend.PIECEWISE,
                    prefill_cuda_graph_num_tokens=[bucket],
                )
                assert eligible and padded == bucket
                assert padded_all_tokens == [bucket] * world_size
                if empty_rank is not None:
                    assert all_tokens[empty_rank] == 0 and max(all_tokens) == 3
                set_per_request_prefill_cuda_graph_flag(eligible)

                metadata.request_ids = [0] if live_tokens else []
                metadata._msa_qo_lens_cpu = torch.tensor(
                    [live_tokens] if live_tokens else [], dtype=torch.int32
                )
                metadata._msa_kv_lens_cpu = metadata._msa_qo_lens_cpu.clone()
                metadata._msa_qo_offset_cpu = torch.zeros(local_contexts, dtype=torch.int32)
                metadata._build_msa_fields()
                assert metadata.msa_out_cache_loc.tolist() == list(
                    range(128, 128 + live_tokens)
                ) + [-1] * (bucket - live_tokens)
                packed.normal_()
                before_main, before_index = main_backing.clone(), index_backing.clone()
                args = (packed, main_cache, index_cache, metadata.msa_out_cache_loc)
                if empty_rank is None:
                    with capture_piecewise_cuda_graph(True):
                        # Three warmups, then capture. Keep the capture outputs
                        # alive because the runner stores non-owning views.
                        for _ in range(4):
                            capture_output = runner(*args)
                    assert runner.entries[bucket].cuda_graph is not None
                else:
                    runner(*args)
                torch.cuda.synchronize()
                assert all(torch.isfinite(output.float()).all() for output in capture_output)

                for cache, before, page in (
                    (main_backing, before_main, 3),
                    (index_backing, before_index, 5),
                ):
                    if live_tokens:
                        assert not torch.equal(
                            cache[page, ..., :live_tokens, :].view(torch.uint8),
                            before[page, ..., :live_tokens, :].view(torch.uint8),
                        ), "prefilling rank did not update its cache"
                        # Exclude only legal live-token writes; padding, other
                        # pages, and interleaved storage must remain unchanged.
                        before[page, ..., :live_tokens, :].copy_(cache[page, ..., :live_tokens, :])
                    assert torch.equal(cache.view(torch.uint8), before.view(torch.uint8)), (
                        f"rank {rank}: empty/padded cache storage was modified"
                    )
    finally:
        set_per_request_prefill_cuda_graph_flag(previous_prefill_flag)
        runner.clear_cuda_graphs()
    return rank


@pytest.mark.skip_less_device(2)
def test_minimax_m3_piecewise_empty_adp_rank_preserves_caches() -> None:
    """An empty ADP rank replays PCG without writing main K/V or index-K."""
    if torch.cuda.device_count() < 2:
        pytest.skip("requires two CUDA devices")
    with MPIPoolExecutor(max_workers=2) as executor:
        ranks = list(executor.map(_run_empty_adp_rank, [2, 2]))
    assert sorted(ranks) == [0, 1]
