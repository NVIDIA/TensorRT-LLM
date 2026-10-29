# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Split-KV reduction and per-bucket CUDA Graph workspace regressions."""

import copy
import importlib
from types import ModuleType, SimpleNamespace

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 10,
    reason="requires an SM100-or-newer CUDA device",
)


def _module(name: str) -> ModuleType:
    return importlib.import_module(f"tensorrt_llm._torch.attention_backend.fp4_mla.{name}")


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("store_stats", [False, True])
@pytest.mark.parametrize(
    "splits,queries,heads,columns,contiguous",
    [
        (1, 1, 3, 1, False),
        (3, 5, 3, 127, False),
        (16, 8, 128, 512, False),
        (32, 33, 3, 512, False),
        (33, 3, 5, 513, False),
        (65, 2, 3, 1025, False),
        (7, 5, 3, 257, True),
    ],
)
def test_splitkv_reduce_strides_empty_splits_and_graph_replay(
    dtype: torch.dtype,
    store_stats: bool,
    splits: int,
    queries: int,
    heads: int,
    columns: int,
    contiguous: bool,
) -> None:
    reduce = _module("fp4_mla_splitkv_reduce").fp4_mla_split_kv_reduce
    torch.manual_seed(17)
    partial = torch.randn((splits, queries, heads, columns), device="cuda").permute(0, 2, 3, 1)
    maxima = (torch.randn((splits, queries, heads), device="cuda") * 15).permute(0, 2, 1)
    sums = (torch.rand((splits, queries, heads), device="cuda") * 32).permute(0, 2, 1)
    sums[:, 0, 0] = 0
    if splits > 1:
        sums[0] = 0
    maxima.masked_fill_(sums == 0, -torch.inf)
    # Empty splits must not contaminate the output, even if their payload is stale.
    partial.masked_fill_((sums == 0).unsqueeze(2), torch.nan)
    out = torch.empty((queries, heads, columns), dtype=dtype, device="cuda").permute(1, 2, 0)
    if contiguous:
        partial, maxima, sums = partial.contiguous(), maxima.contiguous(), sums.contiguous()
        out = out.contiguous()
    final_max = torch.empty((heads, queries), device="cuda") if store_stats else None
    final_sum = torch.empty((heads, queries), device="cuda") if store_stats else None

    def launch() -> None:
        reduce(
            partial,
            maxima,
            sums,
            out,
            final_max,
            final_sum,
            softmax_scale_log2=0.73,
            output_scale=0.25,
        )

    def check() -> None:
        active = sums != 0
        expected_max = maxima.double().masked_fill(~active, -torch.inf).amax(0)
        weights = torch.where(active, torch.exp2((maxima.double() - expected_max) * 0.73), 0)
        expected_sum = (sums.double() * weights).sum(0)
        numerator = (
            torch.where(active.unsqueeze(2), partial.double(), 0) * weights.unsqueeze(2)
        ).sum(0)
        expected = torch.where(
            expected_sum.unsqueeze(1) != 0, numerator / expected_sum.unsqueeze(1) * 0.25, 0
        ).to(dtype)
        tolerance = torch.finfo(dtype).eps
        torch.testing.assert_close(out, expected, rtol=tolerance, atol=tolerance / 8)
        if store_stats:
            torch.testing.assert_close(final_max, expected_max.float(), rtol=0, atol=0)
            torch.testing.assert_close(final_sum, expected_sum.float(), rtol=2e-5, atol=2e-6)

    launch()
    check()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        launch()
    # Replays must consume new numerator and softmax-state contents at the same addresses.
    partial.mul_(0.5)
    sums.mul_(1.5)
    maxima.add_(3.0)
    out.fill_(torch.nan)
    graph.replay()
    check()


def test_splitkv_workspace_is_sized_per_graph_bucket() -> None:
    adapter = _module("fp4_mla_splitkv")
    heads, columns, splits, query_capacity = 128, 512, 4, 32

    def partial_launch(*args, split_kv_buffers, **kwargs) -> None:
        # Isolate workspace ownership from the main attention kernel. These CUDA
        # operations are captured alongside the real adapter and real reducer.
        partial, maxima, sums = split_kv_buffers
        repeated_q = args[0].view(-1, heads, 320)
        partial.copy_(repeated_q[:, :, :1].float().expand(-1, -1, columns))
        maxima.zero_()
        sums.fill_(1.0)

    main = SimpleNamespace(
        SMEM_P4_RUNTIME_MAX_KV=160 * 1024,
        FP4_MLA_P_GLOBAL_SCALE=1.0,
        run_trtllm_fp4_mla_decode_page_native_from_raw=partial_launch,
    )
    base_metadata = SimpleNamespace(is_cuda_graph=False)
    buckets = []
    storage_pointers = set()
    for queries in [1, 4, 8]:
        metadata = copy.copy(base_metadata)
        metadata.is_cuda_graph = True
        q = torch.full((queries, heads, 320), queries, dtype=torch.uint8, device="cuda")
        sf = torch.zeros((queries, heads, 40), dtype=torch.uint8, device="cuda")
        output = torch.empty((queries, heads, columns), dtype=torch.bfloat16, device="cuda")
        page_ids = torch.arange(queries * 2, dtype=torch.int32, device="cuda")
        indptr = torch.arange(queries + 1, dtype=torch.int32, device="cuda") * 2
        lengths = torch.full((queries,), 256, dtype=torch.int32, device="cuda")
        scale = torch.ones((), device="cuda")
        cache = torch.empty(1, device="cuda")

        def forward(
            metadata=metadata,
            q=q,
            sf=sf,
            output=output,
            page_ids=page_ids,
            indptr=indptr,
            lengths=lengths,
            scale=scale,
            cache=cache,
        ) -> None:
            adapter.run_fp4_mla_splitkv(
                main,
                metadata,
                q,
                sf,
                cache,
                cache,
                None,
                cache,
                scale,
                page_ids,
                indptr,
                lengths,
                output,
                splits=splits,
                max_kv_len=256,
                sm_scale=1.0,
                q_global_scale=scale,
                query_len_per_seq=1,
                v_page_offset=0,
                q_batch_capacity=query_capacity,
            )

        forward()
        workspace = {
            key: value
            for key, value in vars(metadata).items()
            if key.startswith("_fp4_mla_splitkv_")
        }
        tasks = splits * queries
        expected_sizes = {
            "q": tasks * heads * 320,
            "q_sf": tasks * heads * 40,
            "pages": tasks * 2,
            "indptr": tasks + 1,
            "lengths": tasks,
            "partial": tasks * heads * columns,
            "maxima": tasks * heads,
            "sums": tasks * heads,
        }
        for name, size in expected_sizes.items():
            tensor = workspace[f"_fp4_mla_splitkv_{name}"]
            assert tensor.numel() == size
            assert tensor.data_ptr() not in storage_pointers
            storage_pointers.add(tensor.data_ptr())
        pointers = {key: value.data_ptr() for key, value in workspace.items()}
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            forward()
        # Keep every captured input alive, including Q scales and page metadata.
        buckets.append((metadata, q, output, graph, pointers, forward))

    for round_idx in range(3):
        for metadata, q, output, graph, pointers, _forward in reversed(buckets):
            value = q.shape[0] + round_idx
            q.fill_(value)
            graph.replay()
            torch.testing.assert_close(output, torch.full_like(output, value), rtol=0, atol=0)
            for name, pointer in pointers.items():
                assert getattr(metadata, name).data_ptr() == pointer
    assert not any(name.startswith("_fp4_mla_splitkv_") for name in vars(base_metadata))


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    "splits,queries,heads,tile_columns",
    [(48, 6, 6, 256), (64, 6, 6, 256), (8, 24, 96, 512), (12, 24, 96, 512)],
)
def test_vector_reduce_logical_heads_and_live_scales(
    dtype: torch.dtype, splits: int, queries: int, heads: int, tile_columns: int
) -> None:
    reduce = _module("fp4_mla_splitkv_vector").reduce_splitkv_vector
    torch.manual_seed(23)
    partial = torch.randn((splits * queries, 128, 512), device="cuda")
    maxima = torch.randn((splits * queries, 128), device="cuda") * 12
    sums = torch.rand((splits * queries, 128), device="cuda") * 16
    sums.view(splits, queries, 128)[0] = 0
    sums.view(splits, queries, 128)[:, 0, 0] = 0
    maxima.masked_fill_(sums == 0, -torch.inf)
    partial.masked_fill_((sums == 0).unsqueeze(-1), torch.nan)
    q_scale = torch.tensor(1.25, device="cuda")
    kv_scale = torch.tensor(0.75, device="cuda")
    output = torch.empty((queries, heads, 1024), dtype=dtype, device="cuda")[:, :, ::2]

    def launch() -> None:
        reduce(
            partial,
            maxima,
            sums,
            output,
            q_scale,
            kv_scale,
            splits=splits,
            queries=queries,
            softmax_scale_log2=0.73,
            output_scale=0.25,
            tile_columns=tile_columns,
        )

    def check() -> None:
        row_sum = sums.view(splits, queries, 128)[:, :, :heads].double()
        active = row_sum != 0
        row_max = maxima.view(splits, queries, 128)[:, :, :heads].double()
        row_max = row_max * (0.73 / (q_scale.double() * kv_scale.double()))
        global_max = row_max.masked_fill(~active, -torch.inf).amax(0)
        weights = torch.where(active, torch.exp2(row_max - global_max), 0)
        values = partial.view(splits, queries, 128, 512)[:, :, :heads].double()
        numerator = (torch.where(active.unsqueeze(-1), values, 0) * weights.unsqueeze(-1)).sum(0)
        denominator = (row_sum * weights).sum(0).unsqueeze(-1)
        expected = torch.where(denominator != 0, numerator / denominator * 0.25, 0).to(dtype)
        tolerance = torch.finfo(dtype).eps
        torch.testing.assert_close(output, expected, rtol=tolerance, atol=tolerance / 8)

    launch()
    check()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        launch()
    q_scale.fill_(0.5)
    kv_scale.fill_(2.0)
    partial.mul_(0.75)
    sums.mul_(1.5)
    output.fill_(torch.nan)
    graph.replay()
    check()


@pytest.mark.parametrize(
    "length,splits", [(6, 4), (255, 4), (256, 4), (257, 4), (259, 4), (512, 8), (1025, 2)]
)
def test_folded_pages_preserve_each_query_causal_prefix(length: int, splits: int) -> None:
    partition = _module("fp4_mla_foldq")._fold_pages_kernel
    capacity, width = 1025, 6
    partition_pages = ((capacity + 255) // 256 + splits - 1) // splits * 2
    source_pages = torch.tensor([7, 3, 8, 1, 6, 2, 5, 0, 4], dtype=torch.int32, device="cuda")
    source_indptr = torch.tensor([0, source_pages.numel()], dtype=torch.int32, device="cuda")
    lengths = torch.tensor([length], dtype=torch.int32, device="cuda")
    pages = torch.empty(splits * partition_pages, dtype=torch.int32, device="cuda")
    indptr = torch.empty(splits + 1, dtype=torch.int32, device="cuda")
    valid = torch.empty(splits, dtype=torch.int32, device="cuda")
    bounds = torch.empty_like(valid)

    def launch() -> None:
        partition[(splits, 1)](
            source_pages,
            source_indptr,
            lengths,
            pages,
            indptr,
            valid,
            bounds,
            1,
            width,
            splits,
            partition_pages,
            128,
        )

    def check(live_length: int) -> None:
        inverse_pages = {page: index for index, page in enumerate(source_pages.tolist())}
        page_lists = pages.view(splits, partition_pages).tolist()
        valid_lengths, first_bounds = valid.tolist(), bounds.tolist()
        assert indptr.tolist() == [index * partition_pages for index in range(splits + 1)]
        for query in range(width):
            visited = []
            for split in range(splits):
                count = min(valid_lengths[split], max(0, first_bounds[split] + query))
                for token in range(count):
                    physical_page = page_lists[split][token // 128]
                    visited.append(inverse_pages[physical_page] * 128 + token % 128)
            assert visited == list(range(max(0, live_length - width + 1 + query)))

    launch()
    check(length)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        launch()
    lengths.fill_(6)
    source_pages.copy_(source_pages.flip(0))
    graph.replay()
    check(6)
