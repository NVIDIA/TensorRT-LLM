# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Packed FP4 MLA decode: row layout policy and the CuTe DSL split reducer."""

import importlib
import math

import pytest
import torch

import tensorrt_llm._torch.attention.backends.fp4_mla.config as fp4_mla_config
import tensorrt_llm._torch.attention.backends.fp4_mla.decode as fp4_mla_decode

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 10,
    reason="requires an SM100-or-newer CUDA device",
)


@pytest.mark.parametrize(
    "num_heads,query_len,expected",
    [
        (96, 8, 768),
        (12, 8, 128),
        (6, 6, 128),
        (128, 4, 512),
        (128, 8, 1024),
        (128, 9, 0),
        (0, 1, 0),
    ],
)
def test_packed_rows_per_seq_pads_each_request_to_128_rows(
    monkeypatch, num_heads: int, query_len: int, expected: int
) -> None:
    monkeypatch.setenv(fp4_mla_config.FP4_MLA_PACKED_Q_ENV, "1")
    monkeypatch.setenv(fp4_mla_config.FP4_MLA_CUTEDSL_FUSED_V_TRANSPOSE_ENV, "0")
    monkeypatch.setattr(fp4_mla_config, "_fp4_mla_attention_backend", lambda: "cutedsl")
    assert fp4_mla_config._fp4_mla_packed_rows_per_seq(num_heads, query_len) == expected
    monkeypatch.setenv(fp4_mla_config.FP4_MLA_PACKED_Q_ENV, "0")
    assert fp4_mla_config._fp4_mla_packed_rows_per_seq(num_heads, query_len) == 0


def test_packed_split_count_fills_one_cluster_wave(monkeypatch) -> None:
    monkeypatch.delenv("TRTLLM_FP4_MLA_SPLIT_KV", raising=False)
    monkeypatch.setattr(fp4_mla_decode, "_get_sm_count", lambda device: 212)
    device = torch.device("cuda")
    select = fp4_mla_decode._select_packed_split_count
    # helix16 B4 MTP7 (6 tiles per request): 106 clusters / 24 -> 4 splits.
    assert select(4, 6, device, max_kv_len=999424, is_cuda_graph=True) == 4
    # A single short request is capped by the KV length when it is known.
    assert select(1, 6, device, max_kv_len=512, is_cuda_graph=False) == 1
    assert select(1, 6, device, max_kv_len=999424, is_cuda_graph=True) == 16
    # Large batches keep one split.
    assert select(32, 6, device, max_kv_len=999424, is_cuda_graph=True) == 1
    monkeypatch.setenv("TRTLLM_FP4_MLA_SPLIT_KV", "3")
    assert select(4, 6, device, max_kv_len=999424, is_cuda_graph=True) == 3


@pytest.mark.skipif(
    torch.cuda.get_device_capability() != (10, 7) if torch.cuda.is_available() else True,
    reason="the CuTe DSL reducer is compiled for Rubin SM107",
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("with_stats", [False, True])
@pytest.mark.parametrize(
    "requests,splits,num_heads,query_len",
    [(4, 4, 96, 8), (1, 16, 12, 8), (3, 3, 6, 6), (2, 2, 128, 4), (5, 1, 96, 1)],
)
def test_packed_split_reduce_matches_reference_and_replays(
    dtype: torch.dtype, with_stats: bool, requests: int, splits: int, num_heads: int, query_len: int
) -> None:
    reduce_mod = importlib.import_module(
        "tensorrt_llm._torch.attention.backends.fp4_mla.fp4_mla_packed_reduce"
    )
    torch.manual_seed(7)
    valid_m = num_heads * query_len
    physical_m = (valid_m + 127) // 128 * 128
    tasks = splits * requests
    partial = torch.randn((tasks, physical_m, 512), device="cuda")
    maxima = torch.randn((tasks, physical_m), device="cuda") * 12
    sums = torch.rand((tasks, physical_m), device="cuda") * 16 + 0.5
    # Split 0 is empty for every row; row 0 of request 0 is empty everywhere.
    sums.view(splits, requests, physical_m)[0] = 0
    sums.view(splits, requests, physical_m)[:, 0, 0] = 0
    maxima.masked_fill_(sums == 0, -torch.inf)
    partial.masked_fill_((sums == 0).unsqueeze(-1), torch.nan)
    q_scale = torch.tensor(1.25, device="cuda")
    kv_scale = torch.tensor(0.75, device="cuda")
    sm_scale, output_scale = 0.1, 0.25
    output = torch.empty((requests * valid_m, 512), dtype=dtype, device="cuda")
    stats = torch.empty((requests * valid_m, 2), device="cuda") if with_stats else None

    def launch() -> None:
        reduce_mod.reduce_packed_splits(
            partial,
            maxima,
            sums,
            output,
            q_scale,
            kv_scale,
            num_requests=requests,
            num_splits=splits,
            valid_m=valid_m,
            physical_m=physical_m,
            sm_scale=sm_scale,
            output_scale=output_scale,
            stats=stats,
            sm_count=212,
        )

    def check() -> None:
        scale = sm_scale * math.log2(math.e) / (q_scale.double() * kv_scale.double())
        row_sum = sums.view(splits, requests, physical_m)[:, :, :valid_m].double()
        active = row_sum != 0
        row_max = maxima.view(splits, requests, physical_m)[:, :, :valid_m].double() * scale
        global_max = row_max.masked_fill(~active, -torch.inf).amax(0)
        weights = torch.where(active, torch.exp2(row_max - global_max), 0.0)
        values = partial.view(splits, requests, physical_m, 512)[:, :, :valid_m].double()
        numerator = (torch.where(active.unsqueeze(-1), values, 0.0) * weights.unsqueeze(-1)).sum(0)
        denominator = (row_sum * weights).sum(0)
        expected = torch.where(
            denominator.unsqueeze(-1) != 0,
            numerator / denominator.unsqueeze(-1) * output_scale,
            0.0,
        )
        expected = expected.reshape(requests * valid_m, 512).to(dtype)
        tolerance = torch.finfo(dtype).eps
        torch.testing.assert_close(output, expected, rtol=4 * tolerance, atol=tolerance)
        if stats is not None:
            expected_max = torch.where(denominator != 0, global_max / math.log2(math.e), -torch.inf)
            torch.testing.assert_close(
                stats[:, 0], expected_max.reshape(-1).float(), rtol=1e-5, atol=1e-5
            )
            torch.testing.assert_close(
                stats[:, 1], denominator.reshape(-1).float(), rtol=2e-5, atol=2e-6
            )

    launch()
    torch.cuda.synchronize()
    check()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        launch()
    # Replays follow the live scales and payloads at the same addresses.
    q_scale.fill_(0.5)
    kv_scale.fill_(2.0)
    partial.mul_(0.75)
    sums.mul_(1.5)
    output.fill_(torch.nan)
    graph.replay()
    torch.cuda.synchronize()
    check()
