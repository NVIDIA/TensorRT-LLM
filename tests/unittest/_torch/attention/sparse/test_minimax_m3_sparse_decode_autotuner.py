# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Focused tests for MiniMax-M3 adaptive sparse-decode tuning."""

import contextlib
from types import SimpleNamespace

import pytest
import torch


@pytest.mark.parametrize(
    ("is_cuda_graph_metadata", "expected_tuning_calls"),
    [(False, 0), (True, 1)],
)
def test_only_graph_warmup_can_seed_adaptive_tuning(
    monkeypatch, is_cuda_graph_metadata, expected_tuning_calls
):
    from tensorrt_llm._torch.attention_backend.sparse.minimax_m3 import sparse_decode_autotuner

    class FakeTensor:
        def __init__(self, shape, dtype, *, is_cuda=True):
            self.shape = torch.Size(shape)
            self.dtype = dtype
            self.is_cuda = is_cuda
            self._stride = torch.empty(shape, device="meta").stride()

        def size(self):
            return self.shape

        def stride(self):
            return self._stride

    tuner = SimpleNamespace(
        profiling_cache=SimpleNamespace(
            search_cache=lambda *args, **kwargs: (False, None, None, None),
            get_cache_key=lambda *args, **kwargs: ("sparse-decode",),
        ),
        mapping=SimpleNamespace(has_pp=lambda: False),
        is_tuning_mode=False,
    )
    fallback_tactics = []
    tuning_calls = []
    choose_calls = []

    def record_autotune():
        tuning_calls.append(True)
        return contextlib.nullcontext()

    def choose_one(*args, **kwargs):
        choose_calls.append(True)
        return args[1][0], -1

    monkeypatch.setattr(sparse_decode_autotuner.AutoTuner, "get", lambda: tuner)
    tuner.choose_one = choose_one
    monkeypatch.setattr(
        sparse_decode_autotuner.torch.cuda,
        "is_current_stream_capturing",
        lambda: False,
    )
    monkeypatch.setattr(sparse_decode_autotuner, "autotune", record_autotune)
    monkeypatch.setattr(sparse_decode_autotuner, "_attempted_tuning_keys", set())
    monkeypatch.setattr(
        sparse_decode_autotuner.MiniMaxM3SparseDecodeRunner,
        "__call__",
        lambda self, inputs, *, tactic, plan: fallback_tactics.append(tactic),
    )

    q = FakeTensor((2, 4, 128), torch.bfloat16)
    k_paged = FakeTensor((8, 2, 128, 128), torch.float8_e4m3fn)
    v_paged = FakeTensor((8, 2, 128, 128), torch.float8_e4m3fn)
    block_indexes = FakeTensor((2, 2, 16), torch.int32)
    block_table = FakeTensor((2, 4), torch.int32)
    seq_lens = FakeTensor((2,), torch.int32)
    output = FakeTensor((2, 4, 128), torch.bfloat16)

    sparse_decode_autotuner.run_adaptive_sparse_decode(
        q,
        k_paged,
        v_paged,
        block_indexes,
        block_table,
        seq_lens,
        output,
        sm_scale=128**-0.5,
        decode_query_len=1,
        plan=object(),
        is_cuda_graph_metadata=is_cuda_graph_metadata,
    )

    assert fallback_tactics == [-1]
    assert len(tuning_calls) == expected_tuning_calls
    assert len(choose_calls) == expected_tuning_calls
    assert len(sparse_decode_autotuner._attempted_tuning_keys) == expected_tuning_calls


@pytest.mark.parametrize("is_cuda_graph", [False, True])
def test_adaptive_dispatch_passes_graph_metadata_state(monkeypatch, is_cuda_graph):
    from tensorrt_llm._torch.attention_backend.fmha import msa_sparse_gqa
    from tensorrt_llm._torch.attention_backend.sparse.minimax_m3 import (
        msa_utils,
        sparse_decode_autotuner,
    )
    from tensorrt_llm._torch.attention_backend.sparse.minimax_m3.msa_backend import _MsaDecodeSpan
    from tensorrt_llm.llmapi.llm_args import MiniMaxM3SparseAttentionConfig

    page_size = head_dim = 128
    block_table = torch.zeros(1, 1, dtype=torch.int32)
    k_paged = torch.zeros(1, 1, page_size, head_dim)
    v_paged = torch.zeros_like(k_paged)
    graph_flags = []
    monkeypatch.setattr(
        msa_utils,
        "msa_paged_kv",
        lambda manager, layer_idx: (k_paged, v_paged),
    )
    monkeypatch.setattr(
        sparse_decode_autotuner,
        "run_adaptive_sparse_decode",
        lambda *args, **kwargs: graph_flags.append(kwargs["is_cuda_graph_metadata"]),
    )

    attention = SimpleNamespace(
        layer_idx=0,
        head_dim=head_dim,
        num_heads=1,
        q_scaling=1.0,
        sparse_params=MiniMaxM3SparseAttentionConfig(
            implementation="msa", decode_backend="adaptive"
        ).to_sparse_params(),
    )
    metadata = SimpleNamespace(
        is_cuda_graph=is_cuda_graph,
        kv_cache_manager=object(),
        _msa_prewritten_layer=None,
        msa_decode_query_len=1,
        msa_decode_span=_MsaDecodeSpan(0, 1, 0, 1, 1),
        msa_block_table=block_table,
        msa_seq_lens_cuda=torch.ones(1, dtype=torch.int32),
        msa_kv_indices=block_table.flatten(),
        msa_qo_lens_cpu=torch.ones(1, dtype=torch.int32),
        msa_kv_lens_cpu=torch.ones(1, dtype=torch.int32),
        msa_qo_offset_cpu=torch.zeros(1, dtype=torch.int32),
    )

    msa_sparse_gqa.run_msa_paged_gqa(
        attention,
        torch.zeros(1, head_dim),
        None,
        None,
        metadata,
        torch.empty(1, head_dim),
        kv_block_indexes=torch.zeros(1, 1, 16, dtype=torch.int32),
        plan=object(),
    )

    assert graph_flags == [is_cuda_graph]
