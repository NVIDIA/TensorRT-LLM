# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Focused tests for MiniMax-M3 adaptive sparse-decode tuning."""

import contextlib
from types import SimpleNamespace

import torch


def test_only_graph_warmup_can_seed_adaptive_tuning(monkeypatch):
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

    cached_tactic = None

    def search_cache(*args, **kwargs):
        del args, kwargs
        if cached_tactic is None:
            return False, 0, -1, float("inf")
        return True, 0, cached_tactic, 1.0

    tuner = SimpleNamespace(
        profiling_cache=SimpleNamespace(
            search_cache=search_cache,
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
        nonlocal cached_tactic
        choose_calls.append(True)
        if cached_tactic is None:
            cached_tactic = "triton"
        return args[1][0], cached_tactic

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

    def run(is_cuda_graph_metadata, plan=object()):
        return sparse_decode_autotuner.run_adaptive_sparse_decode(
            q,
            k_paged,
            v_paged,
            block_indexes,
            block_table,
            seq_lens,
            output,
            sm_scale=128**-0.5,
            decode_query_len=1,
            plan=plan,
            is_cuda_graph_metadata=is_cuda_graph_metadata,
        )

    assert run(False) is None
    assert fallback_tactics == [-1]
    assert not tuning_calls
    assert not choose_calls
    assert not sparse_decode_autotuner._attempted_tuning_keys

    assert run(True) == "triton"
    # Once Triton is cached, dispatch no longer requires an MSA plan.
    assert run(True, plan=None) == "triton"
    assert fallback_tactics == [-1, "triton", "triton"]
    assert len(tuning_calls) == 1
    assert len(choose_calls) == 2
    assert len(sparse_decode_autotuner._attempted_tuning_keys) == 1
