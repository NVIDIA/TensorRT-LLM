# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``features.glue_h1a``: the fused select-accept's save of the device sequence lengths.

``DFlashWorker._prepare_attn_metadata_off_path`` runs the engine's save with the switch off and,
with it on, saves the same fields with the device lengths' copy made on a side stream. The save's
copy, the saved originals and the restore are the same either way, eager and under CUDA graph
replay, when a reader writes the copy after the main stream waits on the returned event (as the
drafter's prepare does after ``select_accept``).
"""

import dataclasses
import importlib
import types
from contextlib import contextmanager

import pytest
import torch

from tensorrt_llm._torch.attention.backends.interface import AttentionMetadata

_FAMILY = "tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31"
_PKG = f"{_FAMILY}.kimi_k31_pro_nvfp4__sm_107__helix16_moetp16ep1"
_FEATURES = f"{_FAMILY}.ops.common.rubin_ull_features"


@contextmanager
def _switch(on: bool):
    features = importlib.import_module(_FEATURES)
    saved = features._ACTIVE
    features._ACTIVE = dataclasses.replace(saved, glue_h1a=on)
    try:
        yield
    finally:
        features._ACTIVE = saved


def test_the_switch_is_a_model_config_field() -> None:
    features = importlib.import_module(_FEATURES)
    assert features.MODEL_CONFIG_FIELDS["glue_h1a"] is bool
    assert isinstance(features.K31Features().glue_h1a, bool)


class _Metadata:
    """The fields and the save / restore of ``AttentionMetadata`` the worker's save touches."""

    prepare_for_spec_dec = AttentionMetadata.prepare_for_spec_dec
    restore_from_spec_dec = AttentionMetadata.restore_from_spec_dec

    def __init__(self, lengths: list):
        self._seq_lens = torch.tensor(lengths, dtype=torch.int32)
        self._seq_lens_cuda = self._seq_lens.cuda()
        self.kv_lens_cuda = torch.arange(100, 100 + len(lengths), dtype=torch.int32, device="cuda")
        self._saved_tensors = {}


def _worker():
    modeling = importlib.import_module(f"{_PKG}.modeling")
    worker = object.__new__(modeling.DFlashWorker)
    worker.mapping = types.SimpleNamespace(rank=0)
    return worker


def _step(worker, md, spec, gens: int, width: int, x: torch.Tensor, w: torch.Tensor):
    """Target head stand-in -> save -> argmax stand-in -> wait -> the prepare's write of the
    generation rows' lengths. Returns (the save's copy, the argmax)."""
    logits = x @ w.t()
    done = worker._prepare_attn_metadata_off_path(md, spec)
    tokens = logits.float().argmax(dim=-1)
    if done is not None:
        torch.cuda.current_stream().wait_event(done)
    md._seq_lens_cuda[:gens].fill_(width)
    return md._seq_lens_cuda, tokens


@pytest.mark.parametrize("on", (False, True))
def test_eager_save_and_restore(on: bool) -> None:
    worker = _worker()
    lengths = [1000, 1001, 1002, 1003]
    x = torch.randn(32, 512, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(4096, 512, device="cuda", dtype=torch.bfloat16)
    for warmup in (False, True):
        md = _Metadata(lengths)
        original, kv = md._seq_lens_cuda, md.kv_lens_cuda
        spec = types.SimpleNamespace(is_cuda_graph=warmup)
        with _switch(on):
            copy, _ = _step(worker, md, spec, 2, 8, x, w)
        torch.cuda.synchronize()
        assert copy is not original
        assert copy.tolist() == [8, 8, 1002, 1003]
        assert original.tolist() == lengths
        assert set(md._saved_tensors) == {"_seq_lens", "_seq_lens_cuda"} | (
            {"kv_lens_cuda"} if warmup else set()
        )
        md.restore_from_spec_dec()
        assert md._seq_lens_cuda is original and md.kv_lens_cuda is kv
        assert not md._saved_tensors


def test_graph_replay_equals_eager_both_ways() -> None:
    worker = _worker()
    x = torch.randn(32, 512, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(4096, 512, device="cuda", dtype=torch.bfloat16)
    spec = types.SimpleNamespace(is_cuda_graph=True)
    results = {}
    for on in (False, True):
        md = _Metadata([5, 6, 7, 8])
        original = md._seq_lens_cuda
        with _switch(on):
            _step(worker, md, types.SimpleNamespace(is_cuda_graph=False), 2, 8, x, w)  # warm
            md.restore_from_spec_dec()
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                copy, tokens = _step(worker, md, spec, 2, 8, x, w)
            md.restore_from_spec_dec()
        out = []
        for base in (2000, 3000, 4000):
            original.copy_(torch.arange(base, base + 4, dtype=torch.int32))
            x.normal_()
            graph.replay()
            torch.cuda.synchronize()
            assert original.tolist() == list(range(base, base + 4))
            out.append((copy.tolist(), tokens.clone()))
            assert copy.tolist() == [8, 8, base + 2, base + 3]
            eager = _Metadata(list(range(base, base + 4)))
            eager_copy, eager_tokens = _step(worker, eager, spec, 2, 8, x, w)
            torch.cuda.synchronize()
            assert eager_copy.tolist() == copy.tolist()
            assert torch.equal(eager_tokens, tokens)
        results[on] = [c for c, _ in out]
        del graph
    assert results[False] == results[True]
