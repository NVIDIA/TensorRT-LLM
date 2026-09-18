# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Steady-state generation fast prepare under CUDA-graph batch padding.

A generation-only batch padded with CUDA-graph dummy requests must, from the
second full pass on, take ``_apply_steady_gen_fast_prepare`` and produce the
same model inputs and attention metadata as the full ``_prepare_tp_inputs``
walk, re-staging the KV block table only on the steps where a real request
starts a new block."""

import os
import sys

import pytest
import torch

import tensorrt_llm
from tensorrt_llm._torch.pyexecutor.model_engine import PyTorchModelEngine
from tensorrt_llm._torch.pyexecutor.resource_manager import (
    KVCacheManager, ResourceManager, ResourceManagerType)
from tensorrt_llm._torch.pyexecutor.sampler.sampler import SampleStateTensors
from tensorrt_llm._torch.pyexecutor.scheduler import ScheduledRequests
from tensorrt_llm.bindings.executor import KvCacheConfig
from tensorrt_llm.llmapi import CudaGraphConfig
from tensorrt_llm.llmapi.llm_args import TorchLlmArgs
from tensorrt_llm.mapping import Mapping

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_pytorch_model_engine import DummyModelEngine, _create_request  # noqa: E402

needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

TOKENS_PER_BLOCK = 4
MAX_BATCH = 13
PROMPT = 6
N_REAL = 5  # pads to the 8-request graph size -> 3 dummies


def _make_engine():
    llm_args = TorchLlmArgs(model="dummy",
                            max_batch_size=MAX_BATCH,
                            max_num_tokens=258,
                            cuda_graph_config=CudaGraphConfig(
                                enable_padding=True, batch_sizes=[1, 2, 4, 8, 16]))
    engine = DummyModelEngine(llm_args, torch.half)
    kv = KVCacheManager(
        KvCacheConfig(max_tokens=258),
        tensorrt_llm.bindings.internal.batch_manager.CacheType.SELF,
        num_layers=1,
        num_kv_heads=engine.model.config.num_key_value_heads,
        head_dim=engine.model.config.head_dim,
        tokens_per_block=TOKENS_PER_BLOCK,
        max_seq_len=258,
        max_batch_size=MAX_BATCH,
        mapping=Mapping(world_size=1, tp_size=1, rank=0),
        dtype=tensorrt_llm.bindings.DataType.HALF,
    )
    return engine, kv


def _snapshot(engine, n):
    md = engine.attn_metadata
    width = -(-int(md.kv_lens[:n].max()) // TOKENS_PER_BLOCK)
    return {
        "pos": engine.model.recorded_position_ids.clone().cpu(),
        "input_ids": engine.input_ids_cuda[:N_REAL].clone().cpu(),
        "kv_lens": md.kv_lens[:n].clone(),
        "kv_lens_cuda": md.kv_lens_cuda[:n].clone().cpu(),
        # the host tensor thop.attention reads for max_past_kv_length (mMaxSeqLenKv)
        "kv_lens_runtime": md.kv_lens_runtime[:n].clone(),
        "blocks": md.kv_cache_block_offsets[:, :n, :, :width].clone().cpu(),
        "prompt_lens": md.prompt_lens_cuda[:n].clone().cpu(),
        "request_types": md.host_request_types[:n].clone(),
        "total_kv_lens": md.host_total_kv_lens.clone(),
        "cached": list(md.kv_cache_params.num_cached_tokens_per_seq),
        "request_ids": list(md.request_ids),
        "num_seqs": md.num_seqs,
        "num_contexts": md.num_contexts,
        "seq_lens": md.seq_lens.clone(),
    }


def _run(fast: bool, mp, num_gen_steps=6, drop_after=None):
    with mp.context() as monkeypatch:
        return _run_patched(fast, monkeypatch, num_gen_steps, drop_after)


def _run_patched(fast: bool, monkeypatch, num_gen_steps, drop_after):
    engine, kv = _make_engine()
    rm = ResourceManager({ResourceManagerType.KV_CACHE_MANAGER: kv})
    counts = {"fast": 0, "block_copies": 0}
    if not fast:
        monkeypatch.setattr(PyTorchModelEngine, "_can_use_steady_gen_fast_prepare",
                            lambda self, *a, **k: False)
    orig_apply = engine._apply_steady_gen_fast_prepare

    def spy_apply(*a, **k):
        counts["fast"] += 1
        return orig_apply(*a, **k)

    monkeypatch.setattr(engine, "_apply_steady_gen_fast_prepare", spy_apply)
    orig_copy = kv.copy_batch_block_offsets

    def spy_copy(*a, **k):
        counts["block_copies"] += 1
        return orig_copy(*a, **k)

    monkeypatch.setattr(kv, "copy_batch_block_offsets", spy_copy)

    reqs = [_create_request(PROMPT, i) for i in range(N_REAL)]
    for i, r in enumerate(reqs):
        r.py_seq_slot = i  # the executor's slot manager does this; needed for the previous-tensor branch
    batch = ScheduledRequests()
    batch.context_requests_last_chunk = list(reqs)
    kv.prepare_resources(batch)
    engine.forward(batch, rm)

    snaps = []
    active = list(reqs)
    for step in range(1, num_gen_steps + 1):
        if drop_after is not None and step == drop_after + 1:
            active = active[1:]  # request 0 finished
        batch = ScheduledRequests()
        batch.generation_requests = list(active)
        kv.prepare_resources(batch)
        # Overlap scheduler order: the forward consumes the previous step's token from the
        # device sample buffer; update_requests appends it to the request afterwards.
        new_tokens = torch.full((1, MAX_BATCH, 1), 100 + step, dtype=torch.int32, device="cuda")
        engine.forward(batch, rm, new_tensors_device=SampleStateTensors(new_tokens=new_tokens))
        torch.cuda.synchronize()
        snaps.append(_snapshot(engine, engine.attn_metadata.num_seqs))
        for r in active:
            r.add_new_token(100 + step, 0)
    kv.shutdown()
    return snaps, counts, engine


@needs_cuda
def test_fast_prepare_fires_with_padding_and_matches_full_pass(monkeypatch):
    ref, ref_counts, _ = _run(False, monkeypatch)
    out, counts, engine = _run(True, monkeypatch)
    assert ref_counts["fast"] == 0
    assert engine._steady_gen_cache is not None, "recording condition never held"
    # the context step already assigns py_batch_idx, so gen step 1 is a full pass that
    # records the layout and steps 2..6 take the fast path
    assert counts["fast"] == 5
    for step, (a, b) in enumerate(zip(out, ref), start=1):
        assert a["num_seqs"] == 8 and a["request_ids"][N_REAL:] == b["request_ids"][N_REAL:]
        for key in a:
            if isinstance(a[key], torch.Tensor):
                torch.testing.assert_close(a[key], b[key], atol=0, rtol=0, msg=f"step {step} {key}")
            else:
                assert a[key] == b[key], (step, key)
    cache = engine._steady_gen_cache
    assert cache is not None and cache["num_requests"] == 8 and cache["num_real"] == N_REAL
    assert cache["attn_metadata"] is engine.attn_metadata
    assert cache["tokens_per_block"] == TOKENS_PER_BLOCK
    # KV block table staging: the full pass restages on every prepare (context + 6 gen);
    # the fast path only when a real request starts a new block. Positions per gen step are
    # PROMPT + step - 1 = 6,7,8,9,10,11 -> only step 3 (position 8) opens a block.
    assert ref_counts["block_copies"] == 7
    assert counts["block_copies"] == 1 + 1 + 1
    # the dummies' rows never move
    dummy_positions = [s["cached"][N_REAL:] for s in out]
    assert all(p == dummy_positions[0] for p in dummy_positions)
    assert [s["cached"][0] for s in out] == [PROMPT + k for k in range(6)]


@needs_cuda
def test_fast_prepare_advances_the_host_runtime_kv_lens(monkeypatch):
    """prepare() binds ``kv_lens_runtime`` (what thop.attention reads as
    host_past_key_value_lengths) to a tensor separate from ``kv_lens``, so the fast
    step has to advance it as well: after the fast steps it must agree with the
    device kv lengths and with ``kv_lens`` minus the extra tokens."""
    out, counts, engine = _run(True, monkeypatch)
    assert counts["fast"] == 5
    md = engine.attn_metadata
    n = md.num_seqs
    torch.testing.assert_close(md.kv_lens_runtime[:n], md.kv_lens_cuda[:n].cpu(), atol=0, rtol=0)
    torch.testing.assert_close(md.kv_lens_runtime[:n],
                               md.kv_lens[:n] - md.kv_cache_params.num_extra_kv_tokens, atol=0, rtol=0)
    # every real request committed one token per gen step
    assert md.kv_lens_runtime[:N_REAL].tolist() == [PROMPT + 6] * N_REAL
    assert [s["kv_lens_runtime"][0].item() for s in out] == [PROMPT + k for k in range(1, 7)]


@needs_cuda
def test_batch_change_records_new_layout(monkeypatch):
    ref, _, _ = _run(False, monkeypatch, num_gen_steps=7, drop_after=4)
    out, counts, engine = _run(True, monkeypatch, num_gen_steps=7, drop_after=4)
    # steps 2,3,4 fast (5 real + 3 dummies); step 5 full pass (4 real, no padding needed),
    # re-records; steps 6,7 fast again
    assert counts["fast"] == 5
    assert engine._steady_gen_cache["num_requests"] == 4
    assert engine._steady_gen_cache["num_real"] == 4
    for step, (a, b) in enumerate(zip(out, ref), start=1):
        for key in a:
            if isinstance(a[key], torch.Tensor):
                torch.testing.assert_close(a[key], b[key], atol=0, rtol=0, msg=f"step {step} {key}")
            else:
                assert a[key] == b[key], (step, key)
