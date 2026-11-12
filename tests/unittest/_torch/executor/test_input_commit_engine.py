# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""``PyTorchModelEngine._prepare_tp_inputs`` with and without the batched input commit.

The commit contract is that no device operation between ``stage`` and ``commit`` reads a staged
destination. This is the engine-level check of it for the staging sites of the engine itself, on a
one-model speculative engine with the overlap scheduler: two identical engines prepare the same
batches, one with the host->device copies and the overlap gathers issued one by one, one under
``InputCommit`` (begin / scope / commit, as ``forward`` does). Every buffer the step writes is
pre-filled with a sentinel, so a staged value that a device operation had consumed before the commit
wrote it would show as a difference, and the two engines must end with the same inputs, gather ids,
draft tokens, accepted counts and previous-batch offset buffers, step after step (arena
alternation, new requests joining, requests leaving, a different order).

The engine here uses the base attention metadata, a mock speculative metadata, the V1 KV cache
manager and no mamba state, so the staging sites of the TRT-LLM attention metadata, the mamba and
state-index metadata, the speculative metadata (seeds, offsets, slot ids, sampling parameters) and
the DFlash worker are not exercised by this file. The verify mode of the commit
(``TRTLLM_GEN_HOST_BUNDLE_VERIFY``) compares every staged destination of a real engine.
"""

import random
from dataclasses import dataclass
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

import tensorrt_llm
from tensorrt_llm._torch.attention_backend.interface import AttentionMetadata
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.pyexecutor import input_commit as ic
from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequest
from tensorrt_llm._torch.pyexecutor.model_engine import PyTorchModelEngine
from tensorrt_llm._torch.pyexecutor.resource_manager import (
    KVCacheManager,
    ResourceManager,
    ResourceManagerType,
)
from tensorrt_llm._torch.pyexecutor.scheduler import ScheduledRequests
from tensorrt_llm._torch.speculative.spec_sampler_base import SampleStateTensorsSpec
from tensorrt_llm.bindings.executor import KvCacheConfig
from tensorrt_llm.llmapi import CudaGraphConfig, SADecodingConfig, SamplingParams
from tensorrt_llm.llmapi.llm_args import TorchLlmArgs
from tensorrt_llm.mapping import Mapping

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")

DRAFT = 3  # draft tokens per step
TOKENS = DRAFT + 1
NUM_SLOTS = 12
SENTINEL = -12345


@dataclass
class _Config:
    torch_dtype: torch.dtype
    num_key_value_heads: int = 16
    num_attention_heads: int = 16
    hidden_size: int = 256
    architectures: list = None

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads


class _DummyModel(torch.nn.Module):
    def __init__(self, dtype: torch.dtype, mapping: Mapping):
        super().__init__()
        self.model_config = ModelConfig(
            pretrained_config=_Config(torch_dtype=dtype), mapping=mapping
        )

    def infer_max_seq_len(self):
        return 2048

    @property
    def config(self):
        return self.model_config.pretrained_config

    def forward(self, *args, **kwargs):
        return {"logits": torch.zeros((kwargs["input_ids"].size(0), 10), device="cuda")}


class _Engine(PyTorchModelEngine):
    def __init__(self, llm_args, spec_config, attention_dp: bool = False):
        mapping = Mapping(
            world_size=tensorrt_llm.mpi_world_size(),
            tp_size=tensorrt_llm.mpi_world_size(),
            rank=tensorrt_llm.mpi_rank(),
            enable_attention_dp=attention_dp,
        )
        super().__init__(
            model_path="dummy",
            mapping=mapping,
            model=_DummyModel(torch.half, mapping),
            llm_args=llm_args,
            spec_config=spec_config,
        )


def _llm_args() -> TorchLlmArgs:
    return TorchLlmArgs(
        model="dummy",
        max_batch_size=13,
        max_num_tokens=258,
        cuda_graph_config=CudaGraphConfig(enable_padding=True, batch_sizes=[1, 2, 4, 8, 16]),
    )


def _engine_and_cache():
    max_tokens, batch_size = 258, 13
    engine = _Engine(
        _llm_args(), SADecodingConfig(max_draft_len=DRAFT, draft_len_schedule={1: DRAFT})
    )
    engine.runtime_draft_len = DRAFT
    cache = KVCacheManager(
        KvCacheConfig(max_tokens=max_tokens),
        tensorrt_llm.bindings.internal.batch_manager.CacheType.SELF,
        num_layers=1,
        num_kv_heads=engine.model.config.num_key_value_heads,
        head_dim=engine.model.config.head_dim,
        tokens_per_block=1,
        max_seq_len=max_tokens,
        max_batch_size=batch_size,
        mapping=Mapping(world_size=1, tp_size=1, rank=0),
        dtype=tensorrt_llm.bindings.DataType.HALF,
    )
    return engine, cache


def _poison(engine: PyTorchModelEngine) -> None:
    """Fill every buffer the step writes with a sentinel (also the ones the commit zeroes)."""
    for name in (
        "input_ids_cuda",
        "position_ids_cuda",
        "gather_ids_cuda",
        "draft_tokens_cuda",
        "num_accepted_draft_tokens_cuda",
        "previous_batch_indices_cuda",
        "previous_pos_indices_cuda",
        "previous_pos_id_offsets_cuda",
        "previous_kv_lens_offsets_cuda",
    ):
        getattr(engine, name).fill_(SENTINEL)


def _request(req_id: int, slot: int, batch_idx, num_tokens: int, rng: random.Random):
    tokens = [rng.randint(1, 900) for _ in range(num_tokens)]
    request = LlmRequest(
        request_id=req_id,
        max_new_tokens=64,
        input_tokens=tokens,
        sampling_config=tensorrt_llm.bindings.SamplingConfig(
            SamplingParams()._get_sampling_config()
        ),
        is_streaming=False,
    )
    request.paged_kv_block_ids = []
    request.py_seq_slot = slot
    request.py_batch_idx = batch_idx  # None: the first step of a request on a GEN worker
    request.py_draft_tokens = (
        [rng.randint(1, 900) for _ in range(DRAFT)] if batch_idx is None else []
    )
    return request


def _overlap_state(rng: random.Random) -> SampleStateTensorsSpec:
    """The previous step's device outputs: [T, slots, 1] tokens, [slots] lengths, [slots, D] drafts."""

    def tensor(shape, low, high):
        values = [rng.randint(low, high) for _ in range(int(torch.tensor(shape).prod()))]
        return torch.tensor(values, dtype=torch.int32, device="cuda").reshape(shape)

    return SampleStateTensorsSpec(
        new_tokens=tensor((TOKENS, NUM_SLOTS, 1), 1, 900),
        new_tokens_lens=tensor((NUM_SLOTS,), 1, TOKENS),
        next_draft_tokens=tensor((NUM_SLOTS, DRAFT), 1, 900),
    )


# (request id, slot, has a previous step in the overlap tensors, prompt tokens)
STEPS = [
    # all requests are new: no previous tensors, draft tokens come from the request
    [(1, 5, False, 40), (2, 2, False, 55)],
    # steady state plus a request that joins (first step on this worker)
    [(1, 5, True, 41), (2, 2, True, 56), (3, 9, False, 33)],
    # one leaves, the joiner now has a previous step, another joins, order differs
    [(3, 9, True, 34), (1, 5, True, 42), (4, 0, False, 77), (5, 11, False, 12)],
    # many in flight (previous-batch gathers across slots, a wider fixup)
    [(i, (7 * i) % NUM_SLOTS, True, 30 + i) for i in range(1, 11)],
    # back to a single request
    [(3, 9, True, 35)],
]


def _run_step(engine, cache, step, state, commit, commit_it: bool = True):
    rng = random.Random(1000 + len(step))
    requests = [_request(i, slot, slot if prev else None, n, rng) for i, slot, prev, n in step]
    scheduled = ScheduledRequests()
    scheduled.generation_requests = requests
    attn_metadata = AttentionMetadata(
        max_num_requests=16, max_num_tokens=64, kv_cache_manager=cache
    )
    attn_metadata.is_cuda_graph = False
    spec_metadata = Mock(_force_non_greedy_for_capture=False)
    resource_manager = ResourceManager({ResourceManagerType.KV_CACHE_MANAGER: cache})
    kwargs = dict(
        scheduled_requests=scheduled,
        kv_cache_manager=cache,
        attn_metadata=attn_metadata,
        spec_metadata=spec_metadata,
        new_tensors_device=state,
        resource_manager=resource_manager,
    )
    if commit is None:
        inputs, gather_ids = engine._prepare_tp_inputs(**kwargs)
    else:
        commit.begin()
        with ic.input_commit_scope(commit):
            inputs, gather_ids = engine._prepare_tp_inputs(**kwargs)
        if commit_it:
            commit.commit()
        else:
            commit.abort()
    return inputs, gather_ids, requests


def _snapshot(engine, inputs, gather_ids, num_requests: int) -> dict:
    num_tokens = num_requests * TOKENS
    return {
        "input_ids": inputs["input_ids"][:num_tokens].cpu(),
        "position_ids": inputs["position_ids"].reshape(-1)[:num_tokens].cpu(),
        "gather_ids": gather_ids.cpu() if gather_ids is not None else None,
        "engine_gather_ids": engine.gather_ids_cuda[:num_tokens].cpu(),
        "draft_tokens": engine.draft_tokens_cuda[: num_requests * DRAFT].cpu(),
        "num_accepted": engine.num_accepted_draft_tokens_cuda[:num_requests].cpu(),
        "previous_pos_id_offsets": engine.previous_pos_id_offsets_cuda.cpu(),
        "previous_kv_lens_offsets": engine.previous_kv_lens_offsets_cuda.cpu(),
    }


def _differences(baseline, ref, staged, got, num_requests: int) -> list:
    """The names of the buffers in which the two engines' prepared inputs differ."""
    want = _snapshot(baseline, *ref[:2], num_requests)
    have = _snapshot(staged, *got[:2], num_requests)
    differing = []
    for key in want:
        if want[key] is None or have[key] is None:
            if (want[key] is None) != (have[key] is None):
                differing.append(key)
        elif not torch.equal(want[key], have[key]):
            differing.append(key)
    return differing


def test_prepare_tp_inputs_staged_matches_unstaged(monkeypatch: pytest.MonkeyPatch) -> None:
    census_lines = []
    monkeypatch.setattr(
        ic, "logger", SimpleNamespace(info=census_lines.append, warning=census_lines.append)
    )
    baseline, baseline_cache = _engine_and_cache()
    staged, staged_cache = _engine_and_cache()
    commit = ic.InputCommit(capacity=256)  # small: the arenas also grow during the steps
    ic.warmup_input_commit_kernels(commit)
    census = ic.bundle_census()
    census.reset()
    census.begin_step(live=True)
    rng = random.Random(7)
    try:
        for index, step in enumerate(STEPS):
            state = _overlap_state(rng)
            _poison(baseline)
            _poison(staged)
            torch.cuda.synchronize()
            ref = _run_step(baseline, baseline_cache, step, state, None)
            got = _run_step(staged, staged_cache, step, state, commit)
            torch.cuda.synchronize()
            assert ref[0]["input_ids"].shape[0] == len(step) * TOKENS
            assert got[0]["input_ids"].shape[0] == len(step) * TOKENS
            assert _differences(baseline, ref, staged, got, len(step)) == [], index
            # The host-side bookkeeping does not depend on the staging either.
            for r_ref, r_got in zip(ref[2], got[2]):
                assert r_ref.py_batch_idx == r_got.py_batch_idx
                assert r_ref.cached_tokens == r_got.cached_tokens
            assert staged.previous_request_ids == baseline.previous_request_ids
        # Every step went through the commit, with the overlap gathers (also when no request has a
        # previous step: the fixup then only zeroes the offset buffers).
        assert census.get("commits") == census.get("overlap_commits") == len(STEPS)
    finally:
        baseline_cache.shutdown()
        staged_cache.shutdown()
        census.log(final=True)
        assert any(line.startswith("InputCommit census (final): steps=") for line in census_lines)
        census.reset()


def test_negative_control_a_step_without_its_commit_differs() -> None:
    # The comparison above must see a missing write: the staged buffers are poisoned, so a step that
    # prepares its inputs but never commits (the commit is aborted) leaves them behind.
    baseline, baseline_cache = _engine_and_cache()
    staged, staged_cache = _engine_and_cache()
    commit = ic.InputCommit()
    ic.warmup_input_commit_kernels(commit)
    state = _overlap_state(random.Random(11))
    step = STEPS[1]
    try:
        _poison(baseline)
        _poison(staged)
        torch.cuda.synchronize()
        ref = _run_step(baseline, baseline_cache, step, state, None)
        got = _run_step(staged, staged_cache, step, state, commit, commit_it=False)
        torch.cuda.synchronize()
        differing = _differences(baseline, ref, staged, got, len(step))
        assert {"input_ids", "position_ids", "draft_tokens", "previous_pos_id_offsets"} <= set(
            differing
        ), differing
    finally:
        baseline_cache.shutdown()
        staged_cache.shutdown()


def test_engine_creates_the_commit_only_where_the_bundle_was_measured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    messages = []
    monkeypatch.setattr(
        ic, "logger", SimpleNamespace(info=messages.append, warning=messages.append)
    )
    spec = SADecodingConfig(max_draft_len=DRAFT, draft_len_schedule={1: DRAFT})
    census = ic.bundle_census()
    census.reset()

    # Tensor-parallel GEN engine with the switch on: the commit exists and was warmed up silently.
    monkeypatch.setattr(ic, "_GEN_HOST_BUNDLE", True)
    tp_engine = _Engine(_llm_args(), spec)
    assert isinstance(tp_engine._input_commit, ic.InputCommit)
    assert ic.gen_host_bundle_kernels_enabled()
    assert tp_engine._input_commit._num_commits == 0
    assert not any("InputCommit" in m for m in messages), messages

    # Attention DP: the engine switches every part off, once, with a warning.
    adp_engine = _Engine(_llm_args(), spec, attention_dp=True)
    assert adp_engine._input_commit is None
    assert not ic.gen_host_bundle_kernels_enabled()
    assert sum("attention DP is enabled" in m for m in messages) == 1, messages

    # The switch off: no commit and no warning.
    monkeypatch.setattr(ic, "_GEN_HOST_BUNDLE", False)
    messages.clear()
    assert _Engine(_llm_args(), spec)._input_commit is None
    assert not messages
