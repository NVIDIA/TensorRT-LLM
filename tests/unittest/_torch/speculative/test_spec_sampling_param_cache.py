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
"""The one-model sampling buffers equal the per-request / per-row Python expansion."""

import random
import types

import pytest
import torch

from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequestState
from tensorrt_llm._torch.pyexecutor.sampler.sampler_common import request_random_seed
from tensorrt_llm._torch.speculative.interface import (
    _RNG_SLOT_SPAN,
    DEFAULT_SAMPLING_SEED,
    SpecMetadata,
    SpeculativeDecodingMode,
)

DRAFT_LEN = 3


class _Request:
    def __init__(self, slot, *, temperature=None, top_k=None, top_p=None, seed=None):
        self.sampling_config = types.SimpleNamespace(
            temperature=temperature, top_k=top_k, top_p=top_p, min_p=None, seed=seed
        )
        self.state = LlmRequestState.GENERATION_IN_PROGRESS
        self.py_seq_slot = slot
        self.py_request_id = 1000 + slot
        self.py_spec_sampling_params = None
        self.py_logits_post_processors = None


def _batch(rng, n):
    requests = []
    for slot in range(n):
        kind = rng.choice(["greedy", "t1", "topk", "seeded"])
        if kind == "greedy":
            request = _Request(slot)
        elif kind == "t1":
            request = _Request(slot, temperature=1.0, top_p=1.0, top_k=0)
        elif kind == "topk":
            request = _Request(slot, temperature=0.7, top_k=20, top_p=0.9)
        else:
            request = _Request(slot, temperature=1.0, seed=rng.randrange(1 << 64))
        if rng.random() < 0.2:
            request.state = LlmRequestState.CONTEXT_INIT
        requests.append(request)
    return requests


def _metadata(max_num_requests=64):
    metadata = SpecMetadata(
        max_num_requests=max_num_requests,
        max_draft_len=DRAFT_LEN,
        max_total_draft_tokens=DRAFT_LEN,
        spec_dec_mode=SpeculativeDecodingMode.MTP,
    )
    metadata.runtime_draft_len = DRAFT_LEN
    return metadata


def _reference_rng_rows(request_seeds, request_offsets, num_tokens):
    seeds, offsets = [], []
    for seed, offset, n in zip(request_seeds, request_offsets, num_tokens):
        seeds.extend(seed for _ in range(n))
        offsets.extend(offset + row * _RNG_SLOT_SPAN for row in range(n))
    return seeds, offsets


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("seed", range(3))
def test_rng_rows_match_the_per_row_expansion(seed):
    rng = random.Random(seed)
    metadata = _metadata()
    reference = _metadata()
    for step in range(4):
        requests = _batch(rng, rng.randrange(1, 40))
        normalized, _ = metadata._scan_one_model_sampling(requests)
        # Mixed row counts, as context rows (1) and ragged verify windows give.
        normalized = [(*entry[:4], rng.randrange(1, 5)) for entry in normalized]
        # All-greedy batches skip the RNG upload.
        metadata.is_all_greedy_sample = False
        metadata._populate_request_rng_state(requests, normalized)

        user_seeds = [request_random_seed(r) for r in requests]
        request_seeds = [DEFAULT_SAMPLING_SEED if s is None else s for s in user_seeds]
        request_offsets = reference._take_rng_window_offsets(
            requests, [s is not None for s in user_seeds]
        )
        seeds, offsets = _reference_rng_rows(
            request_seeds, request_offsets, [n for *_, n in normalized]
        )
        n = len(requests)
        assert metadata.request_seeds[:n].tolist() == request_seeds, f"step {step}"
        assert metadata.request_offsets[:n].tolist() == request_offsets, f"step {step}"
        assert metadata.seeds[: len(seeds)].tolist() == seeds, f"step {step}"
        assert metadata.offsets[: len(offsets)].tolist() == offsets, f"step {step}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("seed", range(3))
def test_sampling_buffers_match_the_per_request_expansion(seed):
    rng = random.Random(seed)
    metadata = _metadata()
    for step in range(6):
        requests = _batch(rng, rng.randrange(1, 40))
        metadata.populate_sampling_params_for_one_model(requests)
        if metadata.is_all_greedy_sample:
            continue
        normalized, _ = metadata._scan_one_model_sampling(requests)
        per_token = [[], [], [], []]
        for *values, n in normalized:
            for column, value in zip(per_token, values):
                column.extend(value for _ in range(n))
        rows, n = len(per_token[0]), len(requests)
        dtypes = (torch.float32, torch.int32, torch.float32, torch.float32)
        per_request = (
            metadata.request_temperatures,
            metadata.request_top_ks,
            metadata.request_top_ps,
            metadata.request_min_ps,
        )
        for column, (buffer, dtype) in enumerate(zip(per_request, dtypes)):
            want = torch.tensor([v[column] for v in normalized], dtype=dtype)
            assert torch.equal(buffer[:n].cpu(), want), f"step {step}"
        expanded = (metadata.temperatures, metadata.top_ks, metadata.top_ps, metadata.min_ps)
        for buffer, dtype, values in zip(expanded, dtypes, per_token):
            assert torch.equal(buffer[:rows].cpu(), torch.tensor(values, dtype=dtype)), (
                f"step {step}"
            )
        disable = torch.iinfo(torch.int32).max
        assert metadata.top_k_max == max(
            (v[1] for v in normalized if 0 < v[1] < disable), default=0
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_sampled_steps_after_greedy_steps_get_the_reference_windows():
    """Greedy steps skip the RNG uploads; later sampled steps still get the reference windows."""
    metadata = _metadata()
    reference = _metadata()
    greedy = [_Request(slot) for slot in range(4)]
    sampled = [_Request(slot, temperature=0.7, seed=slot or None) for slot in range(4)]
    for step, requests in enumerate((greedy, sampled, greedy, greedy, sampled, greedy)):
        metadata.populate_sampling_params_for_one_model(requests)
        user_seeds = [request_random_seed(r) for r in requests]
        expected_offsets = reference._take_rng_window_offsets(
            requests, [s is not None for s in user_seeds]
        )
        assert metadata.is_all_greedy_sample == (requests is greedy)
        if requests is sampled:
            n = len(requests)
            assert metadata.request_offsets[:n].tolist() == expected_offsets, f"step {step}"
            seeds = [DEFAULT_SAMPLING_SEED if s is None else s for s in user_seeds]
            assert metadata.request_seeds[:n].tolist() == seeds, f"step {step}"
