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
"""Regression tests for ``SpecMetadata.dp_num_tokens``.

The attention-DP allgather in ``model_engine`` publishes the speculative token
count *before* ``prepare()`` runs, while every consumer reads the value
``prepare()`` leaves behind. When the two were derived independently they
drifted apart and 3 of 4 spec modes published the wrong per-rank count under
attention DP (nvbugs/6707519). These tests pin the invariant that makes the
single derivation point meaningful:

    dp_num_tokens() == self.num_tokens after prepare()

The invariant is pure integer arithmetic, so these run without a GPU: the
``cpu_allocation`` fixture redirects the ``device='cuda'`` buffer allocations
in ``__post_init__``/``prepare()`` to the CPU. Everything else -- the real
constructors and the real ``prepare()`` -- is exercised unchanged.
"""

import pytest
import torch

from tensorrt_llm._torch.speculative.draft_target import DraftTargetOneModelSpecMetadata
from tensorrt_llm._torch.speculative.eagle3 import Eagle3OneModelSpecMetadata
from tensorrt_llm._torch.speculative.interface import SpeculativeDecodingMode
from tensorrt_llm._torch.speculative.mtp import MTPSpecMetadata

# (id, factory, expected number of tokens published for the batch below)
#
# The batch is fixed for every mode: 2 generation requests out of 3 sequences,
# so a mode that subtracts per generation request produces a count strictly
# below NUM_TOKENS and a regression cannot pass by accident.
NUM_SEQS = 3
NUM_GENERATIONS = 2
NUM_TOKENS = 16
MAX_DRAFT_LEN = 3
MAX_TOTAL_DRAFT_TOKENS = 5


def _mtp_vanilla() -> MTPSpecMetadata:
    return MTPSpecMetadata(
        max_num_requests=NUM_SEQS,
        max_draft_len=MAX_DRAFT_LEN,
        max_total_draft_tokens=MAX_DRAFT_LEN,
        spec_dec_mode=SpeculativeDecodingMode.MTP,
        mtp_num_modules=MAX_DRAFT_LEN,
    )


def _mtp_eagle() -> MTPSpecMetadata:
    return MTPSpecMetadata(
        max_num_requests=NUM_SEQS,
        max_draft_len=MAX_DRAFT_LEN,
        max_total_draft_tokens=MAX_DRAFT_LEN,
        spec_dec_mode=SpeculativeDecodingMode.MTP_EAGLE_ONE_MODEL,
        mtp_num_modules=MAX_DRAFT_LEN,
    )


def _eagle3_linear() -> Eagle3OneModelSpecMetadata:
    return Eagle3OneModelSpecMetadata(
        max_num_requests=NUM_SEQS,
        max_draft_len=MAX_DRAFT_LEN,
        max_total_draft_tokens=MAX_DRAFT_LEN,
        spec_dec_mode=SpeculativeDecodingMode.EAGLE3_ONE_MODEL,
        num_layers=1,
        hidden_size=2,
        max_num_tokens=NUM_TOKENS,
        dtype=torch.float32,
        layers_to_capture={0},
    )


def _eagle3_tree() -> Eagle3OneModelSpecMetadata:
    # A static tree subtracts max_total_draft_tokens rather than max_draft_len.
    metadata = Eagle3OneModelSpecMetadata(
        max_num_requests=NUM_SEQS,
        max_draft_len=MAX_DRAFT_LEN,
        max_total_draft_tokens=MAX_TOTAL_DRAFT_TOKENS,
        spec_dec_mode=SpeculativeDecodingMode.EAGLE3_ONE_MODEL,
        num_layers=1,
        hidden_size=2,
        max_num_tokens=NUM_TOKENS,
        dtype=torch.float32,
        layers_to_capture={0},
    )
    metadata.is_spec_dec_tree = True
    return metadata


def _draft_target() -> DraftTargetOneModelSpecMetadata:
    return DraftTargetOneModelSpecMetadata(
        max_num_requests=NUM_SEQS,
        max_draft_len=MAX_DRAFT_LEN,
        max_total_draft_tokens=MAX_DRAFT_LEN,
        spec_dec_mode=SpeculativeDecodingMode.DRAFT_TARGET_ONE_MODEL,
        max_num_tokens=NUM_TOKENS,
    )


CASES = [
    # MTP vanilla drops one verification position per generation request.
    ("mtp_vanilla", _mtp_vanilla, NUM_TOKENS - NUM_GENERATIONS),
    # MTP Eagle keeps the 1st-draft-forward shape, which matches input_ids.
    ("mtp_eagle_one_model", _mtp_eagle, NUM_TOKENS),
    ("eagle3_one_model_linear", _eagle3_linear, NUM_TOKENS - NUM_GENERATIONS * MAX_DRAFT_LEN),
    ("eagle3_one_model_tree", _eagle3_tree, NUM_TOKENS - NUM_GENERATIONS * MAX_TOTAL_DRAFT_TOKENS),
    ("draft_target_one_model", _draft_target, NUM_TOKENS - NUM_GENERATIONS * MAX_DRAFT_LEN),
]


@pytest.fixture
def cpu_allocation(monkeypatch):
    """Redirect the metadata's ``device='cuda'`` allocations to the CPU.

    The spec metadata classes hardcode ``device='cuda'`` for their index
    buffers, which would otherwise force this arithmetic-only test onto a GPU
    runner. The spec modules all do a plain ``import torch``, so patching the
    factories on the module object covers every allocation site. Copies into
    the redirected buffers stay CPU->CPU and ``prefer_pinned()`` is already
    False when no device is present.
    """

    def to_cpu(fn):
        def wrapper(*args, **kwargs):
            if kwargs.get("device") == "cuda":
                kwargs["device"] = "cpu"
            return fn(*args, **kwargs)

        return wrapper

    for name in ("empty", "arange", "zeros", "tensor"):
        monkeypatch.setattr(torch, name, to_cpu(getattr(torch, name)))


@pytest.mark.parametrize(
    "factory,expected",
    [(factory, expected) for _, factory, expected in CASES],
    ids=[case_id for case_id, _, _ in CASES],
)
def test_dp_num_tokens_matches_prepare(cpu_allocation, factory, expected) -> None:
    """The pre-prepare allgather value must equal the post-prepare count."""
    metadata = factory()
    metadata.request_ids = list(range(NUM_SEQS))
    metadata.seq_lens = [1] * NUM_SEQS
    metadata.num_generations = NUM_GENERATIONS
    metadata.num_tokens = NUM_TOKENS
    metadata.runtime_draft_len = MAX_DRAFT_LEN

    # What model_engine allgathers, read before prepare() rewrites num_tokens.
    published = metadata.dp_num_tokens()
    assert published == expected

    metadata.prepare()

    # What every consumer of the count actually reads.
    assert metadata.num_tokens == published
