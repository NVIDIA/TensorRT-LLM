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
"""Target sampling of the Eagle3 dynamic-tree worker under ``AdvancedSamplingMode.FULL``."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from tensorrt_llm._torch.speculative.eagle3_dynamic_tree import Eagle3OneModelDynamicTreeWorker
from tensorrt_llm.llmapi.llm_args import AdvancedSamplingMode

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")

DEV = "cuda"
VOCAB = 4096
NODES = 3
DISABLE_TOPK = torch.iinfo(torch.int32).max


def _worker(batch: int) -> Eagle3OneModelDynamicTreeWorker:
    worker = object.__new__(Eagle3OneModelDynamicTreeWorker)
    worker.spec_config = SimpleNamespace(max_draft_len=1)
    worker.force_num_accepted_tokens = 0.0
    worker.tokens_per_gen_step = NODES
    worker._max_batch_size = batch
    worker._max_path_len = 2
    worker._accepted_tokens_buf = torch.zeros(batch, 2, dtype=torch.int32, device=DEV)
    worker._num_accepted_tokens_buf = torch.ones(batch, dtype=torch.int32, device=DEV)
    worker._accepted_draft_indices_tensor = torch.full(
        (batch, 1), -1, dtype=torch.int32, device=DEV
    )
    worker._target_tokens_buf = torch.zeros(batch * NODES, dtype=torch.int64, device=DEV)
    worker.spec_tree_manager = None
    return worker


def _metadata(rows: int, seeds: torch.Tensor, *, min_p: float = 0.0) -> SimpleNamespace:
    """Per-row sampling params for ``rows`` context requests, one logits row each."""
    temperatures = torch.ones(rows, dtype=torch.float32, device=DEV)
    top_ks = torch.full((rows,), DISABLE_TOPK, dtype=torch.int32, device=DEV)
    top_ps = torch.ones(rows, dtype=torch.float32, device=DEV)
    min_ps = torch.full((rows,), min_p, dtype=torch.float32, device=DEV)
    return SimpleNamespace(
        advanced_sampling_mode=AdvancedSamplingMode.FULL,
        is_all_greedy_sample=False,
        is_cuda_graph=False,
        temperatures=temperatures,
        top_ks=top_ks,
        top_ps=top_ps,
        min_ps=min_ps,
        seeds=seeds,
        offsets=torch.zeros(rows, dtype=torch.int64, device=DEV),
    )


def _sample_contexts(rejection: bool, logits: torch.Tensor, metadata) -> torch.Tensor:
    rows = logits.shape[0]
    worker = _worker(rows)
    method = (
        worker._sample_and_accept_dynamic_tree_rejection
        if rejection
        else worker._sample_and_accept_dynamic_tree
    )
    accepted, _ = method(logits, None, metadata, rows, rows, 0)
    return accepted[:, 0].clone()


@pytest.mark.parametrize("rejection", [False, True], ids=["plain", "rejection"])
def test_context_token_follows_its_own_seed_wherever_it_sits(rejection: bool) -> None:
    rows = 8
    gen = torch.Generator(device=DEV).manual_seed(0)
    logits = torch.randn(rows, VOCAB, device=DEV, generator=gen)
    seeds = torch.arange(100, 100 + rows, dtype=torch.int64, device=DEV)
    perm = torch.randperm(rows, generator=torch.Generator().manual_seed(1)).to(DEV)

    tokens = _sample_contexts(rejection, logits, _metadata(rows, seeds))
    permuted = _sample_contexts(rejection, logits[perm], _metadata(rows, seeds[perm]))

    assert torch.equal(permuted, tokens[perm])


@pytest.mark.parametrize("rejection", [False, True], ids=["plain", "rejection"])
def test_context_token_respects_min_p(rejection: bool) -> None:
    rows = 512
    logits = torch.zeros(rows, VOCAB, device=DEV)
    logits[:, :4] = 4.0
    probs = torch.softmax(logits[0], dim=-1)
    filtered = probs < 0.5 * probs.max()
    seeds = torch.arange(rows, dtype=torch.int64, device=DEV)

    unfiltered = _sample_contexts(rejection, logits, _metadata(rows, seeds))
    with_min_p = _sample_contexts(rejection, logits, _metadata(rows, seeds, min_p=0.5))

    assert filtered[unfiltered.long()].any()
    assert not filtered[with_min_p.long()].any()


def test_rejection_target_probs_use_each_request_params() -> None:
    num_gens = 2
    gen = torch.Generator(device=DEV).manual_seed(2)
    logits = torch.randn(num_gens * NODES, VOCAB, device=DEV, generator=gen)
    min_ps = torch.tensor([0.5, 0.0], dtype=torch.float32, device=DEV)

    worker = _worker(num_gens)
    worker.spec_tree_manager = SimpleNamespace(
        slot_storage=SimpleNamespace(
            all_ids_buf=torch.arange(num_gens, device=DEV),
            has_tree=torch.ones(num_gens, dtype=torch.bool, device=DEV),
            next_links_from_slots=lambda ids, n: (None, None),
        )
    )
    worker.tree_ops_converter = MagicMock()
    worker.tree_ops_converter.verify_dynamic_tree_rejection_out.return_value = (None, None, None)
    worker._finalize_dynamic_tree_verify_outputs = MagicMock()
    metadata = SimpleNamespace(
        advanced_sampling_mode=AdvancedSamplingMode.FULL,
        is_cuda_graph=False,
        request_temperatures=torch.ones(num_gens, dtype=torch.float32, device=DEV),
        request_top_ks=torch.full((num_gens,), DISABLE_TOPK, dtype=torch.int32, device=DEV),
        request_top_ps=torch.ones(num_gens, dtype=torch.float32, device=DEV),
        request_min_ps=min_ps,
        request_seeds=torch.zeros(num_gens, dtype=torch.int64, device=DEV),
        request_offsets=torch.zeros(num_gens, dtype=torch.int64, device=DEV),
        rng_accept_slot=0,
        draft_tokens=torch.zeros(num_gens * (NODES - 1), dtype=torch.int32, device=DEV),
    )

    worker._sample_and_accept_dynamic_tree_rejection(logits, None, metadata, num_gens, 0, num_gens)

    target_probs = worker.tree_ops_converter.verify_dynamic_tree_rejection_out.call_args.args[1]
    softmax = torch.softmax(logits, dim=-1).reshape(num_gens, NODES, VOCAB)
    kept = softmax >= 0.5 * softmax.amax(dim=-1, keepdim=True)
    assert target_probs.shape == (num_gens, NODES, VOCAB)
    assert torch.all(target_probs[0][~kept[0]] == 0)
    torch.testing.assert_close(target_probs[1], softmax[1])
