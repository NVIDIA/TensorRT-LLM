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
"""First-step draft tokens of a disaggregated generation request."""

from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch.speculative import interface as interface_ops
from tensorrt_llm._torch.speculative.interface import SpecMetadata, SpecWorkerBase
from tensorrt_llm._torch.speculative.utils import disagg_gen_first_draft_tokens

pytestmark = pytest.mark.cpu_only

K, SLOTS, VOCAB = 3, 4, 8
REJECTION = SimpleNamespace(use_rejection_sampling=True)
STRICT = SimpleNamespace(use_rejection_sampling=False)


@pytest.mark.parametrize("ctx_drafts", [None, [], [0] * K, [5, 6, 7]])
def test_rejection_always_verifies_placeholders_against_the_one_hot_row(ctx_drafts):
    assert disagg_gen_first_draft_tokens(ctx_drafts, REJECTION, True, K) == ([0] * K, True)


def test_rejection_pads_a_shorter_context_draft_list_to_this_servers_width():
    assert disagg_gen_first_draft_tokens([5, 6], REJECTION, True, K) == ([0] * K, True)


def test_rejection_installs_as_many_placeholders_as_the_context_server_sent():
    assert disagg_gen_first_draft_tokens([5, 6, 7, 8], REJECTION, True, K) == ([0] * 4, True)


def test_strict_keeps_the_context_server_drafts():
    assert disagg_gen_first_draft_tokens([5, 6, 7], STRICT, True, K) == ([5, 6, 7], False)


def test_strict_pads_a_shorter_context_draft_list_to_this_servers_width():
    assert disagg_gen_first_draft_tokens([5, 6], STRICT, True, K) == ([5, 6, 0], False)


@pytest.mark.parametrize("ctx_drafts", [None, []])
def test_strict_without_drafts_gets_placeholders(ctx_drafts):
    assert disagg_gen_first_draft_tokens(ctx_drafts, STRICT, True, K) == ([0] * K, False)


@pytest.mark.parametrize("ctx_drafts, expected", [(None, []), ([], []), ([5, 6, 7], [5, 6, 7])])
@pytest.mark.parametrize("spec_config", [None, REJECTION], ids=["no_spec_config", "rejection"])
def test_without_speculative_decoding_the_drafts_are_unchanged(ctx_drafts, expected, spec_config):
    assert disagg_gen_first_draft_tokens(ctx_drafts, spec_config, False, K) == (expected, False)


def test_flagged_slot_gets_a_one_hot_row_at_the_placeholder():
    """A flagged slot's row is one-hot at the placeholder; other slots keep theirs."""
    metadata = SimpleNamespace(
        use_rejection_sampling=True,
        draft_probs=torch.full((SLOTS, K, VOCAB), 0.5),
        draft_probs_last_dim=VOCAB,
        draft_probs_vocab_size=VOCAB,
        dummy_slot_row=SLOTS - 1,
    )
    tokens, needs_onehot = disagg_gen_first_draft_tokens(None, REJECTION, True, K)
    assert needs_onehot

    SpecMetadata.write_padding_onehot_draft_probs(metadata, [2], K)

    expected = torch.zeros(K, VOCAB)
    expected[torch.arange(K), torch.tensor(tokens)] = 1.0
    assert torch.equal(metadata.draft_probs[2], expected)
    assert torch.equal(metadata.draft_probs[1], torch.full((K, VOCAB), 0.5))


class _StubSpecWorker(SpecWorkerBase):
    """Concrete ``SpecWorkerBase`` that stubs out the abstract API."""

    @property
    def max_draft_len(self) -> int:
        return K

    def _forward_impl(self, *args: object, **kwargs: object) -> None:
        raise NotImplementedError


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="The one-hot row is at draft id 0, which maps to d2t[0], not to the placeholder (target token 0).",
)
def test_one_hot_row_is_the_placeholder_proposal_for_a_reduced_vocabulary_drafter(monkeypatch):
    draft_vocab, slot = 2, 1
    metadata = SimpleNamespace(
        use_rejection_sampling=True,
        draft_probs=torch.full((SLOTS, K, draft_vocab), 0.5),
        draft_probs_last_dim=draft_vocab,
        draft_probs_vocab_size=draft_vocab,
        dummy_slot_row=SLOTS - 1,
        rejection_sample_method="standard",
        is_ragged_verify=False,
        rng_accept_slot=0,
        temperatures=torch.ones(K + 1),
        top_ks=torch.full((K + 1,), VOCAB, dtype=torch.int32),
        top_ps=torch.ones(K + 1),
        min_ps=torch.zeros(K + 1),
        advanced_sampling_mode=None,
        full_draft_probs=None,
        d2t_target_indices=None,
    )
    tokens, _ = disagg_gen_first_draft_tokens(None, REJECTION, True, K)
    SpecMetadata.write_padding_onehot_draft_probs(metadata, [slot], K)

    worker = _StubSpecWorker.__new__(_StubSpecWorker)
    torch.nn.Module.__init__(worker)
    worker.force_num_accepted_tokens = 0.0
    # Draft ids 0 and 1 map to target tokens 1 and 3; no draft id maps to target token 0.
    worker._d2t = torch.tensor([1, 2])
    worker._compute_probs_from_logits = lambda mode, scores, *params: scores.softmax(-1)
    worker._rng_state_per_request = lambda *args, slot: (None, None)
    kernel = {}

    def rejection(**kwargs):
        kernel.update(kwargs)
        return torch.zeros(1, K + 1, dtype=torch.int32), torch.ones(1, dtype=torch.int32)

    monkeypatch.setattr(interface_ops, "rejection_sampling_one_model", rejection)
    worker._sample_and_accept_draft_tokens_rejection(
        torch.zeros(K + 1, VOCAB),
        torch.tensor([tokens]),
        metadata.draft_probs[[slot]],
        0,
        1,
        metadata,
    )

    placeholder = kernel["draft_token_ids"][0].long()
    proposal = kernel["draft_probs"][0].gather(-1, placeholder[:, None]).squeeze(-1)
    assert proposal.tolist() == [1.0] * K
