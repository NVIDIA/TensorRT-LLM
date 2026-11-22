# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The DFlash worker's override points for drafters that differ from the standalone DFlash one.

A worker subclass whose drafter embeds its own block, reads the step's attention metadata in
its block forward, or refines its logits from the block's draft hidden states overrides
``_draft_block_embedding``, ``_draft_block_forward``, ``_uses_fused_paged_latent_prepare`` and
``_refine_block_logits``. These tests pin the defaults every other drafter keeps, and that the
refinement receives the block's draft hidden states.
"""

from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch.speculative.dflash import DFlashWorker, dflash_noise_block_embedding

pytestmark = pytest.mark.cpu_only


def _worker(cls=DFlashWorker):
    return object.__new__(cls)


def test_block_embedding_defaults_to_the_token_embedding():
    embed_tokens = torch.nn.Embedding(6, 4)
    draft = SimpleNamespace(
        draft_model_full=SimpleNamespace(model=SimpleNamespace(embed_tokens=embed_tokens))
    )
    bonus = torch.tensor([2, 5])
    trained_mask = torch.randn(4)
    actual = _worker()._draft_block_embedding(draft, bonus, 1, 3, trained_mask)
    expected = dflash_noise_block_embedding(embed_tokens, bonus, 1, 3, trained_mask)
    torch.testing.assert_close(actual, expected)


def test_block_forward_defaults_to_dflash_forward_without_the_metadata():
    out = torch.empty(0)
    seen = {}

    def dflash_forward(**kwargs):
        seen.update(kwargs)
        return out

    draft = SimpleNamespace(dflash_forward=dflash_forward)
    actual = _worker()._draft_block_forward(draft, object(), noise_embedding=1, query_positions=2)
    assert actual is out
    assert seen == {"noise_embedding": 1, "query_positions": 2}


@pytest.mark.parametrize(
    ("kv_factor", "lilicorr", "expected"), [(1, False, True), (1, True, False), (2, False, False)]
)
def test_fused_paged_latent_prepare_serves_single_latent_drafters_but_lilicorr(
    kv_factor, lilicorr, expected
):
    draft = SimpleNamespace(_uses_lilicorr=lilicorr)
    assert _worker()._uses_fused_paged_latent_prepare(draft, kv_factor) is expected


@pytest.mark.parametrize("with_hidden_states", [False, True])
def test_refinement_receives_the_block_draft_hidden_states(with_hidden_states):
    class Refined(Exception):
        pass

    seen = {}

    class RecordingWorker(DFlashWorker):
        def _refine_block_logits(
            self, draft_model, gen_logits, inputs, spec_metadata, draft_hidden_states=None
        ):
            seen["hidden_states"] = draft_hidden_states
            raise Refined

    hidden_states = torch.randn(6, 4) if with_hidden_states else None
    with pytest.raises(Refined):
        _worker(RecordingWorker)._sample_block_draft_tokens(
            SimpleNamespace(), torch.zeros(2, 3, 8), hidden_states, {}, SimpleNamespace(), 0, 2
        )
    if with_hidden_states:
        torch.testing.assert_close(seen["hidden_states"], hidden_states.reshape(2, 3, 4))
    else:
        assert seen["hidden_states"] is None
