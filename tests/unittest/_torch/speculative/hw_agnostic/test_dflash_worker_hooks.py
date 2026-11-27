# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The DFlash worker's override points for drafters that differ from the standalone DFlash one.

A worker subclass whose drafter embeds its own block or reads the step's attention metadata in
its block forward overrides ``_draft_block_embedding``, ``_draft_block_forward``, and
``_uses_fused_paged_latent_prepare``. These tests pin the defaults every other drafter keeps
and that ``_refine_block_logits`` keeps its 4-argument contract.
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


def test_refinement_keeps_four_argument_signature():
    class Refined(Exception):
        pass

    seen = {}

    class FourArgWorker(DFlashWorker):
        def _refine_block_logits(self, draft_model, gen_logits, inputs, spec_metadata):
            seen["args"] = (draft_model, gen_logits, inputs, spec_metadata)
            raise Refined

    draft = SimpleNamespace()
    logits = torch.zeros(2, 3, 8)
    inputs = {}
    spec_metadata = SimpleNamespace()
    with pytest.raises(Refined):
        _worker(FourArgWorker)._sample_block_draft_tokens(
            draft, logits, torch.randn(6, 4), inputs, spec_metadata, 0, 2
        )
    assert seen["args"] == (draft, logits, inputs, spec_metadata)
