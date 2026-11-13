# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""DEP_GUMBEL_DRAFT's switch line and contract: the block sampler names each reason an operand is outside its contract
and declines CPU operands. The install into the DSpark ancestral method is
``test_modeling_v2_m31_dep_sampler_overrides.py``'s. CPU only."""

import types

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2._router_index import NoTarget
from tensorrt_llm.llmapi.llm_args import AdvancedSamplingMode

try:
    from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_1_flash_nvfp4__sm_107__dep import (  # noqa: E501
        gumbel_draft,
    )
except NoTarget as exc:
    pytest.skip(str(exc), allow_module_level=True)

pytestmark = pytest.mark.cpu_only

ROWS, K, VOCAB, RANK = 4, 3, 64, 16


def test_the_engagement_line_names_the_switch(monkeypatch):
    monkeypatch.setattr(gumbel_draft, "DEP_GUMBEL_DRAFT", False)
    assert "TRTLLM_M31_DISABLE_DEP_GUMBEL_DRAFT=1" in gumbel_draft.engagement()
    monkeypatch.setattr(gumbel_draft, "DEP_GUMBEL_DRAFT", True)
    monkeypatch.setitem(gumbel_draft._STATE, "fn", lambda *args: None)
    assert gumbel_draft.engagement().startswith("engaged")


def _operands(**over):
    ops = dict(
        gen_logits=torch.zeros(ROWS, K, VOCAB),
        anchors=torch.zeros(ROWS, dtype=torch.int32),
        markov_w1=torch.zeros(VOCAB, RANK, dtype=torch.bfloat16),
        markov_w2=torch.zeros(VOCAB, RANK, dtype=torch.bfloat16),
        temperatures=torch.ones(ROWS),
        slots=torch.arange(ROWS),
        draft_probs=torch.zeros(ROWS + 1, K, VOCAB),
    )
    ops.update(over)
    return ops


@pytest.mark.parametrize(
    "over, needle",
    [
        ({}, "operands on"),  # well formed, on the CPU
        ({"gen_logits": torch.zeros(ROWS, VOCAB)}, "gen_logits"),
        ({"gen_logits": torch.zeros(ROWS, K, VOCAB, dtype=torch.int32)}, "gen_logits"),
        ({"markov_w1": torch.zeros(VOCAB - 1, RANK, dtype=torch.bfloat16)}, "markov_w1"),
        ({"markov_w2": torch.zeros(VOCAB + 8, RANK, dtype=torch.bfloat16)}, "markov_w2"),
        ({"markov_w2": torch.zeros(VOCAB, RANK)}, "markov_w2"),
        ({"anchors": torch.zeros(ROWS, 1, dtype=torch.int32)}, "anchors"),
        ({"temperatures": torch.ones(ROWS, dtype=torch.float64)}, "temperatures"),
        ({"slots": torch.arange(ROWS, dtype=torch.int32)}, "slots"),
        ({"draft_probs": torch.zeros(ROWS + 1, K, VOCAB, dtype=torch.bfloat16)}, "draft_probs"),
        ({"draft_probs": torch.zeros(ROWS + 1, K - 1, VOCAB)}, "draft_probs"),
        ({"draft_probs": torch.zeros(ROWS + 1, K, VOCAB - 1)}, "draft_probs"),
        ({"draft_probs": None}, "operands on"),  # no proposal buffer is fine; still the CPU
    ],
)
def test_the_contract_names_each_reason(over, needle):
    reason = gumbel_draft.block_outside_contract(**_operands(**over))
    assert reason is not None and reason.startswith(needle), reason


class _Worker:
    """The worker surface ``draw_block`` reads, with a deterministic CPU sampler (argmax)."""

    @staticmethod
    def _sample_from_logits_with_probs(
        mode, logits, temperatures, top_ks, top_ps, min_ps, *, seed, offset
    ):
        probs = torch.softmax(logits / temperatures.unsqueeze(-1), dim=-1)
        return probs.argmax(-1).to(torch.int32), probs

    def maybe_gather_sharded_draft_logits(self, gen_logits, spec_metadata):
        return gen_logits

    def _zero_padding_rows(self, logits, spec_metadata, num_contexts, batch_size, rows_per_request):
        return logits

    def _rng_state_per_request(self, spec_metadata, num_contexts, batch_size, *, slot):
        return None, None


def _case():
    gen = torch.Generator().manual_seed(0)
    draft_model = types.SimpleNamespace(
        markov_w1=torch.randn(VOCAB, RANK, generator=gen).to(torch.bfloat16),
        markov_w2=torch.randn(VOCAB, RANK, generator=gen).to(torch.bfloat16),
    )
    meta = types.SimpleNamespace(
        advanced_sampling_mode=AdvancedSamplingMode.NO_TOPK_NO_TOPP,
        draft_probs=torch.zeros(ROWS + 1, K, VOCAB),
        batch_slot_ids=torch.tensor([3, 0, 2, 1]),
        request_temperatures=torch.ones(ROWS),
        rng_draft_slot=lambda step: step,
    )
    inputs = {"first_prev_tokens": torch.tensor([5, -1, VOCAB, 9], dtype=torch.int32)}
    return draft_model, torch.randn(ROWS, K, VOCAB, generator=gen), inputs, meta


def test_draw_block_declines_cpu_operands(monkeypatch):
    monkeypatch.setitem(
        gumbel_draft._STATE, "fn", lambda *args: pytest.fail("the sampler must not run")
    )
    draft_model, gen_logits, inputs, meta = _case()
    assert (
        gumbel_draft.draw_block(_Worker(), draft_model, gen_logits, inputs, meta, 0, ROWS) is None
    )
