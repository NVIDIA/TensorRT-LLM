# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""DEP_MARKOV_GLUE's switch line, and the glue's methods decline CPU tensors. The install into the DSpark ancestral loop
is ``test_modeling_v2_m31_dep_sampler_overrides.py``'s. CPU only."""

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2._router_index import NoTarget

try:
    from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_1_flash_nvfp4__sm_107__dep import (  # noqa: E501
        markov_glue,
    )
except NoTarget as exc:
    pytest.skip(str(exc), allow_module_level=True)

pytestmark = pytest.mark.cpu_only

ROWS, K, VOCAB, RANK = 4, 3, 64, 16


def test_the_engagement_line_names_the_switch(monkeypatch):
    monkeypatch.setattr(markov_glue, "DEP_MARKOV_GLUE", False)
    assert "TRTLLM_M31_DISABLE_DEP_MARKOV_GLUE=1" in markov_glue.engagement()
    monkeypatch.setattr(markov_glue, "DEP_MARKOV_GLUE", True)
    assert markov_glue.engagement().startswith("engaged")


def test_cpu_tensors_are_declined():
    glue = markov_glue.MarkovGlue()
    w1 = torch.zeros(VOCAB, RANK, dtype=torch.bfloat16)
    assert glue.step_bias(torch.zeros(ROWS, dtype=torch.int64), w1, w1) is None
    bias = torch.zeros(ROWS, VOCAB, dtype=torch.bfloat16)
    assert glue.biased_logits(torch.zeros(ROWS, K, VOCAB), 0, bias) is None
    draft_probs = torch.zeros(ROWS, K, VOCAB)
    assert not glue.store_probs(draft_probs, torch.arange(ROWS), 0, torch.ones(ROWS, VOCAB))
    assert torch.count_nonzero(draft_probs) == 0
