# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Per-family prompt_embeds handling: models that read token ids after the
input embedding treat an embedding row like one of their image positions."""

from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch.models.modeling_deepseekv4 import DeepseekV4ForCausalLM
from tensorrt_llm._torch.models.modeling_deepseekv41 import DeepseekV41ForCausalLM
from tensorrt_llm._torch.models.modeling_gemma4 import Gemma4AssistantForCausalLM, Gemma4ForCausalLM
from tensorrt_llm._torch.models.modeling_minimaxm3 import MiniMaxM3VLForConditionalGeneration
from tensorrt_llm._torch.models.modeling_qwen4_exp import Qwen4ExpForCausalLM
from tensorrt_llm._torch.models.modeling_speculative import (
    Eagle3ForCausalLM,
    MistralLarge3EagleForCausalLM,
)
from tensorrt_llm._torch.models.modeling_utils import DecoderModelForCausalLM

pytestmark = pytest.mark.cpu_only

IDS = torch.tensor([11, 0, 0, 14, 0], dtype=torch.int32)
ROWS = torch.tensor([1, 2, 4])


def _gemma4(ple_dim: int, pad_token_id) -> SimpleNamespace:
    return SimpleNamespace(
        model=SimpleNamespace(hidden_size_per_layer_input=ple_dim),
        config=SimpleNamespace(pad_token_id=pad_token_id),
    )


def test_plain_default() -> None:
    plain = SimpleNamespace()
    assert DecoderModelForCausalLM.prompt_embeds_unsupported_reason(plain) is None
    assert DecoderModelForCausalLM.prompt_embeds_model_inputs(plain, IDS, ROWS) == {}


@pytest.mark.parametrize(
    "ple_dim, pad_token_id, ple_ids",
    [(32, 7, [11, 7, 7, 14, 7]), (0, None, None), (32, None, "refused")],
    ids=["ple_pads_embedding_rows", "no_ple", "ple_without_pad"],
)
def test_gemma4(ple_dim: int, pad_token_id, ple_ids) -> None:
    model = _gemma4(ple_dim, pad_token_id)
    reason = Gemma4ForCausalLM.prompt_embeds_unsupported_reason(model)
    if ple_ids == "refused":
        assert reason
        return
    assert reason is None
    extra = Gemma4ForCausalLM.prompt_embeds_model_inputs(model, IDS, ROWS)
    assert (extra["ple_input_ids"].tolist() if extra else None) == ple_ids
    # Plain rows keep their ids, so plain requests batched alongside are unaffected.
    assert IDS.tolist() == [11, 0, 0, 14, 0]


@pytest.mark.parametrize("cls", [DeepseekV4ForCausalLM, Qwen4ExpForCausalLM])
def test_id_routing_reads_placeholders(cls: type) -> None:
    model = SimpleNamespace(config=SimpleNamespace(has_engram=False))
    extra = cls.prompt_embeds_model_inputs(model, IDS, ROWS)
    assert extra["orig_input_ids"] is IDS


def test_deepseek_engram_refuses() -> None:
    with_engram = SimpleNamespace(config=SimpleNamespace(has_engram=True))
    without = SimpleNamespace(config=SimpleNamespace(has_engram=False))
    assert DeepseekV4ForCausalLM.prompt_embeds_unsupported_reason(with_engram)
    assert DeepseekV4ForCausalLM.prompt_embeds_unsupported_reason(without) is None
    assert DeepseekV41ForCausalLM.prompt_embeds_unsupported_reason(without)


@pytest.mark.parametrize(
    "cls",
    [
        Eagle3ForCausalLM,
        MistralLarge3EagleForCausalLM,
        Gemma4AssistantForCausalLM,
        MiniMaxM3VLForConditionalGeneration,
    ],
)
def test_refused_families(cls: type) -> None:
    assert cls.prompt_embeds_unsupported_reason(SimpleNamespace())
