# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Model-specific LLM args are rejected for models that do not declare them."""

from contextlib import nullcontext
from types import SimpleNamespace

import pytest

from tensorrt_llm._torch.models.modeling_qwen3_5 import Qwen3_5MoeForCausalLM, Qwen3_5MoeVLModel
from tensorrt_llm._torch.models.modeling_qwen3_next import Qwen3NextForCausalLM
from tensorrt_llm._torch.models.modeling_utils import DecoderModelForCausalLM
from tensorrt_llm._torch.pyexecutor.model_loader import _reject_unsupported_model_llm_args


@pytest.mark.parametrize(
    "model_cls, supported",
    [
        (DecoderModelForCausalLM, False),
        (Qwen3NextForCausalLM, True),
        (Qwen3_5MoeForCausalLM, True),
        (Qwen3_5MoeVLModel, True),
    ],
)
@pytest.mark.parametrize("name", ["replicate_embedding", "mnnvl_allreduce_fp8_payload"])
def test_model_specific_llm_args(model_cls, supported, name):
    llm_args = SimpleNamespace(replicate_embedding=False, mnnvl_allreduce_fp8_payload=False)
    _reject_unsupported_model_llm_args(llm_args, model_cls)
    setattr(llm_args, name, True)
    with nullcontext() if supported else pytest.raises(ValueError, match=name):
        _reject_unsupported_model_llm_args(llm_args, model_cls)
