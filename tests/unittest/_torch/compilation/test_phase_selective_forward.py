# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import Mock, patch

from tensorrt_llm._torch.compilation.utils import _PhaseSelectiveForward


def _attention_attrs(num_contexts: int) -> dict[str, object]:
    metadata = SimpleNamespace(num_contexts=num_contexts)
    return {"attention_metadata": lambda: metadata}


def test_phase_selective_forward_dispatches_by_batch_phase():
    eager_forward = Mock(return_value="eager")
    compiled_forward = Mock(return_value="compiled")
    forward = _PhaseSelectiveForward(eager_forward, compiled_forward)

    with patch(
        "tensorrt_llm._torch.compilation.utils.get_model_extra_attrs",
        return_value=_attention_attrs(num_contexts=1),
    ):
        assert forward("context") == "compiled"

    with patch(
        "tensorrt_llm._torch.compilation.utils.get_model_extra_attrs",
        return_value=_attention_attrs(num_contexts=0),
    ):
        assert forward("generation") == "eager"

    compiled_forward.assert_called_once_with("context")
    eager_forward.assert_called_once_with("generation")


def test_phase_selective_forward_bypass_is_restored():
    eager_forward = Mock(return_value="eager")
    compiled_forward = Mock(return_value="compiled")
    forward = _PhaseSelectiveForward(eager_forward, compiled_forward)

    with patch(
        "tensorrt_llm._torch.compilation.utils.get_model_extra_attrs",
        return_value=_attention_attrs(num_contexts=1),
    ):
        assert forward() == "compiled"
        with forward.bypass():
            assert forward() == "eager"
        assert forward() == "compiled"
