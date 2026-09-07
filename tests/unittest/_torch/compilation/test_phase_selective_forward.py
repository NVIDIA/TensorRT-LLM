# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import Mock

from tensorrt_llm._torch.compilation.utils import _PhaseSelectiveForward


def test_phase_selective_forward_uses_compiled_by_default() -> None:
    eager_forward = Mock(return_value="eager")
    compiled_forward = Mock(return_value="compiled")
    forward = _PhaseSelectiveForward(eager_forward, compiled_forward)

    assert forward("input") == "compiled"
    compiled_forward.assert_called_once_with("input")
    eager_forward.assert_not_called()


def test_phase_selective_forward_bypass_is_restored() -> None:
    eager_forward = Mock(return_value="eager")
    compiled_forward = Mock(return_value="compiled")
    forward = _PhaseSelectiveForward(eager_forward, compiled_forward)

    assert forward() == "compiled"
    with forward.bypass():
        assert forward() == "eager"
    assert forward() == "compiled"
