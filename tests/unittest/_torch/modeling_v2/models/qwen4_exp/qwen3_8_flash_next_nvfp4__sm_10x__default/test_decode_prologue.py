# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The speculative shell's engine hooks (`_runtime_hooks`), CPU only: the spec loop on the
execution stream and the family's decode prologue (`qsa_spec_step_metadata__decode`), as the FP8
target sets them.

The prologue op's numerics (bitwise the input preparer's corrections and row mapping) are held
by the FP8 target's `test_attention_ops.py`, which drives its cells; the NVFP4 verify steps run
it in the `test_mtp3_gsm8k*` accuracy cases.
"""

import pytest

from tensorrt_llm._torch.modeling_v2.models.qwen4_exp.ops.attention.qsa_spec_step_metadata__decode import (  # noqa: E501
    SpecStepMetadataPrologue,
)
from tensorrt_llm._torch.modeling_v2.models.qwen4_exp.qwen3_8_flash_next_nvfp4__sm_10x__default import (  # noqa: E501
    modeling as M,
)


@pytest.mark.cpu_only
def test_hooks():
    hooks = M._runtime_hooks()
    assert hooks.loop_on_execution_stream
    assert isinstance(hooks.decode_prologue, SpecStepMetadataPrologue)


@pytest.mark.cpu_only
def test_the_decode_forward_reads_the_prologue_positions():
    """The packet is consumed once, for the very tensor the prologue corrected."""
    pos = object()
    md = type("Md", (), {})()
    md.spec_step_positions = (pos, "pos3")
    assert M.take_step_positions(md, object()) is None and md.spec_step_positions is None
    md.spec_step_positions = (pos, "pos3")
    assert M.take_step_positions(md, pos) == "pos3" and md.spec_step_positions is None
