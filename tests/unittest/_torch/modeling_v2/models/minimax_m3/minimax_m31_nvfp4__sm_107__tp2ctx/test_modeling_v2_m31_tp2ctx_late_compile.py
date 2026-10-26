# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The TP2 context target arms the late-compile detector, like the generation targets.

The executor calls mark_server_ready() after warmup only for a model whose RuntimeHooks opt into
late_compile_detect, so without it a context-only server never marks ready. CPU.
"""

from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m31_nvfp4__sm_107__tp2ctx import (
    modeling as m,
)
from tensorrt_llm._torch.pyexecutor.py_executor import _arm_runtime_opt_ins
from tensorrt_llm._torch.pyexecutor.runtime_hooks import RuntimeHooks


def test_tp2ctx_arms_the_late_compile_detector():
    assert m.ModelingV2MiniMaxM31Nvfp4Sm107Tp2Ctx.runtime_hooks.late_compile_detect
    # The executor arms the detector from that field alone (the target's other opt-ins are process-wide).
    assert _arm_runtime_opt_ins(RuntimeHooks(late_compile_detect=True))


def test_a_model_without_the_opt_in_does_not_arm_it():
    assert not _arm_runtime_opt_ins(RuntimeHooks())
