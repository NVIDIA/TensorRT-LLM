# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The step-loop hooks a glm_moe_dsa target declares to the engine (``RuntimeHooks``).

``make_runtime_hooks`` declares the engine-side fast paths that the plan runs: the
decode step's input prologue in one launch (``decode_prologue``), the overlap executor
loop on the execution stream (``execution_stream_loop``), and the executor events of the
post-MoE hop health check (``post_moe_hop``) and the host page retention policy
(``host_page_retention``). Every hook it leaves undeclared stays off, and the engine runs
that part of the step loop as it does for the built-in model.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from tensorrt_llm._torch.pyexecutor.runtime_hooks import RuntimeHooks

if TYPE_CHECKING:
    from tensorrt_llm._torch.models.modeling_deepseekv3 import GlmMoeDsaForCausalLM

    from .dispatch import Plan


class GlmExecutorEvents:
    """The executor events of a glm_moe_dsa target (``ExecutorEvents``).

    A kernel package is imported only when its fast path runs.
    """

    def __init__(self, plan: "Plan") -> None:
        self._plan = plan

    def after_warmup(self) -> None:
        if self._plan.enabled("post_moe_hop"):
            from .kernels.post_moe_hop import check_health

            # Collective over each post-MoE hop workspace's TP group: every rank
            # fails startup together when any hop reported a device error during
            # warmup or graph capture.
            check_health("after_warmup", collective=True)
        if self._plan.enabled("host_page_retention"):
            from .kernels.host_page_retention import install_host_policy

            # From here on this rank's CPU allocator keeps freed pages instead of
            # returning them to the OS at once, so host work during a decode step
            # does not invalidate the GPU's address translations.
            install_host_policy("after_warmup")

    def before_overlap_loop(self) -> None:
        # No fast path of this family acts before the overlap loop.
        return

    def at_shutdown(self) -> None:
        if self._plan.enabled("post_moe_hop"):
            from .kernels.post_moe_hop import check_health

            # End-of-serving report of this rank; rank-local and non-raising inside
            # cleanup (a device error is logged at ERROR).
            check_health("end_of_serving", raise_on_error=False)


def make_runtime_hooks(model: "GlmMoeDsaForCausalLM") -> RuntimeHooks:
    """The model's hooks: every engine-side fast path its plan runs.

    A kernel package is imported only when its fast path runs, as ``dispatch.get_impl``
    imports the packages of the model's own fast paths.
    """
    plan = model.plan
    decode_prologue = None
    if plan.enabled("decode_prologue"):
        from .kernels.decode_prologue import DecodePrologueHook

        # The decode target takes the embedding rows of the prologue's launch from the
        # same object (the core's ``decode_prologue``).
        decode_prologue = model.model.decode_prologue = DecodePrologueHook()
    return RuntimeHooks(
        loop_on_execution_stream=plan.enabled("execution_stream_loop"),
        decode_prologue=decode_prologue,
        executor_events=GlmExecutorEvents(plan),
    )
