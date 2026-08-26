# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Per-denoising-step activation precision for static-FP8 Cosmos3.

A ModelOpt-calibrated checkpoint carries one activation scale per projection,
taken as a max over the whole sampling trajectory. That single scale fits the
outer denoising steps worst, so this module lets those steps run the resident
FP8 weights through a 16-bit GEMM instead: the weight is dequantized with its
own ``weight_scale`` and ``input_scale`` goes unused. Middle steps keep the
checkpoint's fully quantized path.

Nothing extra is read from the checkpoint -- the weights and scales are the
ones already loaded, and no second checkpoint or persistent dequantized copy
is kept. ``first_steps``/``last_steps`` are a runtime policy: the checkpoint
records no per-step information of any kind.

Precision is selected once per denoising step, before any transformer call for
that step, so the conditional and unconditional CFG branches of one step always
agree.
"""

from typing import Iterable, Optional

import torch
import torch.nn.functional as F

from tensorrt_llm._torch.modules.linear import FP8QDQLinearMethod, Linear
from tensorrt_llm.logger import logger


class StepPrecisionController:
    """Holds the activation precision selected for the current denoising step."""

    def __init__(self, first_steps: int, last_steps: int) -> None:
        if first_steps < 0 or last_steps < 0:
            raise ValueError(
                f"first_steps/last_steps must be non-negative, got {first_steps}/{last_steps}"
            )
        self.first_steps = first_steps
        self.last_steps = last_steps
        self.high_precision = False

    def set_step(self, step_index: int, num_steps: int) -> None:
        if num_steps <= 0:
            raise ValueError(f"num_steps must be positive, got {num_steps}")
        if step_index < 0 or step_index >= num_steps:
            raise IndexError(f"step_index must be in [0, {num_steps}), got {step_index}")
        # A single-step schedule is the warmup probe rather than a real
        # request, and treating every step as an edge step would make warmup
        # exercise a path the measured run never takes.
        if num_steps == 1:
            self.high_precision = False
            return
        self.high_precision = (
            step_index < self.first_steps or step_index >= num_steps - self.last_steps
        )

    def reset(self) -> None:
        self.high_precision = False


def apply_fp8_w8a16_linear(
    module: Linear, input: torch.Tensor, bias: Optional[torch.Tensor]
) -> torch.Tensor:
    """16-bit GEMM against the module's resident FP8 weight.

    ``FP8QDQLinearMethod.create_weights`` allocates ``weight`` as ``[out, in]``
    float8, which is the layout ``F.linear`` wants, so no transpose is needed.
    ``weight_scale`` is the per-tensor scalar and broadcasts. ``input_scale`` is
    deliberately unused -- leaving the activation unquantized is the point.
    """
    if input.dtype == torch.float8_e4m3fn:
        raise RuntimeError(
            "step precision: a high-precision step received an already-quantized "
            "activation. A caller that pre-quantizes a shared activation (fused "
            "gate/up or shared q/k/v) must stand down while high_precision is set, "
            "otherwise the step is not actually running in 16-bit."
        )
    weight = module.weight.to(input.dtype) * module.weight_scale.to(input.dtype)
    return F.linear(input, weight, bias)


class StepPrecisionFp8LinearMethod:
    """Dispatches each call to the checkpoint's FP8 path or the 16-bit path."""

    def __init__(self, base_method: FP8QDQLinearMethod, controller: StepPrecisionController):
        self.base_method = base_method
        self.controller = controller

    @property
    def high_precision(self) -> bool:
        """Published so activation-sharing callers can stand down for this step."""
        return self.controller.high_precision

    def __getattr__(self, name: str):
        # Everything not overridden here (create_weights, load_weights,
        # process_weights_after_loading, the quantizes_* properties Linear
        # queries) belongs to the wrapped method.
        return getattr(self.base_method, name)

    def apply(
        self, module: Linear, input: torch.Tensor, bias: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if self.controller.high_precision:
            return apply_fp8_w8a16_linear(module, input, bias)
        return self.base_method.apply(module, input, bias)


def linear_runs_high_precision(module: Optional[Linear]) -> bool:
    """Whether this Linear is currently on the 16-bit path.

    The contract a wrapped quantization method publishes: activation-sharing
    callers quantize once above the Linear, which would defeat the 16-bit step,
    so they consult this before doing so.
    """
    if module is None:
        return False
    return bool(getattr(module.quant_method, "high_precision", False))


def install_step_precision(
    roots: Iterable[torch.nn.Module], controller: StepPrecisionController
) -> int:
    """Wrap every static-FP8 Linear under *roots* for per-step dispatch.

    Must run after weight loading: the wrapper forwards the load-time hooks to
    the base method, but wrapping earlier would put it in the path of the
    loader's ``isinstance`` checks. Returns the number of wrapped modules; 0
    means this is not a static-FP8 model.
    """
    wrapped = 0
    for root in roots:
        for module in root.modules():
            if not isinstance(module, Linear):
                continue
            if not isinstance(module.quant_method, FP8QDQLinearMethod):
                continue
            if isinstance(module.quant_method, StepPrecisionFp8LinearMethod):
                continue
            module.quant_method = StepPrecisionFp8LinearMethod(module.quant_method, controller)
            wrapped += 1
    if wrapped:
        logger.info(
            f"Cosmos3 step precision: {wrapped} FP8 linears will run the first "
            f"{controller.first_steps} and last {controller.last_steps} denoising "
            "steps with BF16 activations."
        )
    return wrapped
