# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""BF16 Linear execution on selected diffusion iterations."""

from typing import Any

import torch

from tensorrt_llm._torch.modules.linear import (
    AllReduceParams,
    Linear,
    LinearMethodBase,
    UnquantizedLinearMethod,
)


class LinearQuantDispatcher:
    """Dispatch a loaded Linear while retaining its ordinary TP forward path.

    The original BF16 weight is a registered buffer on the Linear, so device
    transfers and component offloading move it together with quantized weights.
    Upstream activation-quantizing fusions must stand down for both phases.
    """

    requires_unquantized_activation = True
    quantizes_nvfp4_activations = False
    supports_nccl_symmetric_memory_window_output = False

    def __init__(self, quantized_method: LinearMethodBase, state: dict[str, Any]) -> None:
        self.quantized_method = quantized_method
        self.unquantized_method = UnquantizedLinearMethod()
        self.state = state

    def apply(
        self, module: Linear, input: torch.Tensor, bias: torch.Tensor | None, **kwargs: Any
    ) -> torch.Tensor:
        if self.state.get("linear_quant_ignored", False):
            return self.unquantized_method.apply(
                module, input, bias, weight=module._unquantized_weight
            )
        return self.quantized_method.apply(module, input, bias, **kwargs)

    def prepare_output_slice(self, module: Linear, start: int, end: int) -> None:
        self.quantized_method.prepare_output_slice(module, start, end)

    def apply_linear_allreduce_norm(
        self, module: Linear, input: torch.Tensor, all_reduce_params: AllReduceParams
    ) -> None:
        # Keep reductions on Linear.forward's ordinary path in both phases.
        return None

    def apply_slice(
        self, module: Linear, input: torch.Tensor, start: int, end: int
    ) -> torch.Tensor:
        if self.state.get("linear_quant_ignored", False):
            bias = module.bias[start:end] if module.bias is not None else None
            return self.unquantized_method.apply(
                module, input, bias, weight=module._unquantized_weight[start:end]
            )
        return self.quantized_method.apply_slice(module, input, start, end)


def install_linear_quant_dispatcher(model: torch.nn.Module, state: dict[str, Any]) -> None:
    """Install after native post-load transformations and before compilation."""
    for module in model.modules():
        if "_unquantized_weight" not in module._buffers:
            config = getattr(module, "quant_config", None)
            if (
                "weight" in module._parameters
                and hasattr(module, "quant_method")
                and getattr(config, "quant_algo", None) is not None
            ):
                raise ValueError(
                    "Linear ignored_steps requires retained BF16 weights for every quantized "
                    f"Linear; {type(module).__name__} was loaded without retained unquantized weights."
                )
            continue
        if not isinstance(module.quant_method, LinearQuantDispatcher):
            module.quant_method = LinearQuantDispatcher(module.quant_method, state)
            # The fused quantized GEMM+allreduce bypasses the ordinary method.apply.
            module.use_fused_gemm_allreduce = False
