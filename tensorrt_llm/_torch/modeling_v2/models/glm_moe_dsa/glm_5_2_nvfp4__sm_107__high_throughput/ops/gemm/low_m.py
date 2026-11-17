# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Model-owned BF16 GEMM dispatch for the DEP16 decode and MTP shapes."""

from functools import lru_cache

import torch

from tensorrt_llm._torch.flashinfer_utils import get_env_enable_pdl
from tensorrt_llm._torch.modules.embedding import LMHead
from tensorrt_llm._torch.modules.linear import Linear, UnquantizedLinearMethod


@lru_cache(maxsize=None)
def _device_limits(device: torch.device) -> tuple[int, int]:
    props = torch.cuda.get_device_properties(device)
    return props.multi_processor_count, props.shared_memory_per_block_optin


@torch.library.custom_op(
    "trtllm::glm_rubin_high_throughput_low_m", mutates_args=(), device_types="cuda"
)
def low_m_gemm(
    inputs: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    narrow: bool = False,
    one_wave: bool = False,
    release_early: bool = False,
) -> torch.Tensor:
    from ...kernels.gemm.splitk import (
        narrow_tactic,
        one_wave_tactic,
        run_splitk_dense,
        splitk_default_tactic,
        wide_tuned_launch,
    )

    matrix = inputs.view(-1, inputs.shape[-1])
    m, k = matrix.shape
    n = weight.shape[0]
    sm_count, smem_capacity = _device_limits(inputs.device)
    options = {}
    if m <= 32:
        tactic = splitk_default_tactic(m, n, k, sm_count=sm_count, smem_capacity=smem_capacity)
        if one_wave:
            tactic = one_wave_tactic(tactic, m, n, k, sm_count=sm_count)
    elif narrow:
        tactic = narrow_tactic(m, n, k, smem_capacity=smem_capacity)
        options["reserve_full_smem"] = False
    else:
        launch = wide_tuned_launch(m, n, k, sm_count=sm_count, smem_capacity=smem_capacity)
        tactic = launch.tactic
        options = launch.run_kwargs()
        if release_early:
            options["launch_dependents_early"] = True
    output = torch.empty((m, n), dtype=inputs.dtype, device=inputs.device)
    run_splitk_dense(matrix, weight.t(), bias, output, get_env_enable_pdl(), tactic, **options)
    return output.view(*inputs.shape[:-1], n)


@low_m_gemm.register_fake
def _fake_low_m_gemm(
    inputs: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    narrow: bool = False,
    one_wave: bool = False,
    release_early: bool = False,
) -> torch.Tensor:
    return inputs.new_empty((*inputs.shape[:-1], weight.shape[0]))


class RubinUnquantizedLinearMethod(UnquantizedLinearMethod):
    def __init__(self, max_rows: int = 32, narrow: bool = False, one_wave: bool = False) -> None:
        self.max_rows = max_rows
        self.narrow = narrow
        self.one_wave = one_wave

    def apply(
        self,
        module: Linear,
        input: torch.Tensor,
        bias: torch.Tensor | None,
        weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        weight = module.weight if weight is None else weight
        rows = input.numel() // input.shape[-1]
        if (
            input.dtype == weight.dtype == torch.bfloat16
            and input.is_cuda
            and input.is_contiguous()
            and weight.is_contiguous()
            and input.shape[-1] == weight.shape[-1]
            and input.shape[-1] % 128 == 0
            and weight.shape[0] % 8 == 0
            and 0 < rows <= self.max_rows
            and (bias is None or bias.dtype == input.dtype)
            and not torch.is_grad_enabled()
            and (
                module.all_reduce is None
                or not module.all_reduce.uses_nccl_symmetric_memory_window()
            )
        ):
            return low_m_gemm(input, weight, bias, self.narrow, self.one_wave)
        return super().apply(module, input, bias, weight)


def install_low_m_methods(model: torch.nn.Module) -> None:
    for name, module in model.named_modules():
        if (
            not isinstance(module, Linear)
            or type(module.quant_method) is not UnquantizedLinearMethod
        ):
            continue
        wide = name.endswith(("o_proj", "eh_proj")) or (
            ".mlp." in name and ".experts." not in name and ".shared_experts." not in name
        )
        narrow = name.endswith("shared_experts.down_proj")
        module.quant_method = RubinUnquantizedLinearMethod(
            64 if wide or narrow else 32, narrow, isinstance(module, LMHead)
        )
