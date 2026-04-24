# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""LoRA delta custom op and runtime planner for AutoDeploy.

The lora_delta op computes the LoRA correction for a single (layer, module) target:
    delta = lora_B @ lora_A @ x

It reads per-batch LoRA metadata from _GlobalLoraPlanner, which is populated
each forward pass by ADEngine._prepare_inputs from PeftCacheManager's PEFT table.

This follows the same planner pattern as _TrtllmPlanner (trtllm_attention.py) and
_GlobalFlashInferPlanner (flashinfer_attention.py) for managing host-side runtime state.
"""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch

# =============================================================================
# Global LoRA Planner — manages per-batch host-side LoRA state
# =============================================================================


@dataclass
class LoraModuleParams:
    """Per-(layer, module) LoRA parameters for the current batch."""

    lora_ranks: torch.Tensor  # [num_seqs] int32 — rank per request (0 = no LoRA)
    lora_weight_pointers: torch.Tensor  # [num_seqs, 3] int64 — (ptr_in, ptr_out, ptr_scale)


class _GlobalLoraPlanner:
    """Manages per-batch LoRA runtime state for lora_delta ops.

    All tensors are HOST-side (CPU). Populated each forward by _prepare_inputs.
    lora_delta ops read from here at runtime via get_params().
    """

    _instance: Optional["_GlobalLoraPlanner"] = None

    def __init__(self):
        self._params: Dict[Tuple[int, int], LoraModuleParams] = {}
        self._host_request_types: Optional[torch.Tensor] = None
        self._prompt_lens_cpu: Optional[torch.Tensor] = None
        self._max_rank: int = 0
        self._num_seqs: int = 0
        self._active: bool = False

    @classmethod
    def get(cls) -> "_GlobalLoraPlanner":
        """Get or create the global singleton."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def reset(cls):
        """Reset the global singleton (for testing)."""
        cls._instance = None

    def populate(
        self,
        peft_table: Optional[Dict],
        ordered_requests,
        num_seqs: int,
        num_context_requests: int,
        max_rank: int,
    ):
        """Build per-(layer, module) LoRA params from PeftCacheManager's PEFT table.

        Follows the same logic as PyTorch backend's _get_eager_lora_params_from_requests
        (model_engine.py:3496-3590).

        Args:
            peft_table: {task_id -> [TaskLayerModuleConfig]} from PeftCacheManager
            ordered_requests: list of LlmRequest in batch order (context first, then gen)
            num_seqs: number of sequences in this batch
            num_context_requests: number of context (prefill) requests at the front
            max_rank: maximum LoRA rank across all adapters
        """
        self._params.clear()
        self._num_seqs = num_seqs
        self._max_rank = max_rank

        if peft_table is None or num_seqs == 0:
            self._active = False
            return

        # Build host_request_types and prompt_lens_cpu
        # ordered_requests has context requests first, then generation requests
        host_request_types = torch.zeros(num_seqs, dtype=torch.int32)
        prompt_lens_cpu = torch.zeros(num_seqs, dtype=torch.int32)
        for i in range(num_seqs):
            if i < num_context_requests:
                host_request_types[i] = 0  # context
                request = ordered_requests[i]
                prompt_lens_cpu[i] = request.context_chunk_size
            else:
                host_request_types[i] = 1  # generation
                prompt_lens_cpu[i] = 1
        self._host_request_types = host_request_types
        self._prompt_lens_cpu = prompt_lens_cpu

        # First pass: discover all (layer_id, module_id) pairs from the PEFT table
        all_layer_modules = set()
        for task_id, configs in peft_table.items():
            for config in configs:
                all_layer_modules.add((config.layer_id, config.module_id))

        # Second pass: build per-request tensors for each (layer, module)
        for layer_id, module_id in all_layer_modules:
            ranks = torch.zeros(num_seqs, dtype=torch.int32)
            pointers = torch.zeros(num_seqs, 3, dtype=torch.int64)

            for i, request in enumerate(ordered_requests[:num_seqs]):
                task_id = request.lora_task_id
                if task_id is None or task_id not in peft_table:
                    continue

                for config in peft_table[task_id]:
                    if config.layer_id == layer_id and config.module_id == module_id:
                        ranks[i] = config.adapter_size
                        pointers[i, 0] = config.weights_in_pointer
                        pointers[i, 1] = config.weights_out_pointer
                        scaling_vec = getattr(config, "scaling_vec_pointer", None)
                        pointers[i, 2] = scaling_vec if scaling_vec is not None else 0
                        break

            self._params[(layer_id, module_id)] = LoraModuleParams(
                lora_ranks=ranks,
                lora_weight_pointers=pointers,
            )

        self._active = True

    def has_lora_for(self, layer_id: int, module_id: int) -> bool:
        """Check if any request in the batch has LoRA for this (layer, module)."""
        if not self._active:
            return False
        params = self._params.get((layer_id, module_id))
        if params is None:
            return False
        return params.lora_ranks.any().item()

    def get_params(self, layer_id: int, module_id: int) -> Optional[LoraModuleParams]:
        """Return LoRA params for a specific (layer, module)."""
        return self._params.get((layer_id, module_id))

    @property
    def host_request_types(self) -> Optional[torch.Tensor]:
        return self._host_request_types

    @property
    def prompt_lens_cpu(self) -> Optional[torch.Tensor]:
        return self._prompt_lens_cpu

    @property
    def max_rank(self) -> int:
        return self._max_rank

    @property
    def num_seqs(self) -> int:
        return self._num_seqs


# =============================================================================
# lora_delta custom op
# =============================================================================


@torch.library.custom_op("auto_deploy::lora_delta", mutates_args=())
def lora_delta(
    x: torch.Tensor,
    linear_out: torch.Tensor,
    layer_id: int,
    module_id: int,
) -> torch.Tensor:
    """Compute LoRA delta for one (layer, module) target.

    Args:
        x: Input tensor (same input as the base linear).
        linear_out: Output of the base linear. Used only for shape — the actual
            computation reads weights from _GlobalLoraPlanner. Passing this ensures
            the fake implementation returns a tensor with matching symbolic shape.
        layer_id: Compile-time constant identifying the layer.
        module_id: Compile-time constant identifying the module (ATTENTION_Q, etc.).

    Runtime LoRA state (ranks, weight pointers, request types, prompt lengths)
    is read from _GlobalLoraPlanner, which is populated each forward by _prepare_inputs.
    """
    assert x.dtype in (torch.float16, torch.bfloat16), (
        f"lora_delta requires FP16/BF16 input, got {x.dtype}. "
        "An upstream fusion may have converted the input to FP8."
    )

    output_size = linear_out.shape[-1]

    planner = _GlobalLoraPlanner.get()
    if not planner.has_lora_for(layer_id, module_id):
        return torch.zeros_like(linear_out)

    params = planner.get_params(layer_id, module_id)

    # lora_grouped_gemm expects 2D [num_tokens, hidden] input with remove_input_padding=True.
    # AD graph may have 3D [batch, seq, hidden] — always flatten to 2D before calling kernel.
    orig_shape = x.shape
    x_2d = x.reshape(-1, x.shape[-1]) if x.dim() > 2 else x

    result = torch.ops.trtllm.lora_grouped_gemm(
        x_2d,
        planner.host_request_types[: planner.num_seqs],
        [params.lora_ranks],
        [params.lora_weight_pointers],
        planner.prompt_lens_cpu[: planner.num_seqs],
        [output_size],
        False,  # transA
        True,  # transB
        planner.max_rank,
        0,  # weight_index
        True,  # remove_input_padding — always True after 2D reshape
    )

    # lora_grouped_gemm returns a list for multi-module; we use single-module
    out = result if isinstance(result, torch.Tensor) else result[0]

    # Restore original batch dimensions if input was 3D
    if x.dim() > 2:
        out = out.reshape(*orig_shape[:-1], output_size)
    return out


@lora_delta.register_fake
def lora_delta_fake(
    x: torch.Tensor,
    linear_out: torch.Tensor,
    layer_id: int,
    module_id: int,
) -> torch.Tensor:
    """Fake implementation for torch.export tracing.

    Returns a tensor with the same shape as linear_out, ensuring symbolic
    shapes match for the downstream add(linear_out, lora_delta_out).
    """
    return torch.empty_like(linear_out)
