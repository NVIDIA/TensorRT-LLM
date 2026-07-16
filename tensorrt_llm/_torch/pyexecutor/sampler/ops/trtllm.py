# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Thin Python wrappers over the trtllm C++ sampling ops used by TorchSampler.

These forward to ``torch.ops.trtllm.*`` custom ops (registered from
``cpp/tensorrt_llm/thop``); the wrappers exist to give the sampler a stable,
keyword-only Python surface decoupled from the raw op-registration names.
"""

import torch


def top_p_decay_update(
    *,
    runtime_top_p: torch.Tensor,
    initial_top_p: torch.Tensor,
    top_p_decay: torch.Tensor,
    top_p_min: torch.Tensor,
    reset_ids: torch.Tensor,
    is_decay_slot: torch.Tensor,
    step_tokens: torch.Tensor,
    sampled_slots: torch.Tensor,
) -> None:
    """Fused in-place update of ``runtime_top_p`` for the sampled decay slots.

    For each sampled row whose slot is decay-active (per ``is_decay_slot``)::

        runtime_top_p[slot] = initial_top_p[slot]                         if reset_id >= 0 and token == reset_id
                            = max(runtime_top_p[slot] * decay, top_p_min)  otherwise

    All per-slot tensors are 1-D of length ``max_num_sequences``;
    ``step_tokens`` is a slot-indexed 1-D (possibly strided) int32 view of the
    new-tokens buffer for a fixed step/beam (``new_tokens[step, :, beam]``) --
    the kernel gathers each sampled token in-kernel; ``sampled_slots`` is 1-D of
    length ``num_sampled`` (this iteration's rows). ``runtime_top_p`` is mutated
    in place; nothing is returned.
    """
    torch.ops.trtllm.top_p_decay_update(
        runtime_top_p,
        initial_top_p,
        top_p_decay,
        top_p_min,
        reset_ids,
        is_decay_slot,
        step_tokens,
        sampled_slots,
    )


def top_p_decay_gather(
    *,
    runtime_top_p: torch.Tensor,
    is_decay_slot: torch.Tensor,
    static_top_p: torch.Tensor,
    slots: torch.Tensor,
) -> torch.Tensor:
    """Fused pre-sample per-row top-p gather for decay-active rows.

    Returns a new per-row tensor::

        row_top_p[i] = runtime_top_p[slots[i]]  if is_decay_slot[slots[i]]
                     = static_top_p[i]          otherwise

    replacing the eager ``index_select`` x2 + ``where`` chain with one launch.
    """
    return torch.ops.trtllm.top_p_decay_gather(runtime_top_p, is_decay_slot, static_top_p, slots)
