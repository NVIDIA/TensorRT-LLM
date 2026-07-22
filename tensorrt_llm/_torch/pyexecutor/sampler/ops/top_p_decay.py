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

"""Top-P Decay ops for TorchSampler, fused via ``torch.compile``.

These two ops sit on a host-launch-bound path (per-step work of a few dozen
elements per row), so the eager op sequences are fused with Inductor to keep
the per-step launch count low; the generated Triton kernels match the GPU-side
cost of hand-written fused kernels once the model forward is long enough to
hide the compiled-function entry.

Compile configuration (benchmarked against eager and other modes on H200):

- ``mode="max-autotune-no-cudagraphs"``: CUDA graphs are deliberately
  disabled -- the update mutates persistent per-slot state in place and the
  gather's output is consumed outside the compiled region, both of which are
  unsafe with cudagraph static buffers (outputs get overwritten by subsequent
  replays).
- ``torch._dynamo.mark_dynamic`` on the batch-varying dimensions, so per-step
  changes in batch composition do not trigger recompilation while the
  remaining dimensions stay specialized.

Compilation happens lazily on the first decay-active request (roughly a
second); non-decay workloads never trigger it.
"""

import torch


@torch.compile(mode="max-autotune-no-cudagraphs")
def _top_p_decay_update_impl(
    runtime_top_p: torch.Tensor,
    initial_top_p: torch.Tensor,
    top_p_decay: torch.Tensor,
    top_p_min: torch.Tensor,
    reset_ids: torch.Tensor,
    is_decay_slot: torch.Tensor,
    step_tokens: torch.Tensor,
    sampled_slots: torch.Tensor,
) -> None:
    active = is_decay_slot[sampled_slots]
    current = runtime_top_p[sampled_slots]
    updated = torch.where(
        step_tokens[sampled_slots] == reset_ids[sampled_slots],
        initial_top_p[sampled_slots],
        torch.maximum(current * top_p_decay[sampled_slots], top_p_min[sampled_slots]),
    )
    runtime_top_p[sampled_slots] = torch.where(active, updated, current)


@torch.compile(mode="max-autotune-no-cudagraphs")
def _top_p_decay_gather_impl(
    runtime_top_p: torch.Tensor,
    is_decay_slot: torch.Tensor,
    static_top_p: torch.Tensor,
    slots: torch.Tensor,
) -> torch.Tensor:
    return torch.where(is_decay_slot[slots], runtime_top_p[slots], static_top_p)


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

    Applies the Top-P Decay recurrence (see ``TorchSampler.TopPDecayStore`` for
    the feature-level semantics) to every sampled row whose slot is
    decay-active per ``is_decay_slot``.

    All per-slot tensors are 1-D of length ``max_num_sequences``;
    ``step_tokens`` is a slot-indexed 1-D (possibly strided) int32 view of the
    new-tokens buffer for a fixed step/beam (``new_tokens[step, :, beam]``);
    ``sampled_slots`` is 1-D of length ``num_sampled`` (this iteration's rows).
    ``runtime_top_p`` is mutated in place; nothing is returned.
    """
    torch._dynamo.mark_dynamic(sampled_slots, 0)
    _top_p_decay_update_impl(
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

    ``runtime_top_p`` / ``is_decay_slot`` are per-slot arrays; ``static_top_p``
    and ``slots`` are per-row (length = the group's per-step row count).
    """
    torch._dynamo.mark_dynamic(slots, 0)
    torch._dynamo.mark_dynamic(static_top_p, 0)
    return _top_p_decay_gather_impl(runtime_top_p, is_decay_slot, static_top_p, slots)
