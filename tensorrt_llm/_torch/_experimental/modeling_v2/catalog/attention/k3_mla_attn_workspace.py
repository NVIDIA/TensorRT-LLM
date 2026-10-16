# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Caller-owned state of Kimi K3's MLA decode attention: the per-CTA partial slots and the no-cluster mode's arrival
counters of every ``attention/k3_mla_attn_vb_out`` call (and its ``k3_mla_attn`` / ``k3_mla_attn_out`` forms).

A state type, not an entry: it launches nothing per call. Its constructor is eager; the target builds one per device
and head-group count in ``post_load_weights`` (before any CUDA-graph capture) and passes it to every MLA attention
call of that shape. The contract is the ``## State`` section of ``k3_mla_attn_vb_out.md``.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(eq=False)
class K3MlaAttnWorkspace:
    """One device's attention workspace for calls of ``groups`` head groups (the rank's heads / 6): fp16 partial
    slots for 8 requests x ``groups`` x 16 CTAs, then the no-cluster mode's fp32 (m, l) exchange and its int32
    arrival counters, which only grow. Calls on one workspace must run one at a time (see the contract's
    ``## State``)."""

    buffer: torch.Tensor
    """fp16 [attn_workspace_elems(groups)]: the words the op's ``workspace`` argument names."""
    groups: int

    @classmethod
    def create(cls, device: torch.device, groups: int) -> "K3MlaAttnWorkspace":
        """Allocate and arm a workspace for ``groups`` head groups on ``device``: the counters zeroed, the partial
        slots left as allocated (a call reads only the words it wrote). Eager: it allocates, so it refuses to run
        under CUDA-graph capture; it returns once the arming is complete, so the workspace is ready on any stream."""
        from tensorrt_llm._torch.cute_dsl_kernels.k3_mla import op

        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("K3MlaAttnWorkspace.create allocates: call it before capture")
        device = torch.device(device)
        buffer = op.make_attn_workspace(device, groups)
        torch.cuda.synchronize(device)
        return cls(buffer=buffer, groups=groups)

    @property
    def device(self) -> torch.device:
        return self.buffer.device
