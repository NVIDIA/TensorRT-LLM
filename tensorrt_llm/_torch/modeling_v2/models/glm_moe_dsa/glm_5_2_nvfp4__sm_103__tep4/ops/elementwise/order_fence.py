# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""`order_fence`: one plain (non-PDL) kernel on the current stream, a 1-element fill, so the
next kernel on that stream no longer hangs off the previous producer by a programmatic edge.

The target issues it on the side stream right after a wide decode step's MoE fork, ahead of
the shared expert. In the captured decode graph both the router (main stream) and the shared
expert (side stream) are the pre-MoE collective's PDL dependents, and only one of them
launches at its early release: at 192 rows the shared up GEMM took it and the router started
only when the collective ended; at 128 rows both launched before it ended and the shared GEMM
slowed the router (2-3x). Behind this node the router always takes the early launch and the
shared expert starts once the collective completes, in the shadow of the top-k and FC1 (it
only has to finish before FC2). 4-GPU decode-step chain (real AR, 3 MoE layers per step, 280
alternating replays): -30.6 us per step at 128 rows, -17.9 us at 192. Ordering only: every
byte is the same.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from tensorrt_llm._torch.modeling_v2.catalog._op import OpWrapper

from .._build import Count


@dataclass(frozen=True)
class OrderFenceHandle:
    """The fence's one persistent element (only the launch matters, so one serves every call)
    and the route it is counted as."""

    buf: torch.Tensor
    route: str
    count: Count


def build(device: int, route: str, count: Count) -> OrderFenceHandle:
    """Allocate the fence's element on `device`; at load, as a capture cannot allocate it."""
    return OrderFenceHandle(torch.zeros(1, dtype=torch.int32, device=device), route, count)


class OrderFence(OpWrapper):
    """Launch the fence where `launch` (a step's own fact: a wide decode step's exact rows)
    asks for it, counted as the handle's route; launch nothing otherwise."""

    def __call__(self, launch: bool, handle: OrderFenceHandle) -> None:
        if not launch:
            return
        handle.buf.zero_()
        handle.count(handle.route, True)

    def reference(self, launch: bool, handle: OrderFenceHandle) -> None:
        """Nothing: the fence computes nothing a caller reads."""
        return None


order_fence = OrderFence()
