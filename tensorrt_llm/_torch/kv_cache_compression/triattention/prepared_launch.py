# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Replayable Triton kernel launches frozen at build time."""

from typing import Dict, Tuple

import torch


class PreparedTritonKernelLaunch:
    """Replay one Triton kernel launch frozen at build time.

    ``warmup`` JIT-compiles the kernel once for a fixed grid, bound tensor
    set, and constexpr set; ``__call__`` re-launches the compiled binary
    directly, skipping Triton's per-call dispatch. Constexpr values are
    passed positionally on replay, so their order is validated against the
    kernel's constexpr parameter declaration order at build time.
    """

    def __init__(
        self,
        triton_kernel,
        bound_tensors: Tuple[torch.Tensor, ...],
        constexpr_values: Dict[str, object],
        *,
        grid: Tuple[int, ...],
        num_warps: int,
    ) -> None:
        params = getattr(triton_kernel, "params", None)
        if params is not None:
            declared = [param.name for param in params if param.is_constexpr]
            if list(constexpr_values.keys()) != declared:
                raise ValueError(
                    f"constexpr order {list(constexpr_values.keys())} must match the "
                    f"kernel's declaration order {declared}: replay passes them "
                    "positionally"
                )
        self.device = bound_tensors[0].device
        self.bound_tensors = tuple(bound_tensors)
        self.constexpr_values = dict(constexpr_values)
        with torch.cuda.device(self.device):
            self.build_stream = torch.cuda.current_stream(self.device)
            # warmup() then indexing the compiled cache by grid is the
            # documented-by-use Triton pattern for dispatch-free replay; if a
            # Triton upgrade changes it, this raises here at build time rather
            # than corrupting a replay.
            compiled = triton_kernel.warmup(
                *self.bound_tensors,
                **self.constexpr_values,
                num_warps=num_warps,
                grid=grid,
            )
            self.compiled_kernel_runner = compiled[grid]

    def __call__(self, *replay_tensors: torch.Tensor) -> None:
        """Replay the launch; ``replay_tensors``, if given, substitute the bound tensors."""
        current_stream = torch.cuda.current_stream(self.device)
        if (current_stream.device, current_stream.cuda_stream) != (
            self.build_stream.device,
            self.build_stream.cuda_stream,
        ):
            raise RuntimeError(
                "a prepared Triton kernel launch must run on the stream it was built on"
            )
        self.compiled_kernel_runner(
            *(replay_tensors if replay_tensors else self.bound_tensors),
            *self.constexpr_values.values(),
            stream=self.build_stream.cuda_stream,
        )
