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

"""Lowered parameters for two-stage VisualGen SOL attention."""

from __future__ import annotations

import math
import numbers
import struct
from dataclasses import dataclass, field
from typing import Literal

import torch

from tensorrt_llm._torch.attention.backends.sparse.params import SparseParams


def _as_timestep_float(timestep: object) -> float | None:
    if timestep is None:
        return None
    if isinstance(timestep, torch.Tensor):
        if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
            raise RuntimeError("SOL graph phase must be precomputed before CUDA Graph capture")
        if timestep.numel() == 0:
            return None
        # WAN I2V can carry one timestep per token, with reference tokens fixed
        # at zero. Stay dense until every live token is below the cutoff.
        timestep = timestep.amax().item()
    if isinstance(timestep, bool) or not isinstance(timestep, numbers.Real):
        raise TypeError("timestep must be a real scalar or tensor")
    value = float(timestep)
    if not math.isfinite(value):
        raise ValueError("timestep must be finite")
    return value


@dataclass(frozen=True, slots=True)
class SolParams(SparseParams):
    """Static SOL policy lowered from the user-facing VisualGen config."""

    algorithm: Literal["sol_attn"] = field(init=False, default="sol_attn")
    tau: float = 1.0
    disabled_until_timestep: float | None = None
    dense_layers: frozenset[int] = field(default_factory=frozenset)

    def __post_init__(self) -> None:
        if isinstance(self.tau, bool) or not isinstance(self.tau, numbers.Real):
            raise TypeError("tau must be a finite real number")
        try:
            tau = struct.unpack("=f", struct.pack("=f", float(self.tau)))[0]
        except (OverflowError, ValueError, struct.error) as error:
            raise ValueError("tau must be representable as float32") from error
        if not math.isfinite(tau):
            raise ValueError("tau must be finite")
        object.__setattr__(self, "tau", tau)

        cutoff = self.disabled_until_timestep
        if cutoff is not None:
            if isinstance(cutoff, bool) or not isinstance(cutoff, numbers.Real):
                raise TypeError("disabled_until_timestep must be a real number or None")
            cutoff = float(cutoff)
            if not math.isfinite(cutoff) or not 0.0 < cutoff <= 1.0:
                raise ValueError("disabled_until_timestep must be in (0, 1]")
            object.__setattr__(self, "disabled_until_timestep", cutoff)

        dense_layers = frozenset(self.dense_layers)
        if any(
            isinstance(layer, bool) or not isinstance(layer, int) or layer < 0
            for layer in dense_layers
        ):
            raise ValueError("dense_layers must contain only non-negative integers")
        object.__setattr__(self, "dense_layers", dense_layers)

    @staticmethod
    def get_graph_phase_for_timestep(
        timestep: object,
        *,
        disabled_until_timestep: float | None,
    ) -> int | None:
        """Return 0 for the dense prefix and 1 for the sparse suffix."""

        if disabled_until_timestep is None:
            return None
        value = _as_timestep_float(timestep)
        if value is None:
            return None
        return int(value < disabled_until_timestep)

    def should_use_sparse(
        self,
        *,
        layer_idx: int,
        timestep: object,
        graph_phase: int | None = None,
    ) -> bool:
        """Return whether this layer should execute the SOL sparse path."""

        if layer_idx in self.dense_layers:
            return False
        if graph_phase is not None:
            if graph_phase not in (0, 1):
                raise ValueError("SOL graph_phase must be 0 or 1")
            phase = graph_phase
        else:
            phase = self.get_graph_phase_for_timestep(
                timestep,
                disabled_until_timestep=self.disabled_until_timestep,
            )
        if phase is None:
            if self.disabled_until_timestep is not None:
                raise ValueError(
                    "timestep is required when SOL disabled_until_timestep is configured"
                )
            return True
        return phase == 1


__all__ = ["SolParams"]
