# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""DwdpManager facade for VA-based DWDP.

This module owns three concerns and delegates the rest:
  * Lifecycle / global singleton (``__enter__`` / ``__exit__`` +
    ``set_global_dwdp_manager`` / ``get_global_dwdp_manager``)
  * DWDP MPI sub-communicator creation (``_create_dwdp_comm``)
  * Layer index registration (SSOT: ``add_layer`` → ``_registered_layers``)
    plus runtime entry points that forward to ``DWDPWeightManager``
    (``prefetch_first_layers``, ``wait_and_bind``,
    ``record_compute_and_prefetch_next``)

The heavy lifting — MNNVL handle allocation, composite VA layout,
double-buffer scheduling — lives in ``tensorrt_llm._torch.modules.dwdp``
and is wired up by ``setup_dwdp()`` during ``setup(model)``.
"""
from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn
from mpi4py.MPI import COMM_WORLD

from tensorrt_llm._torch.distributed import MPIDist
from tensorrt_llm._torch.modules.dwdp import DWDPWeightManager, setup_dwdp
from tensorrt_llm._utils import global_mpi_rank
from tensorrt_llm.llmapi.llm_args import DwdpConfig
from tensorrt_llm.logger import logger
from tensorrt_llm.mapping import Mapping


_global_dwdp_manager: Optional["DwdpManager"] = None


def set_global_dwdp_manager(manager: Optional["DwdpManager"]) -> None:
    global _global_dwdp_manager
    _global_dwdp_manager = manager


def get_global_dwdp_manager() -> Optional["DwdpManager"]:
    return _global_dwdp_manager


class DwdpManager:
    """Lifecycle facade that plumbs the VA-based DWDP pipeline.

    Construction creates the DWDP MPI sub-communicator. ``add_layer`` is the
    Single Source of Truth for MoE layer indices — ``configurable_moe``
    registers each MoE layer during model construction, and ``setup(model)``
    passes that list to ``setup_dwdp()`` so ``collect_moe_params`` never has
    to re-discover layers by walking the model tree.

    After ``setup()`` returns, the runtime methods (``prefetch_first_layers``,
    ``wait_and_bind``, ``record_compute_and_prefetch_next``) forward to the
    ``DWDPWeightManager`` instance produced by ``setup_dwdp``.
    """

    def __init__(
        self,
        config: DwdpConfig,
        dist: object,
        mapping: Mapping,
    ) -> None:
        if not isinstance(dist, MPIDist):
            raise RuntimeError("DWDP requires MPI backend (MPIDist)")
        if not mapping.dwdp_enabled:
            raise RuntimeError(
                f"DwdpManager requires mapping.dwdp_enabled (dwdp_size > 1); "
                f"got dwdp_size={mapping.dwdp_size}"
            )

        self.config = config
        self.dist = dist
        self._mapping = mapping
        self.dwdp_size = mapping.dwdp_size
        self.dwdp_rank = mapping.dwdp_rank
        self.num_experts_per_worker = config.num_experts_per_worker
        self.num_groups = config.num_groups
        self.num_prefetch_experts = config.num_prefetch_experts

        self.rank = global_mpi_rank()
        self.dwdp_group = self._create_dwdp_comm()

        # SSOT for MoE layer indices; populated by configurable_moe.add_layer()
        self._registered_layers: List[int] = []

        # Set by setup(model); None until then
        self._weight_manager: Optional[DWDPWeightManager] = None

    # ------------------------------------------------------------------
    # DWDP MPI group
    # ------------------------------------------------------------------

    def _create_dwdp_comm(self):
        """Create an MPI sub-communicator scoped to this rank's DWDP group.

        With num_groups=2, dwdp_size=4:
            Group 0: ranks [0, 1, 2, 3]
            Group 1: ranks [4, 5, 6, 7]
        """
        group_id = self.rank // self.dwdp_size
        group_start = group_id * self.dwdp_size
        ranks = list(range(group_start, group_start + self.dwdp_size))
        return COMM_WORLD.Create_group(COMM_WORLD.group.Incl(ranks))

    # ------------------------------------------------------------------
    # Global singleton lifecycle
    # ------------------------------------------------------------------

    def __enter__(self) -> "DwdpManager":
        if get_global_dwdp_manager() is not None:
            raise RuntimeError("DwdpManager already registered globally")
        set_global_dwdp_manager(self)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        self.cleanup()
        set_global_dwdp_manager(None)
        return False

    def is_enabled(self) -> bool:
        return self.dwdp_size > 1

    def cleanup(self) -> None:
        """Release MPI sub-communicator and reset weight manager. Idempotent."""
        self._weight_manager = None
        if self.dwdp_group is not None:
            self.dwdp_group.Free()
            self.dwdp_group = None

    # ------------------------------------------------------------------
    # Layer registration (SSOT) + setup
    # ------------------------------------------------------------------

    def add_layer(self, layer_idx: int) -> None:
        """Register an MoE layer index. Called from configurable_moe.__init__."""
        if layer_idx in self._registered_layers:
            logger.warning(
                f"[DwdpManager] Layer {layer_idx} already registered; ignoring"
            )
            return
        self._registered_layers.append(layer_idx)

    def setup(self, model: nn.Module) -> Optional[DWDPWeightManager]:
        """Run the VA initialization pipeline on ``model``.

        Delegates to ``setup_dwdp`` (transport + weight buffer + weight
        manager + MoE backend fixup). Stores the returned ``DWDPWeightManager``
        for subsequent runtime calls.

        Called once from ``py_executor_creator`` after weight loading.
        """
        layer_indices = sorted(self._registered_layers)
        device_id = torch.cuda.current_device()
        self._weight_manager = setup_dwdp(
            model=model,
            mapping=self._mapping,
            device_id=device_id,
            comm=self.dwdp_group,
            layer_indices=layer_indices,
        )
        return self._weight_manager

    # ------------------------------------------------------------------
    # Runtime entry points (forwarded to DWDPWeightManager)
    # ------------------------------------------------------------------

    def prefetch_first_layers(self) -> None:
        """Warm-up: kick off prefetch for the first MoE layers (double-buffer depth)."""
        if self._weight_manager is None:
            raise RuntimeError("DwdpManager.setup() has not been called yet")
        first = self._weight_manager.first_moe_layer()
        self._weight_manager.prefetch_layer(first)
        second = self._weight_manager.next_moe_layer(first)
        if second is not None:
            self._weight_manager.prefetch_layer(second)

    def wait_and_bind(self, backend_module: nn.Module, layer_idx: int) -> None:
        """Wait for prefetch and bind full-tensor views onto backend weights."""
        if self._weight_manager is None:
            raise RuntimeError("DwdpManager.setup() has not been called yet")
        self._weight_manager.wait_and_bind(backend_module, layer_idx)

    def record_compute_and_prefetch_next(self, layer_idx: int) -> None:
        """After the current layer finishes compute, kick off the next layer's prefetch.

        The WAR signal for the "consumed" slot is recorded inside
        ``wait_and_bind``; here we just schedule the next prefetch.
        """
        if self._weight_manager is None:
            raise RuntimeError("DwdpManager.setup() has not been called yet")
        next_idx = self._weight_manager.next_moe_layer(layer_idx)
        if next_idx is not None:
            self._weight_manager.prefetch_layer(next_idx)
