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
  * DWDP group creation (``_create_dwdp_comm``): an MPI sub-communicator, or
    a TCPStore rendezvous for independently launched workers
  * Layer index registration (SSOT: ``add_layer`` → ``_registered_layers``)
    plus runtime entry points that forward to ``DWDPWeightManager``
    (``start_forward``, ``wait_and_bind``,
    ``finish_layer``, ``invalidate_prefetch``)

The heavy lifting — MNNVL handle allocation, composite VA layout,
prefetch scheduling — lives in ``tensorrt_llm._torch.modules.dwdp``
and is wired up by ``setup_dwdp()`` during ``setup(model)``.
"""

from __future__ import annotations

import weakref
from typing import List, Optional

import torch
import torch.nn as nn
from mpi4py.MPI import COMM_WORLD

from tensorrt_llm._torch.cute_dsl_utils import set_sm_reserve
from tensorrt_llm._torch.distributed import MPIDist
from tensorrt_llm._torch.modules.dwdp import (
    DWDPWeightManager,
    StoreDwdpGroup,
    dwdp_global_rank,
    fabric_gather,
    setup_dwdp,
    store_rendezvous_enabled,
    teardown_dwdp,
)
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

    Construction creates the DWDP group: an MPI sub-communicator, or a
    ``StoreDwdpGroup`` when the context workers are launched independently
    (see ``modules.dwdp.rendezvous``). ``add_layer`` is the
    Single Source of Truth for MoE layer indices — ``configurable_moe``
    registers each MoE layer during model construction, and ``setup(model)``
    passes that list to ``setup_dwdp()`` so ``collect_moe_params`` never has
    to re-discover layers by walking the model tree.

    After ``setup()`` returns, the runtime methods (``start_forward``,
    ``wait_and_bind``, ``finish_layer``) forward to the
    ``DWDPWeightManager`` instance produced by ``setup_dwdp``.
    """

    def __init__(
        self,
        config: DwdpConfig,
        dist: object,
        mapping: Mapping,
    ) -> None:
        if get_global_dwdp_manager() is not None:
            raise RuntimeError("DwdpManager already registered globally")
        # Independently launched workers each have their own MPI world, so the
        # group rendezvous over a TCPStore instead of COMM_WORLD.
        self._store_rendezvous = store_rendezvous_enabled()
        if not self._store_rendezvous and not isinstance(dist, MPIDist):
            raise RuntimeError(
                "DWDP requires the MPI backend (MPIDist), or a store "
                "rendezvous: set TRTLLM_DWDP_RANK and TRTLLM_DWDP_MASTER_ADDR "
                "when the context workers are launched independently"
            )
        if not mapping.dwdp_enabled:
            raise RuntimeError(
                f"DwdpManager requires mapping.dwdp_enabled (dwdp_size > 1); "
                f"got dwdp_size={mapping.dwdp_size}"
            )

        # Validate DwdpConfig topology against the actual launch.
        # `num_groups` declares the user's intended topology — that the launch
        # contains `num_groups × dwdp_size` CTX worker ranks, partitioned into
        # `num_groups` independent DWDP groups of `dwdp_size` ranks each.
        # Without this validation `num_groups` is a dead schema field, which
        # is how it was treated up through commit `1fbc0d49`.
        if config.num_groups <= 0:
            raise ValueError(f"DwdpConfig.num_groups must be positive, got {config.num_groups}")
        self.rank = dwdp_global_rank()
        group_id = self.rank // mapping.dwdp_size
        if group_id >= config.num_groups:
            raise ValueError(
                f"DwdpManager rank {self.rank} computes group_id={group_id} "
                f"(rank // dwdp_size={mapping.dwdp_size}) which is >= "
                f"DwdpConfig.num_groups ({config.num_groups}). The launch "
                f"started more CTX workers than fit in {config.num_groups} × "
                f"{mapping.dwdp_size} = {config.num_groups * mapping.dwdp_size} "
                f"ranks, or num_groups in the YAML is set too small."
            )
        expected_min_world = config.num_groups * mapping.dwdp_size
        # Under a store rendezvous COMM_WORLD spans only this worker and says
        # nothing about the launch; the group_id check above bounds the rank,
        # and a missing peer surfaces as the first store allgather timing out.
        actual_world = None if self._store_rendezvous else COMM_WORLD.Get_size()
        if actual_world is not None and actual_world < expected_min_world:
            raise ValueError(
                f"DwdpConfig declares {config.num_groups} groups × dwdp_size "
                f"{mapping.dwdp_size} = {expected_min_world} CTX workers, but "
                f"MPI world size is only {actual_world}. Either the launch's "
                f"--ntasks is too small, or num_groups is set too high."
            )

        self.config = config
        self.dist = dist
        self._mapping = mapping
        self.dwdp_size = mapping.dwdp_size
        self.dwdp_rank = mapping.dwdp_rank
        self.num_experts_per_worker = config.num_experts_per_worker
        self.num_groups = config.num_groups
        self.num_prefetch_experts = config.num_prefetch_experts

        # Per-rank expert range. ``num_prefetch_experts`` is the rank-to-rank
        # stride and ``num_experts_per_worker`` is the (uniform across ranks)
        # storage chunk size. When the user requests redundancy
        # (``num_prefetch_experts < num_experts_per_worker``) consecutive
        # ranks' ranges overlap; when the partition is non-uniform (e.g.
        # dwdp_size = 3 over 256 experts) the tail rank's
        # ``[start, end)`` runs past ``num_experts`` and is later capped at
        # weight-load time. ``_init_dwdp_expert_layout`` reads
        # ``start_expert_id`` to install the correct ``slot_start`` /
        # ``slot_end`` on every fused MoE backend before ``create_weights``.
        self.start_expert_id = self.dwdp_rank * self.num_prefetch_experts
        self.end_expert_id = self.start_expert_id + self.num_experts_per_worker

        # SMs the expert gather holds for the whole step; persistent grids leave
        # them out (``set_sm_reserve`` on ``__enter__``, before the model is built).
        self.resident_sms = fabric_gather.resident_sms()

        self.dwdp_group = self._create_dwdp_comm()

        # SSOT for MoE layer indices; populated by configurable_moe.add_layer()
        self._registered_layers: List[int] = []
        # Layers that run several times in a row per pass; see add_repeated_layer()
        self._repeated_layers: List[int] = []

        # Set by setup(model); None until then
        self._weight_manager: Optional[DWDPWeightManager] = None
        self._model_ref: Optional[weakref.ReferenceType[nn.Module]] = None

    # ------------------------------------------------------------------
    # DWDP group
    # ------------------------------------------------------------------

    def _create_dwdp_comm(self):
        """Create the group handle scoped to this rank's DWDP group.

        With num_groups=2, dwdp_size=4:
            Group 0: ranks [0, 1, 2, 3]
            Group 1: ranks [4, 5, 6, 7]

        Returns an MPI sub-communicator, or a ``StoreDwdpGroup`` under a store
        rendezvous; both expose ``allgather`` / ``Barrier`` / ``Free``.
        """
        if self._store_rendezvous:
            return StoreDwdpGroup.from_env(
                global_rank=self.rank,
                dwdp_size=self.dwdp_size,
                num_groups=self.config.num_groups,
            )

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
        set_sm_reserve(self.resident_sms)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        try:
            self.cleanup()
        finally:
            set_sm_reserve(0)
            set_global_dwdp_manager(None)
        return False

    def is_enabled(self) -> bool:
        return self.dwdp_size > 1

    def cleanup(self) -> None:
        """Release the DWDP group (MPI sub-communicator or TCPStore) and reset
        the weight manager. Idempotent."""
        cleanup_error = None
        cleanup_traceback = None
        try:
            model = self._model_ref() if self._model_ref is not None else None
            if model is not None and self._weight_manager is not None:
                teardown_dwdp(model, expected_manager=self._weight_manager)
            elif self._weight_manager is not None:
                self._weight_manager.release()
        except BaseException as exc:
            cleanup_error = exc
            cleanup_traceback = exc.__traceback__
        finally:
            self._weight_manager = None
            self._model_ref = None
            if self.dwdp_group is not None:
                try:
                    self.dwdp_group.Free()
                except BaseException:
                    if cleanup_error is None:
                        raise
                    logger.exception("Failed to free DWDP group after runtime cleanup error")
                finally:
                    self.dwdp_group = None

        if cleanup_error is not None:
            raise cleanup_error.with_traceback(cleanup_traceback)

    # ------------------------------------------------------------------
    # Layer registration (SSOT) + setup
    # ------------------------------------------------------------------

    def add_layer(self, layer_idx: int) -> None:
        """Register an MoE layer index. Called from configurable_moe.__init__."""
        if layer_idx in self._registered_layers:
            logger.warning(f"[DwdpManager] Layer {layer_idx} already registered; ignoring")
            return
        self._registered_layers.append(layer_idx)

    def add_repeated_layer(self, layer_idx: int) -> None:
        """Declare that layer ``layer_idx`` runs several times in a row per forward pass.

        Called by models that reuse a decoder layer, e.g. the one-model MTP-Eagle layer that runs
        once per draft step. Its repeats bind the expert copy still in the prefetch slot instead
        of gathering the layer again. Layers whose MoE does not use DWDP are ignored.
        """
        if layer_idx not in self._repeated_layers:
            self._repeated_layers.append(layer_idx)

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
            num_experts_per_worker=self.num_experts_per_worker,
            num_prefetch_experts=self.num_prefetch_experts,
            contention_opt=self.config.contention_opt,
            num_prefetch_slots=self.config.num_prefetch_slots,
            repeated_layer_indices=[
                layer for layer in self._repeated_layers if layer in self._registered_layers
            ],
        )
        self._model_ref = weakref.ref(model)
        if self._store_rendezvous:
            # Lets DWDP rank 0, which serves the store, release it once every group is set up.
            self.dwdp_group.mark_setup_done()
        return self._weight_manager

    # ------------------------------------------------------------------
    # Runtime entry points (forwarded to DWDPWeightManager)
    # ------------------------------------------------------------------

    def start_forward(self) -> None:
        """Start a forward pass: prefetch its first MoE layers.

        Enqueues nothing when the previous forward pass already prefetched
        them (its last MoE layers gather the first layers of the next pass).
        """
        if self._weight_manager is None:
            raise RuntimeError("DwdpManager.setup() has not been called yet")
        self._weight_manager.start_forward()

    def wait_and_bind(self, backend_module: nn.Module, layer_idx: int) -> None:
        """Wait for prefetch and bind full-tensor views onto backend weights."""
        if self._weight_manager is None:
            raise RuntimeError("DwdpManager.setup() has not been called yet")
        self._weight_manager.wait_and_bind(backend_module, layer_idx)

    def finish_layer(self, layer_idx: int) -> None:
        """Release the layer's buffer slot and prefetch the layers it unblocks.

        Called after the layer's MoE kernel has been enqueued, so the slot's
        consume event covers that kernel.
        """
        if self._weight_manager is None:
            raise RuntimeError("DwdpManager.setup() has not been called yet")
        self._weight_manager.finish_layer(layer_idx)

    def invalidate_prefetch(self) -> None:
        """Drop the prefetches issued ahead of the next forward pass.

        Call after the expert weights may have changed (e.g. a weight update);
        the next forward pass gathers its first layers again.
        """
        if self._weight_manager is not None:
            self._weight_manager.invalidate_prefetch()
