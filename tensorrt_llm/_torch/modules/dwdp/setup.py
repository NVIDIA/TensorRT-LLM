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

"""DWDP setup orchestration: Transport -> WeightBuffer -> WeightManager -> backend patching.

This module provides ``setup_dwdp()``, the single entry point that runs ONCE
after model weight loading and BEFORE any inference forward pass.  It:

1. Collects MoE expert weight parameters from the model layers specified by
   ``layer_indices`` (the Single Source of Truth coming from
   ``DwdpManager._registered_layers``).
2. Builds per-layer WeightSpecs from the collected parameter shapes.
3. Computes the expert partition (``ExpertPartition``), the only source of
   every rank's expert range, and agrees on it across the DWDP group
   (``_agree_on_partition``).
4. Allocates MNNVL fabric handles via ``DWDPTransport.create()``, copying
   local weights into fabric memory and freeing the originals.  Handles
   are exchanged via the DWDP MPI communicator.
5. Constructs a WeightBuffer with composite VA layout (zero-copy local
   + page-pool-backed remote prefetch slots).
6. Fills page-alignment edge bytes at MNNVL region boundaries.
7. Creates a DWDPWeightManager for runtime P2P prefetch scheduling.
8. Exchanges the small per-expert parameters (e.g. fc31_alpha, fc2_alpha,
   an EP-sharded e_score_correction_bias) and checks the per-layer ones in
   one collective (``_exchange_expert_scalars``), which is also the group's
   vote on steps 5-7, then patches each MoE backend so it sees all
   ``num_experts`` (ep_size=1).  Large scale params (w3_w1_weight_scale,
   w2_weight_scale) go through Transport (MNNVL + P2P).

From the handshake to the vote, every check of the setup is made from data
the whole group gathered, the same way on every rank: the handshake, each
transport handle allgather, the transport's import status and the vote.  A
failure on one rank in that span thus raises on every rank after the same
collective, instead of leaving its peers blocked in the next one.  The
backend fixup after the vote is local: it installs the exchanged tensors (a
few KiB of device memory per MoE layer), and a failure there raises on that
rank only.

The function is idempotent: calling it a second time on the same model is a
no-op (returns the existing manager stored on the model).
"""

from __future__ import annotations

import itertools
import traceback
from collections import Counter
from typing import TYPE_CHECKING, Dict, List, NamedTuple, Optional, Sequence, Tuple, Union

import torch
from torch import nn

from tensorrt_llm.logger import logger

from .rendezvous import GroupComm
from .specs import ExpertPartition, LayerWeightSpecs, PeerRanges, WeightSpec, lookup_owner
from .transport import DWDPTransport, _allgather_keeping, _describe_error
from .vmm import get_allocation_granularity
from .weight_buffer import WeightBuffer
from .weight_manager import DWDPWeightManager

if TYPE_CHECKING:
    from tensorrt_llm.mapping import Mapping

# Sentinel attribute to mark idempotent setup completion.
# Must match the attribute name read in DeepseekV3Model.forward().
_DWDP_SETUP_DONE = "dwdp_weight_manager"

# Weight parameter names handled by the full DWDP pipeline:
#   Transport (MNNVL handles) → WeightBuffer (composite VA) → WeightManager (P2P prefetch)
# Includes the main expert weights AND their large EP-sharded scale params.
_EXPERT_WEIGHT_NAMES = (
    "w3_w1_weight",
    "w2_weight",
    "w3_w1_weight_scale",
    "w2_weight_scale",
)

# Name keywords of the small parameters of an experts module that the fixup
# exchanges (``_exchange_expert_scalars``): fc31_alpha, fc2_alpha, the
# activation input scales, ... ; names in _EXPERT_WEIGHT_NAMES go through
# the transport instead.
_SCALE_KEYWORDS = ("scale", "scaling_factor", "dequant", "alpha")

# Small parameters named ``*_act_scale`` are per-layer activation pre-quant scales
# (``fc31_act_scale``, ``[1, hidden_size]``): the NVFP4 AWQ loader keeps one vector for every
# expert of a layer, from the first expert it loads.  They are classified by name, before the
# shape tests, because with one expert per rank a ``[1, ...]`` parameter has a per-expert shape.
_ACT_SCALE_SUFFIX = "_act_scale"


def _release_during_rollback(
    resource: Union[DWDPTransport, WeightBuffer, DWDPWeightManager], what: str
) -> None:
    """Release a partially built resource without masking the in-flight error.

    Rollback runs while an exception is propagating. A raise from ``release()``
    would replace that exception, so the caller would see the cleanup failure
    instead of the reason setup failed.
    """
    try:
        resource.release()
    except BaseException:  # noqa: BLE001 - cleanup must never mask the original
        logger.error(
            f"[DWDP Setup] {what} release failed during rollback\n"
            f"{traceback.format_exc()}"
        )


def _roll_back_setup(
    transport: DWDPTransport,
    weight_buffer: Optional[WeightBuffer],
    weight_manager: Optional[DWDPWeightManager],
) -> None:
    """Release what ``setup_dwdp`` built after the transport, without masking the error in flight.

    The weight manager owns the buffer and the transport once it holds the transport.
    """
    if weight_manager is not None and weight_manager._transport is transport:
        _release_during_rollback(weight_manager, "weight manager")
        return
    if weight_buffer is not None:
        _release_during_rollback(weight_buffer, "weight buffer")
    _release_during_rollback(transport, "transport")


def _refuse_reserved_storages(
    local_params: Dict[Tuple[int, str], torch.Tensor], reserved_storages: frozenset[int]
) -> None:
    """Refuse expert weights whose storage another owner holds, before the transport takes them.

    The transport copies each weight into its own handle and then frees the weight's storage.
    ``DwdpManager.setup`` reserves the slots of the committed attached dense DWDP runtime: a
    weight there is a view dense DWDP repointed into a slot, which happens only when a dense
    root reaches a routed-expert weight.  The transport would then take the slot's bytes for the
    experts and free the slot under the dense runtime.

    Raises:
        RuntimeError: Expert weights live in reserved storages; the message lists them.
    """
    if not reserved_storages:
        return
    taken = [
        f"layers.{layer}.{name}"
        for (layer, name), tensor in sorted(local_params.items(), key=lambda item: item[0])
        if tensor.untyped_storage().data_ptr() in reserved_storages
    ]
    if taken:
        listed = ", ".join(taken[:8]) + (f" and {len(taken) - 8} more" if len(taken) > 8 else "")
        raise RuntimeError(
            f"[DWDP Setup] Expert weights in reserved storage, which the transport would free: "
            f"{listed}. They are views of the committed dense DWDP slots: dense DWDP moved them, "
            f"so the decoder layers' dense_dwdp_roots (or dense_dwdp_attr_allow) reach the routed "
            f"experts, which expert DWDP owns."
        )


def setup_dwdp(
    model: nn.Module,
    mapping: Mapping,
    device_id: int,
    comm: GroupComm,
    layer_indices: List[int],
    num_experts_per_worker: int,
    num_prefetch_experts: int,
    num_prefetch_slots: int,
    contention_opt: bool = False,
    repeated_layer_indices: Sequence[int] = (),
    copy_mode: Optional[str] = None,
    copy_stream: Optional[torch.cuda.Stream] = None,
    reserved_storages: frozenset[int] = frozenset(),
) -> Optional[DWDPWeightManager]:
    """Set up DWDP for a model after weight loading.

    This is the single orchestration entry point.  It must be called exactly
    once, after all model weights are loaded and before any forward pass.

    Args:
        model: The top-level causal-LM model (e.g. DeepseekV3ForCausalLM).
        mapping: ``tensorrt_llm.mapping.Mapping`` instance with DWDP fields.
        device_id: CUDA device ordinal for this rank.
        comm: The DWDP group's communicator, an mpi4py sub-communicator or a
            ``StoreDwdpGroup`` (owned by DwdpManager; lifetime exceeds
            setup_dwdp).
        layer_indices: MoE layer indices collected by
            ``DwdpManager._registered_layers`` (SSOT).  Each index must
            correspond to an MoE layer in the model.
        num_experts_per_worker: ``DwdpConfig.num_experts_per_worker``: the
            experts each rank stores in a strided expert partition, or 0
            (with ``num_prefetch_experts``) for the balanced partition.
        num_prefetch_experts: ``DwdpConfig.num_prefetch_experts``: the
            stride between consecutive ranks' first experts in a strided
            partition, or 0 for the balanced one.  Both go to
            ``ExpertPartition.from_config`` with the model's expert count,
            as in ``DwdpManager.expert_partition``, which sized the MoE
            backends before their weights were created; the handshake checks
            that every rank computed the same partition and loaded the rows
            of its range.  Under redundancy (``num_prefetch_experts <
            num_experts_per_worker``) adjacent ranks' ranges overlap by
            ``num_experts_per_worker - num_prefetch_experts`` experts and
            ``lookup_owner`` resolves the lowest-rank owner.
        num_prefetch_slots: Number of prefetch buffer slots, each holding the
            remote experts of one MoE layer (``DwdpConfig.num_prefetch_slots``).
        contention_opt: Copy-engine prefetch as one interleaved
            ``cudaMemcpyBatchAsync`` per layer (``DwdpConfig.contention_opt``).
        repeated_layer_indices: Layers of ``layer_indices`` that run several
            times in a row within a forward pass (``DwdpManager.add_repeated_layer``).
        copy_mode: The prefetch transfer (``DwdpConfig.copy_mode``); None keeps
            the selection by ``contention_opt`` and ``TRTLLM_DWDP_PREFETCH_GATHER``.
        copy_stream: The weight manager's copy stream (``DWDPWeightManager``
            creates one if None); ``DwdpManager.setup`` passes the stream
            attached dense DWDP was set up with.  A cached manager keeps its own.
        reserved_storages: ``untyped_storage().data_ptr()`` of storages that
            another owner holds, which the transport must not take over:
            ``DwdpManager.setup`` passes the slots of the committed attached
            dense DWDP runtime.  An expert weight in one of them raises a
            RuntimeError before the transport starts.

    Returns:
        A ready-to-use DWDPWeightManager if DWDP is enabled, ``None`` otherwise.
        The manager is also stored on ``model.dwdp_weight_manager`` so that
        a second call returns the cached instance (idempotent).

    Raises:
        RuntimeError: The DWDP group disagrees on the expert partition, a
            rank failed in the transport or in a later step before the vote,
            or the ranks' small expert parameters do not fit together.
            Raised on every rank after the same collective: a rank that
            failed raises its own error, the others an error naming it.  A
            failure in the backend fixup after the vote raises on that rank
            only.
        ValueError: The config gives no valid partition of the model's
            experts (raised on every rank alike).
    """
    # --- Guard: DWDP not enabled ---
    if not mapping.dwdp_enabled:
        return None

    # --- Guard: already set up? ---
    # Check the inner decoder model (where DeepseekV3Model.forward() reads
    # self.dwdp_weight_manager).
    decoder_model = _get_decoder_model(model)
    if hasattr(decoder_model, _DWDP_SETUP_DONE) and getattr(decoder_model, _DWDP_SETUP_DONE) is not None:
        logger.info("[DWDP Setup] Already initialized; returning cached manager.")
        return getattr(decoder_model, _DWDP_SETUP_DONE)

    # Log GPU memory at the start of DWDP setup
    free_mem, total_mem = torch.cuda.mem_get_info(device_id)
    logger.info(
        f"[DWDP Setup] GPU {device_id} memory before setup: "
        f"free={free_mem/(1024**3):.2f}GB / total={total_mem/(1024**3):.2f}GB "
        f"(used={((total_mem-free_mem)/(1024**3)):.2f}GB)"
    )
    logger.info(
        f"[DWDP Setup] Starting DWDP setup: dwdp_rank={mapping.dwdp_rank}, "
        f"dwdp_size={mapping.dwdp_size}, device_id={device_id}, "
        f"layer_indices={layer_indices}"
    )

    if not layer_indices:
        raise ValueError(
            "[DWDP Setup] layer_indices is empty. DwdpManager must register "
            "MoE layers via add_layer() before setup_dwdp() is called."
        )

    # 1. Collect MoE params (SSOT: iterate passed layer_indices only)
    local_params, weight_names = collect_moe_params(model, layer_indices)
    _refuse_reserved_storages(local_params, reserved_storages)
    logger.info(
        f"[DWDP Setup] Collected params from {len(layer_indices)} MoE layers, "
        f"weight_names={weight_names}"
    )

    # Query the model's *actual* gate-side num_experts from the first MoE
    # layer.  This is the routing-side expert count (e.g., 72 for DSv3-Lite,
    # 256 for DeepSeek-V3 / R1, 896 for Kimi-K3), the one the expert
    # partition splits.  All Phase 1+2 plumbing — composite VA size, the
    # partition, validation — uses this gate-side count.
    num_experts_total = _get_model_num_experts(model, layer_indices)

    # 2. Build weight specs (full_shape uses the gate-side num_experts)
    layer_weight_specs = build_weight_specs(
        local_params, weight_names, mapping, num_experts_total
    )
    logger.info(
        f"[DWDP Setup] Built weight specs for {len(layer_weight_specs)} layers, "
        f"num_experts={num_experts_total}"
    )

    # 3. The expert partition, from the function _init_dwdp_expert_layout
    #    sized the backends with, and the group's verdict on it before any
    #    handle exists: one allgather after which every rank goes on, or
    #    every rank raises.
    partition = ExpertPartition.from_config(
        num_experts=num_experts_total,
        dwdp_size=mapping.dwdp_size,
        num_experts_per_worker=num_experts_per_worker,
        num_prefetch_experts=num_prefetch_experts,
    )
    _agree_on_partition(
        comm,
        partition,
        dwdp_rank=mapping.dwdp_rank,
        granularity=get_allocation_granularity(device_id),
        layer_weight_specs=layer_weight_specs,
    )
    local_start, local_end = partition.range_of(mapping.dwdp_rank)
    logger.info(
        f"[DWDP Setup] Expert range: local=[{local_start}, {local_end}) "
        f"({local_end - local_start} experts), total={num_experts_total}, "
        f"partition {partition.describe()}"
    )

    # 4. Transport: MNNVL alloc -> copy -> free originals -> exchange handles
    logger.info("[DWDP Setup] Creating transport (MNNVL handle exchange)...")
    transport = DWDPTransport.create(
        layer_weight_specs=layer_weight_specs,
        local_params=local_params,
        comm=comm,
        dwdp_rank=mapping.dwdp_rank,
        device_id=device_id,
        partition=partition,
    )
    logger.info("[DWDP Setup] Transport created successfully.")

    # 5-7. Local work: an error here (e.g. this rank's prefetch ring does not
    # fit) is only recorded, and raised after the vote in step 8, which every
    # rank enters; the peers then raise there too, instead of waiting for
    # this rank in a later collective.
    weight_buffer = None
    weight_manager = None
    local_error = None
    try:
        free_mem, total_mem = torch.cuda.mem_get_info(device_id)
        logger.info(
            f"[DWDP Setup] GPU {device_id} memory after transport: "
            f"free={free_mem / (1024**3):.2f}GB / total={total_mem / (1024**3):.2f}GB "
            f"(used={((total_mem - free_mem) / (1024**3)):.2f}GB)"
        )
        peer_ranges = partition.peer_ranges

        # 5. WeightBuffer: composite VA layout
        logger.info("[DWDP Setup] Creating weight buffer (composite VA layout)...")
        weight_buffer = WeightBuffer.create(
            layer_weight_specs=layer_weight_specs,
            handles=transport.get_handle_set(),
            local_start=local_start,
            local_end=local_end,
            dwdp_size=mapping.dwdp_size,
            device_id=device_id,
            num_slots=num_prefetch_slots,
        )
        logger.info(
            f"[DWDP Setup] Weight buffer created successfully ({num_prefetch_slots} prefetch "
            f"slots)."
        )
        free_mem, total_mem = torch.cuda.mem_get_info(device_id)
        logger.info(
            f"[DWDP Setup] GPU {device_id} memory after weight buffer: "
            f"free={free_mem / (1024**3):.2f}GB / total={total_mem / (1024**3):.2f}GB "
            f"(used={((total_mem - free_mem) / (1024**3)):.2f}GB)"
        )

        # 6. Fill edge bytes
        logger.info("[DWDP Setup] Filling page-alignment edge bytes...")
        fill_edge_bytes(
            weight_buffer=weight_buffer,
            peer_views=transport.get_peer_views(),
            local_start=local_start,
            local_end=local_end,
            peer_ranges=peer_ranges,
        )
        logger.info("[DWDP Setup] Edge bytes filled.")

        # 7. WeightManager
        logger.info("[DWDP Setup] Creating weight manager...")
        weight_manager = DWDPWeightManager(
            weight_buffer=weight_buffer,
            peer_views=transport.get_peer_views(),
            peer_ranges=peer_ranges,
            moe_layer_indices=layer_indices,
            weight_names=list(weight_names),
            dwdp_rank=mapping.dwdp_rank,
            dwdp_size=mapping.dwdp_size,
            contention_opt=contention_opt,
            repeated_layer_indices=repeated_layer_indices,
            copy_mode=copy_mode,
            copy_stream=copy_stream,
        )
        # Transfer resource ownership before backend fixup so every later failure
        # can be rolled back through one idempotent manager release.
        weight_manager._transport = transport
        logger.info("[DWDP Setup] Weight manager created.")
    except Exception as exc:  # noqa: BLE001 - any failure is this rank's vote below
        local_error = exc
        logger.error(
            f"[DWDP Setup] Rank {mapping.dwdp_rank} failed after the transport with {exc!r}; "
            f"reporting it to the DWDP group"
        )
    except BaseException:
        _roll_back_setup(transport, weight_buffer, weight_manager)
        raise

    # 8. The vote and the backend fixup: _exchange_expert_scalars is the
    #    fixup's only collective, entered by every rank; it raises on every
    #    rank if any rank failed, and otherwise returns the full small
    #    per-expert tensors, which fixup_moe_backends installs locally.
    logger.info("[DWDP Setup] Patching MoE backends...")
    try:
        expert_scalars = _exchange_expert_scalars(
            comm,
            model,
            layer_indices,
            partition,
            error=local_error,
        )
        fixup_moe_backends(model, layer_indices, num_experts_total, expert_scalars)
    except BaseException:
        _roll_back_setup(transport, weight_buffer, weight_manager)
        raise
    logger.info("[DWDP Setup] MoE backends patched.")

    # Store on both the top-level model (for idempotent retrieval) and the
    # inner decoder model (where DeepseekV3Model.forward() reads it).
    setattr(model, _DWDP_SETUP_DONE, weight_manager)
    decoder_model.dwdp_weight_manager = weight_manager

    logger.info("[DWDP Setup] Setup complete.")
    return weight_manager


def teardown_dwdp(
    model: nn.Module,
    expected_manager: Optional[DWDPWeightManager] = None,
) -> None:
    """Detach and release a model's DWDP runtime state.

    Both the top-level model and its decoder hold the same manager so forward
    can reach it without a global lookup. Clear those references before
    unmapping the composite VA, then release the manager exactly once.
    """
    decoder_model = _get_decoder_model(model)
    managers = []
    for owner in (model, decoder_model):
        manager = getattr(owner, _DWDP_SETUP_DONE, None)
        if manager is not None and (
            expected_manager is None or manager is expected_manager
        ) and all(manager is not item for item in managers):
            managers.append(manager)
        if hasattr(owner, _DWDP_SETUP_DONE) and (
            expected_manager is None or manager is expected_manager
        ):
            setattr(owner, _DWDP_SETUP_DONE, None)

    if expected_manager is not None and all(
        expected_manager is not item for item in managers
    ):
        managers.append(expected_manager)

    first_error = None
    for manager in managers:
        try:
            manager.release()
        except BaseException as exc:  # noqa: BLE001 - every manager must be tried
            logger.error(
                f"[DWDP Teardown] Manager release failed\n{traceback.format_exc()}"
            )
            if first_error is None:
                first_error = exc
    if first_error is not None:
        raise first_error


# ---------------------------------------------------------------------------
# Helper: collect MoE params (SSOT: layer_indices is input, not discovered)
# ---------------------------------------------------------------------------


def collect_moe_params(
    model: nn.Module,
    layer_indices: List[int],
) -> Tuple[Dict[Tuple[int, str], torch.Tensor], List[str]]:
    """Extract MoE expert weight/scale tensors from the specified layers.

    Iterates the passed ``layer_indices`` (the authoritative list from
    ``DwdpManager._registered_layers``) and pulls expert parameters off
    each layer's MoE module.  Layer discovery is NOT done here — that is
    the DwdpManager's job via ``add_layer()``.

    Args:
        model: Top-level CausalLM model (e.g. DeepseekV3ForCausalLM).
        layer_indices: Sorted list of MoE decoder layer indices, registered
            by ConfigurableMoE.__init__() → DwdpManager.add_layer().

    Returns:
        Tuple of:
            - local_params: Dict mapping ``(layer_idx, weight_name)`` to the
              parameter tensor (still on device, will be consumed by Transport).
            - weight_names: List of weight parameter names collected (subset of
              ``_EXPERT_WEIGHT_NAMES`` that actually exist on the backend).

    Raises:
        RuntimeError: If any layer in ``layer_indices`` does not have an
            MoE experts module with the expected weight parameters.
    """
    local_params: Dict[Tuple[int, str], torch.Tensor] = {}
    weight_names_set: List[str] = []

    decoder_model = _get_decoder_model(model)
    layers = decoder_model.layers

    for layer_idx in layer_indices:
        layer = layers[layer_idx]
        _moe_module, experts_module = _get_moe_and_experts(layer)
        if experts_module is None:
            raise RuntimeError(
                f"[DWDP Setup] Layer {layer_idx} is registered as an MoE layer "
                f"but no experts module was found. Check DwdpManager.add_layer() "
                f"call sites."
            )

        found_names = []
        for wname in _EXPERT_WEIGHT_NAMES:
            param = getattr(experts_module, wname, None)
            if param is not None and isinstance(param, (torch.Tensor, nn.Parameter)):
                found_names.append(wname)
                local_params[(layer_idx, wname)] = (
                    param.data if isinstance(param, nn.Parameter) else param
                )

        if not found_names:
            raise RuntimeError(
                f"[DWDP Setup] Layer {layer_idx} MoE experts module has none of "
                f"the expected weight parameters {_EXPERT_WEIGHT_NAMES}."
            )

        if not weight_names_set:
            weight_names_set = found_names
        logger.debug(
            f"[DWDP Setup] Layer {layer_idx}: collected {found_names}, "
            f"shapes={[local_params[(layer_idx, n)].shape for n in found_names]}"
        )

    return local_params, weight_names_set


# ---------------------------------------------------------------------------
# Helper: build weight specs
# ---------------------------------------------------------------------------


def build_weight_specs(
    local_params: Dict[Tuple[int, str], torch.Tensor],
    weight_names: List[str],
    mapping: Mapping,
    num_experts_total: int,
) -> LayerWeightSpecs:
    """Convert collected parameters to ``LayerWeightSpecs``.

    Each local parameter has shape ``(local_experts, ...)`` where
    ``local_experts`` is the number of experts this rank stores: the length
    of its range in the expert partition, which ``_init_dwdp_expert_layout``
    sized the backend with (ranks may differ, e.g. 75 and 74).  The
    composite VA's ``full_shape`` uses ``num_experts_total`` (gate-side,
    e.g. 72 for DSv3-Lite) so every expert id in ``[0, num_experts_total)``
    is addressable — and *only* those.  Under redundancy the ranks' counts
    add up to more than ``num_experts_total`` (overlap duplicates), but
    those duplicates live within each peer's MNNVL handle, not in the
    composite VA.

    Args:
        local_params: Dict from ``collect_moe_params``.
        weight_names: Weight names list.
        mapping: Mapping with dwdp_size.
        num_experts_total: Gate-side expert count read from the model
            (``MoE.num_experts``), used as ``WeightSpec.num_experts`` /
            ``full_shape[0]``.

    Returns:
        LayerWeightSpecs: ``Dict[layer_idx, Dict[weight_name, WeightSpec]]``.
    """
    layer_weight_specs: LayerWeightSpecs = {}

    # Group params by layer
    layers_seen: Dict[int, List[str]] = {}
    for (layer_idx, wname) in local_params:
        if layer_idx not in layers_seen:
            layers_seen[layer_idx] = []
        layers_seen[layer_idx].append(wname)

    for layer_idx in sorted(layers_seen.keys()):
        specs: Dict[str, WeightSpec] = {}
        for wname in weight_names:
            key = (layer_idx, wname)
            if key not in local_params:
                continue
            param = local_params[key]
            chunk_shape = tuple(param.shape)
            full_shape = (num_experts_total,) + chunk_shape[1:]
            specs[wname] = WeightSpec(
                num_experts=num_experts_total,
                chunk_shape=chunk_shape,
                full_shape=full_shape,
                dtype=param.dtype,
            )
        layer_weight_specs[layer_idx] = specs

    return layer_weight_specs


def _get_model_num_experts(model: nn.Module, layer_indices: List[int]) -> int:
    """Read the gate-side ``num_experts`` from the first MoE layer.

    The fused MoE backend stores the routing-side expert count
    (``MoE.num_experts`` from the model config, e.g.
    ``n_routed_experts``) on each ``experts_module``.  All MoE layers in
    a given model share the same value, so reading the first one is
    sufficient.

    Args:
        model: Top-level CausalLM model.
        layer_indices: Non-empty list of MoE layer indices.

    Returns:
        Gate-side expert count (positive integer).

    Raises:
        RuntimeError: If the first MoE layer's experts module cannot be
            found or does not expose ``num_experts``.
    """
    decoder_model = _get_decoder_model(model)
    layers = decoder_model.layers
    first_layer_idx = sorted(layer_indices)[0]
    _, experts_module = _get_moe_and_experts(layers[first_layer_idx])
    if experts_module is None:
        raise RuntimeError(
            f"[DWDP Setup] Cannot read num_experts: layer {first_layer_idx} "
            f"has no experts module."
        )
    n = getattr(experts_module, "num_experts", None)
    if not isinstance(n, int) or n <= 0:
        raise RuntimeError(
            f"[DWDP Setup] Layer {first_layer_idx} experts module has "
            f"invalid num_experts={n!r}."
        )
    return n


# ---------------------------------------------------------------------------
# Helper: agree on the expert partition (handshake)
# ---------------------------------------------------------------------------


def _agree_on_partition(
    comm: GroupComm,
    partition: ExpertPartition,
    *,
    dwdp_rank: int,
    granularity: int,
    layer_weight_specs: LayerWeightSpecs,
) -> None:
    """One allgather after which every rank of the DWDP group goes on, or every rank raises.

    Each rank reports its DWDP rank, the partition it computed (expert count, every rank's
    range, kind), its VMM granularity and, for every (layer, weight) expert chunk it loaded,
    the rows, the shape of one expert and the dtype.  The verdict is a function of the gathered
    reports only, so it is the same on every rank: the transport after it is entered by every
    rank or by none.  The transport then maps every peer's handle with the peer's range and its
    own granularity and bytes per expert, which this makes the same on every rank.
    Rejected: a rank that sent no handshake (it runs another TensorRT-LLM tree); a rank whose
    partition (config or model drift), granularity, or loaded (layer, weight) chunks, expert
    shapes or dtypes differ from the most common report's, so that the ranks named are the
    ones that differ from the others, rank 0 included; a chunk whose rows are not its rank's
    expert count (``_init_dwdp_expert_layout`` sized the backend from another partition); and
    a report out of rank order.

    Raises:
        RuntimeError: On every rank, naming each offending rank and item (the first 8).
    """
    chunks = tuple(
        (layer_idx, name, tuple(spec.chunk_shape[1:]), str(spec.dtype), spec.local_experts)
        for layer_idx in sorted(layer_weight_specs)
        for name, spec in sorted(layer_weight_specs[layer_idx].items())
    )
    report = (
        dwdp_rank,
        partition.num_experts,
        partition.ranges,
        partition.kind,
        granularity,
        chunks,
    )
    reports = comm.allgather(report)
    malformed = [
        rank
        for rank, rank_report in enumerate(reports)
        if not (isinstance(rank_report, tuple) and len(rank_report) == len(report))
    ]
    if malformed:
        raise RuntimeError(
            f"[DWDP Setup] DWDP ranks {malformed} sent no expert partition handshake: every rank "
            f"of the DWDP group must run the same TensorRT-LLM tree."
        )

    problems = []
    if len(reports) != partition.dwdp_size:
        problems.append(f"{len(reports)} ranks reported, the partition has {partition.dwdp_size}")
    # The reference is the most common report, rank and rows aside (on a tie, the lowest rank's),
    # so that the ranks named are the ones that differ from the others, rank 0 included.
    keys = [
        (num_experts, ranges, kind, rank_granularity, tuple(chunk[:-1] for chunk in rank_chunks))
        for _, num_experts, ranges, kind, rank_granularity, rank_chunks in reports
    ]
    ref_rank = keys.index(Counter(keys).most_common(1)[0][0])
    _, ref_experts, ref_ranges, ref_kind, ref_granularity, ref_chunks = reports[ref_rank]
    # Every rank must load the same (layer, weight) chunks, of the same expert shape and dtype.
    ref_layout = [chunk[:-1] for chunk in ref_chunks]
    for rank, rank_report in enumerate(reports):
        reported_rank, num_experts, ranges, kind, rank_granularity, rank_chunks = rank_report
        if reported_rank != rank:
            problems.append(f"position {rank} holds the report of DWDP rank {reported_rank}")
            continue
        if (num_experts, ranges, kind) != (ref_experts, ref_ranges, ref_kind):
            problems.append(
                f"rank {rank} computed the {kind} partition {list(ranges)} of {num_experts} "
                f"experts, rank {ref_rank} the {ref_kind} partition {list(ref_ranges)} of "
                f"{ref_experts}"
            )
            continue
        if rank_granularity != ref_granularity:
            problems.append(
                f"rank {rank} maps with a granularity of {rank_granularity} bytes, rank "
                f"{ref_rank} with {ref_granularity}"
            )
        layout = [chunk[:-1] for chunk in rank_chunks]
        if layout != ref_layout:
            mine, ref = next(
                (mine, ref)
                for mine, ref in itertools.zip_longest(layout, ref_layout)
                if mine != ref
            )
            problems.append(
                f"rank {rank} loaded other expert chunks than rank {ref_rank}: (layer, weight, "
                f"expert shape, dtype) {mine}, rank {ref_rank} {ref}"
            )
            continue
        if rank >= len(ranges):
            continue
        start, end = ranges[rank]
        wrong = [(layer_idx, name, n) for layer_idx, name, *_, n in rank_chunks if n != end - start]
        if wrong:
            layer_idx, name, n = wrong[0]
            problems.append(
                f"rank {rank} stores experts [{start}, {end}) ({end - start} experts) but "
                f"loaded {len(wrong)} expert chunks with other row counts, e.g. layer "
                f"{layer_idx} {name} with {n} rows"
            )
    if problems:
        raise RuntimeError(
            f"[DWDP Setup] The DWDP group does not agree on the expert partition: "
            f"{'; '.join(problems[:8])}"
            + (f"; and {len(problems) - 8} more" if len(problems) > 8 else "")
            + ". Every rank must run the same DwdpConfig and model; "
            "_init_dwdp_expert_layout sizes the MoE layers DWDP manages from "
            "DwdpManager.expert_partition()."
        )


# ---------------------------------------------------------------------------
# Helper: fill edge bytes
# ---------------------------------------------------------------------------


def fill_edge_bytes(
    weight_buffer: WeightBuffer,
    peer_views: Dict[Tuple[int, int, str], torch.Tensor],
    local_start: int,
    local_end: int,
    peer_ranges: PeerRanges,
) -> None:
    """Fill page-alignment edge bytes at MNNVL region boundaries.

    When local expert data does not align to page boundaries, the MNNVL pages
    contain edge regions that hold peer data: per (layer, weight) at most one
    *leading edge* ``[page_start, local_start * expert_bytes)`` and one
    *trailing edge* ``[local_end * expert_bytes, page_end)``.  With experts
    smaller than the granularity an edge holds bytes of several experts; only
    the experts next to the local range are filled here.  Every prefetch
    rewrites the remote experts, edges included, before any read.

    The data is read from peer MNNVL tensor views and written into the local
    MNNVL handle's mapped VA.  This is a one-time setup operation.

    Args:
        weight_buffer: WeightBuffer with composite VA layout.
        peer_views: ``{(peer_rank, layer_idx, name): tensor}`` from Transport.
        local_start: First local expert index (inclusive).
        local_end: Last local expert index (exclusive).
        peer_ranges: Every rank's ``(start, end)`` in the expert partition.
            ``lookup_owner`` resolves the owner of ``prev_expert`` /
            ``next_expert`` here, and each peer view has that peer's rows,
            so any partition (balanced, capped, redundant) picks the right
            peer and row.
    """
    if local_start >= local_end:
        return

    for layer_idx in weight_buffer.layer_indices:
        for name in weight_buffer.weight_names(layer_idx):
            edge_info = weight_buffer.get_edge_info(layer_idx, name)

            if edge_info.leading_edge == 0 and edge_info.trailing_edge == 0:
                logger.debug(
                    f"[DWDP Setup] Layer {layer_idx}/{name}: no edge bytes to fill"
                )
                continue

            # Get the full tensor view (composite VA)
            full_tensor = weight_buffer.get_full_tensor(layer_idx, name)

            # Leading edge: bytes in [page_start, local_start_bytes).
            # These belong to the experts before local_start (possibly several).
            if edge_info.leading_edge > 0 and local_start > 0:
                # Fill the expert just before local_start.  Use lookup_owner so
                # non-uniform / redundant partitions pick the lowest-rank peer
                # that owns ``prev_expert``.
                prev_expert = local_start - 1
                peer_rank = lookup_owner(prev_expert, peer_ranges)
                peer_chunk_start, _ = peer_ranges[peer_rank]
                peer_local_offset = prev_expert - peer_chunk_start

                peer_key = (peer_rank, layer_idx, name)
                if peer_key in peer_views:
                    src = peer_views[peer_key]
                    # prev_expert's data, or its tail, lies in our MNNVL
                    # page.  Copy the full expert slice (the composite VA's
                    # expert dimension is page-aligned).
                    full_tensor[prev_expert].copy_(src[peer_local_offset])
                    logger.debug(
                        f"[DWDP Setup] Layer {layer_idx}/{name}: filled leading edge "
                        f"from peer {peer_rank}, expert {prev_expert}"
                    )

            # Trailing edge: bytes in [local_end_bytes, page_end).
            # These belong to the experts from local_end on (possibly several).
            if edge_info.trailing_edge > 0 and local_end < full_tensor.shape[0]:
                next_expert = local_end
                peer_rank = lookup_owner(next_expert, peer_ranges)
                peer_chunk_start, _ = peer_ranges[peer_rank]
                peer_local_offset = next_expert - peer_chunk_start

                peer_key = (peer_rank, layer_idx, name)
                if peer_key in peer_views:
                    src = peer_views[peer_key]
                    full_tensor[next_expert].copy_(src[peer_local_offset])
                    logger.debug(
                        f"[DWDP Setup] Layer {layer_idx}/{name}: filled trailing edge "
                        f"from peer {peer_rank}, expert {next_expert}"
                    )

    torch.cuda.synchronize(weight_buffer.device_id)


# ---------------------------------------------------------------------------
# Helper: exchange the small expert parameters (the vote), fixup MoE backends
# ---------------------------------------------------------------------------


class _ParamReport(NamedTuple):
    """One small parameter of one layer as a rank reports it to ``_exchange_expert_scalars``.

    Attributes:
        shape: The parameter's shape on the rank.
        data: A CPU copy; None when the first dimension is the model's expert count, since a
            full-sized parameter needs no exchange.
    """

    shape: Tuple[int, ...]
    data: Optional[torch.Tensor]


class _LayerScalars(NamedTuple):
    """What ``fixup_moe_backends`` installs on one MoE layer after ``_exchange_expert_scalars``.

    Attributes:
        params: ``[num_experts, ...]`` CPU tensors of the experts module's per-expert small
            parameters, by name, put together from every rank's shard.
        bias: The ``[num_experts]`` CPU gate ``e_score_correction_bias`` put together from every
            rank's shard, or None when it is full-sized on every rank already, or absent.
        act_scales: Rank 0's CPU copy of each per-layer activation scale (``*_act_scale``), by
            name, which every rank copies into its own parameter.
    """

    params: Dict[str, torch.Tensor]
    bias: Optional[torch.Tensor]
    act_scales: Dict[str, torch.Tensor]


def _report_param(tensor: torch.Tensor, num_experts: int) -> _ParamReport:
    """``tensor`` as this rank reports it: its shape, and a CPU copy unless it is full-sized."""
    shape = tuple(tensor.shape)
    if shape and shape[0] == num_experts:
        return _ParamReport(shape, None)
    return _ParamReport(shape, tensor.detach().cpu().contiguous())


def _collect_expert_scalars(
    model: nn.Module, layer_indices: List[int], num_experts: int
) -> Dict[int, Tuple[Dict[str, _ParamReport], Optional[_ParamReport]]]:
    """This rank's small parameters of every MoE layer, as ``_exchange_expert_scalars`` sends them.

    Per layer: the direct parameters of the experts module whose names contain one of
    ``_SCALE_KEYWORDS`` and are not in ``_EXPERT_WEIGHT_NAMES`` (the transport moves those),
    and the gate's ``e_score_correction_bias`` (None without one).
    """
    layers = _get_decoder_model(model).layers
    collected = {}
    for layer_idx in layer_indices:
        moe_module, experts_module = _get_moe_and_experts(layers[layer_idx])
        params = {}
        if experts_module is not None:
            for pname, param in experts_module.named_parameters(recurse=False):
                if pname in _EXPERT_WEIGHT_NAMES or not any(kw in pname for kw in _SCALE_KEYWORDS):
                    continue
                params[pname] = _report_param(param.data, num_experts)
        bias = None
        gate_module = _get_gate_module(moe_module) if moe_module is not None else None
        bias_param = getattr(gate_module, "e_score_correction_bias", None)
        if bias_param is not None:
            bias_data = bias_param.data if isinstance(bias_param, nn.Parameter) else bias_param
            bias = _report_param(bias_data, num_experts)
        collected[layer_idx] = (params, bias)
    return collected


def _bitwise_equal(a: Optional[torch.Tensor], b: Optional[torch.Tensor]) -> bool:
    """Same dtype, shape and bytes (NaN payloads included)."""
    if a is None or b is None or a.dtype != b.dtype or a.shape != b.shape:
        return False
    return torch.equal(a.reshape(-1).view(torch.uint8), b.reshape(-1).view(torch.uint8))


def _ulp_distance(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Elementwise distance of two 16-bit float tensors in units in the last place.

    The bits are read as integers, with the sign-magnitude encoding made monotonic, so
    adjacent representable values are 1 apart and ``+0`` and ``-0`` are 0 apart.
    """

    def ordered(t: torch.Tensor) -> torch.Tensor:
        bits = t.contiguous().view(torch.int16).to(torch.int32)
        return torch.where(bits < 0, -(bits & 0x7FFF), bits)

    return (ordered(a) - ordered(b)).abs()


def _act_scale_matches(value: Optional[torch.Tensor], reference: Optional[torch.Tensor]) -> bool:
    """Whether a rank's per-layer activation scale can stand for rank 0's ``reference``.

    The NVFP4 AWQ loader accepts the experts of a layer whose ``pre_quant_scale`` match the
    first one it loads within ``rtol=1e-5, atol=1e-8``, and casts that first one to the module
    dtype.  Each rank loads its own experts, so two ranks' values may differ by that tolerance
    before the cast, which is one unit in the last place of a 16-bit dtype after it.  Other
    dtypes are compared with the loader's tolerance.
    """
    if value is None or reference is None:
        return False
    if value.dtype != reference.dtype or value.shape != reference.shape:
        return False
    if value.dtype.is_floating_point and value.element_size() == 2:
        return bool((_ulp_distance(value, reference) <= 1).all())
    return torch.allclose(value.double(), reference.double(), rtol=1e-5, atol=1e-8)


def _agree_on_act_scale(
    what: str, reports: List[_ParamReport], problems: List[str]
) -> Optional[torch.Tensor]:
    """Rank 0's copy of a per-layer activation scale, if every rank's can stand for it.

    Returns:
        Rank 0's CPU tensor, which every rank installs; None after appending to ``problems``.
    """
    reference = reports[0].data
    differing = [
        rank
        for rank, report in enumerate(reports)
        if not _act_scale_matches(report.data, reference)
    ]
    if not differing:
        return reference
    dtype = None if reference is None else reference.dtype
    problems.append(
        f"{what} is a per-layer activation scale, but ranks {differing} hold values further "
        f"from rank 0's than one unit in the last place of {dtype} (or another shape or dtype). "
        f"Each rank takes it from the first expert it loads; a load on one rank would require "
        f"every expert's pre_quant_scale to match expert 0's within rtol=1e-5, which this check "
        f"of one value per rank cannot replace"
    )
    return None


def _merge_shards(
    what: str,
    reports: List[_ParamReport],
    partition: ExpertPartition,
    problems: List[str],
    *,
    allow_per_layer: bool,
) -> Tuple[str, Optional[torch.Tensor]]:
    """Classify one parameter from every rank's report and put a per-expert one together.

    Returns:
        ``(category, full)``: ``"per expert"`` with the ``[num_experts, ...]`` tensor,
        ``"full"`` (full-sized on every rank, nothing to install), ``"per layer"`` (the same
        shape on every rank and not per expert; it must be bitwise equal on every rank, nothing
        to install) or ``"invalid"``; every inconsistency is appended to ``problems``.
    """
    num_experts = partition.num_experts
    shapes = [report.shape for report in reports]
    counts = [partition.count_of(rank) for rank in range(len(reports))]
    trailing = {shape[1:] for shape in shapes}
    if len(trailing) == 1 and all(shape and shape[0] == num_experts for shape in shapes):
        return "full", None
    if len(trailing) == 1 and all(
        shape and shape[0] == count for shape, count in zip(shapes, counts)
    ):
        shards = [report.data for report in reports]
        if len({shard.dtype for shard in shards}) != 1:
            problems.append(f"{what}: dtypes {[shard.dtype for shard in shards]} per rank")
            return "invalid", None
        full = _scatter_shards_to_full(
            shards=shards,
            peer_ranges=partition.peer_ranges,
            num_experts_total=num_experts,
            ref=shards[0],
        )
        # Overlapping ranges store an expert on more than one rank: every copy must agree,
        # i.e. each rank's shard is what was put together over its range.
        differing = set()
        for rank, shard in enumerate(shards):
            start, end = partition.range_of(rank)
            if shard.numel() == 0:
                continue
            own = shard.reshape(end - start, -1).view(torch.uint8)
            merged = full[start:end].reshape(end - start, -1).view(torch.uint8)
            rows = (own != merged).any(dim=1).nonzero().flatten().tolist()
            differing.update(start + row for row in rows)
        if differing:
            problems.append(
                f"{what}: the ranks storing experts {sorted(differing)[:8]} hold different "
                f"values for them"
            )
        return "per expert", full
    if allow_per_layer and len(set(shapes)) == 1:
        differing = [
            rank
            for rank, report in enumerate(reports)
            if not _bitwise_equal(report.data, reports[0].data)
        ]
        if differing:
            problem = f"{what} is per layer, but ranks {differing} hold other values than rank 0"
            if not shapes[0]:
                # The activation input scales.
                problem += (
                    f" ({[report.data.item() for report in reports]}); a rank reduces it over the "
                    f"experts it loads (quantization's load_streaming_nvfp4_expert), so all "
                    f"experts of the checkpoint must have the same one"
                )
            problems.append(problem)
        return "per layer", None
    problems.append(
        f"{what}: shapes {shapes} per rank; a per-expert parameter has the rank's expert count "
        f"{counts} or {num_experts} rows on every rank"
    )
    return "invalid", None


def _exchange_expert_scalars(
    comm: GroupComm,
    model: nn.Module,
    layer_indices: List[int],
    partition: ExpertPartition,
    *,
    error: Optional[Exception] = None,
) -> Dict[int, _LayerScalars]:
    """The fixup's one collective, which is also the group's vote on the setup steps before it.

    Every rank enters it, also after a failure, with its status (``error``, None on success)
    and its small expert parameters (``_collect_expert_scalars``) of every MoE layer.  The
    verdict is a function of the gathered reports only, so it is the same on every rank:

    * A rank failed: every rank raises, the failed rank its own error and the others a
      RuntimeError naming it, and each rolls its setup back.
    * Each layer's parameter names must match across ranks.  A parameter with ``count_p`` rows
      on every rank ``p`` (``fc31_alpha``, ``fc2_alpha``) is per expert: the shards are put
      together into ``[num_experts, ...]`` along the partition (``_scatter_shards_to_full``),
      and every rank storing an expert (overlapping ranges) must hold the same values for it.
      One with ``num_experts`` rows on every rank is full-sized already.  A per-layer
      activation scale (``*_act_scale``, e.g. AWQ's ``fc31_act_scale``), classified by name,
      must be within one unit in the last place of rank 0's copy (``_act_scale_matches``),
      which every rank then installs.  A zero-dimensional one (``fc31_input_scale``,
      ``fc2_input_scale``), which the alphas fold in, or any other one of the same shape on
      every rank, is per layer and must be bitwise equal on every rank.  Anything else raises.
    * The gate's ``e_score_correction_bias``: full-sized on every rank (the gate loads it
      whole under ``moe_ep_size = 1``) stays; ``count_p`` entries on every rank are put
      together like a per-expert parameter; anything else raises.

    Returns:
        ``_LayerScalars`` per layer index, for ``fixup_moe_backends``.

    Raises:
        RuntimeError: As above, on every rank after the same collective.
    """
    collected = {}
    if error is None:
        try:
            collected = _collect_expert_scalars(model, layer_indices, partition.num_experts)
        except Exception as exc:  # noqa: BLE001 - reported to the group below
            error = exc
    status = None if error is None else _describe_error(error)
    reports = _allgather_keeping(comm, (status, collected), error)
    if error is not None:
        raise error
    failed = [
        f"rank {rank}: {rank_status}"
        for rank, (rank_status, _) in enumerate(reports)
        if rank_status is not None
    ]
    if failed:
        raise RuntimeError(
            f"[DWDP Setup] DWDP setup failed on {'; '.join(failed)}. Every rank of the DWDP "
            f"group raises this and releases its DWDP resources."
        )

    problems: List[str] = []
    categories: Dict[str, set[str]] = {}
    expert_scalars: Dict[int, _LayerScalars] = {}
    for layer_idx in layer_indices:
        per_rank = [layers.get(layer_idx) for _, layers in reports]
        if any(entry is None for entry in per_rank):
            problems.append(f"layer {layer_idx} is missing on some ranks")
            continue
        names = sorted(per_rank[0][0])
        if any(sorted(params) != names for params, _ in per_rank):
            problems.append(
                f"layer {layer_idx}: the ranks have other small expert parameters "
                f"({[sorted(params) for params, _ in per_rank]})"
            )
            continue
        params = {}
        act_scales = {}
        for pname in names:
            what = f"layer {layer_idx} {pname}"
            reports_of_param = [rank_params[pname] for rank_params, _ in per_rank]
            if pname.endswith(_ACT_SCALE_SUFFIX):
                categories.setdefault("per layer, rank 0's installed", set()).add(pname)
                act_scale = _agree_on_act_scale(what, reports_of_param, problems)
                if act_scale is not None:
                    act_scales[pname] = act_scale
                continue
            category, full = _merge_shards(
                what, reports_of_param, partition, problems, allow_per_layer=True
            )
            categories.setdefault(category, set()).add(pname)
            if full is not None:
                params[pname] = full
        biases = [bias for _, bias in per_rank]
        bias = None
        if all(report is None for report in biases):
            categories.setdefault("absent", set()).add("e_score_correction_bias")
        elif any(report is None for report in biases):
            problems.append(f"layer {layer_idx}: only some ranks have an e_score_correction_bias")
        else:
            category, bias = _merge_shards(
                f"layer {layer_idx} e_score_correction_bias",
                biases,
                partition,
                problems,
                allow_per_layer=False,
            )
            categories.setdefault(category, set()).add("e_score_correction_bias")
        expert_scalars[layer_idx] = _LayerScalars(params, bias, act_scales)
    if problems:
        raise RuntimeError(
            f"[DWDP Setup] The DWDP group's small expert parameters do not fit together: "
            f"{'; '.join(problems[:8])}"
            + (f"; and {len(problems) - 8} more" if len(problems) > 8 else "")
            + ". Every rank of the DWDP group raises this."
        )
    summary = "; ".join(f"{kind}: {sorted(names)}" for kind, names in sorted(categories.items()))
    logger.info(
        f"[DWDP Setup] Exchanged the small expert parameters of {len(layer_indices)} MoE layers "
        f"in one allgather (partition {partition.describe()}): {summary}"
    )
    return expert_scalars


def fixup_moe_backends(
    model: nn.Module,
    moe_layer_indices: List[int],
    num_experts_total: int,
    expert_scalars: Dict[int, _LayerScalars],
) -> None:
    """Patch each MoE backend to see all experts (ep_size=1) after DWDP setup.

    After DWDP setup, each rank conceptually owns *all* experts (the weight
    buffer has a full [num_experts, ...] tensor via composite VA).  The MoE
    backend must therefore be reconfigured:

    - ``ep_size = 1``, ``ep_rank = 0``: no more EP sharding from the backend's
      perspective; DWDP handles the distribution.
    - ``num_experts``: unchanged (already the global count).
    - ``expert_size_per_partition = num_experts``: all experts are "local".
    - ``slot_start = 0``, ``slot_end = num_experts``.
    - ``initial_local_expert_ids = list(range(num_experts))``.
    - ``initial_global_assignments = list(range(num_experts))``.
    - ``num_slots = num_experts``.

    Additionally, it installs the full-sized small parameters that
    ``_exchange_expert_scalars`` put together from every rank's shard
    (``fc31_alpha`` / ``fc2_alpha``; the gate's ``e_score_correction_bias``
    only if it was loaded per expert: on the ``moe_ep_size = 1`` setup the
    gate loads it whole), copies rank 0's per-layer activation scales
    (``*_act_scale``) into this rank's, and rebuilds the quant scales.  Local
    only: the exchange was the setup's last collective.

    Args:
        model: Top-level CausalLM model.
        moe_layer_indices: MoE layer indices to patch.
        num_experts_total: Total number of experts across all ranks.
        expert_scalars: ``_exchange_expert_scalars``'s result, per layer index.
    """
    decoder_model = _get_decoder_model(model)
    layers = decoder_model.layers

    for layer_idx in moe_layer_indices:
        layer = layers[layer_idx]
        moe_module, experts_module = _get_moe_and_experts(layer)
        if experts_module is None:
            logger.warning(
                f"[DWDP Setup] Layer {layer_idx}: could not find experts module "
                f"for backend patching (skipped)"
            )
            continue

        # --- Patch expert-parallel attributes on both the ConfigurableMoE ---
        # --- wrapper (if present) AND the backend. ---
        # ConfigurableMoE has its own ep_size, slot_start, etc. that are used
        # in its forward path.  The backend is the inner module that holds
        # weight parameters.
        # ``moe_module`` is what _get_moe_and_experts() just resolved, so this
        # is model-agnostic: on DeepSeek it is layer.mlp and this stays exactly
        # equivalent to the old getattr(layer.mlp, "experts", None); on K3 it
        # is layer.block_sparse_moe, which has no ``.mlp`` at all.
        configurable_moe = _get_configurable_moe(moe_module)
        targets = [experts_module]
        if configurable_moe is not None and configurable_moe is not experts_module:
            targets.insert(0, configurable_moe)

        old_ep_size = getattr(experts_module, "ep_size", None)
        old_ep_rank = getattr(experts_module, "ep_rank", None)
        old_slot_start = getattr(experts_module, "slot_start", None)
        old_slot_end = getattr(experts_module, "slot_end", None)

        for target in targets:
            target.ep_size = 1
            target.ep_rank = 0
            target.expert_size_per_partition = num_experts_total
            target.slot_start = 0
            target.slot_end = num_experts_total
            target.num_slots = num_experts_total
            target.initial_local_expert_ids = list(range(num_experts_total))
            target.initial_global_assignments = list(range(num_experts_total))

        # ``expert_size_per_partition`` just changed, and it sizes every
        # PER_EXPERT_TENSOR activation constant. This function is the layout's
        # last writer, so re-materialize them here. Backend only: the wrapper
        # declares no ``activation_support`` of its own. Imported locally because
        # ``fused_moe`` reads the DWDP manager, so a module-scope import here
        # would close that cycle. Spelled absolute rather than relative: this
        # module sits one level deeper than ``modules/``, so the relative form
        # needs three dots, and getting that wrong resolves to a package that
        # does not exist without any static check noticing.
        from tensorrt_llm._torch.moe.fused_moe.activation import install_activation_params

        if getattr(experts_module, "activation", None) is not None:
            install_activation_params(experts_module)

        logger.debug(
            f"[DWDP Setup] Layer {layer_idx}: patched "
            f"{'ConfigurableMoE + ' if len(targets) > 1 else ''}backend "
            f"ep_size={old_ep_size}->{1}, ep_rank={old_ep_rank}->{0}, "
            f"slots=[{old_slot_start},{old_slot_end})->[0,{num_experts_total})"
        )

        layer_scalars = expert_scalars.get(layer_idx)
        if layer_scalars is not None:
            # --- The gate's e_score_correction_bias, if it was per expert ---
            gate_module = _get_gate_module(moe_module)
            if layer_scalars.bias is not None and gate_module is not None:
                bias_param = gate_module.e_score_correction_bias
                bias_data = bias_param.data if isinstance(bias_param, nn.Parameter) else bias_param
                full_bias = layer_scalars.bias.to(bias_data.device)
                if isinstance(bias_param, nn.Parameter):
                    gate_module.e_score_correction_bias = nn.Parameter(
                        full_bias, requires_grad=False
                    )
                else:
                    bias_param.data = full_bias

            # --- Small per-expert scale/dequant parameters ---
            # Large scale params (w3_w1_weight_scale, w2_weight_scale) are
            # handled by Transport (MNNVL + P2P) — they are in
            # _EXPERT_WEIGHT_NAMES.  The small ones (fc31_alpha, fc2_alpha)
            # replace the rank's shard with the full-sized tensor here.
            for pname, full in layer_scalars.params.items():
                device = getattr(experts_module, pname).device
                setattr(
                    experts_module,
                    pname,
                    nn.Parameter(full.to(device), requires_grad=False),
                )
                logger.debug(
                    f"[DWDP Setup] Layer {layer_idx}: installed the full {pname} "
                    f"({full.shape[0]} experts)"
                )

            # --- Per-layer activation scales: rank 0's copy, in place ---
            # In place, so whatever already holds the parameter sees the same value on
            # every rank.
            for pname, value in layer_scalars.act_scales.items():
                param = getattr(experts_module, pname)
                param.data.copy_(value.to(device=param.device, dtype=param.dtype))
        _rebuild_quant_scales(experts_module, layer_idx)


def _scatter_shards_to_full(
    *,
    shards: List[torch.Tensor],
    peer_ranges: PeerRanges,
    num_experts_total: int,
    ref: torch.Tensor,
) -> torch.Tensor:
    """Reconstruct a full ``(num_experts_total, *trailing)`` tensor from
    per-peer shards using ``peer_ranges`` as the index map.

    Each peer's shard has shape ``(count, *trailing)``, where ``count`` is
    at least ``end - start`` of the peer's range; counts may differ between
    peers (balanced partition, capped last rank).  The prefix
    ``shard[:end - start]`` is placed at ``full[start:end]``.  For redundancy
    (overlapping ranges) writes overwrite — ``_exchange_expert_scalars``
    checks that the values agree, so the result is deterministic.

    Args:
        shards: Per-peer tensors as returned by ``comm.allgather``.
        peer_ranges: Per-peer ``(start, end)``.
        num_experts_total: Global expert count (= full tensor's first dim).
        ref: Reference tensor for dtype + device.
    """
    if not shards:
        raise ValueError("shards must be non-empty")
    trailing_shape = tuple(shards[0].shape[1:])
    full = torch.zeros(
        (num_experts_total, *trailing_shape),
        dtype=ref.dtype,
        device=ref.device,
    )
    for peer_rank, (start, end) in enumerate(peer_ranges):
        valid = end - start
        if valid <= 0:
            continue
        full[start:end] = shards[peer_rank][:valid].to(ref.device)
    return full


# ---------------------------------------------------------------------------
# Internal utilities
# ---------------------------------------------------------------------------


def _rebuild_quant_scales(experts_module: nn.Module, layer_idx: int) -> None:
    """Rebuild the ``quant_scales`` NamedTuple to reference the current Parameters.

    After setup, the quant_scales NamedTuple may hold stale references:
      - w3_w1_weight_scale / w2_weight_scale: handled by DWDP Transport. Their
        original storage is freed (resized to 0), but the Parameter objects
        still exist. At runtime, wait_and_bind() does param.data = composite_VA,
        which swaps in the full tensor. The NamedTuple references the same
        Parameter, so it sees the updated data.
      - fc31_alpha / fc2_alpha: put together from every rank's shard by
        _exchange_expert_scalars; fixup_moe_backends replaces these with new
        full-sized Parameter objects via setattr(), so the old NamedTuple
        references are stale and must be rebuilt.

    This function reconstructs the NamedTuple from the module's current
    Parameter state so all fields point to the live objects.

    Args:
        experts_module: The MoE backend module whose ``quant_scales`` to rebuild.
        layer_idx: Layer index (for log messages only).
    """
    qs = getattr(experts_module, "quant_scales", None)
    if qs is None:
        return

    # Detect FusedMoEQuantScalesNVFP4 by its field names.
    qs_type = type(qs)
    field_names = getattr(qs_type, "_fields", ())
    if "fc1_weight_block" in field_names and "fc2_weight_block" in field_names:
        # FusedMoEQuantScalesNVFP4:
        #   fc1_act_global   = module.fc31_input_scale   (scalar, not EP-sharded)
        #   fc1_weight_block = module.w3_w1_weight_scale (Transport: empty now, filled at runtime)
        #   fc1_global       = module.fc31_alpha         (exchanged: full-sized)
        #   fc2_act_global   = module.fc2_input_scale    (scalar, not EP-sharded)
        #   fc2_weight_block = module.w2_weight_scale    (Transport: empty now, filled at runtime)
        #   fc2_global       = module.fc2_alpha          (exchanged: full-sized)
        new_qs = qs_type(
            fc1_act_global=experts_module.fc31_input_scale,
            fc1_weight_block=experts_module.w3_w1_weight_scale,
            fc1_global=experts_module.fc31_alpha,
            fc2_act_global=experts_module.fc2_input_scale,
            fc2_weight_block=experts_module.w2_weight_scale,
            fc2_global=experts_module.fc2_alpha,
        )
        experts_module.quant_scales = new_qs
        logger.info(
            f"[DWDP Setup] Layer {layer_idx}: rebuilt FusedMoEQuantScalesNVFP4 "
            f"(weight_scale params via Transport, alpha params via the exchange)"
        )
    else:
        logger.debug(
            f"[DWDP Setup] Layer {layer_idx}: quant_scales type {qs_type.__name__} "
            f"not handled by _rebuild_quant_scales (fields={field_names}); skipping."
        )


def _get_decoder_model(model: nn.Module) -> nn.Module:
    """Navigate from top-level CausalLM to the decoder model with .layers.

    Handles DeepseekV3ForCausalLM -> .model (DeepseekV3Model).
    Falls back to searching common attribute names.

    Args:
        model: Top-level model.

    Returns:
        The decoder model that has a ``.layers`` ModuleList.

    Raises:
        RuntimeError: If the decoder model cannot be found.
    """
    # Direct attribute: model.model
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model

    # Maybe the model itself has layers
    if hasattr(model, "layers"):
        return model

    # Search for common patterns
    for attr_name in ("transformer", "decoder", "backbone"):
        child = getattr(model, attr_name, None)
        if child is not None and hasattr(child, "layers"):
            return child

    raise RuntimeError(
        "[DWDP Setup] Cannot find decoder model with .layers attribute. "
        f"Model type: {type(model).__name__}"
    )


def _get_configurable_moe(moe_module: Optional[nn.Module]) -> Optional[nn.Module]:
    """The ConfigurableMoE wrapper of an MoE module, if the model uses one.

    DeepSeek calls it ``experts``; Kimi K3's ``KimiK3MoERuntime`` calls the
    same thing ``routed_experts``.  Returns None when the module has neither.
    """
    if moe_module is None:
        return None
    experts = getattr(moe_module, "experts", None)
    if experts is None:
        experts = getattr(moe_module, "routed_experts", None)
    return experts


def _get_moe_and_experts(
    layer: nn.Module,
) -> Tuple[Optional[nn.Module], Optional[nn.Module]]:
    """From a decoder layer, find the MoE wrapper and its experts backend.

    The standard path for DeepSeek is:
        layer.mlp (Deepseekv3MoE) -> .experts (MoE backend)

    Kimi K3 spells the same shape differently:
        layer.block_sparse_moe (KimiK3MoERuntime) -> .routed_experts (MoE backend)

    Returns:
        Tuple of (moe_module, experts_module) where moe_module is the wrapper
        (e.g. Deepseekv3MoE) and experts_module is the backend (e.g.
        CutlassFusedMoE, ConfigurableMoE, etc.). Both may be None if the
        layer is not an MoE layer.
    """
    # K3's *dense* layers do carry an ``mlp``, but this function is only ever
    # reached for layer indices that registered themselves from
    # ConfigurableMoE.__init__, so a dense layer never gets here and the
    # ``mlp``-first order stays safe.
    mlp = getattr(layer, "mlp", None)
    if mlp is None:
        mlp = getattr(layer, "block_sparse_moe", None)
    if mlp is None:
        return None, None

    # Check if mlp itself is an MoE backend (has w3_w1_weight)
    if hasattr(mlp, "w3_w1_weight"):
        return mlp, mlp

    # Standard path: mlp.experts (K3: block_sparse_moe.routed_experts)
    experts = _get_configurable_moe(mlp)
    if experts is not None:
        # Prefer the inner backend (ConfigurableMoE wraps it)
        backend = getattr(experts, "backend", None)
        if backend is not None and hasattr(backend, "w3_w1_weight"):
            return mlp, backend
        # Fallback: direct backend without ConfigurableMoE wrapper
        if hasattr(experts, "w3_w1_weight"):
            return mlp, experts

    return None, None


def _get_gate_module(moe_module: nn.Module) -> Optional[nn.Module]:
    """Get the gate module from an MoE wrapper.

    Args:
        moe_module: The MoE wrapper (e.g. Deepseekv3MoE).

    Returns:
        Gate module with e_score_correction_bias, or None.
    """
    gate = getattr(moe_module, "gate", None)
    if gate is not None and hasattr(gate, "e_score_correction_bias"):
        return gate
    return None
