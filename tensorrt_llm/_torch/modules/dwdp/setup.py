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
3. Computes each rank's local expert range.
4. Allocates MNNVL fabric handles via ``DWDPTransport.create()``, copying
   local weights into fabric memory and freeing the originals.  Handles
   are exchanged via the DWDP MPI communicator.
5. Constructs a WeightBuffer with composite VA layout (zero-copy local
   + page-pool-backed remote double buffer).
6. Fills page-alignment edge bytes at MNNVL region boundaries.
7. Creates a DWDPWeightManager for runtime P2P prefetch scheduling.
8. Patches each MoE backend so it sees all ``num_experts`` (ep_size=1) and
   allgathers small EP-sharded parameters (e.g. e_score_correction_bias,
   fc31_alpha, fc2_alpha) via MPI.  Large scale params
   (w3_w1_weight_scale, w2_weight_scale) go through Transport (MNNVL + P2P).

The function is idempotent: calling it a second time on the same model is a
no-op (returns the existing manager stored on the model).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
from torch import nn

from tensorrt_llm.logger import logger

from .specs import LayerWeightSpecs, WeightSpec
from .transport import DWDPTransport
from .weight_buffer import WeightBuffer
from .weight_manager import DWDPWeightManager

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


def setup_dwdp(
    model: nn.Module,
    mapping,
    device_id: int,
    comm,
    layer_indices: List[int],
) -> Optional[DWDPWeightManager]:
    """Set up DWDP for a model after weight loading.

    This is the single orchestration entry point.  It must be called exactly
    once, after all model weights are loaded and before any forward pass.

    Args:
        model: The top-level causal-LM model (e.g. DeepseekV3ForCausalLM).
        mapping: ``tensorrt_llm.mapping.Mapping`` instance with DWDP fields.
        device_id: CUDA device ordinal for this rank.
        comm: mpi4py communicator scoped to the DWDP group (owned by
            DwdpManager; lifetime exceeds setup_dwdp).
        layer_indices: MoE layer indices collected by
            ``DwdpManager._registered_layers`` (SSOT).  Each index must
            correspond to an MoE layer in the model.

    Returns:
        A ready-to-use DWDPWeightManager if DWDP is enabled, ``None`` otherwise.
        The manager is also stored on ``model._dwdp_weight_manager`` so that
        a second call returns the cached instance (idempotent).
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
    logger.info(
        f"[DWDP Setup] Collected params from {len(layer_indices)} MoE layers, "
        f"weight_names={weight_names}"
    )

    # 2. Build weight specs
    layer_weight_specs = build_weight_specs(local_params, weight_names, mapping)
    logger.info(
        f"[DWDP Setup] Built weight specs for {len(layer_weight_specs)} layers"
    )

    # 3. Compute local expert range
    first_spec = _get_first_spec(layer_weight_specs)
    num_experts_total = first_spec.num_experts
    experts_per_rank = num_experts_total // mapping.dwdp_size
    local_start = mapping.dwdp_rank * experts_per_rank
    local_end = local_start + experts_per_rank
    logger.info(
        f"[DWDP Setup] Expert range: local=[{local_start}, {local_end}), "
        f"total={num_experts_total}, per_rank={experts_per_rank}"
    )

    # 4. Transport: MNNVL alloc -> copy -> free originals -> exchange handles
    logger.info("[DWDP Setup] Creating transport (MNNVL handle exchange)...")
    transport = DWDPTransport.create(
        layer_weight_specs=layer_weight_specs,
        local_params=local_params,
        comm=comm,
        dwdp_rank=mapping.dwdp_rank,
        dwdp_size=mapping.dwdp_size,
        device_id=device_id,
        local_start=local_start,
        local_end=local_end,
    )
    logger.info("[DWDP Setup] Transport created successfully.")
    free_mem, total_mem = torch.cuda.mem_get_info(device_id)
    logger.info(
        f"[DWDP Setup] GPU {device_id} memory after transport: "
        f"free={free_mem/(1024**3):.2f}GB / total={total_mem/(1024**3):.2f}GB "
        f"(used={((total_mem-free_mem)/(1024**3)):.2f}GB)"
    )

    # 5. WeightBuffer: composite VAs
    logger.info("[DWDP Setup] Creating weight buffer (composite VA layout)...")
    weight_buffer = WeightBuffer.create(
        layer_weight_specs=layer_weight_specs,
        handles=transport.get_handle_set(),
        local_start=local_start,
        local_end=local_end,
        dwdp_size=mapping.dwdp_size,
        device_id=device_id,
    )
    logger.info("[DWDP Setup] Weight buffer created successfully.")

    # 6. Fill edge bytes
    logger.info("[DWDP Setup] Filling page-alignment edge bytes...")
    fill_edge_bytes(
        weight_buffer=weight_buffer,
        peer_views=transport.get_peer_views(),
        local_start=local_start,
        local_end=local_end,
        dwdp_size=mapping.dwdp_size,
    )
    logger.info("[DWDP Setup] Edge bytes filled.")

    # 7. WeightManager
    logger.info("[DWDP Setup] Creating weight manager...")
    weight_manager = DWDPWeightManager(
        weight_buffer=weight_buffer,
        peer_views=transport.get_peer_views(),
        moe_layer_indices=layer_indices,
        weight_names=list(weight_names),
        dwdp_rank=mapping.dwdp_rank,
        dwdp_size=mapping.dwdp_size,
    )
    logger.info("[DWDP Setup] Weight manager created.")

    # 8. Patch MoE backends
    logger.info("[DWDP Setup] Patching MoE backends...")
    fixup_moe_backends(
        model,
        layer_indices,
        num_experts_total,
        comm,
        mapping.dwdp_rank,
        mapping.dwdp_size,
    )
    logger.info("[DWDP Setup] MoE backends patched.")

    # 9. Keep transport alive (its handles underpin the VA mappings)
    weight_manager._transport = transport

    # Store on both the top-level model (for idempotent retrieval) and the
    # inner decoder model (where DeepseekV3Model.forward() reads it).
    setattr(model, _DWDP_SETUP_DONE, weight_manager)
    decoder_model.dwdp_weight_manager = weight_manager

    logger.info("[DWDP Setup] Setup complete.")
    return weight_manager


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
    mapping,
) -> LayerWeightSpecs:
    """Convert collected parameters to ``LayerWeightSpecs``.

    Each local parameter has shape ``(experts_per_rank, ...)``.  The full shape
    is ``(num_experts_total, ...)`` where ``num_experts_total = experts_per_rank * dwdp_size``.

    Args:
        local_params: Dict from ``collect_moe_params``.
        weight_names: Weight names list.
        mapping: Mapping with dwdp_size.

    Returns:
        LayerWeightSpecs: ``Dict[layer_idx, Dict[weight_name, WeightSpec]]``.
    """
    layer_weight_specs: LayerWeightSpecs = {}
    dwdp_size = mapping.dwdp_size

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
            experts_per_rank = chunk_shape[0]
            num_experts_total = experts_per_rank * dwdp_size
            full_shape = (num_experts_total,) + chunk_shape[1:]
            specs[wname] = WeightSpec(
                num_experts=num_experts_total,
                chunk_shape=chunk_shape,
                full_shape=full_shape,
                dtype=param.dtype,
            )
        layer_weight_specs[layer_idx] = specs

    return layer_weight_specs


# ---------------------------------------------------------------------------
# Helper: fill edge bytes
# ---------------------------------------------------------------------------


def fill_edge_bytes(
    weight_buffer: WeightBuffer,
    peer_views: Dict[Tuple[int, int, str], torch.Tensor],
    local_start: int,
    local_end: int,
    dwdp_size: int,
) -> None:
    """Fill page-alignment edge bytes at MNNVL region boundaries.

    When local expert data does not align to page boundaries, the MNNVL pages
    contain edge regions that must be filled with the correct peer data.  This
    happens at most twice per (layer, weight): a *leading edge* before
    ``local_start`` and a *trailing edge* after ``local_end``.

    The data is read from peer MNNVL tensor views and written into the local
    MNNVL handle's mapped VA.  This is a one-time setup operation.

    Args:
        weight_buffer: WeightBuffer with composite VAs.
        peer_views: ``{(peer_rank, layer_idx, name): tensor}`` from Transport.
        local_start: First local expert index (inclusive).
        local_end: Last local expert index (exclusive).
        dwdp_size: Number of DWDP ranks.
    """
    experts_per_rank = local_end - local_start
    if experts_per_rank == 0:
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
            # These belong to experts just before local_start.
            if edge_info.leading_edge > 0 and local_start > 0:
                # The leading edge is within the expert just before local_start.
                # We need the tail of the previous expert from the peer that owns it.
                prev_expert = local_start - 1
                peer_rank = prev_expert // experts_per_rank
                peer_local_offset = prev_expert % experts_per_rank

                peer_key = (peer_rank, layer_idx, name)
                if peer_key in peer_views:
                    src = peer_views[peer_key]
                    # The edge bytes correspond to the tail portion of prev_expert's
                    # data that spills into our MNNVL page.  Since the full_tensor's
                    # expert dimension is aligned by the composite VA, we can simply
                    # copy the corresponding expert slice.
                    full_tensor[prev_expert].copy_(src[peer_local_offset])
                    logger.debug(
                        f"[DWDP Setup] Layer {layer_idx}/{name}: filled leading edge "
                        f"from peer {peer_rank}, expert {prev_expert}"
                    )

            # Trailing edge: bytes in [local_end_bytes, page_end).
            # These belong to experts just after local_end.
            if edge_info.trailing_edge > 0 and local_end < full_tensor.shape[0]:
                next_expert = local_end
                peer_rank = next_expert // experts_per_rank
                peer_local_offset = next_expert % experts_per_rank

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
# Helper: fixup MoE backends
# ---------------------------------------------------------------------------


def fixup_moe_backends(
    model: nn.Module,
    moe_layer_indices: List[int],
    num_experts_total: int,
    comm,
    dwdp_rank: int,
    dwdp_size: int,
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

    Additionally, the gate's ``e_score_correction_bias`` (which was loaded as
    an EP shard) is allgathered so every rank has the full bias vector.

    Args:
        model: Top-level CausalLM model.
        moe_layer_indices: MoE layer indices to patch.
        num_experts_total: Total number of experts across all ranks.
        comm: DWDP MPI communicator used for small tensor allgathers.
        dwdp_rank: This rank's DWDP rank.
        dwdp_size: Total DWDP ranks.
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
        configurable_moe = getattr(layer.mlp, "experts", None)
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

        logger.debug(
            f"[DWDP Setup] Layer {layer_idx}: patched "
            f"{'ConfigurableMoE + ' if len(targets) > 1 else ''}backend "
            f"ep_size={old_ep_size}->{1}, ep_rank={old_ep_rank}->{0}, "
            f"slots=[{old_slot_start},{old_slot_end})->[0,{num_experts_total})"
        )

        # --- Allgather e_score_correction_bias ---
        gate_module = _get_gate_module(moe_module)
        if gate_module is not None:
            _allgather_e_score_correction_bias(
                gate_module, layer_idx, dwdp_rank, dwdp_size,
                num_experts_total, comm,
            )

        # --- Allgather small expert scale/dequant parameters via MPI ---
        # Large scale params (w3_w1_weight_scale, w2_weight_scale) are already
        # handled by Transport (MNNVL + P2P) — they are in _EXPERT_WEIGHT_NAMES.
        # Small params (fc31_alpha, fc2_alpha) are allgathered here.
        # Pass experts_per_rank explicitly because w3_w1_weight has already been
        # moved to the MNNVL composite VA (size num_experts_total, not per_rank).
        experts_per_rank = num_experts_total // dwdp_size
        _allgather_expert_scales(
            experts_module, layer_idx, dwdp_rank, dwdp_size, comm,
            experts_per_rank=experts_per_rank,
        )
        _rebuild_quant_scales(experts_module, layer_idx)


def _allgather_e_score_correction_bias(
    gate_module: nn.Module,
    layer_idx: int,
    dwdp_rank: int,
    dwdp_size: int,
    num_experts_total: int,
    comm,
) -> None:
    """Allgather the EP-sharded e_score_correction_bias to get the full vector.

    Each rank has a shard of size ``num_experts // ep_size``.  After allgather
    each rank holds the full ``(num_experts,)`` bias vector.

    If the bias is already full-sized (not EP-sharded), this is a no-op.
    The gate's e_score_correction_bias covers ALL experts for routing and
    may not be EP-sharded even when expert weights are.

    Args:
        gate_module: The DeepseekV3Gate (or compatible) module with
            ``e_score_correction_bias`` parameter.
        layer_idx: Layer index (for log messages only).
        dwdp_rank: This rank's DWDP rank.
        dwdp_size: Total DWDP ranks.
        num_experts_total: Total number of experts across all ranks.
        comm: DWDP MPI communicator.
    """
    bias_param = getattr(gate_module, "e_score_correction_bias", None)
    if bias_param is None:
        logger.debug(
            f"[DWDP Setup] Layer {layer_idx}: no e_score_correction_bias found"
        )
        return

    bias_data = bias_param.data if isinstance(bias_param, nn.Parameter) else bias_param
    bias_size = bias_data.shape[0]
    expected_shard_size = num_experts_total // dwdp_size

    logger.info(
        f"[DWDP Setup] Layer {layer_idx}: e_score_correction_bias "
        f"type={type(bias_param).__name__}, shape={bias_data.shape}, "
        f"device={bias_data.device}, "
        f"num_experts_total={num_experts_total}, expected_shard={expected_shard_size}"
    )

    # The gate's e_score_correction_bias covers ALL experts for routing.
    # It may or may not be EP-sharded depending on the model.
    # Only allgather if the bias is actually sharded (size == shard size).
    if bias_size == num_experts_total:
        logger.info(
            f"[DWDP Setup] Layer {layer_idx}: e_score_correction_bias "
            f"already full-sized ({bias_size}), skipping allgather"
        )
        return
    if bias_size != expected_shard_size:
        logger.warning(
            f"[DWDP Setup] Layer {layer_idx}: e_score_correction_bias "
            f"size={bias_size} doesn't match expected shard size "
            f"({expected_shard_size}) or full size ({num_experts_total}), "
            f"skipping allgather"
        )
        return

    # Allgather local shards via MPI.  Move to CPU first so the allgather
    # doesn't require CUDA-aware MPI.
    local_shard = bias_data.cpu().contiguous()
    all_shards = comm.allgather(local_shard)
    full_bias = torch.cat(all_shards, dim=0).to(bias_data.device)

    if isinstance(bias_param, nn.Parameter):
        # Re-create parameter with the full size
        gate_module.e_score_correction_bias = nn.Parameter(
            full_bias, requires_grad=False
        )
    else:
        bias_param.data = full_bias

    logger.info(
        f"[DWDP Setup] Layer {layer_idx}: allgathered e_score_correction_bias "
        f"({expected_shard_size} -> {full_bias.shape[0]})"
    )


def _allgather_expert_scales(
    experts_module: nn.Module,
    layer_idx: int,
    dwdp_rank: int,
    dwdp_size: int,
    comm,
    experts_per_rank: Optional[int] = None,
) -> None:
    """Allgather small EP-sharded quantization scale/dequant parameters via MPI.

    Only handles SMALL scale params (e.g. ``fc31_alpha``, ``fc2_alpha``)
    that are NOT in ``_EXPERT_WEIGHT_NAMES``.  Large scale params
    (``w3_w1_weight_scale``, ``w2_weight_scale``) are handled by DWDP
    Transport (MNNVL + P2P).

    Heuristic: any ``nn.Parameter`` on ``experts_module`` whose name contains
    ``"scale"``, ``"scaling_factor"``, ``"dequant"``, or ``"alpha"``, whose
    ``shape[0] == experts_per_rank``, and which is NOT in
    ``_EXPERT_WEIGHT_NAMES``, is treated as a small EP-sharded tensor.

    Args:
        experts_module: The MoE backend module (e.g. CutlassFusedMoE).
        layer_idx: Layer index (for log messages only).
        dwdp_rank: This rank's DWDP rank.
        dwdp_size: Total DWDP ranks.
        comm: DWDP MPI communicator.
        experts_per_rank: Number of experts per DWDP rank.  If None, inferred
            from ``w3_w1_weight.shape[0]`` (only valid before DWDP weight
            buffer replaces the original tensor).
    """
    _SCALE_KEYWORDS = ("scale", "scaling_factor", "dequant", "alpha")

    # Discover EP-sharded scale parameters.
    if experts_per_rank is None:
        # Fallback: infer from main weight (only valid before DWDP buffer swap).
        ref_param = getattr(experts_module, "w3_w1_weight", None)
        if ref_param is None:
            return
        ref_data = ref_param.data if isinstance(ref_param, nn.Parameter) else ref_param
        experts_per_rank = ref_data.shape[0]
    if experts_per_rank == 0:
        return

    for pname, param in list(experts_module.named_parameters()):
        # Only look at direct parameters (no '.' in name — skip sub-modules).
        if "." in pname:
            continue
        if not any(kw in pname for kw in _SCALE_KEYWORDS):
            continue
        # Large scale params in _EXPERT_WEIGHT_NAMES are handled by Transport
        # (MNNVL + P2P), not MPI allgather.
        if pname in _EXPERT_WEIGHT_NAMES:
            continue

        pdata = param.data
        if pdata.ndim == 0 or pdata.shape[0] != experts_per_rank:
            # Scalar or non-EP-sharded — skip.
            continue

        local_shard = pdata.cpu().contiguous()
        logger.info(
            f"[DWDP Setup] Layer {layer_idx}: allgathering {pname} "
            f"shape={local_shard.shape} dtype={local_shard.dtype} via MPI"
        )

        all_shards = comm.allgather(local_shard)
        full_tensor = torch.cat(all_shards, dim=0).to(pdata.device)

        # Replace parameter with full-sized version.
        setattr(
            experts_module,
            pname,
            nn.Parameter(full_tensor, requires_grad=False),
        )

        logger.info(
            f"[DWDP Setup] Layer {layer_idx}: allgathered {pname} "
            f"({experts_per_rank} -> {full_tensor.shape[0]})"
        )


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
      - fc31_alpha / fc2_alpha: allgathered via MPI. _allgather_expert_scales
        replaces these with new full-sized Parameter objects via setattr(), so the
        old NamedTuple references are stale and must be rebuilt.

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
        #   fc1_global       = module.fc31_alpha         (MPI allgathered: full-sized)
        #   fc2_act_global   = module.fc2_input_scale    (scalar, not EP-sharded)
        #   fc2_weight_block = module.w2_weight_scale    (Transport: empty now, filled at runtime)
        #   fc2_global       = module.fc2_alpha          (MPI allgathered: full-sized)
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
            f"(weight_scale params via Transport, alpha params via MPI allgather)"
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


def _get_moe_and_experts(
    layer: nn.Module,
) -> Tuple[Optional[nn.Module], Optional[nn.Module]]:
    """From a decoder layer, find the MoE wrapper and its experts backend.

    The standard path for DeepSeek is:
        layer.mlp (Deepseekv3MoE) -> .experts (MoE backend)

    Returns:
        Tuple of (moe_module, experts_module) where moe_module is the wrapper
        (e.g. Deepseekv3MoE) and experts_module is the backend (e.g.
        CutlassFusedMoE, ConfigurableMoE, etc.). Both may be None if the
        layer is not an MoE layer.
    """
    mlp = getattr(layer, "mlp", None)
    if mlp is None:
        return None, None

    # Check if mlp itself is an MoE backend (has w3_w1_weight)
    if hasattr(mlp, "w3_w1_weight"):
        return mlp, mlp

    # Standard path: mlp.experts
    experts = getattr(mlp, "experts", None)
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


def _get_first_spec(layer_weight_specs: LayerWeightSpecs) -> WeightSpec:
    """Get the first WeightSpec from layer_weight_specs.

    Args:
        layer_weight_specs: The weight specs dict.

    Returns:
        First WeightSpec found.

    Raises:
        ValueError: If specs are empty.
    """
    for layer_idx, weight_specs in layer_weight_specs.items():
        for name, spec in weight_specs.items():
            return spec
    raise ValueError("layer_weight_specs is empty")
