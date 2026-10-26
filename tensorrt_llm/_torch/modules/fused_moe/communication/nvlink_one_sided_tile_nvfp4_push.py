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

"""Opt-in push-combine overlay for the packed-NVFP4 tile-NVFP4 V2 family.

``NVLinkOneSidedTilePushCombine`` subclasses ``NVLinkOneSidedTileNvfp4V2`` and
reuses its dispatch / workspace / layout machinery; its combine drives the
push-combine Torch ops (``moe_a2a_push_*``: D0 generation-open, K1 producer
push, K2 producer publish, K3 consumer gate, K4 top-k reduce, and the
sequential ``push_combine`` that packages K1->K2->K3->K4).

Contract (fail-closed, mirrors the C++ push op surface in
``moeAlltoAllTileNvfp4V2PushOp.cpp``):

  * ONLY the packed-interleaved [Q|SF] recipe (``nvfp4_packed``, recipe id 3);
    the push landing region C is a STORAGE ALIAS onto the packed SF
    plane (``cft_region_c_alias_v1``).
  * EP in {2, 4, 8, 16, 32}; records_per_token == min(8, ep_size).
  * top_k == 8 and BF16 / H7168 activations. PDL follows the common
    TensorRT-LLM ``TRTLLM_ENABLE_PDL`` policy, including its default when the
    variable is unset.
  * a push build id (``tile_push_combine:v2:<sha256>``); other tile-TMA build
    ids are rejected BY NAME so no DSO can substitute for another.

The combine runs on one owner stream over one cached symmetric workspace under
the common TensorRT-LLM PDL policy.  A device-side push generation is advanced
at the dispatch tail (D0 RMW, replay-safe) rather than a host scalar baked into
a captured graph.  The same D0 launch also advances the NVLinkOneSided A2A
round flag by one (the combine half of the round that the stock prepare-combine
kernel would otherwise advance), so the stock dispatch protocol -- including
its round-parity double-buffered ``recv_counters`` that K1 reads on device --
stays intact.  Compatible same-process strategy objects share the validated
workspace/control plane through reference counts; their communication phases
must remain serialized, as in the layer-wise runner.

CUDA-graph capture contract (generation replay).  The push generation is a
DEVICE word throughout: the D0 dispatch-tail advance
(``moe_a2a_push_d0_generation_open``) computes the opened generation on-device
via an RMW, and the combine ops (``moe_a2a_push_k23`` /
``_k4_topk_reduce`` and the packaged ``moe_a2a_push_combine``) READ that same
device generation word on-device -- no host ``generation`` scalar is passed into
any op.  Because every op re-reads (or re-advances, for D0) the device word on
each replay, dispatch's D0 and the combine ops may be co-captured into one
replayable graph without the frozen-scalar mismatch that a baked host generation
would cause: K2 publishes ``ready[C][P]`` at the replay-advanced device
generation and K3 gates on the same replay-advanced device generation.  The
host-side ``_push_generation`` shadow is a bookkeeping mirror only (used for the
Python phase/identity guards); it never enters an op argument list.

Reference-counted workspace reuse.  The push landing region C is a storage
ALIAS onto the packed SF plane, so Push and a CFT PULL combine must
never drive it concurrently.  Compatible ``NVLinkOneSidedTilePushCombine``
objects in one process may nevertheless coexist: construction validates the
full recipe/layout identity, reuses the one cached symmetric control plane, and
increments reference counts.  This supports the decode and prefill strategy
objects created by the layer-wise runner.  It does not authorize concurrent
Push phases; callers must serialize communication on the owner stream.
"""

from __future__ import annotations

import os
import threading
from typing import Optional

import torch

from tensorrt_llm._mnnvl_utils import MnnvlMemory
from tensorrt_llm.mapping import Mapping

from .nvlink_one_sided import NVLinkOneSided
from .nvlink_one_sided_tile_nvfp4_v2 import NVLinkOneSidedTileNvfp4V2
from .tile_nvfp4_v2_layout import (
    HIDDEN_ELEMENTS,
    PUSH_CONTROL_SLAB_BYTES,
    PUSH_FIRST_GENERATION,
    PUSH_K4_SELECTED_SHARED_STRIDE_DEFAULT_BYTES,
    PUSH_RECIPE,
    TILE_TMA_RECIPE_IDS,
    MoeA2ACombineTileFormat,
    TileNvfp4V1Layout,
    normalize_tile_tma_recipe,
    push_identity_fields,
    workspace_extra_bytes,
)

# TRTLLM_PUSH_D0_FOLD=1 (default off): skip the D0 generation-open launch at the
# dispatch tail and let the paired combine's K23 open the generation (and advance
# the A2A round flag) right before it publishes. Only this rank's K23 reads its own
# generation word, so K23 publishes and gates on exactly the value it would see
# after D0; one kernel launch less per MoE layer. The choice is made per strategy
# object, recorded per dispatch and honoured by that dispatch's combine.
# Kernel-side options of the same DSO (read in C++, all default off):
#   TRTLLM_PUSH_K23_PDL=1       launch K23 with PDL after K1 (overlaps its launch).
#   TRTLLM_PUSH_K4_V2=1         K4 v2 for EP >= 4 (lighter prologue, skips invalid
#                               route slots, bitwise-identical output).
#   TRTLLM_PUSH_K23_DEADLINE=1  trap with a diagnostic instead of spinning forever
#                               when a peer never publishes (TRTLLM_PUSH_K23_DEADLINE_CYCLES=N
#                               sets the budget in SM cycles).
PUSH_D0_FOLD_ENV = "TRTLLM_PUSH_D0_FOLD"
_D0_FOLDED_KEY = "push_d0_folded"


class NVLinkOneSidedTilePushCombine(NVLinkOneSidedTileNvfp4V2):
    """Push-combine tile-NVFP4 V2 selector.

    Reuses the ``NVLinkOneSidedTileNvfp4V2`` dispatch/workspace/layout base and
    re-points the combine at the ``moe_a2a_push_*`` op family.  The wire recipe
    is fixed to ``nvfp4_packed`` (the one legal push target) and EP is restricted
    to {2, 4, 8, 16, 32}.
    """

    TILE_PUSH_SELECTOR = "tile_push_combine_v2"
    BUILD_ID_PREFIX = "tile_push_combine:v2:"
    # Other tile-TMA build ids are rejected BY NAME so a push selector can never
    # bind a non-push DSO.
    UNIFIED_PULL_BUILD_ID_PREFIX = "tile_tma_unified:v1:"
    SEALED_ARM_BUILD_ID_PREFIX = "tile_nvfp4_v2_"
    REQUIRED_TOP_K = 8
    SUPPORTED_EP_SIZES = (2, 4, 8, 16, 32)

    # Phase machine: dispatched -> generation_open -> pushing -> published ->
    # gated -> idle.  ``dispatched`` is the transient state published by the base
    # dispatch; ``generation_open`` is reached after the D0 device advance.
    _PHASE_IDLE = "idle"
    _PHASE_DISPATCHING = "dispatching"
    _PHASE_DISPATCH_FAILED = "dispatch_failed"
    _PHASE_DISPATCHED = "dispatched"
    _PHASE_GENERATION_OPEN = "generation_open"
    _PHASE_PUSHING = "pushing"
    _PHASE_PUBLISHED = "published"
    _PHASE_GATED = "gated"
    _PHASE_COMBINE_FAILED = "combine_failed"

    # Process-wide symmetric control-plane cache (the raw-u64 flag plane), keyed
    # independently from the base symmetric workspace.
    _CONTROL_PLANES: dict = {}
    _CONTROL_PLANE_LOCK = threading.RLock()

    # ------------------------------------------------------------------
    # Workspace sizing: the push path is the packed recipe, so the recipe-keyed
    # second plane is the packed [Q|SF] plane (4480 B/slot), NOT the 896-B split
    # SF plane.  This makes the physical workspace byte-identical to the
    # nvfp4_packed PULL combine (the storage-alias target).
    # ------------------------------------------------------------------

    @staticmethod
    def calculate_required_workspace_size(
        ep_size: int,
        top_k: int,
        max_num_tokens: int,
        hidden_size: int,
        dtype: torch.dtype,
        eplb_stats_num_experts: Optional[int] = None,
        extra_payload_bytes_per_token: int = 0,
        recipe: str = PUSH_RECIPE,
    ) -> int:
        from tensorrt_llm.math_utils import pad_up

        canonical_recipe = normalize_tile_tma_recipe(recipe)
        if canonical_recipe != PUSH_RECIPE:
            raise ValueError(
                f"push-combine requires the {PUSH_RECIPE!r} recipe; got {canonical_recipe!r}"
            )
        legacy_bytes = NVLinkOneSided.calculate_required_workspace_size(
            ep_size,
            top_k,
            max_num_tokens,
            hidden_size,
            dtype,
            eplb_stats_num_experts=eplb_stats_num_experts,
            extra_payload_bytes_per_token=extra_payload_bytes_per_token,
        )
        return pad_up(
            legacy_bytes + workspace_extra_bytes(ep_size, max_num_tokens, recipe=canonical_recipe),
            128,
        )

    def __init__(self, mapping: Mapping, recipe: str = PUSH_RECIPE, *args, **kwargs):
        canonical_recipe = normalize_tile_tma_recipe(recipe)
        if canonical_recipe != PUSH_RECIPE:
            raise ValueError(
                "push-combine is defined only for the packed-interleaved "
                f"[Q|SF] recipe ({PUSH_RECIPE!r}); got {canonical_recipe!r}"
            )
        ep_size = int(mapping.moe_ep_size)
        if ep_size not in self.SUPPORTED_EP_SIZES:
            raise ValueError(
                f"push-combine supports EP in {self.SUPPORTED_EP_SIZES}; got EP{ep_size}"
            )
        top_k = kwargs.get("top_k")
        if top_k is None or int(top_k) != self.REQUIRED_TOP_K:
            raise ValueError(f"push-combine requires top_k={self.REQUIRED_TOP_K}; got {top_k}")
        kernel_build_id = kwargs.get("kernel_build_id")
        if isinstance(kernel_build_id, str) and (
            kernel_build_id.startswith(self.SEALED_ARM_BUILD_ID_PREFIX)
            or kernel_build_id.startswith(self.UNIFIED_PULL_BUILD_ID_PREFIX)
        ):
            raise ValueError(
                "push-combine refuses non-push build ids (sealed per-arm "
                f"{self.SEALED_ARM_BUILD_ID_PREFIX}* or incumbent unified PULL "
                f"{self.UNIFIED_PULL_BUILD_ID_PREFIX}*); the push DSO and the "
                "PULL/sealed DSOs can never substitute for each other"
            )
        if not isinstance(kernel_build_id, str) or not kernel_build_id.startswith(
            self.BUILD_ID_PREFIX
        ):
            raise ValueError(
                "push-combine kernel_build_id must bind the push target and DSO "
                f"SHA ({self.BUILD_ID_PREFIX}<sha256>)"
            )

        self._push_recipe = canonical_recipe
        self._push_recipe_id = int(TILE_TMA_RECIPE_IDS[canonical_recipe])
        self._push_k4_selected_shared_stride_bytes = int(
            PUSH_K4_SELECTED_SHARED_STRIDE_DEFAULT_BYTES
        )
        # A host shadow of the DEVICE push generation (advanced on-device by D0).
        self._push_generation = int(PUSH_FIRST_GENERATION) - 1
        self._push_owner_token = None
        self._control_plane_key = None
        self._fold_d0 = os.environ.get(PUSH_D0_FOLD_ENV) == "1"

        super().__init__(
            mapping,
            *args,
            combine_tile_format=MoeA2ACombineTileFormat.TILE_NVFP4_V1,
            **kwargs,
        )

        # Side-DSO metainfo: the push ops validate a fixed 10-field metainfo
        # (see the C++ op).
        self._push_rc11_metainfo = self._make_rc11_side_dso_metainfo(self.moe_a2a_metainfo)
        self.tile_push_selector = self.TILE_PUSH_SELECTOR

        # Recipe-qualified workspace-reuse key + the push identity fields.  The
        # push identity is a CUDA-graph / reuse-guard concern only; it never
        # enters the base NVLinkOneSided physical-allocation key.
        self._tile_workspace_key = self._tile_workspace_key + (
            "tile_push_combine",
            canonical_recipe,
            self._push_recipe_id,
            workspace_extra_bytes(
                self.ep_size, self.max_num_tokens_per_rank, recipe=canonical_recipe
            ),
            push_identity_fields(
                k4_selected_shared_stride_bytes=(self._push_k4_selected_shared_stride_bytes)
            ),
        )
        workspace_record = self._workspace_state
        if workspace_record is None:
            raise RuntimeError("push-combine workspace was not initialized")
        cached_key = workspace_record.setdefault(
            "tile_push_combine_workspace_key", self._tile_workspace_key
        )
        if cached_key != self._tile_workspace_key:
            raise RuntimeError(
                "refusing to reuse a push-combine workspace with a different "
                "recipe/layout/identity key"
            )

        # Reference-counted Push ownership over one aliased physical workspace.
        # Compatible same-process Push strategies may coexist and share the
        # cached control plane, but their communication phases must be serialized.
        # A CFT PULL owner remains mutually exclusive at execution.
        self._claim_workspace_owner(workspace_record)

        # Allocate (or reuse) the symmetric raw-u64 control plane.  Release the
        # owner token if control-plane init fails, so a partially-constructed
        # instance never permanently blocks the aliased workspace.
        try:
            self._init_control_plane()
        except Exception:
            self._release_workspace_owner()
            raise

    # ------------------------------------------------------------------
    # Owner token: reference-counted reuse of one aliased physical workspace.
    # ------------------------------------------------------------------

    def _claim_workspace_owner(self, workspace_record: dict) -> None:
        # REFERENCE-COUNTED single-workspace ownership.  A process may legitimately
        # construct the push strategy more than once (e.g. one MoE layer object per
        # decoder layer, or the layer-wise benchmark's decode and prefill runners).
        # All same-process strategies share the ONE cached symmetric control plane
        # (see ``_CONTROL_PLANES``, keyed by ep_rank/ep_size); a later construction
        # REUSES that plane (never re-inits it) and its identity was already
        # verified against ``_tile_workspace_key`` above.  They never drive push
        # comm concurrently, and cross-process aliasing is impossible (the token
        # lives in the per-process NVLinkOneSided workspace record).  So we
        # refcount live owners instead of rejecting the second one.
        token = object()
        workspace_record["tile_push_combine_owner_count"] = (
            workspace_record.get("tile_push_combine_owner_count", 0) + 1
        )
        workspace_record["tile_push_combine_owner_token"] = token
        self._push_owner_token = token

    def _release_workspace_owner(self) -> None:
        token = getattr(self, "_push_owner_token", None)
        if token is None:
            return
        workspace_record = getattr(self, "_workspace_state", None)
        if workspace_record:
            count = max(0, workspace_record.get("tile_push_combine_owner_count", 0) - 1)
            workspace_record["tile_push_combine_owner_count"] = count
            if count == 0 and (workspace_record.get("tile_push_combine_owner_token") is token):
                workspace_record["tile_push_combine_owner_token"] = None
        self._push_owner_token = None

    # ------------------------------------------------------------------
    # Symmetric raw-u64 control plane (the push flag ABI plane).
    # ------------------------------------------------------------------

    def _init_control_plane(self) -> None:
        # ONE symmetric MNNVL tensor [ep_size, kPushControlSlabBytes]: each rank's
        # segment is exactly one compact_row_v2 (272 B), so the strided tensor's
        # stride(0) IS the symmetric remote hop (control_peer_base[C] =
        # control_base + C*stride(0)).  The FULL strided tensor is stored and
        # passed to the ops (like self.workspace); the per-rank hop replicates the
        # payload plane's peer-base arithmetic exactly.
        segment_bytes = int(PUSH_CONTROL_SLAB_BYTES)
        key = (self.ep_rank, self.ep_size, segment_bytes)
        with NVLinkOneSidedTilePushCombine._CONTROL_PLANE_LOCK:
            record = NVLinkOneSidedTilePushCombine._CONTROL_PLANES.get(key)
            if record is None:
                MnnvlMemory.initialize()
                mem = MnnvlMemory(self.mapping, segment_bytes)
                strided = mem.as_torch_strided_tensor(torch.uint8)
                # Establish the cleared bootstrap state (ready[]/generation/gate
                # all zero) on device via the push bootstrap op (clears this
                # rank's OWN segment), then drain the device and take a COLLECTIVE
                # barrier so no peer reads a flag before every rank has cleared
                # its own row.  The barrier is host-collective and MUST stay
                # OUTSIDE any CUDA-graph capture.
                torch.ops.trtllm.moe_a2a_push_control_bootstrap(
                    strided,
                    int(self.ep_rank),
                    int(self.ep_size),
                )
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                MnnvlMemory.get_comm(self.mapping).barrier()
                record = {
                    "mnnvl_mem": mem,
                    "strided": strided,
                    "segment_bytes": segment_bytes,
                    "refcount": 0,
                }
                NVLinkOneSidedTilePushCombine._CONTROL_PLANES[key] = record
            record["refcount"] += 1
        self._control_plane_key = key
        self._control_plane = record["strided"]

    def _release_control_plane(self) -> None:
        key = getattr(self, "_control_plane_key", None)
        if key is None:
            return
        with NVLinkOneSidedTilePushCombine._CONTROL_PLANE_LOCK:
            record = NVLinkOneSidedTilePushCombine._CONTROL_PLANES.get(key)
            if record is not None:
                record["refcount"] -= 1
                if record["refcount"] <= 0:
                    NVLinkOneSidedTilePushCombine._CONTROL_PLANES.pop(key, None)
                    record.clear()
        self._control_plane_key = None
        self._control_plane = None

    # ------------------------------------------------------------------
    # Public identity surface.
    # ------------------------------------------------------------------

    @property
    def recipe(self) -> str:
        return self._push_recipe

    @property
    def recipe_id(self) -> int:
        return self._push_recipe_id

    @property
    def records_per_token(self) -> int:
        """Device dispatch axis: EP2->2, EP4->4, EP8/EP16/EP32->8."""

        return min(self.REQUIRED_TOP_K, self.ep_size)

    @property
    def push_generation(self) -> int:
        """Host shadow of the device push generation (advanced by D0)."""

        return int(self._push_generation)

    def supports_post_quant_dispatch(self) -> bool:
        return True

    # ------------------------------------------------------------------
    # Dispatch: base dispatch, republish the PACKED layout, then advance the
    # push generation on-device (D0) at the dispatch TAIL.
    # ------------------------------------------------------------------

    def dispatch(self, *args, **kwargs):
        phase = self._dispatch_state.get("phase")
        if phase != self._PHASE_IDLE:
            raise RuntimeError("push-combine dispatch requires idle state; got " + repr(phase))
        generation = int(self._push_generation) + 1
        if generation < int(PUSH_FIRST_GENERATION):
            generation = int(PUSH_FIRST_GENERATION)
        self._dispatch_state = {
            "phase": self._PHASE_DISPATCHING,
            "push_generation": generation,
        }
        try:
            result = super().dispatch(*args, **kwargs)
        except Exception:
            self._dispatch_state["phase"] = self._PHASE_DISPATCH_FAILED
            raise
        if self._dispatch_state.get("phase") != self._PHASE_DISPATCHED:
            self._dispatch_state["phase"] = self._PHASE_DISPATCH_FAILED
            raise RuntimeError("push-combine dispatch did not publish dispatched state")
        try:
            self._publish_push_recipe_layout()
            # Device-side generation advance at the dispatch TAIL.  The D0 RMW is
            # replay-safe (the opened generation is computed on-device, never a
            # host scalar), so a captured graph advances the device word on every
            # replay instead of freezing a Python int.
            # With TRTLLM_PUSH_D0_FOLD the paired combine's K23 opens it instead.
            if not self._fold_d0:
                with torch.cuda.nvtx.range("tile_push/d0_generation_open"):
                    torch.ops.trtllm.moe_a2a_push_d0_generation_open(
                        self._control_plane,
                        self.workspace,
                        self._push_rc11_metainfo,
                        int(self.ep_rank),
                        int(self.ep_size),
                    )
        except Exception:
            self._dispatch_state["phase"] = self._PHASE_DISPATCH_FAILED
            raise
        self._push_generation = generation
        self._dispatch_state["push_generation"] = generation
        self._dispatch_state[_D0_FOLDED_KEY] = self._fold_d0
        self._dispatch_state["phase"] = self._PHASE_GENERATION_OPEN
        return result

    def _publish_push_recipe_layout(self) -> None:
        """Rebuild the published descriptor for the PACKED recipe BY VALUE."""

        state = self._dispatch_state
        layout_fields = state.get("tile_nvfp4_v2_layout")
        if layout_fields is None:
            raise RuntimeError(
                "push-combine layout requires the base dispatch to have "
                "published tile_nvfp4_v2_layout"
            )
        layout = TileNvfp4V1Layout.create(
            ep_size=int(layout_fields["ep_size"]),
            runtime_max_tokens_per_rank=int(layout_fields["runtime_max_tokens_per_rank"]),
            graph_static_max_tokens_per_rank=int(layout_fields["graph_static_max_tokens_per_rank"]),
            q_plane_offset_bytes=int(layout_fields["q_plane_offset_bytes"]),
            workspace_size_per_rank_bytes=int(layout_fields["workspace_size_per_rank_bytes"]),
            recipe=self._push_recipe,
        )
        state["tile_nvfp4_v2_layout"] = layout.as_dict()
        state["combine_graph_key"] = self._compose_graph_key(layout)

    def _compose_graph_key(self, layout: TileNvfp4V1Layout) -> tuple:
        return (
            self._tile_workspace_key,
            self._dispatch_state["dispatch_payload_schema"],
            layout.graph_key(kernel_build_id=self.kernel_build_id),
            push_identity_fields(
                k4_selected_shared_stride_bytes=(self._push_k4_selected_shared_stride_bytes)
            ),
        )

    def get_combine_graph_key(self) -> tuple:
        phase = self._dispatch_state.get("phase")
        if phase not in (self._PHASE_DISPATCHED, self._PHASE_GENERATION_OPEN):
            raise RuntimeError(
                "push-combine graph key requested outside dispatched/"
                "generation_open; got " + repr(phase)
            )
        graph_key = self._dispatch_state.get("combine_graph_key")
        if graph_key is None:
            raise RuntimeError("push-combine dispatch state has no graph key")
        return graph_key

    # ------------------------------------------------------------------
    # Combine state guard.
    # ------------------------------------------------------------------

    def _require_push_state(self, expected_phase: str):
        state = self._dispatch_state
        phase = state.get("phase")
        if phase != expected_phase:
            raise RuntimeError(
                "push-combine requires " + expected_phase + " state; got " + repr(phase)
            )
        required = (
            "local_num_tokens",
            "runtime_max_tokens_per_rank",
            "tile_nvfp4_v2_layout",
            "dispatch_payload_schema",
            "combine_graph_key",
            "push_generation",
        )
        missing = [name for name in required if state.get(name) is None]
        if missing:
            raise RuntimeError("push-combine combine state is incomplete: " + str(missing))
        local_num_tokens = int(state["local_num_tokens"])
        runtime_max = int(state["runtime_max_tokens_per_rank"])
        generation = int(state["push_generation"])
        if generation != int(self._push_generation):
            raise RuntimeError("push-combine generation mismatch")
        if generation < int(PUSH_FIRST_GENERATION):
            raise RuntimeError("push-combine generation below the first live epoch")
        layout = TileNvfp4V1Layout(**state["tile_nvfp4_v2_layout"])
        if (
            layout.ep_size != int(self.ep_size)
            or layout.runtime_max_tokens_per_rank != runtime_max
            or layout.graph_static_max_tokens_per_rank != int(self.max_num_tokens_per_rank)
        ):
            raise RuntimeError("push-combine layout/runtime identity mismatch")
        if state["combine_graph_key"] != self._compose_graph_key(layout):
            raise RuntimeError("push-combine layout graph key mismatch")
        return state, local_num_tokens, runtime_max, layout, generation

    def _normalize_combine_payload(self, final_hidden_states, runtime_max):
        if final_hidden_states.dim() == 2:
            expected_shape = (int(self.ep_size) * int(runtime_max), HIDDEN_ELEMENTS)
            if tuple(int(x) for x in final_hidden_states.shape) != expected_shape:
                raise ValueError("push-combine final_hidden_states 2D shape must be [E*R,7168]")
            payload = final_hidden_states.view(self.ep_size, int(runtime_max), HIDDEN_ELEMENTS)
        elif final_hidden_states.dim() == 3:
            payload = final_hidden_states
        else:
            raise ValueError("push-combine final_hidden_states must be 2D or 3D")
        if (
            payload.dtype is not torch.bfloat16
            or tuple(payload.shape) != (self.ep_size, int(runtime_max), HIDDEN_ELEMENTS)
            or not payload.is_contiguous()
            or getattr(payload.device, "type", None) != "cuda"
        ):
            raise ValueError("push-combine requires contiguous CUDA BF16 [E,R,7168]")
        # The K1 producer reads a contiguous BF16 [tokens, 7168] source plane.
        return payload.view(int(self.ep_size) * int(runtime_max), HIDDEN_ELEMENTS)

    def _pop_common_combine_kwargs(self, kwargs) -> None:
        if kwargs.pop("use_low_precision_combine", False):
            raise ValueError("push-combine cannot use the legacy FP8 override")
        all_rank_max = kwargs.pop("all_rank_max_num_tokens", None)
        runtime_max = self._dispatch_state.get("runtime_max_tokens_per_rank")
        if all_rank_max is not None and int(all_rank_max) != int(runtime_max):
            raise ValueError("all_rank_max_num_tokens must match dispatch runtime R")

    def _pop_cft_incumbent_live(self, kwargs) -> bool:
        # The push alias is the sole live owner of region C (guarded by the
        # Python owner token); a CFT PULL owner must not be live.
        cft_incumbent_live = bool(kwargs.pop("cft_incumbent_live", False))
        if cft_incumbent_live:
            raise ValueError(
                "push-combine may not run while an incumbent-CFT PULL owner is "
                "live on region C (cft_incumbent_live must be False)"
            )
        return cft_incumbent_live

    # ------------------------------------------------------------------
    # Combine: the preferred single-op sequential push_combine (K1->K23->K4)
    # on the single owner stream under the common TensorRT-LLM PDL policy.
    # ------------------------------------------------------------------

    def combine(self, final_hidden_states: torch.Tensor, **kwargs) -> torch.Tensor:
        state, local_num_tokens, runtime_max, layout, generation = self._require_push_state(
            self._PHASE_GENERATION_OPEN
        )
        # payload_in_workspace is HONORED (not rejected).  The real generation
        # model routes FC2 output into the workspace Q band via
        # get_combine_payload_tensor_in_workspace and toggles the flag on; the
        # push DSO's resolveK1Source reads the workspace Q plane
        # (rankBase + q_plane_offset) when the flag is True and TORCH_CHECKs that
        # payload.data_ptr() == that base (exactly the view we handed out), so no
        # separate PULL workspace-view op is needed.  When the flag is False
        # (bench / backend opt-out) K1 reads the explicit payload as before.
        self._pop_common_combine_kwargs(kwargs)
        cft_incumbent_live = self._pop_cft_incumbent_live(kwargs)
        if kwargs:
            raise TypeError("unsupported push-combine kwargs: " + str(sorted(kwargs)))
        payload = self._normalize_combine_payload(final_hidden_states, runtime_max)

        state["phase"] = self._PHASE_PUSHING
        try:
            # The push kernels source the generation from the DEVICE control word
            # (advanced by the replay-safe D0 RMW), so no host ``generation``
            # scalar is baked into this call; a captured graph re-reads the
            # device generation on every replay.
            with torch.cuda.nvtx.range("tile_push/combine"):
                output = torch.ops.trtllm.moe_a2a_push_combine(
                    payload,
                    int(local_num_tokens),
                    self.workspace,
                    self._control_plane,
                    self._push_rc11_metainfo,
                    int(runtime_max),
                    int(self.max_num_tokens_per_rank),
                    int(self.ep_rank),
                    int(self.ep_size),
                    int(layout.q_plane_offset_bytes),
                    bool(self.payload_in_workspace),
                    int(self._push_recipe_id),
                    int(self._push_k4_selected_shared_stride_bytes),
                    bool(cft_incumbent_live),
                    open_generation=bool(state.get(_D0_FOLDED_KEY, False)),
                )
        except Exception:
            state["phase"] = self._PHASE_COMBINE_FAILED
            raise
        self._validate_output(output, payload, local_num_tokens)
        self._dispatch_state = {"phase": self._PHASE_IDLE}
        return output

    def combine_stepwise(self, final_hidden_states: torch.Tensor, **kwargs) -> torch.Tensor:
        """Drive K1 -> K2 -> K3 -> K4 as four ops on the single owner stream.

        Functionally equivalent to :meth:`combine` (the packaged op), but walks
        the full pushing -> published -> gated phase machine explicitly so each
        push step is independently launchable / microbenchmarkable.  Ordering is
        the stream itself (kernel boundaries): K2 publishes after K1 drains, K3
        gates on the published readiness, K4 acquires localGate[C] after K3.
        """

        state, local_num_tokens, runtime_max, layout, generation = self._require_push_state(
            self._PHASE_GENERATION_OPEN
        )
        # payload_in_workspace is HONORED (not rejected).  The real generation
        # model routes FC2 output into the workspace Q band via
        # get_combine_payload_tensor_in_workspace and toggles the flag on; the
        # push DSO's resolveK1Source reads the workspace Q plane
        # (rankBase + q_plane_offset) when the flag is True and TORCH_CHECKs that
        # payload.data_ptr() == that base (exactly the view we handed out), so no
        # separate PULL workspace-view op is needed.  When the flag is False
        # (bench / backend opt-out) K1 reads the explicit payload as before.
        self._pop_common_combine_kwargs(kwargs)
        cft_incumbent_live = self._pop_cft_incumbent_live(kwargs)
        if kwargs:
            raise TypeError("unsupported push-combine kwargs: " + str(sorted(kwargs)))
        payload = self._normalize_combine_payload(final_hidden_states, runtime_max)

        state["phase"] = self._PHASE_PUSHING
        try:
            with torch.cuda.nvtx.range("tile_push/k1"):
                torch.ops.trtllm.moe_a2a_push_k1(
                    payload,
                    int(local_num_tokens),
                    self.workspace,
                    self._push_rc11_metainfo,
                    int(runtime_max),
                    int(self.max_num_tokens_per_rank),
                    int(self.ep_rank),
                    int(self.ep_size),
                    int(layout.q_plane_offset_bytes),
                    bool(self.payload_in_workspace),
                    int(self._push_recipe_id),
                    bool(cft_incumbent_live),
                )
            state["phase"] = self._PHASE_PUBLISHED
            # K23/K4 source the generation from the DEVICE control word (the
            # replay-safe D0 RMW), so no host ``generation`` scalar is passed;
            # ordering is the stream itself and a captured graph re-reads the
            # device generation on every replay.
            # Fused K23: K2 publish + K3 gate in one launch (replaces the former
            # separate moe_a2a_push_k2_publish + moe_a2a_push_k3_gate ops).
            with torch.cuda.nvtx.range("tile_push/k23_control"):
                torch.ops.trtllm.moe_a2a_push_k23(
                    self._control_plane,
                    int(self.ep_rank),
                    int(self.ep_size),
                    open_generation=bool(state.get(_D0_FOLDED_KEY, False)),
                    workspace=self.workspace,
                    metainfo=self._push_rc11_metainfo,
                )
            state["phase"] = self._PHASE_GATED
            with torch.cuda.nvtx.range("tile_push/k4_topk_reduce"):
                output = torch.ops.trtllm.moe_a2a_push_k4_topk_reduce(
                    int(local_num_tokens),
                    self.workspace,
                    self._control_plane,
                    self._push_rc11_metainfo,
                    int(runtime_max),
                    int(self.max_num_tokens_per_rank),
                    int(self.ep_rank),
                    int(self.ep_size),
                    int(layout.q_plane_offset_bytes),
                    int(self._push_recipe_id),
                    int(self._push_k4_selected_shared_stride_bytes),
                    bool(cft_incumbent_live),
                )
        except Exception:
            state["phase"] = self._PHASE_COMBINE_FAILED
            raise
        self._validate_output(output, payload, local_num_tokens)
        self._dispatch_state = {"phase": self._PHASE_IDLE}
        return output

    @staticmethod
    def _validate_output(output, payload, local_num_tokens) -> None:
        if (
            output.dim() != 2
            or tuple(int(x) for x in output.shape) != (int(local_num_tokens), HIDDEN_ELEMENTS)
            or output.dtype is not torch.bfloat16
            or not output.is_contiguous()
            or output.device != payload.device
            or getattr(output.device, "type", None) != "cuda"
        ):
            raise ValueError(
                "push-combine output must be contiguous CUDA BF16 "
                "[local_tokens,7168] on the payload device"
            )

    # ------------------------------------------------------------------
    # Workspace view + quiescent release.
    # ------------------------------------------------------------------

    def get_combine_payload_tensor_in_workspace(
        self, runtime_max_tokens_per_rank: int, hidden_size: int, dtype: torch.dtype
    ) -> torch.Tensor:
        # PURE-PYTHON workspace view (no PULL-DSO op): the push forced path
        # loads ONLY the push DSO (``moeAlltoAllTileNvfp4V2PushOp.cpp``), which
        # does NOT register the tile_tma workspace-view op.  The combine SOURCE
        # plane is the graph-static BF16 [E,R,7168] band starting at the layout's
        # q_plane_offset in THIS rank's own workspace segment; slice/view it here
        # by value rather than binding a PULL-only op.
        phase = self._dispatch_state.get("phase")
        if phase not in (self._PHASE_DISPATCHED, self._PHASE_GENERATION_OPEN):
            raise RuntimeError(
                "push-combine get_combine_payload_tensor_in_workspace requested "
                "outside dispatched/generation_open; got " + repr(phase)
            )
        layout_fields = self._dispatch_state.get("tile_nvfp4_v2_layout")
        if layout_fields is None:
            raise RuntimeError("push-combine dispatch state has no layout")
        layout = TileNvfp4V1Layout(**layout_fields)
        runtime_max = int(runtime_max_tokens_per_rank)
        if runtime_max != int(layout.runtime_max_tokens_per_rank):
            raise ValueError("push-combine workspace view R must match the dispatch graph key")
        if int(hidden_size) != HIDDEN_ELEMENTS or dtype is not torch.bfloat16:
            raise ValueError("push-combine workspace view is fixed to BF16/H7168")
        ep_size = int(self.ep_size)
        qoff = int(layout.q_plane_offset_bytes)
        # BF16 SOURCE band bytes: E*R*7168 elements * 2 bytes each.
        q_bytes = self.workspace[self.ep_rank][
            qoff : qoff + ep_size * runtime_max * HIDDEN_ELEMENTS * 2
        ]
        assert q_bytes.data_ptr() % 128 == 0, (
            "push-combine BF16 source band must be 128-byte aligned"
        )
        return (
            q_bytes.view(torch.bfloat16)
            .view(ep_size, runtime_max, HIDDEN_ELEMENTS)
            .view(ep_size * runtime_max, HIDDEN_ELEMENTS)
        )

    def destroy(self):
        """Quiescently release the owner token and control plane, then super()."""

        if getattr(self, "_destroyed", False):
            return
        # Quiescent release: drain the owner stream so no in-flight push step
        # still references the aliased workspace / control plane, THEN drop the
        # owner token and control-plane reference before the base tears the
        # symmetric workspace down.
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self._release_workspace_owner()
        self._release_control_plane()
        self._push_rc11_metainfo = None
        super().destroy()
