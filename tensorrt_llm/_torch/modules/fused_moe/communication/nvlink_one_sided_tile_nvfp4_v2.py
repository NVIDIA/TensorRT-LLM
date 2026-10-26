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

"""Opt-in NVLinkOneSided tile-wise NVFP4 V2 base (dispatch + workspace layout).

The legacy ``NVLinkOneSided`` class and its ``use_low_precision_combine`` FP8
boolean retain their original meaning.  This base reuses the stock
NVLinkOneSided dispatch unchanged and only adds the graph-static tile-NVFP4
workspace layout (BF16 combine source plane + a recipe-keyed second plane) that
a tile-wise combine (for example the Push-Combine selector in
``nvlink_one_sided_tile_nvfp4_push.py``) consumes.  The larger workspace gives
it its own entry in the NVLinkOneSided workspace cache.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch

from tensorrt_llm.bindings import internal as _tllm_internal
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.math_utils import pad_up

from .nvlink_one_sided import NVLinkOneSided
from .tile_nvfp4_v2_layout import (
    HIDDEN_ELEMENTS,
    MoeA2ACombineTileFormat,
    TileNvfp4V1Layout,
    graph_static_q_offset,
    workspace_extra_bytes,
)


class NVLinkOneSidedTileNvfp4V2(NVLinkOneSided):
    """NVLink one-sided communication base for tile-wise NVFP4 combines.

    This class is deliberately opt-in.  It supports only BF16/H7168/top-k=8
    and rejects the legacy FP8 boolean instead of silently changing formats.
    Subclasses provide ``combine`` and ``get_combine_payload_tensor_in_workspace``.
    """

    def supports_post_quant_dispatch(self) -> bool:
        """Quantize locally to NVFP4 before dispatch (stock FP4+SF wire).

        The tile-NVFP4 combine side is independent of the dispatch payload
        dtype: prepare quantizes the FC2 BF16 outputs into the Q plane.
        Dispatching BF16 here triples the wire bytes and forces the activation
        quantize onto the received (up to ExR) tokens after dispatch.
        """

        return True

    @staticmethod
    def _make_rc11_side_dso_metainfo(metainfo: torch.Tensor) -> torch.Tensor:
        """Copy current metainfo into the fixed 10-field side-DSO order.

        Dispatch and Q-offset calculation continue to consume the original
        metainfo in the current runtime ABI. Only the separately built
        tile-NVFP4 side DSO consumes this ten-field semantic projection, so it
        does not depend on the field count/order of the main library.
        """

        if not isinstance(metainfo, torch.Tensor):
            raise TypeError("RC11 side-DSO metainfo source must be a torch.Tensor")
        if metainfo.device.type != "cpu":
            raise RuntimeError("RC11 side-DSO metainfo source must remain on CPU")
        if metainfo.dtype != torch.int64:
            raise RuntimeError("RC11 side-DSO metainfo source must use int64")
        if metainfo.dim() != 1:
            raise RuntimeError("RC11 side-DSO metainfo source must be one-dimensional")
        if not metainfo.is_contiguous():
            raise RuntimeError("RC11 side-DSO metainfo source must be contiguous")

        try:
            thop = _tllm_internal.thop
            current_num_fields = int(thop.MOE_A2A_NUM_METAINFO_FIELDS)
            source_indices = (
                int(thop.MOE_A2A_FLAG_VAL_OFFSET_INDEX),
                int(thop.MOE_A2A_LOCAL_TOKEN_COUNTER_OFFSET_INDEX),
                int(thop.MOE_A2A_SEND_COUNTERS_OFFSET_INDEX),
                int(thop.MOE_A2A_RECV_COUNTERS_OFFSET_INDEX),
                int(thop.MOE_A2A_DISPATCH_COMPLETION_FLAGS_OFFSET_INDEX),
                int(thop.MOE_A2A_COMBINE_COMPLETION_FLAGS_OFFSET_INDEX),
                int(thop.MOE_A2A_TOPK_TARGET_RANKS_OFFSET_INDEX),
                int(thop.MOE_A2A_TOPK_SEND_INDICES_OFFSET_INDEX),
                int(thop.MOE_A2A_EPLB_GATHERED_STATS_OFFSET_INDEX),
                int(thop.MOE_A2A_PAYLOAD_DATA_OFFSET_INDEX),
            )
        except (AttributeError, TypeError, ValueError) as error:
            raise RuntimeError(
                "current thop does not expose the complete MoE A2A metainfo ABI"
            ) from error

        if current_num_fields <= 0 or metainfo.numel() != current_num_fields:
            raise RuntimeError("RC11 side-DSO metainfo source field count does not match thop")
        if len(set(source_indices)) != 10:
            raise RuntimeError("RC11 side-DSO metainfo semantic indices are not unique")
        if any(index < 0 or index >= current_num_fields for index in source_indices):
            raise RuntimeError("RC11 side-DSO metainfo semantic index is outside the current ABI")

        index_tensor = torch.tensor(source_indices, dtype=torch.int64, device=metainfo.device)
        side_metainfo = metainfo.index_select(0, index_tensor)
        if (
            side_metainfo.device.type != "cpu"
            or side_metainfo.dtype != torch.int64
            or side_metainfo.dim() != 1
            or side_metainfo.numel() != 10
            or not side_metainfo.is_contiguous()
        ):
            raise RuntimeError("failed to materialize valid RC11 side-DSO metainfo")
        return side_metainfo

    @staticmethod
    def calculate_required_workspace_size(
        ep_size: int,
        top_k: int,
        max_num_tokens: int,
        hidden_size: int,
        dtype: torch.dtype,
        eplb_stats_num_experts: Optional[int] = None,
        extra_payload_bytes_per_token: int = 0,
    ) -> int:
        legacy_bytes = NVLinkOneSided.calculate_required_workspace_size(
            ep_size,
            top_k,
            max_num_tokens,
            hidden_size,
            dtype,
            eplb_stats_num_experts=eplb_stats_num_experts,
            extra_payload_bytes_per_token=extra_payload_bytes_per_token,
        )
        # The Q source plane is the existing graph-static BF16 combine
        # reservation.  V2 adds only the separate E*M*896 SF plane.
        return pad_up(legacy_bytes + workspace_extra_bytes(ep_size, max_num_tokens), 128)

    def __init__(
        self,
        mapping: Mapping,
        num_slots: int,
        top_k: int,
        max_num_tokens_per_rank: int,
        payload_in_workspace: bool = False,
        hidden_size: Optional[int] = None,
        dtype: Optional[torch.dtype] = None,
        num_experts: Optional[int] = None,
        use_low_precision_combine: bool = False,
        *,
        combine_tile_format: MoeA2ACombineTileFormat = (MoeA2ACombineTileFormat.TILE_NVFP4_V1),
        kernel_build_id: str,
    ):
        try:
            combine_tile_format = MoeA2ACombineTileFormat(combine_tile_format)
        except ValueError as error:
            raise ValueError(f"unknown combine tile format: {combine_tile_format}") from error
        if combine_tile_format is not MoeA2ACombineTileFormat.TILE_NVFP4_V1:
            raise ValueError("NVLinkOneSidedTileNvfp4V2 requires tile_nvfp4_v1")
        if use_low_precision_combine:
            raise ValueError(
                "tile_nvfp4_v1 and legacy use_low_precision_combine (FP8) are mutually exclusive"
            )
        if top_k != 8:
            raise ValueError("tile_nvfp4_v1 currently requires top_k=8")
        if hidden_size != HIDDEN_ELEMENTS or dtype is not torch.bfloat16:
            raise ValueError("tile_nvfp4_v1 requires hidden_size=7168 and dtype=torch.bfloat16")
        if not kernel_build_id:
            raise ValueError("kernel_build_id is required for the CUDA graph key")

        if num_experts is not None:
            raise ValueError("tile_nvfp4_v1 does not support EPLB statistics gathering")

        self.combine_tile_format = combine_tile_format
        self.kernel_build_id = kernel_build_id

        # The base sizes the workspace through self.calculate_required_workspace_size
        # (overridden below), so the tile-wise layout gets its own entry in the
        # NVLinkOneSided workspace cache and never aliases a stock workspace.
        super().__init__(
            mapping,
            num_slots,
            top_k,
            max_num_tokens_per_rank,
            hidden_size=hidden_size,
            dtype=dtype,
            num_experts=None,
            use_low_precision_combine=False,
        )
        # ExternalCommMoEScheduler sets this per forward when the MoE backend
        # writes its output directly into the workspace combine source plane.
        self.payload_in_workspace = payload_in_workspace

        self._tile_workspace_key = (
            "moe_a2a_tile_nvfp4_v2_workspace",
            self.kernel_build_id,
            self.combine_tile_format.value,
            self.ep_size,
            self.max_num_tokens_per_rank,
            HIDDEN_ELEMENTS,
            str(torch.bfloat16),
            workspace_extra_bytes(self.ep_size, self.max_num_tokens_per_rank),
        )
        workspace_record = self._workspace_state
        if workspace_record is None:
            raise RuntimeError("tile-wise V2 workspace was not initialized")
        cached_key = workspace_record.setdefault(
            "tile_nvfp4_v2_workspace_key", self._tile_workspace_key
        )
        if cached_key != self._tile_workspace_key:
            raise RuntimeError(
                "refusing to reuse a tile-wise V2 workspace with a different layout key"
            )

    def dispatch(
        self,
        hidden_states: torch.Tensor,
        hidden_states_sf: Optional[torch.Tensor],
        token_selected_slots: torch.Tensor,
        token_final_scales: Optional[torch.Tensor],
        all_rank_num_tokens: List[int],
        use_dp_padding: Optional[bool] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor, Optional[torch.Tensor]]:
        if hidden_states_sf is None:
            # Unquantized (BF16) dispatch path, kept for bench/compat use.
            if (
                hidden_states.dtype != torch.bfloat16
                or hidden_states.dim() != 2
                or hidden_states.shape[1] != HIDDEN_ELEMENTS
            ):
                raise ValueError("tile_nvfp4_v1 dispatch requires BF16 [tokens,7168]")
        else:
            # Post-quant dispatch: FP4-packed activations + block scale factors.
            # Detailed payload validation is owned by the stock dispatch below.
            if (
                hidden_states.dim() != 2
                or hidden_states_sf.dim() != 2
                or hidden_states_sf.shape[0] != hidden_states.shape[0]
            ):
                raise ValueError("quantized dispatch payload/sf token dimensions must match")
        if (
            token_selected_slots.dtype != torch.int32
            or token_selected_slots.dim() != 2
            or token_selected_slots.shape[1] != self.top_k
        ):
            raise ValueError("tile_nvfp4_v1 dispatch slots must be int32 [tokens,8]")
        if token_selected_slots.shape[0] != hidden_states.shape[0]:
            raise ValueError("dispatch payload token dimensions must match")
        if token_final_scales is not None:
            if (
                token_final_scales.dtype not in (torch.float32, torch.bfloat16)
                or token_final_scales.dim() != 2
                or token_final_scales.shape[1] != self.top_k
                or token_final_scales.shape[0] != hidden_states.shape[0]
                or not token_final_scales.is_contiguous()
            ):
                raise ValueError(
                    "final scales must be contiguous float32/bfloat16 [tokens,8] when present"
                )

        def _payload_bytes_per_token(tensor: torch.Tensor) -> int:
            bytes_per_token = tensor.element_size()
            for extent in tensor.shape[1:]:
                bytes_per_token *= int(extent)
            return bytes_per_token

        # Order MUST mirror the stock dispatch payload registration
        # (nvlink_one_sided.py dispatch: hidden_states, sf?, slots, scales?)
        # so the 128B-aligned Q-plane offset math matches the device layout.
        payloads_for_layout = [("hidden_states", hidden_states)]
        if hidden_states_sf is not None:
            payloads_for_layout.append(("hidden_states_sf", hidden_states_sf))
        payloads_for_layout.append(("slots", token_selected_slots))
        if token_final_scales is not None:
            payloads_for_layout.append(("final_scales", token_final_scales))
        payload_schema = tuple(
            (
                name,
                str(tensor.dtype),
                tuple(int(extent) for extent in tensor.shape[1:]),
                _payload_bytes_per_token(tensor),
            )
            for name, tensor in payloads_for_layout
        )
        result = super().dispatch(
            hidden_states,
            hidden_states_sf,
            token_selected_slots,
            token_final_scales,
            all_rank_num_tokens,
            use_dp_padding,
            **kwargs,
        )
        runtime_max = self._dispatch_state.get("runtime_max_tokens_per_rank")
        legacy_q_offset = self._dispatch_state.get("combine_payload_offset")
        if runtime_max is None or legacy_q_offset is None:
            raise RuntimeError("legacy dispatch did not publish R/q_offset")
        payload_data_offset = int(self.moe_a2a_metainfo[self.PAYLOAD_DATA_OFFSET_INDEX].item())
        q_offset = graph_static_q_offset(
            payload_data_offset_bytes=payload_data_offset,
            ep_size=self.ep_size,
            graph_static_max_tokens_per_rank=self.max_num_tokens_per_rank,
            payload_bytes_per_token=tuple(item[3] for item in payload_schema),
        )
        layout = TileNvfp4V1Layout.create(
            ep_size=self.ep_size,
            runtime_max_tokens_per_rank=int(runtime_max),
            graph_static_max_tokens_per_rank=self.max_num_tokens_per_rank,
            q_plane_offset_bytes=int(q_offset),
            workspace_size_per_rank_bytes=int(self.workspace.size(1)),
        )
        # These values are copied into state so eager and captured paths can
        # compare exactly the same immutable descriptor/key.
        self._dispatch_state["legacy_runtime_combine_payload_offset"] = int(legacy_q_offset)
        self._dispatch_state["combine_payload_offset"] = int(q_offset)
        self._dispatch_state["dispatch_payload_schema"] = payload_schema
        self._dispatch_state["tile_nvfp4_v2_layout"] = layout.as_dict()
        self._dispatch_state["combine_graph_key"] = (
            self._tile_workspace_key,
            payload_schema,
            layout.graph_key(kernel_build_id=self.kernel_build_id),
        )
        return result

    def get_combine_graph_key(self) -> tuple:
        if self._dispatch_state.get("phase") != "dispatched":
            raise RuntimeError("combine graph key requested outside the dispatched phase")
        graph_key = self._dispatch_state.get("combine_graph_key")
        if graph_key is None:
            raise RuntimeError("tile-wise V2 dispatch state has no graph key")
        return graph_key

    def combine(self, final_hidden_states: torch.Tensor, **kwargs) -> torch.Tensor:
        raise NotImplementedError(
            f"{type(self).__name__} does not implement a tile-wise combine; use a "
            "concrete selector such as NVLinkOneSidedTilePushCombine"
        )

    def get_combine_payload_tensor_in_workspace(
        self, runtime_max_tokens_per_rank: int, hidden_size: int, dtype: torch.dtype
    ) -> torch.Tensor:
        raise NotImplementedError(
            f"{type(self).__name__} does not implement a tile-wise combine workspace view"
        )
