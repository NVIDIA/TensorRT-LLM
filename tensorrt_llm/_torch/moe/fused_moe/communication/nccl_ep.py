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
"""NCCL EP (Expert Parallelism) Communication Strategy for MoE -- LL rank-major.

Targets the embedded ``nccl.ep`` Python package built from pinned
NCCL-Extensions source. The dispatch
returns rank-major LL outputs:

  * ``recv_x``            : 3D ``[ep_size, max_tokens_per_rank, hidden]`` bf16,
                            reshaped to 2D for the downstream MoE pipeline.
  * ``recv_topk_idx``     : 2D ``[..., top_k]`` int32 with real expert IDs (-1 for invalid rows)
  * ``recv_topk_weights`` : 2D ``[..., top_k]`` float32 (the original router weights)

This matches NVLinkOneSided's contract directly, so NO
``_modify_output_to_adapt_fused_moe`` adapter is needed. The MoE backend's
``fused_moe`` runs top_k experts per row, applies the weights, and produces one
reduced output per row. ``handle.combine`` then sums per-source-rank
contributions back to the home rank.

Persistent handle: ``Group.create_handle`` is called ONCE (first dispatch);
subsequent dispatches call ``handle.update(topk_idx, ...)`` to rebind routing.
CUDA-graph capture is supported once the handle exists.
"""

from types import SimpleNamespace
from typing import List, Optional, Tuple

import torch

from tensorrt_llm.logger import logger
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models.modeling_utils import QuantConfig

from .base import Communication

_NCCL_RUNTIME_ERRORS = (RuntimeError, OSError)


class NcclEP(Communication):
    """NCCL EP Low-Latency rank-major communication strategy for MoE expert parallelism."""

    def __init__(
        self,
        mapping: Mapping,
        num_slots: int,
        hidden_size: int,
        max_num_tokens: int = 1024,
        moe_max_num_tokens: Optional[int] = None,
        top_k: int = 8,
        quant_config: Optional[QuantConfig] = None,
        use_low_precision_combine: bool = False,
    ):
        super().__init__(mapping)

        from tensorrt_llm._torch.moe.fused_moe.nccl_ep_utils import is_nccl_ep_installed

        if not is_nccl_ep_installed():
            raise RuntimeError("nccl-ep is not installed.")

        self.num_slots = num_slots
        self.num_experts = num_slots
        self.hidden_size = hidden_size
        self.num_local_experts = num_slots // self.ep_size
        self.max_top_k = top_k
        self.quant_config = quant_config
        self.use_fp8 = self._has_deepseek_fp8_block_scales()
        self.use_external_fp8 = self._has_fp8_qdq()
        self.use_external_nvfp4 = self._has_nvfp4()
        if self.use_fp8 or self.use_external_fp8 or self.use_external_nvfp4:
            from tensorrt_llm._torch.moe.fused_moe.nccl_ep_utils import nccl_ep_supports_version

            if not nccl_ep_supports_version("0.2"):
                raise RuntimeError("NCCL-EP quantized dispatch requires libnccl_ep >= 0.2.")

        if self.use_external_nvfp4 and hidden_size % 256 != 0:
            raise RuntimeError(
                "NCCL-EP NVFP4 dispatch requires hidden_size divisible by 256 "
                "for 16-byte token and scale rows."
            )

        self.use_low_precision_combine = (
            use_low_precision_combine and self.supports_low_precision_combine()
        )

        self.max_tokens_per_rank = (
            max_num_tokens
            if moe_max_num_tokens is None
            else min(max_num_tokens, moe_max_num_tokens)
        )
        self.max_recv_tokens = self.ep_size * self.max_tokens_per_rank

        # Singleton NCCL EP context: owns the EP group, RDMA buffers, and
        # persistent OUTPUT Tensor descriptors. Allocate it lazily on first
        # dispatch because full-model construction runs under MetaInitMode,
        # which redirects torch.empty to the meta device even when a CUDA
        # device is passed explicitly.
        self._ctx = None

        # Persistent dispatch handle. Created on first dispatch via
        # group.create_handle; reused thereafter via handle.update so
        # subsequent dispatches are CUDA-graph-safe.
        self._handle = None  # nccl.ep.Handle | None
        self._dispatch_state: dict = {}

    @staticmethod
    def is_platform_supported() -> bool:
        from tensorrt_llm._torch.moe.fused_moe.nccl_ep_utils import is_nccl_ep_installed

        return is_nccl_ep_installed()

    def is_workload_feasible(self, all_rank_num_tokens: List[int], num_chunks: int) -> bool:
        if num_chunks > 1:
            return False
        if max(all_rank_num_tokens) > self.max_tokens_per_rank:
            return False
        return True

    def supports_post_quant_dispatch(self) -> bool:
        return self.use_external_fp8 or self.use_external_nvfp4

    def uses_internal_dispatch_quantization(self) -> bool:
        return self.use_fp8

    def supports_low_precision_combine(self) -> bool:
        """Return whether the experimental LL NVFP4 combine path is available."""
        return self.use_external_nvfp4 and self.hidden_size % 512 == 0

    def _has_deepseek_fp8_block_scales(self) -> bool:
        return (
            self.quant_config is not None
            and self.quant_config.layer_quant_mode.has_fp8_block_scales()
        )

    def _has_fp8_qdq(self) -> bool:
        return self.quant_config is not None and self.quant_config.layer_quant_mode.has_fp8_qdq()

    def _has_nvfp4(self) -> bool:
        return self.quant_config is not None and self.quant_config.layer_quant_mode.has_nvfp4()

    def _get_context(self):
        if self._ctx is None:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError(
                    "NcclEP context must be initialized before CUDA graph capture. "
                    "Run an eager warmup forward before enabling or capturing CUDA graphs."
                )
            from nccl.ep import Layout

            from tensorrt_llm._torch.moe.fused_moe.nccl_ep_utils import get_nccl_ep_context

            self._ctx = get_nccl_ep_context(
                self.mapping,
                self.num_experts,
                self.max_tokens_per_rank,
                self.hidden_size,
                self.max_top_k,
                self.use_fp8,
                Layout.RANK_MAJOR,
                external_fp8=self.use_external_fp8,
                external_nvfp4=self.use_external_nvfp4,
            )
        return self._ctx

    def _setup_handle(self, ctx, topk_nd, stream):
        """Ensure self._handle exists; rebind topk via handle.update on subsequent calls."""
        if self._handle is None:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError(
                    "NcclEP dispatch handle must be initialized before CUDA graph capture. "
                    "Run an eager warmup forward before enabling or capturing CUDA graphs."
                )
            self._handle = ctx.ep_group.create_handle(
                ctx.layout,
                topk_nd,
                stream=stream,
            )
        else:
            self._handle.update(topk_nd, stream=stream)
        return self._handle

    # ------------------------------------------------------------------
    # Dispatch -- rank-major LL
    # ------------------------------------------------------------------

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
        """Dispatch tokens via NCCL EP LL rank-major.

        Returns rank-major-shaped tensors directly:
          (recv_hs [N, H], recv_sf [N, H/128] or None, recv_slots [N, top_k] int32,
           recv_scales [N, top_k] float32)

        where N = ep_size * max_tokens_per_rank. Rows beyond
        ``recv_rank_counter[r]`` for source rank r have recv_slots = -1
        (sentinel), naturally skipped by the MoE backend.
        """
        from nccl.ep import DispatchInputs, DispatchOutputs, Tensor

        from tensorrt_llm.bindings.internal.thop import BufferKind

        ctx = self._get_context()

        all_rank_max_num_tokens = max(all_rank_num_tokens)
        if all_rank_max_num_tokens > self.max_tokens_per_rank:
            raise ValueError(
                f"all_rank_max_num_tokens={all_rank_max_num_tokens} > "
                f"max_tokens_per_rank={self.max_tokens_per_rank}"
            )

        num_tokens = hidden_states.shape[0]
        top_k = token_selected_slots.shape[1]
        if top_k > self.max_top_k:
            raise ValueError(f"top_k={top_k} exceeds configured max_top_k={self.max_top_k}")
        if token_final_scales is None:
            raise RuntimeError(
                "NcclEP rank-major dispatch requires token_final_scales "
                "(router weights) -- it is an INPUT to handle.dispatch."
            )

        stream = ctx.get_stream()

        # NCCL-EP takes FP32 top-k weights. Router outputs are commonly
        # BF16/FP16, so convert them at the binding boundary.
        if not hidden_states.is_contiguous() or not token_final_scales.is_contiguous():
            raise ValueError("NCCL-EP dispatch requires contiguous token and routing tensors.")
        hidden_states_c = hidden_states
        weights_f32_c = token_final_scales.float()

        input_tokens_nd = Tensor(hidden_states_c)
        input_topk_weights_nd = Tensor(weights_f32_c)
        input_scales_nd = None
        if self.use_external_fp8:
            if hidden_states.dtype != torch.float8_e4m3fn:
                raise ValueError(
                    "NCCL-EP external FP8 dispatch requires float8_e4m3fn tokens, "
                    f"got {hidden_states.dtype}."
                )
            # TODO(NCCL-EP): accept raw FP8 descriptors under NONE so this
            # byte-preserving compatibility view is unnecessary.
            # NONE accepts BF16 but not FP8 descriptors. Mirror DeepEP LL by
            # transporting the FP8 payload as a BF16 byte view, then restoring
            # the FP8 view on the receive buffer below.
            hidden_states_c = hidden_states.view(torch.bfloat16)
            input_tokens_nd = Tensor(hidden_states_c)
        elif self.use_external_nvfp4:
            expected_token_shape = (num_tokens, self.hidden_size // 2)
            expected_scale_shape = (num_tokens, self.hidden_size // 16)
            if hidden_states.dtype != torch.uint8 or hidden_states.shape != expected_token_shape:
                raise ValueError(
                    "NCCL-EP NVFP4 dispatch requires uint8 tokens with shape "
                    f"{expected_token_shape}, got dtype={hidden_states.dtype}, "
                    f"shape={tuple(hidden_states.shape)}."
                )
            if (
                hidden_states_sf is None
                or hidden_states_sf.dtype != torch.uint8
                or hidden_states_sf.shape != expected_scale_shape
            ):
                got_dtype = None if hidden_states_sf is None else hidden_states_sf.dtype
                got_shape = None if hidden_states_sf is None else tuple(hidden_states_sf.shape)
                raise ValueError(
                    "NCCL-EP NVFP4 dispatch requires uint8 scales with shape "
                    f"{expected_scale_shape}, got dtype={got_dtype}, shape={got_shape}."
                )
            scales_c = (
                hidden_states_sf
                if hidden_states_sf.is_contiguous()
                else hidden_states_sf.contiguous()
            )
            # nccl4py does not expose ncclFloat4x2. Transport the packed FP4
            # bytes through a BF16 view; FWD preserves the physical row bytes.
            hidden_states_c = hidden_states.view(torch.bfloat16)
            input_tokens_nd = Tensor(hidden_states_c)
            input_scales_nd = Tensor(scales_c)

        # NCCL-EP 0.2 resets inactive rank-major recv_topk_idx rows in the
        # dispatch kernel. Retain the v0.1 pre-dispatch fallback.
        if not ctx.kernel_resets_recv_topk_idx:
            ctx.recv_topk_idx_buf.fill_(-1)

        topk_idx_dev = (
            token_selected_slots
            if token_selected_slots.dtype == ctx.topk_idx_dtype
            and token_selected_slots.is_contiguous()
            else token_selected_slots.to(ctx.topk_idx_dtype).contiguous()
        )
        topk_nd = Tensor(topk_idx_dev)
        dispatch_outputs = ctx.dispatch_outputs
        window_tokens = None
        # CUDA graph capture requires stable context-owned output addresses.
        # Eager dispatch keeps the operation-scoped NCCL-window allocation.
        if ctx.zerocopy_enabled and not torch.cuda.is_current_stream_capturing():
            window_tokens, actual_kind, window_handle = (
                torch.ops.trtllm.allocate_output_with_nccl_window(
                    ctx.output_tokens_buf,
                    int(BufferKind.NCCL_WINDOW),
                    self.mapping.moe_ep_group,
                )
            )
            if actual_kind == int(BufferKind.NCCL_WINDOW) and window_handle:
                # The TRT-LLM pool owns this already-registered raw ncclWindow_t.
                window = SimpleNamespace(handle=window_handle)
                dispatch_outputs = DispatchOutputs(
                    tokens=Tensor(window_tokens, window=window, window_offset=0),
                    topk_weights=ctx.recv_topk_weights_nd,
                    topk_idx=ctx.recv_topk_idx_nd,
                    scales=ctx.scales_nd,
                )
            else:
                # The allocator returned a distinct ordinary tensor. Dispatch
                # still targets the persistent descriptor, so return it below.
                window_tokens = None

        handle = self._setup_handle(ctx, topk_nd, stream)
        inputs = DispatchInputs(
            tokens=input_tokens_nd,
            topk_weights=input_topk_weights_nd,
            scales=input_scales_nd,
        )
        handle.dispatch(
            inputs,
            dispatch_outputs,
            layout_info=ctx.dispatch_layout_info,
            config=ctx.dispatch_config,
            stream=stream,
        )

        # The handle internally references topk_nd; keep both the Tensor
        # descriptor and its backing torch tensor alive until combine completes.
        self._dispatch_state = {
            "num_tokens": num_tokens,
            "topk_nd": topk_nd,
            "topk_idx_dev": topk_idx_dev,
            "window_tokens": window_tokens,
        }

        # Match NVLinkOneSided's contract: token_selected_slots in
        # [0, num_experts) for valid rows, -1 for invalid. When the kernel
        # writes GLOBAL ids directly (opt-in detected at ctx init), the
        # buffer is already in the right space and we pass it through.
        # Otherwise the kernel writes LOCAL ids in [0, num_local_experts)
        # and we add ep_rank * num_local_experts to restore the global
        # numbering downstream consumers expect.
        # The dispatch buffer is 3D [ep_size, max_tokens_per_rank, max_top_k]
        # per the LL rank-major contract; flatten to 2D for downstream.
        recv_topk_idx_flat = ctx.recv_topk_idx_buf.view(self.max_recv_tokens, self.max_top_k)
        if ctx.kernel_writes_global_ids:
            recv_slots_global = recv_topk_idx_flat
        else:
            recv_slots_global = torch.where(
                recv_topk_idx_flat >= 0,
                recv_topk_idx_flat + self.ep_rank * self.num_local_experts,
                recv_topk_idx_flat,
            )

        # Output buffers are 3D [ep_size, max_tokens_per_rank, ...] per the
        # LL rank-major contract; downstream MoE pipeline expects 2D --
        # flatten via view.
        output_tokens = window_tokens if window_tokens is not None else ctx.output_tokens_buf
        if self.use_external_fp8:
            output_tokens = output_tokens.view(torch.float8_e4m3fn)
        elif self.use_external_nvfp4:
            output_tokens = output_tokens.view(torch.uint8)
        return (
            output_tokens.view(
                self.max_recv_tokens,
                self.hidden_size // 2 if self.use_external_nvfp4 else self.hidden_size,
            ),
            (
                ctx.scales_buf.view(
                    self.max_recv_tokens,
                    self.hidden_size // 128 if self.use_fp8 else self.hidden_size // 16,
                )
                if (self.use_fp8 or self.use_external_nvfp4)
                else None
            ),
            recv_slots_global,
            ctx.recv_topk_weights_buf.view(self.max_recv_tokens, self.max_top_k),
        )

    # ------------------------------------------------------------------
    # Combine -- rank-major LL
    # ------------------------------------------------------------------

    def combine(
        self,
        final_hidden_states: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """Combine MoE-reduced rank-major output back to the home rank.

        Input: [max_recv_tokens, hidden] -- already weighted per-row by fused_moe.
        Output: [num_tokens, hidden] -- combined to original token order.
        """
        from nccl.ep import (
            CombineConfig,
            CombineInputs,
            CombineOutputs,
            CombineQuantizationRecipe,
            Tensor,
        )

        ctx = self._ctx
        if ctx is None:
            raise RuntimeError("NcclEP.combine called before dispatch.")
        state = self._dispatch_state
        stream = ctx.get_stream()

        num_tokens = state["num_tokens"]

        # Combine input for LL rank-major must be 3D
        # [ep_size, max_tokens_per_rank, hidden] -- reshape if caller passed
        # 2D [max_recv, H] or a per-expert [E, max_recv, H] layout.
        if final_hidden_states.dim() == 3 and final_hidden_states.shape[0] != self.ep_size:
            final_hidden_states = final_hidden_states.reshape(-1, self.hidden_size)
        if final_hidden_states.dim() == 2:
            if final_hidden_states.shape[0] != self.max_recv_tokens:
                raise ValueError(
                    f"combine input rows={final_hidden_states.shape[0]} "
                    f"expected={self.max_recv_tokens}"
                )
            final_hidden_states = final_hidden_states.view(
                self.ep_size,
                self.max_tokens_per_rank,
                self.hidden_size,
            )

        combine_input_c = final_hidden_states.contiguous()
        combine_output = torch.empty(
            num_tokens,
            self.hidden_size,
            dtype=torch.bfloat16,
            device=combine_input_c.device,
        )

        combine_input_nd = Tensor(combine_input_c)
        combine_output_nd = Tensor(combine_output)

        if self.use_low_precision_combine:
            # Match DeepEP LL's NVFP4 contract.  The rank-major leading
            # dimension is the source rank, so recv_rank_counter_buf supplies
            # the valid rows for each scale-kernel group.
            combine_scales = torch.ops.trtllm.calculate_nvfp4_global_scale(
                combine_input_c,
                ctx.recv_rank_counter_buf,
            )
            expected_scale_shape = (*combine_input_c.shape[:-1], 1)
            if (
                combine_scales.dtype != torch.float32
                or tuple(combine_scales.shape) != expected_scale_shape
            ):
                raise RuntimeError(
                    "calculate_nvfp4_global_scale must return FP32 scales with shape "
                    f"{expected_scale_shape}, got dtype={combine_scales.dtype}, "
                    f"shape={tuple(combine_scales.shape)}"
                )
            self._handle.combine(
                CombineInputs(
                    tokens=combine_input_nd,
                    scales=Tensor(combine_scales.contiguous()),
                ),
                CombineOutputs(tokens=combine_output_nd),
                config=CombineConfig(
                    quantization_recipe=CombineQuantizationRecipe.NVFP4,
                ),
                stream=stream,
            )
        else:
            # Rank-major combine: no layout_info or config required.
            self._handle.combine(
                CombineInputs(tokens=combine_input_nd),
                CombineOutputs(tokens=combine_output_nd),
                stream=stream,
            )

        self._dispatch_state = {}
        return combine_output

    def destroy(self):
        """Release per-instance NCCL EP resources (handle).

        NcclEpContext is shared across instances and released through a
        refcounted cache.
        """
        if self._handle is not None:
            try:
                self._handle.destroy()
            except _NCCL_RUNTIME_ERRORS as e:
                logger.warning(f"Handle.destroy error during destroy: {e}")
            self._handle = None

        from tensorrt_llm._torch.moe.fused_moe.nccl_ep_utils import release_nccl_ep_context

        if self._ctx is not None:
            release_nccl_ep_context(self._ctx)
        self._ctx = None
        self._dispatch_state = {}
