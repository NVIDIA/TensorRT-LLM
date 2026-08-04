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

Targets the ``nccl.ep`` Python package shipped in the nccl4py wheel (built
against an NCCL master tree containing ``contrib/nccl_ep``). The dispatch
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

from typing import List, Optional, Tuple

import torch

from tensorrt_llm.logger import logger
from tensorrt_llm.mapping import Mapping

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
    ):
        super().__init__(mapping)

        from tensorrt_llm._torch.modules.fused_moe.nccl_ep_utils import is_nccl_ep_installed

        if not is_nccl_ep_installed():
            raise RuntimeError("nccl-ep is not installed.")

        if self.ep_size <= 0:
            raise ValueError(f"NcclEP requires moe_ep_size > 0, got {self.ep_size}")
        if not 0 <= self.ep_rank < self.ep_size:
            raise ValueError(
                f"NcclEP requires 0 <= moe_ep_rank < moe_ep_size, "
                f"got {self.ep_rank=}, {self.ep_size=}"
            )
        if num_slots <= 0:
            raise ValueError(f"NcclEP requires num_slots > 0, got {num_slots}")
        if num_slots % self.ep_size != 0:
            raise ValueError(
                f"NcclEP requires num_slots divisible by moe_ep_size, "
                f"got {num_slots=}, {self.ep_size=}"
            )
        if hidden_size <= 0:
            raise ValueError(f"NcclEP requires hidden_size > 0, got {hidden_size}")
        if top_k <= 0 or top_k > num_slots:
            raise ValueError(f"NcclEP requires 0 < top_k <= num_slots, got {top_k=}, {num_slots=}")
        if max_num_tokens <= 0:
            raise ValueError(f"NcclEP requires max_num_tokens > 0, got {max_num_tokens}")
        if moe_max_num_tokens is not None and moe_max_num_tokens <= 0:
            raise ValueError(
                f"NcclEP requires moe_max_num_tokens > 0 when provided, got {moe_max_num_tokens}"
            )

        self.num_slots = num_slots
        self.num_experts = num_slots
        self.hidden_size = hidden_size
        self.num_local_experts = num_slots // self.ep_size
        self.max_top_k = top_k

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
        from tensorrt_llm._torch.modules.fused_moe.nccl_ep_utils import is_nccl_ep_installed

        return is_nccl_ep_installed()

    def is_workload_feasible(self, all_rank_num_tokens: List[int], num_chunks: int) -> bool:
        if num_chunks > 1:
            return False
        if max(all_rank_num_tokens) > self.max_tokens_per_rank:
            return False
        return True

    def _get_context(self):
        if self._ctx is None:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError(
                    "NcclEP context must be initialized before CUDA graph capture. "
                    "Run an eager warmup forward before enabling or capturing CUDA graphs."
                )
            from nccl.ep import Layout

            from tensorrt_llm._torch.modules.fused_moe.nccl_ep_utils import get_nccl_ep_context

            self._ctx = get_nccl_ep_context(
                self.mapping,
                self.num_experts,
                self.max_tokens_per_rank,
                self.hidden_size,
                self.max_top_k,
                Layout.RANK_MAJOR,
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
        from nccl.ep import DispatchConfig, DispatchInputs, DispatchOutputs, LayoutInfo, Tensor

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

        # TODO(NCCL): topk_weights still requires float32; once bf16/native
        # weights are accepted upstream, drop this conversion too.
        weights_f32 = (
            token_final_scales
            if token_final_scales.dtype == torch.float32
            else token_final_scales.to(torch.float32)
        )
        hidden_states_c = hidden_states.contiguous()
        weights_f32_c = weights_f32.contiguous()

        input_tokens_nd = Tensor(hidden_states_c)
        input_topk_weights_nd = Tensor(weights_f32_c)

        # Mark padding rows with the -1 sentinel so fused_moe skips them.
        # The dispatch kernel only writes recv_topk_idx for slots that
        # received tokens; rows beyond `recv_rank_counter[r]` keep stale
        # data from prior dispatches. recv_rank_counter is written fresh
        # by the dispatch kernel (low_latency.cu:877) so it does not need
        # pre-zeroing, and recv_topk_weights on -1 rows is don't-care
        # (fused_moe ignores the weight when the expert id is -1).
        ctx.recv_topk_idx_buf.fill_(-1)

        outputs = DispatchOutputs(
            tokens=ctx.output_tokens_nd,
            topk_weights=ctx.recv_topk_weights_nd,
            topk_idx=ctx.recv_topk_idx_nd,
            scales=None,
        )
        layout_info = LayoutInfo(src_rank_counters=ctx.recv_rank_counter_nd)
        # The v0.2-gated capability path asks the kernel to emit global
        # expert ids directly; v0.1 retains the default local-id contract
        # and uses the translation below.
        if ctx._expert_id_kind_global is not None:
            layout_info._lowpp.recv_topk_idx_kind = ctx._expert_id_kind_global

        topk_idx_dev = token_selected_slots.to(ctx.topk_idx_dtype).contiguous()
        topk_nd = Tensor(topk_idx_dev)
        handle = self._setup_handle(ctx, topk_nd, stream)
        inputs = DispatchInputs(
            tokens=input_tokens_nd,
            topk_weights=input_topk_weights_nd,
        )
        handle.dispatch(
            inputs,
            outputs,
            layout_info=layout_info,
            config=DispatchConfig(round_scales=0),
            stream=stream,
        )

        # The handle internally references topk_nd; keep both the Tensor
        # descriptor and its backing torch tensor alive until combine completes.
        self._dispatch_state = {
            "num_tokens": num_tokens,
            "topk_nd": topk_nd,
            "topk_idx_dev": topk_idx_dev,
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
        return (
            ctx.output_tokens_buf.view(self.max_recv_tokens, self.hidden_size),
            None,
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
        from nccl.ep import CombineInputs, CombineOutputs, Tensor

        ctx = self._ctx
        if ctx is None:
            raise RuntimeError("NcclEP.combine called before dispatch.")
        state = self._dispatch_state
        stream = ctx.get_stream()

        num_tokens = state["num_tokens"]

        # NCCL-EP LL combine consumes rank-major tokens with shape
        # [ep_size, max_tokens_per_rank, hidden]. The scheduler normally
        # provides the equivalent 2D [max_recv_tokens, hidden] view.
        if final_hidden_states.dim() == 2:
            expected_shape = (self.max_recv_tokens, self.hidden_size)
            if tuple(final_hidden_states.shape) != expected_shape:
                raise ValueError(
                    f"combine input shape={tuple(final_hidden_states.shape)} "
                    f"expected={expected_shape}"
                )
            final_hidden_states = final_hidden_states.view(
                self.ep_size,
                self.max_tokens_per_rank,
                self.hidden_size,
            )
        elif final_hidden_states.dim() == 3:
            expected_shape = (self.ep_size, self.max_tokens_per_rank, self.hidden_size)
            if tuple(final_hidden_states.shape) != expected_shape:
                raise ValueError(
                    f"combine input shape={tuple(final_hidden_states.shape)} "
                    f"expected={expected_shape}"
                )
        else:
            raise ValueError(
                "NcclEP combine input must be 2D [max_recv_tokens, hidden] or "
                "3D [ep_size, max_tokens_per_rank, hidden], got "
                f"shape={tuple(final_hidden_states.shape)}"
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

        # Rank-major combine: no layout_info, no config required (send_only=0
        # is the default; defaults round-trip fine).
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

        from tensorrt_llm._torch.modules.fused_moe.nccl_ep_utils import release_nccl_ep_context

        if self._ctx is not None:
            release_nccl_ep_context(self._ctx)
        self._ctx = None
        self._dispatch_state = {}
