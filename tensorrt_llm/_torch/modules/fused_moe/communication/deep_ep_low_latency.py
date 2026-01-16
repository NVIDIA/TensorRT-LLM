# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""
DeepEP Low Latency Communication Strategy

This module implements the DeepEP Low Latency communication method for MoE.
DeepEP Low Latency is optimized for small token counts with minimal communication overhead.
"""

import os
from typing import List, Optional, Tuple

import torch

from tensorrt_llm._torch.modules.fused_moe.deep_ep_utils import buffer_pool, deep_ep_installed
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models.modeling_utils import QuantConfig

from .base import Communication


class DeepEPLowLatency(Communication):
    """
    DeepEP Low Latency strategy supporting both pre-quant and post-quant
    """

    def __init__(
        self,
        mapping: Mapping,
        num_slots: int,
        hidden_size: int,
        weight_dtype: torch.dtype,
        quant_config: QuantConfig,
        expert_size_per_partition: int = 0,
        max_num_tokens: int = 1024,
        use_low_precision_combine: bool = False,
        moe_max_num_tokens: Optional[int] = None,
    ):
        super().__init__(mapping)

        # Store needed parameters
        self.num_slots = num_slots
        self.hidden_size = hidden_size
        self.weight_dtype = weight_dtype
        self.quant_config = quant_config
        self.moe_max_num_tokens = moe_max_num_tokens

        self.expert_size_per_partition = expert_size_per_partition
        self.use_low_precision_combine = (
            use_low_precision_combine and self.supports_low_precision_combine()
        )
        # Read from environment variable, same as wideEP
        self.enable_postquant_alltoall = (
            os.environ.get("TRTLLM_MOE_POST_QUANT_ALLTOALLV", "1") == "1"
        )

        # Calculate deep_ep_max_num_tokens
        assert moe_max_num_tokens is not None
        default_limit = min(max_num_tokens, moe_max_num_tokens)
        self.deep_ep_max_num_tokens = int(
            os.environ.get("TRTLLM_DEEP_EP_TOKEN_LIMIT", str(default_limit))
        )

        # Set nvshmem queue pair depth larger than the number of on-flight WRs
        # (ref: https://github.com/deepseek-ai/DeepEP/issues/427)
        os.environ["NVSHMEM_QP_DEPTH"] = str(2 * (self.deep_ep_max_num_tokens + 1))

        self.deep_ep_buffer = buffer_pool.get_low_latency_buffer(mapping)
        self.deep_ep_buffer.reserve(self.deep_ep_max_num_tokens, hidden_size, num_slots)

    @staticmethod
    def is_platform_supported() -> bool:
        """
        Check if DeepEP Low Latency is supported on the current platform
        """
        if os.environ.get("TRTLLM_CAN_USE_DEEP_EP", "0") != "1":
            return False
        if not deep_ep_installed:
            return False
        return True

    def supports_post_quant_dispatch(self) -> bool:
        """
        DeepEP Low Latency supports post-quant for: fp8_qdq, nvfp4, w4afp8
        """
        if not self.enable_postquant_alltoall:
            return False
        return self._has_nvfp4() or self._has_fp8_qdq() or self._has_w4afp8()

    def supports_low_precision_combine(self) -> bool:
        """
        DeepEP Low Latency supports low-precision combine for: fp8_qdq, nvfp4, w4afp8
        """
        return self._has_nvfp4() or self._has_fp8_qdq() or self._has_w4afp8()

    def is_workload_feasible(self, all_rank_num_tokens: List[int], num_chunks: int) -> bool:
        """
        Check if DeepEP Low Latency is feasible for the given workload at runtime.

        This method performs runtime checks based on workload characteristics such as
        token counts, number of chunks, and weight dtype compatibility.
        """
        if num_chunks > 1:
            return False
        all_rank_max_num_tokens = max(all_rank_num_tokens)
        if all_rank_max_num_tokens > self.deep_ep_max_num_tokens:
            return False
        if self.weight_dtype != torch.bfloat16:
            return False
        return True

    def dispatch(
        self,
        hidden_states: torch.Tensor,
        hidden_states_sf: Optional[torch.Tensor],
        token_selected_slots: torch.Tensor,
        token_final_scales: Optional[torch.Tensor],
        all_rank_num_tokens: List[int],
        use_dp_padding: Optional[bool] = None,
        pre_quant_scale: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor, Optional[torch.Tensor]]:
        """
        DeepEP Low Latency dispatch
        """
        all_rank_max_num_tokens = max(all_rank_num_tokens)

        assert all_rank_max_num_tokens <= self.deep_ep_max_num_tokens

        deep_ep_topk_idx = token_selected_slots
        deep_ep_topk_weights = token_final_scales

        if not self.supports_post_quant_dispatch():
            # Pre-quant dispatch (unquantized data)
            hidden_states, recv_expert_count, deep_ep_handle = (
                self.deep_ep_buffer.low_latency_dispatch(
                    hidden_states, deep_ep_topk_idx, all_rank_max_num_tokens, self.num_slots
                )
            )

            hidden_states, _, token_selected_slots, token_final_scales = (
                self._modify_output_to_adapt_fused_moe(
                    hidden_states, None, recv_expert_count, token_final_scales.dtype
                )
            )

            # Store dispatch state for combine
            self._dispatch_state = {
                "deep_ep_handle": deep_ep_handle,
                "deep_ep_topk_idx": deep_ep_topk_idx,
                "deep_ep_topk_weights": deep_ep_topk_weights,
                "recv_expert_count": recv_expert_count,
            }

        else:
            # Post-quant dispatch (quantized data)
            if self._has_fp8_qdq():
                assert hidden_states.dtype == torch.float8_e4m3fn and hidden_states_sf is None, (
                    "hidden_states should be torch.float8_e4m3fn and hidden_states_sf should be None "
                    "in fp8 postquant alltoall"
                )

                hidden_states = hidden_states.view(torch.bfloat16)
                hidden_states, recv_expert_count, deep_ep_handle = (
                    self.deep_ep_buffer.low_latency_dispatch(
                        hidden_states, deep_ep_topk_idx, all_rank_max_num_tokens, self.num_slots
                    )
                )
                hidden_states = hidden_states.view(torch.float8_e4m3fn)

            elif self._has_nvfp4():
                token_num = hidden_states.shape[0]
                # For nvfp4, hidden_states.shape[1] is the quantized dimension (hidden_size // 2)
                # We need to calculate the original hidden_size
                # note: we use uint8 to store 2 fp4 values
                hidden_size = hidden_states.shape[1] * 2

                # Pre-dispatch assertions
                assert (
                    hidden_states.dtype == torch.uint8
                    and hidden_states_sf is not None
                    and hidden_states_sf.dtype == torch.uint8
                )
                assert hidden_size % 32 == 0, (
                    "HiddenSize should be divisible by 32 in nvfp4 postquant alltoall"
                )
                assert (
                    hidden_states_sf.shape[0] == token_num
                    and hidden_states_sf.shape[1] == hidden_size // 16
                )
                assert (
                    hidden_states.shape[0] == token_num
                    and hidden_states.shape[1] == hidden_size // 2
                )

                hidden_states, hidden_states_sf, recv_expert_count, deep_ep_handle = (
                    self.deep_ep_buffer.low_latency_dispatch_fp4(
                        hidden_states,
                        hidden_states_sf,
                        deep_ep_topk_idx,
                        all_rank_max_num_tokens,
                        self.num_slots,
                    )
                )

                # Post-dispatch assertions
                assert hidden_states.dtype == torch.uint8 and hidden_states_sf.dtype == torch.uint8
                assert hidden_states.dim() == 3 and hidden_states_sf.dim() == 3
                assert (
                    hidden_states.shape[2] == hidden_size // 2
                    and hidden_states_sf.shape[2] == hidden_size // 16
                )

            elif self._has_w4afp8():
                assert pre_quant_scale is not None, "W4AFP8 requires pre_quant_scale"
                assert (
                    pre_quant_scale.shape == (1, hidden_states.shape[1])
                    and pre_quant_scale.dtype == hidden_states.dtype
                )

                hidden_states = (
                    (hidden_states * pre_quant_scale).to(torch.float8_e4m3fn).view(torch.bfloat16)
                )
                hidden_states, recv_expert_count, deep_ep_handle = (
                    self.deep_ep_buffer.low_latency_dispatch(
                        hidden_states, deep_ep_topk_idx, all_rank_max_num_tokens, self.num_slots
                    )
                )
                hidden_states = hidden_states.view(torch.float8_e4m3fn)

            else:
                raise ValueError("Unsupported quantization mode for post-quant DeepEPLowLatency")

            hidden_states, hidden_states_sf, token_selected_slots, token_final_scales = (
                self._modify_output_to_adapt_fused_moe(
                    hidden_states, hidden_states_sf, recv_expert_count, token_final_scales.dtype
                )
            )

            # Store dispatch state for combine
            self._dispatch_state = {
                "deep_ep_handle": deep_ep_handle,
                "deep_ep_topk_idx": deep_ep_topk_idx,
                "deep_ep_topk_weights": deep_ep_topk_weights,
                "recv_expert_count": recv_expert_count,
            }

        return hidden_states, hidden_states_sf, token_selected_slots, token_final_scales

    def combine(
        self,
        final_hidden_states: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """
        DeepEP Low Latency combine - reads from self._dispatch_state
        """
        deep_ep_handle = self._dispatch_state["deep_ep_handle"]
        deep_ep_topk_idx = self._dispatch_state["deep_ep_topk_idx"]
        deep_ep_topk_weights = self._dispatch_state["deep_ep_topk_weights"]
        recv_expert_count = self._dispatch_state["recv_expert_count"]

        all_rank_max_num_tokens = kwargs.get("all_rank_max_num_tokens")
        assert all_rank_max_num_tokens is not None, (
            "all_rank_max_num_tokens must be provided in kwargs"
        )
        num_tokens_per_expert = self.mapping.moe_ep_size * all_rank_max_num_tokens

        final_hidden_states = final_hidden_states.view(
            self.expert_size_per_partition, num_tokens_per_expert, self.hidden_size
        )

        if self.use_low_precision_combine:
            if self._has_nvfp4():
                precision = "nvfp4"
                global_scales = torch.ops.trtllm.calculate_nvfp4_global_scale(
                    final_hidden_states, recv_expert_count
                )
            else:
                precision = "fp8"
                global_scales = None

            final_hidden_states = self.deep_ep_buffer.low_latency_combine_low_precision(
                precision,
                final_hidden_states,
                global_scales,
                deep_ep_topk_idx,
                deep_ep_topk_weights,
                deep_ep_handle,
            )
        else:
            final_hidden_states = self.deep_ep_buffer.low_latency_combine(
                final_hidden_states, deep_ep_topk_idx, deep_ep_topk_weights, deep_ep_handle
            )

        return final_hidden_states

    def _has_nvfp4(self) -> bool:
        """Check if NVFP4 quantization is enabled"""
        return self.quant_config is not None and self.quant_config.layer_quant_mode.has_nvfp4()

    def _has_fp8_qdq(self) -> bool:
        """Check if FP8 QDQ quantization is enabled"""
        return self.quant_config is not None and self.quant_config.layer_quant_mode.has_fp8_qdq()

    def _has_w4afp8(self) -> bool:
        """Check if W4AFP8 quantization is enabled"""
        return (
            self.quant_config is not None
            and self.quant_config.quant_mode.is_int4_weight_only_per_group()
        )

    def _modify_output_to_adapt_fused_moe(
        self,
        hidden_states: torch.Tensor,
        hidden_states_sf: Optional[torch.Tensor],
        recv_expert_count: torch.Tensor,
        final_scales_dtype: torch.dtype,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor, torch.Tensor]:
        """
        Adapter for DeepEP output to match fused_moe interface

        hidden_states shape: [#local experts, EP size * all_rank_max_num_tokens, hidden_size]
        recv_expert_count shape: [#local experts]

        TODO: remove the adapter by changing `torch.ops.trtllm.fused_moe` API
        """
        mask = torch.arange(
            hidden_states.shape[1], dtype=torch.int32, device=hidden_states.device
        ).expand(hidden_states.shape[0], hidden_states.shape[1]) < recv_expert_count.unsqueeze(1)

        token_selected_slots = torch.where(
            mask,
            torch.arange(
                hidden_states.shape[0] * self.mapping.moe_ep_rank,
                hidden_states.shape[0] * (self.mapping.moe_ep_rank + 1),
                dtype=torch.int32,
                device=hidden_states.device,
            ).unsqueeze(1),
            self.num_slots,
        )

        hidden_states = hidden_states.reshape(
            hidden_states.shape[0] * hidden_states.shape[1], hidden_states.shape[2]
        )
        if hidden_states_sf is not None:
            hidden_states_sf = hidden_states_sf.reshape(
                hidden_states_sf.shape[0] * hidden_states_sf.shape[1], hidden_states_sf.shape[2]
            )

        token_selected_slots = token_selected_slots.view(hidden_states.shape[0], 1)
        token_final_scales = torch.ones_like(token_selected_slots, dtype=final_scales_dtype)

        return hidden_states, hidden_states_sf, token_selected_slots, token_final_scales
