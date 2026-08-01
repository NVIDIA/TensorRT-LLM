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

import torch
from torch import nn

from tensorrt_llm._torch.models.checkpoints.hf.weight_mapper import HfWeightMapper
from tensorrt_llm._torch.models.modeling_utils import register_mapper
from tensorrt_llm._torch.modules.fused_moe.interface import MoE


@register_mapper("HF", "AfmoeForCausalLM")
class AfmoeHfWeightMapper(HfWeightMapper):
    def __init__(self):
        super().__init__()

        self.params_map = {
            # MoE expert weights: gate_proj->w1, up_proj->w3, down_proj->w2
            r"(.*experts\.\d+\.)gate_proj(.*)": r"\1w1\2",
            r"(.*experts\.\d+\.)up_proj(.*)": r"\1w3\2",
            r"(.*experts\.\d+\.)down_proj(.*)": r"\1w2\2",
            # HF router weight path -> TRT-LLM gate path
            r"(.*)\.router\.gate\.(.*)": r"\1.gate.\2",
            # expert_bias -> gate.e_score_correction_bias
            r"(.*)\.mlp\.expert_bias(.*)": r"\1.mlp.gate.e_score_correction_bias\2",
        }

    def preprocess_weights(self, weights: dict) -> dict:
        weights = self.rename_by_params_map(self.params_map, weights)
        weights = self._fuse_attention_gate(weights)
        return weights

    def _fuse_attention_gate(self, weights: dict) -> dict:
        """Fuse the separate attention ``gate_proj`` into ``q_proj``.

        AfmoeAttention uses ``attn_output_gate=True``, so the gate weights are
        interleaved with the query weights per head and loaded through the fused
        QKV projection. The HF checkpoint stores ``q_proj`` and ``gate_proj`` as
        two separate matrices of shape ``[num_heads * head_dim, hidden]``; the
        fused QKV projection expects the query slot laid out per head as
        ``[head0_q, head0_gate, head1_q, head1_gate, ...]`` (see
        ``Attention.forward`` where ``q_gate`` is viewed as
        ``[..., num_heads, 2 * head_dim]`` and chunked into q/gate).
        """
        marker = ".self_attn.gate_proj."
        gate_keys = [k for k in weights if marker in k]
        if not gate_keys:
            return weights

        num_heads = self.model.config.num_attention_heads
        for gate_key in gate_keys:
            prefix, suffix = gate_key.split(marker)
            q_key = f"{prefix}.self_attn.q_proj.{suffix}"
            if q_key not in weights:
                continue
            weights[q_key] = self._interleave_per_head(weights[q_key], weights[gate_key], num_heads)
            del weights[gate_key]
        return weights

    @staticmethod
    def _interleave_per_head(q: torch.Tensor, gate: torch.Tensor, num_heads: int) -> torch.Tensor:
        """Interleave q and gate rows per head: ``[h0_q, h0_gate, h1_q, ...]``.

        Works for 2D weights ``[num_heads * per_head, hidden]`` as well as 1D
        biases and FP8 block scales, since the split is always taken along the
        leading (output) dimension.
        """
        assert q.shape[0] % num_heads == 0, (
            f"q_proj rows {q.shape[0]} not divisible by num_heads {num_heads}"
        )
        assert gate.shape == q.shape, f"gate_proj shape {gate.shape} != q_proj shape {q.shape}"
        per_head = q.shape[0] // num_heads
        tail = q.shape[1:]
        q = q.reshape(num_heads, per_head, *tail)
        gate = gate.reshape(num_heads, per_head, *tail)
        fused = torch.stack([q, gate], dim=1)
        return fused.reshape(num_heads * 2 * per_head, *tail).contiguous()

    def is_special_instance_module(self, module: nn.Module) -> bool:
        return isinstance(module, MoE)

    def handle_special_instance_module(
        self,
        module: nn.Module,
        module_name: str,
        module_weights: dict,
        allow_partial_loading: bool = False,
    ) -> None:
        if isinstance(module, MoE):
            module.load_weights(
                weights=[module_weights],
                allow_partial_loading=allow_partial_loading,
            )
