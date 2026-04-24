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

"""DeepSeek V4 model variant with explicit AutoDeploy sharding IR hints.

This module is imported only when ``AD_USE_IR_MODELS`` is set.  It keeps the
base DeepSeek V4 module hierarchy and checkpoint key names intact while making
the attention head/group reshapes and rowwise output collective explicit.
"""

import torch
from torch import nn

from tensorrt_llm._torch.auto_deploy.models.factory import ModelFactoryRegistry
from tensorrt_llm._torch.auto_deploy.models.hf import AutoModelForCausalLMFactory

from .modeling_deepseek_v4 import DeepseekV4Attention as _BaseDeepseekV4Attention
from .modeling_deepseek_v4 import (
    DeepseekV4AutoModelForCausalLMFactory as _BaseDeepseekV4AutoModelForCausalLMFactory,
)
from .modeling_deepseek_v4 import DeepseekV4Block as _BaseDeepseekV4Block
from .modeling_deepseek_v4 import (
    DeepseekV4Config,
    _apply_rope,
    _compress_topk_idxs,
    _window_topk_idxs,
)
from .modeling_deepseek_v4 import DeepseekV4ForCausalLM as _BaseDeepseekV4ForCausalLM


class DeepseekV4Attention(_BaseDeepseekV4Attention):
    """DeepSeek V4 attention that emits explicit sharding-aware IR ops."""

    def forward(self, x: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        freqs_cis_table = self.rotary_emb()
        freqs_cis = freqs_cis_table[position_ids]

        qr = self.q_norm(self.wq_a(x, layer_type="mla"))
        q = self.wq_b(qr, tp_mode="colwise", layer_type="mla")
        q = torch.ops.auto_deploy.view(
            q,
            [batch_size, seq_len, self.num_heads, self.head_dim],
            tp_scaled_dim=2,
            layer_type="mla",
        )
        q = q * torch.rsqrt(q.square().mean(-1, keepdim=True) + self.eps)
        q_rope = _apply_rope(q[..., -self.rope_head_dim :], freqs_cis)
        q = torch.cat([q[..., : -self.rope_head_dim], q_rope], dim=-1)

        kv = self.kv_norm(self.wkv(x, layer_type="mla"))
        kv_rope = _apply_rope(kv[..., -self.rope_head_dim :], freqs_cis)
        kv = torch.cat([kv[..., : -self.rope_head_dim], kv_rope], dim=-1)

        topk_idxs = _window_topk_idxs(self.window_size, batch_size, seq_len, x.device)
        if self.compress_ratio:
            compressed_kv = self.compressor(x, position_ids, freqs_cis_table)
            compressed_idxs = _compress_topk_idxs(
                self.compress_ratio,
                batch_size,
                seq_len,
                seq_len,
                x.device,
                self.compressor.max_compressed_len,
            )
            topk_idxs = torch.cat([topk_idxs, compressed_idxs], dim=-1)
            kv = torch.cat([kv, compressed_kv], dim=1)

        o = torch.ops.auto_deploy.torch_deepseek_v4_sparse_attention(
            q,
            kv,
            self.attn_sink,
            topk_idxs.int(),
            self.softmax_scale,
            enable_sharding=True,
            layer_type="mla",
        )
        o_rope = _apply_rope(o[..., -self.rope_head_dim :], freqs_cis, inverse=True)
        o = torch.cat([o[..., : -self.rope_head_dim], o_rope], dim=-1)

        o = torch.ops.auto_deploy.view(
            o,
            [batch_size, seq_len, self.num_groups, -1],
            tp_scaled_dim=2,
            layer_type="mla",
        )
        wo_a = self.wo_a.weight.view(self.num_groups, self.o_lora_rank, -1)
        o = torch.einsum("bsgd,grd->bsgr", o, wo_a)
        o = self.wo_b(o.flatten(2), tp_mode="rowwise", layer_type="mla")
        return torch.ops.auto_deploy.all_reduce(o, layer_type="mla")


class DeepseekV4Block(_BaseDeepseekV4Block):
    """DeepSeek V4 block that swaps in the sharding-aware attention module."""

    def __init__(self, config: DeepseekV4Config, layer_idx: int) -> None:
        super().__init__(config, layer_idx)
        self.attn = DeepseekV4Attention(config, layer_idx)


class DeepseekV4ForCausalLM(_BaseDeepseekV4ForCausalLM):
    """DeepSeek V4 causal LM using sharding-aware decoder blocks."""

    def __init__(self, config: DeepseekV4Config, **kwargs) -> None:
        super().__init__(config, **kwargs)
        self.layers = nn.ModuleList(
            [DeepseekV4Block(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.post_init()


AutoModelForCausalLMFactory.register_custom_model_cls("DeepseekV4Config", DeepseekV4ForCausalLM)


@ModelFactoryRegistry.register("DeepseekV4AutoModelForCausalLM")
class DeepseekV4AutoModelForCausalLMFactory(_BaseDeepseekV4AutoModelForCausalLMFactory):
    pass


DeepseekV4AutoModelForCausalLMFactory.register_custom_model_cls(
    "DeepseekV4Config", DeepseekV4ForCausalLM
)
