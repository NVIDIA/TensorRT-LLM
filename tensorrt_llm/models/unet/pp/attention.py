# SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from ....functional import allgather, split
from ....mapping import Mapping
from ....module import Module
from ..attention import CrossAttention, SelfAttention, _attention


class DistriSelfAttentionPP(Module):

    def __init__(self, module: SelfAttention, mapping: Mapping = Mapping()):
        super().__init__()
        self.mapping = mapping
        self.module = module

    def forward(self, hidden_states):
        mapping = self.mapping
        attn = self.module

        batch_size, sequence_length, _ = hidden_states.shape

        qkv = attn.to_qkv(hidden_states)

        query, kv = split(qkv, [attn.inner_dim, attn.inner_dim * 2], dim=2)

        if mapping.tp_size == 1:
            full_kv = kv
        else:
            full_kv = allgather(kv, group=mapping.tp_group, gather_dim=1)

        key, value = split(full_kv, full_kv.shape[-1] // 2, dim=-1)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view([batch_size, -1, attn.heads,
                            head_dim]).transpose(1, 2)
        key = key.view([batch_size, -1, attn.heads, head_dim]).transpose(1, 2)
        value = value.view([batch_size, -1, attn.heads,
                            head_dim]).transpose(1, 2)

        hidden_states = _attention(query, key, value, attn.scale)

        hidden_states = hidden_states.view(
            [batch_size, -1, attn.heads * head_dim])

        # linear proj
        hidden_states = attn.to_out(hidden_states)

        return hidden_states


class DistriCrossAttentionPP(Module):

    def __init__(self, module: CrossAttention, mapping: Mapping = Mapping()):
        super().__init__()
        self.mapping = mapping
        self.module = module
        self.kv_cache = None

    def forward(self, hidden_states, context):
        attn = self.module
        recompute_kv = self.kv_cache is None

        if context is None:
            context = hidden_states

        batch_size, sequence_length, _ = context.shape

        query = attn.to_q(hidden_states)

        if recompute_kv or self.kv_cache is None:
            kv = attn.to_kv(context)
            self.kv_cache = kv
        else:
            kv = self.kv_cache
        key, value = split(kv, kv.shape[-1] // 2, dim=-1)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view([batch_size, -1, attn.heads,
                            head_dim]).transpose(1, 2)
        key = key.view([batch_size, -1, attn.heads, head_dim]).transpose(1, 2)
        value = value.view([batch_size, -1, attn.heads,
                            head_dim]).transpose(1, 2)

        hidden_states = _attention(query, key, value, scale=attn.scale)

        hidden_states = hidden_states.view(
            [batch_size, -1, attn.heads * head_dim])

        # linear proj
        hidden_states = attn.to_out(hidden_states)
        return hidden_states
