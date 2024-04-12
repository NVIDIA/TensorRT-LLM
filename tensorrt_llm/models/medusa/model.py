# SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from collections import OrderedDict

import tensorrt as trt

from tensorrt_llm.models.llama.model import LLaMAForCausalLM

from ..._common import default_net
from ..._utils import pad_vocab_size
from ...functional import ACT2FN, Tensor, stack
from ...layers import ColumnLinear
from ...mapping import Mapping
from ...module import Module, ModuleList
from ..generation_mixin import GenerationMixin


class MedusaLayer(Module):

    def __init__(
            self,
            hidden_size,
            hidden_act="silu",
            dtype=None,
            mapping=Mapping(),
    ):
        super().__init__()
        self.linear = ColumnLinear(hidden_size,
                                   hidden_size,
                                   dtype=dtype,
                                   tp_group=mapping.tp_group,
                                   tp_size=mapping.tp_size,
                                   gather_output=True)
        self.hidden_act = hidden_act

    def forward(self, x):
        return x + ACT2FN[self.hidden_act](self.linear(x))


class MedusaHead(Module):

    def __init__(
            self,
            num_layers,
            hidden_size,
            vocab_size,
            hidden_act="silu",
            dtype=None,
            mapping=Mapping(),
    ):
        super().__init__()
        self.medusa_layers = ModuleList([
            MedusaLayer(hidden_size=hidden_size,
                        hidden_act=hidden_act,
                        dtype=dtype,
                        mapping=mapping) for _ in range(num_layers)
        ])
        self.lm_head = ColumnLinear(hidden_size,
                                    vocab_size,
                                    bias=False,
                                    dtype=dtype,
                                    tp_group=mapping.tp_group,
                                    tp_size=mapping.tp_size,
                                    gather_output=True)
        return

    def forward(self, x):
        hidden_states = x
        for layer in self.medusa_layers:
            hidden_states = layer(hidden_states)
        return self.lm_head(hidden_states)


class MedusaForCausalLm(LLaMAForCausalLM):

    def __init__(self, config):

        super().__init__(config)
        self.num_medusa_heads = config.num_medusa_heads
        self.num_medusa_layers = config.num_medusa_layers
        self.hidden_size = config.hidden_size
        self.vocab_size = config.vocab_size
        vocab_size_padded = pad_vocab_size(self.vocab_size,
                                           config.mapping.tp_size)
        self.medusa_heads = ModuleList([
            MedusaHead(num_layers=self.num_medusa_layers,
                       hidden_size=config.hidden_size,
                       vocab_size=vocab_size_padded,
                       hidden_act=config.hidden_act,
                       dtype=config.dtype,
                       mapping=config.mapping)
            for _ in range(self.num_medusa_heads)
        ])
        self.max_medusa_token_len = config.max_draft_len

    def forward(self, *args, **kwargs):
        output_original = True
        hidden_states = super().forward(*args, **kwargs)

        if kwargs['use_cache']:
            if default_net().plugin_config.paged_kv_cache:
                lm_logits, hidden_states = hidden_states
            else:
                lm_logits, presents, hidden_states = hidden_states

        if self.mapping.is_last_pp_rank():
            medusa_logits = []
            for i in range(self.num_medusa_heads):
                medusa_logits.append(self.medusa_heads[i](hidden_states))
            # [num_medusa_heads, batch_size, num_medusa_tokens + 1, padded_vocab_size].
            # Remove padding [num_medusa_heads, batch_size * num_medusa_tokens + 1, padded_vocab_size].
            medusa_logits = stack(medusa_logits, dim=0)
            medusa_logits.mark_output('medusa_logits', self.config.logits_dtype)
        else:
            hidden_states.mark_output('hidden_states_output', self.config.dtype)

        if kwargs['use_cache'] and default_net(
        ).plugin_config.paged_kv_cache == False:
            if self.mapping.is_last_pp_rank():
                if output_original:
                    return (medusa_logits, lm_logits, presents)
                return (medusa_logits, presents)
            return (hidden_states, presents)
        else:
            if self.mapping.is_last_pp_rank():
                if output_original:
                    return medusa_logits, lm_logits
                return medusa_logits
            return hidden_states

    def prepare_inputs(self, *args, **kwargs):
        kwargs['max_draft_len'] = self.max_medusa_token_len
        inputs = super().prepare_inputs(*args, **kwargs)
        num_profiles = len(inputs['input_ids'].profiles)
        max_gen_token_len = self.max_medusa_token_len + 1
        medusa_mask_len_range = [[0, max_gen_token_len, max_gen_token_len]
                                 ] * num_profiles
        medusa_position_len_range = [[0, max_gen_token_len, max_gen_token_len]
                                     ] * num_profiles
        # # 32 bits packed mask aligned.
        num_packed_medusa_masks = (self.max_medusa_token_len + 1 + 32 - 1) // 32
        packed_medusa_mask_len_range = [[0, 1, num_packed_medusa_masks]
                                        ] * num_profiles

        # batch beam range (different sequence may have different medusa offsets or packed masks).
        bb_range_cxt = GenerationMixin.default_range(kwargs['max_batch_size'])
        bb_range_gen = GenerationMixin.default_range(kwargs['max_batch_size'] *
                                                     kwargs['max_beam_width'])
        # enable_two_optimization_profiles
        if num_profiles == 2:
            bb_range = [bb_range_cxt, bb_range_gen]
        else:
            bb_range = [bb_range_gen]

        # medusa position offsets that are fixed during the whole session.
        # it will be shared among all sequences.
        medusa_position_offsets = Tensor(
            name='medusa_position_offsets',
            dtype=trt.int32,
            shape=[-1, -1],
            dim_range=OrderedDict([
                ('batch_size_beam_width', bb_range),
                ('medusa_position_ids_dim0', medusa_position_len_range),
            ]),
        )

        medusa_packed_mask = Tensor(
            name='medusa_packed_mask',
            dtype=trt.int32,
            shape=[-1, -1, -1],
            dim_range=OrderedDict([
                ('batch_size_beam_width', bb_range),
                ('medusa_packed_mask_dim0', medusa_mask_len_range),
                ('medusa_packed_mask_dim1', packed_medusa_mask_len_range),
            ]),
        )
        inputs['medusa_packed_mask'] = medusa_packed_mask
        inputs['medusa_position_offsets'] = medusa_position_offsets
        return inputs
