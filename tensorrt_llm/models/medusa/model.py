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
import inspect
from collections import OrderedDict
from typing import Optional

import tensorrt as trt

from tensorrt_llm.models.generation_mixin import GenerationMixin

from ..._common import default_net
from ..._utils import pad_vocab_size
from ...functional import ACT2FN, Tensor, gather_last_token_logits, stack
from ...layers import ColumnLinear
from ...mapping import Mapping
from ...module import Module, ModuleList


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
        return

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


class MedusaLM(Module, GenerationMixin):

    def __init__(
        self,
        base_model: Module,
        mapping=Mapping(),
        num_medusa_heads=4,
        num_medusa_layers=1,
        hidden_act='silu',
    ):
        super().__init__()
        self.base_model = base_model
        self.mapping = mapping
        self.hidden_size = base_model.hidden_size
        self.vocab_size = base_model.vocab_size
        self.num_medusa_heads = num_medusa_heads
        self.num_medusa_layers = num_medusa_layers
        self.hidden_act = hidden_act
        vocab_size_padded = pad_vocab_size(self.vocab_size, mapping.tp_size)
        self.medusa_heads = ModuleList([
            MedusaHead(num_layers=num_medusa_layers,
                       hidden_size=self.hidden_size,
                       vocab_size=vocab_size_padded,
                       hidden_act=self.hidden_act,
                       dtype=base_model.dtype,
                       mapping=mapping) for _ in range(num_medusa_heads)
        ])
        return

    def forward(
        self,
        input_ids,
        position_ids=None,
        use_cache=False,
        last_token_ids=None,
        attention_mask=None,
        medusa_position_offsets=None,
        medusa_packed_mask=None,
        kv_cache_params=None,
        attention_params=None,
        hidden_states=None,
        prompt_embedding_table: Optional[Tensor] = None,
        prompt_tasks: Optional[Tensor] = None,
        prompt_vocab_size: Optional[Tensor] = None,
        lora_params=None,
        output_original=True,
    ):
        parent_model = inspect.getmro(type(
            self.base_model))[1]  # get the pre-LMHead model
        hidden_states = parent_model.forward(
            self.base_model, input_ids, position_ids, use_cache, attention_mask,
            medusa_position_offsets, medusa_packed_mask, kv_cache_params,
            attention_params, hidden_states, prompt_embedding_table,
            prompt_tasks, prompt_vocab_size, lora_params)
        if use_cache:
            hidden_states, presents = hidden_states

        if self.mapping.is_last_pp_rank():
            hidden_states = gather_last_token_logits(
                hidden_states,
                last_token_ids,
                default_net().plugin_config.remove_input_padding,
            )
            # [batch_size, hidden_size] -> [batch_size, vocab_size]
            if output_original:
                lm_logits = self.base_model.lm_head(hidden_states)
                lm_logits.mark_output('logits', self.base_model.logits_dtype)

            medusa_logits = []
            for i in range(self.num_medusa_heads):
                medusa_logits.append(self.medusa_heads[i](hidden_states))
            medusa_logits = stack(medusa_logits, dim=1)
            medusa_logits.mark_output('medusa_logits',
                                      self.base_model.logits_dtype)
        else:
            hidden_states.mark_output('hidden_states_output',
                                      self.base_model.dtype)

        if use_cache and default_net().plugin_config.paged_kv_cache == False:
            for i, present in zip(
                    self.mapping.pp_layers(self.base_model.num_layers),
                    presents):
                present.mark_output(f'present_key_value_{i}',
                                    self.base_model.kv_dtype)
            if self.mapping.is_last_pp_rank():
                if output_original:
                    return (medusa_logits, lm_logits, presents)
                return (medusa_logits, presents)
            return (hidden_states, presents)
        else:
            if self.mapping.is_last_pp_rank():
                if output_original:
                    return medusa_logits, lm_logits
                return lm_logits
            return hidden_states

    def prepare_inputs(
        self,
        max_batch_size,
        max_input_len,
        max_new_tokens,
        use_cache,
        max_medusa_tokens_len,
        max_beam_width,
        max_num_tokens: int = None,
        prompt_embedding_table_size: int = 0,
    ):
        base_model_inputs = self.base_model.prepare_inputs(
            max_batch_size,
            max_input_len,
            max_new_tokens,
            use_cache,
            max_beam_width,
            max_num_tokens,
            prompt_embedding_table_size,
            max_draft_len=max_medusa_tokens_len)
        num_profiles = len(base_model_inputs[0].profiles)
        max_gen_token_len = max_medusa_tokens_len + 1
        medusa_mask_len_range = [[0, max_gen_token_len, max_gen_token_len]
                                 ] * num_profiles
        medusa_position_len_range = [[0, max_gen_token_len, max_gen_token_len]
                                     ] * num_profiles
        # 32 bits packed mask aligned.
        num_packed_medusa_masks = (max_medusa_tokens_len + 1 + 32 - 1) // 32
        packed_medusa_mask_len_range = [[0, 1, num_packed_medusa_masks]
                                        ] * num_profiles

        # medusa position offsets that are fixed during the whole session.
        # it will be shared among all sequences.
        medusa_position_offsets = Tensor(
            name='medusa_position_offsets',
            dtype=trt.int32,
            shape=[-1],
            dim_range=OrderedDict([
                ('medusa_position_ids_dim0', medusa_position_len_range),
            ]),
        )

        medusa_packed_mask = Tensor(
            name='medusa_packed_mask',
            dtype=trt.int32,
            shape=[-1, -1],
            dim_range=OrderedDict([
                ('medusa_packed_mask_dim0', medusa_mask_len_range),
                ('medusa_packed_mask_dim1', packed_medusa_mask_len_range),
            ]),
        )

        return base_model_inputs[:5] + (
            medusa_position_offsets,
            medusa_packed_mask,
        ) + base_model_inputs[5:]
