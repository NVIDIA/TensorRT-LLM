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

import json
from typing import Optional, Union

from transformers import AutoModelForCausalLM

from tensorrt_llm._utils import numpy_to_torch
from tensorrt_llm.models.llama.model import LLaMAForCausalLM
from tensorrt_llm.models.medusa.weight import load_medusa_hf
from tensorrt_llm.models.qwen.model import QWenForCausalLM

from ..._common import default_net
from ..._utils import pad_vocab_size
from ...functional import ACT2FN, stack
from ...layers import ColumnLinear
from ...mapping import Mapping
from ...module import Module, ModuleList
from ..modeling_utils import PretrainedModel, QuantConfig
from .config import MedusaConfig
from .weight import convert_hf_llama


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


# MedusaForCausalLm is a thin wrapper that picks parent class for GenericMedusaForCausalLM.
# All medusa functionality is defined in GenericMedusaForCausalLM.
class MedusaForCausalLm(PretrainedModel):
    config_class = MedusaConfig

    def __init__(self, config: MedusaConfig):
        super().__init__(config)

        BaseLM = QWenForCausalLM if hasattr(
            config,
            "model_type") and "qwen" in config.model_type else LLaMAForCausalLM

        class GenericMedusaForCausalLM(BaseLM):

            def __init__(self, config: MedusaConfig):
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
                        lm_logits, hidden_states, _ = hidden_states
                    else:
                        lm_logits, presents, hidden_states = hidden_states

                if self.mapping.is_last_pp_rank():
                    medusa_logits = []
                    for i in range(self.num_medusa_heads):
                        medusa_logits.append(
                            self.medusa_heads[i](hidden_states))
                    # [num_medusa_heads, batch_size, num_medusa_tokens + 1, padded_vocab_size].
                    # Remove padding [num_medusa_heads, batch_size * num_medusa_tokens + 1, padded_vocab_size].
                    medusa_logits = stack(medusa_logits, dim=0)
                    medusa_logits.mark_output('medusa_logits',
                                              self.config.logits_dtype)
                else:
                    hidden_states.mark_output('hidden_states_output',
                                              self.config.dtype)

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
                kwargs['speculative_decoding_draft_tokens_external'] = False
                kwargs['max_draft_len'] = self.max_medusa_token_len
                return super().prepare_inputs(*args, **kwargs)

        self.model = GenericMedusaForCausalLM(config)

    # Specialization to redirect accesses to self.model
    def __getattribute__(self, name):
        if name == 'model' or '__' in name:
            return object.__getattribute__(self, name)
        else:
            model = object.__getattribute__(self, 'model')
            return model.__getattribute__(name)

    # Override specialized __setattr__ defined in Module
    def __setattr__(self, name, value) -> None:
        object.__setattr__(self, name, value)

    @classmethod
    def from_hugging_face(
            cls,
            hf_model_or_dir: Union[str, 'transformers.PreTrainedModel'],
            dtype: str = 'auto',
            mapping: Optional[Mapping] = None,
            quant_config: Optional[QuantConfig] = None,
            **kwargs):
        import transformers

        assert hf_model_or_dir is not None
        speculative_model_dir = kwargs.get('speculative_model_dir', None)

        use_preloading = isinstance(hf_model_or_dir,
                                    transformers.PreTrainedModel)
        if use_preloading:
            hf_model = hf_model_or_dir
            hf_config_or_dir = hf_model.config
        else:
            hf_model_dir = hf_model_or_dir
            hf_config_or_dir = hf_model_or_dir

        config = MedusaConfig.from_hugging_face(hf_config_or_dir,
                                                dtype=dtype,
                                                mapping=mapping,
                                                quant_config=quant_config,
                                                **kwargs)

        # ModelOpt ckpt has combined base model and Medusa-head
        is_modelopt_ckpt = True if not speculative_model_dir else False

        if not use_preloading:
            trust_remote_code = kwargs.pop('trust_remote_code', True)

            if is_modelopt_ckpt:
                hf_model = LLaMAForCausalLM.from_hugging_face(
                    hf_model_dir,
                    dtype,
                    mapping=mapping,
                    quant_config=quant_config,
                    **kwargs)
            else:
                hf_model = AutoModelForCausalLM.from_pretrained(
                    hf_model_dir,
                    dtype="auto",
                    trust_remote_code=trust_remote_code)

                assert isinstance(hf_model, transformers.PreTrainedModel)

        if is_modelopt_ckpt:
            weights = {
                name: numpy_to_torch(param.raw_value)
                for name, param in hf_model.named_parameters()
            }
        else:
            weights = convert_hf_llama(
                hf_model,
                config.mapping,
                dtype='float16',
                use_parallel_embedding=config.use_parallel_embedding)

        model = cls(config)

        if is_modelopt_ckpt:
            num_medusa_heads = config.config.num_medusa_heads
            num_medusa_layers = config.config.num_medusa_layers
            speculative_model_dir = hf_model_or_dir
        else:
            config_file = speculative_model_dir / "config.json"
            with open(config_file) as fp:
                model_config = json.load(fp)

            num_medusa_heads = kwargs[
                'speculative_config'].num_medusa_heads if 'speculative_config' in kwargs else model_config.get(
                    'medusa_num_heads', None)
            num_medusa_layers = model_config.get('medusa_num_layers', None)
        medusa_weights = load_medusa_hf(medusa_path=speculative_model_dir,
                                        num_medusa_heads=num_medusa_heads,
                                        num_medusa_layers=num_medusa_layers,
                                        mapping=mapping,
                                        dtype="float16",
                                        is_modelopt_ckpt=is_modelopt_ckpt)
        weights.update(medusa_weights)
        model.load(weights)
        return model
