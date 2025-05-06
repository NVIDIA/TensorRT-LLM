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
from typing import Optional, OrderedDict, Union

import numpy as np
import tensorrt as trt
import torch
import transformers

from tensorrt_llm.models.modeling_utils import PretrainedModel

from ..._common import default_net
from ...functional import (ACT2FN, Tensor, concat, constant, cumsum, expand,
                           index_select, select, shape, slice, unsqueeze)
from ...layers import MLP, BertAttention, Embedding, LayerNorm, Linear
from ...mapping import Mapping
from ...module import Module, ModuleList
from ..modeling_utils import QuantConfig
from .config import BERTConfig
from .convert import (load_hf_bert_base, load_hf_bert_cls, load_hf_bert_qa,
                      load_weights_from_hf_model)


class BertEmbedding(Module):

    def __init__(self,
                 vocab_size,
                 hidden_size,
                 max_position_embeddings,
                 type_vocab_size,
                 dtype=None):
        super().__init__()
        self.vocab_embedding = Embedding(vocab_size, hidden_size, dtype=dtype)
        self.position_embedding = Embedding(max_position_embeddings,
                                            hidden_size,
                                            dtype=dtype)
        self.token_embedding = Embedding(type_vocab_size,
                                         hidden_size,
                                         dtype=dtype)
        self.max_position_embeddings = max_position_embeddings

        self.embedding_ln = LayerNorm(normalized_shape=hidden_size, dtype=dtype)

    def forward(self, input_ids, position_ids, token_type_ids):
        x = self.vocab_embedding(input_ids)
        x = x + self.position_embedding(position_ids)
        x = x + self.token_embedding(token_type_ids)
        x = self.embedding_ln(x)
        return x


class BertEncoderLayer(Module):

    def __init__(self,
                 hidden_size,
                 num_attention_heads,
                 max_position_embeddings,
                 hidden_act='relu',
                 tp_group=None,
                 tp_size=1,
                 dtype=None):
        super().__init__()
        self.input_layernorm = LayerNorm(normalized_shape=hidden_size,
                                         dtype=dtype)

        self.attention = BertAttention(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            max_position_embeddings=max_position_embeddings,
            tp_group=tp_group,
            tp_size=tp_size,
            dtype=dtype)
        self.mlp = MLP(hidden_size=hidden_size,
                       ffn_hidden_size=hidden_size * 4,
                       hidden_act=hidden_act,
                       tp_group=tp_group,
                       tp_size=tp_size,
                       dtype=dtype)
        self.post_layernorm = LayerNorm(normalized_shape=hidden_size,
                                        dtype=dtype)

    def forward(self,
                hidden_states,
                attention_mask=None,
                input_lengths=None,
                max_input_length=None):
        residual = hidden_states

        attention_output = self.attention(hidden_states,
                                          attention_mask=attention_mask,
                                          input_lengths=input_lengths,
                                          max_input_length=max_input_length)

        hidden_states = residual + attention_output

        hidden_states = self.input_layernorm(hidden_states)

        residual = hidden_states

        hidden_states = self.mlp(hidden_states)

        hidden_states = residual + hidden_states

        hidden_states = self.post_layernorm(hidden_states)

        return hidden_states


class BertBase(PretrainedModel):
    '''
    Base class that provides from_huggingface() and prepare_inputs() methods
    '''
    config_class = BERTConfig

    def __init__(self, config: BERTConfig):
        super().__init__(config)

    @classmethod
    def load_hf_bert(cls, model_dir: str, load_model_on_cpu: bool,
                     dtype: torch.dtype):
        """
        Use as the abstractmethod, load corresponding HF model.
        Subclass must implement this method!
        """

        assert cls.__name__ != "BertBase", f"Never call from BertBase class!"

        if cls.__name__ == "BertModel":
            return load_hf_bert_base(model_dir, load_model_on_cpu, dtype)
        elif cls.__name__ == "BertForQuestionAnswering":
            return load_hf_bert_qa(model_dir, load_model_on_cpu, dtype)
        elif cls.__name__ == "BertForSequenceClassification":
            return load_hf_bert_cls(model_dir, load_model_on_cpu, dtype)
        else:
            assert False, f"Unknown class {cls.__name__}!"

    @classmethod
    def from_hugging_face(
            cls,
            hf_model_or_dir: Union[str, 'transformers.PreTrainedModel'],
            dtype: str = 'float16',
            mapping: Optional[Mapping] = None,
            quant_config: Optional[QuantConfig] = None,
            **kwargs):
        """
        Create a BertModel object from give parameters
        """
        import transformers

        assert hf_model_or_dir is not None
        use_preloading = isinstance(hf_model_or_dir,
                                    transformers.PreTrainedModel)
        if use_preloading:
            hf_model = hf_model_or_dir
            hf_config_or_dir = hf_model.config
        else:
            hf_model_dir = hf_model_or_dir
            hf_config_or_dir = hf_model_or_dir

        load_model_on_cpu = kwargs.pop('load_model_on_cpu', False)
        tllm_config = BERTConfig.from_hugging_face(
            hf_config_or_dir=hf_config_or_dir,
            dtype=dtype,
            mapping=mapping,
            quant_config=quant_config,
            **kwargs)
        #NOTE: override architecture info
        RobertaCls_mapping = {
            "BertModel": "RobertaModel",
            "BertForQuestionAnswering": "RobertaForQuestionAnswering",
            "BertForSequenceClassification": "RobertaForSequenceClassification",
        }
        if tllm_config.is_roberta:
            setattr(tllm_config, 'architecture',
                    RobertaCls_mapping[cls.__name__])
        else:
            setattr(tllm_config, 'architecture', cls.__name__)

        torch_dtype = torch.float16 if dtype == 'float16' else torch.float32
        if not use_preloading:
            hf_model = cls.load_hf_bert(model_dir=hf_model_dir,
                                        load_model_on_cpu=load_model_on_cpu,
                                        dtype=torch_dtype)
        weights = load_weights_from_hf_model(hf_model=hf_model,
                                             config=tllm_config)
        model = cls(tllm_config)
        model.load(weights)

        return model

    # Override the PretrainedModel's meothd, can unify in the future.
    def prepare_inputs(self, max_batch_size, max_input_len, **kwargs):
        remove_input_padding = default_net().plugin_config.remove_input_padding
        # opt_shape is set to half of max batch_size and seq_len by default
        # tune this according to real data distribution
        bs_range = [1, (max_batch_size + 1) // 2, max_batch_size]
        inlen_range = [1, (max_input_len + 1) // 2, max_input_len]
        num_tokens_range = [
            1,
            (max_input_len * max_batch_size + 1) // 2,
            max_input_len * max_batch_size,
        ]
        if not remove_input_padding:
            input_ids = Tensor(
                name='input_ids',
                dtype=trt.int32,
                shape=[-1, -1],
                dim_range=OrderedDict([('batch_size', [bs_range]),
                                       ('input_len', [inlen_range])]),
            )
            # also called segment_ids
            token_type_ids = Tensor(
                name='token_type_ids',
                dtype=trt.int32,
                shape=[-1, -1],
                dim_range=OrderedDict([('batch_size', [bs_range]),
                                       ('input_len', [inlen_range])]),
            )
        else:
            input_ids = Tensor(
                name="input_ids",
                dtype=trt.int32,
                shape=[-1],
                dim_range=OrderedDict([("num_tokens", [num_tokens_range])]),
            )
            token_type_ids = Tensor(
                name='token_type_ids',
                dtype=trt.int32,
                shape=[-1],
                dim_range=OrderedDict([('num_tokens', [num_tokens_range])]),
            )
            position_ids = Tensor(
                name='position_ids',
                dtype=trt.int32,
                shape=[-1],
                dim_range=OrderedDict([('num_tokens', [num_tokens_range])]),
            )
            max_input_length = Tensor(
                name="max_input_length",
                dtype=trt.int32,
                shape=[-1],
                dim_range=OrderedDict([("max_input_length", [inlen_range])]),
            )
        input_lengths = Tensor(name='input_lengths',
                               dtype=trt.int32,
                               shape=[-1],
                               dim_range=OrderedDict([('batch_size', [bs_range])
                                                      ]))

        inputs = {
            'input_ids': input_ids,
            'input_lengths': input_lengths,
            'token_type_ids': token_type_ids,
        }

        if remove_input_padding:
            inputs['position_ids'] = position_ids
            inputs['max_input_length'] = max_input_length

        return inputs


class BertModel(BertBase):

    def __init__(self, config: BERTConfig):
        super().__init__(config)

        self.config = config
        self.max_position_embeddings = config.max_position_embeddings
        self.padding_idx = config.pad_token_id
        self.is_roberta = config.is_roberta
        self.embedding = BertEmbedding(
            vocab_size=config.vocab_size,
            hidden_size=config.hidden_size,
            max_position_embeddings=config.max_position_embeddings,
            type_vocab_size=config.type_vocab_size,
            dtype=config.dtype)

        self.layers = ModuleList([
            BertEncoderLayer(
                hidden_size=config.hidden_size,
                num_attention_heads=config.num_attention_heads,
                max_position_embeddings=config.max_position_embeddings,
                hidden_act=config.hidden_act,
                tp_group=config.mapping.tp_group,
                tp_size=config.mapping.tp_size,
                dtype=config.dtype) for _ in range(config.num_hidden_layers)
        ])

    def forward(self,
                input_ids=None,
                input_lengths=None,
                position_ids=None,
                token_type_ids=None,
                hidden_states=None,
                max_input_length=None):
        # remove_input_padding requires these fields as explicit input
        mask = None
        if not default_net().plugin_config.remove_input_padding:
            seq_len_2d = concat([1, shape(input_ids, 1)])

            # create position ids
            position_ids_buffer = constant(
                np.expand_dims(
                    np.arange(self.max_position_embeddings).astype(np.int32),
                    0))
            tmp_position_ids = slice(position_ids_buffer,
                                     starts=[0, 0],
                                     sizes=seq_len_2d)
            tmp_position_ids = expand(tmp_position_ids, shape(input_ids))  #BxL
            tmp_input_lengths = unsqueeze(input_lengths, 1)  #Bx1
            tmp_input_lengths = expand(tmp_input_lengths,
                                       shape(input_ids))  #BxL
            mask = tmp_position_ids < tmp_input_lengths  # BxL
            mask = mask.cast('int32')

            if position_ids is None:
                if self.is_roberta:
                    # see create_position_ids_from_input_ids() in https://github.com/huggingface/transformers/blob/main/src/transformers/models/roberta/modeling_roberta.py
                    position_ids = (tmp_position_ids + 1) * mask
                    position_ids = position_ids + self.padding_idx
                else:
                    position_ids = slice(position_ids_buffer,
                                         starts=[0, 0],
                                         sizes=seq_len_2d)
                    position_ids = expand(position_ids, shape(input_ids))

            # create token_type_ids
            if token_type_ids is None:
                token_type_ids_buffer = constant(
                    np.expand_dims(
                        np.zeros(self.max_position_embeddings).astype(np.int32),
                        0))
                token_type_ids = slice(token_type_ids_buffer,
                                       starts=[0, 0],
                                       sizes=seq_len_2d)
                token_type_ids = expand(token_type_ids, shape(input_ids))

        hidden_states = self.embedding(input_ids, position_ids, token_type_ids)
        self.register_network_output('embedding_output', hidden_states)

        for idx, layer in enumerate(self.layers):
            hidden_states = layer(hidden_states=hidden_states,
                                  input_lengths=input_lengths,
                                  attention_mask=mask,
                                  max_input_length=max_input_length)
            # keep the last layer output name as hidden_states
            if ((idx == (self.config.num_hidden_layers - 1)) and
                (self.config.architecture in ["BertModel", "RobertaModel"])):
                hidden_states.mark_output('hidden_states', self.config.dtype)
            else:
                self.register_network_output(f"layer_{idx}_output",
                                             hidden_states)

        return hidden_states


RobertaModel = BertModel


class BertForQuestionAnswering(BertBase):

    def __init__(self, config: BERTConfig):
        super().__init__(config)
        self.bert = BertModel(config)
        self.num_labels = config.num_labels
        self.qa_outputs = Linear(config.hidden_size,
                                 config.num_labels,
                                 dtype=config.dtype)

    def forward(self,
                input_ids=None,
                input_lengths=None,
                token_type_ids=None,
                position_ids=None,
                hidden_states=None,
                max_input_length=None):

        remove_input_padding = default_net().plugin_config.remove_input_padding
        if remove_input_padding:
            assert token_type_ids is not None and \
                   position_ids is not None and \
                   max_input_length is not None, \
                   "token_type_ids, position_ids, max_input_length is required " \
                   "in remove_input_padding mode"
        hidden_states = self.bert.forward(input_ids=input_ids,
                                          input_lengths=input_lengths,
                                          token_type_ids=token_type_ids,
                                          position_ids=position_ids,
                                          hidden_states=hidden_states,
                                          max_input_length=max_input_length)

        logits = self.qa_outputs(hidden_states)
        logits.mark_output('logits', self.config.logits_dtype)

        return logits


RobertaForQuestionAnswering = BertForQuestionAnswering


class BertPooler(Module):

    def __init__(self, hidden_size, dtype):
        super().__init__()
        self.dense = Linear(hidden_size, hidden_size, dtype=dtype)
        self.activation = ACT2FN['tanh']

    def forward(self, hidden_states, input_lengths, remove_input_padding):
        if not remove_input_padding:
            # We "pool" the model by simply taking the hidden state corresponding
            # to the first token.
            first_token_tensor = select(hidden_states, 1, 0)
        else:
            # when remove_input_padding is enabled, the shape of hidden_states is [num_tokens, hidden_size]
            # We can take the first token of each sequence according to input_lengths,
            # and then do pooling similar to padding mode.
            # For example, if input_lengths is [8, 5, 6], then the indices of first tokens
            # should be [0, 8, 13]
            first_token_indices = cumsum(
                concat([
                    0,
                    slice(input_lengths,
                          starts=[0],
                          sizes=(shape(input_lengths) -
                                 constant(np.array([1], dtype=np.int32))))
                ]), 0)
            first_token_tensor = index_select(hidden_states, 0,
                                              first_token_indices)

        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class RobertaClassificationHead(Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, hidden_size, dtype, num_labels):
        super().__init__()
        self.dense = Linear(hidden_size, hidden_size, dtype=dtype)
        self.out_proj = Linear(hidden_size, num_labels)

    def forward(self, hidden_states, input_lengths, remove_input_padding):

        if not remove_input_padding:
            # We "pool" the model by simply taking the hidden state corresponding
            # to the first token.
            first_token_tensor = select(hidden_states, 1, 0)
        else:
            # when remove_input_padding is enabled, the shape of hidden_states is [num_tokens, hidden_size]
            # We can take the first token of each sequence according to input_lengths,
            # and then do pooling similar to padding mode.
            # For example, if input_lengths is [8, 5, 6], then the indices of first tokens
            # should be [0, 8, 13]
            first_token_indices = cumsum(
                concat([
                    0,
                    slice(input_lengths,
                          starts=[0],
                          sizes=(shape(input_lengths) -
                                 constant(np.array([1], dtype=np.int32))))
                ]), 0)
            first_token_tensor = index_select(hidden_states, 0,
                                              first_token_indices)

        x = self.dense(first_token_tensor)
        x = ACT2FN['tanh'](x)
        x = self.out_proj(x)
        return x


class BertForSequenceClassification(BertBase):

    def __init__(self, config: BERTConfig):
        super().__init__(config)

        self.config = config
        self.is_roberta = config.is_roberta
        self.bert = BertModel(config)
        self.num_labels = config.num_labels

        if not config.is_roberta:
            self.pooler = BertPooler(hidden_size=config.hidden_size,
                                     dtype=config.dtype)
            self.classifier = Linear(config.hidden_size,
                                     config.num_labels,
                                     dtype=config.dtype)
        else:
            self.classifier = RobertaClassificationHead(
                hidden_size=config.hidden_size,
                num_labels=config.num_labels,
                dtype=config.dtype)

    def forward(self,
                input_ids,
                input_lengths,
                token_type_ids=None,
                position_ids=None,
                hidden_states=None,
                max_input_length=None):

        remove_input_padding = default_net().plugin_config.remove_input_padding

        # required as explicit input in remove_input_padding mode
        # see examples/models/core/bert/run_remove_input_padding.py for how to create them from input_ids and input_lengths
        if remove_input_padding:
            assert token_type_ids is not None and \
                   position_ids is not None and \
                   max_input_length is not None, \
                   "token_type_ids, position_ids, max_input_length is required " \
                   "in remove_input_padding mode"

        hidden_states = self.bert.forward(input_ids=input_ids,
                                          input_lengths=input_lengths,
                                          token_type_ids=token_type_ids,
                                          position_ids=position_ids,
                                          hidden_states=hidden_states,
                                          max_input_length=max_input_length)

        if not self.is_roberta:
            pooled_output = self.pooler(
                hidden_states=hidden_states,
                input_lengths=input_lengths,
                remove_input_padding=remove_input_padding)
            logits = self.classifier(pooled_output)
        else:
            logits = self.classifier(hidden_states=hidden_states,
                                     input_lengths=input_lengths,
                                     remove_input_padding=remove_input_padding)

        logits.mark_output('logits', self.config.logits_dtype)
        return logits


RobertaForSequenceClassification = BertForSequenceClassification
