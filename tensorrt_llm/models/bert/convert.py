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
import time
from typing import Union

import torch

# isort: off
from transformers import (AutoModel, AutoModelForQuestionAnswering,
                          AutoModelForSequenceClassification)
from transformers import (BertPreTrainedModel, RobertaPreTrainedModel)
# isort: on
from ...logger import logger
from ..convert_utils import split, split_qkv_bias_tp, split_qkv_tp
from .config import BERTConfig


def extract_layer_idx(name):
    ss = name.split('.')
    for s in ss:
        if s.isdigit():
            return s
    return None


def _load_weights_from_hf_bert_model(hf_model: Union[BertPreTrainedModel,
                                                     RobertaPreTrainedModel],
                                     model_config: BERTConfig,
                                     torch_dtype: torch.dtype = torch.float16):
    weights = {}
    no_match = {}
    mapping = model_config.mapping
    # use different prefix because BertModel is used both individually and as part of model
    trtllm_prefix = "" if (model_config.architecture
                           in ["BertModel", "RobertaModel"]) else "bert."
    for k, v in hf_model.state_dict().items():
        key = None
        v = v.to(torch_dtype).cpu()
        if 'embeddings.word_embeddings.weight' in k:
            key = f'{trtllm_prefix}embedding.vocab_embedding.weight'
        elif 'embeddings.position_embeddings.weight' in k:
            key = f'{trtllm_prefix}embedding.position_embedding.weight'
        elif 'embeddings.token_type_embeddings.weight' in k:
            key = f'{trtllm_prefix}embedding.token_embedding.weight'
        elif 'embeddings.LayerNorm.weight' in k:
            key = f'{trtllm_prefix}embedding.embedding_ln.weight'
        elif 'embeddings.LayerNorm.bias' in k:
            key = f'{trtllm_prefix}embedding.embedding_ln.bias'
        else:
            layer_idx = extract_layer_idx(k)
            if layer_idx is None:
                no_match[k] = v
                continue
            idx = int(layer_idx)
            if 'attention.output.dense.weight' in k:
                #TODO: add TP support
                key = f'{trtllm_prefix}layers.{idx}.attention.dense.weight'
                v_clone = v.clone()
                v = split(v=v_clone,
                          tp_size=mapping.tp_size,
                          idx=mapping.tp_rank,
                          dim=1)
            elif 'attention.output.dense.bias' in k:
                key = f'{trtllm_prefix}layers.{idx}.attention.dense.bias'
            elif 'attention.output.LayerNorm.weight' in k:
                key = f'{trtllm_prefix}layers.{idx}.input_layernorm.weight'
            elif 'attention.output.LayerNorm.bias' in k:
                key = f'{trtllm_prefix}layers.{idx}.input_layernorm.bias'
            elif 'intermediate.dense.weight' in k:
                key = f'{trtllm_prefix}layers.{idx}.mlp.fc.weight'
                v_clone = v.clone()
                v = split(v=v_clone,
                          tp_size=mapping.tp_size,
                          idx=mapping.tp_rank,
                          dim=0)
            elif 'intermediate.dense.bias' in k:
                key = f'{trtllm_prefix}layers.{idx}.mlp.fc.bias'
                v_clone = v.clone()
                v = split(v=v_clone,
                          tp_size=mapping.tp_size,
                          idx=mapping.tp_rank,
                          dim=0)
            elif 'output.dense.weight' in k:
                key = f'{trtllm_prefix}layers.{idx}.mlp.proj.weight'
                v_clone = v.clone()
                v = split(v=v_clone,
                          tp_size=mapping.tp_size,
                          idx=mapping.tp_rank,
                          dim=1)
            elif 'output.dense.bias' in k:
                key = f'{trtllm_prefix}layers.{idx}.mlp.proj.bias'
            elif 'output.LayerNorm.weight' in k:
                key = f'{trtllm_prefix}layers.{idx}.post_layernorm.weight'
            elif 'output.LayerNorm.bias' in k:
                key = f'{trtllm_prefix}layers.{idx}.post_layernorm.bias'
            elif 'attention.self.query.weight' in k:
                key = f'{trtllm_prefix}layers.{idx}.attention.q.weight'
            elif 'attention.self.query.bias' in k:
                key = f'{trtllm_prefix}layers.{idx}.attention.q.bias'
            elif 'attention.self.key.weight' in k:
                key = f'{trtllm_prefix}layers.{idx}.attention.k.weight'
            elif 'attention.self.key.bias' in k:
                key = f'{trtllm_prefix}layers.{idx}.attention.k.bias'
            elif 'attention.self.value.weight' in k:
                key = f'{trtllm_prefix}layers.{idx}.attention.v.weight'
            elif 'attention.self.value.bias' in k:
                key = f'{trtllm_prefix}layers.{idx}.attention.v.bias'
            else:
                no_match[k] = v
                continue
        weights[key] = v

    for idx in range(model_config.num_hidden_layers):
        qkv_key = f'{trtllm_prefix}layers.{idx}.attention.qkv'
        q_key = f'{trtllm_prefix}layers.{idx}.attention.q'
        k_key = f'{trtllm_prefix}layers.{idx}.attention.k'
        v_key = f'{trtllm_prefix}layers.{idx}.attention.v'
        for postfix in ['weight', 'bias']:
            v = torch.cat(
                (weights[f'{q_key}.{postfix}'], weights[f'{k_key}.{postfix}'],
                 weights[f'{v_key}.{postfix}']),
                dim=0)
            v_clone = v.clone()
            split_v = v_clone
            if postfix == 'weight':
                split_v = split_qkv_tp(v_clone,
                                       model_config.num_attention_heads,
                                       model_config.hidden_size,
                                       mapping.tp_size, mapping.tp_rank)

            elif postfix == 'bias':
                split_v = split_qkv_bias_tp(v_clone,
                                            model_config.num_attention_heads,
                                            model_config.hidden_size,
                                            mapping.tp_size, mapping.tp_rank)
            else:
                assert True, f"Unknown postfix={postfix}!"
            #add qkv weight/bias
            weights[f'{qkv_key}.{postfix}'] = split_v
            #remove separate q, k , v
            del weights[f'{q_key}.{postfix}']
            del weights[f'{k_key}.{postfix}']
            del weights[f'{v_key}.{postfix}']
    return (weights, no_match)


def _load_weights_from_hf_bert_qa_model(
        hf_model: Union[BertPreTrainedModel, RobertaPreTrainedModel],
        model_config: BERTConfig,
        torch_dtype: torch.dtype = torch.float16):
    weights, no_match = _load_weights_from_hf_bert_model(
        hf_model, model_config, torch_dtype)

    weights['qa_outputs.weight'] = no_match['qa_outputs.weight']

    weights['qa_outputs.bias'] = no_match['qa_outputs.bias']
    del no_match['qa_outputs.weight']
    del no_match['qa_outputs.bias']

    return (weights, no_match)


def _load_weights_from_hf_bert_cls_model(
        hf_model: Union[BertPreTrainedModel, RobertaPreTrainedModel],
        model_config: BERTConfig,
        torch_dtype: torch.dtype = torch.float16):

    weights, no_match = _load_weights_from_hf_bert_model(
        hf_model, model_config, torch_dtype)

    if model_config.is_roberta:
        # roberta Version
        weights['classifier.dense.weight'] = no_match['classifier.dense.weight']
        weights['classifier.dense.bias'] = no_match['classifier.dense.bias']
        weights['classifier.out_proj.weight'] = no_match[
            'classifier.out_proj.weight']
        weights['classifier.out_proj.bias'] = no_match[
            'classifier.out_proj.bias']
        del no_match['classifier.dense.weight']
        del no_match['classifier.dense.bias']
        del no_match['classifier.out_proj.weight']
        del no_match['classifier.out_proj.bias']
    else:
        weights['pooler.dense.weight'] = no_match['bert.pooler.dense.weight']
        weights['pooler.dense.bias'] = no_match['bert.pooler.dense.bias']
        weights['classifier.weight'] = no_match['classifier.weight']
        weights['classifier.bias'] = no_match['classifier.bias']
        del no_match['bert.pooler.dense.weight']
        del no_match['bert.pooler.dense.bias']
        del no_match['classifier.weight']
        del no_match['classifier.bias']

    return (weights, no_match)


def load_hf_bert_base(model_dir: str,
                      load_model_on_cpu: bool = False,
                      dtype: torch.dtype = torch.float16):
    """
    load huggingface BertModel and RobertaModel model
    """
    model = AutoModel.from_pretrained(
        model_dir,
        trust_remote_code=True,
    )
    if not load_model_on_cpu:
        model.cuda().to(dtype)
    model.eval()
    return model


def load_hf_bert_qa(model_dir: str,
                    load_model_on_cpu: bool = False,
                    dtype: torch.dtype = torch.float16):
    """
    load huggingface BertForQuestionAnswering and RobertaForQuestionAnswering
    """
    model = AutoModelForQuestionAnswering.from_pretrained(
        model_dir,
        trust_remote_code=True,
    )
    if not load_model_on_cpu:
        model.cuda().to(dtype)
    model.eval()
    return model


def load_hf_bert_cls(model_dir: str,
                     load_model_on_cpu: bool = False,
                     dtype: torch.dtype = torch.float16):
    """
    load huggingface BertForSequenceClassification and RobertaForSequenceClassification
    """
    model = AutoModelForSequenceClassification.from_pretrained(
        model_dir,
        trust_remote_code=True,
    )
    if not load_model_on_cpu:
        model.cuda().to(dtype)
    model.eval()
    return model


def load_weights_from_hf_model(
    hf_model,
    config: BERTConfig,
):
    """
    load trtllm weights from hf model

    return a dict of weights, with trtllm weights naming

    """
    #TODO: add quantization support
    weights = {}
    tik = time.time()

    torch_dtype = getattr(torch, config.dtype)

    #NOTE: Bert
    no_match = None
    if config.architecture in [
            "BertForQuestionAnswering", "RobertaForQuestionAnswering"
    ]:
        weights, no_match = _load_weights_from_hf_bert_qa_model(
            hf_model=hf_model, model_config=config, torch_dtype=torch_dtype)
    elif config.architecture in ["BertModel", "RobertaModel"]:
        weights, no_match = _load_weights_from_hf_bert_model(
            hf_model=hf_model, model_config=config, torch_dtype=torch_dtype)
    elif config.architecture in [
            "BertForSequenceClassification", "RobertaForSequenceClassification"
    ]:
        weights, no_match = _load_weights_from_hf_bert_cls_model(
            hf_model=hf_model, model_config=config, torch_dtype=torch_dtype)
    else:
        assert False, f"Unknown BERT model {config.architecture}"

    if no_match is not None:
        logger.warning(
            f"These weights from huggingface model are not used:\n {[key for key in no_match.keys()]}"
        )

    tok = time.time()
    t = time.strftime('%H:%M:%S', time.gmtime(tok - tik))
    print(f'Weights loaded. Total time: {t}')
    return weights


def quantize(hf_model_dir: str,
             output_dir: str,
             config: BERTConfig,
             device: str = 'cuda',
             calib_dataset: str = 'cnn_dailymail'):
    '''
        Quantize the save the model as TRT-LLM checkpoint to output_dir
    '''
    logger.warning(f"FP8 Support for Bert will come soon!")
