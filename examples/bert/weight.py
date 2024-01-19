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
import numpy as np
import torch


def extract_layer_idx(name):
    ss = name.split('.')
    for s in ss:
        if s.isdigit():
            return s
    return None


def split(v, tp_size, idx, dim=0):
    if tp_size == 1:
        return v
    if len(v.shape) == 1:
        return np.ascontiguousarray(np.split(v, tp_size)[idx].copy())
    elif len(v.shape) == 2:
        return np.ascontiguousarray(np.split(v, tp_size, axis=dim)[idx].copy())
    return None


def load_from_hf_bert(tensorrt_llm_bert,
                      hf_bert,
                      hf_bert_config,
                      rank=0,
                      tensor_parallel=1,
                      fp16=False):
    qkv_weight = [[None, None, None]
                  for _ in range(hf_bert_config.num_hidden_layers)]

    qkv_bias = [[None, None, None]
                for _ in range(hf_bert_config.num_hidden_layers)]

    for k, v in hf_bert.state_dict().items():
        torch_dtype = torch.float16 if fp16 else torch.float32
        v = v.to(torch_dtype).cpu().numpy()
        if 'embeddings.word_embeddings.weight' in k:
            tensorrt_llm_bert.embedding.vocab_embedding.weight.value = v
        elif 'embeddings.position_embeddings.weight' in k:
            tensorrt_llm_bert.embedding.position_embedding.weight.value = v
        elif 'embeddings.token_type_embeddings.weight' in k:
            tensorrt_llm_bert.embedding.token_embedding.weight.value = v
        elif 'embeddings.LayerNorm.weight' in k:
            tensorrt_llm_bert.embedding.embedding_ln.weight.value = v
        elif 'embeddings.LayerNorm.bias' in k:
            tensorrt_llm_bert.embedding.embedding_ln.bias.value = v
        else:
            layer_idx = extract_layer_idx(k)
            if layer_idx is None:
                continue
            idx = int(layer_idx)
            if 'attention.output.dense.weight' in k:
                tensorrt_llm_bert.layers[
                    idx].attention.dense.weight.value = split(v,
                                                              tensor_parallel,
                                                              rank,
                                                              dim=1)
            elif 'attention.output.dense.bias' in k:
                tensorrt_llm_bert.layers[idx].attention.dense.bias.value = v
            elif 'attention.output.LayerNorm.weight' in k:
                tensorrt_llm_bert.layers[idx].input_layernorm.weight.value = v
            elif 'attention.output.LayerNorm.bias' in k:
                tensorrt_llm_bert.layers[idx].input_layernorm.bias.value = v
            elif 'intermediate.dense.weight' in k:
                tensorrt_llm_bert.layers[idx].mlp.fc.weight.value = split(
                    v, tensor_parallel, rank)
            elif 'intermediate.dense.bias' in k:
                tensorrt_llm_bert.layers[idx].mlp.fc.bias.value = split(
                    v, tensor_parallel, rank)
            elif 'output.dense.weight' in k:
                tensorrt_llm_bert.layers[idx].mlp.proj.weight.value = split(
                    v, tensor_parallel, rank, dim=1)
            elif 'output.dense.bias' in k:
                tensorrt_llm_bert.layers[idx].mlp.proj.bias.value = v
            elif 'output.LayerNorm.weight' in k:
                tensorrt_llm_bert.layers[idx].post_layernorm.weight.value = v
            elif 'output.LayerNorm.bias' in k:
                tensorrt_llm_bert.layers[idx].post_layernorm.bias.value = v
            elif 'attention.self.query.weight' in k:
                qkv_weight[idx][0] = v
            elif 'attention.self.query.bias' in k:
                qkv_bias[idx][0] = v
            elif 'attention.self.key.weight' in k:
                qkv_weight[idx][1] = v
            elif 'attention.self.key.bias' in k:
                qkv_bias[idx][1] = v
            elif 'attention.self.value.weight' in k:
                qkv_weight[idx][2] = v
            elif 'attention.self.value.bias' in k:
                qkv_bias[idx][2] = v

    for i in range(hf_bert_config.num_hidden_layers):
        tensorrt_llm_bert.layers[i].attention.qkv.weight.value = split(
            np.concatenate(qkv_weight[i]), tensor_parallel, rank)
        tensorrt_llm_bert.layers[i].attention.qkv.bias.value = split(
            np.concatenate(qkv_bias[i]), tensor_parallel, rank)


def load_from_hf_qa_bert(tensorrt_llm_qa_bert,
                         hf_qa_bert,
                         hf_bert_config,
                         rank=0,
                         tensor_parallel=1,
                         fp16=False):
    load_from_hf_bert(tensorrt_llm_qa_bert.bert, hf_qa_bert, hf_bert_config,
                      rank, tensor_parallel, fp16)
    states = hf_qa_bert.state_dict()

    torch_dtype = torch.float16 if fp16 else torch.float32

    tensorrt_llm_qa_bert.qa_outputs.weight.value = states[
        'qa_outputs.weight'].to(torch_dtype).cpu().numpy()
    tensorrt_llm_qa_bert.qa_outputs.bias.value = states['qa_outputs.bias'].to(
        torch_dtype).cpu().numpy()
