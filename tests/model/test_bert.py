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
import tempfile
import unittest
from collections import OrderedDict
from itertools import product
from typing import Union

import numpy as np
import parameterized

# isort: off
import torch
import tensorrt as trt
# isort: on
import os
import sys

from parameterized import parameterized

import tensorrt_llm
import tensorrt_llm.runtime
from tensorrt_llm import Builder
from tensorrt_llm._utils import str_dtype_to_trt, trt_dtype_to_torch
from tensorrt_llm.network import net_guard
from tensorrt_llm.plugin.plugin import ContextFMHAType
from tensorrt_llm.runtime import TensorInfo

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.util import skip_fp32_accum_pre_ampere, unittest_name_func

from transformers import AutoTokenizer  # isort:skip
from transformers import BertConfig, BertPreTrainedModel, BertForQuestionAnswering, BertForSequenceClassification, BertModel  # isort:skip
from transformers import RobertaConfig, RobertaPreTrainedModel, RobertaForQuestionAnswering, RobertaForSequenceClassification, RobertaModel  # isort:skip


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
        return np.ascontiguousarray(np.split(v, tp_size)[idx])
    elif len(v.shape) == 2:
        return np.ascontiguousarray(np.split(v, tp_size, axis=dim)[idx])
    return None


def load_from_hf_model(tensorrt_llm_model: tensorrt_llm.module.Module,
                       hf_model: Union[BertPreTrainedModel,
                                       RobertaPreTrainedModel],
                       hf_model_config: Union[BertConfig, RobertaConfig],
                       rank=0,
                       tensor_parallel=1,
                       fp16=False):
    qkv_weight = [[None, None, None]
                  for _ in range(hf_model_config.num_hidden_layers)]

    qkv_bias = [[None, None, None]
                for _ in range(hf_model_config.num_hidden_layers)]

    torch_dtype = torch.float16 if fp16 else torch.float32
    for k, v in hf_model.state_dict().items():
        v = v.to(torch_dtype).cpu().numpy()
        if 'embeddings.word_embeddings.weight' in k:
            tensorrt_llm_model.embedding.vocab_embedding.weight.value = v
        elif 'embeddings.position_embeddings.weight' in k:
            tensorrt_llm_model.embedding.position_embedding.weight.value = v
        elif 'embeddings.token_type_embeddings.weight' in k:
            tensorrt_llm_model.embedding.token_embedding.weight.value = v
        elif 'embeddings.LayerNorm.weight' in k:
            tensorrt_llm_model.embedding.embedding_ln.weight.value = v
        elif 'embeddings.LayerNorm.bias' in k:
            tensorrt_llm_model.embedding.embedding_ln.bias.value = v
        else:
            layer_idx = extract_layer_idx(k)
            if layer_idx is None:
                continue
            idx = int(layer_idx)
            if 'attention.output.dense.weight' in k:
                tensorrt_llm_model.layers[
                    idx].attention.dense.weight.value = split(v,
                                                              tensor_parallel,
                                                              rank,
                                                              dim=1)
            elif 'attention.output.dense.bias' in k:
                tensorrt_llm_model.layers[idx].attention.dense.bias.value = v
            elif 'attention.output.LayerNorm.weight' in k:
                tensorrt_llm_model.layers[idx].input_layernorm.weight.value = v
            elif 'attention.output.LayerNorm.bias' in k:
                tensorrt_llm_model.layers[idx].input_layernorm.bias.value = v
            elif 'intermediate.dense.weight' in k:
                tensorrt_llm_model.layers[idx].mlp.fc.weight.value = split(
                    v, tensor_parallel, rank)
            elif 'intermediate.dense.bias' in k:
                tensorrt_llm_model.layers[idx].mlp.fc.bias.value = split(
                    v, tensor_parallel, rank)
            elif 'output.dense.weight' in k:
                tensorrt_llm_model.layers[idx].mlp.proj.weight.value = split(
                    v, tensor_parallel, rank, dim=1)
            elif 'output.dense.bias' in k:
                tensorrt_llm_model.layers[idx].mlp.proj.bias.value = v
            elif 'output.LayerNorm.weight' in k:
                tensorrt_llm_model.layers[idx].post_layernorm.weight.value = v
            elif 'output.LayerNorm.bias' in k:
                tensorrt_llm_model.layers[idx].post_layernorm.bias.value = v
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

    for i in range(hf_model_config.num_hidden_layers):
        tensorrt_llm_model.layers[i].attention.qkv.weight.value = split(
            np.concatenate(qkv_weight[i]), tensor_parallel, rank)
        tensorrt_llm_model.layers[i].attention.qkv.bias.value = split(
            np.concatenate(qkv_bias[i]), tensor_parallel, rank)


def load_from_hf_qa_model(tensorrt_llm_qa_model: tensorrt_llm.module.Module,
                          hf_qa_model: Union[BertPreTrainedModel,
                                             RobertaPreTrainedModel],
                          hf_bert_config: Union[BertConfig, RobertaConfig],
                          rank=0,
                          tensor_parallel=1,
                          fp16=False):
    load_from_hf_model(tensorrt_llm_qa_model.bert, hf_qa_model, hf_bert_config,
                       rank, tensor_parallel, fp16)
    states = hf_qa_model.state_dict()

    torch_dtype = torch.float16 if fp16 else torch.float32

    tensorrt_llm_qa_model.qa_outputs.weight.value = states[
        'qa_outputs.weight'].to(torch_dtype).cpu().numpy()
    tensorrt_llm_qa_model.qa_outputs.bias.value = states['qa_outputs.bias'].to(
        torch_dtype).cpu().numpy()


def load_from_hf_cls_model(tensorrt_llm_cls_model: tensorrt_llm.models.
                           BertForSequenceClassification,
                           hf_qa_model: Union[BertPreTrainedModel,
                                              RobertaPreTrainedModel],
                           hf_bert_config: Union[BertConfig, RobertaConfig],
                           rank=0,
                           tensor_parallel=1,
                           fp16=False):
    load_from_hf_model(tensorrt_llm_cls_model.bert, hf_qa_model, hf_bert_config,
                       rank, tensor_parallel, fp16)
    states = hf_qa_model.state_dict()

    torch_dtype = torch.float16 if fp16 else torch.float32

    if isinstance(hf_qa_model, BertPreTrainedModel):
        tensorrt_llm_cls_model.pooler.dense.weight.value = states[
            'bert.pooler.dense.weight'].to(torch_dtype).cpu().numpy()
        tensorrt_llm_cls_model.pooler.dense.bias.value = states[
            'bert.pooler.dense.bias'].to(torch_dtype).cpu().numpy()

        tensorrt_llm_cls_model.classifier.weight.value = states[
            'classifier.weight'].to(torch_dtype).cpu().numpy()
        tensorrt_llm_cls_model.classifier.bias.value = states[
            'classifier.bias'].to(torch_dtype).cpu().numpy()
    else:
        tensorrt_llm_cls_model.classifier.dense.weight.value = states[
            'classifier.dense.weight'].to(torch_dtype).cpu().numpy()
        tensorrt_llm_cls_model.classifier.dense.bias.value = states[
            'classifier.dense.bias'].to(torch_dtype).cpu().numpy()

        tensorrt_llm_cls_model.classifier.out_proj.weight.value = states[
            'classifier.out_proj.weight'].to(torch_dtype).cpu().numpy()
        tensorrt_llm_cls_model.classifier.out_proj.bias.value = states[
            'classifier.out_proj.bias'].to(torch_dtype).cpu().numpy()


class TestBert(unittest.TestCase):

    def load_test_cases():
        bert_models = [
            'BertModel',
            'BertForQuestionAnswering',
            'BertForSequenceClassification',
        ]
        roberta_models = [
            'RobertaModel',
            'RobertaForQuestionAnswering',
            'RobertaForSequenceClassification',
        ]
        bert_model_dirs = ['']  # add more tests for read data.
        roberta_model_dirs = ['roberta-base']  # add more tests for read data.
        test_cases = []
        test_cases += product(bert_models, [False], [False], [False],
                              [ContextFMHAType.disabled], ['float32'],
                              bert_model_dirs)
        test_cases += product(bert_models, [False], [True], [True], [
            ContextFMHAType.disabled, ContextFMHAType.enabled,
            ContextFMHAType.enabled_with_fp32_acc
        ], ['float16'], bert_model_dirs)
        test_cases += product(roberta_models, [False], [True], [True], [
            ContextFMHAType.disabled, ContextFMHAType.enabled,
            ContextFMHAType.enabled_with_fp32_acc
        ], ['float32'], roberta_model_dirs)

        return test_cases

    @parameterized.expand(load_test_cases, name_func=unittest_name_func)
    def test_bert(self, model, use_refit, use_plugin, fast_building,
                  context_fmha_type, dtype, model_dir):

        skip_fp32_accum_pre_ampere(context_fmha_type)

        tensorrt_llm.logger.set_level('error')
        if 'Roberta' in model:
            model_type = 'Roberta'
        else:
            model_type = 'Bert'
        fp16 = (dtype == 'float16')
        world_size = 1
        rank = 0
        batch_size = 8
        input_len = 128
        vocab_size = 51200
        num_layers = 12
        num_heads = 12
        hidden_act = 'gelu'
        max_position_embeddings = 512
        hidden_size = 768
        bs_range = [1, (batch_size + 1) // 2, batch_size]
        inlen_range = [1, (input_len + 1) // 2, input_len]
        torch_dtype = torch.float16 if fp16 else torch.float32
        trt_dtype = trt.float16 if fp16 else trt.float32
        timing_cache = 'model.cache'

        torch.manual_seed(0)

        builder = Builder()
        with tempfile.TemporaryDirectory() as tmpdirname:
            builder_config = builder.create_builder_config(
                name=model,
                precision='float16' if fp16 else 'float32',
                timing_cache=timing_cache,
                tensor_parallel=world_size,  # TP only
                use_refit=use_refit)
            network = builder.create_network()
            network.plugin_config.to_legacy_setting()
            if use_plugin:
                network.plugin_config.bert_attention_plugin = dtype
            if fast_building:
                network.plugin_config.gemm_plugin = dtype
            network.plugin_config.set_context_fmha(context_fmha_type)
            with net_guard(network):
                # Prepare inputs
                # TODO: could class be better than dict for profiles?
                input_ids = tensorrt_llm.Tensor(name='input_ids',
                                                dtype=trt.int32,
                                                shape=[-1, -1],
                                                dim_range=OrderedDict([
                                                    ('batch_size', [bs_range]),
                                                    ('input_len', [inlen_range])
                                                ]))
                input_lengths = tensorrt_llm.Tensor(name='input_lengths',
                                                    dtype=trt.int32,
                                                    shape=[-1],
                                                    dim_range=OrderedDict([
                                                        ('batch_size',
                                                         [bs_range])
                                                    ]))
                # Initialize model
                if model_dir:
                    bert_config = globals(
                    )[f'{model_type}Config'].from_pretrained(
                        model_dir, torch_dtype=torch_dtype)
                    vocab_size = bert_config.vocab_size
                    hidden_size = bert_config.hidden_size
                    num_layers = bert_config.num_hidden_layers
                    num_heads = bert_config.num_attention_heads
                    hidden_size = bert_config.intermediate_size // 4
                    hidden_act = bert_config.hidden_act
                    max_position_embeddings = bert_config.max_position_embeddings
                else:
                    bert_config = globals()[f'{model_type}Config'](
                        vocab_size=vocab_size,
                        hidden_size=hidden_size,
                        num_hidden_layers=num_layers,
                        num_attention_heads=num_heads,
                        intermediate_size=4 * hidden_size,
                        hidden_act=hidden_act,
                        max_position_embeddings=max_position_embeddings,
                        torch_dtype=torch_dtype,
                    )

                output_name = "hidden_states"
                if model == 'BertModel' or model == 'RobertaModel':
                    if model_dir:
                        hf_bert = globals(
                        )[f'{model_type}Model'].from_pretrained(
                            model_dir).cuda().to(torch_dtype).eval()
                    else:
                        hf_bert = globals()[f'{model_type}Model'](
                            bert_config, add_pooling_layer=False).cuda().to(
                                torch_dtype).eval()
                    tensorrt_llm_bert = tensorrt_llm.models.BertModel(
                        num_layers=num_layers,
                        num_heads=num_heads,
                        hidden_size=hidden_size,
                        vocab_size=vocab_size,
                        hidden_act=hidden_act,
                        max_position_embeddings=max_position_embeddings,
                        type_vocab_size=bert_config.type_vocab_size,
                        pad_token_id=bert_config.pad_token_id,
                        is_roberta=(model_type == 'Roberta'),
                        mapping=tensorrt_llm.Mapping(
                            world_size=world_size,
                            rank=rank,
                            tp_size=world_size),  # TP only
                        dtype=trt_dtype)
                    load_from_hf_model(tensorrt_llm_bert,
                                       hf_bert,
                                       bert_config,
                                       rank=rank,
                                       tensor_parallel=world_size,
                                       fp16=fp16)
                elif model == 'BertForQuestionAnswering' or model == 'RobertaForQuestionAnswering':
                    if model_dir:
                        hf_bert = globals(
                        )[f'{model_type}ForQuestionAnswering'].from_pretrained(
                            model_dir).cuda().to(torch_dtype).eval()
                    else:
                        hf_bert = globals(
                        )[f'{model_type}ForQuestionAnswering'](
                            bert_config).cuda().to(torch_dtype).eval()
                    output_name = "logits"
                    tensorrt_llm_bert = tensorrt_llm.models.BertForQuestionAnswering(
                        num_layers=num_layers,
                        num_heads=num_heads,
                        hidden_size=hidden_size,
                        vocab_size=vocab_size,
                        hidden_act=hidden_act,
                        max_position_embeddings=max_position_embeddings,
                        type_vocab_size=bert_config.type_vocab_size,
                        pad_token_id=bert_config.pad_token_id,
                        is_roberta=(model_type == 'Roberta'),
                        num_labels=
                        2,  # just make it a const here, seems to me not worth as a config
                        mapping=tensorrt_llm.Mapping(
                            world_size=world_size,
                            rank=rank,
                            tp_size=world_size),  # TP only
                        dtype=trt_dtype)
                    load_from_hf_qa_model(tensorrt_llm_bert,
                                          hf_bert,
                                          bert_config,
                                          rank=rank,
                                          tensor_parallel=world_size,
                                          fp16=fp16)
                elif model == 'BertForSequenceClassification' or model == 'RobertaForSequenceClassification':
                    if model_dir:
                        hf_bert = globals(
                        )[f'{model_type}ForSequenceClassification'].from_pretrained(
                            model_dir).cuda().to(torch_dtype).eval()
                    else:
                        hf_bert = globals(
                        )[f'{model_type}ForSequenceClassification'](
                            bert_config).cuda().to(torch_dtype).eval()
                    output_name = "logits"
                    tensorrt_llm_bert = tensorrt_llm.models.BertForSequenceClassification(
                        num_layers=num_layers,
                        num_heads=num_heads,
                        hidden_size=hidden_size,
                        vocab_size=vocab_size,
                        hidden_act=hidden_act,
                        max_position_embeddings=max_position_embeddings,
                        type_vocab_size=bert_config.type_vocab_size,
                        pad_token_id=bert_config.pad_token_id,
                        is_roberta=(model_type == 'Roberta'),
                        num_labels=
                        2,  # just make it a const here, seems to me not worth as a config
                        mapping=tensorrt_llm.Mapping(
                            world_size=world_size,
                            rank=rank,
                            tp_size=world_size),  # TP only
                        dtype=trt_dtype)
                    load_from_hf_cls_model(tensorrt_llm_bert,
                                           hf_bert,
                                           bert_config,
                                           rank=rank,
                                           tensor_parallel=world_size,
                                           fp16=fp16)
                else:
                    assert False, f"Unknown model {model}"
                # Prepare
                network.set_named_parameters(
                    tensorrt_llm_bert.named_parameters())

                # Forward
                output = tensorrt_llm_bert(input_ids=input_ids,
                                           input_lengths=input_lengths)

                # Mark outputs
                output_dtype = trt.float16 if fp16 else trt.float32
                output.mark_output(output_name, output_dtype)

                for k, v in tensorrt_llm_bert.named_network_outputs():
                    network._mark_output(v, k, str_dtype_to_trt(dtype))

            # Build engine
            engine_buffer = builder.build_engine(network, builder_config)
            session = tensorrt_llm.runtime.Session.from_serialized_engine(
                engine_buffer)
            stream = torch.cuda.current_stream().cuda_stream

            # Inference
            # The dtype of input_ids should be queried from the engine,
            # for testing purpose, int32 is fine for now.
            attention_mask = None
            if model_dir:
                hf_tokenizer = AutoTokenizer.from_pretrained(model_dir)
                input_strings = ['Hello world!' for _ in range(batch_size)]
                input_ids_with_padding = hf_tokenizer(input_strings,
                                                      padding='max_length',
                                                      max_length=input_len)
                input_ids_without_padding = hf_tokenizer(input_strings)
                input_ids = torch.tensor(
                    input_ids_with_padding['input_ids']).int().cuda()
                input_lengths = [
                    len(x) for x in input_ids_without_padding['input_ids']
                ]
                input_lengths = torch.tensor(input_lengths,
                                             device=input_ids.device,
                                             dtype=torch.int32)
                attention_mask = torch.tensor(
                    input_ids_with_padding['attention_mask'],
                    device=input_ids.device,
                    dtype=torch.int32)
            else:
                input_ids = torch.randint(bert_config.vocab_size,
                                          (batch_size, input_len)).int().cuda()
                input_lengths = input_len * torch.ones(
                    (batch_size, ), dtype=torch.int32, device='cuda')

            output_info = session.infer_shapes([
                TensorInfo('input_ids', trt.DataType.INT32,
                           (batch_size, input_len)),
                TensorInfo('input_lengths', trt.DataType.INT32, (batch_size, ))
            ])
            session._print_engine_info()

            outputs = {
                t.name: torch.empty(tuple(t.shape),
                                    dtype=trt_dtype_to_torch(t.dtype),
                                    device='cuda')
                for t in output_info
            }
            assert output_name in outputs, f'{output_name} not found in outputs'
            session.run(inputs={
                'input_ids': input_ids,
                'input_lengths': input_lengths
            },
                        outputs=outputs,
                        stream=stream)
            torch.cuda.synchronize()
            res = outputs[output_name]

            with torch.no_grad():
                if model_dir:
                    hf_outputs = hf_bert.forward(input_ids=input_ids,
                                                 attention_mask=attention_mask)
                else:
                    hf_outputs = hf_bert.forward(input_ids)
            torch.cuda.synchronize()

            if model == 'BertModel' or model == 'RobertaModel':
                ref = hf_outputs.last_hidden_state
                if use_plugin and model_dir:
                    # when we use_plugin and have real-data model_dir and input
                    # We do not need to care about the output of padding positions:
                    attention_mask_tmp = attention_mask.unsqueeze(-1)
                    ref = ref * attention_mask_tmp
                    res = res * attention_mask_tmp

                np.testing.assert_allclose(ref.cpu().numpy(),
                                           res.cpu().numpy(),
                                           atol=1e-2,
                                           rtol=1e-2)
            elif model == 'BertForQuestionAnswering' or model == 'RobertaForQuestionAnswering':
                res_start_logits, res_end_logits = torch.split(res, 1, -1)
                res_start_logits = res_start_logits.squeeze()
                res_end_logits = res_end_logits.squeeze()

                ref_start_logits = hf_outputs.start_logits
                ref_end_logits = hf_outputs.end_logits
                if use_plugin and model_dir:
                    # when we use_plugin and have real-data model_dir and input
                    # We do not need to care about the output of padding positions:
                    ref_start_logits = ref_start_logits * attention_mask
                    ref_end_logits = ref_end_logits * attention_mask
                    res_start_logits = res_start_logits * attention_mask
                    res_end_logits = res_end_logits * attention_mask

                np.testing.assert_allclose(ref_start_logits.cpu().numpy(),
                                           res_start_logits.cpu().numpy(),
                                           atol=1.5e-2)
                np.testing.assert_allclose(ref_end_logits.cpu().numpy(),
                                           res_end_logits.cpu().numpy(),
                                           atol=1.5e-2)
            elif model == 'BertForSequenceClassification' or model == 'RobertaForSequenceClassification':
                ref = hf_outputs.logits
                np.testing.assert_allclose(ref.cpu().numpy(),
                                           res.cpu().numpy(),
                                           atol=1e-2,
                                           rtol=1e-2)


if __name__ == '__main__':
    unittest.main()
