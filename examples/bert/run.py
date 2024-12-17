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
import argparse
import json
import os

# isort: off
import torch
import tensorrt as trt
# isort: on

from transformers import AutoConfig, AutoTokenizer
from utils import (compare_bertcls_result, compare_bertqa_result,
                   decode_bertcls_output, decode_bertqa_output, get_engine_name,
                   intermediate_check, prepare_text_inputs, process_input,
                   temporary_datasets_config)

import tensorrt_llm
from tensorrt_llm import logger
from tensorrt_llm._utils import trt_dtype_to_torch
from tensorrt_llm.runtime import Session, TensorInfo

from transformers import BertConfig, BertPreTrainedModel, BertForQuestionAnswering, BertForSequenceClassification, BertModel  # isort:skip
from transformers import RobertaConfig, RobertaPreTrainedModel, RobertaForQuestionAnswering, RobertaForSequenceClassification, RobertaModel  # isort:skip

OUTPUT_NAME_MAPPING = {
    'BertModel': 'hidden_states',
    'BertForQuestionAnswering': 'logits',
    'BertForSequenceClassification': 'logits',
    'RobertaModel': 'hidden_states',
    'RobertaForQuestionAnswering': 'logits',
    'RobertaForSequenceClassification': 'logits'
}


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_level', type=str, default='info')
    parser.add_argument('--engine_dir', type=str, default='bert_outputs')
    parser.add_argument('--hf_model_dir', type=str, required=True)
    parser.add_argument('--run_hf_test', action='store_true')
    parser.add_argument('--remove_input_padding', action='store_true')
    parser.add_argument('--debug', action='store_true')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()

    tensorrt_llm.logger.set_level(args.log_level)

    config_path = os.path.join(args.engine_dir, 'config.json')
    with open(config_path, 'r') as f:
        config = json.load(f)

    remove_padding = config['build_config']['plugin_config'][
        'remove_input_padding']
    assert args.remove_input_padding == remove_padding, \
        f"The engine is build with remove_input_padding={remove_padding}, \
        but the inference runtime is performed with remove_input_padding={args.remove_input_padding}!"

    world_size = config['pretrained_config']['mapping']['world_size']
    assert world_size == tensorrt_llm.mpi_world_size(), \
        f'Engine world size ({world_size}) != Runtime world size ({tensorrt_llm.mpi_world_size()})'

    model_name = config['pretrained_config']['architecture']

    # Roberta doesn't have token_type_ids, use all zeros to replace
    is_roberta = "Roberta" in model_name

    runtime_rank = tensorrt_llm.mpi_rank() if world_size > 1 else 0

    runtime_mapping = tensorrt_llm.Mapping(world_size,
                                           runtime_rank,
                                           tp_size=world_size)
    torch.cuda.set_device(runtime_rank % runtime_mapping.gpus_per_node)

    serialize_path = get_engine_name(runtime_rank)
    serialize_path = os.path.join(args.engine_dir, serialize_path)

    stream = torch.cuda.current_stream().cuda_stream
    logger.info(f'Loading engine from {serialize_path}')
    with open(serialize_path, 'rb') as f:
        engine_buffer = f.read()
    logger.info(f'Creating session from engine')
    session = Session.from_serialized_engine(engine_buffer)
    if args.debug: session._print_engine_info()

    #NOTE: prepare input
    with temporary_datasets_config(HF_DATASETS_OFFLINE=False):
        test_inputs = prepare_text_inputs(model_name)
    hf_tokenizer = AutoTokenizer.from_pretrained(args.hf_model_dir)
    if args.remove_input_padding:
        #NOTE:Remove padding
        inputs_without_padding = hf_tokenizer(**test_inputs)
        input_ids_list = [
            torch.tensor(ids).int().cuda() \
            for ids in inputs_without_padding['input_ids']
            ]
        # attention_mask_list = inputs_without_padding['attention_mask'],
        if is_roberta:
            token_type_ids_list = [
                torch.zeros_like(torch.tensor(ids)).int().cuda() \
                for ids in inputs_without_padding['input_ids']
            ]
        else:
            token_type_ids_list = [
                torch.tensor(ids).int().cuda() \
                for ids in inputs_without_padding['token_type_ids']
                ]

        input_ids, input_lengths, token_type_ids, position_ids, max_input_length = \
            process_input(input_ids_list=input_ids_list,
                          token_type_ids_list=token_type_ids_list,
                          is_roberta=is_roberta,
                          padding_idx=config['pretrained_config']['pad_token_id'])

    else:

        #NOTE:Padding: pad to longest seq len
        inputs_with_padding = hf_tokenizer(
            **test_inputs,
            padding=True,
        )

        inputs_without_padding = hf_tokenizer(**test_inputs)
        input_ids = torch.tensor(inputs_with_padding['input_ids']).int().cuda()
        input_lengths = [len(x) for x in inputs_without_padding['input_ids']]
        input_lengths = torch.tensor(input_lengths,
                                     device=input_ids.device,
                                     dtype=torch.int32)
        attention_mask = torch.tensor(inputs_with_padding['attention_mask'],
                                      device=input_ids.device,
                                      dtype=torch.int32)
        if is_roberta:
            token_type_ids = torch.zeros_like(torch.tensor(
                inputs_with_padding['input_ids']),
                                              device=input_ids.device,
                                              dtype=torch.int32)
        else:
            token_type_ids = torch.tensor(inputs_with_padding['token_type_ids'],
                                          device=input_ids.device,
                                          dtype=torch.int32)

    # NOTE: TRT-LLM perform inference
    output_name = OUTPUT_NAME_MAPPING[model_name]
    if args.remove_input_padding:
        # NOTE: Remove padding:
        inputs = {
            "input_ids": input_ids,
            "input_lengths": input_lengths,
            "token_type_ids": token_type_ids,
            "position_ids": position_ids,
            "max_input_length": max_input_length
        }
        output_info = session.infer_shapes([
            TensorInfo("input_ids", trt.DataType.INT32, input_ids.shape),
            TensorInfo("input_lengths", trt.DataType.INT32,
                       input_lengths.shape),
            TensorInfo("token_type_ids", trt.DataType.INT32,
                       token_type_ids.shape),
            TensorInfo("position_ids", trt.DataType.INT32, position_ids.shape),
            TensorInfo("max_input_length", trt.DataType.INT32,
                       max_input_length.shape)
        ])
    else:
        #NOTE: Padding:
        inputs = {
            'input_ids': input_ids,
            'input_lengths': input_lengths,
            'token_type_ids': token_type_ids,
        }
        output_info = session.infer_shapes([
            TensorInfo('input_ids', trt.DataType.INT32, input_ids.shape),
            TensorInfo('input_lengths', trt.DataType.INT32,
                       input_lengths.shape),
            TensorInfo('token_type_ids', trt.DataType.INT32,
                       token_type_ids.shape)
        ])

    outputs = {
        t.name:
        torch.empty(tuple(t.shape),
                    dtype=trt_dtype_to_torch(t.dtype),
                    device='cuda')
        for t in output_info
    }
    assert output_name in outputs, f'{output_name} not found in outputs, check if build.py set output name correctly'

    logger.info(f"Rank{runtime_rank} is running inference...")
    ok = session.run(inputs=inputs, outputs=outputs, stream=stream)
    assert ok, "Runtime execution failed"
    torch.cuda.synchronize()
    res = outputs[output_name]
    if args.debug: logger.info(f"Outputs:{outputs.keys()}")

    # NOTE: load hf model and perform inference as reference (only on rank0)
    if tensorrt_llm.mpi_rank() == 0:
        logger.info(f"Rank{runtime_rank} is generating HF reference...")
        if args.run_hf_test:
            hf_bert = globals()[f'{model_name}'].from_pretrained(
                args.hf_model_dir).cuda().to(torch.float16).eval()
            hf_inputs = hf_tokenizer(**test_inputs,
                                     padding=True,
                                     return_tensors="pt")
            hf_inputs = hf_inputs.to(hf_bert.device)
            with torch.no_grad():
                hf_outputs = hf_bert.forward(output_hidden_states=args.debug,
                                             **hf_inputs)

            torch.cuda.synchronize()
    # NOTE: Decode output (only on rank0)

    if tensorrt_llm.mpi_rank() == 0:
        logger.info(f"Rank{runtime_rank} is comparing with HF reference...")
        if model_name == "BertModel" or model_name == "RobertaModel":
            if args.remove_input_padding:
                # reshape result back to [batch_size, ...],
                # and then "padding" so we could compare the tensor
                from torch.nn.utils.rnn import pad_sequence
                res = torch.split(res, input_lengths.tolist(), dim=0)
                res = pad_sequence(list(res), batch_first=True, padding_value=0)
            else:
                # applied attention mask on trtllm res
                attention_mask_tmp = attention_mask.unsqueeze(-1)
                res = res * attention_mask_tmp
            if args.run_hf_test:
                ref = hf_outputs.last_hidden_state
                ref = ref * hf_inputs['attention_mask'].unsqueeze(-1)

                if args.debug:
                    intermediate_check(outputs, hf_outputs['hidden_states'],
                                       attention_mask_tmp, logger)

                if world_size == 1:
                    torch.testing.assert_close(actual=res.half(),
                                               expected=ref,
                                               rtol=1.5e-2,
                                               atol=1.5e-2)
                else:
                    # the arithmetic order of TP>1 is different from HF ref, which is always TP=1 for convenience.
                    torch.testing.assert_close(actual=res.half(),
                                               expected=ref,
                                               rtol=4e-2,
                                               atol=2e-2)
                print(f"{model_name} result is all close to HF reference!")

        if model_name == 'BertForQuestionAnswering' or model_name == 'RobertaForQuestionAnswering':
            if args.remove_input_padding:

                # [num_tokens, 2] -> [num_tokens, 1]
                res_start_logits, res_end_logits = torch.split(res, 1, -1)

                # reshape result back to [batch_size, ...]
                res_start_logits = torch.split(res_start_logits,
                                               input_lengths.tolist(),
                                               dim=0)
                res_start_logits = tuple(t.squeeze() for t in res_start_logits)
                res_end_logits = torch.split(res_end_logits,
                                             input_lengths.tolist(),
                                             dim=0)
                res_end_logits = tuple(t.squeeze() for t in res_end_logits)

            else:
                #NOTE: Padding
                # [B, Padding_len, 2] -> [B, Padding_len, 1]
                res_start_logits, res_end_logits = torch.split(res, 1, -1)
                # [B, Padding_len, 1] -> [B, Padding_len]
                res_start_logits = res_start_logits.squeeze()
                res_end_logits = res_end_logits.squeeze()
                res_start_logits = res_start_logits * attention_mask
                res_end_logits = res_end_logits * attention_mask

                res_start_logits = torch.split(res_start_logits, 1, dim=0)
                res_start_logits = tuple(t.squeeze(0) for t in res_start_logits)
                res_end_logits = torch.split(res_end_logits, 1, dim=0)
                res_end_logits = tuple(t.squeeze(0) for t in res_end_logits)


            decode_res = decode_bertqa_output( inputs_text=test_inputs, \
                                hf_tokenizer=hf_tokenizer, start_logits=res_start_logits, \
                                end_logits=res_end_logits)

            if args.run_hf_test:
                ref_start_logits = hf_outputs.start_logits
                ref_end_logits = hf_outputs.end_logits
                # when we use_plugin and have real-data model_dir and input
                # We do not need to care about the output of padding positions:
                ref_start_logits = ref_start_logits * hf_inputs['attention_mask']
                ref_end_logits = ref_end_logits * hf_inputs['attention_mask']

                decode_ref = decode_bertqa_output( inputs_text=test_inputs, \
                                    hf_tokenizer=hf_tokenizer, start_logits=ref_start_logits, \
                                    end_logits=ref_end_logits)
                compare_bertqa_result(inputs_text=test_inputs,
                                      res_answers=decode_res,
                                      ref_answers=decode_ref)

        elif model_name == 'BertForSequenceClassification' or model_name == 'RobertaForSequenceClassification':
            hf_config = AutoConfig.from_pretrained(args.hf_model_dir)
            decode_res = decode_bertcls_output(logits=res,
                                               hf_model_config=hf_config,
                                               inputs_text=test_inputs)

            if args.run_hf_test:
                ref = hf_outputs.logits
                if world_size == 1:
                    torch.testing.assert_close(actual=res.half(),
                                               expected=ref,
                                               rtol=1.5e-2,
                                               atol=1.5e-2)
                else:
                    # the arithmetic order of TP>1 is different from HF ref, which is always TP=1 for convenience.
                    torch.testing.assert_close(actual=res.half(),
                                               expected=ref,
                                               rtol=4e-2,
                                               atol=2e-2)
                decode_ref = decode_bertcls_output(logits=hf_outputs.logits,
                                                   hf_model_config=hf_config,
                                                   inputs_text=test_inputs)
                compare_bertcls_result(inputs_text=test_inputs,
                                       res_answers=decode_res,
                                       ref_answers=decode_res)
