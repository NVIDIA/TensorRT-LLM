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
import argparse
import copy
import json
import os

import numpy as np
import torch
from datasets import load_dataset, load_metric
from transformers import AutoModelForCausalLM, LlamaTokenizer

import tensorrt_llm
import tensorrt_llm.profiler as profiler
from tensorrt_llm.logger import logger
from tensorrt_llm.quantization import QuantMode

from build import get_engine_name  # isort:skip


def TRTLLaMA(args, config):
    dtype = config['builder_config']['precision']
    tp_size = config['builder_config']['tensor_parallel']
    pp_size = config['builder_config']['pipeline_parallel']
    world_size = tp_size * pp_size

    assert world_size == tensorrt_llm.mpi_world_size(), \
        f'Engine world size ({world_size}) != Runtime world size ({tensorrt_llm.mpi_world_size()})'

    num_heads = config['builder_config']['num_heads'] // tp_size
    hidden_size = config['builder_config']['hidden_size'] // tp_size
    vocab_size = config['builder_config']['vocab_size']
    num_layers = config['builder_config']['num_layers']
    use_gpt_attention_plugin = bool(
        config['plugin_config']['gpt_attention_plugin'])
    remove_input_padding = config['plugin_config']['remove_input_padding']
    num_kv_heads = config['builder_config'].get('num_kv_heads', num_heads)
    paged_kv_cache = config['plugin_config']['paged_kv_cache']
    tokens_per_block = config['plugin_config']['tokens_per_block']
    use_custom_all_reduce = config['plugin_config'].get('use_custom_all_reduce',
                                                        False)

    quant_mode = QuantMode(config['builder_config']['quant_mode'])
    if config['builder_config'].get('multi_query_mode', False):
        tensorrt_llm.logger.warning(
            "`multi_query_mode` config is deprecated. Please rebuild the engine."
        )
        num_kv_heads = 1
    num_kv_heads = (num_kv_heads + tp_size - 1) // tp_size

    model_config = tensorrt_llm.runtime.ModelConfig(
        vocab_size=vocab_size,
        num_layers=num_layers,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        hidden_size=hidden_size,
        paged_kv_cache=paged_kv_cache,
        tokens_per_block=tokens_per_block,
        gpt_attention_plugin=use_gpt_attention_plugin,
        remove_input_padding=remove_input_padding,
        use_custom_all_reduce=use_custom_all_reduce,
        dtype=dtype,
        quant_mode=quant_mode)

    runtime_rank = tensorrt_llm.mpi_rank()
    runtime_mapping = tensorrt_llm.Mapping(world_size,
                                           runtime_rank,
                                           tp_size=tp_size,
                                           pp_size=pp_size)
    torch.cuda.set_device(runtime_rank % runtime_mapping.gpus_per_node)

    engine_name = get_engine_name('llama', dtype, tp_size, pp_size,
                                  runtime_rank)
    serialize_path = os.path.join(args.engine_dir, engine_name)

    tensorrt_llm.logger.set_level(args.log_level)

    profiler.start('load tensorrt_llm engine')
    with open(serialize_path, 'rb') as f:
        engine_buffer = f.read()
    decoder = tensorrt_llm.runtime.GenerationSession(model_config,
                                                     engine_buffer,
                                                     runtime_mapping)
    profiler.stop('load tensorrt_llm engine')
    tensorrt_llm.logger.info(
        f'Load engine takes: {profiler.elapsed_time_in_sec("load tensorrt_llm engine")} sec'
    )
    return decoder


def main(args):
    runtime_rank = tensorrt_llm.mpi_rank()
    logger.set_level(args.log_level)

    test_hf = args.test_hf and runtime_rank == 0  # only run hf on rank 0
    test_trt_llm = args.test_trt_llm
    hf_model_location = args.hf_model_location
    profiler.start('load tokenizer')
    tokenizer = LlamaTokenizer.from_pretrained(hf_model_location,
                                               legacy=False,
                                               padding_side='left')
    profiler.stop('load tokenizer')
    tensorrt_llm.logger.info(
        f'Load tokenizer takes: {profiler.elapsed_time_in_sec("load tokenizer")} sec'
    )
    tokenizer.pad_token = tokenizer.eos_token

    dataset_cnn = load_dataset("ccdv/cnn_dailymail",
                               '3.0.0',
                               cache_dir=args.dataset_path)

    max_batch_size = args.batch_size

    # runtime parameters
    # repetition_penalty = 1
    top_k = args.top_k
    output_len = 100
    test_token_num = 923
    # top_p = 0.0
    # random_seed = 5
    temperature = 1
    num_beams = args.num_beams

    pad_id = tokenizer.encode(tokenizer.pad_token, add_special_tokens=False)[0]
    end_id = tokenizer.encode(tokenizer.eos_token, add_special_tokens=False)[0]

    if test_trt_llm:
        config_path = os.path.join(args.engine_dir, 'config.json')
        with open(config_path, 'r') as f:
            config = json.load(f)

        tensorrt_llm_llama = TRTLLaMA(args, config)

    if test_hf:
        profiler.start('load HF model')
        model = AutoModelForCausalLM.from_pretrained(hf_model_location)
        profiler.stop('load HF model')
        tensorrt_llm.logger.info(
            f'Load HF model takes: {profiler.elapsed_time_in_sec("load HF model")} sec'
        )
        if args.data_type == 'fp16':
            model.half()
        model.cuda()

    def summarize_tensorrt_llm(datapoint):
        batch_size = len(datapoint['article'])

        line = copy.copy(datapoint['article'])
        line_encoded = []
        input_lengths = []
        for i in range(batch_size):
            line[i] = line[i] + ' TL;DR: '

            line[i] = line[i].strip()
            line[i] = line[i].replace(" n't", "n't")

            input_id = tokenizer.encode(line[i],
                                        return_tensors='pt').type(torch.int32)
            input_id = input_id[:, -test_token_num:]

            line_encoded.append(input_id)
            input_lengths.append(input_id.shape[-1])

        # do padding, should move outside the profiling to prevent the overhead
        max_length = max(input_lengths)
        if tensorrt_llm_llama.remove_input_padding:
            line_encoded = [
                torch.tensor(t, dtype=torch.int32).cuda() for t in line_encoded
            ]
        else:
            # do padding, should move outside the profiling to prevent the overhead
            for i in range(batch_size):
                pad_size = max_length - input_lengths[i]

                pad = torch.ones([1, pad_size]).type(torch.int32) * pad_id
                line_encoded[i] = torch.cat(
                    [torch.tensor(line_encoded[i], dtype=torch.int32), pad],
                    axis=-1)

            line_encoded = torch.cat(line_encoded, axis=0).cuda()
            input_lengths = torch.tensor(input_lengths,
                                         dtype=torch.int32).cuda()

        sampling_config = tensorrt_llm.runtime.SamplingConfig(
            end_id=end_id, pad_id=pad_id, top_k=top_k, num_beams=num_beams)

        with torch.no_grad():
            tensorrt_llm_llama.setup(batch_size,
                                     max_context_length=max_length,
                                     max_new_tokens=output_len,
                                     beam_width=num_beams)

            if tensorrt_llm_llama.remove_input_padding:
                output_ids = tensorrt_llm_llama.decode_batch(
                    line_encoded, sampling_config)
            else:
                output_ids = tensorrt_llm_llama.decode(
                    line_encoded,
                    input_lengths,
                    sampling_config,
                )

            torch.cuda.synchronize()

        # Extract a list of tensors of shape beam_width x output_ids.
        if tensorrt_llm_llama.mapping.is_first_pp_rank():
            output_beams_list = [
                tokenizer.batch_decode(output_ids[batch_idx, :,
                                                  input_lengths[batch_idx]:],
                                       skip_special_tokens=True)
                for batch_idx in range(batch_size)
            ]
            return output_beams_list, output_ids[:, :, max_length:].tolist()
        return [], []

    def summarize_hf(datapoint):
        batch_size = len(datapoint['article'])
        if batch_size > 1:
            logger.warning(
                f"HF does not support batch_size > 1 to verify correctness due to padding. Current batch size is {batch_size}"
            )

        line = copy.copy(datapoint['article'])
        for i in range(batch_size):
            line[i] = line[i] + ' TL;DR: '

            line[i] = line[i].strip()
            line[i] = line[i].replace(" n't", "n't")

        line_encoded = tokenizer(line,
                                 return_tensors='pt',
                                 padding=True,
                                 truncation=True)["input_ids"].type(torch.int64)

        line_encoded = line_encoded[:, -test_token_num:]
        line_encoded = line_encoded.cuda()

        with torch.no_grad():
            output = model.generate(line_encoded,
                                    max_length=len(line_encoded[0]) +
                                    output_len,
                                    top_k=top_k,
                                    temperature=temperature,
                                    eos_token_id=tokenizer.eos_token_id,
                                    pad_token_id=tokenizer.pad_token_id,
                                    num_beams=num_beams,
                                    num_return_sequences=num_beams,
                                    early_stopping=True)

        tokens_list = output[:, len(line_encoded[0]):].tolist()
        output = output.reshape([batch_size, num_beams, -1])
        output_lines_list = [
            tokenizer.batch_decode(output[:, i, len(line_encoded[0]):],
                                   skip_special_tokens=True)
            for i in range(num_beams)
        ]

        return output_lines_list, tokens_list

    if test_trt_llm:
        datapoint = dataset_cnn['test'][0:1]
        summary, _ = summarize_tensorrt_llm(datapoint)
        if runtime_rank == 0:
            logger.info(
                "---------------------------------------------------------")
            logger.info("TensorRT-LLM Generated : ")
            logger.info(f" Article : {datapoint['article']}")
            logger.info(f"\n Highlights : {datapoint['highlights']}")
            logger.info(f"\n Summary : {summary}")
            logger.info(
                "---------------------------------------------------------")

    if test_hf:
        datapoint = dataset_cnn['test'][0:1]
        summary, _ = summarize_hf(datapoint)
        logger.info("---------------------------------------------------------")
        logger.info("HF Generated : ")
        logger.info(f" Article : {datapoint['article']}")
        logger.info(f"\n Highlights : {datapoint['highlights']}")
        logger.info(f"\n Summary : {summary}")
        logger.info("---------------------------------------------------------")

    metric_tensorrt_llm = [load_metric("rouge") for _ in range(num_beams)]
    metric_hf = [load_metric("rouge") for _ in range(num_beams)]
    for i in range(num_beams):
        metric_tensorrt_llm[i].seed = 0
        metric_hf[i].seed = 0

    ite_count = 0
    data_point_idx = 0
    while (data_point_idx < len(dataset_cnn['test'])) and (ite_count <
                                                           args.max_ite):
        if runtime_rank == 0:
            logger.debug(
                f"run data_point {data_point_idx} ~ {data_point_idx + max_batch_size}"
            )
        datapoint = dataset_cnn['test'][data_point_idx:(data_point_idx +
                                                        max_batch_size)]

        if test_trt_llm:
            profiler.start('tensorrt_llm')
            summary_tensorrt_llm, tokens_tensorrt_llm = summarize_tensorrt_llm(
                datapoint)
            profiler.stop('tensorrt_llm')

        if test_hf:
            profiler.start('hf')
            summary_hf, tokens_hf = summarize_hf(datapoint)
            profiler.stop('hf')

        if runtime_rank == 0:
            if test_trt_llm:
                for batch_idx in range(len(summary_tensorrt_llm)):
                    for beam_idx in range(num_beams):
                        metric_tensorrt_llm[beam_idx].add_batch(
                            predictions=[
                                summary_tensorrt_llm[batch_idx][beam_idx]
                            ],
                            references=[datapoint['highlights'][batch_idx]])
            if test_hf:
                for beam_idx in range(num_beams):
                    for batch_idx in range(len(summary_hf[beam_idx])):
                        metric_hf[beam_idx].add_batch(
                            predictions=[summary_hf[beam_idx][batch_idx]],
                            references=[datapoint['highlights'][batch_idx]])

            logger.debug('-' * 100)
            logger.debug(f"Article : {datapoint['article']}")
            if test_trt_llm:
                logger.debug(f'TensorRT-LLM Summary: {summary_tensorrt_llm}')
            if test_hf:
                logger.debug(f'HF Summary: {summary_hf}')
            logger.debug(f"highlights : {datapoint['highlights']}")

        data_point_idx += max_batch_size
        ite_count += 1

    if runtime_rank == 0:
        if test_trt_llm:
            np.random.seed(0)  # rouge score use sampling to compute the score
            logger.info(
                f'TensorRT-LLM (total latency: {profiler.elapsed_time_in_sec("tensorrt_llm")} sec)'
            )
            for beam_idx in range(num_beams):
                logger.info(f"TensorRT-LLM beam {beam_idx} result")
                computed_metrics_tensorrt_llm = metric_tensorrt_llm[
                    beam_idx].compute()
                for key in computed_metrics_tensorrt_llm.keys():
                    logger.info(
                        f'  {key} : {computed_metrics_tensorrt_llm[key].mid[2]*100}'
                    )

                if args.check_accuracy and beam_idx == 0:
                    assert computed_metrics_tensorrt_llm['rouge1'].mid[
                        2] * 100 > args.tensorrt_llm_rouge1_threshold
        if test_hf:
            np.random.seed(0)  # rouge score use sampling to compute the score
            logger.info(
                f'Hugging Face (total latency: {profiler.elapsed_time_in_sec("hf")} sec)'
            )
            for beam_idx in range(num_beams):
                logger.info(f"HF beam {beam_idx} result")
                computed_metrics_hf = metric_hf[beam_idx].compute()
                for key in computed_metrics_hf.keys():
                    logger.info(
                        f'  {key} : {computed_metrics_hf[key].mid[2]*100}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--hf_model_location',
                        type=str,
                        default='/workspace/models/llama-models/llama-7b-hf')
    parser.add_argument('--test_hf', action='store_true')
    parser.add_argument('--test_trt_llm', action='store_true')
    parser.add_argument('--data_type',
                        type=str,
                        choices=['fp32', 'fp16'],
                        default='fp16')
    parser.add_argument('--dataset_path', type=str, default='')
    parser.add_argument('--log_level', type=str, default='info')
    parser.add_argument('--engine_dir', type=str, default='llama_outputs')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--max_ite', type=int, default=20)
    parser.add_argument('--check_accuracy', action='store_true')
    parser.add_argument('--tensorrt_llm_rouge1_threshold',
                        type=float,
                        default=15.0)
    parser.add_argument('--num_beams', type=int, default=1)
    parser.add_argument('--top_k', type=int, default=1)

    args = parser.parse_args()

    main(args)
