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

import torch
from datasets import load_dataset
from evaluate import load
from transformers import AutoModelForCausalLM, LlamaTokenizer

import tensorrt_llm
import tensorrt_llm.profiler as profiler
from tensorrt_llm.bindings import KVCacheType
from tensorrt_llm.logger import logger
from tensorrt_llm.quantization import QuantMode


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--hf_model_location',
                        type=str,
                        default='/tmp/models/Mistral-7B-v0.1')
    parser.add_argument('--test_hf', action='store_true')
    parser.add_argument('--test_trt_llm', action='store_true')
    parser.add_argument('--data_type',
                        type=str,
                        choices=['fp16'],
                        default='fp16')
    parser.add_argument('--dataset_path', type=str, default='/tmp/data')
    parser.add_argument(
        '--max_attention_window_size',
        type=int,
        default=4096,
        help=
        'The attention window size that controls the sliding window attention kv cache behavior'
    )
    parser.add_argument(
        '--max_input_len',
        type=int,
        default=6400,
        help='The max input length TensorRT LLM engine was built with')
    parser.add_argument('--log_level', type=str, default='info')
    parser.add_argument('--max_ite', type=int, default=5)
    parser.add_argument(
        '--engine_dir',
        type=str,
        default='/code/tensorrt_llm/mistral_trtllm/llama_style_merge_long_v2')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_beams', type=int, default=1)
    parser.add_argument('--top_k', type=int, default=1)
    parser.add_argument('--output_len', type=int, default=128)
    parser.add_argument('--temperature', type=float, default=1)
    parser.add_argument('--check_accuracy', action='store_true')
    parser.add_argument('--tensorrt_llm_rouge1_threshold',
                        type=float,
                        default=15.0)
    parser.add_argument(
        '--rouge_dir',
        default=None,
        type=str,
        help=
        "datasets.load_metrics('rouge') will attempt to pull rouge package from HF. Use cached rouge can avoid network outage of host or HF."
    )
    parser.add_argument(
        '--multi_block_mode',
        type=lambda s: s.lower() in
        ("yes", "true", "t", "1"
         ),  # custom boolean function to convert input string to boolean
        default=True,
        help=
        "Distribute the work across multiple CUDA thread-blocks on the GPU for masked MHA kernel."
    )
    parser.add_argument('--enable_context_fmha_fp32_acc',
                        action='store_true',
                        help="Enable FMHA runner FP32 accumulation.")

    args = parser.parse_args()
    return args


def TRTLLaMA(args, config):

    pretrained_config = config['pretrained_config']
    quantization_config = pretrained_config['quantization']

    build_config = config['build_config']
    kv_cache_type = KVCacheType.from_string(build_config['kv_cache_type'])
    plugin_config = build_config['plugin_config']

    dtype = pretrained_config['dtype']
    tp_size = pretrained_config['mapping']['tp_size']
    pp_size = pretrained_config['mapping']['pp_size']
    world_size = tp_size * pp_size

    assert world_size == tensorrt_llm.mpi_world_size(), \
        f'Engine world size ({world_size}) != Runtime world size ({tensorrt_llm.mpi_world_size()})'

    num_heads = pretrained_config['num_attention_heads'] // tp_size
    hidden_size = pretrained_config['hidden_size'] // tp_size

    max_batch_size = build_config['max_batch_size']
    vocab_size = pretrained_config['vocab_size']
    num_layers = pretrained_config['num_hidden_layers']
    use_gpt_attention_plugin = bool(plugin_config['gpt_attention_plugin'])
    remove_input_padding = plugin_config['remove_input_padding']
    num_kv_heads = pretrained_config['num_key_value_heads']
    tokens_per_block = plugin_config['tokens_per_block']

    quant_mode = QuantMode.from_quant_algo(
        quant_algo=quantization_config['quant_algo'],
        kv_cache_quant_algo=quantization_config['kv_cache_quant_algo'])
    if pretrained_config.get('multi_query_mode', False):
        tensorrt_llm.logger.warning(
            "`multi_query_mode` config is deprecated. Please rebuild the engine."
        )
        num_kv_heads = 1
    num_kv_heads = (num_kv_heads + tp_size - 1) // tp_size

    model_config = tensorrt_llm.runtime.ModelConfig(
        max_batch_size=max_batch_size,
        max_beam_width=args.num_beams,
        vocab_size=vocab_size,
        num_layers=num_layers,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        hidden_size=hidden_size,
        kv_cache_type=kv_cache_type,
        tokens_per_block=tokens_per_block,
        gpt_attention_plugin=use_gpt_attention_plugin,
        remove_input_padding=remove_input_padding,
        dtype=dtype,
        quant_mode=quant_mode)

    runtime_rank = tensorrt_llm.mpi_rank()
    runtime_mapping = tensorrt_llm.Mapping(world_size,
                                           runtime_rank,
                                           tp_size=tp_size,
                                           pp_size=pp_size)
    torch.cuda.set_device(runtime_rank % runtime_mapping.gpus_per_node)

    engine_name = f'rank{runtime_rank}.engine'
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


def get_long_texts(dataset_openweb):
    for datapoint in dataset_openweb["train"]:
        text = datapoint["text"]
        approximate_tokens = len(text.split())
        if (approximate_tokens
                > args.max_attention_window_size) and (approximate_tokens
                                                       < args.max_input_len):
            yield text


def prepare_prompt(text):
    text = text.replace("\n", " ")
    text = text + '\n TL;DR: '
    text = text.strip()
    text = text.replace(" n't", "n't")
    return text


def summarize_hf(datapoint, tokenizer, hf_model, args):

    line_encoded = tokenizer(datapoint,
                             return_tensors='pt',
                             padding=True,
                             truncation=True)["input_ids"].type(torch.int32)

    line_encoded = line_encoded.cuda()

    with torch.no_grad():
        output = hf_model.generate(line_encoded,
                                   max_new_tokens=args.output_len,
                                   temperature=args.temperature,
                                   eos_token_id=tokenizer.eos_token_id,
                                   pad_token_id=tokenizer.pad_token_id,
                                   num_beams=args.num_beams,
                                   top_k=args.top_k,
                                   do_sample=True,
                                   early_stopping=True)

    tokens_list = output[:, len(line_encoded[0]):].tolist()
    output = output.reshape([args.batch_size, args.num_beams, -1])
    output_lines_list = [
        tokenizer.batch_decode(output[:, i, len(line_encoded[0]):],
                               skip_special_tokens=True)
        for i in range(args.num_beams)
    ]

    return output_lines_list, tokens_list


def summarize_tensorrt_llm(datapoint, tokenizer, tensorrt_llm_llama, args):
    line_encoded = []
    input_id = tokenizer.encode(datapoint,
                                return_tensors='pt').type(torch.int32)
    line_encoded.append(input_id)
    input_lengths = []
    input_lengths.append(input_id.shape[-1])
    max_length = max(input_lengths)

    pad_id = tokenizer.encode(tokenizer.pad_token, add_special_tokens=False)[0]
    end_id = tokenizer.encode(tokenizer.eos_token, add_special_tokens=False)[0]

    if tensorrt_llm_llama.remove_input_padding:
        line_encoded = [
            torch.tensor(t, dtype=torch.int32).cuda() for t in line_encoded
        ]
    else:
        # do padding, should move outside the profiling to prevent the overhead
        for i in range(args.batch_size):
            pad_size = max_length - input_lengths[i]

            pad = torch.ones([1, pad_size]).type(torch.int32) * pad_id
            line_encoded[i] = torch.cat(
                [torch.tensor(line_encoded[i], dtype=torch.int32), pad],
                axis=-1)

            line_encoded = torch.cat(line_encoded, axis=0).cuda()

    input_lengths = torch.tensor(input_lengths, dtype=torch.int32).cuda()

    sampling_config = tensorrt_llm.runtime.SamplingConfig(
        end_id=end_id,
        pad_id=pad_id,
        top_k=args.top_k,
        num_beams=args.num_beams)

    with torch.no_grad():
        tensorrt_llm_llama.setup(
            batch_size=args.batch_size,
            max_context_length=max_length,
            max_new_tokens=args.output_len,
            beam_width=args.num_beams,
            max_attention_window_size=args.max_attention_window_size,
            multi_block_mode=args.multi_block_mode,
            enable_context_fmha_fp32_acc=args.enable_context_fmha_fp32_acc)
        logger.info(f"Generation session set up with the parameters: \
            batch_size: {tensorrt_llm_llama.batch_size}, \
            max_context_length: {tensorrt_llm_llama.max_context_length}, \
            max_new_tokens: {tensorrt_llm_llama.max_new_tokens}, \
            beam_width: {tensorrt_llm_llama.beam_width}, \
            max_attention_window_size: {tensorrt_llm_llama.max_attention_window_size}, \
            multi_block_mode: {tensorrt_llm_llama.multi_block_mode}, \
            enable_context_fmha_fp32_acc: {tensorrt_llm_llama.enable_context_fmha_fp32_acc}"
                    )

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

    logger.info(f"Decoded output of shape{output_ids.shape}")

    # Extract a list of tensors of shape beam_width x output_ids.
    if tensorrt_llm_llama.mapping.is_first_pp_rank():
        output_beams_list = [
            tokenizer.batch_decode(output_ids[batch_idx, :,
                                              input_lengths[batch_idx]:],
                                   skip_special_tokens=True)
            for batch_idx in range(args.batch_size)
        ]
        return output_beams_list, output_ids[:, :, max_length:].tolist()
    return [], []


def main(args):
    runtime_rank = tensorrt_llm.mpi_rank()
    logger.set_level(args.log_level)

    profiler.start('load tokenizer')
    tokenizer = LlamaTokenizer.from_pretrained(args.hf_model_location,
                                               legacy=False,
                                               padding_side='left')
    profiler.stop('load tokenizer')
    tensorrt_llm.logger.info(
        f'Load tokenizer takes: {profiler.elapsed_time_in_sec("load tokenizer")} sec'
    )
    tokenizer.pad_token = tokenizer.eos_token

    dataset_openweb = load_dataset("stas/openwebtext-10k",
                                   cache_dir=args.dataset_path,
                                   trust_remote_code=True)
    long_texts = get_long_texts(dataset_openweb)  # generator

    # get datapoints
    try:
        datapoints = [
            prepare_prompt(next(long_texts)) for i in range(args.max_ite)
        ]
    except StopIteration:
        logger.warning(
            f"No test data of sufficient length ({args.max_attention_window_size}). Try decreasing the max_attention_window_size parameter"
        )
        return

    if args.test_trt_llm:
        config_path = os.path.join(args.engine_dir, 'config.json')
        with open(config_path, 'r') as f:
            config = json.load(f)
        tensorrt_llm_llama = TRTLLaMA(args, config)

        trt_llm_summary = []
        for ite in range(args.max_ite):
            trt_llm_summary.append(
                summarize_tensorrt_llm(datapoints[ite], tokenizer,
                                       tensorrt_llm_llama, args)[0])

        if runtime_rank == 0:
            logger.info(
                "---------------------------------------------------------")
            logger.info("TRT LLM Generated : ")
            logger.info(f" Article : {datapoints[0]}")
            logger.info(f"\n Summary : {trt_llm_summary[0]}")
            logger.info(
                "---------------------------------------------------------")

        del tensorrt_llm_llama

    test_hf = args.test_hf and runtime_rank == 0  # only run hf on rank 0
    if test_hf:
        profiler.start('load HF model')
        hf_model = AutoModelForCausalLM.from_pretrained(
            args.hf_model_location,
            dtype=torch.float16,
            use_flash_attention_2=True)
        profiler.stop('load HF model')
        tensorrt_llm.logger.info(
            f'Load HF model takes: {profiler.elapsed_time_in_sec("load HF model")} sec'
        )
        hf_model.cuda()

        hf_summary = []
        for ite in range(args.max_ite):
            hf_summary.append(
                summarize_hf(datapoints[ite], tokenizer, hf_model, args)[0])
        logger.info("---------------------------------------------------------")
        logger.info("HF Generated : ")
        logger.info(f" Article : {datapoints[0]}")
        logger.info(f"\n Summary : {hf_summary[0]}")
        logger.info("---------------------------------------------------------")

    # no ground truth, compare with hf
    if runtime_rank == 0 and args.test_hf and args.test_trt_llm:

        rouge_dir = args.rouge_dir if args.rouge_dir and os.path.exists(
            args.rouge_dir) else "rouge"
        metric_tensorrt_llm = [load(rouge_dir) for _ in range(args.num_beams)]

        for i in range(args.num_beams):
            metric_tensorrt_llm[i].seed = 0

        for ite in range(args.max_ite):
            for batch_idx in range(len(trt_llm_summary[0])):
                for beam_idx in range(args.num_beams):
                    metric_tensorrt_llm[beam_idx].add_batch(
                        predictions=[trt_llm_summary[ite][batch_idx][beam_idx]],
                        references=[hf_summary[ite][beam_idx][batch_idx]])

        for beam_idx in range(args.num_beams):
            logger.info(f"TensorRT LLM beam {beam_idx} result")
            computed_metrics_tensorrt_llm = metric_tensorrt_llm[
                beam_idx].compute()
            for key in computed_metrics_tensorrt_llm.keys():
                logger.info(
                    f'  {key} : {computed_metrics_tensorrt_llm[key]*100}')

            if args.check_accuracy and beam_idx == 0:
                assert computed_metrics_tensorrt_llm[
                    'rouge1'] * 100 > args.tensorrt_llm_rouge1_threshold


if __name__ == '__main__':
    args = parse_args()
    main(args)
