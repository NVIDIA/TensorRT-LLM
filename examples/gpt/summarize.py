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
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset, load_metric
from transformers import AutoModelForCausalLM, AutoTokenizer, T5Tokenizer

import tensorrt_llm
import tensorrt_llm.profiler as profiler
from tensorrt_llm.logger import logger
from tensorrt_llm.quantization import QuantMode
from tensorrt_llm.tools.ppl import ppl

from build import find_engines  # isort:skip


def TRTGPT(args, config):
    dtype = config['builder_config']['precision']
    world_size = config['builder_config']['tensor_parallel']
    assert world_size == tensorrt_llm.mpi_world_size(), \
        f'Engine world size ({world_size}) != Runtime world size ({tensorrt_llm.mpi_world_size()})'

    world_size = config['builder_config']['tensor_parallel']
    num_heads = config['builder_config']['num_heads'] // world_size
    hidden_size = config['builder_config']['hidden_size'] // world_size
    vocab_size = config['builder_config']['vocab_size']
    num_layers = config['builder_config']['num_layers']
    use_gpt_attention_plugin = bool(
        config['plugin_config']['gpt_attention_plugin'])
    remove_input_padding = config['plugin_config']['remove_input_padding']
    multi_query_mode = config['builder_config']['multi_query_mode']
    num_kv_heads = 1 if multi_query_mode else num_heads
    paged_kv_cache = config['plugin_config']['paged_kv_cache']
    tokens_per_block = config['plugin_config']['tokens_per_block']
    gather_all_token_logits = config['builder_config'].get(
        'gather_all_token_logits', False)
    assert not (args.eval_ppl and not gather_all_token_logits), \
        "PPL evaluation requires engine built with gather_all_token_logits enabled"

    use_custom_all_reduce = config['plugin_config']['use_custom_all_reduce']
    quant_mode = QuantMode(config['builder_config'].get('quant_mode', 0))

    model_config = tensorrt_llm.runtime.ModelConfig(
        vocab_size=vocab_size,
        num_layers=num_layers,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        hidden_size=hidden_size,
        gpt_attention_plugin=use_gpt_attention_plugin,
        remove_input_padding=remove_input_padding,
        tokens_per_block=tokens_per_block,
        paged_kv_cache=paged_kv_cache,
        dtype=dtype,
        quant_mode=quant_mode,
        gather_all_token_logits=gather_all_token_logits,
        use_custom_all_reduce=use_custom_all_reduce,
    )

    runtime_rank = tensorrt_llm.mpi_rank()
    runtime_mapping = tensorrt_llm.Mapping(world_size,
                                           runtime_rank,
                                           tp_size=world_size)
    torch.cuda.set_device(runtime_rank % runtime_mapping.gpus_per_node)

    serialize_path = find_engines(args.engine_dir,
                                  dtype=dtype,
                                  tp_size=world_size,
                                  rank=runtime_rank)[0]

    tensorrt_llm.logger.set_level(args.log_level)

    with open(serialize_path, 'rb') as f:
        engine_buffer = f.read()
    decoder = tensorrt_llm.runtime.GenerationSession(model_config,
                                                     engine_buffer,
                                                     runtime_mapping)

    return decoder


def main(args):
    runtime_rank = tensorrt_llm.mpi_rank()
    logger.set_level(args.log_level)

    test_hf = args.test_hf and runtime_rank == 0  # only run hf on rank 0
    test_trt_llm = args.test_trt_llm
    hf_model_location = args.hf_model_location

    if args.vocab_file is not None:
        tokenizer = T5Tokenizer(vocab_file=args.vocab_file, padding_side='left')
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer,
                                                  padding_side='left')

    tokenizer.pad_token = tokenizer.eos_token

    if args.eval_type == 'code_completion':
        dataset_name = "openai_humaneval"
        dataset_revision = None
        dataset_input_key = 'prompt'
        dataset_output_key = 'canonical_solution'
    elif args.eval_type == 'summarize':
        dataset_name = "ccdv/cnn_dailymail"
        dataset_revision = "3.0.0"
        dataset_input_key = 'article'
        dataset_output_key = 'highlights'
    dataset = load_dataset(dataset_name,
                           dataset_revision,
                           cache_dir=args.dataset_path)

    config_path = str(args.engine_dir / 'config.json')
    with open(config_path, 'r') as f:
        config = json.load(f)

    max_batch_size = args.batch_size

    # runtime parameters
    # repetition_penalty = 1
    top_k = args.top_k
    output_len = args.output_len
    test_token_num = 923
    # top_p = 0.0
    # random_seed = 5
    temperature = 1
    num_beams = args.num_beams
    length_penalty = args.length_penalty

    pad_id = tokenizer.encode(tokenizer.pad_token, add_special_tokens=False)[0]
    end_id = tokenizer.encode(tokenizer.eos_token, add_special_tokens=False)[0]

    if test_trt_llm:
        tensorrt_llm_gpt = TRTGPT(args, config)

    if test_hf:
        model = AutoModelForCausalLM.from_pretrained(hf_model_location,
                                                     trust_remote_code=True)
        model.cuda()
        if args.data_type == 'fp16':
            model.half()

    def eval_tensorrt_llm(datapoint, eval_type='summarize'):
        batch_size = len(datapoint)
        append_str = ' TL;DR: ' if eval_type == 'summarize' else ''
        line = copy.copy(datapoint)
        line_encoded = []
        input_lengths = []
        for i in range(batch_size):
            line[i] = line[i] + append_str

            line[i] = line[i].strip()
            line[i] = line[i].replace(" n't", "n't")

            input_id = tokenizer.encode(line[i],
                                        return_tensors='pt',
                                        add_special_tokens=False).type(
                                            torch.int32)
            input_id = input_id[:, -test_token_num:]

            line_encoded.append(input_id)
            input_lengths.append(input_id.shape[-1])

        max_length = max(input_lengths)

        if tensorrt_llm_gpt.remove_input_padding:
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
            end_id=end_id,
            pad_id=pad_id,
            top_k=top_k,
            num_beams=num_beams,
            length_penalty=length_penalty)

        with torch.no_grad():
            tensorrt_llm_gpt.setup(batch_size,
                                   max_context_length=max_length,
                                   max_new_tokens=output_len,
                                   beam_width=num_beams)

            if tensorrt_llm_gpt.remove_input_padding:
                outputs = tensorrt_llm_gpt.decode_batch(
                    line_encoded,
                    sampling_config,
                    output_sequence_lengths=True,
                    return_dict=True)
            else:
                outputs = tensorrt_llm_gpt.decode(line_encoded,
                                                  input_lengths,
                                                  sampling_config,
                                                  output_sequence_lengths=True,
                                                  return_dict=True)
            torch.cuda.synchronize()

        # Extract a list of tensors of shape beam_width x output_ids.
        if tensorrt_llm_gpt.mapping.is_first_pp_rank():
            output_ids = outputs['output_ids']
            output_beams_list = [
                tokenizer.batch_decode(output_ids[batch_idx, :,
                                                  input_lengths[batch_idx]:],
                                       skip_special_tokens=True)
                for batch_idx in range(batch_size)
            ]

            ppls = []
            if args.eval_ppl:
                seq_lens = outputs['sequence_lengths']
                context_logits = outputs['context_logits']
                if tensorrt_llm_gpt.remove_input_padding:
                    context_logits = context_logits.flatten(end_dim=1)
                    seg_points = [0] + np.cumsum(input_lengths).tolist()
                    context_logits = [
                        context_logits[s:e]
                        for s, e in zip(seg_points[:-1], seg_points[1:])
                    ]
                else:
                    context_logits = [
                        context_logits[bidx, :input_lengths[bidx]]
                        for bidx in range(batch_size)
                    ]

                # Remove the first generation logits which are same to last context logits
                # Step dim at 1
                generation_logits = torch.stack(
                    outputs['generation_logits'][1:], dim=1)
                for bidx in range(batch_size):
                    # [batch, beam, step]
                    curr_len = seq_lens[bidx, 0]
                    curr_ctx_len = input_lengths[bidx]
                    curr_gen_len = curr_len - curr_ctx_len

                    curr_ids = output_ids[bidx, 0, 1:curr_len]
                    curr_logits = torch.cat([
                        context_logits[bidx],
                        generation_logits[bidx, :curr_gen_len - 1]
                    ],
                                            dim=0)
                    curr_ppl = ppl(curr_logits, curr_ids)
                    ppls.append(curr_ppl)
                    logger.debug(
                        f"TensorRT-LLM PPL: {curr_ppl:.3f} | Generation length: {curr_gen_len}"
                    )

            return output_beams_list, output_ids[:, :,
                                                 max_length:].tolist(), ppls
        return [], [], []

    def eval_hf(datapoint, eval_type='summarize'):
        batch_size = len(datapoint)
        append_str = ' TL;DR: ' if eval_type == 'summarize' else ''
        if batch_size > 1:
            logger.warning(
                f"HF does not support batch_size > 1 to verify correctness due to padding and attention mask. Current batch size is {batch_size}"
            )

        line = copy.copy(datapoint)
        line_encoded = []
        input_lengths = []
        for i in range(batch_size):
            line[i] = line[i] + append_str

            line[i] = line[i].strip()
            line[i] = line[i].replace(" n't", "n't")

            input_id = tokenizer.encode(line[i],
                                        return_tensors='pt',
                                        add_special_tokens=False).type(
                                            torch.int64)
            input_id = input_id[:, -test_token_num:]

            line_encoded.append(input_id)
            input_lengths.append(input_id.shape[-1])

        max_length = max(input_lengths)

        for i in range(batch_size):
            pad_size = max_length - input_lengths[i]

            pad = torch.ones([1, pad_size]).type(torch.int64) * pad_id
            line_encoded[i] = torch.cat(
                [pad, torch.tensor(line_encoded[i], dtype=torch.int64)],
                axis=-1)

        line_encoded = torch.cat(line_encoded, axis=0).cuda()

        with torch.no_grad():
            outputs = model.generate(line_encoded,
                                     max_length=len(line_encoded[0]) +
                                     output_len,
                                     top_k=top_k,
                                     temperature=temperature,
                                     eos_token_id=tokenizer.eos_token_id,
                                     pad_token_id=tokenizer.pad_token_id,
                                     num_beams=num_beams,
                                     num_return_sequences=num_beams,
                                     early_stopping=True,
                                     length_penalty=length_penalty,
                                     output_scores=True,
                                     return_dict_in_generate=True)
            # model.generate cannot return context logits?
            context_outputs = model(line_encoded)

        output_ids = outputs['sequences']
        tokens_list = output_ids[:, len(line_encoded[0]):].tolist()
        output_ids = output_ids.reshape([batch_size, num_beams, -1])
        output_lines_list = [
            tokenizer.batch_decode(output_ids[:, i, len(line_encoded[0]):],
                                   skip_special_tokens=True)
            for i in range(num_beams)
        ]

        ppls = []
        if args.eval_ppl and batch_size == 1:
            # Only for batch size of 1
            seq_lens = [output_ids.size(-1) for _ in range(batch_size)]
            context_logits = context_outputs['logits']
            # Remove the first generation logits which are same to last context logits
            generation_logits = torch.stack(outputs['scores'][1:], dim=1)

            ppls = []
            for bidx in range(batch_size):
                curr_len = seq_lens[bidx]
                curr_ctx_len = input_lengths[bidx]
                curr_gen_len = curr_len - curr_ctx_len

                curr_ids = output_ids[bidx, 0, 1:curr_len]
                curr_logits = torch.cat([
                    context_logits[bidx],
                    generation_logits[bidx, :curr_gen_len - 1]
                ],
                                        dim=0)
                curr_ppl = ppl(curr_logits, curr_ids)
                ppls.append(curr_ppl)
                logger.debug(
                    f"HF PPL: {curr_ppl:.3f} | Generation length: {curr_gen_len}"
                )

        return output_lines_list, tokens_list, ppls

    if test_trt_llm:
        datapoint = dataset['test'][0:1]
        output, *_ = eval_tensorrt_llm(datapoint[dataset_input_key],
                                       eval_type=args.eval_type)
        if runtime_rank == 0:
            logger.info(
                "---------------------------------------------------------")
            logger.info("TensorRT-LLM Generated : ")
            logger.info(f" Input : {datapoint[dataset_input_key]}")
            logger.info(f"\n Reference : {datapoint[dataset_output_key]}")
            logger.info(f"\n Output : {output}")
            logger.info(
                "---------------------------------------------------------")

    if test_hf:
        datapoint = dataset['test'][0:1]
        output, *_ = eval_hf(datapoint[dataset_input_key],
                             eval_type=args.eval_type)
        logger.info("---------------------------------------------------------")
        logger.info("HF Generated : ")
        logger.info(f" Input : {datapoint[dataset_input_key]}")
        logger.info(f"\n Reference : {datapoint[dataset_output_key]}")
        logger.info(f"\n Output : {output}")
        logger.info("---------------------------------------------------------")

    metric_tensorrt_llm = [load_metric("rouge") for _ in range(num_beams)]
    metric_hf = [load_metric("rouge") for _ in range(num_beams)]
    for i in range(num_beams):
        metric_tensorrt_llm[i].seed = 0
        metric_hf[i].seed = 0
    ppls_trt_llm, ppls_hf = [], []

    ite_count = 0
    data_point_idx = 0
    while (data_point_idx < len(dataset['test'])) and (ite_count <
                                                       args.max_ite):
        if runtime_rank == 0:
            logger.debug(
                f"run data_point {data_point_idx} ~ {data_point_idx + max_batch_size}"
            )
        datapoint = dataset['test'][data_point_idx:(data_point_idx +
                                                    max_batch_size)]

        if test_trt_llm:
            profiler.start('tensorrt_llm')
            output_tensorrt_llm, _, curr_ppls_trt_llm = eval_tensorrt_llm(
                datapoint[dataset_input_key])
            profiler.stop('tensorrt_llm')

        if test_hf:
            profiler.start('hf')
            output_hf, _, curr_ppls_hf = eval_hf(datapoint[dataset_input_key])
            profiler.stop('hf')

        if runtime_rank == 0:
            if test_trt_llm:
                for batch_idx in range(len(output_tensorrt_llm)):
                    for beam_idx in range(num_beams):
                        metric_tensorrt_llm[beam_idx].add_batch(
                            predictions=[
                                output_tensorrt_llm[batch_idx][beam_idx]
                            ],
                            references=[
                                datapoint[dataset_output_key][batch_idx]
                            ])
                ppls_trt_llm.extend(curr_ppls_trt_llm)
            if test_hf:
                for beam_idx in range(num_beams):
                    for batch_idx in range(len(output_hf[beam_idx])):
                        metric_hf[beam_idx].add_batch(
                            predictions=[output_hf[beam_idx][batch_idx]],
                            references=[
                                datapoint[dataset_output_key][batch_idx]
                            ])
                ppls_hf.extend(curr_ppls_hf)

            logger.debug('-' * 100)
            logger.debug(f"Input : {datapoint[dataset_input_key]}")
            if test_trt_llm:
                logger.debug(f'TensorRT-LLM Output: {output_tensorrt_llm}')
            if test_hf:
                logger.debug(f'HF Output: {output_hf}')
            logger.debug(f"highlights : {datapoint[dataset_output_key]}")

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
            if args.eval_ppl:
                logger.info(f"  Per-token perplexity: {np.mean(ppls_trt_llm)}")
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
            if args.eval_ppl and args.batch_size == 1:
                logger.info(f"  Per-token perplexity: {np.mean(ppls_hf)}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--hf_model_location', type=str, default='gpt2')
    parser.add_argument(
        '--tokenizer',
        default=None,
        help='tokenizer path; defaults to hf_model_location if left unspecified'
    )
    parser.add_argument('--vocab_file')
    parser.add_argument('--test_hf', action='store_true')
    parser.add_argument('--test_trt_llm', action='store_true')
    parser.add_argument('--data_type',
                        type=str,
                        choices=['fp32', 'fp16'],
                        default='fp32')
    parser.add_argument('--dataset_path', type=str, default='')
    parser.add_argument('--log_level', type=str, default='info')
    parser.add_argument('--engine_dir', type=Path, default='gpt_outputs')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--max_ite', type=int, default=20)
    parser.add_argument('--output_len', type=int, default=100)
    parser.add_argument('--check_accuracy', action='store_true')
    parser.add_argument('--tensorrt_llm_rouge1_threshold',
                        type=float,
                        default=15.0)
    parser.add_argument('--num_beams', type=int, default=1)
    parser.add_argument('--top_k', type=int, default=1)
    parser.add_argument('--eval_type',
                        type=str,
                        default='summarize',
                        choices=['summarize', 'code_completion'])
    parser.add_argument('--length_penalty', type=float, default=1.0)
    parser.add_argument('--eval_ppl', action='store_true')

    args = parser.parse_args()
    if args.tokenizer == None:
        args.tokenizer = args.hf_model_location
    main(args)
