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
import json
from pathlib import Path

import evaluate
import numpy as np
import torch
from datasets import load_dataset
from qwen.utils.utils import make_context
from transformers import (AutoModel, AutoModelForCausalLM,
                          AutoModelForSeq2SeqLM, AutoTokenizer,
                          GenerationConfig, T5Tokenizer)

import tensorrt_llm
import tensorrt_llm.profiler as profiler
from tensorrt_llm.logger import logger
from tensorrt_llm.runtime import ModelRunner
from tensorrt_llm.tools.ppl import ppl

DEFAULT_HF_MODEL_DIRS = {
    'baichuan': 'baichuan-inc/Baichuan-13B-Chat',
    'bloom': 'bigscience/bloom-560m',
    'chatglm_6b': 'THUDM/chatglm-6b',
    'chatglm2_6b': 'THUDM/chatglm2-6b',
    'chatglm2_6b_32k': 'THUDM/chatglm2-6b-32k',
    'chatglm3_6b': 'THUDM/chatglm3-6b',
    'chatglm3_6b_base': 'THUDM/chatglm3-6b-base',
    'chatglm3_6b_32k': 'THUDM/chatglm3-6b-32k',
    'falcon': 'tiiuae/falcon-rw-1b',
    'glm_10b': 'THUDM/glm-10b',
    'gpt': 'gpt2-medium',
    'gptj': 'EleutherAI/gpt-j-6b',
    'gptneox': 'EleutherAI/gpt-neox-20b',
    'internlm': 'internlm/internlm-chat-7b',
    'llama': 'meta-llama/Llama-2-7b-hf',
    'opt': 'facebook/opt-350m',
    'qwen': 'Qwen/Qwen-7B',
}

DTYPE_STR_MAPPING = {
    'fp32': torch.float32,
    'fp16': torch.float16,
    'bf16': torch.bfloat16,
    'float32': torch.float32,
    'float16': torch.float16,
    'bfloat16': torch.bfloat16,
}


def read_model_name_from_config(config_path: Path):
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config['builder_config']['name']


def main(args):
    runtime_rank = tensorrt_llm.mpi_rank()
    logger.set_level(args.log_level)

    model_name = read_model_name_from_config(
        Path(args.engine_dir) / "config.json")
    if args.hf_model_dir is None:
        args.hf_model_dir = DEFAULT_HF_MODEL_DIRS[model_name]
    if args.tokenizer_dir is None:
        args.tokenizer_dir = args.hf_model_dir

    test_hf = args.test_hf and runtime_rank == 0  # only run hf on rank 0
    test_trt_llm = args.test_trt_llm
    profiler.start('load tokenizer')
    if args.vocab_file is None:
        # Should set both padding_side and truncation_side to be 'left'
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_dir,
                                                  legacy=False,
                                                  padding_side='left',
                                                  truncation_side='left',
                                                  trust_remote_code=True)
    else:
        # From gpt-next
        tokenizer = T5Tokenizer(vocab_file=args.vocab_file,
                                padding_side='left',
                                truncation_side='left')
    profiler.stop('load tokenizer')
    logger.info(
        f'Load tokenizer takes: {profiler.elapsed_time_in_sec("load tokenizer")} sec'
    )

    # TODO: The below lines are used to be compatible with the original code; may need fix
    if model_name.startswith('chatglm'):
        pass
    elif model_name == 'falcon' and tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    elif model_name == 'qwen':
        tokenizer.pad_token_id = tokenizer.im_end_id
    else:
        tokenizer.pad_token = tokenizer.eos_token

    if model_name == 'falcon':
        pad_id = tokenizer.pad_token_id
        end_id = tokenizer.eos_token_id
    elif model_name == 'qwen':
        pad_id = tokenizer.pad_token_id
        end_id = tokenizer.im_end_id
    else:
        pad_id = tokenizer.encode(tokenizer.pad_token,
                                  add_special_tokens=False)[0]
        end_id = tokenizer.encode(tokenizer.eos_token,
                                  add_special_tokens=False)[0]

    if args.eval_task == 'code_completion':
        dataset_name = "openai_humaneval"
        dataset_revision = None
        dataset_input_key = 'prompt'
        dataset_output_key = 'canonical_solution'
    elif args.eval_task == 'summarize':
        dataset_name = "ccdv/cnn_dailymail"
        dataset_revision = "3.0.0"
        dataset_input_key = 'article'
        dataset_output_key = 'highlights'
    dataset = load_dataset(dataset_name,
                           dataset_revision,
                           cache_dir=args.dataset_path)

    max_batch_size = args.batch_size

    # runtime parameters
    top_k = args.top_k
    top_p = args.top_p
    output_len = args.output_len
    test_token_num = args.max_input_length
    # random_seed = 5
    temperature = 1
    num_beams = args.num_beams
    length_penalty = args.length_penalty
    repetition_penalty = args.repetition_penalty

    if test_trt_llm:
        runner = ModelRunner.from_dir(args.engine_dir,
                                      rank=runtime_rank,
                                      debug_mode=args.debug_mode)
        assert not (args.eval_ppl and not runner.session.gather_all_token_logits), \
            "PPL evaluation requires engine built with gather_all_token_logits enabled"

    if test_hf:
        profiler.start('load HF model')
        torch_dtype = DTYPE_STR_MAPPING[args.data_type]
        if model_name.startswith('chatglm'):
            auto_model_cls = AutoModel
        elif model_name.startswith('glm'):
            auto_model_cls = AutoModelForSeq2SeqLM
        else:
            auto_model_cls = AutoModelForCausalLM
        model = auto_model_cls.from_pretrained(
            args.hf_model_dir,
            trust_remote_code=True,
            torch_dtype=torch_dtype,
            device_map='auto' if args.hf_device_map_auto else None)
        if not args.hf_device_map_auto:
            model.cuda()
        if model_name == 'qwen':
            model.generation_config = GenerationConfig.from_pretrained(
                args.hf_model_dir, trust_remote_code=True)
        profiler.stop('load HF model')
        logger.info(
            f'Load HF model takes: {profiler.elapsed_time_in_sec("load HF model")} sec'
        )

    output_dir = Path(args.output_dir) if args.output_dir else None
    if output_dir is not None:
        output_dir.mkdir(exist_ok=True, parents=True)
        if test_trt_llm:
            with (output_dir / 'trtllm.out').open('w') as f:
                f.write(f'Engine path: {args.engine_dir}\n')
                f.write(f'Tokenizer path: {args.tokenizer_dir}\n')
        if test_hf:
            with (output_dir / 'hf.out').open('w') as f:
                f.write(f'Model path: {args.hf_model_dir}\n')
                f.write(f'Tokenizer path: {args.tokenizer_dir}\n')

    def _prepare_inputs(batch_input_texts,
                        eval_task='summarize',
                        add_special_tokens=True):
        batch_size = len(batch_input_texts)
        append_str = ' TL;DR: ' if eval_task == 'summarize' else ''
        batch_input_ids = []
        for i in range(batch_size):
            curr_text = batch_input_texts[i] + append_str
            curr_text = curr_text.strip().replace(" n't", "n't")

            # TODO: The below lines are used to be compatible with the original code; may need fix
            if model_name.startswith(('chatglm2', 'chatglm3')):
                input_ids = tokenizer.encode(curr_text, return_tensors='pt')
                input_ids = input_ids[:, :test_token_num]
            elif model_name == 'qwen':
                # use make_content to generate prompt
                system_prompt = "You are a useful assistant, please directly output the corresponding summary according to the article entered by the user."
                _, input_id_list = make_context(
                    tokenizer=tokenizer,
                    query=curr_text,
                    history=[],
                    system=system_prompt,
                    max_input_length=test_token_num,
                )
                input_ids = torch.tensor(input_id_list).unsqueeze(0)
            else:
                input_ids = tokenizer.encode(
                    curr_text,
                    return_tensors='pt',
                    add_special_tokens=add_special_tokens,
                    truncation=True,
                    max_length=test_token_num)

            batch_input_ids.append(input_ids)
        return batch_input_ids

    def eval_trt_llm(datapoint,
                     eval_task='summarize',
                     eval_ppl=False,
                     add_special_tokens=True):
        batch_size = len(datapoint[dataset_input_key])
        batch_input_ids = _prepare_inputs(datapoint[dataset_input_key],
                                          eval_task=eval_task,
                                          add_special_tokens=add_special_tokens)
        input_lengths = [x.size(1) for x in batch_input_ids]

        with torch.no_grad():
            outputs = runner.generate(
                batch_input_ids,
                max_new_tokens=output_len,
                max_kv_cache_length=args.max_kv_cache_length,
                end_id=end_id,
                pad_id=pad_id,
                top_k=top_k,
                top_p=top_p,
                num_beams=num_beams,
                length_penalty=length_penalty,
                repetition_penalty=repetition_penalty,
                output_sequence_lengths=True,
                return_dict=True)
            torch.cuda.synchronize()

        # Extract a list of tensors of shape beam_width x output_ids.
        if runner.session.mapping.is_first_pp_rank():
            output_ids = outputs['output_ids']
            output_beams_list = [
                tokenizer.batch_decode(output_ids[batch_idx, :,
                                                  input_lengths[batch_idx]:],
                                       skip_special_tokens=True)
                for batch_idx in range(batch_size)
            ]
            output_ids_list = [
                output_ids[batch_idx, :, input_lengths[batch_idx]:]
                for batch_idx in range(batch_size)
            ]

            ppls = []
            if eval_ppl:
                seq_lengths = outputs['sequence_lengths']
                context_logits = outputs['context_logits']
                # Remove the first generation logits which are same to last context logits
                generation_logits = torch.stack(
                    outputs['generation_logits'][1:], dim=1)
                for bidx in range(batch_size):
                    # [batch, beam, step]
                    curr_len = seq_lengths[bidx, 0]
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

            return output_beams_list, output_ids_list, ppls
        return [], [], []

    def eval_hf(datapoint,
                eval_task='summarize',
                eval_ppl=False,
                add_special_tokens=True):
        batch_size = len(datapoint[dataset_input_key])
        if batch_size > 1:
            logger.warning(
                f"HF does not support batch_size > 1 to verify correctness due to padding. Current batch size is {batch_size}"
            )
        batch_input_ids = _prepare_inputs(datapoint[dataset_input_key],
                                          eval_task=eval_task,
                                          add_special_tokens=add_special_tokens)
        input_lengths = [x.size(1) for x in batch_input_ids]
        # Left padding for HF
        max_length = max(input_lengths)
        paddings = [
            torch.ones(max_length - l, dtype=torch.int32) * pad_id
            for l in input_lengths
        ]
        batch_input_ids = [
            torch.cat([pad, x.squeeze(0)])
            for x, pad in zip(batch_input_ids, paddings)
        ]
        batch_input_ids = torch.stack(batch_input_ids)
        batch_input_ids = batch_input_ids.cuda()

        with torch.no_grad():
            outputs = model.generate(batch_input_ids,
                                     max_new_tokens=output_len,
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
            if eval_ppl and batch_size == 1:
                # model.generate cannot return context logits?
                # Will cause additional latency
                context_outputs = model(batch_input_ids)

        output_ids = outputs['sequences']
        tokens_list = output_ids[:, len(batch_input_ids[0]):].tolist()
        output_ids = output_ids.reshape([batch_size, num_beams, -1])
        output_lines_list = [
            tokenizer.batch_decode(output_ids[:, i,
                                              len(batch_input_ids[0]):],
                                   skip_special_tokens=True)
            for i in range(num_beams)
        ]

        ppls = []
        if eval_ppl and batch_size == 1:
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
        output, *_ = eval_trt_llm(datapoint,
                                  eval_task=args.eval_task,
                                  eval_ppl=args.eval_ppl,
                                  add_special_tokens=args.add_special_tokens)
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
        output, *_ = eval_hf(datapoint,
                             eval_task=args.eval_task,
                             eval_ppl=args.eval_ppl,
                             add_special_tokens=args.add_special_tokens)
        logger.info("---------------------------------------------------------")
        logger.info("HF Generated : ")
        logger.info(f" Input : {datapoint[dataset_input_key]}")
        logger.info(f"\n Reference : {datapoint[dataset_output_key]}")
        logger.info(f"\n Output : {output}")
        logger.info("---------------------------------------------------------")

    # TODO: Add random_seed flag in gptj
    metric_tensorrt_llm = [evaluate.load("rouge") for _ in range(num_beams)]
    metric_hf = [evaluate.load("rouge") for _ in range(num_beams)]
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
            output_tensorrt_llm, _, curr_ppls_trt_llm = eval_trt_llm(
                datapoint,
                eval_task=args.eval_task,
                eval_ppl=args.eval_ppl,
                add_special_tokens=args.add_special_tokens)
            profiler.stop('tensorrt_llm')

        if test_hf:
            profiler.start('hf')
            output_hf, _, curr_ppls_hf = eval_hf(
                datapoint,
                eval_task=args.eval_task,
                eval_ppl=args.eval_ppl,
                add_special_tokens=args.add_special_tokens)
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
                if output_dir is not None:
                    # yapf: disable
                    for i in range(len(output_tensorrt_llm[0])):
                        for beam_idx in range(num_beams):
                            with (output_dir / 'trtllm.out').open('a') as f:
                                f.write(f'[{data_point_idx + i}] [Beam {beam_idx}] {output_tensorrt_llm[beam_idx][i]}\n')
                    # yapf: enable
                ppls_trt_llm.extend(curr_ppls_trt_llm)
            if test_hf:
                for beam_idx in range(num_beams):
                    for batch_idx in range(len(output_hf[beam_idx])):
                        metric_hf[beam_idx].add_batch(
                            predictions=[output_hf[beam_idx][batch_idx]],
                            references=[
                                datapoint[dataset_output_key][batch_idx]
                            ])
                if output_dir is not None:
                    # yapf: disable
                    for i in range(len(output_hf[0])):
                        for beam_idx in range(num_beams):
                            with (output_dir / 'hf.out').open('a') as f:
                                f.write(f'[{data_point_idx + i}] [Beam {beam_idx}] {output_hf[beam_idx][i]}\n')
                    # yapf: enable
                ppls_hf.extend(curr_ppls_hf)

            logger.debug('-' * 100)
            logger.debug(f"Input : {datapoint[dataset_input_key]}")
            if test_trt_llm:
                logger.debug(f'TensorRT-LLM Output: {output_tensorrt_llm}')
            if test_hf:
                logger.debug(f'HF Output: {output_hf}')
            logger.debug(f"Reference : {datapoint[dataset_output_key]}")

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
                        f'  {key} : {computed_metrics_tensorrt_llm[key]*100}')

                if args.check_accuracy and beam_idx == 0:
                    assert computed_metrics_tensorrt_llm[
                        'rouge1'] * 100 > args.tensorrt_llm_rouge1_threshold
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
                    logger.info(f'  {key} : {computed_metrics_hf[key]*100}')
            if args.eval_ppl and args.batch_size == 1:
                logger.info(f"  Per-token perplexity: {np.mean(ppls_hf)}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--hf_model_dir', type=str, default=None)
    parser.add_argument(
        '--tokenizer_dir',
        default=None,
        help='tokenizer path; defaults to hf_model_dir if left unspecified')
    parser.add_argument('--vocab_file')
    parser.add_argument('--test_hf', action='store_true')
    parser.add_argument('--test_trt_llm', action='store_true')
    parser.add_argument(
        '--data_type',
        type=str,
        choices=['fp32', 'fp16', 'bf16', 'float32', 'float16', 'bfloat16'],
        default='fp16')
    parser.add_argument('--engine_dir', type=str, default='engine_outputs')
    parser.add_argument('--eval_task',
                        type=str,
                        default='summarize',
                        choices=['summarize', 'code_completion'])
    parser.add_argument('--eval_ppl', action='store_true')
    parser.add_argument('--check_accuracy', action='store_true')
    parser.add_argument('--tensorrt_llm_rouge1_threshold',
                        type=float,
                        default=15.0)
    parser.add_argument('--dataset_path', type=str, default='')
    parser.add_argument('--log_level', type=str, default='info')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--max_ite', type=int, default=20)
    parser.add_argument('--output_len', type=int, default=100)
    parser.add_argument('--max_input_length', type=int, default=923)
    parser.add_argument('--max_kv_cache_length',
                        type=int,
                        default=None,
                        help='The max kv cache length. \
              If the final sequence length exceeds the kv cache length, we will enable cyclic kv cache. \
              If it is set to None, we will use the max sequence length.')
    parser.add_argument('--num_beams', type=int, default=1)
    parser.add_argument('--top_k', type=int, default=1)
    parser.add_argument('--top_p', type=float, default=0.0)
    parser.add_argument('--length_penalty', type=float, default=1.0)
    parser.add_argument('--repetition_penalty', type=float, default=1.0)
    parser.add_argument('--debug_mode',
                        default=False,
                        action='store_true',
                        help="Whether or not to turn on the debug mode")
    parser.add_argument('--no_add_special_tokens',
                        dest='add_special_tokens',
                        default=True,
                        action='store_false',
                        help="Whether or not to add special tokens")
    parser.add_argument(
        '--hf_device_map_auto',
        action='store_true',
        help="Use device map 'auto' to load a pretrained HF model. This may "
        "help to test a large model that cannot fit into a singlue GPU.")
    parser.add_argument(
        '--output_dir',
        type=str,
        default=None,
        help="Directory where to save output sentences. 'trtllm.out' for "
        "TensorRT-LLM outputs, and 'hf.out' for HF outputs.  If None, do not "
        "save outputs.")

    args = parser.parse_args()

    main(args)
