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
import ast
from pathlib import Path

import evaluate
import numpy as np
import torch
from datasets import load_dataset
from qwen.utils.utils import make_context
from transformers import (AutoModel, AutoModelForCausalLM,
                          AutoModelForSeq2SeqLM, GenerationConfig)
from utils import DEFAULT_HF_MODEL_DIRS, load_tokenizer, read_model_name

import tensorrt_llm
import tensorrt_llm.profiler as profiler
from tensorrt_llm._utils import str_dtype_to_torch
from tensorrt_llm.logger import logger
from tensorrt_llm.runtime import PYTHON_BINDINGS, ModelRunner
from tensorrt_llm.tools.ppl import ppl

if PYTHON_BINDINGS:
    from tensorrt_llm.runtime import ModelRunnerCpp


def main(args):
    runtime_rank = tensorrt_llm.mpi_rank()
    logger.set_level(args.log_level)

    model_name = read_model_name(args.engine_dir)
    if args.hf_model_dir is None:
        args.hf_model_dir = DEFAULT_HF_MODEL_DIRS[model_name]
    if args.tokenizer_dir is None:
        args.tokenizer_dir = args.hf_model_dir

    test_hf = args.test_hf and runtime_rank == 0  # only run hf on rank 0
    test_trt_llm = args.test_trt_llm
    profiler.start('load tokenizer')
    tokenizer, pad_id, end_id = load_tokenizer(
        tokenizer_dir=args.tokenizer_dir,
        vocab_file=args.vocab_file,
        model_name=model_name,
    )
    profiler.stop('load tokenizer')
    logger.info(
        f'Load tokenizer takes: {profiler.elapsed_time_in_sec("load tokenizer")} sec'
    )

    if args.eval_task == 'code_completion':
        dataset_name = "openai_humaneval"
        dataset_revision = None
        dataset_input_key = 'prompt'
        dataset_output_key = 'canonical_solution'
        dataset_split = 'test'
    elif args.eval_task == 'summarize':
        dataset_name = "ccdv/cnn_dailymail"
        dataset_revision = "3.0.0"
        dataset_input_key = 'article'
        dataset_output_key = 'highlights'
        dataset_split = 'test'
    elif args.eval_task == 'summarize_long':
        dataset_name = "tau/zero_scrolls"
        dataset_revision = 'squality'
        dataset_input_key = 'input'
        dataset_output_key = 'output'
        dataset_split = 'validation'  # only this split contains reference strings
    dataset = load_dataset(dataset_name,
                           dataset_revision,
                           cache_dir=args.dataset_path,
                           split=dataset_split)

    max_batch_size = args.batch_size

    # runtime parameters
    top_k = args.top_k
    top_p = args.top_p
    output_len = args.output_len
    test_token_num = args.max_input_length
    max_attention_window_size = args.max_attention_window_size
    sink_token_length = args.sink_token_length

    # random_seed = 5
    temperature = args.temperature
    num_beams = args.num_beams
    length_penalty = args.length_penalty
    repetition_penalty = args.repetition_penalty
    presence_penalty = args.presence_penalty
    frequency_penalty = args.frequency_penalty

    if test_trt_llm:
        if not PYTHON_BINDINGS and not args.use_py_session:
            logger.warning(
                "Python bindings of C++ session is unavailable, fallback to Python session."
            )
            args.use_py_session = True
        runner_cls = ModelRunner if args.use_py_session else ModelRunnerCpp
        runner_kwargs = dict(engine_dir=args.engine_dir,
                             rank=runtime_rank,
                             debug_mode=args.debug_mode)
        if args.medusa_choices is not None:
            args.medusa_choices = ast.literal_eval(args.medusa_choices)
            assert args.use_py_session, "Medusa is only supported by py_session"
            assert args.temperature == 0, "Medusa should use temperature == 0"
            assert args.num_beams == 1, "Medusa should use num_beams == 1"
            runner_kwargs.update(medusa_choices=args.medusa_choices)
        if not args.use_py_session:
            runner_kwargs.update(
                max_batch_size=max_batch_size,
                max_input_len=test_token_num,
                max_output_len=output_len,
                max_beam_width=num_beams,
                max_attention_window_size=max_attention_window_size,
                sink_token_length=sink_token_length)
        runner = runner_cls.from_dir(**runner_kwargs)
        assert not (args.eval_ppl and not (runner.gather_context_logits and runner.gather_generation_logits)), \
            "PPL evaluation requires engine built with gather_all_token_logits enabled"

    if test_hf:
        profiler.start('load HF model')
        dtype_alias_mapping = {
            'fp32': 'float32',
            'fp16': 'float16',
            'bf16': 'bfloat16'
        }
        args.data_type = dtype_alias_mapping.get(args.data_type, args.data_type)
        if model_name.startswith('chatglm'):
            auto_model_cls = AutoModel
        elif model_name.startswith('glm'):
            auto_model_cls = AutoModelForSeq2SeqLM
        else:
            auto_model_cls = AutoModelForCausalLM
        model = auto_model_cls.from_pretrained(
            args.hf_model_dir,
            trust_remote_code=True,
            torch_dtype=str_dtype_to_torch(args.data_type),
            device_map='auto' if args.hf_device_map_auto else None)
        try:
            model.to_bettertransformer()
        except ValueError as e:
            logger.warning(
                f'Fail to call model.to_bettertransformer(), exception:\n{str(e)}'
            )
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
                input_ids = tokenizer.encode(curr_text,
                                             return_tensors='pt').squeeze(0)
                input_ids = input_ids[:test_token_num]
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
                input_ids = torch.tensor(input_id_list)
            else:
                input_ids = tokenizer.encode(
                    curr_text,
                    return_tensors='pt',
                    add_special_tokens=add_special_tokens,
                    truncation=True,
                    max_length=test_token_num).squeeze(0)

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
        input_lengths = [x.size(0) for x in batch_input_ids]

        with torch.no_grad():
            outputs = runner.generate(
                batch_input_ids,
                max_new_tokens=output_len,
                max_attention_window_size=max_attention_window_size,
                sink_token_length=sink_token_length,
                end_id=end_id,
                pad_id=pad_id,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                num_beams=num_beams,
                length_penalty=length_penalty,
                repetition_penalty=repetition_penalty,
                presence_penalty=presence_penalty,
                frequency_penalty=frequency_penalty,
                output_sequence_lengths=True,
                return_dict=True,
                medusa_choices=args.medusa_choices)
            torch.cuda.synchronize()

        # Extract a list of tensors of shape beam_width x output_ids.
        if runtime_rank == 0:
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

            ppls = [[] for _ in range(batch_size)]
            seq_lengths_array = outputs["sequence_lengths"].cpu().tolist()
            lengths_info = {
                'input_lengths': input_lengths,
                'seq_lengths': seq_lengths_array
            }
            if eval_ppl:
                seq_lengths = outputs['sequence_lengths']
                context_logits = outputs['context_logits']
                # Remove the first generation logits which are same to last context logits
                generation_logits = outputs['generation_logits'][:, :, 1:]
                for batch_idx in range(batch_size):
                    # [batch, beam, step]
                    for beam_idx in range(num_beams):
                        curr_len = seq_lengths[batch_idx, beam_idx]
                        curr_ctx_len = input_lengths[batch_idx]
                        curr_gen_len = curr_len - curr_ctx_len

                        curr_ids = output_ids[batch_idx, beam_idx, 1:curr_len]
                        curr_logits = torch.cat([
                            context_logits[batch_idx],
                            generation_logits[batch_idx,
                                              beam_idx, :curr_gen_len - 1]
                        ],
                                                dim=0)
                        curr_ppl = ppl(curr_logits, curr_ids)
                        logger.debug(
                            f"TensorRT-LLM PPL: {curr_ppl:.3f} | Generation length: {curr_gen_len}"
                        )
                        ppls[batch_idx].append(curr_ppl)

            return output_beams_list, output_ids_list, ppls, lengths_info
        return [], [], [], {}

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
        input_lengths = [x.size(0) for x in batch_input_ids]
        # Left padding for HF
        max_length = max(input_lengths)
        paddings = [
            torch.ones(max_length - l, dtype=torch.int32) * pad_id
            for l in input_lengths
        ]
        batch_input_ids = [
            torch.cat([pad, x]) for x, pad in zip(batch_input_ids, paddings)
        ]
        batch_input_ids = torch.stack(batch_input_ids)
        batch_input_ids = batch_input_ids.cuda()

        with torch.no_grad():
            outputs = model.generate(batch_input_ids,
                                     max_new_tokens=output_len,
                                     top_k=top_k,
                                     temperature=temperature,
                                     eos_token_id=end_id,
                                     pad_token_id=pad_id,
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

        ppls = [[] for _ in range(batch_size)]
        if eval_ppl and batch_size == 1:
            # Only for batch size of 1
            seq_lens = (output_ids != end_id).logical_and(
                output_ids != pad_id).sum(dim=-1)
            context_logits = context_outputs['logits']
            # Remove the first generation logits which are same to last context logits
            generation_logits = torch.stack(outputs['scores'][1:], dim=1)
            _, max_gen_len, voc_size = generation_logits.size()
            generation_logits = generation_logits.view(batch_size, num_beams,
                                                       max_gen_len, voc_size)
            for batch_idx in range(batch_size):
                for beam_idx in range(num_beams):
                    curr_len = seq_lens[batch_idx, beam_idx]
                    curr_ctx_len = input_lengths[batch_idx]
                    curr_gen_len = curr_len - curr_ctx_len

                    curr_ids = output_ids[batch_idx, beam_idx, 1:curr_len]
                    curr_logits = torch.cat([
                        context_logits[batch_idx],
                        generation_logits[batch_idx,
                                          beam_idx, :curr_gen_len - 1]
                    ],
                                            dim=0)
                    curr_ppl = ppl(curr_logits, curr_ids)
                    logger.debug(
                        f"HF PPL: {curr_ppl:.3f} | Generation length: {curr_gen_len}"
                    )
                    ppls[batch_idx].append(curr_ppl)

        return output_lines_list, tokens_list, ppls

    if test_trt_llm:
        datapoint = dataset[0:1]
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
        datapoint = dataset[0:1]
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
    ppls_trt_llm = [[] for _ in range(num_beams)]
    ppls_hf = [[] for _ in range(num_beams)]

    ite_count = 0
    data_point_idx = 0
    total_output_token_count_trt_llm = 0  # only valid for runtime_rank == 0
    while (data_point_idx < len(dataset)) and (ite_count < args.max_ite):
        if runtime_rank == 0:
            logger.debug(
                f"run data_point {data_point_idx} ~ {data_point_idx + max_batch_size}"
            )
        datapoint = dataset[data_point_idx:(data_point_idx + max_batch_size)]

        if test_trt_llm:
            profiler.start('tensorrt_llm')
            output_tensorrt_llm, output_ids_trt_llm, curr_ppls_trt_llm, lengths_info = eval_trt_llm(
                datapoint,
                eval_task=args.eval_task,
                eval_ppl=args.eval_ppl,
                add_special_tokens=args.add_special_tokens)
            profiler.stop('tensorrt_llm')
            if runtime_rank == 0:
                input_lengths = lengths_info['input_lengths']
                seq_lengths = lengths_info['seq_lengths']
                output_token_count_trt_llm = sum(
                    seq_lengths[idx][0] - input_lengths[idx]
                    for idx in range(len(input_lengths)))
                total_output_token_count_trt_llm += output_token_count_trt_llm

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
                        if args.eval_ppl:
                            ppls_trt_llm[beam_idx].append(
                                curr_ppls_trt_llm[batch_idx][beam_idx])
                if output_dir is not None:
                    # yapf: disable
                    for i in range(len(output_tensorrt_llm[0])):
                        for beam_idx in range(num_beams):
                            with (output_dir / 'trtllm.out').open('a') as f:
                                f.write(f'[{data_point_idx + i}] [Beam {beam_idx}] {output_tensorrt_llm[beam_idx][i]}\n')
                    # yapf: enable
            if test_hf:
                for beam_idx in range(num_beams):
                    for batch_idx in range(len(output_hf[beam_idx])):
                        metric_hf[beam_idx].add_batch(
                            predictions=[output_hf[beam_idx][batch_idx]],
                            references=[
                                datapoint[dataset_output_key][batch_idx]
                            ])
                        if args.eval_ppl and args.batch_size == 1:
                            ppls_hf[beam_idx].append(
                                curr_ppls_hf[batch_idx][beam_idx])
                if output_dir is not None:
                    # yapf: disable
                    for i in range(len(output_hf[0])):
                        for beam_idx in range(num_beams):
                            with (output_dir / 'hf.out').open('a') as f:
                                f.write(f'[{data_point_idx + i}] [Beam {beam_idx}] {output_hf[beam_idx][i]}\n')
                    # yapf: enable

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
            logger.info(
                f'TensorRT-LLM (total output tokens: {total_output_token_count_trt_llm})'
            )
            logger.info(
                f'TensorRT-LLM (tokens per second: {total_output_token_count_trt_llm / profiler.elapsed_time_in_sec("tensorrt_llm")})'
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
                    logger.info(
                        f"  Per-token perplexity: {np.mean(ppls_trt_llm[beam_idx])}"
                    )
                    if args.check_accuracy and beam_idx == 0:
                        assert np.mean(ppls_trt_llm[beam_idx]
                                       ) < args.tensorrt_llm_ppl_threshold
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
                    logger.info(
                        f"  Per-token perplexity: {np.mean(ppls_hf[beam_idx])}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--hf_model_dir', '--model_dir', type=str, default=None)
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
    parser.add_argument('--use_py_session',
                        default=False,
                        action='store_true',
                        help="Whether or not to use Python runtime session")
    parser.add_argument(
        '--eval_task',
        type=str,
        default='summarize',
        choices=['summarize', 'summarize_long', 'code_completion'])
    parser.add_argument('--check_accuracy', action='store_true')
    parser.add_argument('--tensorrt_llm_rouge1_threshold',
                        type=float,
                        default=15.0)
    parser.add_argument('--eval_ppl', action='store_true')
    parser.add_argument('--tensorrt_llm_ppl_threshold',
                        type=float,
                        default=15.0)
    parser.add_argument('--dataset_path', type=str, default='')
    parser.add_argument('--log_level', type=str, default='info')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--max_ite', type=int, default=20)
    parser.add_argument('--output_len', type=int, default=100)
    parser.add_argument('--max_input_length', type=int, default=923)
    parser.add_argument(
        '--max_attention_window_size',
        type=int,
        default=None,
        help=
        'The attention window size that controls the sliding window attention / cyclic kv cache behaviour'
    )
    parser.add_argument('--sink_token_length',
                        type=int,
                        default=None,
                        help='The sink token length.')
    parser.add_argument('--num_beams', type=int, default=1)
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--top_k', type=int, default=1)
    parser.add_argument('--top_p', type=float, default=0.0)
    parser.add_argument('--length_penalty', type=float, default=1.0)
    parser.add_argument('--repetition_penalty', type=float, default=1.0)
    parser.add_argument('--presence_penalty', type=float, default=0.0)
    parser.add_argument('--frequency_penalty', type=float, default=0.0)
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
    parser.add_argument(
        '--medusa_choices',
        type=str,
        default=None,
        help="Medusa choice to use, if not none, will use Medusa decoding."
        "   E.g.: [[0, 0, 0, 0], [0, 1, 0], [1, 0], [1, 1]] for 9 medusa tokens."
    )

    args = parser.parse_args()

    main(args)
