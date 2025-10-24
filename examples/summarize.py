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
import os
from pathlib import Path

import evaluate
import numpy as np
import torch
from datasets import load_dataset
from transformers import (AutoModel, AutoModelForCausalLM,
                          AutoModelForSeq2SeqLM, GenerationConfig)
from utils import (DEFAULT_HF_MODEL_DIRS, add_common_args, get_beam_width_array,
                   load_tokenizer, read_model_name, supports_inflight_batching)

import tensorrt_llm
import tensorrt_llm.profiler as profiler
from tensorrt_llm._utils import mpi_broadcast, str_dtype_to_torch
from tensorrt_llm.builder import EngineConfig
from tensorrt_llm.functional import RopeEmbeddingUtils, RotaryScalingType
from tensorrt_llm.layers import MropeParams
from tensorrt_llm.logger import logger
from tensorrt_llm.models.qwen.utils import make_context
from tensorrt_llm.runtime import PYTHON_BINDINGS, ModelRunner
from tensorrt_llm.tools.ppl import ppl

if PYTHON_BINDINGS:
    from tensorrt_llm.runtime import ModelRunnerCpp

from ngram.run_dtm_ngram import run_dtm_ngram


def ensemble_mrope_params(batch_input_ids, max_position_embeddings,
                          rotary_embedding_dim, theta):
    mrope_params = MropeParams()
    batch_size = len(batch_input_ids)

    _, rotary_cos_sin = RopeEmbeddingUtils.create_sinusoidal_positions_for_attention_plugin(
        num_pos=max_position_embeddings,
        dim=rotary_embedding_dim,
        theta=1000000.0,
        scale_type=RotaryScalingType.mrope,
    )
    rotary_cos_sin = torch.tensor(rotary_cos_sin).to(batch_input_ids[0].device)
    rotary_cos_sin = rotary_cos_sin.reshape(max_position_embeddings,
                                            int(rotary_embedding_dim / 2), 2)

    cos_ori = rotary_cos_sin[:, :, 0]
    sin_ori = rotary_cos_sin[:, :, 1]

    mrope_position_ids_padding = torch.zeros(
        (batch_size, max_position_embeddings), dtype=torch.int32)
    for i in range(batch_size):
        seq_len = batch_input_ids[i].shape[-1]
        mrope_position_ids_padding[i, :seq_len] = torch.arange(
            seq_len, device=batch_input_ids[i].device)

    cos = cos_ori[mrope_position_ids_padding].unsqueeze(-1)
    sin = sin_ori[mrope_position_ids_padding].unsqueeze(-1)

    mrope_params.mrope_rotary_cos_sin = torch.concatenate(
        (cos, sin), axis=-1).reshape(batch_size, -1)
    mrope_params.mrope_position_deltas = torch.zeros(
        [batch_size, 1], device=batch_input_ids[0].device)

    return mrope_params


def main(args):
    is_integration_test = os.getenv('INTEGRATION_TEST', '0') == '1'
    if is_integration_test:
        logger.info(
            "Running in integration test mode - will only run one batch and skip accuracy checks"
        )
        logger.info(
            "Setting max_ite=1 and check_accuracy=False for integration test")
        args.max_ite = 1
        args.check_accuracy = False

    runtime_rank = tensorrt_llm.mpi_rank()
    logger.set_level(args.log_level)

    test_hf = args.test_hf and runtime_rank == 0  # only run hf on rank 0
    test_trt_llm = args.test_trt_llm
    model_name, model_version = read_model_name(
        args.engine_dir if not test_hf else args.hf_model_dir, test_hf)
    if args.hf_model_dir is None:
        logger.warning(
            "hf_model_dir is not specified. Try to infer from model_name, but this may be incorrect."
        )
        if model_name in DEFAULT_HF_MODEL_DIRS:
            args.hf_model_dir = DEFAULT_HF_MODEL_DIRS[model_name]
        else:
            args.hf_model_dir = None
    if args.tokenizer_dir is None:
        args.tokenizer_dir = args.hf_model_dir

    profiler.start('load tokenizer')
    tokenizer, pad_id, end_id = load_tokenizer(
        tokenizer_dir=args.tokenizer_dir,
        vocab_file=args.vocab_file,
        model_name=model_name,
        model_version=model_version,
        tokenizer_type=args.tokenizer_type,
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
    elif args.eval_task == "eval_context_ppl":
        dataset_name = "SlimPajama-6B"
        dataset_revision = None
        dataset_input_key = 'text'
        dataset_output_key = 'text'
        dataset_split = 'test'
        args.output_len = 1  # Only want to compute the ppl of context
        args.eval_ppl = True
        logger.warning(
            f"Run task '{args.eval_task}', setting 'output_len' to 1, and enable 'eval_ppl'."
        )
    if args.dataset_dir is not None and isinstance(args.dataset_dir, str):
        args.dataset_dir = args.dataset_dir.rstrip('/')
        if args.dataset_dir.endswith(dataset_name):
            dataset_name = args.dataset_dir
        else:
            dataset_name = f"{args.dataset_dir}/{dataset_name}"
    dataset = load_dataset(dataset_name,
                           dataset_revision,
                           cache_dir=args.dataset_cache_dir,
                           split=dataset_split,
                           trust_remote_code=True)
    dataset = dataset.shuffle(args.random_seed)

    max_batch_size = args.batch_size

    # runtime parameters
    top_k = args.top_k
    top_p = args.top_p
    output_len = args.output_len
    test_token_num = args.max_input_length
    max_attention_window_size = args.max_attention_window_size
    sink_token_length = args.sink_token_length

    if args.end_id:
        end_id = args.end_id

    stop_words_list = None
    if args.stop_words:
        stop_words_list = tensorrt_llm.runtime.decode_words_list(
            args.stop_words, tokenizer)
    if model_version == 'glm4':  # add default stop token ids for GLM-4
        glm4_stop_ids = [[151329], [151336], [151338]]
        if stop_words_list is None:
            stop_words_list = [glm4_stop_ids] * args.batch_size
        else:
            for req_stop_words_list in stop_words_list:
                req_stop_words_list.extend(glm4_stop_ids)

    bad_words_list = None
    if args.bad_words:
        bad_words_list = tensorrt_llm.runtime.decode_words_list(
            args.bad_words, tokenizer)

    if args.beam_width_array is not None:
        logger.info("Use Variable-Beam-Width-Search")
        args.beam_width_array, args.num_beams = get_beam_width_array(
            args.beam_width_array)

    num_beams = args.num_beams
    num_return_sequences = args.num_return_sequences
    num_sequences = args.num_return_sequences or num_beams
    assert num_beams == 1 or num_sequences <= num_beams

    temperature = args.temperature
    length_penalty = args.length_penalty
    early_stopping = args.early_stopping
    beam_width_array = args.beam_width_array
    repetition_penalty = args.repetition_penalty
    presence_penalty = args.presence_penalty
    frequency_penalty = args.frequency_penalty
    random_seed = args.random_seed
    torch.manual_seed(random_seed)

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

    rouge_dir = args.rouge_dir if args.rouge_dir and os.path.exists(
        args.rouge_dir) else "rouge"
    metric_tensorrt_llm = [
        evaluate.load(rouge_dir) for _ in range(num_sequences)
    ]
    metric_hf = [evaluate.load(rouge_dir) for _ in range(num_sequences)]
    for i in range(num_sequences):
        metric_tensorrt_llm[i].seed = 0
        metric_hf[i].seed = 0
    ppls_trt_llm = [[] for _ in range(num_sequences)]
    ppls_hf = [[] for _ in range(num_sequences)]

    def _prepare_inputs(batch_input_texts,
                        eval_task='summarize',
                        add_special_tokens=True,
                        min_input_length=0):
        batch_size = len(batch_input_texts)
        append_str = ' TL;DR: ' if eval_task == 'summarize' else ''
        batch_input_ids = []
        for i in range(batch_size):
            curr_text = batch_input_texts[i] + append_str
            curr_text = curr_text.strip().replace(" n't", "n't")

            # TODO: The below lines are used to be compatible with the original code; may need fix
            if 'GLM' in model_name and model_version in ('chatglm2',
                                                         'chatglm3'):
                input_ids = tokenizer.encode(curr_text,
                                             return_tensors='pt').squeeze(0)
                input_ids = input_ids[:test_token_num]
            elif 'qwen' in model_name.lower() and model_version == 'qwen':
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
                if 'qwen' in model_name.lower() and 'qwen2' in model_version:
                    messages = [{
                        "role":
                        "system",
                        "content":
                        "You are a helpful assistant, please summarize the article entered by the user with one or two sentences."
                    }, {
                        "role": "user",
                        "content": curr_text
                    }]
                    curr_text = tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True)
                input_ids = tokenizer.encode(
                    curr_text,
                    return_tensors='pt',
                    add_special_tokens=add_special_tokens,
                    truncation=True,
                    max_length=test_token_num).squeeze(0)

            if input_ids.numel() > min_input_length:
                batch_input_ids.append(input_ids)
        return batch_input_ids

    def eval_trt_llm(datapoint,
                     eval_task='summarize',
                     eval_ppl=False,
                     add_special_tokens=True,
                     min_input_length=0,
                     runner=None):
        batch_size = len(datapoint[dataset_input_key])
        batch_input_ids = _prepare_inputs(datapoint[dataset_input_key],
                                          eval_task=eval_task,
                                          add_special_tokens=add_special_tokens,
                                          min_input_length=min_input_length)
        # Generate mrope params for qwen model
        engine_config = EngineConfig.from_json_file(
            f"{args.engine_dir}/config.json")
        pretrain_config = engine_config.pretrained_config
        mrope_params = None
        if 'qwen' in model_name.lower():
            mrope_params = ensemble_mrope_params(
                batch_input_ids,
                max_position_embeddings=pretrain_config.max_position_embeddings,
                rotary_embedding_dim=pretrain_config.rotary_embedding_dim,
                theta=pretrain_config.rotary_base,
            )

        if batch_size == 0 or len(batch_input_ids) == 0:
            return [], [], [], {}
        input_lengths = [x.size(0) for x in batch_input_ids]

        if args.ngram_config is not None:
            # Speculative decoding of NGram
            outputs = run_dtm_ngram(batch_input_ids,
                                    args,
                                    runtime_rank,
                                    end_id,
                                    pad_id,
                                    stop_words_list,
                                    bad_words_list,
                                    tokenizer.vocab_size,
                                    target_runner=runner)
            if not args.streaming:  # Unpack runner from the return value in No-Streaming mode
                outputs, runner = list(outputs)[0]
        else:  # Normal run
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
                    stop_words_list=stop_words_list,
                    bad_words_list=bad_words_list,
                    num_beams=num_beams,
                    num_return_sequences=num_return_sequences,
                    length_penalty=length_penalty,
                    early_stopping=early_stopping,
                    beam_width_array=beam_width_array,
                    repetition_penalty=repetition_penalty,
                    presence_penalty=presence_penalty,
                    frequency_penalty=frequency_penalty,
                    lora_uids=args.lora_task_uids,
                    lookahead_config=args.lookahead_config,
                    output_sequence_lengths=True,
                    output_generation_logits=eval_ppl,
                    return_dict=True,
                    random_seed=random_seed,
                    medusa_choices=args.medusa_choices,
                    eagle_choices=args.eagle_choices,
                    mrope_params=mrope_params)
                torch.cuda.synchronize()

        # Extract a list of tensors of shape beam_width x output_ids.
        if runtime_rank == 0:
            output_ids = outputs['output_ids']
            output_beams_list = [
                tokenizer.batch_decode(beam_tokens[:, input_lengths[i]:],
                                       skip_special_tokens=True)
                for i, beam_tokens in enumerate(output_ids)
            ]
            output_ids_list = [
                beam_tokens[:, input_lengths[i]:]
                for i, beam_tokens in enumerate(output_ids)
            ]

            ppls = [[] for _ in range(batch_size)]
            lengths_info = {
                'input_lengths': input_lengths,
                'seq_lengths': outputs["sequence_lengths"].cpu().tolist(),
            }
            if eval_ppl:
                seq_lengths = outputs['sequence_lengths']
                context_logits = outputs['context_logits']
                # Remove the first generation logits which are same to last
                # context logits.
                generation_logits = outputs['generation_logits'][:, :, 1:]
                for batch_idx in range(batch_size):
                    # [batch, beam, step]
                    for beam_idx in range(num_sequences):
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
                        logger.debug(f"TensorRT LLM PPL: {curr_ppl:.3f} | "
                                     f"Generation length: {curr_gen_len}")
                        ppls[batch_idx].append(curr_ppl)
            return output_beams_list, output_ids_list, ppls, lengths_info
        return [], [], [], {}

    def eval_hf(datapoint,
                eval_task='summarize',
                eval_ppl=False,
                add_special_tokens=True,
                min_input_length=0):
        batch_size = len(datapoint[dataset_input_key])
        if batch_size > 1:
            logger.warning(
                f"HF does not support batch_size > 1 to verify correctness due to padding. Current batch size is {batch_size}"
            )
        batch_input_ids = _prepare_inputs(datapoint[dataset_input_key],
                                          eval_task=eval_task,
                                          add_special_tokens=add_special_tokens,
                                          min_input_length=min_input_length)
        batch_size = len(batch_input_ids)
        if batch_size == 0:
            return [], [], [], [[] for _ in range(batch_size)]
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

        # specialization for HF
        if early_stopping in [0, 1]:
            local_early_stopping = bool(early_stopping)
        else:
            local_early_stopping = "never"

        with torch.no_grad():
            hf_config = {}
            if num_beams == 1:
                hf_config.update({
                    "top_k": top_k,
                    "top_p": top_p,
                    "do_sample": True,
                })
            else:
                hf_config.update({
                    "num_beams": num_beams,
                    "early_stopping": local_early_stopping,
                })

            outputs = model.generate(batch_input_ids,
                                     max_new_tokens=output_len,
                                     num_return_sequences=num_sequences,
                                     temperature=temperature,
                                     eos_token_id=end_id,
                                     pad_token_id=pad_id,
                                     length_penalty=length_penalty,
                                     output_scores=True,
                                     return_dict_in_generate=True,
                                     **hf_config)
            if eval_ppl and batch_size == 1:
                # model.generate cannot return context logits?
                # Will cause additional latency
                context_outputs = model(batch_input_ids)

        output_ids = outputs['sequences']
        tokens_list = output_ids[:, max_length:].tolist()
        output_ids = output_ids.reshape([batch_size, num_sequences, -1])
        output_lines_list = [
            tokenizer.batch_decode(output_ids[:, i, max_length:],
                                   skip_special_tokens=True)
            for i in range(num_sequences)
        ]

        ppls = [[] for _ in range(batch_size)]
        if eval_ppl and batch_size == 1:
            # Only for batch size of 1
            seq_lens = (output_ids
                        != end_id).logical_and(output_ids != pad_id).sum(dim=-1)
            context_logits = context_outputs['logits']
            # Remove the first generation logits which are same to last context logits
            generation_logits = outputs['scores'][1:]
            # When output_len is 1, generation_logits would be () and lead to error if we do torch.stack
            if len(generation_logits) == 0:
                generation_logits = torch.empty(
                    [context_logits.shape[0], 0, context_logits.shape[-1]],
                    device=context_logits.device)
            else:
                generation_logits = torch.stack(generation_logits, dim=1)
            _, max_gen_len, voc_size = generation_logits.size()
            generation_logits = generation_logits.view(batch_size, num_beams,
                                                       max_gen_len, voc_size)
            for batch_idx in range(batch_size):
                for beam_idx in range(num_sequences):
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
        if not supports_inflight_batching(args.engine_dir):
            logger.warning(
                "The given engine does not support in-flight batching, fallback to python session"
            )
            args.use_py_session = True

        if not PYTHON_BINDINGS and not args.use_py_session:
            logger.warning(
                "Python bindings of C++ session is unavailable, fallback to Python session."
            )
            args.use_py_session = True
        if args.return_all_generated_tokens:
            raise ValueError(
                "Returning all the generated tokens at each step is not supported in summarize.py"
            )

        logger.info(
            f"Using {'Python' if args.use_py_session else 'C++'} session")

        runner_cls = ModelRunner if args.use_py_session else ModelRunnerCpp
        runner_kwargs = dict(
            engine_dir=args.engine_dir,
            rank=runtime_rank,
            debug_mode=args.debug_mode,
            gpu_weights_percent=args.gpu_weights_percent,
            enable_context_fmha_fp32_acc=args.enable_context_fmha_fp32_acc,
        )
        if not args.use_py_session:
            runner_kwargs.update(
                lora_dir=args.lora_dir,
                lora_ckpt_source=args.lora_ckpt_source,
                max_batch_size=max_batch_size,
                max_input_len=test_token_num,
                max_output_len=output_len,
                max_beam_width=num_beams,
                max_attention_window_size=max_attention_window_size,
                sink_token_length=sink_token_length,
                max_tokens_in_paged_kv_cache=args.max_tokens_in_paged_kv_cache,
                kv_cache_enable_block_reuse=args.kv_cache_enable_block_reuse,
                kv_cache_free_gpu_memory_fraction=args.
                kv_cache_free_gpu_memory_fraction,
                enable_chunked_context=args.enable_chunked_context,
                multi_block_mode=args.multi_block_mode,
                cuda_graph_mode=args.cuda_graph_mode,
                gather_generation_logits=args.eval_ppl,
                use_gpu_direct_storage=args.use_gpu_direct_storage,
            )

        if args.medusa_choices is not None:
            args.medusa_choices = ast.literal_eval(args.medusa_choices)
            assert args.temperature == 1.0, "Medusa should use temperature == 1.0"
            assert args.num_beams == 1, "Medusa should use num_beams == 1"
            runner_kwargs.update(medusa_choices=args.medusa_choices)
        if args.eagle_choices is not None or args.eagle_posterior_threshold is not None or args.eagle_use_dynamic_tree:
            assert args.num_beams == 1, "Eagle should use num_beams == 1"
            if args.eagle_choices is not None and not args.eagle_use_dynamic_tree:
                args.eagle_choices = ast.literal_eval(args.eagle_choices)
                runner_kwargs.update(eagle_choices=args.eagle_choices)
            if args.eagle_posterior_threshold is not None:
                runner_kwargs.update(
                    eagle_posterior_threshold=args.eagle_posterior_threshold)
            if args.eagle_use_dynamic_tree:
                runner_kwargs.update(
                    eagle_use_dynamic_tree=args.eagle_use_dynamic_tree)
                assert args.eagle_dynamic_tree_max_top_k is not None and args.eagle_dynamic_tree_max_top_k > 0
                runner_kwargs.update(eagle_dynamic_tree_max_top_k=args.
                                     eagle_dynamic_tree_max_top_k)
        if args.lookahead_config is not None:
            args.lookahead_config = ast.literal_eval(args.lookahead_config)
            assert len(
                args.lookahead_config
            ) == 3, "Lookahead needs [max_window_size, max_ngram_size, max_verification_set_size]"
            runner_kwargs.update(lookahead_config=args.lookahead_config)
        if args.ngram_config is not None:
            assert args.kv_cache_enable_block_reuse, "`--kv_cache_enable_block_reuse` must be specified in speculative decoding."
            assert not args.use_py_session, "`--use_py_session` is not supported in Speculative decoding."
            assert args.num_beams == 1, "`--num_beams>1` is not supported in Speculative decoding."
            max_draft_len, _, target_device_list = ast.literal_eval(
                args.ngram_config)
            args.max_output_len = output_len  # Specialization for NGram
            runner_kwargs.update(is_orchestrator_mode=True,
                                 device_ids=target_device_list,
                                 max_input_len=test_token_num + max_draft_len +
                                 output_len)

        runner = runner_cls.from_dir(**runner_kwargs)
        assert not (args.eval_ppl and not runner.gather_context_logits), \
            "PPL evaluation requires engine built with gather_context_logits enabled"

        datapoint = dataset[0:1]
        output, *_ = eval_trt_llm(datapoint,
                                  eval_task=args.eval_task,
                                  eval_ppl=args.eval_ppl,
                                  add_special_tokens=args.add_special_tokens,
                                  min_input_length=args.min_input_length,
                                  runner=runner)
        if runtime_rank == 0 and args.eval_task != "eval_context_ppl":
            logger.info(
                "---------------------------------------------------------")
            logger.info("TensorRT LLM Generated: ")
            logger.info(f" Input: {datapoint[dataset_input_key]}")
            logger.info(f"\n Reference: {datapoint[dataset_output_key]}")
            logger.info(f"\n Output: {output}")
            logger.info(
                "---------------------------------------------------------")

        ite_count = 0
        data_point_idx = 0
        total_output_token_count_trt_llm = 0  # only valid for runtime_rank == 0
        while (data_point_idx < len(dataset)) and (ite_count < args.max_ite):
            if runtime_rank == 0:
                logger.debug(
                    f"run data_point {data_point_idx} ~ {data_point_idx + max_batch_size}"
                )
            datapoint = dataset[data_point_idx:(data_point_idx +
                                                max_batch_size)]

            profiler.start('tensorrt_llm')
            output_tensorrt_llm, output_ids_trt_llm, curr_ppls_trt_llm, lengths_info = eval_trt_llm(
                datapoint,
                eval_task=args.eval_task,
                eval_ppl=args.eval_ppl,
                add_special_tokens=args.add_special_tokens,
                min_input_length=args.min_input_length,
                runner=runner)
            profiler.stop('tensorrt_llm')

            empty_batch = runtime_rank == 0 and len(output_tensorrt_llm) == 0
            empty_batch = mpi_broadcast(empty_batch, 0)
            if empty_batch:
                # No valid samples in the current batch, skip this iteration
                data_point_idx += max_batch_size
                continue

            if runtime_rank == 0:
                input_lengths = lengths_info['input_lengths']
                seq_lengths = lengths_info['seq_lengths']
                output_token_count_trt_llm = sum(
                    beam_len - input_lengths[batch_idx]
                    for batch_idx, beam_lens in enumerate(seq_lengths)
                    for beam_len in beam_lens)
                total_output_token_count_trt_llm += output_token_count_trt_llm
                for batch_idx, output_beams in enumerate(output_tensorrt_llm):
                    reference = datapoint[dataset_output_key][batch_idx]
                    for beam_idx, output_beam in enumerate(output_beams):
                        metric_tensorrt_llm[beam_idx].add_batch(
                            predictions=[output_beam], references=[reference])
                        if args.eval_ppl:
                            ppls_trt_llm[beam_idx].append(
                                curr_ppls_trt_llm[batch_idx][beam_idx])
                if output_dir is not None:
                    for i in range(len(output_tensorrt_llm[0])):
                        for beam_idx in range(num_sequences):
                            with (output_dir / 'trtllm.out').open('a') as f:
                                f.write(
                                    f'[{data_point_idx + i}] [Beam {beam_idx}] {output_tensorrt_llm[beam_idx][i]}\n'
                                )

                logger.debug('-' * 100)
                logger.debug(f"Input: {datapoint[dataset_input_key]}")
                logger.debug(f'TensorRT LLM Output: {output_tensorrt_llm}')
                logger.debug(f"Reference: {datapoint[dataset_output_key]}")

            data_point_idx += max_batch_size
            ite_count += 1
        del runner

    if test_hf and runtime_rank == 0:
        profiler.start('load HF model')
        dtype_alias_mapping = {
            'fp32': 'float32',
            'fp16': 'float16',
            'bf16': 'bfloat16'
        }
        args.hf_data_type = dtype_alias_mapping.get(args.hf_data_type,
                                                    args.hf_data_type)
        if 'GLM' in model_name and model_version == 'glm':
            auto_model_cls = AutoModelForSeq2SeqLM
        elif 'GLM' in model_name and model_version == 'chatglm':
            auto_model_cls = AutoModel
        else:
            auto_model_cls = AutoModelForCausalLM
        # TODO: args.hf_device_map_auto is not being correctly set
        # remove in future version
        if model_name == 'DeepseekV2ForCausalLM':
            args.hf_device_map_auto = True
        model = auto_model_cls.from_pretrained(
            args.hf_model_dir,
            trust_remote_code=True,
            dtype=str_dtype_to_torch(args.hf_data_type),
            device_map='auto' if args.hf_device_map_auto else None)
        try:
            model.to_bettertransformer()
        except Exception as e:
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

        datapoint = dataset[0:1]
        output, *_ = eval_hf(datapoint,
                             eval_task=args.eval_task,
                             eval_ppl=args.eval_ppl,
                             add_special_tokens=args.add_special_tokens,
                             min_input_length=args.min_input_length)
        if runtime_rank == 0 and args.eval_task != "eval_context_ppl":
            logger.info(
                "---------------------------------------------------------")
            logger.info("HF Generated: ")
            logger.info(f" Input: {datapoint[dataset_input_key]}")
            logger.info(f"\n Reference: {datapoint[dataset_output_key]}")
            logger.info(f"\n Output: {output}")
            logger.info(
                "---------------------------------------------------------")

        ite_count = 0
        data_point_idx = 0
        total_output_token_count_hf = 0  # only valid for runtime_rank == 0
        while (data_point_idx < len(dataset)) and (ite_count < args.max_ite):
            if runtime_rank == 0:
                logger.debug(
                    f"run data_point {data_point_idx} ~ {data_point_idx + max_batch_size}"
                )
            datapoint = dataset[data_point_idx:(data_point_idx +
                                                max_batch_size)]

            profiler.start('hf')
            output_hf, token_list, curr_ppls_hf = eval_hf(
                datapoint,
                eval_task=args.eval_task,
                eval_ppl=args.eval_ppl,
                add_special_tokens=args.add_special_tokens,
                min_input_length=args.min_input_length)
            profiler.stop('hf')

            # HF model runs on rank 0 only
            empty_batch = len(output_hf) == 0
            if empty_batch:
                # No valid samples in the current batch, skip this iteration
                data_point_idx += max_batch_size
                continue

            if runtime_rank == 0:
                seq_lengths = [len(tokens) for tokens in token_list]
                total_output_token_count_hf += sum(seq_lengths)
                for beam_idx in range(num_sequences):
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
                    for i in range(len(output_hf[0])):
                        for beam_idx in range(num_sequences):
                            with (output_dir / 'hf.out').open('a') as f:
                                f.write(
                                    f'[{data_point_idx + i}] [Beam {beam_idx}] {output_hf[beam_idx][i]}\n'
                                )

                logger.debug('-' * 100)
                logger.debug(f"Input: {datapoint[dataset_input_key]}")
                logger.debug(f'HF Output: {output_hf}')
                logger.debug(f"Reference: {datapoint[dataset_output_key]}")

            data_point_idx += max_batch_size
            ite_count += 1
        del model

    if runtime_rank == 0 and args.max_ite > 0:
        if test_trt_llm:
            np.random.seed(0)  # rouge score use sampling to compute the score
            logger.info(
                f'TensorRT LLM (total latency: {profiler.elapsed_time_in_sec("tensorrt_llm")} sec)'
            )

            logger.info(
                f'TensorRT LLM (total output tokens: {total_output_token_count_trt_llm})'
            )
            logger.info(
                f'TensorRT LLM (tokens per second: {total_output_token_count_trt_llm / profiler.elapsed_time_in_sec("tensorrt_llm")})'
            )
            for beam_idx in range(num_sequences):
                logger.info(f"TensorRT LLM beam {beam_idx} result")
                if args.eval_task != "eval_context_ppl":
                    if args.estimate_accuracy_std_dev:
                        computed_metrics_tensorrt_llm = metric_tensorrt_llm[
                            beam_idx].compute(use_aggregator=False)
                        computed_std_dev_tensorrt_llm = {
                            key: np.std(scores)
                            for key, scores in
                            computed_metrics_tensorrt_llm.items()
                        }
                        computed_metrics_tensorrt_llm = {
                            key: np.mean(scores)
                            for key, scores in
                            computed_metrics_tensorrt_llm.items()
                        }
                        for key in computed_metrics_tensorrt_llm.keys():
                            logger.info(
                                f"  {key}: {computed_metrics_tensorrt_llm[key]*100} ({computed_std_dev_tensorrt_llm[key]*100})"
                            )
                    else:
                        computed_metrics_tensorrt_llm = metric_tensorrt_llm[
                            beam_idx].compute()
                        for key in computed_metrics_tensorrt_llm.keys():
                            logger.info(
                                f"  {key}: {computed_metrics_tensorrt_llm[key]*100}"
                            )
                    if args.check_accuracy and beam_idx == 0:
                        rouge1 = computed_metrics_tensorrt_llm['rouge1'] * 100
                        assert rouge1 > args.tensorrt_llm_rouge1_threshold, f"[FAILED] rouge1 ({rouge1}) is smaller than threshold ({args.tensorrt_llm_rouge1_threshold})."
                if args.eval_ppl:
                    logger.info(
                        f"  Per-token perplexity: {np.mean(ppls_trt_llm[beam_idx])}"
                    )
                    if args.check_accuracy and beam_idx == 0:
                        avg_ppl = np.mean(ppls_trt_llm[beam_idx])
                        assert avg_ppl < args.tensorrt_llm_ppl_threshold, f"[FAILED] average PPL ({avg_ppl}) is larger than threshold ({args.tensorrt_llm_ppl_threshold})."
        if test_hf:
            np.random.seed(0)  # rouge score use sampling to compute the score
            logger.info(
                f'Hugging Face (total latency: {profiler.elapsed_time_in_sec("hf")} sec)'
            )
            logger.info(
                f'Hugging Face (total output tokens: {total_output_token_count_hf})'
            )
            logger.info(
                f'Hugging Face (tokens per second: {total_output_token_count_hf / profiler.elapsed_time_in_sec("hf")})'
            )

            for beam_idx in range(num_sequences):
                logger.info(f"HF beam {beam_idx} result")
                computed_metrics_hf = metric_hf[beam_idx].compute()
                if args.eval_task != "eval_context_ppl":
                    for key in computed_metrics_hf.keys():
                        logger.info(f'  {key}: {computed_metrics_hf[key]*100}')
                if args.eval_ppl and args.batch_size == 1:
                    logger.info(
                        f"  Per-token perplexity: {np.mean(ppls_hf[beam_idx])}")


if __name__ == '__main__':
    # see `add_common_args` for extended list of arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_hf', action='store_true')
    parser.add_argument('--test_trt_llm', action='store_true')
    parser.add_argument('--eval_task',
                        type=str,
                        default='summarize',
                        choices=[
                            'summarize', 'summarize_long', 'code_completion',
                            'eval_context_ppl'
                        ])
    parser.add_argument('--check_accuracy', action='store_true')
    parser.add_argument('--estimate_accuracy_std_dev', action='store_true')
    parser.add_argument('--tensorrt_llm_rouge1_threshold',
                        type=float,
                        default=15.0)
    parser.add_argument('--eval_ppl', action='store_true')
    parser.add_argument('--tensorrt_llm_ppl_threshold',
                        type=float,
                        default=15.0)
    parser.add_argument(
        '--dataset_dir',
        type=str,
        default=None,
        help="The local directory of the dataset for evaluation; "
        "will download the dataset from huggingface hub if not specified.")
    parser.add_argument(
        '--dataset_cache_dir',
        type=str,
        default=None,
        help="The local cache directory for dataset; "
        "will use `~/.cache/huggingface/datasets` if not specified.")
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--max_ite', type=int, default=20)
    parser.add_argument('--output_len', type=int, default=100)
    parser.add_argument('--max_input_length', type=int, default=923)
    parser.add_argument(
        '--min_input_length',
        type=int,
        default=0,
        help='skip the sentences which are shorter than min_input_length.')
    parser.add_argument(
        '--output_dir',
        type=str,
        default=None,
        help="Directory where to save output sentences. 'trtllm.out' for "
        "TensorRT LLM outputs, and 'hf.out' for HF outputs.  If None, do not "
        "save outputs.")
    parser.add_argument(
        '--rouge_dir',
        default=None,
        type=str,
        help=
        "evaluate.load('rouge') will attempt to pull rouge package from HF. Use cached rouge can avoid network outage of host or HF."
    )
    parser.add_argument("--use_gpu_direct_storage",
                        default=False,
                        action="store_true",
                        help="Use GPUDirect Storage (GDS) to load the engine")
    parser = add_common_args(parser)
    args = parser.parse_args()

    main(args)
