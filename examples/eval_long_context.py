# MIT License

# Copyright (c) 2023 OpenBMB

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

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

# reference: https://github.com/OpenBMB/InfiniteBench/blob/main/src/eval_yarn_mistral.py

import argparse
import ast
import json
from pathlib import Path

import torch
from infinitebench.compute_scores import compute_scores
from infinitebench.eval_utils import (DATA_NAME_TO_MAX_NEW_TOKENS,
                                      create_prompt, dump_jsonl, get_answer,
                                      load_data)
from utils import (DEFAULT_HF_MODEL_DIRS, DEFAULT_PROMPT_TEMPLATES,
                   add_common_args, load_tokenizer, read_model_name)

import tensorrt_llm
import tensorrt_llm.profiler as profiler
from tensorrt_llm.logger import logger
from tensorrt_llm.runtime import PYTHON_BINDINGS, ModelRunner

if PYTHON_BINDINGS:
    from tensorrt_llm.runtime import ModelRunnerCpp

MAX_POSITION_ID = 128 * 1024  # Determined by the model
TRUNCATE_LEN = 128 * 1024


def parse_arguments(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--max_input_length', type=int, default=923)
    parser.add_argument('--output_log_probs_npy',
                        type=str,
                        help='Numpy file where the log_probs are stored',
                        default=None)

    parser.add_argument('--output_cum_log_probs_npy',
                        type=str,
                        help='Numpy file where the cum_log_probs are stored',
                        default=None)

    parser.add_argument(
        "--task",
        type=str,
        choices=['passkey', 'kv_retrieval'],
        required=True,
        help=
        "Which task to use. Note that \"all\" can only be used in `compute_scores.py`.",  # noqa
    )
    parser.add_argument('--data_dir',
                        type=str,
                        default='./',
                        help="The directory of data.")
    parser.add_argument("--output_dir",
                        type=str,
                        default=None,
                        help="Where to dump the prediction results.")  # noqa
    parser.add_argument(
        "--start_idx",
        type=int,
        default=0,
        help=
        "The index of the first example to infer on. This is used if you want to evaluate on a (contiguous) subset of the data."
    )  # noqa
    parser.add_argument(
        "--stop_idx",
        type=int,
        help=
        "The index of the last example to infer on. This is used if you want to evaluate on a (contiguous) subset of the data. Defaults to the length of dataset."
    )  # noqa
    parser.add_argument('--tensorrt_llm_accuracy_threshold',
                        type=float,
                        default=99)
    parser = add_common_args(parser)

    return parser.parse_args(args=args)


def parse_input(tokenizer,
                input_text=None,
                prompt_template=None,
                add_special_tokens=True,
                max_input_length=923,
                pad_id=None,
                num_prepend_vtokens=[],
                model_name=None,
                model_version=None):
    if pad_id is None:
        pad_id = tokenizer.pad_token_id

    batch_input_ids = []
    for curr_text in input_text:
        if prompt_template is not None:
            curr_text = prompt_template.format(input_text=curr_text)
        input_ids = tokenizer.encode(curr_text,
                                     add_special_tokens=add_special_tokens,
                                     truncation=True,
                                     max_length=max_input_length)
        batch_input_ids.append(input_ids)

    if num_prepend_vtokens:
        assert len(num_prepend_vtokens) == len(batch_input_ids)
        base_vocab_size = tokenizer.vocab_size - len(
            tokenizer.special_tokens_map.get('additional_special_tokens', []))
        for i, length in enumerate(num_prepend_vtokens):
            batch_input_ids[i] = list(
                range(base_vocab_size,
                      base_vocab_size + length)) + batch_input_ids[i]

    if 'GLM' in model_name and model_version == 'glm':
        for ids in batch_input_ids:
            ids.append(tokenizer.sop_token_id)

    batch_input_ids = [
        torch.tensor(x, dtype=torch.int32) for x in batch_input_ids
    ]
    return batch_input_ids


def main(args):
    # model_name = "yarn-mistral"
    runtime_rank = tensorrt_llm.mpi_rank()
    logger.set_level(args.log_level)

    print(json.dumps(vars(args), indent=4))
    data_name = args.task

    # Model
    max_tokens = DATA_NAME_TO_MAX_NEW_TOKENS[data_name]

    model_name, model_version = read_model_name(args.engine_dir)
    if args.tokenizer_dir is None:
        logger.warning(
            "tokenizer_dir is not specified. Try to infer from model_name, but this may be incorrect."
        )
        args.tokenizer_dir = DEFAULT_HF_MODEL_DIRS[model_name]

    tokenizer, pad_id, end_id = load_tokenizer(
        tokenizer_dir=args.tokenizer_dir,
        vocab_file=args.vocab_file,
        model_name=model_name,
        model_version=model_version,
        tokenizer_type=args.tokenizer_type,
    )

    if not PYTHON_BINDINGS and not args.use_py_session:
        logger.warning(
            "Python bindings of C++ session is unavailable, fallback to Python session."
        )
        args.use_py_session = True
    if args.debug_mode and not args.use_py_session:
        logger.warning(
            "Debug mode is not supported in C++ session for now, fallback to Python session."
        )
        args.use_py_session = True
    runner_cls = ModelRunner if args.use_py_session else ModelRunnerCpp
    runner_kwargs = dict(
        engine_dir=args.engine_dir,
        lora_dir=args.lora_dir,
        rank=runtime_rank,
        debug_mode=args.debug_mode,
        lora_ckpt_source=args.lora_ckpt_source,
        gpu_weights_percent=args.gpu_weights_percent,
    )
    if args.medusa_choices is not None:
        args.medusa_choices = ast.literal_eval(args.medusa_choices)
        assert args.temperature == 1.0, "Medusa should use temperature == 1.0"
        assert args.num_beams == 1, "Medusa should use num_beams == 1"
        runner_kwargs.update(medusa_choices=args.medusa_choices)
    if not args.use_py_session:
        runner_kwargs.update(
            max_batch_size=args.batch_size,
            max_input_len=args.max_input_length,
            max_output_len=max_tokens,
            max_beam_width=args.num_beams,
            max_attention_window_size=args.max_attention_window_size,
            sink_token_length=args.sink_token_length,
            max_tokens_in_paged_kv_cache=args.max_tokens_in_paged_kv_cache,
            kv_cache_enable_block_reuse=args.kv_cache_enable_block_reuse,
            kv_cache_free_gpu_memory_fraction=args.
            kv_cache_free_gpu_memory_fraction,
            enable_chunked_context=args.enable_chunked_context,
        )
    runner = runner_cls.from_dir(**runner_kwargs)

    # Data
    examples = load_data(data_name, data_dir=args.data_dir)
    if args.stop_idx is None:
        args.stop_idx = len(examples)

    output_path = None
    if runtime_rank == 0:
        if args.output_dir is not None:
            result_dir = Path(args.output_dir, model_name)
            result_dir.mkdir(exist_ok=True, parents=True)

            if args.stop_idx is None:
                output_path = (result_dir / f"preds_{data_name}.jsonl")
            else:
                output_path = (
                    result_dir /
                    f"preds_{data_name}_{args.start_idx}-{args.stop_idx}.jsonl"  # noqa
                )

    prompt_template = None
    if args.use_prompt_template and model_name in DEFAULT_PROMPT_TEMPLATES:
        prompt_template = DEFAULT_PROMPT_TEMPLATES[model_name]

    if runtime_rank == 0:
        preds = []
        logger.info("==== Evaluation ====")
        logger.info(f"# examples: {len(examples)}")
        logger.info(f"Start index: {args.start_idx}")
        logger.info(f"Stop index: {args.stop_idx}")
        logger.info(f"Max tokens: {max_tokens}")
    assert args.batch_size == 1
    profiler.start('Evaluation')
    for i in range(args.start_idx, args.stop_idx):
        eg = examples[i]
        input_text = [create_prompt(eg, data_name, args.data_dir)]
        batch_input_ids = parse_input(
            tokenizer=tokenizer,
            input_text=input_text,
            prompt_template=prompt_template,
            add_special_tokens=args.add_special_tokens,
            max_input_length=args.max_input_length,
            pad_id=pad_id,
            num_prepend_vtokens=args.num_prepend_vtokens,
            model_name=model_name,
            model_version=model_version)
        input_lengths = [x.size(0) for x in batch_input_ids]

        if runtime_rank == 0:
            logger.debug(f"====== Example {i} ======")
            logger.debug(f"input_lengths: {input_lengths}")
            logger.debug(f"input_text: {input_text}")
            logger.debug(f"answer: {get_answer(eg, data_name)}")
        outputs = runner.generate(
            batch_input_ids,
            max_new_tokens=max_tokens,
            max_attention_window_size=args.max_attention_window_size,
            sink_token_length=args.sink_token_length,
            end_id=end_id,
            pad_id=pad_id,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            num_beams=args.num_beams,
            length_penalty=args.length_penalty,
            early_stopping=args.early_stopping,
            beam_width_array=args.beam_width_array,
            repetition_penalty=args.repetition_penalty,
            presence_penalty=args.presence_penalty,
            frequency_penalty=args.frequency_penalty,
            # stop_words_list=stop_words_list,
            # bad_words_list=bad_words_list,
            output_cum_log_probs=(args.output_cum_log_probs_npy != None),
            output_log_probs=(args.output_log_probs_npy != None),
            lora_uids=args.lora_task_uids,
            prompt_table=args.prompt_table_path,
            prompt_tasks=args.prompt_tasks,
            streaming=args.streaming,
            output_sequence_lengths=True,
            return_dict=True,
            medusa_choices=args.medusa_choices)
        torch.cuda.synchronize()
        if runtime_rank == 0:
            output_ids = outputs['output_ids']
            output_beams_list = [
                tokenizer.batch_decode(output_ids[batch_idx, :,
                                                  input_lengths[batch_idx]:],
                                       skip_special_tokens=True)
                for batch_idx in range(args.batch_size)
            ]

            logger.debug(f"preds: {output_beams_list[0]}")
            preds.append({
                "id": i,
                "prediction": output_beams_list[0][0],
                "ground_truth": get_answer(eg, data_name),
                "input_lengths": input_lengths,
            })
            if output_path is not None:
                dump_jsonl(preds, output_path)
    profiler.stop('Evaluation')

    if runtime_rank == 0:
        logger.info(
            f'Evaluation takes: {profiler.elapsed_time_in_sec("Evaluation")} sec.'
        )
        logger.info("Compute the score")
        acc = compute_scores(preds, args.task) * 100
        logger.info(f"{args.task} accuracy: {acc:.2f} ({len(preds)})")

        if args.tensorrt_llm_accuracy_threshold is not None:
            assert acc >= args.tensorrt_llm_accuracy_threshold, f"acc ({acc}) < tensorrt_llm_accuracy_threshold ({args.tensorrt_llm_accuracy_threshold})"


if __name__ == "__main__":
    args = parse_arguments()
    main(args)
