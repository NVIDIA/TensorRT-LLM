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
import os

import aiohttp
import datasets
import torch
from transformers import AutoProcessor, MllamaForConditionalGeneration
from utils import add_common_args

import tensorrt_llm.profiler as profiler
from tensorrt_llm import logger
from tensorrt_llm.runtime import MultimodalModelRunner


def prepare_prompts(task, data):
    if task == 'lmms-lab/ai2d':
        prompts = f"<|image|><|begin_of_text|> {data['question']}"
        if prompts[-1] != '?':
            prompts += '?'
        for j, option in enumerate(data['options']):
            prompts += f" ({j}) {option}"
        prompts += "; answer: "

    return prompts


os.environ["TOKENIZERS_PARALLELISM"] = "false"
parser = argparse.ArgumentParser()
parser = add_common_args(parser)
parser.add_argument('--test_trtllm',
                    action='store_true',
                    default=None,
                    help="Evaluate the TensorRT-LLM.")
parser.add_argument('--test_hf',
                    action='store_true',
                    default=None,
                    help="Evaluate the Huggingface.")
parser.add_argument('--max_ite', type=int, default=20)
parser.add_argument('--eval_task',
                    type=str,
                    default='lmms-lab/ai2d',
                    choices=[
                        'lmms-lab/ai2d',
                    ])
parser.add_argument(
    '--accuracy_threshold',
    type=float,
    default=None,
    help=
    'used to check the accuracy of test_trtllm. Should be between 0 and 100.')
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
args = parser.parse_args()

logger.set_level(args.log_level)

dataset = datasets.load_dataset(args.dataset_dir,
                                storage_options={
                                    'client_kwargs': {
                                        'timeout':
                                        aiohttp.ClientTimeout(total=3600)
                                    }
                                },
                                cache_dir=args.dataset_cache_dir,
                                split='test')

processor = AutoProcessor.from_pretrained(args.hf_model_dir)

hf_model = None
if args.test_hf:
    profiler.start('load HF model')
    hf_model = MllamaForConditionalGeneration.from_pretrained(
        args.hf_model_dir,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    profiler.stop('load HF model')
    logger.info(
        f'Load HF model takes: {profiler.elapsed_time_in_sec("load HF model")} sec'
    )

trtllm_model = None
if args.test_trtllm:
    profiler.start('load TensorRT-LLM model')
    trtllm_model = MultimodalModelRunner(args)
    profiler.stop('load TensorRT-LLM model')
    logger.info(
        f'Load TensorRT-LLM model takes: {profiler.elapsed_time_in_sec("load TensorRT-LLM model")} sec'
    )

if trtllm_model or hf_model:
    trtllm_correct = 0 if trtllm_model else None
    hf_correct = 0 if hf_model else None
    for i in range(args.max_ite):
        logger.debug(f"Ite: {i:3d}")
        data = dataset[i]

        prompts = prepare_prompts(args.eval_task, data)
        answer = data['answer']
        image = data['image']

        hf_result = None
        if hf_model:
            profiler.start('hf')
            inputs = processor(
                image,
                prompts,
                return_tensors="pt",
            ).to(hf_model.device)
            input_length = inputs.input_ids.shape[-1]

            hf_output = hf_model.generate(**inputs, max_new_tokens=1)
            hf_result = processor.decode(hf_output[0][input_length:])
            if answer == hf_result:
                hf_correct += 1
            profiler.stop('hf')

        trtllm_result = None
        if trtllm_model:
            profiler.start('tensorrt_llm')
            input_text, output_text = trtllm_model.run(prompts,
                                                       image,
                                                       max_new_tokens=1)
            trtllm_result = output_text[0][0]
            if answer == trtllm_result:
                trtllm_correct += 1
            profiler.stop('tensorrt_llm')

        logger.debug(f"prompts: {prompts}")
        logger.debug(f"reference answer: {answer}")
        if hf_result:
            logger.debug(f"HF's answer: {hf_result}")
        if trtllm_result:
            logger.debug(f"TRT-LLM's answer: {trtllm_result}")

    logger.info(f"total iterations: {args.max_ite}")
    if hf_correct is not None:
        logger.info(f"HF's accuracy: {100 * hf_correct / args.max_ite:4.2f}%")
    if trtllm_correct is not None:
        logger.info(
            f"TRT-LLM's accuracy: {100 * trtllm_correct / args.max_ite:4.2f}%")
else:
    logger.info("Neither enable test_trtllm nor enable test_hf")
