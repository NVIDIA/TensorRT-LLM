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
from transformers import AutoProcessor
from utils import add_common_args

import tensorrt_llm
import tensorrt_llm.profiler as profiler
from tensorrt_llm import logger
from tensorrt_llm.runtime import MultimodalModelRunner

SUPPORTED_MODEL_TYPES = {
    'blip2': 'Blip2ForConditionalGeneration',
    'fuyu': 'FuyuForCausalLM',
    'kosmos-2': 'Kosmos2ForConditionalGeneration',
    'llava': 'LlavaForConditionalGeneration',
    'llava_next': 'LlavaNextForConditionalGeneration',
    'llava_onevision': 'LlavaOnevisionForConditionalGeneration',
    'phi-3-vision': 'AutoModelForCausalLM',
    'qwen2_vl': 'Qwen2VLForConditionalGeneration',  # not tested for TRT-LLM yet
    'mllama': 'MllamaForConditionalGeneration',
    'vila': None,
    'cogvlm': None,  # not tested for TRT-LLM yet
    'neva': None,  # not tested for TRT-LLM yet
    'internvl': None,
}
EVAL_TASKS = ['lmms-lab/ai2d', 'lmms-lab/VQAv2', 'lmms-lab/MME']


def parse_arguments(args=None):
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
                        choices=EVAL_TASKS,
                        default='lmms-lab/VQAv2')
    parser.add_argument('--model_type',
                        type=str,
                        default=None,
                        choices=SUPPORTED_MODEL_TYPES.keys())
    parser.add_argument(
        '--accuracy_threshold',
        type=float,
        default=None,
        help=
        'used to check the accuracy of test_trtllm. Should be between 0 and 100.'
    )
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
    return parser.parse_args(args=args)


def load_dataset(args) -> datasets.Dataset:
    split_name = 'validation' if 'VQAv2' in args.eval_task else 'test'

    if args.dataset_dir is not None and os.path.exists(
            os.path.join(args.dataset_dir, "dataset_info.json")):
        logger.info(f"load dataset by load_from_disk from {args.dataset_dir}")
        dataset = datasets.load_from_disk(args.dataset_dir)

    else:
        logger.info(
            f"load dataset by load_dataset from {args.dataset_dir or args.eval_task}"
        )
        dataset = datasets.load_dataset(
            args.dataset_dir or args.eval_task,
            cache_dir=args.dataset_cache_dir,
            split=split_name,
            storage_options={
                'client_kwargs': {
                    'timeout': aiohttp.ClientTimeout(total=3600)
                }
            },
            trust_remote_code=True,
        )
    return dataset


def load_hf_model(args):
    if SUPPORTED_MODEL_TYPES[args.model_type] is None:
        raise ValueError(f"Unsupported HF model_type: {args.model_type}")
    profiler.start('load HF model')
    model_class = getattr(__import__('transformers'),
                          SUPPORTED_MODEL_TYPES[args.model_type])
    hf_model = model_class.from_pretrained(args.hf_model_dir,
                                           dtype=torch.float16,
                                           device_map="cuda:0",
                                           trust_remote_code=True)
    profiler.stop('load HF model')

    logger.info(
        f'Load HF model takes: {profiler.elapsed_time_in_sec("load HF model")} sec'
    )
    return hf_model


def load_trtllm_model(args):
    profiler.start('load TensorRT LLM model')
    trtllm_model = MultimodalModelRunner(args)
    profiler.stop('load TensorRT LLM model')
    logger.info(
        f'Load TensorRT LLM model takes: {profiler.elapsed_time_in_sec("load TensorRT LLM model")} sec'
    )
    return trtllm_model


def prepare_prompts(task, data, model_type, processor) -> str:
    prompts = None
    question = data['question']
    if question[-1] != '?':
        question += '?'

    if task == 'lmms-lab/ai2d':
        for j, option in enumerate(data['options']):
            question += f" ({j}) {option}"

    if model_type in ['blip2', 'neva']:
        prompts = f"Question: {question} Answer: "
    elif model_type == 'fuyu':
        prompts = f"Answer the following {task} question based on the image: {question}"
    elif model_type == 'kosmos-2':
        prompts = f"<grounding> Question: {question} Answer: "
    elif model_type == 'cogvlm':
        prompts = f"A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: {question} ASSISTANT:"
    elif model_type in ['llava', 'llava_next', 'llava_onevision']:
        conversation = [
            {
                "role":
                "user",
                "content": [
                    {
                        "type": "image"
                    },
                    {
                        "type": "text",
                        "text": question
                    },
                ],
            },
        ]
        prompts = processor.apply_chat_template(conversation,
                                                add_generation_prompt=True)
    elif model_type in ['vila', 'internvl']:
        prompts = f"<image>\n{question}"
    elif model_type == 'phi-3-vision':
        messages = [
            {
                "role": "user",
                "content": f"<|image_1|>\n{question}"
            },
        ]
        prompts = processor.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
    elif model_type == 'qwen2_vl':
        conversation = [{
            "role":
            "user",
            "content": [{
                "type": "image",
            }, {
                "type": "text",
                "text": question
            }]
        }]
        prompts = processor.apply_chat_template(conversation,
                                                add_generation_prompt=True)
    elif model_type == 'mllama':
        prompts = processor.apply_chat_template(images=data['image'],
                                                text=question + "; answer: ")
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")

    return prompts


def eval(output, task, data) -> bool:
    output = output.strip().lower()
    if task == 'lmms-lab/VQAv2':
        return any(answer['answer'] in output for answer in data['answers'])
    else:
        return data['answer'].lower() in output


os.environ["TOKENIZERS_PARALLELISM"] = "false"
args = parse_arguments()
if args.model_type not in SUPPORTED_MODEL_TYPES:
    raise ValueError(f"Unsupported model_type: {args.model_type}")

logger.set_level(args.log_level)

runtime_rank = tensorrt_llm.mpi_rank()
dataset = load_dataset(args)
hf_model = load_hf_model(args) if args.test_hf else None
trtllm_model = load_trtllm_model(args) if args.test_trtllm else None
if SUPPORTED_MODEL_TYPES[args.model_type] is None:
    hf_processor = None
else:
    hf_processor = AutoProcessor.from_pretrained(args.hf_model_dir,
                                                 trust_remote_code=True)
hf_correct = trtllm_correct = 0

if args.model_type == 'mllama':
    from tensorrt_llm.runtime.processor_wrapper import MllamaProcessorWrapper
    hf_processor = MllamaProcessorWrapper(hf_processor, logger)

torch.random.manual_seed(0)
profiler.start('evaluation')
if args.test_trtllm or args.test_hf:
    for i in range(args.max_ite):
        logger.debug(f"Ite: {i:3d}")
        data = dataset[i]
        if i > len(dataset):
            break
        prompts = prepare_prompts(args.eval_task, data, args.model_type,
                                  hf_processor)
        image = data['image']

        if args.test_hf:
            assert hf_model is not None, f"Unsupported HF model_type: {args.model_type}"
            profiler.start('hf')
            inputs = hf_processor(
                images=image,
                text=prompts,
                return_tensors="pt",
            ).to(hf_model.device,
                 torch.float16)  # add torch.float16 for llava-onevision
            input_length = inputs.input_ids.shape[-1]
            hf_output = hf_model.generate(**inputs,
                                          max_new_tokens=args.max_new_tokens)
            hf_result = (hf_processor.batch_decode(
                hf_output, skip_special_tokens=True)[0] if args.model_type in [
                    'blip2'
                ] else hf_processor.decode(hf_output[0][input_length:],
                                           skip_special_tokens=True))
            hf_correct += eval(hf_result, args.eval_task, data)
            profiler.stop('hf')

        if args.test_trtllm:
            profiler.start('tensorrt_llm')
            _, output_text = trtllm_model.run(
                input_text=prompts,
                input_image=image,
                input_audio=None,
                max_new_tokens=args.max_new_tokens)
            if runtime_rank == 0:
                trtllm_result = output_text[0][0]
                trtllm_correct += eval(trtllm_result, args.eval_task, data)
            profiler.stop('tensorrt_llm')

        if runtime_rank == 0:
            if args.eval_task == 'lmms-lab/VQAv2':
                answer = data['answers']
            else:
                answer = data['answer']
            logger.debug(f"prompts: {prompts}")
            logger.debug(f"reference answer: {answer}")
            if args.test_hf:
                logger.debug(f"HF's answer: {hf_result}")
            if args.test_trtllm:
                logger.debug(f"TRT-LLM's answer: {trtllm_result}")

    if runtime_rank == 0:
        logger.info(f"total iterations: {args.max_ite}")
        if args.test_hf:
            logger.info(
                f"HF's accuracy: {100 * hf_correct / args.max_ite:4.2f}%")
        if args.test_trtllm:
            logger.info(
                f"TRT-LLM's accuracy: {100 * trtllm_correct / args.max_ite:4.2f}%"
            )
    # check if the accuracy is above the threshold
    if args.accuracy_threshold is not None and args.test_trtllm:
        assert trtllm_correct / args.max_ite >= args.accuracy_threshold / 100, \
            f"TRT-LLM's accuracy is below the threshold: {args.accuracy_threshold}%."
else:
    logger.info("Neither enable test_trtllm nor enable test_hf")

profiler.stop('evaluation')
logger.info(
    f'Evaluation takes: {profiler.elapsed_time_in_sec("evaluation")} sec')
