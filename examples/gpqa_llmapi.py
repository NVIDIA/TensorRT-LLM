# MIT License
#
# Copyright (c) 2020 Dan Hendrycks
# Copyright (c) 2023 Deep Cognition and Language Research (DeCLaRe) Lab
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# Not a contribution
# Changes made by NVIDIA CORPORATION & AFFILIATES or otherwise documented as
# NVIDIA-proprietary are not a contribution and subject to the following terms and conditions:
# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
"""A duplication of examples/mmlu_llmapi.py and tensorrt_llm/bench/benchmark/utils/asynchronous.py, but targeting GPQA task.
The duplication is used to get a quick GPQA score in the CI test.
TODO: Should be merged with examples/mmlu_llmapi.py
Example usage:
    python gpqa.py --hf_model_dir <HF model path> --data_dir <GPQA csv data path>
or with more optimizations:
    python gpqa.py --hf_model_dir <HF model path> --data_dir <GPQA csv data path> \
        --limit 0.1 --tp_size 8 --ep_size 8 --use_cuda_graph --enable_overlap_scheduler \
        --concurrency 8 --mtp_nextn 3 --print_iter_log  --batch_size 32 --max_num_tokens 4096
"""

import argparse
import asyncio
import os
import random
import re
import time
from contextlib import asynccontextmanager
from typing import List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer

from tensorrt_llm import SamplingParams
from tensorrt_llm._torch.llm import LLM as PyTorchLLM
from tensorrt_llm._torch.pyexecutor.config import PyTorchConfig
from tensorrt_llm.builder import BuildConfig
from tensorrt_llm.llmapi import KvCacheConfig, MTPDecodingConfig

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Template for multiple choice questions
QUERY_TEMPLATE_MULTICHOICE = """
Answer the following multiple choice question. The last line of your response should be of the following format: 'Answer: $LETTER' (without quotes) where LETTER is one of ABCD. Think step by step before answering.
{Question}
A) {A}
B) {B}
C) {C}
D) {D}
""".strip()

# Pattern to extract the answer from the response
ANSWER_PATTERN_MULTICHOICE = r"(?i)Answer[ \t]*:[ \t]*([A-D])"


class RandomSeedGenerator:
    """A deterministic seed generator for reproducible random number generation.

    This implementation guarantees consistent results across different machines,
    Python versions, and platforms by using integer-based seed generation.
    """

    def __init__(self, initial_seed: int = 42):
        self.initial_seed = initial_seed
        self.random_generator = random.Random(initial_seed)

    def gen_seed(self, idx: int, sub_idx: Optional[int] = None) -> int:
        # This ensures consistent behavior across platforms
        if sub_idx is not None:
            # Combine seeds using prime numbers and bit operations
            # to minimize collisions and maintain reproducibility
            complex_seed = self.initial_seed
            complex_seed = (complex_seed * 2147483647) + idx  # Use prime number
            complex_seed = (complex_seed * 2147483647) + (sub_idx if sub_idx
                                                          is not None else 0)
        else:
            complex_seed = (self.initial_seed * 2147483647) + idx

        self.random_generator.seed(complex_seed)
        return self.random_generator.randint(0, 2**32 - 1)


class DataShuffle:
    '''
    A class to shuffle the data with fixed seed.
    '''

    def __init__(self, seed: int = 42):
        self.seed = seed
        self.random_generator = random.Random(self.seed)

    def shuffle(self, data: List[dict]) -> List[dict]:
        self.random_generator.shuffle(data)
        return data


# Class to manage tasks for processing requests
class TaskManager:

    def __init__(self,
                 model: PyTorchLLM,
                 outbox: asyncio.Queue[Tuple[int, float]],
                 concurrency: int = -1) -> None:
        self.model = model
        self._inbox = asyncio.Queue()
        self._outbox = outbox

        self._stop = asyncio.Event()
        self._tasks: Set[asyncio.Task] = set()
        self._backend_task = None
        self._concurrency_semaphore = asyncio.Semaphore(
            concurrency) if concurrency > 0 else None

    # Function to extract the answer from the response and calculate the score
    def get_answer(self, response: str, answer: str) -> float:
        match = re.search(ANSWER_PATTERN_MULTICHOICE, response)
        extracted_answer = match.group(1) if match else None
        score = 1.0 if extracted_answer == answer else 0.0
        return score

    # Function to process a single request
    async def process_request(self, idx: int, request: str, answer: str,
                              sampling_params: SamplingParams) -> float:
        async with semaphore_guard(self._concurrency_semaphore):
            output = self.model.generate_async(request,
                                               sampling_params=sampling_params)

            gen_output = await output.aresult()
            # Extract generated tokens
            response = gen_output.outputs[0].text
            score = self.get_answer(response, answer)
        await self._outbox.put((idx, score))

    # Worker function to continuously process requests
    async def worker(self) -> None:
        while not self._stop.is_set():
            idx, request, answer, sampling_params = await self._inbox.get()
            task = asyncio.create_task(
                self.process_request(idx, request, answer, sampling_params))
            self._tasks.add(task)
            task.add_done_callback(self._tasks.discard)

    # Function to stop the worker
    def stop(self) -> None:
        self._stop.set()
        for task in self._tasks:
            task.cancel()
        self._backend_task.cancel()

    # Property to check if the worker is busy
    @property
    def busy(self) -> bool:
        return bool(self._tasks)

    # Function to start the worker
    def run(self) -> None:
        self._backend_task = asyncio.create_task(self.worker())

    # Function to enqueue a request
    async def enqueue(self, idx: int, request: str, answer: str,
                      sampling_params: SamplingParams) -> None:
        await self._inbox.put((idx, request, answer, sampling_params))


def format_multichoice_question(row: dict) -> str:
    return QUERY_TEMPLATE_MULTICHOICE.format(**row)


def load_data(data_dir: str,
              dataset_shuffle: DataShuffle,
              limit: Optional[float] = None,
              num_runs: int = 1) -> List[List[dict]]:
    assert data_dir.endswith('.csv'), "The provided file is not a CSV file."
    df = pd.read_csv(data_dir)
    dataset = [row.to_dict() for _, row in df.iterrows()]
    if limit is not None:
        dataset = dataset[:int(len(dataset) * limit) + 1]
    shuffled_datasets = []
    for _ in range(num_runs):
        shuffled_datasets.append(dataset_shuffle.shuffle(dataset.copy()))
    return shuffled_datasets


# Function to generate a prompt and the correct answer
def gen_prompt(row: dict, tokenizer: AutoTokenizer,
               dataset_shuffle: DataShuffle) -> Tuple[str, str]:
    choices = dataset_shuffle.shuffle([
        row["Correct Answer"],
        row["Incorrect Answer 1"],
        row["Incorrect Answer 2"],
        row["Incorrect Answer 3"],
    ])
    correct_index = choices.index(row["Correct Answer"])
    answer = "ABCD"[correct_index]
    choices_dict = dict(A=choices[0],
                        B=choices[1],
                        C=choices[2],
                        D=choices[3],
                        Question=row["Question"])
    msg = [{
        "role": "user",
        "content": str(format_multichoice_question(choices_dict))
    }]
    prompt = tokenizer.apply_chat_template(msg,
                                           tokenize=False,
                                           add_generation_prompt=True)
    return prompt, answer


# Async context manager for semaphore
@asynccontextmanager
async def semaphore_guard(semaphore: Optional[asyncio.Semaphore] = None):
    if semaphore is not None:
        await semaphore.acquire()
    try:
        yield
    finally:
        if semaphore is not None:
            semaphore.release()


# Function to enqueue messages for processing
async def enqueue_messages(backend: TaskManager, dataset: List[dict],
                           tokenizer: AutoTokenizer,
                           sampling_params: SamplingParams,
                           submit_finished: asyncio.Event,
                           seed_generator: RandomSeedGenerator,
                           dataset_shuffle: DataShuffle) -> None:
    for idx, row in enumerate(dataset):
        prompt, answer = gen_prompt(row, tokenizer, dataset_shuffle)
        idx_seed = seed_generator.gen_seed(idx=idx)
        sampling_params.seed = idx_seed
        await backend.enqueue(idx, prompt, answer, sampling_params)
    submit_finished.set()


# Function to benchmark the model asynchronously
async def async_benchmark(
    model: PyTorchLLM,
    sampling_params: SamplingParams,
    dataset: List[dict],
    tokenizer: AutoTokenizer,
    seed_generator: RandomSeedGenerator,
    dataset_shuffle: DataShuffle,
    concurrency: int = -1,
) -> List[float]:
    outbox = asyncio.Queue()
    submit_finished = asyncio.Event()
    results = []

    try:
        backend = TaskManager(model, outbox, concurrency=concurrency)
        backend.run()

        num_requests = len(dataset)
        enqueue_task = asyncio.create_task(
            enqueue_messages(backend, dataset, tokenizer, sampling_params,
                             submit_finished, seed_generator, dataset_shuffle))

        with tqdm(total=num_requests, desc="Processing requests") as pbar:
            while not submit_finished.is_set() or not outbox.empty() or len(
                    results) < num_requests:
                try:
                    idx, item = await asyncio.wait_for(outbox.get(),
                                                       timeout=3600)
                    results.append((idx, item))
                    pbar.update(1)
                except asyncio.TimeoutError:
                    print("No items in queue. Continuing.")
                    if not backend.busy:
                        break
        results.sort(key=lambda x: x[0])
        return results

    except asyncio.CancelledError:
        enqueue_task.cancel()

    finally:
        backend.stop()


# Function to parse command line arguments
def parse_args():
    # Model args
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf_model_dir",
                        type=str,
                        required=True,
                        default=None,
                        help="HF model dir")
    parser.add_argument("--tokenizer_dir",
                        type=str,
                        default=None,
                        help="Tokenizer dir")
    parser.add_argument('--load_format',
                        type=str,
                        default='auto',
                        help='Load format for the model')
    parser.add_argument("--top_p",
                        type=float,
                        default=1e-5,
                        help="Top-p for sampling")
    parser.add_argument("--temperature",
                        type=float,
                        default=0.0,
                        help="Temperature for sampling")

    # PyTorch backend settings
    parser.add_argument("--backend",
                        type=str,
                        choices=["pytorch"],
                        default="pytorch",
                        help="Choose the backend to run the model")
    parser.add_argument('--attn_backend',
                        type=str,
                        default='TRTLLM',
                        choices=['TRTLLM', 'FLASHINFER'],
                        help='Attention kernel for PyTorch flow.')
    parser.add_argument("--max_generation_tokens",
                        type=int,
                        default=32768,
                        help="Maximum generation tokens")
    parser.add_argument("--concurrency",
                        type=int,
                        default=-1,
                        help="Concurrency for dataset items")
    parser.add_argument('--batch_size',
                        type=int,
                        default=32,
                        help="Max batch size")
    parser.add_argument("--max_num_tokens",
                        type=int,
                        default=4096,
                        help="Maximum number of tokens")
    parser.add_argument("--tp_size",
                        type=int,
                        default=1,
                        help="Tensor Parallel size (only for pytorch backend)")
    parser.add_argument("--ep_size",
                        type=int,
                        default=1,
                        help="Expert Parallel size (only for pytorch backend)")

    # KV cache
    parser.add_argument('--kv_cache_dtype',
                        type=str,
                        default='auto',
                        help='KV cache dtype')
    parser.add_argument('--kv_cache_disable_block_reuse',
                        default=False,
                        action='store_true',
                        help='Disable block reuse for KV cache')

    # TODO: change the default value back to 0.95
    parser.add_argument("--kv_cache_fraction",
                        type=float,
                        default=0.85,
                        help='Fraction of KV cache to use')

    # Optimizations
    parser.add_argument('--use_cuda_graph',
                        default=False,
                        action='store_true',
                        help='Use CUDA graph for inference')
    parser.add_argument('--torch_compile',
                        action="store_true",
                        help="Enable torch compile for pytorch backend")
    parser.add_argument("--enable_attention_dp",
                        default=False,
                        action='store_true')
    parser.add_argument('--print_iter_log',
                        default=False,
                        action='store_true',
                        help='Print iteration logs during execution')
    parser.add_argument('--enable_overlap_scheduler',
                        default=False,
                        action='store_true')

    # Speculative decoding
    parser.add_argument('--mtp_nextn',
                        type=int,
                        default=0,
                        help='Number of next-n layers to predict')

    # GPQA args
    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help="Path to the data directory. If not available, "
        "download from https://huggingface.co/datasets/Idavidrein/gpqa")
    parser.add_argument("--limit",
                        type=float,
                        default=None,
                        help="Limit the number of samples to run")
    parser.add_argument('--check_accuracy', action='store_true')
    parser.add_argument('--accuracy_threshold', type=float, default=0.67)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--out_dir', type=str, default=None)
    parser.add_argument('--num_runs', type=int, default=1)

    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    if args.tokenizer_dir is None:
        args.tokenizer_dir = args.hf_model_dir
    random.seed(args.seed)
    np.random.seed(args.seed)

    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.hf_model_dir)

    # Configure the PyTorch model
    build_config = BuildConfig(max_batch_size=args.batch_size,
                               max_num_tokens=args.max_num_tokens)
    pytorch_config = PyTorchConfig(
        attn_backend=args.attn_backend,
        enable_overlap_scheduler=args.enable_overlap_scheduler,
        torch_compile_enabled=args.torch_compile,
        kv_cache_dtype=args.kv_cache_dtype,
        use_cuda_graph=args.use_cuda_graph,
        load_format=args.load_format,
        print_iter_log=args.print_iter_log,
        # TODO: there is a known issue in autotuner_enabled warmup,
        # and it will be fixed in the near future
        autotuner_enabled=False)
    kv_cache_config = KvCacheConfig(
        enable_block_reuse=not args.kv_cache_disable_block_reuse,
        free_gpu_memory_fraction=args.kv_cache_fraction)
    mtp_config = MTPDecodingConfig(
        num_nextn_predict_layers=args.mtp_nextn) if args.mtp_nextn > 0 else None

    model = PyTorchLLM(model=args.hf_model_dir,
                       tokenizer=tokenizer,
                       tensor_parallel_size=args.tp_size,
                       kv_cache_config=kv_cache_config,
                       speculative_config=mtp_config,
                       moe_expert_parallel_size=args.ep_size,
                       pytorch_backend_config=pytorch_config,
                       build_config=build_config,
                       enable_attention_dp=args.enable_attention_dp)

    # Configure the sampling params
    sampling_params = SamplingParams(max_tokens=args.max_generation_tokens,
                                     top_p=args.top_p,
                                     temperature=args.temperature,
                                     end_id=tokenizer.eos_token_id,
                                     pad_id=tokenizer.pad_token_id)

    # Load the dataset
    seed_generator = RandomSeedGenerator(initial_seed=args.seed)
    dataset_shuffle = DataShuffle(seed=args.seed)
    datasets = load_data(args.data_dir,
                         dataset_shuffle,
                         limit=args.limit,
                         num_runs=args.num_runs)

    t = time.time()
    try:
        # Run the benchmark
        results = []
        for i in range(args.num_runs):
            dataset = datasets[i]
            result = asyncio.run(
                async_benchmark(model,
                                sampling_params,
                                dataset,
                                tokenizer,
                                seed_generator,
                                dataset_shuffle,
                                concurrency=args.concurrency))
            results.append(result)
    finally:
        if model is not None:
            model.__exit__(None, None, None)
    t = time.time() - t
    print(f"Finished in {t:.3f} seconds")

    # Calculate and print the accuracy
    acc = [np.mean([res[1] for res in result]) for result in results]
    acc_mean = np.mean(acc)
    for i in range(args.num_runs):
        print(f"Run {i+1} accuracy: {acc[i]:.3f}")
    print("Average accuracy: {:.3f}".format(acc_mean))
    if args.check_accuracy:
        assert acc_mean >= args.accuracy_threshold, f"Expected accuracy >= {args.accuracy_threshold} while got {acc_mean}"

    return acc


if __name__ == "__main__":
    main()
