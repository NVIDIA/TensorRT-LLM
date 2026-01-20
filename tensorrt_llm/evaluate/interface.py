# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import copy
import json
import os
import random
from abc import ABC, abstractmethod
from typing import Any, Iterable, List, Optional, Union

import numpy as np
import torch
from tqdm import tqdm

import tensorrt_llm.profiler as profiler

from ..llmapi import RequestOutput
from ..logger import logger
from ..sampling_params import SamplingParams


class Evaluator(ABC):

    def __init__(self,
                 random_seed: int = 0,
                 apply_chat_template: bool = False,
                 fewshot_as_multiturn: bool = False,
                 system_prompt: Optional[str] = None,
                 chat_template_kwargs: Optional[dict[str, Any]] = None,
                 output_dir: Optional[str] = None):
        random.seed(random_seed)
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
        self.apply_chat_template = apply_chat_template
        self.fewshot_as_multiturn = fewshot_as_multiturn
        self.system_prompt = system_prompt
        self.chat_template_kwargs = chat_template_kwargs
        self.output_dir = output_dir

    @abstractmethod
    def generate_samples(self) -> Iterable[tuple]:
        raise NotImplementedError()

    @abstractmethod
    def compute_score(self, outputs: List[RequestOutput], references: List[str],
                      *auxiliaries) -> float:
        raise NotImplementedError()

    def do_apply_chat_template(self, llm: Any,
                               prompt: Union[str, List[dict]]) -> str:
        if isinstance(prompt, str):
            messages = [{"role": "user", "content": prompt}]
        else:
            messages = prompt
        if self.system_prompt is not None:
            messages = [{
                "role": "system",
                "content": self.system_prompt
            }] + messages
        return llm.tokenizer.apply_chat_template(messages,
                                                 tokenize=False,
                                                 add_generation_prompt=True,
                                                 **(self.chat_template_kwargs
                                                    or {}))

    def _get_sampline_params(self, sampling_params: Optional[SamplingParams],
                             sampling_args: Optional[dict]) -> SamplingParams:
        if sampling_params is None:
            sampling_params = SamplingParams()
        else:
            sampling_params = copy.deepcopy(sampling_params)

        if sampling_args is not None:
            for key, value in sampling_args.items():
                setattr(sampling_params, key, value)
        return sampling_params

    def evaluate(self,
                 llm: Any,
                 sampling_params: Optional[SamplingParams] = None,
                 streaming: bool = False) -> float:
        profiler.start("trtllm exec")
        outputs, references, auxiliaries = [], [], []
        for prompt, sampling_args, reference, *aux in tqdm(
                self.generate_samples(), desc="Submitting requests"):
            if self.apply_chat_template:
                prompt = self.do_apply_chat_template(llm, prompt)
            sampling_params = self._get_sampline_params(sampling_params,
                                                        sampling_args)
            output = llm.generate_async(
                prompt,
                sampling_params,
                streaming=streaming,
            )
            outputs.append(output)
            references.append(reference)
            auxiliaries.append(aux)
        results = []
        for output in tqdm(outputs, desc="Fetching responses"):
            results.append(output.result())

        if self.output_dir:
            dump_inference_results(self.output_dir, results,
                                   getattr(llm, 'tokenizer', None))

        profiler.stop("trtllm exec")
        elapsed_time = profiler.elapsed_time_in_sec("trtllm exec")
        logger.info(f"TRTLLM execution time: {elapsed_time:.3f} seconds.")
        profiler.reset("trtllm exec")

        score = self.compute_score(results, references, *zip(*auxiliaries))
        return score

    @staticmethod
    def command(ctx, *args, **kwargs) -> None:
        raise NotImplementedError()


def dump_inference_results(output_dir: str, results: List[dict],
                           tokenizer: Any):
    if not output_dir:
        return

    os.makedirs(output_dir, exist_ok=True)

    # Collect results
    results_list = []
    for task_id, result in enumerate(results):
        output_ids = result.outputs[0].token_ids
        output_text = result.outputs[0].text.strip()
        input_text = result.prompt.strip()
        input_ids = tokenizer.encode(input_text)
        results_list.append({
            "task_id": task_id,
            "input_ids": input_ids,
            "output_ids": output_ids,
            "input_text": input_text,
            "output_text": output_text
        })

    # Dump token ids
    ids_path = os.path.join(output_dir, "dumped_ids.json")
    try:
        with open(ids_path, "w") as f:
            for item in results_list:
                data = {
                    "task_id": item["task_id"],
                    "input_ids": item["input_ids"],
                    "output_ids": item["output_ids"],
                    "input_tokens": len(item["input_ids"]),
                    "output_tokens": len(item["output_ids"])
                }
                f.write(json.dumps(data) + "\n")
        logger.info(f"Dumped IDs to {ids_path}")
    except Exception as e:
        logger.warning(f"Failed to dump IDs to {ids_path}: {e}")

    # Dump text
    text_path = os.path.join(output_dir, "dumped_text.json")
    try:
        with open(text_path, "w") as f:
            for item in results_list:
                data = {
                    "task_id": item["task_id"],
                    "input_text": item["input_text"],
                    "output_text": item["output_text"],
                    "input_len": len(item["input_text"]),
                    "output_len": len(item["output_text"])
                }
                f.write(json.dumps(data) + "\n")
        logger.info(f"Dumped text to {text_path}")
    except Exception as e:
        logger.warning(f"Failed to dump text to {text_path}: {e}")
