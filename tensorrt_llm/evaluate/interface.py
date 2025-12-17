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
                 dump_path: Optional[str] = None,
                 dump_as_text: bool = False):
        random.seed(random_seed)
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
        self.apply_chat_template = apply_chat_template
        self.fewshot_as_multiturn = fewshot_as_multiturn
        self.system_prompt = system_prompt
        self.chat_template_kwargs = chat_template_kwargs
        self.dump_path = dump_path
        self.dump_as_text = dump_as_text

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
        task_id = 0
        if self.dump_path:
            self.dump_path = prepare_dump_path(self.dump_path)
            logger.info(f"Dumping data to {self.dump_path}")
        for output in tqdm(outputs, desc="Fetching responses"):
            res = output.result()
            results.append(res)
            dump_inference_result(self.dump_path, res, task_id,
                                  self.dump_as_text,
                                  getattr(llm, 'tokenizer', None))
            task_id += 1

        profiler.stop("trtllm exec")
        elapsed_time = profiler.elapsed_time_in_sec("trtllm exec")
        logger.info(f"TRTLLM execution time: {elapsed_time:.3f} seconds.")
        profiler.reset("trtllm exec")

        score = self.compute_score(results, references, *zip(*auxiliaries))
        return score

    @staticmethod
    def command(ctx, *args, **kwargs) -> None:
        raise NotImplementedError()


def prepare_dump_path(dump_path: str) -> str:
    if dump_path:
        if os.path.isdir(dump_path) or dump_path.endswith(os.sep):
            dump_path = os.path.join(dump_path, "dumped_data.json")
        os.makedirs(os.path.dirname(dump_path), exist_ok=True)
        if os.path.exists(dump_path):
            os.remove(dump_path)
    return dump_path


def dump_inference_result(dump_path: str, result: RequestOutput, task_id: int,
                          dump_as_text: bool, tokenizer: Any):
    if not dump_path:
        return
    try:
        with open(dump_path, "a") as f:
            input_ids = result.prompt_token_ids
            output_ids = result.outputs[0].token_ids

            if tokenizer is None:
                logger.warning("Tokenizer not found, dumping raw token ids")
                dump_as_text = False

            if dump_as_text:
                input_content = tokenizer.decode(input_ids)
                output_content = tokenizer.decode(output_ids)
            else:
                input_content = input_ids
                output_content = output_ids

            if dump_as_text:
                data = {
                    "task_id": task_id,
                    "input_text": input_content,
                    "output_text": output_content,
                    "input_lens": len(input_content),
                    "output_lens": len(output_content)
                }
            else:
                data = {
                    "task_id": task_id,
                    "input_ids": input_ids,
                    "output_ids": output_ids,
                    "input_tokens": len(input_content),
                    "output_tokens": len(output_content)
                }
            f.write(json.dumps(data) + "\n")
    except Exception as e:
        logger.warning(f"Failed to dump data to {dump_path}: {e}")
