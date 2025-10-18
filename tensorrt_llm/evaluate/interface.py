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
                 chat_template_kwargs: Optional[dict[str, Any]] = None):
        random.seed(random_seed)
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
        self.apply_chat_template = apply_chat_template
        self.fewshot_as_multiturn = fewshot_as_multiturn
        self.system_prompt = system_prompt
        self.chat_template_kwargs = chat_template_kwargs

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
        profiler.stop("trtllm exec")
        elapsed_time = profiler.elapsed_time_in_sec("trtllm exec")
        logger.info(f"TRTLLM execution time: {elapsed_time:.3f} seconds.")
        profiler.reset("trtllm exec")

        score = self.compute_score(results, references, *zip(*auxiliaries))
        return score

    @staticmethod
    def command(ctx, *args, **kwargs) -> None:
        raise NotImplementedError()
