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
from abc import ABC, abstractmethod, abstractstaticmethod
from typing import Iterable, List, Optional, Union

from tqdm import tqdm

import tensorrt_llm.profiler as profiler

from .._torch import LLM as PyTorchLLM
from ..llmapi import LLM, RequestOutput
from ..logger import logger
from ..sampling_params import SamplingParams


class Evaluator(ABC):

    def __init__(self,
                 apply_chat_template: bool = False,
                 system_prompt: Optional[str] = None):
        self.apply_chat_template = apply_chat_template
        self.system_prompt = system_prompt

    @abstractmethod
    def generate_samples(self) -> Iterable[tuple]:
        raise NotImplementedError()

    @abstractmethod
    def compute_score(self, outputs: List[RequestOutput], references: List[str],
                      *auxiliaries) -> float:
        raise NotImplementedError()

    def do_apply_chat_template(self, llm: Union[LLM, PyTorchLLM],
                               prompt: str) -> str:
        messages = [{"role": "user", "content": prompt}]
        if self.system_prompt is not None:
            messages = [{
                "role": "system",
                "content": self.system_prompt
            }] + messages
        return llm.tokenizer.apply_chat_template(messages,
                                                 tokenize=False,
                                                 add_generation_prompt=True)

    def evaluate(self,
                 llm: Union[LLM, PyTorchLLM],
                 sampling_params: Optional[SamplingParams] = None) -> float:
        profiler.start("trtllm exec")
        outputs, references, auxiliaries = [], [], []
        for prompt, reference, *aux in tqdm(self.generate_samples(),
                                            desc="Submitting requests"):
            if self.apply_chat_template:
                prompt = self.do_apply_chat_template(llm, prompt)
            output = llm.generate_async(prompt, sampling_params)
            outputs.append(output)
            references.append(reference)
            auxiliaries.append(aux)
        for output in tqdm(outputs, desc="Fetching responses"):
            output.result()
        profiler.stop("trtllm exec")
        elapsed_time = profiler.elapsed_time_in_sec("trtllm exec")
        logger.info(f"TRTLLM execution time: {elapsed_time:.3f} seconds.")

        score = self.compute_score(outputs, references, *zip(*auxiliaries))
        return score

    @abstractstaticmethod
    def command(ctx, *args, **kwargs) -> None:
        raise NotImplementedError()
