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
from typing import Iterable, List, Optional, Union

import click
import datasets
import evaluate

from .. import LLM as PyTorchLLM
from .._tensorrt_engine import LLM
from ..llmapi import RequestOutput
from ..logger import logger
from ..sampling_params import SamplingParams
from .interface import Evaluator


class CnnDailymail(Evaluator):

    def __init__(self,
                 dataset_path: Optional[str] = None,
                 num_samples: Optional[int] = None,
                 random_seed: int = 0,
                 rouge_path: Optional[str] = None,
                 apply_chat_template: bool = False,
                 system_prompt: Optional[str] = None):
        super().__init__(random_seed=random_seed,
                         apply_chat_template=apply_chat_template,
                         system_prompt=system_prompt)
        if dataset_path is None:
            dataset_path = "ccdv/cnn_dailymail"
        self.data = datasets.load_dataset(dataset_path,
                                          "3.0.0",
                                          split="test",
                                          trust_remote_code=True)
        self.data = self.data.shuffle(random_seed)
        if num_samples is None:
            self.num_samples = self.data.num_rows
        else:
            self.num_samples = min(num_samples, self.data.num_rows)
        if rouge_path is None:
            rouge_path = "rouge"
        self.rouge = evaluate.load(rouge_path)

    def generate_samples(self) -> Iterable[tuple]:
        for i, sample in enumerate(self.data):
            if i >= self.num_samples:
                break
            prompt = sample["article"] + " TL;DR:"
            prompt = prompt.strip().replace(" n't", "n't")
            yield prompt, None, sample["highlights"]

    def compute_score(self, outputs: List[RequestOutput],
                      references: List[str]) -> float:
        for beam_idx in range(len(outputs[0].outputs)):
            metrics = self.rouge.compute(
                predictions=[output.outputs[0].text for output in outputs],
                references=references)
            logger.info(f"Beam {beam_idx} rouge scores:")
            for key in metrics.keys():
                logger.info(f"\t{key}: {metrics[key]*100:.3f}")
            if beam_idx == 0:
                rouge1 = metrics["rouge1"] * 100
        return rouge1

    @click.command("cnn_dailymail")
    @click.option("--dataset_path",
                  type=str,
                  default=None,
                  help="The path to CNN Dailymail dataset. "
                  "If unspecified, the dataset is downloaded from HF hub.")
    @click.option(
        "--num_samples",
        type=int,
        default=None,
        help="Number of samples to run the evaluation; None means full dataset."
    )
    @click.option("--random_seed",
                  type=int,
                  default=0,
                  help="Random seed for dataset processing.")
    @click.option("--rouge_path",
                  type=str,
                  default=None,
                  help="The path to rouge repository."
                  "If unspecified, the repository is downloaded from HF hub.")
    @click.option("--apply_chat_template",
                  is_flag=True,
                  default=False,
                  help="Whether to apply chat template.")
    @click.option("--system_prompt",
                  type=str,
                  default=None,
                  help="System prompt.")
    @click.option("--max_input_length",
                  type=int,
                  default=924,
                  help="Maximum prompt length.")
    @click.option("--max_output_length",
                  type=int,
                  default=100,
                  help="Maximum generation length.")
    @click.pass_context
    @staticmethod
    def command(ctx, dataset_path: Optional[str], num_samples: int,
                random_seed: int, rouge_path: Optional[str],
                apply_chat_template: bool, system_prompt: Optional[str],
                max_input_length: int, max_output_length: int) -> None:
        llm: Union[LLM, PyTorchLLM] = ctx.obj
        sampling_params = SamplingParams(
            max_tokens=max_output_length,
            truncate_prompt_tokens=max_input_length)
        evaluator = CnnDailymail(dataset_path,
                                 num_samples=num_samples,
                                 random_seed=random_seed,
                                 rouge_path=rouge_path,
                                 apply_chat_template=apply_chat_template,
                                 system_prompt=system_prompt)
        evaluator.evaluate(llm, sampling_params)
        llm.shutdown()
