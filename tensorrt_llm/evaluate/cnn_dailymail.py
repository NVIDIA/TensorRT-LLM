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
from typing import Iterable, List, Union

import click
import datasets
import evaluate

from .._torch import LLM as PyTorchLLM
from ..llmapi import LLM, RequestOutput
from ..logger import logger
from ..sampling_params import SamplingParams
from .interface import Evaluator


class CnnDailymail(Evaluator):

    def __init__(self,
                 dataset_path: str = "ccdv/cnn_dailymail",
                 num_samples: int = None,
                 random_seed: int = 0,
                 rouge_path: str = "rouge"):
        self.data = datasets.load_dataset(dataset_path, "3.0.0", split="test")
        self.data = self.data.shuffle(random_seed)
        if num_samples is None:
            self.num_samples = self.data.num_rows
        else:
            self.num_samples = min(num_samples, self.data.num_rows)
        self.rouge = evaluate.load(rouge_path)

    def generate_samples(self) -> Iterable[tuple]:
        for i, sample in enumerate(self.data):
            if i >= self.num_samples:
                break
            prompt = sample["article"].strip().replace(" n't",
                                                       "n't") + " TL;DR: "
            reference = sample["highlights"].strip().replace(" n't", "n't")
            yield prompt, reference

    def compute_score(self, outputs: List[RequestOutput],
                      references: List[str]) -> float:
        metrics = self.rouge.compute(
            predictions=[output.outputs[0].text for output in outputs],
            references=references)
        for key in metrics.keys():
            logger.info(f"  {key}: {metrics[key]*100:.3f}")
        return metrics["rouge1"] * 100

    @click.command("cnn_dailymail")
    @click.option("--dataset_path", type=str, default="ccdv/cnn_dailymail")
    @click.option("--num_samples", type=int, default=None)
    @click.option("--random_seed", type=int, default=0)
    @click.option("--rouge_path", type=str, default="rouge")
    @click.option("--max_input_length", type=int, default=924)
    @click.option("--max_output_length", type=int, default=100)
    @click.option("--check_accuracy", is_flag=True, default=False)
    @click.option("--accuracy_threshold", type=float, default=15)
    @click.pass_context
    @staticmethod
    def command(ctx, dataset_path: str, num_samples: int, random_seed: int,
                rouge_path: str, max_input_length: int, max_output_length: int,
                check_accuracy: bool, accuracy_threshold: float) -> None:
        llm: Union[LLM, PyTorchLLM] = ctx.obj
        sampling_params = SamplingParams(
            max_tokens=max_output_length,
            truncate_prompt_tokens=max_input_length)
        evaluator = CnnDailymail(dataset_path,
                                 num_samples=num_samples,
                                 random_seed=random_seed,
                                 rouge_path=rouge_path)
        accuracy = evaluator.evaluate(llm, sampling_params)
        llm.shutdown()

        if check_accuracy:
            assert accuracy >= accuracy_threshold, f"Expected accuracy >= {accuracy_threshold}, but got {accuracy}"
