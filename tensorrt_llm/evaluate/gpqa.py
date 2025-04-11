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
from typing import Iterable, List, Optional, Tuple, Union

import click
import datasets
import numpy as np

from .._torch import LLM as PyTorchLLM
from ..llmapi import LLM, RequestOutput
from ..logger import logger
from ..sampling_params import SamplingParams
from .interface import Evaluator

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


def format_multichoice_question(row: dict) -> str:
    return QUERY_TEMPLATE_MULTICHOICE.format(**row)


def gen_prompt(row: dict) -> Tuple[str, str]:
    choices = [
        row["Correct Answer"],
        row["Incorrect Answer 1"],
        row["Incorrect Answer 2"],
        row["Incorrect Answer 3"],
    ]
    correct_index = choices.index(row["Correct Answer"])
    answer = "ABCD"[correct_index]
    choices_dict = dict(A=choices[0],
                        B=choices[1],
                        C=choices[2],
                        D=choices[3],
                        Question=row["Question"])
    prompt = str(format_multichoice_question(choices_dict))
    return prompt, answer


def get_answer(response: str, answer: str) -> float:
    import re
    match = re.search(ANSWER_PATTERN_MULTICHOICE, response)
    extracted_answer = match.group(1) if match else None
    score = 1.0 if extracted_answer == answer else 0.0
    return score


class GPQA(Evaluator):

    def __init__(self,
                 dataset_path: str = "Idavidrein/gpqa",
                 num_samples: Optional[int] = None,
                 random_seed: int = 0,
                 apply_chat_template: bool = False,
                 system_prompt: Optional[str] = None):
        super().__init__(random_seed=random_seed,
                         apply_chat_template=apply_chat_template,
                         system_prompt=system_prompt)
        self.data = datasets.load_dataset(dataset_path,
                                          "gpqa_diamond",
                                          split="train")
        self.data = self.data.shuffle(random_seed)
        if num_samples is None:
            self.num_samples = self.data.num_rows
        else:
            self.num_samples = min(num_samples, self.data.num_rows)

    def generate_samples(self) -> Iterable[tuple]:
        for i, sample in enumerate(self.data):
            if i >= self.num_samples:
                break
            prompt, ref = gen_prompt(sample)
            yield prompt, ref

    def compute_score(self, outputs: List[RequestOutput],
                      references: List[str]) -> float:
        corrections = []
        for output, ref in zip(outputs, references):
            correction = get_answer(output.outputs[0].text, ref)
            corrections.append(correction)

        acc = np.mean(corrections) * 100
        logger.info(f"GPQA average accuracy: {acc:.2f} ({len(corrections)})")
        return acc

    @click.command("gpqa")
    @click.option("--dataset_path", type=str, default="Idavidrein/gpqa")
    @click.option("--num_samples", type=int, default=None)
    @click.option("--random_seed", type=int, default=0)
    @click.option("--max_input_length", type=int, default=4094)
    @click.option("--max_output_length", type=int, default=32768)
    @click.option("--apply_chat_template", is_flag=True, default=False)
    @click.option("--check_accuracy", is_flag=True, default=False)
    @click.option("--accuracy_threshold", type=float, default=30)
    @click.pass_context
    @staticmethod
    def command(ctx, dataset_path: str, num_samples: int, random_seed: int,
                max_input_length: int, max_output_length: int,
                apply_chat_template: bool, check_accuracy: bool,
                accuracy_threshold: float) -> None:
        llm: Union[LLM, PyTorchLLM] = ctx.obj
        sampling_params = SamplingParams(
            max_tokens=max_output_length,
            truncate_prompt_tokens=max_input_length)
        evaluator = GPQA(dataset_path,
                         num_samples=num_samples,
                         random_seed=random_seed,
                         apply_chat_template=apply_chat_template)
        accuracy = evaluator.evaluate(llm, sampling_params)
        llm.shutdown()

        if check_accuracy:
            assert accuracy >= accuracy_threshold, f"Expected accuracy >= {accuracy_threshold}, but got {accuracy}"
