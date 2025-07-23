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
import json
import os
from typing import Iterable, List, Optional, Union

import click
import datasets
import jsonschema
import numpy as np

from .. import LLM as PyTorchLLM
from .._tensorrt_engine import LLM
from ..llmapi import RequestOutput
from ..logger import logger
from ..sampling_params import GuidedDecodingParams, SamplingParams
from .interface import Evaluator


class JsonModeEval(Evaluator):

    def __init__(self,
                 dataset_path: Optional[str] = None,
                 num_samples: Optional[int] = None,
                 random_seed: int = 0,
                 apply_chat_template: bool = True,
                 system_prompt: Optional[str] = None):
        if not apply_chat_template:
            raise ValueError(
                f"{self.__class__.__name__} requires apply_chat_template=True.")
        super().__init__(random_seed=random_seed,
                         apply_chat_template=apply_chat_template,
                         system_prompt=system_prompt)
        if dataset_path is None:
            dataset_path = "NousResearch/json-mode-eval"
        self.data = datasets.load_dataset(dataset_path,
                                          split="train",
                                          trust_remote_code=True)
        self.data = self.data.shuffle(random_seed)
        if num_samples is None:
            self.num_samples = self.data.num_rows
        else:
            self.num_samples = min(num_samples, self.data.num_rows)

    def generate_samples(self) -> Iterable[tuple]:
        for i, sample in enumerate(self.data):
            if i >= self.num_samples:
                break
            schema = sample["schema"]
            if os.environ.get("TRTLLM_XGUIDANCE_LENIENT") == "1":
                schema = json.loads(schema)
                schema["x-guidance"] = {"lenient": True}
                schema = json.dumps(schema)
            sampling_args = {
                "guided_decoding": GuidedDecodingParams(json=schema)
            }
            yield sample["prompt"], sampling_args, sample["completion"], sample[
                "schema"]

    def compute_score(self, outputs: List[RequestOutput], references: List[str],
                      schemas: List[str]) -> float:
        all_corrections, all_grammar_corrections = [], []
        for output, ref, schema in zip(outputs, references, schemas):
            try:
                output_json = json.loads(output.outputs[0].text)
                jsonschema.validate(output_json, json.loads(schema))
            except (json.JSONDecodeError, jsonschema.ValidationError):
                all_corrections.append(False)
                all_grammar_corrections.append(False)
                continue
            all_corrections.append(output_json == json.loads(ref))
            all_grammar_corrections.append(True)

        acc = np.mean(all_corrections) * 100
        logger.info(
            f"JSON Mode Eval accuracy: {acc:.2f} ({len(all_corrections)})")
        grammar_acc = np.mean(all_grammar_corrections) * 100
        logger.info(
            f"JSON Mode Eval grammar accuracy: {grammar_acc:.2f} ({len(all_grammar_corrections)})"
        )
        return acc

    @click.command("json_mode_eval")
    @click.option("--dataset_path",
                  type=str,
                  default=None,
                  help="The path to JSON Mode Eval dataset. "
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
    @click.option("--system_prompt",
                  type=str,
                  default=None,
                  help="System prompt.")
    @click.option("--max_input_length",
                  type=int,
                  default=1024,
                  help="Maximum prompt length.")
    @click.option("--max_output_length",
                  type=int,
                  default=512,
                  help="Maximum generation length.")
    @click.pass_context
    @staticmethod
    def command(ctx, dataset_path: Optional[str], num_samples: int,
                random_seed: int, system_prompt: Optional[str],
                max_input_length: int, max_output_length: int) -> None:
        llm: Union[LLM, PyTorchLLM] = ctx.obj
        sampling_params = SamplingParams(
            max_tokens=max_output_length,
            truncate_prompt_tokens=max_input_length)
        evaluator = JsonModeEval(dataset_path,
                                 num_samples=num_samples,
                                 random_seed=random_seed,
                                 apply_chat_template=True,
                                 system_prompt=system_prompt)
        evaluator.evaluate(llm, sampling_params)
        llm.shutdown()
