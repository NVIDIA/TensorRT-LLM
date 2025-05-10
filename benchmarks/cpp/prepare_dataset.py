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

import logging
from typing import Any, Optional, Tuple

import click
from pydantic import BaseModel, field_validator, model_validator
from transformers import AutoTokenizer
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast
from utils.abstractions import (ConstantTaskIdDistribution,
                                UniformTaskIdDistribution)
from utils.prepare_real_data import dataset
from utils.prepare_synthetic_data import token_norm_dist, token_unif_dist


class RootArgs(BaseModel):
    tokenizer: str
    output: str
    random_seed: int
    task_id: int = None
    std_out: bool
    rand_task_id: Optional[Tuple[int, int]] = None
    export_format: str
    task_id_distribution: Any = None

    @field_validator("tokenizer")
    def get_tokenizer(cls,
                      v: str) -> PreTrainedTokenizer | PreTrainedTokenizerFast:
        try:
            tokenizer = AutoTokenizer.from_pretrained(v, padding_side="left")
        except EnvironmentError as e:
            raise ValueError(
                f"Cannot find a tokenizer from the given string because of {e}\nPlease set tokenizer to the directory that contains the tokenizer, or set to a model name in HuggingFace."
            )
        tokenizer.pad_token = tokenizer.eos_token
        return tokenizer

    @model_validator(mode="after")
    def get_task_id_distribution(self) -> "RootArgs":
        # Check that exactly one of task_id or rand_task_id is set
        if (self.task_id is None and self.rand_task_id is None) or (
                self.task_id is not None and self.rand_task_id is not None):
            raise ValueError(
                "Exactly one of 'task_id' or 'rand_task_id' must be specified")

        # Set the task_id_distribution based on which field is set
        if self.task_id is not None:
            # Use constant distribution with the specific task_id
            self.task_id_distribution = ConstantTaskIdDistribution(
                task_id=self.task_id)
        else:
            # Use uniform distribution within the rand_task_id range
            min_id, max_id = self.rand_task_id
            self.task_id_distribution = UniformTaskIdDistribution(min_id=min_id,
                                                                  max_id=max_id)

        return self


@click.group()
@click.option(
    "--tokenizer",
    required=True,
    type=str,
    help=
    "Tokenizer dir for the model run by gptManagerBenchmark, or the model name from HuggingFace.",
)
@click.option(
    "--output",
    type=str,
    help="Output json filename.",
    default="preprocessed_dataset.json",
)
@click.option(
    "--stdout",
    is_flag=True,
    help=
    "Print output to stdout with a JSON dataset entry on each line. Deprecated.Prefer using / extending --export-format instead when your goal is to export a dataset for tllm-bench or a new benchmarking entrypoint.",
    default=False,
)
@click.option(
    "--random-seed",
    required=False,
    type=int,
    help="random seed for token_ids",
    default=420,
)
@click.option("--task-id", type=int, default=-1, help="LoRA task id")
@click.option("--rand-task-id",
              type=int,
              default=None,
              nargs=2,
              help="Random LoRA Tasks")
@click.option(
    "--log-level",
    default="info",
    type=click.Choice(["info", "debug"]),
    help="Logging level.",
)
@click.option(
    "--export-format",
    default="gpt-manager-benchmark",
    type=click.Choice(["gpt-manager-benchmark", "tllm-bench"]),
    help=
    "Sets the export format so that the dataset can be consumed by your target benchmarking entrypoint.",
)
@click.pass_context
def cli(ctx: click.Context, **kwargs):
    """This script generates dataset input for gptManagerBenchmark."""
    if kwargs["log_level"] == "info":
        logging.basicConfig(level=logging.INFO)
    elif kwargs["log_level"] == "debug":
        logging.basicConfig(level=logging.DEBUG)
    else:
        raise ValueError(f"Unsupported logging level {kwargs['log_level']}")

    ctx.obj = RootArgs(
        tokenizer=kwargs["tokenizer"],
        output=kwargs["output"],
        std_out=kwargs["stdout"],
        random_seed=kwargs["random_seed"],
        task_id=kwargs["task_id"],
        rand_task_id=kwargs["rand_task_id"],
        export_format=kwargs["export_format"],
    )


cli.add_command(dataset)
cli.add_command(token_norm_dist)
cli.add_command(token_unif_dist)

if __name__ == "__main__":
    cli()
