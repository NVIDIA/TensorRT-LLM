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
from typing import Optional, Tuple

import click
from pydantic import BaseModel, field_validator
from transformers import AutoTokenizer
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast
from utils.prepare_real_data import dataset
from utils.prepare_synthetic_data import token_norm_dist, token_unif_dist


class RootArgs(BaseModel):
    tokenizer: str
    output: str
    random_seed: int
    task_id: int
    std_out: bool
    rand_task_id: Optional[Tuple[int, int]]

    @field_validator('tokenizer')
    def get_tokenizer(cls,
                      v: str) -> PreTrainedTokenizer | PreTrainedTokenizerFast:
        try:
            tokenizer = AutoTokenizer.from_pretrained(v, padding_side='left')
        except EnvironmentError as e:
            raise ValueError(
                f"Cannot find a tokenizer from the given string because of {e}\nPlease set tokenizer to the directory that contains the tokenizer, or set to a model name in HuggingFace."
            )
        tokenizer.pad_token = tokenizer.eos_token
        return tokenizer


@click.group()
@click.option(
    "--tokenizer",
    required=True,
    type=str,
    help=
    "Tokenizer dir for the model run by gptManagerBenchmark, or the model name from HuggingFace."
)
@click.option("--output",
              type=str,
              help="Output json filename.",
              default="preprocessed_dataset.json")
@click.option(
    "--stdout",
    is_flag=True,
    help="Print output to stdout with a JSON dataset entry on each line.",
    default=False)
@click.option("--random-seed",
              required=False,
              type=int,
              help="random seed for token_ids",
              default=420)
@click.option("--task-id", type=int, default=-1, help="LoRA task id")
@click.option("--rand-task-id",
              type=int,
              default=None,
              nargs=2,
              help="Random LoRA Tasks")
@click.option("--log-level",
              default="info",
              type=click.Choice(['info', 'debug']),
              help="Logging level.")
@click.pass_context
def cli(ctx, **kwargs):
    """This script generates dataset input for gptManagerBenchmark."""
    if kwargs['log_level'] == 'info':
        logging.basicConfig(level=logging.INFO)
    elif kwargs['log_level'] == 'debug':
        logging.basicConfig(level=logging.DEBUG)
    else:
        raise ValueError(f"Unsupported logging level {kwargs['log_level']}")

    ctx.obj = RootArgs(tokenizer=kwargs['tokenizer'],
                       output=kwargs['output'],
                       std_out=kwargs['stdout'],
                       random_seed=kwargs['random_seed'],
                       task_id=kwargs['task_id'],
                       rand_task_id=kwargs['rand_task_id'])


cli.add_command(dataset)
cli.add_command(token_norm_dist)
cli.add_command(token_unif_dist)

if __name__ == "__main__":
    cli()
