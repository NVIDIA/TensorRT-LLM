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
from pathlib import Path
from typing import Optional, Tuple

import click
from pydantic import BaseModel, model_validator
from transformers import AutoTokenizer

from tensorrt_llm.bench.dataset.prepare_real_data import real_dataset
from tensorrt_llm.bench.dataset.prepare_synthetic_data import token_norm_dist, token_unif_dist


class RootArgs(BaseModel):
    tokenizer: str
    output: str
    random_seed: int
    task_id: int
    trust_remote_code: bool = False
    rand_task_id: Optional[Tuple[int, int]]
    lora_dir: Optional[str] = None

    @model_validator(mode="after")
    def validate_tokenizer(self):
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                self.tokenizer, padding_side="left", trust_remote_code=self.trust_remote_code
            )
        except EnvironmentError as e:
            raise ValueError(
                "Cannot find a tokenizer from the given string because of "
                f"{e}\nPlease set tokenizer to the directory that contains "
                "the tokenizer, or set to a model name in HuggingFace."
            )
        tokenizer.pad_token = tokenizer.eos_token
        self.tokenizer = tokenizer

        return self


@click.group(name="prepare-dataset")
@click.option(
    "--output", type=str, help="Output json filename.", default="preprocessed_dataset.json"
)
@click.option(
    "--random-seed", required=False, type=int, help="random seed for token_ids", default=420
)
@click.option("--task-id", type=int, default=-1, help="LoRA task id")
@click.option("--rand-task-id", type=int, default=None, nargs=2, help="Random LoRA Tasks")
@click.option("--lora-dir", type=str, default=None, help="Directory containing LoRA adapters")
@click.option(
    "--log-level", default="info", type=click.Choice(["info", "debug"]), help="Logging level."
)
@click.option(
    "--trust-remote-code",
    is_flag=True,
    default=False,
    envvar="TRUST_REMOTE_CODE",
    help="Trust remote code.",
)
@click.pass_context
def prepare_dataset(ctx, **kwargs):
    """Prepare dataset for benchmarking with trtllm-bench."""
    model = ctx.obj.model or ctx.obj.checkpoint_path
    output_path = Path(kwargs["output"])
    output_path.parent.mkdir(parents=True, exist_ok=True)

    ctx.obj = RootArgs(
        tokenizer=model,
        output=kwargs["output"],
        random_seed=kwargs["random_seed"],
        task_id=kwargs["task_id"],
        rand_task_id=kwargs["rand_task_id"],
        lora_dir=kwargs["lora_dir"],
        trust_remote_code=kwargs["trust_remote_code"],
    )


prepare_dataset.add_command(real_dataset)
prepare_dataset.add_command(token_norm_dist)
prepare_dataset.add_command(token_unif_dist)
