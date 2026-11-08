#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from typing import Optional

import click

from tensorrt_llm.llmapi import LLM, SamplingParams


@click.command()
@click.option("--model_dir", type=str, required=True)
@click.option("--tp_size", type=int, default=1)
@click.option("--backend", type=str, default=None)
def main(model_dir: str, tp_size: int, backend: Optional[str]):
    backend = backend or "pytorch"
    assert backend == "pytorch"

    # The logging smoke test only generates one request.
    llm = LLM(model_dir, tensor_parallel_size=tp_size, max_batch_size=1)

    sampling_params = SamplingParams(max_tokens=10, end_id=-1)

    prompt_token_ids = [45, 12, 13]
    for output in llm.generate([prompt_token_ids],
                               sampling_params=sampling_params):
        print(output)


if __name__ == '__main__':
    main()
