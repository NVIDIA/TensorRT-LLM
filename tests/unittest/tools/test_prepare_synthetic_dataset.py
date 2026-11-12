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
"""Length distributions of ``trtllm-bench prepare-dataset`` synthetic datasets (no tokenizer download)."""

import json
from types import SimpleNamespace

import numpy as np
import pytest
from click.testing import CliRunner

from tensorrt_llm.bench.dataset import prepare_synthetic_data

pytestmark = pytest.mark.cpu_only

_NUM_REQUESTS = 200


class _Tokenizer:
    vocab_size = 100000
    eos_token = "</s>"

    def encode(self, text, add_special_tokens=True):
        return [self.vocab_size - 1]


def _lengths(tmp_path, command, args):
    output = tmp_path / "dataset.jsonl"
    root_args = SimpleNamespace(
        tokenizer=_Tokenizer(),
        output=str(output),
        random_seed=420,
        task_id=-1,
        rand_task_id=None,
        lora_dir=None,
    )
    result = CliRunner().invoke(
        command, ["--num-requests", str(_NUM_REQUESTS), *args], obj=root_args
    )
    assert result.exit_code == 0, result.output
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert len(rows) == _NUM_REQUESTS
    return ([len(row["input_ids"]) for row in rows], [row["output_tokens"] for row in rows])


@pytest.mark.parametrize(
    "command, args",
    [
        pytest.param(
            prepare_synthetic_data.token_norm_dist,
            [
                "--input-mean",
                "1000",
                "--input-stdev",
                "200",
                "--output-mean",
                "500",
                "--output-stdev",
                "100",
            ],
            id="norm",
        ),
        pytest.param(
            prepare_synthetic_data.token_unif_dist,
            [
                "--input-min",
                "100",
                "--input-max",
                "2000",
                "--output-min",
                "50",
                "--output-max",
                "1000",
            ],
            id="unif",
        ),
    ],
)
def test_input_and_output_lengths_are_not_correlated(tmp_path, command, args):
    input_lens, output_lens = _lengths(tmp_path, command, args)
    # Both lengths used to come from one random sequence, which made them perfectly correlated.
    assert len(set(input_lens)) > 1 and len(set(output_lens)) > 1
    assert abs(np.corrcoef(input_lens, output_lens)[0, 1]) < 0.3
