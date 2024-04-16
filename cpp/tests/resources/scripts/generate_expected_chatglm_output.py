#!/usr/bin/env python3
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

import run

resources_dir = Path(__file__).parent.resolve().parent
model_path = resources_dir / "models"


def generate_output(
    model_name: str = "",
    num_beams: int = 1,
    max_output_len: int = 8,
    engine_kind: str = "fp32-plugin",
    output_logits: bool = False,
    output_cum_log_probs: bool = False,
    output_log_probs: bool = False,
):
    hf_path = model_path / model_name
    # we do not distinguish TP / PP / engine_kind yet
    tp_size = 1
    pp_size = 1
    tp_pp_dir = f"tp{tp_size}-pp{pp_size}-gpu/"
    engine_dir = model_path / 'rt_engine' / model_name / engine_kind / tp_pp_dir
    data_input_file_name = resources_dir / "data" / f"input_tokens_{model_name}.npy"
    if num_beams <= 1:
        output_dir = resources_dir / "data" / model_name / "sampling"
    else:
        output_dir = resources_dir / "data" / model_name / f"beam_search_{num_beams}"
    output_dir.mkdir(exist_ok=True, parents=True)
    data_output_npy_file_name = output_dir / f"output_tokens.npy"
    data_output_csv_file_name = output_dir / f"output_tokens.csv"

    args_list = [
        '--engine_dir',
        str(engine_dir),
        '--tokenizer_dir',
        str(hf_path),
        '--input_file',
        str(data_input_file_name),
        '--output_npy',
        str(data_output_npy_file_name),
        '--output_csv',
        str(data_output_csv_file_name),
        '--max_output_len',
        str(max_output_len),
        '--num_beams',
        str(num_beams),
        #'--use_py_session',
    ]

    output_logits_npy = None
    if output_logits:
        output_logits_npy = str(output_dir / 'logits.npy')
        args_list.extend(['--output_logits_npy', str(output_logits_npy)])

    output_cum_log_probs_npy = None
    if output_cum_log_probs:
        output_cum_log_probs_npy = str(output_dir / 'cum_log_probs.npy')
        args_list.extend(
            ['--output_cum_log_probs_npy',
             str(output_cum_log_probs_npy)])

    output_log_probs_npy = None
    if output_log_probs:
        output_log_probs_npy = str(output_dir / 'log_probs.npy')
        args_list.extend(['--output_log_probs_npy', str(output_log_probs_npy)])

    args = run.parse_arguments(args_list)
    run.main(args)


if __name__ == '__main__':
    generate_output(model_name='chatglm-6b', num_beams=1)
    generate_output(model_name='chatglm-6b', num_beams=2)
    generate_output(model_name='chatglm2-6b', num_beams=1)
    generate_output(model_name='chatglm2-6b', num_beams=2)
    generate_output(model_name='chatglm3-6b', num_beams=1)
    generate_output(model_name='chatglm3-6b', num_beams=2)
    print("Done")
