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

import os
from pathlib import Path

import numpy as np

# isort: off
import run
# isort: on

import tensorrt_llm.bindings as _tb
from tensorrt_llm.bindings.internal.testing import ModelSpec

resources_dir = Path(__file__).parent.resolve().parent
model_path = resources_dir / "models"


def generate_output(
    model_name: str = "",
    num_beams: int = 1,
    max_output_len: int = 8,
    output_logits: bool = False,
    output_cum_log_probs: bool = False,
    output_log_probs: bool = False,
):
    hf_path = model_path / model_name
    tp_size = 1
    pp_size = 1
    cp_size = 1
    tp_pp_cp_dir = f"tp{tp_size}-pp{pp_size}-cp{cp_size}-gpu/"
    input_file = f"input_tokens_{model_name}.npy"

    data_input_file_name = resources_dir / "data" / input_file
    if num_beams == 1:
        output_dir = resources_dir / "data" / model_name / "sampling"
    else:
        output_dir = resources_dir / "data" / model_name / f"beam_search_{num_beams}"
    output_dir.mkdir(exist_ok=True, parents=True)

    model_spec_obj_list = [
        ModelSpec(input_file,
                  _tb.DataType.HALF).use_gpt_plugin().set_kv_cache_type(
                      _tb.KVCacheType.CONTINUOUS),
        ModelSpec(input_file, _tb.DataType.HALF).use_gpt_plugin().
        use_packed_input().set_kv_cache_type(_tb.KVCacheType.PAGED),
    ]

    for model_spec_obj in model_spec_obj_list:
        engine_dir = model_path / 'rt_engine' / model_name / model_spec_obj.get_model_path(
        ) / tp_pp_cp_dir
        base_output_name = os.path.splitext(
            model_spec_obj.get_results_file())[0]
        output_npy_file_name = output_dir / f'{base_output_name}.npy'
        output_csv_file_name = output_dir / f'{base_output_name}.csv'

        args_list = [
            '--engine_dir',
            str(engine_dir),
            '--tokenizer_dir',
            str(hf_path),
            '--input_file',
            str(data_input_file_name),
            '--output_npy',
            str(output_npy_file_name),
            '--output_csv',
            str(output_csv_file_name),
            '--max_output_len',
            str(max_output_len),
            '--num_beams',
            str(num_beams),
            '--use_py_session',
        ]

        if output_logits:
            file_name = str(output_npy_file_name)[:-4] + "_logits.npy"
            args_list.extend(['--output_logits_npy', file_name])

        if output_cum_log_probs:
            file_name = str(output_npy_file_name)[:-4] + "_cum_log_probs.npy"
            args_list.extend(['--output_cum_log_probs_npy', file_name])

        if output_log_probs:
            file_name = str(output_npy_file_name)[:-4] + "_log_probs.npy"
            args_list.extend(['--output_log_probs_npy', file_name])

        args = run.parse_arguments(args_list)
        run.main(args)

        # Convert pad_id to end_id in .npy out put file
        data = np.load(str(output_npy_file_name))
        if model_name == 'chatglm-6b':
            data[data == 3] = 130005
        elif model_name == 'chatglm2-6b' or model_name == 'chatglm3-6b':
            data[data == 0] = 2
        elif model_name == 'glm-10b':
            data[data == 50256] = 50258
        else:
            raise NameError('bad model name')

        np.save(str(output_npy_file_name), data)


if __name__ == '__main__':
    generate_output(model_name='chatglm-6b', num_beams=1)
    generate_output(model_name='chatglm-6b', num_beams=2)
    generate_output(model_name='chatglm2-6b', num_beams=1)
    generate_output(model_name='chatglm2-6b', num_beams=2)
    generate_output(model_name='chatglm3-6b', num_beams=1)
    generate_output(model_name='chatglm3-6b', num_beams=2)
    generate_output(model_name='glm-10b', num_beams=1)
    print("Done")
