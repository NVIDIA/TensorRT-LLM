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

import argparse as _arg
import os
from pathlib import Path

# isort: off
import run
# isort: on

import tensorrt_llm.bindings as _tb
from tensorrt_llm.bindings.internal.testing import ModelSpec


def generate_output(engine: str,
                    model_spec_obj: ModelSpec,
                    max_output_len: int = 8):

    model = 'vicuna-7b-medusa'
    hf_model = 'vicuna-7b-v1.3'
    resources_dir = Path(__file__).parent.resolve().parent
    models_dir = resources_dir / 'models'
    hf_dir = models_dir / hf_model
    tp_pp_dir = 'tp1-pp1-cp1-gpu/'
    engine_dir = models_dir / 'rt_engine' / model / engine / tp_pp_dir

    data_dir = resources_dir / 'data'
    input_file = data_dir / 'input_vicuna.npy'
    model_data_dir = data_dir / model
    output_dir = model_data_dir / 'sampling'

    base_output_name = os.path.splitext(model_spec_obj.get_results_file())[0]

    args = run.parse_arguments([
        '--engine_dir',
        str(engine_dir), '--input_file',
        str(input_file), '--tokenizer_dir',
        str(hf_dir), '--output_npy',
        str(output_dir / (base_output_name + '.npy')), '--output_csv',
        str(output_dir / (base_output_name + '.csv')), '--max_output_len',
        str(max_output_len), '--use_py_session',
        '--medusa_choices=[[0], [0, 0], [1], [0, 1], [2], [0, 0, 0], [1, 0], [0, 2], [3], [0, 3], [4], [0, 4], [2, 0], [0, 5], [0, 0, 1], [5], [0, 6], [6], [0, 7], [0, 1, 0], [1, 1], [7], [0, 8], [0, 0, 2], [3, 0], [0, 9], [8], [9], [1, 0, 0], [0, 2, 0], [1, 2], [0, 0, 3], [4, 0], [2, 1], [0, 0, 4], [0, 0, 5], [0, 0, 0, 0], [0, 1, 1], [0, 0, 6], [0, 3, 0], [5, 0], [1, 3], [0, 0, 7], [0, 0, 8], [0, 0, 9], [6, 0], [0, 4, 0], [1, 4], [7, 0], [0, 1, 2], [2, 0, 0], [3, 1], [2, 2], [8, 0], [0, 5, 0], [1, 5], [1, 0, 1], [0, 2, 1], [9, 0], [0, 6, 0], [0, 0, 0, 1], [1, 6], [0, 7, 0]]',
        '--temperature', '1.0'
    ])
    run.main(args)
    print(f"Output saved at {str(output_dir / base_output_name)}.[npy|csv]")


def generate_outputs():
    print(f'Generating outputs for Medusa FP16')
    max_output_len = 128
    model_spec_obj = ModelSpec('input_tokens_long.npy', _tb.DataType.HALF)
    model_spec_obj.use_gpt_plugin()
    model_spec_obj.set_max_output_length(max_output_len)
    model_spec_obj.use_packed_input()
    model_spec_obj.set_kv_cache_type(_tb.KVCacheType.PAGED)
    model_spec_obj.use_medusa()

    generate_output(engine=model_spec_obj.get_model_path(),
                    model_spec_obj=model_spec_obj,
                    max_output_len=max_output_len)


if __name__ == '__main__':
    parser = _arg.ArgumentParser()
    parser.add_argument(
        "--only_multi_gpu",
        action="store_true",
        help="Generate data with Pipeline and Tensor Parallelism")

    args = parser.parse_args()

    generate_outputs()
    print("Done")
