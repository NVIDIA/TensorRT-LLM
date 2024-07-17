#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from pathlib import Path

import run


def generate_output(engine: str, output_name: str, max_output_len: int = 8):

    model = 'vicuna-explicit_draft'
    resources_dir = Path(__file__).parent.resolve().parent
    models_dir = resources_dir / 'models'
    hf_dir = models_dir / 'vicuna-7b-v1.3'
    tp_pp_dir = 'tp1-pp1-gpu/'
    engine_dir = models_dir / 'rt_engine' / model / engine / tp_pp_dir

    data_dir = resources_dir / 'data'
    input_file = data_dir / 'input_tokens.npy'
    model_data_dir = data_dir / model
    output_dir = model_data_dir / 'sampling'

    output_name += '_tp1_pp1'

    args = run.parse_arguments([
        '--engine_dir',
        str(engine_dir),
        '--input_file',
        str(input_file),
        '--tokenizer_dir',
        str(hf_dir),
        '--output_npy',
        str(output_dir / (output_name + '.npy')),
        '--output_csv',
        str(output_dir / (output_name + '.csv')),
        '--max_output_len',
        str(max_output_len),
        '--use_py_session',
    ])
    run.main(args)


def generate_outputs():
    print(f'Generating outputs for FP16')
    generate_output(engine='fp16-plugin-packed-paged',
                    output_name='output_tokens_fp16_plugin_packed_paged',
                    max_output_len=128)


if __name__ == '__main__':
    parser = _arg.ArgumentParser()
    args = parser.parse_args()

    generate_outputs()
    print("Done")
