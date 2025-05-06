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

# isort: off
import run
# isort: on

import tensorrt_llm.bindings as _tb
from tensorrt_llm.bindings.internal.testing import ModelSpec


def generate_output(engine: str,
                    num_beams: int,
                    input_name: str,
                    model_spec_obj: ModelSpec,
                    max_output_len: int = 8,
                    output_logits: bool = False):
    tp_size = 1
    pp_size = 1
    cp_size = 1
    model = 'mamba-2.8b-hf'
    resources_dir = Path(__file__).parent.resolve().parent
    models_dir = resources_dir / 'models'
    tp_pp_cp_dir = 'tp' + str(tp_size) + '-pp' + str(pp_size) + '-cp' + str(
        cp_size) + '-gpu/'
    engine_dir = models_dir / 'rt_engine' / model / engine / tp_pp_cp_dir

    data_dir = resources_dir / 'data'
    input_file = data_dir / input_name
    model_data_dir = data_dir / model
    if num_beams <= 1:
        output_dir = model_data_dir / 'sampling'
    else:
        output_dir = model_data_dir / ('beam_search_' + str(num_beams))

    base_output_name = os.path.splitext(model_spec_obj.get_results_file())[0]

    output_logits_npy = None
    if output_logits:
        output_logits_npy = str(output_dir /
                                (base_output_name + '_logits' + '.npy'))

    args = run.parse_arguments([
        '--engine_dir',
        str(engine_dir), '--input_file',
        str(input_file), '--tokenizer_dir',
        str(models_dir / 'gpt-neox-20b'), '--output_npy',
        str(output_dir / (base_output_name + '.npy')), '--output_csv',
        str(output_dir / (base_output_name + '.csv')), '--max_output_len',
        str(max_output_len), '--num_beams',
        str(num_beams), '--output_logits_npy',
        str(output_logits_npy), '--use_py_session'
    ])
    run.main(args)


def generate_outputs(num_beams):
    print('Generating Mamba FP16 outputs')
    input_name = 'input_tokens.npy'
    model_spec_obj = ModelSpec(input_name, _tb.DataType.HALF)
    model_spec_obj.set_kv_cache_type(_tb.KVCacheType.CONTINUOUS)

    generate_output(engine=model_spec_obj.get_model_path(),
                    num_beams=num_beams,
                    input_name=input_name,
                    model_spec_obj=model_spec_obj)

    print('Generating Mamba FP16-plugin outputs')
    model_spec_obj.use_gpt_plugin()
    generate_output(engine=model_spec_obj.get_model_path(),
                    num_beams=num_beams,
                    input_name=input_name,
                    model_spec_obj=model_spec_obj)

    print('Generating Mamba FP16-plugin-packed outputs')
    model_spec_obj.use_packed_input()
    generate_output(engine=model_spec_obj.get_model_path(),
                    num_beams=num_beams,
                    input_name=input_name,
                    model_spec_obj=model_spec_obj)

    print('Generating Mamba FP16-plugin-packed-paged outputs')
    model_spec_obj.set_kv_cache_type(_tb.KVCacheType.PAGED)
    generate_output(engine=model_spec_obj.get_model_path(),
                    num_beams=num_beams,
                    input_name=input_name,
                    model_spec_obj=model_spec_obj)


if __name__ == '__main__':
    generate_outputs(num_beams=1)
