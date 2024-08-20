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

import argparse
from pathlib import Path

import run
from build_engines_utils import init_model_spec_module

init_model_spec_module()

import os
import shutil

import model_spec

import tensorrt_llm.bindings as _tb


def get_model_data_dir():
    resources_dir = Path(__file__).parent.resolve().parent
    data_dir = resources_dir / 'data'
    return data_dir / 'gpt2'


def generate_output(engine: str,
                    num_beams: int,
                    input_name: str,
                    model_spec_obj: model_spec.ModelSpec,
                    max_output_len: int = 8,
                    output_logits: bool = False,
                    output_cum_log_probs: bool = False,
                    output_log_probs: bool = False):
    tp_size = 1
    pp_size = 1
    model = 'gpt2'
    resources_dir = Path(__file__).parent.resolve().parent
    models_dir = resources_dir / 'models'
    tp_pp_dir = 'tp' + str(tp_size) + '-pp' + str(pp_size) + '-gpu/'
    engine_dir = models_dir / 'rt_engine' / model / engine / tp_pp_dir

    data_dir = resources_dir / 'data'
    input_file = data_dir / input_name
    model_data_dir = get_model_data_dir()
    if num_beams <= 1:
        output_dir = model_data_dir / 'sampling'
    else:
        output_dir = model_data_dir / ('beam_search_' + str(num_beams))

    model_spec_obj.use_tensor_parallelism(tp_size).use_pipeline_parallelism(
        pp_size)

    base_output_name = os.path.splitext(model_spec_obj.get_results_file())[0]

    output_logits_npy = None
    if output_logits:
        logits_file = base_output_name + '_logits.npy'
        output_logits_npy = str(output_dir / logits_file)

    results_file = str(output_dir / (base_output_name + '.npy'))
    results_csv = str(output_dir / (base_output_name + '.csv'))

    args_list = [
        '--engine_dir',
        str(engine_dir), '--input_file',
        str(input_file), '--tokenizer_dir',
        str(models_dir / model), '--output_npy', results_file, '--output_csv',
        results_csv, '--max_output_len',
        str(max_output_len), '--num_beams',
        str(num_beams), '--output_logits_npy',
        str(output_logits_npy), '--use_py_session'
    ]

    # Generate context_fmha_fp32_acc enabled results for GptExecutorTest.GenerationLogitsEarlyStop
    if model_spec_obj.get_enable_context_fmha_fp32_acc():
        args_list.extend(["--enable_context_fmha_fp32_acc"])

    assert not os.path.exists(results_file) and not os.path.exists(results_csv)

    if output_cum_log_probs:
        args_list.extend([
            '--output_cum_log_probs_npy',
            f'{output_dir / model_spec_obj.get_cum_log_probs_file()}'
        ])

    if output_log_probs:
        args_list.extend([
            '--output_log_probs_npy',
            f'{output_dir / model_spec_obj.get_log_probs_file()}'
        ])

    args = run.parse_arguments(args_list)
    run.main(args)


def generate_outputs(num_beams):
    input_name = 'input_tokens.npy'
    input_name_long = 'input_tokens_long.npy'

    print('Generating GPT2 FP32 outputs')
    model_spec_obj = model_spec.ModelSpec(input_name, _tb.DataType.FLOAT)
    model_spec_obj.set_kv_cache_type(_tb.KVCacheType.CONTINUOUS)
    if num_beams == 1:
        generate_output(engine=model_spec_obj.get_model_path(),
                        num_beams=num_beams,
                        input_name=input_name,
                        model_spec_obj=model_spec_obj)
    model_spec_obj.use_gpt_plugin()
    generate_output(engine=model_spec_obj.get_model_path(),
                    num_beams=num_beams,
                    input_name=input_name,
                    model_spec_obj=model_spec_obj)

    print('Generating GPT2 FP16 outputs')
    model_spec_obj = model_spec.ModelSpec(input_name, _tb.DataType.HALF)
    model_spec_obj.set_kv_cache_type(_tb.KVCacheType.CONTINUOUS)
    if num_beams == 1:
        generate_output(engine=model_spec_obj.get_model_path(),
                        num_beams=num_beams,
                        input_name=input_name,
                        model_spec_obj=model_spec_obj)
    model_spec_obj.use_gpt_plugin()
    generate_output(engine=model_spec_obj.get_model_path(),
                    num_beams=num_beams,
                    input_name=input_name,
                    model_spec_obj=model_spec_obj)
    model_spec_obj.use_packed_input()
    generate_output(engine=model_spec_obj.get_model_path(),
                    num_beams=num_beams,
                    input_name=input_name,
                    model_spec_obj=model_spec_obj)
    model_spec_obj.set_kv_cache_type(_tb.KVCacheType.PAGED)
    model_spec_obj.gather_logits()
    generate_output(engine=model_spec_obj.get_model_path(),
                    num_beams=num_beams,
                    input_name=input_name,
                    model_spec_obj=model_spec_obj,
                    output_logits=True,
                    output_log_probs=True,
                    output_cum_log_probs=True)
    # GptExecutorTest.GenerationLogitsEarlyStop requires to use context_fmha_fp32_acc flag in runtime
    model_spec_obj.enable_context_fmha_fp32_acc()
    generate_output(engine=model_spec_obj.get_model_path(),
                    num_beams=num_beams,
                    input_name=input_name,
                    model_spec_obj=model_spec_obj,
                    output_logits=True,
                    output_log_probs=True,
                    output_cum_log_probs=True)

    model_spec_obj = model_spec.ModelSpec(input_name, _tb.DataType.HALF)
    model_spec_obj.use_gpt_plugin()
    model_spec_obj.set_kv_cache_type(_tb.KVCacheType.PAGED)
    model_spec_obj.use_packed_input()
    generate_output(engine=model_spec_obj.get_model_path(),
                    num_beams=num_beams,
                    input_name=input_name,
                    model_spec_obj=model_spec_obj,
                    output_logits=False,
                    output_log_probs=True,
                    output_cum_log_probs=True)
    model_spec_obj.set_max_output_length(128)
    generate_output(engine=model_spec_obj.get_model_path(),
                    num_beams=num_beams,
                    input_name=input_name,
                    model_spec_obj=model_spec_obj,
                    output_logits=False,
                    max_output_len=128)

    model_spec_obj = model_spec.ModelSpec(input_name_long, _tb.DataType.HALF)
    model_spec_obj.use_gpt_plugin()
    model_spec_obj.use_packed_input()
    model_spec_obj.set_kv_cache_type(_tb.KVCacheType.PAGED)
    generate_output(engine=model_spec_obj.get_model_path(),
                    num_beams=num_beams,
                    input_name=input_name_long,
                    model_spec_obj=model_spec_obj,
                    output_logits=False)

    model_spec_obj = model_spec.ModelSpec(input_name, _tb.DataType.HALF)
    model_spec_obj.use_gpt_plugin()
    model_spec_obj.use_packed_input()
    model_spec_obj.set_kv_cache_type(_tb.KVCacheType.PAGED)
    model_spec_obj.set_quant_method(model_spec.QuantMethod.SMOOTH_QUANT)
    generate_output(engine=model_spec_obj.get_model_path(),
                    num_beams=num_beams,
                    input_name=input_name,
                    model_spec_obj=model_spec_obj,
                    output_logits=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--clean',
                        action='store_true',
                        default=False,
                        help='Clean target folders before building engines')
    args = parser.parse_args()
    if args.clean:
        model_data_dir = get_model_data_dir()
        print(f'Cleaning target folder {model_data_dir}')
        shutil.rmtree(model_data_dir, ignore_errors=True)
    generate_outputs(num_beams=1)
    generate_outputs(num_beams=2)
