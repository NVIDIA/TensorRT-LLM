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

from mpi4py.MPI import COMM_WORLD

# isort: off
import run
# isort: on

import tensorrt_llm.bindings as _tb
from tensorrt_llm.bindings.internal.testing import ModelSpec


def generate_output(engine: str,
                    num_beams: int,
                    model_spec_obj: ModelSpec,
                    tp_size: int = 1,
                    pp_size: int = 1,
                    cp_size: int = 1,
                    max_output_len: int = 8):

    model = 'Llama-3.2-1B'
    resources_dir = Path(__file__).parent.resolve().parent
    models_dir = resources_dir / 'models'
    hf_dir = models_dir / model
    tp_pp_cp_dir = 'tp' + str(tp_size) + '-pp' + str(pp_size) + '-cp' + str(
        cp_size) + '-gpu/'
    engine_dir = models_dir / 'rt_engine' / model / engine / tp_pp_cp_dir

    data_dir = resources_dir / 'data'
    input_file = data_dir / 'input_tokens_llama.npy'
    model_data_dir = data_dir / model
    if num_beams <= 1:
        output_dir = model_data_dir / 'sampling'
    else:
        output_dir = model_data_dir / ('beam_search_' + str(num_beams))

    base_output_name = os.path.splitext(model_spec_obj.get_results_file())[0]

    args = run.parse_arguments([
        '--engine_dir',
        str(engine_dir), '--input_file',
        str(input_file), '--tokenizer_dir',
        str(hf_dir), '--output_npy',
        str(output_dir / (base_output_name + '.npy')), '--output_csv',
        str(output_dir / (base_output_name + '.csv')), '--max_output_len',
        str(max_output_len), '--num_beams',
        str(num_beams), '--use_py_session'
    ])
    run.main(args)


def generate_outputs(num_beams, only_multi_gpu=False):
    if not only_multi_gpu:
        tp_pp_cp_sizes = [(1, 1, 1)]
    elif COMM_WORLD.size == 4:
        tp_pp_cp_sizes = [(4, 1, 1), (2, 2, 1), (1, 4, 1)]
    elif COMM_WORLD.size == 2:
        tp_pp_cp_sizes = [(1, 2, 1), (2, 1, 1)]
    else:
        raise RuntimeError(
            f"The world size of MPI {COMM_WORLD.size} is not equal to 1, 2, or 4."
        )
    model_spec_obj = ModelSpec('input_tokens_llama.npy', _tb.DataType.HALF)
    model_spec_obj.use_gpt_plugin()
    model_spec_obj.set_kv_cache_type(_tb.KVCacheType.PAGED)
    model_spec_obj.use_packed_input()

    for tp_size, pp_size, cp_size in tp_pp_cp_sizes:
        print(
            f'Generating outputs for Llama FP16 with TP={tp_size}, PP={pp_size} and CP={cp_size}'
        )
        model_spec_obj.use_tensor_parallelism(tp_size)
        model_spec_obj.use_pipeline_parallelism(pp_size)
        model_spec_obj.use_context_parallelism(cp_size)
        generate_output(engine=model_spec_obj.get_model_path(),
                        num_beams=num_beams,
                        tp_size=tp_size,
                        pp_size=pp_size,
                        cp_size=cp_size,
                        model_spec_obj=model_spec_obj)


if __name__ == '__main__':
    parser = _arg.ArgumentParser()
    parser.add_argument(
        "--only_multi_gpu",
        action="store_true",
        help="Generate data with Pipeline and Tensor Parallelism")

    args = parser.parse_args()

    generate_outputs(num_beams=1, only_multi_gpu=args.only_multi_gpu)
    generate_outputs(num_beams=2, only_multi_gpu=args.only_multi_gpu)
    print("Done")
