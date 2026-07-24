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
import time
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
                    max_output_len: int = 8,
                    output_logits: bool = False,
                    output_cum_log_probs: bool = False,
                    output_log_probs: bool = False):

    model = 'Llama-3.2-1B'
    resources_dir = Path(__file__).parent.resolve().parent
    models_dir = resources_dir / 'models'
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

    args_list = [
        f'--engine_dir={engine_dir}',
        f'--input_file={input_file}',
        f'--tokenizer_dir={models_dir / model}',
        f'--output_npy={output_dir / (base_output_name + ".npy")}',
        f'--output_csv={output_dir / (base_output_name + ".csv")}',
        f'--max_output_len={max_output_len}',
        f'--num_beams={num_beams}',
        '--use_py_session',
    ]

    if output_logits:
        args_list.extend([
            f'--output_logits_npy={output_dir / (base_output_name + "_logits.npy")}',
            '--output_generation_logits',
        ])

    if output_cum_log_probs:
        args_list.extend([
            f'--output_cum_log_probs_npy={output_dir / model_spec_obj.get_cum_log_probs_file()}'
        ])

    if output_log_probs:
        args_list.extend([
            f'--output_log_probs_npy={output_dir / model_spec_obj.get_log_probs_file()}'
        ])

    args = run.parse_arguments(args_list)
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
            f'Generating outputs for Llama FP16 with TP={tp_size}, PP={pp_size}, CP={cp_size}, BW={num_beams}'
        )
        start_time = time.time()

        output_logits = False
        output_log_probs = False
        output_cum_log_probs = False
        if tp_size == 4 and pp_size == 1:
            output_logits = True
            output_log_probs = True
            output_cum_log_probs = True

        model_spec_obj.use_tensor_parallelism(tp_size)
        model_spec_obj.use_pipeline_parallelism(pp_size)
        model_spec_obj.use_context_parallelism(cp_size)
        generate_output(engine=model_spec_obj.get_model_path(),
                        num_beams=num_beams,
                        tp_size=tp_size,
                        pp_size=pp_size,
                        cp_size=cp_size,
                        model_spec_obj=model_spec_obj,
                        output_logits=output_logits,
                        output_log_probs=output_log_probs,
                        output_cum_log_probs=output_cum_log_probs)

        duration = time.time() - start_time
        print(
            f"Generating outputs for Llama FP16 with TP={tp_size}, PP={pp_size}, CP={cp_size}, BW={num_beams} took {duration} seconds"
        )


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
