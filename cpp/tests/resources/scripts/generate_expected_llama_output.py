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
from pathlib import Path

import run


def generate_output(engine: str,
                    num_beams: int,
                    output_name: str,
                    tp_size: int = 1,
                    pp_size: int = 1,
                    max_output_len: int = 8):

    model = 'llama-7b-hf'
    resources_dir = Path(__file__).parent.resolve().parent
    models_dir = resources_dir / 'models'
    hf_dir = models_dir / model
    tp_pp_dir = 'tp' + str(tp_size) + '-pp' + str(pp_size) + '-gpu/'
    engine_dir = models_dir / 'rt_engine' / model / engine / tp_pp_dir

    data_dir = resources_dir / 'data'
    input_file = data_dir / 'input_tokens.npy'
    model_data_dir = data_dir / model
    if num_beams <= 1:
        output_dir = model_data_dir / 'sampling'
    else:
        output_dir = model_data_dir / ('beam_search_' + str(num_beams))

    output_name += '_tp' + str(tp_size) + '_pp' + str(pp_size)

    args = run.parse_arguments([
        '--engine_dir',
        str(engine_dir), '--input_file',
        str(input_file), '--tokenizer_dir',
        str(hf_dir), '--output_npy',
        str(output_dir / (output_name + '.npy')), '--output_csv',
        str(output_dir / (output_name + '.csv')), '--max_output_len',
        str(max_output_len), '--num_beams',
        str(num_beams), '--use_py_session'
    ])
    run.main(args)


def generate_outputs(num_beams, only_multi_gpu=False):
    tp_pp_sizes = [(1, 1)] if not only_multi_gpu else [(4, 1), (2, 2), (1, 4)]
    for tp_size, pp_size in tp_pp_sizes:
        print(
            f'Generating outputs for Llama FP16 with TP={tp_size} and PP={pp_size}'
        )
        generate_output(engine='fp16-plugin-packed-paged',
                        num_beams=num_beams,
                        tp_size=tp_size,
                        pp_size=pp_size,
                        output_name='output_tokens_fp16_plugin_packed_paged')


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
