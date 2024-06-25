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
import pathlib as _pl
import platform as _pf
import sys as _sys

from build_engines_utils import run_command, wincopy


def build_engine(weight_dir: _pl.Path, medusa_dir: _pl.Path,
                 engine_dir: _pl.Path, *args):

    covert_cmd = [_sys.executable, "examples/medusa/convert_checkpoint.py"] + (
        ['--model_dir', str(weight_dir)] if weight_dir else []) + [
            '--medusa_model_dir', str(medusa_dir), \
            '--output_dir', str(engine_dir), '--dtype=float16', '--num_medusa_heads=4'
        ] + list(args)

    run_command(covert_cmd)

    build_args = ["trtllm-build"] + (
        ['--checkpoint_dir', str(engine_dir)] if engine_dir else []) + [
            '--output_dir',
            str(engine_dir),
            '--gemm_plugin=float16',
            '--max_batch_size=8',
            '--max_input_len=12',
            '--max_seq_len=140',
            '--log_level=error',
            '--paged_kv_cache=enable',
            '--remove_input_padding=enable',
            '--speculative_decoding_mode=medusa',
        ]

    run_command(build_args)


def build_engines(model_cache: str):
    resources_dir = _pl.Path(__file__).parent.resolve().parent
    models_dir = resources_dir / 'models'
    model_name = 'vicuna-7b-v1.3'
    medusa_name = 'medusa-vicuna-7b-v1.3'

    if model_cache:
        print("Copy model from model_cache")
        model_cache_dir = _pl.Path(model_cache) / model_name
        medusa_cache_dir = _pl.Path(model_cache) / medusa_name
        assert model_cache_dir.is_dir()
        assert medusa_cache_dir.is_dir()

        if _pf.system() == "Windows":
            wincopy(source=str(model_cache_dir),
                    dest=model_name,
                    isdir=True,
                    cwd=models_dir)
            wincopy(source=str(medusa_cache_dir),
                    dest=medusa_name,
                    isdir=True,
                    cwd=models_dir)
        else:
            run_command(
                ["rsync", "-av", str(model_cache_dir), "."], cwd=models_dir)
            run_command(
                ["rsync", "-av", str(medusa_cache_dir), "."], cwd=models_dir)

    model_dir = models_dir / model_name
    medusa_dir = models_dir / medusa_name
    assert model_dir.is_dir()
    assert medusa_dir.is_dir()

    engine_dir = models_dir / 'rt_engine' / model_name

    print(f"\nBuilding fp16 engine")
    build_engine(model_dir, medusa_dir,
                 engine_dir / 'fp16-plugin-packed-paged/tp1-pp1-gpu')

    print("Done.")


if __name__ == "__main__":
    parser = _arg.ArgumentParser()
    parser.add_argument("--model_cache",
                        type=str,
                        help="Directory where models are stored")

    build_engines(**vars(parser.parse_args()))
