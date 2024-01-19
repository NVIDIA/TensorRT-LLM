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


def build_engine(weight_dir: _pl.Path, engine_dir: _pl.Path, *args):
    build_args = [_sys.executable, "examples/llama/build.py"] + (
        ['--model_dir', str(weight_dir)] if weight_dir else []) + [
            '--output_dir',
            str(engine_dir), '--dtype=float16',
            '--use_gpt_attention_plugin=float16', '--use_custom_all_reduce',
            '--use_gemm_plugin=float16', '--max_batch_size=32',
            '--max_input_len=40', '--max_output_len=20', '--max_beam_width=2',
            '--log_level=error', '--use_inflight_batching', '--paged_kv_cache',
            '--remove_input_padding'
        ] + list(args)
    run_command(build_args)


def build_engines(model_cache: str, only_multi_gpu: bool):
    resources_dir = _pl.Path(__file__).parent.resolve().parent
    models_dir = resources_dir / 'models'
    model_name = 'llama-7b-hf'

    if model_cache:
        print("Copy model from model_cache")
        model_cache_dir = _pl.Path(model_cache) / 'llama-models' / model_name
        assert (model_cache_dir.is_dir())

        if _pf.system() == "Windows":
            wincopy(source=str(model_cache_dir),
                    dest=model_name,
                    isdir=True,
                    cwd=models_dir)
        else:
            run_command(
                ["rsync", "-av", str(model_cache_dir), "."], cwd=models_dir)

    hf_dir = models_dir / model_name
    assert hf_dir.is_dir()

    engine_dir = models_dir / 'rt_engine' / model_name

    tp_pp_sizes = [(1, 1)]
    if only_multi_gpu:
        tp_pp_sizes = [(1, 4), (4, 1), (2, 2)]
    for tp_size, pp_size in tp_pp_sizes:
        tp_pp_dir = f"tp{tp_size}-pp{pp_size}-gpu"
        world_size = tp_size * pp_size
        print(f"\nBuilding fp16 tp{tp_size} pp{pp_size} engine")
        build_engine(hf_dir,
                     engine_dir / f'fp16-plugin-packed-paged/{tp_pp_dir}',
                     f'--world_size={world_size}', f'--tp_size={tp_size}',
                     f'--pp_size={pp_size}')

    print("Done.")


if __name__ == "__main__":
    parser = _arg.ArgumentParser()
    parser.add_argument("--model_cache",
                        type=str,
                        help="Directory where models are stored")
    parser.add_argument(
        "--only_multi_gpu",
        action="store_true",
        help="Flag to build only for Tensor and Pipeline parallelism")

    build_engines(**vars(parser.parse_args()))
