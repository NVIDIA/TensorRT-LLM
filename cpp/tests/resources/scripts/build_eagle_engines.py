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
import pathlib as _pl
import platform as _pf
import sys as _sys

from build_engines_utils import run_command, wincopy

import tensorrt_llm.bindings as _tb
from tensorrt_llm.bindings.internal.testing import ModelSpec


def build_engine(base_model_dir: _pl.Path, eagle_model_dir: _pl.Path,
                 engine_dir: _pl.Path, build_base_model: bool, *args):

    if build_base_model:
        checkpoint_path = "examples/models/core/llama/convert_checkpoint.py"
    else:
        checkpoint_path = "examples/eagle/convert_checkpoint.py"

    covert_cmd = [_sys.executable, checkpoint_path] + (
        ['--model_dir', str(base_model_dir)] if base_model_dir else []) + [
            '--output_dir', str(engine_dir), '--dtype=float16'
        ] + list(args)

    if not build_base_model:
        covert_cmd += [
            '--eagle_model_dir',
            str(eagle_model_dir), '--num_eagle_layers=4', '--max_draft_len=63'
        ]

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
            '--use_paged_context_fmha=enable',
        ]

    if not build_base_model:
        build_args += ['--speculative_decoding_mode=eagle']

    run_command(build_args)


def build_engines(model_cache: str):
    resources_dir = _pl.Path(__file__).parent.resolve().parent
    models_dir = resources_dir / 'models'
    model_name = 'vicuna-7b-eagle'
    base_model_name = 'vicuna-7b-v1.3'
    eagle_model_name = 'EAGLE-Vicuna-7B-v1.3'

    if model_cache:
        print(f"Copy model from {model_cache}")
        base_model_cache_dir = _pl.Path(model_cache) / base_model_name
        eagle_cache_dir = _pl.Path(model_cache) / eagle_model_name
        assert base_model_cache_dir.is_dir(), base_model_cache_dir
        assert eagle_cache_dir.is_dir(), eagle_cache_dir

        if _pf.system() == "Windows":
            wincopy(source=str(base_model_cache_dir),
                    dest=base_model_name,
                    isdir=True,
                    cwd=models_dir)
            wincopy(source=str(eagle_cache_dir),
                    dest=eagle_model_name,
                    isdir=True,
                    cwd=models_dir)
        else:
            run_command(["rsync", "-rlptD",
                         str(base_model_cache_dir), "."],
                        cwd=models_dir)
            run_command(["rsync", "-rlptD",
                         str(eagle_cache_dir), "."],
                        cwd=models_dir)

    base_model_dir = models_dir / base_model_name
    eagle_model_dir = models_dir / eagle_model_name
    assert base_model_dir.is_dir()
    assert eagle_model_dir.is_dir()

    eagle_engine_dir = models_dir / 'rt_engine' / model_name
    base_engine_dir = models_dir / 'rt_engine' / base_model_name

    model_spec_obj = ModelSpec('input_tokens.npy', _tb.DataType.HALF)
    model_spec_obj.use_gpt_plugin()
    model_spec_obj.set_kv_cache_type(_tb.KVCacheType.PAGED)
    model_spec_obj.use_packed_input()

    base_full_engine_path = base_engine_dir / model_spec_obj.get_model_path(
    ) / 'tp1-pp1-cp1-gpu'
    print(f"\nBuilding fp16 engine at {str(base_full_engine_path)}")
    build_engine(base_model_dir,
                 eagle_model_dir,
                 base_full_engine_path,
                 build_base_model=True)

    model_spec_obj = ModelSpec('input_tokens.npy', _tb.DataType.HALF)
    model_spec_obj.use_gpt_plugin()
    model_spec_obj.set_kv_cache_type(_tb.KVCacheType.PAGED)
    model_spec_obj.use_packed_input()
    model_spec_obj.use_eagle()
    eagle_full_engine_path = eagle_engine_dir / model_spec_obj.get_model_path(
    ) / 'tp1-pp1-cp1-gpu'
    print(f"\nBuilding fp16 engine at {str(eagle_full_engine_path)}")
    build_engine(base_model_dir,
                 eagle_model_dir,
                 eagle_full_engine_path,
                 build_base_model=False)

    print("Done.")


if __name__ == "__main__":
    parser = _arg.ArgumentParser()
    parser.add_argument("--model_cache",
                        type=str,
                        help="Directory where models are stored")

    build_engines(**vars(parser.parse_args()))
