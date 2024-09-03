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

from build_engines_utils import run_command, wincopy


def build_engine(weight_dir: _pl.Path, engine_dir: _pl.Path,
                 has_tllm_checkpoint: bool):

    if not has_tllm_checkpoint:
        raise RuntimeError(
            'Convert checkpoint is not supported for ReDrafter. '
            'Provide a path that contains a checkpoint in the tllm_ckpt folder and set --has_tllm_checkpoint flag'
        )
    else:
        checkpoint_dir = weight_dir / 'tllm_ckpt'

    build_args = ["trtllm-build"] + (
        ['--checkpoint_dir', str(checkpoint_dir)] if engine_dir else []) + [
            '--output_dir',
            str(engine_dir),
            '--gemm_plugin=float16',
            '--max_batch_size=8',
            '--max_input_len=64',
            '--max_seq_len=1024',
            '--log_level=error',
            '--paged_kv_cache=enable',
            '--remove_input_padding=enable',
            '--speculative_decoding_mode=explicit_draft_tokens',
        ]

    run_command(build_args)


def build_engines(model_cache: str, has_tllm_checkpoint: bool):
    resources_dir = _pl.Path(__file__).parent.resolve().parent
    models_dir = resources_dir / 'models'
    model_name = 'vicuna_redrafter'
    if has_tllm_checkpoint:
        base_model_name = 'vicuna-7b-v1.3'
    # FIXME(nkorobov): rename folder in the cache
    # model_name = 'redrafter-vicuna-7b-v1.3'

    if model_cache:
        print("Copy model from model_cache")
        model_cache_dir = _pl.Path(model_cache) / model_name
        assert model_cache_dir.is_dir()
        if has_tllm_checkpoint:
            base_model_cache_dir = _pl.Path(model_cache) / base_model_name
            assert base_model_cache_dir.is_dir()

        if _pf.system() == "Windows":
            wincopy(source=str(model_cache_dir),
                    dest=model_name,
                    isdir=True,
                    cwd=models_dir)
            if has_tllm_checkpoint:
                wincopy(source=str(base_model_cache_dir),
                        dest=base_model_name,
                        isdir=True,
                        cwd=models_dir)
        else:
            run_command(["rsync", "-rlptD",
                         str(model_cache_dir), "."],
                        cwd=models_dir)
            if has_tllm_checkpoint:
                run_command(["rsync", "-rlptD",
                             str(base_model_cache_dir), "."],
                            cwd=models_dir)

    model_dir = models_dir / model_name
    assert model_dir.is_dir()

    engine_dir = models_dir / 'rt_engine' / model_name

    print(f"\nBuilding fp16 engine")
    build_engine(model_dir, engine_dir / 'fp16-plugin-packed-paged/tp1-pp1-gpu',
                 has_tllm_checkpoint)

    print("Done.")


if __name__ == "__main__":
    parser = _arg.ArgumentParser()
    parser.add_argument("--model_cache",
                        type=str,
                        help="Directory where models are stored")
    parser.add_argument(
        "--has_tllm_checkpoint",
        action='store_true',
        help="True if the provided path contains the trt-llm checkpoint.")

    build_engines(**vars(parser.parse_args()))
