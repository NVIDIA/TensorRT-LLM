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
import os as _os
import pathlib as _pl
import platform as _pf
import sys as _sys
import typing as _tp

from build_engines_utils import run_command, wincopy


def build_engine(weight_dir: _pl.Path, ckpt_dir: _pl.Path, engine_dir: _pl.Path,
                 *args):
    convert_args = [_sys.executable, "examples/mamba/convert_checkpoint.py"] + (
        ['--model_dir', str(weight_dir)] if weight_dir else []) + [
            '--output_dir',
            str(ckpt_dir),
            '--dtype=float16',
        ]
    run_command(convert_args)
    build_args = ["trtllm-build"] + ['--checkpoint_dir',
                                     str(ckpt_dir)] + [
                                         '--output_dir',
                                         str(engine_dir),
                                         '--gpt_attention_plugin=disable',
                                         '--paged_kv_cache=disable',
                                         '--gemm_plugin=disable',
                                         '--max_batch_size=8',
                                         '--max_input_len=924',
                                         '--max_seq_len=1024',
                                         '--max_beam_width=1',
                                     ] + list(args)
    run_command(build_args)


def build_engines(model_cache: _tp.Optional[str] = None):
    resources_dir = _pl.Path(__file__).parent.resolve().parent
    models_dir = resources_dir / 'models'
    model_name = 'mamba-2.8b-hf'

    if model_cache:
        print("Copy model from model_cache")
        model_cache_dir = _pl.Path(model_cache) / 'mamba' / model_name
        if _pf.system() == "Windows":
            wincopy(source=str(model_cache_dir),
                    dest=model_name,
                    isdir=True,
                    cwd=models_dir)
        else:
            run_command(
                ["rsync", "-av", str(model_cache_dir), "."], cwd=models_dir)
    else:
        print("Clone model from HF")
        hf_dir = _pl.Path(models_dir) / model_name
        run_command(
            [
                "git", "clone",
                "https://huggingface.co/state-spaces/mamba-2.8b-hf", model_name
            ],
            cwd=models_dir,
        )
    hf_dir = models_dir / model_name
    assert (hf_dir.is_dir())

    # Clone or update the tokenizer directory without lfs
    tokenizer_name = 'gpt-neox-20b'
    tokenizer_hf_dir = models_dir / tokenizer_name
    if tokenizer_hf_dir.exists():
        assert tokenizer_hf_dir.is_dir()
        run_command(["git", "pull"], cwd=tokenizer_hf_dir)
    else:
        if _pf.system() == "Windows":
            url_prefix = ""
        else:
            url_prefix = "file://"
        tokenizer_url = url_prefix + str(
            _pl.Path(model_cache) / tokenizer_name
        ) if model_cache else "https://huggingface.co/EleutherAI/gpt-neox-20b"
        run_command([
            "git", "clone", tokenizer_url, "--single-branch", "--no-local",
            tokenizer_name
        ],
                    cwd=tokenizer_hf_dir.parent,
                    env={
                        **_os.environ, "GIT_LFS_SKIP_SMUDGE": "1"
                    })

    tp_size = 1
    pp_size = 1
    tp_pp_dir = f"tp{tp_size}-pp{pp_size}-gpu"

    ckpt_dir = models_dir / 'rt_ckpt' / model_name
    engine_dir = models_dir / 'rt_engine' / model_name

    print("\nBuilding fp16 engine")
    build_engine(hf_dir, ckpt_dir / 'fp16-default' / tp_pp_dir,
                 engine_dir / 'fp16-default' / tp_pp_dir,
                 '--remove_input_padding=disable', '--paged_state=disable',
                 '--mamba_conv1d_plugin=disable')
    print("\nBuilding fp16-plugin engine")
    build_engine(hf_dir, ckpt_dir / 'fp16-plugin' / tp_pp_dir,
                 engine_dir / 'fp16-plugin' / tp_pp_dir,
                 '--remove_input_padding=disable', '--paged_state=disable')
    print("\nBuilding fp16-plugin-packed engine")
    build_engine(hf_dir, ckpt_dir / 'fp16-plugin-packed' / tp_pp_dir,
                 engine_dir / 'fp16-plugin-packed' / tp_pp_dir,
                 '--remove_input_padding=enable', '--paged_state=disable')
    print("\nBuilding fp16-plugin-packed-paged engine")
    build_engine(hf_dir, ckpt_dir / 'fp16-plugin-packed-paged' / tp_pp_dir,
                 engine_dir / 'fp16-plugin-packed-paged' / tp_pp_dir,
                 '--remove_input_padding=enable', '--paged_state=enable')
    print("Done.")


if __name__ == "__main__":
    parser = _arg.ArgumentParser()
    parser.add_argument("--model_cache",
                        type=str,
                        help="Directory where models are stored")

    build_engines(**vars(parser.parse_args()))
