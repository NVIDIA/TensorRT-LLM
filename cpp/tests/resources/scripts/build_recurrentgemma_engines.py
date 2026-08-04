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

import tensorrt_llm.bindings as _tb
from tensorrt_llm.bindings.internal.testing import ModelSpec


def build_engine(weight_dir: _pl.Path, ckpt_dir: _pl.Path, engine_dir: _pl.Path,
                 *args):
    convert_args = [
        _sys.executable,
        "examples/models/core/recurrentgemma/convert_checkpoint.py"
    ] + (['--model_dir', str(weight_dir)] if weight_dir else []) + [
        '--output_dir',
        str(ckpt_dir),
        '--ckpt_type=hf',
        '--dtype=float16',
    ]
    run_command(convert_args)
    build_args = ["trtllm-build"] + ['--checkpoint_dir',
                                     str(ckpt_dir)] + [
                                         '--output_dir',
                                         str(engine_dir),
                                         '--gpt_attention_plugin=float16',
                                         '--paged_kv_cache=enable',
                                         '--gemm_plugin=float16',
                                         '--max_batch_size=8',
                                         '--max_input_len=924',
                                         '--max_seq_len=1024',
                                         '--max_beam_width=1',
                                     ] + list(args)
    run_command(build_args)


def build_engines(model_cache: _tp.Optional[str] = None):
    resources_dir = _pl.Path(__file__).parent.resolve().parent
    models_dir = resources_dir / 'models'
    model_name = 'recurrentgemma-2b'
    hf_dir = models_dir / model_name

    # Clone or update the model directory without lfs
    if model_cache:
        print("Copy model from model_cache")
        model_cache_dir = _pl.Path(model_cache) / 'recurrentgemma' / model_name
        print(model_cache_dir)
        assert (model_cache_dir.is_dir())
        if _pf.system() == "Windows":
            wincopy(source=str(model_cache_dir),
                    dest=model_name,
                    isdir=True,
                    cwd=models_dir)
        else:
            run_command(["rsync", "-rlptD",
                         str(model_cache_dir), "."],
                        cwd=models_dir)
    else:
        if not hf_dir.is_dir():
            if _pf.system() == "Windows":
                url_prefix = ""
            else:
                url_prefix = "file://"
            model_url = "https://huggingface.co/google/recurrentgemma-2b"
            run_command([
                "git", "clone", model_url, "--single-branch", "--no-local",
                model_name
            ],
                        cwd=models_dir,
                        env={
                            **_os.environ, "GIT_LFS_SKIP_SMUDGE": "1"
                        })

    assert (hf_dir.is_dir())

    # Download the model file
    model_file_name = "*"
    if not model_cache:
        run_command(["git", "lfs", "pull", "--include", model_file_name],
                    cwd=hf_dir)

    tp_size = 1
    pp_size = 1
    cp_size = 1
    tp_pp_cp_dir = f"tp{tp_size}-pp{pp_size}-cp{cp_size}-gpu"

    ckpt_dir = models_dir / 'rt_ckpt' / model_name
    engine_dir = models_dir / 'rt_engine' / model_name

    python_exe = _sys.executable
    run_command([python_exe, "-m", "pip", "install", "transformers>=4.40.0"],
                env=_os.environ,
                timeout=300)
    input_file = 'input_tokens.npy'
    model_spec_obj = ModelSpec(input_file, _tb.DataType.HALF)
    model_spec_obj.use_gpt_plugin()
    model_spec_obj.use_packed_input()
    model_spec_obj.set_kv_cache_type(_tb.KVCacheType.PAGED)

    print("\nBuilding fp16-plugin-packed-paged engine")
    build_engine(hf_dir,
                 ckpt_dir / model_spec_obj.get_model_path() / tp_pp_cp_dir,
                 engine_dir / model_spec_obj.get_model_path() / tp_pp_cp_dir,
                 '--remove_input_padding=enable', '--paged_state=enable')

    print("Done.")


if __name__ == "__main__":
    parser = _arg.ArgumentParser()
    parser.add_argument("--model_cache",
                        type=str,
                        help="Directory where models are stored")

    build_engines(**vars(parser.parse_args()))
