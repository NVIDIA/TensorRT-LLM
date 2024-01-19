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


def build_engine(weight_dir: _pl.Path, engine_dir: _pl.Path, *args):
    build_args = [_sys.executable, "examples/gptj/build.py"] + (
        ['--model_dir', str(weight_dir)] if weight_dir else []) + [
            '--output_dir',
            str(engine_dir),
            '--dtype=float16',
            '--logits_dtype=float16',
            '--use_gemm_plugin=float16',
            '--max_batch_size=32',
            '--max_input_len=40',
            '--max_output_len=20',
            '--max_beam_width=2',
            '--log_level=error',
        ] + list(args)
    run_command(build_args)


def build_engines(model_cache: _tp.Optional[str] = None, only_fp8=False):
    resources_dir = _pl.Path(__file__).parent.resolve().parent
    models_dir = resources_dir / 'models'
    model_name = 'gpt-j-6b'

    # Clone or update the model directory without lfs
    hf_dir = models_dir / model_name
    if hf_dir.exists():
        assert hf_dir.is_dir()
        run_command(["git", "pull"], cwd=hf_dir)
    else:
        if _pf.system() == "Windows":
            url_prefix = ""
        else:
            url_prefix = "file://"
        model_url = url_prefix + str(
            _pl.Path(model_cache) / model_name
        ) if model_cache else "https://huggingface.co/EleutherAI/gpt-j-6b"
        run_command([
            "git", "clone", model_url, "--single-branch", "--no-local",
            model_name
        ],
                    cwd=hf_dir.parent,
                    env={
                        **_os.environ, "GIT_LFS_SKIP_SMUDGE": "1"
                    })

    assert (hf_dir.is_dir())

    # Download the model file
    model_file_name = "pytorch_model.bin"
    if model_cache:
        if _pf.system() == "Windows":
            wincopy(source=str(
                _pl.Path(model_cache) / model_name / model_file_name),
                    dest=model_file_name,
                    isdir=False,
                    cwd=hf_dir)
        else:
            run_command([
                "rsync", "-av",
                str(_pl.Path(model_cache) / model_name / model_file_name), "."
            ],
                        cwd=hf_dir)
    else:
        run_command(["git", "lfs", "pull", "--include", model_file_name],
                    cwd=hf_dir)

    assert ((hf_dir / model_file_name).is_file())

    engine_dir = models_dir / 'rt_engine' / model_name

    # TODO add Tensor and Pipeline parallelism to GPT-J
    tp_size = 1
    pp_size = 1
    tp_pp_dir = f"tp{tp_size}-pp{pp_size}-gpu"

    if only_fp8:
        # with ifb, new plugin
        print(
            "\nBuilding fp8-plugin engine using gpt_attention_plugin with inflight-batching, packed"
        )
        # TODO: use dummy scales atm; to re-enable when data is uploaded to the model cache
        # quantized_fp8_model_arg = '--quantized_fp8_model_path=' + \
        #     str(_pl.Path(model_cache) / 'fp8-quantized-ammo' / 'gptj_tp1_rank0.npz')
        build_engine(hf_dir, engine_dir / 'fp8-plugin' / tp_pp_dir,
                     '--use_gpt_attention_plugin=float16', '--enable_fp8',
                     '--fp8_kv_cache', '--use_inflight_batching',
                     '--paged_kv_cache', '--remove_input_padding')
    else:
        print("\nBuilding fp16-plugin engine")
        build_engine(hf_dir, engine_dir / 'fp16-plugin' / tp_pp_dir,
                     '--use_gpt_attention_plugin=float16')

        print("\nBuilding fp16-plugin-packed engine")
        build_engine(hf_dir, engine_dir / 'fp16-plugin-packed' / tp_pp_dir,
                     '--use_gpt_attention_plugin=float16',
                     '--remove_input_padding')

        print("\nBuilding fp16-plugin-packed-paged engine")
        build_engine(hf_dir,
                     engine_dir / 'fp16-plugin-packed-paged' / tp_pp_dir,
                     '--use_gpt_attention_plugin=float16',
                     '--use_inflight_batching')
        print("Done.")


if __name__ == "__main__":
    parser = _arg.ArgumentParser()
    parser.add_argument("--model_cache",
                        type=str,
                        help="Directory where models are stored")
    parser.add_argument(
        "--only_fp8",
        action="store_true",
        help="Build engines for only FP8 tests. Implemented for H100 runners.")

    build_engines(**vars(parser.parse_args()))
