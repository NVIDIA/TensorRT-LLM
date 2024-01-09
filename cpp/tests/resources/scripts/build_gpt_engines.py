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
import typing as _tp

import hf_gpt_convert as _egc
import torch.multiprocessing as _mp
from build_engines_utils import run_command, wincopy

import build as _egb  # isort:skip


def build_engine(weight_dir: _pl.Path, engine_dir: _pl.Path, world_size, *args):
    args = [
        '--log_level=error',
        '--model_dir',
        str(weight_dir),
        '--output_dir',
        str(engine_dir),
        '--max_batch_size=256',
        '--max_input_len=512',
        '--max_output_len=20',
        '--max_beam_width=2',
        '--builder_opt=0',
        f'--world_size={world_size}',
    ] + list(args)
    print("Running: " + " ".join(args))
    _egb.run_build(args)


def build_engines(model_cache: _tp.Optional[str] = None, world_size: int = 1):
    # TODO add support of Pipeline parallelism to GPT
    tp_size = world_size
    pp_size = 1

    resources_dir = _pl.Path(__file__).parent.resolve().parent
    models_dir = resources_dir / 'models'
    model_name = 'gpt2'

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
            _pl.Path(model_cache) /
            model_name) if model_cache else "https://huggingface.co/gpt2"
        run_command([
            "git", "clone", model_url, "--single-branch", "--no-local",
            model_name
        ],
                    cwd=hf_dir.parent,
                    env={
                        **_os.environ, "GIT_LFS_SKIP_SMUDGE": "1"
                    })

    assert hf_dir.is_dir()

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

    safetensor_file = hf_dir / "model.safetensors"
    has_safetensor = safetensor_file.exists()
    if has_safetensor:
        safetensor_file.rename(str(safetensor_file) + ".bak")

    assert (hf_dir / model_file_name).is_file()

    weight_dir = models_dir / 'c-model' / model_name
    engine_dir = models_dir / 'rt_engine' / model_name

    print("\nConverting to fp32")
    tp_pp_dir = f"tp{tp_size}-pp{pp_size}-gpu"
    fp32_weight_dir = weight_dir / 'fp32'
    _egc.run_conversion(
        _egc.ProgArgs(in_file=str(hf_dir),
                      out_dir=str(fp32_weight_dir),
                      storage_type='float32',
                      tensor_parallelism=tp_size))

    print("\nBuilding fp32 engines")
    tp_dir = f"{world_size}-gpu"
    fp32_weight_dir_x_gpu = fp32_weight_dir / tp_dir
    build_engine(fp32_weight_dir_x_gpu, engine_dir / 'fp32-default' / tp_pp_dir,
                 tp_size, '--dtype=float32')
    build_engine(fp32_weight_dir_x_gpu, engine_dir / 'fp32-plugin' / tp_pp_dir,
                 tp_size, '--dtype=float32',
                 '--use_gpt_attention_plugin=float32')

    print("\nConverting to fp16")
    fp16_weight_dir = weight_dir / 'fp16'
    _egc.run_conversion(
        _egc.ProgArgs(in_file=str(hf_dir),
                      out_dir=str(fp16_weight_dir),
                      storage_type='float16',
                      tensor_parallelism=tp_size))

    print("\nBuilding fp16 engines")
    fp16_weight_dir_x_gpu = fp16_weight_dir / tp_dir
    build_engine(fp16_weight_dir_x_gpu, engine_dir / 'fp16-default' / tp_pp_dir,
                 tp_size, '--dtype=float16')
    build_engine(fp16_weight_dir_x_gpu, engine_dir / 'fp16-plugin' / tp_pp_dir,
                 tp_size, '--dtype=float16',
                 '--use_gpt_attention_plugin=float16')
    build_engine(fp16_weight_dir_x_gpu,
                 engine_dir / 'fp16-plugin-packed' / tp_pp_dir, tp_size,
                 '--dtype=float16', '--use_gpt_attention_plugin=float16',
                 '--remove_input_padding')
    # this engine can be use for in-flight batching
    ifb_args = [
        '--dtype=float16',
        '--use_gpt_attention_plugin=float16',
        '--remove_input_padding',
        '--paged_kv_cache',
        '--enable_context_fmha_fp32_acc',
        '--max_num_tokens=10000',
        '--use_paged_context_fmha',
    ]
    build_engine(fp16_weight_dir_x_gpu,
                 engine_dir / 'fp16-plugin-packed-paged' / tp_pp_dir, tp_size,
                 '--max_draft_len=5', *ifb_args)

    # We build almost the same engine twice. But this engine has gather_all_token_logits
    # to extract logits from python runtime and uses context FMHA for generation to match draft model executions,
    # which uses context FMHA for draft tokens prediction.
    # Currently the gather_all_token_logits is not supported with target model of speculative decoding
    build_engine(fp16_weight_dir_x_gpu,
                 engine_dir / 'fp16-plugin-packed-paged-gather' / tp_pp_dir,
                 tp_size, '--gather_all_token_logits',
                 '--use_context_fmha_for_generation', *ifb_args)
    build_engine(
        fp16_weight_dir_x_gpu, engine_dir /
        'fp16-plugin-packed-paged-context-fmha-for-gen' / tp_pp_dir, tp_size,
        '--use_context_fmha_for_generation', '--max_draft_len=5', *ifb_args)

    # build engine with lora enabled
    build_engine(fp16_weight_dir_x_gpu,
                 engine_dir / "fp16-plugin-packed-paged-lora" / tp_pp_dir,
                 tp_size, '--use_lora_plugin=float16',
                 '--lora_target_modules=attn_qkv', *ifb_args)

    if has_safetensor:
        _pl.Path(str(safetensor_file) + ".bak").rename(safetensor_file)

    print("Done.")


if __name__ == "__main__":
    parser = _arg.ArgumentParser()
    parser.add_argument("--model_cache",
                        type=str,
                        help="Directory where models are stored")

    parser.add_argument('--world_size',
                        type=int,
                        default=1,
                        help='world size, only support tensor parallelism now')

    _mp.set_start_method("spawn")

    build_engines(**vars(parser.parse_args()))
