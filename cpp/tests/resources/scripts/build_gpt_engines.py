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

import argparse
import os
import platform
import shutil
import sys
from pathlib import Path
from typing import Optional

from build_engines_utils import run_command, wincopy

import tensorrt_llm.bindings as _tb
from tensorrt_llm.bindings.internal.testing import ModelSpec, QuantMethod


def convert_ckpt(model_dir: str,
                 output_dir: str,
                 *args,
                 world_size: int = 1,
                 dtype: str = 'float16'):
    convert_cmd = [
        sys.executable, "examples/models/core/gpt/convert_checkpoint.py",
        f"--model_dir={model_dir}", f"--output_dir={output_dir}",
        f"--dtype={dtype}", f"--tp_size={world_size}"
    ] + list(args)
    run_command(convert_cmd)


def build_engine(
    checkpoint_dir: str,
    engine_dir: str,
    *args,
    max_input_len: int = 256,
    max_seq_len: int = 384,
):

    build_cmd = [
        "trtllm-build",
        '--log_level=error',
        f'--checkpoint_dir={checkpoint_dir}',
        f'--output_dir={engine_dir}',
        '--max_batch_size=64',
        f'--max_input_len={max_input_len}',
        f'--max_seq_len={max_seq_len}',
        '--max_beam_width=2',
        '--kv_cache_type=continuous',
    ]
    legacy_args = [
        "--gpt_attention_plugin=disable",
        "--context_fmha=disable",
        "--remove_input_padding=disable",
    ]
    build_cmd = build_cmd + legacy_args + list(args)
    run_command(build_cmd)


def build_engines(model_cache: Optional[str] = None,
                  world_size: int = 1,
                  clean: Optional[bool] = False):
    # TODO add support of Pipeline parallelism to GPT
    tp_size = world_size
    pp_size = 1
    cp_size = 1

    resources_dir = Path(__file__).parent.resolve().parent
    models_dir = resources_dir / 'models'
    model_name = 'gpt2'

    # Clone or update the model directory without lfs
    hf_dir = models_dir / model_name
    if hf_dir.exists():
        assert hf_dir.is_dir()
        run_command(["git", "pull"], cwd=hf_dir)
    else:
        if platform.system() == "Windows":
            url_prefix = ""
        else:
            url_prefix = "file://"

        model_url = url_prefix + str(
            Path(model_cache) /
            model_name) if model_cache else "https://huggingface.co/gpt2"
        run_command([
            "git", "clone", model_url, "--single-branch", "--no-local",
            model_name
        ],
                    cwd=hf_dir.parent,
                    env={
                        **os.environ, "GIT_LFS_SKIP_SMUDGE": "1"
                    })

    assert hf_dir.is_dir()

    # Download the model file
    model_file_name = "pytorch_model.bin"
    if model_cache:
        if platform.system() == "Windows":
            wincopy(source=str(
                Path(model_cache) / model_name / model_file_name),
                    dest=model_file_name,
                    isdir=False,
                    cwd=hf_dir)
        else:
            run_command([
                "rsync", "-rlptD",
                str(Path(model_cache) / model_name / model_file_name), "."
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

    ckpt_dir = models_dir / 'c-model' / model_name
    engine_dir = models_dir / 'rt_engine' / model_name

    if clean:
        target_dir = Path(engine_dir)
        print('clean up target folder ', target_dir)
        if target_dir.is_dir():
            shutil.rmtree(target_dir, ignore_errors=True)

    tp_pp_cp_dir = f"tp{tp_size}-pp{pp_size}-cp{cp_size}-gpu"
    tp_dir = f"{world_size}-gpu"

    print("\nConverting to fp16")
    fp16_ckpt_dir = ckpt_dir / 'fp16' / tp_dir
    convert_ckpt(str(hf_dir),
                 str(fp16_ckpt_dir),
                 world_size=tp_size,
                 dtype='float16')

    print("\nBuilding fp16 engines")

    input_file = 'input_tokens.npy'
    # this engine can be use for in-flight batching
    ifb_base_args = [
        '--gpt_attention_plugin=float16',
        '--remove_input_padding=enable',
        '--context_fmha=enable',
        '--max_num_tokens=10000',
        '--use_paged_context_fmha=enable',
    ]

    paged_kv_cache_args = ['--kv_cache_type=paged']

    no_kv_cache_args = ['--kv_cache_type=disabled']

    def get_ifb_args(kv_cache_type):
        if kv_cache_type == _tb.KVCacheType.DISABLED:
            return ifb_base_args + no_kv_cache_args
        elif kv_cache_type == _tb.KVCacheType.PAGED:
            return ifb_base_args + paged_kv_cache_args
        else:
            assert False, f"Unsupported kv_cache_type: {kv_cache_type}"

    model_spec_obj = ModelSpec(input_file, _tb.DataType.HALF)
    model_spec_obj.use_gpt_plugin()
    model_spec_obj.set_kv_cache_type(_tb.KVCacheType.PAGED)
    model_spec_obj.use_packed_input()

    model_spec_current = model_spec_obj.__copy__()

    for kv_cache_type in [_tb.KVCacheType.DISABLED, _tb.KVCacheType.PAGED]:
        model_spec_current.set_kv_cache_type(kv_cache_type)
        build_engine(
            str(fp16_ckpt_dir),
            str(engine_dir / model_spec_current.get_model_path() /
                tp_pp_cp_dir), *get_ifb_args(kv_cache_type))

    model_spec_current = model_spec_obj.__copy__()
    max_draft_tokens = 5
    model_spec_current.use_draft_tokens_external_decoding()
    model_spec_current.set_draft_tokens(max_draft_tokens)

    build_engine(
        str(fp16_ckpt_dir),
        str(engine_dir / model_spec_current.get_model_path() / tp_pp_cp_dir),
        f'--max_draft_len={max_draft_tokens}',
        '--speculative_decoding_mode=draft_tokens_external',
        *get_ifb_args(_tb.KVCacheType.PAGED))

    model_spec_current = model_spec_obj.__copy__()
    model_spec_current.use_multiple_profiles()

    build_engine(
        str(fp16_ckpt_dir),
        str(engine_dir / model_spec_current.get_model_path() / tp_pp_cp_dir),
        '--multiple_profiles=enable', *get_ifb_args(_tb.KVCacheType.PAGED))

    model_spec_current = model_spec_obj.__copy__()
    max_input_len = 128
    model_spec_current.set_max_input_length(max_input_len)

    build_engine(str(fp16_ckpt_dir),
                 str(engine_dir / model_spec_current.get_model_path() /
                     tp_pp_cp_dir),
                 *get_ifb_args(_tb.KVCacheType.PAGED),
                 max_input_len=max_input_len)

    # We build almost the same engine twice. But this engine has gather_context_logits
    # to extract logits from python runtime and uses context FMHA for generation to match draft model executions,
    # which uses context FMHA for draft tokens prediction.
    # Currently the gather_context_logits is not supported with target model of speculative decoding
    model_spec_current = model_spec_obj.__copy__()
    model_spec_current.gather_logits()

    build_engine(
        str(fp16_ckpt_dir),
        str(engine_dir / model_spec_current.get_model_path() / tp_pp_cp_dir),
        '--gather_context_logits', *get_ifb_args(_tb.KVCacheType.PAGED))

    # build engine with lora enabled
    model_spec_current = model_spec_obj.__copy__()
    model_spec_current.use_lora_plugin()
    build_engine(
        str(fp16_ckpt_dir),
        str(engine_dir / model_spec_current.get_model_path() / tp_pp_cp_dir),
        "--lora_target_modules=attn_qkv", '--lora_plugin=float16',
        *get_ifb_args(_tb.KVCacheType.PAGED))

    if model_cache:
        llm_datasets_root = Path(model_cache) / "datasets"
        calib_dataset = llm_datasets_root / "cimec/lambada/"
    else:
        calib_dataset = "lambada"
    print("\nConverting to fp16 SQ")
    fp16_sq_ckpt_dir = ckpt_dir / 'fp16-sq' / tp_dir
    convert_ckpt(str(hf_dir),
                 str(fp16_sq_ckpt_dir),
                 "--smoothquant=0.5",
                 f"--calib_dataset={calib_dataset}",
                 world_size=tp_size,
                 dtype='float16')

    print("\nBuilding fp16 SQ engines")
    model_spec_current = ModelSpec(input_file, _tb.DataType.HALF)
    model_spec_current.use_gpt_plugin()
    model_spec_current.use_packed_input()
    model_spec_current.set_quant_method(QuantMethod.SMOOTH_QUANT)

    for kv_cache_type in [_tb.KVCacheType.DISABLED, _tb.KVCacheType.PAGED]:
        model_spec_current.set_kv_cache_type(kv_cache_type)
        build_engine(
            str(fp16_sq_ckpt_dir),
            str(engine_dir / model_spec_current.get_model_path() /
                tp_pp_cp_dir), *get_ifb_args(kv_cache_type))

    if has_safetensor:
        Path(str(safetensor_file) + ".bak").rename(safetensor_file)

    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_cache",
                        type=str,
                        help="Directory where models are stored")

    parser.add_argument('--world_size',
                        type=int,
                        default=1,
                        help='World size, only support tensor parallelism now')

    parser.add_argument('--clean',
                        action='store_true',
                        default=False,
                        help='Clean target folders before building engines')

    build_engines(**vars(parser.parse_args()))
