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
import sys
from pathlib import Path
from typing import Optional

from build_engines_utils import run_command, wincopy


def convert_ckpt(model_dir: str,
                 output_dir: str,
                 *args,
                 world_size: int = 1,
                 dtype: str = 'float16'):
    convert_cmd = [
        sys.executable, "examples/gpt/convert_checkpoint.py",
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
        '--builder_opt=0',
    ]
    legacy_args = [
        "--gpt_attention_plugin=disable",
        "--context_fmha=disable",
        "--paged_kv_cache=disable",
        "--remove_input_padding=disable",
        "--enable_xqa=disable",
    ]
    build_cmd = build_cmd + legacy_args + list(args)
    run_command(build_cmd)


def build_engines(model_cache: Optional[str] = None, world_size: int = 1):
    # TODO add support of Pipeline parallelism to GPT
    tp_size = world_size
    pp_size = 1

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
                "rsync", "-av",
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

    tp_pp_dir = f"tp{tp_size}-pp{pp_size}-gpu"
    tp_dir = f"{world_size}-gpu"

    print("\nConverting to fp32")
    fp32_ckpt_dir = ckpt_dir / 'fp32' / tp_dir
    convert_ckpt(str(hf_dir),
                 str(fp32_ckpt_dir),
                 world_size=tp_size,
                 dtype='float32')

    print("\nBuilding fp32 engines")
    build_engine(str(fp32_ckpt_dir),
                 str(engine_dir / 'fp32-default' / tp_pp_dir))
    build_engine(str(fp32_ckpt_dir),
                 str(engine_dir / 'fp32-plugin' / tp_pp_dir),
                 '--gpt_attention_plugin=float32', '--context_fmha=enable',
                 '--context_fmha_fp32_acc=enable')

    print("\nConverting to fp16")
    fp16_ckpt_dir = ckpt_dir / 'fp16' / tp_dir
    convert_ckpt(str(hf_dir),
                 str(fp16_ckpt_dir),
                 world_size=tp_size,
                 dtype='float16')

    print("\nBuilding fp16 engines")
    build_engine(str(fp16_ckpt_dir),
                 str(engine_dir / 'fp16-default' / tp_pp_dir))
    build_engine(str(fp16_ckpt_dir),
                 str(engine_dir / 'fp16-plugin' / tp_pp_dir),
                 '--gpt_attention_plugin=float16')
    build_engine(str(fp16_ckpt_dir),
                 str(engine_dir / 'fp16-plugin-packed' / tp_pp_dir),
                 '--gpt_attention_plugin=float16',
                 '--remove_input_padding=enable')

    # this engine can be use for in-flight batching
    ifb_args = [
        '--gpt_attention_plugin=float16',
        '--remove_input_padding=enable',
        '--paged_kv_cache=enable',
        '--context_fmha=enable',
        '--context_fmha_fp32_acc=enable',
        '--max_num_tokens=10000',
        '--use_paged_context_fmha=enable',
    ]
    build_engine(str(fp16_ckpt_dir),
                 str(engine_dir / 'fp16-plugin-packed-paged' / tp_pp_dir),
                 *ifb_args)
    build_engine(
        str(fp16_ckpt_dir),
        str(engine_dir / 'fp16-plugin-packed-paged-draft-tokens' / tp_pp_dir),
        '--max_draft_len=5',
        '--speculative_decoding_mode=draft_tokens_external', *ifb_args)
    build_engine(
        str(fp16_ckpt_dir),
        str(engine_dir / 'fp16-plugin-packed-paged-nprofiles' / tp_pp_dir),
        '--multiple_profiles=enable', *ifb_args)
    build_engine(str(fp16_ckpt_dir),
                 str(engine_dir / 'fp16-plugin-packed-paged-in128' / tp_pp_dir),
                 *ifb_args,
                 max_input_len=128)

    # Build the target model with return accepted token logits
    # Build with '--max_draft_len', '--speculative_decoding_mode' and '--gather_generation_logits'
    build_engine(
        str(fp16_ckpt_dir),
        str(engine_dir /
            'fp16-plugin-packed-paged-return-accepted-tokens-logits' /
            tp_pp_dir), '--max_draft_len=5',
        '--speculative_decoding_mode=draft_tokens_external',
        '--gather_generation_logits', *ifb_args)

    # We build almost the same engine twice. But this engine has gather_all_token_logits
    # to extract logits from python runtime and uses context FMHA for generation to match draft model executions,
    # which uses context FMHA for draft tokens prediction.
    # Currently the gather_all_token_logits is not supported with target model of speculative decoding
    build_engine(
        str(fp16_ckpt_dir),
        str(engine_dir / 'fp16-plugin-packed-paged-gather' / tp_pp_dir),
        '--gather_all_token_logits', *ifb_args)

    build_engine(
        str(fp16_ckpt_dir),
        str(engine_dir / 'fp16-plugin-packed-paged-la-decoding' / tp_pp_dir),
        '--max_draft_len=64', '--speculative_decoding_mode=lookahead_decoding',
        *ifb_args)

    # build engine with lora enabled
    build_engine(str(fp16_ckpt_dir),
                 str(engine_dir / "fp16-plugin-packed-paged-lora" / tp_pp_dir),
                 "--lora_target_modules=attn_qkv", '--lora_plugin=float16',
                 *ifb_args)

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
    build_engine(str(fp16_sq_ckpt_dir),
                 str(engine_dir / 'fp16-plugin-packed-paged-sq' / tp_pp_dir),
                 *ifb_args)

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
                        help='world size, only support tensor parallelism now')

    build_engines(**vars(parser.parse_args()))
