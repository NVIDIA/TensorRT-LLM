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
import platform
import shutil
import sys
import typing
from pathlib import Path

from build_engines_utils import run_command, wincopy

resources_dir = Path(
    __file__).parent.parent.parent.parent.parent / "examples/chatglm"
sys.path.insert(0, str(resources_dir))

engine_target_path = Path(__file__).parent.parent / "models/rt_engine"


def convert_ckpt(model_dir: str, output_dir: str, world_size: int):
    convert_cmd = [
        sys.executable,
        str(resources_dir / "convert_checkpoint.py"), "--dtype=float16",
        f"--model_dir={model_dir}", f"--output_dir={output_dir}",
        f"--tp_size={world_size}"
    ]
    print("Running: " + " ".join(convert_cmd))
    run_command(convert_cmd)


def build_engine(ckpt_dir: str, engine_dir: str):
    build_cmd = [
        "trtllm-build", f"--checkpoint_dir={ckpt_dir}",
        f"--output_dir={engine_dir}", "--log_level=error", "--max_batch_size=2",
        "--max_beam_width=4", "--max_input_len=512", "--max_output_len=512",
        "--gpt_attention_plugin=float16", "--gemm_plugin=float16",
        "--builder_opt=0", "--remove_input_padding=disable",
        "--paged_kv_cache=disable"
    ]
    print("Running: " + " ".join(build_cmd))
    run_command(build_cmd)


def build_engines(model_cache: typing.Optional[str] = None,
                  world_size: int = 1):
    Path(engine_target_path).mkdir(parents=True, exist_ok=True)

    run_command(
        ["pip", "install", "-r",
         str(resources_dir) + "/requirements.txt"],
        cwd=resources_dir)

    for model_name in ["chatglm_6b", "chatglm2_6b", "chatglm3_6b"]:
        # Get original model
        model_cache_dir = Path(model_cache) / model_name.replace("_", "-")
        if model_cache_dir.is_dir():
            print("Copy model from model_cache")
            if platform.system() == "Windows":
                wincopy(source=str(model_cache_dir),
                        dest=model_name,
                        isdir=True,
                        cwd=resources_dir)
            else:
                run_command(
                    ["rsync", "-av", str(model_cache_dir), "."],
                    cwd=resources_dir)
            shutil.move(resources_dir / model_name.replace("_", "-"),
                        resources_dir / model_name)
        else:
            print("Clone model from HF")
            run_command(
                [
                    "git", "clone",
                    f"https://huggingface.co/THUDM/{model_name.replace('_', '-')}",
                    model_name
                ],
                cwd=resources_dir,
            )

        # Build engines
        print(f"Building {model_name}")
        weight_dir = Path(resources_dir) / model_name
        ckpt_dir = Path(resources_dir) / "trt_ckpt" / model_name
        trt_dir = Path(resources_dir) / "trt_engines" / model_name

        # fix remained error in chatglm_6b, hope to remove this in the future
        if model_name == "chatglm_6b":
            shutil.copy(
                Path(resources_dir) / "tokenization_chatglm.py",
                weight_dir,
            )

        convert_ckpt(weight_dir, ckpt_dir, world_size)
        build_engine(ckpt_dir, trt_dir)
        shutil.move(trt_dir, engine_target_path)

    print("Done")


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
