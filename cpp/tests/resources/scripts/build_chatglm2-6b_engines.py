#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import subprocess as _sp
import sys
import typing as _tp
from glob import glob as _glob

import torch.multiprocessing as _mp

resources_dir = _pl.Path(
    __file__).parent.parent.parent.parent.parent / "examples/chatglm2-6b"
sys.path.insert(0, str(resources_dir))

engine_target_path = _pl.Path(
    __file__).parent.parent / "models/rt_engine/chatglm2-6b"

import build as _ecb


def build_engine(weight_dir: _pl.Path, engine_dir: _pl.Path, world_size, *args):
    args = [
        '--log_level=error',
        '--model_dir',
        str(weight_dir),
        '--output_dir',
        str(engine_dir),
        '--max_batch_size=2',
        '--max_beam_width=2',
        '--builder_opt=0',
        f'--world_size={world_size}',
    ] + list(args)
    print("Running: " + " ".join(args))
    _ecb.run_build(args)


def run_command(command: _tp.Sequence[str], *, cwd=None, **kwargs) -> None:

    command = [str(i) for i in command]
    print(f"Running: cd %s && %s" %
          (str(cwd or _pl.Path.cwd()), " ".join(command)))
    _sp.check_call(command, cwd=cwd, **kwargs)


def build_engines(model_cache: _tp.Optional[str] = None, world_size: int = 1):

    # Clone the model directory
    hf_dir = resources_dir / "pyTorchModel"
    trt_dir = resources_dir / "trtModel"

    run_command(
        ["pip", "install", "-r",
         str(resources_dir) + "/requirements.txt"],
        cwd=resources_dir)

    if not _os.path.exists(hf_dir):
        _os.mkdir(hf_dir)

    if len(_glob(str(hf_dir) + "/*")) == 0:
        run_command(
            [
                "git",
                "clone",
                "https://huggingface.co/THUDM/chatglm2-6b",
                hf_dir,
            ],
            cwd=resources_dir,
        )

    print("\nBuilding engine")
    build_engine(hf_dir, trt_dir, world_size, "--dtype", "float16",
                 "--use_gpt_attention_plugin", "float16", "--use_gemm_plugin",
                 "float16")

    if not _os.path.exists(str(engine_target_path)):
        _os.system(f"mkdir -p {str(engine_target_path)}")

    _os.system(f"cp -r {str(trt_dir) + '/*'} {engine_target_path}")

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
