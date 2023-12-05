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
import pathlib as _pl
import shutil as _shutil
import subprocess as _sp
import sys
import typing as _tp
from pathlib import Path as _Path

import torch.multiprocessing as _mp

resources_dir = _pl.Path(
    __file__).parent.parent.parent.parent.parent / "examples/chatglm"
sys.path.insert(0, str(resources_dir))

engine_target_path = _pl.Path(__file__).parent.parent / "models/rt_engine"

import build as _ecb


def build_engine(model_name: str, weight_dir: _pl.Path, engine_dir: _pl.Path,
                 world_size, *args):
    args = [
        '-m',
        str(model_name),
        '--log_level=error',
        '--model_dir',
        str(weight_dir),
        '--output_dir',
        str(engine_dir),
        '--max_batch_size=2',
        '--max_beam_width=2',
        "--max_input_len=512",
        "--max_output_len=512",
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

    model_name_list = ["chatglm_6b", "chatglm2_6b", "chatglm3_6b"]
    hf_dir_list = [resources_dir / model_name for model_name in model_name_list]
    trt_dir_list = [
        resources_dir / ("output_" + model_name)
        for model_name in model_name_list
    ]

    run_command(
        ["pip", "install", "-r",
         str(resources_dir) + "/requirements.txt"],
        cwd=resources_dir)

    # Clone the model directory
    for model_name, hf_dir in zip(model_name_list, hf_dir_list):
        if not _Path(hf_dir).exists():
            run_command(
                [
                    "git",
                    "clone",
                    "https://huggingface.co/THUDM/" +
                    model_name.replace("_", "-"),
                    model_name,
                ],
                cwd=resources_dir,
            )

    print("\nBuilding engines")
    for model_name, hf_dir, trt_dir in zip(model_name_list, hf_dir_list,
                                           trt_dir_list):
        print("Building %s" % model_name)
        build_engine(model_name, hf_dir, trt_dir, world_size)

    if not _Path(engine_target_path).exists():
        _Path(engine_target_path).mkdir(parents=True, exist_ok=True)
    for model_name in model_name_list:
        _shutil.move(
            _Path(resources_dir) / ("output_" + model_name),
            engine_target_path / model_name)

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
