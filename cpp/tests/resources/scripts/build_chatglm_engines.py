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
import shutil as _shutil
import sys as _sys
import typing as _tp
from pathlib import Path as _Path

from build_engines_utils import run_command, wincopy

resources_dir = _pl.Path(
    __file__).parent.parent.parent.parent.parent / "examples/chatglm"
_sys.path.insert(0, str(resources_dir))

engine_target_path = _pl.Path(__file__).parent.parent / "models/rt_engine"

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
        "--max_input_len=512",
        "--max_output_len=512",
        '--builder_opt=0',
        f'--world_size={world_size}',
    ] + list(args)
    print("Running: " + " ".join(args))
    _ecb.run_build(args)


def build_engines(model_cache: _tp.Optional[str] = None, world_size: int = 1):
    _Path(engine_target_path).mkdir(parents=True, exist_ok=True)

    run_command(
        ["pip", "install", "-r",
         str(resources_dir) + "/requirements.txt"],
        cwd=resources_dir)

    for model_name in ["chatglm_6b", "chatglm2_6b", "chatglm3_6b"]:
        # Get original model
        model_cache_dir = _pl.Path(model_cache) / model_name.replace("_", "-")
        if model_cache_dir.is_dir():
            print("Copy model from model_cache")
            if _pf.system() == "Windows":
                wincopy(source=str(model_cache_dir),
                        dest=model_name,
                        isdir=True,
                        cwd=resources_dir)
            else:
                run_command(
                    ["rsync", "-av", str(model_cache_dir), "."],
                    cwd=resources_dir)
            _shutil.move(resources_dir / model_name.replace("_", "-"),
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
        print("Building %s" % model_name)
        weight_dir = _Path(resources_dir) / model_name
        trt_dir = _Path(resources_dir) / ("output_" + model_name)

        # fix remained error in chatglm_6b, hope to remove this in the future
        if model_name == "chatglm_6b":
            _shutil.copy(
                _Path(resources_dir) / "tokenization_chatglm.py",
                weight_dir,
            )

        build_engine(weight_dir, trt_dir, world_size)
        _shutil.move(trt_dir, engine_target_path / model_name)

    print("Done")


if __name__ == "__main__":
    parser = _arg.ArgumentParser()
    parser.add_argument("--model_cache",
                        type=str,
                        help="Directory where models are stored")

    parser.add_argument('--world_size',
                        type=int,
                        default=1,
                        help='world size, only support tensor parallelism now')

    build_engines(**vars(parser.parse_args()))
