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
    __file__).parent.parent.parent.parent.parent / "examples/chatglm6b"
sys.path.insert(0, str(resources_dir))

engine_target_path = _pl.Path(
    __file__).parent.parent / "models/rt_engine/chatglm6b"

import build as _ecb


def build_engine(weigth_dir: _pl.Path, engine_dir: _pl.Path, *args):
    print("Additional parameters: " + " ".join(args[0]))
    arg = _ecb.parse_arguments()
    arg.model_dir = str(weigth_dir)
    arg.output_dir = str(engine_dir)
    arg.max_batch_size = 2
    arg.max_beam_width = 2
    for item in args[0]:
        key, value = item.split(" ")
        if key[2:] in dir(arg):
            arg.__setattr__(key, value)
        else:
            print("Error parameter name:", key)
            return
    _ecb.build(0, arg)


def run_command(command: _tp.Sequence[str], *, cwd=None, **kwargs) -> None:

    command = [str(i) for i in command]
    print(f"Running: cd %s && %s" %
          (str(cwd or _pl.Path.cwd()), " ".join(command)))
    _sp.check_call(command, cwd=cwd, **kwargs)


def build_engines(model_cache: _tp.Optional[str] = None, world_size: int = 1):

    # Clone the model directory
    hf_dir = resources_dir / "pyTorchModel"
    ft_dir = resources_dir / "ftModel"
    trt_dir = resources_dir / "trtModel"

    run_command(
        ["pip", "install", "-r",
         str(resources_dir) + "/requirements.txt"],
        cwd=resources_dir)

    if not _os.path.exists(hf_dir):
        _os.mkdir(hf_dir)

    if len(_glob(str(hf_dir) + "/*")) == 0:
        run_command(
            ["git", "clone", "https://huggingface.co/THUDM/chatglm-6b", hf_dir],
            cwd=resources_dir)

    if not _os.path.exists(resources_dir / "lm.npy"):
        print("Exporting weight of LM")
        run_command([
            "cp",
            str(hf_dir) + "/modeling_chatglm.py",
            str(hf_dir) + "/modeling_chatglm.py-backup"
        ],
                    cwd=resources_dir)
        run_command([
            "cp",
            str(resources_dir) + "/modeling_chatglm.py",
            str(hf_dir) + "/modeling_chatglm.py"
        ],
                    cwd=resources_dir)
        run_command(["python3", str(resources_dir) + "/exportLM.py"],
                    cwd=resources_dir)
        assert (_os.path.exists(resources_dir / "lm.npy"))
        run_command([
            "mv",
            str(hf_dir) + "/modeling_chatglm.py-backup",
            str(hf_dir) + "/modeling_chatglm.py"
        ],
                    cwd=resources_dir)

    if len(_glob(str(ft_dir) + "/*")) == 0:
        print("\nConverting weight")
        run_command([
            "python3",
            str(resources_dir) + "/hf_chatglm6b_convert.py", "-i", hf_dir, "-o",
            ft_dir, "--storage-type", "fp16"
        ],
                    cwd=resources_dir)

    if len(_glob(str(trt_dir) + "/*")) == 0:
        print("\nBuilding engine")
        arg_list = [
            "--dtype float16",
            "--use_gpt_attention_plugin float16",
            "--use_gemm_plugin float16",
        ]
        build_engine(ft_dir / "1-gpu", trt_dir, arg_list)

    if not _os.path.exists(str(engine_target_path)):
        _os.system(f"mkdir -p {str(engine_target_path)}")
        _os.system(f"cp -r {str(trt_dir) + '/*'} {engine_target_path}")

    print("Done.")


if __name__ == "__main__":
    parser = _arg.ArgumentParser()
    parser.add_argument("--model_cache",
                        type=str,
                        default="examples/chatglm6b/pyTorchModel",
                        help="Directory where models are stored")

    parser.add_argument('--world_size',
                        type=int,
                        default=1,
                        help='world size, only support tensor parallelism now')

    _mp.set_start_method("spawn")

    build_engines(**vars(parser.parse_args()))
