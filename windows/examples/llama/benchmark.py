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
import subprocess
import sys
from pathlib import Path


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Benchmark llama TensorRT-LLM model.")
    parser.add_argument('--model_dir',
                        type=str,
                        default=None,
                        help="Directory with HF model")
    parser.add_argument('--quant_ckpt_path',
                        type=str,
                        default=None,
                        help="Path with quanitzed weights")
    parser.add_argument('--engine_dir',
                        type=str,
                        default=None,
                        help="Directory to store engines")

    return parser.parse_args()


def main(args):
    dir_level = 3  # How nested is the current file?
    top_level_path = Path(__file__).resolve().parents[dir_level]
    example_path = top_level_path / "examples" / "llama"
    benchmark_path = top_level_path / "benchmarks" / "python"

    input_seqlen = 100
    output_seqlen = 100
    batch_size = 8

    build_args = [
        sys.executable,
        str(example_path / "build.py"), "--model_dir",
        str(args.model_dir), "--quant_ckpt_path",
        str(args.quant_ckpt_path), "--dtype", "float16", "--log_level", "info",
        "--use_gpt_attention_plugin", "float16", "--use_gemm_plugin", "float16",
        "--enable_context_fmha", "--use_weight_only", "--weight_only_precision",
        "int4_gptq", "--per_group", "--max_input_len",
        str(input_seqlen), "--max_output_len",
        str(output_seqlen), "--n_positions",
        str(input_seqlen + output_seqlen + 1), "--max_batch_size",
        str(batch_size), "--output_dir",
        str(args.engine_dir)
    ]
    benchmark_args = [
        sys.executable,
        str(benchmark_path / "benchmark.py"), "--engine_dir",
        str(args.engine_dir), "--mode", "plugin", "-m", "llama_7b", "--dtype",
        "float16", "--log_level", "info", "--batch_size",
        str(batch_size), "--input_output_len",
        f"{input_seqlen},{output_seqlen}", "--num_beams", "1", "--warm_up", "1",
        "--num_runs", "3", "--duration", "10", "--csv"
    ]

    def run(pass_args):
        print("Running {}".format(" ".join(pass_args)))
        subprocess.run(pass_args)

    run(build_args)
    run(benchmark_args)


if __name__ == '__main__':
    args = parse_arguments()
    assert args.model_dir, "Please pass in path to model"
    assert args.quant_ckpt_path, "Please pass in path to quantized weights"
    if not args.engine_dir:
        args.engine_dir = Path.cwd() / "engines"

    assert Path(
        args.model_dir).exists(), "Please pass a valid, existing model path"
    assert Path(args.quant_ckpt_path).exists(
    ), "Please pass a valid, existing path to quantized weights"

    args.model_dir = Path(args.model_dir)
    args.quant_ckpt_path = Path(args.quant_ckpt_path)
    args.engine_dir = Path(args.engine_dir)

    main(args)
