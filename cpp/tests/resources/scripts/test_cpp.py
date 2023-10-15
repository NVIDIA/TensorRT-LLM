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
import glob as _gl
import logging as _log
import os as _os
import pathlib as _pl
import subprocess as _sp
import sys as _sys
import typing as _tp


def find_dir_containing(files: _tp.Sequence[str],
                        start_dir: _tp.Optional[_pl.Path] = None) -> _pl.Path:
    if start_dir is None:
        start_dir = _pl.Path.cwd().absolute()

    assert isinstance(start_dir, _pl.Path)
    assert start_dir.is_dir()

    if set(files).issubset({f.name for f in start_dir.iterdir()}):
        return start_dir
    elif start_dir.parent is not start_dir:
        return find_dir_containing(files, start_dir.parent)
    else:
        raise FileNotFoundError(files)


def find_root_dir(start_dir: _tp.Optional[_pl.Path] = None) -> _pl.Path:
    return find_dir_containing(("scripts", "examples", "cpp"), start_dir)


def run_tests(cuda_architectures: _tp.Optional[str] = None,
              build_dir: _tp.Optional[str] = None,
              dist_dir: _tp.Optional[str] = None,
              model_cache: _tp.Optional[str] = None,
              skip_gptj=False,
              skip_llama=False,
              skip_chatglm6b=False,
              only_fp8=False,
              trt_root: _tp.Optional[str] = None) -> None:
    root_dir = find_root_dir()
    _log.info("Using root directory: %s", str(root_dir))

    def run_command(command: _tp.Sequence[str],
                    *,
                    cwd=root_dir,
                    shell=False,
                    env=None) -> None:
        _log.info("Running: cd %s && %s", str(cwd), " ".join(command))
        _sp.check_call(command, cwd=cwd, shell=shell, env=env)

    python_exe = _sys.executable

    # Build wheel again to WAR issue that the "google-tests" target needs the cmake generated files
    # which were not packaged when running the build job
    # eventually it should be packaged in build job, and run test only on test node
    cuda_architectures = cuda_architectures if cuda_architectures is not None else "80"
    build_dir = _pl.Path(
        build_dir) if build_dir is not None else _pl.Path("cpp") / "build"
    dist_dir = _pl.Path(dist_dir) if dist_dir is not None else _pl.Path("build")
    build_wheel = [
        python_exe, "scripts/build_wheel.py", "--cuda_architectures",
        cuda_architectures, "--build_dir",
        str(build_dir), "--dist_dir",
        str(dist_dir)
    ]
    if trt_root is not None:
        build_wheel += ["--trt_root", str(trt_root)]

    run_command(build_wheel)

    dist_dir = dist_dir if dist_dir.is_absolute() else root_dir / dist_dir
    wheels = _gl.glob(str(dist_dir / "tensorrt_llm-*.whl"))
    assert len(wheels) > 0, "No wheels found"
    install_wheel = [python_exe, "-m", "pip", "install", "--upgrade", *wheels]
    run_command(install_wheel)

    resources_dir = _pl.Path("cpp") / "tests" / "resources"
    scripts_dir = resources_dir / "scripts"
    model_cache = ["--model_cache", model_cache] if model_cache else []
    only_fp8_arg = ["--only_fp8"] if only_fp8 else []

    gpt_env = {**_os.environ, "PYTHONPATH": "examples/gpt"}
    build_gpt_engines = [python_exe,
                         str(scripts_dir / "build_gpt_engines.py")
                         ] + model_cache
    run_command(build_gpt_engines, env=gpt_env)

    generate_expected_gpt_output = [
        python_exe,
        str(scripts_dir / "generate_expected_gpt_output.py")
    ]
    run_command(generate_expected_gpt_output, env=gpt_env)

    if not skip_gptj:
        build_gptj_engines = [
            python_exe, str(scripts_dir / "build_gptj_engines.py")
        ] + model_cache + only_fp8_arg
        run_command(build_gptj_engines)

        gptj_env = {**_os.environ, "PYTHONPATH": "examples/gptj"}
        generate_expected_gptj_output = [
            python_exe,
            str(scripts_dir / "generate_expected_gptj_output.py")
        ] + only_fp8_arg
        run_command(generate_expected_gptj_output, env=gptj_env)
    else:
        _log.info("Skipping GPT-J tests")

    if not skip_llama:
        build_llama_engines = [
            python_exe, str(scripts_dir / "build_llama_engines.py")
        ] + model_cache
        run_command(build_llama_engines)

        llama_env = {**_os.environ, "PYTHONPATH": "examples/llama"}
        generate_expected_llama_output = [
            python_exe,
            str(scripts_dir / "generate_expected_llama_output.py")
        ]
        run_command(generate_expected_llama_output, env=llama_env)
    else:
        _log.info("Skipping Lllama tests")

    if not skip_chatglm6b:
        build_chatglm6b_engines = [
            python_exe,
            str(scripts_dir / "build_chatglm6b_engines.py")
        ]
        run_command(build_chatglm6b_engines)

        chatglm6b_env = {**_os.environ, "PYTHONPATH": "examples/chatglm6b"}
        generate_expected_chatglm6b_output = [
            python_exe,
            str(scripts_dir / "generate_expected_chatglm6b_output.py")
        ]  # only_fp8 is not supported by ChatGLM-6B now
        run_command(generate_expected_chatglm6b_output, env=chatglm6b_env)
    else:
        _log.info("Skipping ChatGLM6B tests")

    build_dir = build_dir if build_dir.is_absolute() else root_dir / build_dir

    make_google_tests = [
        "cmake", "--build", ".", "--config", "Release", "-j", "--target",
        "google-tests"
    ]
    run_command(make_google_tests, cwd=build_dir)

    cpp_env = {**_os.environ}
    ctest = ["ctest", "--output-on-failure", "--output-junit", "report.xml"]
    excluded_tests = []
    if skip_gptj:
        excluded_tests.append(".*Gptj.*")
    if skip_llama:
        excluded_tests.append(".*Llama.*")
    if only_fp8:
        ctest.extend(["-R", ".*FP8.*"])
    else:
        excluded_tests.append(".*FP8.*")
    if excluded_tests:
        ctest.extend(["-E", "|".join(excluded_tests)])
    run_command(ctest, cwd=build_dir, env=cpp_env)

    make_benchmarks = [
        "cmake", "--build", ".", "--config", "Release", "-j", "--target",
        "benchmarks"
    ]
    run_command(make_benchmarks, cwd=build_dir)

    benchmark_exe_dir = build_dir / "benchmarks"
    gpt_engine_dir = resources_dir / "models" / "rt_engine" / "gpt2"
    benchmark = [
        str(benchmark_exe_dir / "gptSessionBenchmark"), "--model", "gpt",
        "--engine_dir",
        str(gpt_engine_dir / "fp16-plugin" / "tp1-pp1-gpu"), "--batch_size",
        "8", "--input_output_len", "10,20", "--duration", "10"
    ]
    run_command(benchmark)

    generate_batch_manager_data = [
        python_exe,
        str(scripts_dir / "generate_batch_manager_data.py")
    ]
    run_command(generate_batch_manager_data)

    benchmark_src_dir = _pl.Path("benchmarks") / "cpp"
    data_dir = resources_dir / "data"
    prepare_dataset = [
        python_exe,
        str(benchmark_src_dir / "prepare_dataset.py"), "--dataset",
        str(data_dir / "dummy_cnn.json"), "--max_input_len", "20",
        "--tokenizer_dir",
        str(resources_dir / "models" / "gpt2"), "--output",
        str(data_dir / "prepared_dummy_cnn.json")
    ]
    run_command(prepare_dataset)

    benchmark = [
        str(benchmark_exe_dir / "gptManagerBenchmark"), "--model", "gpt",
        "--engine_dir",
        str(gpt_engine_dir / "fp16-plugin-packed-paged" / "tp1-pp1-gpu"),
        "--type", "IFB", "--dataset",
        str(data_dir / "prepared_dummy_cnn.json")
    ]
    run_command(benchmark)
    benchmark = [
        str(benchmark_exe_dir / "gptManagerBenchmark"), "--model", "gpt",
        "--engine_dir",
        str(gpt_engine_dir / "fp16-plugin-packed-paged" / "tp1-pp1-gpu"),
        "--type", "V1", "--dataset",
        str(data_dir / "prepared_dummy_cnn.json")
    ]
    run_command(benchmark)


if __name__ == "__main__":
    _log.basicConfig(level=_log.INFO)
    parser = _arg.ArgumentParser()

    parser.add_argument("--cuda_architectures", "-a")
    parser.add_argument("--build_dir",
                        type=str,
                        help="Directory where cpp sources are built")
    parser.add_argument("--trt_root",
                        type=str,
                        help="Directory of the TensorRT install")
    parser.add_argument("--dist_dir",
                        type=str,
                        help="Directory where python wheels are built")
    parser.add_argument("--model_cache",
                        type=str,
                        help="Directory where models are stored")
    parser.add_argument("--skip_gptj",
                        action="store_true",
                        help="Skip the tests for GPT-J")
    parser.add_argument("--skip_llama",
                        action="store_true",
                        help="Skip the tests for Llama")
    parser.add_argument("--skip_chatglm6b",
                        action="store_true",
                        help="Skip the tests for ChatGLM6B")

    parser.add_argument(
        "--only_fp8",
        action="store_true",
        help="Run only FP8 tests. Implemented for H100 runners.")
    run_tests(**vars(parser.parse_args()))
