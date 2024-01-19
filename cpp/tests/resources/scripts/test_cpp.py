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
import glob as _gl
import logging as _log
import os as _os
import pathlib as _pl
import subprocess as _sp
import sys as _sys
import typing as _tp
from multiprocessing import cpu_count


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


def run_command(command: _tp.Sequence[str],
                cwd: _pl.Path,
                *,
                shell=False,
                env=None) -> None:
    _log.info("Running: cd %s && %s", str(cwd), " ".join(command))
    _sp.check_call(command, cwd=cwd, shell=shell, env=env)


def build_trt_llm(python_exe: str,
                  root_dir: _pl.Path,
                  build_dir: _pl.Path,
                  cuda_architectures: _tp.Optional[str] = None,
                  use_ccache: _tp.Optional[bool] = False,
                  dist_dir: _tp.Optional[str] = None,
                  trt_root: _tp.Optional[str] = None,
                  job_count: _tp.Optional[int] = None):
    # Build wheel again to WAR issue that the "google-tests" target needs the cmake generated files
    # which were not packaged when running the build job
    # eventually it should be packaged in build job, and run test only on test node
    cuda_architectures = cuda_architectures if cuda_architectures is not None else "80"
    dist_dir = _pl.Path(dist_dir) if dist_dir is not None else _pl.Path("build")
    build_wheel = [
        python_exe, "scripts/build_wheel.py", "--cuda_architectures",
        cuda_architectures, "--build_dir",
        str(build_dir), "--dist_dir",
        str(dist_dir)
    ]

    if use_ccache:
        build_wheel.append("--use_ccache")

    if trt_root is not None:
        build_wheel += ["--trt_root", str(trt_root)]

    if job_count is not None:
        build_wheel += ["-j", str(job_count)]

    run_command(build_wheel, cwd=root_dir, env=_os.environ)

    dist_dir = dist_dir if dist_dir.is_absolute() else root_dir / dist_dir
    wheels = _gl.glob(str(dist_dir / "tensorrt_llm-*.whl"))
    assert len(wheels) > 0, "No wheels found"
    install_wheel = [python_exe, "-m", "pip", "install", "--upgrade", *wheels]
    run_command(install_wheel, cwd=root_dir)


def run_tests(cuda_architectures: _tp.Optional[str] = None,
              build_dir: _tp.Optional[str] = None,
              dist_dir: _tp.Optional[str] = None,
              model_cache: _tp.Optional[str] = None,
              skip_gpt=False,
              skip_gptj=False,
              skip_llama=False,
              skip_chatglm=False,
              run_fp8=False,
              only_multi_gpu=False,
              trt_root: _tp.Optional[str] = None,
              build_only=False,
              use_ccache=False,
              job_count: _tp.Optional[int] = None) -> None:
    root_dir = find_root_dir()
    _log.info("Using root directory: %s", str(root_dir))

    python_exe = _sys.executable
    build_dir = _pl.Path(
        build_dir) if build_dir is not None else _pl.Path("cpp") / "build"

    build_trt_llm(python_exe=python_exe,
                  root_dir=root_dir,
                  build_dir=build_dir,
                  cuda_architectures=cuda_architectures,
                  use_ccache=use_ccache,
                  dist_dir=dist_dir,
                  trt_root=trt_root,
                  job_count=job_count)

    build_dir = build_dir if build_dir.is_absolute() else root_dir / build_dir
    resources_dir = _pl.Path("cpp") / "tests" / "resources"

    generate_lora_data_args_tp1 = [
        python_exe,
        str(resources_dir / "scripts" / "generate_test_lora_weights.py"),
        "--out-dir=cpp/tests/resources/data/lora-test-weights-tp1",
        "--tp-size=1"
    ]

    generate_lora_data_args_tp2 = [
        python_exe,
        str(resources_dir / "scripts" / "generate_test_lora_weights.py"),
        "--out-dir=cpp/tests/resources/data/lora-test-weights-tp2",
        "--tp-size=2"
    ]

    run_command(generate_lora_data_args_tp1, cwd=root_dir)
    run_command(generate_lora_data_args_tp2, cwd=root_dir)

    if not only_multi_gpu:
        prepare_all_model_tests(python_exe=python_exe,
                                root_dir=root_dir,
                                resources_dir=resources_dir,
                                model_cache=model_cache,
                                skip_gpt=skip_gpt,
                                skip_gptj=skip_gptj,
                                skip_llama=skip_llama,
                                skip_chatglm=skip_chatglm,
                                run_fp8=run_fp8)

        if build_only:
            return

        run_google_tests(build_dir=build_dir,
                         skip_gpt=skip_gpt,
                         skip_gptj=skip_gptj,
                         skip_llama=skip_llama,
                         skip_chatglm=skip_chatglm,
                         run_fp8=run_fp8)

        if not skip_gpt:
            run_benchmarks(python_exe=python_exe,
                           root_dir=root_dir,
                           build_dir=build_dir,
                           resources_dir=resources_dir)
        else:
            _log.info("Skipping benchmarks")

    else:
        prepare_multi_gpu_model_tests(python_exe=python_exe,
                                      root_dir=root_dir,
                                      resources_dir=resources_dir,
                                      model_cache=model_cache)

        if build_only:
            return

        run_multi_gpu_tests(build_dir=build_dir)


def prepare_all_model_tests(python_exe: str,
                            root_dir: _pl.Path,
                            resources_dir: _pl.Path,
                            model_cache: _tp.Optional[str] = None,
                            skip_gpt=False,
                            skip_gptj=False,
                            skip_llama=False,
                            skip_chatglm=False,
                            run_fp8=False):
    model_cache_arg = ["--model_cache", model_cache] if model_cache else []
    only_fp8_arg = ["--only_fp8"]

    if not skip_gpt:
        prepare_model_tests(model_name="gpt",
                            python_exe=python_exe,
                            root_dir=root_dir,
                            resources_dir=resources_dir,
                            model_cache_arg=model_cache_arg)
    else:
        _log.info("Skipping GPT tests")

    if not skip_gptj:
        prepare_model_tests(model_name="gptj",
                            python_exe=python_exe,
                            root_dir=root_dir,
                            resources_dir=resources_dir,
                            model_cache_arg=model_cache_arg)
        if run_fp8:
            prepare_model_tests(model_name="gptj",
                                python_exe=python_exe,
                                root_dir=root_dir,
                                resources_dir=resources_dir,
                                model_cache_arg=model_cache_arg,
                                only_fp8_arg=only_fp8_arg)
    else:
        _log.info("Skipping GPT-J tests")

    if not skip_llama:
        prepare_model_tests(model_name="llama",
                            python_exe=python_exe,
                            root_dir=root_dir,
                            resources_dir=resources_dir,
                            model_cache_arg=model_cache_arg)
    else:
        _log.info("Skipping Lllama tests")

    if not skip_chatglm:
        prepare_model_tests(model_name="chatglm",
                            python_exe=python_exe,
                            root_dir=root_dir,
                            resources_dir=resources_dir,
                            model_cache_arg=model_cache_arg)
    else:
        _log.info("Skipping ChatGLM tests")


def prepare_multi_gpu_model_tests(python_exe: str,
                                  root_dir: _pl.Path,
                                  resources_dir: _pl.Path,
                                  model_cache: _tp.Optional[str] = None):
    model_cache_arg = ["--model_cache", model_cache] if model_cache else []
    only_multi_gpu_arg = ["--only_multi_gpu"]

    prepare_model_tests(model_name="llama",
                        python_exe=python_exe,
                        root_dir=root_dir,
                        resources_dir=resources_dir,
                        model_cache_arg=model_cache_arg,
                        only_multi_gpu_arg=only_multi_gpu_arg)


def prepare_model_tests(model_name: str,
                        python_exe: str,
                        root_dir: _pl.Path,
                        resources_dir: _pl.Path,
                        model_cache_arg=[],
                        only_fp8_arg=[],
                        only_multi_gpu_arg=[]):
    scripts_dir = resources_dir / "scripts"

    model_env = {**_os.environ, "PYTHONPATH": f"examples/{model_name}"}
    build_engines = [
        python_exe,
        str(scripts_dir / f"build_{model_name}_engines.py")
    ] + model_cache_arg + only_fp8_arg + only_multi_gpu_arg
    run_command(build_engines, cwd=root_dir, env=model_env)

    model_env["PYTHONPATH"] = "examples"
    generate_expected_output = [
        python_exe,
        str(scripts_dir / f"generate_expected_{model_name}_output.py")
    ] + only_fp8_arg + only_multi_gpu_arg
    if only_multi_gpu_arg:
        generate_expected_output = [
            "mpirun", "-n", "4", "--allow-run-as-root", "--timeout", "600"
        ] + generate_expected_output
    run_command(generate_expected_output, cwd=root_dir, env=model_env)


def run_google_tests(build_dir: _pl.Path, skip_gpt, skip_gptj, skip_llama,
                     skip_chatglm, run_fp8):
    make_google_tests = [
        "cmake", "--build", ".", "--config", "Release", "-j", "--target",
        "google-tests"
    ]
    run_command(make_google_tests, cwd=build_dir)

    cpp_env = {**_os.environ}
    ctest = ["ctest", "--output-on-failure", "--output-junit", "results.xml"]
    excluded_tests = []
    if skip_gpt:
        excluded_tests.append(
            ".*GptTest.*|.*GptSessionTest.*|.*GptManagerTest.*|.*TrtGptModelTest.*"
        )
    if skip_gptj:
        excluded_tests.append(".*Gptj.*")
    if skip_llama:
        excluded_tests.append(".*Llama.*")
    if skip_chatglm:
        excluded_tests.append(".*ChatGlm.*")
    if not run_fp8:
        excluded_tests.append(".*FP8.*")
    if excluded_tests:
        ctest.extend(["-E", "|".join(excluded_tests)])
    run_command(ctest, cwd=build_dir, env=cpp_env)


def run_multi_gpu_tests(build_dir: _pl.Path):
    make_google_tests = [
        "cmake", "--build", ".", "--config", "Release", "-j", "--target",
        "google-tests"
    ]
    run_command(make_google_tests, cwd=build_dir)

    tests_dir = build_dir / "tests"
    cpp_env = {**_os.environ}
    session_test = [
        "mpirun", "-n", "4", "--allow-run-as-root", "gptSessionTest",
        "--gtest_filter=*TP*:*PP*"
    ]
    run_command(session_test, cwd=tests_dir, env=cpp_env)


def run_benchmarks(python_exe: str, root_dir: _pl.Path, build_dir: _pl.Path,
                   resources_dir: _pl.Path):
    scripts_dir = resources_dir / "scripts"

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
    run_command(benchmark, cwd=root_dir)

    prompt_flags = [None, "--long_prompt"]
    prompt_files = ["dummy_cnn.json", "dummy_long_cnn.json"]
    token_files = ["prepared_" + s for s in prompt_files]
    max_input_lens = ["20", "512"]

    for flag, prompt_f, tokens_f, len in zip(prompt_flags, prompt_files,
                                             token_files, max_input_lens):
        generate_batch_manager_data = [
            python_exe,
            str(scripts_dir / "generate_batch_manager_data.py"),
            "--output_filename", prompt_f
        ]
        if flag is not None:
            generate_batch_manager_data.append(flag)
        run_command(generate_batch_manager_data, cwd=root_dir)

        benchmark_src_dir = _pl.Path("benchmarks") / "cpp"
        data_dir = resources_dir / "data"
        prepare_dataset = [
            python_exe,
            str(benchmark_src_dir / "prepare_dataset.py"), "--dataset",
            str(data_dir / prompt_f), "--max_input_len", len, "--tokenizer_dir",
            str(resources_dir / "models" / "gpt2"), "--output",
            str(data_dir / tokens_f)
        ]
        run_command(prepare_dataset, cwd=root_dir)

        benchmark = [
            str(benchmark_exe_dir / "gptManagerBenchmark"), "--model", "gpt",
            "--engine_dir",
            str(gpt_engine_dir / "fp16-plugin-packed-paged" / "tp1-pp1-gpu"),
            "--type", "IFB", "--dataset",
            str(data_dir / tokens_f)
        ]
        run_command(benchmark, cwd=root_dir)

        benchmark = [
            str(benchmark_exe_dir / "gptManagerBenchmark"), "--model", "gpt",
            "--engine_dir",
            str(gpt_engine_dir / "fp16-plugin-packed-paged" / "tp1-pp1-gpu"),
            "--type", "V1", "--dataset",
            str(data_dir / tokens_f)
        ]
        run_command(benchmark, cwd=root_dir)


if __name__ == "__main__":
    _log.basicConfig(level=_log.INFO)
    parser = _arg.ArgumentParser()

    parser.add_argument("--cuda_architectures", "-a")
    parser.add_argument("--use_ccache",
                        action="store_true",
                        help="Use ccache in cmake building stage")
    parser.add_argument("--job_count",
                        "-j",
                        type=int,
                        const=cpu_count(),
                        nargs="?",
                        help="Parallel job count for compiling TensorRT-LLM")
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
    parser.add_argument("--only_gpt",
                        action="store_true",
                        help="Run only the tests for GPT")
    parser.add_argument("--only_gptj",
                        action="store_true",
                        help="Run only the tests for GPT-J")
    parser.add_argument("--only_llama",
                        action="store_true",
                        help="Run only the tests for Llama")
    parser.add_argument("--only_chatglm",
                        action="store_true",
                        help="Run only the tests for ChatGLM")
    parser.add_argument("--skip_gpt",
                        action="store_true",
                        help="Skip the tests for GPT")
    parser.add_argument("--skip_gptj",
                        action="store_true",
                        help="Skip the tests for GPT-J")
    parser.add_argument("--skip_llama",
                        action="store_true",
                        help="Skip the tests for Llama")
    parser.add_argument("--skip_chatglm",
                        action="store_true",
                        help="Skip the tests for ChatGLM")
    parser.add_argument(
        "--run_fp8",
        action="store_true",
        help="Additionally run FP8 tests. Implemented for H100 runners.")
    parser.add_argument(
        "--only_multi_gpu",
        action="store_true",
        help="Run only mulit-GPU tests. Implemented for 4 GPUs.")
    parser.add_argument("--build_only",
                        action="store_true",
                        help="Build only, do not run tests.")

    args = parser.parse_args()

    if (args.only_gpt + args.only_gptj + args.only_llama + args.only_chatglm >
            1):
        parser.error('Cannot combine multiple only_* arguments.')

    if args.only_gpt:
        args.skip_gptj = True
        args.skip_llama = True
        args.skip_chatglm = True

    if args.only_gptj:
        args.skip_gpt = True
        args.skip_llama = True
        args.skip_chatglm = True

    if args.only_llama:
        args.skip_gpt = True
        args.skip_gptj = True
        args.skip_chatglm = True

    if args.only_chatglm:
        args.skip_gpt = True
        args.skip_gptj = True
        args.skip_llama = True

    del args.only_gpt
    del args.only_gptj
    del args.only_llama
    del args.only_chatglm

    run_tests(**vars(args))
