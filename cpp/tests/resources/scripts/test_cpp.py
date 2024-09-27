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
import copy
import functools
import glob
import logging as _log
import os as _os
import pathlib as _pl
import platform
import signal
import subprocess as _sp
import sys as _sys
import time as _time
import typing as _tp

build_script_dir = _pl.Path(
    __file__).parent.resolve().parent.parent.parent.parent / 'scripts'
assert build_script_dir.is_dir()
_sys.path.append(str(build_script_dir))
from build_wheel import add_arguments as add_build_arguments
from build_wheel import get_build_dir
from build_wheel import main as build_trt_llm


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
                env=None,
                timeout=None) -> None:
    _log.info("Running: cd %s && %s", str(cwd), " ".join(command))
    override_timeout = int(_os.environ.get("CPP_TEST_TIMEOUT_OVERRIDDEN", "-1"))
    if override_timeout > 0 and (timeout is None or override_timeout > timeout):
        _log.info("Overriding the command timeout: %s (before) and %s (after)",
                  timeout, override_timeout)
        timeout = override_timeout
    _sp.check_call(command, cwd=cwd, shell=shell, env=env, timeout=timeout)


def merge_report(parallel, retry, output):
    import xml.etree.ElementTree as ElementTree
    base = ElementTree.parse(parallel)
    extra = ElementTree.parse(retry)

    base_suite = base.getroot()
    extra_suite = extra.getroot()

    base_suite.attrib['failures'] = extra_suite.attrib['failures']
    base_suite.attrib['time'] = str(
        int(base_suite.attrib['time']) + int(extra_suite.attrib['time']))

    case_names = {element.attrib['name'] for element in extra_suite}
    base_suite[:] = [
        element
        for element in base_suite if element.attrib['name'] not in case_names
    ] + list(extra_suite)

    base.write(output, encoding="UTF-8", xml_declaration=True)


def add_parallel_info(report, parallel):
    import xml.etree.ElementTree as ElementTree
    try:
        document = ElementTree.parse(report)
    except FileNotFoundError:
        return
    root = document.getroot()
    root.attrib['parallel'] = str(parallel)
    document.write(report, encoding="UTF-8", xml_declaration=True)


def parallel_run_ctest(
    command: _tp.Sequence[str],
    cwd: _pl.Path,
    *,
    shell=False,
    env=None,
    timeout=None,
    parallel=2,
) -> None:
    if parallel == 1:
        return run_command(command,
                           cwd=cwd,
                           shell=shell,
                           env=env,
                           timeout=timeout)

    env = {} if env is None else env
    env['CTEST_PARALLEL_LEVEL'] = str(parallel)

    def get_report():
        reports = glob.glob("results-*.xml", root_dir=cwd)
        if not reports:
            return ''

        return reports[0]

    report = None
    try:
        run_command(command, cwd=cwd, shell=shell, env=env, timeout=timeout)
    except _sp.CalledProcessError:
        report = get_report()
        if report == '':
            # Some catastrophic fail happened that there's no report generated
            raise

        parallel_report = 'parallel-' + report
        _os.rename(cwd / report, cwd / parallel_report)

        try:
            _log.info("Parallel test failed, retry serial on failed tests")
            del env['CTEST_PARALLEL_LEVEL']
            command = [*command, "--rerun-failed"]
            run_command(command, cwd=cwd, shell=shell, env=env, timeout=timeout)
        finally:
            if not _os.path.exists(cwd / report):
                # Some catastrophic fail happened that there's no report generated
                # Use parallel result as final report
                _os.rename(cwd / parallel_report, cwd / report)
            else:
                retry_report = 'retry-' + report
                _os.rename(cwd / report, cwd / retry_report)
                merge_report(cwd / parallel_report, cwd / retry_report,
                             cwd / report)
    finally:
        if report is None:
            report = get_report()
        if report:
            add_parallel_info(cwd / report, parallel)


def run_tests(build_dir: _pl.Path,
              model_cache: _tp.Optional[str] = None,
              skip_unit_tests=False,
              run_gpt=False,
              run_gptj=False,
              run_llama=False,
              run_chatglm=False,
              run_medusa=False,
              run_mamba=False,
              run_recurrentgemma=False,
              run_encoder=False,
              run_bart=False,
              run_t5=False,
              run_redrafter=False,
              run_fp8=False,
              only_multi_gpu=False,
              build_only=False,
              test_timeout=3600) -> None:
    root_dir = find_root_dir()
    _log.info("Using root directory: %s", str(root_dir))

    python_exe = _sys.executable

    if run_mamba:
        run_command(
            [python_exe, "-m", "pip", "install", "transformers>=4.39.0"],
            cwd=root_dir,
            env=_os.environ,
            timeout=300)

    if run_recurrentgemma:
        run_command([
            "git", "clone",
            "https://github.com/google-deepmind/recurrentgemma.git"
        ],
                    cwd=root_dir,
                    env=_os.environ,
                    timeout=300)
        run_command(
            [python_exe, "-m", "pip", "install", "./recurrentgemma[full]"],
            cwd=root_dir,
            env=_os.environ,
            timeout=300)

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

    generate_multi_lora_tp2_args = [
        python_exe,
        str(resources_dir / "scripts" / "generate_test_lora_weights.py"),
        "--out-dir=cpp/tests/resources/data/multi_lora",
        "--tp-size=2",
        "--num-loras=128",
    ]

    generate_gpt2_lora_data_args_tp1 = [
        python_exe,
        str(resources_dir / "scripts" / "generate_test_lora_weights.py"),
        "--out-dir=cpp/tests/resources/data/lora-test-weights-gpt2-tp1",
        "--tp-size=1", "--hidden-size=768", "--num-layers=12",
        "--config-ids-filter=0", "--no-generate-cache-pages"
    ]

    run_command(generate_lora_data_args_tp1, cwd=root_dir, timeout=100)
    run_command(generate_lora_data_args_tp2, cwd=root_dir, timeout=100)
    run_command(generate_multi_lora_tp2_args, cwd=root_dir, timeout=100)
    run_command(generate_gpt2_lora_data_args_tp1, cwd=root_dir, timeout=100)

    if not skip_unit_tests:
        run_unit_tests(build_dir=build_dir, timeout=test_timeout)
    else:
        _log.info("Skipping unit tests")

    if not only_multi_gpu:
        prepare_all_model_tests(python_exe=python_exe,
                                root_dir=root_dir,
                                resources_dir=resources_dir,
                                model_cache=model_cache,
                                run_gpt=run_gpt,
                                run_gptj=run_gptj,
                                run_llama=run_llama,
                                run_chatglm=run_chatglm,
                                run_medusa=run_medusa,
                                run_mamba=run_mamba,
                                run_recurrentgemma=run_recurrentgemma,
                                run_encoder=run_encoder,
                                run_bart=run_bart,
                                run_t5=run_t5,
                                run_redrafter=run_redrafter,
                                run_fp8=run_fp8)

        if build_only:
            return

        run_single_gpu_tests(build_dir=build_dir,
                             run_gpt=run_gpt,
                             run_gptj=run_gptj,
                             run_llama=run_llama,
                             run_chatglm=run_chatglm,
                             run_medusa=run_medusa,
                             run_mamba=run_mamba,
                             run_recurrentgemma=run_recurrentgemma,
                             run_encoder=run_encoder,
                             run_bart=run_bart,
                             run_t5=run_t5,
                             run_redrafter=run_redrafter,
                             run_fp8=run_fp8,
                             timeout=test_timeout)

        if run_gpt:
            run_benchmarks(model_name="gpt",
                           python_exe=python_exe,
                           root_dir=root_dir,
                           build_dir=build_dir,
                           resources_dir=resources_dir,
                           model_cache=model_cache,
                           test_gpt_session_benchmark=True,
                           batching_types=["IFB", "V1"],
                           api_types=["gptManager", "executor"])
        elif run_t5:
            run_benchmarks(model_name="t5",
                           python_exe=python_exe,
                           root_dir=root_dir,
                           build_dir=build_dir,
                           resources_dir=resources_dir,
                           model_cache=model_cache,
                           test_gpt_session_benchmark=False,
                           batching_types=["IFB"],
                           api_types=["executor"])
        elif run_bart:
            run_benchmarks(model_name="bart",
                           python_exe=python_exe,
                           root_dir=root_dir,
                           build_dir=build_dir,
                           resources_dir=resources_dir,
                           model_cache=model_cache,
                           test_gpt_session_benchmark=False,
                           batching_types=["IFB"],
                           api_types=["executor"])
        else:
            _log.info("Skipping benchmarks")

    elif platform.system() != "Windows":
        prepare_multi_gpu_model_tests(python_exe=python_exe,
                                      root_dir=root_dir,
                                      resources_dir=resources_dir,
                                      model_cache=model_cache)

        if build_only:
            return

        run_multi_gpu_tests(build_dir=build_dir, timeout=test_timeout)


def prepare_all_model_tests(python_exe: str,
                            root_dir: _pl.Path,
                            resources_dir: _pl.Path,
                            model_cache: _tp.Optional[str] = None,
                            run_gpt=False,
                            run_gptj=False,
                            run_llama=False,
                            run_chatglm=False,
                            run_medusa=False,
                            run_mamba=False,
                            run_recurrentgemma=False,
                            run_encoder=False,
                            run_bart=False,
                            run_t5=False,
                            run_redrafter=False,
                            run_fp8=False):
    model_cache_arg = ["--model_cache", model_cache] if model_cache else []

    if run_gpt:
        prepare_model_tests(model_name="gpt",
                            python_exe=python_exe,
                            root_dir=root_dir,
                            resources_dir=resources_dir,
                            model_cache_arg=model_cache_arg)
    else:
        _log.info("Skipping GPT tests")

    if run_gptj:
        prepare_model_tests(model_name="gptj",
                            python_exe=python_exe,
                            root_dir=root_dir,
                            resources_dir=resources_dir,
                            model_cache_arg=model_cache_arg)
        if run_fp8:
            only_fp8_arg = ["--only_fp8"]
            prepare_model_tests(model_name="gptj",
                                python_exe=python_exe,
                                root_dir=root_dir,
                                resources_dir=resources_dir,
                                model_cache_arg=model_cache_arg,
                                only_fp8_arg=only_fp8_arg)
    else:
        _log.info("Skipping GPT-J tests")

    if run_llama:
        prepare_model_tests(model_name="llama",
                            python_exe=python_exe,
                            root_dir=root_dir,
                            resources_dir=resources_dir,
                            model_cache_arg=model_cache_arg)
    else:
        _log.info("Skipping Lllama tests")

    if run_chatglm:
        prepare_model_tests(model_name="chatglm",
                            python_exe=python_exe,
                            root_dir=root_dir,
                            resources_dir=resources_dir,
                            model_cache_arg=model_cache_arg)
    else:
        _log.info("Skipping ChatGLM tests")

    if run_medusa:
        prepare_model_tests(model_name="medusa",
                            python_exe=python_exe,
                            root_dir=root_dir,
                            resources_dir=resources_dir,
                            model_cache_arg=model_cache_arg)
    else:
        _log.info("Skipping Medusa tests")

    if run_mamba:
        prepare_model_tests(model_name="mamba",
                            python_exe=python_exe,
                            root_dir=root_dir,
                            resources_dir=resources_dir,
                            model_cache_arg=model_cache_arg)
    else:
        _log.info("Skipping Mamba tests")

    if run_recurrentgemma:
        prepare_model_tests(model_name="recurrentgemma",
                            python_exe=python_exe,
                            root_dir=root_dir,
                            resources_dir=resources_dir,
                            model_cache_arg=model_cache_arg)
    else:
        _log.info("Skipping RecurrentGemma tests")

    if run_encoder:
        prepare_model_tests(model_name="enc_dec",
                            python_exe=python_exe,
                            root_dir=root_dir,
                            resources_dir=resources_dir,
                            model_cache_arg=model_cache_arg)
    else:
        _log.info("Skipping encoder tests")

    if run_bart:
        prepare_model_tests(model_name="bart",
                            python_exe=python_exe,
                            root_dir=root_dir,
                            resources_dir=resources_dir,
                            model_cache_arg=model_cache_arg)
    else:
        _log.info("Skipping BART tests")

    if run_t5:
        prepare_model_tests(model_name="t5",
                            python_exe=python_exe,
                            root_dir=root_dir,
                            resources_dir=resources_dir,
                            model_cache_arg=model_cache_arg)
    else:
        _log.info("Skipping T5 tests")

    if run_redrafter:
        prepare_model_tests(model_name="redrafter",
                            python_exe=python_exe,
                            root_dir=root_dir,
                            resources_dir=resources_dir,
                            model_cache_arg=model_cache_arg)
    else:
        _log.info("Skipping ReDrafter tests")


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

    prepare_model_tests(model_name="t5",
                        python_exe=python_exe,
                        root_dir=root_dir,
                        resources_dir=resources_dir,
                        model_cache_arg=model_cache_arg,
                        only_multi_gpu_arg=['--tp', '4', '--pp', '1'])

    prepare_model_tests(model_name="gpt",
                        python_exe=python_exe,
                        root_dir=root_dir,
                        resources_dir=resources_dir,
                        model_cache_arg=model_cache_arg)

    prepare_model_tests(model_name="chatglm",
                        python_exe=python_exe,
                        root_dir=root_dir,
                        resources_dir=resources_dir,
                        model_cache_arg=model_cache_arg)


def prepare_model_tests(model_name: str,
                        python_exe: str,
                        root_dir: _pl.Path,
                        resources_dir: _pl.Path,
                        model_cache_arg=[],
                        only_fp8_arg=[],
                        only_multi_gpu_arg=[]):
    scripts_dir = resources_dir / "scripts"

    model_env = {**_os.environ, "PYTHONPATH": f"examples/{model_name}"}
    enc_dec_model_name_arg = []
    beams_arg = []
    if model_name in ('bart', 't5'):
        enc_dec_model_name_arg = [
            '--hf_repo_name',
            'facebook/bart-large-cnn' if model_name == 'bart' else 't5-small'
        ]
        if model_name == 't5' and (not only_multi_gpu_arg):
            beams_arg = ['--beams', '1,2']
        model_name = 'enc_dec'

    build_engines = [
        python_exe,
        str(scripts_dir / f"build_{model_name}_engines.py")
    ] + model_cache_arg + only_fp8_arg + only_multi_gpu_arg + enc_dec_model_name_arg + beams_arg

    if model_name in ['gpt']:
        build_engines += ['--clean']
    run_command(build_engines, cwd=root_dir, env=model_env, timeout=1800)

    model_env["PYTHONPATH"] = "examples"
    generate_expected_output = [
        python_exe,
        str(scripts_dir / f"generate_expected_{model_name}_output.py")
    ] + only_fp8_arg + only_multi_gpu_arg + enc_dec_model_name_arg
    if "enc_dec" in model_name:
        generate_expected_output += model_cache_arg
        generate_expected_output += beams_arg

    if model_name in ['gpt']:
        generate_expected_output += ['--clean']

    if only_multi_gpu_arg and model_name != 'enc_dec':
        for world_size in (2, 4):
            generate_command = [
                "mpirun", "-n",
                str(world_size), "--allow-run-as-root", "--timeout", "600"
            ] + generate_expected_output
            run_command(generate_command,
                        cwd=root_dir,
                        env=model_env,
                        timeout=600)
    else:
        run_command(generate_expected_output,
                    cwd=root_dir,
                    env=model_env,
                    timeout=600)


def build_tests(build_dir: _pl.Path):
    make_google_tests = [
        "cmake", "--build", ".", "--config", "Release", "-j", "--target",
        "google-tests"
    ]
    run_command(make_google_tests, cwd=build_dir, timeout=300)


def with_memory_monitor(func):
    if not _os.environ.get('LLM_MEMORY_PROFILING', False):
        return func

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        memory_collector = _sp.Popen([
            "/usr/bin/python3",
            find_root_dir() /
            "tests/llm-test-defs/turtle/defs/memory_collector.py",
            "-p",
            str(_os.getpid()),
            "-i",
            "0.2",
        ])
        try:
            func(*args, **kwargs)
        finally:
            memory_collector.send_signal(signal.SIGINT)
            memory_collector.wait()

    return wrapper


@with_memory_monitor
def run_unit_tests(build_dir: _pl.Path, timeout=1800):
    build_tests(build_dir=build_dir)

    cpp_env = {**_os.environ}
    ctest = [
        "ctest", "--output-on-failure", "--output-junit",
        "results-unit-tests.xml"
    ]
    excluded_tests = []
    excluded_tests.append("Gpt[^j]")
    excluded_tests.append("Gptj")
    excluded_tests.append("Llama")
    excluded_tests.append("ChatGlm")
    excluded_tests.append("Medusa")
    excluded_tests.append("ExplicitDraftTokensDecoding")
    excluded_tests.append("Mamba")
    excluded_tests.append("RecurrentGemma")
    excluded_tests.append("Encoder")
    excluded_tests.append("EncDec")
    ctest.extend(["-E", "|".join(excluded_tests)])
    parallel_run_ctest(ctest, cwd=build_dir, env=cpp_env, timeout=timeout)


@with_memory_monitor
def run_single_gpu_tests(build_dir: _pl.Path,
                         run_gpt,
                         run_gptj,
                         run_llama,
                         run_chatglm,
                         run_medusa,
                         run_mamba,
                         run_recurrentgemma,
                         run_encoder,
                         run_bart,
                         run_t5,
                         run_redrafter,
                         run_fp8,
                         timeout=3600):
    build_tests(build_dir=build_dir)

    cpp_env = {**_os.environ}
    ctest = [
        "ctest", "--output-on-failure", "--output-junit",
        "results-single-gpu.xml"
    ]

    included_tests = []
    if run_gpt:
        included_tests.append("Gpt[^j]")
    if run_gptj:
        included_tests.append("Gptj")
    if run_llama:
        included_tests.append("Llama")
    if run_chatglm:
        included_tests.append("ChatGlm")
    if run_medusa:
        included_tests.append("Medusa")
    if run_mamba:
        included_tests.append("Mamba")
    if run_recurrentgemma:
        included_tests.append("RecurrentGemma")
    if run_encoder:
        included_tests.append("EncoderModelTestSingleGPU")
    if run_bart:
        included_tests.append("BartBasicTest")
    if run_t5:
        included_tests.append("T5BasicTest")
        included_tests.append("T5Beam2Test")
    if run_redrafter:
        included_tests.append("ExplicitDraftTokens")

    excluded_tests = []
    if not run_fp8:
        excluded_tests.append("FP8")

    if included_tests:
        ctest.extend(["-R", "|".join(included_tests)])
        if excluded_tests:
            ctest.extend(["-E", "|".join(excluded_tests)])
        parallel_run_ctest(ctest, cwd=build_dir, env=cpp_env, timeout=timeout)


def produce_mpirun_command(*, global_commands, nranks, local_commands,
                           leader_commands):
    l = global_commands
    for rank in range(nranks):
        l += ["-n", "1"] + local_commands + (leader_commands
                                             if rank == 0 else []) + [":"]
    return l[:-1]


@with_memory_monitor
def run_multi_gpu_tests(build_dir: _pl.Path, timeout=1500):
    build_tests(build_dir=build_dir)

    tests_dir = build_dir / "tests"
    cpp_env = {**_os.environ}
    # Utils tests
    mpi_utils_test = [
        "mpirun",
        "-n",
        "4",
        "--allow-run-as-root",
        "mpiUtilsTest",
    ]
    run_command(mpi_utils_test, cwd=tests_dir, env=cpp_env, timeout=300)

    # Cache transceiver tests
    cache_trans_test = [
        "mpirun",
        "-n",
        "2",
        "--allow-run-as-root",
        "batch_manager/cacheTransceiverTest",
    ]
    run_command(cache_trans_test, cwd=tests_dir, env=cpp_env, timeout=300)

    # UCX transceiver tests, the test may not be built if ENABLE_UCX is 0
    if _os.path.exists(
            _os.path.join(tests_dir, "batch_manager/ucxDataTransceiverTest")):
        ucx_trans_test = [
            "mpirun",
            "-n",
            "2",
            "--allow-run-as-root",
            "batch_manager/ucxDataTransceiverTest",
        ]
        run_command(ucx_trans_test, cwd=tests_dir, env=cpp_env, timeout=300)

    xml_output_file = build_dir / "results-multi-gpu-real-decoder.xml"
    trt_model_test = produce_mpirun_command(
        global_commands=["mpirun", "--allow-run-as-root"],
        nranks=4,
        local_commands=[
            "batch_manager/trtGptModelRealDecoderTest",
            "--gtest_filter=*TP*:*PP*"
        ],
        leader_commands=[f"--gtest_output=xml:{xml_output_file}"])
    run_command(trt_model_test, cwd=tests_dir, env=cpp_env,
                timeout=timeout)  # expecting ~ 1200s

    #Executor test in leader mode
    new_env = copy.copy(cpp_env)
    xml_output_file = build_dir / "results-multi-gpu-llama-exec-leader-mode.xml"
    new_env["RUN_LLAMA_MULTI_GPU"] = "true"
    trt_model_test = produce_mpirun_command(
        global_commands=["mpirun", "--allow-run-as-root"],
        nranks=4,
        local_commands=[
            "executor/executorTest",
            "--gtest_filter=*LlamaExecutorTest*LeaderMode*:*LlamaMultiExecutorTest*LeaderMode*"
        ],
        leader_commands=[f"--gtest_output=xml:{xml_output_file}"])
    run_command(trt_model_test, cwd=tests_dir, env=new_env, timeout=1500)

    # Executor test in orchestrator mode
    # https://nvbugs/4690328 - Disabled BW2 tests because of spurious failure
    xml_output_file = build_dir / "results-multi-gpu-llama-exec-orch-mode.xml"
    trt_model_test = [
        "mpirun", "-n", "1", "--allow-run-as-root", "executor/executorTest",
        "--gtest_filter=*LlamaExecutorTest*OrchMode*:-*BW2*",
        f"--gtest_output=xml:{xml_output_file}"
    ]
    run_command(trt_model_test, cwd=tests_dir, env=new_env, timeout=1500)

    #EncDec test in leader mode
    new_env = copy.copy(cpp_env)
    xml_output_file = build_dir / "results-multi-gpu-t5-exec-leader-mode.xml"
    trt_model_test = produce_mpirun_command(
        global_commands=["mpirun", "--allow-run-as-root"],
        nranks=4,
        local_commands=[
            "executor/executorTest",
            "--gtest_filter=T5MultiGPUTest/EncDecParamsTest.Forward*"
        ],
        leader_commands=[f"--gtest_output=xml:{xml_output_file}"],
    )
    run_command(trt_model_test, cwd=tests_dir, env=new_env, timeout=1500)

    #Logits processor test in leader mode
    new_env = copy.copy(cpp_env)
    xml_output_file = build_dir / "results-multi-gpu-logits-proc.xml"
    trt_model_test = produce_mpirun_command(
        global_commands=["mpirun", "--allow-run-as-root"],
        nranks=4,
        local_commands=[
            "executor/executorTest",
            "--gtest_filter=LlamaExecutorTest/LogitsProcParamsTest*tp2_pp2*"
        ],
        leader_commands=[f"--gtest_output=xml:{xml_output_file}"])
    run_command(trt_model_test, cwd=tests_dir, env=new_env, timeout=1500)

    new_env = copy.copy(cpp_env)
    xml_output_file = build_dir / "results-multi-gpu-dist-executor_gpt.xml"
    trt_model_test = produce_mpirun_command(
        global_commands=["mpirun", "--allow-run-as-root"],
        nranks=2,
        local_commands=[
            "executor/executorTest",
            "--gtest_filter=DistExecutorTest.GPTTokenComparison"
        ],
        leader_commands=[f"--gtest_output=xml:{xml_output_file}"])
    run_command(trt_model_test, cwd=tests_dir, env=new_env, timeout=1500)

    new_env = copy.copy(cpp_env)
    xml_output_file = build_dir / "results-multi-gpu-dist-executor_chatglm.xml"
    trt_model_test = produce_mpirun_command(
        global_commands=["mpirun", "--allow-run-as-root"],
        nranks=2,
        local_commands=[
            "executor/executorTest",
            "--gtest_filter=DistExecutorTest.ChatGLMTokenComparison"
        ],
        leader_commands=[f"--gtest_output=xml:{xml_output_file}"])
    run_command(trt_model_test, cwd=tests_dir, env=new_env, timeout=1500)


def run_benchmarks(model_name: str, python_exe: str, root_dir: _pl.Path,
                   build_dir: _pl.Path, resources_dir: _pl.Path,
                   model_cache: str, test_gpt_session_benchmark: bool,
                   batching_types: list[str], api_types: list[str]):

    # At this moment, CI env might not installed tensorrt_llm before, so tensorrt_llm module might not be available.
    import pathlib
    import sys

    import model_spec
    src_root_dir = pathlib.Path(
        __file__).parent.resolve().parent.parent.parent.parent

    sys.path.insert(0, str(src_root_dir))
    import tensorrt_llm.bindings as _tb

    make_benchmarks = [
        "cmake", "--build", ".", "--config", "Release", "-j", "--target",
        "benchmarks"
    ]
    run_command(make_benchmarks, cwd=build_dir, timeout=300)

    benchmark_exe_dir = build_dir / "benchmarks"
    if model_name == "gpt":
        model_engine_dir = resources_dir / "models" / "rt_engine" / "gpt2"
        tokenizer_dir = resources_dir / "models" / "gpt2"
    elif model_name in ('bart', 't5'):
        if model_name == "t5":
            hf_repo_name = "t5-small"
        elif model_name == "bart":
            hf_repo_name = "bart-large-cnn"
        model_engine_dir = resources_dir / "models" / "enc_dec" / "trt_engines" / hf_repo_name
        tokenizer_dir = model_cache + "/" + hf_repo_name
        model_engine_path = model_engine_dir / "1-gpu" / "float16" / "decoder"
        encoder_model_engine_path = model_engine_dir / "1-gpu" / "float16" / "encoder"
        model_name = "enc_dec"
    else:
        _log.info(
            f"run_benchmark test does not support {model_name}. Skipping benchmarks"
        )
        return NotImplementedError

    if test_gpt_session_benchmark:
        if model_name == "gpt":
            input_file = 'input_tokens.npy'
            model_spec_obj = model_spec.ModelSpec(input_file, _tb.DataType.HALF)
            model_spec_obj.set_kv_cache_type(_tb.KVCacheType.CONTINUOUS)
            model_spec_obj.use_gpt_plugin()
            model_engine_path = model_engine_dir / model_spec_obj.get_model_path(
            ) / "tp1-pp1-gpu"
        else:
            _log.info(
                f"gptSessionBenchmark test does not support {model_name}. Skipping benchmarks"
            )
            return NotImplementedError

        benchmark = [
            str(benchmark_exe_dir / "gptSessionBenchmark"), "--engine_dir",
            str(model_engine_path), "--batch_size", "8", "--input_output_len",
            "10,20", "--duration", "10"
        ]
        run_command(benchmark, cwd=root_dir, timeout=600)

    prompt_datasets_args = [{
        '--dataset-name': "cnn_dailymail",
        '--dataset-config-name': "3.0.0",
        '--dataset-split': "validation",
        '--dataset-input-key': "article",
        '--dataset-prompt': "Summarize the following article:",
        '--dataset-output-key': "highlights"
    }, {
        '--dataset-name': "Open-Orca/1million-gpt-4",
        '--dataset-split': "train",
        '--dataset-input-key': "question",
        '--dataset-prompt-key': "system_prompt",
        '--dataset-output-key': "response"
    }]
    token_files = [
        "prepared_" + s['--dataset-name'].replace('/', '_')
        for s in prompt_datasets_args
    ]
    max_input_lens = ["256", "20"]
    num_reqs = ["50", "10"]

    if model_name == "gpt":
        input_file = 'input_tokens.npy'
        model_spec_obj = model_spec.ModelSpec(input_file, _tb.DataType.HALF)
        model_spec_obj.set_kv_cache_type(_tb.KVCacheType.PAGED)
        model_spec_obj.use_gpt_plugin()
        model_spec_obj.use_packed_input()
        model_engine_path = model_engine_dir / model_spec_obj.get_model_path(
        ) / "tp1-pp1-gpu"

    for prompt_ds_args, tokens_f, len, num_req in zip(prompt_datasets_args,
                                                      token_files,
                                                      max_input_lens, num_reqs):

        benchmark_src_dir = _pl.Path("benchmarks") / "cpp"
        data_dir = resources_dir / "data"
        prepare_dataset = [
            python_exe,
            str(benchmark_src_dir / "prepare_dataset.py"), "--tokenizer",
            str(tokenizer_dir), "--output",
            str(data_dir / tokens_f), "dataset", "--max-input-len", len,
            "--num-requests", num_req
        ]
        for k, v in prompt_ds_args.items():
            prepare_dataset += [k, v]
        # https://nvbugs/4658787
        # WAR before the prepare dataset can use offline cached dataset
        run_command(prepare_dataset,
                    cwd=root_dir,
                    timeout=300,
                    env={'HF_DATASETS_OFFLINE': '0'})

        for batching_type in batching_types:
            for api_type in api_types:
                benchmark = [
                    str(benchmark_exe_dir / "gptManagerBenchmark"),
                    "--engine_dir",
                    str(model_engine_path), "--type",
                    str(batching_type), "--api",
                    str(api_type), "--dataset",
                    str(data_dir / tokens_f)
                ]
                if model_name == "enc_dec":
                    benchmark += [
                        "--encoder_engine_dir",
                        str(encoder_model_engine_path)
                    ]

                run_command(benchmark, cwd=root_dir, timeout=600)
                req_rate_benchmark = benchmark + ["--request_rate", "100"]
                run_command(req_rate_benchmark, cwd=root_dir, timeout=600)
                concurrency_benchmark = benchmark + ["--concurrency", "30"]
                run_command(concurrency_benchmark, cwd=root_dir, timeout=600)

        if "IFB" in batching_type and "executor" in api_types:
            # executor streaming test
            benchmark = [
                str(benchmark_exe_dir / "gptManagerBenchmark"), "--engine_dir",
                str(model_engine_path), "--type", "IFB", "--dataset",
                str(data_dir / tokens_f), "--api", "executor", "--streaming"
            ]
            if model_name == "enc_dec":
                benchmark += [
                    "--encoder_engine_dir",
                    str(encoder_model_engine_path)
                ]
            run_command(benchmark, cwd=root_dir, timeout=600)

        if "IFB" in batching_type and "gptManager" in api_type:
            # gptManager streaming test
            benchmark = [
                str(benchmark_exe_dir / "gptManagerBenchmark"), "--engine_dir",
                str(model_engine_path), "--type", "IFB", "--dataset",
                str(data_dir / tokens_f), "--api", "gptManager", "--streaming"
            ]
            if model_name == "enc_dec":
                benchmark += [
                    "--encoder_engine_dir",
                    str(encoder_model_engine_path)
                ]
            run_command(benchmark, cwd=root_dir, timeout=600)

            # gptManager streaming test with delay
            benchmark = [
                str(benchmark_exe_dir / "gptManagerBenchmark"), "--engine_dir",
                str(model_engine_path), "--type", "IFB", "--dataset",
                str(data_dir / tokens_f), "--api", "gptManager", "--streaming",
                "request_rate", "100", "--enable_exp_delays"
            ]
            if model_name == "enc_dec":
                benchmark += [
                    "--encoder_engine_dir",
                    str(encoder_model_engine_path)
                ]
            run_command(benchmark, cwd=root_dir, timeout=600)


if __name__ == "__main__":
    _log.basicConfig(level=_log.INFO)
    parser = _arg.ArgumentParser()

    build_config_group = "Build config"
    build_config_parser = parser.add_argument_group(
        build_config_group, "Configure TensorRT-LLM build")
    add_build_arguments(build_config_parser)
    build_config_parser.set_defaults(install=True, skip_building_wheel=True)

    test_config_group = "Tests config"
    tests_config_parser = parser.add_argument_group(test_config_group,
                                                    "Configure tests")

    tests_config_parser.add_argument("--model_cache",
                                     type=str,
                                     help="Directory where models are stored")
    tests_config_parser.add_argument(
        "--build_only",
        action="store_true",
        help=
        "Only build engines and generate expected outputs, do not run tests.")
    tests_config_parser.add_argument(
        "--skip_unit_tests",
        action="store_true",
        help="Skip unit tests. Only run model tests.")
    tests_config_parser.add_argument("--run_all_models",
                                     action="store_true",
                                     help="Run the tests for all models")
    tests_config_parser.add_argument("--run_gpt",
                                     action="store_true",
                                     help="Run the tests for GPT")
    tests_config_parser.add_argument("--run_gptj",
                                     action="store_true",
                                     help="Run the tests for GPT-J")
    tests_config_parser.add_argument("--run_llama",
                                     action="store_true",
                                     help="Run the tests for Llama")
    tests_config_parser.add_argument("--run_chatglm",
                                     action="store_true",
                                     help="Run the tests for ChatGLM")
    tests_config_parser.add_argument("--run_medusa",
                                     action="store_true",
                                     help="Run the tests for Medusa")
    tests_config_parser.add_argument("--run_mamba",
                                     action="store_true",
                                     help="Run the tests for Mamba")
    tests_config_parser.add_argument("--run_recurrentgemma",
                                     action="store_true",
                                     help="Run the tests for RecurrentGemma")
    tests_config_parser.add_argument("--run_encoder",
                                     action="store_true",
                                     help="Run the tests for BART encoder")
    tests_config_parser.add_argument("--run_bart",
                                     action="store_true",
                                     help="Run the tests for BART")
    tests_config_parser.add_argument("--run_t5",
                                     action="store_true",
                                     help="Run the tests for T5")
    tests_config_parser.add_argument("--run_redrafter",
                                     action="store_true",
                                     help="Run the tests for ReDrafter")
    tests_config_parser.add_argument(
        "--run_fp8",
        action="store_true",
        help="Additionally run FP8 tests. Implemented for H100 runners.")
    tests_config_parser.add_argument(
        "--only_multi_gpu",
        action="store_true",
        help="Run only mulit-GPU tests. Implemented for 4 GPUs.")
    tests_config_parser.add_argument("--test_timeout",
                                     type=int,
                                     help="Timeout for tests.")

    args = parser.parse_args()

    arg_groups = {}
    for group in parser._action_groups:
        group_dict = {
            a.dest: getattr(args, a.dest, None)
            for a in group._group_actions
        }
        arg_groups[group.title] = _arg.Namespace(**group_dict)

    build_args = arg_groups[build_config_group]
    build_trt_llm(**vars(build_args))

    test_args = arg_groups[test_config_group]
    test_args.build_dir = get_build_dir(build_args.build_dir,
                                        build_args.build_type)
    # Make modelSpec module since build engine and generate output scripts will need it.
    make_modelSpec = [
        "cmake", "--build",
        test_args.build_dir.__str__(), "--config", build_args.build_type, "-j",
        "--target", "modelSpec"
    ]
    run_command(make_modelSpec, cwd=build_args.build_dir, timeout=300)

    from build_engines_utils import init_model_spec_module

    init_model_spec_module(force_init_trtllm_bindings=False)

    if test_args.run_all_models:
        test_args.run_gpt = True
        test_args.run_gptj = True
        test_args.run_llama = True
        test_args.run_chatglm = True
        test_args.run_mamba = True
        test_args.run_recurrentgemma = True
        test_args.run_encoder = True
        test_args.run_bart = True
        test_args.run_t5 = True
        test_args.run_medusa = True
        test_args.run_redrafter = True

    del test_args.run_all_models

    do_memory_profiling = _os.environ.get('LLM_MEMORY_PROFILING', False)
    if do_memory_profiling:
        unix_socket = "/tmp/profiling_scribe.unix"

        scribe = _sp.Popen([
            "/usr/bin/python3",
            find_root_dir() /
            "tests/llm-test-defs/turtle/defs/profiling_scribe.py", "-l",
            unix_socket
        ])

        while not _os.path.exists(unix_socket):
            _time.sleep(0.1)

    try:
        run_tests(**vars(test_args))
    finally:
        if do_memory_profiling:
            scribe.send_signal(signal.SIGINT)
            scribe.wait(timeout=10)
            scribe.kill()
