import copy
import logging as _logger
import os as _os
import pathlib as _pl
from typing import List

import defs.cpp.cpp_common as _cpp
import pytest


def run_single_gpu_tests(build_dir: _pl.Path,
                         test_list: List[str],
                         run_fp8=False,
                         timeout=3600):

    cpp_env = {**_os.environ}
    tests_dir = build_dir / "tests" / "e2e_tests"

    included_tests = list(_cpp.generate_included_model_tests(test_list))

    fname_list = list(_cpp.generate_result_file_name(test_list,
                                                     run_fp8=run_fp8))
    resultFileName = "-".join(fname_list) + ".xml"

    excluded_tests = ["FP8"] if not run_fp8 else []

    excluded_tests.extend(list(_cpp.generate_excluded_test_list(test_list)))

    ctest = ["ctest", "--output-on-failure", "--output-junit", resultFileName]

    if included_tests:
        ctest.extend(["-R", "|".join(included_tests)])
        if excluded_tests:
            ctest.extend(["-E", "|".join(excluded_tests)])

        parallel = _cpp.default_test_parallel
        if parallel_override := _os.environ.get("LLM_TEST_PARALLEL_OVERRIDE",
                                                None):
            parallel = int(parallel_override)

        _cpp.parallel_run_ctest(ctest,
                                cwd=tests_dir,
                                env=cpp_env,
                                timeout=timeout,
                                parallel=parallel)
    if "gpt" in test_list:
        xml_output_file = build_dir / "results-single-gpu-disagg-executor_gpt.xml"
        new_env = copy.copy(cpp_env)
        new_env["TRTLLM_USE_MPI_KVCACHE"] = "1"
        trt_model_test = _cpp.produce_mpirun_command(
            global_commands=["mpirun", "--allow-run-as-root"],
            nranks=2,
            local_commands=[
                "executor/disaggExecutorTest",
                "--gtest_filter=*GptSingleDeviceDisaggSymmetricExecutorTest*"
            ],
            leader_commands=[f"--gtest_output=xml:{xml_output_file}"])
        _cpp.run_command(trt_model_test,
                         cwd=tests_dir,
                         env=new_env,
                         timeout=timeout)

        run_spec_dec_tests(build_dir=build_dir)


def run_benchmarks(
    model_name: str,
    python_exe: str,
    root_dir: _pl.Path,
    build_dir: _pl.Path,
    resources_dir: _pl.Path,
    model_cache: str,
    batching_types: list[str],
    api_types: list[str],
):
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
        _logger.info(
            f"run_benchmark test does not support {model_name}. Skipping benchmarks"
        )
        return NotImplementedError

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
        model_engine_path = model_engine_dir / "fp16_plugin_packed_paged" / "tp1-pp1-cp1-gpu"

        # WAR: Currently importing the bindings here causes a segfault in pybind 11 during shutdown
        # As this just builds a path we hard-code for now to obviate the need for import of bindings

        # model_spec_obj = model_spec.ModelSpec(input_file, _tb.DataType.HALF)
        # model_spec_obj.set_kv_cache_type(_tb.KVCacheType.PAGED)
        # model_spec_obj.use_gpt_plugin()
        # model_spec_obj.use_packed_input()
        # model_engine_path = model_engine_dir / model_spec_obj.get_model_path(
        # ) / "tp1-pp1-cp1-gpu"

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
        _cpp.run_command(prepare_dataset,
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

                _cpp.run_command(benchmark, cwd=root_dir, timeout=600)
                req_rate_benchmark = benchmark + [
                    "--request_rate", "100", "--enable_exp_delays"
                ]
                _cpp.run_command(req_rate_benchmark, cwd=root_dir, timeout=600)
                concurrency_benchmark = benchmark + ["--concurrency", "30"]
                _cpp.run_command(concurrency_benchmark,
                                 cwd=root_dir,
                                 timeout=600)

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
            _cpp.run_command(benchmark, cwd=root_dir, timeout=600)


def run_spec_dec_tests(build_dir: _pl.Path):
    xml_output_file = build_dir / "results-spec-dec-fast-logits.xml"
    cpp_env = {**_os.environ}
    tests_dir = build_dir / "tests" / "e2e_tests"
    trt_model_test = _cpp.produce_mpirun_command(
        global_commands=["mpirun", "--allow-run-as-root"],
        nranks=3,
        local_commands=[
            "executor/executorTest", "--gtest_filter=*SpecDecFastLogits*"
        ],
        leader_commands=[f"--gtest_output=xml:{xml_output_file}"])
    _cpp.run_command(trt_model_test, cwd=tests_dir, env=cpp_env, timeout=1500)


@pytest.fixture(scope="session")
def run_model_tests(build_dir, lora_setup):

    def _run(model_name: str, run_fp8: bool):
        run_single_gpu_tests(
            build_dir=build_dir,
            test_list=[model_name],
            timeout=_cpp.default_test_timeout,
            run_fp8=run_fp8,
        )

    return _run


@pytest.fixture(scope="session")
def run_model_benchmarks(root_dir, build_dir, cpp_resources_dir, python_exe,
                         model_cache):

    def _run(
        model_name: str,
        batching_types: List[str],
        api_types: List[str],
    ):

        run_benchmarks(
            model_name=model_name,
            python_exe=python_exe,
            root_dir=root_dir,
            build_dir=build_dir,
            resources_dir=cpp_resources_dir,
            model_cache=model_cache,
            batching_types=batching_types,
            api_types=api_types,
        )

    return _run


@pytest.mark.parametrize("build_google_tests", ["80", "86", "89", "90"],
                         indirect=True)
@pytest.mark.parametrize("model", [
    "bart", "chatglm", "eagle", "encoder", "enc_dec_language_adapter", "gpt",
    "gpt_executor", "gpt_tests", "llama", "mamba", "medusa", "recurrentgemma",
    "redrafter", "t5"
])
@pytest.mark.parametrize("run_fp8", [False, True], ids=["", "fp8"])
def test_model(build_google_tests, model, prepare_model, run_model_tests,
               run_fp8):

    if model == "recurrentgemma":
        pytest.skip(
            "TODO: fix recurrentgemma OOM with newest version of transformers")
        return

    prepare_model(model, run_fp8)

    run_model_tests(model, run_fp8)


@pytest.mark.skip(reason="https://nvbugs/5601670")
@pytest.mark.parametrize("build_google_tests", ["80", "86", "89", "90"],
                         indirect=True)
@pytest.mark.parametrize("model", ["bart", "gpt", "t5"])
def test_benchmarks(build_benchmarks, model, prepare_model,
                    run_model_benchmarks):

    prepare_model(model)

    batching_types = ["IFB"]
    api_types = ["executor"]

    run_model_benchmarks(
        model_name=model,
        batching_types=batching_types,
        api_types=api_types,
    )
