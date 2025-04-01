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
import json
import os
import re
import shutil
import sys
import tempfile
from difflib import SequenceMatcher
from pathlib import Path
from typing import Optional, Tuple, Union

import pytest
import yaml
from defs.common import convert_weights
from defs.trt_test_alternative import (check_call, check_call_negative_test,
                                       check_output, exists, makedirs)

from .common import (PluginOptions, convert_weights, prune_checkpoint,
                     quantize_data, refit_model, venv_check_call)
from .conftest import (llm_models_root, skip_nvlink_inactive, skip_pre_ada,
                       skip_pre_blackwell, skip_pre_hopper, tests_path,
                       unittest_path)

sys.path.append(os.path.join(str(tests_path()), '/../examples/apps'))


def test_gpt3_175b_1layers_build_only(llm_root, llm_venv, engine_dir):
    "Build GPT-3 175B: 96 layer w/ plugins"
    example_root = os.path.join(llm_root, "examples", "gpt")
    engine_dir = os.path.join(engine_dir, "gpt-175-96layers-build-only")

    dtype = 'float16'
    convert_cmd = [
        f"{example_root}/../generate_checkpoint_config.py",
        f"--output_path={engine_dir}/ckpt_config.json",
        "--architecture=GPTForCausalLM", f"--dtype={dtype}",
        "--num_hidden_layers=1", "--num_attention_heads=96",
        "--hidden_size=12288", "--vocab_size=51200", "--tp_size=8"
    ]
    venv_check_call(llm_venv, convert_cmd)

    print("Building engines...")
    build_cmd = [
        "trtllm-build",
        f"--model_config={engine_dir}/ckpt_config.json",
        f"--output_dir={engine_dir}",
        "--max_batch_size=256",
        "--max_input_len=200",
        "--max_seq_len=400",
        "--max_beam_width=1",
        f"--gpt_attention_plugin={dtype}",
    ]
    check_call(" ".join(build_cmd), shell=True, env=llm_venv._new_env)


@pytest.mark.parametrize("additional_build_option", ["", "--multi_query_mode"],
                         ids=lambda x: x.strip("-"))
@pytest.mark.parametrize("use_py_session", [False, True],
                         ids=["use_cpp_session", "use_py_session"])
def test_gpt_fp32(llm_root, llm_venv, additional_build_option, use_py_session,
                  engine_dir):
    example_root = os.path.join(llm_root, "examples", "gpt")
    engine_dir = os.path.join(engine_dir, "gpt2")

    dtype = 'float32'
    convert_cmd = [
        f"{example_root}/../generate_checkpoint_config.py",
        f"--output_path={engine_dir}/ckpt_config.json",
        "--architecture=GPTForCausalLM", f"--dtype={dtype}",
        "--num_hidden_layers=2", "--num_attention_heads=16",
        "--hidden_size=1024", "--vocab_size=51200"
    ]
    if 'multi_query_mode' in additional_build_option:
        convert_cmd.append("--num_key_value_heads=1")
    venv_check_call(llm_venv, convert_cmd)

    print("Building engines...")
    build_cmd = [
        "trtllm-build",
        f"--model_config={engine_dir}/ckpt_config.json",
        f"--output_dir={engine_dir}",
        "--max_batch_size=256",
        "--max_input_len=200",
        "--max_seq_len=400",
        "--max_beam_width=1",
        f"--gpt_attention_plugin={dtype}",
    ]
    check_call(" ".join(build_cmd), shell=True, env=llm_venv._new_env)

    print("Running inference...")
    run_cmd = [
        f"{example_root}/../run.py", "--max_output_len=1",
        f"--engine_dir={engine_dir}"
    ]
    if use_py_session:
        run_cmd.extend(["--use_py_session"])
    venv_check_call(llm_venv, run_cmd)


@pytest.mark.parametrize("prune", [False, True], ids=["", "prune"])
@pytest.mark.parametrize(
    "additional_build_option",
    ["", "remove_input_padding", "quantization int8_sq_per_tensor"],
    ids=lambda x: x.replace(" ", "_"))
@pytest.mark.parametrize("use_py_session", [False, True],
                         ids=["use_cpp_session", "use_py_session"])
def test_llama_e2e(llama_example_root, llama_tokenizer_model_root, llm_venv,
                   cmodel_dir, engine_dir, additional_build_option,
                   use_py_session, prune):

    model_name = 'llama-e2e'
    model_dir = convert_weights(
        llm_venv=llm_venv,
        example_root=llama_example_root,
        cmodel_dir=cmodel_dir,
        model=model_name,
        model_path=llama_tokenizer_model_root,
    )

    unpruned_model_dir = model_dir
    if prune:
        print("Pruning checkpoint...")
        model_dir = prune_checkpoint(llm_venv, model_dir)

    build_cmd = [
        "trtllm-build", f"--checkpoint_dir={model_dir}",
        f"--output_dir={engine_dir}", f"--max_beam_width=4",
        f"--max_batch_size={1}", f"--max_input_len={1024}",
        f"--gpt_attention_plugin=float16", f"--gemm_plugin=float16"
    ]

    print("Build engines...")

    if additional_build_option == "":
        build_cmd += [f"--remove_input_padding=disable"]
    elif additional_build_option == "remove_input_padding":
        build_cmd += [f"--remove_input_padding=enable"]
    else:
        build_cmd += [f"--{additional_build_option}"]

    if prune:
        build_cmd.append("--strip_plan")

    build_cmd.extend(PluginOptions("float16", None, "float16", None).to_args())

    check_call(" ".join(build_cmd), shell=True, env=llm_venv._new_env)

    if prune:
        print("Refitting engine...")
        engine_dir = refit_model(llm_venv, engine_dir, unpruned_model_dir)

    print("Run inference...")
    run_cmd = [
        f"{llama_example_root}/../run.py",
        "--max_output_len=1",
        f"--tokenizer_dir={llama_tokenizer_model_root}",
        "--log_level=verbose",
        f"--engine_dir={engine_dir}",
    ]
    if use_py_session:
        run_cmd.extend(["--use_py_session"])
    venv_check_call(llm_venv, run_cmd)


@pytest.mark.parametrize("prune", [False, True], ids=["", "prune"])
@pytest.mark.parametrize("enable_fp8", [False, True], ids=["", "enable_fp8"])
@pytest.mark.parametrize("additional_build_option",
                         ["", "remove_input_padding"],
                         ids=lambda x: x)
@pytest.mark.parametrize("use_py_session", [False, True],
                         ids=["use_cpp_session", "use_py_session"])
def test_mistral_e2e(llama_example_root, llama_tokenizer_model_root, llm_venv,
                     cmodel_dir, engine_dir, enable_fp8,
                     additional_build_option, use_py_session, prune):

    model_name = 'mistral-e2e'
    if enable_fp8:
        model_dir = quantize_data(llm_venv=llm_venv,
                                  example_root=llama_example_root,
                                  model_dir=llama_tokenizer_model_root,
                                  dtype='float16',
                                  qformat='fp8',
                                  quantize_dir=cmodel_dir,
                                  kv_cache_dtype='fp8',
                                  calib_size=32)
    else:
        model_dir = convert_weights(llm_venv=llm_venv,
                                    example_root=llama_example_root,
                                    cmodel_dir=cmodel_dir,
                                    model=model_name,
                                    model_path=llama_tokenizer_model_root,
                                    enable_fp8=enable_fp8)

    unpruned_model_dir = model_dir
    if prune:
        print("Pruning checkpoint...")
        model_dir = prune_checkpoint(llm_venv, model_dir)

    build_cmd = [
        "trtllm-build",
        f"--checkpoint_dir={model_dir}",
        f"--output_dir={engine_dir}",
        f"--max_batch_size=1",
        f"--max_input_len=1024",
        f"--max_num_tokens=1024",
        f"--max_beam_width=4",
        f"--gemm_plugin=float16",
    ]
    print("Build engines...")

    if additional_build_option == "":
        if not enable_fp8:
            build_cmd += [f"--remove_input_padding=disable"]
    elif additional_build_option == "remove_input_padding":
        build_cmd += [f"--remove_input_padding=enable"]
    else:
        build_cmd += [f"--{additional_build_option}"]

    if enable_fp8:
        build_cmd.append("--use_fp8_context_fmha=enable")
    else:
        build_cmd.append("--context_fmha=disable")
        build_cmd.append("--gpt_attention_plugin=float16")
        build_cmd.extend(
            PluginOptions("float16", None, "float16", None).to_args())
    if prune:
        build_cmd.append("--strip_plan")

    os.path.join(cmodel_dir, ".internal_trt.cfg")
    check_call(" ".join(build_cmd), shell=True, env=llm_venv._new_env)

    if prune:
        print("Refitting engine...")
        engine_dir = refit_model(llm_venv, engine_dir, unpruned_model_dir)

    print("Run inference...")
    run_cmd = [
        f"{llama_example_root}/../run.py",
        "--max_output_len=1",
        f"--tokenizer_dir={llama_tokenizer_model_root}",
        "--log_level=verbose",
        "--max_attention_window_size=5",
        f"--engine_dir={engine_dir}",
    ]
    if use_py_session:
        run_cmd.extend(["--use_py_session"])
    venv_check_call(llm_venv, run_cmd)


def trtllm_bench_prolog(
        llm_root,
        llm_venv,
        engine_dir: Optional[str],
        model_subdir,
        model_name: str,
        quant: str,
        streaming: bool,
        skip_engine_build: bool = False
) -> Union[Tuple[Path, Path, Path], Path]:
    ''' Optionally build engine and generate dataset for benchmark.

    Returns:
        Union[Tuple[Path, Path, Path], Path]:
            - Tuple containing model_path, engine_path, and dataset_path.
            - A single dataset_path object if skip_engine_build is True.
    '''

    llm_models = llm_models_root()
    # skip when llm_models_root is None
    if llm_models is None:
        return

    model_path = Path(llm_models, model_subdir).absolute()
    engine_path = None
    quant_name = quant if quant is not None else "FP16"
    stream_mode = "streaming" if streaming else "non-streaming"
    benchmark_name = f"trtllm-bench-sanity-{quant_name}-{stream_mode}"
    benchmark_name += "-pytorch-backend" if skip_engine_build else benchmark_name
    dataset_tool = Path(llm_root, "benchmarks", "cpp", "prepare_dataset.py")

    work_dir = Path(tempfile.TemporaryDirectory().name
                    ) if skip_engine_build else Path(engine_dir)
    dataset_path = Path(work_dir, f"{benchmark_name}.txt")
    # Clean up an existing directory if it exists
    shutil.rmtree(work_dir, ignore_errors=True)
    # Generate a small dataset to run a test.
    work_dir.mkdir(parents=True)
    dataset_output = llm_venv.run_cmd(
        [
            f"{dataset_tool.resolve()}",
            "--stdout",
            "--tokenizer",
            f"{model_path}",
            "token-norm-dist",
            "--input-mean",
            "128",
            "--output-mean",
            "128",
            "--input-stdev",
            "0",
            "--output-stdev",
            "0",
            "--num-requests",
            "10",
        ],
        caller=check_output,
    )
    # Grab the stdout and write it to a dataset file for passing to suite.
    with open(dataset_path, "w") as dataset:
        dataset.write(dataset_output)

    if not skip_engine_build:
        build_cmd = \
            f"trtllm-bench " \
            f"--model {model_name} " \
            f"--model_path {model_path} " \
            f"--workspace {work_dir} " \
            f"build --tp_size 1"

        if quant is not None:
            build_cmd = f"{build_cmd} --quantization {quant}"

        build_cmd = f"{build_cmd} --dataset {dataset_path}"
        build_output = check_output(build_cmd, shell=True)

        for line in build_output.split("\n")[::-1]:
            if line.startswith("ENGINE SAVED:"):
                engine_path = Path(line.split(":")[1])
                break

    return model_path, engine_path, dataset_path


@pytest.fixture
def temp_extra_llm_api_options_file(request):
    if request.node.callspec.params['use_extra_config']:
        temp_dir = tempfile.gettempdir()
        temp_file_path = os.path.join(temp_dir, "extra_llm_api_options.yaml")
        try:
            extra_llm_api_options_dict = {
                "enable_chunked_prefill": False,
                "kv_cache_config": {
                    "enable_block_reuse": False,
                    "max_tokens": 40000
                }
            }

            if request.node.callspec.params['pytorch_backend_config']:
                extra_llm_api_options_dict["pytorch_backend_config"] = {
                    "enable_overlap_scheduler": True,
                    "use_cuda_graph": True,
                    "cuda_graph_batch_sizes": [1, 2, 3],
                }

            with open(temp_file_path, 'w') as f:
                yaml.dump(extra_llm_api_options_dict, f)

            yield temp_file_path
        finally:
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)
    else:
        assert not request.node.callspec.params['pytorch_backend_config']
        yield None


@pytest.mark.parametrize("model_subdir", [
    "llama-3.1-model/Meta-Llama-3.1-8B",
],
                         ids=lambda x: x.strip("-"))
@pytest.mark.parametrize(
    "model_name",
    [
        "meta-llama/Llama-3.1-8B",
    ],
)
@pytest.mark.parametrize("quant", [None, "FP8"], ids=["FP16", "FP8"])
@pytest.mark.parametrize("streaming", ["", "--streaming"],
                         ids=["non-streaming", "streaming"])
@pytest.mark.parametrize("use_extra_config", [True, False],
                         ids=["extra_config", ""])
@pytest.mark.parametrize("pytorch_backend_config", [False], ids=[""])
def test_trtllm_bench_sanity(llm_root, llm_venv, engine_dir, model_subdir,
                             model_name, quant, streaming, use_extra_config,
                             pytorch_backend_config,
                             temp_extra_llm_api_options_file):
    '''
    sanity check on the new benchmark script to make sure it works
    - meta-llama/Llama-3.1-8B for baseline
    - fp16 and fp8 to test quantization
    '''

    model_path, engine_path, dataset_path = trtllm_bench_prolog(
        llm_root, llm_venv, engine_dir, model_subdir, model_name, quant,
        "streaming" in streaming)

    benchmark_cmd = \
        f"trtllm-bench --model {model_name} --model_path {model_path} " \
        f"throughput --engine_dir {engine_path} " \
        f"--dataset {dataset_path} {streaming}"

    assert not pytorch_backend_config
    if use_extra_config:
        benchmark_cmd += f" --extra_llm_api_options {temp_extra_llm_api_options_file}"
    check_call(benchmark_cmd, shell=True)


@pytest.mark.parametrize(
    "model_name, llama_model_root, use_extra_config, pytorch_backend_config",
    [('meta-llama/Llama-3.1-8B', 'llama-3.1-8b', False, False),
     pytest.param('meta-llama/Llama-3.1-8B',
                  'llama-3.1-8b-instruct-hf-fp8',
                  True,
                  False,
                  marks=skip_pre_hopper),
     pytest.param('meta-llama/Llama-3.1-8B',
                  'llama-3.1-8b-instruct-hf-fp8',
                  True,
                  True,
                  marks=skip_pre_hopper),
     pytest.param('meta-llama/Llama-3.1-8B',
                  'llama-3.1-8b-hf-nvfp4',
                  False,
                  False,
                  marks=skip_pre_blackwell)],
    indirect=['llama_model_root'])
def test_trtllm_bench_pytorch_backend_sanity(llm_root, llm_venv,
                                             llama_model_root, model_name,
                                             use_extra_config,
                                             pytorch_backend_config,
                                             temp_extra_llm_api_options_file):
    '''
    sanity check on latency benchmark for LLM API with PyTorch backend
    '''
    model_path, _, dataset_path = trtllm_bench_prolog(llm_root,
                                                      llm_venv,
                                                      None,
                                                      llama_model_root,
                                                      model_name,
                                                      False,
                                                      False,
                                                      skip_engine_build=True)

    benchmark_cmd = \
        f"trtllm-bench --model {model_name} --model_path {model_path} " \
        f"throughput " \
        f"--dataset {dataset_path} --backend 'pytorch'"

    if use_extra_config:
        benchmark_cmd += f" --extra_llm_api_options {temp_extra_llm_api_options_file}"

    check_call(benchmark_cmd, shell=True)


def test_trtllm_bench_mgmn(llm_root, llm_venv):
    model_name = "meta-llama/Llama-3.1-8B"
    llama_model_dir = Path(
        llm_models_root()) / "llama-3.1-model/Llama-3.1-8B-Instruct"
    dataset_path = trtllm_bench_prolog(llm_root,
                                       llm_venv,
                                       engine_dir=None,
                                       model_subdir=llama_model_dir,
                                       model_name=model_name,
                                       quant=None,
                                       streaming=False,
                                       skip_engine_build=True)
    benchmark_cmd = \
        f"mpirun -n 2 trtllm-llmapi-launch trtllm-bench --model {model_name} " \
        f"--model_path {llama_model_dir} " \
        f"throughput " \
        f"--dataset {dataset_path} --backend pytorch --tp 2"

    check_call(benchmark_cmd, shell=True)


@pytest.mark.parametrize("model_subdir", [
    "llama-3.1-model/Meta-Llama-3.1-8B",
],
                         ids=lambda x: x.strip("-"))
@pytest.mark.parametrize(
    "model_name",
    [
        "meta-llama/Llama-3.1-8B",
    ],
)
@pytest.mark.parametrize("quant", [None, "FP8"], ids=["FP16", "FP8"])
def test_trtllm_bench_latency_sanity(llm_root, llm_venv, engine_dir,
                                     model_subdir, model_name, quant):
    '''
    sanity check on the new benchmark script to make sure it works
    - meta-llama/Llama-3.1-8B for baseline
    - fp16 and fp8 to test quantization
    '''

    model_path, engine_path, dataset_path = trtllm_bench_prolog(llm_root,
                                                                llm_venv,
                                                                engine_dir,
                                                                model_subdir,
                                                                model_name,
                                                                quant,
                                                                streaming=True)

    benchmark_cmd = \
        f"trtllm-bench --model {model_path} latency --engine_dir {engine_path} " \
        f"--dataset {dataset_path}"
    check_call(benchmark_cmd, shell=True)


@pytest.mark.parametrize(
    "model_name",
    [
        "meta-llama/Llama-3.1-8B",
    ],
)
def test_trtllm_bench_help_sanity(model_name):
    '''
    Sanity check that the options are defined properly by printing out help
    '''
    check_call("trtllm-bench --help", shell=True)
    check_call(f"trtllm-bench --model {model_name} build --help", shell=True)
    check_call(f"trtllm-bench --model {model_name} throughput --help",
               shell=True)
    check_call(f"trtllm-bench --model {model_name} latency --help", shell=True)


@pytest.mark.parametrize("request_rate", [False, True],
                         ids=["", "enable_request_rate"])
@pytest.mark.parametrize("concurrency", [False, True],
                         ids=["", "enable_concurrency"])
def test_trtllm_bench_request_rate_and_concurrency(llm_root, llm_venv,
                                                   engine_dir, request_rate,
                                                   concurrency):
    '''
    sanity check on the trtllm-bench new request rate and concurrency API
    '''
    model_subdir = "llama-3.1-model/Meta-Llama-3.1-8B"
    model_name = "meta-llama/Llama-3.1-8B"

    model_path, engine_path, dataset_path = trtllm_bench_prolog(llm_root,
                                                                llm_venv,
                                                                engine_dir,
                                                                model_subdir,
                                                                model_name,
                                                                quant=None,
                                                                streaming=False)

    benchmark_cmd = \
        f"trtllm-bench --model {model_path} throughput --engine_dir {engine_path} " \
        f"--dataset {dataset_path}"

    if request_rate:
        benchmark_cmd += " --request_rate 100"
    if concurrency:
        benchmark_cmd += " --concurrency 100"

    if request_rate and concurrency:
        # negative test, request rate and concurrency should not be turned on at the same time
        check_call_negative_test(benchmark_cmd, shell=True)
    else:
        check_call(benchmark_cmd, shell=True)


@pytest.mark.parametrize("model_subdir", [
    "llama-3.1-model/Meta-Llama-3.1-8B",
],
                         ids=lambda x: x.strip("-"))
@pytest.mark.parametrize(
    "model_name",
    [
        "meta-llama/Llama-3.1-8B",
    ],
)
@pytest.mark.parametrize("streaming", [True, False],
                         ids=["non-streaming", "streaming"])
@pytest.mark.parametrize("backend", [None, "pytorch"], ids=["TRT", "PyTorch"])
def test_trtllm_bench_iteration_log(llm_root, llm_venv, model_name,
                                    model_subdir, streaming, backend):
    '''
    Test the iteration log functionality with necessary options
    '''
    iteration_log = None
    engine_dir = None

    try:
        skip_engine_build = backend is not None
        iteration_log = tempfile.mkstemp(dir="/tmp", suffix=".txt")[1]
        if not skip_engine_build:
            engine_dir = tempfile.mkdtemp(dir="/tmp")

        model_path, engine_path, dataset_path = trtllm_bench_prolog(
            llm_root,
            llm_venv,
            engine_dir,
            model_subdir,
            model_name,
            quant=None,
            skip_engine_build=skip_engine_build,
            streaming=streaming)

        benchmark_cmd = \
            f"trtllm-bench --model {model_path} throughput " \
            f"--dataset {dataset_path} --iteration_log {iteration_log}"

        if streaming:
            benchmark_cmd += " --streaming"

        if skip_engine_build:
            assert engine_path is None, "Engine path should be None"
            benchmark_cmd += f" --backend {backend}"
        else:
            assert engine_path is not None, "Engine path should not be None"
            benchmark_cmd += f" --engine_dir {engine_path}"

        check_call(benchmark_cmd, shell=True)

        assert os.path.exists(
            iteration_log
        ), f"Iteration log file {iteration_log} was not created."
        if os.path.getsize(iteration_log) == 0:
            raise AssertionError(
                f"Iteration log file {iteration_log} is empty.")
    finally:
        if iteration_log:
            shutil.rmtree(iteration_log, ignore_errors=True)
        if engine_dir:
            shutil.rmtree(engine_dir, ignore_errors=True)


@pytest.mark.parametrize("model_name", [
    "gpt_350m", "gpt_350m_sq_per_tensor", "llama_70b", "bert_base",
    "falcon_40b", "t5_base", "roberta_base"
],
                         ids=lambda x: x.strip("-"))
def test_benchmark_sanity(llm_root, llm_venv, model_name, engine_dir):
    '''
    sanity check on the benchmark script to make sure it works
    - gpt_350m for gpt baseline.
    - gpt_350m_sq_per_tensor for testing SQ
    - llama_70b for GQA (num_kv_heads < num_heads) in gpt benchmark script.
    - bert_base for bert baseline.
    - t5_base for t5 baseline.
    '''
    build_script_root = os.path.join(llm_root, "tests/integration/defs/perf")
    benchmark_root = os.path.join(llm_root, "benchmarks", "python")
    engine_dir = os.path.join(engine_dir, model_name, "benchmark-sanity")
    if not exists(engine_dir):
        makedirs(engine_dir)

    # max batch size 256 (default) is OOM on A30, changing to a smaller one to just test sanity
    build_args = f"-m {model_name} --force_num_layer_1 --max_input_len 512 --max_batch_size 8"
    # test OOTB path in one of the model
    if model_name == "gpt_350m":
        build_args += " --mode ootb"
    build_cmd = f'{build_script_root}/build.py --output_dir {engine_dir} {build_args}'.split(
        " ")

    benchmark_args = f"--batch_size 1;2 --duration 0 --num_runs 1"
    if 'bert' in model_name:
        benchmark_args += " --input_len 20;60"
        benchmark_args += " --m enc"
    else:
        benchmark_args += " --input_output_len 20,60;60,20"
        if 't5' in model_name or 'roberta' in model_name:
            benchmark_args += " --m enc-dec"
    load_cmd = f'{benchmark_root}/benchmark.py --engine_dir {engine_dir} {benchmark_args}'.split(
        " ")

    venv_check_call(llm_venv, build_cmd)
    venv_check_call(llm_venv, load_cmd)


@skip_pre_ada
@pytest.mark.parametrize("model_name",
                         ["llama_7b", "gptj_6b", "gpt_350m", "falcon_40b"],
                         ids=lambda x: x.strip("-"))
def test_benchmark_sanity_enable_fp8(llm_root, llm_venv, model_name,
                                     engine_dir):
    '''
    sanity check on the benchmark script to make sure it works
    '''
    build_script_root = os.path.join(llm_root, "tests/integration/defs/perf")
    benchmark_root = os.path.join(llm_root, "benchmarks", "python")
    engine_dir = os.path.join(engine_dir, model_name, "benchmark-sanity")
    if not exists(engine_dir):
        makedirs(engine_dir)
    build_args = f"-m {model_name} --force_num_layer_1 --quantization fp8"
    build_cmd = f'{build_script_root}/build.py --output_dir {engine_dir} {build_args}'.split(
        " ")

    benchmark_args = f"--batch_size 1;2 --duration 0 --num_runs 1 --quantization fp8"
    if 'bert' in model_name:
        benchmark_args += " --input_len 20;60"
        benchmark_args += " --m enc"
    else:
        benchmark_args += " --input_output_len 20,60;60,20"
    load_cmd = f'{benchmark_root}/benchmark.py --engine_dir {engine_dir} {benchmark_args}'.split(
        " ")
    venv_check_call(llm_venv, build_cmd)
    venv_check_call(llm_venv, load_cmd)


def test_chatglm_6b_sanity(chatglm_6b_example_root, llm_venv, cmodel_dir,
                           engine_dir, update_transformers):
    llm_models = llm_models_root()

    # skip when llm_models_root is None
    if llm_models is None:
        return

    # Use `chatglm_6b_example_root` as temporary tokenizer path since we need replace the `tokenization_chatglm.py`
    model_path = Path(llm_models) / 'chatglm-6b'
    for file in (list(model_path.glob("*.py")) +
                 list(model_path.glob("*.json")) +
                 list(model_path.glob("ice_text.model"))):
        print(file.name)
        if "tokenization_chatglm.py" in file.name:
            continue
        shutil.copy(
            file,
            chatglm_6b_example_root + "/chatglm-6b/tokenization_chatglm.py")

    dtype = 'float16'
    ckpt_dir = convert_weights(llm_venv=llm_venv,
                               example_root=chatglm_6b_example_root,
                               cmodel_dir=cmodel_dir,
                               model='chatglm-6b',
                               model_path=str(model_path),
                               data_type=dtype)
    build_cmd = [
        "trtllm-build",
        f"--checkpoint_dir={ckpt_dir}",
        f"--output_dir={engine_dir}",
        f"--max_batch_size={8}",
        f"--max_input_len={924}",
        f"--max_seq_len={1024}",
        f"--max_beam_width={1}",
        f"--gemm_plugin={dtype}",
        f"--gpt_attention_plugin={dtype}",
        "--context_fmha=disable",
    ]
    check_call(" ".join(build_cmd), shell=True, env=llm_venv._new_env)
    run_cmd = [
        f"{chatglm_6b_example_root}/../run.py",
        f"--engine_dir={engine_dir}",
        f"--tokenizer_dir={chatglm_6b_example_root}",
        "--max_output_len=10",
    ]
    venv_check_call(llm_venv, run_cmd)


def test_chatglm2_6b_sanity(chatglm2_6b_example_root, llm_venv, cmodel_dir,
                            engine_dir):
    llm_models = llm_models_root()
    # skip when llm_models_root is None
    if llm_models is None:
        return

    dtype = 'float16'
    ckpt_dir = convert_weights(llm_venv=llm_venv,
                               example_root=chatglm2_6b_example_root,
                               cmodel_dir=cmodel_dir,
                               model='chatglm2-6b',
                               model_path=f'{llm_models}/chatglm2-6b',
                               data_type=dtype)
    build_cmd = [
        "trtllm-build",
        f"--checkpoint_dir={ckpt_dir}",
        f"--output_dir={engine_dir}",
        f"--max_batch_size={8}",
        f"--max_input_len={924}",
        f"--max_seq_len={1024}",
        f"--max_beam_width={1}",
        f"--gemm_plugin={dtype}",
        f"--gpt_attention_plugin={dtype}",
    ]
    check_call(" ".join(build_cmd), shell=True, env=llm_venv._new_env)
    run_cmd = [
        f"{chatglm2_6b_example_root}/../run.py", f"--engine_dir={engine_dir}",
        f"--tokenizer_dir={llm_models}/chatglm2-6b", "--max_output_len=10"
    ]
    venv_check_call(llm_venv, run_cmd)


def test_chatglm3_6b_sanity(chatglm3_6b_example_root, llm_venv, cmodel_dir,
                            engine_dir):
    llm_models = llm_models_root()
    # skip when llm_models_root is None
    if llm_models is None:
        return

    dtype = 'float16'
    ckpt_dir = convert_weights(llm_venv=llm_venv,
                               example_root=chatglm3_6b_example_root,
                               cmodel_dir=cmodel_dir,
                               model='chatglm3-6b',
                               model_path=f'{llm_models}/chatglm3-6b',
                               data_type=dtype)
    build_cmd = [
        "trtllm-build",
        f"--checkpoint_dir={ckpt_dir}",
        f"--output_dir={engine_dir}",
        f"--max_batch_size={8}",
        f"--max_input_len={924}",
        f"--max_seq_len={1024}",
        f"--max_beam_width={1}",
        f"--gemm_plugin={dtype}",
        f"--gpt_attention_plugin={dtype}",
    ]
    check_call(" ".join(build_cmd), shell=True, env=llm_venv._new_env)
    run_cmd = [
        f"{chatglm3_6b_example_root}/../run.py", f"--engine_dir={engine_dir}",
        f"--tokenizer_dir={llm_models}/chatglm3-6b", "--max_output_len=10"
    ]
    venv_check_call(llm_venv, run_cmd)


@pytest.mark.parametrize("data_type", ["float16", "bfloat16"])
def test_glm_10b_sanity(glm_10b_example_root, llm_venv, data_type, cmodel_dir,
                        engine_dir):
    llm_models = llm_models_root()
    # skip when llm_models_root is None
    if llm_models is None:
        return

    dtype = 'float16'
    ckpt_dir = convert_weights(llm_venv=llm_venv,
                               example_root=glm_10b_example_root,
                               cmodel_dir=cmodel_dir,
                               model='glm-10b',
                               model_path=f'{llm_models}/glm-10b',
                               data_type=dtype)
    build_cmd = [
        "trtllm-build",
        f"--checkpoint_dir={ckpt_dir}",
        f"--output_dir={engine_dir}",
        f"--max_batch_size={8}",
        f"--max_input_len={924}",
        f"--max_seq_len={1024}",
        f"--max_beam_width={1}",
        f"--gemm_plugin={dtype}",
        f"--gpt_attention_plugin={dtype}",
        "--context_fmha=disable",
    ]
    check_call(" ".join(build_cmd), shell=True, env=llm_venv._new_env)
    run_cmd = [
        f"{glm_10b_example_root}/../run.py", f"--engine_dir={engine_dir}",
        f"--tokenizer_dir={llm_models}/glm-10b", "--max_output_len=10"
    ]
    venv_check_call(llm_venv, run_cmd)


@pytest.mark.parametrize("query_type", ["mha", "mqa", "gqa"])
@pytest.mark.parametrize("use_py_session", [False, True],
                         ids=["use_cpp_session", "use_py_session"])
@pytest.mark.parametrize("gpu_weight_percent", [-1, 0, 0.8],
                         ids=["", "gpu_percent_0", "gpu_percent_0_8"])
def test_falcon_e2e(falcon_example_root, llm_venv, engine_dir, query_type,
                    use_py_session, gpu_weight_percent):
    print(f"Build engines... query_type: {query_type}")

    dtype = "float16"
    config = {
        'architecture': 'FalconForCausalLM',
        'dtype': dtype,
        'num_hidden_layers': 2,
        'num_attention_heads': 16,
        'num_key_value_heads': 16,
        'hidden_size': 4096,
        'vocab_size': 65024,
        'position_embedding_type': 'rope_gpt_neox',
        'max_position_embeddings': 2048,
        'hidden_act': 'gelu',
        'bias': False,
        'parallel_attention': False,
        'new_decoder_architecture': False,
    }
    if query_type == 'mha':
        config['position_embedding_type'] = 'alibi_with_scale'
    elif query_type == 'mqa':
        config['num_key_value_heads'] = 1
        config['parallel_attention'] = True
    elif query_type == 'gqa':
        config['num_key_value_heads'] = 4
        config['new_decoder_architecture'] = True

    # Save the dummy-weight checkpoint config.json to engine_dir
    if not os.path.exists(engine_dir):
        os.makedirs(engine_dir)
    ckpt_config_path = os.path.join(engine_dir, 'ckpt_config.json')
    with open(ckpt_config_path, 'w') as f:
        json.dump(config, f, indent=4)

    build_cmd = [
        "trtllm-build",
        f"--model_config={ckpt_config_path}",
        f"--output_dir={engine_dir}",
        "--log_level=verbose",
        f"--max_batch_size={1}",
        f"--max_input_len={1024}",
        f"--output_dir={engine_dir}",
        "--log_level=verbose",
    ]

    if gpu_weight_percent == -1:
        build_cmd.append(f"--gemm_plugin={dtype}")
    else:
        build_cmd.extend(["--gemm_plugin=disable", "--weight_streaming"])

    if query_type in ('mqa', 'gqa'):
        build_cmd.extend([f"--gpt_attention_plugin={dtype}"])
    check_call(" ".join(build_cmd), shell=True, env=llm_venv._new_env)

    print("Run inference...")
    run_cmd = [
        f"{falcon_example_root}/../run.py",
        "--max_output_len=2",
        "--log_level=verbose",
        f"--engine_dir={engine_dir}",
    ]
    if use_py_session:
        run_cmd.extend(["--use_py_session"])
    if gpu_weight_percent != -1:
        run_cmd.append(f"--gpu_weights_percent={gpu_weight_percent}")

    venv_check_call(llm_venv, run_cmd)


@pytest.mark.parametrize("enable_fp8", [False, True],
                         ids=["enable_fp8", "disable_fp8"])
@pytest.mark.parametrize("enable_ibf", [False, True],
                         ids=["enable_ibf", "disable_ibf"])
@pytest.mark.parametrize("use_py_session", [False, True],
                         ids=["use_cpp_session", "use_py_session"])
def test_falcon_gqa_e2e(falcon_example_root, llm_venv, engine_dir, enable_fp8,
                        enable_ibf, use_py_session):
    dtype = "float16"
    config = {
        'architecture': 'FalconForCausalLM',
        'dtype': dtype,
        'num_hidden_layers': 2,
        'num_attention_heads': 16,
        'num_key_value_heads': 4,
        'hidden_size': 4096,
        'vocab_size': 65024,
        'position_embedding_type': 'rope_gpt_neox',
        'max_position_embeddings': 2048,
        'hidden_act': 'gelu',
        'bias': False,
        'parallel_attention': False,
        'new_decoder_architecture': True,
    }
    if enable_fp8:
        config['quantization'] = {
            'quant_algo': 'FP8',
            'kv_cache_quant_algo': 'FP8'
        }

    # Save the dummy-weight checkpoint config.json to engine_dir
    if not os.path.exists(engine_dir):
        os.makedirs(engine_dir)
    ckpt_config_path = os.path.join(engine_dir, 'ckpt_config.json')
    with open(ckpt_config_path, 'w') as f:
        json.dump(config, f, indent=4)

    build_cmd = [
        "trtllm-build", f"--model_config={ckpt_config_path}",
        f"--output_dir={engine_dir}", "--log_level=verbose",
        f"--gemm_plugin={dtype}", f"--gpt_attention_plugin={dtype}",
        "--max_batch_size=8"
    ]
    if enable_ibf:
        build_cmd.extend(
            ["--remove_input_padding=enable", "--paged_kv_cache=enable"])
    check_call(" ".join(build_cmd), shell=True, env=llm_venv._new_env)

    print("Run inference...")
    run_cmd = [
        f"{falcon_example_root}/../run.py",
        "--max_output_len=2",
        "--log_level=verbose",
        f"--engine_dir={engine_dir}",
    ]
    if use_py_session:
        run_cmd.extend(["--use_py_session"])
    venv_check_call(llm_venv, run_cmd)


run_llm_path = os.path.join(os.path.dirname(__file__), "_run_llmapi_llm.py")


@pytest.mark.parametrize("model_name,model_path", [
    ("llama", "llama-models/llama-7b-hf"),
    ("gptj", "gpt-j-6b"),
    ("falcon", "falcon-7b-instruct"),
    ("llama", "codellama/CodeLlama-7b-Instruct-hf"),
])
def test_llmapi_load_engine_from_build_command(llm_root, llm_venv, engine_dir,
                                               model_name, model_path):
    llama_example_root = os.path.join(llm_root, "examples", model_name)
    dtype = 'float16'
    cmodel_dir = os.path.join(engine_dir, f"{model_name}-engine")

    ckpt_dir = convert_weights(llm_venv=llm_venv,
                               example_root=llama_example_root,
                               cmodel_dir=cmodel_dir,
                               model=model_name,
                               model_path=f'{llm_models_root()}/{model_path}',
                               data_type=dtype)

    engine_dir = os.path.join(engine_dir, f"{model_name}-engine")

    build_cmd = [
        "trtllm-build",
        f"--checkpoint_dir={ckpt_dir}",
        f"--output_dir={engine_dir}",
        f"--max_batch_size={8}",
        f"--max_input_len={924}",
        f"--max_seq_len={1024}",
        f"--max_beam_width={1}",
        f"--gemm_plugin={dtype}",
        f"--gpt_attention_plugin={dtype}",
    ]
    check_call(" ".join(build_cmd), shell=True, env=llm_venv._new_env)

    venv_check_call(llm_venv, [
        run_llm_path,
        "--model_dir",
        engine_dir,
    ])


@pytest.mark.parametrize("model_name,model_path", [
    ("llama", "llama-models-v2/llama-v2-7b-hf"),
])
def test_llmapi_load_engine_from_build_command_with_lora(
        llm_root, llm_venv, engine_dir, model_name, model_path):
    llama_example_root = os.path.join(llm_root, "examples", model_name)
    dtype = 'bfloat16'
    cmodel_dir = os.path.join(engine_dir, f"{model_name}-engine")

    ckpt_dir = convert_weights(llm_venv=llm_venv,
                               example_root=llama_example_root,
                               cmodel_dir=cmodel_dir,
                               model=model_name,
                               model_path=f'{llm_models_root()}/{model_path}',
                               data_type=dtype)

    engine_dir = os.path.join(engine_dir, f"{model_name}-engine")

    build_cmd = [
        "trtllm-build",
        f"--checkpoint_dir={ckpt_dir}",
        f"--output_dir={engine_dir}",
        f"--max_batch_size={1}",
        f"--max_input_len={924}",
        f"--max_seq_len={1024}",
        f"--max_beam_width={1}",
        f"--gemm_plugin={dtype}",
        f"--gpt_attention_plugin={dtype}",
        f"--lora_plugin={dtype}",
        f"--lora_target_modules=attn_q",
    ]
    check_call(" ".join(build_cmd), shell=True, env=llm_venv._new_env)

    venv_check_call(llm_venv, [
        run_llm_path,
        "--model_dir",
        engine_dir,
    ])


@pytest.mark.parametrize("model_name,model_path", [
    ("llama", "llama-models-v2/TinyLlama-1.1B-Chat-v1.0"),
])
def test_llmapi_build_command_parameters_align(llm_root, llm_venv, engine_dir,
                                               model_name, model_path):
    from tensorrt_llm._utils import release_gc
    from tensorrt_llm.llmapi import LLM
    from tensorrt_llm.llmapi.llm_utils import BuildConfig
    llama_example_root = os.path.join(llm_root, "examples", model_name)
    dtype = 'float16'
    cmodel_dir = os.path.join(engine_dir, f"{model_name}-engine")

    ckpt_dir = convert_weights(llm_venv=llm_venv,
                               example_root=llama_example_root,
                               cmodel_dir=cmodel_dir,
                               model=model_name,
                               model_path=f'{llm_models_root()}/{model_path}',
                               data_type=dtype)

    engine_dir = os.path.join(engine_dir, f"{model_name}-engine")

    build_cmd = [
        "trtllm-build",
        f"--checkpoint_dir={ckpt_dir}",
        f"--output_dir={engine_dir}",
        f"--max_batch_size={4}",
        f"--max_input_len={111}",
        f"--max_seq_len={312}",
        f"--max_beam_width={4}",
        f"--gemm_plugin={dtype}",
        f"--gpt_attention_plugin={dtype}",
    ]
    check_call(" ".join(build_cmd), shell=True, env=llm_venv._new_env)

    build_config = BuildConfig()
    # change some building parameters
    build_config.max_batch_size = 4
    build_config.max_beam_width = 4
    build_config.max_input_len = 111
    build_config.strongly_typed = True
    build_config.max_seq_len = 312
    build_config.plugin_config._gemm_plugin = dtype
    build_config.plugin_config._gpt_attention_plugin = dtype

    llm = LLM(model=f'{llm_models_root()}/{model_path}',
              build_config=build_config)
    tmpdir = tempfile.TemporaryDirectory()
    llm.save(tmpdir.name)
    build_cmd_cfg = None
    build_llmapi_cfg = None

    with open(os.path.join(engine_dir, "config.json"), "r") as f:
        engine_config = json.load(f)

        build_cmd_cfg = BuildConfig.from_dict(
            engine_config["build_config"]).to_dict()

    with open(os.path.join(tmpdir.name, "config.json"), "r") as f:
        llm_api_engine_cfg = json.load(f)

        build_llmapi_cfg = BuildConfig.from_dict(
            llm_api_engine_cfg["build_config"]).to_dict()

    assert build_cmd_cfg == build_llmapi_cfg
    del LLM
    release_gc()


def test_llmapi_load_ckpt_from_convert_command(llm_root, llm_venv, engine_dir):
    llama_example_root = os.path.join(llm_root, "examples", "llama")
    dtype = 'float16'
    cmodel_dir = os.path.join(engine_dir, "llama-7b-cmodel")

    ckpt_dir = convert_weights(
        llm_venv=llm_venv,
        example_root=llama_example_root,
        cmodel_dir=cmodel_dir,
        model='llama-7b',
        model_path=f'{llm_models_root()}/llama-models/llama-7b-hf',
        data_type=dtype)

    venv_check_call(llm_venv, [
        run_llm_path,
        "--model_dir",
        ckpt_dir,
    ])


def test_llmapi_exit(llm_venv):
    llm_exit_script = unittest_path() / "llmapi/run_llm_exit.py"
    llama_model_dir = Path(
        llm_models_root()) / "llama-models-v2/TinyLlama-1.1B-Chat-v1.0"

    run_command = [
        str(llm_exit_script), "--model_dir",
        str(llama_model_dir), "--tp_size", "1"
    ]
    venv_check_call(llm_venv, run_command)


@pytest.mark.skip_less_device(2)
def test_llmapi_exit_multi_gpu(llm_venv):
    llm_exit_script = unittest_path() / "llmapi/run_llm_exit.py"
    llama_model_dir = Path(
        llm_models_root()) / "llama-models-v2/TinyLlama-1.1B-Chat-v1.0"

    run_command = [
        str(llm_exit_script), "--model_dir",
        str(llama_model_dir), "--tp_size", "2"
    ]
    venv_check_call(llm_venv, run_command)


def test_llmapi_chat_example(llm_root, llm_venv):
    # Test for the examples/apps/chat.py
    example_root = Path(os.path.join(llm_root, "examples", "apps"))
    test_root = unittest_path() / "llmapi" / "apps"
    llm_venv.run_cmd([
        "-m", "pip", "install", "-r",
        os.path.join(example_root, "requirements.txt")
    ])

    llm_venv.run_cmd(["-m", "pytest", str(test_root / "_test_llm_chat.py")])


def test_llmapi_server_example(llm_root, llm_venv):
    # Test for the examples/apps/fastapi_server.py
    test_root = unittest_path() / "llmapi" / "apps"
    llm_venv.run_cmd(["-m", "pytest", str(test_root / "_test_llm_server.py")])


def test_openai_misc_example(llm_root, llm_venv):
    # Test for the examples/apps/openai_server.py
    test_root = unittest_path() / "llmapi" / "apps"
    llm_venv.run_cmd(["-m", "pytest", str(test_root / "_test_openai_misc.py")])


def test_openai_completions_example(llm_root, llm_venv):
    # Test for the examples/apps/openai_server.py
    test_root = unittest_path() / "llmapi" / "apps"
    llm_venv.run_cmd(
        ["-m", "pytest",
         str(test_root / "_test_openai_completions.py")])


def test_openai_chat_example(llm_root, llm_venv):
    # Test for the examples/apps/openai_server.py
    example_root = Path(os.path.join(llm_root, "examples", "apps"))
    test_root = unittest_path() / "llmapi" / "apps"
    llm_venv.run_cmd([
        "-m", "pip", "install", "-r",
        os.path.join(example_root, "requirements.txt")
    ])

    llm_venv.run_cmd(["-m", "pytest", str(test_root / "_test_openai_chat.py")])


@pytest.mark.skip_less_device(2)
@pytest.mark.skip_less_device_memory(40000)
def test_openai_multi_chat_example(llm_root, llm_venv):
    "Test for the examples/apps/openai_server.py"
    example_root = Path(os.path.join(llm_root, "examples", "apps"))
    test_root = unittest_path() / "llmapi" / "apps"
    llm_venv.run_cmd([
        "-m", "pip", "install", "-r",
        os.path.join(example_root, "requirements.txt")
    ])

    llm_venv.run_cmd(
        ["-m", "pytest",
         str(test_root / "_test_openai_multi_chat.py")])


@skip_nvlink_inactive
@pytest.mark.skip_less_device(4)
@pytest.mark.skip_less_device_memory(80000)
def test_openai_consistent_chat(llm_root, llm_venv):
    "Test for the examples/apps/openai_server.py"
    example_root = Path(os.path.join(llm_root, "examples", "apps"))
    test_root = unittest_path() / "llmapi" / "apps"
    llm_venv.run_cmd([
        "-m", "pip", "install", "-r",
        os.path.join(example_root, "requirements.txt")
    ])

    llm_venv.run_cmd(
        ["-m", "pytest",
         str(test_root / "_test_openai_consistent_chat.py")])


@skip_nvlink_inactive
@pytest.mark.skip_less_device(4)
@pytest.mark.skip_less_device_memory(80000)
def test_openai_multinodes_chat_tp16pp1(llm_root, llm_venv):
    "Test for the examples/apps/openai_server.py"
    example_root = Path(os.path.join(llm_root, "examples", "apps"))
    test_root = unittest_path() / "llmapi" / "apps"
    llm_venv.run_cmd([
        "-m", "pip", "install", "-r",
        os.path.join(example_root, "requirements.txt")
    ])

    llm_venv.run_cmd([
        "-m", "pytest", "-k", "tp16pp1",
        str(test_root / "_test_openai_multi_nodes.py")
    ])


@skip_nvlink_inactive
@pytest.mark.skip_less_device(4)
@pytest.mark.skip_less_device_memory(80000)
def test_openai_multinodes_chat_tp8pp2(llm_root, llm_venv):
    "Test for the examples/apps/openai_server.py"
    example_root = Path(os.path.join(llm_root, "examples", "apps"))
    test_root = unittest_path() / "llmapi" / "apps"
    llm_venv.run_cmd([
        "-m", "pip", "install", "-r",
        os.path.join(example_root, "requirements.txt")
    ])

    llm_venv.run_cmd([
        "-m", "pytest", "-k", "tp8pp2",
        str(test_root / "_test_openai_multi_nodes.py")
    ])


def test_build_time_benchmark_sanity(llm_root, llm_venv):
    temp = tempfile.TemporaryDirectory()
    llm_venv.run_cmd([
        str(Path(llm_root) / "tests/microbenchmarks/build_time_dashboard.py"),
        '-m',
        temp.name,
    ])


### LLMAPI examples
def _run_llmapi_example(llm_root, engine_dir, llm_venv, script_name: str,
                        *args):
    example_root = Path(llm_root) / "examples" / "llm-api"
    engine_dir = Path(engine_dir) / "llmapi"
    if not engine_dir.exists():
        engine_dir.mkdir(parents=True)
    examples_script = example_root / script_name

    run_command = [str(examples_script)] + list(args)

    # Create llm models softlink to avoid duplicated downloading for llm api example
    src_dst_dict = {
        f"{llm_models_root()}/llama-models-v2/TinyLlama-1.1B-Chat-v1.0":
        f"{llm_venv.get_working_directory()}/TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        f"{llm_models_root()}/vicuna-7b-v1.3":
        f"{llm_venv.get_working_directory()}/lmsys/vicuna-7b-v1.3",
        f"{llm_models_root()}/medusa-vicuna-7b-v1.3":
        f"{llm_venv.get_working_directory()}/FasterDecoding/medusa-vicuna-7b-v1.3",
        f"{llm_models_root()}/llama3.1-medusa-8b-hf_v0.1":
        f"{llm_venv.get_working_directory()}/nvidia/Llama-3.1-8B-Medusa-FP8",
    }

    for src, dst in src_dst_dict.items():
        if not os.path.islink(dst):
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            os.symlink(src, dst, target_is_directory=True)

    cnn_dailymail_src = f"{llm_models_root()}/datasets/cnn_dailymail"
    cnn_dailymail_dst = f"{llm_venv.get_working_directory()}/cnn_dailymail"
    if not os.path.islink(cnn_dailymail_dst):
        os.symlink(cnn_dailymail_src,
                   cnn_dailymail_dst,
                   target_is_directory=True)

    venv_check_call(llm_venv, run_command)


def test_llmapi_quickstart(llm_root, engine_dir, llm_venv):
    _run_llmapi_example(llm_root, engine_dir, llm_venv, "quickstart_example.py")


def test_llmapi_example_inference(llm_root, engine_dir, llm_venv):
    _run_llmapi_example(llm_root, engine_dir, llm_venv, "llm_inference.py")


def test_llmapi_example_customize(llm_root, engine_dir, llm_venv):
    _run_llmapi_example(llm_root, engine_dir, llm_venv,
                        "llm_inference_customize.py")


def test_llmapi_example_inference_async(llm_root, engine_dir, llm_venv):
    _run_llmapi_example(llm_root, engine_dir, llm_venv,
                        "llm_inference_async.py")


def test_llmapi_example_inference_async_streaming(llm_root, engine_dir,
                                                  llm_venv):
    _run_llmapi_example(llm_root, engine_dir, llm_venv,
                        "llm_inference_async_streaming.py")


def test_llmapi_example_quantization(llm_root, engine_dir, llm_venv):
    _run_llmapi_example(llm_root, engine_dir, llm_venv, "llm_quantization.py")


def test_llmapi_example_logits_processor(llm_root, engine_dir, llm_venv):
    _run_llmapi_example(llm_root, engine_dir, llm_venv,
                        "llm_logits_processor.py")


def test_llmapi_example_multilora(llm_root, engine_dir, llm_venv):
    _run_llmapi_example(llm_root, engine_dir, llm_venv, "llm_multilora.py")


def test_llmapi_example_guided_decoding(llm_root, engine_dir, llm_venv):
    _run_llmapi_example(llm_root, engine_dir, llm_venv,
                        "llm_guided_decoding.py")


def test_llmapi_example_lookahead_decoding(llm_root, engine_dir, llm_venv):
    _run_llmapi_example(llm_root, engine_dir, llm_venv,
                        "llm_lookahead_decoding.py")


def test_llmapi_example_medusa_decoding(llm_root, engine_dir, llm_venv):
    _run_llmapi_example(llm_root, engine_dir, llm_venv,
                        "llm_medusa_decoding.py")


def test_llmapi_example_medusa_decoding_use_modelopt(llm_root, engine_dir,
                                                     llm_venv):
    _run_llmapi_example(
        llm_root, engine_dir, llm_venv, "llm_medusa_decoding.py",
        "--use_modelopt_ckpt",
        f"--model_dir={llm_models_root()}/llama3.1-medusa-8b-hf_v0.1")


def test_llmapi_example_eagle_decoding(llm_root, engine_dir, llm_venv):
    _run_llmapi_example(llm_root, engine_dir, llm_venv, "llm_eagle_decoding.py")


@pytest.mark.skip_less_device(2)
def test_llmapi_example_distributed_tp2(llm_root, engine_dir, llm_venv):
    _run_llmapi_example(llm_root, engine_dir, llm_venv,
                        "llm_inference_distributed.py")


@pytest.mark.skip_less_device(2)
def test_llmapi_example_distributed_autopp_tp2(llm_root, engine_dir, llm_venv):
    _run_llmapi_example(llm_root, engine_dir, llm_venv, "llm_auto_parallel.py")


def test_llmapi_quickstart_atexit(llm_root, engine_dir, llm_venv):
    script_path = Path(
        llm_root
    ) / "tests/integration/defs/examples/run_llm_quickstart_atexit.py"
    llm_venv.run_cmd([str(script_path)])


def test_llmapi_quant_llama_70b(llm_root, engine_dir, llm_venv):
    # Test quantizing llama-70b model with only 2 H100 GPUs
    # The background: there is a bug preventing quantization of llama-70b model with <tp-size> GPUs
    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(',')
    if len(visible_devices) < 2:
        visible_devices = ['0', '1']
    visible_devices = visible_devices[:2]

    env = {
        'CUDA_VISIBLE_DEVICES': ','.join(visible_devices),
    }
    print(f'env: {env}')

    script_path = Path(
        llm_root
    ) / "tests/integration/defs/examples/run_llm_fp8_quant_llama_70b.py"
    llm_venv.run_cmd([str(script_path)], env=env)


# End of HLAPI examples


### Pivot-To-Python examples
def test_ptp_quickstart(llm_root, llm_venv):
    example_root = Path(os.path.join(llm_root, "examples", "pytorch"))

    src = f"{llm_models_root()}/llama-3.1-model/Llama-3.1-8B-Instruct"
    dst = f"{llm_venv.get_working_directory()}/meta-llama/Llama-3.1-8B-Instruct"
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    os.symlink(src, dst, target_is_directory=True)

    venv_check_call(llm_venv, [str(example_root / "quickstart.py")])


@pytest.mark.parametrize("model_name,model_path", [
    ("Llama3.1-8B-BF16", "llama-3.1-model/Meta-Llama-3.1-8B"),
    ("Nemotron4_4B-BF16", "nemotron/Minitron-4B-Base"),
    pytest.param('Llama3.1-8B-NVFP4',
                 'nvfp4-quantized/Meta-Llama-3.1-8B',
                 marks=skip_pre_blackwell),
    pytest.param('Llama3.1-8B-FP8',
                 'llama-3.1-model/Llama-3.1-8B-Instruct-FP8',
                 marks=skip_pre_hopper),
])
def test_ptp_quickstart_advanced(llm_root, llm_venv, model_name, model_path):
    print(f"Testing {model_name}.")
    example_root = Path(os.path.join(llm_root, "examples", "pytorch"))
    llm_venv.run_cmd([
        str(example_root / "quickstart_advanced.py"),
        "--enable_overlap_scheduler",
        "--enable_chunked_prefill",
        "--model_dir",
        f"{llm_models_root()}/{model_path}",
    ])


@pytest.mark.parametrize("model_name,model_path", [
    ("DeepSeek-V3-Lite-BF16", "DeepSeek-V3-Lite/bf16"),
])
def test_ptq_quickstart_advanced_mtp(llm_root, llm_venv, model_name,
                                     model_path):
    print(f"Testing {model_name}.")
    example_root = Path(os.path.join(llm_root, "examples", "pytorch"))
    llm_venv.run_cmd([
        str(example_root / "quickstart_advanced.py"),
        "--enable_overlap_scheduler",
        "--use_cuda_graph",
        "--spec_decode_nextn",
        "1",  # test 1 MTP module
        "--spec_decode_algo",
        "MTP",
        "--model_dir",
        f"{llm_models_root()}/{model_path}",
    ])


@pytest.mark.parametrize("model_name,model_path,eagle_model_path", [
    ("Llama-3.1-8b-Instruct", "llama-3.1-model/Llama-3.1-8B-Instruct",
     "EAGLE3-LLaMA3.1-Instruct-8B"),
])
def test_ptp_quickstart_advanced_eagle3(llm_root, llm_venv, model_name,
                                        model_path, eagle_model_path):
    print(f"Testing {model_name}.")
    example_root = Path(os.path.join(llm_root, "examples", "pytorch"))
    llm_venv.run_cmd([
        str(example_root / "quickstart_advanced.py"),
        "--spec_decode_nextn",
        "4",
        "--spec_decode_algo",
        "eagle3",
        "--model_dir",
        f"{llm_models_root()}/{model_path}",
        "--eagle_model_dir",
        f"{llm_models_root()}/{eagle_model_path}",
        "--kv_cache_enable_block_reuse",
    ])


@pytest.mark.skip_less_device_memory(110000)
@pytest.mark.skip_less_device(8)
@pytest.mark.parametrize("model_name,model_path", [
    pytest.param(
        'DeepSeek-R1', 'DeepSeek-R1/DeepSeek-R1', marks=skip_pre_hopper),
])
def test_ptp_quickstart_advanced_deepseek_r1_8gpus(llm_root, llm_venv,
                                                   model_name, model_path):
    print(f"Testing {model_name}.")
    example_root = Path(os.path.join(llm_root, "examples", "pytorch"))
    llm_venv.run_cmd([
        str(example_root / "quickstart_advanced.py"),
        "--enable_overlap_scheduler",
        "--model_dir",
        f"{llm_models_root()}/{model_path}",
        "--moe_tp_size=1",
        "--moe_ep_size=8",
        "--tp_size=8",
        "--use_cuda_graph",
        "--enable_attention_dp",
        "--kv_cache_fraction=0.95",
        "--max_batch_size=1",
        "--max_seq_len=3000",
        "--kv_cache_enable_block_reuse",
    ])


@pytest.mark.skip_less_device_memory(80000)
@pytest.mark.skip_less_device(8)
@pytest.mark.parametrize("model_name,model_path", [
    ("Llama3.1-70B-BF16", "llama-3.1-model/Meta-Llama-3.1-70B"),
    ("Mixtral-8x7B-BF16", "Mixtral-8x7B-v0.1"),
    pytest.param('Llama3.1-70B-FP8',
                 'llama-3.1-model/Llama-3.1-70B-Instruct-FP8',
                 marks=skip_pre_hopper),
    pytest.param('Llama3.1-405B-FP8',
                 'llama-3.1-model/Llama-3.1-405B-Instruct-FP8',
                 marks=skip_pre_hopper),
    pytest.param('Mixtral-8x7B-NVFP4',
                 'nvfp4-quantized/Mixtral-8x7B-Instruct-v0.1',
                 marks=skip_pre_blackwell),
])
def test_ptp_quickstart_advanced_8gpus(llm_root, llm_venv, model_name,
                                       model_path):
    print(f"Testing {model_name}.")
    example_root = Path(os.path.join(llm_root, "examples", "pytorch"))
    llm_venv.run_cmd([
        str(example_root / "quickstart_advanced.py"),
        "--enable_overlap_scheduler",
        "--enable_chunked_prefill",
        "--model_dir",
        f"{llm_models_root()}/{model_path}",
        "--tp_size=8",
    ])


@skip_pre_blackwell
def test_ptp_quickstart_advanced_mixed_precision(llm_root, llm_venv):
    example_root = Path(os.path.join(llm_root, "examples", "pytorch"))
    model_path = "Llama-3_1-8B-Instruct_fp8_nvfp4_hf"
    llm_venv.run_cmd([
        str(example_root / "quickstart_advanced.py"),
        "--model_dir",
        f"{llm_models_root()}/{model_path}",
    ])


@pytest.mark.parametrize("modality", ["image", "video"])
@pytest.mark.parametrize("model_name,model_path", [
    ("NVILA-8B-FP16", "vila/NVILA-8B"),
    ("llava-v1.6-mistral-7b", "llava-v1.6-mistral-7b-hf"),
    ("qwen2-vl-7b-instruct", "Qwen2-VL-7B-Instruct"),
])
def test_ptp_quickstart_multimodal(llm_root, llm_venv, model_name, model_path,
                                   modality):
    llm_venv.run_cmd(
        ['-m', 'pip', 'install', 'flash-attn==2.7.3', '--no-build-isolation'])

    example_root = Path(os.path.join(llm_root, "examples", "pytorch"))
    test_data_root = Path(
        os.path.join(llm_models_root(), "multimodals", "test_data"))
    print(f"Accuracy test {model_name} {modality} mode with example inputs.")
    accuracy_inputs = {
        "image": {
            "prompt": [
                "Describe the natural environment in the image.",
                "Describe the object and the weather condition in the image.",
                "Describe the traffic condition on the road in the image.",
            ],
            "media": [
                str(test_data_root / "seashore.png"),
                str(test_data_root / "inpaint.png"),
                str(test_data_root / "61.jpg"),
            ],
        },
        "video": {
            "prompt": [
                "Tell me what you see in the video briefly.",
                "Describe the scene in the video briefly.",
            ],
            "media": [
                str(test_data_root / "OAI-sora-tokyo-walk.mp4"),
                str(test_data_root / "world.mp4"),
            ],
        },
    }
    expected_answers = {
        "NVILA-8B-FP16": {
            "image": [
                [
                    "The image features a stormy ocean with large waves crashing, a gray sky with white clouds, and a dark gray horizon.",
                    "The image features a stormy ocean with large waves crashing, a dark gray sky with white clouds, and a grayish-blue water surface."
                ],
                "The object is a large rock formation, and the weather condition is sunny with a blue sky and white clouds.",
                [
                    "The road is busy with multiple cars, including a blue car, a silver SUV, and a black car, all driving in the same direction.",
                    "The road is busy with multiple cars, including a blue car, a white car, a black car, and a silver car, all driving in the same direction.",
                    "The road is busy with multiple cars, including a blue car, a white car, a black car, and a green double-decker bus."
                ],
            ],
            "video": [
                [
                    "The video depicts a woman walking down a city street at night. She is wearing a black leather jacket, a red dress, and black boots. The woman is carrying a black purse and has sunglasses on. The street is wet, and there are many people walking around. The woman is looking at the camera.",
                    "The video depicts a woman walking down a city street at night. She is wearing a black leather jacket, a red dress, and black boots. The woman is carrying a black purse and is wearing sunglasses. The street is wet, and there are many people walking around. The woman is walking towards the camera, and the"
                ],
                [
                    "The video depicts a stunning view of Earth from space, showcasing the planet's curvature and the vastness of space. The Earth is illuminated by the sun, with the left side appearing darker and the right side brighter. The image captures the beauty of our home planet, highlighting its unique features and the contrast between day and night",
                    "The video depicts a stunning view of Earth from space, showcasing the planet's vibrant blue oceans and the intricate patterns of city lights illuminating the continents. The image captures the curvature of the Earth, with the dark side of the planet visible, and the bright side displaying the illuminated city lights. The contrast between the illuminated and"
                ],
            ],
        },
        "llava-v1.6-mistral-7b": {
            "image": [
                [
                    "The image depicts a dramatic ocean scene under a cloudy sky. The ocean is characterized by large, powerful waves that are breaking and crashing onto the shore. The waves are white and frothy, indicating that they are in the process of breaking. The water appears to be a deep blue-green color, suggesting",
                    "The image depicts a dramatic natural environment. The sky is overcast with dark, heavy clouds, suggesting a stormy or gloomy weather condition. The ocean is in motion, with large waves that are breaking and crashing onto the shore. The water appears choppy and turbulent, with white foam and spray visible",
                ],
                [
                    "The image shows a scenic landscape with a prominent rock formation, which appears to be a large, flat-topped mountain or butte. The rock formation is rugged and has a smooth, flat top, suggesting it could be a natural landmark or a geological feature. The sky is clear with a few",
                    "The image shows a majestic mountain with a flat top, which is characteristic of buttes. The mountain is prominently featured in the background, with a clear blue sky above it and a few scattered clouds. The weather appears to be and clear, with no visible signs of rain or storms.",
                ],
                "The image shows a multi-lane highway with several vehicles in motion. There are cars and a bus visible, and the traffic appears to be moderate, with no significant congestion. The road is divided by a central divider, and there are green trees lining the sides of the highway, indicating a suburban",
            ],
        },
        "qwen2-vl-7b-instruct": {
            "image": [
                [
                    "The image depicts a vast ocean with waves crashing against the shore. The sky is filled with dark clouds, creating a dramatic and moody atmosphere. The waves are powerful and turbulent, suggesting a stormy weather condition. The overall scene conveys a sense of raw natural beauty and the raw power of the ocean.",
                    "The image depicts a vast ocean with waves crashing against the shore. The sky is filled with dark clouds, creating a dramatic and moody atmosphere. The waves are powerful and turbulent, with white foam at their crests, indicating strong winds and rough sea conditions. The overall scene conveys a sense of raw natural power and"
                ],
                [
                    "The image depicts a scenic mountainous landscape. The central object is a large, prominent rock formation known as Half Dome, which is a well-known landmark in Yosemite National Park, California. The weather appears to be clear and sunny, with a bright blue sky and some scattered clouds. The visibility is excellent, allowing for a",
                    "The image depicts a scenic mountainous landscape with a prominent rock formation in the background. The rock formation is a large, steep, and pointed peak, which appears to be a well-known natural landmark. The sky is clear with a few scattered clouds, indicating fair weather conditions. The lighting suggests it is a sunny day,",
                    "The image depicts a scenic mountainous landscape with a prominent, steep, and rocky peak in the background. The peak is characterized by its sharp, jagged edges and a smooth, polished surface, suggesting it might be a well-known natural landmark. The sky is clear with a few scattered clouds, indicating fair weather conditions."
                ],
                [
                    "The traffic condition on the road in the image appears to be moderate. There are several vehicles traveling in both directions, including cars, a bus, and a police car. The road is divided into multiple lanes, and the vehicles are maintaining a safe distance from each other. The overall scene suggests a typical day with moderate traffic",
                    "The traffic condition on the road in the image appears to be moderate. There are several vehicles traveling in both directions, including cars, a bus, and a truck. The road is divided into multiple lanes, and the vehicles are maintaining a safe distance from each other. The overall flow of traffic seems to be smooth, with",
                    "The traffic condition on the road in the image appears to be moderate. There are several vehicles traveling in both directions, including cars, a bus, and a police car. The road is divided into multiple lanes, and the vehicles are maintaining a safe distance from each other. The overall flow of traffic seems to be smooth,"
                ],
            ],
            "video": [
                [
                    "The video shows a person walking down a busy city street at night. The street is illuminated by numerous bright lights and signs, creating a vibrant and lively atmosphere. The person is wearing a black leather jacket, a red dress, and large sunglasses, and is carrying a black handbag. The street appears to be wet,",
                    "The video shows a person walking down a busy city street at night. The street is illuminated by numerous bright lights and signs, creating a vibrant and lively atmosphere. The person is wearing a black leather jacket, a red dress, and large sunglasses, and is carrying a black bag. The street appears to be wet, reflecting"
                ],
                [
                    "The video shows a spinning Earth with a black background. The Earth is mostly dark, with some parts illuminated by lights."
                ],
            ],
        },
    }

    cmd = [
        str(example_root / "quickstart_multimodal.py"),
        "--model_dir",
        f"{llm_models_root()}/{model_path}",
        "--modality",
        modality,
        "--prompt",
        *accuracy_inputs[modality]["prompt"],
        "--media",
        *accuracy_inputs[modality]["media"],
    ]
    # NOTE
    # Qwen2-VL model need larger max_num_tokens.
    if model_name == "qwen2-vl-7b-instruct" and modality == "video":
        cmd.append("--max_num_tokens=16384")
    output = llm_venv.run_cmd(cmd, caller=check_output)

    def parse_output(text):
        results = []
        text_lists = re.split(r"\[\d+\] Prompt:", text)
        for item in text_lists:
            item = item.replace(os.linesep, "")
            while True:
                match = re.search(r"(Generated text: \'(.*?)\')", item,
                                  re.MULTILINE)
                if match is None:
                    break
                _, end = match.span(1)
                results.append(match.group(2))
                item = item[end:]
        return results

    match_ratio = 0.9
    for output, expected_answer in zip(parse_output(output),
                                       expected_answers[model_name][modality]):
        if not isinstance(expected_answer, list):
            expected_answer = [expected_answer]
        assert any(
            SequenceMatcher(a=output, b=answer).ratio() > match_ratio
            for answer in expected_answer
        ), f"Wrong answer!\nGenerated \"{output}\"\nExpected \"{expected_answer}\"\nMatch ratio: {[SequenceMatcher(a=output, b=answer).ratio() for answer in expected_answer]} all below threshold {match_ratio}"

    print("All answers are correct!")

    if not any(name in model_name for name in ["NVILA"]):
        print(f"Skipping functionality test for {model_name}.")
        return

    print(f"Functionality test {model_name} {modality} mode.")
    functionality_inputs = {
        "image": {
            "prompt":
            "Describe the two images in detail.",
            "media": [
                "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/inpaint.png",
                "https://huggingface.co/datasets/Sayali9141/traffic_signal_images/resolve/main/61.jpg",
            ],
        },
        "video": {
            "prompt":
            "Tell me what you see in the video briefly.",
            "media": [
                "https://huggingface.co/datasets/Efficient-Large-Model/VILA-inference-demos/resolve/main/OAI-sora-tokyo-walk.mp4",
                "https://huggingface.co/datasets/Efficient-Large-Model/VILA-inference-demos/resolve/main/world.mp4",
            ],
        },
    }
    llm_venv.run_cmd([
        str(example_root / "quickstart_multimodal.py"),
        "--model_dir",
        f"{llm_models_root()}/{model_path}",
        "--modality",
        modality,
        "--prompt",
        functionality_inputs[modality]["prompt"],
        "--media",
        *functionality_inputs[modality]["media"],
    ])


@pytest.mark.parametrize("model_name,model_path", [
    ("BertForSequenceClassification", "bert/bert-base-uncased-yelp-polarity"),
])
def test_ptp_quickstart_bert(llm_root,
                             llm_venv,
                             model_name,
                             model_path,
                             backend='VANILLA'):
    print(f"Testing {model_name}.")
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    from tensorrt_llm import SamplingParams
    from tensorrt_llm._torch import LLM
    from tensorrt_llm._torch.pyexecutor.config import PyTorchConfig
    from tensorrt_llm.sampling_params import SamplingParams
    prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
    ]
    model_dir = f"{llm_models_root()}/{model_path}"
    # NOTE: Bert model return logits for now
    sampling_param = SamplingParams(max_tokens=32, return_context_logits=True)
    with LLM(
            model=model_dir,
            pytorch_backend_config=PyTorchConfig(attn_backend=backend),
    ) as llm:

        outputs = llm.generate(prompts, sampling_params=sampling_param)
    # Print the outputs.
    tllm_logits = []
    for output in outputs:
        prompt = output.prompt
        tllm_logit = output.context_logits.cpu()
        print(f"Prompt: {prompt!r}, Context logits: {tllm_logit}")
        tllm_logits += [tllm_logit]
    # Stack the output
    tllm_logits = torch.stack(tllm_logits)

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    # NOTE: assume the model is BertForSequenceClassification for now
    # load BertForSequenceClassification model
    hf_model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    hf_model = hf_model.half().to(tllm_logits.device)

    with torch.inference_mode():
        inputs = tokenizer(prompts, return_tensors="pt",
                           padding='longest').to(hf_model.device)
        hf_outputs = hf_model(**inputs)
        hf_logit = hf_outputs.logits.float()

    torch.testing.assert_close(tllm_logits, hf_logit, rtol=1.5e-2, atol=1.5e-2)
    # If assert passes, print success message.
    print("Success: HF model logits match TRTLLM logits!")


@pytest.mark.parametrize("model_name,model_path", [
    ("Llama3.1-8B-BF16", "llama-3.1-model/Meta-Llama-3.1-8B"),
])
def test_ptp_star_attention_example(llm_root, llm_venv, model_name, model_path,
                                    star_attention_input_root):
    print(f"Testing {model_name}.")
    workspace = llm_venv.get_working_directory()
    example_root = Path(os.path.join(llm_root, "examples", "pytorch"))
    input_file = Path(
        os.path.join(star_attention_input_root,
                     "test_star_attention_input.jsonl"))
    output_file = Path(os.path.join(workspace, "star_attention_output.jsonl"))
    llm_venv.run_cmd([
        str(example_root / "star_attention.py"),
        "--model_path",
        f"{llm_models_root()}/{model_path}",
        "--sa_block_size=200",
        "--sa_anchor_size=200",
        f"--input_file={input_file}",
        f"--output_file={output_file}",
    ])


@pytest.mark.skip_less_device_memory(80000)
@pytest.mark.parametrize("model_name,model_path", [
    ("DeepSeek-R1-Distill-Qwen-7B", "DeepSeek-R1/DeepSeek-R1-Distill-Qwen-7B"),
])
def test_ptp_scaffolding(llm_root, llm_venv, model_name, model_path):
    print(f"Testing scaffolding {model_name}.")
    example_root = Path(os.path.join(llm_root, "examples", "scaffolding"))
    input_file = Path(os.path.join(example_root, "test.jsonl"))
    llm_venv.run_cmd([
        str(example_root / "aime24_test.py"),
        "--generation_dir",
        f"{llm_models_root()}/{model_path}",
        f"--jsonl_file={input_file}",
        "--threshold=0.5",
    ])


# End of Pivot-To-Python examples
