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
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Optional, Tuple, Union

import pytest
import yaml
from defs.common import convert_weights
from defs.trt_test_alternative import (check_call, check_call_negative_test,
                                       check_output)

from .common import (PluginOptions, convert_weights, get_mmlu_accuracy,
                     prune_checkpoint, quantize_data, refit_model,
                     venv_check_call)
from .conftest import (get_device_count, llm_models_root, skip_no_sm120,
                       skip_nvlink_inactive, skip_post_blackwell, skip_pre_ada,
                       skip_pre_blackwell, skip_pre_hopper, tests_path,
                       unittest_path)

sys.path.append(os.path.join(str(tests_path()), '/../examples/apps'))

TEST_MEM_USAGE = os.environ.get('TEST_MEM_USAGE', True)

if TEST_MEM_USAGE:
    os.environ['TLLM_LOG_LEVEL'] = 'INFO'

_MEM_FRACTION_50 = 0.5
_MEM_FRACTION_80 = 0.8
_MEM_FRACTION_95 = 0.95


def _get_mem_info_from_log(file, ranks_num):
    import re

    # Peak memory size, model memory size and extra memory size are printed
    # only when TLLM_LOG_LEVEL=INFO
    pattern = re.compile(r"\[MemUsageChange] Allocated ([\d]+\.[\d]+) GiB ")
    fraction_pattern = re.compile(r"fraction is set ([\d]+\.[\d]+), ")
    total_mem_pattern = re.compile(r"device total memory ([\d]+\.[\d]+) GiB")
    peak_mem_pattern = re.compile(
        r"Peak memory during memory usage profiling \(torch \+ non-torch\): ([\d]+\.[\d]+) GiB"
    )
    extra_mem_pattern = re.compile(
        r"Memory used outside torch \(e\.g\., NCCL and CUDA graphs\) in memory usage profiling: ([\d]+\.[\d]+) GiB"
    )
    activation_pattern = re.compile(
        r"Memory dynamically allocated during inference \(inside torch\) in memory usage profiling: ([\d]+\.[\d]+) GiB"
    )
    model_pattern = re.compile(
        r"Memory used after loading model weights \(inside torch\) in memory usage profiling: ([\d]+\.[\d]+) GiB"
    )
    tmp_kv_patterm = re.compile(r"tmp kv_mem ([\d]+\.[\d]+) GiB")
    start_time_mem_pattern = re.compile(
        r"Memory used after loading model weights \(outside torch\) in memory usage profiling: ([\d]+\.[\d]+) GiB"
    )

    fraction = 0.90
    kv_mem_size = []
    total_memory = []
    peak_memory = []
    extra_memory = []
    activation_memory = []
    model_memory = []
    tmp_kv = []
    start_time_mem = []
    file.seek(0)
    lines = file.readlines()
    for line in lines:
        match = pattern.findall(line)
        if len(match) > 0:
            kv_mem_size.append(float(match[0]))
        match = fraction_pattern.findall(line)
        if len(match) > 0:
            fraction = float(match[0])
        match = total_mem_pattern.findall(line)
        if len(match) > 0:
            total_memory.append(float(match[0]))
        match = peak_mem_pattern.findall(line)
        if len(match) > 0:
            peak_memory.append(float(match[0]))
        match = extra_mem_pattern.findall(line)
        if len(match) > 0:
            extra_memory.append(float(match[0]))
        match = activation_pattern.findall(line)
        if len(match) > 0:
            activation_memory.append(float(match[0]))
        match = model_pattern.findall(line)
        if len(match) > 0:
            model_memory.append(float(match[0]))
        match = tmp_kv_patterm.findall(line)
        if len(match) > 0:
            tmp_kv.append(float(match[0]))
        match = start_time_mem_pattern.findall(line)
        if len(match) > 0:
            start_time_mem.append(float(match[0]))

    assert len(
        kv_mem_size) % 2 == 0, "no enough memory usage information in log"
    kv_mem_size = kv_mem_size[len(kv_mem_size) // 2:]
    return peak_memory, model_memory, sum(
        kv_mem_size
    ) / ranks_num, extra_memory, fraction, total_memory, activation_memory, sum(
        tmp_kv) / ranks_num, sum(start_time_mem) - ranks_num


def _get_kv_mem_size_candidate(total_Gib, used_Gib, fraction):
    return (total_Gib - used_Gib) * fraction


def _check_mem_usage(file, mem_info, ranks_num=1):
    if file is None or not TEST_MEM_USAGE:
        return
    delta = 0.3  # 0.3 GB as buffer
    peak, model_size, kv_mem_size, extra, fraction, total_memory, activation_memory, tmp_kv, start_time_mem = _get_mem_info_from_log(
        file, ranks_num)

    peak = max(peak)
    min_total = min(total_memory)
    e_peak, e_model_size, e_kv_mem_size, e_extra = mem_info
    import torch
    _, total = torch.cuda.mem_get_info()
    e_kv_mem_size = _get_kv_mem_size_candidate(min_total,
                                               (e_peak + start_time_mem),
                                               fraction)
    print(
        f"Expected memory usage: peak mem {e_peak + start_time_mem}, model mem {e_model_size}, kv mem {e_kv_mem_size:.2f}, extra {e_extra}, total {total / (1 << 30):.2f}"
    )
    print(
        f"Running memory information: peak mem {peak}, model mem {model_size}, kv mem {kv_mem_size}, extra {extra}, total {min_total}, activation {activation_memory}, tmp_kv {tmp_kv}, fraction  {fraction}, none-torch memory at starttime {start_time_mem}"
    )

    assert peak - tmp_kv <= e_peak + start_time_mem + delta, f"peak memory {peak} is larger than expected {e_peak}"
    assert kv_mem_size >= e_kv_mem_size - delta, f"kv memory size {kv_mem_size} is smaller than expected {e_kv_mem_size}"
    # assert model_size <= e_model_size + delta, f"model memory {model_size} is larger than expected {e_model_size}"
    # assert max(extra) <= e_extra + delta, f"extra memory size {extra} is larger than expected {e_extra}"


def test_gpt3_175b_1layers_build_only(llm_root, llm_venv, engine_dir):
    "Build GPT-3 175B: 96 layer w/ plugins"
    example_root = os.path.join(llm_root, "examples", "models", "core", "gpt")
    engine_dir = os.path.join(engine_dir, "gpt-175-96layers-build-only")

    dtype = 'float16'
    convert_cmd = [
        f"{example_root}/../../../generate_checkpoint_config.py",
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
    example_root = os.path.join(llm_root, "examples", "models", "core", "gpt")
    engine_dir = os.path.join(engine_dir, "gpt2")

    dtype = 'float32'
    convert_cmd = [
        f"{example_root}/../../../generate_checkpoint_config.py",
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
        f"{example_root}/../../../run.py", "--max_output_len=1",
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
        f"{llama_example_root}/../../../run.py",
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
        f"{llama_example_root}/../../../run.py",
        "--max_output_len=1",
        f"--tokenizer_dir={llama_tokenizer_model_root}",
        "--log_level=verbose",
        "--max_attention_window_size=5",
        f"--engine_dir={engine_dir}",
    ]
    if use_py_session:
        run_cmd.extend(["--use_py_session"])
    venv_check_call(llm_venv, run_cmd)


@pytest.mark.parametrize("model_name,model_path", [
    ("DeepSeek-R1-Distill-Qwen-1.5B", "DeepSeek-R1-Distill-Qwen-1.5B"),
])
def test_qwen_e2e_cpprunner_large_new_tokens(model_name, model_path, llm_venv,
                                             qwen_example_root, cmodel_dir,
                                             engine_dir):
    "RCCA: https://nvbugs/5238105"
    model_dir = convert_weights(
        llm_venv=llm_venv,
        example_root=qwen_example_root,
        cmodel_dir=cmodel_dir,
        model=model_name,
        model_path=f"{llm_models_root()}/{model_path}",
    )

    build_cmd = [
        "trtllm-build", f"--checkpoint_dir={model_dir}",
        f"--output_dir={engine_dir}", f"--gemm_plugin=float16",
        "--max_num_tokens=32768"
    ]

    check_call(" ".join(build_cmd), shell=True, env=llm_venv._new_env)

    from transformers import AutoTokenizer

    from tensorrt_llm.runtime import PYTHON_BINDINGS

    if PYTHON_BINDINGS:
        from tensorrt_llm.runtime import ModelRunnerCpp
    tokenizer = AutoTokenizer.from_pretrained(
        f"{llm_models_root()}/{model_path}",
        trust_remote_code=True,
        use_fast=False)

    message = r"<｜begin▁of▁sentence｜><｜User｜>The operation $\otimes$ is defined for all nonzero numbers by $a \otimes b = \frac{a^{2}}{b}$. Determine $[(1 \otimes 2) \otimes 3] - [1 \otimes (2 \otimes 3)]$. Let's think step by step and output the final answer within \boxed{}.<｜Assistant｜>"

    inputs = tokenizer(message, return_tensors='pt',
                       add_special_tokens=False)['input_ids']

    runner = ModelRunnerCpp.from_dir(engine_dir=f"{engine_dir}",
                                     max_input_len=128,
                                     max_output_len=4096,
                                     max_batch_size=8)

    outputs = runner.generate(inputs,
                              end_id=tokenizer.eos_token_id,
                              pad_id=tokenizer.pad_token_id,
                              temperature=0.6,
                              top_p=1.0,
                              top_k=1024,
                              max_new_tokens=1024,
                              return_dict=True,
                              min_length=1,
                              num_return_sequences=4,
                              output_sequence_lengths=True)

    seq_lengths = outputs['sequence_lengths']
    assert not (seq_lengths == 0).any(
    ), f"Found zero length in sequence_lengths tensor: {seq_lengths}"


# TODO replace the trtllm_bench_prolog
class BenchRunner:

    def __init__(self,
                 llm_root: str,
                 llm_venv: Any,
                 model_subdir: str,
                 model_name: str,
                 streaming: bool,
                 tp_size: int,
                 use_pytorch_backend: bool = False,
                 skip_engine_build: bool = False,
                 quant: Optional[str] = None,
                 extra_llm_api_options: Optional[str] = None,
                 use_mpirun: bool = False,
                 concurrency: Optional[int] = None,
                 num_requests: int = 10):

        llm_models = llm_models_root()
        assert llm_models is not None
        self.llm_root = llm_root
        self.llm_venv = llm_venv
        self.model_path = Path(llm_models, model_subdir).absolute()
        self.model_name = model_name
        self.quant = quant
        self.streaming = streaming
        self.skip_engine_build = skip_engine_build
        self.use_pytorch_backend = use_pytorch_backend
        self.use_mpirun = use_mpirun
        self.tp_size = tp_size
        self.quant_name = self.quant if self.quant is not None else "FP16"
        self.extra_llm_api_options = extra_llm_api_options

        self.work_dir = Path(tempfile.TemporaryDirectory().name)

        self.dataset_path = os.path.join(self.work_dir, f"data.txt")
        if self.use_mpirun:
            self.mpirun_cmd = f"mpirun --allow-run-as-root -n {self.tp_size} trtllm-llmapi-launch"
        else:
            self.mpirun_cmd = ""
        self.engine_path = None
        self.concurrency = concurrency
        self.num_requests = num_requests

    def __call__(self):
        self.prepare_dataset()
        if not (self.skip_engine_build or self.use_pytorch_backend):
            self.build_engine()
        return self.run_bench()

    def prepare_dataset(self):
        dataset_tool = Path(self.llm_root, "benchmarks", "cpp",
                            "prepare_dataset.py")

        # Generate a small dataset to run a test.
        self.work_dir.mkdir(parents=True)
        command = [
            f"{dataset_tool.resolve()}",
            "--stdout",
            "--tokenizer",
            f"{self.model_path}",
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
            str(self.num_requests),
        ]
        print(f"Running command: {' '.join(command)}")
        dataset_output = self.llm_venv.run_cmd(
            command,
            caller=check_output,
        )
        # Grab the stdout and write it to a dataset file for passing to suite.
        with open(self.dataset_path, "w") as dataset:
            dataset.write(dataset_output)

    def build_engine(self):
        if self.skip_engine_build:
            return

        build_cmd = \
            f"{self.mpirun_cmd} " \
            f"trtllm-bench " \
            f"--model {self.model_name} " \
            f"--model_path {self.model_path} " \
            f"--workspace {self.work_dir} " \
            f"build --tp_size {self.tp_size}"

        if self.quant is not None:
            build_cmd = f"{build_cmd} --quantization {self.quant}"

        build_cmd = f"{build_cmd} --dataset {self.dataset_path}"
        build_output = check_output(build_cmd,
                                    shell=True,
                                    env=self.llm_venv._new_env)

        for line in build_output.split("\n")[::-1]:
            if line.startswith("ENGINE SAVED:"):
                self.engine_path = Path(line.split(":")[1])
                break

    def run_bench(self):
        streaming = "--streaming" if self.streaming else ""
        benchmark_cmd = \
            f"{self.mpirun_cmd} " \
            f"trtllm-bench --model {self.model_name} --model_path {self.model_path} " \
            f"throughput " \
            f"--tp {self.tp_size} "
        if self.engine_path:
            benchmark_cmd += f"--engine_dir {self.engine_path} "
        benchmark_cmd += f" --dataset {self.dataset_path} {streaming}"

        if self.use_pytorch_backend:
            benchmark_cmd += " --backend pytorch"
        else:
            benchmark_cmd += " --backend tensorrt"

        if self.extra_llm_api_options:
            benchmark_cmd += f" --extra_llm_api_options {self.extra_llm_api_options}"
        if self.concurrency:
            benchmark_cmd += f" --concurrency {self.concurrency}"
        if self.num_requests:
            benchmark_cmd += f" --num_requests {self.num_requests}"

        benchmark_output = check_output(benchmark_cmd,
                                        shell=True,
                                        env=self.llm_venv._new_env)
        return self.parse_benchmark_output(benchmark_output)

    def parse_benchmark_output(self, output):
        """Parse the benchmark output to extract key metrics."""
        result = {
            'concurrency': self.concurrency,
            'num_requests': self.num_requests,
            'throughput': 0,
            'latency': 0
        }

        lines = output.split('\n')
        for line in lines:
            line = line.strip()
            if 'total token throughput' in line.lower(
            ) and 'tokens/sec' in line.lower():
                try:
                    throughput = line.split(":")[1].strip()
                    result['throughput'] = throughput
                except (IndexError, ValueError) as e:
                    print(
                        f"Failed to parse throughput from line: {line}. Error: {e}"
                    )
            elif 'total latency' in line.lower() and 'ms' in line.lower():
                try:
                    latency = line.split(":")[1].strip()
                    result['latency'] = latency
                except (IndexError, ValueError) as e:
                    print(
                        f"Failed to parse latency from line: {line}. Error: {e}"
                    )

        return result


@pytest.mark.parametrize("model_name", ["meta-llama/Meta-Llama-3-8B-Instruct"],
                         ids=["llama3-8b"])
@pytest.mark.parametrize("model_subdir",
                         ["llama-models-v3/llama-v3-8b-instruct-hf"],
                         ids=["llama-v3"])
@pytest.mark.parametrize("use_pytorch_backend", [True, False],
                         ids=["pytorch_backend", "trt_backend"])
def test_trtllm_bench_llmapi_launch(llm_root, llm_venv, model_name,
                                    model_subdir, use_pytorch_backend):
    runner = BenchRunner(llm_root=llm_root,
                         llm_venv=llm_venv,
                         model_name=model_name,
                         model_subdir=model_subdir,
                         streaming=False,
                         use_pytorch_backend=use_pytorch_backend,
                         use_mpirun=True,
                         tp_size=2)
    runner()


@skip_pre_hopper
@pytest.mark.skip_less_device_memory(80000)
@pytest.mark.parametrize("model_name", ["meta/Meta-Llama-3.1-8B"],
                         ids=["llama3_1-8b"])
@pytest.mark.parametrize("model_subdir", ["llama-3.1-model/Meta-Llama-3.1-8B"],
                         ids=["llama_v3_1"])
@pytest.mark.parametrize("use_pytorch_backend", [False], ids=["trt_backend"])
def test_trtllm_bench_mig_launch(llm_root, llm_venv, model_name, model_subdir,
                                 use_pytorch_backend):
    "run bench mark in MIG mode, check if the throughput is increasing by concurrency"
    skip_engine_build = False
    results = {}
    concurrency_list = [1, 32, 64, 128]

    for concurrency in concurrency_list:
        num_requests = concurrency * 10
        runner = BenchRunner(llm_root=llm_root,
                             llm_venv=llm_venv,
                             model_name=model_name,
                             model_subdir=model_subdir,
                             streaming=False,
                             use_pytorch_backend=use_pytorch_backend,
                             use_mpirun=False,
                             tp_size=1,
                             concurrency=concurrency,
                             num_requests=num_requests,
                             skip_engine_build=skip_engine_build)

        output = runner()
        results[concurrency] = output

    print(f"\n=== Benchmark Results Comparison ===")
    print(f"Model: {model_name}")
    print(f"Backend: {'PyTorch' if use_pytorch_backend else 'TensorRT'}")
    print(
        f"{'Concurrency':<15} {'Throughput':<15} {'Latency':<15} {'Num Requests':<15}"
    )
    print("-" * 60)

    for idx, val in enumerate(concurrency_list):
        metrics = results.get(val)
        if not isinstance(metrics, dict):
            pytest.fail(
                f"Unexpected benchmark result type for concurrency {val}: {type(metrics)}"
            )
        try:
            throughput = float(metrics.get('throughput', 0))
            latency = float(metrics.get('latency', 0))
            num_requests = int(metrics.get('num_requests', 0))
        except (ValueError, TypeError) as e:
            pytest.fail(
                f"Failed to parse benchmark results for concurrency {val}: {e}")
        assert throughput > 0, f"Throughput is 0 for concurrency {val}"
        assert latency > 0, f"Latency is 0 for concurrency {val}"
        print(f"{val:<15} {throughput:<15} {latency:<15} {num_requests:<15}")
        if idx > 0:
            prev_throughput = float(results[concurrency_list[idx - 1]].get(
                'throughput', 0))
            assert throughput > prev_throughput * 1.3, f"Throughput is not increasing for concurrency {concurrency_list[idx]}"


@pytest.mark.parametrize(
    "model_name, llama_model_root",
    [pytest.param("TinyLlama-1.1B-Chat-v1.0", "TinyLlama-1.1B-Chat-v1.0")],
    indirect=["llama_model_root"])
def test_trtllm_bench_invalid_token_pytorch(llm_root, llm_venv, model_name,
                                            llama_model_root):
    # Prepare dataset with invalid tokens
    _, _, dataset_path = trtllm_bench_prolog(llm_root,
                                             llm_venv,
                                             engine_dir=None,
                                             model_subdir=llama_model_root,
                                             model_name=model_name,
                                             quant=None,
                                             streaming=False,
                                             skip_engine_build=True)
    with open(dataset_path) as f:
        dataset = [json.loads(line) for line in f.readlines()]
    dataset[0]["input_ids"][-1] = -1
    with open(dataset_path, "w") as f:
        f.writelines(f"{json.dumps(data)}\n" for data in dataset)

    # Run benchmark
    extra_options = {
        "cuda_graph_config": {
            "enable_padding": True,
            "batch_sizes": [1, 2, 4, 8, 16, 32, 64, 128, 256, 384],
        },
    }
    with tempfile.TemporaryDirectory() as tmpdir:
        extra_options_path = Path(tmpdir) / "extra-llm-api-options.yml"
        with open(extra_options_path, "w") as f:
            yaml.dump(extra_options, f)

        output_path = Path(tmpdir) / "stdout.log"
        benchmark_cmd = \
                f"trtllm-bench --model {model_name} " \
                f"--model_path {llama_model_root} " \
                f"throughput " \
                f"--dataset {str(dataset_path)} --backend pytorch " \
                f"--extra_llm_api_options {extra_options_path} " \
                f"> {output_path} 2>&1"
        # Check clean shutdown (no hang)
        with pytest.raises(subprocess.CalledProcessError) as exc_info:
            check_call(benchmark_cmd, shell=True, env=llm_venv._new_env)
        # Check non-zero exit code
        assert exc_info.value.returncode != 0
        with open(output_path) as f:
            stdout = f.read()

    # Check that error is reported correctly
    assert "Requests failed: Token ID out of range (1 requests)" in stdout


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
def get_tmp_file():
    return tempfile.mkstemp()


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
                },
                "num_postprocess_workers": 2,
            }

            pytorch_backend_config = {}
            if request.node.callspec.params['pytorch_backend_config']:
                pytorch_backend_config = {
                    "cuda_graph_config": {},
                    # trtllm-bench will set cuda_max_batch_size to
                    # max_batch_size, so the cuda_graph_batch_sizes is not
                    # needed.
                    # "cuda_graph_batch_sizes": [1, 2, 3],
                }
            # Flatten the pytorch_backend_config
            extra_llm_api_options_dict.update(pytorch_backend_config)

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
        f"--backend tensorrt " \
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
        f"--dataset {dataset_path} --backend pytorch"

    mapping = {
        "Meta-Llama-3.1-8B": 19.4,
        "Llama-3.1-8B-Instruct-FP8": 12.0,
        "Meta-Llama-3.1-8B-NVFP4": 10.2
    }
    if use_extra_config:
        benchmark_cmd += f" --extra_llm_api_options {temp_extra_llm_api_options_file}"

    model_id = llama_model_root.split(r"/")[-1]
    if "nvfp4-quantized" in llama_model_root:
        model_id += "-NVFP4"
    with tempfile.NamedTemporaryFile(mode='w+t',
                                     suffix=f".{model_id}.log",
                                     dir="./",
                                     delete=True,
                                     delete_on_close=True) as running_log:
        check_call(benchmark_cmd, shell=True, stdout=running_log)
        if model_id in mapping and not use_extra_config:
            # extra config defines max kv cache tokens number to be 40000 which makes the checking
            # the checking process not unified.
            _check_mem_usage(running_log, [mapping[model_id], 0, 0, 0])


def test_trtllm_bench_mgmn(llm_root, llm_venv):
    model_name = "meta-llama/Llama-3.1-8B"
    llama_model_dir = Path(
        llm_models_root()) / "llama-3.1-model/Llama-3.1-8B-Instruct"
    _, _, dataset_path = trtllm_bench_prolog(llm_root,
                                             llm_venv,
                                             engine_dir=None,
                                             model_subdir=llama_model_dir,
                                             model_name=model_name,
                                             quant=None,
                                             streaming=False,
                                             skip_engine_build=True)

    benchmark_cmd = \
            f"mpirun --allow-run-as-root -n 2 trtllm-llmapi-launch trtllm-bench --model {model_name} " \
            f"--model_path {llama_model_dir} " \
            f"throughput " \
            f"--dataset {str(dataset_path)} --backend pytorch --tp 2"

    model_name = model_name.split(r"/")[-1]
    with tempfile.NamedTemporaryFile(mode='w+t',
                                     suffix=f".{model_name}.log",
                                     dir="./",
                                     delete=True,
                                     delete_on_close=True) as running_log:
        check_call(benchmark_cmd,
                   shell=True,
                   stdout=running_log,
                   env=llm_venv._new_env)
        _check_mem_usage(running_log, [30, 0, 0, 0])


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
        f"trtllm-bench --model {model_name} --model_path {model_path} latency " \
        f"--engine_dir {engine_path} --dataset {dataset_path} --backend tensorrt"
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
        f"trtllm-bench --model {model_name} --model_path {model_path} throughput " \
        f"--engine_dir {engine_path} --dataset {dataset_path} --backend tensorrt"

    if request_rate:
        benchmark_cmd += " --request_rate 100"
    if concurrency:
        benchmark_cmd += " --concurrency 100"

    print(f"cmd: {benchmark_cmd}")

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
@pytest.mark.parametrize("backend", ["tensorrt", "pytorch"],
                         ids=["TRT", "PyTorch"])
def test_trtllm_bench_iteration_log(llm_root, llm_venv, model_name,
                                    model_subdir, streaming, backend):
    '''
    Test the iteration log functionality with necessary options
    '''
    iteration_log = None
    engine_dir = None

    try:
        skip_engine_build = backend != "tensorrt"
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
            f"trtllm-bench --model {model_name} --model_path {model_path} " \
            f"throughput --dataset {dataset_path} --iteration_log {iteration_log}"

        if streaming:
            benchmark_cmd += " --streaming"

        benchmark_cmd += f" --backend {backend}"
        if skip_engine_build:
            assert engine_path is None, "Engine path should be None"
        else:
            assert engine_path is not None, "Engine path should not be None"
            benchmark_cmd += f" --engine_dir {engine_path}"

        if skip_engine_build:
            model_name = model_name.split("/")[-1]
            with tempfile.NamedTemporaryFile(
                    mode='w+t',
                    suffix=f".{model_name}_{streaming}.log",
                    dir="./",
                    delete=True,
                    delete_on_close=True) as running_log:
                check_call(benchmark_cmd, shell=True, stdout=running_log)
                _check_mem_usage(running_log, [19.4, 0, 0, 0])
        else:
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


def test_chatglm_6b_sanity(chatglm_6b_example_root, llm_venv, cmodel_dir,
                           engine_dir):
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


def test_mistral_large_hidden_vocab_size(llama_example_root, llm_venv,
                                         llama_tokenizer_model_root,
                                         engine_dir):
    """RCCA https://nvbugs/4753548"""
    config = {
        "architecture": "LlamaForCausalLM",
        "dtype": "float16",
        "vocab_size": 131072,
        "hidden_size": 16384,
        "num_hidden_layers": 1,
        "num_attention_heads": 96,
        "hidden_act": "silu",
        "logits_dtype": "float32",
        "norm_epsilon": 1e-06,
        "position_embedding_type": "rope_gpt_neox",
        "max_position_embeddings": 131072,
        "num_key_value_heads": 8,
        "intermediate_size": 36864,
        "head_size": 128,
    }

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
        "--max_input_len=8096",
        "--max_seq_len=52488",
        "--max_num_tokens=52488",
        "--gemm_plugin=float16",
        "--gpt_attention_plugin=float16",
        "--paged_kv_cache=enable",
        "--remove_input_padding=enable",
        "--max_batch_size=32",
    ]
    check_call(" ".join(build_cmd), shell=True, env=llm_venv._new_env)

    print("Run inference...")
    run_cmd = [
        f"{llama_example_root}/../../../run.py",
        "--max_output_len=20",
        f"--engine_dir={engine_dir}",
        f"--tokenizer_dir={llama_tokenizer_model_root}",
    ]
    venv_check_call(llm_venv, run_cmd)


def test_trtllm_serve_example(llm_root, llm_venv):
    example_root = Path(os.path.join(llm_root, "examples", "serve"))
    test_root = unittest_path() / "llmapi" / "apps"
    llm_venv.run_cmd([
        "-m", "pip", "install", "-r",
        os.path.join(example_root, "requirements.txt")
    ])
    llm_venv.run_cmd(
        ["-m", "pytest",
         str(test_root / "_test_trtllm_serve_example.py")])


def test_trtllm_serve_multimodal_example(llm_root, llm_venv):
    example_root = Path(os.path.join(llm_root, "examples", "serve"))
    test_root = unittest_path() / "llmapi" / "apps"
    llm_venv.run_cmd([
        "-m", "pip", "install", "-r",
        os.path.join(example_root, "requirements.txt")
    ])
    llm_venv.run_cmd([
        "-m", "pytest",
        str(test_root / "_test_trtllm_serve_multimodal_example.py")
    ])


def test_trtllm_serve_lora_example(llm_root, llm_venv):
    example_root = Path(os.path.join(llm_root, "examples", "serve"))
    test_root = unittest_path() / "llmapi" / "apps"
    llm_venv.run_cmd([
        "-m", "pip", "install", "-r",
        os.path.join(example_root, "requirements.txt")
    ])
    llm_venv.run_cmd(
        ["-m", "pytest",
         str(test_root / "_test_trtllm_serve_lora.py")])


@pytest.mark.parametrize("backend", ["pytorch", "trt"])
def test_trtllm_serve_top_logprobs(llm_root, llm_venv, backend: str):
    example_root = Path(os.path.join(llm_root, "examples", "serve"))
    test_root = unittest_path() / "llmapi" / "apps"
    llm_venv.run_cmd([
        "-m", "pip", "install", "-r",
        os.path.join(example_root, "requirements.txt")
    ])
    llm_venv.run_cmd([
        "-m", "pytest",
        str(test_root / "_test_trtllm_serve_top_logprobs.py"), "-k", backend
    ])


@pytest.mark.parametrize("backend", ["pytorch", "trt"])
def test_openai_misc_example(llm_root, llm_venv, backend: str):
    test_root = unittest_path() / "llmapi" / "apps"
    llm_venv.run_cmd([
        "-m", "pytest",
        str(test_root / "_test_openai_misc.py"), "-k", backend
    ])


def test_openai_cache_salt(llm_root, llm_venv):
    example_root = Path(os.path.join(llm_root, "examples", "serve"))
    test_root = unittest_path() / "llmapi" / "apps"
    llm_venv.run_cmd([
        "-m", "pip", "install", "-r",
        os.path.join(example_root, "requirements.txt")
    ])
    llm_venv.run_cmd(
        ["-m", "pytest",
         str(test_root / "_test_openai_cache_salt.py")])


@pytest.mark.parametrize("backend", ["pytorch", "trt"])
def test_openai_completions_example(llm_root, llm_venv, backend: str):
    test_root = unittest_path() / "llmapi" / "apps"
    filter_expr = f"{backend} and not sampler"
    llm_venv.run_cmd([
        "-m", "pytest",
        str(test_root / "_test_openai_completions.py"), "-k", filter_expr
    ])


@pytest.mark.parametrize("backend", ["pytorch", "trt"])
def test_openai_chat_example(llm_root, llm_venv, backend: str):
    test_root = unittest_path() / "llmapi" / "apps"
    filter_expr = f"{backend} and not sampler"
    llm_venv.run_cmd([
        "-m", "pytest",
        str(test_root / "_test_openai_chat.py"), "-k", filter_expr
    ])


@pytest.mark.parametrize("backend", ["pytorch", "trt"])
def test_openai_reasoning(llm_root, llm_venv, backend: str):
    test_root = unittest_path() / "llmapi" / "apps"
    llm_venv.run_cmd([
        "-m", "pytest",
        str(test_root / "_test_openai_reasoning.py"), "-k", backend
    ])


@pytest.mark.parametrize("sampler", ["torch_sampler", "trtllm_sampler"])
def test_openai_completions_with_logit_bias(llm_root, llm_venv, sampler: str):
    test_root = unittest_path() / "llmapi" / "apps"
    llm_venv.run_cmd([
        "-m", "pytest",
        str(test_root / "_test_openai_completions.py"), "-k", sampler
    ])


@pytest.mark.parametrize("sampler", ["torch_sampler", "trtllm_sampler"])
def test_openai_chat_with_logit_bias(llm_root, llm_venv, sampler: str):
    test_root = unittest_path() / "llmapi" / "apps"
    llm_venv.run_cmd([
        "-m", "pytest",
        str(test_root / "_test_openai_chat.py"), "-k", sampler
    ])


def test_openai_perf_metrics(llm_root, llm_venv):
    test_root = unittest_path() / "llmapi" / "apps"
    llm_venv.run_cmd(
        ["-m", "pytest",
         str(test_root / "_test_openai_perf_metrics.py")])


@skip_pre_hopper
def test_openai_chat_harmony(llm_root, llm_venv):
    test_root = unittest_path() / "llmapi" / "apps"
    llm_venv.run_cmd(
        ["-m", "pytest",
         str(test_root / "_test_openai_chat_harmony.py")])


def test_openai_responses(llm_root, llm_venv):
    test_root = unittest_path() / "llmapi" / "apps"
    llm_venv.run_cmd(
        ["-m", "pytest",
         str(test_root / "_test_openai_responses.py")])


def test_openai_prometheus(llm_root, llm_venv):
    test_root = unittest_path() / "llmapi" / "apps"
    llm_venv.run_cmd(
        ["-m", "pytest",
         str(test_root / "_test_openai_prometheus.py")])


def test_openai_lora(llm_root, llm_venv):
    test_root = unittest_path() / "llmapi" / "apps"
    llm_venv.run_cmd(["-m", "pytest", str(test_root / "_test_openai_lora.py")])


@pytest.mark.skip(reason="https://nvbugs/5596377")
def test_openai_chat_multimodal_example(llm_root, llm_venv):
    test_root = unittest_path() / "llmapi" / "apps"
    llm_venv.run_cmd(
        ["-m", "pytest",
         str(test_root / "_test_openai_chat_multimodal.py")])


def test_openai_mmencoder_example(llm_root, llm_venv):
    test_root = unittest_path() / "llmapi" / "apps"
    llm_venv.run_cmd(
        ["-m", "pytest",
         str(test_root / "_test_openai_mmencoder.py")])


def test_openai_chat_guided_decoding(llm_root, llm_venv):
    test_root = unittest_path() / "llmapi" / "apps"
    llm_venv.run_cmd([
        "-m", "pytest",
        str(test_root / "_test_openai_chat_guided_decoding.py")
    ])


@pytest.mark.skip_less_device(2)
@pytest.mark.skip_less_device_memory(40000)
def test_openai_multi_chat_example(llm_root, llm_venv):
    test_root = unittest_path() / "llmapi" / "apps"
    llm_venv.run_cmd(
        ["-m", "pytest",
         str(test_root / "_test_openai_multi_chat.py")])


@skip_nvlink_inactive
@pytest.mark.skip_less_device(4)
@pytest.mark.skip_less_device_memory(80000)
def test_openai_consistent_chat(llm_root, llm_venv):
    test_root = unittest_path() / "llmapi" / "apps"
    llm_venv.run_cmd(
        ["-m", "pytest",
         str(test_root / "_test_openai_consistent_chat.py")])


@skip_nvlink_inactive
@pytest.mark.skip_less_device(4)
@pytest.mark.skip_less_device_memory(80000)
def test_openai_multinodes_chat_tp16pp1(llm_root, llm_venv):
    test_root = unittest_path() / "llmapi" / "apps"
    llm_venv.run_cmd([
        "-m", "pytest", "-k", "tp16pp1",
        str(test_root / "_test_openai_multi_nodes.py")
    ])


@skip_nvlink_inactive
@pytest.mark.skip_less_device(4)
@pytest.mark.skip_less_device_memory(80000)
def test_openai_multinodes_chat_tp8pp2(llm_root, llm_venv):
    test_root = unittest_path() / "llmapi" / "apps"
    llm_venv.run_cmd([
        "-m", "pytest", "-k", "tp8pp2",
        str(test_root / "_test_openai_multi_nodes.py")
    ])


@pytest.mark.skip_less_device_memory(80000)
@pytest.mark.parametrize("model_name", [
    "llama-3.1-model/Meta-Llama-3.1-8B",
    pytest.param("gpt_oss/gpt-oss-20b", marks=skip_pre_hopper)
])
def test_trtllm_benchmark_serving(llm_venv, model_name):
    test_root = unittest_path() / "llmapi" / "apps"
    llm_venv.run_cmd([
        "-m", "pytest",
        str(test_root /
            f"_test_trtllm_serve_benchmark.py::test_trtllm_serve_benchmark[{model_name}]"
            )
    ])


def test_build_time_benchmark_sanity(llm_root, llm_venv):
    temp = tempfile.TemporaryDirectory()
    llm_venv.run_cmd([
        str(Path(llm_root) / "tests/microbenchmarks/build_time_dashboard.py"),
        '-m',
        temp.name,
    ])


@pytest.mark.skip_less_device_memory(80000)
def test_trtllm_multimodal_benchmark_serving(llm_root, llm_venv):
    test_root = unittest_path() / "llmapi" / "apps"
    llm_venv.run_cmd([
        "-m", "pytest",
        str(test_root / "_test_trtllm_serve_multimodal_benchmark.py")
    ])


@pytest.mark.skip_less_device(4)
@pytest.mark.skip_less_device_memory(40000)
@pytest.mark.parametrize("gen_config",
                         ["gen_tp2pp1", "gen_tp1pp2", "gen_tp1pp1"])
@pytest.mark.parametrize("ctx_config",
                         ["ctx_tp2pp1", "ctx_tp1pp2", "ctx_tp1pp1"])
def test_openai_disagg_multi_nodes_completion(llm_root, llm_venv, ctx_config,
                                              gen_config):
    test_root = unittest_path() / "llmapi" / "apps"
    llm_venv.run_cmd([
        "-m",
        "pytest",
        str(test_root /
            f"_test_disagg_serving_multi_nodes.py::test_completion[{ctx_config}-{gen_config}]"
            ),
    ])


### PyTorch examples


def parse_output(text):
    results = []
    text_lists = re.split(r"\[\d+\] Prompt:", text)
    for item in text_lists:
        item = item.replace(os.linesep, "")
        while True:
            match = re.search(r'Generated text: ([\'"])(.*?)\1', item,
                              re.MULTILINE)
            if match is None:
                break
            _, end = match.span(1)
            results.append(match.group(2))
            item = item[end:]
    return results


def test_ptp_quickstart(llm_root, llm_venv):
    example_root = Path(os.path.join(llm_root, "examples", "llm-api"))

    src = f"{llm_models_root()}/llama-3.1-model/Llama-3.1-8B-Instruct"
    dst = f"{llm_venv.get_working_directory()}/meta-llama/Llama-3.1-8B-Instruct"
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    os.symlink(src, dst, target_is_directory=True)

    with tempfile.NamedTemporaryFile(mode='w+t',
                                     suffix=".Llama-3.1-8B-Instruct.log",
                                     dir="./",
                                     delete=True,
                                     delete_on_close=True) as running_log:
        venv_check_call(llm_venv, [str(example_root / "quickstart_example.py")],
                        stdout=running_log)
        _check_mem_usage(running_log, [4.60, 0, 0, 0])


@pytest.mark.parametrize("model_name,model_path", [
    ("Llama3.1-8B-BF16", "llama-3.1-model/Meta-Llama-3.1-8B"),
    ("Llama3.2-11B-BF16", "llama-3.2-models/Llama-3.2-11B-Vision"),
    ("Nemotron4_4B-BF16", "nemotron/Minitron-4B-Base"),
    ("Nemotron-H-8B", "Nemotron-H-8B-Base-8K"),
    pytest.param('Llama3.1-8B-NVFP4',
                 'nvfp4-quantized/Meta-Llama-3.1-8B',
                 marks=skip_pre_blackwell),
    pytest.param('Llama3.1-8B-FP8',
                 'llama-3.1-model/Llama-3.1-8B-Instruct-FP8',
                 marks=skip_pre_hopper),
    pytest.param('Llama3.1-70B-NVFP4',
                 'nvfp4-quantized/Meta-Llama-3.1-70B',
                 marks=skip_pre_blackwell),
    pytest.param('Llama3.1-70B-FP8',
                 'llama-3.1-model/Llama-3.1-70B-Instruct-FP8',
                 marks=skip_pre_hopper),
    pytest.param('Nemotron-Super-49B-v1-NVFP4',
                 'nvfp4-quantized/Llama-3_3-Nemotron-Super-49B-v1_nvfp4_hf',
                 marks=skip_pre_hopper),
    pytest.param('Nemotron-Super-49B-v1-FP8',
                 'nemotron-nas/Llama-3_3-Nemotron-Super-49B-v1-FP8',
                 marks=skip_pre_hopper),
    pytest.param('Mixtral-8x7B-NVFP4',
                 'nvfp4-quantized/Mixtral-8x7B-Instruct-v0.1',
                 marks=skip_pre_blackwell),
    pytest.param('Mixtral-8x7B-FP8',
                 'Mixtral-8x7B-Instruct-v0.1-fp8',
                 marks=skip_pre_blackwell),
    pytest.param('Qwen3-30B-A3B',
                 'Qwen3/Qwen3-30B-A3B',
                 marks=pytest.mark.skip_less_device_memory(80000)),
    pytest.param(
        'Qwen3-30B-A3B_fp8_hf',
        'Qwen3/saved_models_Qwen3-30B-A3B_fp8_hf',
        marks=(skip_pre_hopper, pytest.mark.skip_less_device_memory(40000))),
    pytest.param(
        'Qwen3-30B-A3B_nvfp4_hf',
        'Qwen3/saved_models_Qwen3-30B-A3B_nvfp4_hf',
        marks=(skip_pre_blackwell, pytest.mark.skip_less_device_memory(20000))),
    pytest.param(
        'Llama3.3-70B-FP8',
        'modelopt-hf-model-hub/Llama-3.3-70B-Instruct-fp8',
        marks=(skip_pre_blackwell, pytest.mark.skip_less_device_memory(96000))),
    pytest.param('Llama3.3-70B-FP4',
                 'modelopt-hf-model-hub/Llama-3.3-70B-Instruct-fp4',
                 marks=skip_pre_blackwell),
    pytest.param('Nemotron-Super-49B-v1-BF16',
                 'nemotron-nas/Llama-3_3-Nemotron-Super-49B-v1',
                 marks=skip_pre_blackwell),
    pytest.param('Mixtral-8x7B-BF16',
                 'Mixtral-8x7B-Instruct-v0.1',
                 marks=skip_pre_blackwell),
    pytest.param('Mistral-Nemo-12b-Base',
                 'Mistral-Nemo-Base-2407',
                 marks=skip_pre_blackwell),
    pytest.param('DeepSeek-R1-Distill-Qwen-32B',
                 'DeepSeek-R1/DeepSeek-R1-Distill-Qwen-32B',
                 marks=skip_pre_blackwell),
    pytest.param('GPT-OSS-20B', 'gpt_oss/gpt-oss-20b',
                 marks=skip_pre_blackwell),
    pytest.param(
        'GPT-OSS-120B', 'gpt_oss/gpt-oss-120b', marks=skip_pre_blackwell),
])
def test_ptp_quickstart_advanced(llm_root, llm_venv, model_name, model_path):
    print(f"Testing {model_name}.")
    example_root = Path(os.path.join(llm_root, "examples", "llm-api"))
    if model_name == "Nemotron-H-8B":
        llm_venv.run_cmd([
            str(example_root / "quickstart_advanced.py"),
            "--disable_kv_cache_reuse",
            "--max_batch_size=8",
            "--model_dir",
            f"{llm_models_root()}/{model_path}",
        ])
    else:
        mapping = {
            "Llama3.1-8B-BF16": 18.60,
            "Llama3.2-11B-BF16": 18.88,
            "Nemotron4_4B-BF16": 12.50,
            "Llama3.1-8B-FP8": 13.05,
            "Llama3.1-8B-NVFP4": 10.2
        }
        with tempfile.NamedTemporaryFile(mode='w+t',
                                         suffix=f".{model_name}.log",
                                         dir="./",
                                         delete=True,
                                         delete_on_close=True) as running_log:
            cmds = [
                str(example_root / "quickstart_advanced.py"),
                "--enable_chunked_prefill",
                f"--model_dir={llm_models_root()}/{model_path}",
            ]
            if "Qwen3" in model_name:
                cmds.append(f"--kv_cache_fraction=0.6")
            if "Llama3.1-70B" in model_name:
                cmds.append(f"--max_num_tokens=1024")
            llm_venv.run_cmd(cmds, stdout=running_log)
            if model_name in mapping:
                _check_mem_usage(running_log, [mapping[model_name], 0, 0, 0])


@pytest.mark.parametrize("model_name,model_path", [
    ("DeepSeek-V3-Lite-BF16", "DeepSeek-V3-Lite/bf16"),
])
def test_ptp_quickstart_advanced_mtp(llm_root, llm_venv, model_name,
                                     model_path):
    print(f"Testing {model_name}.")
    example_root = Path(os.path.join(llm_root, "examples", "llm-api"))
    with tempfile.NamedTemporaryFile(mode='w+t',
                                     suffix=f".{model_name}.log",
                                     dir="./",
                                     delete=True,
                                     delete_on_close=True) as running_log:
        llm_venv.run_cmd(
            [
                str(example_root / "quickstart_advanced.py"),
                "--use_cuda_graph",
                "--spec_decode_max_draft_len",
                "1",  # test 1 MTP module
                "--spec_decode_algo",
                "MTP",
                "--model_dir",
                f"{llm_models_root()}/{model_path}",
                "--use_one_model",
            ],
            stdout=running_log)
        _check_mem_usage(running_log, [54.60, 0, 0, 0])


@pytest.mark.parametrize("model_name,model_path", [
    ("DeepSeek-V3-Lite-BF16", "DeepSeek-V3-Lite/bf16"),
])
def test_ptp_quickstart_advanced_mtp_eagle(llm_root, llm_venv, model_name,
                                           model_path):
    print(f"Testing {model_name}.")
    example_root = Path(os.path.join(llm_root, "examples", "llm-api"))
    with tempfile.NamedTemporaryFile(mode='w+t',
                                     suffix=f".{model_name}.log",
                                     dir="./",
                                     delete=True,
                                     delete_on_close=True) as running_log:
        llm_venv.run_cmd([
            str(example_root / "quickstart_advanced.py"),
            "--use_cuda_graph",
            "--spec_decode_max_draft_len",
            "3",
            "--spec_decode_algo",
            "MTP",
            "--model_dir",
            f"{llm_models_root()}/{model_path}",
        ],
                         stdout=running_log)
        # 74.60 is the memory usage for DeepSeek-V3-Lite-BF16 with MTP Eagle 2 two model style as one extra kv cache is needed for draft model.
        _check_mem_usage(running_log, [74.60, 0, 0, 0])


@pytest.mark.skip_less_device(4)
def test_ptp_quickstart_advanced_bs1(llm_root, llm_venv):
    model_name = "DeepSeek-V3-Lite-FP8"
    model_path = "DeepSeek-V3-Lite/fp8"
    print(f"Testing {model_name}.")
    example_root = Path(os.path.join(llm_root, "examples", "llm-api"))
    llm_venv.run_cmd([
        str(example_root / "quickstart_advanced.py"),
        "--use_cuda_graph",
        "--cuda_graph_padding_enabled",
        "--cuda_graph_batch_sizes",
        "8",
        "--disable_overlap_scheduler",
        "--enable_attention_dp",
        "--tp_size",
        "4",
        "--moe_ep_size",
        "4",
        "--prompt",
        "\"NVIDIA is a great company because\"",
        "--model_dir",
        f"{llm_models_root()}/{model_path}",
    ])


@pytest.mark.skip_less_device_memory(80000)
@pytest.mark.skip_less_device(8)
@skip_pre_hopper
@pytest.mark.parametrize("model_path", [
    pytest.param('DeepSeek-V3', marks=skip_post_blackwell),
    pytest.param('DeepSeek-V3-0324', marks=skip_post_blackwell),
    pytest.param('DeepSeek-R1/DeepSeek-R1-0528-FP4', marks=skip_pre_blackwell),
])
def test_ptp_quickstart_advanced_deepseek_multi_nodes(llm_root, llm_venv,
                                                      model_path):
    # "RCCA https://nvbugs/5163844"
    print(f"Testing {model_path}.")
    example_root = Path(os.path.join(llm_root, "examples", "llm-api"))
    run_cmd = [
        "trtllm-llmapi-launch",
        "python3",
        str(example_root / "quickstart_advanced.py"),
        f"--model_dir={llm_models_root()}/{model_path}",
        "--moe_ep_size=8",
        "--tp_size=16",
        "--use_cuda_graph",
        f"--kv_cache_fraction={_MEM_FRACTION_50}",
        "--max_batch_size=32",
        "--max_num_tokens=2048",
        "--disable_kv_cache_reuse",
    ]
    check_call(" ".join(run_cmd), shell=True, env=llm_venv._new_env)


@pytest.mark.parametrize("model_name,model_path,eagle_model_path", [
    ("Llama-3.1-8b-Instruct", "llama-3.1-model/Llama-3.1-8B-Instruct",
     "EAGLE3-LLaMA3.1-Instruct-8B"),
])
def test_ptp_quickstart_advanced_eagle3(llm_root, llm_venv, model_name,
                                        model_path, eagle_model_path):
    print(f"Testing {model_name}.")
    example_root = Path(os.path.join(llm_root, "examples", "llm-api"))
    with tempfile.NamedTemporaryFile(mode='w+t',
                                     suffix=f".{model_name}.log",
                                     dir="./",
                                     delete=True,
                                     delete_on_close=True) as running_log:
        llm_venv.run_cmd([
            str(example_root / "quickstart_advanced.py"),
            "--spec_decode_max_draft_len",
            "4",
            "--spec_decode_algo",
            "eagle3",
            "--model_dir",
            f"{llm_models_root()}/{model_path}",
            "--draft_model_dir",
            f"{llm_models_root()}/{eagle_model_path}",
            "--disable_kv_cache_reuse",
            "--disable_overlap_scheduler",
        ],
                         stdout=running_log)
        _check_mem_usage(running_log, [25.2, 0, 0, 0])


@pytest.mark.parametrize("model_name,model_path", [
    ("Llama-3.1-8B-Instruct", "llama-3.1-model/Llama-3.1-8B-Instruct"),
])
def test_ptp_quickstart_advanced_ngram(llm_root, llm_venv, model_name,
                                       model_path):
    print(f"Testing {model_name}.")
    example_root = Path(os.path.join(llm_root, "examples", "llm-api"))
    with tempfile.NamedTemporaryFile(mode='w+t',
                                     suffix=f".{model_name}.log",
                                     dir="./",
                                     delete=True,
                                     delete_on_close=True) as running_log:
        llm_venv.run_cmd([
            str(example_root / "quickstart_advanced.py"),
            "--model_dir",
            f"{llm_models_root()}/{model_path}",
            "--spec_decode_algo",
            "NGRAM",
            "--spec_decode_max_draft_len",
            "4",
            "--max_matching_ngram_size",
            "2",
            "--use_cuda_graph",
            "--disable_kv_cache_reuse",
            "--disable_overlap_scheduler",
        ],
                         stdout=running_log)
        _check_mem_usage(running_log, [27.0, 0, 0, 0])


@pytest.mark.parametrize("model_name,model_path", [
    ("Llama-3.1-8B-Instruct", "llama-3.1-model/Llama-3.1-8B-Instruct"),
])
def test_ptp_quickstart_advanced_auto(llm_root, llm_venv, model_name,
                                      model_path):
    print(f"Testing {model_name}.")
    example_root = Path(os.path.join(llm_root, "examples", "llm-api"))
    with tempfile.NamedTemporaryFile(mode='w+t',
                                     suffix=f".{model_name}.log",
                                     dir="./",
                                     delete=True,
                                     delete_on_close=True) as running_log:
        llm_venv.run_cmd([
            str(example_root / "quickstart_advanced.py"),
            "--model_dir",
            f"{llm_models_root()}/{model_path}",
            "--spec_decode_algo",
            "AUTO",
            "--use_cuda_graph",
            "--max_batch_size=4",
        ],
                         stdout=running_log)
        _check_mem_usage(running_log, [27.0, 0, 0, 0])


@skip_post_blackwell
@pytest.mark.skip_less_device_memory(80000)
@pytest.mark.skip_less_device(4)
@pytest.mark.parametrize("model_name,model_path", [
    pytest.param(
        'DeepSeek-V3-Lite-FP8', 'DeepSeek-V3-Lite/fp8', marks=skip_pre_hopper),
])
def test_ptp_quickstart_advanced_deepseek_v3_lite_4gpus_adp_balance(
        llm_root, llm_venv, model_name, model_path):
    print(f"Testing {model_name}.")
    example_root = Path(os.path.join(llm_root, "examples", "llm-api"))
    with tempfile.NamedTemporaryFile(mode='w+t',
                                     suffix=f".{model_name}.log",
                                     dir="./",
                                     delete=True,
                                     delete_on_close=True) as running_log:
        llm_venv.run_cmd([
            str(example_root / "quickstart_advanced.py"),
            "--model_dir",
            f"{llm_models_root()}/{model_path}",
            "--moe_tp_size=1",
            "--moe_ep_size=4",
            "--tp_size=4",
            "--use_cuda_graph",
            "--enable_attention_dp",
            f"--kv_cache_fraction={_MEM_FRACTION_95}",
            "--max_batch_size=1",
            "--max_seq_len=3000",
            "--disable_kv_cache_reuse",
            "--attention_dp_enable_balance",
            "--attention_dp_time_out_iters",
            "10",
            "--attention_dp_batching_wait_iters",
            "10",
        ],
                         stdout=running_log)
        _check_mem_usage(running_log, [106.3, 0, 0, 0], 8)


@skip_post_blackwell
@pytest.mark.skip_less_device_memory(110000)
@pytest.mark.skip_less_device(8)
@pytest.mark.parametrize("model_name,model_path", [
    pytest.param(
        'DeepSeek-R1', 'DeepSeek-R1/DeepSeek-R1', marks=skip_pre_hopper),
    pytest.param('DeepSeek-R1-0528-FP4',
                 'DeepSeek-R1/DeepSeek-R1-0528-FP4',
                 marks=skip_pre_blackwell),
])
def test_ptp_quickstart_advanced_deepseek_r1_8gpus(llm_root, llm_venv,
                                                   model_name, model_path):
    print(f"Testing {model_name}.")
    example_root = Path(os.path.join(llm_root, "examples", "llm-api"))
    with tempfile.NamedTemporaryFile(mode='w+t',
                                     suffix=f".{model_name}.log",
                                     dir="./",
                                     delete=True,
                                     delete_on_close=True) as running_log:
        llm_venv.run_cmd([
            str(example_root / "quickstart_advanced.py"),
            "--model_dir",
            f"{llm_models_root()}/{model_path}",
            "--moe_tp_size=1",
            "--moe_ep_size=8",
            "--tp_size=8",
            "--use_cuda_graph",
            "--enable_attention_dp",
            f"--kv_cache_fraction={_MEM_FRACTION_95}",
            "--max_batch_size=1",
            "--max_seq_len=3000",
            "--disable_kv_cache_reuse",
        ],
                         stdout=running_log)
        _check_mem_usage(running_log, [106.3, 0, 0, 0], 8)


@skip_post_blackwell
@pytest.mark.skip_less_device_memory(110000)
@pytest.mark.skip_less_device(8)
@pytest.mark.parametrize("model_name,model_path", [
    pytest.param(
        'DeepSeek-R1', 'DeepSeek-R1/DeepSeek-R1', marks=skip_pre_hopper),
])
def test_relaxed_acceptance_quickstart_advanced_deepseek_r1_8gpus(
        llm_root, llm_venv, model_name, model_path):
    print(f"Testing {model_name}.")
    example_root = Path(os.path.join(llm_root, "examples", "llm-api"))
    with tempfile.NamedTemporaryFile(mode='w+t',
                                     suffix=f".{model_name}.log",
                                     dir="./",
                                     delete=True,
                                     delete_on_close=True) as running_log:
        llm_venv.run_cmd([
            str(example_root / "quickstart_advanced.py"),
            "--model_dir",
            f"{llm_models_root()}/{model_path}",
            "--moe_tp_size=1",
            "--moe_ep_size=8",
            "--tp_size=8",
            "--use_cuda_graph",
            f"--kv_cache_fraction={_MEM_FRACTION_95}",
            "--max_batch_size=1",
            "--max_seq_len=3000",
            "--disable_kv_cache_reuse",
            "--spec_decode_algo",
            "MTP",
            "--spec_decode_max_draft_len",
            "5",
            "--use_relaxed_acceptance_for_thinking",
            "--relaxed_topk=10",
            "--relaxed_delta=0.5",
            "--enable_attention_dp",
            "--use_one_model",
        ],
                         stdout=running_log)
        _check_mem_usage(running_log, [85.6, 0, 0, 0], 8)


@skip_pre_ada
@skip_post_blackwell
@pytest.mark.skip_less_device_memory(80000)
@pytest.mark.skip_less_device(8)
@pytest.mark.parametrize("model_name,model_path", [
    pytest.param('DeepSeek-R1-W4AFP8',
                 'DeepSeek-R1/DeepSeek-R1-W4AFP8',
                 marks=skip_pre_hopper),
])
def test_ptp_quickstart_advanced_deepseek_r1_w4afp8_8gpus(
        llm_root, llm_venv, model_name, model_path):
    print(f"Testing {model_name}.")
    example_root = Path(os.path.join(llm_root, "examples", "llm-api"))
    with tempfile.NamedTemporaryFile(mode='w+t',
                                     suffix=f".{model_name}.log",
                                     dir="./",
                                     delete=True,
                                     delete_on_close=True) as running_log:
        llm_venv.run_cmd([
            str(example_root / "quickstart_advanced.py"),
            "--model_dir",
            f"{llm_models_root()}/{model_path}",
            "--moe_tp_size=1",
            "--moe_ep_size=8",
            "--tp_size=8",
            "--use_cuda_graph",
            f"--kv_cache_fraction={_MEM_FRACTION_50}",
            "--max_batch_size=1",
            "--max_seq_len=512",
        ],
                         stdout=running_log)
        _check_mem_usage(running_log, [50.0, 0, 0, 0], 8)


@pytest.mark.skip_less_device_memory(80000)
@pytest.mark.parametrize("model_name,model_path,gpu_count", [
    ("Llama3.1-70B-BF16", "llama-3.1-model/Meta-Llama-3.1-70B", 2),
    ("Mixtral-8x7B-BF16", "Mixtral-8x7B-v0.1", 8),
    pytest.param('Llama3.1-70B-FP8',
                 'llama-3.1-model/Llama-3.1-70B-Instruct-FP8',
                 2,
                 marks=skip_pre_hopper),
    pytest.param('Llama3.1-405B-FP8',
                 'llama-3.1-model/Llama-3.1-405B-Instruct-FP8',
                 8,
                 marks=(skip_pre_hopper, pytest.mark.timeout(7200))),
    pytest.param('Mixtral-8x7B-NVFP4',
                 'nvfp4-quantized/Mixtral-8x7B-Instruct-v0.1',
                 8,
                 marks=skip_pre_blackwell),
    pytest.param('Nemotron-Ultra-253B',
                 'nemotron-nas/Llama-3_1-Nemotron-Ultra-253B-v1',
                 8,
                 marks=(skip_pre_hopper, pytest.mark.timeout(12600))),
    pytest.param('DeepSeek-V3-671B-FP8',
                 'DeepSeek-V3-0324',
                 8,
                 marks=(skip_post_blackwell,
                        pytest.mark.skip_less_device_memory(140000))),
])
def test_ptp_quickstart_advanced_multi_gpus(llm_root, llm_venv, model_name,
                                            model_path, gpu_count):
    print(f"Testing {model_name}.")
    if gpu_count > get_device_count():
        pytest.skip(f"Not enough GPUs for {model_name}")
    example_root = Path(os.path.join(llm_root, "examples", "llm-api"))
    mapping = {
        "Llama3.1-70B-BF16": 91.0,
        "Mixtral-8x7B-BF16": 16.5,
        "Llama3.1-70B-FP8": 58.5,
        "Llama3.1-405B-FP8": 63.2,
        "Mixtral-8x7B-NVFP4": 9.9,
        "Nemotron-Ultra-253B": 72.3,
        "DeepSeek-V3-671B-FP8": 83.8
    }
    with tempfile.NamedTemporaryFile(mode='w+t',
                                     suffix=f".{model_name}.log",
                                     dir="./",
                                     delete=True,
                                     delete_on_close=True) as running_log:
        llm_venv.run_cmd([
            str(example_root / "quickstart_advanced.py"),
            "--enable_chunked_prefill",
            "--model_dir",
            f"{llm_models_root()}/{model_path}",
            f"--tp_size={gpu_count}",
            "--max_batch_size=32",
            "--max_num_tokens=256",
        ],
                         stdout=running_log)
        if model_name in mapping:
            _check_mem_usage(running_log, [mapping[model_name], 0, 0, 0],
                             gpu_count)


@skip_pre_hopper
@pytest.mark.skip_less_device(8)
@pytest.mark.parametrize("cuda_graph", [False, True])
@pytest.mark.parametrize("model_name,model_path", [
    ("Llama-4-Maverick-17B-128E-Instruct-FP8",
     "llama4-models/nvidia/Llama-4-Maverick-17B-128E-Instruct-FP8"),
    ("Llama-4-Scout-17B-16E-Instruct-FP8",
     "llama4-models/Llama-4-Scout-17B-16E-Instruct-FP8"),
    pytest.param('Llama-4-Scout-17B-16E-Instruct-FP4',
                 'llama4-models/Llama-4-Scout-17B-16E-Instruct-FP4',
                 marks=skip_pre_blackwell),
])
def test_ptp_quickstart_advanced_8gpus_chunked_prefill_sq_22k(
        llm_root, llm_venv, model_name, model_path, cuda_graph):
    print(f"Testing {model_name} on 8 GPUs.")
    example_root = Path(os.path.join(llm_root, "examples", "llm-api"))
    cmd = [
        str(example_root / "quickstart_advanced.py"),
        "--enable_chunked_prefill",
        "--model_dir",
        f"{llm_models_root()}/{model_path}",
        "--tp_size=8",
        "--moe_ep_size=8",
        "--max_seq_len=22000",
        "--kv_cache_fraction=0.1",
    ]
    if cuda_graph:
        cmd.extend([
            "--use_cuda_graph",
            "--cuda_graph_padding_enabled",
        ])
    llm_venv.run_cmd(cmd)


# This test is specifically to be run on 2 GPUs on Blackwell RTX 6000 Pro (SM120) architecture
# TODO: remove once we have a node with 8 GPUs and reuse test_ptp_quickstart_advanced_8gpus
@skip_no_sm120
@pytest.mark.skip_less_device_memory(80000)
@pytest.mark.skip_less_device(2)
@pytest.mark.parametrize("model_name,model_path", [
    ('Nemotron-Super-49B-v1-BF16',
     'nemotron-nas/Llama-3_3-Nemotron-Super-49B-v1'),
    ("Mixtral-8x7B-BF16", "Mixtral-8x7B-Instruct-v0.1"),
    pytest.param('Llama3.1-70B-BF16',
                 'llama-3.1-model/Meta-Llama-3.1-70B',
                 marks=pytest.mark.skip_less_device_memory(95000)),
])
def test_ptp_quickstart_advanced_2gpus_sm120(llm_root, llm_venv, model_name,
                                             model_path):
    print(f"Testing {model_name} on 2 GPUs (SM120+).")
    example_root = Path(os.path.join(llm_root, "examples", "llm-api"))
    llm_venv.run_cmd([
        str(example_root / "quickstart_advanced.py"),
        "--enable_chunked_prefill",
        "--model_dir",
        f"{llm_models_root()}/{model_path}",
        "--tp_size=2",
        "--max_num_tokens=256",
    ])


@skip_pre_blackwell
def test_ptp_quickstart_advanced_mixed_precision(llm_root, llm_venv):
    example_root = Path(os.path.join(llm_root, "examples", "llm-api"))
    model_path = "Llama-3_1-8B-Instruct_fp8_nvfp4_hf"
    with tempfile.NamedTemporaryFile(mode='w+t',
                                     suffix=f".{model_path}.log",
                                     dir="./",
                                     delete=True,
                                     delete_on_close=True) as running_log:
        llm_venv.run_cmd([
            str(example_root / "quickstart_advanced.py"),
            "--model_dir",
            f"{llm_models_root()}/{model_path}",
        ],
                         stdout=running_log)
        _check_mem_usage(running_log, [12.0, 0, 0, 0])


@pytest.mark.parametrize("use_cuda_graph", [False, True])
@pytest.mark.parametrize("modality", ["image", "video", "mixture_text_image"])
@pytest.mark.parametrize("model_name,model_path", [
    ("NVILA-8B-FP16", "vila/NVILA-8B"),
    ("NVILA-15B-FP16", "NVILA-15B"),
    ("llava-v1.6-mistral-7b", "llava-v1.6-mistral-7b-hf"),
    ("qwen2-vl-7b-instruct", "Qwen2-VL-7B-Instruct"),
    ("qwen2.5-vl-7b-instruct", "Qwen2.5-VL-7B-Instruct"),
    pytest.param("mistral-small-3.1-24b-instruct",
                 "Mistral-Small-3.1-24B-Instruct-2503",
                 marks=pytest.mark.skip_less_device_memory(80000)),
    pytest.param("gemma-3-27b-it",
                 "gemma/gemma-3-27b-it",
                 marks=(pytest.mark.skip_less_device_memory(80000),
                        skip_post_blackwell)),
    pytest.param(
        "Nano-v2-VLM",
        "Nano-v2-VLM",
        marks=pytest.mark.skip(reason="Nano V2 VLM ckpt is not released yet.")),
])
def test_ptp_quickstart_multimodal(llm_root, llm_venv, model_name, model_path,
                                   modality, use_cuda_graph):
    # NOTE: individual tests need to be enabled in
    # tests/integration/test_lists/qa/examples_test_list.txt

    example_root = Path(os.path.join(llm_root, "examples", "llm-api"))
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
        "mixture_text_image": {
            "prompt": [
                "Who invented the internet?",
                "Describe the scene in the image briefly.",
            ],
            "media": [
                "",
                str(test_data_root / "inpaint.png"),
            ],
        }
    }

    expected_keywords = {
        "Nano-v2-VLM": {
            "image": [
                ["natural", "ocean", "waves", "stormy", "overcast", "sea"],
                ["mountain", "rock", "large", "clear", "blue", "sky"],
                ["road", "lane", "vehicle", "bus", "cars", "traffic"],
            ],
            "video": [
                [
                    "person", "red", "dress", "walking", "street", "black",
                    "leather", "jacket"
                ],
                ["space", "earth", "black", "frame", "city", "lights", "dark"],
            ],
            "mixture_text_image":
            [["invented", "internet", "person", "people", "computers"],
             ["large", "rock", "mountain", "center", "sky", "clear", "trees"]]
        },
        "NVILA-8B-FP16": {
            "image": [
                ["stormy", "ocean", "waves", "cloudy", "sunlight", "sky"],
                ["rock", "formation", "sunny", "sky", "clouds"],
                ["road", "busy", "car", "black", "blue"],
            ],
            "video": [
                ["woman", "street", "night", "walking", "camera"],
                [
                    "stunning", "earth", "space", "planet", "curvature", "dark",
                    "bright", "contrast", "illuminate"
                ],
            ],
        },
        "llava-v1.6-mistral-7b": {
            "image": [
                ["ocean", "sky", "large", "waves", "shore", "blue"],
                ['mountain', 'flat', 'clouds', 'road', 'sky'],
                ["highway", "vehicles", "traffic", "bus", "suburban"],
            ],
        },
        "qwen2-vl-7b-instruct": {
            "image": [
                ["ocean", "waves", "atmosphere", "stormy", "clouds", "intense"],
                ["trees", "winding", "road", "sunny", "sky", "atmosphere"],
                ["traffic", "vehicles", "moderate", "lanes", "road", "cars"],
            ],
            "video": [
                ["city", "night", "lights", "jacket", "wet"],
                ["earth", "spinning", "black"],
            ],
        },
        "qwen2.5-vl-7b-instruct": {
            "image": [
                ["dramatic", "moody", "ocean", "stormy", "sky", "waves"],
                ["large", "dome", "yosemite", "landmark", "rock", "road"],
                [
                    "highway", "traffic", "vehicles", "lanes", "congestion",
                    "road"
                ],
            ],
            "video": [
                ["woman", "neon", "night", "jacket", "wet"],
                ["earth", "world", "night", "lights", "cities"],
            ],
        },
        "mistral-small-3.1-24b-instruct": {
            "image": [
                ["dramatic", "seascape", "ocean", "turbulent", "waves", "dark"],
                ["scenic", "rock", "landscape", "monolith", "formation"],
                [
                    "multi-lane", "highway", "moderate", "traffic", "flow",
                    "vehicles", "congestion"
                ],
            ],
            "mixture_text_image":
            [["invention", "person", "scientists", "Lick", "engineers"],
             ["landscape", "trees", "road", "depicts", "scenic"]]
        },
        "gemma-3-27b-it": {
            "image": [
                ["natural", "turbulent", "dramatic", "scene", "wave"],
                ["image", "famous", "rock", "granite", "landmark"],
                ["traffic", "moderate", "heavy", "flowing", "cars"],
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
        # TODO: remove this once kv cache reuse is supported for all VLM models
        "--disable_kv_cache_reuse",
    ]
    # NOTE: Qwen2-VL and Qwen2-5-VL model need larger max_num_tokens for video.
    if model_name in ["qwen2-vl-7b-instruct", "qwen2.5-vl-7b-instruct"
                      ] and modality == "video":
        cmd.append("--max_num_tokens=16384")
    if use_cuda_graph:
        cmd.append("--use_cuda_graph")
    # Gemma3 VLM needs a custom mask which is only supported by flashinfer backend currently.
    # Custom mask involves bidirectional masking of image tokens in context phase. To get this
    # correct, chunked prefill and kv cache reuse need to be turned off.
    if model_name == "gemma-3-27b-it":
        cmd.append("--image_format=pil")
        cmd.append("--attention_backend=FLASHINFER")
        cmd.append("--disable_kv_cache_reuse")
        cmd.append("--kv_cache_fraction=0.5")
        cmd.append("--max_seq_len=1024")
    # Nano V2 VLM needs smaller max_batch_size to save memory.
    # Also need to disable kv cache reuse for Nemotron-H architecture.
    if model_name == "Nano-v2-VLM":
        cmd.append("--max_batch_size=128")
        cmd.append("--disable_kv_cache_reuse")
        if modality == "video":
            cmd.append("--max_num_tokens=20480")

    output = llm_venv.run_cmd(cmd, caller=check_output)

    match_ratio = 4.0 / 5
    if model_name == "qwen2-vl-7b-instruct" and modality == "image":
        match_ratio = 4.0 / 6

    parsed_outputs = parse_output(output)
    for prompt_output, prompt_keywords in zip(
            parsed_outputs, expected_keywords[model_name][modality]):
        matches = [
            keyword in prompt_output.lower() for keyword in prompt_keywords
        ]
        obs_match_ratio = 1. * sum(matches) / len(matches)
        assert obs_match_ratio >= match_ratio, f"Incorrect output!\nGenerated \"{prompt_output}\"\nExpected keywords \"{prompt_keywords}\"\n Matched keywords: {matches}\n Observed match ratio {obs_match_ratio} below threshold {match_ratio}\n\nParsed output for all prompts: {parsed_outputs}"

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

    mapping = {
        "NVILA-8B-FP16": [72.3, 0.6],
    }

    with tempfile.NamedTemporaryFile(mode='w+t',
                                     suffix=f".{model_name}.log",
                                     dir="./",
                                     delete=True,
                                     delete_on_close=True) as running_log:
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
            "--disable_kv_cache_reuse",
        ],
                         stdout=running_log)

        if model_name in mapping:
            peak, fraction = mapping[model_name]
            _check_mem_usage(running_log, [peak, 0, 0, 0])


@pytest.mark.parametrize("modality", ["image", "video"])
@pytest.mark.parametrize(
    "model_name,model_path,match_ratio",
    [
        ("llava-v1.6-mistral-7b", "llava-v1.6-mistral-7b-hf", 0.8),
        ("qwen2.5-vl-7b-instruct", "Qwen2.5-VL-7B-Instruct", 0.8),
        ("phi4-multimodal-instruct", "multimodals/Phi-4-multimodal-instruct",
         0.8),
        pytest.param("phi4-multimodal-instruct-fp4",
                     "multimodals/Phi-4-multimodal-instruct-FP4",
                     0.8,
                     marks=skip_pre_blackwell),
        pytest.param("phi4-multimodal-instruct-fp8",
                     "multimodals/Phi-4-multimodal-instruct-FP8",
                     0.8,
                     marks=skip_pre_hopper),
        pytest.param(
            "mistral-small-3.1-24b-instruct",
            "Mistral-Small-3.1-24B-Instruct-2503",
            # Lower threshold to give some wiggle room for flakiness.
            0.6,
            marks=pytest.mark.skip_less_device_memory(80000)),
    ])
def test_ptp_quickstart_multimodal_kv_cache_reuse(llm_root, llm_venv,
                                                  model_name, model_path,
                                                  modality, match_ratio):
    # NOTE: individual tests need to be enabled in
    # tests/integration/test_lists/qa/examples_test_list.txt

    example_root = Path(os.path.join(llm_root, "examples", "llm-api"))
    test_data_root = Path(
        os.path.join(llm_models_root(), "multimodals", "test_data"))
    print(f"Accuracy test {model_name} {modality} mode with example inputs.")
    if modality == "video" and model_name in {
            "llava-v1.6-mistral-7b", "mistral-small-3.1-24b-instruct",
            "phi4-multimodal-instruct", "phi4-multimodal-instruct-fp4",
            "phi4-multimodal-instruct-fp8"
    }:
        pytest.skip(f"Skipping video modality test for {model_name}")

    num_same_requests = 3  # test kv cache reuse with multiple same requests
    accuracy_inputs = {
        "image": {
            "prompt": [
                "Describe the natural environment in the image.",
            ] * num_same_requests,
            "media": [
                str(test_data_root / "seashore.png"),
            ] * num_same_requests,
        },
        "video": {
            "prompt": [
                "Tell me what you see in the video briefly.",
            ] * num_same_requests,
            "media": [
                str(test_data_root / "OAI-sora-tokyo-walk.mp4"),
            ] * num_same_requests,
        },
    }

    expected_keywords = {
        "llava-v1.6-mistral-7b": {
            "image": [
                ["ocean", "sky", "large", "waves", "shore", "blue"],
            ] * num_same_requests,
        },
        "qwen2.5-vl-7b-instruct": {
            "image": [
                ["dramatic", "moody", "ocean", "stormy", "sky", "waves"],
            ] * num_same_requests,
            "video": [
                ["woman", "neon", "night", "jacket", "wet"],
            ] * num_same_requests,
        },
        "mistral-small-3.1-24b-instruct": {
            "image": [
                [
                    "cloud", "dramatic", "seascape", "ocean", "turbulent",
                    "waves"
                ],
            ] * num_same_requests,
        },
        "phi4-multimodal-instruct": {
            "image": [
                [
                    "image", "depicts", "natural", "environment", "ocean",
                    "water", "waves", "sky"
                ],
            ] * num_same_requests,
        },
        "phi4-multimodal-instruct-fp4": {
            "image": [
                [
                    "image", "depicts", "natural", "environment", "ocean",
                    "water", "waves", "sky"
                ],
            ] * num_same_requests,
        },
        "phi4-multimodal-instruct-fp8": {
            "image": [
                [
                    "image", "depicts", "natural", "environment", "ocean",
                    "water", "waves", "sky"
                ],
            ] * num_same_requests,
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
        "--max_batch_size",  # single request at a time to test kv cache reuse
        "1",
    ]
    # NOTE: Qwen2-VL and Qwen2-5-VL model need larger max_num_tokens for video.
    if model_name in ["qwen2-vl-7b-instruct", "qwen2.5-vl-7b-instruct"
                      ] and modality == "video":
        cmd.append("--max_num_tokens=16384")

    if model_name.startswith("phi4-multimodal-instruct"):
        cmd.append("--max_seq_len=4096")
        cmd.append("--load_lora")
        cmd.append("--auto_model_name")
        cmd.append("Phi4MMForCausalLM")

    output = llm_venv.run_cmd(cmd, caller=check_output)
    match_ratio = 4.0 / 5
    for prompt_output, prompt_keywords in zip(
            parse_output(output), expected_keywords[model_name][modality]):
        matches = [
            keyword in prompt_output.lower() for keyword in prompt_keywords
        ]
        obs_match_ratio = 1. * sum(matches) / len(matches)
        print(
            f"Prompt output: {prompt_output}\nExpected keywords: {prompt_keywords}\n Matched keywords: {matches}\n Observed match ratio {obs_match_ratio} given threshold {match_ratio}"
        )
        assert obs_match_ratio >= match_ratio, f"Incorrect output!\nGenerated \"{prompt_output}\"\nExpected keywords \"{prompt_keywords}\"\n Matched keywords: {matches}\n Observed match ratio {obs_match_ratio} below threshold {match_ratio}"
    # TODO: Setting max_batch_size=1 and repeating the same request helps test KV cache reuse indirectly,
    # but does not directly measure the KV cache hit rate. For a more direct test, we would need to enable
    # return_perf_metrics=True, which is not currently supported by the quickstart example CLI.
    print("All answers are correct!")


@pytest.mark.parametrize("modality", ["image", "video"])
@pytest.mark.parametrize(
    "model_name,model_path,match_ratio",
    [
        ("llava-v1.6-mistral-7b", "llava-v1.6-mistral-7b-hf", 0.8),
        ("qwen2.5-vl-7b-instruct", "Qwen2.5-VL-7B-Instruct", 0.8),
        ("phi4-multimodal-instruct", "multimodals/Phi-4-multimodal-instruct",
         0.8),
        pytest.param("phi4-multimodal-instruct-fp4",
                     "multimodals/Phi-4-multimodal-instruct-FP4",
                     0.8,
                     marks=skip_pre_blackwell),
        pytest.param("phi4-multimodal-instruct-fp8",
                     "multimodals/Phi-4-multimodal-instruct-FP8",
                     0.8,
                     marks=skip_pre_hopper),
        pytest.param(
            "mistral-small-3.1-24b-instruct",
            "Mistral-Small-3.1-24B-Instruct-2503",
            # Lower threshold to give some wiggle room for flakiness.
            0.6,
            marks=pytest.mark.skip_less_device_memory(80000)),
    ])
def test_ptp_quickstart_multimodal_chunked_prefill(llm_root, llm_venv,
                                                   model_name, model_path,
                                                   modality, match_ratio):
    # NOTE: individual tests need to be enabled in
    # tests/integration/test_lists/qa/examples_test_list.txt

    example_root = Path(os.path.join(llm_root, "examples", "llm-api"))
    test_data_root = Path(
        os.path.join(llm_models_root(), "multimodals", "test_data"))
    print(f"Accuracy test {model_name} {modality} mode with example inputs.")
    if modality == "video" and model_name in {
            "llava-v1.6-mistral-7b", "mistral-small-3.1-24b-instruct",
            "phi4-multimodal-instruct", "phi4-multimodal-instruct-fp4",
            "phi4-multimodal-instruct-fp8"
    }:
        pytest.skip(f"Skipping video modality test for {model_name}")
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

    expected_keywords = {
        "llava-v1.6-mistral-7b": {
            "image": [
                ["ocean", "sky", "large", "waves", "shore", "blue"],
                ['mountain', 'flat', 'clouds', 'road', 'sky'],
                ["highway", "vehicles", "traffic", "bus", "suburban"],
            ],
        },
        "qwen2.5-vl-7b-instruct": {
            "image": [
                ["dramatic", "moody", "ocean", "stormy", "sky", "waves"],
                ["large", "dome", "yosemite", "landmark", "rock", "road"],
                [
                    "highway", "traffic", "vehicles", "lanes", "congestion",
                    "road"
                ],
            ],
            "video": [
                ["woman", "neon", "night", "jacket", "wet"],
                ["earth", "world", "night", "lights", "cities"],
            ],
        },
        "mistral-small-3.1-24b-instruct": {
            "image": [
                [
                    "cloud", "dramatic", "seascape", "ocean", "turbulent",
                    "waves"
                ],
                ["scenic", "rock", "landscape", "monolith", "formation"],
                [
                    "multi-lane", "highway", "moderate", "traffic", "flow",
                    "vehicles", "congestion"
                ],
            ],
        },
        "phi4-multimodal-instruct": {
            "image": [
                [
                    "image", "depicts", "natural", "environment", "ocean",
                    "water", "waves", "sky"
                ],
                [
                    "object", "mountain", "weather", "condition", "clear",
                    "visible"
                ],
                [
                    "traffic", "condition", "road", "moderate", "vehicles",
                    "lanes", "cars", "bus"
                ],
            ],
        },
        "phi4-multimodal-instruct-fp8": {
            "image": [
                [
                    "image", "depicts", "natural", "environment", "ocean",
                    "water", "waves", "sky"
                ],
                [
                    "object", "mountain", "weather", "condition", "clear",
                    "visible"
                ],
                [
                    "traffic", "condition", "road", "moderate", "vehicles",
                    "lanes", "cars", "bus"
                ],
            ],
        },
        "phi4-multimodal-instruct-fp4": {
            "image": [
                [
                    "image", "depicts", "natural", "environment", "ocean",
                    "water", "waves", "sky"
                ],
                ["rock", "formation", "sunny", "sky", "clouds"],
                [
                    "traffic", "condition", "road", "moderate", "vehicles",
                    "lane", "flow", "traffic"
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
        "--enable_chunked_prefill",
        "--max_num_tokens=256",
    ]
    if model_name.startswith("phi4-multimodal-instruct"):
        cmd.append("--max_seq_len=4096")
        cmd.append("--load_lora")
        cmd.append("--auto_model_name")
        cmd.append("Phi4MMForCausalLM")

    output = llm_venv.run_cmd(cmd, caller=check_output)
    for prompt_output, prompt_keywords in zip(
            parse_output(output), expected_keywords[model_name][modality]):
        matches = [
            keyword in prompt_output.lower() for keyword in prompt_keywords
        ]
        obs_match_ratio = 1. * sum(matches) / len(matches)
        print(
            f"Prompt output: {prompt_output}\nExpected keywords: {prompt_keywords}\n Matched keywords: {matches}\n Observed match ratio {obs_match_ratio} given threshold {match_ratio}"
        )
        assert obs_match_ratio >= match_ratio, f"Incorrect output!\nGenerated \"{prompt_output}\"\nExpected keywords \"{prompt_keywords}\"\n Matched keywords: {matches}\n Observed match ratio {obs_match_ratio} below threshold {match_ratio}"
    print("All answers are correct!")


@pytest.mark.parametrize("modality", ["image", "audio", "image_audio"])
@pytest.mark.parametrize("model_name,model_path", [
    ("phi4-multimodal-instruct", "multimodals/Phi-4-multimodal-instruct"),
    pytest.param("phi4-multimodal-instruct-fp4",
                 "multimodals/Phi-4-multimodal-instruct-FP4",
                 marks=skip_pre_blackwell),
    pytest.param("phi4-multimodal-instruct-fp8",
                 "multimodals/Phi-4-multimodal-instruct-FP8",
                 marks=skip_pre_hopper),
])
def test_ptp_quickstart_multimodal_phi4mm(llm_root, llm_venv, model_name,
                                          model_path, modality):
    example_root = Path(os.path.join(llm_root, "examples", "llm-api"))
    test_data_root = Path(
        os.path.join(llm_models_root(), "multimodals", "test_data"))
    audio_data_root = Path(
        os.path.join(llm_models_root(), "multimodals",
                     "Phi-4-multimodal-instruct", "examples"))
    print(f"Accuracy test {model_name} {modality} mode with example inputs.")
    accuracy_inputs = {
        "image": {
            "prompt": [
                "Describe the object and the weather condition in the image.",
                "Describe the traffic condition on the road in the image.",
            ],
            "media": [
                str(test_data_root / "inpaint.png"),
                str(test_data_root / "61.jpg"),
            ],
        },
        "audio": {
            "prompt": [
                "Transcribe the audio clip into text, please don't add other text.",
                "Transcribe the audio clip into text, please don't add other text.",
            ],
            "media": [
                str(audio_data_root /
                    "what_is_the_traffic_sign_in_the_image.wav"),
                str(audio_data_root / "what_is_shown_in_this_image.wav"),
            ],
        },
        "image_audio": {
            "prompt": [
                "",
            ],
            "media": [
                str(test_data_root / "inpaint.png"),
                str(audio_data_root / "what_is_shown_in_this_image.wav"),
            ],
        }
    }
    expected_keywords = {
        "image": [
            ["object", "mountain", "weather", "clear", "clouds"],
            ["traffic", "road", "vehicles", "cars", "bus"],
        ],
        "audio": [
            ["what", "is", "the", "traffic", "sign", "in", "image"],
            ["what", "is", "shown", "in", "this", "image"],
        ],
        "image_audio": [
            ["image", "depicts", "scenic", "famous", "landmark"],
        ],
    }

    if model_name == "phi4-multimodal-instruct-fp4":
        expected_keywords["image_audio"] = [
            ["image", "shows", "mountain", "El", "Capitan", "road", "trees"],
        ]

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
        # Set max_seq_len to 4096 to use short rope factor.
        "--max_seq_len=4096",
        "--load_lora",
        "--auto_model_name",
        "Phi4MMForCausalLM",
    ]
    output = llm_venv.run_cmd(cmd, caller=check_output)

    match_ratio = 0.6
    parsed_outputs = parse_output(output)
    for prompt_output, prompt_keywords in zip(parsed_outputs,
                                              expected_keywords[modality]):
        matches = [
            keyword in prompt_output.lower() for keyword in prompt_keywords
        ]
        obs_match_ratio = 1. * sum(matches) / len(matches)
        assert obs_match_ratio >= match_ratio, f"Incorrect output!\nGenerated \"{prompt_output}\"\nExpected keywords \"{prompt_keywords}\"\n Matched keywords: {matches}\n Observed match ratio {obs_match_ratio} below threshold {match_ratio}\n\nParsed output for all prompts: {parsed_outputs}"

    print("All answers are correct!")


@pytest.mark.skip_less_device(2)
@pytest.mark.skip_less_device_memory(80000)
@pytest.mark.parametrize("model_name,model_path", [
    pytest.param(
        "gemma-3-27b-it", "gemma/gemma-3-27b-it", marks=skip_post_blackwell),
    ("mistral-small-3.1-24b-instruct", "Mistral-Small-3.1-24B-Instruct-2503"),
    ("phi4-multimodal-instruct", "multimodals/Phi-4-multimodal-instruct"),
    pytest.param("phi4-multimodal-instruct-fp4",
                 "multimodals/Phi-4-multimodal-instruct-FP4",
                 marks=skip_pre_blackwell),
    pytest.param("phi4-multimodal-instruct-fp8",
                 "multimodals/Phi-4-multimodal-instruct-FP8",
                 marks=skip_pre_hopper),
])
def test_ptp_quickstart_multimodal_2gpu(llm_root, llm_venv, model_name,
                                        model_path):
    example_root = Path(os.path.join(llm_root, "examples", "llm-api"))
    test_data_root = Path(
        os.path.join(llm_models_root(), "multimodals", "test_data"))

    print(f"Accuracy test {model_name} image mode with example inputs.")

    # Define accuracy inputs for image modality
    accuracy_inputs = {
        "image": {
            "prompt": [
                "Describe the object and the weather condition in the image.",
                "Describe the traffic condition on the road in the image.",
            ],
            "media": [
                str(test_data_root / "inpaint.png"),
                str(test_data_root / "61.jpg"),
            ],
        }
    }

    # Define expected keywords for each model
    expected_keywords = {
        "gemma-3-27b-it": {
            "image": [
                ["half", "dome", "yosemite", "landmark", "rounded"],
                ["flowing", "traffic", "vehicles", "road", "Changi"],
            ],
        },
        "mistral-small-3.1-24b-instruct": {
            "image": [
                ["scenic", "rock", "landscape", "monolith", "formation"],
                [
                    "multi-lane", "highway", "moderate", "traffic", "flow",
                    "vehicles", "congestion"
                ],
            ],
        },
        "phi4-multimodal-instruct": {
            "image": [
                ["object", "mountain", "weather", "clear", "clouds"],
                ["traffic", "road", "vehicles", "cars", "bus"],
            ],
        },
        "phi4-multimodal-instruct-fp4": {
            "image": [
                ["object", "mountain", "weather", "clear", "clouds"],
                ["traffic", "road", "vehicles", "cars", "bus"],
            ],
        },
        "phi4-multimodal-instruct-fp8": {
            "image": [
                ["object", "mountain", "weather", "clear", "clouds"],
                ["traffic", "road", "vehicles", "cars", "bus"],
            ],
        },
    }

    # Build command for image modality
    cmd = [
        str(example_root / "quickstart_multimodal.py"),
        "--model_dir",
        f"{llm_models_root()}/{model_path}",
        "--modality",
        "image",
        "--prompt",
        *accuracy_inputs["image"]["prompt"],
        "--media",
        *accuracy_inputs["image"]["media"],
        "--tp_size",
        "2",
    ]

    # Add model-specific configurations
    if model_name == "gemma-3-27b-it":
        # Gemma3 VLM needs a custom mask which is only supported by flashinfer backend currently.
        # Custom mask involves bidirectional masking of image tokens in context phase. To get this
        # correct, chunked prefill and kv cache reuse need to be turned off.
        cmd.append("--image_format=pil")
        cmd.append("--attention_backend=FLASHINFER")
        cmd.append("--disable_kv_cache_reuse")
        cmd.append("--kv_cache_fraction=0.5")
        cmd.append("--max_seq_len=1024")
    elif model_name.startswith("phi4-multimodal-instruct"):
        # Set max_seq_len to 4096 to use short rope factor.
        cmd.append("--max_seq_len=4096")
        cmd.append("--load_lora")
        cmd.append("--auto_model_name")
        cmd.append("Phi4MMForCausalLM")
    elif model_name == "mistral-small-3.1-24b-instruct":
        # TODO: remove this once kv cache reuse is supported for Mistral
        cmd.append("--disable_kv_cache_reuse")

    output = llm_venv.run_cmd(cmd, caller=check_output)

    # Set match ratio based on model
    match_ratio = 4.0 / 5
    if model_name.startswith("phi4-multimodal-instruct"):
        match_ratio = 0.6

    # Check output accuracy
    parsed_outputs = parse_output(output)
    for prompt_output, prompt_keywords in zip(
            parsed_outputs, expected_keywords[model_name]["image"]):
        matches = [
            keyword in prompt_output.lower() for keyword in prompt_keywords
        ]
        obs_match_ratio = 1. * sum(matches) / len(matches)
        assert obs_match_ratio >= match_ratio, f"Incorrect output!\nGenerated \"{prompt_output}\"\nExpected keywords \"{prompt_keywords}\"\n Matched keywords: {matches}\n Observed match ratio {obs_match_ratio} below threshold {match_ratio}\n\nParsed output for all prompts: {parsed_outputs}"

    print("All answers are correct!")


@pytest.mark.skip_less_device_memory(80000)
@pytest.mark.parametrize("model_name,model_path", [
    ("mistral-small-3.1-24b-instruct", "Mistral-Small-3.1-24B-Instruct-2503"),
    ("phi4-multimodal-instruct", "multimodals/Phi-4-multimodal-instruct"),
    pytest.param("phi4-multimodal-instruct-fp4",
                 "multimodals/Phi-4-multimodal-instruct-FP4",
                 marks=skip_pre_blackwell),
    pytest.param("phi4-multimodal-instruct-fp8",
                 "multimodals/Phi-4-multimodal-instruct-FP8",
                 marks=skip_pre_hopper),
    pytest.param(
        "gemma-3-27b-it", "gemma/gemma-3-27b-it", marks=skip_post_blackwell),
])
def test_ptp_quickstart_multimodal_multiturn(llm_root, llm_venv, model_name,
                                             model_path):
    example_root = Path(os.path.join(llm_root, "examples", "llm-api"))
    test_data_root = Path(
        os.path.join(llm_models_root(), "multimodals", "test_data"))

    print(f"Accuracy test {model_name} image mode with example inputs.")

    # Define accuracy inputs for image modality
    accuracy_inputs = {
        "image": {
            "prompt": [
                "Describe what you see in this image.",
                "How would you describe the atmosphere of this scene?",
            ],
            "media": [
                str(test_data_root / "inpaint.png"),
            ],
        }
    }

    # Define expected keywords for each model
    expected_keywords = {
        "gemma-3-27b-it": {
            "image": [
                ["description", "image", "half", "dome", "park"],
                ["atmosphere", "peaceful", "majestic", "scene", "sky"],
            ],
        },
        "mistral-small-3.1-24b-instruct": {
            "image": [
                [
                    "depicts", "scenic", "landscape", "rock", "formation",
                    "background"
                ],
                ["atmosphere", "serene", "majestic", "clear", "sky", "trees"],
            ],
        },
        "phi4-multimodal-instruct": {
            "image": [
                ["depicts", "landscape", "mountain", "half", "dome"],
                ["atmosphere", "serene", "sense", "scene", "majestic"],
            ],
        },
        "phi4-multimodal-instruct-fp4": {
            "image": [
                ["depicts", "landscape", "mountain", "half", "dome"],
                ["atmosphere", "serene", "sense", "scene", "majestic"],
            ],
        },
        "phi4-multimodal-instruct-fp8": {
            "image": [
                ["depicts", "landscape", "mountain", "half", "dome"],
                ["atmosphere", "serene", "sense", "scene", "majestic"],
            ],
        },
    }
    # Build command for image modality
    cmd = [
        str(example_root / "quickstart_multimodal.py"),
        "--model_dir",
        f"{llm_models_root()}/{model_path}",
        "--modality",
        "image",
        "--multiturn",
        "--prompt",
        *accuracy_inputs["image"]["prompt"],
        "--media",
        *accuracy_inputs["image"]["media"],
    ]

    # Add model-specific configurations
    if model_name == "gemma-3-27b-it":
        # Gemma3 VLM needs a custom mask which is only supported by flashinfer backend currently.
        # Custom mask involves bidirectional masking of image tokens in context phase. To get this
        # correct, chunked prefill and kv cache reuse need to be turned off.
        cmd.append("--image_format=pil")
        cmd.append("--attention_backend=FLASHINFER")
        cmd.append("--disable_kv_cache_reuse")
        cmd.append("--kv_cache_fraction=0.5")
        cmd.append("--max_seq_len=1024")

    elif model_name.startswith("phi4-multimodal-instruct"):
        # Set max_seq_len to 4096 to use short rope factor.
        cmd.append("--max_seq_len=4096")
        cmd.append("--load_lora")
        cmd.append("--auto_model_name")
        cmd.append("Phi4MMForCausalLM")

    elif model_name == "mistral-small-3.1-24b-instruct":
        # TODO: remove this once kv cache reuse is supported for Mistral
        cmd.append("--disable_kv_cache_reuse")

    output = llm_venv.run_cmd(cmd, caller=check_output)
    print("output:", output)
    # Set match ratio based on model
    match_ratio = 4.0 / 5
    if model_name.startswith("Phi-4-multimodal-instruct"):
        match_ratio = 0.6

    # Check output accuracy
    parsed_outputs = parse_output(output)
    for prompt_output, prompt_keywords in zip(
            parsed_outputs, expected_keywords[model_name]["image"]):
        matches = [
            keyword in prompt_output.lower() for keyword in prompt_keywords
        ]
        obs_match_ratio = 1. * sum(matches) / len(matches)
        print("prompt_output:", prompt_output)
        print("prompt_keywords:", prompt_keywords)
        print("matches:", matches)
        print("obs_match_ratio:", obs_match_ratio)
        assert obs_match_ratio >= match_ratio, f"Incorrect output!\nGenerated \"{prompt_output}\"\nExpected keywords \"{prompt_keywords}\"\n Matched keywords: {matches}\n Observed match ratio {obs_match_ratio} below threshold {match_ratio}\n\nParsed output for all prompts: {parsed_outputs}"

    print("All answers are correct!")


@pytest.mark.parametrize("model_name,model_path", [
    ("BertForSequenceClassification", "bert/bert-base-uncased-yelp-polarity"),
])
@pytest.mark.parametrize("backend", ["VANILLA", "TRTLLM"])
def test_ptp_quickstart_bert(llm_root, llm_venv, model_name, model_path,
                             backend):
    print(f"Testing {model_name} with {backend} backend.")
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    from tensorrt_llm import LLM, SamplingParams
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
            attn_backend=backend,
            disable_overlap_scheduler=True,
    ) as llm:

        outputs = llm.generate(prompts, sampling_params=sampling_param)
    # Print the outputs.
    tllm_logits = []
    for output in outputs:
        prompt = output.prompt
        tllm_logit = output.context_logits.cpu()[0, :]
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
    example_root = Path(os.path.join(llm_root, "examples", "llm-api"))
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
        str(example_root / "run_majority_vote_aime24.py"),
        "--model_dir",
        f"{llm_models_root()}/{model_path}",
        f"--jsonl_file={input_file}",
        "--threshold=0.5",
    ])


@pytest.mark.skip_less_device_memory(80000)
@pytest.mark.skip_less_device(4)
@pytest.mark.parametrize("model_path", [
    pytest.param('llama-3.3-models/Llama-3.3-70B-Instruct',
                 marks=(skip_pre_hopper, pytest.mark.timeout(5400))),
    pytest.param('llama4-models/Llama-4-Maverick-17B-128E-Instruct',
                 marks=skip_pre_hopper),
])
def test_ptp_quickstart_advanced_llama_multi_nodes(llm_root, llm_venv,
                                                   model_path):
    print(f"Testing {model_path}.")
    tp_size, pp_size = 16, 1
    if "Llama-4" in model_path:
        tp_size, pp_size = 8, 2

    example_root = Path(os.path.join(llm_root, "examples", "llm-api"))
    run_cmd = [
        "trtllm-llmapi-launch",
        "python3",
        str(example_root / "quickstart_advanced.py"),
        f"--model_dir={llm_models_root()}/{model_path}",
        "--moe_ep_size=8",
        f"--tp_size={tp_size}",
        f"--pp_size={pp_size}",
        "--use_cuda_graph",
        f"--kv_cache_fraction={_MEM_FRACTION_50}",
        "--max_batch_size=32",
        "--max_num_tokens=2048",
        "--disable_kv_cache_reuse",
    ]
    check_call(" ".join(run_cmd), shell=True, env=llm_venv._new_env)


@pytest.mark.timeout(7200)
@pytest.mark.skip_less_device_memory(80000)
@pytest.mark.skip_less_device(4)
@pytest.mark.parametrize("eval_task", ["mmlu"])
@pytest.mark.parametrize("tp_size,pp_size,ep_size", [(16, 1, 8), (8, 2, 8)],
                         ids=["tp16", "tp8pp2"])
@pytest.mark.parametrize("model_path", [
    pytest.param('llama-3.3-models/Llama-3.3-70B-Instruct',
                 marks=skip_pre_hopper),
    pytest.param('llama4-models/Llama-4-Maverick-17B-128E-Instruct',
                 marks=skip_pre_hopper),
    pytest.param('llama4-models/nvidia/Llama-4-Maverick-17B-128E-Instruct-FP8',
                 marks=skip_pre_hopper),
    pytest.param('Qwen3/Qwen3-235B-A22B', marks=skip_pre_hopper),
    pytest.param('Qwen3/saved_models_Qwen3-235B-A22B_nvfp4_hf',
                 marks=skip_pre_blackwell),
    pytest.param('DeepSeek-R1/DeepSeek-R1-0528-FP4', marks=skip_pre_blackwell),
    pytest.param('Kimi-K2-Instruct',
                 marks=(skip_pre_hopper, skip_post_blackwell)),
    pytest.param('nemotron-nas/Llama-3_1-Nemotron-Ultra-253B-v1',
                 marks=skip_pre_hopper),
])
def test_multi_nodes_eval(llm_venv, model_path, tp_size, pp_size, ep_size,
                          eval_task, mmlu_dataset_root):
    if "Llama-4" in model_path and tp_size == 16:
        pytest.skip("Llama-4 with tp16 is not supported")

    mmlu_threshold = 81.5
    model_dir = f"{llm_models_root()}/{model_path}"
    run_cmd = [
        "trtllm-llmapi-launch",
        "trtllm-eval",
        f"--model={model_dir}",
        f"--ep_size={ep_size}",
        f"--tp_size={tp_size}",
        f"--pp_size={pp_size}",
        f"--kv_cache_free_gpu_memory_fraction={_MEM_FRACTION_80}",
        "--max_batch_size=32",
        "--backend=pytorch",
    ]

    if "Kimi" in model_path:
        run_cmd.append("--trust_remote_code")
    else:
        run_cmd.append(f"--tokenizer={model_dir}")

    run_cmd.extend([eval_task, f"--dataset_path={mmlu_dataset_root}"])

    llm_venv._new_env["TRT_LLM_DISABLE_LOAD_WEIGHTS_IN_PARALLEL"] = "1"
    output = check_output(" ".join(run_cmd), shell=True, env=llm_venv._new_env)

    if os.environ.get("SLURM_PROCID", '0') == '0':
        mmlu_accuracy = get_mmlu_accuracy(output)
        assert mmlu_accuracy > mmlu_threshold, f"MMLU accuracy {mmlu_accuracy} is less than threshold {mmlu_threshold}"


@pytest.mark.skip_less_device_memory(80000)
@pytest.mark.parametrize("return_generation_logits", [True, False])
@pytest.mark.parametrize("model_path", [
    ("llama-3.1-model/Llama-3.1-8B-Instruct"),
    pytest.param("llama-3.3-models/Llama-3.3-70B-Instruct",
                 marks=pytest.mark.skip_less_device(8)),
])
def test_llmapi_generation_logits(llm_venv, model_path,
                                  return_generation_logits):
    """
    RCCA: https://nvbugspro.nvidia.com/bug/5501805
    """

    import asyncio

    from tensorrt_llm import LLM, SamplingParams

    seq_len, max_tokens = 131072, 100000
    if return_generation_logits:
        # use short seq_len and max_tokens for testing when return_generation_logits is True
        seq_len, max_tokens = 1024, 1000
    tp_size = 8 if "70B" in model_path else 1
    # Model parameters
    params = {
        "cuda_graph_config": {
            "batch_sizes": [512]
        },
        "enable_chunked_prefill": True,
        "guided_decoding_backend": "xgrammar",
        "kv_cache_config": {
            "cross_kv_cache_fraction": None,
            "enable_block_reuse": False,
            "free_gpu_memory_fraction": 0.9,
            "max_attention_window": None
        },
        "max_seq_len": seq_len,
        "tensor_parallel_size": tp_size,
    }

    # Sampling parameters
    sampling_params = SamplingParams(
        max_tokens=max_tokens,
        return_context_logits=False,
        return_generation_logits=return_generation_logits,
    )

    # Test prompt (token IDs)
    prompt = [
        128000, 128006, 9125, 128007, 271, 38766, 1303, 33025, 2696, 25, 6790,
        220, 2366, 18, 198, 15724, 2696, 25, 220, 2545, 17907, 220, 2366, 20,
        271, 67, 10319, 7422, 389, 128009, 128006, 882, 128007, 271, 3923, 374,
        701, 836, 30, 128009, 128006, 78191, 128007, 271
    ]

    async def async_generation_test():
        """Async generation test function"""
        model_path_full = f"{llm_models_root()}/{model_path}"
        llm = LLM(**params, model=model_path_full, tokenizer=model_path_full)

        try:
            outputs = []
            async for output in llm.generate_async(
                    prompt,
                    sampling_params,
                    streaming=True,
            ):
                outputs.append(output)
                print(f"Generated: {output}")

            # Verify that we got some output
            assert len(outputs) > 0, "No output generated"
            print(f"Successfully generated {len(outputs)} streaming outputs")

        finally:
            llm.shutdown()

    # Run the async test
    loop = asyncio.get_event_loop()
    loop.run_until_complete(async_generation_test())
