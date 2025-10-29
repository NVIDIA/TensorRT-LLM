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

import os
from pathlib import Path

import pytest
from defs.common import (convert_weights, generate_summary_cmd, quantize_data,
                         venv_check_call, venv_mpi_check_call)
from defs.conftest import get_sm_version, skip_fp8_pre_ada, skip_post_blackwell
from defs.trt_test_alternative import check_call

# skip trt flow cases on post-Blackwell-Ultra
if get_sm_version() >= 103:
    pytest.skip(
        "TRT workflow tests are not supported on post Blackwell-Ultra architecture",
        allow_module_level=True)


@skip_post_blackwell
@pytest.mark.parametrize("gemm_plugin", [True, False],
                         ids=["enable_gemm_plugin", "disable_gemm_plugin"])
@pytest.mark.parametrize("gpt_attention_plugin", [True, False],
                         ids=["enable_attn_plugin", "disable_attn_plugin"])
@pytest.mark.parametrize("dtype", ["bfloat16", "float16"])
@pytest.mark.parametrize("qformat", [None, "fp8", "int8_sq", "int4_awq"],
                         ids=["disable_quant", "fp8", "int8_sq", "int4_awq"])
@pytest.mark.parametrize("paged_cache", [True, False],
                         ids=["use_paged_cache", "no_paged_cache"])
@pytest.mark.parametrize("recurrentgemma_model_root", [
    "recurrentgemma-2b",
    "recurrentgemma-2b-flax",
],
                         indirect=True)
@pytest.mark.parametrize("use_py_session", [False, True],
                         ids=["use_cpp_session", "use_py_session"])
def test_llm_recurrentgemma_1gpu(recurrentgemma_example_root,
                                 recurrentgemma_model_root, llm_datasets_root,
                                 llm_rouge_root, llm_venv, gemm_plugin,
                                 gpt_attention_plugin, paged_cache, dtype,
                                 qformat, use_py_session, cmodel_dir,
                                 engine_dir, qcache_dir):
    "Build & Run recurrentgemma model with one gpu"
    if not gpt_attention_plugin and paged_cache:
        pytest.skip(
            "Skip the test with paged_kv_cache=True and gpt_attention_plugin=disable."
        )
    skip_fp8_pre_ada(use_fp8=qformat == "fp8")

    ckpt_type = "jax" if "flax" in str(recurrentgemma_model_root) else "hf"
    if qformat is not None and ckpt_type == "jax":
        pytest.skip("PTQ is not supported for jax checkpoints")

    # Fix the issue that the typeguard==2.13.3 is used by recurrentgemma.
    llm_venv.run_cmd(["-m", "pip", "install", "typeguard==2.13.3", "--upgrade"])

    print("Build engines...")
    if ckpt_type == "jax":
        tokenizer_root = os.path.join(
            os.path.dirname(recurrentgemma_model_root), "tokenizer.model")
        model_name = Path(recurrentgemma_model_root).parents[1].name
    else:
        tokenizer_root = recurrentgemma_model_root
        model_name = Path(recurrentgemma_model_root).parents[0].name

    if qformat is not None:
        kv_cache_dtype = "fp8" if qformat == "fp8" else "int8"
        model_dir = quantize_data(
            llm_venv,
            recurrentgemma_example_root,
            model_dir=recurrentgemma_model_root,
            calib_dataset=f"{llm_datasets_root}/cnn_dailymail",
            dtype=dtype,
            qformat=qformat,
            quantize_dir=qcache_dir,
            calib_size=512,
            kv_cache_dtype=kv_cache_dtype)
    else:
        model_dir = convert_weights(llm_venv=llm_venv,
                                    example_root=recurrentgemma_example_root,
                                    cmodel_dir=cmodel_dir,
                                    model=model_name,
                                    model_path=recurrentgemma_model_root,
                                    data_type=dtype,
                                    ckpt_type=ckpt_type)
    build_cmd = [
        "trtllm-build",
        f"--checkpoint_dir={model_dir}",
        f"--output_dir={engine_dir}",
        "--mamba_conv1d_plugin=auto",
        "--max_batch_size=8",
        "--max_seq_len=2048",
    ]
    if not gpt_attention_plugin:
        build_cmd.append(f"--gpt_attention_plugin=disable")
        build_cmd.append(f"--remove_input_padding=disable")
    if gemm_plugin:
        build_cmd.append("--gemm_plugin=auto")
    if paged_cache:
        build_cmd.append("--paged_kv_cache=enable")
        build_cmd.append("--paged_state=enable")
    else:
        build_cmd.append("--paged_kv_cache=disable")
        build_cmd.append("--paged_state=disable")

    check_call(" ".join(build_cmd), shell=True, env=llm_venv._new_env)

    print(f"Run {model_name}...")
    tensorrt_llm_rouge1_threshold = 19.0 if qformat is not None else 20.0
    if ckpt_type == "jax":
        summary_cmd = generate_summary_cmd(
            recurrentgemma_example_root,
            hf_model_dir=recurrentgemma_model_root,
            vocab_file=tokenizer_root,
            data_type=dtype,
            engine_dir=engine_dir,
            batch_size=8,
            max_attention_window_size=2048,
            tensorrt_llm_rouge1_threshold=tensorrt_llm_rouge1_threshold,
            dataset_dir=llm_datasets_root,
            rouge_dir=llm_rouge_root)
    else:
        summary_cmd = generate_summary_cmd(
            recurrentgemma_example_root,
            hf_model_dir=recurrentgemma_model_root,
            tokenizer_dir=tokenizer_root,
            data_type=dtype,
            engine_dir=engine_dir,
            batch_size=8,
            max_attention_window_size=2048,
            tensorrt_llm_rouge1_threshold=tensorrt_llm_rouge1_threshold,
            dataset_dir=llm_datasets_root,
            rouge_dir=llm_rouge_root)
    if use_py_session:
        summary_cmd.extend(["--use_py_session"])

    venv_check_call(llm_venv, summary_cmd)


@pytest.mark.parametrize("recurrentgemma_model_root", [
    "recurrentgemma-2b",
    "recurrentgemma-2b-flax",
],
                         indirect=True)
def test_llm_recurrentgemma_2gpu(recurrentgemma_example_root,
                                 recurrentgemma_model_root, llm_datasets_root,
                                 llm_rouge_root, llm_venv, cmodel_dir,
                                 engine_dir):
    "Build & Run recurrentgemma model with two gpu"

    print("Build engines...")

    ckpt_type = "jax" if "flax" in str(recurrentgemma_model_root) else "hf"

    if ckpt_type == "jax":
        tokenizer_root = os.path.join(
            os.path.dirname(recurrentgemma_model_root), "tokenizer.model")
        model_name = Path(recurrentgemma_model_root).parents[1].name
    else:
        tokenizer_root = recurrentgemma_model_root
        model_name = Path(recurrentgemma_model_root).parents[0].name

    model_dir = convert_weights(llm_venv=llm_venv,
                                example_root=recurrentgemma_example_root,
                                cmodel_dir=cmodel_dir,
                                model=model_name,
                                model_path=recurrentgemma_model_root,
                                data_type='float16',
                                ckpt_type=ckpt_type,
                                tp_size=2)
    build_cmd = [
        "trtllm-build",
        f"--checkpoint_dir={model_dir}",
        f"--output_dir={engine_dir}",
        "--gemm_plugin=auto",
        "--max_batch_size=8",
        "--paged_kv_cache=enable",
        "--paged_state=enable",
    ]

    check_call(" ".join(build_cmd), shell=True, env=llm_venv._new_env)

    print(f"Run {model_name}...")
    if ckpt_type == "jax":
        summary_cmd = generate_summary_cmd(
            recurrentgemma_example_root,
            hf_model_dir=recurrentgemma_model_root,
            vocab_file=tokenizer_root,
            data_type='float16',
            engine_dir=engine_dir,
            batch_size=8,
            max_attention_window_size=2048,
            tensorrt_llm_rouge1_threshold="20.0",
            dataset_dir=llm_datasets_root,
            rouge_dir=llm_rouge_root)
    else:
        summary_cmd = generate_summary_cmd(
            recurrentgemma_example_root,
            hf_model_dir=recurrentgemma_model_root,
            tokenizer_dir=tokenizer_root,
            data_type='float16',
            engine_dir=engine_dir,
            batch_size=8,
            max_attention_window_size=2048,
            tensorrt_llm_rouge1_threshold="20.0",
            dataset_dir=llm_datasets_root,
            rouge_dir=llm_rouge_root)

    venv_mpi_check_call(llm_venv, ["mpirun", "-n", "2", "--allow-run-as-root"],
                        summary_cmd)
