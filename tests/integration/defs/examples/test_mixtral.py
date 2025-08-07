# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import csv
import os

import pytest
from defs.common import (convert_weights, generate_summary_cmd, quantize_data,
                         venv_check_call, venv_mpi_check_call)
from defs.conftest import (get_sm_version, llm_models_root, skip_post_blackwell,
                           skip_pre_ada)
from defs.trt_test_alternative import check_call

# skip trt flow cases on post-Blackwell-Ultra
if get_sm_version() >= 103:
    pytest.skip(
        "TRT workflow tests are not supported on post Blackwell-Ultra architecture",
        allow_module_level=True)


@skip_post_blackwell
@pytest.mark.parametrize("model_name", ['mixtral-8x7b-v0.1-AWQ'])
def test_llm_mixtral_int4_awq_1gpu_summary(llama_example_root,
                                           llm_datasets_root, model_name,
                                           llm_rouge_root, llm_venv, cmodel_dir,
                                           engine_dir,
                                           qcache_dir_without_install_package):
    models_root = llm_models_root()
    model_dir = os.path.join(models_root, model_name)
    ckpt_dir = os.path.join(cmodel_dir, model_name)

    print("Convert checkpoint...")
    convert_cmd = [
        f"{llama_example_root}/convert_checkpoint.py",
        "--model_dir",
        model_dir,
        "--output_dir",
        ckpt_dir,
    ]
    venv_check_call(llm_venv, convert_cmd)

    print("Build engines...")
    build_cmd = [
        "trtllm-build",
        f"--checkpoint_dir={ckpt_dir}",
        f"--output_dir={engine_dir}",
    ]
    check_call(" ".join(build_cmd), shell=True, env=llm_venv._new_env)

    print("Run inference")
    summary_cmd = generate_summary_cmd(llama_example_root,
                                       hf_model_dir=model_dir,
                                       data_type="fp16",
                                       tensorrt_llm_rouge1_threshold=19.5,
                                       engine_dir=engine_dir,
                                       dataset_dir=llm_datasets_root,
                                       rouge_dir=llm_rouge_root)

    venv_check_call(llm_venv, summary_cmd)


@pytest.mark.skip_less_device(8)
@pytest.mark.skip_less_device_memory(80000)
@pytest.mark.parametrize("test_type", ['build', 'infer'])
@pytest.mark.parametrize(
    "moe_tp_size", [1, 4, 8],
    ids=['expert_parallel', 'mixed_parallel', 'tensor_parallel'])
@pytest.mark.parametrize("moe_renorm_mode", [0, 1],
                         ids=['no_renormalize', 'renormalize'])
@pytest.mark.parametrize("mode", [0, 1], ids=['plugin', 'ootb_except_mha'])
@pytest.mark.parametrize("llm_mixtral_model_root",
                         ['Mixtral-8x7B-v0.1', 'Mixtral-8x22B-v0.1'],
                         indirect=True)
def test_llm_mixtral_2nodes_8gpus(llama_example_root, llm_mixtral_model_root,
                                  llm_datasets_root, llm_rouge_root, llm_venv,
                                  cmodel_dir, engine_dir, moe_tp_size,
                                  moe_renorm_mode, mode, test_type):
    "Run test on 2x8 gpus with moe_renorm_mode"
    data_type = "float16"
    tp_size, pp_size = 8, 2
    world_size = tp_size * pp_size
    model_name = os.path.basename(llm_mixtral_model_root)
    engine_dir = os.path.join(llama_example_root, "engines", model_name,
                              data_type, f"{world_size}-gpu",
                              f"tp{tp_size}pp{pp_size}moe{moe_tp_size}",
                              f"renorm_{moe_renorm_mode}", f"mode_{mode}")

    if test_type == "build":
        model_dir = convert_weights(llm_venv=llm_venv,
                                    example_root=llama_example_root,
                                    cmodel_dir=cmodel_dir,
                                    model="mixtral",
                                    model_path=llm_mixtral_model_root,
                                    tp_size=tp_size,
                                    moe_tp_size=moe_tp_size,
                                    moe_ep_size=tp_size // moe_tp_size,
                                    pp_size=pp_size,
                                    data_type=data_type,
                                    moe_renorm_mode=moe_renorm_mode)

        gemm_plugin = "disable" if mode == "ootb-except-mha" else data_type
        moe_plugin = "disable" if mode == "ootb-except-mha" else data_type

        print("Build engines...")
        build_cmd = [
            "trtllm-build",
            f"--checkpoint_dir={model_dir}",
            f"--output_dir={engine_dir}",
            f"--gemm_plugin={gemm_plugin}",
            f"--moe_plugin={moe_plugin}",
            f"--workers={8}",
            "--max_input_len=1024",
            "--max_batch_size=1",
            "--context_fmha=enable",
            "--max_beam_width=4",
        ]

        check_call(" ".join(build_cmd), shell=True, env=llm_venv._new_env)

    if test_type == "infer":
        print("Run summarize...")
        summary_cmd = generate_summary_cmd(llama_example_root,
                                           hf_model_dir=llm_mixtral_model_root,
                                           data_type="fp16",
                                           num_beams=4,
                                           engine_dir=engine_dir,
                                           tensorrt_llm_rouge1_threshold=23,
                                           dataset_dir=llm_datasets_root,
                                           rouge_dir=llm_rouge_root)

        venv_check_call(llm_venv, summary_cmd)


@pytest.mark.skip_less_device(4)
@pytest.mark.skip_less_device_memory(45000)
@pytest.mark.parametrize("llm_lora_model_root", ["chinese-mixtral-lora"],
                         indirect=True)
@pytest.mark.parametrize("llm_mixtral_model_root", ["Mixtral-8x7B-v0.1"],
                         indirect=True)
def test_llm_mixtral_moe_plugin_lora_4gpus(
    llama_example_root,
    llm_mixtral_model_root,
    llm_venv,
    cmodel_dir,
    engine_dir,
    llm_lora_model_root,
):
    "run Mixtral MoE lora test on 4 gpu."
    print("Build engines...")
    dtype = 'float16'
    model_name = os.path.basename(llm_mixtral_model_root)
    ckpt_dir = convert_weights(llm_venv=llm_venv,
                               example_root=llama_example_root,
                               cmodel_dir=cmodel_dir,
                               model=model_name,
                               tp_size=4,
                               pp_size=1,
                               model_path=llm_mixtral_model_root,
                               data_type=dtype)

    print("Build engines...")
    build_cmd = [
        "trtllm-build",
        f"--checkpoint_dir={ckpt_dir}",
        f"--output_dir={engine_dir}",
        "--lora_plugin=auto",
        "--moe_plugin=auto",
        f"--lora_dir={llm_lora_model_root}",
        "--worker=4",
        "--max_batch_size=8",
    ]
    check_call(" ".join(build_cmd), shell=True, env=llm_venv._new_env)

    ref_1 = [
        1, 28705, 29242, 30731, 31182, 235, 158, 142, 234, 182, 152, 28924,
        29926, 28971, 29242, 28988
    ]
    ref_2 = [
        1, 315, 2016, 285, 4284, 526, 5680, 28723, 28705, 28740, 28723, 661
    ]

    input_text = "我爱吃蛋糕"
    print("Run inference with lora id 0...")
    run_cmd = [
        f"{llama_example_root}/../../../run.py",
        "--max_output_len=5",
        f"--input_text={input_text}",
        "--lora_task_uids=0",
        f"--tokenizer_dir={llm_lora_model_root}",
        f"--engine_dir={engine_dir}",
        f"--output_csv={llm_venv.get_working_directory()}/use_lora.csv",
        "--use_py_session",
    ]
    venv_mpi_check_call(llm_venv, ["mpirun", "-n", "4", "--allow-run-as-root"],
                        run_cmd)

    with open(f"{llm_venv.get_working_directory()}/use_lora.csv") as f:
        predict = csv.reader(f)
        predict = next(predict)
    predict = [int(p) for p in predict]
    assert ref_1 == predict

    print("Run inference with lora id -1...")
    input_text = "I love french quiche"
    run_cmd = [
        f"{llama_example_root}/../../../run.py",
        "--max_output_len=5",
        f"--input_text={input_text}",
        "--lora_task_uids=-1",
        f"--tokenizer_dir={llm_lora_model_root}",
        f"--engine_dir={engine_dir}",
        f"--output_csv={llm_venv.get_working_directory()}/no_lora.csv",
        "--use_py_session",
    ]
    venv_mpi_check_call(llm_venv, ["mpirun", "-n", "4", "--allow-run-as-root"],
                        run_cmd)

    with open(f"{llm_venv.get_working_directory()}/no_lora.csv") as f:
        predict = csv.reader(f)
        predict = next(predict)
    predict = [int(p) for p in predict]
    assert ref_2 == predict


@skip_pre_ada
@pytest.mark.skip_less_device(4)
@pytest.mark.skip_less_device_memory(80000)
@pytest.mark.parametrize("llm_lora_model_root", ["chinese-mixtral-lora"],
                         indirect=True)
@pytest.mark.parametrize("llm_mixtral_model_root", ["Mixtral-8x7B-v0.1"],
                         indirect=True)
def test_llm_mixtral_moe_plugin_fp8_lora_4gpus(
    llama_example_root,
    llm_mixtral_model_root,
    llm_venv,
    qcache_dir,
    engine_dir,
    llm_lora_model_root,
):
    "run Mixtral MoE lora test on 4 gpu."
    print("Build engines...")
    dtype = 'float16'
    tp_size = 4
    pp_size = 1
    workers = tp_size * pp_size

    print("Quantizing engine...")
    model_dir = quantize_data(llm_venv,
                              llama_example_root,
                              model_dir=llm_mixtral_model_root,
                              dtype=dtype,
                              qformat="fp8",
                              kv_cache_dtype="fp8",
                              quantize_dir=qcache_dir,
                              tp_size=tp_size,
                              pp_size=pp_size)

    print("Build engines...")
    build_cmd = [
        "trtllm-build",
        f"--checkpoint_dir={model_dir}",
        f"--workers={workers}",
        "--max_batch_size=8",
        f"--output_dir={engine_dir}",
        f"--lora_dir={llm_lora_model_root}",
        f"--lora_plugin={dtype}",
        f"--moe_plugin={dtype}",
    ]

    check_call(" ".join(build_cmd), shell=True, env=llm_venv._new_env)

    ref_1 = [
        1, 28705, 29242, 30731, 31182, 235, 158, 142, 234, 182, 152, 28924,
        29926, 28971, 29242, 28988
    ]

    input_text = "我爱吃蛋糕"
    print("Run inference with lora id 0...")
    run_cmd = [
        f"{llama_example_root}/../../../run.py",
        "--max_output_len=5",
        f"--input_text={input_text}",
        "--lora_task_uids=0",
        f"--tokenizer_dir={llm_lora_model_root}",
        f"--engine_dir={engine_dir}",
        f"--output_csv={llm_venv.get_working_directory()}/use_lora.csv",
        "--use_py_session",
    ]
    venv_mpi_check_call(llm_venv,
                        ["mpirun", "-n", f"{workers}", "--allow-run-as-root"],
                        run_cmd)

    with open(f"{llm_venv.get_working_directory()}/use_lora.csv") as f:
        predict = csv.reader(f)
        predict = next(predict)
    predict = [int(p) for p in predict]
    assert ref_1 == predict

    ref_2 = [
        1, 315, 2016, 285, 4284, 526, 5680, 28723, 315, 2016, 272, 1439, 469,
        28725
    ]
    print("Run inference with lora id -1...")
    input_text = "I love french quiche. I"
    run_cmd = [
        f"{llama_example_root}/../../../run.py",
        "--max_output_len=5",
        f"--input_text={input_text}",
        "--lora_task_uids=-1",
        f"--tokenizer_dir={llm_lora_model_root}",
        f"--engine_dir={engine_dir}",
        f"--output_csv={llm_venv.get_working_directory()}/no_lora.csv",
        "--use_py_session",
    ]
    venv_mpi_check_call(llm_venv,
                        ["mpirun", "-n", f"{workers}", "--allow-run-as-root"],
                        run_cmd)

    with open(f"{llm_venv.get_working_directory()}/no_lora.csv") as f:
        predict = csv.reader(f)
        predict = next(predict)
    predict = [int(p) for p in predict]
    assert ref_2 == predict
