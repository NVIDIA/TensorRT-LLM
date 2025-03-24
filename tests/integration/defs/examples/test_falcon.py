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

import pytest
from defs.common import (convert_weights, generate_summary_cmd, venv_check_call,
                         venv_mpi_check_call)
from defs.conftest import skip_fp8_pre_ada, skip_pre_ada
from defs.trt_test_alternative import check_call


@pytest.mark.parametrize("num_beams", [1, 2, 4],
                         ids=lambda num_beams: f'nb:{num_beams}')
@pytest.mark.parametrize(
    "use_gpt_attention_plugin", [True, False],
    ids=["enable_attention_plugin", "disable_attention_plugin"])
@pytest.mark.parametrize("context_fmha_type",
                         ["enabled", "enabled_with_fp32_acc", "disabled"])
@pytest.mark.parametrize("dtype", ['float16', 'bfloat16'])
@pytest.mark.parametrize("use_py_session", [False, True],
                         ids=["use_cpp_session", "use_py_session"])
def test_llm_falcon_rw_1b_1node_1gpus(
        falcon_example_root, llm_falcon_rw_1b_model_root, llm_datasets_root,
        llm_rouge_root, llm_venv, cmodel_dir, engine_dir,
        use_gpt_attention_plugin, context_fmha_type, dtype, use_py_session,
        num_beams):
    # Build & Run falcon-rw-1b with one gpu
    print("Converting checkpoint...")
    model_name = os.path.basename(llm_falcon_rw_1b_model_root)
    ckpt_dir = convert_weights(llm_venv=llm_venv,
                               example_root=falcon_example_root,
                               cmodel_dir=cmodel_dir,
                               model=model_name,
                               model_path=llm_falcon_rw_1b_model_root,
                               data_type=dtype)

    print("Building engines...")
    build_cmd = [
        "trtllm-build ",
        f"--checkpoint_dir={ckpt_dir}",
        f"--output_dir={engine_dir}",
        f"--max_batch_size={8}",
        f"--max_input_len={924}",
        f"--max_seq_len={1024}",
        f"--max_beam_width={5}",
        f"--gemm_plugin={dtype}",
        "--gather_context_logits",
    ]
    if use_gpt_attention_plugin:
        build_cmd.append(f"--gpt_attention_plugin={dtype}")
    else:
        build_cmd.extend([
            "--gpt_attention_plugin=disable",
            "--context_fmha=disable",
            "--paged_kv_cache=disable",
            "--remove_input_padding=disable",
        ])
    if context_fmha_type == "enabled":
        build_cmd.append("--context_fmha=enable")
    elif context_fmha_type == "disabled":
        build_cmd.append("--context_fmha=disable")

    check_call(" ".join(build_cmd), shell=True, env=llm_venv._new_env)

    print('Run falcon-rw-1b...')
    data_type = "fp16" if dtype == "float16" else "bf16"

    # disable kv cache reuse for now.
    # TODO(tjohnsen) enable kv cache reuse when https://nvbugspro.nvidia.com/bug/5048858 fixed
    summary_cmd = [
        f"{falcon_example_root}/../summarize.py",
        "--test_trt_llm",
        "--hf_model_dir",
        llm_falcon_rw_1b_model_root,
        "--engine_dir",
        engine_dir,
        "--data_type",
        data_type,
        "--check_accuracy",
        f"--num_beams={num_beams}",
        "--eval_ppl",
        f"--dataset_dir={llm_datasets_root}",
        f"--rouge_dir={llm_rouge_root}",
        '--kv_cache_free_gpu_memory_fraction=0.5',
        "--no-kv_cache_enable_block_reuse",
    ]
    if use_py_session:
        summary_cmd.extend(["--use_py_session"])
    if context_fmha_type == "enabled_with_fp32_acc":
        summary_cmd.append("--enable_context_fmha_fp32_acc")
    venv_check_call(llm_venv, summary_cmd)


@pytest.mark.skip_less_device(2)
@pytest.mark.skip_less_device_memory(50000)
@pytest.mark.skip_less_host_memory(500000)
@pytest.mark.parametrize("num_beams", [1, 2, 4],
                         ids=lambda num_beams: f'nb:{num_beams}')
@pytest.mark.parametrize("embedding_sharding_dim", [-1, 0, 1],
                         ids=[
                             "disable_parallel_embedding",
                             "embedding_sharding_dim:0",
                             "embedding_sharding_dim:1"
                         ])
@pytest.mark.parametrize(
    "use_gpt_attention_plugin", [True, False],
    ids=["enable_attention_plugin", "disable_attention_plugin"])
@pytest.mark.parametrize("use_py_session", [False, True],
                         ids=["use_cpp_session", "use_py_session"])
def test_llm_falcon_rw_1b_1node_2gpus(
        falcon_example_root, llm_falcon_rw_1b_model_root, llm_datasets_root,
        llm_rouge_root, llm_venv, cmodel_dir, engine_dir,
        embedding_sharding_dim, use_gpt_attention_plugin, use_py_session,
        num_beams):
    # Test for Falcon ALiBi on TP>1
    print("Converting checkpoint...")
    dtype = 'float16'
    # Disable parallel embedding if embedding_sharding_dim < 0
    use_parallel_embedding = (embedding_sharding_dim >= 0)
    embedding_sharding_dim = max(0, embedding_sharding_dim)
    model_name = os.path.basename(llm_falcon_rw_1b_model_root)
    ckpt_dir = convert_weights(llm_venv=llm_venv,
                               example_root=falcon_example_root,
                               cmodel_dir=cmodel_dir,
                               model=model_name,
                               model_path=llm_falcon_rw_1b_model_root,
                               data_type=dtype,
                               gpus=2,
                               tp_size=2,
                               use_parallel_embedding=use_parallel_embedding,
                               embedding_sharding_dim=embedding_sharding_dim)

    print("Building engines...")
    build_cmd = [
        "trtllm-build",
        f"--checkpoint_dir={ckpt_dir}",
        f"--output_dir={engine_dir}",
        f"--max_batch_size={8}",
        f"--max_input_len={924}",
        f"--max_seq_len={1024}",
        f"--max_beam_width={5}",
        f"--gemm_plugin={dtype}",
    ]
    if use_gpt_attention_plugin:
        build_cmd.append(f"--gpt_attention_plugin={dtype}")
    else:
        build_cmd.extend([
            "--gpt_attention_plugin=disable",
            "--context_fmha=disable",
            "--paged_kv_cache=disable",
            "--remove_input_padding=disable",
        ])

    check_call(" ".join(build_cmd), shell=True, env=llm_venv._new_env)

    print('Run falcon-rw-1b...')
    # Reference Rouge1 score (HF): 1=15.62, 2=18.82, 4=20.26
    rouge1_threshold = {1: 14.85, 2: 17.8, 4: 19}[num_beams]
    summary_cmd = [
        f"{falcon_example_root}/../summarize.py",
        "--test_trt_llm",
        "--hf_model_dir",
        llm_falcon_rw_1b_model_root,
        "--engine_dir",
        engine_dir,
        "--data_type",
        "fp16",
        "--check_accuracy",
        f"--num_beams={num_beams}",
        f"--dataset_dir={llm_datasets_root}",
        f"--rouge_dir={llm_rouge_root}",
        f"--tensorrt_llm_rouge1_threshold={rouge1_threshold}",
    ]
    if use_py_session:
        summary_cmd.extend(["--use_py_session"])
    venv_mpi_check_call(llm_venv, ["mpirun", "-n", "2", "--allow-run-as-root"],
                        summary_cmd)


@pytest.mark.skip_less_device(2)
@pytest.mark.skip_less_device_memory(50000)
@pytest.mark.skip_less_host_memory(500000)
@pytest.mark.parametrize("num_beams", [1, 2, 4],
                         ids=lambda num_beams: f'nb:{num_beams}')
@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
@pytest.mark.parametrize("context_fmha_type",
                         ["enabled", "enabled_with_fp32_acc", "disabled"])
@pytest.mark.parametrize("use_py_session", [False, True],
                         ids=["use_cpp_session", "use_py_session"])
def test_llm_falcon_40b_1node_2gpus(falcon_example_root,
                                    llm_falcon_40b_model_root,
                                    llm_datasets_root, llm_rouge_root, llm_venv,
                                    cmodel_dir, engine_dir, dtype,
                                    context_fmha_type, use_py_session,
                                    num_beams):
    # Build & Run falcon 40b with two gpus
    print("Converting checkpoint...")
    model_name = os.path.basename(llm_falcon_40b_model_root)
    ckpt_dir = convert_weights(llm_venv=llm_venv,
                               example_root=falcon_example_root,
                               cmodel_dir=cmodel_dir,
                               model=model_name,
                               model_path=llm_falcon_40b_model_root,
                               data_type=dtype,
                               gpus=2,
                               tp_size=2,
                               workers=2)
    print("Building engines...")
    build_cmd = [
        "trtllm-build",
        f"--checkpoint_dir={ckpt_dir}",
        f"--output_dir={engine_dir}",
        f"--max_batch_size={8}",
        f"--max_input_len={924}",
        f"--max_seq_len={1024}",
        f"--max_beam_width={5}",
        "--remove_input_padding=enable",
        f"--gpt_attention_plugin={dtype}",
        f"--gemm_plugin={dtype}",
    ]
    if context_fmha_type == "enabled":
        build_cmd.append("--context_fmha=enable")
    elif context_fmha_type == "disabled":
        build_cmd.append("--context_fmha=disable")

    check_call(" ".join(build_cmd), shell=True, env=llm_venv._new_env)

    print('Run falcon 40b...')
    data_type = "fp16" if dtype == "float16" else "bf16"
    summary_cmd = [
        f"{falcon_example_root}/../summarize.py",
        "--test_trt_llm",
        "--hf_model_dir",
        llm_falcon_40b_model_root,
        "--engine_dir",
        engine_dir,
        "--data_type",
        data_type,
        "--check_accuracy",
        f"--num_beams={num_beams}",
        f"--dataset_dir={llm_datasets_root}",
        f"--rouge_dir={llm_rouge_root}",
        '--kv_cache_free_gpu_memory_fraction=0.8',
    ]
    if use_py_session:
        summary_cmd.extend(["--use_py_session"])
    if context_fmha_type == "enabled_with_fp32_acc":
        summary_cmd.append("--enable_context_fmha_fp32_acc")
    venv_mpi_check_call(llm_venv, ["mpirun", "-n", "2", "--allow-run-as-root"],
                        summary_cmd)


@pytest.mark.parametrize("num_beams", [1, 2, 4],
                         ids=lambda num_beams: f'nb:{num_beams}')
@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
@pytest.mark.parametrize("context_fmha_type",
                         ["enabled", "enabled_with_fp32_acc", "disabled"])
@pytest.mark.parametrize("enable_block_reuse", [True, False],
                         ids=["enable_block_reuse", "disable_block_reuse"])
def test_llm_falcon_7b_1node_1gpus(falcon_example_root,
                                   llm_falcon_7b_model_root, llm_datasets_root,
                                   llm_venv, cmodel_dir, engine_dir, dtype,
                                   context_fmha_type, enable_block_reuse,
                                   num_beams, llm_rouge_root):
    "Build & Run falcon-7b with one gpu"
    if num_beams > 1 and enable_block_reuse:
        pytest.skip(
            "Block reuse is currently not supported with beam width > 1.")

    print("Converting checkpoint...")
    model_name = os.path.basename(llm_falcon_7b_model_root)
    ckpt_dir = convert_weights(llm_venv,
                               falcon_example_root,
                               cmodel_dir,
                               model_name,
                               llm_falcon_7b_model_root,
                               data_type=dtype)

    print("Building engines...")
    build_cmd = [
        "trtllm-build",
        f"--checkpoint_dir={ckpt_dir}",
        f"--output_dir={engine_dir}",
        f"--max_batch_size={8}",
        f"--max_input_len={924}",
        f"--max_seq_len={1024}",
        f"--max_beam_width={5}",
        "--remove_input_padding=enable",
        f"--gpt_attention_plugin={dtype}",
        f"--gemm_plugin={dtype}",
        "--gather_context_logits",
        "--use_paged_context_fmha=enable",
    ]
    if context_fmha_type == "enabled":
        build_cmd.append("--context_fmha=enable")

    check_call(" ".join(build_cmd), shell=True, env=llm_venv._new_env)

    print('Run falcon-7b...')
    data_type = "fp16" if dtype == "float16" else "bf16"
    summary_cmd = [
        f"{falcon_example_root}/../summarize.py",
        "--test_trt_llm",
        "--hf_model_dir",
        llm_falcon_7b_model_root,
        "--engine_dir",
        engine_dir,
        "--data_type",
        data_type,
        "--check_accuracy",
        f"--num_beams={num_beams}",
        f"--dataset_dir={llm_datasets_root}",
        f"--rouge_dir={llm_rouge_root}",
    ]
    if enable_block_reuse:
        summary_cmd.extend(["--kv_cache_enable_block_reuse"])
    if context_fmha_type == "enabled_with_fp32_acc":
        summary_cmd.append("--enable_context_fmha_fp32_acc")

    venv_check_call(llm_venv, summary_cmd)


@pytest.mark.skip_less_device(8)
@pytest.mark.skip_less_device_memory(80000)
@pytest.mark.skip_less_host_memory(1000000)
@pytest.mark.parametrize("num_beams", [1, 2, 4],
                         ids=lambda num_beams: f'nb:{num_beams}')
@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
@pytest.mark.parametrize("context_fmha_type",
                         ["enabled", "enabled_with_fp32_acc", "disabled"])
@pytest.mark.parametrize(
    "tp_pp_size", [(8, 1), (4, 2)],
    ids=lambda tp_pp_size: f'tp{tp_pp_size[0]}pp{tp_pp_size[1]}')
@pytest.mark.parametrize("use_py_session", [False, True],
                         ids=["use_cpp_session", "use_py_session"])
def test_llm_falcon_180b_1node_8gpus(falcon_example_root,
                                     llm_falcon_180b_model_root,
                                     llm_datasets_root, llm_rouge_root,
                                     llm_venv, cmodel_dir, engine_dir, dtype,
                                     context_fmha_type, tp_pp_size,
                                     use_py_session, num_beams):
    "Build & Run falcon 180b with 8 gpus"
    print("Converting checkpoint...")
    tp_size, pp_size = tp_pp_size
    world_size = tp_size * pp_size
    model_name = os.path.basename(llm_falcon_180b_model_root)
    ckpt_dir = convert_weights(llm_venv=llm_venv,
                               example_root=falcon_example_root,
                               cmodel_dir=cmodel_dir,
                               model=model_name,
                               model_path=llm_falcon_180b_model_root,
                               data_type=dtype,
                               gpus=world_size,
                               tp_size=tp_size,
                               pp_size=pp_size,
                               load_by_shard=True,
                               workers=world_size)

    print("Building engines...")
    build_cmd = [
        "trtllm-build",
        f"--checkpoint_dir={ckpt_dir}",
        f"--output_dir={engine_dir}",
        f"--max_batch_size={8}",
        f"--max_input_len={924}",
        f"--max_seq_len={1024}",
        f"--max_beam_width={num_beams}",
        "--remove_input_padding=enable",
        f"--gpt_attention_plugin={dtype}",
        f"--gemm_plugin={dtype}",
        f"--workers={world_size}",
    ]
    if context_fmha_type == "enabled":
        build_cmd.append("--context_fmha=enable")
    elif context_fmha_type == "disabled":
        build_cmd.append("--context_fmha=disable")

    check_call(" ".join(build_cmd), shell=True, env=llm_venv._new_env)

    print('Run falcon 180b...')
    data_type = "fp16" if dtype == "float16" else "bf16"
    summary_cmd = [
        f"{falcon_example_root}/../summarize.py", "--test_trt_llm",
        "--hf_model_dir", llm_falcon_180b_model_root, "--engine_dir",
        engine_dir, "--data_type", data_type, "--check_accuracy",
        f"--num_beams={num_beams}", f"--dataset_dir={llm_datasets_root}",
        f"--rouge_dir={llm_rouge_root}"
    ]
    if use_py_session:
        summary_cmd.extend(["--use_py_session"])
    if context_fmha_type == "enabled_with_fp32_acc":
        summary_cmd.append("--enable_context_fmha_fp32_acc")
    venv_mpi_check_call(llm_venv, ["mpirun", "-n", "8", "--allow-run-as-root"],
                        summary_cmd)


@skip_pre_ada
@pytest.mark.parametrize("use_py_session", [False, True],
                         ids=["use_cpp_session", "use_py_session"])
def test_llm_falcon_rw_1b_fp8_1node_1gpus(falcon_example_root,
                                          llm_falcon_rw_1b_model_root,
                                          llm_datasets_root, llm_rouge_root,
                                          llm_venv, cmodel_dir, engine_dir,
                                          use_py_session):
    "Build & Run falcon-rw-1b fp8 with 1 gpu"

    # Quantize HF falcon-rw-1b checkpoint into FP8 format
    print("Quantizing and converting checkpoint...")
    model_name = os.path.basename(llm_falcon_rw_1b_model_root)
    dtype = "float16"
    ckpt_dir = f"{cmodel_dir}/{model_name}/fp8/1-gpu"

    quantize_cmd = [
        f"{falcon_example_root}/../quantization/quantize.py",
        f"--model_dir={llm_falcon_rw_1b_model_root}",
        f"--calib_dataset={llm_datasets_root}/cnn_dailymail",
        f"--dtype={dtype}",
        "--qformat=fp8",
        f"--output_dir={ckpt_dir}",
    ]

    venv_check_call(llm_venv, quantize_cmd)

    print("Building engines...")
    build_cmd = [
        "trtllm-build",
        f"--checkpoint_dir={ckpt_dir}",
        f"--output_dir={engine_dir}",
        f"--max_batch_size={8}",
        f"--max_input_len={924}",
        f"--max_seq_len={1024}",
        "--remove_input_padding=enable",
        f"--gpt_attention_plugin={dtype}",
        f"--gemm_plugin={dtype}",
    ]

    check_call(" ".join(build_cmd), shell=True, env=llm_venv._new_env)

    print('Run falcon-rw-1b...')
    summary_cmd = generate_summary_cmd(falcon_example_root,
                                       hf_model_dir=llm_falcon_rw_1b_model_root,
                                       engine_dir=engine_dir,
                                       data_type='fp16',
                                       tensorrt_llm_rouge1_threshold=15.5,
                                       use_py_session=use_py_session,
                                       dataset_dir=llm_datasets_root,
                                       rouge_dir=llm_rouge_root,
                                       max_ite=100)

    venv_check_call(llm_venv, summary_cmd)


@skip_pre_ada
@pytest.mark.skip_less_device(8)
@pytest.mark.skip_less_device_memory(80000)
@pytest.mark.skip_less_host_memory(1000000)
@pytest.mark.parametrize("use_py_session", [False, True],
                         ids=["use_cpp_session", "use_py_session"])
def test_llm_falcon_180b_fp8_1node_8gpus(falcon_example_root,
                                         llm_falcon_180b_model_root,
                                         llm_datasets_root, llm_rouge_root,
                                         llm_venv, cmodel_dir, engine_dir,
                                         use_py_session):
    "Build & Run falcon 180b fp8 with 8 gpus"

    # Quantize HF Falcon 180B checkpoint into FP8 format
    print("Quantizing and converting checkpoint...")
    model_name = os.path.basename(llm_falcon_180b_model_root)
    dtype = "float16"
    tp_size, pp_size = 8, 1
    world_size = tp_size * pp_size
    ckpt_dir = f"{cmodel_dir}/{model_name}/fp8/tp{tp_size}-pp{pp_size}"

    quantize_cmd = [
        f"{falcon_example_root}/../quantization/quantize.py",
        f"--model_dir={llm_falcon_180b_model_root}",
        f"--calib_dataset={llm_datasets_root}/cnn_dailymail",
        f"--dtype={dtype}",
        "--qformat=fp8",
        f"--output_dir={ckpt_dir}",
        "--calib_size=16",
        f"--tp_size={tp_size}",
        f"--pp_size={pp_size}",
    ]

    venv_check_call(llm_venv, quantize_cmd)

    print("Building engines...")
    build_cmd = [
        "trtllm-build",
        f"--checkpoint_dir={ckpt_dir}",
        f"--output_dir={engine_dir}",
        f"--max_batch_size={8}",
        f"--max_input_len={924}",
        f"--max_seq_len={1024}",
        "--remove_input_padding=enable",
        f"--gpt_attention_plugin={dtype}",
        f"--gemm_plugin={dtype}",
        f"--workers={world_size}",
    ]

    check_call(" ".join(build_cmd), shell=True, env=llm_venv._new_env)

    print('Run falcon 180b...')
    summary_cmd = generate_summary_cmd(falcon_example_root,
                                       hf_model_dir=llm_falcon_180b_model_root,
                                       engine_dir=engine_dir,
                                       data_type='fp16',
                                       use_py_session=use_py_session,
                                       dataset_dir=llm_datasets_root,
                                       rouge_dir=llm_rouge_root)

    venv_mpi_check_call(llm_venv, [
        "mpirun", "-n", f"{world_size}", "--allow-run-as-root",
        "--oversubscribe"
    ], summary_cmd)


@pytest.mark.parametrize("quant_algo", ["w4a8_awq", "w4a16_awq"])
@pytest.mark.parametrize("use_py_session", [False, True],
                         ids=["use_cpp_session", "use_py_session"])
def test_llm_falcon_rw_1b_awq_1node_1gpus(falcon_example_root,
                                          llm_falcon_rw_1b_model_root,
                                          llm_datasets_root, llm_rouge_root,
                                          llm_venv, cmodel_dir, engine_dir,
                                          quant_algo, use_py_session):
    "Build & Run falcon-rw-1b int4_awq with 1 gpu"
    skip_fp8_pre_ada("w4a8_awq" in quant_algo)

    # Quantize HF falcon-rw-1b checkpoint into int4_awq format
    print("Quantizing and converting checkpoint...")
    model_name = os.path.basename(llm_falcon_rw_1b_model_root)
    dtype = "float16"
    qformat = "int4_awq" if quant_algo == "w4a16_awq" else quant_algo
    ckpt_dir = f"{cmodel_dir}/{model_name}/{quant_algo}/1-gpu"

    quantize_cmd = [
        f"{falcon_example_root}/../quantization/quantize.py",
        f"--model_dir={llm_falcon_rw_1b_model_root}",
        f"--calib_dataset={llm_datasets_root}/cnn_dailymail",
        f"--dtype={dtype}",
        f"--qformat={qformat}",
        f"--output_dir={ckpt_dir}",
    ]

    venv_check_call(llm_venv, quantize_cmd)

    print("Building engines...")
    build_cmd = [
        "trtllm-build",
        f"--checkpoint_dir={ckpt_dir}",
        f"--output_dir={engine_dir}",
        f"--gpt_attention_plugin={dtype}",
        f"--gemm_plugin={dtype}",
        "--remove_input_padding=enable",
        "--context_fmha=enable",
    ]

    check_call(" ".join(build_cmd), shell=True, env=llm_venv._new_env)

    print('Run falcon-rw-1b...')
    summary_cmd = generate_summary_cmd(falcon_example_root,
                                       hf_model_dir=llm_falcon_rw_1b_model_root,
                                       engine_dir=engine_dir,
                                       data_type='fp16',
                                       tensorrt_llm_rouge1_threshold=13.5,
                                       use_py_session=use_py_session,
                                       dataset_dir=llm_datasets_root,
                                       rouge_dir=llm_rouge_root)

    venv_check_call(llm_venv, summary_cmd)


@pytest.mark.skip_less_device(2)
@pytest.mark.skip_less_device_memory(80000)
@pytest.mark.skip_less_host_memory(1000000)
@pytest.mark.parametrize("quant_algo", ["w4a8_awq", "w4a16_awq"])
@pytest.mark.parametrize("use_py_session", [False, True],
                         ids=["use_cpp_session", "use_py_session"])
def test_llm_falcon_180b_awq_1node_2gpus(falcon_example_root,
                                         llm_falcon_180b_model_root,
                                         llm_datasets_root, llm_rouge_root,
                                         llm_venv, cmodel_dir, engine_dir,
                                         quant_algo, use_py_session):
    "Build & Run falcon 180b int4_awq with 2 gpus"
    skip_fp8_pre_ada("w4a8_awq" in quant_algo)

    # Quantize HF Falcon 180B checkpoint into int4_awq format
    print("Quantizing and converting checkpoint...")
    model_name = os.path.basename(llm_falcon_180b_model_root)
    dtype = "float16"
    qformat = "int4_awq" if quant_algo == "w4a16_awq" else quant_algo
    tp_size, pp_size = 2, 1
    world_size = tp_size * pp_size
    ckpt_dir = f"{cmodel_dir}/{model_name}/{quant_algo}/tp{tp_size}-pp{pp_size}"

    quantize_cmd = [
        f"{falcon_example_root}/../quantization/quantize.py",
        f"--model_dir={llm_falcon_180b_model_root}",
        f"--calib_dataset={llm_datasets_root}/cnn_dailymail",
        f"--dtype={dtype}",
        f"--qformat={qformat}",
        f"--output_dir={ckpt_dir}",
        "--calib_size=16",
        f"--tp_size={tp_size}",
        f"--pp_size={pp_size}",
    ]

    venv_check_call(llm_venv, quantize_cmd)

    print("Building engines...")
    build_cmd = [
        "trtllm-build",
        f"--checkpoint_dir={ckpt_dir}",
        f"--output_dir={engine_dir}",
        f"--gpt_attention_plugin={dtype}",
        f"--gemm_plugin={dtype}",
        f"--workers={world_size}",
        "--remove_input_padding=enable",
        "--context_fmha=enable",
    ]

    check_call(" ".join(build_cmd), shell=True, env=llm_venv._new_env)

    print('Run falcon 180b...')
    summary_cmd = generate_summary_cmd(falcon_example_root,
                                       hf_model_dir=llm_falcon_180b_model_root,
                                       engine_dir=engine_dir,
                                       data_type='fp16',
                                       use_py_session=use_py_session,
                                       dataset_dir=llm_datasets_root,
                                       rouge_dir=llm_rouge_root)
    venv_mpi_check_call(llm_venv, [
        "mpirun", "-n", f"{world_size}", "--allow-run-as-root",
        "--oversubscribe"
    ], summary_cmd)


@pytest.mark.parametrize("num_beams", [1, 2, 4],
                         ids=lambda num_beams: f'nb:{num_beams}')
@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
@pytest.mark.parametrize("context_fmha_type",
                         ["enabled", "enabled_with_fp32_acc", "disabled"])
@pytest.mark.parametrize("enable_block_reuse", [True, False],
                         ids=["enable_block_reuse", "disable_block_reuse"])
def test_llm_falcon_11b_1node_1gpus(falcon_example_root,
                                    llm_falcon_11b_model_root,
                                    llm_datasets_root, llm_venv, cmodel_dir,
                                    engine_dir, dtype, context_fmha_type,
                                    enable_block_reuse, num_beams,
                                    llm_rouge_root):
    "Build & Run falcon-11B with one gpu"
    if num_beams > 1 and enable_block_reuse:
        pytest.skip(
            "Block reuse is currently not supported with beam width > 1.")

    print("Converting checkpoint...")
    model_name = os.path.basename(llm_falcon_11b_model_root)
    ckpt_dir = convert_weights(llm_venv,
                               falcon_example_root,
                               cmodel_dir,
                               model_name,
                               llm_falcon_11b_model_root,
                               data_type=dtype)

    print("Building engines...")
    build_cmd = [
        "trtllm-build",
        f"--checkpoint_dir={ckpt_dir}",
        f"--output_dir={engine_dir}",
        f"--max_batch_size={8}",
        f"--max_input_len={924}",
        f"--max_seq_len={1024}",
        f"--max_beam_width={5}",
        "--remove_input_padding=enable",
        f"--gpt_attention_plugin={dtype}",
        f"--gemm_plugin={dtype}",
        "--gather_context_logits",
        "--use_paged_context_fmha=enable",
    ]
    if context_fmha_type == "enabled":
        build_cmd.append("--context_fmha=enable")

    check_call(" ".join(build_cmd), shell=True, env=llm_venv._new_env)

    print('Run falcon-11B...')
    data_type = "fp16" if dtype == "float16" else "bf16"
    summary_cmd = [
        f"{falcon_example_root}/../summarize.py",
        "--test_trt_llm",
        "--hf_model_dir",
        llm_falcon_11b_model_root,
        "--engine_dir",
        engine_dir,
        "--data_type",
        data_type,
        "--check_accuracy",
        f"--num_beams={num_beams}",
        f"--dataset_dir={llm_datasets_root}",
        f"--rouge_dir={llm_rouge_root}",
    ]
    if enable_block_reuse:
        summary_cmd.extend(["--kv_cache_enable_block_reuse"])
    if context_fmha_type == "enabled_with_fp32_acc":
        summary_cmd.append("--enable_context_fmha_fp32_acc")

    venv_check_call(llm_venv, summary_cmd)
