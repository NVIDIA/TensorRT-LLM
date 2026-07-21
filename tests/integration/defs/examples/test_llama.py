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
import copy
import csv
import os

import defs.ci_profiler
import pytest
from defs.common import (convert_weights, generate_summary_cmd, quantize_data,
                         test_llm_torch_multi_lora_support,
                         test_multi_lora_support, venv_check_call,
                         venv_mpi_check_call)
# yapf: disable
from defs.conftest import (get_device_count, get_device_memory,
                           skip_fp8_pre_ada, skip_post_blackwell, skip_pre_ada)
# yapf: enable
from defs.trt_test_alternative import check_call

# skip trt flow cases on post-Blackwell-Ultra
# if get_sm_version() >= 103:
#     pytest.skip(
#         "TRT workflow tests are not supported on post Blackwell-Ultra architecture",
#         allow_module_level=True)

INPUT_TEXT_1 = "After Washington had returned to Williamsburg, " + \
               "Dinwiddie ordered him to lead a larger force to assist Trent in his work. " + \
               "While en route, Washington learned of Trent's retreat. " + \
               "Since Tanaghrisson had promised support to the British, " + \
               "Washington continued toward Fort Duquesne and met with the Mingo leader. " + \
               "Learning of a French scouting party in the area, Washington, " + \
               "with Tanaghrisson and his party, surprised the Canadians on May 28 " + \
               "in what became known as the Battle of Jumonville Glen. " + \
               "They killed many of the Canadians, including their commanding officer, " + \
               "Joseph Coulon de Jumonville, whose head was reportedly split open by " + \
               "Tanaghrisson with a tomahawk. The historian Fred Anderson suggests that " + \
               "Tanaghrisson was acting to gain the support of the British and regain " + \
               "authority over his own people. They had been inclined to support the French, " + \
               "with whom they had long trading relationships. One of Tanaghrisson's men told " + \
               "Contrecoeur that Jumonville had been killed by British musket fire. " + \
               "Question: Upon learning of a French scounting party in the area, " + \
               "what did Washington do? Answer:"

INPUT_TEXT_2 = "Born in north-east France, Soyer trained as a"


@pytest.mark.parametrize("num_beams", [1, 2, 4],
                         ids=lambda num_beams: f'nb:{num_beams}')
@pytest.mark.parametrize("run_type", ['inference', 'summarization'])
@pytest.mark.parametrize("data_type", ['bfloat16', 'float16'])
@pytest.mark.parametrize("fp8_cache", [True, False],
                         ids=["enable_fp8", "disable_fp8"])
@pytest.mark.parametrize(
    "llama_model_root",
    ['llama-v3-8b-instruct-hf', 'llama-3.1-8b-instruct-hf-fp8'],
    indirect=True)
def test_llm_llama_1gpu(run_type, data_type, fp8_cache, llama_example_root,
                        llama_model_root, llm_datasets_root, llm_rouge_root,
                        llm_venv, cmodel_dir, engine_dir,
                        qcache_dir_without_install_package, num_beams):
    if num_beams > 2 and get_device_memory() < 80000:
        pytest.skip("device memory is insufficient.")

    use_fp8 = fp8_cache if "fp8" not in llama_model_root.lower() else True
    skip_fp8_pre_ada(use_fp8=use_fp8)

    model_name = os.path.basename(llama_model_root)

    if llama_model_root.endswith('Llama-3.1-8B-Instruct-FP8'):
        model_dir = convert_weights(llm_venv=llm_venv,
                                    example_root=llama_example_root,
                                    cmodel_dir=cmodel_dir,
                                    model="llama_v3_finegrained_fp8",
                                    model_path=llama_model_root,
                                    fp8_kv_cache=fp8_cache,
                                    data_type=data_type)
    elif fp8_cache:
        # Quantize HF llama checkpoint into FP8 format
        model_dir = quantize_data(
            llm_venv,
            llama_example_root,
            model_dir=llama_model_root,
            calib_dataset=f"{llm_datasets_root}/cnn_dailymail",
            dtype=data_type,
            qformat="fp8",
            quantize_dir=qcache_dir_without_install_package,
            calib_size=512,
            kv_cache_dtype="fp8")
    else:
        model_dir = convert_weights(
            llm_venv=llm_venv,
            example_root=llama_example_root,
            cmodel_dir=cmodel_dir,
            model=model_name,
            model_path=llama_model_root,
            data_type=data_type,
            enable_fp8=fp8_cache,
            fp8_kv_cache=fp8_cache,
            quant_ckpt_path=
            f"{qcache_dir_without_install_package}/quantized_fp8/llama_tp1_rank0.npz"
            if fp8_cache else None)

    print("Build engines...")
    build_cmd = [
        "trtllm-build",
        f"--checkpoint_dir={model_dir}",
        f"--output_dir={engine_dir}",
        f"--gpt_attention_plugin={data_type}",
        f"--gemm_plugin={data_type}",
        "--remove_input_padding=enable",
        f"--max_beam_width={num_beams}",
    ]

    check_call(" ".join(build_cmd), shell=True, env=llm_venv._new_env)

    if run_type == "inference":
        print("Run inference...")
        venv_check_call(llm_venv, [
            f"{llama_example_root}/../run.py",
            "--max_output_len=50",
            f"--tokenizer_dir={llama_model_root}",
            f"--engine_dir={engine_dir}",
            f"--num_beams={num_beams}",
        ])
    elif run_type == "summarization":
        print("Run summarize...")
        tensorrt_llm_rouge1_threshold = {
            1: 14,
            2: 19,
            4: 19,
        }[num_beams]

        summary_cmd = generate_summary_cmd(
            llama_example_root,
            hf_model_dir=llama_model_root,
            data_type="fp16",
            engine_dir=engine_dir,
            tensorrt_llm_rouge1_threshold=tensorrt_llm_rouge1_threshold,
            num_beams=num_beams,
            dataset_dir=llm_datasets_root,
            rouge_dir=llm_rouge_root)

        venv_check_call(llm_venv, summary_cmd)


@pytest.mark.parametrize(
    "data_type", ['float16', 'fp8', 'sq_ootb', 'awq', 'int8_wo'],
    ids=['base_fp16', 'base_fp8', 'base_sq_ootb', 'base_awq', 'base_int8_wo'])
@pytest.mark.parametrize("llama_model_root", ['llama-v3-8b-hf'], indirect=True)
@pytest.mark.parametrize("llm_dora_model_root",
                         ['commonsense-llama-v3-8b-dora-r32'],
                         indirect=True)
def test_llm_llama_v3_dora_1gpu(data_type, llama_example_root, llama_model_root,
                                llm_dora_model_root, llm_datasets_root,
                                llm_venv, cmodel_dir, engine_dir,
                                qcache_dir_without_install_package):
    "run llama dora test on 1gpu"
    print("Build engines...")

    model_name = 'llama_v3-dora'
    if data_type == 'fp8':
        skip_fp8_pre_ada(use_fp8=True)

        model_dir = quantize_data(
            llm_venv,
            llama_example_root,
            model_dir=llama_model_root,
            calib_dataset=f"{llm_datasets_root}/cnn_dailymail",
            dtype="float16",
            qformat="fp8",
            quantize_dir=qcache_dir_without_install_package,
            calib_size=512,
            kv_cache_dtype="fp8")
    elif data_type == 'sq_ootb':
        model_dir = quantize_data(
            llm_venv,
            llama_example_root,
            model_dir=llama_model_root,
            calib_dataset=f"{llm_datasets_root}/cnn_dailymail",
            dtype="float16",
            qformat="int8_sq",
            quantize_dir=qcache_dir_without_install_package,
            calib_size=32)
    elif data_type == 'awq':
        model_dir = quantize_data(
            llm_venv,
            llama_example_root,
            model_dir=llama_model_root,
            calib_dataset=f"{llm_datasets_root}/cnn_dailymail",
            dtype="float16",
            qformat="int4_awq",
            awq_block_size=128,
            quantize_dir=qcache_dir_without_install_package,
            calib_size=32)
    elif data_type == 'int8_wo':
        model_dir = convert_weights(llm_venv=llm_venv,
                                    example_root=llama_example_root,
                                    cmodel_dir=cmodel_dir,
                                    model=model_name,
                                    model_path=llama_model_root,
                                    use_weight_only=True,
                                    weight_only_precision='int8')
    else:
        model_dir = convert_weights(llm_venv=llm_venv,
                                    example_root=llama_example_root,
                                    cmodel_dir=cmodel_dir,
                                    model=model_name,
                                    model_path=llama_model_root)

    # normalize dora magnitude
    dora_weights = f"{llm_venv.get_working_directory()}/dora_weights"

    normalize_cmd = [
        f"{llama_example_root}/../../../dora/normalize_weights.py", "-i",
        llm_dora_model_root, "-b", llama_model_root, "-o", dora_weights
    ]

    venv_check_call(llm_venv, normalize_cmd)

    print("Build engines...")
    build_cmd = [
        "trtllm-build",
        f"--checkpoint_dir={model_dir}",
        f"--output_dir={engine_dir}",
        "--lora_plugin=auto",
        "--dora_plugin=enable",
        "--remove_input_padding=enable",  # otherwise no cpp runtime
        "--kv_cache_type=paged",  # otherwise no cpp runtime
        "--gemm_plugin=auto",
        f"--lora_dir={dora_weights}",
    ]
    check_call(" ".join(build_cmd), shell=True, env=llm_venv._new_env)

    input_tokens = [
        128000, 39314, 374, 459, 7754, 430, 16964, 264, 3465, 11, 35526, 449,
        459, 1988, 430, 5825, 4726, 2317, 13, 9842, 264, 2077, 430, 36001,
        45695, 279, 1715, 382, 394, 17010, 30151, 512, 394, 5321, 5268, 279,
        4495, 4320, 311, 279, 3488, 25, 578, 842, 1121, 304, 279, 1920, 315,
        7397, 74767, 374, 279, 5788, 315, 13465, 323, 24463, 13, 16299, 3094,
        17738, 279, 7314, 315, 7397, 74767, 31931, 16533, 16, 25, 36424, 4907,
        374, 42101, 1555, 279, 20282, 13, 22559, 17, 25, 8828, 4907, 374, 16489,
        311, 11742, 4907, 13, 22559, 18, 25, 92479, 5237, 25734, 304, 279,
        16312, 41255, 3177, 4907, 13, 22559, 19, 25, 8219, 4238, 374, 16489,
        1139, 37833, 5237, 25734, 4286, 16533, 3645, 25, 4320, 16, 14, 9399, 17,
        14, 9399, 18, 14, 9399, 19, 271, 394, 17010, 5688, 512, 72348, 394,
        17010, 6075, 1473
    ]

    out_ref = [
        128000, 39314, 374, 459, 7754, 430, 16964, 264, 3465, 11, 35526, 449,
        459, 1988, 430, 5825, 4726, 2317, 13, 9842, 264, 2077, 430, 36001,
        45695, 279, 1715, 382, 394, 17010, 30151, 512, 394, 5321, 5268, 279,
        4495, 4320, 311, 279, 3488, 25, 578, 842, 1121, 304, 279, 1920, 315,
        7397, 74767, 374, 279, 5788, 315, 13465, 323, 24463, 13, 16299, 3094,
        17738, 279, 7314, 315, 7397, 74767, 31931, 16533, 16, 25, 36424, 4907,
        374, 42101, 1555, 279, 20282, 13, 22559, 17, 25, 8828, 4907, 374, 16489,
        311, 11742, 4907, 13, 22559, 18, 25, 92479, 5237, 25734, 304, 279,
        16312, 41255, 3177, 4907, 13, 22559, 19, 25, 8219, 4238, 374, 16489,
        1139, 37833, 5237, 25734, 4286, 16533, 3645, 25, 4320, 16, 14, 9399, 17,
        14, 9399, 18, 14, 9399, 19, 271, 394, 17010, 5688, 512, 72348, 394,
        17010, 6075, 1473, 394, 279, 4495, 4320, 374, 4320, 18, 128001, 128001,
        128001, 128001, 128001, 128001, 128001, 128001, 128001, 128001, 128001,
        128001, 128001, 128001, 128001, 128001, 128001, 128001, 128001, 128001,
        128001, 128001, 128001, 128001, 128001
    ]

    in_csv = f"{llm_venv.get_working_directory()}/input.csv"
    out_csv = f"{llm_venv.get_working_directory()}/output.csv"
    with open(in_csv, "w") as f:
        writer = csv.writer(f)
        writer.writerow(input_tokens)

    base_run_cmd = [
        f"{llama_example_root}/../../../run.py", "--max_output_len=20",
        f"--input_file={in_csv}", f"--tokenizer_dir={llama_model_root}",
        f"--engine_dir={engine_dir}", "--max_output_len=32"
    ]

    for use_py_session in [True, False]:
        if use_py_session:
            print("Run inference with Python runtime...")
        else:
            print("Run inference with C++ runtime...")

        print(f"Run inference with lora id 0...")
        run_cmd = copy.deepcopy(base_run_cmd)
        run_cmd.extend(["--lora_task_uids=0", f"--output_csv={out_csv}"])
        if use_py_session:
            run_cmd.append("--use_py_session")
        venv_check_call(llm_venv, run_cmd)

        with open(out_csv) as f:
            predict = csv.reader(f)
            predict = next(predict)

        predict = [int(p) for p in predict]
        assert out_ref == predict or data_type != "float16"


@pytest.mark.skip_less_device(8)
@pytest.mark.skip_less_device_memory(80000)
@pytest.mark.parametrize("num_beams", [1, 4],
                         ids=lambda num_beams: f'nb:{num_beams}')
@pytest.mark.parametrize(
    "tp_pp_size", [(8, 1), (4, 2)],
    ids=lambda tp_pp_size: f'tp{tp_pp_size[0]}pp{tp_pp_size[1]}')
@pytest.mark.parametrize("test_case", ["pg64317"], indirect=True)
def test_llm_llama_long_alpaca_8gpu_summary(llama_example_root,
                                            llm_long_alpaca_model_root,
                                            llm_datasets_root, llm_rouge_root,
                                            llm_venv, cmodel_dir, engine_dir,
                                            num_beams, tp_pp_size, test_case):
    "llama test for long alpaca"
    tp_size, pp_size = tp_pp_size
    world_size = 8
    assert tp_size * pp_size == world_size, \
        f'tp_size({tp_size}) x pp_size({pp_size}) != 8'

    model_name = 'llama_long_alpaca'
    model_dir = convert_weights(llm_venv=llm_venv,
                                example_root=llama_example_root,
                                cmodel_dir=cmodel_dir,
                                model=model_name,
                                model_path=llm_long_alpaca_model_root,
                                gpus=world_size,
                                tp_size=tp_size,
                                pp_size=pp_size,
                                data_type="bfloat16")

    print("Build engines...")
    build_cmd = [
        "trtllm-build",
        f"--checkpoint_dir={model_dir}",
        f"--output_dir={engine_dir}",
        "--gpt_attention_plugin=bfloat16",
        "--remove_input_padding=enable",
        "--gemm_plugin=bfloat16",
        f"--max_beam_width={num_beams}",
        "--max_input_len=32768",
        "--max_seq_len=49152",
        "--max_batch_size=1",
        "--max_num_tokens=32768",
    ]
    print("Build engines...")

    check_call(" ".join(build_cmd), shell=True, env=llm_venv._new_env)

    print("Run summarize...")
    max_output_len = test_case["max_output_len"]
    run_cmd = [
        f"{llama_example_root}/../../../run.py",
        f"--max_output_len={max_output_len}",
        f"--input_file={test_case['input_file']}", f"--engine_dir={engine_dir}",
        f"--num_beams={num_beams}",
        f"--tokenizer_dir={llm_long_alpaca_model_root}",
        "--max_input_length=32768"
    ]

    venv_mpi_check_call(
        llm_venv, ["mpirun", "-n", f"{world_size}", "--allow-run-as-root"],
        run_cmd)

    summary_cmd = generate_summary_cmd(llama_example_root,
                                       hf_model_dir=llm_long_alpaca_model_root,
                                       max_input_length=16384,
                                       output_len=max_output_len,
                                       data_type="fp16",
                                       num_beams=num_beams,
                                       engine_dir=engine_dir,
                                       dataset_dir=llm_datasets_root,
                                       rouge_dir=llm_rouge_root)

    venv_mpi_check_call(
        llm_venv, ["mpirun", "-n", f"{world_size}", "--allow-run-as-root"],
        summary_cmd)


@pytest.mark.parametrize("fp8_quant", [
    'disable_fp8',
    pytest.param('enable_fp8', marks=skip_post_blackwell),
    pytest.param('enable_fp8_meta_recipe', marks=skip_post_blackwell)
])
@pytest.mark.parametrize("llama_model_root", ['llama-3.1-8b', 'llama-3.2-1b'],
                         indirect=True)
def test_llm_llama_v3_1_1node_single_gpu(llama_example_root, llama_model_root,
                                         llm_venv, cmodel_dir,
                                         llm_datasets_root, llm_rouge_root,
                                         engine_dir, fp8_quant):
    "Run llama3.1 test on 1 gpu."
    data_type = "bfloat16"
    model_name = os.path.basename(llama_model_root)

    use_fp8_rowwise = False
    use_meta_fp8_rowwise_recipe = False
    if fp8_quant == 'enable_fp8':
        use_fp8_rowwise = True
    elif fp8_quant == 'enable_fp8_meta_recipe':
        use_fp8_rowwise = True
        use_meta_fp8_rowwise_recipe = True

    print("Convert weight...")
    model_dir = convert_weights(
        llm_venv=llm_venv,
        example_root=llama_example_root,
        cmodel_dir=cmodel_dir,
        model=model_name,
        model_path=llama_model_root,
        data_type=data_type,
        tp_size=1,
        pp_size=1,
        use_fp8_rowwise=use_fp8_rowwise,
        use_meta_fp8_rowwise_recipe=use_meta_fp8_rowwise_recipe)

    print("Build engines...")
    build_cmd = [
        "trtllm-build", f"--checkpoint_dir={model_dir}",
        f"--output_dir={engine_dir}", f"--max_batch_size={8}",
        f"--max_seq_len={2048}"
    ]
    check_call(" ".join(build_cmd), shell=True, env=llm_venv._new_env)

    print("Run summarize...")
    summary_cmd = [
        f"{llama_example_root}/../../../summarize.py",
        "--test_trt_llm",
        f"--hf_model_dir={llama_model_root}",
        f"--engine_dir={engine_dir}",
        "--check_accuracy",
        f"--tensorrt_llm_rouge1_threshold={14}",
        f"--dataset_dir={llm_datasets_root}",
        f"--rouge_dir={llm_rouge_root}",
    ]
    venv_check_call(llm_venv, summary_cmd)


@skip_pre_ada
@pytest.mark.skip_less_device_memory(80000)
@pytest.mark.parametrize(
    "llama_model_root",
    ['llama-v3-8b-instruct-hf', 'llama-3.1-8b', 'llama-3.2-1b', 'llama-3.2-3b'],
    indirect=True)
def test_llama_3_x_fp8_with_bf16_lora(llama_example_root, llm_datasets_root,
                                      qcache_dir_without_install_package,
                                      llm_venv, engine_dir, llama_model_root):
    "Run Llama 3.1 and 3.2 models with multiple dummy LoRAs."

    print("Quantizing model to fp8...")

    defs.ci_profiler.start("quantize_model")
    qmodel_dir = quantize_data(
        llm_venv,
        llama_example_root,
        model_dir=llama_model_root,
        calib_dataset=f"{llm_datasets_root}/cnn_dailymail",
        dtype="bfloat16",
        qformat="fp8",
        quantize_dir=qcache_dir_without_install_package,
        calib_size=32,
        kv_cache_dtype="fp8")
    defs.ci_profiler.stop("quantize_model")
    print(
        f"quantize_model: {defs.ci_profiler.elapsed_time_in_sec('quantize_model')} sec"
    )

    defs.ci_profiler.start("test_multi_lora_support")
    test_multi_lora_support(
        hf_model_dir=llama_model_root,
        tllm_ckpt_dir=qmodel_dir,
        engine_dir=engine_dir,
        llm_venv=llm_venv,
        example_root=llama_example_root,
        num_loras=2,
        lora_rank=8,
        target_hf_modules=["q_proj", "k_proj", "v_proj"],
        target_trtllm_modules=["attn_q", "attn_k", "attn_v"],
        zero_lora_weights=True,
    )
    defs.ci_profiler.stop("test_multi_lora_support")
    print(
        f"test_multi_lora_support: {defs.ci_profiler.elapsed_time_in_sec('test_multi_lora_support')} sec"
    )


@skip_pre_ada
@pytest.mark.skip_less_device_memory(80000)
@pytest.mark.parametrize("llama_model_root", [
    'llama-v3-8b-instruct-hf',
    'llama-3.1-8b-instruct',
    'llama-3.2-1b-instruct',
    'llama-3.2-3b-instruct',
    'llama-3.3-70b-instruct',
],
                         indirect=True)
def test_llama_3_x_with_bf16_lora_torch(llama_example_root, llm_datasets_root,
                                        qcache_dir_without_install_package,
                                        llm_venv, engine_dir, llama_model_root):
    """Run Llama models with multiple dummy LoRAs using LLM-API Torch backend."""

    if "llama-3.3-70b-instruct" in llama_model_root.lower():
        tensor_parallel_size = 8
        if get_device_count() < 8:
            pytest.skip(
                "Skipping: llama-3.3-70b-instruct model requires 8 GPUs")
    else:
        tensor_parallel_size = 1

    print("Testing with LLM-API Torch backend...")

    defs.ci_profiler.start("test_llm_torch_multi_lora_support")

    test_llm_torch_multi_lora_support(
        hf_model_dir=llama_model_root,
        llm_venv=llm_venv,
        num_loras=2,
        lora_rank=8,
        target_hf_modules=["q_proj", "k_proj", "v_proj"],
        zero_lora_weights=True,
        tensor_parallel_size=tensor_parallel_size)
    defs.ci_profiler.stop("test_llm_torch_multi_lora_support")
    print(
        f"test_llm_torch_multi_lora_support: {defs.ci_profiler.elapsed_time_in_sec('test_llm_torch_multi_lora_support')} sec"
    )
