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
import json
import os

import defs.ci_profiler
import pytest
from defs.common import (convert_weights, generate_summary_cmd, parse_output,
                         quantize_data, similar,
                         test_llm_torch_multi_lora_support,
                         test_multi_lora_support, venv_check_call,
                         venv_check_output, venv_mpi_check_call)
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
@pytest.mark.parametrize("llama_model_root", [
    'llama-v2-7b-hf', 'llama-v3-8b-instruct-hf', 'llama-3.1-8b-instruct-hf-fp8'
],
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


@skip_pre_ada
@pytest.mark.parametrize("data_type", ['bfloat16'])
@pytest.mark.parametrize("llama_model_root", ['llama-v2-7b-hf'], indirect=True)
def test_llm_llama_1gpu_fp8_kv_cache(
    data_type,
    llama_example_root,
    llama_model_root,
    llm_datasets_root,
    llm_rouge_root,
    llm_venv,
    cmodel_dir,
    engine_dir,
    qcache_dir_without_install_package,
):
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

    print("Build engines...")
    build_cmd = [
        "trtllm-build",
        f"--checkpoint_dir={model_dir}",
        f"--output_dir={engine_dir}",
        f"--gpt_attention_plugin={data_type}",
        f"--gemm_plugin={data_type}",
        "--remove_input_padding=enable",
        "--use_paged_context_fmha=enable",
        "--use_fp8_context_fmha=enable",
        "--max_beam_width=1",
    ]

    check_call(" ".join(build_cmd), shell=True, env=llm_venv._new_env)
    with open(f"{engine_dir}/config.json") as f:
        engine_config = json.load(f)

    assert engine_config["build_config"]["plugin_config"][
        "use_fp8_context_fmha"] == True
    assert engine_config["pretrained_config"]["quantization"][
        "kv_cache_quant_algo"] == "FP8"


@pytest.mark.parametrize("llama_model_root", ['llama-7b'], indirect=True)
def test_llm_llama_v1_1gpu_kv_cache_reuse_with_prompt_table(
        llama_example_root, llama_model_root, llm_datasets_root, llm_rouge_root,
        llm_venv, cmodel_dir, engine_dir):
    max_prompt_embedding_table_size = 16
    hidden_size = 4096
    vocab_size = 32000
    input_len = 42

    print("Convert checkpoint...")
    model_name = 'llama_v1-kv_cache_reuse_w_prompt_table'
    model_dir = convert_weights(llm_venv=llm_venv,
                                example_root=llama_example_root,
                                cmodel_dir=cmodel_dir,
                                model=model_name,
                                model_path=llama_model_root)

    print("Build engines...")
    build_cmd = [
        "trtllm-build", f"--checkpoint_dir={model_dir}",
        f"--output_dir={engine_dir}/engines", "--gpt_attention_plugin=float16",
        "--gemm_plugin=float16", "--remove_input_padding=enable",
        "--max_batch_size=1",
        f"--tokens_per_block={max_prompt_embedding_table_size}",
        "--paged_kv_cache=enable", "--use_paged_context_fmha=enable",
        f"--max_prompt_embedding_table_size={max_prompt_embedding_table_size}"
    ]

    check_call(" ".join(build_cmd), shell=True, env=llm_venv._new_env)

    # generate input ids, dummy prompt table and extra ids
    input_file = f"{engine_dir}/input_ids.npy"
    prompt_table_path = f"{engine_dir}/prompt_table.npy"
    extra_ids_file = f"{engine_dir}/extra_ids.npy"
    # run the script inside venv since it depends on numpy
    venv_script = f'''
    import numpy as np
    input_ids = [[
        i + {vocab_size} if i < {max_prompt_embedding_table_size} else i + 1000
        for i in range({input_len})
    ]]
    np.save("{input_file}", np.array(input_ids))

    prompt_table_shape = (1, {max_prompt_embedding_table_size}, {hidden_size})
    prompt_table = np.random.rand(*prompt_table_shape).astype(np.float16)
    np.save("{prompt_table_path}", prompt_table)

    extra_ids = [[
        1 if i < {max_prompt_embedding_table_size} else 0
        for i in range({input_len})
    ]]
    np.save("{extra_ids_file}", np.array(extra_ids))
    '''
    llm_venv.run(venv_script)

    # add --run_profiling to run the request for multiple times
    print("Run inference")
    run_cmd = [
        f"{llama_example_root}/../../../run.py", "--max_output_len=10",
        f"--tokenizer_dir={llama_model_root}",
        f"--engine_dir={engine_dir}/engines", f"--input_file={input_file}",
        f"--prompt_table_path={prompt_table_path}",
        "--kv_cache_enable_block_reuse",
        f"--input_token_extra_ids_file={extra_ids_file}", "--run_profiling"
    ]
    venv_check_output(llm_venv, run_cmd)


@pytest.mark.parametrize("llama_model_root", ['llama-7b'], indirect=True)
def test_llm_llama_1gpu_batched_beam_search(llama_example_root,
                                            llama_model_root, llm_datasets_root,
                                            llm_venv, engine_dir,
                                            qcache_dir_without_install_package):
    "llama run batched beam search on 1 gpu"
    qmodel_dir = quantize_data(llm_venv,
                               llama_example_root,
                               model_dir=llama_model_root,
                               dtype="float16",
                               quantize_dir=qcache_dir_without_install_package)

    print("Build engines...")
    build_cmd = [
        "trtllm-build",
        f"--checkpoint_dir={qmodel_dir}",
        f"--output_dir={engine_dir}",
        f"--gpt_attention_plugin=float16",
        "--remove_input_padding=enable",
        "--paged_kv_cache=enable",
        "--max_batch_size=4",
        "--max_beam_width=4",
        "--max_input_len=512",
        "--max_seq_len=532",
        "--gemm_plugin=float16",
    ]

    check_call(" ".join(build_cmd), shell=True, env=llm_venv._new_env)

    # run.py test.
    num_beams = 4
    run_cmd = [
        f"{llama_example_root}/../../../run.py",
        "--max_output_len=20",
        f"--tokenizer_dir={llama_model_root}",
        f"--engine_dir={engine_dir}",
        "--no_add_special_tokens",
        f"--num_beams={num_beams}",
        "--input_text",
        "Miguel de Cervantes wrote",
        "Diego Velazquez painted his most famous painting,",
        "Miguel de Cervantes wrote",
        "Diego Velazquez painted his most famous painting,",
    ]

    output = venv_check_output(llm_venv, run_cmd)
    output = parse_output(output)

    for idx in [0, 1]:
        assert all(
            [
                a == b for (a, b) in zip(
                    output[num_beams * idx:num_beams * idx +
                           num_beams], output[num_beams * (idx + 2):num_beams *
                                              (idx + 2) + num_beams])
            ]
        ), f"outputs {idx} and {idx+2} don't match: {output[num_beams * idx:num_beams * idx + num_beams]}, {output[num_beams * (idx + 2):num_beams * (idx + 2) + num_beams]}"

    expected_output = [
        ["Don Quixote in 1605. The book is considered the first modern novel."],
        [
            "Las Meninas, in 1656. The painting is a portrait of King Philip IV",
            "\"Las Meninas\" in 1656. The painting depicts King Philip"
        ],
    ]

    for idx, result in enumerate(output):
        assert any(
            [
                similar(item, result)
                for item in expected_output[(idx // num_beams) % 2]
            ]
        ), f"output {result} is not similar to any of {expected_output[(idx // num_beams) % 2]}"


@pytest.mark.parametrize(
    "data_type", [
        'float16', 'fp8',
        pytest.param('sq_ootb', marks=skip_post_blackwell),
        pytest.param('awq', marks=skip_post_blackwell),
        pytest.param('int8_wo', marks=skip_post_blackwell)
    ],
    ids=['base_fp16', 'base_fp8', 'base_sq_ootb', 'base_awq', 'base_int8_wo'])
@pytest.mark.parametrize("lora_data_type", ['float16'], ids=['lora_fp16'])
@pytest.mark.parametrize("llama_model_root", ['llama-v2-13b-hf'], indirect=True)
@pytest.mark.parametrize("llm_lora_model_root", ['chinese-llama-2-lora-13b'],
                         indirect=True)
def test_llm_llama_v2_lora_1gpu(data_type, lora_data_type, llama_example_root,
                                llama_model_root, llm_datasets_root, llm_venv,
                                cmodel_dir, engine_dir, llm_lora_model_root,
                                qcache_dir_without_install_package):
    "run llama lora test on 1gpu"
    print("Build engines...")

    model_name = 'llama_v2-lora'
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

    print("Build engines...")
    build_cmd = [
        "trtllm-build",
        f"--checkpoint_dir={model_dir}",
        f"--output_dir={engine_dir}",
        "--lora_plugin=auto",
        "--gemm_plugin=auto",
        f"--lora_dir={llm_lora_model_root}",
    ]
    check_call(" ".join(build_cmd), shell=True, env=llm_venv._new_env)

    ref_1 = [
        29871, 32160, 33657, 33281, 30214, 30672, 30780, 33820, 32024, 30214,
        32083, 33820, 30755, 37432, 32030, 30313, 30214, 30417, 30210, 30505,
        34870, 30214, 30417, 30210, 30505, 31656, 39298, 30214, 32063, 30210
    ]
    ref_2 = [
        29871, 32160, 33657, 33281, 30214, 30672, 30780, 33820, 32024, 30214,
        33759, 41026, 31381, 30769, 31811, 31900, 30214, 36869, 31900, 36869,
        31900, 30214, 36869, 31900, 36869, 31900, 31900, 31900, 31900, 31900
    ]

    input_text = "今天天气很好，我到公园的时候，"
    # TODO change to chinese evaluation task in the future

    base_run_cmd = [
        f"{llama_example_root}/../../../run.py",
        "--max_output_len=20",
        f"--input_text={input_text}",
        f"--tokenizer_dir={llm_lora_model_root}",
        f"--engine_dir={engine_dir}",
        "--no_add_special_tokens",
    ]

    for use_py_session in [True, False]:
        if use_py_session:
            print("Run inference with Python runtime...")
        else:
            print("Run inference with C++ runtime...")

        print(f"Run inference with lora id 0...")
        run_cmd = copy.deepcopy(base_run_cmd)
        run_cmd.extend([
            "--lora_task_uids=0",
            f"--output_csv={llm_venv.get_working_directory()}/use_lora.csv"
        ])
        if use_py_session:
            run_cmd.append("--use_py_session")
        venv_check_call(llm_venv, run_cmd)

        with open(f"{llm_venv.get_working_directory()}/use_lora.csv") as f:
            predict = csv.reader(f)
            predict = next(predict)
        predict = [int(p) for p in predict]
        assert ref_1 == predict or data_type != "float16"

        print(f"Run inference with lora id -1...")
        run_cmd = copy.deepcopy(base_run_cmd)
        run_cmd.extend([
            "--lora_task_uids=-1",
            f"--output_csv={llm_venv.get_working_directory()}/no_lora.csv"
        ])
        if use_py_session:
            run_cmd.append("--use_py_session")
        venv_check_call(llm_venv, run_cmd)

        with open(f"{llm_venv.get_working_directory()}/no_lora.csv") as f:
            predict = csv.reader(f)
            predict = next(predict)
        predict = [int(p) for p in predict]
        assert ref_2 == predict or data_type != "float16"


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


@skip_post_blackwell
@pytest.mark.skip_less_device_memory(30000)
@pytest.mark.parametrize("num_beams", [1, 4],
                         ids=lambda num_beams: f'nb:{num_beams}')
@pytest.mark.parametrize("weight_only_precision", ["int4", "int8"])
@pytest.mark.parametrize("llama_model_root", ['llama-7b'], indirect=True)
def test_llm_llama_wo_1gpu_summary(llama_example_root, llama_model_root,
                                   llm_datasets_root, llm_rouge_root, llm_venv,
                                   engine_dir, num_beams, cmodel_dir,
                                   weight_only_precision):

    skip_fp8_pre_ada(use_fp8=True)

    llm_venv.get_working_directory()
    model_name = os.path.basename(llama_example_root)

    ckpt_dir = convert_weights(llm_venv=llm_venv,
                               example_root=llama_example_root,
                               cmodel_dir=cmodel_dir,
                               model=model_name,
                               model_path=llama_model_root,
                               data_type="float16",
                               use_weight_only=True,
                               weight_only_precision=weight_only_precision,
                               gpus=1,
                               tp_size=1,
                               pp_size=1)

    print("Build engines...")
    build_cmd = [
        "trtllm-build",
        f"--checkpoint_dir={ckpt_dir}",
        f"--output_dir={engine_dir}",
        "--gpt_attention_plugin=float16",
        "--remove_input_padding=enable",
        "--gemm_plugin=float16",
        f"--max_beam_width={num_beams}",
    ]

    check_call(" ".join(build_cmd), shell=True, env=llm_venv._new_env)

    print("Run inference")

    summary_cmd = generate_summary_cmd(llama_example_root,
                                       hf_model_dir=llama_model_root,
                                       data_type="fp16",
                                       num_beams=num_beams,
                                       tensorrt_llm_rouge1_threshold=20.2 if
                                       weight_only_precision == 'int8' else 16,
                                       engine_dir=engine_dir,
                                       dataset_dir=llm_datasets_root,
                                       rouge_dir=llm_rouge_root)

    venv_check_call(llm_venv, summary_cmd)


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
@pytest.mark.parametrize("llama_model_root", [
    'llama-v2-7b-hf', 'llama-v3-8b-instruct-hf', 'llama-3.1-8b', 'llama-3.2-1b',
    'llama-3.2-3b'
],
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

    expected_outputs = {
        'llama-v3-8b-instruct-hf': [
            " I hope you're having a great day! I just wanted to reach out and say hi, and see if you're doing okay. I know things",
            " Seattle, Washington is known for its mild and wet climate, with over 200 days of precipitation per year. The city experiences a significant amount of rainfall",
            " No, it is not recommended to fill diesel in a petrol car. Diesel and petrol are two different types of fuel, and using the wrong type of",
            " I'm curious to know what's currently popular.\nI can help you with that! As of now, the top 5 trending songs on Spotify are",
            " Paris\nWhat is the capital of Germany? Berlin\nWhat is the capital of Italy? Rome\nWhat is the capital of Spain? Madrid\nWhat"
        ],
        'llama-3.1-8b-instruct': [
            " I'm doing pretty well, thanks for asking. I just got back from a great vacation in Hawaii and I'm still feeling pretty relaxed. I'm",
            " Seattle, Washington is known for its rainy and overcast weather, but the city's climate is actually quite mild and temperate. The city experiences a",
            " | What happens if you put diesel in a petrol car?\nFilling a petrol car with diesel is a common mistake that can cause serious damage to the",
            " I need to know what's hot right now.\nI can check the top 5 trending songs on Spotify for you. However, please note that the",
            " Paris\nWhat is the capital of France?\nThe capital of France is Paris. Paris is the largest city in France and is known for its iconic landmarks"
        ],
        'llama-3.2-1b-instruct': [
            " I'm doing great, thanks for asking! I just got back from a fantastic weekend getaway to the beach, and I'm feeling refreshed and rejuvenated",
            " Right now?\nI'm planning a trip to Seattle and I want to know what the weather is like. I'm looking for a general idea of what",
            " Filling a diesel car with petrol is not recommended, and it can cause serious damage to the engine. Diesel and petrol are two different types of fuel",
            " based on the last 24 hours?\nI can provide you with the top 5 trending songs on Spotify based on the last 24 hours, but",
            " Paris.\nThe capital of France is Paris. Paris is the most populous city in France and is known for its rich history, art, fashion, and"
        ],
        'llama-3.2-3b-instruct': [
            " I'm doing alright, just got back from a long hike and I'm feeling pretty exhausted. Nothing like a good hike to clear the mind and get",
            " (Current Weather)\nI'm happy to help you with the current weather in Seattle, WA! However, I'm a large language model, I don",
            " and what are the types of fuel that can be used in a diesel engine?\nDiesel engines are designed to run on diesel fuel, which is a",
            " and provide the 5 most popular artists on Spotify?\nAccording to Spotify's current charts, here are the top 5 trending songs and the 5",
            " Paris\nWhat is the capital of France?\nThe capital of France is indeed Paris. Located in the north-central part of the country, Paris is a"
        ],
        'llama-3.3-70b-instruct': [
            " I hope you are having a great day. I am doing well, thanks for asking. I was just thinking about how much I love the fall season",
            " Is it always rainy?\nSeattle, WA is known for its overcast and rainy weather, but it's not always rainy. The city experiences a mild",
            " No, it is not recommended to fill diesel in a petrol car. Diesel fuel is not designed to be used in petrol engines, and using it can",
            " I want to know what's popular right now.\nAs of my knowledge cutoff, I don't have real-time access to current Spotify trends. However,",
            " Paris\nWhat is the capital of Germany? Berlin\nWhat is the capital of Italy? Rome\nWhat is the capital of Spain? Madrid\nWhat"
        ],
    }

    print("Testing with LLM-API Torch backend...")

    defs.ci_profiler.start("test_llm_torch_multi_lora_support")

    model_name = os.path.basename(llama_model_root).lower()
    test_llm_torch_multi_lora_support(
        hf_model_dir=llama_model_root,
        llm_venv=llm_venv,
        num_loras=2,
        lora_rank=8,
        target_hf_modules=["q_proj", "k_proj", "v_proj"],
        zero_lora_weights=True,
        tensor_parallel_size=tensor_parallel_size,
        expected_outputs=expected_outputs[model_name])
    defs.ci_profiler.stop("test_llm_torch_multi_lora_support")
    print(
        f"test_llm_torch_multi_lora_support: {defs.ci_profiler.elapsed_time_in_sec('test_llm_torch_multi_lora_support')} sec"
    )
