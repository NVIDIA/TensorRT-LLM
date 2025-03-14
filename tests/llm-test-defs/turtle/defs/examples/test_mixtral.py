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
import uuid

import pytest
from defs.common import (convert_weights, generate_mmlu_cmd,
                         generate_summary_cmd, quantize_data, venv_check_call,
                         venv_mpi_check_call)
from defs.conftest import (evaltool_mmlu_post_process,
                           evaltool_wikilingua_post_process, llm_models_root,
                           skip_pre_ada, skip_pre_blackwell)
from defs.trt_test_alternative import check_call
from evaltool.constants import (EVALTOOL_INFERENCE_SERVER_STARTUP_SCRIPT,
                                EVALTOOL_INFERENCE_SERVER_STOP_SCRIPT,
                                EVALTOOL_MMLU_CONFIG, EVALTOOL_MMLU_RESULT_FILE,
                                EVALTOOL_WIKILINGUA_CONFIG,
                                EVALTOOL_WIKILINGUA_RESULT_FILE)


@pytest.mark.skip_less_device(2)
@pytest.mark.skip_less_device_memory(80000)
@pytest.mark.parametrize("num_beams", [1, 4],
                         ids=lambda num_beams: f'nb:{num_beams}')
@pytest.mark.parametrize("weight_only_precision", ["int4", "int8"])
@pytest.mark.parametrize("llm_mixtral_model_root", ['Mixtral-8x7B-v0.1'],
                         indirect=True)
def test_llm_mixtral_wo_2gpus_summary(llama_example_root,
                                      llm_mixtral_model_root, llm_datasets_root,
                                      llm_rouge_root, llm_venv, cmodel_dir,
                                      engine_dir, num_beams,
                                      weight_only_precision):
    "run mixtral on 2gpus"
    model_name = 'mixtral'

    ckpt_dir = convert_weights(llm_venv=llm_venv,
                               example_root=llama_example_root,
                               cmodel_dir=cmodel_dir,
                               model=model_name,
                               model_path=llm_mixtral_model_root,
                               data_type="float16",
                               use_weight_only=True,
                               weight_only_precision=weight_only_precision,
                               tp_size=2,
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
        "--workers=2",
    ]

    check_call(" ".join(build_cmd), shell=True, env=llm_venv._new_env)

    print("Run inference")
    thresholds = {'int8': 22.0, 'int4': 18.0}

    summary_cmd = generate_summary_cmd(
        llama_example_root,
        hf_model_dir=llm_mixtral_model_root,
        data_type="fp16",
        num_beams=num_beams,
        tensorrt_llm_rouge1_threshold=thresholds[weight_only_precision],
        engine_dir=engine_dir,
        dataset_dir=llm_datasets_root,
        rouge_dir=llm_rouge_root)

    venv_mpi_check_call(llm_venv, ["mpirun", "-n", "2", "--allow-run-as-root"],
                        summary_cmd)


@skip_pre_ada
@pytest.mark.parametrize("model_name", ['Mixtral-8x7B-Instruct-v0.1-fp8'])
def test_llm_mixtral_4gpus_fp8_mmlu_llmapi(
    mmlu_dataset_root,
    llmapi_example_root,
    model_name,
    llm_venv,
):
    models_root = llm_models_root()
    model_dir = os.path.join(models_root, model_name)

    print("Run MMLU test")
    mmlu_cmd = [
        f"{llmapi_example_root}/../mmlu_llmapi.py",
        "--data_dir",
        f"{mmlu_dataset_root}",
        "--hf_model_dir",
        f"{model_dir}",
        "--backend=tensorrt",
        "--check_accuracy",
        "--tp_size=4",
        f"--accuracy_threshold=0.695",
    ]

    venv_check_call(llm_venv, mmlu_cmd)


@skip_pre_ada
@pytest.mark.skip_less_device(4)
@pytest.mark.skip_less_device_memory(80000)
@pytest.mark.parametrize("num_beams", [1, 4],
                         ids=lambda num_beams: f'nb:{num_beams}')
@pytest.mark.parametrize("llm_mixtral_model_root",
                         ['Mixtral-8x7B-v0.1', 'Mixtral-8x22B-v0.1'],
                         indirect=True)
def test_llm_mixtral_fp8_4gpus_summary(llama_example_root,
                                       llm_mixtral_model_root,
                                       llm_datasets_root, llm_rouge_root,
                                       llm_venv, engine_dir, num_beams,
                                       qcache_dir):
    "run mixtral fp8 on 4gpus"
    data_type = "bfloat16"
    tp_size, pp_size = 2, 2
    world_size = tp_size * pp_size

    print("Quantizing engine...")
    # Quantize HF llama checkpoint into FP8 format
    model_dir = quantize_data(
        llm_venv,
        llama_example_root,
        model_dir=llm_mixtral_model_root,
        calib_dataset=f"{llm_datasets_root}/cnn_dailymail",
        dtype=data_type,
        qformat="fp8",
        quantize_dir=qcache_dir,
        tp_size=tp_size,
        pp_size=pp_size,
        calib_size=32)

    print("Build engines...")
    build_cmd = [
        "trtllm-build",
        f"--checkpoint_dir={model_dir}",
        f"--output_dir={engine_dir}",
        f"--gpt_attention_plugin={data_type}",
        f"--gemm_plugin={data_type}",
        f"--moe_plugin={data_type}",
        "--remove_input_padding=enable",
        f"--max_beam_width={num_beams}",
        "--max_input_len=2048",
        "--max_seq_len=4096",
        f"--workers={world_size}",
        "--use_paged_context_fmha=enable",
    ]

    check_call(" ".join(build_cmd), shell=True, env=llm_venv._new_env)

    print("Run summarize...")
    tensorrt_llm_rouge1_threshold = 21.5
    summary_cmd = generate_summary_cmd(
        llama_example_root,
        hf_model_dir=llm_mixtral_model_root,
        data_type="fp16",
        num_beams=num_beams,
        tensorrt_llm_rouge1_threshold=tensorrt_llm_rouge1_threshold,
        engine_dir=engine_dir,
        dataset_dir=llm_datasets_root,
        rouge_dir=llm_rouge_root)

    venv_mpi_check_call(
        llm_venv, ["mpirun", "-n", f"{world_size}", "--allow-run-as-root"],
        summary_cmd)

    print("Run mmlu...")
    mmlu_cmd = generate_mmlu_cmd(llama_example_root,
                                 tokenizer_dir=llm_mixtral_model_root,
                                 engine_dir=engine_dir,
                                 accuracy_threshold=0.70,
                                 num_beams=num_beams,
                                 data_dir=f"{llm_datasets_root}/mmlu")

    venv_mpi_check_call(
        llm_venv, ["mpirun", "-n", f"{world_size}", "--allow-run-as-root"],
        mmlu_cmd)


@pytest.mark.skip_less_device(4)
@pytest.mark.skip_less_device_memory(80000)
@pytest.mark.parametrize("llm_mixtral_model_root", ['Mixtral-8x7B-v0.1'],
                         indirect=True)
def test_llm_mixtral_fp8_managed_weights_4gpus_summary(llama_example_root,
                                                       llm_mixtral_model_root,
                                                       llm_datasets_root,
                                                       llm_rouge_root, llm_venv,
                                                       engine_dir, qcache_dir):
    data_type = "bfloat16"
    tp_size, pp_size = 2, 2
    world_size = tp_size * pp_size

    print("Quantizing engine...")
    # Quantize HF llama checkpoint into FP8 format
    model_dir = quantize_data(
        llm_venv,
        llama_example_root,
        model_dir=llm_mixtral_model_root,
        calib_dataset=f"{llm_datasets_root}/cnn_dailymail",
        dtype=data_type,
        qformat="fp8",
        quantize_dir=qcache_dir,
        tp_size=tp_size,
        pp_size=pp_size,
        calib_size=32)

    print("Build engines...")
    build_cmd = [
        "trtllm-build", f"--checkpoint_dir={model_dir}",
        f"--output_dir={engine_dir}", f"--gpt_attention_plugin={data_type}",
        f"--gemm_plugin={data_type}", f"--moe_plugin={data_type}",
        "--remove_input_padding=enable", f"--max_beam_width=1",
        "--max_input_len=2048", "--max_seq_len=4096", f"--worker={world_size}",
        "--fast_build"
    ]

    check_call(" ".join(build_cmd), shell=True, env=llm_venv._new_env)

    print("Run summarize...")
    tensorrt_llm_rouge1_threshold = 21.5
    summary_cmd = generate_summary_cmd(
        llama_example_root,
        hf_model_dir=llm_mixtral_model_root,
        data_type="fp16",
        num_beams=1,
        tensorrt_llm_rouge1_threshold=tensorrt_llm_rouge1_threshold,
        engine_dir=engine_dir,
        dataset_dir=llm_datasets_root,
        rouge_dir=llm_rouge_root)

    venv_mpi_check_call(
        llm_venv, ["mpirun", "-n", f"{world_size}", "--allow-run-as-root"],
        summary_cmd)

    print("Run mmlu...")
    mmlu_cmd = generate_mmlu_cmd(llama_example_root,
                                 tokenizer_dir=llm_mixtral_model_root,
                                 engine_dir=engine_dir,
                                 accuracy_threshold=0.70,
                                 num_beams=1,
                                 data_dir=f"{llm_datasets_root}/mmlu")

    venv_mpi_check_call(
        llm_venv, ["mpirun", "-n", f"{world_size}", "--allow-run-as-root"],
        mmlu_cmd)


@pytest.mark.skip_less_device(4)
@pytest.mark.parametrize("llm_mixtral_model_root", ['Mixtral-8x7B-v0.1'],
                         indirect=True)
def test_llm_mixtral_v1_smooth_quant_4gpus(llama_example_root,
                                           llm_mixtral_model_root,
                                           llm_datasets_root, llm_rouge_root,
                                           llm_venv, cmodel_dir, engine_dir):
    "Run smooth quant test on 4 gpus"
    data_type = "float16"
    model_dir = convert_weights(
        llm_venv=llm_venv,
        example_root=llama_example_root,
        cmodel_dir=cmodel_dir,
        model="mixtral-sq",
        model_path=llm_mixtral_model_root,
        tp_size=2,
        pp_size=2,
        smoothquant=0.5,
        per_channel=True,
        per_token=True,
        data_type=data_type,
        calib_dataset=f"{llm_datasets_root}/ccdv/cnn_dailymail",
        workers=4)

    print("Build engines...")
    build_cmd = [
        "trtllm-build",
        f"--checkpoint_dir={model_dir}",
        f"--output_dir={engine_dir}",
        f"--gpt_attention_plugin={data_type}",
        f"--gemm_plugin={data_type}",
        "--max_input_len=1024",
        "--max_batch_size=1",
        "--context_fmha=enable",
        "--max_beam_width=4",
        "--workers=4",
    ]

    check_call(" ".join(build_cmd), shell=True, env=llm_venv._new_env)

    print("Run summarize...")
    summary_cmd = generate_summary_cmd(llama_example_root,
                                       hf_model_dir=llm_mixtral_model_root,
                                       data_type="fp16",
                                       num_beams=4,
                                       engine_dir=engine_dir,
                                       tensorrt_llm_rouge1_threshold=23,
                                       dataset_dir=llm_datasets_root,
                                       rouge_dir=llm_rouge_root)

    venv_mpi_check_call(llm_venv, ["mpirun", "-n", "4", "--allow-run-as-root"],
                        summary_cmd)


@pytest.mark.skip_less_device(8)
@pytest.mark.skip_less_device_memory(45000)
@pytest.mark.parametrize(
    "moe_tp_size", [1, 4, 8],
    ids=['expert_parallel', 'mixed_parallel', 'tensor_parallel'])
@pytest.mark.parametrize("moe_renorm_mode", [0, 1],
                         ids=['no_renormalize', 'renormalize'])
@pytest.mark.parametrize("mode", [0, 1], ids=['plugin', 'ootb_except_mha'])
@pytest.mark.parametrize("llm_mixtral_model_root",
                         ['Mixtral-8x7B-v0.1', 'Mixtral-8x22B-v0.1'],
                         indirect=True)
def test_llm_mixtral_v1_8gpus_summary(llama_example_root,
                                      llm_mixtral_model_root, llm_datasets_root,
                                      llm_rouge_root, llm_venv, cmodel_dir,
                                      engine_dir, moe_tp_size, moe_renorm_mode,
                                      mode):
    "Run test on 8 gpus with moe_renorm_mode"
    data_type = "float16"

    tp_size, pp_size = 8, 1
    world_size = tp_size * pp_size
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
                                moe_renorm_mode=moe_renorm_mode,
                                workers=world_size)
    gemm_plugin = "disable" if mode == "ootb-except-mha" else data_type
    moe_plugin = "disable" if mode == "ootb-except-mha" else data_type

    print("Build engines...")
    build_cmd = [
        "trtllm-build",
        f"--checkpoint_dir={model_dir}",
        f"--output_dir={engine_dir}",
        f"--gemm_plugin={gemm_plugin}",
        f"--moe_plugin={moe_plugin}",
        f"--workers={world_size}",
        "--max_input_len=1024",
        "--max_batch_size=1",
        "--context_fmha=enable",
        "--max_beam_width=4",
        f"--max_seq_len={8192}",
    ]

    check_call(" ".join(build_cmd), shell=True, env=llm_venv._new_env)

    print("Run summarize...")
    summary_cmd = generate_summary_cmd(llama_example_root,
                                       hf_model_dir=llm_mixtral_model_root,
                                       data_type="fp16",
                                       num_beams=4,
                                       engine_dir=engine_dir,
                                       tensorrt_llm_rouge1_threshold=21,
                                       dataset_dir=llm_datasets_root,
                                       rouge_dir=llm_rouge_root)

    venv_mpi_check_call(
        llm_venv, ["mpirun", "-n", f"{world_size}", "--allow-run-as-root"],
        summary_cmd)


@pytest.mark.skip_less_device(4)
@pytest.mark.parametrize("llm_mixtral_model_root", ['Mixtral-8x7B-v0.1'],
                         indirect=True)
def test_mixtal_evaltool(llama_example_root, evaltool_root,
                         llm_mixtral_model_root, llm_venv, engine_dir,
                         cmodel_dir):

    print("Build engines...")

    data_type = "float16"
    model_dir = convert_weights(llm_venv=llm_venv,
                                example_root=llama_example_root,
                                cmodel_dir=cmodel_dir,
                                model='mixtral',
                                model_path=llm_mixtral_model_root,
                                tp_size=4,
                                pp_size=1,
                                data_type=data_type,
                                workers=4)

    print("Build engines...")
    build_cmd = [
        "trtllm-build",
        f"--checkpoint_dir={model_dir}",
        f"--output_dir={engine_dir}",
        f"--gpt_attention_plugin={data_type}",
        f"--gemm_plugin={data_type}",
        "--gather_context_logits",
        "--max_batch_size=8",
        "--max_input_len=7000",
        "--max_seq_len=7048",
        "--workers=4",
    ]
    check_call(" ".join(build_cmd), shell=True, env=llm_venv._new_env)

    print("Human eval")
    start_inference_server = [
        EVALTOOL_INFERENCE_SERVER_STARTUP_SCRIPT, "-e", engine_dir, "-t",
        llm_mixtral_model_root, "-d", evaltool_root, "-m", "1024", "-c", "4"
    ]
    check_call(" ".join(start_inference_server),
               shell=True,
               env=llm_venv._new_env)

    task_list = ['wikilingua', 'mmlu']
    try:
        for task in task_list:
            project_id = str(uuid.uuid4())
            if task == "wikilingua":
                config_file = EVALTOOL_WIKILINGUA_CONFIG
                result_file = EVALTOOL_WIKILINGUA_RESULT_FILE

            if task == "mmlu":
                config_file = EVALTOOL_MMLU_CONFIG
                result_file = EVALTOOL_MMLU_RESULT_FILE

            # Update config dynamically
            import yaml
            with open(config_file, 'r') as f:
                lm_eval_config = yaml.safe_load(f)
                lm_eval_config['model']['llm_name'] = llm_mixtral_model_root
                lm_eval_config['model']['tokenizer_path'] = model_dir

            config_file = os.path.join(llm_venv.get_working_directory(),
                                       "lm_eval_config.yaml")
            with open(config_file, 'w') as f:
                yaml.dump(lm_eval_config, f)

            # launch evaluation
            run_cmd = [
                f"cd {evaltool_root}",
                "&&",
                "source .venv/bin/activate",
                "&&",
                "python3",
                f"evaltool/interfaces/cli/main.py",
                "project",
                "launch",
                f"--eval_project_config_file '{config_file}'",
                "--infra_name local",
                f"--output_dir '{llm_venv.get_working_directory()}'",
                f"--project_id {project_id}",
            ]
            # venv_mpi_check_call(llm_venv, [
            #     "mpirun", "--allow-run-as-root", "--oversubscribe", "-np", "4"
            # ], run_cmd)
            check_call(" ".join(run_cmd), shell=True, executable="/bin/bash")

            # process result
            result_path = f"{llm_venv.get_working_directory()}/{project_id}/{result_file}"
            check_call(f"cat {result_path}", shell=True)

            if task == 'mmlu':
                evaltool_mmlu_post_process(result_path, 0.71775, 0.006)
            if task == 'wikilingua':
                evaltool_wikilingua_post_process(result_path, 0.2776, 0.006)
    finally:
        # stop the server
        end_inference_server = [
            EVALTOOL_INFERENCE_SERVER_STOP_SCRIPT, "-c", "4"
        ]
        check_call(" ".join(end_inference_server),
                   shell=True,
                   env=llm_venv._new_env)


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
        f"{llama_example_root}/../run.py",
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
        f"{llama_example_root}/../run.py",
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
        f"{llama_example_root}/../run.py",
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
        f"{llama_example_root}/../run.py",
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


@pytest.mark.skip_less_device(4)
@pytest.mark.skip_less_device_memory(45000)
@pytest.mark.parametrize("llm_mixtral_model_root", ['Mixtral-8x7B-v0.1'],
                         indirect=True)
def test_llm_mixtral_pp_reduce_scatter_4gpus(llama_example_root,
                                             llm_mixtral_model_root,
                                             llm_datasets_root, llm_rouge_root,
                                             llm_venv, cmodel_dir, engine_dir):
    "Run PP reduce scatter test on 4 gpus"
    data_type = "float16"
    model_dir = convert_weights(llm_venv=llm_venv,
                                example_root=llama_example_root,
                                cmodel_dir=cmodel_dir,
                                model="mixtral",
                                model_path=llm_mixtral_model_root,
                                tp_size=2,
                                pp_size=2,
                                data_type=data_type,
                                workers=4)

    print("Build engines...")
    build_cmd = [
        "trtllm-build",
        f"--checkpoint_dir={model_dir}",
        f"--output_dir={engine_dir}",
        f"--gpt_attention_plugin={data_type}",
        f"--gemm_plugin={data_type}",
        "--max_input_len=1024",
        "--max_batch_size=1",
        "--context_fmha=enable",
        "--pp_reduce_scatter=enable",
        "--max_beam_width=4",
        "--workers=4",
    ]

    check_call(" ".join(build_cmd), shell=True, env=llm_venv._new_env)

    print("Run summarize...")
    summary_cmd = generate_summary_cmd(llama_example_root,
                                       hf_model_dir=llm_mixtral_model_root,
                                       data_type="fp16",
                                       num_beams=4,
                                       engine_dir=engine_dir,
                                       tensorrt_llm_rouge1_threshold=23,
                                       dataset_dir=llm_datasets_root,
                                       rouge_dir=llm_rouge_root)

    venv_mpi_check_call(llm_venv, ["mpirun", "-n", "4", "--allow-run-as-root"],
                        summary_cmd)


@skip_pre_blackwell
@pytest.mark.skip_less_device_memory(180000)
@pytest.mark.parametrize("fp4_type", ["plugin", "ootb", "disable"],
                         ids=["fp4_plugin", "fp4_ootb", "disable_fp4"])
@pytest.mark.parametrize("llm_mixtral_model_root", ['Mixtral-8x7B-v0.1'],
                         indirect=True)
def test_llm_mixtral_1gpu_fp4(
    mmlu_dataset_root,
    fp4_type,
    llama_example_root,
    llm_mixtral_model_root,
    llm_venv,
    cmodel_dir,
    engine_dir,
    qcache_dir,
    llm_datasets_root,
):
    model_name = os.path.basename(llm_mixtral_model_root)

    if fp4_type != "disable":
        model_dir = quantize_data(
            llm_venv,
            llama_example_root,
            model_dir=llm_mixtral_model_root,
            calib_dataset=f"{llm_datasets_root}/cnn_dailymail",
            dtype="float16",
            qformat="nvfp4",
            kv_cache_dtype="fp8",
            quantize_dir=qcache_dir)
    else:
        model_dir = convert_weights(llm_venv=llm_venv,
                                    example_root=llama_example_root,
                                    cmodel_dir=cmodel_dir,
                                    model=model_name,
                                    model_path=llm_mixtral_model_root,
                                    data_type='float16')
    print("Build engines...")
    build_cmd = [
        "trtllm-build", f"--checkpoint_dir={model_dir}",
        f"--output_dir={engine_dir}", "--max_input_len=2048"
    ]
    if fp4_type != "disable":
        build_cmd.extend([
            "--gemm_plugin=disable"
            if fp4_type == "ootb" else "--gemm_plugin=nvfp4"
        ])
    if fp4_type == "plugin":
        build_cmd.extend([
            "--use_paged_context_fmha=enable", "--use_fp8_context_fmha=enable"
        ])

    check_call(" ".join(build_cmd), shell=True, env=llm_venv._new_env)

    print("Run MMLU test")
    acc_thres = 0.680
    mmlu_cmd = [
        f"{llama_example_root}/../mmlu.py",
        "--data_dir",
        f"{mmlu_dataset_root}",
        "--hf_model_dir",
        f"{llm_mixtral_model_root}",
        "--test_trt_llm",
        f"--engine_dir={engine_dir}",
        "--check_accuracy",
        f"--accuracy_threshold={acc_thres}",
    ]

    venv_check_call(llm_venv, mmlu_cmd)


@skip_pre_blackwell
@pytest.mark.parametrize("model_name", ['Mixtral-8x7B-Instruct-v0.1'])
def test_llm_mixtral_1gpu_fp4_llmapi(
    mmlu_dataset_root,
    llmapi_example_root,
    model_name,
    llm_venv,
):
    models_root = llm_models_root()
    model_dir = os.path.join(models_root, "nvfp4-quantized", model_name)

    print("Run MMLU test")
    mmlu_cmd = [
        f"{llmapi_example_root}/../mmlu_llmapi.py",
        "--data_dir",
        f"{mmlu_dataset_root}",
        "--hf_model_dir",
        f"{model_dir}",
        "--backend=tensorrt",
        "--check_accuracy",
        f"--accuracy_threshold=0.680",
    ]

    venv_check_call(llm_venv, mmlu_cmd)


@pytest.mark.parametrize("model_name", ['mixtral-8x7b-v0.1-AWQ'])
def test_llm_mixtral_int4_awq_1gpu_summary(llama_example_root,
                                           llm_datasets_root, model_name,
                                           llm_rouge_root, llm_venv, cmodel_dir,
                                           engine_dir):
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
