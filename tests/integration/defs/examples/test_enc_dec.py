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

import pytest
from defs.common import (convert_weights, quantize_data, venv_check_call,
                         venv_mpi_check_call)
from defs.conftest import (get_device_count, get_sm_version, skip_fp8_pre_ada,
                           skip_post_blackwell)
from defs.trt_test_alternative import check_call

# skip trt flow cases on post-Blackwell-Ultra
if get_sm_version() >= 103:
    pytest.skip(
        "TRT workflow tests are not supported on post Blackwell-Ultra architecture",
        allow_module_level=True)


@pytest.mark.parametrize("use_fp8", [True, False],
                         ids=["enable_fp8", "disable_fp8"])
@pytest.mark.parametrize("num_beams", [1, 2, 3],
                         ids=lambda num_beams: f'nb:{num_beams}')
@pytest.mark.parametrize("pp_size", [1, 2], ids=lambda pp_size: f'pp:{pp_size}')
@pytest.mark.parametrize("tp_size", [1, 2], ids=lambda tp_size: f'tp:{tp_size}')
@pytest.mark.parametrize(
    "use_paged_kv_cache", [True, False],
    ids=["enable_paged_kv_cache", "disable_paged_kv_cache"])
@pytest.mark.parametrize(
    "use_attention_plugin",
    [pytest.param(True, marks=skip_post_blackwell), False],
    ids=["enable_attention_plugin", "disable_attention_plugin"])
@pytest.mark.parametrize("use_gemm_plugin", [True, False],
                         ids=["enable_gemm_plugin", "disable_gemm_plugin"])
@pytest.mark.parametrize("data_type", ['bfloat16', 'float16', 'float32'])
@pytest.mark.parametrize("enc_dec_model_root", [
    pytest.param('t5-small', marks=skip_post_blackwell),
    pytest.param('flan-t5-small', marks=skip_post_blackwell),
    pytest.param('byt5-small', marks=skip_post_blackwell), 'bart-large-cnn',
    pytest.param('mbart-large-50-many-to-one-mmt', marks=skip_post_blackwell),
    'wmt14'
],
                         indirect=True)
@pytest.mark.parametrize("compare_hf_fp32", [True, False],
                         ids=["compare_hf", "no_compare_hf"])
def test_llm_enc_dec_general(llm_venv, cmodel_dir, engine_dir, data_type,
                             use_attention_plugin, use_gemm_plugin,
                             enc_dec_example_root, enc_dec_model_root, tp_size,
                             pp_size, num_beams, compare_hf_fp32,
                             use_paged_kv_cache, use_fp8, llm_datasets_root):

    world_size = tp_size * pp_size

    if get_device_count() < world_size:
        pytest.skip(
            f"Running world size {world_size} on a node with only {get_device_count()} devices. Skip the test..."
        )

    skip_fp8_pre_ada(use_fp8)

    print("Locate model checkpoints in test storage...")
    tllm_model_name, model_ckpt_path = enc_dec_model_root

    print("Converting Encoder-Decoder model into binary format...")
    # ckpt from llm_models/<model_name> --> cmodels/<model_name>/<dtype>
    model_name = tllm_model_name
    model_type = None
    if "t5" in model_name or "ul2" in model_name:
        if data_type != "float32":
            pytest.skip("transformer:issue/34264")
        model_type = "t5"
    elif "bart" in model_name:
        model_type = "bart"
    elif "wmt" in model_name:
        model_type = "nmt"

    if use_fp8:
        assert use_paged_kv_cache and use_attention_plugin
        # a known apex huggingface bug for t5 only
        # t5 only takes float32 in quantization loop
        # https://github.com/huggingface/transformers/issues/34264
        converted_weight_dir = quantize_data(
            llm_venv,
            enc_dec_example_root,
            model_dir=model_ckpt_path,
            dtype=data_type,
            quantize_dir=cmodel_dir,
            qformat="fp8",
            tp_size=tp_size,
            pp_size=pp_size,
            kv_cache_dtype="fp8",
            batch_size=1,
            calib_dataset=f"{llm_datasets_root}/cnn_dailymail")

        enc_dec_engine_dir = f"{engine_dir}/{tllm_model_name}/{world_size}-gpu/fp8"
    else:
        converted_weight_dir = convert_weights(
            llm_venv=llm_venv,
            example_root=enc_dec_example_root,
            cmodel_dir=cmodel_dir,
            model=model_name,
            model_path=model_ckpt_path,
            data_type=data_type,
            tp_size=tp_size,
            pp_size=pp_size,
            model_type=model_type)

        enc_dec_engine_dir = f"{engine_dir}/{tllm_model_name}/{world_size}-gpu/{data_type}"

    print("Build engines...")

    # change plugins precision to auto if testing fp8
    data_type = "auto" if use_fp8 else data_type

    for component in ["encoder", "decoder"]:
        build_cmd = [
            "trtllm-build",
            f"--checkpoint_dir={converted_weight_dir}/{component}",
            f"--output_dir={enc_dec_engine_dir}/{component}",
            f"--max_beam_width={num_beams}",
            "--moe_plugin=disable",
            "--max_batch_size=8",
        ]

        if component == "encoder":
            build_cmd.append(f"--max_input_len=512")
        else:
            build_cmd.append(f"--max_input_len=1")
            build_cmd.append(f"--max_seq_len=201")
            build_cmd.append(f"--max_encoder_input_len=512")

        if use_paged_kv_cache and component == "decoder":
            # paged_kv_cache only applies to decoder component
            # As for now, we only support num_beams=1 for decoder paged kv cache in python runtime
            build_cmd.append(f"--paged_kv_cache=enable")
        else:
            build_cmd.append(f"--paged_kv_cache=disable")

        if use_gemm_plugin:
            build_cmd.append(f"--gemm_plugin={data_type}")
        else:
            build_cmd.append(f"--gemm_plugin=disable")

        if use_attention_plugin:
            # TODO: remove skip after support bert_attention_plugin on B200
            build_cmd.append(f"--bert_attention_plugin={data_type}")
            build_cmd.append(f"--gpt_attention_plugin={data_type}")
            build_cmd.append("--remove_input_padding=enable")

            # for non-T5 models, FP16/BF16
            if model_type == "t5" or data_type == "float32":
                build_cmd.append("--context_fmha=disable")
            elif use_fp8:
                build_cmd.append("--use_fp8_context_fmha=enable")
        else:
            build_cmd.append(f"--bert_attention_plugin=disable")
            build_cmd.append(f"--gpt_attention_plugin=disable")
            build_cmd.append("--remove_input_padding=disable")

        check_call(" ".join(build_cmd), shell=True, env=llm_venv._new_env)

    print("Run inference...")
    if use_paged_kv_cache and pp_size == 1:
        # use paged engines to cover ModelRunnerCpp tests
        run_cmd = [
            f"{enc_dec_example_root}/../../../run.py",
            f"--engine_dir={enc_dec_engine_dir}",
            f"--tokenizer_dir={model_ckpt_path}",
            "--max_output_len=24",
            f"--num_beams={num_beams}",
            "--input_text='translate English to German: The house is wonderful.'",
        ]
    else:
        # old Python runtime tests
        run_cmd = [
            f"{enc_dec_example_root}/run.py",
            f"--engine_dir={enc_dec_engine_dir}",
            f"--engine_name={model_name}",
            f"--model_name={model_ckpt_path}",  # use ckpt path so we can use local copy rather than cloning from HF
            "--max_new_tokens=24",  # shorter than 3rd example input length to capture any bug
            f"--num_beams={num_beams}",
        ]
        if compare_hf_fp32:
            run_cmd.extend(["--compare_hf_fp32"])

    if world_size == 1:
        venv_check_call(llm_venv, run_cmd)
    else:
        venv_mpi_check_call(
            llm_venv, ["mpirun", "-n",
                       str(world_size), "--allow-run-as-root"], run_cmd)


@pytest.mark.parametrize("use_fp8", [True, False],
                         ids=["enable_fp8", "disable_fp8"])
@pytest.mark.parametrize("num_beams", [1, 2, 3],
                         ids=lambda num_beams: f'nb:{num_beams}')
@pytest.mark.parametrize("pp_size", [1, 2], ids=lambda pp_size: f'pp:{pp_size}')
@pytest.mark.parametrize("tp_size", [1, 2], ids=lambda tp_size: f'tp:{tp_size}')
@pytest.mark.parametrize("data_type", ['bfloat16', 'float16', 'float32'])
@pytest.mark.parametrize("enc_dec_model_root", ['flan-t5-small', 'flan-t5-xl'],
                         indirect=True)
def test_llm_enc_dec_mmlu(llm_venv, cmodel_dir, engine_dir, data_type,
                          enc_dec_example_root, enc_dec_model_root, tp_size,
                          pp_size, num_beams, mmlu_dataset_root, use_fp8,
                          llm_datasets_root):

    world_size = tp_size * pp_size

    if get_device_count() < world_size:
        pytest.skip(
            f"Running world size {world_size} on a node with only {get_device_count()} devices. Skip the test..."
        )

    skip_fp8_pre_ada(use_fp8)

    print("Locate model checkpoints in test storage...")
    tllm_model_name, model_ckpt_path = enc_dec_model_root

    print("Converting Encoder-Decoder model into binary format...")
    # ckpt from llm_models/<model_name> --> cmodels/<model_name>/<dtype>
    model_name = tllm_model_name
    model_type = None
    if "t5" in model_name or "ul2" in model_name:
        model_type = "t5"
    elif "bart" in model_name:
        model_type = "bart"
    elif "wmt" in model_name:
        model_type = "nmt"

    if use_fp8:
        # a known apex huggingface bug for t5 only
        # t5 only takes float32 in quantization loop
        # https://github.com/huggingface/transformers/issues/34264
        converted_weight_dir = quantize_data(
            llm_venv,
            enc_dec_example_root,
            model_dir=model_ckpt_path,
            dtype=data_type,
            quantize_dir=cmodel_dir,
            qformat="fp8",
            tp_size=tp_size,
            pp_size=pp_size,
            kv_cache_dtype="fp8",
            batch_size=1,
            calib_dataset=f"{llm_datasets_root}/cnn_dailymail")

        enc_dec_engine_dir = f"{engine_dir}/{tllm_model_name}/{world_size}-gpu/fp8"
    else:
        converted_weight_dir = convert_weights(
            llm_venv=llm_venv,
            example_root=enc_dec_example_root,
            cmodel_dir=cmodel_dir,
            model=model_name,
            model_path=model_ckpt_path,
            data_type=data_type,
            tp_size=tp_size,
            pp_size=pp_size,
            model_type=model_type)

        enc_dec_engine_dir = f"{engine_dir}/{tllm_model_name}/{world_size}-gpu/{data_type}"

    print("Build engines...")

    max_input_len = 2048

    # change plugins precision to auto if testing fp8
    data_type = "auto" if use_fp8 else data_type

    for component in ["encoder", "decoder"]:
        build_cmd = [
            "trtllm-build",
            f"--checkpoint_dir={converted_weight_dir}/{component}",
            f"--output_dir={enc_dec_engine_dir}/{component}",
            f"--max_beam_width={num_beams}",
            "--moe_plugin=disable",
            "--max_batch_size=8",
        ]

        if component == "encoder":
            build_cmd.append(f"--max_input_len={max_input_len}")
        else:
            build_cmd.append(f"--max_input_len=1")
            build_cmd.append(f"--max_seq_len=201")
            build_cmd.append(f"--max_encoder_input_len={max_input_len}")

        if component == "decoder":
            # paged_kv_cache only applies to decoder component
            # As for now, we only support num_beams=1 for decoder paged kv cache in python runtime
            build_cmd.append(f"--paged_kv_cache=enable")

        build_cmd.append(f"--gemm_plugin={data_type}")
        build_cmd.append(f"--bert_attention_plugin={data_type}")
        build_cmd.append(f"--gpt_attention_plugin={data_type}")
        build_cmd.append("--remove_input_padding=enable")

        # for non-T5 models, FP16/BF16
        if model_type == "t5" or data_type == "float32":
            build_cmd.append("--context_fmha=disable")
        elif use_fp8:
            build_cmd.append("--use_fp8_context_fmha=enable")
        check_call(" ".join(build_cmd), shell=True, env=llm_venv._new_env)

    print("Run MMLU test")
    accuracy_threshold_map = {
        "flan-t5-xl": {
            "float32": 0.440,  # 0.444
        },
        "flan-t5-small": {
            "float32": 0.280,  # 0.282
            "float16": 0.280,  # 0.283
            "float8": 0.280,  # 0.284
        }
    }
    precision = "float8" if use_fp8 else data_type
    accuracy_threshold = accuracy_threshold_map[tllm_model_name][precision]

    mmlu_cmd = [
        f"{enc_dec_example_root}/../../../mmlu.py",
        f"--data_dir={mmlu_dataset_root}",
        f"--hf_model_dir={model_ckpt_path}",
        "--test_trt_llm",
        f"--engine_dir={enc_dec_engine_dir}",
        "--kv_cache_free_gpu_memory_fraction=0.45",
        "--cross_kv_cache_fraction=0.45",
        "--check_accuracy",
        f"--accuracy_threshold={accuracy_threshold}",
    ]

    venv_check_call(llm_venv, mmlu_cmd)
