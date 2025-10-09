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
from defs.common import convert_weights, venv_check_call
from defs.conftest import get_sm_version, skip_post_blackwell
from defs.trt_test_alternative import check_call

# skip trt flow cases on post-Blackwell-Ultra
if get_sm_version() >= 103:
    pytest.skip(
        "TRT workflow tests are not supported on post Blackwell-Ultra architecture",
        allow_module_level=True)


@skip_post_blackwell
@pytest.mark.parametrize("use_cpp_runtime", [True, False],
                         ids=["use_cpp_runtime", "use_python_runtime"])
@pytest.mark.parametrize("num_beams", [1, 4],
                         ids=lambda num_beams: f'nb:{num_beams}')
@pytest.mark.parametrize("data_type", ['float16'])
@pytest.mark.parametrize("weight_only_precision", [
    'disable_weight_only',
    pytest.param('int8', marks=skip_post_blackwell),
    pytest.param('int4', marks=skip_post_blackwell)
])
@pytest.mark.parametrize(
    "use_attention_plugin", [True, False],
    ids=["enable_attention_plugin", "disable_attention_plugin"])
@pytest.mark.parametrize("use_gemm_plugin", [True, False],
                         ids=["enable_gemm_plugin", "disable_gemm_plugin"])
@pytest.mark.parametrize("whisper_model_root", ['large-v3', 'large-v2'],
                         indirect=True)
def test_llm_whisper_general(llm_venv, engine_dir, data_type,
                             weight_only_precision, use_attention_plugin,
                             use_gemm_plugin, whisper_example_root,
                             whisper_model_root, num_beams, use_cpp_runtime,
                             whisper_example_audio_file, llm_datasets_root):
    print("Locate model checkpoints in test storage...")
    tllm_model_name, model_ckpt_dir = whisper_model_root

    if any((not use_attention_plugin, use_gemm_plugin, 'v3'
            not in tllm_model_name)) and use_cpp_runtime:
        pytest.skip(f"Plugins might not support C++ runtime. Skip the test...")

    whisper_engine_dir = f"{engine_dir}/{tllm_model_name}/{data_type}_{weight_only_precision}"

    if 'int' in weight_only_precision:
        use_weight_only = True
    else:
        use_weight_only = False
        weight_only_precision = None
    converted_weight_dir = convert_weights(
        llm_venv=llm_venv,
        example_root=whisper_example_root,
        cmodel_dir=whisper_engine_dir,
        model=tllm_model_name,
        model_path=model_ckpt_dir,
        use_weight_only=use_weight_only,
        weight_only_precision=weight_only_precision)
    print("Build engines...")
    for component in ["encoder", "decoder"]:
        build_cmd = [
            "trtllm-build",
            f"--checkpoint_dir={converted_weight_dir}/{component}",
            f"--output_dir={whisper_engine_dir}/{component}",
            "--paged_kv_cache=disable",
            "--moe_plugin=disable",
            "--max_batch_size=8",
        ]
        if use_cpp_runtime:
            build_cmd.extend(
                ("--paged_kv_cache enable", "--remove_input_padding enable"))
        else:
            build_cmd.append("--remove_input_padding=disable")

        if component == "encoder":
            build_cmd.append(
                f"--max_input_len=3000"
            )  # check against actual encoder features length (3000,...) in C++ runtime
            build_cmd.append(f"--max_seq_len=3000")
        if component == "decoder":
            build_cmd.append(f"--max_input_len=14")
            build_cmd.append(f"--max_seq_len=114")
            build_cmd.append(f"--max_encoder_input_len=3000")
            build_cmd.append(f"--max_beam_width={num_beams}")

        if use_gemm_plugin:
            build_cmd.append(f"--gemm_plugin={data_type}")
        else:
            build_cmd.append(f"--gemm_plugin=disable")

        if use_attention_plugin:
            build_cmd.append(f"--bert_attention_plugin={data_type}")
            build_cmd.append(f"--gpt_attention_plugin={data_type}")
        else:
            build_cmd.append(f"--bert_attention_plugin=disable")
            build_cmd.append(f"--gpt_attention_plugin=disable")

        check_call(" ".join(build_cmd), shell=True, env=llm_venv._new_env)

    if use_cpp_runtime:
        print("Run inference using Python bindings of C++ runtime...")
        run_cmd = [
            f'{whisper_example_root}/../../../run.py',
            f'--multimodal_input_file={whisper_example_audio_file}',
            f'--engine_dir={whisper_engine_dir}',
            f'--max_output_len=96',
        ]
    else:
        print("Run inference using Whisper's custom Python runtime...")
        run_cmd = [
            f"{whisper_example_root}/run.py",
            f"--dataset={llm_datasets_root}/hf-internal-testing/librispeech_asr_dummy",
            f"--engine_dir={whisper_engine_dir}",
            f"--assets_dir={model_ckpt_dir}",
            f"--num_beams={num_beams}",
            f"--dtype={data_type}",
            f"--use_py_session",
            f"--accuracy_check",
        ]
    # https://nvbugs/4658787
    # WAR before whisper tests can work offline
    env = {"HF_DATASETS_OFFLINE": "0"}
    venv_check_call(llm_venv, run_cmd, env=env)
