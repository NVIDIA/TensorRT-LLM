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
"""Module test_bert test bert examples."""
import pytest
from defs.common import convert_weights, venv_check_call, venv_mpi_check_call
from defs.conftest import get_device_count, get_sm_version
from defs.trt_test_alternative import check_call

# skip trt flow cases on post-Blackwell-Ultra
if get_sm_version() >= 103:
    pytest.skip(
        "TRT workflow tests are not supported on post Blackwell-Ultra architecture",
        allow_module_level=True)


# # Build parameters
@pytest.mark.parametrize(
    "model, hf_bert_model_root",
    [("BertModel", 'bert/bert-base-uncased'),
     ("BertForQuestionAnswering", 'bert/bert-base-cased-squad2'),
     ("BertForSequenceClassification", 'bert/bert-base-uncased-yelp-polarity'),
     ("RobertaModel", 'bert/roberta-base'),
     ("RobertaForQuestionAnswering", 'bert/roberta-base-squad2'),
     ("RobertaForSequenceClassification", 'bert/twitter-roberta-base-emotion')])
@pytest.mark.parametrize("dtype", ["float32", "float16"])
@pytest.mark.parametrize("pp_size", [1], ids=lambda pp_size: f'pp:{pp_size}')
@pytest.mark.parametrize("tp_size", [1, 2], ids=lambda tp_size: f'tp:{tp_size}')
@pytest.mark.parametrize(
    "use_attention_plugin, context_fmha_type", [(True, 'enabled'),
                                                (True, 'enabled_with_fp32_acc'),
                                                (True, 'disabled'),
                                                (False, 'disabled')],
    ids=[
        'use_attention_plugin-enable_context_fmha',
        'use_attention_plugin-enable_context_fmha_fp32_acc',
        'use_attention_plugin-disable_context_fmha',
        'disable_attention_plugin-disable_context_fmha',
    ])
@pytest.mark.parametrize(
    "remove_input_padding", [True, False],
    ids=["enable_remove_input_padding", "disable_remove_input_padding"])
# Run parameters
@pytest.mark.parametrize("compare_hf", [True], ids=["compare_hf"])
def test_llm_bert_general(bert_example_root, llm_venv, model, dtype, pp_size,
                          tp_size, use_attention_plugin, context_fmha_type,
                          hf_bert_model_root, bert_model_root, compare_hf,
                          cmodel_dir, engine_dir, remove_input_padding):
    "Run bert for float16 and float32"
    world_size = tp_size * pp_size

    if get_device_count() < world_size:
        pytest.skip(
            f"Running world size {world_size} on a node with only {get_device_count()} devices. Skip the test..."
        )

    print("Locate model checkpoints in test storage...")
    hf_model_name, model_ckpt_path = bert_model_root

    remove_padding = remove_input_padding
    if not use_attention_plugin:
        remove_padding = False
    else:
        if get_sm_version() >= 100 and get_sm_version() < 120:
            pytest.skip("Attention plugin is not supported on SM100")

    # Convert checkpoints
    converted_weight_dir = convert_weights(llm_venv=llm_venv,
                                           example_root=bert_example_root,
                                           cmodel_dir=cmodel_dir,
                                           model=model,
                                           model_path=model_ckpt_path,
                                           data_type=dtype,
                                           tp_size=tp_size)

    # Build Engine
    bert_engine_dir = f"{engine_dir}/{model}/{world_size}-gpus/{dtype}/remove_padding_{remove_padding}"
    build_cmd = [
        "trtllm-build",
        f"--checkpoint_dir={converted_weight_dir}",
        f"--output_dir={bert_engine_dir}",
        "--max_batch_size=8",
    ]

    if use_attention_plugin:
        build_cmd.append(f"--bert_attention_plugin={dtype}")
    else:
        build_cmd.append(f"--bert_attention_plugin=disable")
    if remove_input_padding and use_attention_plugin:
        build_cmd.extend(["--remove_input_padding=enable"])
    else:
        build_cmd.extend(["--remove_input_padding=disable"])

    if context_fmha_type == 'enabled':
        build_cmd.extend(["--context_fmha=enable"])
    if context_fmha_type == 'enabled_with_fp32_acc':
        build_cmd.extend(["--bert_context_fmha_fp32_acc=enable"])

    check_call(" ".join(build_cmd), shell=True, env=llm_venv._new_env)
    # Run Engine
    print("Run inference...")
    run_cmd = [
        f"{bert_example_root}/run.py",
        f"--engine_dir={bert_engine_dir}",
        f"--hf_model_dir={model_ckpt_path}",
    ]
    if remove_input_padding and use_attention_plugin:
        run_cmd.extend(["--remove_input_padding"])
    if compare_hf:
        run_cmd.extend(["--run_hf_test"])
    if world_size == 1:
        venv_check_call(llm_venv, run_cmd)
    else:
        venv_mpi_check_call(
            llm_venv, ["mpirun", "-n",
                       str(world_size), "--allow-run-as-root"], run_cmd)
