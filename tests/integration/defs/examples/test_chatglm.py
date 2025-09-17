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
"""Module test_chatglm test chatglm examples."""
import os
import shutil

import pytest
from defs.common import convert_weights, venv_check_call
from defs.conftest import get_sm_version, skip_post_blackwell
from defs.trt_test_alternative import check_call, exists

# skip trt flow cases on post-Blackwell-Ultra
if get_sm_version() >= 103:
    pytest.skip(
        "TRT workflow tests are not supported on post Blackwell-Ultra architecture",
        allow_module_level=True)


# TODO: add more test case for input_padding, paged_kv_cache, num_beams
@pytest.mark.skip_less_device_memory(24000)
@pytest.mark.parametrize("use_weight_only",
                         [pytest.param(True, marks=skip_post_blackwell), False],
                         ids=["enable_weight_only", "disable_weight_only"])
@pytest.mark.parametrize("llm_glm_4_9b_model_root",
                         ["glm-4-9b", "glm-4-9b-chat"],
                         indirect=True)
def test_llm_glm_4_9b_single_gpu_summary(glm_4_9b_example_root,
                                         llm_glm_4_9b_model_root,
                                         llm_datasets_root, llm_rouge_root,
                                         llm_venv, cmodel_dir, engine_dir,
                                         use_weight_only):
    "Build & run glm-4-9b on single gpu."
    print("Converting checkpoint...")
    dtype = 'float16'
    model_name = os.path.basename(llm_glm_4_9b_model_root)
    ckpt_dir = convert_weights(llm_venv=llm_venv,
                               example_root=glm_4_9b_example_root,
                               cmodel_dir=cmodel_dir,
                               model=model_name,
                               model_path=llm_glm_4_9b_model_root,
                               data_type=dtype,
                               use_weight_only=use_weight_only)

    print("Building engines...")
    build_cmd = [
        "trtllm-build", f"--checkpoint_dir={ckpt_dir}",
        f"--output_dir={engine_dir}", f"--max_batch_size={8}",
        f"--max_input_len={924}", f"--max_seq_len={1024}",
        f"--gpt_attention_plugin={dtype}"
    ]

    check_call(" ".join(build_cmd), shell=True, env=llm_venv._new_env)

    print("Running inference...")

    # fix HF error in glm-4-9b, hope to remove this in the future
    # nvbug 5025895
    model_temp_dir = glm_4_9b_example_root + "/model_temp_dir"
    if not exists(model_temp_dir):
        shutil.copytree(llm_glm_4_9b_model_root, model_temp_dir)
        shutil.copy(glm_4_9b_example_root + "/tokenization_chatglm.py",
                    model_temp_dir)

    summary_cmd = [
        f"{glm_4_9b_example_root}/../../../summarize.py", "--test_trt_llm",
        "--hf_model_dir", f"{model_temp_dir}", "--data_type", "fp16",
        "--check_accuracy", f"--engine_dir={engine_dir}",
        f"--dataset_dir={llm_datasets_root}", f"--rouge_dir={llm_rouge_root}"
    ]

    venv_check_call(llm_venv, summary_cmd)
