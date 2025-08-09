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
from defs.common import venv_check_call
from defs.conftest import get_gpu_device_list, get_sm_version
from defs.trt_test_alternative import check_call

# skip trt flow cases on post-Blackwell-Ultra
if get_sm_version() >= 103:
    pytest.skip(
        "TRT workflow tests are not supported on post Blackwell-Ultra architecture",
        allow_module_level=True)

INPUT_TEXT = """
Write a Python function `find_max(words)` to solve the following problem:\nWrite a function that accepts a list of strings.\nThe list contains different words. Return the word with maximum number\nof unique characters. If multiple strings have maximum number of unique\ncharacters, return the one which comes first in lexicographical order.\nfind_max(["name", "of", "string"]) == "string"\nfind_max(["name", "enam", "game"]) == "enam"\nfind_max(["aaaaaaa", "bb" ,"cc"]) == ""aaaaaaa"
"""


def test_llm_gptj_fp8_manage_weights_summary(gptj_example_root,
                                             llm_gptj_model_root,
                                             llm_datasets_root, llm_rouge_root,
                                             llm_venv, engine_dir):

    gpus = ["L40S", "H20", "H100"]
    if all(x not in get_gpu_device_list()[0] for x in gpus):
        pytest.skip("FP8 cannot be enabled on Pre-Ada Arch.")

    print("Quantizing model...")
    qcache_dir = "/tmp/cache"
    ckpt_dir = f"{qcache_dir}/quantized_model_cache"
    quantize_cmd = [
        f"{gptj_example_root}/../quantization/quantize.py",
        f"--model_dir={llm_gptj_model_root}",
        f"--calib_dataset={llm_datasets_root}/cnn_dailymail",
        f"--output_dir={ckpt_dir}",
        "--dtype=float16",
        "--calib_size=16",
        "--qformat=fp8",
    ]

    venv_check_call(llm_venv, quantize_cmd)

    print("Building engines...")
    build_cmd = [
        "trtllm-build", f"--checkpoint_dir={ckpt_dir}",
        f"--output_dir={engine_dir}", "--max_batch_size=4",
        "--max_input_len=1024", "--max_seq_len=1152", "--max_beam_width=5",
        "--fast_build"
    ]
    check_call(" ".join(build_cmd), shell=True, env=llm_venv._new_env)

    print("Run summary...")
    summary_cmd = [
        f"{gptj_example_root}/../summarize.py", "--engine_dir", engine_dir,
        "--hf_model_dir", llm_gptj_model_root, "--batch_size", "1",
        "--test_trt_llm", "--tensorrt_llm_rouge1_threshold", "14",
        "--data_type", "fp16", "--check_accuracy",
        f"--dataset_dir={llm_datasets_root}", f"--rouge_dir={llm_rouge_root}"
    ]

    venv_check_call(llm_venv, summary_cmd)
