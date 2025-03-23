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

import csv
from copy import deepcopy

import pytest
from defs.common import convert_weights, venv_check_call
from defs.conftest import get_device_memory, skip_post_blackwell
from defs.trt_test_alternative import check_call


# TODO: remove skip after enable Blackwell for Speculative Decoding
@skip_post_blackwell
@pytest.mark.parametrize("batch_size", [1, 2], ids=['bs1', 'bs2'])
@pytest.mark.parametrize("data_type", ['float16'])
@pytest.mark.parametrize("draft_len", [4, 8],
                         ids=['draft_len_4', 'draft_len_8'])
@pytest.mark.parametrize("use_logits", [False, True],
                         ids=['use_tokens', 'use_logits'])
@pytest.mark.parametrize("use_py_session", [False], ids=["use_cpp_session"])
@pytest.mark.parametrize("draft_target_model_roots", ["gpt2", "llama_v2"],
                         indirect=True)
@pytest.mark.parametrize("streaming", [False, True],
                         ids=["no_streaming", "streaming"])
def test_llm_draft_target_model_1gpu(batch_size, data_type, draft_len,
                                     use_logits, use_py_session,
                                     draft_target_model_roots, streaming,
                                     draft_target_model_example_root,
                                     llm_datasets_root, llm_rouge_root,
                                     llm_venv, cmodel_dir, engine_dir):
    if "llama" in draft_target_model_roots[1]:
        if get_device_memory() < 80000:
            pytest.skip("GPU memory is insufficient.")

    model_name = "draft_target_model"

    print("Build checkpoint ...")
    model_dir = convert_weights(llm_venv=llm_venv,
                                example_root=draft_target_model_example_root,
                                cmodel_dir=cmodel_dir,
                                model=model_name,
                                model_path=draft_target_model_roots[1],
                                data_type=data_type)

    print("Build engines ...")
    draft_engine_dir = engine_dir + "-draft"
    target_engine_dir = engine_dir + "-target"
    baseline_engine_dir = engine_dir + "-baseline"
    common_build_cmd = [
        "trtllm-build",
        f"--checkpoint_dir={model_dir}",
        f"--max_batch_size={batch_size}",
        f"--max_beam_width=1",
        "--max_input_len=1024",
        "--max_seq_len=1536",
        "--use_paged_context_fmha=enable",
        f"--gpt_attention_plugin={data_type}",
        f"--gemm_plugin={data_type}",
        "--gather_generation_logits",
    ]
    draft_model_build_cmd = deepcopy(common_build_cmd)
    draft_model_build_cmd.extend([
        f"--output_dir={draft_engine_dir}",
    ])
    target_model_build_cmd = deepcopy(common_build_cmd)
    target_model_build_cmd.extend([
        f"--output_dir={target_engine_dir}",
        "--speculative_decoding_mode=draft_tokens_external",
        f"--max_draft_len={draft_len}",
    ])
    baseline_model_build_cmd = deepcopy(common_build_cmd)
    baseline_model_build_cmd.extend([
        f"--output_dir={baseline_engine_dir}",
    ])

    check_call(" ".join(draft_model_build_cmd),
               shell=True,
               env=llm_venv._new_env)
    check_call(" ".join(target_model_build_cmd),
               shell=True,
               env=llm_venv._new_env)
    check_call(" ".join(baseline_model_build_cmd),
               shell=True,
               env=llm_venv._new_env)

    print("Run inferences ...")
    draft_model_config = f"[{draft_len},[0],[0],{use_logits}]"
    common_run_cmd = [
        f"{draft_target_model_example_root}/../run.py",
        f"--tokenizer_dir={draft_target_model_roots[1]}",
        "--max_output_len=64",
        "--kv_cache_enable_block_reuse",
        "--kv_cache_free_gpu_memory_fraction=0.25",
    ]
    if streaming:
        common_run_cmd.extend(["--streaming", "--streaming_interval=1"])
    if batch_size == 1:
        common_run_cmd.extend(["--input_text", "'How are you?'"])
    elif batch_size == 2:
        common_run_cmd.extend(["--input_text", "'Hello'", "'How are you?'"])
    else:
        assert False, "Only batch_size <=2 is supported in test."
    assert not use_py_session, "Only CPP session is supported in Draft-Target-Model."

    run_cmd = deepcopy(common_run_cmd)
    run_cmd.extend([
        f"--engine_dir={target_engine_dir}",
        f"--draft_engine_dir={draft_engine_dir}",
        f"--draft_target_model_config={draft_model_config}",
        f"--output_csv={engine_dir}/draft_target_output.csv",
    ])
    baseline_run_cmd = deepcopy(common_run_cmd)
    baseline_run_cmd.extend([
        f"--engine_dir={baseline_engine_dir}",
        f"--output_csv={engine_dir}/baseline_output.csv",
    ])

    venv_check_call(llm_venv, run_cmd)
    venv_check_call(llm_venv, baseline_run_cmd)

    print("Compare outputs ...")
    with open(f"{engine_dir}/draft_target_output.csv") as dt_f, open(
            f"{engine_dir}/baseline_output.csv") as b_f:
        for bs, (dt_request,
                 b_request) in enumerate(zip(csv.reader(dt_f),
                                             csv.reader(b_f))):
            assert (
                len(dt_request) == len(b_request)
            ), f"Output length at ({bs=}) is different ({len(dt_request)} v.s. {len(b_request)})."
            for index, (dt, b) in enumerate(zip(dt_request, b_request)):
                assert (
                    int(dt) == int(b)
                ), f"Output at ({bs=}, {index=}) is different ({dt} v.s. {b})."
