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
from defs.conftest import get_sm_version, skip_post_blackwell
from defs.trt_test_alternative import check_call

# skip trt flow cases on post-Blackwell-Ultra
if get_sm_version() >= 103:
    pytest.skip(
        "TRT workflow tests are not supported on post Blackwell-Ultra architecture",
        allow_module_level=True)


# TODO: remove skip after support NGram on B200
@skip_post_blackwell
@pytest.mark.parametrize("batch_size", [1, 2], ids=['bs1', 'bs2'])
@pytest.mark.parametrize("data_type", ['float16'])
@pytest.mark.parametrize("max_draft_len", [4, 8],
                         ids=['max_draft_len_4', 'max_draft_len_8'])
@pytest.mark.parametrize(
    "max_matching_ngram_size", [2, 4],
    ids=['max_matching_ngram_size_2', 'max_matching_ngram_size_4'])
@pytest.mark.parametrize("use_logits", [False, True],
                         ids=['use_tokens', 'use_logits'])  # useless yet
@pytest.mark.parametrize("use_py_session", [False], ids=["use_cpp_session"])
@pytest.mark.parametrize("ngram_root", ["gpt2"], indirect=True)
@pytest.mark.parametrize("streaming", [False, True],
                         ids=["no_streaming", "streaming"])
def test_llm_ngram_1gpu(batch_size, data_type, max_draft_len,
                        max_matching_ngram_size, use_logits, use_py_session,
                        ngram_root, streaming, ngram_example_root,
                        llm_datasets_root, llm_rouge_root, llm_venv, cmodel_dir,
                        engine_dir):
    model_name = "ngram"

    print("Build checkpoint ...")
    model_dir = convert_weights(llm_venv=llm_venv,
                                example_root=ngram_example_root,
                                cmodel_dir=cmodel_dir,
                                model=model_name,
                                model_path=ngram_root,
                                data_type=data_type)

    print("Build engines ...")
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
    ]
    target_model_build_cmd = deepcopy(common_build_cmd)
    target_model_build_cmd.extend([
        f"--output_dir={target_engine_dir}",
        "--speculative_decoding_mode=draft_tokens_external",
        f"--max_draft_len={max_draft_len+1}",
    ])
    baseline_model_build_cmd = deepcopy(common_build_cmd)
    baseline_model_build_cmd.extend([
        f"--output_dir={baseline_engine_dir}",
    ])

    check_call(" ".join(target_model_build_cmd),
               shell=True,
               env=llm_venv._new_env)
    check_call(" ".join(baseline_model_build_cmd),
               shell=True,
               env=llm_venv._new_env)

    print("Run inferences ...")
    common_run_cmd = [
        f"{ngram_example_root}/../run.py",
        f"--tokenizer_dir={ngram_root}",
        f"--max_output_len=64",
        f"--kv_cache_enable_block_reuse",
        f"--kv_cache_free_gpu_memory_fraction=0.25",
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
    ngram_config = f"[{max_draft_len},{max_matching_ngram_size},[0]]"
    run_cmd.extend([
        f"--engine_dir={target_engine_dir}",
        f"--ngram_config={ngram_config}",
        f"--output_csv={engine_dir}/ngram_output.csv",
    ])
    baseline_run_cmd = deepcopy(common_run_cmd)
    baseline_run_cmd.extend([
        f"--engine_dir={baseline_engine_dir}",
        f"--output_csv={engine_dir}/baseline_output.csv",
    ])

    venv_check_call(llm_venv, run_cmd)
    venv_check_call(llm_venv, baseline_run_cmd)

    print("Compare outputs ...")
    with open(f"{engine_dir}/ngram_output.csv") as dt_f, open(
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

    if batch_size > 1 or streaming:  # Summarize tests for only batch_size=1 and streaming=False.
        return

    print("Run summarize...")
    ngram_config = f"[{max_draft_len},{max_matching_ngram_size},[0]]"

    run_cmd = [
        f"{ngram_example_root}/../summarize.py",
        "--test_hf",
        "--test_trt_llm",
        "--check_accuracy",
        "--batch_size=1",
        f"--hf_model_dir={ngram_root}",
        f"--engine_dir={target_engine_dir}",
        f"--dataset_dir={llm_datasets_root}",
        f"--rouge_dir={llm_rouge_root}",
        "--kv_cache_enable_block_reuse",
        f"--ngram_config={ngram_config}",
        "--tensorrt_llm_rouge1_threshold=20",
        f"--kv_cache_free_gpu_memory_fraction=0.25",
    ]

    venv_check_call(llm_venv, run_cmd)
