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
@pytest.mark.parametrize("batch_size", [8], ids=['bs8'])
@pytest.mark.parametrize("redrafter_num_beams", [5, 8], ids=['nb5', 'nb8'])
@pytest.mark.parametrize("redrafter_draft_len_per_beam", [5], ids=['dl5'])
@pytest.mark.parametrize("data_type", ['bfloat16'])
@pytest.mark.parametrize("redrafter_model_roots", ["redrafter-vicuna-7b-v1.3"],
                         indirect=True)
@pytest.mark.parametrize("use_py_session", [False, True],
                         ids=["use_cpp_session", "use_py_session"])
def test_llm_redrafter_1gpu(batch_size, data_type, redrafter_model_roots,
                            redrafter_num_beams, redrafter_draft_len_per_beam,
                            redrafter_example_root, llama_example_root,
                            llm_datasets_root, llm_rouge_root, llm_venv,
                            cmodel_dir, cmodel_base_dir, engine_dir,
                            use_py_session):
    print("Build engines...")
    model_name = "redrafter"
    base_model_name = "llama"
    base_example_root = llama_example_root

    base_model_dir = convert_weights(llm_venv=llm_venv,
                                     example_root=base_example_root,
                                     cmodel_dir=cmodel_base_dir,
                                     model=base_model_name,
                                     model_path=redrafter_model_roots[0],
                                     data_type=data_type)

    redrafter_convert_roots = (base_model_dir, redrafter_model_roots[1])

    model_dir = convert_weights(
        llm_venv=llm_venv,
        example_root=redrafter_example_root,
        cmodel_dir=cmodel_dir,
        model=model_name,
        model_path=redrafter_convert_roots,
        data_type=data_type,
        redrafter_num_beams=redrafter_num_beams,
        redrafter_draft_len_per_beam=redrafter_draft_len_per_beam)

    build_cmd = [
        "trtllm-build",
        f"--checkpoint_dir={model_dir}",
        f"--output_dir={engine_dir}",
        f"--gpt_attention_plugin={data_type}",
        f"--gemm_plugin={data_type}",
        f"--max_beam_width=1",
        "--remove_input_padding=enable",
        "--context_fmha=enable",
        "--max_input_len=1024",
        "--max_seq_len=1536",
        f"--max_batch_size={batch_size}",
        "--kv_cache_type=paged",
        '--speculative_decoding_mode=explicit_draft_tokens',
    ]

    check_call(" ".join(build_cmd), shell=True, env=llm_venv._new_env)

    print("Run summarize...")

    summary_cmd = [
        f"{redrafter_example_root}/../summarize.py", "--test_trt_llm",
        "--hf_model_dir", f"{redrafter_model_roots[0]}", "--tokenizer_dir",
        f"{redrafter_model_roots[0]}", f"--engine_dir={engine_dir}",
        "--check_accuracy", "--tensorrt_llm_rouge1_threshold=24",
        f"--temperature=1.0", f"--max_ite=40", f"--batch_size={batch_size}",
        f"--dataset_dir={llm_datasets_root}", f"--rouge_dir={llm_rouge_root}"
    ]

    if use_py_session:
        summary_cmd.append("--use_py_session")

    venv_check_call(llm_venv, summary_cmd)
