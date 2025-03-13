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
import os

import pytest
from defs.common import (convert_weights, generate_summary_cmd,
                         venv_mpi_check_call)
from defs.trt_test_alternative import check_call


@pytest.fixture(autouse=True, scope="module")
def grok_example_root(llm_venv, llm_root):
    "get grok example path"
    example_root = os.path.join(llm_root, "examples", "grok")
    try:
        llm_venv.run_cmd([
            "-m", "pip", "install", "-r",
            os.path.join(example_root, "requirements.txt")
        ])
    except Exception:
        print("pip install error!")

    return example_root


@pytest.mark.skip_less_device(8)
@pytest.mark.skip_less_device_memory(80000)
def test_llm_grok_wo_1node_8gpus_summary(grok_example_root, cmodel_dir,
                                         grok_model_root, grok_code_root,
                                         llm_datasets_root, llm_venv,
                                         engine_dir, llm_rouge_root):
    "test grok on 8 gpus with weight only"
    dtype = "bfloat16"
    tp_size, pp_size = 8, 1
    workers = tp_size * pp_size
    model_name = os.path.basename(grok_model_root)
    model_dir = convert_weights(llm_venv=llm_venv,
                                example_root=grok_example_root,
                                cmodel_dir=cmodel_dir,
                                model=model_name,
                                model_path=grok_code_root,
                                use_weight_only=True,
                                data_type=dtype,
                                tp_size=tp_size,
                                pp_size=pp_size,
                                workers=workers,
                                weights_dir=grok_model_root)

    print("Building engines...")
    build_cmd = [
        "trtllm-build",
        f"--checkpoint_dir={model_dir}",
        f"--output_dir={engine_dir}",
        f"--workers={workers}",
        f"--gpt_attention_plugin={dtype}",
        f"--gemm_plugin={dtype}",
        f"--moe_plugin={dtype}",
        "--paged_kv_cache=enable",
        "--remove_input_padding=enable",
    ]

    check_call(" ".join(build_cmd), shell=True, env=llm_venv._new_env)

    print("Run engines...")
    vocab_file = f"{grok_code_root}/tokenizer.model"
    run_cmd = [
        f"{grok_example_root}/../run.py",
        "--input_text",
        "The answer to life the universe and everything is of course",
        f"--engine_dir={engine_dir}",
        "--max_output_len=50",
        "--top_p=1",
        "--top_k=8",
        "--temperature=0.3",
        "--random_seed=0",
        f"--vocab_file={vocab_file}",
    ]

    venv_mpi_check_call(llm_venv, ["mpirun", "-n", "8", "--allow-run-as-root"],
                        run_cmd)

    summary_cmd = generate_summary_cmd(grok_example_root,
                                       engine_dir=engine_dir,
                                       data_type=dtype,
                                       vocab_file=vocab_file,
                                       tensorrt_llm_rouge1_threshold=15,
                                       eval_task="summarize",
                                       dataset_dir=llm_datasets_root,
                                       rouge_dir=llm_rouge_root)

    venv_mpi_check_call(llm_venv,
                        ["mpirun", "-n", f"{workers}", "--allow-run-as-root"],
                        summary_cmd)
