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
"""Module test_gpt test gpt examples."""
import csv
import os
import re
from pathlib import Path

import pytest
from defs.common import (convert_weights, generate_summary_cmd, parse_mpi_cmd,
                         parse_output, quantize_data, run_and_check, similar,
                         similarity_score, test_multi_lora_support,
                         venv_check_call, venv_check_output,
                         venv_mpi_check_call, venv_mpi_check_output)
from defs.conftest import (get_device_memory, get_sm_version, skip_fp8_pre_ada,
                           skip_post_blackwell, skip_pre_ada)
from defs.trt_test_alternative import check_call

# skip trt flow cases on post-Blackwell-Ultra
if get_sm_version() >= 103:
    pytest.skip(
        "TRT workflow tests are not supported on post Blackwell-Ultra architecture",
        allow_module_level=True)

INPUT_TEXT_1 = "After Washington had returned to Williamsburg, " + \
               "Dinwiddie ordered him to lead a larger force to assist Trent in his work. " + \
               "While en route, Washington learned of Trent's retreat. " + \
               "Since Tanaghrisson had promised support to the British, " + \
               "Washington continued toward Fort Duquesne and met with the Mingo leader. " + \
               "Learning of a French scouting party in the area, Washington, " + \
               "with Tanaghrisson and his party, surprised the Canadians on May 28 " + \
               "in what became known as the Battle of Jumonville Glen. " + \
               "They killed many of the Canadians, including their commanding officer, " + \
               "Joseph Coulon de Jumonville, whose head was reportedly split open by " + \
               "Tanaghrisson with a tomahawk. The historian Fred Anderson suggests that " + \
               "Tanaghrisson was acting to gain the support of the British and regain " + \
               "authority over his own people. They had been inclined to support the French, " + \
               "with whom they had long trading relationships. One of Tanaghrisson's men told " + \
               "Contrecoeur that Jumonville had been killed by British musket fire. " + \
               "Question: Upon learning of a French scounting party in the area, " + \
               "what did Washington do? Answer:"

INPUT_TEXT_2 = "You hold the job title in the Wizarding World of Harry Potter where you " + \
               "say random words looking for spells"


@pytest.mark.parametrize("num_beams", [1, 4],
                         ids=["num_beams_1", "num_beams_4"])
@pytest.mark.parametrize(
    "return_all_generated_tokens", [True, False],
    ids=["return_all_generated_tokens", "disable_return_all_generated_tokens"])
@pytest.mark.parametrize("batch_size", [1, 3],
                         ids=["batch_size_1", "batch_size_3"])
def test_streaming_beam(gpt_example_root, llm_venv, llm_gpt2_model_root,
                        engine_dir, cmodel_dir, num_beams,
                        return_all_generated_tokens, batch_size):
    """ Test the correctness of beam search + streaming versus the outputs of
    non-streaming beam search. Both use the cpp runtime.
    The num_beams=1 test acts as a test for `return_all_generated_tokens`"""

    dtype = 'float16'
    output_len = 10
    texts = ["want to", "Movies are just", "Soyer was"]
    input_text = texts[:batch_size]

    ckpt_dir = convert_weights(llm_venv=llm_venv,
                               example_root=gpt_example_root,
                               cmodel_dir=cmodel_dir,
                               model="gpt2",
                               model_path=llm_gpt2_model_root,
                               data_type=dtype)

    print("Building engines...")
    build_cmd = [
        "trtllm-build",
        f"--checkpoint_dir={ckpt_dir}",
        f"--output_dir={engine_dir}",
        f"--max_batch_size={8}",
        f"--max_input_len={924}",
        f"--max_seq_len={1024}",
        f"--gpt_attention_plugin={dtype}",
        f"--gemm_plugin={dtype}",
        f"--max_beam_width={num_beams}",
        "--context_fmha=enable",
        "--use_paged_context_fmha=enable",
        "--paged_kv_cache=enable",
        "--remove_input_padding=enable",
    ]
    check_call(" ".join(build_cmd), shell=True, env=llm_venv._new_env)

    print("Running inference...")

    streaming_command = [
        f"{gpt_example_root}/../../../run.py", "--no_add_special_tokens",
        f"--max_output_len={output_len}", f"--engine_dir={engine_dir}",
        f"--tokenizer_dir={llm_gpt2_model_root}", f"--streaming",
        f"--streaming_interval=1", f"--num_beams={num_beams}", f"--input_text",
        *input_text
    ]
    if return_all_generated_tokens:
        streaming_command += ["--return_all_generated_tokens"]
    streaming_outputs = venv_check_output(llm_venv, streaming_command)

    joined_nonstreamed_outputs = ""
    for length_iterator in range(1, output_len + 1):
        command = [
            f"{gpt_example_root}/../../../run.py", "--no_add_special_tokens",
            f"--max_output_len={length_iterator}", f"--engine_dir={engine_dir}",
            f"--tokenizer_dir={llm_gpt2_model_root}",
            f"--num_beams={num_beams}", f"--input_text", *input_text
        ]
        if return_all_generated_tokens:
            command += ["--return_all_generated_tokens"]

        non_streaming_output = venv_check_output(llm_venv, command)
        joined_nonstreamed_outputs += "Output from command" + str(
            command) + "\n" + non_streaming_output

    def parse_output(text: str) -> list[str]:
        results = []
        while True:
            match = re.search(
                r"Output \[Text \d+ Beam \d+\]: \"([^\"]*)\"\r?\n", text)
            if match is None:
                break
            _, end = match.span()
            results.append(match.group(1))
            text = text[end:]
        return results

    print("STREAMING OUTPUT HERE\n\n\n",
          streaming_outputs,
          "\n\n\n",
          sep="----")
    print("NON-STREAMING OUTPUT HERE\n\n\n",
          joined_nonstreamed_outputs,
          "\n\n\n",
          sep="----")
    parsed_streamed_outputs = parse_output(streaming_outputs)
    parsed_nonstreamed_outputs = parse_output(joined_nonstreamed_outputs)

    def ordered_subset(s1, s2):
        """
        Use this to check if the streamed outputs are an ordered subset of nonstreamed
        Streaming can sometimes skip outputs
        """
        s2 = iter(s2)
        try:
            for c in s1:
                while next(s2) != c:
                    pass
            else:
                return True
        except StopIteration:
            return False

    streaming_is_subset = ordered_subset(parsed_streamed_outputs,
                                         parsed_nonstreamed_outputs)
    print("streaming_is_subset ", streaming_is_subset)
    assert streaming_is_subset
    is_equal = (parsed_streamed_outputs == parsed_nonstreamed_outputs)
    print("is_equal", is_equal)
    if not is_equal:
        print("Differences:")
        for streamed, nonstreamed in zip(parsed_streamed_outputs,
                                         parsed_nonstreamed_outputs):
            if (streamed != nonstreamed):
                print("Streamed:", streamed)
                print("Nonstreamed:", nonstreamed)

    # streaming can can skip outputs, if the next set of outputs arrive.
    # this means that the is_equal flag is currently flaky: https://nvbugspro.nvidia.com/bug/4851644
    # assert is_equal


def test_llm_gpt2_kv_cache_1gpu(gpt_example_root, llm_venv, llm_gpt2_model_root,
                                engine_dir, cmodel_dir):
    "gpt2 cases on 1 gpu"
    print("Converting checkpoint...")
    dtype = 'float16'
    ckpt_dir = convert_weights(llm_venv=llm_venv,
                               example_root=gpt_example_root,
                               cmodel_dir=cmodel_dir,
                               model="gpt2",
                               model_path=llm_gpt2_model_root,
                               data_type=dtype)

    print("Building engines...")
    build_cmd = [
        "trtllm-build",
        f"--checkpoint_dir={ckpt_dir}",
        f"--output_dir={engine_dir}",
        f"--max_batch_size={8}",
        f"--max_input_len={924}",
        f"--max_seq_len={1024}",
        f"--gpt_attention_plugin={dtype}",
        f"--gemm_plugin={dtype}",
        "--context_fmha=enable",
        "--use_paged_context_fmha=enable",
        "--paged_kv_cache=enable",
        "--remove_input_padding=enable",
    ]
    check_call(" ".join(build_cmd), shell=True, env=llm_venv._new_env)

    print("Running inference...")

    venv_check_call(llm_venv, [
        f"{gpt_example_root}/../../../summarize.py",
        f"--engine_dir={engine_dir}",
        "--test_hf",
        "--batch_size=1",
        "--test_trt_llm",
        f"--hf_model_dir={llm_gpt2_model_root}",
        "--check_accuracy",
        "--tensorrt_llm_rouge1_threshold=13.5",
        "--no_add_special_tokens",
        "--max_tokens_in_paged_kv_cache=1024",
    ])

    venv_check_call(llm_venv, [
        f"{gpt_example_root}/../../../summarize.py",
        f"--engine_dir={engine_dir}",
        "--test_hf",
        "--batch_size=1",
        "--test_trt_llm",
        f"--hf_model_dir={llm_gpt2_model_root}",
        "--check_accuracy",
        "--tensorrt_llm_rouge1_threshold=13.5",
        "--no_add_special_tokens",
        "--kv_cache_enable_block_reuse",
        "--kv_cache_free_gpu_memory_fraction=0.5",
    ])


@pytest.mark.parametrize(
    "use_attention_plugin", [True, False],
    ids=["enable_attention_plugin", "disable_attention_plugin"])
@pytest.mark.parametrize("use_gemm_plugin", [True, False],
                         ids=["enable_gemm_plugin", "disable_gemm_plugin"])
def test_llm_gpt2_1gpu(gpt_example_root, llm_venv, llm_gpt2_model_root,
                       llm_datasets_root, llm_rouge_root, engine_dir,
                       cmodel_dir, use_attention_plugin, use_gemm_plugin):
    "gpt2 cases on 1 gpu"
    print("Converting checkpoint...")
    dtype = 'float16'
    ckpt_dir = convert_weights(llm_venv=llm_venv,
                               example_root=gpt_example_root,
                               cmodel_dir=cmodel_dir,
                               model="gpt2",
                               model_path=llm_gpt2_model_root,
                               data_type=dtype)

    print("Building engines...")
    build_cmd = [
        "trtllm-build",
        f"--checkpoint_dir={ckpt_dir}",
        f"--output_dir={engine_dir}",
        f"--max_batch_size={8}",
        f"--max_input_len={924}",
        f"--max_seq_len={1024}",
    ]

    if use_attention_plugin:
        build_cmd.extend([f"--gpt_attention_plugin={dtype}"])
    else:
        build_cmd.extend([
            "--gpt_attention_plugin=disable",
            "--context_fmha=disable",
            "--paged_kv_cache=disable",
            "--remove_input_padding=disable",
        ])
    if use_gemm_plugin:
        build_cmd.extend([f"--gemm_plugin={dtype}"])

    check_call(" ".join(build_cmd), shell=True, env=llm_venv._new_env)

    print("Running inference...")
    venv_check_call(llm_venv, [
        f"{gpt_example_root}/../../../summarize.py",
        f"--engine_dir={engine_dir}", "--test_hf", "--batch_size=1",
        "--test_trt_llm", f"--hf_model_dir={llm_gpt2_model_root}",
        "--check_accuracy", "--tensorrt_llm_rouge1_threshold=13.5",
        "--no_add_special_tokens", f"--dataset_dir={llm_datasets_root}",
        f"--rouge_dir={llm_rouge_root}"
    ])

    if not use_gemm_plugin:
        print("Checking embedding sharing...")
        # Embedding sharing should be enabled automatically.
        # Gpt2 has 124M parameters among which 36.8M are shared between embedding and lm_head.
        # If embedding sharing is enabled, the FP16 engine size should be about 248 MB;
        # otherwise, the engine size should be about 321.6 MB.
        engine_size = os.path.getsize(f"{engine_dir}/rank0.engine") / (1024**2)
        assert engine_size < 280


@pytest.mark.parametrize("use_gemm_plugin", [True, False],
                         ids=["enable_gemm_plugin", "disable_gemm_plugin"])
@pytest.mark.parametrize("use_py_session", [False, True],
                         ids=["use_cpp_session", "use_py_session"])
@pytest.mark.parametrize("streaming", [False, True],
                         ids=["non_streaming", "streaming"])
def test_llm_gpt2_medium_1gpu(gpt_example_root, llm_venv,
                              llm_gpt2_medium_model_root, cmodel_dir,
                              engine_dir, use_gemm_plugin, use_py_session,
                              streaming):
    "gpt2-medium build & run"
    print("Converting checkpoint...")
    dtype = 'float16'
    ckpt_dir = convert_weights(llm_venv=llm_venv,
                               example_root=gpt_example_root,
                               cmodel_dir=cmodel_dir,
                               model="gpt2-medium",
                               model_path=llm_gpt2_medium_model_root,
                               data_type=dtype)

    print("Building engines...")
    build_cmd = [
        "trtllm-build",
        f"--checkpoint_dir={ckpt_dir}",
        f"--output_dir={engine_dir}",
        f"--max_batch_size={8}",
        f"--max_input_len={924}",
        f"--max_seq_len={1024}",
        f"--gpt_attention_plugin={dtype}",
        "--paged_kv_cache=enable",
        "--remove_input_padding=enable",
    ]

    if use_gemm_plugin:
        build_cmd.extend([f"--gemm_plugin={dtype}"])

    check_call(" ".join(build_cmd), shell=True, env=llm_venv._new_env)

    run_cmd = [
        f"{gpt_example_root}/../../../run.py", "--max_output_len=8",
        f"--engine_dir={engine_dir}",
        f"--tokenizer_dir={llm_gpt2_medium_model_root}",
        "--no_add_special_tokens"
    ]

    if streaming:
        run_cmd.append("--streaming")
    if use_py_session:
        run_cmd.append("--use_py_session")

    print("Running inference...")
    output = venv_check_output(llm_venv, run_cmd)

    valid_outputs = [
        "chef before moving to London in the early",
        "chef before moving to London in the late",
        "chef and eventually became a chef at a",
    ]

    if not streaming:
        output = parse_output(output)[0]
        assert any([similar(output, expect)
                    for expect in valid_outputs]), f"output is: {output}"
    else:
        # Fetch all outputs and expect a monotonically increasing similarity
        similarities = []
        for suboutput in parse_output(output):
            similarities.append(
                max([
                    similarity_score(suboutput, expect)
                    for expect in valid_outputs
                ]))
        assert (
            all(x <= y for x, y in zip(similarities, similarities[1:]))
        ), f"streaming outputs must have a monotonically increasing similarity score. similarities: {similarities}"
        output = parse_output(output)[-1]
        assert any([similar(output, expect)
                    for expect in valid_outputs]), f"output is: {output}"


@pytest.mark.parametrize("use_py_session", [False, True],
                         ids=["use_cpp_session", "use_py_session"])
@pytest.mark.parametrize("streaming", [False, True],
                         ids=["non_streaming", "streaming"])
def test_llm_gpt2_medium_bad_words_1gpu(gpt_example_root, llm_venv,
                                        llm_gpt2_medium_model_root, cmodel_dir,
                                        engine_dir, use_py_session, streaming):
    "gpt2 build & run"

    if use_py_session and streaming:
        pytest.skip(
            "Streaming with py session does not return complete sequence to reliably check stop words"
        )

    print("Converting checkpoint...")
    dtype = 'float16'
    ckpt_dir = convert_weights(llm_venv=llm_venv,
                               example_root=gpt_example_root,
                               cmodel_dir=cmodel_dir,
                               model="gpt2-medium",
                               model_path=llm_gpt2_medium_model_root,
                               data_type=dtype)

    print("Building engines...")
    build_cmd = [
        "trtllm-build",
        f"--checkpoint_dir={ckpt_dir}",
        f"--output_dir={engine_dir}",
        f"--max_batch_size={8}",
        f"--max_input_len={924}",
        f"--max_seq_len={1024}",
        f"--gpt_attention_plugin={dtype}",
        "--paged_kv_cache=enable",
        "--remove_input_padding=enable",
    ]

    check_call(" ".join(build_cmd), shell=True, env=llm_venv._new_env)

    run_cmd = [
        f"{gpt_example_root}/../../../run.py", "--max_output_len=8",
        f"--engine_dir={engine_dir}",
        f"--tokenizer_dir={llm_gpt2_medium_model_root}",
        "--no_add_special_tokens"
    ]

    if streaming:
        run_cmd.append("--streaming")
    if use_py_session:
        run_cmd.append("--use_py_session")

    valid_outputs = [
        "chef before moving to the UK in the",
        "chef and eventually became a chef at a",
    ]
    bad_words_args = ["--bad_words", " London"]
    run_and_check(llm_venv,
                  run_cmd + bad_words_args,
                  valid_outputs,
                  streaming=streaming)

    bad_words_args = ["--bad_words", " to London", " irrelevant words"]
    run_and_check(llm_venv,
                  run_cmd + bad_words_args,
                  valid_outputs,
                  streaming=streaming)

    bad_words_args = ["--bad_words", " irrelevant words", " to London"]
    run_and_check(llm_venv,
                  run_cmd + bad_words_args,
                  valid_outputs,
                  streaming=streaming)


@pytest.mark.parametrize("use_py_session", [False, True],
                         ids=["use_cpp_session", "use_py_session"])
@pytest.mark.parametrize("streaming", [False, True],
                         ids=["non_streaming", "streaming"])
def test_llm_gpt2_medium_stop_words_1gpu(gpt_example_root, llm_venv,
                                         llm_gpt2_medium_model_root, cmodel_dir,
                                         engine_dir, use_py_session, streaming):
    "gpt2 build & run"
    if use_py_session and streaming:
        pytest.skip(
            "Streaming with py session does not return complete sequence to reliably check stop words"
        )

    print("Converting checkpoint...")
    dtype = 'float16'
    ckpt_dir = convert_weights(llm_venv=llm_venv,
                               example_root=gpt_example_root,
                               cmodel_dir=cmodel_dir,
                               model="gpt2-medium",
                               model_path=llm_gpt2_medium_model_root,
                               data_type=dtype)

    print("Building engines...")
    build_cmd = [
        "trtllm-build",
        f"--checkpoint_dir={ckpt_dir}",
        f"--output_dir={engine_dir}",
        f"--max_batch_size={8}",
        f"--max_input_len={924}",
        f"--max_seq_len={1024}",
        f"--gpt_attention_plugin={dtype}",
        "--paged_kv_cache=enable",
        "--remove_input_padding=enable",
    ]

    check_call(" ".join(build_cmd), shell=True, env=llm_venv._new_env)

    run_cmd = [
        f"{gpt_example_root}/../../../run.py", "--max_output_len=8",
        f"--engine_dir={engine_dir}",
        f"--tokenizer_dir={llm_gpt2_medium_model_root}",
        "--no_add_special_tokens"
    ]

    if streaming:
        run_cmd.append("--streaming")
    if use_py_session:
        run_cmd.append("--use_py_session")

    valid_outputs = [
        "chef before moving to London",
        "chef and eventually became",
    ]
    stop_words_args = ["--stop_words", " London", " became"]
    run_and_check(llm_venv,
                  run_cmd + stop_words_args,
                  valid_outputs,
                  streaming=streaming)

    stop_words_args = [
        "--stop_words", " eventually became", " to London", " irrelevant output"
    ]
    run_and_check(llm_venv,
                  run_cmd + stop_words_args,
                  valid_outputs,
                  streaming=streaming)

    stop_words_args = [
        "--stop_words", " to London", " eventually became", " irrelevant output"
    ]
    run_and_check(llm_venv,
                  run_cmd + stop_words_args,
                  valid_outputs,
                  streaming=streaming)


@pytest.mark.skip_less_device(8)
@pytest.mark.parametrize(
    "use_attention_plugin", [True, False],
    ids=["enable_attention_plugin", "disable_attention_plugin"])
@pytest.mark.parametrize("use_gemm_plugin", [True, False],
                         ids=["enable_gemm_plugin", "disable_gemm_plugin"])
def test_llm_gpt3_175b_2layers_1node_8gpus(gpt_example_root, llm_venv,
                                           engine_dir, use_attention_plugin,
                                           use_gemm_plugin):
    "Build & run GPT-3 175B: 2 layer w/ plugins, regression test for issues #20"
    dtype = 'float16'
    convert_cmd = [
        f"{gpt_example_root}/../../../generate_checkpoint_config.py",
        f"--output_path={engine_dir}/ckpt_config.json",
        "--architecture=GPTForCausalLM", f"--dtype={dtype}",
        "--num_hidden_layers=2", "--num_attention_heads=96",
        "--hidden_size=12288", "--vocab_size=51200", "--tp_size=8"
    ]
    venv_check_call(llm_venv, convert_cmd)

    print("Building engines...")
    build_cmd = [
        "trtllm-build",
        f"--model_config={engine_dir}/ckpt_config.json",
        f"--output_dir={engine_dir}",
        f"--max_batch_size={256}",
        f"--max_input_len={924}",
        f"--max_seq_len={1024}",
    ]

    if use_attention_plugin:
        build_cmd.extend([f"--gpt_attention_plugin={dtype}"])
    else:
        build_cmd.extend([
            "--gpt_attention_plugin=disable",
            "--context_fmha=disable",
            "--paged_kv_cache=disable",
            "--remove_input_padding=disable",
        ])
    if use_gemm_plugin:
        build_cmd.extend([f"--gemm_plugin={dtype}"])

    check_call(" ".join(build_cmd), shell=True, env=llm_venv._new_env)

    venv_mpi_check_call(
        llm_venv,
        ["mpirun", "--allow-run-as-root", "--oversubscribe", "-np", "8"], [
            f"{gpt_example_root}/../../../run.py", "--max_output_len=8",
            f"--engine_dir={engine_dir}", "--no_add_special_tokens"
        ])


@pytest.mark.parametrize(
    "use_attention_plugin", [True, False],
    ids=["enable_attention_plugin", "disable_attention_plugin"])
@pytest.mark.parametrize("use_gemm_plugin", [True, False],
                         ids=["enable_gemm_plugin", "disable_gemm_plugin"])
def test_llm_gpt3_175b_96layers_build_only(gpt_example_root, llm_venv,
                                           engine_dir, use_attention_plugin,
                                           use_gemm_plugin):
    "Build GPT-3 175B: 96 layer w/ plugins"
    dtype = 'float16'
    convert_cmd = [
        f"{gpt_example_root}/../../../generate_checkpoint_config.py",
        f"--output_path={engine_dir}/ckpt_config.json",
        "--architecture=GPTForCausalLM", f"--dtype={dtype}",
        "--num_hidden_layers=96", "--num_attention_heads=96",
        "--hidden_size=12288", "--vocab_size=51200", "--tp_size=8"
    ]
    venv_check_call(llm_venv, convert_cmd)

    print("Building engines...")
    build_cmd = [
        "trtllm-build",
        f"--model_config={engine_dir}/ckpt_config.json",
        f"--output_dir={engine_dir}",
        f"--max_batch_size={64}",
        f"--max_input_len={924}",
        f"--max_seq_len={1024}",
    ]

    if use_attention_plugin:
        build_cmd.extend([f"--gpt_attention_plugin={dtype}"])
    else:
        build_cmd.extend([
            "--gpt_attention_plugin=disable",
            "--context_fmha=disable",
            "--paged_kv_cache=disable",
            "--remove_input_padding=disable",
        ])
    if use_gemm_plugin:
        build_cmd.extend([f"--gemm_plugin={dtype}"])

    check_call(" ".join(build_cmd), shell=True, env=llm_venv._new_env)


@pytest.mark.skip_less_device(8)
@pytest.mark.skip_less_device_memory(80000)
@pytest.mark.parametrize(
    "use_attention_plugin", [True, False],
    ids=["enable_attention_plugin", "disable_attention_plugin"])
@pytest.mark.parametrize("use_gemm_plugin", [True, False],
                         ids=["enable_gemm_plugin", "disable_gemm_plugin"])
@pytest.mark.parametrize("context_fmha", [True, False],
                         ids=["enable_fmha", "disable_fmha"])
@pytest.mark.parametrize("parallel_build", [True, False],
                         ids=["parallel_build", "serial_build"])
def test_llm_gpt3_175b_1node_8gpus(gpt_example_root, llm_venv, engine_dir,
                                   use_attention_plugin, use_gemm_plugin,
                                   context_fmha, parallel_build,
                                   timeout_manager):
    "Build & Run GPT-3 175B: 96 layer w/ plugins"
    dtype = 'float16'

    # Convert checkpoint with timeout management
    with timeout_manager.timed_operation("convert"):
        convert_cmd = [
            f"{gpt_example_root}/../../../generate_checkpoint_config.py",
            f"--output_path={engine_dir}/ckpt_config.json",
            "--architecture=GPTForCausalLM", f"--dtype={dtype}",
            "--num_hidden_layers=96", "--num_attention_heads=96",
            "--hidden_size=12288", "--vocab_size=51200", "--tp_size=8"
        ]
        venv_check_call(llm_venv,
                        convert_cmd,
                        timeout=timeout_manager.remaining_timeout)

    # Build engines with timeout management
    print("Building engines...")
    with timeout_manager.timed_operation("build"):
        build_cmd = [
            "trtllm-build",
            f"--model_config={engine_dir}/ckpt_config.json",
            f"--output_dir={engine_dir}",
            f"--max_batch_size={32}",
            f"--max_input_len={924}",
            f"--max_seq_len={1024}",
        ]

        if use_attention_plugin:
            build_cmd.extend([f"--gpt_attention_plugin={dtype}"])
            if context_fmha:
                build_cmd.extend(["--context_fmha=enable"])
            else:
                build_cmd.extend(["--context_fmha=disable"])
        else:
            build_cmd.extend([
                "--gpt_attention_plugin=disable",
                "--context_fmha=disable",
                "--paged_kv_cache=disable",
                "--remove_input_padding=disable",
            ])
        if use_gemm_plugin:
            build_cmd.extend([f"--gemm_plugin={dtype}"])
        if parallel_build:
            build_cmd.extend(["--workers=8"])

        check_call(" ".join(build_cmd),
                   shell=True,
                   env=llm_venv._new_env,
                   timeout=timeout_manager.remaining_timeout)

    # Run inference with timeout management
    print('Run gpt3-175b...')
    with timeout_manager.timed_operation("run"):
        venv_mpi_check_call(
            llm_venv,
            ["mpirun", "--allow-run-as-root", "--oversubscribe", "-np", "8"], [
                f"{gpt_example_root}/../../../run.py", "--max_output_len=8",
                f"--engine_dir={engine_dir}", "--no_add_special_tokens"
            ],
            timeout=timeout_manager.remaining_timeout)


@skip_post_blackwell
@pytest.mark.parametrize("per_token_channel", [True, False],
                         ids=["enable_ptpc", "disable_ptpc"])
def test_llm_gpt2_smooth_single_gpu_summary(gpt_example_root, llm_venv,
                                            llm_gpt2_model_root,
                                            llm_datasets_root, llm_rouge_root,
                                            cmodel_dir, engine_dir,
                                            per_token_channel):
    "gpt2-smooth test on single gpu"
    print("Converting checkpoint...")
    dtype = 'float16'
    ckpt_dir = convert_weights(
        llm_venv=llm_venv,
        example_root=gpt_example_root,
        cmodel_dir=cmodel_dir,
        model="gpt2-smooth",
        model_path=llm_gpt2_model_root,
        data_type=dtype,
        per_token=per_token_channel,
        per_channel=per_token_channel,
        calib_dataset=f"{llm_datasets_root}/cimec/lambada")

    print("Building engines...")
    # NOTE: SQ does not support OOTB path for attention for now.
    # Check tensorrt_llm/quantization/layers.py::SmoothQuantAttention for details.
    build_cmd = [
        "trtllm-build",
        f"--checkpoint_dir={ckpt_dir}",
        f"--output_dir={engine_dir}",
        f"--max_batch_size={8}",
        f"--max_input_len={924}",
        f"--max_seq_len={1024}",
        f"--gpt_attention_plugin={dtype}",
    ]

    check_call(" ".join(build_cmd), shell=True, env=llm_venv._new_env)

    print("Running inference...")
    venv_check_call(llm_venv, [
        f"{gpt_example_root}/../../../run.py", "--max_output_len=8",
        f"--engine_dir={engine_dir}", f"--tokenizer_dir={llm_gpt2_model_root}",
        "--no_add_special_tokens"
    ])


@skip_post_blackwell
def test_llm_gpt2_int8_kv_1gpu(gpt_example_root, llm_venv, llm_gpt2_model_root,
                               llm_datasets_root, engine_dir, cmodel_dir):
    "gpt2 INT8 KV Cache test on 1 gpu"
    print("Converting checkpoint...")
    dtype = 'float16'
    ckpt_dir = convert_weights(
        llm_venv=llm_venv,
        example_root=gpt_example_root,
        cmodel_dir=cmodel_dir,
        model="gpt2-int8-kv",
        model_path=llm_gpt2_model_root,
        data_type=dtype,
        calib_dataset=f"{llm_datasets_root}/cimec/lambada")

    print("Building engines...")
    # TODO: This case only support enable gpt attention plugin.
    # https://nvbugs/4175869
    build_cmd = [
        "trtllm-build",
        f"--checkpoint_dir={ckpt_dir}",
        f"--output_dir={engine_dir}",
        f"--max_batch_size={8}",
        f"--max_input_len={924}",
        f"--max_seq_len={1024}",
        f"--gpt_attention_plugin={dtype}",
    ]

    check_call(" ".join(build_cmd), shell=True, env=llm_venv._new_env)

    print("Running inference...")
    venv_check_call(llm_venv, [
        f"{gpt_example_root}/../../../run.py", "--max_output_len=8",
        f"--engine_dir={engine_dir}", f"--tokenizer_dir={llm_gpt2_model_root}",
        "--no_add_special_tokens"
    ])


@skip_pre_ada
@pytest.mark.parametrize("quant_lm_head", [True, False])
@pytest.mark.parametrize("qformat", ["fp8", "fp8_pc_pt"])
def test_llm_gpt2_medium_fp8(gpt_example_root, llm_gpt2_medium_model_root,
                             llm_datasets_root, llm_rouge_root, llm_venv,
                             cmodel_dir, engine_dir, quant_lm_head, qformat):
    if qformat == "fp8_pc_pt" and quant_lm_head:
        pytest.skip("Skipping test for fp8_pc_pt with quant_lm_head")
    "Build & Run gpt2-medium fp8 with 1 gpu"
    print("Quantizing and converting checkpoint...")
    dtype = "float16"
    ckpt_dir = f"{cmodel_dir}/gpt2-medium/fp8/1-gpu"

    quantize_cmd = [
        f"{gpt_example_root}/../../../quantization/quantize.py",
        f"--model_dir={llm_gpt2_medium_model_root}",
        f"--calib_dataset={llm_datasets_root}/cnn_dailymail",
        f"--dtype={dtype}",
        f"--qformat={qformat}",
        f"--output_dir={ckpt_dir}",
    ]
    if quant_lm_head:
        quantize_cmd.append("--quantize_lm_head")
    venv_check_call(llm_venv, quantize_cmd)

    print("Building engines...")
    build_cmd = [
        "trtllm-build",
        f"--checkpoint_dir={ckpt_dir}",
        f"--output_dir={engine_dir}",
        f"--max_batch_size={1}",
        f"--max_input_len={924}",
        f"--max_seq_len={1024}",
        f"--max_num_tokens={924}",
        f"--gemm_plugin={dtype}",
    ]
    check_call(" ".join(build_cmd), shell=True, env=llm_venv._new_env)

    print('Run engines...')
    rouge1_threshold = 22.8 if qformat == "fp8_pc_pt" else (
        20.9 if quant_lm_head else 21.7)
    summary_cmd = [
        f"{gpt_example_root}/../../../summarize.py",
        f"--engine_dir={engine_dir}",
        f"--hf_model_dir={llm_gpt2_medium_model_root}", "--test_trt_llm",
        "--check_accuracy",
        f"--tensorrt_llm_rouge1_threshold={rouge1_threshold}",
        f"--dataset_dir={llm_datasets_root}", f"--rouge_dir={llm_rouge_root}"
    ]
    venv_check_call(llm_venv, summary_cmd)


@skip_pre_ada
@pytest.mark.skip_less_device(2)
@pytest.mark.parametrize("llm_gpt2_starcoder_model_root",
                         ["starcoder", "starcoderplus", "starcoder2"],
                         indirect=True)
def test_starcoder_fp8_quantization_2gpu(gpt_example_root,
                                         llm_gpt2_starcoder_model_root,
                                         llm_datasets_root, llm_rouge_root,
                                         llm_venv, cmodel_dir, engine_dir):
    "Build & Run gpt2-starcoder fp8 with 2 gpus"
    print("Quantizing and converting checkpoint...")
    dtype = "bfloat16"
    ckpt_dir = f"{cmodel_dir}/gpt2-starcoder/fp8/2-gpu"

    tp_size, pp_size = 2, 1
    world_size = tp_size * pp_size
    quantize_cmd = [
        f"{gpt_example_root}/../../../quantization/quantize.py",
        f"--model_dir={llm_gpt2_starcoder_model_root}",
        f"--calib_dataset={llm_datasets_root}/cnn_dailymail",
        f"--dtype={dtype}",
        "--qformat=fp8",
        "--kv_cache_dtype=fp8",
        f"--calib_tp_size={tp_size}",
        f"--tp_size={tp_size}",
        f"--output_dir={ckpt_dir}",
    ]
    venv_check_call(llm_venv, quantize_cmd)

    print("Building engines...")
    build_cmd = [
        "trtllm-build",
        f"--checkpoint_dir={ckpt_dir}",
        f"--output_dir={engine_dir}",
        f"--max_batch_size={1}",
        f"--max_input_len={924}",
        f"--max_seq_len={1024}",
        f"--max_num_tokens={924}",
        f"--gemm_plugin={dtype}",
        f"--workers={world_size}",
    ]
    check_call(" ".join(build_cmd), shell=True, env=llm_venv._new_env)

    print('Run engines...')
    summary_cmd = [
        f"{gpt_example_root}/../../../summarize.py",
        f"--engine_dir={engine_dir}",
        f"--hf_model_dir={llm_gpt2_starcoder_model_root}", "--test_trt_llm",
        "--check_accuracy", "--tensorrt_llm_rouge1_threshold=17.5",
        f"--dataset_dir={llm_datasets_root}", f"--rouge_dir={llm_rouge_root}"
    ]
    venv_mpi_check_call(
        llm_venv, ["mpirun", "-n", f"{world_size}", "--allow-run-as-root"],
        summary_cmd)


def test_llm_gpt2_next_1gpu(gpt_example_root, llm_venv,
                            llm_gpt2_next_model_root, engine_dir, cmodel_dir):
    "RoPE is only supported with GPTAttention plugin"
    print("Converting checkpoint...")
    dtype = "bfloat16"
    ckpt_dir = convert_weights(llm_venv=llm_venv,
                               example_root=gpt_example_root,
                               cmodel_dir=cmodel_dir,
                               model="gpt2-next",
                               model_path=llm_gpt2_next_model_root,
                               data_type=dtype)

    print("Building engines...")
    build_cmd = [
        "trtllm-build",
        f"--checkpoint_dir={ckpt_dir}",
        f"--output_dir={engine_dir}",
        f"--max_batch_size={8}",
        f"--max_input_len={924}",
        f"--max_seq_len={1024}",
        f"--gpt_attention_plugin={dtype}",
    ]

    check_call(" ".join(build_cmd), shell=True, env=llm_venv._new_env)

    print("Running inference...")
    venv_check_call(llm_venv, [
        f"{gpt_example_root}/../../../run.py", "--max_output_len=8",
        f"--engine_dir={engine_dir}",
        f"--vocab_file={ckpt_dir}/tokenizer.model", "--no_add_special_tokens"
    ])


# transformers compatibility issues
@pytest.mark.parametrize("tensor_parallel", [1, 2], ids=["tp1", "tp2"])
@pytest.mark.parametrize("use_py_session", [False, True],
                         ids=["use_cpp_session", "use_py_session"])
def test_llm_gpt2_next_prompt_tuning(gpt_example_root, llm_venv,
                                     llm_gpt2_next_model_root, cmodel_dir,
                                     engine_dir, tensor_parallel,
                                     use_py_session):
    f"gpt-next prompt tuning on {tensor_parallel} gpu(s)"
    dtype = "bfloat16"
    ckpt_dir = convert_weights(llm_venv=llm_venv,
                               example_root=gpt_example_root,
                               cmodel_dir=cmodel_dir,
                               model="gpt2-next",
                               model_path=llm_gpt2_next_model_root,
                               gpus=tensor_parallel,
                               tp_size=tensor_parallel,
                               data_type=dtype)

    print("Building engines...")
    build_cmd = [
        "trtllm-build",
        f"--checkpoint_dir={ckpt_dir}",
        f"--output_dir={engine_dir}",
        f"--max_batch_size=4",
        f"--max_input_len=924",
        f"--max_seq_len=1024",
        f"--gpt_attention_plugin={dtype}",
        "--max_prompt_embedding_table_size=200",
    ]
    check_call(" ".join(build_cmd), shell=True, env=llm_venv._new_env)

    print("Converting prompt-tuning table...")
    squad_table_nemo = Path(llm_gpt2_next_model_root
                            ).parent / "p-tuning" / "gpt2b_gpt2-squad-vt60.nemo"
    squad_table = Path(gpt_example_root) / "prompt_table_squad.npy"
    train900_table_nemo = Path(
        llm_gpt2_next_model_root
    ).parent / "p-tuning" / "gpt2b_gpt2b-train900-v2.nemo"
    train900_table = Path(gpt_example_root) / "prompt_table_train900.npy"
    for (in_file, out_file) in [(squad_table_nemo, squad_table),
                                (train900_table_nemo, train900_table)]:
        table_conv_cmd = [
            f"{gpt_example_root}/nemo_prompt_convert.py", "-i",
            str(in_file), "-o",
            str(out_file)
        ]
        venv_check_call(llm_venv, table_conv_cmd)

    merged_table = Path(gpt_example_root) / "prompt_table_train900.npy"
    table_merge_cmd = [
        f"{gpt_example_root}/merge_ptuning_tables.py",
        str(squad_table),
        str(train900_table),
        str(merged_table)
    ]
    venv_check_call(llm_venv, table_merge_cmd)

    inference_params = {
        "squad": {
            "num_v_tokens":
            50,
            "input":
            "Context: In Hinduism the spiritual teacher is known as a guru, and, in many traditions of Hinduism - especially those common in the West - the emphasis on spiritual mentorship is extremely high, with gurus often exercising a great deal of control over the lives of their disciples.\n\nQuestion: Who do gurus control?\n\nAnswer:",
            "outputs": [
                "The answer is, of course, the disciple.",
                "The guru controls the disciple's life, but",
                "The guru is the one who controls the disciple."
            ],
        },
        "train900": {
            "num_v_tokens": 20,
            "input":
            "Context: Carlsen faced Anand in the World Chess Championship 2013, at Hyatt Regency in Chennai, India, from 9 to 22 November. Carlsen won the match 6.5–3.5 by winning games five, six and nine and drawing the remainder, becoming the new World Chess Champion.\n\nQuestion: When did Carlsen become World Chess Champion?\n\nAnswer:",
            "outputs":
            ["2013", "2013" + os.linesep + os.linesep + "Question: Who"],
        }
    }

    print("Running inference...")

    def parse_output(text: str) -> list[str]:
        results = []
        while True:
            match = re.search(
                r"Output \[Text \d+ Beam \d+\]: \"([^\"]*)\"" + os.linesep,
                text, re.MULTILINE)
            if match is None:
                break
            _, end = match.span()
            results.append(match.group(1))
            text = text[end:]
        return results

    # test model without p-tuning dict
    run_cmd = [
        f"{gpt_example_root}/../../../run.py",
        "--no_add_special_tokens",
        "--max_output_len=10",
        f"--engine_dir={engine_dir}",
        f"--vocab_file={ckpt_dir}/tokenizer.model",
        f"--input_text={inference_params['squad']['input']}",
    ]

    if use_py_session:
        run_cmd.append("--use_py_session")

    output = venv_mpi_check_output(
        llm_venv, ["mpirun", "-n", f"{tensor_parallel}", "--allow-run-as-root"],
        run_cmd)
    assert any(
        similar(parse_output(output)[0][:len(ref) + 1], ref)
        for ref in inference_params["squad"]["outputs"]), "incorrect output"

    # test p-tuning task separately
    run_cmd = [
        f"{gpt_example_root}/../../../run.py",
        "--no_add_special_tokens",
        "--max_output_len=10",
        f"--engine_dir={engine_dir}",
        f"--vocab_file={ckpt_dir}/tokenizer.model",
        f"--prompt_table={squad_table}",
        f"--num_prepend_vtokens={inference_params['squad']['num_v_tokens']}",
        f"--input_text={inference_params['squad']['input']}",
        f"--no-kv_cache_enable_block_reuse",
    ]

    if use_py_session:
        run_cmd.append("--use_py_session")

    output = venv_mpi_check_output(
        llm_venv, ["mpirun", "-n", f"{tensor_parallel}", "--allow-run-as-root"],
        run_cmd)
    assert any(
        similar(parse_output(output)[0][:len(ref) + 1], ref)
        for ref in inference_params["squad"]["outputs"]), "incorrect output"

    run_cmd = [
        f"{gpt_example_root}/../../../run.py",
        "--no_add_special_tokens",
        "--max_output_len=10",
        f"--engine_dir={engine_dir}",
        f"--vocab_file={ckpt_dir}/tokenizer.model",
        f"--prompt_table={train900_table}",
        f"--num_prepend_vtokens={inference_params['train900']['num_v_tokens']}",
        f"--input_text={inference_params['train900']['input']}",
        f"--no-kv_cache_enable_block_reuse",
    ]

    if use_py_session:
        run_cmd.append("--use_py_session")

    output = venv_mpi_check_output(
        llm_venv, ["mpirun", "-n", f"{tensor_parallel}", "--allow-run-as-root"],
        run_cmd)
    assert any(
        similar(parse_output(output)[0][:len(ref) + 1], ref)
        for ref in inference_params["train900"]["outputs"]), "incorrect output"

    # test batched p-tuning tasks
    run_cmd = [
        f"{gpt_example_root}/../../../run.py",
        "--no_add_special_tokens",
        "--max_output_len=10",
        f"--engine_dir={engine_dir}",
        f"--vocab_file={ckpt_dir}/tokenizer.model",
        f"--prompt_table={merged_table}",
        f"--num_prepend_vtokens",
        str(inference_params['squad']['num_v_tokens']),
        str(inference_params['train900']['num_v_tokens']),
        f"--prompt_tasks=0,1",
        f"--input_text",
        inference_params["squad"]["input"],
        inference_params['train900']['input'],
        f"--no-kv_cache_enable_block_reuse",
    ]

    if use_py_session:
        run_cmd.append("--use_py_session")

    output = venv_mpi_check_output(
        llm_venv, ["mpirun", "-n", f"{tensor_parallel}", "--allow-run-as-root"],
        run_cmd)

    outputs = parse_output(output)
    assert any(
        similar(outputs[0][:len(ref) + 1], ref)
        for ref in inference_params["squad"]["outputs"]), "incorrect output"
    assert any(
        similar(outputs[1][:len(ref) + 1], ref)
        for ref in inference_params["train900"]["outputs"]), "incorrect output"

    # test batched and streamed p-tuning tasks
    # Streaming with py session does not return complete sequence to reliably check stop words"

    if not use_py_session and tensor_parallel == 1:
        run_cmd = [
            f"{gpt_example_root}/../../../run.py",
            "--no_add_special_tokens",
            "--max_output_len=10",
            f"--engine_dir={engine_dir}",
            f"--vocab_file={ckpt_dir}/tokenizer.model",
            f"--prompt_table={merged_table}",
            f"--num_prepend_vtokens",
            str(inference_params['squad']['num_v_tokens']),
            str(inference_params['train900']['num_v_tokens']),
            f"--prompt_tasks=0,1",
            "--streaming",
            f"--input_text",
            inference_params["squad"]["input"],
            inference_params['train900']['input'],
            f"--no-kv_cache_enable_block_reuse",
        ]

        output = venv_mpi_check_output(
            llm_venv,
            ["mpirun", "-n", f"{tensor_parallel}", "--allow-run-as-root"],
            run_cmd)

        outputs = parse_output(output)
        squad_outputs = outputs[::2]
        train900_outputs = outputs[1::2]
        for outputs, valid_outputs in [
            (squad_outputs, inference_params["squad"]["outputs"]),
            (train900_outputs, inference_params["train900"]["outputs"])
        ]:
            assert any(
                similar(outputs[-1][:len(ref) + 1], ref)
                for ref in valid_outputs), "incorrect output"
            similarities = []
            for suboutput in outputs:
                similarities.append(
                    max([
                        similarity_score(suboutput, expect)
                        for expect in valid_outputs
                    ]))
            assert (
                all(x <= y for x, y in zip(similarities, similarities[1:]))
            ), f"streaming outputs must have a monotonically increasing similarity score. valid_outputs: {valid_outputs}, outputs: {outputs}, similarities: {similarities}"


@pytest.mark.skip_less_device(4)
@pytest.mark.parametrize(
    "tp_pp_size", [(4, 1), (2, 2), (1, 4)],
    ids=lambda tp_pp_size: f'tp{tp_pp_size[0]}pp{tp_pp_size[1]}')
def test_llm_gpt2_medium_1node_4gpus(gpt_example_root,
                                     llm_gpt2_medium_model_root,
                                     llm_datasets_root, llm_rouge_root,
                                     llm_venv, cmodel_dir, engine_dir,
                                     tp_pp_size):
    print("Converting checkpoint...")
    dtype = 'float16'
    tp_size, pp_size = tp_pp_size
    world_size = tp_size * pp_size
    ckpt_dir = convert_weights(llm_venv=llm_venv,
                               example_root=gpt_example_root,
                               cmodel_dir=cmodel_dir,
                               model="gpt2-medium",
                               model_path=llm_gpt2_medium_model_root,
                               data_type=dtype,
                               gpus=world_size,
                               tp_size=tp_size,
                               pp_size=pp_size,
                               workers=world_size)

    print("Building engines...")
    build_cmd = [
        "trtllm-build",
        f"--checkpoint_dir={ckpt_dir}",
        f"--output_dir={engine_dir}",
        "--max_batch_size=8",
        "--max_input_len=924",
        "--max_seq_len=1024",
        f"--gemm_plugin={dtype}",
        f"--workers={world_size}",
    ]
    check_call(" ".join(build_cmd), shell=True, env=llm_venv._new_env)

    print("Run engines...")
    summary_cmd = [
        f"{gpt_example_root}/../../../summarize.py", "--test_trt_llm",
        f"--engine_dir={engine_dir}",
        f"--hf_model_dir={llm_gpt2_medium_model_root}", "--check_accuracy",
        "--tensorrt_llm_rouge1_threshold=19",
        f"--dataset_dir={llm_datasets_root}", f"--rouge_dir={llm_rouge_root}"
    ]
    venv_mpi_check_call(
        llm_venv, ["mpirun", "-n", f"{world_size}", "--allow-run-as-root"],
        summary_cmd)


@pytest.mark.skip_less_device(4)
@pytest.mark.parametrize(
    "use_attention_plugin", [True, False],
    ids=["enable_attention_plugin", "disable_attention_plugin"])
@pytest.mark.parametrize("use_gemm_plugin", [True, False],
                         ids=["enable_gemm_plugin", "disable_gemm_plugin"])
@pytest.mark.parametrize("context_fmha", [True, False],
                         ids=["enable_fmha", "disable_fmha"])
@pytest.mark.parametrize("parallel_build", [True, False],
                         ids=["parallel_build", "serial_build"])
def test_llm_gpt2_santacoder_1node_4gpus(gpt_example_root,
                                         llm_gpt2_santacoder_model_root,
                                         llm_venv, engine_dir, cmodel_dir,
                                         use_attention_plugin, use_gemm_plugin,
                                         context_fmha, parallel_build):
    "Build & Run GPT2 variant santacoder"
    print("Converting checkpoint...")
    dtype = 'float16'
    ckpt_dir = convert_weights(llm_venv=llm_venv,
                               example_root=gpt_example_root,
                               cmodel_dir=cmodel_dir,
                               model="gpt2-santacoder",
                               model_path=llm_gpt2_santacoder_model_root,
                               data_type=dtype,
                               gpus=4,
                               tp_size=4)

    print("Building engines...")
    build_cmd = [
        "trtllm-build",
        f"--checkpoint_dir={ckpt_dir}",
        f"--output_dir={engine_dir}",
        f"--max_batch_size={8}",
        f"--max_input_len={924}",
        f"--max_seq_len={1024}",
    ]

    if use_attention_plugin:
        build_cmd.extend([f"--gpt_attention_plugin={dtype}"])
        if context_fmha:
            build_cmd.extend(["--context_fmha=enable"])
        else:
            build_cmd.extend(["--context_fmha=disable"])
    else:
        build_cmd.extend([
            "--gpt_attention_plugin=disable",
            "--context_fmha=disable",
            "--paged_kv_cache=disable",
            "--remove_input_padding=disable",
        ])
    if use_gemm_plugin:
        build_cmd.extend([f"--gemm_plugin={dtype}"])
    if parallel_build:
        build_cmd.extend(["--workers=4"])

    check_call(" ".join(build_cmd), shell=True, env=llm_venv._new_env)

    print('Run gpt2-santacoder...')
    venv_mpi_check_call(
        llm_venv,
        ["mpirun", "--allow-run-as-root", "--oversubscribe", "-np", "4"], [
            f"{gpt_example_root}/../../../run.py", "--max_output_len=20",
            f"--engine_dir={engine_dir}", "--tokenizer_dir",
            llm_gpt2_santacoder_model_root, "--input_text",
            "def print_hello_world():", "--no_add_special_tokens"
        ])


@pytest.mark.skip_less_device(4)
@pytest.mark.parametrize(
    "use_attention_plugin", [True, False],
    ids=["enable_attention_plugin", "disable_attention_plugin"])
@pytest.mark.parametrize("use_gemm_plugin", [True, False],
                         ids=["enable_gemm_plugin", "disable_gemm_plugin"])
@pytest.mark.parametrize("context_fmha", [True, False],
                         ids=["enable_fmha", "disable_fmha"])
@pytest.mark.parametrize("llm_gpt2_starcoder_model_root",
                         ["starcoder", "starcoderplus", "starcoder2"],
                         indirect=True)
def test_llm_gpt2_starcoder_1node_4gpus(gpt_example_root,
                                        llm_gpt2_starcoder_model_root,
                                        llm_datasets_root, llm_rouge_root,
                                        llm_venv, cmodel_dir, engine_dir,
                                        use_attention_plugin, use_gemm_plugin,
                                        context_fmha):
    "Build & Run GPT2 variant starcoder"
    print("Converting checkpoint...")
    dtype = 'float16'
    ckpt_dir = convert_weights(llm_venv=llm_venv,
                               example_root=gpt_example_root,
                               cmodel_dir=cmodel_dir,
                               model="gpt2-starcoder",
                               model_path=llm_gpt2_starcoder_model_root,
                               data_type=dtype,
                               gpus=4,
                               tp_size=4)

    print("Building engines...")
    build_cmd = [
        "trtllm-build",
        f"--checkpoint_dir={ckpt_dir}",
        f"--output_dir={engine_dir}",
        f"--max_batch_size={8}",
        f"--max_input_len={924}",
        f"--max_seq_len={1024}",
        "--workers=4",
    ]

    if use_attention_plugin:
        build_cmd.extend([f"--gpt_attention_plugin={dtype}"])
        if context_fmha:
            build_cmd.extend(["--context_fmha=enable"])
        else:
            build_cmd.extend(["--context_fmha=disable"])
    else:
        build_cmd.extend([
            "--gpt_attention_plugin=disable",
            "--context_fmha=disable",
            "--paged_kv_cache=disable",
            "--remove_input_padding=disable",
        ])
    if use_gemm_plugin:
        build_cmd.extend([f"--gemm_plugin={dtype}"])

    check_call(" ".join(build_cmd), shell=True, env=llm_venv._new_env)

    print('Run gpt2-starcoder...')
    venv_mpi_check_call(
        llm_venv,
        ["mpirun", "--allow-run-as-root", "--oversubscribe", "-np", "4"], [
            f"{gpt_example_root}/../../../run.py",
            "--max_output_len=20",
            f"--engine_dir={engine_dir}",
            "--tokenizer_dir",
            llm_gpt2_starcoder_model_root,
            "--input_text",
            "def print_hello_world():",
            "--no_add_special_tokens",
        ])

    summary_cmd = generate_summary_cmd(
        gpt_example_root,
        "no_add_special_tokens",
        batch_size=1,
        engine_dir=engine_dir,
        eval_task="code_completion",
        hf_model_dir=llm_gpt2_starcoder_model_root,
        max_attention_window_size=4096,
        tensorrt_llm_rouge1_threshold=25,
        dataset_dir=llm_datasets_root,
        rouge_dir=llm_rouge_root)

    print('Run gpt2-starcoder summarize...')
    venv_mpi_check_call(
        llm_venv,
        ["mpirun", "--allow-run-as-root", "--oversubscribe", "-np", "4"],
        summary_cmd)


@skip_post_blackwell
@pytest.mark.skip_less_host_memory(250000)
def test_llm_gpt2_starcoder_1gpus(gpt_example_root,
                                  llm_gpt2_starcoder_model_root, llm_venv,
                                  engine_dir, cmodel_dir):
    "Build & Run GPT2 variant starcoder on single gpu"
    print("Converting checkpoint...")
    print(f"cmodel dir is {cmodel_dir}")
    dtype = 'float16'
    ckpt_dir = convert_weights(llm_venv=llm_venv,
                               example_root=gpt_example_root,
                               cmodel_dir=cmodel_dir,
                               model="gpt2-starcoder",
                               model_path=llm_gpt2_starcoder_model_root,
                               data_type=dtype)

    print("Building engines...")
    build_cmd = [
        "trtllm-build",
        f"--checkpoint_dir={ckpt_dir}",
        f"--output_dir={engine_dir}",
        f"--max_batch_size={8}",
        f"--max_input_len={924}",
        f"--max_seq_len={1024}",
        f"--gpt_attention_plugin={dtype}",
        "--context_fmha=enable",
        f"--gemm_plugin={dtype}",
    ]

    check_call(" ".join(build_cmd), shell=True, env=llm_venv._new_env)

    print('Run gpt2-starcoder...')
    summary_cmd = [
        f"{gpt_example_root}/../../../run.py", "--max_output_len=20",
        f"--engine_dir={engine_dir}", "--tokenizer_dir",
        llm_gpt2_starcoder_model_root, "--input_text",
        "def print_hello_world():", "--no_add_special_tokens"
    ]

    venv_check_call(llm_venv, summary_cmd)


@skip_post_blackwell
@pytest.mark.skip_less_host_memory(250000)
@pytest.mark.parametrize("dtype", ["float16"])
@pytest.mark.parametrize("precision", ["int8", "int4"])
@pytest.mark.parametrize("llm_gpt2_starcoder_model_root",
                         ["starcoder", "starcoderplus", "starcoder2"],
                         indirect=True)
def test_llm_gpt2_starcoder_weight_only(gpt_example_root,
                                        llm_gpt2_starcoder_model_root,
                                        llm_datasets_root, llm_rouge_root,
                                        llm_venv, cmodel_dir, engine_dir, dtype,
                                        precision):
    "Build & Run GPT2 variant starcoder with int8/int4 weight only"

    print("Converting checkpoint...")
    ckpt_dir = convert_weights(llm_venv=llm_venv,
                               example_root=gpt_example_root,
                               cmodel_dir=cmodel_dir,
                               model="gpt2-starcoder",
                               model_path=llm_gpt2_starcoder_model_root,
                               data_type=dtype,
                               use_weight_only=True,
                               weight_only_precision=precision)

    print("Building engines...")
    build_cmd = [
        "trtllm-build",
        f"--checkpoint_dir={ckpt_dir}",
        f"--output_dir={engine_dir}",
        f"--max_batch_size={8}",
        f"--max_input_len={924}",
        f"--max_seq_len={1024}",
        f"--gpt_attention_plugin={dtype}",
        "--context_fmha=enable",
        f"--gemm_plugin={dtype}",
    ]

    check_call(" ".join(build_cmd), shell=True, env=llm_venv._new_env)

    print('Run gpt2-starcoder...')
    summary_cmd = [
        f"{gpt_example_root}/../../../run.py",
        "--max_output_len=20",
        f"--engine_dir={engine_dir}",
        "--tokenizer_dir",
        llm_gpt2_starcoder_model_root,
        "--input_text",
        "def print_hello_world():",
        "--no_add_special_tokens",
    ]

    venv_check_call(llm_venv, summary_cmd)

    summary_cmd = generate_summary_cmd(
        gpt_example_root,
        "no_add_special_tokens",
        batch_size=1,
        engine_dir=engine_dir,
        eval_task="code_completion",
        hf_model_dir=llm_gpt2_starcoder_model_root,
        max_attention_window_size=4096,
        tensorrt_llm_rouge1_threshold=25,
        dataset_dir=llm_datasets_root,
        rouge_dir=llm_rouge_root)

    print('Run gpt2-starcoder summarize...')
    venv_check_call(llm_venv, summary_cmd)


@pytest.mark.parametrize("tensor_parallel", [1, 2], ids=["tp1", "tp2"])
@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
def test_llm_gpt2_starcoder2(gpt_example_root, llm_gpt2_starcoder2_model_root,
                             llm_datasets_root, llm_rouge_root, llm_venv,
                             cmodel_dir, engine_dir, dtype, tensor_parallel):
    "Build & Run GPT2 variant starcoder2 on single gpu"
    print("Converting checkpoint...")
    print(f"cmodel dir is {cmodel_dir}")
    ckpt_dir = convert_weights(llm_venv=llm_venv,
                               example_root=gpt_example_root,
                               cmodel_dir=cmodel_dir,
                               model="gpt2-starcoder2",
                               model_path=llm_gpt2_starcoder2_model_root,
                               data_type=dtype,
                               gpus=tensor_parallel,
                               tp_size=tensor_parallel)

    print("Building engines...")
    build_cmd = [
        "trtllm-build",
        f"--checkpoint_dir={ckpt_dir}",
        f"--output_dir={engine_dir}",
        f"--max_batch_size={1}",
        f"--max_input_len={1024}",
        f"--max_seq_len={1024}",
        f"--gpt_attention_plugin={dtype}",
        "--context_fmha=enable",
        f"--gemm_plugin={dtype}",
    ]

    check_call(" ".join(build_cmd), shell=True, env=llm_venv._new_env)

    print('Run gpt2-starcoder...')
    venv_mpi_check_call(
        llm_venv,
        parse_mpi_cmd([
            "mpirun", "--allow-run-as-root", "--oversubscribe", "-np",
            str(tensor_parallel)
        ]), [
            f"{gpt_example_root}/../../../summarize.py", "--batch_size=1",
            f"--engine_dir={engine_dir}", "--test_trt_llm", "--check_accuracy",
            "--eval_task=code_completion",
            f"--hf_model_dir={llm_gpt2_starcoder2_model_root}",
            "--no_add_special_tokens", "--max_attention_window_size=4096",
            "--tensorrt_llm_rouge1_threshold=25",
            f"--dataset_dir={llm_datasets_root}",
            f"--rouge_dir={llm_rouge_root}"
        ])


@pytest.mark.parametrize("qformat", ["fp8", "full_prec"])
@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
@pytest.mark.parametrize("minitron_model_root", ["4b"], indirect=True)
def test_llm_minitron(gpt_example_root, minitron_model_root, llm_datasets_root,
                      llm_rouge_root, llm_venv, cmodel_dir, engine_dir, dtype,
                      qformat):
    skip_fp8_pre_ada(qformat == 'fp8')
    "Build & Run GPT2 variant minitron on single gpu"

    if qformat == 'fp8':
        print("Quantizing and converting checkpoint...")
        ckpt_dir = f"{cmodel_dir}/minitron/fp8/1-gpu"

        quantize_cmd = [
            f"{gpt_example_root}/../../../quantization/quantize.py",
            f"--model_dir={minitron_model_root}",
            f"--calib_dataset={llm_datasets_root}/cnn_dailymail",
            f"--dtype={dtype}",
            "--qformat=fp8",
            "--kv_cache_dtype=fp8",
            f"--output_dir={ckpt_dir}",
        ]
        venv_check_call(llm_venv, quantize_cmd)
    else:
        print(f"Converting checkpoint...")
        ckpt_dir = convert_weights(llm_venv=llm_venv,
                                   example_root=gpt_example_root,
                                   cmodel_dir=cmodel_dir,
                                   model="gpt2-minitron",
                                   model_path=minitron_model_root,
                                   data_type=dtype,
                                   gpus=1,
                                   tp_size=1)

    print("Building engines...")
    build_cmd = [
        "trtllm-build",
        f"--checkpoint_dir={ckpt_dir}",
        f"--output_dir={engine_dir}",
        f"--max_batch_size={1}",
        f"--max_input_len={1024}",
        f"--max_seq_len={1024}",
        f"--gpt_attention_plugin={dtype}",
        "--context_fmha=enable",
        f"--gemm_plugin={dtype}",
    ]

    check_call(" ".join(build_cmd), shell=True, env=llm_venv._new_env)

    print('Run Minitron...')
    venv_mpi_check_call(
        llm_venv,
        parse_mpi_cmd(
            ["mpirun", "--allow-run-as-root", "--oversubscribe", "-np", "1"]), [
                f"{gpt_example_root}/../../../summarize.py", "--batch_size=1",
                f"--engine_dir={engine_dir}", "--test_trt_llm",
                "--check_accuracy", "--eval_task", "code_completion",
                "--hf_model_dir", minitron_model_root,
                "--no_add_special_tokens", "--max_attention_window_size=4096",
                "--tensorrt_llm_rouge1_threshold=29",
                f"--dataset_dir={llm_datasets_root}",
                f"--rouge_dir={llm_rouge_root}"
            ])


@pytest.mark.skip_less_device(2)
@pytest.mark.parametrize("embedding_sharding_dim", [0, 1])
@pytest.mark.parametrize("dtype", ["float16"])
def test_llm_gpt2_parallel_embedding_2gpu(gpt_example_root, llm_venv,
                                          llm_gpt2_model_root,
                                          llm_datasets_root, llm_rouge_root,
                                          cmodel_dir, engine_dir,
                                          embedding_sharding_dim, dtype):
    "GPT2 with parallel embedding"
    print("Converting checkpoint...")
    ckpt_dir = convert_weights(llm_venv=llm_venv,
                               example_root=gpt_example_root,
                               cmodel_dir=cmodel_dir,
                               model="gpt2",
                               model_path=llm_gpt2_model_root,
                               data_type=dtype,
                               gpus=2,
                               tp_size=2,
                               use_parallel_embedding=True,
                               embedding_sharding_dim=embedding_sharding_dim)

    print("Building engines...")
    build_cmd = [
        "trtllm-build",
        f"--checkpoint_dir={ckpt_dir}",
        f"--output_dir={engine_dir}",
        f"--max_batch_size={8}",
        f"--max_input_len={1000}",
        f"--max_seq_len={1024}",
        f"--gpt_attention_plugin={dtype}",
        "--workers=2",
    ]
    check_call(" ".join(build_cmd), shell=True, env=llm_venv._new_env)

    print("Running inference...")
    venv_mpi_check_call(llm_venv, [
        "mpirun", "--allow-run-as-root", "--oversubscribe", "-np", "2"
    ], [
        f"{gpt_example_root}/../../../summarize.py", "--batch_size=8",
        "--test_trt_llm", "--check_accuracy",
        "--tensorrt_llm_rouge1_threshold=13.5", f"--engine_dir={engine_dir}",
        f"--hf_model_dir={llm_gpt2_model_root}", "--no_add_special_tokens",
        f"--dataset_dir={llm_datasets_root}", f"--rouge_dir={llm_rouge_root}"
    ])


@pytest.mark.parametrize("llm_gpt2b_lora_model_root",
                         [("gpt2b_lora-900.nemo", "gpt2b_lora-stories.nemo")],
                         ids=["900_stories"],
                         indirect=True)
def test_llm_gpt2_multi_lora_1gpu(gpt_example_root, llm_venv,
                                  llm_gpt2_next_model_root, cmodel_dir,
                                  engine_dir, llm_gpt2b_lora_model_root):
    "gpt2 run lora with nemo checkpoint on 1 gpu"
    print("Converting checkpoint...")
    dtype = "float16"
    ckpt_dir = convert_weights(llm_venv=llm_venv,
                               example_root=gpt_example_root,
                               cmodel_dir=cmodel_dir,
                               model="gpt2-next-lora",
                               model_path=llm_gpt2_next_model_root,
                               data_type=dtype)

    print("Building engines...")
    lora_900, lora_stories = llm_gpt2b_lora_model_root.split(",")
    build_cmd = [
        "trtllm-build",
        f"--checkpoint_dir={ckpt_dir}",
        f"--output_dir={engine_dir}",
        f"--max_batch_size={4}",
        f"--max_input_len={512}",
        f"--max_seq_len={562}",
        f"--max_beam_width={2}",
        f"--gpt_attention_plugin={dtype}",
        "--remove_input_padding=enable",
        "--paged_kv_cache=enable",
        "--context_fmha=enable",
        f"--lora_plugin={dtype}",
        "--lora_dir",
        lora_900,
        lora_stories,
        "--lora_ckpt_source=nemo",
        "--lora_target_modules",
        "attn_qkv",
    ]

    check_call(" ".join(build_cmd), shell=True, env=llm_venv._new_env)

    run_cmd = [
        f"{gpt_example_root}/../../../run.py",
        "--max_output_len=20",
        "--use_py_session",
        f"--vocab_file={ckpt_dir}/tokenizer.model",
        f"--engine_dir={engine_dir}",
        "--lora_task_uids",
        "0",
        "-1",
        "1",
        "--no_add_special_tokens",
        "--input_text",
        INPUT_TEXT_1,
        INPUT_TEXT_2,
        INPUT_TEXT_2,
    ]

    output = venv_check_output(llm_venv, run_cmd)
    output = parse_output(output)
    expected_output = [
        [
            "He surprised the Canadians on May 28 in what became known as the Battle of Jumonville",
            "Washington, with Tanaghrisson and his party, surprised the Canadians on May 28 in"
        ],
        [
            "The game is played with a deck of cards, and the player who has the most"
        ],
        [
            "You are a wizard who is a wizard. You are a wizard who is",
            'The job title is "Spellcaster" and the job description is "Spell"'
        ],
    ]

    for idx, result in enumerate(output):
        assert any([similar(item, result)
                    for item in expected_output[idx]]), f"output is {output}"


@skip_post_blackwell
@pytest.mark.skip_less_device_memory(50000)
@pytest.mark.parametrize("data_type", ['float16', 'fp8'],
                         ids=['base_fp16', 'base_fp8'])
@pytest.mark.parametrize("lora_data_type", ['float16'], ids=['lora_fp16'])
@pytest.mark.parametrize("llm_gpt2_starcoder_model_root", ['starcoder2'],
                         indirect=True)
@pytest.mark.parametrize("llm_lora_model_root",
                         ['peft-lora-starcoder2-15b-unity-copilot'],
                         indirect=True)
def test_llm_gpt_starcoder_lora_1gpu(data_type, lora_data_type,
                                     gpt_example_root,
                                     llm_gpt2_starcoder_model_root,
                                     llm_datasets_root, llm_venv, cmodel_dir,
                                     engine_dir, llm_lora_model_root,
                                     qcache_dir):
    "run starcoder2 lora test on 1gpu"
    if data_type == 'fp8':
        skip_fp8_pre_ada(use_fp8=True)
    else:
        if get_device_memory() < 80000:
            pytest.skip("GPU memory is not sufficient.")

    print("Converting checkpoint...")
    model_name = 'starcoder2-lora'

    if data_type == 'fp8':
        model_dir = quantize_data(
            llm_venv,
            gpt_example_root,
            model_dir=llm_gpt2_starcoder_model_root,
            calib_dataset=f"{llm_datasets_root}/cnn_dailymail",
            dtype="float16",
            qformat="fp8",
            kv_cache_dtype="fp8",
            quantize_dir=qcache_dir,
            calib_size=512)
    else:
        model_dir = convert_weights(llm_venv=llm_venv,
                                    example_root=gpt_example_root,
                                    cmodel_dir=cmodel_dir,
                                    model=model_name,
                                    model_path=llm_gpt2_starcoder_model_root)

    print("Build engines...")
    build_cmd = [
        "trtllm-build",
        f"--checkpoint_dir={model_dir}",
        f"--output_dir={engine_dir}",
        "--lora_plugin=auto",
        "--gemm_plugin=auto",
        f"--lora_dir={llm_lora_model_root}",
    ]
    check_call(" ".join(build_cmd), shell=True, env=llm_venv._new_env)

    ref_1 = [
        610, 1489, 100, 7670, 100, 5879, 2284, 303, 1489, 459, 8302, 10914,
        16013, 222, 222, 610, 1489, 100, 7670, 100, 5879, 100, 115, 100, 5598,
        45, 115
    ]
    ref_2 = [
        610, 1489, 100, 7670, 100, 5879, 2284, 303, 1489, 459, 8302, 10914, 678,
        222, 222, 610, 1489, 100, 7670, 100, 5879, 100, 115, 100, 5598, 45, 115
    ]

    input_text = "def print_hello_world():"

    print(f"Run inference with lora id 0...")
    venv_check_call(llm_venv, [
        f"{gpt_example_root}/../../../run.py",
        "--max_output_len=20",
        f"--input_text={input_text}",
        "--lora_task_uids=0",
        f"--tokenizer_dir={llm_gpt2_starcoder_model_root}",
        f"--engine_dir={engine_dir}",
        f"--output_csv={llm_venv.get_working_directory()}/use_lora.csv",
        "--no_add_special_tokens",
        "--use_py_session",
    ])

    with open(f"{llm_venv.get_working_directory()}/use_lora.csv") as f:
        predict = csv.reader(f)
        predict = next(predict)
    predict = [int(p) for p in predict]
    assert ref_1 == predict or data_type != "float16"

    print(f"Run inference with lora id -1...")
    venv_check_call(llm_venv, [
        f"{gpt_example_root}/../../../run.py",
        "--max_output_len=20",
        f"--input_text={input_text}",
        "--lora_task_uids=-1",
        f"--tokenizer_dir={llm_gpt2_starcoder_model_root}",
        f"--engine_dir={engine_dir}",
        f"--output_csv={llm_venv.get_working_directory()}/no_lora.csv",
        "--no_add_special_tokens",
        "--use_py_session",
    ])

    with open(f"{llm_venv.get_working_directory()}/no_lora.csv") as f:
        predict = csv.reader(f)
        predict = next(predict)
    predict = [int(p) for p in predict]
    assert ref_2 == predict or data_type != "float16"


@pytest.mark.parametrize("llm_gpt2_starcoder_model_root", ['starcoder2'],
                         indirect=True)
def test_llm_starcoder2_sqootb_single_gpu(gpt_example_root, llm_venv,
                                          llm_gpt2_starcoder_model_root,
                                          llm_datasets_root, llm_rouge_root,
                                          cmodel_dir, engine_dir):
    "Starcoder2-smooth test on single gpu"
    print("Quantization...")
    dtype = 'float16'
    ckpt_dir = f"{cmodel_dir}/starcoder2/int8_sq/1-gpu"

    quantize_cmd = [
        f"{gpt_example_root}/../../../quantization/quantize.py",
        f"--model_dir={llm_gpt2_starcoder_model_root}",
        f"--calib_dataset={llm_datasets_root}/cnn_dailymail",
        f"--dtype={dtype}",
        "--qformat=int8_sq",
        f"--output_dir={ckpt_dir}",
    ]
    venv_check_call(llm_venv, quantize_cmd)

    print("Building engines...")
    build_cmd = [
        "trtllm-build",
        f"--checkpoint_dir={ckpt_dir}",
        f"--output_dir={engine_dir}",
        f"--max_batch_size={8}",
        f"--max_seq_len={4096}",
        f"--gpt_attention_plugin={dtype}",
    ]

    check_call(" ".join(build_cmd), shell=True, env=llm_venv._new_env)

    print('Run starcoder2...')
    venv_mpi_check_call(
        llm_venv,
        parse_mpi_cmd(
            ["mpirun", "--allow-run-as-root", "--oversubscribe", "-np", "1"]), [
                f"{gpt_example_root}/../../../summarize.py", "--batch_size=1",
                f"--engine_dir={engine_dir}", "--test_trt_llm",
                "--check_accuracy", "--eval_task", "code_completion",
                f"--hf_model_dir={llm_gpt2_starcoder_model_root}",
                "--no_add_special_tokens", "--max_attention_window_size=4096",
                "--tensorrt_llm_rouge1_threshold=25",
                f"--dataset_dir={llm_datasets_root}",
                f"--rouge_dir={llm_rouge_root}"
            ])


@skip_pre_ada
@pytest.mark.parametrize("minitron_model_root", ["4b"], indirect=True)
def test_llm_minitron_fp8_with_pseudo_loras(gpt_example_root,
                                            minitron_model_root,
                                            llm_datasets_root,
                                            llm_venv,
                                            cmodel_dir,
                                            engine_dir,
                                            dtype='bfloat16'):
    "Run Minitron model with multiple pseudo LoRAs."

    # Quantize the base model to fp8.
    print("Quantizing and converting checkpoint...")
    ckpt_dir = f"{cmodel_dir}/minitron/fp8/1-gpu"

    quantize_cmd = [
        f"{gpt_example_root}/../../../quantization/quantize.py",
        f"--model_dir={minitron_model_root}",
        f"--calib_dataset={llm_datasets_root}/cnn_dailymail",
        f"--dtype={dtype}",
        "--qformat=fp8",
        "--kv_cache_dtype=fp8",
        f"--output_dir={ckpt_dir}",
    ]
    venv_check_call(llm_venv, quantize_cmd)

    test_multi_lora_support(
        hf_model_dir=minitron_model_root,
        tllm_ckpt_dir=ckpt_dir,
        engine_dir=engine_dir,
        llm_venv=llm_venv,
        example_root=gpt_example_root,
        num_loras=2,
        lora_rank=8,
        target_hf_modules=["q_proj", "k_proj", "v_proj"],
        target_trtllm_modules=["attn_q", "attn_k", "attn_v"],
        zero_lora_weights=True,
    )
