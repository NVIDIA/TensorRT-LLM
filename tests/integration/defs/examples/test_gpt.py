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

import defs.ci_profiler
import pytest
from defs.common import (convert_weights, parse_output, quantize_data,
                         run_and_check, similar, similarity_score,
                         test_multi_lora_support, venv_check_call,
                         venv_check_output, venv_mpi_check_output)
from defs.conftest import (get_device_memory, get_sm_version, skip_fp8_pre_ada,
                           skip_post_blackwell, skip_pre_ada)
from defs.trt_test_alternative import check_call

from tensorrt_llm import LLM
from tensorrt_llm.executor.request import LoRARequest
from tensorrt_llm.lora_manager import LoraConfig
from tensorrt_llm.sampling_params import SamplingParams

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

# streaming can can skip outputs, if the next set of outputs arrive.
# this means that the is_equal flag is currently flaky: https://nvbugspro.nvidia.com/bug/4851644
# assert is_equal


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


@pytest.mark.skip_less_device_memory(
    20000)  # Conservative 20GB requirement for GPT-OSS-20B
@pytest.mark.parametrize("gpt_oss_model_root", [
    "gpt-oss-20b",
], indirect=True)
@pytest.mark.parametrize("llm_lora_model_root",
                         ['gpt-oss-20b-lora-adapter_NIM_r8'],
                         indirect=True)
def test_gpt_oss_20b_lora_torch(gpt_example_root, llm_venv, gpt_oss_model_root,
                                llm_datasets_root, llm_rouge_root, engine_dir,
                                cmodel_dir, llm_lora_model_root):
    """Run GPT-OSS-20B with LoRA adapter using Torch backend."""

    print(f"Using LoRA from: {llm_lora_model_root}")

    defs.ci_profiler.start("test_gpt_oss_20b_lora_torch")

    lora_config = LoraConfig(
        lora_dir=[llm_lora_model_root],
        max_lora_rank=8,  # Match adapter_config.json "r": 8
        max_loras=1,
        max_cpu_loras=1,
    )

    with LLM(model=gpt_oss_model_root, lora_config=lora_config) as llm:

        prompts = [
            "User: Message Mason saying that we should compete in next week's football tournament, and tell him that the winner will get $100.\n\nAssistant: "
        ]

        sampling_params = SamplingParams(max_tokens=50)

        lora_request = [LoRARequest("gpt-oss-lora", 0, llm_lora_model_root)]

        print("Running inference with real LoRA adapter...")
        outputs = llm.generate(prompts,
                               sampling_params,
                               lora_request=lora_request)

        expected_output = " Hey Mason! I hope you're doing well. I was thinking about the next week's football tournament and I wanted to give you a hint that we should compete in it. The winner will be a great opportunity for us to win $100.\n\nUser:"

        for i, output in enumerate(outputs):
            print(f"Prompt {i+1}: {prompts[i]}")
            print(f"Response {i+1}: {output.outputs[0].text}")
            print("-" * 50)

        assert len(outputs) == 1
        assert len(outputs[0].outputs) > 0
        generated_text = outputs[0].outputs[0].text
        similarity = similarity_score(generated_text, expected_output)
        assert similar(generated_text, expected_output, threshold=0.8), \
            f"Output similarity too low (similarity={similarity:.2%})!\nExpected: {repr(expected_output)}\nGot: {repr(generated_text)}"

    defs.ci_profiler.stop("test_gpt_oss_20b_lora_torch")
    print(
        f"test_gpt_oss_20b_lora_torch: {defs.ci_profiler.elapsed_time_in_sec('test_gpt_oss_20b_lora_torch')} sec"
    )
