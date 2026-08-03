# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import copy
import os
import platform
import random
import re
import socket
import tempfile
import time
from difflib import SequenceMatcher
from typing import Any

import yaml
from packaging import version

from tensorrt_llm import LLM as LLM_torch
from tensorrt_llm._utils import get_free_port
from tensorrt_llm.executor.request import LoRARequest
from tensorrt_llm.lora_manager import LoraConfig
from tensorrt_llm.sampling_params import SamplingParams

from .trt_test_alternative import (check_call, check_output, print_info,
                                   print_warning)


def venv_check_call(venv, cmd, env=None, **kwargs):

    def _war_check_call(*args, **kwargs):
        kwargs["cwd"] = venv.get_working_directory()
        return check_call(*args, **kwargs)

    venv.run_cmd(cmd, caller=_war_check_call, env=env, **kwargs)


def venv_check_output(venv, cmd, env=None, **kwargs):

    def _war_check_output(*args, **kwargs):
        kwargs["cwd"] = venv.get_working_directory()
        output = check_output(*args, **kwargs)
        return output

    return venv.run_cmd(cmd, caller=_war_check_output, env=env, **kwargs)


def resolve_llm_model_path(model_path: str) -> str:
    """Resolve a model subpath relative to the test LLM model root."""
    if os.path.isabs(model_path):
        return model_path

    from .conftest import llm_models_root
    return os.path.join(llm_models_root(), model_path)


def venv_mpi_check_call(venv, mpi_cmd, python_cmd, **kwargs):
    """
    This function WAR check_call() to run python_cmd with mpi.
    If mpi_cmd = ["mpirun", "-n", "2"] and python_cmd = ["run.py"], the command will be:

    "mpirun -n 2 <venv python> run.py"

    """

    def _war_check_call(*args, **kwargs):
        assert len(args) == 1, "bad args"
        arg_list, = args
        merged_cmd = copy.deepcopy(mpi_cmd)
        merged_cmd.extend(arg_list)
        kwargs["cwd"] = venv.get_working_directory()
        return check_call(merged_cmd, **kwargs)

    venv.run_cmd(python_cmd, caller=_war_check_call, **kwargs)


def venv_mpi_check_output(venv, mpi_cmd, python_cmd, env=None, **kwargs):
    """
    This function WAR check_output() to run python_cmd with mpi.
    If mpi_cmd = ["mpirun", "-n", "2"] and python_cmd = ["run.py"], the command will be:

    "mpirun -n 2 <venv python> run.py"

    """

    def _war_check_output(*args, **kwargs):
        assert len(args) == 1, "bad args"
        arg_list, = args
        merged_cmd = copy.deepcopy(mpi_cmd)
        merged_cmd.extend(arg_list)
        kwargs["cwd"] = venv.get_working_directory()
        return check_output(merged_cmd, **kwargs)

    return venv.run_cmd(python_cmd, caller=_war_check_output, env=env, **kwargs)


def parse_mpi_cmd(cmd):
    if platform.system() == "Windows":
        # Simply fetch necessary args from Linux cmd then fill Windows cmd because:
        # 1. We use Microsoft MPI on Windows, while Open-MPI on Linux. Args are not compatible.
        # 2. Multi-GPU is actually not supported on Windows for now.
        flags = ("-n", "-np")
        # append None if not found
        indices = [idx for idx in range(len(cmd)) if cmd[idx] in flags] + [
            None,
        ]
        index = indices[0]
        return ["mpiexec", cmd[index], cmd[index + 1]] if index else cmd
    else:
        return cmd


def similarity_score(a, b):
    "similar compare a and b "
    return SequenceMatcher(None, a, b).ratio()


def similar(a, b, threshold=0.8):
    "similar compare a and b "
    return similarity_score(a, b) >= threshold


def generate_summary_cmd(example_root, *args, **kwargs):
    "generate summary command"
    summarize_script = f"{example_root}/../../../summarize.py" if "core" in example_root else f"{example_root}/../summarize.py"
    summary_cmd = [summarize_script, "--test_trt_llm", "--check_accuracy"]

    for key, value in kwargs.items():
        if isinstance(value, bool):
            if value:
                summary_cmd.append(f"--{key}")
        elif isinstance(value, list):  # Support max_attention_window
            summary_cmd.extend([f"--{key}", *map(str, value)])
        else:
            summary_cmd.extend([f"--{key}", f"{value}"])

    for arg in args:
        summary_cmd.append(f"--{arg}")

    return summary_cmd


def get_trt_llm_lib_dir(venv):
    output = venv.run_raw(
        "import tensorrt_llm; print(f'{tensorrt_llm.__path__[0]}/libs')",
        caller=check_output).strip()

    if "TensorRT LLM version: " in output:
        output = output.split('\n')[-1]

    return output.strip()


def trt_gte(venv, major: int, minor: int = 0):
    """
    Check if TRT version is greater than or equal to major.minor
    """
    ver = venv.run_output("import tensorrt;print(tensorrt.__version__)")
    trt_ver = version.parse(ver)
    return trt_ver.major >= major and trt_ver.minor >= minor


def parse_output(text):
    "parse output"
    results = []
    text_lists = re.split(r"Input \[Text \d\]:", text)
    for item in text_lists:
        item = item.replace(os.linesep, "")
        while True:
            match = re.search(
                r"(Output \[Text \d+ Beam \d+\]: \"(.*?)\")(Output|Input|$)",
                item, re.MULTILINE)
            if match is None:
                break
            _, end = match.span(1)
            results.append(match.group(2))
            item = item[end:]

    return results


def run_and_check(llm_venv, run_cmd, valid_outputs, streaming=False):
    print("Running inference...")
    output = venv_check_output(llm_venv, run_cmd)

    if not streaming:
        output = parse_output(output)[0]
        assert any([
            similar(output, expect, threshold=0.95) for expect in valid_outputs
        ]), f"output is: {output}"
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
        assert any([
            similar(output, expect, threshold=0.95) for expect in valid_outputs
        ]), f"output is: {output}"


def generate_dummy_loras(
        hf_model_dir,
        lora_output_dir,
        num_loras=1,
        lora_rank=8,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        zero_weights=False):

    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM

    print("Creating pseudo LoRAs...")

    # Avoid meta tensors by loading model to CPU first (ensures all parameters are materialized)
    try:
        model = AutoModelForCausalLM.from_pretrained(
            hf_model_dir,
            dtype=torch.float16,
            device_map=None,  # Load everything to CPU first
            trust_remote_code=True,
            low_cpu_mem_usage=False,
        )
    except Exception:
        # Fallback to auto device mapping if CPU loading fails
        print(
            "Warning: Loading model to CPU failed, falling back to auto device mapping"
        )
        model = AutoModelForCausalLM.from_pretrained(
            hf_model_dir,
            dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )

    lora_config = LoraConfig(r=lora_rank,
                             target_modules=target_modules,
                             bias="none",
                             task_type="CAUSAL_LM")
    lora_output_paths = []
    for lora_idx in range(num_loras):
        lora_model = get_peft_model(model, lora_config)
        if zero_weights:
            for param in lora_model.parameters():
                param.data.zero_()

        pseudo_lora_dir = f"{lora_output_dir}/pseudo_lora_{lora_idx}"
        lora_model.save_pretrained(pseudo_lora_dir)
        lora_output_paths.append(pseudo_lora_dir)
    return lora_output_paths


def get_test_prompts(use_code_prompts: bool = False) -> list[str]:
    """Get test prompts for LoRA testing.

    Args:
        use_code_prompts: If True, return code-related prompts. If False, return general prompts.

    Returns:
        List of test prompts.
    """
    if use_code_prompts:
        return [
            "Write a function that outputs the fibonacci sequence.",
            "Convert the following C++ code to Python:  x = 0;x++;",
            "Find the largest prime factor of 42.",
            "write a unit test for this function: $(cat fib.py)",
            "# A simple python function to remove whitespace from a string:",
            "How to load CodeLlama from HuggingFace?",
        ]
    else:
        return [
            "Hey how are you doing today?",
            "How is the weather in Seattle, WA?",
            "Is it ok to fill diesel in a petrol car?",
            "Can you check the top 5 trending songs on spotify?",
            "What is the capital of France?",
            "How to load CodeLlama from HuggingFace?",
        ]


def get_test_prompts_for_torch() -> list[str]:
    """Get test prompts for LoRA Torch testing.

    Returns:
        List of test prompts.
    """
    return [
        "Hey how are you doing today?",
        "How is the weather in Seattle, WA?",
        "Is it ok to fill diesel in a petrol car?",
        "Can you check the top 5 trending songs on spotify?",
        "What is the capital of France?",
    ]


def test_multi_lora_support(
    hf_model_dir,
    tllm_ckpt_dir,
    engine_dir,
    llm_venv,
    example_root,
    num_loras=2,
    lora_rank=8,
    target_hf_modules=["q_proj", "k_proj", "v_proj"],
    target_trtllm_modules=["attn_q", "attn_k", "attn_v"],
    zero_lora_weights=True,
    use_code_prompts=False,
):
    start_time = time.time()
    print("Creating dummy LoRAs...")
    lora_start = time.time()
    lora_paths = generate_dummy_loras(
        hf_model_dir=hf_model_dir,
        lora_output_dir=llm_venv.get_working_directory(),
        num_loras=num_loras,
        lora_rank=lora_rank,
        target_modules=target_hf_modules,
        zero_weights=zero_lora_weights)
    lora_end = time.time()
    print(
        f"Creating dummy LoRAs completed in {(lora_end - lora_start):.2f} seconds."
    )

    print("Build engines...")
    build_start = time.time()
    build_cmd = [
        "trtllm-build",
        f"--checkpoint_dir={tllm_ckpt_dir}",
        f"--output_dir={engine_dir}",
        "--remove_input_padding=enable",
        "--context_fmha=enable",
        "--gemm_plugin=auto",
        "--lora_plugin=auto",
        "--max_batch_size=8",
        "--max_input_len=512",
        "--max_seq_len=562",
        "--lora_dir",
        f"{lora_paths[0]}",
        f"{lora_paths[1]}",
        "--max_lora_rank=8",
        "--lora_target_modules",
        *target_trtllm_modules,
        "--max_beam_width=1",
    ]
    check_call(" ".join(build_cmd), shell=True, env=llm_venv._new_env)
    build_end = time.time()
    print(
        f"Build engines completed in {(build_end - build_start):.2f} seconds.")

    input_prompts = get_test_prompts(use_code_prompts)

    print("Run inference with C++ runtime with pybind...")
    inference_start = time.time()
    run_script = f"{example_root}/../../../run.py" if "core" in example_root else f"{example_root}/../run.py"
    run_cmd = [
        run_script,
        f"--tokenizer_dir={hf_model_dir}",
        f"--engine_dir={engine_dir}",
        "--input_text",
        *input_prompts,
        "--lora_task_uids",
        "-1",
        "0",
        "1",
        "-1",
        "0",
        "1",
        "--top_p=0.5",
        "--top_k=0",
        "--random_seed=0",
        "--max_output_len=30",
    ]
    venv_check_call(llm_venv, run_cmd)
    inference_end = time.time()
    print(
        f"Inference completed in {(inference_end - inference_start):.2f} seconds."
    )

    total_time = time.time() - start_time
    print(
        f"Total test_multi_lora_support execution time: {total_time:.2f} seconds"
    )


def test_llm_torch_multi_lora_support(
        hf_model_dir,
        llm_venv,
        num_loras=2,
        lora_rank=8,
        target_hf_modules=["q_proj", "k_proj", "v_proj"],
        target_trtllm_modules=["attn_q", "attn_k", "attn_v"],
        zero_lora_weights=True,
        tensor_parallel_size=1,
        pipeline_parallel_size=1):
    """Test multi-LoRA support with LLM-API Torch backend.

    When zero_lora_weights=True, validates that LoRA outputs match base model
    outputs (since zero-weight LoRAs should not alter behavior).
    """

    assert zero_lora_weights, (
        "This test compares LoRA outputs against base model outputs, "
        "which is only valid when zero_lora_weights=True.")

    start_time = time.time()
    print("Creating dummy LoRAs...")
    lora_start = time.time()

    lora_paths = generate_dummy_loras(
        hf_model_dir=hf_model_dir,
        lora_output_dir=llm_venv.get_working_directory(),
        num_loras=num_loras,
        lora_rank=lora_rank,
        target_modules=target_hf_modules,
        zero_weights=zero_lora_weights)
    lora_end = time.time()
    print(
        f"Creating dummy LoRAs completed in {(lora_end - lora_start):.2f} seconds."
    )

    lora_config = LoraConfig(lora_dir=lora_paths,
                             max_lora_rank=lora_rank,
                             max_loras=num_loras,
                             max_cpu_loras=num_loras,
                             lora_target_modules=target_trtllm_modules)

    input_prompts = get_test_prompts_for_torch()

    sampling_params = SamplingParams(max_tokens=30,
                                     top_p=0.5,
                                     top_k=0,
                                     temperature=0.0)

    # Step 1: Get base model outputs (no LoRA) as the ground truth.
    print("Initializing LLM_torch without LoRA for base model outputs...")
    init_start = time.time()

    with LLM_torch(model=hf_model_dir,
                   tensor_parallel_size=tensor_parallel_size,
                   pipeline_parallel_size=pipeline_parallel_size,
                   dtype="bfloat16",
                   max_batch_size=8,
                   max_input_len=512,
                   max_seq_len=562,
                   max_beam_width=1) as base_llm:

        init_end = time.time()
        print(
            f"Base LLM_torch initialization completed in {(init_end - init_start):.2f} seconds."
        )

        print("Running base model inference (no LoRA)...")
        base_inference_start = time.time()

        base_outputs = base_llm.generate(input_prompts,
                                         sampling_params=sampling_params)

        base_inference_end = time.time()
        print(
            f"Base inference completed in {(base_inference_end - base_inference_start):.2f} seconds."
        )

    expected_outputs = [o.outputs[0].text for o in base_outputs]
    for i, text in enumerate(expected_outputs):
        print(f"Base output {i+1}: {text!r}")

    # Step 2: Run with LoRA adapters and compare against base outputs.
    print("Initializing LLM_torch with LoRA support...")
    init_start = time.time()

    with LLM_torch(model=hf_model_dir,
                   lora_config=lora_config,
                   tensor_parallel_size=tensor_parallel_size,
                   pipeline_parallel_size=pipeline_parallel_size,
                   dtype="bfloat16",
                   max_batch_size=8,
                   max_input_len=512,
                   max_seq_len=562,
                   max_beam_width=1) as llm:

        init_end = time.time()
        print(
            f"LLM_torch initialization completed in {(init_end - init_start):.2f} seconds."
        )

        print("Running inference with LLM-API Torch backend...")
        inference_start = time.time()

        # Create LoRA requests cycling through available adapters.
        lora_requests = []
        lora_counter = 0
        for i in range(len(input_prompts)):
            if i % 2 == 1:
                lora_requests.append(None)
            else:
                lora_idx = lora_counter % num_loras
                lora_counter += 1
                lora_requests.append(
                    LoRARequest(f"lora-{lora_idx}", lora_idx,
                                lora_paths[lora_idx]))

        outputs = llm.generate(input_prompts,
                               sampling_params=sampling_params,
                               lora_request=lora_requests)

        inference_end = time.time()
        print(
            f"Inference completed in {(inference_end - inference_start):.2f} seconds."
        )

        # Validate that LoRA outputs match base model outputs.
        print("Validating outputs against base model...")
        assert len(outputs) == len(expected_outputs), \
            f"Expected {len(expected_outputs)} outputs, got {len(outputs)}"

        for i, (output, expected) in enumerate(zip(outputs, expected_outputs)):
            actual_text = output.outputs[0].text
            print(f"Prompt {i+1}: {input_prompts[i]}")
            print(
                f"LoRA: {lora_requests[i].lora_int_id if lora_requests[i] else 'None'}"
            )
            print(f"Expected (base): {expected!r}")
            print(f"Actual (LoRA):   {actual_text!r}")
            print("-" * 50)

            assert actual_text == expected, \
                f"Output {i+1} mismatch:\nExpected (base): {expected!r}\nActual (LoRA):   {actual_text!r}"

    total_time = time.time() - start_time
    print(f"Total test execution time: {total_time:.2f} seconds")


def get_dummy_spec_decoding_heads(hf_model_dir,
                                  save_dir,
                                  mode='medusa',
                                  num_heads=4,
                                  num_layers=1):

    import os

    import modelopt.torch.opt as mto
    import modelopt.torch.speculative as mtsp
    import transformers
    from modelopt.torch.export import export_hf_checkpoint

    # Create the base model.
    model = transformers.AutoModelForCausalLM.from_pretrained(
        hf_model_dir, trust_remote_code=True)

    if mode == "medusa":
        config = {
            "medusa_num_heads": num_heads,
            "medusa_num_layers": num_layers,
        }
    elif mode == "eagle":
        config = {
            "eagle_num_layers": num_layers,
            "use_input_layernorm_in_first_layer": True,
            "use_last_layernorm": False,
        }
    else:
        raise NotImplementedError(f"Unknown mode {mode}.")
    mtsp.convert(model, [(mode, config)])

    tokenizer = transformers.AutoTokenizer.from_pretrained(hf_model_dir)
    tokenizer.pad_token_id = tokenizer.eos_token_id

    # Create a dummy trainer.
    trainer = transformers.Trainer(model=model, tokenizer=tokenizer)
    trainer._move_model_to_device(model, 'cuda')

    # Enable HF checkpointing so that the saved model will contain the speculative decoding module.
    mto.enable_huggingface_checkpointing()
    trainer.save_model(os.path.join(save_dir, 'native'))
    tokenizer.save_pretrained(os.path.join(save_dir, 'native'))

    import modelopt.torch.quantization as mtq
    import modelopt.torch.utils.dataset_utils as dataset_utils

    mto.enable_huggingface_checkpointing()

    model = transformers.AutoModelForCausalLM.from_pretrained(
        os.path.join(save_dir, 'native'))
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        os.path.join(save_dir, 'native'))

    calib_dataloader = dataset_utils.get_dataset_dataloader(
        dataset_name="cnn_dailymail",
        tokenizer=tokenizer,
        batch_size=1,
        num_samples=1,
        device=model.device,
        include_labels=False,
    )

    quant_cfg = getattr(mtq, "FP8_DEFAULT_CFG")
    # Following quantizers are needed for KV cache quantization.
    quant_cfg["quant_cfg"]["*output_quantizer"] = {
        "num_bits": (4, 3),
        "axis": None,
        "enable": True,
    }
    quant_cfg["quant_cfg"]["*k_bmm_quantizer"] = {
        "num_bits": (4, 3),
        "axis": None,
        "enable": True,
    }
    quant_cfg["quant_cfg"]["*v_bmm_quantizer"] = {
        "num_bits": (4, 3),
        "axis": None,
        "enable": True,
    }

    calibrate_loop = dataset_utils.create_forward_loop(
        calib_dataloader, dataloader=calib_dataloader)
    model = mtq.quantize(model, quant_cfg, forward_loop=calibrate_loop)
    mtq.print_quant_summary(model)

    export_hf_checkpoint(model,
                         dtype=model.config.torch_dtype,
                         export_dir=os.path.join(save_dir, 'fp8'))


def get_mmlu_accuracy(output):
    mmlu_line = None
    for line in output.split('\n'):
        if "MMLU weighted average accuracy:" in line:
            mmlu_line = line
            break

    if mmlu_line is None:
        raise Exception(
            f"Could not find 'MMLU weighted average accuracy:' in output. Full output:\n{output}"
        )

    mmlu_accuracy = float(
        mmlu_line.split("MMLU weighted average accuracy: ")[1].split(" (")[0])

    print(f"MMLU weighted average accuracy is: {mmlu_accuracy}")

    return mmlu_accuracy


def wait_for_server(host, port, timeout_seconds=180):
    start_time = time.time()
    while time.time() - start_time < timeout_seconds:
        try:
            with socket.create_connection((host, port), timeout=5):
                return True
        except (socket.error, ConnectionRefusedError, OSError):
            time.sleep(2)
    return False


PORTS_IN_USE = set()


def get_free_port_in_ci(max_attempts=100):
    """
    Get a free port in the range [CONTAINER_PORT_START, CONTAINER_PORT_START + CONTAINER_PORT_NUM - 1]
    If CONTAINER_PORT_START and CONTAINER_PORT_NUM are not set or all ports are already in use, fallback to get_free_port
    """
    global PORTS_IN_USE

    pid = os.getpid()
    container_port_start = int(os.environ.get("CONTAINER_PORT_START", -1))
    container_port_num = int(os.environ.get("CONTAINER_PORT_NUM", -1))
    if container_port_start != -1 and container_port_num != -1:
        port_range = (container_port_start,
                      container_port_start + container_port_num - 1)
        available_ports = [
            port for port in range(container_port_start, container_port_start +
                                   container_port_num)
            if port not in PORTS_IN_USE
        ]
        num_candidates = len(available_ports)

        for attempt in range(1, num_candidates + 1):
            # Get a random port from the available ports
            port = random.choice(available_ports)

            # Check if the port is free
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                try:
                    s.bind(("localhost", port))
                    PORTS_IN_USE.add(port)
                    print_info(
                        f"[get_free_port_in_ci] pid={pid} allocated port={port} "
                        f"from CI range {port_range} after {attempt} attempt(s); "
                        f"{len(PORTS_IN_USE)} reserved in-process. The probe "
                        f"socket is now closed, so another process may take the "
                        f"port before the caller rebinds it (TOCTOU).")
                    return port
                except OSError as e:
                    print_info(
                        f"[get_free_port_in_ci] pid={pid} candidate port={port} "
                        f"in CI range {port_range} is busy ({e}); trying another."
                    )
                    available_ports.remove(port)
                    continue

        print_warning(
            f"[get_free_port_in_ci] pid={pid} exhausted all {num_candidates} "
            f"candidate ports in CI range {port_range}; falling back to a "
            f"system-assigned ephemeral port.")

    # No port found in the range, try to get a random free port from the system
    for _ in range(max_attempts):
        port = get_free_port()
        if port not in PORTS_IN_USE:
            PORTS_IN_USE.add(port)
            print_info(
                f"[get_free_port_in_ci] pid={pid} allocated system ephemeral "
                f"port={port}; {len(PORTS_IN_USE)} reserved in-process. Another "
                f"process may take it before the caller rebinds it (TOCTOU).")
            return port

    raise Exception(
        f"Failed to find a free port both in container port range and system after {max_attempts} attempts"
    )


def revise_disaggregated_server_config_urls_with_free_ports(
        disaggregated_server_config: dict[str, Any]) -> dict[str, Any]:
    # Revise serve port
    disaggregated_server_config['port'] = get_free_port_in_ci()

    # Revise context and generation server urls
    ctx_urls = disaggregated_server_config["context_servers"]["urls"]
    gen_urls = disaggregated_server_config["generation_servers"]["urls"]
    url_map = dict()
    for url in set(ctx_urls + gen_urls):
        url_map[url] = (url.split(':')[0], get_free_port_in_ci())

    for i, url in enumerate(ctx_urls):
        disaggregated_server_config["context_servers"]["urls"][
            i] = f"{url_map[url][0]}:{url_map[url][1]}"

    for i, url in enumerate(gen_urls):
        disaggregated_server_config["generation_servers"]["urls"][
            i] = f"{url_map[url][0]}:{url_map[url][1]}"

    return disaggregated_server_config


def revise_disagg_config_file_with_free_ports(disagg_config_file: str) -> str:
    # Revise the config file to use free ports
    new_config = None
    with open(disagg_config_file, 'r') as f:
        config = yaml.safe_load(f)
        new_config = revise_disaggregated_server_config_urls_with_free_ports(
            config)

    temp_fd, new_config_file = tempfile.mkstemp(suffix='.yaml')
    with os.fdopen(temp_fd, 'w') as f:
        yaml.dump(new_config, f)

    return new_config_file


def parse_gsm8k_output(output_text: str) -> float:
    """
    Parse accuracy value from lm_eval output for GSM8K flexible-extract exact_match

    Args:
        output_text: The output text from gsm8k command

    Returns:
        float: The accuracy value (0.7582 in the example)
    """

    # Look for the specific pattern:
    # |gsm8k|...|flexible-extract|     5|exact_match|↑  |0.7559|±  |0.0118|
    # lm-eval pads table cells, so allow whitespace around the value.
    patterns = [
        r'flexible-extract\s*\|\s*\d+\s*\|\s*exact_match\s*\|\s*↑\s*\|\s*(\d+(?:\.\d+)?)',
    ]

    for pattern in patterns:
        match = re.search(pattern, output_text)
        if match:
            accuracy_value = float(match.group(1))
            print_info(f"Extracted GSM8K accuracy value: {accuracy_value}")
            return accuracy_value

    print_warning("Could not find GSM8K accuracy value in gsm8k output")
    print_warning(f"Output text: {output_text}")

    return 0.0
