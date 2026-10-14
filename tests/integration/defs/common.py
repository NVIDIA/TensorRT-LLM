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
from typing import Any, Optional

import yaml
from packaging import version

from tensorrt_llm._utils import get_free_port

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


def wait_for_reported_addr(addr_path: str,
                           timeout: float,
                           process=None) -> tuple[str, int]:
    """Read the address a server reported to its --report_addr file.

    The file only appears once trtllm-serve has bound its socket, and that
    socket stays bound from then on, so the address cannot be stolen between
    this read and its use -- unlike a port reserved before the server starts.

    Args:
        addr_path: Path passed to the server's --report_addr.
        timeout: Seconds to wait for the file to appear.
        process: Optional Popen of the server, polled so that a crash fails
            fast instead of burning the whole timeout.

    Returns:
        tuple[str, int]: The host and port the server bound.
    """
    deadline = time.time() + timeout
    while time.time() < deadline:
        if process is not None and process.poll() is not None:
            raise RuntimeError(
                f"server exited with code {process.returncode} before "
                f"reporting its address to {addr_path}")
        try:
            with open(addr_path) as f:
                reported = f.read().strip()
        except FileNotFoundError:
            reported = ""
        if reported:
            host, _, port = reported.rpartition(":")
            return host, int(port)
        time.sleep(0.5)
    raise TimeoutError(f"server did not report its address to {addr_path} "
                       f"within {timeout}s")


PORTS_IN_USE = set()

# Size of the window carved out just below the kernel's ephemeral range, used
# when CONTAINER_PORT_START is unset (e.g. the SLURM multi-node path).
STATIC_PORT_RANGE_SIZE = 4096


def get_ephemeral_port_range() -> Optional[tuple[int, int]]:
    """Return the kernel's ephemeral port range as (low, high), or None.

    These are the ports bind(('', 0)) hands out. None means the range could
    not be read.
    """
    try:
        with open("/proc/sys/net/ipv4/ip_local_port_range") as f:
            low, high = (int(value) for value in f.read().split())
    except (OSError, ValueError) as e:
        print_info(f"[get_free_port_in_ci] could not read the ephemeral port "
                   f"range ({e}); assuming none is reserved.")
        return None
    # Nonsense bounds would make get_static_port_range() hand out ports outside
    # 1-65535, and bind() raises OverflowError (not OSError) for those, so
    # reserve_port_from_range would propagate it instead of trying another port.
    if not 1 <= low <= high <= 65535:
        print_warning(f"[get_free_port_in_ci] ignoring implausible ephemeral "
                      f"port range ({low}, {high}).")
        return None
    return low, high


def get_static_port_range() -> Optional[tuple[int, int]]:
    """Return a (low, high) window just below the kernel's ephemeral range.

    None is returned if the ephemeral range cannot be determined.

    Ports here are never handed out by bind(('', 0)), so a port reserved from
    this window cannot be stolen by a sibling process launched with --port 0 --
    which is how a reserved disaggregated server port was lost to the test's
    own worker.
    """
    ephemeral_range = get_ephemeral_port_range()
    if ephemeral_range is None:
        return None
    high = ephemeral_range[0] - 1
    low = max(1024, high - STATIC_PORT_RANGE_SIZE + 1)
    if low > high:
        return None
    return low, high


def reserve_port_from_range(port_range: tuple[int, int],
                            source: str) -> Optional[int]:
    """Probe-bind random ports from an inclusive (low, high) window.

    The first port found free is recorded in PORTS_IN_USE and returned;
    None is returned once every candidate in the window is taken.
    """
    global PORTS_IN_USE

    pid = os.getpid()
    low, high = port_range
    available_ports = [
        port for port in range(low, high + 1) if port not in PORTS_IN_USE
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
                    f"from {source} range {port_range} after {attempt} "
                    f"attempt(s); {len(PORTS_IN_USE)} reserved in-process. The "
                    f"probe socket is now closed, so another process may take "
                    f"the port before the caller rebinds it (TOCTOU).")
                return port
            except OSError as e:
                print_info(
                    f"[get_free_port_in_ci] pid={pid} candidate port={port} "
                    f"in {source} range {port_range} is busy ({e}); trying "
                    f"another.")
                available_ports.remove(port)
                continue

    print_warning(
        f"[get_free_port_in_ci] pid={pid} exhausted all {num_candidates} "
        f"candidate ports in {source} range {port_range}.")
    return None


def get_free_port_in_ci(max_attempts: int = 100) -> int:
    """Get a free port from the CI-assigned container port range.

    The range is [CONTAINER_PORT_START, CONTAINER_PORT_START + CONTAINER_PORT_NUM - 1].
    If those are unset, or every port in the range is already in use, fall back to
    a port just below the kernel's ephemeral range, and only then to get_free_port.
    """
    global PORTS_IN_USE

    pid = os.getpid()
    container_port_start = int(os.environ.get("CONTAINER_PORT_START", -1))
    container_port_num = int(os.environ.get("CONTAINER_PORT_NUM", -1))
    if container_port_start != -1 and container_port_num != -1:
        port = reserve_port_from_range(
            (container_port_start,
             container_port_start + container_port_num - 1), "CI")
        if port is not None:
            return port

    # No CI range configured, or every port in it is taken. Prefer a port below
    # the ephemeral range over a system-assigned one: the latter is drawn from
    # the same pool that trtllm-serve's own --port 0 workers bind from, so a
    # sibling worker can take it before the caller rebinds it.
    static_port_range = get_static_port_range()
    if static_port_range is not None:
        port = reserve_port_from_range(static_port_range, "static")
        if port is not None:
            return port

    # Last resort: a system-assigned ephemeral port.
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
