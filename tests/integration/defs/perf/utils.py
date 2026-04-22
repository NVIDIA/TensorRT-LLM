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
import abc
import contextlib
import copy
import io
import os
import re
import signal
import subprocess
import threading
import time
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional

from _pytest.nodes import Item
from _pytest.python import Function
from defs.trt_test_alternative import print_error, print_info

from ..common import get_trt_llm_lib_dir
from ..local_venv import PythonVenvRunnerImpl
from ..test_list_parser import parse_test_list
from .data_export import (GPU_MONITORING_FORMAT_KEYS, write_csv,
                          write_gpu_monitoring_no_test_results, write_yaml)
from .gpu_clock_lock import GPUClockLock, InvalidGPUMonitoringResultError
from .session_data_writer import SessionDataWriter


def clean_myelin_time(log: str):
    # When __LUNOWUD = "-time_pass=on", myelin would generate time logs for each myelin layers.
    # These logs start/end with "="*36, the line targets to remove them.
    time_log_pattern = r"\={36}\n[\s\S]*?" * 4 + r"\={36}\n"
    log = re.sub(time_log_pattern, "", log)
    return log


def collect_and_clean_myelin_time(log: str):
    ws = lambda n: " " * n
    total_time_list = re.findall(f"{ws(8)}Total Time(?:\\(ms\\))?: \\d+", log)

    if len(total_time_list) == 0:
        return log

    region_time_list = re.findall(f"{ws(4)}[^\\. ]*: \\d+", log)
    sub_region_time_list = re.findall(f"{ws(4)}.*\\..*: \\d+", log)
    log = clean_myelin_time(log)
    total_time_ms = 0
    for total_ in total_time_list:
        total_time_ms += float(total_.strip().split(':')[1])
    log += f"Total Myelin Build Time : {total_time_ms/1000}s\n"

    def log_sorted_time(time_list):
        time_map = {}
        log = ""

        for time in time_list:
            key, value = time.strip().rsplit(':', 1)
            time_map[key] = time_map.get(key, 0) + float(value)

        time_map = sorted(time_map.items(), key=lambda kv: kv[1], reverse=True)

        for region, time_ms in time_map[0:10]:
            log += (f"{ws(4)}{region}: {time_ms/1000}s\n")

        return log

    log += log_sorted_time(region_time_list)
    log += "Sub Regions: \n"
    log += log_sorted_time(sub_region_time_list)
    return log


class PerfMetricType(str, Enum):
    """
    An string-enum type to define what kind of perf metric it is. It is used by QA to
    set up special threshold criteria for each type of metrics (like >50MB for engine size increase, etc.).
    """
    INFERENCE_TIME = "INFERENCE_TIME"
    MEDIAN_INFERENCE_TIME = "MEDIAN_INFERENCE_TIME"
    P99_INFERENCE_TIME = "P99_INFERENCE_TIME"
    INTER_TOKEN_TIME = "INTER_TOKEN_TIME"
    MEDIAN_INTER_TOKEN_TIME = "MEDIAN_INTER_TOKEN_TIME"
    P99_INTER_TOKEN_TIME = "P99_INTER_TOKEN_TIME"
    FIRST_TOKEN_TIME = "FIRST_TOKEN_TIME"
    MEDIAN_FIRST_TOKEN_TIME = "MEDIAN_FIRST_TOKEN_TIME"
    P99_FIRST_TOKEN_TIME = "P99_FIRST_TOKEN_TIME"
    OUTPUT_TOKEN_TIME = "OUTPUT_TOKEN_TIME"
    MEDIAN_OUTPUT_TOKEN_TIME = "MEDIAN_OUTPUT_TOKEN_TIME"
    P99_OUTPUT_TOKEN_TIME = "P99_OUTPUT_TOKEN_TIME"
    TOKEN_THROUGHPUT = "TOKEN_THROUGHPUT"
    TOTAL_TOKEN_THROUGHPUT = "TOTAL_TOKEN_THROUGHPUT"
    USER_THROUGHPUT = "USER_THROUGHPUT"
    BUILD_TIME = "BUILD_TIME"
    BUILD_PEAK_CPU_MEMORY = "BUILD_PEAK_CPU_MEMORY"
    BUILD_PEAK_GPU_MEMORY = "BUILD_PEAK_GPU_MEMORY"
    INFERENCE_PEAK_GPU_MEMORY = "INFERENCE_PEAK_GPU_MEMORY"
    ENGINE_SIZE = "ENGINE_SIZE"
    CONTEXT_GPU_MEMORY = "CONTEXT_GPU_MEMORY"
    SEQ_THROUGHPUT = "SEQ_THROUGHPUT"
    SEQ_LATENCY = "SEQ_LATENCY"
    KV_CACHE_SIZE = "KV_CACHE_SIZE"
    PER_USER_OUTPUT_THROUGHPUT = "PER_USER_OUTPUT_THROUGHPUT"
    PER_GPU_OUTPUT_THROUGHPUT = "PER_GPU_OUTPUT_THROUGHPUT"


@contextlib.contextmanager
def temp_wd(path):
    """A context manager to temporarily change the working directory."""
    prev_cwd = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(prev_cwd)


def add_host_port_to_cmd(cmd: List[str], host: str, port: int) -> List[str]:
    return cmd + ["--host", host, "--port", str(port)]


#if hang time > 30 mins, it will be killed
_STALL_TIMEOUT = 1800
#if hang with error time > 3 mins, it will be killed
_ERROR_STALL_TIMEOUT = 180

_FATAL_PATTERNS = [
    'Segmentation fault',
    'Fatal Python error:',
    'terminate called',
    'RuntimeError: [TensorRT-LLM][ERROR]',
]


def _run_command_with_captured_output(cmd: list[str],
                                      env: dict[str, str] | None = None) -> str:
    """Run a command, reading stdout line-by-line in a background thread.

    Compared to subprocess.check_output() this has two advantages:
    1. Output is accumulated incrementally, so even if the process is killed
       (by our stall detector or by pytest-timeout SIGALRM) the partial
       output is available and forwarded to Allure.
    2. A stall detector monitors the output stream.  If a fatal error pattern
       is seen and the process then produces no output for _ERROR_STALL_TIMEOUT
       seconds it is killed immediately instead of waiting for the full
       pytest-timeout (typically 3600 s).  A general _STALL_TIMEOUT applies
       when no error pattern has been seen.
    """
    if env is not None:
        env = env.copy()
        env.setdefault("PYTHONUNBUFFERED", "1")

    proc = subprocess.Popen(cmd,
                            env=env,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT,
                            start_new_session=True)

    output_lines: list = []
    lock = threading.Lock()
    last_output_time = [time.monotonic()]
    has_error = [False]

    def _reader():
        try:
            while True:
                raw = proc.stdout.readline()
                if not raw:
                    break
                line = raw.decode('utf-8', errors='replace')
                with lock:
                    output_lines.append(line)
                    last_output_time[0] = time.monotonic()
                    if not has_error[0]:
                        for pat in _FATAL_PATTERNS:
                            if pat in line:
                                has_error[0] = True
                                break
        except (ValueError, OSError):
            pass

    thread = threading.Thread(target=_reader, daemon=True)
    thread.start()

    def _cleanup_after_abort():
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except OSError:
            pass
        proc.wait()
        thread.join(timeout=10)

    try:
        while proc.poll() is None:
            time.sleep(10)
            now = time.monotonic()
            with lock:
                idle = now - last_output_time[0]
                errored = has_error[0]

            limit = _ERROR_STALL_TIMEOUT if errored else _STALL_TIMEOUT
            if idle > limit:
                tag = "errored and stalled" if errored else "stalled"
                print_info(f"Process {tag} with no output for {idle:.0f}s "
                           f"(limit={limit}s), killing")
                os.killpg(proc.pid, signal.SIGKILL)
                break

        thread.join(timeout=30)
        proc.wait()

        with lock:
            output = ''.join(output_lines)

        if proc.returncode != 0:
            err = subprocess.CalledProcessError(proc.returncode, cmd)
            err.stdout = output.encode()
            err.stderr = None
            raise err

        return output

    except subprocess.CalledProcessError:
        raise

    except Exception as exc:
        _cleanup_after_abort()
        with lock:
            partial = ''.join(output_lines)
        if partial:
            rc = proc.returncode if proc.returncode is not None else -9
            err = subprocess.CalledProcessError(rc, cmd)
            err.stdout = partial.encode()
            err.stderr = None
            raise err from exc
        raise

    except BaseException:
        _cleanup_after_abort()
        raise


class PerfBenchScriptTestCmds(NamedTuple):
    data_cmds: List[List[str]]
    build_cmd: List[str]
    benchmark_cmds: List[List[str]]
    mpi_cmd: List[str]

    def run_cmd(self, cmd_idx: int, venv) -> str:
        output = ""
        mpi_cmd = self.mpi_cmd
        build_cmd_str = self.get_cmd_str(len(self.data_cmds))
        current_cmd_str = self.get_cmd_str(cmd_idx)
        if cmd_idx <= len(self.data_cmds) - 1:
            print_info(f'Running prepare dataset command')
            prepare_cmd = self.data_cmds[cmd_idx]
            prepare_cmd_str = " ".join(prepare_cmd)
            envs = copy.deepcopy(os.environ)
            prepare_cmds = prepare_cmd_str.split(';')
            for prepare_cmd in prepare_cmds:
                print_info(f'Now running prepare data command: "{prepare_cmd}"')
                if '>' in prepare_cmd:
                    cmd = prepare_cmd.split('>')[0]
                    dataset_file = prepare_cmd.split('>')[1].split()[0]
                else:
                    cmd = prepare_cmd
                    dataset_file = None
                output += subprocess.check_output(cmd.split(),
                                                  env=envs).decode()
                if dataset_file:
                    with open(f"{dataset_file}", 'w+') as f:
                        f.write(output)

        elif cmd_idx == len(self.data_cmds):
            #running build
            if len(self.build_cmd) == 0:
                pass
            else:
                envs = copy.deepcopy(os.environ)
                print_info(
                    f'Running engine building command: "{build_cmd_str}"')
                command = self.build_cmd
                output += _run_command_with_captured_output(command, env=envs)
        else:
            #running throughput
            print_info(f'Now running benchmarking command: "{current_cmd_str}"')
            command = self.benchmark_cmds[cmd_idx - 1 - len(self.data_cmds)]
            envs = copy.deepcopy(os.environ)
            envs[
                "LD_LIBRARY_PATH"] = f'{get_trt_llm_lib_dir(venv)}:{os.path.dirname(command[0])}:{envs.get("LD_LIBRARY_PATH", "")}'
            print(f'Augmented LD_LIBRARY_PATH={envs["LD_LIBRARY_PATH"]}')
            benchmark_cmd = mpi_cmd + command
            output += _run_command_with_captured_output(benchmark_cmd, env=envs)
            match = re.search(r'--engine_dir=([^\s]+)', current_cmd_str)
            if match:
                engine_dir = match.group(1)
                print_info(f'writing config.json in {engine_dir} to output log')
                with open(os.path.join(engine_dir, "config.json"), "r") as f:
                    config_content = f.read()
                    output += "\n" + "=" * 50 + "\n"
                    output += "ENGINE CONFIG:\n"
                    output += "=" * 50 + "\n"
                    output += config_content
                    output += "\n" + "=" * 50 + "\n"
        return output

    def get_cmd_str(self, cmd_idx) -> List[str]:
        mpi_cmd_str = (" ".join(self.mpi_cmd) +
                       " ") if len(self.mpi_cmd) > 0 else ""
        if cmd_idx <= len(self.data_cmds) - 1:
            cmd_str = " ".join(self.data_cmds[cmd_idx])
        elif cmd_idx == len(self.data_cmds):
            if len(self.build_cmd) == 0:
                cmd_str = ''
            else:
                cmd_str = mpi_cmd_str + " ".join(self.build_cmd)
        else:
            cmd_str = mpi_cmd_str + " ".join(
                self.benchmark_cmds[cmd_idx - 1 - len(self.data_cmds)])

        return cmd_str


def _wait_for_server_ready(url: str,
                           timeout: int = 600,
                           server_proc: subprocess.Popen = None) -> None:
    import requests

    start = time.monotonic()
    while time.monotonic() - start < timeout:
        if server_proc is not None:
            exit_code = server_proc.poll()
            if exit_code is not None:
                raise RuntimeError(
                    f"Server exited with code {exit_code} before becoming ready."
                )
        try:
            time.sleep(1)
            if requests.get(url, timeout=5).status_code == 200:
                print_info(f"Server endpoint {url} is ready")
                return
        except requests.exceptions.RequestException:
            pass
    raise RuntimeError(f"Server {url} did not become ready within {timeout}s")


class PerfServeScriptTestCmds:
    """Commands for serve runtime perf tests (server-client model)."""

    def __init__(self, server_cmd: List[str], client_cmds: List[List[str]],
                 data_cmds: List[List[str]], server_env: Dict[str, str],
                 server_timeout: int):
        self.server_cmd = server_cmd
        self.client_cmds = client_cmds
        self.data_cmds = data_cmds
        self.server_env = server_env
        self.server_timeout = server_timeout
        self._server_proc = None
        self._server_log_path = None

    def start_server(self) -> None:
        if self._server_proc is not None:
            return
        from tensorrt_llm._utils import get_free_port
        self._port = get_free_port()
        self._host = "localhost"
        cmd = self.server_cmd + [
            "--host", self._host, "--port",
            str(self._port)
        ]
        self._server_log_path = os.path.join(os.getcwd(),
                                             "trtllm-serve-perf.log")
        print_info(f"Starting trtllm-serve: {' '.join(cmd)}")
        self._server_log_file = open(self._server_log_path, "w")
        self._server_proc = subprocess.Popen(cmd,
                                             env=self.server_env,
                                             stdout=self._server_log_file,
                                             stderr=subprocess.STDOUT)
        _wait_for_server_ready(f"http://{self._host}:{self._port}/health",
                               timeout=self.server_timeout,
                               server_proc=self._server_proc)

    def stop_server(self) -> None:
        if self._server_proc is not None:
            self._server_proc.terminate()
            try:
                self._server_proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self._server_proc.kill()
                self._server_proc.wait()
            self._server_proc = None
        if hasattr(self, '_server_log_file') and self._server_log_file:
            self._server_log_file.close()
            self._server_log_file = None

    def run_cmd(self, cmd_idx: int, venv) -> str:
        output = ""
        if cmd_idx <= len(self.data_cmds) - 1:
            print_info(f"Running prepare dataset command")
            prepare_cmd = self.data_cmds[cmd_idx]
            prepare_cmd_str = " ".join(prepare_cmd)
            envs = copy.deepcopy(os.environ)
            for sub_cmd in prepare_cmd_str.split(';'):
                print_info(f'Now running prepare data command: "{sub_cmd}"')
                if '>' in sub_cmd:
                    cmd = sub_cmd.split('>')[0]
                    dataset_file = sub_cmd.split('>')[1].split()[0]
                else:
                    cmd = sub_cmd
                    dataset_file = None
                output += subprocess.check_output(cmd.split(),
                                                  env=envs).decode()
                if dataset_file:
                    with open(f"{dataset_file}", 'w+') as f:
                        f.write(output)
        elif cmd_idx == len(self.data_cmds):
            self.start_server()
        else:
            client_cmd = self.client_cmds[cmd_idx - 1 - len(self.data_cmds)]
            client_cmd_with_port = client_cmd + [
                "--host", self._host, "--port",
                str(self._port)
            ]
            print_info(
                f"Running benchmark client: {' '.join(client_cmd_with_port)}")
            output = _run_command_with_captured_output(client_cmd_with_port,
                                                       env=copy.deepcopy(
                                                           os.environ))
        return output

    def get_cmd_str(self, cmd_idx) -> str:
        if cmd_idx <= len(self.data_cmds) - 1:
            return " ".join(self.data_cmds[cmd_idx])
        elif cmd_idx == len(self.data_cmds):
            return " ".join(self.server_cmd)
        else:
            return " ".join(self.client_cmds[cmd_idx - 1 - len(self.data_cmds)])


class AbstractPerfScriptTestClass(abc.ABC):
    """
    Abstract class for all script-based perf tests.
    """

    @abc.abstractmethod
    def get_test_name(self) -> str:
        """
        Define the test name for this test, which will appear in the "[...]" part of the generate test names.
        WARNING: Please keep backward compatibility in get_test_name() method when adding new tests!
        Changing test names means we will lose test history in NvRegress and in PerfDB!
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def set_runtime_configs(self, *args) -> None:
        """
        Set the runtime configs (like directory paths, compute capability, etc.) for the test.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def get_commands(self) -> PerfBenchScriptTestCmds:
        """
        Get the commands to run the test. Should return an PerfScriptTestCmds instance.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def get_perf_result(self, outputs: List[str]) -> float:
        """
        Get the perf result (latency) from the output logs of each command.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def get_threshold(self) -> float:
        """
        Get the relative threshold used to flag a perf regression compared to perf baseline.
        """
        raise NotImplementedError()

    def get_absolute_threshold(self) -> float:
        """
        Get the absolute threshold used to flag a perf regression compared to perf baseline.
        Perf comparison will only fail if it exceeds both relative and absolute thresholds.
        """
        return 0.0

    def get_metric_type(self) -> PerfMetricType:
        """
        Get the type of perf metric. QA uses this field to set up special
        threshold criteria depending on the metric type.
        """
        return PerfMetricType.INFERENCE_TIME

    def get_working_dir(self) -> str:
        """
        Get the working directory to run the commands in. Default is the current working directory.
        Derived classes can override this function if a different working directory is needed.
        """
        return os.getcwd()

    def get_result_state(self) -> str:
        """
        Get the result_state of current test-run
        """
        return self._result_state

    def get_error(self) -> str:
        """
        Get the error of current test-run
        """
        return self._error

    def _check_benchmark_output_for_errors(self, output: str) -> None:
        """
        Check whether the benchmark output contains error messages (e.g., failed requests).
        """
        if not output:
            return

        # Check for non-zero failed requests
        failed_requests_match = re.search(r'Failed requests:\s+(\d+)', output)
        if failed_requests_match:
            failed_count = int(failed_requests_match.group(1))
            if failed_count > 0:
                self._result_state = "failed"
                self._error = Exception(
                    f"Benchmark has {failed_count} failed requests")
                print_error(
                    f"Benchmark output contains {failed_count} failed requests. Marking test as failed."
                )
                return

        # Check for explicit failure markers
        if "!FAILED REQUESTS!" in output or "!CHECK LOG FOR ERRORS!" in output:
            self._result_state = "failed"
            self._error = Exception("Benchmark output contains failure markers")
            print_error(
                "Benchmark output contains failure markers. Marking test as failed."
            )

    def run_ex(self,
               commands,
               full_test_name: str,
               metric_type: PerfMetricType,
               venv: Optional[PythonVenvRunnerImpl],
               gpu_clock_lock: GPUClockLock,
               session_data_writer: SessionDataWriter,
               output_dir: str,
               cmd_idx: int = 0,
               outputs: Dict[int, str] = {},
               original_test_name: str = None,
               **kwargs) -> List[str]:
        """
        Run the commands and write the results to the output csv and/or yaml files.
        """

        # Avoid modifying argument directly
        outputs = outputs.copy()

        # Initialize result status.
        self._perf_result = None
        self._result_state = "valid"
        self._error = None
        self._gpu_clock_lock = gpu_clock_lock
        tmpDir = temp_wd(self.get_working_dir())

        cmd_str = commands.get_cmd_str(cmd_idx)
        is_prepare_dataset_cmd = 'prepare_dataset' in cmd_str or "prepare-dataset" in cmd_str
        is_setup_cmd = is_prepare_dataset_cmd or metric_type is None
        # Start the timer.
        self._start_timestamp = datetime.utcnow()
        try:
            if cmd_idx not in outputs:
                # Capture the stdout from _gpu_clock_lock because the pipeline JUnit update script tries to parse
                # the log to find the GPU clocks.
                with io.StringIO() as buf:
                    if self._gpu_clock_lock:
                        # Lock GPU clock and start monitoring.
                        with contextlib.redirect_stdout(
                                buf), self._gpu_clock_lock, tmpDir:
                            output = commands.run_cmd(cmd_idx, venv)
                            # Print the output log to buf.
                            print(collect_and_clean_myelin_time(output))
                    else:
                        with contextlib.redirect_stdout(buf), tmpDir:
                            output = commands.run_cmd(cmd_idx, venv)
                            # Print the output log to buf.
                            # if not is_prepare_dataset_cmd:
                            print(collect_and_clean_myelin_time(output))

                    if not is_setup_cmd:
                        self._check_benchmark_output_for_errors(output)

                    if is_prepare_dataset_cmd:
                        # For prepare_dataset commands, only print the prepare command info
                        for line in buf.getvalue().split('\n'):
                            if 'Now running prepare data command' in line:
                                print(line)
                    else:
                        print(buf.getvalue())
                    outputs[cmd_idx] = buf.getvalue()
            else:
                print_info(f"Reusing cached logs for command index {cmd_idx}.")

        except InvalidGPUMonitoringResultError as e:
            # Mark result state as invalid when GPU monitoring result is invalid.
            self._result_state = "invalid"
            self._error = e
            print_error(
                f"Test result is invalid due to GPU monitoring issue. Error: {e}"
            )

        except Exception as e:
            # Mark result state as failed if anything else went wrong.
            self._result_state = "failed"
            self._error = e
            print_error(f"Test command failed. Error: {e}")
            partial_output = None
            if isinstance(e, subprocess.CalledProcessError):
                if e.stdout:
                    partial_output = clean_myelin_time(
                        e.stdout.decode("utf-8", errors="replace"))
                print_error("--- stdout ---")
                print_error(partial_output if partial_output else "<empty>")
                print_error("--------------")
                print_error("--- stderr ---")
                print_error(
                    e.stderr.decode("utf-8", errors="replace") if e.
                    stderr else "<empty>")
                print_error("--------------")
            if partial_output and cmd_idx not in outputs:
                outputs[cmd_idx] = partial_output
                print(f"\n{'=' * 60}")
                print(
                    f"PARTIAL OUTPUT (captured before failure, cmd_idx={cmd_idx}):"
                )
                print(f"{'=' * 60}")
                print(partial_output)
                print(f"{'=' * 60}\n")

        if self._result_state == "valid":
            if is_setup_cmd:
                print_info(
                    f"skip writing perf result for setup command (cmd_idx={cmd_idx})."
                )
                outputs.pop(cmd_idx, None)
            else:
                self._perf_result = self.get_perf_result(outputs)

                # Stop the timer
                self._end_timestamp = datetime.utcnow()

                # Write results to output csv and/or yaml files.
                self._write_result(full_test_name, session_data_writer,
                                   output_dir, outputs, original_test_name,
                                   metric_type, cmd_idx)

        return outputs

    def _write_result(self, full_test_name: str,
                      session_data_writer: SessionDataWriter, output_dir: str,
                      outputs: Dict[int, str], original_test_name: str,
                      metric_type: PerfMetricType, cmd_idx: int) -> None:
        """
        Store the test results in the _test_results.
        Write the test results and GPU monitoring data to the output csv and/or yaml files.
        """
        # Get GPU monitoring data
        self._gpu_monitor_data = self._gpu_clock_lock.get_state_data()
        if not self._gpu_monitor_data:
            print("WARNING: No GPU monitoring data available!")
            gpu_clocks = {}
        else:
            # Get the last record of _gpu_monitor_data
            gpu_clocks = self._gpu_monitor_data[-1].__dict__

        gpu_prop = self._gpu_clock_lock.get_gpu_properties()
        gpu_idx = None
        if gpu_prop:
            gpu_idx = gpu_prop.get("index", self._gpu_clock_lock.get_gpu_id())
        # Remove the prefix, which includes the platform info, for network_hash.
        short_test_name = full_test_name.split("::")[-1]
        # Get device subtype for autodeploy tests
        device_subtype = None
        if self._gpu_clock_lock:
            device_subtype = self._gpu_clock_lock.get_device_subtype()

        test_description_dict = {
            "network_name": self.get_test_name(),
            "network_hash":
            short_test_name,  # This is used by the PerfDB to identify a test.
            "sm_clk": gpu_clocks.get("gpu_clock__MHz", None),
            "mem_clk": gpu_clocks.get("memory_clock__MHz", None),
            "gpu_idx": gpu_idx,
            "device_subtype": device_subtype
        }

        # Serialize the commands.
        serialized_cmd = self.get_commands().get_cmd_str(cmd_idx)
        # Save engine building log + benchmarking log in the csv file.
        raw_result = ""
        if 0 in outputs:
            raw_result += outputs[0]
        if cmd_idx > 0 and cmd_idx in outputs:
            raw_result += "\n" + outputs[cmd_idx]

        # Construct test result dict.
        test_result_dict = {
            "perf_case_name":
            full_test_name,
            "test_name":
            short_test_name,
            "original_test_name":
            original_test_name
            if original_test_name is not None else full_test_name,
            "perf_metric":
            self._perf_result,
            "total_time__sec":
            (self._end_timestamp - self._start_timestamp).total_seconds(),
            "start_timestamp":
            self._start_timestamp.strftime("%Y-%m-%d %H:%M:%S %z").rstrip(),
            "end_timestamp":
            self._end_timestamp.strftime("%Y-%m-%d %H:%M:%S %z").rstrip(),
            "state":
            self._result_state,
            "command":
            serialized_cmd,
            "threshold":
            self.get_threshold(),
            "absolute_threshold":
            self.get_absolute_threshold(),
            "metric_type":
            self.get_metric_type().value,
        }

        for result_format in session_data_writer._output_formats:
            print(
                f"""------------------Writing results to {result_format} format...--------------------------"""
            )
        # Write results in CSV format.
        if "csv" in session_data_writer._output_formats:
            csv_name = "perf_script_test_results.csv"
            cvs_result_dict = {**test_description_dict, **test_result_dict}
            if "raw_result" in cvs_result_dict:
                cvs_result_dict["raw_result"] = cvs_result_dict[
                    "raw_result"].replace("\n", "\\n")
            write_csv(output_dir,
                      csv_name, [cvs_result_dict],
                      list(cvs_result_dict.keys()),
                      append_mode=os.path.exists(
                          os.path.join(output_dir, csv_name)))
            if self._gpu_monitor_data is not None:
                write_gpu_monitoring_no_test_results(output_dir,
                                                     self._gpu_monitor_data,
                                                     output="csv",
                                                     append_mode=True)

        # Write results in YAML format.
        if "yaml" in session_data_writer._output_formats:
            yaml_name = "perf_script_test_results.yaml"
            monitor_data_dict = [{
                key: getattr(i, key)
                for key in GPU_MONITORING_FORMAT_KEYS
            } for i in self._gpu_monitor_data]
            yaml_result_dict = {
                "monitor_data": {
                    "cpu": [],
                    "os": [],
                    "gpu": monitor_data_dict,
                },
                "test_description": test_description_dict,
                "test_result": test_result_dict,
            }
            yaml_result_dict = {
                yaml_result_dict["test_result"]["test_name"]: yaml_result_dict
            }
            write_yaml(output_dir,
                       yaml_name,
                       yaml_result_dict,
                       append_mode=os.path.exists(
                           os.path.join(output_dir, yaml_name)))


def generate_one_test_node(session, config, domain_name, test_name, test_func):
    """
    A helper function to create a PyTest item with the specific name and specific test function.
    """

    # Create the parent Item node.
    # Pytest 8.x upgrade compatibility.
    # We should never import Pytest internals within test-definitions.
    # TODO: TRT-23565
    parent = None
    if hasattr(Item, "from_parent"):

        class TrtexecItem(Item):

            def __init__(self, **kwargs):
                super().__init__(**kwargs)

            def runtest(self):
                return test_func()

        parent = TrtexecItem.from_parent(session,
                                         name=domain_name,
                                         nodeid=domain_name)
    else:
        parent = Item(name=domain_name,
                      parent=session,
                      config=config,
                      session=session,
                      nodeid=domain_name)

    parent.obj = None

    # Create the Function node for the test.
    # For pytest 8.x compatibility
    # TODO: TRT-23565
    item = None
    if hasattr(Function, "from_parent"):
        item = Function.from_parent(parent, name=test_name, callobj=test_func)
    else:
        item = Function(name=test_name,
                        parent=parent,
                        config=config,
                        session=session,
                        callobj=test_func)

    item.obj = test_func

    # This has to be set but can be random as it isn't used.
    item.path = Path(os.getcwd())

    return item


def generate_test_nodes(session, config, items, valid_prefixes: List[str],
                        test_func):
    """
    A helper function to generate test nodes that have specific prefixes based on the test lists.
    """

    test_list = config.getoption("--test-list")
    test_prefix = config.getoption("--test-prefix") if config.getoption(
        "--test-prefix") else ""

    # Read all the test names.
    all_tests = []
    try:
        test_names, _ = parse_test_list(test_list, test_prefix=test_prefix)
        all_tests.extend(test_names)
    except FileNotFoundError:
        pass

    # Go through all test names and find the ones that need to be generated.
    for test_name in all_tests:

        # Remove the prefix if it exists.
        if len(test_prefix) > 0 and test_name.startswith(test_prefix):
            test_name = test_name.replace(test_prefix + "/", "", 1)

        if any([test_name.startswith(p) for p in valid_prefixes]):
            domain_name = "::".join(test_name.split("::")[:-1])
            short_test_name = test_name.replace(domain_name + "::", "", 1)
            # Generate the test node and append it to the Pytest item list.
            items.append(
                generate_one_test_node(session, config, domain_name,
                                       short_test_name, test_func))
            print(f"Dynamically generated test node: {test_name}")

    return items
