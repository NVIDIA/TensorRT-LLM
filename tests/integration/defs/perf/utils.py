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
import abc
import contextlib
import copy
import io
import os
import re
import subprocess
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional

from _pytest.nodes import Item
from _pytest.python import Function
from defs.trt_test_alternative import check_output, print_error, print_info

from ..common import get_trt_llm_lib_dir, venv_mpi_check_output
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
    An string-enum type to define what kind of perf metric it is. While it is not used by TURTLE, it is used by QA to
    set up special threshold criteria for each type of metrics (like >50MB for engine size increase, etc.).
    """
    INFERENCE_TIME = "INFERENCE_TIME"
    FIRST_TOKEN_TIME = "FIRST_TOKEN_TIME"
    OUTPUT_TOKEN_TIME = "OUTPUT_TOKEN_TIME"
    TOKEN_THROUGHPUT = "TOKEN_THROUGHPUT"
    BUILD_TIME = "BUILD_TIME"
    BUILD_PEAK_CPU_MEMORY = "BUILD_PEAK_CPU_MEMORY"
    BUILD_PEAK_GPU_MEMORY = "BUILD_PEAK_GPU_MEMORY"
    INFERENCE_PEAK_GPU_MEMORY = "INFERENCE_PEAK_GPU_MEMORY"
    ENGINE_SIZE = "ENGINE_SIZE"
    CONTEXT_GPU_MEMORY = "CONTEXT_GPU_MEMORY"
    SEQ_THROUGHPUT = "SEQ_THROUGHPUT"
    SEQ_LATENCY = "SEQ_LATENCY"
    KV_CACHE_SIZE = "KV_CACHE_SIZE"


class PerfScriptTestCmds(NamedTuple):
    convert_cmd: List[str]
    build_cmd: List[str]
    data_cmds: List[List[str]]
    benchmark_cmds: List[List[str]]
    mpi_cmd: List[str]
    is_python: bool

    def run_cmd(self, cmd_idx: int, venv) -> str:
        output = ""
        mpi_cmd = self.mpi_cmd
        build_cmd_str = self.get_cmd_str(0)
        current_cmd_str = self.get_cmd_str(cmd_idx)
        if cmd_idx == 0:
            if self.build_cmd[0].endswith('.py'):
                print_info(
                    f'Running engine building command: "{build_cmd_str}"')
                if len(mpi_cmd) > 0:
                    output += venv_mpi_check_output(venv, mpi_cmd,
                                                    self.build_cmd)
                else:
                    output += venv.run_cmd(self.build_cmd, caller=check_output)
            else:
                envs = copy.deepcopy(os.environ)
                if len(self.convert_cmd) > 0:
                    convert_cmd_str = " ".join(self.convert_cmd)
                    convert_cmds = convert_cmd_str.split(';')
                    for convert_cmd in convert_cmds:
                        print_info(
                            f'Running convert weights command: "{convert_cmd}"')
                        output += subprocess.check_output(convert_cmd.split(),
                                                          env=envs).decode()
                print_info(
                    f'Running engine building command: "{build_cmd_str}"')
                command = self.build_cmd
                output += subprocess.check_output(command, env=envs).decode()
        else:
            print_info(f'Engine building command was: "{build_cmd_str}"')

            if len(self.data_cmds) >= cmd_idx:
                prepare_cmd = self.data_cmds[cmd_idx - 1]
                prepare_cmd_str = " ".join(prepare_cmd)
                envs = copy.deepcopy(os.environ)
                prepare_cmds = prepare_cmd_str.split(';')
                for cmd in prepare_cmds:
                    print(f'Now running prepare data command: "{cmd}"')
                    output += subprocess.check_output(cmd.split(),
                                                      env=envs).decode()

            print(f'Now running benchmarking command: "{current_cmd_str}"')
            command = self.benchmark_cmds[cmd_idx - 1]
            if self.is_python:
                if len(mpi_cmd) > 0:
                    output += venv_mpi_check_output(venv, mpi_cmd, command)
                else:
                    output += venv.run_cmd(command, caller=check_output)
            else:
                envs = copy.deepcopy(os.environ)
                # Set LD_LIBRARY_PATH to the directory where the binary is located to find libtensorrt_llm.so and
                # libnvinfer_plugin_tensorrt_llm.so.x.
                envs[
                    "LD_LIBRARY_PATH"] = f'{get_trt_llm_lib_dir(venv)}:{os.path.dirname(command[0])}:{envs.get("LD_LIBRARY_PATH", "")}'
                print(
                    f'Augmented cpp runtime LD_LIBRARY_PATH={envs["LD_LIBRARY_PATH"]}'
                )
                benchmark_cmd = mpi_cmd + command
                output += subprocess.check_output(benchmark_cmd,
                                                  env=envs).decode()
        return output

    def get_cmd_str(self, cmd_idx) -> List[str]:
        mpi_cmd_str = (" ".join(self.mpi_cmd) +
                       " ") if len(self.mpi_cmd) > 0 else ""

        if cmd_idx == 0:
            if self.build_cmd[0].endswith('.py'):
                cmd_str = mpi_cmd_str + "python3 " + " ".join(self.build_cmd)
            else:
                cmd_str = mpi_cmd_str + "  " + " ".join(self.build_cmd)
        else:
            python_str = "python3 " if self.is_python else ""
            cmd_str = mpi_cmd_str + python_str + " ".join(
                self.benchmark_cmds[cmd_idx - 1])

        return cmd_str


@contextlib.contextmanager
def temp_wd(path):
    """A context manager to temporarily change the working directory."""
    prev_cwd = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(prev_cwd)


class PerfBenchScriptTestCmds(NamedTuple):
    data_cmds: List[List[str]]
    build_cmd: List[str]
    benchmark_cmds: List[List[str]]
    mpi_cmd: List[str]
    is_python: bool

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
                print(f'Now running prepare data command: "{prepare_cmd}"')
                cmd = prepare_cmd.split('>')[0]
                dataset_file = prepare_cmd.split('>')[1].split()[0]
                output += subprocess.check_output(cmd.split(),
                                                  env=envs).decode()
                with open(f"{dataset_file}", 'w+') as f:
                    f.write(output)

        elif cmd_idx == len(self.data_cmds):
            #running build
            if len(self.build_cmd) == 0:
                pass
            elif self.build_cmd[0].endswith('.py'):
                print_info(
                    f'Running engine building command: "{build_cmd_str}"')
                if len(mpi_cmd) > 0:
                    output += venv_mpi_check_output(venv, mpi_cmd,
                                                    self.build_cmd)
                else:
                    output += venv.run_cmd(self.build_cmd, caller=check_output)
            else:
                envs = copy.deepcopy(os.environ)
                print_info(
                    f'Running engine building command: "{build_cmd_str}"')
                command = self.build_cmd
                output += subprocess.check_output(command, env=envs).decode()
        else:
            #running throughput
            print(f'Now running benchmarking command: "{current_cmd_str}"')
            command = self.benchmark_cmds[cmd_idx - 1 - len(self.data_cmds)]
            if self.is_python:
                if len(mpi_cmd) > 0:
                    output += venv_mpi_check_output(venv, mpi_cmd, command)
                else:
                    output += venv.run_cmd(command, caller=check_output)
            else:
                envs = copy.deepcopy(os.environ)
                # Set LD_LIBRARY_PATH to the directory where the binary is located to find libtensorrt_llm.so and
                # libnvinfer_plugin_tensorrt_llm.so.x.
                envs[
                    "LD_LIBRARY_PATH"] = f'{get_trt_llm_lib_dir(venv)}:{os.path.dirname(command[0])}:{envs.get("LD_LIBRARY_PATH", "")}'
                print(
                    f'Augmented cpp runtime LD_LIBRARY_PATH={envs["LD_LIBRARY_PATH"]}'
                )
                benchmark_cmd = mpi_cmd + command
                output += subprocess.check_output(benchmark_cmd,
                                                  env=envs).decode()
        return output

    def get_cmd_str(self, cmd_idx) -> List[str]:
        mpi_cmd_str = (" ".join(self.mpi_cmd) +
                       " ") if len(self.mpi_cmd) > 0 else ""
        if cmd_idx <= len(self.data_cmds) - 1:
            cmd_str = " ".join(self.data_cmds[cmd_idx])
        elif cmd_idx == len(self.data_cmds):  #for trtllm-bench build command
            if len(self.build_cmd) == 0:
                cmd_str = ''
            elif self.build_cmd[0].endswith('.py'):
                cmd_str = mpi_cmd_str + "python3 " + " ".join(self.build_cmd)
            else:
                cmd_str = mpi_cmd_str + "  " + " ".join(self.build_cmd)
        else:
            python_str = "python3 " if self.is_python else ""
            cmd_str = mpi_cmd_str + python_str + " ".join(
                self.benchmark_cmds[cmd_idx - 1 - len(self.data_cmds)])

        return cmd_str


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
    def get_commands(self) -> PerfScriptTestCmds:
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
        Note: This is not honored by TURTLE for now, but we can add the support later.
        """
        return 0.0

    def get_metric_type(self) -> PerfMetricType:
        """
        Get the type of perf metric. This does not affect TURTLE for now, but QA uses this field to set up special
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

    def run_ex(self,
               full_test_name: str,
               venv: Optional[PythonVenvRunnerImpl],
               gpu_clock_lock: GPUClockLock,
               session_data_writer: SessionDataWriter,
               output_dir: str,
               outputs: Dict[int, str] = {},
               original_test_name: str = None,
               cmd_idx: int = 0,
               **kwargs) -> List[str]:
        """
        Run the commands and write the results to the output csv and/or yaml files.
        """

        # Get the commands.
        commands = self.get_commands()

        # Avoid modifying argument directly
        outputs = outputs.copy()

        # Initialize result status.
        self._perf_result = None
        self._result_state = "valid"
        self._error = None
        self._gpu_clock_lock = gpu_clock_lock
        tmpDir = temp_wd(self.get_working_dir())

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
                            print(collect_and_clean_myelin_time(output))

                    # Print the output log to stdout and cache it.
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
            if isinstance(e, subprocess.CalledProcessError):
                print_error("--- stdout ---")
                if e.stdout:
                    print_error(clean_myelin_time(e.stdout.decode()))
                else:
                    print_error("<empty>")
                print_error("--------------")
                print_error("--- stderr ---")
                print_error(e.stderr.decode() if e.stderr else "<empty>")
                print_error("--------------")

        # Only save perf result if the result is valid.
        if self._result_state == "valid":
            # Parse the perf result from the test outputs.
            if self._config.runtime == 'bench' and cmd_idx == 0:
                print_info(
                    f"skip writing perf result when calling generating dataset in trtllm-bench"
                )
            else:
                self._perf_result = self.get_perf_result(outputs)

                # Stop the timer
                self._end_timestamp = datetime.utcnow()

                # Write results to output csv and/or yaml files.
                self._write_result(full_test_name, session_data_writer,
                                   output_dir, outputs, original_test_name,
                                   cmd_idx)

        return outputs

    def _write_result(self, full_test_name: str,
                      session_data_writer: SessionDataWriter, output_dir: str,
                      outputs: Dict[int, str], original_test_name: str,
                      cmd_idx: int) -> None:
        """
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
        test_description_dict = {
            "network_name": self.get_test_name(),
            "network_hash":
            short_test_name,  # This is used by the PerfDB to identify a test.
            "sm_clk": gpu_clocks.get("gpu_clock__MHz", None),
            "mem_clk": gpu_clocks.get("memory_clock__MHz", None),
            "gpu_idx": gpu_idx
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
            "raw_result":
            raw_result,
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
