# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Functions to output results files in csv. Generates three files:
test_results.csv, session_properties.csv, and gpu_monitoring.csv
Output files are the same as RABBIT formats.
Keep remapping global arrays are used to map internal values to
export keys. Used so that outputs conform to a certain structure for
backwards compatibility.
"""

import csv
import datetime
import os
import re
from collections import OrderedDict

import oyaml
from defs.trt_test_alternative import print_warning

#
# Export specific keys
#

# Used by to verify gpu properties integrity
REQUIRED_GPU_PROPERTIES = [
    "serial_number", "pci_device_id", "architecture", "chip",
    "device_product_name", "multiprocessor_count", "compute_capability",
    "vbios_version", "device_brand_name", "mem_total__MB", "pci_bus_id",
    "pci_sub_system_id", "power_limit__W", "uuid", "interface", "index"
]

#
# Key Remapping Constructs
#

# Maps original key to output key name
SESSION_FORMAT_KEYS = [
    "username",
    "start_timestamp",
    "hostname",
    "ip",
    "nvidia_driver_version",
    "nvidia_device_count",
    "os_properties",
    "cpu_properties",
    "gpu_properties",
    "trt_change_id",
    "trt_branch",
    "commit_timestamp",
    "cuda_version",
    "cublas_version",
    "cudnn_version",
    #"trt_version",
    "end_timestamp"
]

# Maps original key to output key name
# Uses tuple to do (src, dest) key names
TEST_DESCRIPTION_KEYS = [
    "perf_case_name", "network_name", "framework", "sm_clk", "mem_clk",
    "gpu_idx", "network_hash", "flags", "device_subtype"
]

# Tuples are used if the internal dictionary keys are different from output dictionary keys.
# Tuples should be: ("internal key", "output key").
TEST_OTHER_KEYS = [
    ("log_file", "raw_result"),  # Requires also return_code
    ("engine_build_time", "engine_build_time__sec"),
    ("engine_load_time", "engine_load_time__sec"),
    ("engine_file_size", "engine_file_size__MB"),
    ("throughput", "throughput__qps"),
    ("perf_time", "run_time__msec"),
    ("trt_peak_cpu_mem", "trt_peak_cpu_mem__MB"),
    ("trt_peak_gpu_mem", "trt_peak_gpu_mem__MB"),
    ("build_engine_allocated_cpu_mem", "build_engine_allocated_cpu_mem__MB"),
    ("build_engine_allocated_gpu_mem", "build_engine_allocated_gpu_mem__MB"),
    ("deserialize_engine_allocated_cpu_mem",
     "deserialize_engine_allocated_cpu_mem__MB"),
    ("deserialize_engine_allocated_gpu_mem",
     "deserialize_engine_allocated_gpu_mem__MB"),
    ("execution_context_allocated_cpu_mem",
     "execution_context_allocated_cpu_mem__MB"),
    ("execution_context_allocated_gpu_mem",
     "execution_context_allocated_gpu_mem__MB"),
    ("total_time", "total_time__sec"),
    ("start_time", "start_timestamp"),
    ("end_time", "end_timestamp"),
    "state",
    ("command_str", "command")
]

TEST_FORMAT_KEYS = TEST_DESCRIPTION_KEYS + TEST_OTHER_KEYS

GPU_MONITORING_FORMAT_KEYS = [
    "gpu_id",
    "timestamp",
    "gpu_clock__MHz",
    "memory_clock__MHz",
    "graphics_clock__MHz",
    "gpu_utilization__pct",
    "memory_utilization__pct",
    "encoder_utilization__pct",
    "decoder_utilization__pct",
    "gpu_temperature__C",
    "memory_temperature__C",
    "fan_speed__pct",
    "perf_state",
    "power_draw__W",
    "process_num",
]

# Key used in YAML output for key in GPU_MONITORING
GPU_MONITOR_KEY = "monitor_data"
TEST_DESC_KEY = "test_description"
TEST_RESULT_KEY = "test_result"

# Keys that require transforming from file name string to physical dump of file
EXPAND_TO_LOG_DUMP_KEY = ["log_file"]


def write_csv(fpath, fname, write_list, keys_list, append_mode=False):
    """Function to write a list of dictionary values from _generate_write_list to CSV."""
    filename = os.path.join(fpath, fname)

    # Create fieldnames
    fieldnames = []
    for val in keys_list:
        if isinstance(val, tuple):
            fieldnames.append(val[1])
        else:
            fieldnames.append(val)

    write_mode = "a+" if append_mode else "w"
    # This has to be checked before open() is called since the file may be created otherwise
    is_new_file = not os.path.exists(filename)
    with open(filename, write_mode, newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        # Only write the header if we haven't written the header before (new file)
        # or if we are in non-append mode.
        if is_new_file or not append_mode:
            writer.writeheader()

        # Write the actual rows of data
        for write_dict in write_list:
            writer.writerow(write_dict)


def write_yaml(fpath, fname, write_dict, append_mode=False):
    filename = os.path.join(fpath, fname)
    # If append is set, use append to file instead of update() included
    # in oyaml library because update() attempt to parse the file first, which
    # takes extra cycles.
    with open(filename, "a+" if append_mode else "w") as yamlfile:
        res = oyaml.dump(write_dict)
        yamlfile.write(res)


def write_gpu_monitoring_results(fpath,
                                 test_results,
                                 gpu_monitor,
                                 gpu_uuid,
                                 output="yaml",
                                 include_keys=None,
                                 append_mode=True):
    """
    Main function to call all three gpu writing functions depending on output type.
    Writing functions use globally defined list variables with KEYS to determine
    which data values to output to csv, yaml etc. Since csv and yaml files are slightly
    different in formatting, the lists present a common interface between the output data
    and the internal representation of the format.
    """
    if output == "csv":
        # Generate monitoring data without test name because of csv output
        # Since gpu_monitor can contain None values (to indicate a wiped GPU monitoring cache and occurs if a run fails before GPU monitoring is set)
        # we need to process gpu_monitor
        monitor_values = [
            m if m is not None else [] for m in gpu_monitor.values()
        ]

        gpu_monitor_list = [
            storage for gpm in monitor_values for storage in gpm
        ]
        write_gpu_monitoring_no_test_results(fpath,
                                             gpu_monitor_list,
                                             output=output,
                                             append_mode=append_mode)

        # Output only lists of values because of csv output
        write_test_results(fpath,
                           list(test_results.values()),
                           output=output,
                           include_keys=include_keys,
                           append_mode=append_mode)
    else:
        # When outputting YAML, the test_results file varies slightly with gpu_monitoring baked into the
        # the program, they are packaged together instead
        write_test_results_with_gpu_monitoring(fpath,
                                               test_results,
                                               gpu_monitor,
                                               gpu_uuid,
                                               output=output,
                                               include_keys=include_keys,
                                               append_mode=append_mode)


def write_test_results_with_gpu_monitoring(fpath,
                                           test_results,
                                           gpu_monitor,
                                           gpu_uuid,
                                           output="yaml",
                                           include_keys=None,
                                           append_mode=True):
    """
    Yaml version of output that appends gpu_monitoring data atop.
    Despite the output type option, mainly used for yaml.
    Supports appending for long gpu data that requires dumping.
    """
    output_dict = {}

    # Get a subset of the output if possible
    description_keys = _get_subset_keys(TEST_DESCRIPTION_KEYS, include_keys)
    other_keys = _get_subset_keys(TEST_OTHER_KEYS, include_keys)

    # This for loop mainly enforces test keys constraints as the program
    # goes through each list.
    for test_name, test in test_results.items():
        # Create test format list
        # wrap in a list because the generate command assumes multiple tests
        test_write_list = _generate_write_list(other_keys, [test],
                                               _write_test_results_helper)
        # Exclusive to yaml files, requires extra key
        test_desc_write_list = _generate_write_list(description_keys, [test],
                                                    _write_test_results_helper)
        # Create gpu_monitor write_list
        gpu_monitor_write_list = _generate_write_list(
            GPU_MONITORING_FORMAT_KEYS, gpu_monitor[test_name],
            _write_gpu_monitoring_helper)

        # Add values to write list, strip list.
        # Concatenate the two dicts
        output_dict[test_name] = {
            GPU_MONITOR_KEY: {
                "cpu": [],
                "os": [],
                "gpu": {
                    gpu_uuid: gpu_monitor_write_list
                }
            },
            TEST_DESC_KEY: test_desc_write_list[0],
            TEST_RESULT_KEY: test_write_list[0]
        }

    # Finally output results
    write_yaml(fpath, "test_results.yml", output_dict, append_mode)


def write_test_results(logpath,
                       test_result_list,
                       output="csv",
                       include_keys=None,
                       append_mode=True):
    """
    Writes perf test results to logs folder.
    Uses TEST_FORMAT_KEYS to get mapping from keys to values.
    Also uses EXPAND_TO_LOG_DUMP_KEY to expand an attribute with logfile name
    into the content of the logs.

    Args:
        fpath: Path to log directory
        test_result_list: list of PerfResults or any class that supports getattr().
    """
    format_keys = _get_subset_keys(TEST_FORMAT_KEYS, include_keys)

    write_list = _generate_write_list(format_keys, test_result_list,
                                      _write_test_results_helper)

    if output == "csv":
        write_csv(logpath,
                  "perf_test_results.csv",
                  write_list,
                  format_keys,
                  append_mode=append_mode)
    elif output == "yaml":
        # Write yaml requires a dictionary, create a dictionary instead
        write_yaml(logpath,
                   "perf_test_results.yml",
                   dict(zip((format_keys, write_list))),
                   append_mode=append_mode)
    else:
        print_warning("Invalid print option given.")


def write_session_properties(logpath, session_data_list, output="csv"):
    write_list = _generate_write_list(SESSION_FORMAT_KEYS, session_data_list,
                                      _write_session_properties_helper)

    if output == "csv":
        write_csv(logpath, "session_properties.csv", write_list,
                  SESSION_FORMAT_KEYS)
    elif output == "yaml":
        # Write yaml requires a dictionary, create a dictionary instead
        write_yaml(logpath, "session_data.yml", write_list[0])
    else:
        print_warning("Invalid print option given.")


def write_gpu_monitoring_no_test_results(logpath,
                                         gpu_monitoring_data_list,
                                         output="csv",
                                         append_mode=True):
    """
    Writes perf test results to logs folder.
    Uses GPU_MONITORING_FORMAT_KEYS to get mapping from keys to values.
    into the content of the logs.
    Supports append mode for GPU dumping.

    Args:
        fpath: Path to log directory
        gpu_monitoring_data_list: List of GPU monitoring data from GPUMonitor.
    """
    write_list = _generate_write_list(GPU_MONITORING_FORMAT_KEYS,
                                      gpu_monitoring_data_list,
                                      _write_gpu_monitoring_helper)

    if output == "csv":
        write_csv(logpath, "gpu_monitoring.csv", write_list,
                  GPU_MONITORING_FORMAT_KEYS, append_mode)
    elif output == "yaml":
        # Legacy code, can be deprecated.
        # Write yaml requires a dictionary, create a dictionary instead
        write_yaml(logpath, "gpu_monitoring.yml",
                   dict(zip(GPU_MONITORING_FORMAT_KEYS, write_list)),
                   append_mode)
    else:
        print_warning("Invalid print option given.")


def get_log(fpath):
    """
    Converts log output into an ordered dict of stdout and stderr.
    Used for raw_result for test_result.

    Args:
        fpath: File path for the log file.
    """
    log_odict = OrderedDict()
    regex = re.compile(r'>>> (.*) for .*\n')
    cur_output = None
    with open(os.path.join(fpath), "r") as logfile:
        for line in logfile:
            # Use match to only look at beginning of string, faster.
            result = regex.match(line)
            if result:
                cur_output = result.group(1)
                log_odict[cur_output] = ""
            else:
                log_odict[cur_output] = log_odict.get(cur_output, "") + line

    return log_odict


#
# Internal Helper Functions
#


def _generate_write_list(keys_list, data_list, apply_func):
    """
    Helper function for generating list to write in csv, yaml, etc. from
    list of data values that support getattr(). apply_function returns a
    modified write_dict.

    Args:
        data_list: list of class types that support getattr() for each key.
        apply_func: helper function applied to each datatype.
                    Input: write_dict, test_result, src_key, dst_key
    """
    # Calculate the test results
    if data_list is None:
        return []

    write_list = []
    for data in data_list:
        write_dict = {}
        for key in keys_list:
            src_key = key
            dst_key = key

            if type(key) is tuple:
                src_key = key[0]
                dst_key = key[1]

            # Run custom function to do job
            write_dict = apply_func(write_dict, data, src_key, dst_key)

        write_list.append(write_dict)

    return write_list


def _str_convert_helper(data, key):
    # Gets the string of the value
    if isinstance(data, datetime.timedelta):
        return str(data)
    elif isinstance(data, datetime.datetime):
        # rstrip is necessary for some timestreams that don't have proper zones
        return data.strftime("%Y-%m-%d %H:%M:%S %z").rstrip()

    return data


# Helper function to transformat data in _generate_write_list
def _write_test_results_helper(write_dict, test_result, src_key, dst_key):
    # See if need to inject logs instead
    if src_key in EXPAND_TO_LOG_DUMP_KEY:
        # Inject the log in the place file name key with new key as
        # defined in EXPAND_TO_LOG_DUMP_KEY
        log_file_name = getattr(test_result, src_key)
        logdict = "No Logs Injected or Registered"

        # Check if someone the log could not be found or set
        if log_file_name is None:
            print_warning("Unable to inject log.")
        else:
            logdict = get_log(log_file_name)

        # Add return code to dict
        logdict["returncode"] = getattr(test_result, "return_code")
        write_dict[dst_key] = str(logdict)
    else:
        # Use a get operator so that dictionaries or custom classes can be used
        try:
            write_dict[dst_key] = _str_convert_helper(
                getattr(test_result, src_key), dst_key)

        except:
            print_warning(
                "Missing key {} in exporting test results.".format(src_key))
            write_dict[dst_key] = None

    return write_dict


def _write_gpu_monitoring_helper(write_dict, monitoring_data, src_key, dst_key):
    # Use a get operator so that dictionaries or custom classes can be used
    try:
        write_dict[dst_key] = getattr(monitoring_data, src_key)
    except:
        print_warning(
            "Missing key {} in exporting gpu results.".format(src_key))
        write_dict[dst_key] = None

    return write_dict


def _write_session_properties_helper(write_dict, session_data_list, src_key,
                                     dst_key):
    try:
        write_dict[dst_key] = _str_convert_helper(
            getattr(session_data_list, src_key), dst_key)
    except Exception:
        print_warning(
            "Missing key {} in exporting session results.".format(src_key))
        write_dict[dst_key] = None

    return write_dict


def _get_subset_keys(key_list, include_keys=[]):
    """Returns a subset of keys from a key list"""

    if include_keys is None:
        return key_list

    res = []
    for k in key_list:
        if type(k) is tuple and k[0] in include_keys:
            res.append(k)
        elif k in include_keys:
            res.append(k)
    return res
