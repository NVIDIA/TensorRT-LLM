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
"""
File to house data classes related to perf runs.
Compatible with data_export.py for exporting data.
"""


class Data:
    """Base class to support easier ways to interface and assign data to class."""

    def __init__(self, **kwargs):
        """
        Simple constructor that supports optional params
        to hook to class objects.
        """
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise KeyError("Unable to assign value to PerfResult: " + key)

    def __str__(self):
        return str(self.__dict__)

    def __repr__(self):
        return str(self.__dict__)

    def get(self, key, *args, **kwargs):
        """
            Args:
                key - Get key of the data
                default (optional) - Returns given default value if key is None.
        """
        if len(args) > 1:
            raise RuntimeError(
                "Expected 1-2 parameters, but {} were given: {}".format(
                    len(args), args[1:]))

        has_default = False
        default = None
        if "default" in kwargs and len(args) == 1:
            raise RuntimeError(
                "Expected default={} but did not expect {} in parameters".
                format(kwargs["default"], args[0]))
        if "default" in kwargs:
            has_default = True
            default = kwargs["default"]
        elif len(args) == 1:
            has_default = True
            default = args[0]

        # Used so that the class can act like a dictionary
        # Works for now instead of meta class
        return getattr(self, key, default) if has_default else getattr(
            self, key)

    def update(self, dct):
        for key, value in dct.items():
            setattr(self, key, value)


class SessionData(Data):
    """
    Class to store session specific results for perf runs.
    """

    start_timestamp = None
    end_timestamp = None

    # Native NRSU specific collection
    os_properties = None
    cpu_properties = None
    gpu_properties = None
    nvidia_device_count = None
    nvidia_driver_version = None

    # OS Specific
    username = None
    hostname = None
    ip = None

    # TensorRT specific properties
    trt_change_id = None
    trt_branch = None
    commit_timestamp = None
    cuda_version = None
    cublas_version = None
    cudnn_version = None
    #trt_version = None


class PerfResult(Data):
    """
    Stores all relevant data from a perf run. Using class over dictionary
    for easier documentation and reference to what values are stored
    inside class. Can be more intuitive via key mappings with meta class.
    """

    # Class attributes can be set in initialization
    start_time = None
    end_time = None
    total_time = None
    engine_build_time = None
    engine_load_time = None
    engine_file_size = None
    throughput = None
    perf_time = None
    trt_peak_cpu_mem = None
    trt_peak_gpu_mem = None
    build_engine_allocated_cpu_mem = None
    build_engine_allocated_gpu_mem = None
    deserialize_engine_allocated_cpu_mem = None
    deserialize_engine_allocated_gpu_mem = None
    execution_context_allocated_cpu_mem = None
    execution_context_allocated_gpu_mem = None
    state = None
    gpu_monitor = None

    # Session specific information
    sm_clk = None
    mem_clk = None
    gpu_idx = None
    raw_result = None
    command_str = None

    # ARIA v3 Specific
    network_name = None
    framework = None
    network_hash = None
    flags = None

    # Log File Location, Used for Reading Later
    log_file = None
    return_code = None
