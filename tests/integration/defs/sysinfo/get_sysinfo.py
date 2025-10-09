# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import argparse
import json
import logging
import os
import platform
import re
import socket
import sys

import psutil
import pynvml

try:
    from cuda.bindings import driver as cuda
except ImportError:
    from cuda import cuda

# Logger
logger = logging.getLogger(__name__)

# Globals
chip_name_mapping = {}


def load_chip_name_mapping(file_path):
    global chip_name_mapping
    try:
        with open(file_path, "r") as fp:
            chip_name_mapping = json.load(fp)
    except Exception as e:
        logger.warning(f"Failed to load chip name mapping: {e}")


def get_host_memory_info():
    # Get memory information
    memory = psutil.virtual_memory()

    # Convert to MiB
    total_memory_mib = memory.total / (1024 * 1024)
    available_memory_mib = memory.available / (1024 * 1024)

    return round(total_memory_mib, 2), round(available_memory_mib, 2)


def ASSERT_DRV(err):
    if isinstance(err, cuda.CUresult):
        if err != cuda.CUresult.CUDA_SUCCESS:
            raise RuntimeError('Cuda Error: {}'.format(err))
    elif isinstance(err, nvrtc.nvrtcResult):
        if err != nvrtc.nvrtcResult.NVRTC_SUCCESS:
            raise RuntimeError('Nvrtc Error: {}'.format(err))
    else:
        raise RuntimeError('Unknown error type: {}'.format(err))


def get_compute_capability(device_id=0):
    # Init
    err, = cuda.cuInit(0)
    ASSERT_DRV(err)

    # Device
    err, cuDevice = cuda.cuDeviceGet(0)
    ASSERT_DRV(err)

    # Get target architecture
    err, sm_major = cuda.cuDeviceGetAttribute(
        cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
        cuDevice)
    ASSERT_DRV(err)
    err, sm_minor = cuda.cuDeviceGetAttribute(
        cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
        cuDevice)
    ASSERT_DRV(err)

    return ".".join((str(sm_major), str(sm_minor)))


def is_aarch64():
    """Function to detect if running on ARM"""
    return platform.processor().startswith("aarch64")


def is_linux():
    """Function to detect if running on Linux"""
    try:
        return os.uname()[0].lower() == "linux"
    except:
        return False


def is_power():
    """Function to detect if running on IBM Power"""
    return platform.processor() == "ppc64le"


def get_linux_distribution():
    try:
        import distro
        return (distro.id(), distro.version(), distro.codename())
    except:
        logger.warning(
            "Unable to use distro module, defaulting operating system to ('na', 'na', 'na')"
        )
        return ("na", "na", "na")


def is_windows():
    """ Function used to detect if either running on native or WSL """
    return is_native_windows() or is_wsl()


def try_set(write_dict, key, callable_func):
    try:
        write_dict[key] = callable_func()
    except Exception as e:
        logger.warning("Unable to set mako variable {}: {}".format(key, e))
    return write_dict


def is_wsl():
    try:
        with open("/proc/version", "r") as f:
            for line in f:
                if re.search("microsoft", line, re.I):
                    return True
        return False
    except:
        return False


def is_native_windows():
    """Function used to detect if running native windows"""
    return sys.platform == "win32"


def parse_key_value(string):
    try:
        key, value = string.split('=', 1)
        return key.strip(), value.strip()
    except ValueError:
        raise argparse.ArgumentTypeError("Invalid KEY=VALUE format")


def construct_gpu_properties(mako_opts, device_index=0):
    # Initialize NVML
    pynvml.nvmlInit()
    mako_opt_dict = dict()

    try:
        # Get handle for the specified GPU device
        num_gpus = pynvml.nvmlDeviceGetCount()
        handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)

        # Get GPU name
        full_name = pynvml.nvmlDeviceGetName(handle)
        if "Graphics Device" in full_name:
            # Use chip name to get the GPU name
            pci_info = pynvml.nvmlDeviceGetPciInfo(handle)
            if pci_info is None:
                logger.warning(
                    "Failed to get PCI information. PCI parameters and chip name won't be reported"
                )
            else:
                pci_device_id = hex(pci_info.pciDeviceId)
            if pci_device_id is not None and chip_name_mapping:
                entity = str(pci_device_id).lower()
                if entity in chip_name_mapping:
                    chip = str(chip_name_mapping[entity])
                    full_name = "{} BRING-UP BOARD".format(chip.upper())
                else:
                    err_msg = r"Could not find a chip name associated with this device id - {}. Chip name won't be reported".format(
                        entity)
                    logger.warning(err_msg)

        gpu_name = full_name.replace("NVIDIA", "").strip().upper()
        assert gpu_name != "", "device_product_name is empty after removing substring 'NVIDIA' and leading/trailing whitespaces."

        compute_capability = get_compute_capability(device_index)
        gpu_memory = pynvml.nvmlDeviceGetMemoryInfo(handle).total / (1024**2)
        # Gather GPU information
        mako_opt_dict["gpu"] = gpu_name
        mako_opt_dict["gpu_memory"] = gpu_memory

        # Get memory info for this OS
        host_mem_total_mib, host_mem_available_mib = get_host_memory_info()
        mako_opt_dict["host_mem_available_mib"] = host_mem_available_mib
        mako_opt_dict["host_mem_total_mib"] = host_mem_total_mib

        mako_opt_dict = try_set(mako_opt_dict, "hostname", socket.gethostname)
        mako_opt_dict = try_set(mako_opt_dict, "is_aarch64", is_aarch64)
        mako_opt_dict = try_set(mako_opt_dict, "is_linux", is_linux)
        mako_opt_dict = try_set(mako_opt_dict, "is_native_windows",
                                is_native_windows)
        mako_opt_dict = try_set(mako_opt_dict, "is_power", is_power)

        mako_opt_dict = try_set(mako_opt_dict, "is_wsl", is_wsl)
        mako_opt_dict["system_gpu_count"] = num_gpus

        # Set linux specific information
        linux_distribution_info = get_linux_distribution()
        mako_opt_dict["linux_distribution_name"] = linux_distribution_info[0]
        mako_opt_dict["linux_version"] = linux_distribution_info[1]
        mako_opt_dict["linux_codename"] = linux_distribution_info[2]

        if compute_capability is not None:
            mako_opt_dict = try_set(mako_opt_dict, "compute_capability",
                                    lambda: float(compute_capability))
            mako_opt_dict = try_set(
                mako_opt_dict, "supports_tf32",
                lambda: compute_capability and float(compute_capability) >= 8.0)
            mako_opt_dict = try_set(
                mako_opt_dict, "supports_int8",
                lambda: compute_capability and float(compute_capability) >= 6.1)
            mako_opt_dict = try_set(
                mako_opt_dict, "supports_fp8",
                lambda: compute_capability and float(compute_capability) >= 8.9)

        try:
            if is_native_windows():
                sysname, _, _, _, machine, _ = platform.uname()
            else:
                sysname, _, _, _, machine = os.uname()
            mako_opt_dict["cpu"] = machine
            mako_opt_dict["sysname"] = sysname
        except Exception as e:
            logger.warning(
                "Unable to set cpu and sysname mako info: {}".format(e))

        #Set is_android, cuda_version, cudnn_version, and cublas_version to None
        #Set is_remote to False temporarily
        mako_opt_dict["cuda_version"] = None
        mako_opt_dict["cublas_version"] = None
        mako_opt_dict["cudnn_version"] = None
        mako_opt_dict["is_android"] = None
        mako_opt_dict["is_remote"] = False

        if mako_opts:
            mako_opt_dict.update(dict(mako_opts))
        print("Mako options:")
        for key, value in mako_opt_dict.items():
            print(f"{key}={value}")

    finally:
        # Shutdown NVML
        pynvml.nvmlShutdown()


def main():
    parser = argparse.ArgumentParser(
        description="Get arguments when generating system/gpu info")
    parser.add_argument("--device",
                        type=int,
                        default=0,
                        help="Index of device.")
    parser.add_argument(
        "-M",
        "--mako-opt",
        metavar="KEY=VALUE",
        type=parse_key_value,
        action="append",
        help=
        "Specify a KEY=VALUE pair.  Each KEY is assigned to a Mako context variable with value VALUE when "
        "pre-processing the test list provided by the --filter-file "
        "argument.  Multiple --mako-opts arguments may be specified on the command line."
    )
    parser.add_argument("--chip-mapping-file",
                        type=str,
                        required=True,
                        help="Path to the GPU chip mapping JSON file.")
    args = parser.parse_args()
    load_chip_name_mapping(args.chip_mapping_file)
    construct_gpu_properties(args.mako_opt, args.device)


if __name__ == '__main__':
    main()
