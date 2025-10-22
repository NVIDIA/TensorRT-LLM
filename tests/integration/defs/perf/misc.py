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
# -*- coding: utf-8 -*-
"""
Miscellaneous utility code used by trt_test. Should contain all code that is agnostic to remote mode vs local mode.
"""
import subprocess as sp

_GPU_DEVICE_PRODUCT_NAME_MAPPING = {"A100-PCIE-80GB": "A100 80GB PCIe"}


def get_device_subtype(device_product_name: str) -> str:
    """
    Get device subtype based on device product name.

    Simply converts the cleaned device product name to a consistent format
    by replacing spaces and hyphens with underscores.

    Args:
        device_product_name: Cleaned device product name from NVML

    Returns:
        Device subtype string with consistent formatting
    """
    # Convert device name to consistent subtype format (replace spaces and hyphens with underscores)
    return device_product_name.replace(" ", "_").replace("-", "_")


def clean_device_product_name(device_product_name):
    cleaned_name = device_product_name
    cleaned_name = cleaned_name.replace("NVIDIA", "").strip()
    assert cleaned_name != "", "device_product_name is empty after removing substring 'NVIDIA' and leading/trailing whitespaces."

    # Match name reported by older and newer nrsu versions:
    # Old: Jetson AGX Orin Developer Kit
    # New: jetson-agx-orin-developer-kit
    #
    if "jetson" in cleaned_name.lower():
        cleaned_name = cleaned_name.lower().replace(" ", "-")

    if cleaned_name in _GPU_DEVICE_PRODUCT_NAME_MAPPING:
        return _GPU_DEVICE_PRODUCT_NAME_MAPPING[cleaned_name]

    return cleaned_name


def check_output(command: list) -> str:
    """
    Executes a command and returns its output.
    """
    result = sp.run(command, stdout=sp.PIPE, stderr=sp.PIPE)
    if result.returncode != 0:
        raise sp.CalledProcessError(result.returncode,
                                    command,
                                    output=result.stdout,
                                    stderr=result.stderr)
    return result.stdout.decode('utf-8')
