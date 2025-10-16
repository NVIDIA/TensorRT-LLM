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

# Mapping of PCI device IDs to device subtypes
# This allows distinguishing between different variants of the same GPU family
_PCI_DEVICE_ID_TO_SUBTYPE = {
    # H100 variants
    0x2330: "H100_SXM",
    0x2331: "H100_PCIe",
    0x2339: "H100_NVL",
    # A100 variants
    0x20B0: "A100_SXM4_40GB",
    0x20B1: "A100_PCIe_40GB",
    0x20B2: "A100_SXM4_80GB",
    0x20B3: "A100_PCIe_80GB",
    0x20B5: "A100_SXM4_80GB",
    0x20F0: "A100X",
    0x20F1: "A100_PCIe_80GB",
    0x20F3: "A100_PCIe_40GB",
    0x20F5: "A100_SXM4_80GB",
    # L40S variants
    0x26B1: "L40S",
    0x26B2: "L40",
    0x26B5: "L40S",
    # Add more mappings as needed for other GPU families
}


def get_device_subtype(pci_device_id: int, device_product_name: str) -> str:
    """
    Get device subtype based on PCI device ID and product name.

    Args:
        pci_device_id: PCI device ID from NVML
        device_product_name: Cleaned device product name

    Returns:
        Device subtype string, or cleaned product name if no specific subtype found
    """
    # First try to get subtype from PCI device ID mapping
    if pci_device_id in _PCI_DEVICE_ID_TO_SUBTYPE:
        return _PCI_DEVICE_ID_TO_SUBTYPE[pci_device_id]

    # Fallback to using cleaned product name
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
