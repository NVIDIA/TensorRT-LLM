# SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Shared utilities for packaging and distribution."""


def get_license_files():
    """Get the list of license files to include in the package based on platform.

    Returns:
        List[str]: Platform-specific license files.
    """
    import sysconfig
    platform_tag = sysconfig.get_platform()
    if "x86_64" in platform_tag:
        return [
            "LICENSE", "ATTRIBUTIONS-CPP-x86_64.md", "ATTRIBUTIONS-Python.md"
        ]
    elif "arm64" in platform_tag or "aarch64" in platform_tag:
        return [
            "LICENSE", "ATTRIBUTIONS-CPP-aarch64.md", "ATTRIBUTIONS-Python.md"
        ]
    else:
        raise RuntimeError(f"Unrecognized CPU architecture: {platform_tag}")
