# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""The remote-level tests skip a transport this machine lacks; on Grace Blackwell the fabric one must be there."""

import os
import platform
import sys

import pytest

_TEST_DIR = os.path.dirname(os.path.abspath(__file__))
_ADDED_TEST_DIR = _TEST_DIR not in sys.path
if _ADDED_TEST_DIR:
    sys.path.insert(0, _TEST_DIR)
try:
    from remote_test_utils import fabric_available, gpu_count
finally:
    if _ADDED_TEST_DIR:
        sys.path.remove(_TEST_DIR)


def _grace_blackwell() -> bool:
    if platform.machine() != "aarch64" or gpu_count() == 0:
        return False
    import torch

    return torch.cuda.get_device_capability(0)[0] == 10


@pytest.mark.skipif(
    not _grace_blackwell(), reason="the fabric transport is expected on Grace Blackwell only"
)
def test_fabric_transport_is_available_on_grace_blackwell():
    assert fabric_available(), (
        "no fabric transport (driver, IMEX channel or host-NUMA fabric allocation): the remote-level fabric tests "
        "would skip"
    )
