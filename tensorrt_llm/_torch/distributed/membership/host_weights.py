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
"""Names of the EPLB host weights that a Wide-EP deployment keeps in /dev/shm."""

import os

from .signed_store import shared_secret_digest


def host_weights_prefix() -> str:
    """Return the /dev/shm name prefix that every deployment's EPLB host weights share."""
    return f"{os.environ.get('TRTLLM_EPLB_SHM_NAME', 'moe_shared')}_"


def host_weights_name(node: int) -> str:
    """Return the /dev/shm base name of this deployment's EPLB host weights on ``node``.

    The node index keeps the segments of a node supervisor started on another node's host apart
    from that node's own.
    """
    digest = shared_secret_digest(b"eplb host weights")[:16]
    return f"{host_weights_prefix()}{digest}_node{node}"
