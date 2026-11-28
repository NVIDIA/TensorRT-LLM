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
"""Rendezvous store keys and address for Wide-EP fault tolerance."""

import os

from tensorrt_llm._wide_ep_fault_tolerance import RENDEZVOUS_ENV

LIVENESS_KEY = "liveness"
RECORD_KEY = "record"
TRANSACTION_KEY = "last_transaction"
# The Unix time in milliseconds since which the active ranks have waited on a rank that makes no
# progress, or empty once no rank stalls them. It is unset until the first stall, so read it after
# checking that it is set.
STALLED_SINCE_KEY = "stalled_since"


def rendezvous_address() -> str:
    """Return the rendezvous address ``host:port`` that ``RENDEZVOUS_ENV`` holds.

    Raises:
        ValueError: ``RENDEZVOUS_ENV`` is not set.
    """
    address = os.environ.get(RENDEZVOUS_ENV)
    if not address:
        raise ValueError(f"Wide-EP fault tolerance requires host:port in {RENDEZVOUS_ENV}")
    return address


def rendezvous_host_port(address: str) -> tuple[str, int]:
    """Split a rendezvous address ``host:port`` into its host and port."""
    host, port = address.rsplit(":", 1)
    return host, int(port)
