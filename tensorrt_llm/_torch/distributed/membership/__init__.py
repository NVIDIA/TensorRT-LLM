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
"""Membership for Wide-EP fault tolerance.

The authority process decides which ranks are active and publishes each decision as a membership
record. Every rank keeps a liveness connection to it, and the ranks exchange values through the
rendezvous store it hosts.
"""

from tensorrt_llm._wide_ep_fault_tolerance import RENDEZVOUS_ENV, SHARED_SECRET_ENV

from ..startup_record import REPLACEMENT_CANNOT_START, ReplacementCannotStart
from .authority import Authority
from .authority_process import AuthorityProcess
from .liveness import (
    MembershipClient,
    open_liveness_connection,
    open_node_supervisor_connection,
    open_store,
    rendezvous_store,
)
from .pause_word import open_pause_word, pause_word_path, read_pause_word
from .records import (
    AUTHORITY_SILENCE_TIMEOUT_S,
    DEAD_TIMEOUT_S,
    HANG_TIMEOUT_S,
    HELLO_TIMEOUT_S,
    RENDEZVOUS_TIMEOUT_S,
    REPORT_INTERVAL_S,
    MembershipRecord,
    MembershipState,
    NodeReport,
    NodeView,
    RankReport,
    Reply,
    ResumeState,
    TransactionSummary,
    all_ranks,
    has_rank,
    latest_epoch,
    next_epoch,
    rank_env,
    rank_mask,
    remove_cpu_affinity_constraints,
)
from .rendezvous import (
    LIVENESS_KEY,
    RECORD_KEY,
    STALLED_SINCE_KEY,
    TRANSACTION_KEY,
    StoreGroup,
    rendezvous_host_port,
    wait_for_keys,
)
from .signed_store import SignedStore, UnsignedValueError, parse_shared_secret

__all__ = [
    "AUTHORITY_SILENCE_TIMEOUT_S",
    "Authority",
    "AuthorityProcess",
    "DEAD_TIMEOUT_S",
    "HANG_TIMEOUT_S",
    "HELLO_TIMEOUT_S",
    "LIVENESS_KEY",
    "MembershipClient",
    "MembershipRecord",
    "MembershipState",
    "NodeReport",
    "NodeView",
    "RECORD_KEY",
    "RENDEZVOUS_ENV",
    "RENDEZVOUS_TIMEOUT_S",
    "REPLACEMENT_CANNOT_START",
    "REPORT_INTERVAL_S",
    "RankReport",
    "ReplacementCannotStart",
    "Reply",
    "ResumeState",
    "SHARED_SECRET_ENV",
    "STALLED_SINCE_KEY",
    "SignedStore",
    "StoreGroup",
    "TRANSACTION_KEY",
    "TransactionSummary",
    "UnsignedValueError",
    "all_ranks",
    "has_rank",
    "latest_epoch",
    "next_epoch",
    "open_liveness_connection",
    "open_node_supervisor_connection",
    "open_pause_word",
    "open_store",
    "parse_shared_secret",
    "pause_word_path",
    "rank_env",
    "rank_mask",
    "read_pause_word",
    "remove_cpu_affinity_constraints",
    "rendezvous_host_port",
    "rendezvous_store",
    "wait_for_keys",
]
