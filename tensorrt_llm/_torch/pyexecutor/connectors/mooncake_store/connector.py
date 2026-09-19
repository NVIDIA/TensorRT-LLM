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
"""Placeholders for the connector that moves KV pages in and out of the pool.

The store side stands on its own: a pool can be provisioned, lent memory, and
shared with an engine elsewhere without a connector here. Moving pages needs the
KV cache layout description, so neither class is implemented here.

`CONNECTOR_REGISTRY` names both and `py_executor_creator` resolves them by name,
so they exist to turn `connector: mooncake-store` into a clear refusal instead
of an `AttributeError` raised once the pool is already provisioned.

Neither subclasses `KvCacheConnectorWorker` or `KvCacheConnectorScheduler`,
whose `__init_subclass__` checks a method set there is nothing here to satisfy.
"""

__all__ = ["MooncakeStoreConnectorScheduler", "MooncakeStoreConnectorWorker"]

_UNAVAILABLE = (
    "The mooncake-store KV cache connector is not available in this build, so an "
    "engine here cannot read or write the pool. The pool itself is: "
    "'trtllm-serve mooncake_master' owns one, 'trtllm-serve mooncake_donor' and "
    "the mooncake_donation setting lend it host memory, and an engine that does "
    "have the connector can share it. Drop kv_connector_config.connector to "
    "serve without one."
)


class _Unavailable:
    """Reports what this build can and cannot do with a Mooncake pool."""

    def __init__(self, llm_args):
        del llm_args
        raise NotImplementedError(_UNAVAILABLE)


class MooncakeStoreConnectorWorker(_Unavailable):
    """Moves this rank's KV pages between its cache and the pool."""


class MooncakeStoreConnectorScheduler(_Unavailable):
    """Decides which pages a request loads from the pool and saves to it."""
