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
"""Shared WideEP fault-tolerance options for MoE communication paths."""

from __future__ import annotations

import os
from typing import Any, Optional

from tensorrt_llm._torch.alltoall_watchdog import (
    DEFAULT_ALLTOALL_WATCHDOG_POLL_INTERVAL_S,
    DEFAULT_ALLTOALL_WATCHDOG_TIMEOUT_S,
)

from .ep_group_health import EPGroupHealth

_ENABLE_ENV = "TRTLLM_ENABLE_WIDE_EP_FT"
_TIMEOUT_ENV = "TRTLLM_ALLTOALL_WATCHDOG_TIMEOUT_S"
_POLL_INTERVAL_ENV = "TRTLLM_ALLTOALL_WATCHDOG_POLL_INTERVAL_S"

_HEALTH_KEY = "wide_ep_ft_ep_group_health"
_TIMEOUT_KEY = "alltoall_watchdog_timeout_s"
_POLL_INTERVAL_KEY = "alltoall_watchdog_poll_interval_s"


def _env_enabled() -> bool:
    return os.environ.get(_ENABLE_ENV, "0").lower() in {"1", "true", "yes", "on"}


def _float_option(extra_attrs: dict, key: str, env_name: str, default: float) -> float:
    if key in extra_attrs:
        return float(extra_attrs[key])
    if env_name in os.environ:
        return float(os.environ[env_name])
    return default


def get_wide_ep_ft_options(
    model_config: Any,
) -> tuple[Optional[EPGroupHealth], Optional[float], float]:
    """Return the shared EP health object and watchdog timing for a model.

    WideEP FT remains opt-in until the integration PR wires a public model
    option.  Callers can either inject ``wide_ep_ft_ep_group_health`` through
    ``ModelConfig.extra_attrs`` or set ``TRTLLM_ENABLE_WIDE_EP_FT=1`` to create
    one process-local health object shared by all MoE communication layers.
    """

    extra_attrs = getattr(model_config, "extra_attrs", {})
    health = extra_attrs.get(_HEALTH_KEY) or extra_attrs.get("ep_group_health")
    if health is None and _env_enabled():
        health = EPGroupHealth(model_config.mapping.moe_ep_size)
        extra_attrs[_HEALTH_KEY] = health

    poll_interval_s = _float_option(
        extra_attrs,
        _POLL_INTERVAL_KEY,
        _POLL_INTERVAL_ENV,
        DEFAULT_ALLTOALL_WATCHDOG_POLL_INTERVAL_S,
    )
    if health is None:
        return None, None, poll_interval_s

    timeout_s = _float_option(
        extra_attrs,
        _TIMEOUT_KEY,
        _TIMEOUT_ENV,
        DEFAULT_ALLTOALL_WATCHDOG_TIMEOUT_S,
    )
    return health, timeout_s, poll_interval_s
