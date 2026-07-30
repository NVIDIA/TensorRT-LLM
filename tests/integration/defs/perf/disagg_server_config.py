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

"""Helpers for generating perf-sanity disaggregated-server configs."""

from typing import Any, Mapping, Optional

_ALLOWED_SCHEDULE_STYLES = frozenset({"context_first", "generation_first"})


def build_disagg_server_config(
    hostname: str,
    port: int,
    num_ctx_servers: int,
    num_gen_servers: int,
    ctx_hostnames: list[str],
    gen_hostnames: list[str],
    server_config_extra: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    """Build the runtime config with an optional schedule-style override."""
    extras = dict(server_config_extra or {})
    unsupported_keys = sorted(set(extras) - {"schedule_style"})
    if unsupported_keys:
        raise ValueError(
            "server_config_extra supports only 'schedule_style'; "
            f"unsupported keys: {unsupported_keys}"
        )
    if "schedule_style" in extras and extras["schedule_style"] not in _ALLOWED_SCHEDULE_STYLES:
        raise ValueError(
            "server_config_extra.schedule_style must be one of "
            f"{sorted(_ALLOWED_SCHEDULE_STYLES)}, got {extras['schedule_style']!r}"
        )

    server_config = {
        "hostname": hostname,
        "port": port,
        "backend": "pytorch",
        "context_servers": {
            "num_instances": num_ctx_servers,
            "urls": ctx_hostnames,
        },
        "generation_servers": {
            "num_instances": num_gen_servers,
            "urls": gen_hostnames,
        },
    }
    server_config.update(extras)
    return server_config
