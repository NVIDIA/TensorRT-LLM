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
"""Opaque completed-prefix references; addresses are resolved by the engine."""

import re
from collections.abc import Mapping
from dataclasses import fields
from typing import Annotated

from pydantic import ConfigDict, Field, StrictInt, StrictStr
from pydantic.dataclasses import dataclass


@dataclass(frozen=True, slots=True, config=ConfigDict(extra="forbid"))
class RetainedPrefixParams:
    version: Annotated[StrictInt, Field(ge=1, le=1)]
    peer_id: Annotated[StrictStr, Field(pattern=r"^[A-Za-z0-9_.-]{1,128}$")]
    engine_generation: Annotated[StrictStr, Field(pattern=r"^[0-9a-f]{32}$")]
    export_id: Annotated[StrictStr, Field(pattern=r"^[0-9a-f]{64}$")]
    prefix_tokens: Annotated[StrictInt, Field(gt=0)]
    identity: Annotated[StrictStr, Field(pattern=r"^[0-9a-f]{64}$")]


def parse_retained_prefix(value) -> RetainedPrefixParams:
    """Validate/copy wire metadata without granting source authority.

    The configured provider independently verifies the opaque export and its
    exact content identity with the trusted source before native contact.
    """
    if isinstance(value, RetainedPrefixParams):
        value = {field.name: getattr(value, field.name) for field in fields(value)}
    if not isinstance(value, Mapping):
        raise ValueError("retained_prefix must be an object")
    expected = {field.name for field in fields(RetainedPrefixParams)}
    if set(value) != expected:
        raise ValueError("retained_prefix has missing or unknown fields")
    if type(value["version"]) is not int or value["version"] != 1:
        raise ValueError("unsupported retained_prefix version")
    if type(value["prefix_tokens"]) is not int or value["prefix_tokens"] <= 0:
        raise ValueError("retained_prefix requires a positive token boundary")
    peer_id = value["peer_id"]
    if not isinstance(peer_id, str) or not re.fullmatch(r"[A-Za-z0-9_.-]{1,128}", peer_id):
        raise ValueError("retained_prefix requires a configured peer alias")
    for key, length in (("engine_generation", 32), ("export_id", 64), ("identity", 64)):
        if not isinstance(value[key], str) or not re.fullmatch(
            rf"[0-9a-f]{{{length}}}", value[key]
        ):
            raise ValueError(f"invalid retained_prefix {key}")
    return RetainedPrefixParams(**dict(value))
