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
"""Backend-agnostic weight-sharing utilities (MX, GMS, ...)."""

from tensorrt_llm._torch.weight_sharing.source_identity import (
    SOURCE_IDENTITY_FORMAT_VERSION,
    IdentityCheckDecision,
    IdentityCheckPolicy,
    IdentityMatchResult,
    SourceIdentity,
    SourceIdentityMismatchError,
    check_weight_sharing_compatibility,
)

__all__ = [
    "SOURCE_IDENTITY_FORMAT_VERSION",
    "SourceIdentity",
    "IdentityMatchResult",
    "IdentityCheckPolicy",
    "IdentityCheckDecision",
    "SourceIdentityMismatchError",
    "check_weight_sharing_compatibility",
]
