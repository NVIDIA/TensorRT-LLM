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
"""draft_front: MTP draft front-end normalization and TP-local packing (the draft_front contract).

Registered entry points (``dispatch.get_impl("draft_front", name)``):

* ``draft_front``: :func:`pack_draft_front`, returns the TP-local ``eh_proj``
  input or ``None`` to request the native path.
* ``unsupported_reason``: :func:`unsupported_reason`, the host-side
  support-domain check used by ``draft_front``.
* ``native_norm_reason``: :func:`native_norm_reason`, for the caller that owns
  the ``enorm``/``hnorm`` modules: ``None`` when they run the FlashInfer CuTe DSL
  RMSNorm that draft_front mirrors, else why the native draft front must stay.
"""

from ... import dispatch
from .draft_front import Segment, pack_draft_front, plan_segments, unsupported_reason
from .native_norm import native_norm_reason

dispatch.register("draft_front", "draft_front", pack_draft_front)
dispatch.register("draft_front", "unsupported_reason", unsupported_reason)
dispatch.register("draft_front", "native_norm_reason", native_norm_reason)

__all__ = [
    "Segment",
    "native_norm_reason",
    "pack_draft_front",
    "plan_segments",
    "unsupported_reason",
]
