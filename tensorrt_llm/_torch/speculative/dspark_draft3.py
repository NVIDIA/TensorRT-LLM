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
"""Fewer launches on the MLA DSpark drafter's critical path (helix drafter, FP8 draft pool).

Each piece falls back to the current path where it does not apply:

* on drafting Kimi K3.1 engines without attention DP (:func:`draft3_glue_enabled`): per-step glue of the drafter
  forward off the critical path, same values: the split-KV plan of the block decode on a side stream (joined before
  the first decode), the shape-only scratch indices built once per shape, no copy of the first layer's residual, no
  unused dtype conversion of the slots, the drafter LM head reading a contiguous row range of the block output in
  place of a gather.
"""

from ..pyexecutor.input_commit import k31_draft_tp_engine


def draft3_glue_enabled() -> bool:
    """Whether the drafter forward takes its per-step glue off the critical path (same values)."""
    return k31_draft_tp_engine()
