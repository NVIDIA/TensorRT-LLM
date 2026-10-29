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
"""draft_tail: draft selection tail.

Entry points (``dispatch.get_impl("draft_tail", name)``):

* ``select_draft_tail(worker, draft_model, spec_metadata)`` -> callable or
  None; the callable maps the draft LM-head input rows to int32 draft tokens
  (or None for a step that must stay native).
* ``get_runner(mapping, options=None)`` -> ``DraftTailRunner``.
"""

from ...dispatch import register
from .draft_tail import DraftTailOptions, DraftTailRunner, get_runner, select_draft_tail

register("draft_tail", "select_draft_tail", select_draft_tail)
register("draft_tail", "get_runner", get_runner)

__all__ = ["DraftTailRunner", "DraftTailOptions", "get_runner", "select_draft_tail"]
