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
"""Drafting from a suffix automaton over the generated tokens."""

from __future__ import annotations

from .base import DraftMetadata


class SuffixAutomatonDraftMetadata(DraftMetadata):
    """Drafting from a suffix automaton over the tokens generated so far.

    The draft is a pattern match, not a model forward: it reads no attention
    metadata and has no draft KV cache, so this mode holds nothing and its
    hooks do nothing.
    """
