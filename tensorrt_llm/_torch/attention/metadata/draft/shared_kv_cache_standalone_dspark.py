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
"""A standalone DSpark drafter without a draft KV cache of its own."""

from __future__ import annotations

from .base import DraftMetadata


class SharedKVCacheStandaloneDSparkDraftMetadata(DraftMetadata):
    """A standalone DSpark drafter without a draft KV cache of its own.

    The drafter runs once per target step, on the generation rows, with its
    own attention. Its layers are extra layers of the target's KV cache, which
    it addresses through the target's block offsets, or it keeps its context
    in a buffer of the spec worker. Either way it reads nothing that the
    target's metadata do not already hold, so this mode holds nothing.
    """
