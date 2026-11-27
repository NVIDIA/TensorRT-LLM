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
"""The CSA2 request state in the draft steps of a linear chain."""

from __future__ import annotations

from ..csa2 import CSA2Metadata
from .base import DraftMetadata
from .linear_lengths import LinearDraftLengthsMetadata


class CSA2DraftMetadata(DraftMetadata):
    """The CSA2 request state in the draft steps of a linear chain.

    Before each draft step it resolves the positions and owner slots of the
    draft set again, from the KV lengths the step runs with.
    """

    requires = (LinearDraftLengthsMetadata, CSA2Metadata)

    def begin_draft_step(self, step: int) -> None:
        request_state = self.draft_set.get_metadata(CSA2Metadata)
        if request_state._csa2_ready_for_kv_update:
            request_state._refresh_live_endpoints()
