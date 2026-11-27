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
"""DeepSeek-V4's sparse MLA state in the draft steps of a linear chain."""

from __future__ import annotations

from ..deepseek_v4 import DeepseekV4Metadata
from .base import DraftMetadata
from .linear_lengths import LinearDraftLengthsMetadata


class DeepseekV4DraftMetadata(DraftMetadata):
    """DeepSeek-V4's sparse MLA state in the draft steps of a linear chain.

    Before each draft step it derives the sparse state of the draft set again,
    from the rows and lengths the step runs with.
    """

    requires = (LinearDraftLengthsMetadata, DeepseekV4Metadata)

    def begin_draft_step(self, step: int) -> None:
        self.draft_set.get_metadata(DeepseekV4Metadata).derive()
