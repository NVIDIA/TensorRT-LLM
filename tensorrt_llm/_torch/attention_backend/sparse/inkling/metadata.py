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
"""Attention metadata for Inkling: the base plus one pre-capture hook."""

from ...trtllm import TrtllmAttentionMetadata


class InklingAttentionMetadata(TrtllmAttentionMetadata):
    """``TrtllmAttentionMetadata`` plus the short-conv pool's per-step slot write.

    Carries no fields of its own. The write is a host->device copy into a buffer
    the captured decode graph aliases, so it has to run every step and outside
    that region -- ``prepare()`` is the only hook that does both and still sees
    the CUDA-graph padding rows.
    """

    def prepare(self) -> None:
        super().prepare()
        cache = getattr(self.kv_cache_manager, "conv_state_cache", None)
        if cache is not None and self.request_ids is not None:
            cache.write_state_indices(self.request_ids)
