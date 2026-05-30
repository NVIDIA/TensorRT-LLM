# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Flat-item multimodal encoding plan.

Per-model code supplies (a) an extractor that walks each :class:`MultimodalParams`
and yields :class:`MultimodalItem` instances, and (b) per-modality encoder adapters
that bridge a bucket of items to the model's existing encoder call. This module
owns partition, index-tensor build, and per-modality scatter assembly.
"""

from __future__ import annotations

import torch  # noqa: F401

from tensorrt_llm.inputs.multimodal import MMItemOrder, MultimodalParams  # noqa: F401
