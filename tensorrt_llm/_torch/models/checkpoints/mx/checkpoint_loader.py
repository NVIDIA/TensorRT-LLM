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

"""Compatibility shim for the ModelExpress TRT-LLM checkpoint loader."""

from __future__ import annotations

from typing import Any

from tensorrt_llm._torch.models.modeling_utils import register_checkpoint_loader


@register_checkpoint_loader("modelexpress")
class MXCheckpointLoader:
    """Instantiate the ModelExpress-owned TRT-LLM checkpoint loader."""

    def __new__(cls, *args: Any, **kwargs: Any):
        try:
            from modelexpress.engines.trtllm.loader import MXCheckpointLoader as _MXCheckpointLoader
        except ImportError as exc:
            raise ImportError(
                "checkpoint_format='modelexpress' requires the modelexpress Python "
                "package with TRT-LLM support installed."
            ) from exc

        return _MXCheckpointLoader(*args, **kwargs)


__all__ = ["MXCheckpointLoader"]
