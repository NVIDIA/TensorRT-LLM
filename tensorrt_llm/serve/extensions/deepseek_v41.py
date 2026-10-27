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
"""Serving extension for DeepSeek-V4.1 checkpoints (``model_type == "deepseek_v41"``).

Prompts are rendered by the ``deepseek_v41`` custom tokenizer, so the only
serving behavior declared here is which token runs mark a generation that has
broken down.
"""

from typing import Any, Optional

from tensorrt_llm.serve.serving_extensions import (
    DegenerateTokens,
    ServingExtension,
    last_vocabulary_id,
    register_serving_extension,
)


@register_serving_extension(model_types=("deepseek_v41", "deepseek_v41_text"))
class DeepseekV41ServingExtension(ServingExtension):
    """DeepSeek-V4.1 serving behavior."""

    def degenerate_tokens(self, hf_config: Any) -> Optional[DegenerateTokens]:
        """``<｜▁pad▁｜>`` and the last vocabulary id, ``<|place_holder_mm_span_0442|>``.

        Neither belongs in generated text: the pad token fills batches, and the
        last vocabulary id is a multimodal placeholder that no prompt uses. The
        pad id comes from the model config, not the tokenizer, whose pad token
        is ``<｜end▁of▁sentence｜>``.
        """
        pad_token_id = getattr(hf_config, "pad_token_id", None)
        if isinstance(pad_token_id, bool) or not isinstance(pad_token_id, int) or pad_token_id < 0:
            pad_token_id = None
        candidates = (pad_token_id, last_vocabulary_id(hf_config))
        token_ids = tuple(dict.fromkeys(i for i in candidates if i is not None))
        return DegenerateTokens(token_ids=token_ids) if token_ids else None
