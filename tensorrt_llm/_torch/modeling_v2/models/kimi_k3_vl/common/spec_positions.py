# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The speculative-decoding position IDs of a step's first ``num_tokens`` tokens.

:func:`_slice_spec_position_ids` is Modeling V1's
(``tensorrt_llm/_torch/models/modeling_speculative.py``) token for token; only the imports differ.
``test_k3_common.py`` checks it against its source.
"""

from typing import Optional

import torch

__all__ = ["_slice_spec_position_ids"]


def _slice_spec_position_ids(
    position_ids: Optional[torch.Tensor], num_tokens: int
) -> Optional[torch.Tensor]:
    """Slice speculative position IDs along the token dimension."""
    if position_ids is None:
        return None
    return position_ids[..., :num_tokens]
