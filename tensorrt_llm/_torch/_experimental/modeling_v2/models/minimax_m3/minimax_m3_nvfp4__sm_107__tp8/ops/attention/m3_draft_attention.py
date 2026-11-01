# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The Eagle3 drafter layer's attention on the TRTLLM backend: RoPE, KV-cache append and causal attention."""

from __future__ import annotations

import torch as torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell
from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import OpWrapper as OpWrapper
from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.ops.attention.draft_attention import (
    GroupedDraftAttentionBase,
)
from tensorrt_llm._torch.attention.backends.interface import (
    AttentionForwardArgs as AttentionForwardArgs,
)
from tensorrt_llm._torch.attention.backends.interface import (
    PredefinedAttentionMask as PredefinedAttentionMask,
)
from tensorrt_llm.functional import PositionEmbeddingType as PositionEmbeddingType
from tensorrt_llm.functional import RotaryScalingType as RotaryScalingType


class M3DraftAttention(GroupedDraftAttentionBase):
    """The drafter layer's attention over its fused QKV rows ``qkv`` [T, (q + 2 kv) heads x 128] (bf16): the
    backend applies the drafter's RoPE (fused), appends K / V to the drafter layer's paged cache (FP8, the target's
    NVFP4 cache keeps its draft layer in FP8) and runs causal attention over the step's live rows (``metadata``
    ``num_tokens``); returns [num_tokens, q heads x 128] bf16.

    ``backend`` (``TrtllmAttention``) and ``metadata`` are the runtime's objects for the drafter layer and step.
    """

    ARCHS = frozenset({Arch.SM_107})

    CELLS: tuple[Cell, ...] = (
        Cell(why="first draft pass of a 2-request verify (2 x 4 rows)", spec=dict(reqs=2, qlen=4)),
        Cell(why="a later draft pass, one row per request", spec=dict(reqs=2, qlen=1)),
        Cell(why="a context chunk's draft rows", spec=dict(reqs=1, context=512)),
    )

    note = """
    `reference` is exact (float64) causal softmax attention: Q rotated by the backend's NeoX RoPE at each token's
    position, over the drafter layer's K / V rows read back from its FP8 cache (dequantized by the backend's
    ``kv_scale_quant_orig``). What remains between the two is the FMHA's own arithmetic (Q's rounding to the
    kernel's input precision, its P rounding and accumulation order), which the test bounds.
    """


m3_draft_attention = M3DraftAttention()
