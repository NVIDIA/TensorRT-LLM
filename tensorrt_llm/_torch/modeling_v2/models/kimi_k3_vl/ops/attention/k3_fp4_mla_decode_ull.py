# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""K3's packed FP4 MLA generation decode (CuTe DSL decode kernel and its split reduction)."""

from __future__ import annotations

from typing import Any

import torch

from tensorrt_llm._torch.modeling_v2.catalog._op import Arch, Cell, OpWrapper

from . import _fp4_mla_decode_ull as _decode


class K3Fp4MlaDecode(OpWrapper):
    """``run_generation``: the absorbed generation attention of the FP4 MLA layer behind ``attn``
    (its generation ``K3MlaAttnState``) over the FP4 cache, for the packed FP4 query rows the
    generation cache update (or ``k3_mla_q_fused``) wrote; ``q`` ``[T, H * 576]`` provides
    shapes. Returns ``[T, H * 512]`` in ``q``'s dtype; ``softmax_stats_tensor`` (Helix) receives
    the FP32 ``[T, H, 2]`` (max, sum) statistics."""

    ARCHS = frozenset({Arch.SM_107})
    CELLS: tuple[Cell, ...] = ()

    note = """
    Per call: the packed split plan (once per step, fused with the generation lengths), the
    device page table (once per step), the packed decode kernel over (split, request) tasks and
    the split reduction. Inside a `DeferredPackedReduce` context the reduction is recorded instead
    of launched (the Helix tail folds the partials); the rows of single-task requests are then the
    only ones written. Plan, partials and maxima live in workspaces on `metadata.fp4_mla_state`;
    `begin_step` forgets the plan of the previous step and `prelaunch_generation_metadata` may
    launch the plan and page table early on another stream (the result does not change).
    `features` fields select FP16 partials, the reduction's PDL launch, the page plan and the
    decode's PDL wait after the fused query (all as 1007804: off, off, on, off); the wave plan and
    8 task slots per request are fixed. Consumes the prequantized query on the state.
    """

    def __call__(
        self,
        attn: Any,
        q: torch.Tensor,
        metadata: Any,
        softmax_stats_tensor: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return _decode.run_generation(
            attn,
            q,
            metadata,
            softmax_stats_tensor=softmax_stats_tensor,
        )

    def reference(
        self,
        attn: Any,
        q: torch.Tensor,
        metadata: Any,
        softmax_stats_tensor: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Not modelled yet: attention of the dequantized prequantized query rows over each
        request's dequantized FP4 cache rows, the HP ring tail and the packed FP4 V copy, all in
        layouts the KV cache manager and ``metadata.fp4_mla_state`` own. Until it is, the entry
        has no cells; the kernel is checked against the FP4 MLA backend's decode
        (``test_k3_fp4_mla_decode.py``)."""
        raise NotImplementedError("K3Fp4MlaDecode.reference: no plain-torch model yet")


k3_fp4_mla_decode = K3Fp4MlaDecode()
