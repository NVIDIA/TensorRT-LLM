# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""KV-only FP4 MLA generation cache update of the fused query (Triton)."""

from __future__ import annotations

from typing import Any

import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper

from ...kernels.k3r_fp4_mla import fp4_mla_kv_append as _append


class Fp4MlaKvAppend(OpWrapper):
    """``fp4_mla_kv_append_generation``: append this step's latent rows (BF16 ``[T, 576]``) of the
    FP4 MLA layer behind ``mqa`` to its FP4 cache, HP ring, V scales and packed V, and prepare
    (not write) the packed-Q buffers of ``num_q_heads`` heads on ``metadata.fp4_mla_state``."""

    ARCHS = frozenset({Arch.SM_107})
    CELLS: tuple[Cell, ...] = ()

    note = """
    The FP4 MLA backend's fused generation cache update with its query rows switched off
    (`WRITE_Q_ROWS`) and a grid of the KV programs only: the cache bytes are those of the full
    update. The cache pools, page tables, lengths and Q buffers are the backend's, reached through
    `metadata` (the KV cache manager owns their layout); `mqa` supplies the layer index, the RoPE
    table and the local layer of the pools. The fused query (`k3_mla_q_fused`) then writes the
    query rows into the buffers prepared here, on another stream.
    """

    def __call__(
        self, mqa: Any, latent_cache: torch.Tensor, metadata: Any, num_q_heads: int
    ) -> None:
        _append.fp4_mla_kv_append_generation(mqa, latent_cache, metadata, num_q_heads)

    def reference(
        self, mqa: Any, latent_cache: torch.Tensor, metadata: Any, num_q_heads: int
    ) -> None:
        """Not modelled yet: the post-state of the FP4 cache page, HP ring, V scale and packed V
        pools the KV cache manager owns, in their swizzled layouts. Until it is, the entry has no
        cells; the kernel is checked bitwise against the backend's full generation update
        (``test_k3_mla_q_fused.py``)."""
        raise NotImplementedError("Fp4MlaKvAppend.reference: no plain-torch model yet")


fp4_mla_kv_append = Fp4MlaKvAppend()
