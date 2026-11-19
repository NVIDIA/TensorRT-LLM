# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The DFlash2 drafter's window attention on the TRTLLM-Gen sparse MLA decode kernel.

Not a CuTe DSL kernel: flashinfer's TRTLLM-Gen sparse MLA generation kernel (the one the DSA
targets decode with) run over the drafter's per-query window index lists, every query row as
its own request of one token.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn.functional as F

from tensorrt_llm._torch.modeling_v2.catalog._op import Arch, Cell, OpWrapper

from ..attention.dflash2_window import sparse_mla_rows_eager

#: Kimi K3.1 DFlash2 drafter: kv_lora_rank 512 + rope 64, qk_nope 128, window 1024,
#: softmax scale mscale(32, 1)^2 / sqrt(192).
_LATENT = 512
_ROPE = 64
_NOPE = 128
_WINDOW = 1024
_SOFTMAX_SCALE = 0.13086080


def padded_heads(num_heads: int) -> int:
    """The head count the TRTLLM-Gen MLA decode runs: it has no kernel for 64 < H < 128."""
    return 128 if 64 < num_heads < 128 else num_heads


def dflash2_window_attention_buffers(
    max_rows: int, num_heads: int, query_dtype: torch.dtype, device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor]:
    """``(workspace, counter)`` for up to ``max_rows`` query rows: allocate once, before capture.

    Both are zero-initialized; the kernel leaves its multi-CTA counters at zero after every
    launch, so they are reused as they are.
    """
    from flashinfer.utils import get_trtllm_gen_multi_ctas_kv_counter_bytes

    from tensorrt_llm.bindings.internal import thop

    heads = padded_heads(num_heads)
    # Arguments: dtype, batch, tokens, heads, head size, rotary dim, KV heads.
    size = max(
        int(
            thop.get_trtllm_gen_generation_workspace_layout(
                query_dtype, max_rows, max_rows, heads, d, r, 1
            )["trtllm_gen_workspace_size"]
        )
        for d, r in ((_LATENT, 0), (_LATENT + _ROPE, _ROPE))
    )
    sm_count = torch.cuda.get_device_properties(device).multi_processor_count
    counter = get_trtllm_gen_multi_ctas_kv_counter_bytes(max_rows, heads, sm_count)
    return (
        torch.zeros(size, dtype=torch.uint8, device=device),
        torch.zeros(counter, dtype=torch.uint8, device=device),
    )


def _spec(batch: int, width: int, heads: int, **overrides) -> dict:
    spec = dict(
        batch=batch,
        width=width,
        heads=heads,
        latent=_LATENT,
        rope=_ROPE,
        window=_WINDOW,
        scale=_SOFTMAX_SCALE,
        page_size=64,
        layers=1,
    )
    spec.update(overrides)
    return spec


class K31DFlash2WindowAttention(OpWrapper):
    """Absorbed MLA of each query row over its window rows of the paged latent pool.

    ``out[r, h] = softmax_k(scale * query[r, h] . kv[k]) @ kv[k, :latent_dim]`` over the rows
    ``k`` of ``rows[r]`` (``-1`` past ``lengths[r]``), ``kv`` the pool's latent rows. Returns
    ``[R, H, latent_dim]`` in the query's dtype.

    ``pool`` is one layer's ``[pages, page_size, D]`` view (its pages may be strided, as the KV
    cache manager's layer-interleaved pool is); ``rows`` hold global rows
    ``page * page_size + offset`` (``dflash2_window_rows``). ``workspace`` and ``counter`` come
    from ``dflash2_window_attention_buffers`` sized for at least ``R`` rows.
    """

    ARCHS = frozenset({Arch.SM_103})

    CELLS: tuple[Cell, ...] = (
        Cell(why="c1, width 5, TP1: 96 heads (run padded to 128)", spec=_spec(1, 5, 96)),
        Cell(why="8 requests at width 8, TP2: 48 heads", spec=_spec(8, 8, 48)),
        Cell(why="64 requests at width 8, TP4: 24 heads, 512 query rows", spec=_spec(64, 8, 24)),
        Cell(why="16 requests at width 5, TP8: 12 heads", spec=_spec(16, 5, 12)),
        Cell(
            why="4 requests at width 8, TP16: 6 heads, 32-token pages",
            spec=_spec(4, 8, 6, page_size=32),
        ),
        Cell(why="2 requests at width 5, TP32: 3 heads", spec=_spec(2, 5, 3)),
        Cell(
            why="the pool's pages strided by 5 layers (the KV cache manager's layout)",
            spec=_spec(4, 8, 96, layers=5),
        ),
    )

    note = """
    Not CuTe DSL and not bitwise: the TRTLLM-Gen kernel (flashinfer, `backend="trtllm-gen"`)
    multiplies BF16 Q and K into FP32 scores, rounds the probabilities to BF16 for P.V and the
    output to BF16, where the reference (`sparse_mla_rows_eager`) stays in FP32 until the
    output. The test bounds the difference. Whether that moves the drafter's acceptance
    length has to be checked end to end before the call site switches.

    Heads per rank: 96 / TP. The kernel has no 64 < H < 128 variant, so 96 heads run as 128
    with zero queries in the padding heads (one extra pad launch; a caller that allocates its
    query with 128 heads avoids it). Workspace and counters are the caller's
    (`dflash2_window_attention_buffers`, allocated before CUDA-graph capture).
    """

    def __call__(
        self,
        query: torch.Tensor,
        pool: torch.Tensor,
        rows: torch.Tensor,
        lengths: torch.Tensor,
        softmax_scale: float,
        latent_dim: int,
        workspace: torch.Tensor,
        counter: torch.Tensor,
    ) -> torch.Tensor:
        from flashinfer.mla import trtllm_batch_decode_with_kv_cache_mla

        num_heads, head_dim = query.shape[1], query.shape[2]
        q = query
        heads = padded_heads(num_heads)
        if heads != num_heads:
            q = F.pad(q, (0, 0, 0, heads - num_heads))
        out = trtllm_batch_decode_with_kv_cache_mla(
            query=q.unsqueeze(1),
            kv_cache=pool.unsqueeze(1),
            workspace_buffer=workspace,
            qk_nope_head_dim=_NOPE,
            kv_lora_rank=latent_dim,
            qk_rope_head_dim=head_dim - latent_dim,
            block_tables=rows.unsqueeze(1),
            seq_lens=lengths,
            max_seq_len=rows.shape[1],
            sparse_mla_top_k=rows.shape[1],
            bmm1_scale=softmax_scale,
            bmm2_scale=1.0,
            backend="trtllm-gen",
            multi_ctas_kv_counter_buffer=counter,
            use_fp16_softmax=None,
        ).squeeze(1)
        return out[:, :num_heads] if heads != num_heads else out

    def reference(
        self,
        query: torch.Tensor,
        pool: torch.Tensor,
        rows: torch.Tensor,
        lengths: torch.Tensor,
        softmax_scale: float,
        latent_dim: int,
        workspace: torch.Tensor,
        counter: torch.Tensor,
    ) -> torch.Tensor:
        """The drafter's eager FP32 decode (``lengths``, ``workspace``, ``counter`` unused)."""
        pool_rows = pool.reshape(-1, pool.shape[-1])
        return sparse_mla_rows_eager(query, pool_rows, rows, softmax_scale, latent_dim)

    def is_valid(
        self,
        query: torch.Tensor,
        pool: torch.Tensor,
        rows: torch.Tensor,
        lengths: torch.Tensor,
        softmax_scale: float,
        latent_dim: int,
        workspace: torch.Tensor,
        counter: torch.Tensor,
    ) -> None:
        assert query.dtype == torch.bfloat16 and query.dim() == 3, "BF16 [R, H, D] query"
        assert pool.dtype == torch.bfloat16 and pool.dim() == 3
        assert pool.stride(-1) == 1 and pool.stride(-2) == pool.shape[-1], (
            "each page's rows must be contiguous"
        )
        assert query.shape[-1] == pool.shape[-1] == latent_dim + _ROPE and latent_dim == _LATENT
        assert rows.dtype == torch.int32 and lengths.dtype == torch.int32
        assert rows.shape[0] == query.shape[0] == lengths.shape[0], "one list per query row"
        # A list must hold its live entries first: the kernel reads the first lengths[r].
        assert (lengths >= 1).all() and (lengths <= rows.shape[1]).all(), "lengths in [1, K]"


dflash2_window_attention = K31DFlash2WindowAttention()
