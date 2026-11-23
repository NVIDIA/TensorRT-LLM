# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Sparse MLA over the latent cache: FlashInfer's TRTLLM-GEN MLA decode kernel, rope-64 geometry.

Every query row ``r`` (a decode, verify or compacted context row alike) is its own sequence of
one token: it attends, with all its heads, to the latent rows ``packed[r, :lengths[r]]`` of the
cache, which serve as both keys and values. A row is ``[ckv 512 | kpe 64]``: the scores are
``q . k * bmm1_scale`` over all 576 columns, the softmax runs in FP32, and the output
``softmax @ v[..., :512] * bmm2_scale`` is BF16 and 512 wide (the absorbed value projection to
``v_head_dim`` 256 is the caller's ``bmm_out``). For an FP8 cache with dequantization scale
``s`` and the model's softmax scale, ``bmm1_scale = s * softmax_scale`` and ``bmm2_scale = s``.

Adapted from glm5_next's target entry
(``models/glm5_next/glm_5_3_flash_fp8__sm_10x__tp4/ops/attention/sparse_mla_decode.py``), which
wraps the same FlashInfer callable for GLM-5.3's native NoPE geometry (``qk_rope_head_dim=0``,
one 512-wide cache row, per-token top-k lengths). This copy is Hy4-Flash's geometry -- 192
no-PE + 64 rope columns, 512-wide latent values, 4 and 8 heads per rank -- certified on sm_103.

Buffers the caller owns and passes on every call:

- ``workspace``: uint8, at least ``WORKSPACE_BYTES``, zeroed before its first use. The kernel
  keeps the partial results of a row split across CTAs there.
- ``counter``: uint8, contiguous, zeroed once, at least ``counter_bytes(rows, heads, sm_count)``
  for the call's rows and heads. The kernel resets its semaphores after every launch, so
  ordered launches on one stream (and replays of one CUDA graph) share it; launches that can
  run concurrently each need their own.

A CUDA-graph capture needs both buffers allocated before it starts.
"""

from __future__ import annotations

import torch
from flashinfer.mla import trtllm_batch_decode_with_kv_cache_mla
from flashinfer.utils import get_trtllm_gen_multi_ctas_kv_counter_bytes

from tensorrt_llm._torch.flashinfer_utils import get_env_enable_pdl
from tensorrt_llm._torch.modeling_v2.catalog._op import Arch, Cell, OpWrapper, true_fp32_matmul

#: Workspace bytes the kernel needs: 32 MiB, the TRTLLM-GEN share of TensorRT-LLM's generation
#: attention workspace.
WORKSPACE_BYTES = 32 * 1024 * 1024

#: Hy4-Flash's absorbed MLA geometry (config: qk_nope_head_dim 192, qk_rope_head_dim 64,
#: kv_lora_rank 512), the latent cache read as 32-row pages (the index pool's page).
QK_NOPE_HEAD_DIM = 192
QK_ROPE_HEAD_DIM = 64
KV_LORA_RANK = 512
PAGE_ROWS = 32

#: One query row's absorbed width: the latent columns the scores read plus the rope columns.
QUERY_DIM = KV_LORA_RANK + QK_ROPE_HEAD_DIM

#: Heads one rank attends with: 32 heads / tp4, and 32 / tp8.
HEADS_PER_RANK = (4, 8)


def counter_bytes(rows: int, num_heads: int, sm_count: int) -> int:
    """Bytes of multi-CTA semaphores a call of ``rows`` query rows of ``num_heads`` heads needs:
    one 4-byte counter per (row, head), at least one per SM, rounded up to 8 counters."""
    return get_trtllm_gen_multi_ctas_kv_counter_bytes(rows, num_heads, sm_count)


class SparseMlaDecode(OpWrapper):
    """``query`` ``[R, H, 576]`` over ``kv_rows`` ``[N, 576]`` -> BF16 ``[R, H, 512]``.

    ``query`` and ``kv_rows`` are both BF16 (the core cache) or both E4M3 (the kv_fp8 variant).
    ``packed`` int32 ``[R, W]`` holds row ``r``'s latent row ids (rows of the flat latent pool,
    not page ids) in its first ``lengths[r]`` columns and ``-1`` after them; ``lengths`` int32
    ``[R]`` is at least 1 and is the row's visible-token count, which the kernel clamps to ``W``.
    ``bmm1_scale`` and ``bmm2_scale`` are Python floats (fixed in a captured graph) or FP32 device
    tensors of one element.
    """

    ARCHS = frozenset({Arch.SM_103})

    # Hy4-Flash's sparse call sites: one decode row per request, the 6-row verify block, and up
    # to 5120 compacted context rows. A rank attends with 8 heads at tp4 and 4 at tp8; every
    # row selects up to 2048 latent rows of the pool.
    CELLS: tuple[Cell, ...] = (
        Cell(
            why="single-token decode of one request, 8 heads (tp4)",
            spec=dict(heads=8, rows=1, dtype="bf16"),
        ),
        Cell(
            why="single-token decode of one request, 4 heads (tp8)",
            spec=dict(heads=4, rows=1, dtype="bf16"),
        ),
        Cell(
            why="six-row verify block of one request, 8 heads: five drafts plus the anchor",
            spec=dict(heads=8, rows=6, dtype="bf16"),
        ),
        Cell(
            why="5120 compacted context rows, 8 heads: the prefill envelope's largest row count",
            spec=dict(heads=8, rows=5120, dtype="bf16"),
        ),
        Cell(
            why="single-token decode with the FP8 latent cache (kv_fp8 variant), 8 heads",
            spec=dict(heads=8, rows=1, dtype="e4m3"),
        ),
    )

    note = """
    The rope-64 path takes no per-token top-k length argument: FlashInfer refuses
    `sparse_mla_top_k_lens` for `qk_rope_head_dim != 0` (only the native NoPE geometry uses
    it). A row's visible-token count travels as `seq_lens` instead, and the kernel reads the
    first `min(lengths[r], W)` columns of that row's selection: measured on sm_103 (GB300),
    a row whose list holds `n` valid ids and then `-1` padding matches FP32 attention over
    exactly its first `n` columns when `lengths[r]` is `n` or its physical token count, and
    is wrong (relative RMS 0.32 against 0.0019) when `lengths[r]` is longer than `n`. The
    selection is a count, not a position bound: ids far above `lengths[r]` are still read
    (measured: 8 ids from [1000, 1100) with `lengths[r]` 8, relative RMS 0.0021). With a
    multi-row query `[B, q_len, H, 576]` the kernel derives row `j`'s count as
    `lengths[b] - q_len + 1 + j` -- the dense causal bound used as a count -- which for a
    verify block ending at `lengths[b]` is each row's own visible count; this entry passes one
    row per sequence (`q_len` 1), so every row carries its own. `is_valid` refuses a zero
    length and a length whose first `min(length, W)` columns are not all rows of the cache.

    The FP8 kernel rounds the softmax probabilities to E4M3 for the value product, so an FP8
    output moves by a few percent of its own scale against FP32 attention; the test derives its
    band from that rounding. The core path is BF16 in and BF16 out with FP32 softmax and
    accumulation.

    Measured against FP32 attention on sm_103 (GB300), relative RMS: 2.6e-3 (one decode row, 8
    heads), 2.5e-3 (4 heads), 5.1e-4 (six-row verify block), 1.8e-3 (5120 context rows), 1.2e-2
    (FP8 cache); cosine at least 0.999928 throughout. Judged against a different selection of
    the same width the error is 0.87, 358x the correct one's -- the test's band is 3x the
    worst measured core cell and 5x the FP8 one.

    The query is `[R, H, 576]`; `kv_rows` is the flat row view of the latent pool (row ids are
    pool rows, the kernel splits them into 32-row pages itself), so its row count is a
    multiple of 32. Heads: the target attends with 8 (tp4) and 4 (tp8); TRTLLM-GEN has no MLA
    kernel for 64 < H < 128 and raises there. `max_seq_len` does not change the result for
    this path (measured equal at `W` and at the pool's length), so the entry passes `W`. The
    launch uses programmatic dependent launch unless `TRTLLM_ENABLE_PDL=0`.
    """

    def __call__(
        self,
        query: torch.Tensor,
        kv_rows: torch.Tensor,
        packed: torch.Tensor,
        lengths: torch.Tensor,
        workspace: torch.Tensor,
        counter: torch.Tensor,
        bmm1_scale: float | torch.Tensor,
        bmm2_scale: float | torch.Tensor,
    ) -> torch.Tensor:
        return trtllm_batch_decode_with_kv_cache_mla(
            query=query.unsqueeze(1),
            kv_cache=kv_rows.view(-1, 1, PAGE_ROWS, QUERY_DIM),
            workspace_buffer=workspace,
            qk_nope_head_dim=QK_NOPE_HEAD_DIM,
            kv_lora_rank=KV_LORA_RANK,
            qk_rope_head_dim=QK_ROPE_HEAD_DIM,
            block_tables=packed.unsqueeze(1),
            seq_lens=lengths,
            max_seq_len=packed.shape[1],
            sparse_mla_top_k=packed.shape[1],
            bmm1_scale=bmm1_scale,
            bmm2_scale=bmm2_scale,
            enable_pdl=get_env_enable_pdl(),
            backend="trtllm-gen",
            multi_ctas_kv_counter_buffer=counter,
            use_fp16_softmax=False,
        ).squeeze(1)

    def reference(
        self,
        query: torch.Tensor,
        kv_rows: torch.Tensor,
        packed: torch.Tensor,
        lengths: torch.Tensor,
        workspace: torch.Tensor,
        counter: torch.Tensor,
        bmm1_scale: float | torch.Tensor,
        bmm2_scale: float | torch.Tensor,
    ) -> torch.Tensor:
        """FP32 attention over each row's selected latent rows, one BF16 rounding.

        Keys are the full 576-wide rows, values their first 512 columns. ``workspace`` and
        ``counter`` are not read.
        """
        rows, heads, _ = query.shape
        # Gathered as bytes: advanced indexing does not take every FP8 dtype.
        latent = kv_rows.reshape(-1, QUERY_DIM).view(torch.uint8)
        out = torch.empty(rows, heads, KV_LORA_RANK, dtype=torch.bfloat16, device=query.device)
        col = torch.arange(packed.shape[1], device=query.device)
        for r0 in range(0, rows, 16):
            part = slice(r0, min(r0 + 16, rows))
            selected = col[None, :] < lengths[part, None].long()
            ids = packed[part].long().clamp(min=0)
            values = latent[ids].view(kv_rows.dtype).float()  # [c, W, 576]
            # The padding past each length is never read: no value of it may reach the sums.
            values = values.masked_fill(~selected[:, :, None], 0.0)
            with true_fp32_matmul():
                logits = torch.einsum("rhd,rwd->rhw", query[part].float(), values) * bmm1_scale
            probs = logits.masked_fill(~selected[:, None, :], float("-inf")).softmax(dim=-1)
            with true_fp32_matmul():
                attended = (
                    torch.einsum("rhw,rwd->rhd", probs, values[..., :KV_LORA_RANK]) * bmm2_scale
                )
            out[part] = attended.to(torch.bfloat16)
        return out

    def is_valid(
        self,
        query: torch.Tensor,
        kv_rows: torch.Tensor,
        packed: torch.Tensor,
        lengths: torch.Tensor,
        workspace: torch.Tensor,
        counter: torch.Tensor,
        bmm1_scale: float | torch.Tensor,
        bmm2_scale: float | torch.Tensor,
    ) -> None:
        assert query.dtype == kv_rows.dtype and query.dtype in (
            torch.bfloat16,
            torch.float8_e4m3fn,
        ), f"query and cache must share one dtype, BF16 or E4M3; got {query.dtype}, {kv_rows.dtype}"
        assert query.dim() == 3 and query.shape[2] == QUERY_DIM and query.is_contiguous()
        rows, heads, _ = query.shape
        assert heads in HEADS_PER_RANK, (
            f"{heads} heads; this target attends with {HEADS_PER_RANK} (32 heads / tp)"
        )
        assert kv_rows.shape[-1] == QUERY_DIM and kv_rows.shape[0] % PAGE_ROWS == 0
        assert packed.dtype == torch.int32 and packed.dim() == 2 and packed.is_contiguous()
        assert lengths.dtype == torch.int32 and lengths.shape == (rows,)
        assert workspace.dtype == torch.uint8 and workspace.is_contiguous()
        # The kernel's split-KV scratch is "the rest of the workspace": a short one is
        # overrun, not refused.
        assert workspace.numel() >= WORKSPACE_BYTES, (
            f"workspace of {workspace.numel()} bytes < {WORKSPACE_BYTES}"
        )
        # Semaphores left nonzero by anything but this kernel desynchronize the split rows.
        assert not bool(counter.any()), "the multi-CTA counter must be zero between launches"
        if rows == 0:
            return
        # A zero-width selection is not sparse: `sparse_mla_top_k=0` selects dense MLA over the
        # whole cache, a different operation with the same call shape.
        assert packed.shape[1] >= 1, "a zero-width selection asks for dense MLA, not sparse"
        # A zero length is unsupported by the kernel. A length past the row's valid picks makes
        # the kernel read the -1 padding (measured: relative RMS 0.32 against 0.0019), so every
        # column the row's length covers must be a row of the cache.
        assert int(lengths.min()) >= 1, "a zero-length row leaves the kernel no row to read"
        col = torch.arange(packed.shape[1], device=packed.device)
        selected = packed[col[None, :] < lengths[:, None].long()]
        # An id outside the cache is read, not refused.
        assert int(selected.min()) >= 0 and int(selected.max()) < kv_rows.shape[0], (
            "every selected id must be a row of kv_rows"
        )


sparse_mla_decode = SparseMlaDecode()
