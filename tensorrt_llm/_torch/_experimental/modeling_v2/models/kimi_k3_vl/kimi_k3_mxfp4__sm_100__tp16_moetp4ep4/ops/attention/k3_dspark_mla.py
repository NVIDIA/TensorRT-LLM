# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The MLA DSpark drafter's kernels (Inferact-Kimi-K3-DSpark): the CuTe DSL latent RMSNorm + adjacent-pair RoPE, the
q rope rotation into the absorbed query, and the CuTe DSL MLA decode over the paged latent context and the block with
per-token KV bounds. All three are ``_torch/custom_ops`` ops; these wrappers state their contracts."""

from __future__ import annotations

import math
from functools import lru_cache
from typing import Optional

import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import (
    Arch,
    Cell,
    OpWrapper,
    true_fp32_matmul,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl import (
    drafter_config as _drafter_config,
)
from tensorrt_llm._torch.cute_dsl_utils import IS_CUTLASS_DSL_AVAILABLE
from tensorrt_llm._utils import get_sm_version, is_sm_100f

if IS_CUTLASS_DSL_AVAILABLE:
    import cutlass

    from tensorrt_llm._torch.custom_ops.cute_dsl_custom_ops import CuteDSLNVMlaDecodeBlackwellRunner
    from tensorrt_llm._torch.custom_ops.dspark_rmsnorm_rope_custom_op import (
        cute_dsl_dspark_rmsnorm_rope as _rmsnorm_rope_kernel,
    )
    from tensorrt_llm._torch.custom_ops.dspark_rmsnorm_rope_custom_op import (
        is_fused_dspark_rmsnorm_rope_supported as _fused_supported,
    )


def fused_rope_supported(
    x: torch.Tensor,
    weight: Optional[torch.Tensor],
    freqs_real: torch.Tensor,
    num_heads: int,
    rope_dim: int,
    norm_dim: Optional[int],
) -> bool:
    """Whether `k3_dspark_rmsnorm_rope` / `k3_dspark_rope_into` take these operands: the CuTe DSL build on sm_100f
    and the kernel's own operand contract (``is_fused_dspark_rmsnorm_rope_supported``), as the built-in MLA DSpark
    drafter decides."""
    if not (IS_CUTLASS_DSL_AVAILABLE and is_sm_100f()):
        return False
    return _fused_supported(x, weight, freqs_real, num_heads, rope_dim, norm_dim)


def _rotate_adjacent_pairs(
    t: torch.Tensor, freqs_cis: torch.Tensor, num_heads: int
) -> torch.Tensor:
    """The last dim of ``t`` [..., rd] read as (re, im) pairs; its flattened row r rotated by row r // num_heads of
    ``freqs_cis`` [rows / num_heads, >= rd / 2] (the kernel's ``freq_row``), in fp32; the input dtype out."""
    half = t.shape[-1] // 2
    xc = torch.view_as_complex(t.float().reshape(-1, num_heads, half, 2))
    fc = freqs_cis[:, :half].reshape(-1, 1, half)
    return torch.view_as_real(xc * fc).reshape(t.shape).to(t.dtype)


class K3DSparkRmsnormRope(OpWrapper):
    """``t`` [G, s, (h,) D] with the RMSNorm (weight ``weight``, ``eps``) of its first ``norm_dim`` columns (all of
    them for None) and the adjacent-pair RoPE of its last ``rope_dim`` columns by ``freqs_real`` [G * s, rd / 2, 2]
    fp32 (``torch.view_as_real`` of the complex [G, s, rd / 2] phases, flattened to one row per (G, s) as the built-in
    ``_rmsnorm_rope_batched`` passes it): a new tensor. ``apply_weight`` / ``apply_rmsnorm`` False skip
    the scale / the normalization."""

    ARCHS = frozenset({Arch.SM_100, Arch.SM_103})
    CELLS: tuple[Cell, ...] = (
        *(
            Cell(
                why=f"block latent, B={b}, T={t}: [B, T, 576], norm over the 512-wide latent, k_pe roped",
                spec=dict(
                    shape=(b, t, 576), heads=1, rope=64, norm_dim=512, weight=True, norm=True
                ),
            )
            for b, t in ((1, 7), (8, 7), (3, 8))
        ),
        *(
            Cell(
                why=f"context latent of {n} tokens, one layer: [1, N, 576]",
                spec=dict(
                    shape=(1, n, 576), heads=1, rope=64, norm_dim=512, weight=True, norm=True
                ),
            )
            for n in (1, 56, 300)
        ),
        Cell(
            why="q heads, rope only (no norm, no weight): [B, T, 4, 192] at TP16",
            spec=dict(
                shape=(2, 7, 4, 192), heads=4, rope=64, norm_dim=None, weight=False, norm=False
            ),
        ),
    )

    note = """
    The kernel the built-in MLA DSpark drafter runs (cute_dsl_dspark_rmsnorm_rope)
    on sm_100f with the CuTe DSL build, where `fused_rope_supported` holds: bf16
    rows with a regular row stride, D and the norm prefix multiples of 32,
    rope_dim // 2 a multiple of 32, fp32 contiguous freqs with one row per
    (G, s). Elsewhere the drafter runs the reference, as the built-in one does. The RoPE
    is DeepSeek's adjacent-pair form; the YaRN mscale rides in the table.
    The kernel reduces in fp32 and rounds once, as the reference does.
    """

    def __call__(
        self,
        t: torch.Tensor,
        weight: Optional[torch.Tensor],
        freqs_real: torch.Tensor,
        num_heads: int,
        rope_dim: int,
        eps: float,
        apply_weight: bool,
        apply_rmsnorm: bool,
        norm_dim: Optional[int],
    ) -> torch.Tensor:
        return _rmsnorm_rope_kernel(
            t,
            weight,
            freqs_real,
            num_heads,
            rope_dim,
            eps,
            apply_weight,
            apply_rmsnorm,
            False,
            norm_dim,
        )

    def reference(
        self,
        t: torch.Tensor,
        weight: Optional[torch.Tensor],
        freqs_real: torch.Tensor,
        num_heads: int,
        rope_dim: int,
        eps: float,
        apply_weight: bool,
        apply_rmsnorm: bool,
        norm_dim: Optional[int],
    ) -> torch.Tensor:
        """The built-in eager fallback (``_rmsnorm_rope_batched``), step for step."""
        freqs_cis = torch.view_as_complex(freqs_real.contiguous())
        split_norm = norm_dim is not None and norm_dim != t.shape[-1]
        head, tail = (t[..., :norm_dim], t[..., norm_dim:]) if split_norm else (t, None)
        if apply_rmsnorm:
            if apply_weight:
                hf = head.float()
                hf = hf * torch.rsqrt(hf.square().mean(-1, keepdim=True) + eps)
                head = (weight.float() * hf).to(head.dtype)
            else:
                head = head * torch.rsqrt(head.square().mean(-1, keepdim=True) + eps)
        elif apply_weight:
            head = (head.float() * weight.float()).to(head.dtype)
        t = head if tail is None else torch.cat([head, tail], dim=-1)
        if rope_dim > 0:
            rope = _rotate_adjacent_pairs(t[..., -rope_dim:], freqs_cis, num_heads)
            t = torch.cat([t[..., :-rope_dim], rope], dim=-1)
        return t

    def is_valid(
        self,
        t: torch.Tensor,
        weight: Optional[torch.Tensor],
        freqs_real: torch.Tensor,
        num_heads: int,
        rope_dim: int,
        eps: float,
        apply_weight: bool,
        apply_rmsnorm: bool,
        norm_dim: Optional[int],
    ) -> None:
        assert fused_rope_supported(
            t, weight if apply_weight else None, freqs_real, num_heads, rope_dim, norm_dim
        ), "operands outside the kernel's contract: the caller runs the reference"


class K3DSparkRopeInto(OpWrapper):
    """Rotate the last ``rope_dim`` columns of ``q`` [B, T, h, Dq] (adjacent pairs, ``freqs_real`` [B * T, rd / 2, 2])
    into ``out[..., out_rope_offset:out_rope_offset + rope_dim]`` of ``out`` [B, T, h, Do]; nothing else of ``out``
    is written. Returns None."""

    ARCHS = frozenset({Arch.SM_100, Arch.SM_103})
    CELLS: tuple[Cell, ...] = tuple(
        Cell(
            why=f"q of 4 heads (TP16 rank) into the 576-wide absorbed query, B={b}, T={t}",
            spec=dict(batch=b, block=t, heads=4, q_dim=192, out_dim=576, rope=64, offset=512),
        )
        for b, t in ((1, 7), (8, 7), (4, 4))
    )

    note = """
    The kernel the built-in drafter runs (cute_dsl_dspark_rope_into) where
    `fused_rope_supported(q, None, freqs, heads, rope, Dq - rope)` holds; else
    the drafter rotates eagerly and copies. q is read only; out must share
    q's dtype and device and have a regular row stride.
    """

    def __call__(
        self,
        q: torch.Tensor,
        freqs_real: torch.Tensor,
        out: torch.Tensor,
        num_heads: int,
        rope_dim: int,
        out_rope_offset: int,
    ) -> None:
        torch.ops.trtllm.cute_dsl_dspark_rope_into(
            q, freqs_real, out, num_heads, rope_dim, out_rope_offset
        )

    def reference(
        self,
        q: torch.Tensor,
        freqs_real: torch.Tensor,
        out: torch.Tensor,
        num_heads: int,
        rope_dim: int,
        out_rope_offset: int,
    ) -> None:
        freqs_cis = torch.view_as_complex(freqs_real.contiguous())
        out[..., out_rope_offset : out_rope_offset + rope_dim] = _rotate_adjacent_pairs(
            q[..., -rope_dim:], freqs_cis, num_heads
        )

    def is_valid(
        self,
        q: torch.Tensor,
        freqs_real: torch.Tensor,
        out: torch.Tensor,
        num_heads: int,
        rope_dim: int,
        out_rope_offset: int,
    ) -> None:
        assert fused_rope_supported(
            q, None, freqs_real, num_heads, rope_dim, q.shape[-1] - rope_dim
        )
        assert out.dtype == q.dtype and out_rope_offset + rope_dim <= out.shape[-1]


@lru_cache(maxsize=1)
def mla_decode_unavailability_reason() -> Optional[str]:
    """Why the CuTe DSL MLA decode with per-token KV bounds cannot run here, or None (the checks of the built-in
    ``cute_dsl_mla_decode_unavailability_reason``)."""
    if (sm := get_sm_version()) not in (100, 103, 107):
        return f"requires SM100, SM103 or SM107, got SM{sm}"
    if not IS_CUTLASS_DSL_AVAILABLE:
        return "the cute-dsl kernels are not importable"
    op = getattr(torch.ops.trtllm, "cute_dsl_mla_decode_fp16_blackwell", None)
    if op is None:
        return "trtllm::cute_dsl_mla_decode_fp16_blackwell is not registered"
    if not any(
        arg.name == "kv_bounds"
        for overload in op.overloads()
        for arg in getattr(op, overload)._schema.arguments
    ):
        return (
            "the registered cute_dsl_mla_decode_fp16_blackwell takes no kv_bounds, so per-token KV bounds are "
            "unavailable in this build"
        )
    return None


def mla_decode_workspace_bytes(num_heads: int, block: int, latent: int, batch: int) -> int:
    """The decode's workspace for ``batch`` requests of ``block`` query rows (the runner's padded size)."""
    return CuteDSLNVMlaDecodeBlackwellRunner.get_max_padded_workspace_size(
        num_heads, block, latent, batch, cutlass.Float32
    )


def mla_decode_can_implement(
    batch: int,
    block: int,
    page_size: int,
    num_heads: int,
    latent: int,
    rope_dim: int,
    dtype: torch.dtype,
) -> bool:
    """Whether the kernel the runner picks serves this shape (the runner's own ``can_implement`` probe, through its
    constants: one tiler, split_kv 1)."""
    in_dtype = cutlass.Float16 if dtype == torch.float16 else cutlass.BFloat16
    runner = CuteDSLNVMlaDecodeBlackwellRunner
    return bool(
        runner._KERNEL_CLASS_BY_DTYPE[in_dtype].can_implement(
            batch,
            block,
            page_size,
            num_heads,
            latent,
            rope_dim,
            in_dtype,
            in_dtype,
            cutlass.Float32,
            cutlass.Float32,
            (128, 128),
            (128, 256),
            1,
            False,
            runner._IS_VAR_SEQ,
            runner._IS_VAR_SPLIT_KV,
            page_size,
        )
    )


class K3DSparkMlaDecode(OpWrapper):
    """Absorbed-MLA attention of ``query`` [B, T, H, L + R] (latent then rope) over each request's paged latent
    context: request b, query row t attends to the first ``kv_bounds[b * T + t]`` rows of its pages
    (``page_table`` row b, of the latent pool ``pool`` [P, page, L + R]), scaled by ``softmax_scale``. Returns the
    latent-space output [B, T, H, L]. ``seq_lens`` [B] is each request's KV length; ``workspace`` the kernel's
    scratch (`mla_decode_workspace_bytes`)."""

    ARCHS = frozenset({Arch.SM_100, Arch.SM_103})
    CELLS: tuple[Cell, ...] = tuple(
        Cell(
            why=f"4 heads (TP16 rank), latent 512 + rope 64, page 64, B={b}, T={t}, every row the whole block",
            spec=dict(batch=b, block=t, heads=4, latent=512, rope=64, page=64),
        )
        for b, t in ((1, 7), (2, 7), (8, 7), (3, 8))
    )

    note = """
    The call the built-in MLA DSpark drafter makes on its CUTEDSL backend
    (cute_dsl_mla_decode_fp16_blackwell with kv_bounds = seq_lens repeated per
    query row: every block row sees the whole block, no causal fixup), the
    page table passed as `.t()` without a copy and the output written through
    a permuted view. The block's latents are in the pool before the call.
    Builds without kv_bounds are refused (`mla_decode_unavailability_reason`);
    a shape the kernel cannot serve is refused (`mla_decode_can_implement`).
    """

    def __call__(
        self,
        query: torch.Tensor,
        pool: torch.Tensor,
        page_table: torch.Tensor,
        seq_lens: torch.Tensor,
        kv_bounds: torch.Tensor,
        latent: int,
        softmax_scale: float,
        workspace: torch.Tensor,
    ) -> torch.Tensor:
        batch, block, num_heads, _ = query.shape
        page_size = pool.shape[1]
        q_latent = query.detach()[..., :latent].permute(2, 3, 1, 0)
        q_rope = query.detach()[..., latent:].permute(2, 3, 1, 0)
        c_latent = pool.detach()[..., :latent].permute(1, 2, 0)
        c_rope = pool.detach()[..., latent:].permute(1, 2, 0)
        out = torch.empty(batch, block, num_heads, latent, dtype=query.dtype, device=query.device)
        torch.ops.trtllm.cute_dsl_mla_decode_fp16_blackwell(
            q_latent,
            q_rope,
            c_latent,
            c_rope,
            page_table.t(),
            seq_lens,
            out.permute(2, 3, 1, 0),
            workspace,
            num_heads,
            block,
            page_size,
            softmax_scale,
            1.0,
            batch,
            None,
            kv_bounds,
        )
        return out

    def reference(
        self,
        query: torch.Tensor,
        pool: torch.Tensor,
        page_table: torch.Tensor,
        seq_lens: torch.Tensor,
        kv_bounds: torch.Tensor,
        latent: int,
        softmax_scale: float,
        workspace: torch.Tensor,
    ) -> torch.Tensor:
        """fp32; ignores ``seq_lens`` and ``workspace``."""
        batch, block, num_heads, _ = query.shape
        page_size = pool.shape[1]
        out = torch.empty(batch, block, num_heads, latent, dtype=query.dtype, device=query.device)
        with true_fp32_matmul():
            for b in range(batch):
                for t in range(block):
                    n = int(kv_bounds[b * block + t])
                    pages = page_table[b, : (n + page_size - 1) // page_size].long()
                    kv = pool[pages].reshape(-1, pool.shape[-1])[:n].float()
                    scores = query[b, t].float() @ kv.T * softmax_scale
                    out[b, t] = (torch.softmax(scores, -1) @ kv[:, :latent]).to(query.dtype)
        return out

    def is_valid(
        self,
        query: torch.Tensor,
        pool: torch.Tensor,
        page_table: torch.Tensor,
        seq_lens: torch.Tensor,
        kv_bounds: torch.Tensor,
        latent: int,
        softmax_scale: float,
        workspace: torch.Tensor,
    ) -> None:
        batch, block, num_heads, width = query.shape
        assert mla_decode_unavailability_reason() is None
        assert pool.shape[-1] == width and query.stride(-1) == 1 and pool.stride(-1) == 1
        assert kv_bounds.dtype == torch.int32 and kv_bounds.numel() == batch * block
        assert page_table.dtype == torch.int32 and seq_lens.dtype == torch.int32
        assert mla_decode_can_implement(
            batch, block, pool.shape[1], num_heads, latent, width - latent, query.dtype
        )


def mla_paged_decode_kernel():
    """flashinfer's absorbed-MLA paged decode (``flashinfer.mla.trtllm_batch_decode_with_kv_cache_mla``), or None
    where it is not importable (``drafter_config.mla_paged_decode_kernel``, which ``route`` probes too)."""
    return _drafter_config.mla_paged_decode_kernel()


def mla_paged_decode_unavailability_reason() -> Optional[str]:
    """Why this build cannot run `k3_dspark_mla_paged_decode`, or None
    (``drafter_config.mla_paged_decode_unavailability_reason``)."""
    return _drafter_config.mla_paged_decode_unavailability_reason()


class K3DSparkMlaPagedDecode(OpWrapper):
    """flashinfer TRTLLM-Gen absorbed-MLA attention of ``query`` [B, T, H, L + R] over the paged latent context
    ``kv_cache`` [P, 1, page, L + R] (pages ``page_tables[b]``, int32), causal within the query block: request b's
    row t attends to its first ``seq_lens[b] - T + t + 1`` rows (the block's latents already in the pages). Writes
    the latent-space output ``out`` [B, T, H, L] and ``lse`` [B T, H] fp32, the log-sum-exp of the scaled scores in
    base 2. ``nope_dim`` is the un-absorbed q / k head width the kernel is told about; ``max_kv_len`` the pages'
    capacity per request; ``workspace`` its scratch. Returns None."""

    ARCHS = frozenset({Arch.SM_100, Arch.SM_103})
    CELLS: tuple[Cell, ...] = tuple(
        Cell(
            why=f"4 heads (TP16 rank), latent 512 + rope 64, nope 128, page 64, B={b}, T={t}",
            spec=dict(batch=b, block=t, heads=4, nope=128, latent=512, rope=64, page=64),
        )
        for b, t in ((1, 7), (2, 7), (8, 7), (3, 8))
    )

    note = """
    The call the built-in MLA DSpark drafter makes on its TRTLLM backend
    (``_mla_paged_attention``), with its positional arguments: backend
    "trtllm-gen", variable sequence lengths, shared paged KV index, LSE
    returned. The kernel is causal across the query block; the drafter then
    attends each row to the block keys after it (the strict upper triangle)
    eagerly and merges the two by LSE. The LSE is base 2 (the drafter
    multiplies it by ln 2). The workspace is a 256 MiB zeroed buffer.
    Needs flashinfer's MLA module (`mla_paged_decode_unavailability_reason`).
    The reference accumulates in fp32; the kernel rounds P to bf16.
    """

    def __call__(
        self,
        query: torch.Tensor,
        kv_cache: torch.Tensor,
        workspace: torch.Tensor,
        nope_dim: int,
        latent: int,
        rope_dim: int,
        page_tables: torch.Tensor,
        seq_lens: torch.Tensor,
        max_kv_len: int,
        softmax_scale: float,
        out: torch.Tensor,
        lse: torch.Tensor,
    ) -> None:
        mla_paged_decode_kernel()(
            query,
            kv_cache,
            workspace,
            nope_dim,
            latent,
            rope_dim,
            page_tables,
            seq_lens,
            max_kv_len,
            0,
            out,
            softmax_scale,
            1.0,
            None,
            None,
            None,
            backend="trtllm-gen",
            is_var_seq=True,
            uses_shared_paged_kv_idx=True,
            lse=lse,
            return_lse=True,
        )

    def reference(
        self,
        query: torch.Tensor,
        kv_cache: torch.Tensor,
        workspace: torch.Tensor,
        nope_dim: int,
        latent: int,
        rope_dim: int,
        page_tables: torch.Tensor,
        seq_lens: torch.Tensor,
        max_kv_len: int,
        softmax_scale: float,
        out: torch.Tensor,
        lse: torch.Tensor,
    ) -> None:
        """fp32; ignores ``workspace``, ``nope_dim`` and ``max_kv_len``."""
        batch, block, num_heads, _ = query.shape
        page_size = kv_cache.shape[2]
        pool = kv_cache[:, 0]
        with true_fp32_matmul():
            for b in range(batch):
                n = int(seq_lens[b])
                pages = page_tables[b, : (n + page_size - 1) // page_size].long()
                kv = pool[pages].reshape(-1, pool.shape[-1])[:n].float()
                for t in range(block):
                    keys = kv[: n - block + t + 1]
                    scores = query[b, t].float() @ keys.T * softmax_scale  # [H, n_t]
                    out[b, t] = (torch.softmax(scores, -1) @ keys[:, :latent]).to(out.dtype)
                    lse[b * block + t] = torch.logsumexp(scores, -1) / math.log(2.0)

    def is_valid(
        self,
        query: torch.Tensor,
        kv_cache: torch.Tensor,
        workspace: torch.Tensor,
        nope_dim: int,
        latent: int,
        rope_dim: int,
        page_tables: torch.Tensor,
        seq_lens: torch.Tensor,
        max_kv_len: int,
        softmax_scale: float,
        out: torch.Tensor,
        lse: torch.Tensor,
    ) -> None:
        assert mla_paged_decode_unavailability_reason() is None
        assert query.dtype == kv_cache.dtype == out.dtype == torch.bfloat16, "bf16 only"
        assert (
            kv_cache.dim() == 4
            and kv_cache.shape[1] == 1
            and kv_cache.shape[-1] == latent + rope_dim
        )
        assert query.shape[-1] == latent + rope_dim
        assert page_tables.dtype == torch.int32 and seq_lens.dtype == torch.int32
        assert lse.dtype == torch.float32 and lse.shape == (
            query.shape[0] * query.shape[1],
            query.shape[2],
        )


k3_dspark_rmsnorm_rope = K3DSparkRmsnormRope()
k3_dspark_rope_into = K3DSparkRopeInto()
k3_dspark_mla_decode = K3DSparkMlaDecode()
k3_dspark_mla_paged_decode = K3DSparkMlaPagedDecode()
