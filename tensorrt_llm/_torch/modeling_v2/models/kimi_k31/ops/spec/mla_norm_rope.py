# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The DFlash2 MLA drafter's latent RMSNorm + interleaved RoPE as CuTe DSL launches.

Launchers over the kernels of ``kernels/spec/mla_norm_rope.py``:

- :func:`k31_dflash2_mla_ctx_norm_rope`: the context latents of every drafter layer
  (``precompute_context_kv``), one launch;
- :func:`k31_dflash2_mla_ctx_norm_rope_store`: the same latents stored into the paged pool of every
  layer, one launch;
- :func:`k31_dflash2_mla_block_rope`: one layer's block work in ``dflash_forward`` -- the
  query's rope half and the block latents stored into the paged pool -- one launch.

and one OpWrapper per launcher. All are bitwise the torch reference: the kernels round every square,
sum and product on its own and sum the 512 squares in the order torch's CUDA mean takes.

It also holds the host side of the drafter's YaRN RoPE: the rope parameters read from the drafter
config (:func:`resolve_mla_rope_params`), YaRN's attention-scale factor (:func:`yarn_get_mscale`) and
the ``freqs`` table the kernels index (:func:`build_mla_yarn_freqs_cis`).
"""

from __future__ import annotations

import functools
import math
from typing import Dict, Tuple

import cutlass
import cutlass.cute as cute
import torch

from tensorrt_llm._torch.modeling_v2.catalog._op import Arch, Cell, OpWrapper

from ...kernels.spec.mla_norm_rope import (
    K31DraftMlaBlockRoPEFp8Kernel,
    K31DraftMlaBlockRoPEKernel,
    K31DraftMlaCtxNormRoPEKernel,
    K31DraftMlaCtxNormRoPEStoreKernel,
)
from .draft_context_cache import to_draft_fp8

_CUTE_INTS = {torch.int32: cutlass.Int32, torch.int64: cutlass.Int64}

#: Kimi K3.1 DFlash2 drafter: 5 MLA layers, kv_lora_rank 512, rope 64, qk_nope 128, 96 heads.
_LAYERS = 5
_RANK = 512
_ROPE = 64
_NOPE = 128
_EPS = 1e-5


def yarn_get_mscale(scale: float, mscale: float = 1.0) -> float:
    """YaRN's attention-scale factor: ``0.1 * mscale * ln(scale) + 1``, or 1 for ``scale <= 1``."""
    if scale <= 1:
        return 1.0
    return 0.1 * mscale * math.log(scale) + 1.0


def resolve_mla_rope_params(pretrained_config, model_config=None, slack: int = 0) -> Dict:
    """The drafter's YaRN RoPE settings, from either HF spelling (``rope_parameters`` / ``rope_scaling``).

    Only YaRN is accepted. ``max_positions`` is the table length: the served ``max_seq_len`` (else the
    checkpoint's ``max_position_embeddings``) plus ``slack``; the YaRN correction range uses
    ``original_max_positions``, so the length changes no table entry. The worker publishes the positions it
    serves (``_runtime_position_ceiling``), which the drafter prefers over this length.
    """
    scaling = (
        getattr(pretrained_config, "rope_parameters", None)
        or getattr(pretrained_config, "rope_scaling", None)
        or {}
    )
    rope_type = scaling.get("rope_type") or scaling.get("type") or "default"
    if rope_type != "yarn":
        raise ValueError(
            f"The DFlash2 MLA drafter declares rope_type={rope_type!r}; only 'yarn' is supported "
            "(the block decode builds its own YaRN table)."
        )
    theta = scaling.get("rope_theta") or getattr(pretrained_config, "rope_theta", 10000.0)
    runtime = getattr(model_config, "max_seq_len", None)
    return {
        "theta": float(theta),
        "scaling_factor": float(scaling["factor"]),
        "original_max_positions": int(scaling["original_max_position_embeddings"]),
        "max_positions": int(
            runtime or getattr(pretrained_config, "max_position_embeddings", 163840)
        )
        + slack,
        "beta_fast": float(scaling.get("beta_fast", 32.0)),
        "beta_slow": float(scaling.get("beta_slow", 1.0)),
        "mscale": float(scaling.get("mscale", 1.0)),
        "mscale_all_dim": float(scaling.get("mscale_all_dim", 0.0)),
    }


def build_mla_yarn_freqs_cis(
    *,
    dim: int,
    base: float,
    scaling_factor: float,
    original_max_position_embeddings: int,
    max_position_embeddings: int,
    beta_fast: float = 32.0,
    beta_slow: float = 1.0,
    mscale: float = 1.0,
    mscale_all_dim: float = 0.0,
    device: str = "cuda",
) -> torch.Tensor:
    """``[max_position_embeddings, dim // 2]`` complex64 YaRN table of the drafter's rope slice.

    Adjacent-pair convention (``rope_interleave: true``): entry ``[p, i]`` rotates lanes ``2i`` / ``2i + 1`` at
    position ``p``; ``mscale`` rides in the modulus. The HF DeepSeek-V2 YaRN transcription, computed in FP32 on the
    host and then copied to ``device``. The cos / sin are taken over the frequency axis written twice (the layout
    every measured drafter run used; CPU ``cos`` / ``sin`` round 6 of 8224 entries 1 ulp differently over the
    single-width layout), then the first half is kept.
    """
    dtype = torch.float32

    def find_correction_dim(num_rotations):
        return (
            dim * math.log(original_max_position_embeddings / (num_rotations * 2 * math.pi))
        ) / (2 * math.log(base))

    low = math.floor(find_correction_dim(beta_fast))
    high = math.ceil(find_correction_dim(beta_slow))
    if low < 0:
        low = 0
    if high > dim - 1:
        high = dim - 1
    if low == high:
        high += 0.001  # Prevent singularity
    ramp = torch.clamp((torch.arange(dim // 2, dtype=dtype) - low) / (high - low), 0, 1)

    pos_freqs = base ** (torch.arange(0, dim, 2, dtype=dtype) / dim)
    freq_extra = 1.0 / pos_freqs
    freq_inter = 1.0 / (scaling_factor * pos_freqs)
    inv_freq_mask = 1 - ramp
    inv_freq = freq_inter * (1 - inv_freq_mask) + freq_extra * inv_freq_mask
    t = torch.arange(max_position_embeddings, dtype=dtype)
    sinusoid_inp = torch.einsum("i,j -> ij", t, inv_freq).unsqueeze(-1)
    scale = float(
        yarn_get_mscale(scaling_factor, mscale) / yarn_get_mscale(scaling_factor, mscale_all_dim)
    )
    emb = torch.cat((sinusoid_inp, sinusoid_inp), dim=-2)
    table = torch.cat((torch.cos(emb) * scale, torch.sin(emb) * scale), dim=-1)
    cos = table[..., 0].contiguous().to(device)
    sin = table[..., 1].contiguous().to(device)
    half = cos.shape[-1] // 2
    return torch.complex(cos[..., :half], sin[..., :half]).contiguous()


@functools.cache
def _compile_ctx(
    num_layers: int, kv_lora_rank: int, rope_dim: int, eps: float, positions_dtype: torch.dtype
):
    head_dim = kv_lora_rank + rope_dim
    tokens = cute.sym_int()
    x_fake = cute.runtime.make_fake_tensor(
        cutlass.BFloat16, (tokens, num_layers * head_dim), stride=(cute.sym_int64(), 1)
    )
    weight_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.BFloat16, (num_layers, kv_lora_rank), stride_order=(1, 0)
    )
    positions_fake = cute.runtime.make_fake_tensor(
        _CUTE_INTS[positions_dtype], (tokens,), stride=(cute.sym_int64(),)
    )
    freqs_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Float32, (cute.sym_int(), rope_dim // 2, 2), stride_order=(2, 1, 0)
    )
    output_fake = cute.runtime.make_fake_tensor(
        cutlass.BFloat16,
        (num_layers, tokens, head_dim),
        stride=(cute.sym_int64(), cute.sym_int64(), 1),
    )
    stream_fake = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    kernel = K31DraftMlaCtxNormRoPEKernel(num_layers, kv_lora_rank, rope_dim, eps)
    return cute.compile(
        kernel,
        x_fake,
        weight_fake,
        positions_fake,
        freqs_fake,
        output_fake,
        stream_fake,
        options="--opt-level 2 --enable-tvm-ffi",
    )


@functools.cache
def _compile_ctx_store(
    num_layers: int, kv_lora_rank: int, rope_dim: int, eps: float, positions_dtype: torch.dtype
):
    head_dim = kv_lora_rank + rope_dim
    tokens = cute.sym_int()
    x_fake = cute.runtime.make_fake_tensor(
        cutlass.BFloat16, (tokens, num_layers * head_dim), stride=(cute.sym_int64(), 1)
    )
    weight_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.BFloat16, (num_layers, kv_lora_rank), stride_order=(1, 0)
    )
    positions_fake = cute.runtime.make_fake_tensor(
        _CUTE_INTS[positions_dtype], (tokens,), stride=(cute.sym_int64(),)
    )
    freqs_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Float32, (cute.sym_int(), rope_dim // 2, 2), stride_order=(2, 1, 0)
    )
    pool_fake = cute.runtime.make_fake_tensor(
        cutlass.BFloat16,
        (num_layers, cute.sym_int(), head_dim),
        stride=(cute.sym_int64(), cute.sym_int64(), 1),
    )
    rows_fake = cute.runtime.make_fake_tensor(cutlass.Int64, (tokens,), stride=(1,))
    keep_fake = cute.runtime.make_fake_tensor(cutlass.Uint8, (tokens,), stride=(1,))
    stream_fake = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    kernel = K31DraftMlaCtxNormRoPEStoreKernel(num_layers, kv_lora_rank, rope_dim, eps)
    return cute.compile(
        kernel,
        x_fake,
        weight_fake,
        positions_fake,
        freqs_fake,
        pool_fake,
        rows_fake,
        keep_fake,
        stream_fake,
        options="--opt-level 2 --enable-tvm-ffi",
    )


@functools.cache
def _compile_block(
    num_heads: int,
    qk_nope_dim: int,
    kv_lora_rank: int,
    rope_dim: int,
    eps: float,
    positions_dtype: torch.dtype,
    index_dtype: torch.dtype,
    fp8_pool: bool,
):
    head_dim = kv_lora_rank + rope_dim
    q_rows = cute.sym_int()
    tokens = cute.sym_int()
    q_fake = cute.runtime.make_fake_tensor(
        cutlass.BFloat16, (q_rows, qk_nope_dim + rope_dim), stride=(cute.sym_int64(), 1)
    )
    query_fake = cute.runtime.make_fake_tensor(
        cutlass.BFloat16, (q_rows, head_dim), stride=(cute.sym_int64(), 1)
    )
    kv_fake = cute.runtime.make_fake_tensor(
        cutlass.BFloat16, (tokens, head_dim), stride=(cute.sym_int64(), 1)
    )
    weight_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.BFloat16, (kv_lora_rank,), stride_order=(0,)
    )
    positions_fake = cute.runtime.make_fake_tensor(
        _CUTE_INTS[positions_dtype], (tokens,), stride=(cute.sym_int64(),)
    )
    freqs_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Float32, (cute.sym_int(), rope_dim // 2, 2), stride_order=(2, 1, 0)
    )
    pool_fake = cute.runtime.make_fake_tensor(
        cutlass.Uint8 if fp8_pool else cutlass.BFloat16,
        (cute.sym_int(), cute.sym_int(), head_dim),
        stride=(cute.sym_int64(), cute.sym_int64(), 1),
    )
    pages_fake = cute.runtime.make_fake_tensor(
        _CUTE_INTS[index_dtype], (tokens,), stride=(cute.sym_int64(),)
    )
    offsets_fake = cute.runtime.make_fake_tensor(
        _CUTE_INTS[index_dtype], (tokens,), stride=(cute.sym_int64(),)
    )
    stream_fake = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    if fp8_pool:
        kernel = K31DraftMlaBlockRoPEFp8Kernel(
            num_heads, qk_nope_dim, kv_lora_rank, rope_dim, eps, fp8_pool=True
        )
    else:
        kernel = K31DraftMlaBlockRoPEKernel(num_heads, qk_nope_dim, kv_lora_rank, rope_dim, eps)
    return cute.compile(
        kernel,
        q_fake,
        query_fake,
        kv_fake,
        weight_fake,
        positions_fake,
        freqs_fake,
        pool_fake,
        pages_fake,
        offsets_fake,
        stream_fake,
        options="--opt-level 2 --enable-tvm-ffi",
    )


def k31_dflash2_mla_ctx_norm_rope(
    latent: torch.Tensor,
    norm_weight: torch.Tensor,
    positions: torch.Tensor,
    freqs: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """RMSNorm + RoPE of the context latents of every drafter layer, one launch.

    Args:
        latent: ``[N, L, kv_lora_rank + rope_dim]`` BF16 with contiguous ``(L, head_dim)`` rows
            (any token stride).
        norm_weight: ``[L, kv_lora_rank]`` BF16 latent RMSNorm weights, contiguous.
        positions: ``[N]`` int32/int64 RoPE positions.
        freqs: ``[P, rope_dim // 2, 2]`` FP32 (cos, sin) table indexed by position, contiguous.
        eps: RMSNorm epsilon.
    Returns:
        ``[N, L, kv_lora_rank + rope_dim]`` BF16, contiguous.
    """
    num_tokens, num_layers, head_dim = latent.shape
    out = torch.empty((num_tokens, num_layers, head_dim), dtype=latent.dtype, device=latent.device)
    if num_tokens == 0:
        return out
    kv_lora_rank = norm_weight.shape[1]
    compiled = _compile_ctx(
        num_layers, kv_lora_rank, head_dim - kv_lora_rank, float(eps), positions.dtype
    )
    # detach(): DLPack export refuses tensors that require grad (parameters).
    compiled(
        latent.detach().as_strided((num_tokens, num_layers * head_dim), (latent.stride(0), 1)),
        norm_weight.detach(),
        positions.reshape(-1),
        freqs,
        out.transpose(0, 1),
    )
    return out


def k31_dflash2_mla_ctx_norm_rope_store(
    latent: torch.Tensor,
    norm_weight: torch.Tensor,
    positions: torch.Tensor,
    freqs: torch.Tensor,
    pool: torch.Tensor,
    rows: torch.Tensor,
    keep: torch.Tensor,
    eps: float,
) -> None:
    """``k31_dflash2_mla_ctx_norm_rope`` stored into the paged pool, one launch: token ``t``'s latent of layer
    ``l`` goes to ``pool[l, rows[t]]``, times zero where ``keep[t]`` is 0.

    Args:
        latent, norm_weight, positions, freqs, eps: as ``k31_dflash2_mla_ctx_norm_rope``.
        pool: ``[L, pool_rows, kv_lora_rank + rope_dim]`` BF16 (unit inner stride; ``pool_rows`` = pages x
            page size).
        rows: ``[N]`` int64 destination rows. keep: ``[N]`` uint8.
    """
    num_tokens, num_layers, head_dim = latent.shape
    if num_tokens == 0:
        return
    kv_lora_rank = norm_weight.shape[1]
    compiled = _compile_ctx_store(
        num_layers, kv_lora_rank, head_dim - kv_lora_rank, float(eps), positions.dtype
    )
    compiled(
        latent.detach().as_strided((num_tokens, num_layers * head_dim), (latent.stride(0), 1)),
        norm_weight.detach(),
        positions.reshape(-1),
        freqs,
        pool,
        rows.reshape(-1),
        keep.reshape(-1),
    )


def k31_dflash2_mla_block_rope(
    q: torch.Tensor,
    query: torch.Tensor,
    kv: torch.Tensor,
    norm_weight: torch.Tensor,
    positions: torch.Tensor,
    freqs: torch.Tensor,
    pool: torch.Tensor,
    pages: torch.Tensor,
    offsets: torch.Tensor,
    eps: float,
) -> None:
    """The block RoPE work of one MLA drafter layer in one launch.

    Args:
        q: ``[T, H, qk_nope_dim + rope_dim]`` BF16 query rows (unit inner stride, ``H`` heads
            per token).
        query: ``[T, H, kv_lora_rank + rope_dim]`` BF16 absorbed query; only its rope columns
            are written.
        kv: ``[T, kv_lora_rank + rope_dim]`` BF16 block latents (row-strided).
        norm_weight: ``[kv_lora_rank]`` BF16 latent RMSNorm weight, contiguous.
        positions: ``[T]`` int32/int64 RoPE positions of the block tokens.
        freqs: ``[P, rope_dim // 2, 2]`` FP32 (cos, sin) table indexed by position, contiguous.
        pool: ``[pages, page_size, kv_lora_rank + rope_dim]`` BF16 or FP8 E4M3 paged latent
            pool (unit inner stride).
        pages, offsets: ``[T]`` int32/int64 destination page and in-page offset.
        eps: RMSNorm epsilon.
    """
    num_tokens, num_heads, q_dim = q.shape
    if num_tokens == 0:
        return
    head_dim = query.shape[-1]
    kv_lora_rank = norm_weight.shape[0]
    rope_dim = head_dim - kv_lora_rank
    fp8_pool = pool.dtype == torch.float8_e4m3fn
    compiled = _compile_block(
        num_heads,
        q_dim - rope_dim,
        kv_lora_rank,
        rope_dim,
        float(eps),
        positions.dtype,
        pages.dtype,
        fp8_pool,
    )
    compiled(
        q.detach().as_strided((num_tokens * num_heads, q_dim), (q.stride(1), 1)),
        query.as_strided((num_tokens * num_heads, head_dim), (query.stride(1), 1)),
        kv.detach(),
        norm_weight.detach(),
        positions.reshape(-1),
        freqs,
        pool.view(torch.uint8) if fp8_pool else pool,
        pages.reshape(-1),
        offsets.reshape(-1),
    )


def _torch_mla_rope_pairs(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    """Rotate ``x``'s last dim as adjacent (re, im) pairs by ``freqs_cis`` (FP32 math).

    ``x`` is ``[..., rope_dim]``; ``freqs_cis`` complex ``[..., rope_dim // 2]`` broadcastable
    to ``x``'s leading dims. The interleaved (GPT-J) rotation of ``rope_interleave: true``.
    """
    xc = torch.view_as_complex(x.float().unflatten(-1, (-1, 2)).contiguous())
    return torch.view_as_real(xc * freqs_cis).flatten(-2).to(x.dtype)


def _torch_mla_latent_norm_rope(
    latent: torch.Tensor,
    norm_weight: torch.Tensor,
    eps: float,
    freqs_cis: torch.Tensor,
    latent_dim: int,
) -> torch.Tensor:
    """``[RMSNorm(latent[..., :latent_dim]) | RoPE(latent[..., latent_dim:])]`` (FP32 math)."""
    c = latent[..., :latent_dim].float()
    c = c * torch.rsqrt(c.square().mean(-1, keepdim=True) + eps) * norm_weight.float()
    k_pe = _torch_mla_rope_pairs(latent[..., latent_dim:], freqs_cis)
    return torch.cat([c.to(latent.dtype), k_pe], dim=-1)


def _ctx_spec(tokens: int, **overrides) -> dict:
    spec = dict(tokens=tokens, layers=_LAYERS, rank=_RANK, rope=_ROPE, eps=_EPS, positions=None)
    spec.update(overrides)
    return spec


class K31DFlash2MlaCtxNormRope(OpWrapper):
    """``[RMSNorm_l(latent[:, l, :R]) | RoPE(latent[:, l, R:], positions)]`` for every layer ``l``.

    The drafter's context latents (``precompute_context_kv``): ``latent`` is the fused
    ``kv_a`` projection of every layer ``[N, L, R + rope]``, ``norm_weight`` the stacked
    ``kv_a_layernorm`` weights ``[L, R]``, ``freqs_cis`` the drafter's complex64 YaRN table
    ``[P, rope / 2]``, ``positions`` the tokens' positions. Returns a new ``[N, L, R + rope]``.
    """

    ARCHS = frozenset({Arch.SM_103, Arch.SM_107})

    CELLS: tuple[Cell, ...] = (
        Cell(why="one context row: 5 layer rows, torch's 128-thread reduction", spec=_ctx_spec(1)),
        Cell(why="two context rows: 10 layer rows, torch's 64-thread reduction", spec=_ctx_spec(2)),
        Cell(why="decode-step context rows: 64 requests x up to 8 accepted", spec=_ctx_spec(512)),
        Cell(why="a prefill chunk of 8192 tokens", spec=_ctx_spec(8192)),
        Cell(
            why="int32 positions past the original 32k YaRN context",
            spec=_ctx_spec(37, positions="far"),
        ),
        Cell(why="sm_107 TEP8 generation: 1 request x 8 accepted rows", spec=_ctx_spec(8)),
        Cell(
            why="sm_107 DEP16 / Helix CP 8 generation: 96 accepted rows a step", spec=_ctx_spec(96)
        ),
        Cell(why="sm_107 context server: a 16k prefill chunk", spec=_ctx_spec(16384)),
    )

    note = """
    One launch for all 5 layers in place of the reference's chain (gather, cast, square, mean,
    add, rsqrt, two products, the complex rotation, cat). Bitwise the reference on every cell:
    the norm sums the squares in torch's order for a contiguous 512-wide FP32 mean, which
    depends on the row count the reference reduces (N * layers here; 32, 64 or 128 threads per
    row), so the cells cover all three regimes. A torch upgrade that changed that reduction
    would show up as this entry's test failing, not as a silent drift.

    TP does not change the latents (the drafter's kv_a projection is replicated). Compiles
    once per (layers, rank, rope, eps, positions dtype); the first call must be eager.
    kv_lora_rank must be 512 (the drafter's; the torch order is written for that width).
    """

    def __call__(
        self,
        latent: torch.Tensor,
        norm_weight: torch.Tensor,
        eps: float,
        freqs_cis: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        return k31_dflash2_mla_ctx_norm_rope(
            latent, norm_weight, positions, torch.view_as_real(freqs_cis), eps
        )

    def reference(
        self,
        latent: torch.Tensor,
        norm_weight: torch.Tensor,
        eps: float,
        freqs_cis: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        """The target's torch math (``precompute_context_kv`` = ``_torch_mla_latent_norm_rope`` with
        the stacked weights and the table gathered at ``positions``)."""
        freqs = freqs_cis[positions.view(-1).long()].unsqueeze(1)
        return _torch_mla_latent_norm_rope(latent, norm_weight, eps, freqs, norm_weight.shape[-1])

    def is_valid(
        self,
        latent: torch.Tensor,
        norm_weight: torch.Tensor,
        eps: float,
        freqs_cis: torch.Tensor,
        positions: torch.Tensor,
    ) -> None:
        assert latent.dtype == torch.bfloat16 and norm_weight.dtype == torch.bfloat16, (
            f"BF16 latents and weights, got {latent.dtype} / {norm_weight.dtype}"
        )
        num_tokens, num_layers, head_dim = latent.shape
        assert latent.stride(2) == 1 and latent.stride(1) == head_dim, (
            "each token's (layer, head_dim) block must be contiguous"
        )
        assert norm_weight.shape[0] == num_layers and norm_weight.is_contiguous(), (
            f"norm_weight must be a contiguous [{num_layers}, kv_lora_rank]"
        )
        rank = norm_weight.shape[1]
        assert rank % 32 == 0 and ((head_dim - rank) // 2) % 32 == 0, (
            "kv_lora_rank and rope_dim / 2 must be multiples of 32 (one warp per row)"
        )
        assert freqs_cis.dtype == torch.complex64 and freqs_cis.is_contiguous(), (
            "freqs_cis must be a contiguous complex64 [positions, rope_dim / 2]"
        )
        assert freqs_cis.shape[1] == (head_dim - rank) // 2, "one rotation per rope pair"
        assert positions.dtype in (torch.int32, torch.int64) and positions.shape == (num_tokens,)


class K31DFlash2MlaCtxNormRopeStore(OpWrapper):
    """``K31DFlash2MlaCtxNormRope`` of the accepted rows' context latents, times their keep mask, stored into
    the paged latent pool of every layer: ``pool[l, rows[t]] = out[t, l] * keep[t]`` for
    ``out = ctx_norm_rope(latent, ...)`` (BF16 products, as the torch path's ``latent.mul_(mask)`` and its
    indexed store).

    Args:
        latent: ``[N, L, R + rope]`` BF16. norm_weight: ``[L, R]``. eps. freqs_cis: ``[P, rope / 2]``
            complex64. positions: ``[N]``. pool: ``[L, pool_rows, R + rope]`` BF16 (written).
        rows: ``[N]`` int64 distinct pool rows. keep: ``[N]`` uint8.
    """

    ARCHS = frozenset({Arch.SM_103, Arch.SM_107})

    CELLS: tuple[Cell, ...] = (
        Cell(why="the ll step: 8 accepted rows, 3 kept", spec=_ctx_spec(8, keep=3)),
        Cell(why="the mid step: 2 requests x 5 rows", spec=_ctx_spec(10, keep=7)),
        Cell(why="the M2 step: 12 requests x 5 rows", spec=_ctx_spec(60, keep=40)),
        Cell(why="every row kept, 32 requests x 8", spec=_ctx_spec(256, keep=256)),
    )

    note = """
    Kernel: ``K31DraftMlaCtxNormRoPEStoreKernel`` (the context kernel's math, its stores redirected to the
    pool rows and zeroed by the mask). One launch where the torch path takes three (the norm / RoPE launch,
    the masked multiply, the indexed store). Bitwise. Rows must be distinct (the torch store keeps an
    unspecified one of repeated rows).
    """

    def __call__(
        self,
        latent: torch.Tensor,
        norm_weight: torch.Tensor,
        eps: float,
        freqs_cis: torch.Tensor,
        positions: torch.Tensor,
        pool: torch.Tensor,
        rows: torch.Tensor,
        keep: torch.Tensor,
    ) -> None:
        k31_dflash2_mla_ctx_norm_rope_store(
            latent, norm_weight, positions, torch.view_as_real(freqs_cis), pool, rows, keep, eps
        )

    def reference(
        self,
        latent: torch.Tensor,
        norm_weight: torch.Tensor,
        eps: float,
        freqs_cis: torch.Tensor,
        positions: torch.Tensor,
        pool: torch.Tensor,
        rows: torch.Tensor,
        keep: torch.Tensor,
    ) -> None:
        """The torch path: the context latents, ``mul_`` by the mask, the indexed store."""
        out = dflash2_mla_ctx_norm_rope.reference(latent, norm_weight, eps, freqs_cis, positions)
        out.mul_(keep.view(-1, 1, 1).to(out.dtype))
        pool[:, rows] = out.transpose(0, 1)

    def is_valid(
        self,
        latent: torch.Tensor,
        norm_weight: torch.Tensor,
        eps: float,
        freqs_cis: torch.Tensor,
        positions: torch.Tensor,
        pool: torch.Tensor,
        rows: torch.Tensor,
        keep: torch.Tensor,
    ) -> None:
        dflash2_mla_ctx_norm_rope.is_valid(latent, norm_weight, eps, freqs_cis, positions)
        assert pool.dtype == torch.bfloat16 and pool.stride(-1) == 1, (
            "a BF16 pool, unit inner stride"
        )
        assert pool.shape[0] == latent.shape[1] and pool.shape[2] == latent.shape[2]
        assert rows.dtype == torch.int64 and keep.dtype == torch.uint8
        assert int(rows.min()) >= 0 and int(rows.max()) < pool.shape[1], "rows inside the pool"
        assert rows.unique().numel() == rows.numel(), "distinct rows"


def _block_spec(batch: int, width: int, heads: int, **overrides) -> dict:
    spec = dict(
        batch=batch,
        width=width,
        heads=heads,
        nope=_NOPE,
        rank=_RANK,
        rope=_ROPE,
        eps=_EPS,
        page_size=64,
        pool="bf16",
        index_dtype=torch.int64,
    )
    spec.update(overrides)
    return spec


class K31DFlash2MlaBlockRope(OpWrapper):
    """One drafter layer's block RoPE: ``query[..., R:]`` and the block latents in the pool.

    Writes ``query[..., R:] = RoPE(q[..., nope:])`` and
    ``pool[pages, offsets] = [RMSNorm(kv[:, :R]) | RoPE(kv[:, R:])]`` (an FP8 pool takes the
    BF16 row saturated to E4M3, as ``to_draft_fp8``); the rest of ``query`` and ``pool`` is
    untouched. Returns ``(query, pool)``.
    """

    ARCHS = frozenset({Arch.SM_103, Arch.SM_107})

    CELLS: tuple[Cell, ...] = (
        Cell(why="c1, width 5, TP1: 96 heads (5 latent rows)", spec=_block_spec(1, 5, 96)),
        Cell(why="c1, width 8, TP2: 48 heads (8 latent rows)", spec=_block_spec(1, 8, 48)),
        Cell(why="64 requests at width 8, TP1: 512 tokens x 96 heads", spec=_block_spec(64, 8, 96)),
        Cell(why="16 requests at width 5, TP8: 12 heads", spec=_block_spec(16, 5, 12)),
        Cell(
            why="8 requests at width 8, TP32: 3 heads, 32-token pages, int32 indices",
            spec=_block_spec(8, 8, 3, page_size=32, index_dtype=torch.int32),
        ),
        Cell(
            why="sm_107 TEP8 generation: c1, width 8, TP8 (12 heads), FP8 pool",
            spec=_block_spec(1, 8, 12, pool="fp8"),
        ),
        Cell(
            why="sm_107 DEP16 generation: 12 requests at width 8, 96 heads, FP8 pool",
            spec=_block_spec(12, 8, 96, pool="fp8"),
        ),
        Cell(
            why="sm_107 Helix CP 8 generation: 16 requests at width 6, 12 heads, FP8 one-row pages",
            spec=_block_spec(16, 6, 12, pool="fp8", page_size=1),
        ),
    )

    note = """
    One launch per layer in place of the reference's query rotation, latent norm + rotation,
    cat and pool index_put. Bitwise the reference on every cell (the context entry's note: the
    block latents are normalized as T rows, T = requests * width). Heads per rank: 96 / TP for
    TP 1..32 (the drafter's 96 heads).
    """

    def __call__(
        self,
        q: torch.Tensor,
        query: torch.Tensor,
        kv: torch.Tensor,
        norm_weight: torch.Tensor,
        eps: float,
        freqs_cis: torch.Tensor,
        positions: torch.Tensor,
        pool: torch.Tensor,
        store_pages: torch.Tensor,
        store_offsets: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        k31_dflash2_mla_block_rope(
            q,
            query,
            kv,
            norm_weight,
            positions,
            torch.view_as_real(freqs_cis),
            pool,
            store_pages,
            store_offsets,
            eps,
        )
        return query, pool

    def reference(
        self,
        q: torch.Tensor,
        query: torch.Tensor,
        kv: torch.Tensor,
        norm_weight: torch.Tensor,
        eps: float,
        freqs_cis: torch.Tensor,
        positions: torch.Tensor,
        pool: torch.Tensor,
        store_pages: torch.Tensor,
        store_offsets: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """``dflash_forward``'s torch writes, applied to copies of ``query`` and ``pool``."""
        rank = norm_weight.shape[0]
        nope = q.shape[-1] - (query.shape[-1] - rank)
        freqs = freqs_cis[positions.reshape(-1).long()]
        query = query.clone()
        pool = pool.clone()
        query[..., rank:] = _torch_mla_rope_pairs(q[..., nope:], freqs.unsqueeze(1))
        blk_kv = _torch_mla_latent_norm_rope(kv, norm_weight, eps, freqs, rank)
        pages, offsets = store_pages.long(), store_offsets.long()
        if pool.dtype == torch.float8_e4m3fn:
            pool.view(torch.uint8)[pages, offsets] = to_draft_fp8(blk_kv).view(torch.uint8)
        else:
            pool[pages, offsets] = blk_kv.to(pool.dtype)
        return query, pool

    def is_valid(
        self,
        q: torch.Tensor,
        query: torch.Tensor,
        kv: torch.Tensor,
        norm_weight: torch.Tensor,
        eps: float,
        freqs_cis: torch.Tensor,
        positions: torch.Tensor,
        pool: torch.Tensor,
        store_pages: torch.Tensor,
        store_offsets: torch.Tensor,
    ) -> None:
        assert q.dtype == query.dtype == kv.dtype == norm_weight.dtype == torch.bfloat16, (
            "BF16 q, query, kv and norm weight"
        )
        assert pool.dtype in (torch.bfloat16, torch.float8_e4m3fn), f"pool dtype {pool.dtype}"
        num_tokens, num_heads, _ = q.shape
        head_dim = query.shape[-1]
        assert query.shape[:2] == (num_tokens, num_heads) and kv.shape == (num_tokens, head_dim)
        # Rows are addressed as token * H + head: each token's heads must be evenly strided.
        assert q.stride(2) == 1 and q.stride(0) == num_heads * q.stride(1), "q rows per head"
        assert query.stride(2) == 1 and query.stride(0) == num_heads * query.stride(1), (
            "query rows per head"
        )
        assert kv.stride(1) == 1 and pool.stride(-1) == 1 and pool.shape[-1] == head_dim
        assert pool.dim() == 3, "pool must be [pages, page_size, head_dim]"
        assert norm_weight.is_contiguous() and norm_weight.shape[0] % 32 == 0
        assert freqs_cis.dtype == torch.complex64 and freqs_cis.is_contiguous()
        for name, t in (
            ("positions", positions),
            ("pages", store_pages),
            ("offsets", store_offsets),
        ):
            assert t.dtype in (torch.int32, torch.int64) and t.numel() == num_tokens, name
        assert store_pages.dtype == store_offsets.dtype, "pages and offsets share a dtype"


dflash2_mla_ctx_norm_rope = K31DFlash2MlaCtxNormRope()
dflash2_mla_block_rope = K31DFlash2MlaBlockRope()
dflash2_mla_ctx_norm_rope_store = K31DFlash2MlaCtxNormRopeStore()
