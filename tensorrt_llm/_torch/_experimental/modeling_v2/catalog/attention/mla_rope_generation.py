# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""MLA generation-phase RoPE + latent KV-cache append + FMHA scheduler-buffer fill."""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import torch

import tensorrt_llm._torch.custom_ops  # noqa: F401 — registers torch.ops.trtllm.*

from .._op import Arch, Cell, OpWrapper

#: quant_mode bit for an fp8 KV cache.
QM_FP8_KV_CACHE = 128

#: Half an e4m3 ulp. e4m3 carries 3 explicit mantissa bits, so a
#: round-to-nearest lands within 2**-3 relative.
E4M3_ULP_RTOL = 2.0**-3

#: How much of a quantized output may miss bit-exactness before the gate calls
#: it drift rather than rounding. See `compare`.
E4M3_MAX_INEXACT_FRACTION = 1e-3

#: DeepSeek-R1's YaRN attention temperature, carried as `q_scaling = 1/mscale^2`
#: with `mscale = 0.1 * ln(40) + 1`. The target passes this, which is why a
#: `q_scaling != 1` cell exists: it lands in `mla_bmm1_scale` and nowhere else.
_R1_MSCALE = 0.1 * math.log(40.0) + 1.0
Q_SCALING_R1 = 1.0 / (_R1_MSCALE * _R1_MSCALE)


class _MlaRopeGeneration(OpWrapper):
    """One decode step's MLA preprocessing, over the batch's generation
    sequences only.

    Each generation sequence carries `P = predicted_tokens_per_seq` new query
    tokens -- 1 is an ordinary decode, `P > 1` the MTP shape. The per-token
    tensors hold `G * P` rows token-major within a sequence, and row `(g, t)`
    sits at absolute position `sequence_length[num_contexts + g] - P + t`, so
    each row is rotated at its own angle.

    Over an fp8 latent pool the call reads `fused_q`, writes the quantized
    query into `quant_q_buffer`, appends the quantized latent row to the pool,
    and fills `mla_bmm1_scale` / `mla_bmm2_scale` plus the two scheduler
    buffers. It returns None, so the cells below read the buffers and the pool
    rather than a return value.
    """

    ARCHS = frozenset({Arch.SM_103})

    # The DeepSeek-R1-0528 cell: 128 heads, 512/64 latent geometry, fp8 pool,
    # single-layer. Fixed by the checkpoint, so the cells vary what a step
    # varies -- how many new tokens, the page size, and whether the scales are
    # passed explicitly.
    _R1 = dict(
        num_heads=128,
        kv_lora_rank=512,
        qk_rope_head_dim=64,
        qk_nope_head_dim=128,
        quant_mode=QM_FP8_KV_CACHE,
    )

    CELLS: tuple[Cell, ...] = (
        Cell(
            why="production decode: P=1, page 32, scales omitted, R1's YaRN q_scaling",
            spec=dict(
                predicted_tokens_per_seq=1,
                tokens_per_block=32,
                q_scaling=Q_SCALING_R1,
                write_scale=None,
                read_scale=None,
                **_R1,
            ),
        ),
        Cell(
            why="the MTP shape: P=3 consecutive positions per sequence, page 32",
            spec=dict(
                predicted_tokens_per_seq=3,
                tokens_per_block=32,
                q_scaling=Q_SCALING_R1,
                write_scale=None,
                read_scale=None,
                **_R1,
            ),
        ),
        Cell(
            why="explicit reciprocal scales: w and r reach different outputs, so both must be driven",
            spec=dict(
                predicted_tokens_per_seq=1,
                tokens_per_block=32,
                q_scaling=1.0,
                write_scale=0.5,
                read_scale=2.0,
                **_R1,
            ),
        ),
    )

    note = """
    Only the fp8-e4m3 latent pool (`quant_mode=128`) is certified. The op also
    has a bf16-pool path, where `quant_q_buffer` is unused and the roped q lands
    in `fused_q`'s tail instead; nothing here drives it, so it is not certified
    -- the prose contract this class replaced claimed it was, with fifteen tests
    behind the claim and all fifteen on the fp8 path.

    The schema annotation is wrong in both directions: it marks `fused_q` and
    `q_pe` mutable, but `q_pe` is never modified and on this path neither is
    `fused_q`, while `cu_q_seqlens`, `cu_kv_seqlens`, `fmha_scheduler_counter`,
    `quant_q_buffer` and both scale buffers -- all filled by the call -- carry no
    annotation at all. Do not rely on its alias info under functionalization.

    Re-running a prepared step is idempotent rather than double-appending: the
    positions come from `sequence_length`, so the same slots are rewritten.

    Not certified: `P` above 4, `block_ids_per_seq` (the flash-MLA layout),
    helix, `out_scale`, an `attention_window_size` below the KV length, and
    multi-layer pools.
    """

    def __call__(
        self,
        fused_q: torch.Tensor,
        q_pe: torch.Tensor,
        latent_cache: torch.Tensor,
        rotary_cos_sin: Optional[torch.Tensor],
        cu_q_seqlens: torch.Tensor,
        cu_kv_seqlens: torch.Tensor,
        fmha_scheduler_counter: torch.Tensor,
        mla_bmm1_scale: Optional[torch.Tensor],
        mla_bmm2_scale: Optional[torch.Tensor],
        quant_q_buffer: Optional[torch.Tensor],
        sequence_length: torch.Tensor,
        host_past_key_value_lengths: torch.Tensor,
        host_context_lengths: torch.Tensor,
        num_contexts: int,
        kv_cache_block_offsets: Optional[torch.Tensor],
        host_kv_cache_pool_pointers: Optional[torch.Tensor],
        host_kv_cache_pool_mapping: Optional[torch.Tensor],
        kv_scale_orig_quant: Optional[torch.Tensor],
        kv_scale_quant_orig: Optional[torch.Tensor],
        kv_cache_scale_orig_quant: Optional[torch.Tensor],
        out_scale: Optional[torch.Tensor],
        block_ids_per_seq: Optional[torch.Tensor],
        helix_tensor_params: List[Optional[torch.Tensor]],
        predicted_tokens_per_seq: int,
        layer_idx: int,
        num_heads: int,
        num_kv_heads: int,
        head_size: int,
        residual_dim: int,
        tokens_per_block: int,
        attention_window_size: int,
        beam_width: int,
        quant_mode: int,
        q_scaling: float,
        q_lora_rank: int,
        kv_lora_rank: int,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        v_head_dim: int,
        rope_append: bool,
        kv_norm_weight: Optional[torch.Tensor] = None,
        kv_norm_eps: float = 1e-6,
        precomputed_cu_seqlens: bool = False,
        precomputed_fmha_scheduler: bool = False,
        kv_only: bool = False,
        kv_done_elsewhere: bool = False,
        quant_scale_qkv: Optional[torch.Tensor] = None,
    ) -> None:
        torch.ops.trtllm.mla_rope_generation(
            fused_q,
            q_pe,
            latent_cache,
            rotary_cos_sin,
            cu_q_seqlens,
            cu_kv_seqlens,
            fmha_scheduler_counter,
            mla_bmm1_scale,
            mla_bmm2_scale,
            quant_q_buffer,
            sequence_length,
            host_past_key_value_lengths,
            host_context_lengths,
            num_contexts,
            kv_cache_block_offsets,
            host_kv_cache_pool_pointers,
            host_kv_cache_pool_mapping,
            kv_scale_orig_quant,
            kv_scale_quant_orig,
            kv_cache_scale_orig_quant,
            out_scale,
            block_ids_per_seq,
            helix_tensor_params,
            predicted_tokens_per_seq,
            layer_idx,
            num_heads,
            num_kv_heads,
            head_size,
            residual_dim,
            tokens_per_block,
            attention_window_size,
            beam_width,
            quant_mode,
            q_scaling,
            q_lora_rank,
            kv_lora_rank,
            qk_nope_head_dim,
            qk_rope_head_dim,
            v_head_dim,
            rope_append,
            kv_norm_weight,
            kv_norm_eps,
            precomputed_cu_seqlens,
            precomputed_fmha_scheduler,
            kv_only,
            kv_done_elsewhere,
            quant_scale_qkv,
        )

    def reference(
        self,
        row: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        rope_size: int,
        kv_scale_orig_quant: Optional[torch.Tensor] = None,
        store_dtype: torch.dtype = torch.float8_e4m3fn,
    ) -> torch.Tensor:
        """One quantized `[.. , C+R]` row this op is supposed to write.

        The same documented exception to mirroring as the other paged-cache
        entries: the op's effect is on buffers and its arguments only say where.
        One function covers both quantized outputs, because they are the same
        computation on different inputs -- `[fused_q[..., :C] | q_pe]` for a
        `quant_q_buffer` row, `[compressed_kv | k_pe]` for an appended cache
        row. `cos` / `sin` are the rope table's halves already selected for the
        row's absolute position; the test reads them out of the environment,
        which is the only place they exist.

        Two details decide whether a mirror agrees with the kernel:

        The pairing is **GPT-J interleaved** -- consecutive elements form a pair
        -- not the split-half neox form. The two are indistinguishable by shape
        and give different answers.

        The rotation is rounded to the activation dtype **before** quantizing,
        which is what the kernel does. Quantizing the fp32 rotation directly
        disagrees on ~3% of the roped elements: two roundings, not one.

        The passthrough half is not re-rounded -- it arrives in `row`'s dtype
        and is only scaled. The scale is applied once, so the stored bytes are a
        function of the row rather than of the order the kernel visited it in.
        """
        pe = row[..., row.shape[-1] - rope_size :]
        lora = row[..., : row.shape[-1] - rope_size]
        pairs = pe.float().reshape(*pe.shape[:-1], rope_size // 2, 2)
        roped = torch.empty_like(pairs)
        roped[..., 0] = pairs[..., 0] * cos - pairs[..., 1] * sin
        roped[..., 1] = pairs[..., 0] * sin + pairs[..., 1] * cos
        # Round to the activation dtype first: the kernel's own intermediate.
        out = torch.cat([lora, roped.reshape(pe.shape).to(row.dtype)], dim=-1).float()
        if kv_scale_orig_quant is not None:
            out = out * kv_scale_orig_quant[0]
        return out.to(store_dtype)

    def reference_bmm_scales(
        self,
        kv_scale_quant_orig: Optional[torch.Tensor],
        q_scaling: float,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """The `(mla_bmm1_scale, mla_bmm2_scale)` this call is supposed to fill.

        A per-batch scalar pair, not per sequence, and `P` does not enter it.
        The read-side factor `r` appears squared in one and plain in the other
        because the consuming fp8 FMHA dequantizes both operands of BMM1 and one
        of BMM2 -- see `is_valid` for what that costs when `r != 1/w`.

        Computed in double and rounded once, so the comparison measures the
        kernel: every way of getting this scale wrong misses by a factor rather
        than by ulps.
        """
        r = 1.0 if kv_scale_quant_orig is None else float(kv_scale_quant_orig[0])
        x = r * r / (q_scaling * math.sqrt(qk_nope_head_dim + qk_rope_head_dim))
        return (
            torch.tensor([x, x * math.log2(math.e)], dtype=torch.float32),
            torch.tensor([r], dtype=torch.float32),
        )

    def reference_scheduler_buffers(
        self,
        kv_lens: List[int],
        num_heads: int,
        predicted_tokens_per_seq: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """The `(cu_q_seqlens, cu_kv_seqlens)` this call is supposed to fill.

        `kv_lens` is the generation sequences' total KV lengths, in batch order,
        already including all `P` of this step's tokens. The q buffer is in
        units of q rows -- `num_heads` rows per generation *token*, so
        `num_heads * P` per sequence -- and the kv buffer in tokens. Exactly
        `len(kv_lens) + 1` entries are written whatever `P` is; anything past
        that is left alone.
        """
        p = predicted_tokens_per_seq
        cu_q = [i * num_heads * p for i in range(len(kv_lens) + 1)]
        cu_kv = [0]
        for length in kv_lens:
            cu_kv.append(cu_kv[-1] + length)
        return (
            torch.tensor(cu_q, dtype=torch.int32),
            torch.tensor(cu_kv, dtype=torch.int32),
        )

    def compare(self, out: torch.Tensor, ref: torch.Tensor) -> None:
        """Half an e4m3 ulp, *and* a floor on how much may miss bit-exactness.

        The rotation is where kernel and reference can disagree: both evaluate
        `x*cos -+ y*sin` and may do it in different orders. e4m3's coarse
        rounding absorbs that in practice -- every quantized element measured on
        this architecture, at every scale and every cell, came back bit-exact,
        so neither half of this gate has been approached.

        Both halves are kept because they fail differently: a tolerance alone
        passes a mirror that is uniformly a little wrong, and a bit-exact demand
        alone fails the first time a reordering does survive the rounding.
        """
        assert out.shape == ref.shape, (out.shape, ref.shape)
        torch.testing.assert_close(out.float(), ref.float(), rtol=E4M3_ULP_RTOL, atol=0.0)
        inexact = int(
            (out.reshape(-1).view(torch.uint8) != ref.reshape(-1).view(torch.uint8)).sum()
        )
        allowed = max(1, int(E4M3_MAX_INEXACT_FRACTION * out.numel()))
        assert inexact <= allowed, (
            f"not bit-exact enough: {inexact}/{out.numel()} bytes differ (allowed {allowed})"
        )

    def is_valid(
        self,
        fused_q: torch.Tensor,
        q_pe: torch.Tensor,
        latent_cache: torch.Tensor,
        rotary_cos_sin: Optional[torch.Tensor],
        cu_q_seqlens: torch.Tensor,
        cu_kv_seqlens: torch.Tensor,
        fmha_scheduler_counter: torch.Tensor,
        mla_bmm1_scale: Optional[torch.Tensor],
        mla_bmm2_scale: Optional[torch.Tensor],
        quant_q_buffer: Optional[torch.Tensor],
        sequence_length: torch.Tensor,
        host_past_key_value_lengths: torch.Tensor,
        host_context_lengths: torch.Tensor,
        num_contexts: int,
        kv_cache_block_offsets: Optional[torch.Tensor],
        host_kv_cache_pool_pointers: Optional[torch.Tensor],
        host_kv_cache_pool_mapping: Optional[torch.Tensor],
        kv_scale_orig_quant: Optional[torch.Tensor],
        kv_scale_quant_orig: Optional[torch.Tensor],
        kv_cache_scale_orig_quant: Optional[torch.Tensor],
        out_scale: Optional[torch.Tensor],
        block_ids_per_seq: Optional[torch.Tensor],
        helix_tensor_params: List[Optional[torch.Tensor]],
        predicted_tokens_per_seq: int,
        layer_idx: int,
        num_heads: int,
        num_kv_heads: int,
        head_size: int,
        residual_dim: int,
        tokens_per_block: int,
        attention_window_size: int,
        beam_width: int,
        quant_mode: int,
        q_scaling: float,
        q_lora_rank: int,
        kv_lora_rank: int,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        v_head_dim: int,
        rope_append: bool,
        kv_norm_weight: Optional[torch.Tensor] = None,
        kv_norm_eps: float = 1e-6,
        precomputed_cu_seqlens: bool = False,
        precomputed_fmha_scheduler: bool = False,
        kv_only: bool = False,
        kv_done_elsewhere: bool = False,
        quant_scale_qkv: Optional[torch.Tensor] = None,
    ) -> None:
        """The four inputs this op takes and answers wrongly.

        The signature mirrors `__call__` position for position because
        `validating` forwards a call's arguments verbatim and every caller here
        passes them positionally: a narrower signature would collect
        `residual_dim` and `quant_mode` into a catch-all and check nothing,
        silently. Only `quant_q_buffer`, `kv_scale_orig_quant`,
        `kv_scale_quant_orig`, `residual_dim`, `quant_mode` and
        `kv_norm_weight` are read.
        """
        # `residual_dim` must be 0 or `rope_size`, and non-zero requires an FP4
        # KV pool. This entry certifies an fp8-e4m3 pool, against which a
        # non-zero value reads a layout that does not exist.
        assert residual_dim == 0, (
            f"residual_dim must be 0 for an fp8-e4m3 KV pool; got {residual_dim}. "
            "A non-zero value is only legal against an FP4 pool"
        )

        # Over an fp8 pool the quantized query has nowhere to go without this
        # buffer, and its absence is not checked: the kernel launches anyway,
        # the run dies with an illegal memory access, the CUDA context is lost
        # and the process aborts with nothing pointing here.
        if quant_mode & QM_FP8_KV_CACHE:
            assert quant_q_buffer is not None, (
                "an fp8 KV cache (quant_mode & 128) requires quant_q_buffer; the op does "
                "not presence-check it and launches into a null pointer, killing the "
                "CUDA context"
            )

        # A non-None weight folds the kv_a_layernorm into this kernel, which
        # then reads latent_cache RAW. A caller that already normalized -- every
        # caller here -- would be normalizing twice, and the result is plausible.
        assert kv_norm_weight is None, (
            "kv_norm_weight folds kv_a_layernorm into this kernel, which then reads "
            "latent_cache unnormalized; a caller that already normalized would "
            "normalize twice"
        )

        # The two scales are used independently and are not derived from each
        # other: `w` quantizes what this call writes, `r` is folded into the
        # scales the consuming fp8 FMHA dequantizes with. That cancellation is
        # only correct when `r = 1/w`, and nothing downstream can detect a
        # non-reciprocal pair -- the decode is silently mis-scaled. Driving them
        # apart on purpose is a characterization of the op, not a call a target
        # makes, so such a case belongs outside `validating()`.
        if kv_scale_orig_quant is not None or kv_scale_quant_orig is not None:
            w = 1.0 if kv_scale_orig_quant is None else float(kv_scale_orig_quant[0])
            r = 1.0 if kv_scale_quant_orig is None else float(kv_scale_quant_orig[0])
            assert w != 0.0 and math.isclose(r, 1.0 / w, rel_tol=1e-6), (
                f"kv_scale_quant_orig ({r}) must be the reciprocal of kv_scale_orig_quant "
                f"({w}): the consuming FMHA undoes the write scale with it, and a "
                "non-reciprocal pair mis-scales the decode with nothing to show for it"
            )


mla_rope_generation = _MlaRopeGeneration()
