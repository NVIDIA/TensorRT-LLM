# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Decode indexer logits over a paged MXFP4 key cache, and the work schedule they require.

Two entries over two callables of the vendored DeepGEMM (`tensorrt_llm.deep_gemm`, a pybind
module built from the `deepgemm` entry of `3rdparty/fetch_content.json`), one call each:

- `deepgemm_fp8_fp4_paged_mqa_logits` -> `tensorrt_llm.deep_gemm.fp8_fp4_paged_mqa_logits`
- `deepgemm_paged_mqa_logits_metadata` -> `tensorrt_llm.deep_gemm.get_paged_mqa_logits_metadata`

The schedule is a precondition of the logits call and has no other consumer, so both live here.
"""

from __future__ import annotations

from typing import Optional

import torch

from tensorrt_llm import deep_gemm
from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper

_E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0)
_CODE_BYTES = 64  # packed E2M1 bytes per key
_KEY_BYTES = 68  # 64 code bytes + one 4-byte UE8M0 word per key

#: (ratio, page entries): the INDEXER_COMPRESS page holds tokens_per_block / ratio entries;
#: tokens_per_block 128 gives (2, 64) and (1, 128), tokens_per_block 256 gives (2, 128).
_PAGES = ((2, 64), (1, 128), (2, 128))

#: First-row positions of a decode step's requests, by regime.
_REGIMES = (
    ("boundary", "lengths on page and 256-column split edges"),
    ("gate", "positions 1024..16383, the accuracy gates"),
    ("long", "positions 20480..40959, long context"),
    ("64k", "about 64K compressed positions"),
)

#: The logits cells; the schedule is certified at exactly these, as their input.
_LOGITS_CELLS: tuple[Cell, ...] = (
    tuple(
        Cell(
            why=f"{regime} B={batch} next_n={next_n} r={ratio} page {bk}: {what}",
            spec=dict(
                ratio=ratio, block_kv=bk, next_n=next_n, heads=32, batch=batch, regime=regime
            ),
        )
        for regime, what in _REGIMES
        for batch in (1, 8, 32)
        for next_n in (1, 6)
        for ratio, bk in _PAGES
    )
    + tuple(
        Cell(
            why=f"H={heads} next_n={next_n} r=2 page 64: a head split's share",
            spec=dict(
                ratio=2, block_kv=64, next_n=next_n, heads=heads, positions=[3, 130, 999, 4100]
            ),
        )
        for heads in (8, 16, 64)
        for next_n in (1, 6)
    )
    + (
        Cell(
            why="bf16 logits with fp32 weights: next_n=6 B=5 r=2 page 64",
            spec=dict(
                ratio=2,
                block_kv=64,
                next_n=6,
                heads=32,
                positions=[3, 130, 999, 4100, 20001],
                logits_dtype=torch.bfloat16,
            ),
        ),
    )
)


def _decode(packed: torch.Tensor, sf: torch.Tensor) -> torch.Tensor:
    """int8 `[..., 64]` packed E2M1 + int32 `[...]` packed UE8M0 -> float64 `[..., 128]`, exactly.

    Element `2j` is the low nibble of byte `j`; byte `b` of the little-endian
    word is the exponent of elements `32b .. 32b+31`.
    """
    lut = torch.tensor(_E2M1, dtype=torch.float64, device=packed.device)
    p = packed.view(torch.uint8).long()
    v = torch.stack([lut[p & 0xF], lut[p >> 4]], dim=-1).flatten(-2)
    e = sf.contiguous().view(torch.uint8).view(*sf.shape, 4).long() - 127
    scale = ((e + 1023) << 52).view(torch.float64)  # 2^e from the exponent bits
    return (v.unflatten(-1, (4, 32)) * scale.unsqueeze(-1)).flatten(-2)


class DeepgemmFp8Fp4PagedMqaLogits(OpWrapper):
    """Weighted-ReLU MQA logits of `B * next_n` query rows against each request's paged keys.

    The callable is `tensorrt_llm.deep_gemm.fp8_fp4_paged_mqa_logits` (vendored DeepGEMM, pybind),
    one call. For query row `i = b * next_n + j` of request `b`:

        logits[i, s] = sum_h relu(dec(q[b, j, h]) . dec(k[s])) * weights[i, h]   for s < context_lens[b, j]

    where `k[s]` is slot `s % block_kv` of pool page `block_table[b, s // block_kv]`
    and every other column of the row is not specified (nothing is masked to
    `-inf`). `q` is int8 `[B, next_n, H, 64]` packed E2M1 with int32
    `q_sf [B, next_n, H]` (four UE8M0 bytes), `kv_cache` uint8
    `[pages, block_kv, 1, 68]` holding per page `block_kv x 64` code bytes then
    `block_kv x 4` scale bytes, `weights` fp32 `[B * next_n, H]`. No global
    scale is applied: the `128^-0.5 * 32^-0.5` fold belongs in `weights`.

    `clean_logits` is fixed to False (the only value 2-D `context_lens`
    accepts) and the varlen `indices` mode is not exposed.

    Fusion boundary: the paged key gather, the MXFP4 MMA, relu, head weighting
    and the head reduction. The caller owns q / k quantization and packing,
    writing k into the pool, the weights projection and its scale fold, the
    schedule (`deepgemm_paged_mqa_logits_metadata`), the candidate-block mask
    and the top-k.

    Returns a fresh `[B * next_n, max_context_len]` tensor of `logits_dtype`,
    a row-strided view (see `note`). Inputs are not modified.
    """

    ARCHS = frozenset({Arch.SM_103})

    CELLS: tuple[Cell, ...] = _LOGITS_CELLS

    note = """
    Cells: queries `randn * 2` and keys `randn` packed by a native-torch MXFP4
    recipe (the checkpoint's `fp4_act_quant`: UE8M0 `2^ceil(log2(amax / 6))`
    per 32-block, E2M1 round-to-nearest-even), weights `randn * 2^-4` rounded
    to bf16 and passed as fp32; per-row lengths `(p + j + 1) // ratio`; a padding
    request (length 0, an all `-1` table row) in the middle of every batch of 8
    or more; pools of 3x the pages the batch needs plus 8 under a random
    permutation, with the slots past each request's length NaN-scaled.

    Metadata consumed: none as an object. For DeepSeek-V4.1, `context_lens[b, j]
    = (p_b + j + 1) // ratio` for request `b`'s first query position `p_b`
    (padding requests 0); `block_table` is the index-source layer's per-ratio
    compressed-page table in pool pages, the pool its INDEXER_COMPRESS buffer;
    `schedule_meta` comes from `deepgemm_paged_mqa_logits_metadata` over the
    same `context_lens` in the same step and can be shared by that ratio's
    layers. `weights = weights_proj(x) * 2^-6`.

    Output: row stride `max_context_len` rounded up to 256 (fp32) or 512
    (bf16). Columns at or past a row's own length can hold NaN, +inf or finite
    scores of keys it must not see; bound every row by its own length.
    Numerics: per-head dots are exact; the heads are reduced in fp32 in a fixed
    order (four FMA chains of `2 * relu` over heads `h = l mod 4`, then
    `((c0 + c2) + (c1 + c3)) / 2`), bit-identical to that emulation. bf16 logits
    are that value rounded once.

    Preconditions no metadata check can see (each measured silently wrong, so
    there is no `is_valid`): the pool in the kernel's page layout, not
    key-interleaved; `block_kv` the page's true entry count (32, 64 or 128);
    a valid page in every table entry below `ceil(max_j context_lens[b, j] /
    block_kv)` (`-1` past it is fine; a narrower table is read past);
    `max_context_len >= max(context_lens)` -- unchecked, a smaller width
    writes out of bounds, so pass the static bound in captured steps; and a
    schedule from the same lengths in the same step (a stale one gave wrong
    logits or a sticky illegal-address error). Loud rejections: 1-D lengths,
    H not a multiple of 4 in 4..128, bf16 weights with fp32 logits, fp16 logits,
    `weights.stride(1) != 1`, int64 tables or lengths, uint8 q, an int8 pool,
    a 256-entry page, a schedule built for another SM count. H 24 is taken and
    measured against the reference by a test, without being certified.

    JIT: both kernels compile on the first call per (`next_n`, page); one
    variant then serves every batch and width. Make that call outside capture;
    the schedule and the logits capture together and replay with new lengths,
    tables, keys, queries and weights.

    Certified beyond the cells: every B in 1..256 eagerly; every width up to
    1,048,832 (ratio 1) / 524,544 (ratio 2) with lengths up to 1,048,595 /
    524,297; capture at G in 1..32, 40..128 step 8, 192, 256 at the envelope's
    graph-copy widths; mixed steps as row-offset views (q, q_sf, weights after
    `tc` context rows; the lengths a view of the metadata's buffer; table rows
    from `nc`); pools up to 3,245,959 blocks at ratio 1 and 2,615,990 at ratio
    2 in the cache manager's layout (each ratio-2 owner's view), past 2^34 bytes.

    Not certified: other architectures; `block_kv` 32; `next_n` other than 1
    and 6; H other than 32 beyond the cells; B above 256; capture at other
    batches or widths, of a first-ever call, or of a mixed step; bf16 weights
    (a coarser bf16 reduction); the MXFP8, FP8 and varlen `indices` modes; a
    page stride larger than `block_kv * 68`; pools above the bounds.
    """

    def raw_call(
        self,
        q: torch.Tensor,
        q_sf: torch.Tensor,
        kv_cache: torch.Tensor,
        weights: torch.Tensor,
        context_lens: torch.Tensor,
        block_table: torch.Tensor,
        schedule_meta: torch.Tensor,
        max_context_len: int,
        logits_dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        return deep_gemm.fp8_fp4_paged_mqa_logits(
            (q, q_sf),
            kv_cache,
            weights,
            context_lens,
            block_table,
            schedule_meta,
            max_context_len,
            False,
            logits_dtype,
            None,
        )

    def reference(
        self,
        q: torch.Tensor,
        q_sf: torch.Tensor,
        kv_cache: torch.Tensor,
        weights: torch.Tensor,
        context_lens: torch.Tensor,
        block_table: torch.Tensor,
        schedule_meta: torch.Tensor,
        max_context_len: int,
        logits_dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """`sum_h relu(q_h . k) * w_h` over exactly decoded operands, in float64, rounded once.

        Each request's keys are read out of `kv_cache` through its table row in
        the kernel's page layout. NaN marks every column the op does not
        specify (at or past the row's own length); `compare` skips those.
        `schedule_meta` is ignored: it only splits the work. Nothing is mutated.
        """
        batch, next_n = q.shape[0], q.shape[1]
        block_kv = kv_cache.shape[1]
        out = torch.full(
            (batch * next_n, max_context_len), float("nan"), dtype=torch.float64, device=q.device
        )
        wd = weights.double()
        lens = context_lens.long()
        for b in range(batch):
            longest = int(lens[b].max())
            if longest == 0:
                continue
            npg = -(-longest // block_kv)
            pages = kv_cache[block_table[b, :npg].long()].reshape(npg, block_kv * _KEY_BYTES)
            codes = pages[:, : block_kv * _CODE_BYTES].reshape(npg * block_kv, _CODE_BYTES)
            sf = pages[:, block_kv * _CODE_BYTES :].reshape(npg * block_kv, 4)
            sf = sf[:longest].contiguous().view(torch.int32).squeeze(-1)
            kd = _decode(codes[:longest].view(torch.int8), sf)
            dots = torch.einsum("jhd,sd->jhs", _decode(q[b], q_sf[b]), kd)
            rows = slice(b * next_n, (b + 1) * next_n)
            score = (dots.clamp_min(0) * wd[rows].unsqueeze(-1)).sum(1)
            for j in range(next_n):
                length = int(lens[b, j])
                out[b * next_n + j, :length] = score[j, :length]
        return out.to(logits_dtype)

    def compare(self, out: torch.Tensor, ref: torch.Tensor) -> None:
        """torch's default band for the logits dtype, on every element the op specifies.

        The per-head dots are exact and the fp32 head reduction is the only
        rounding: the largest max abs difference from float64 the probes printed
        was 1.9e-6, inside the default fp32 band (rtol 1.3e-6, atol 1e-5). The
        op is also bit-identical to an fp32 emulation of its documented
        reduction order, which the test asserts on top of this gate. bf16
        logits are that fp32 value rounded once. Controls (rolled table rows,
        64-entry pages read as 128, a key-interleaved pool, no relu, reversed
        head weights) put 0.99 to 1.0 of the visible elements outside the band.
        """
        assert out.dtype == ref.dtype, (out.dtype, ref.dtype)
        assert out.shape == ref.shape, (out.shape, ref.shape)
        specified = ~torch.isnan(ref)
        torch.testing.assert_close(out[specified], ref[specified])


class DeepgemmPagedMqaLogitsMetadata(OpWrapper):
    """The per-SM work schedule `deepgemm_fp8_fp4_paged_mqa_logits` walks over these lengths.

    The callable is `tensorrt_llm.deep_gemm.get_paged_mqa_logits_metadata` (vendored DeepGEMM,
    pybind), one call, one small kernel. `context_lens` is int32 `[B, next_n]`, the same tensor
    holding the same values the logits call receives. `num_sms=None` resolves to
    `deep_gemm.get_num_sms()` on the host, the only value the logits call accepts.

    Fusion boundary: the split of the lengths' work over the SMs and nothing else.
    Returns a fresh int32 `[num_sms + 1, 2]` tensor.
    """

    ARCHS = frozenset({Arch.SM_103})

    CELLS: tuple[Cell, ...] = _LOGITS_CELLS

    note = """
    The schedule's values are DeepGEMM's own split of the work and have no
    plain-torch reference: they are certified only through their consumer.
    `reference` therefore states the contract the logits call checks (shape,
    dtype, device), and every cell gates the values end to end -- the schedule
    built from that cell's lengths drives `deepgemm_fp8_fp4_paged_mqa_logits`
    to its reference.

    Metadata consumed: `context_lens`, as the logits call gets it. Build one
    schedule per `context_lens` tensor per step; a ratio's schedule can be
    shared by all index-source layers of that ratio in the step.

    Preconditions: rebuild it whenever the lengths change -- a schedule from
    other lengths was measured to give wrong logits (all-zero lengths) or a
    sticky illegal-address error (2x lengths); that some other mismatches
    happened to give right logits is not a contract. Use the device's SM count.
    On sm_103 `block_kv` is only range-checked (32, 64 or 128; 256 raises); the
    kernel takes its page size from the pool. 1-D lengths raise.

    Captured together with the logits call at every capture the logits entry
    certifies.

    Not certified: other architectures; a schedule consumed by anything else.
    """

    def raw_call(
        self,
        context_lens: torch.Tensor,
        block_kv: int,
        num_sms: Optional[int] = None,
    ) -> torch.Tensor:
        if num_sms is None:
            num_sms = deep_gemm.get_num_sms()
        return deep_gemm.get_paged_mqa_logits_metadata(context_lens, block_kv, num_sms)

    def reference(
        self,
        context_lens: torch.Tensor,
        block_kv: int,
        num_sms: Optional[int] = None,
    ) -> torch.Tensor:
        """The schedule's contract, not its values: an int32 `[num_sms + 1, 2]` tensor on the lengths' device.

        Returned as an uninitialized tensor of that shape, dtype and device;
        `compare` reads only those. The values are gated by the consumer: see
        `note`. `block_kv` is ignored. Nothing is mutated.
        """
        if num_sms is None:
            num_sms = deep_gemm.get_num_sms()
        return torch.empty((num_sms + 1, 2), dtype=torch.int32, device=context_lens.device)

    def compare(self, out: torch.Tensor, ref: torch.Tensor) -> None:
        """Shape, dtype, device and contiguity: what the logits call asserts of a schedule.

        A schedule built for another SM count is rejected by the logits call
        with `_schedule_meta_size == num_sms / num_ctas_per_cluster + 1`; its values
        are gated through the logits (see `note`).
        """
        assert out.dtype == ref.dtype == torch.int32, (out.dtype, ref.dtype)
        assert out.shape == ref.shape, (out.shape, ref.shape)
        assert out.device == ref.device, (out.device, ref.device)
        assert out.is_contiguous()


deepgemm_fp8_fp4_paged_mqa_logits = DeepgemmFp8Fp4PagedMqaLogits()
deepgemm_paged_mqa_logits_metadata = DeepgemmPagedMqaLogitsMetadata()
