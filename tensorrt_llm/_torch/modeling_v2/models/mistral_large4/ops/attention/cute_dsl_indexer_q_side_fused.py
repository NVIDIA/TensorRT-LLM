# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The DSA indexer's q side after its projection, in one CuTe DSL launch."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch

from tensorrt_llm._torch.modeling_v2.catalog._op import Arch, Cell, OpWrapper
from tensorrt_llm._torch.modeling_v2.models.mistral_large4.kernels import (
    _indexer_q_side_fused_kernel as _kernel,
)

#: The indexer geometry the kernel is specialised for: 32 heads of 128, the
#: first 64 dims of each head rotated.
_HEADS = 32
_HEAD_DIM = 128
_ROPE_DIM = 64
_HALF = _ROPE_DIM // 2
_E4M3_MAX = 448.0
_MIN_AMAX = 1.0e-12


@dataclass(frozen=True)
class CompiledIndexerQSide:
    """The compiled launcher and the rope mode it was compiled for."""

    fn: Callable
    is_neox: bool


def build(device_id: int, *, is_neox: bool) -> CompiledIndexerQSide:
    """Compile the variant the target runs: ue8m0 row scales, PDL launch.

    `is_neox` is a compile-time constant of the kernel. One compile, a few
    seconds, cached per process by the kernel module; allocates nothing.
    """
    with torch.cuda.device(device_id):
        fn = _kernel.compile_indexer_q_side_fused(device_id, bool(is_neox), True, use_pdl=True)
    return CompiledIndexerQSide(fn=fn, is_neox=bool(is_neox))


#: What every Mistral Large 4 FP8 cell shares: interleaved rope over its YaRN
#: (factor 512) + LeRoPE table of max_seq_len 9216 rows, positions drawn from
#: all of it.
_ML4 = dict(is_neox=False, table="yarn", max_pos=9216, positions="spread")
#: What every Mistral Large 4 NVFP4 cell shares: the same rope over that
#: checkpoint's YaRN factor 2048 + LeRoPE table of max_seq_len 32768 rows,
#: positions drawn from all of it.
_ML4_32K = dict(is_neox=False, table="yarn", factor=2048, max_pos=32768, positions="spread")
#: What every Mistral Large 4 NVFP4 cell at the checkpoint's own context shares (max_seq_len
#: 1048576, chunked prefill): the same rope over the factor 2048 + LeRoPE table at
#: 1048576 rows, positions drawn from all of it.
_ML4_1M = dict(is_neox=False, table="yarn", factor=2048, max_pos=1048576, positions="spread")


class CuteDslIndexerQSideFused(OpWrapper):
    """Rope the indexer query, pack it to e4m3, and fold its scale into the head weights.

    `q` is the unrotated `[T, 32, 128]` bf16 indexer query, each head `[pe(64) |
    nope(64)]`. For every (token, head) row the op

    1. rotates `pe` at `position_ids[t]` from `cos_sin[pos] = [cos(32) | sin(32)]`
       and rounds the result to bf16: NeoX halves (`is_neox=True`: dims `i` and
       `i + 32` rotate by column `i`) or interleaved pairs (`is_neox=False`:
       dims `2i` and `2i + 1` rotate by column `i`);
    2. takes `amax` over the rotated `pe` and the untouched `nope`, sets the row
       scale to `max(amax, 1e-12) / 448` rounded up to a power of two, and writes
       `q_fp8_out[t, h] = e4m3([pe | nope] / scale)`;
    3. writes `weights_out[t, h] = (wq[t, h] * scale) * weight_scale`.

    `q` is not modified. Both outputs are caller-owned and returned. What the
    caller still owns: the query's projection and any scaling of it before
    rope (Mistral Large 4's SSMax), and the head weights' projection and their
    cast to a contiguous fp32 `[T, 32]`.

    The call is `indexer_q_side_fused`, a CuTe DSL kernel's Python launcher
    (no `torch.ops` registration), launched once. It lives in the family's
    private kernel module `_indexer_q_side_fused_kernel`.
    """

    ARCHS = frozenset({Arch.SM_103})

    # One compiled launcher per rope mode serves every row count. Rows are
    # generation tokens on a decode step and the whole step on a prefill step;
    # positions go up to the engine's max_seq_len.
    # Mistral Large 4 runs interleaved rope on its 26 RoPE indexer layers and an
    # identity table (cos 1, sin 0) on its 18 NoPE indexer layers: decode
    # batches 1..16, CUDA-graph captured at 1, 2, 4, 8 and 16, and eager prefill
    # steps (PrefillTarget._indexer: every row of a step, a mixed step's
    # generation rows included). The FP8 target serves max_num_tokens 9216 at
    # positions 0..9215 on its factor 512 table; the NVFP4 target serves
    # max_seq_len = max_num_tokens 32768 with chunked prefill off, so one step
    # carries up to 32768 rows at positions 0..32767 on its factor 2048 table,
    # and its decode rows (CUDA graphs 1, 2, 4, 8, 16, padded) read positions up
    # to 32767. At its own context (max_seq_len 1048576, chunked prefill on,
    # max_num_tokens 32768) the NVFP4 tep4, dep4 and tep8 deployments build a 1048576-row
    # table and a step still carries up to 32768 rows: a chunk's rows sit at cached ..
    # cached + n - 1 behind the request's cached prefix (the last chunk of a 1M prompt
    # at 1015808..1048575), a mixed step's rows at any position, and its decode rows
    # read any position up to 1048575. The NVFP4 tp4 deployment makes the same call: its
    # indexer is replicated (32 heads on every rank), its tables come from the same
    # `_rope_tables`, and under the engine's query split every rank still ropes every
    # row of the step (the split divides the selection, after this op). At dep4
    # (attention DP, no indexer query split) each rank runs every row of its own
    # requests, so the step is the rank's: up to 32768 rows. Its boot captures each
    # decode graph 1, 2, 4, 8 and 16 on the warm-up batch: the longest dummy
    # (max_seq_len - 1 = 1048575 keys, so its decode row reads position 1048574) in
    # row 0 beside one-key fillers at position 0; the graph then replays with
    # attention DP's one live row behind one-key padding rows, or a full batch. A
    # cell's spec is one of the bases above with its own keys over it: `cached` starts
    # an `arange` step at that position, `position_sweep` (lo, hi) reads every row
    # lo .. hi - 1 in calls of `tokens` rows, `graph` captures at the cell's own row
    # count (True) or at each batch it lists, each replayed on new data, `first` is
    # row 0's position under `positions="first"` (every other row at 0), and
    # `tail_rows` is how many of the table's last rows `positions="tail"` draws from
    # (64 unless given).
    CELLS: tuple[Cell, ...] = (
        Cell(
            why="interleaved rope, decode batch 1 (captured)",
            spec=dict(_ML4, tokens=1, graph=True),
        ),
        Cell(
            why="interleaved rope, decode batch 16 (captured)",
            spec=dict(_ML4, tokens=16, graph=True),
        ),
        Cell(
            why="a NoPE indexer layer at decode batch 16 (captured): identity table, no rotation",
            spec=dict(_ML4, tokens=16, graph=True, table="identity"),
        ),
        Cell(
            why="NVFP4 interleaved prefill at max_num_tokens 32768 rows, positions 0..32767",
            spec=dict(_ML4_32K, tokens=32768, positions="arange"),
        ),
        Cell(
            why="NVFP4 NoPE indexer layer at 32768 rows: identity table over positions 0..32767",
            spec=dict(_ML4_32K, tokens=32768, positions="arange", table="identity"),
        ),
        Cell(
            why="NVFP4 table as the target holds it, cos and sin rounded through bf16, "
            "32768 rows at positions 0..32767",
            spec=dict(_ML4_32K, tokens=32768, positions="arange", table_dtype="bfloat16"),
        ),
        Cell(
            why="NVFP4 1M dep4 tail step: 32768 rows at positions drawn from the last chunk's "
            "1015808..1048575 in any order (a packed or mixed step's far rows)",
            spec=dict(_ML4_1M, tokens=32768, positions="tail", tail_rows=32768),
        ),
        Cell(
            why="NVFP4 1M dep4 NoPE tail step: identity table of 1048576 rows, 32768 rows drawn "
            "from 1015808..1048575",
            spec=dict(_ML4_1M, tokens=32768, positions="tail", tail_rows=32768, table="identity"),
        ),
    )

    note = """
    Fused CuTe DSL indexer Q-side RoPE + FP8 quantization. Covers decode graphs (batch 1/16),
    NoPE identity table, 32K prefill (bf16-rounded cos/sin), and 1M tail steps.
    """

    def __call__(
        self,
        q: torch.Tensor,
        position_ids: torch.Tensor,
        cos_sin: torch.Tensor,
        wq: torch.Tensor,
        weight_scale: float,
        q_fp8_out: torch.Tensor,
        weights_out: torch.Tensor,
        *,
        is_neox: bool,
        compiled: CompiledIndexerQSide,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return _kernel.indexer_q_side_fused(
            q,
            position_ids,
            cos_sin,
            wq,
            weight_scale,
            q_fp8_out,
            weights_out,
            is_neox=is_neox,
            use_ue8m0=True,
            compiled_fn=compiled.fn,
            use_pdl=True,
        )

    def reference(
        self,
        q: torch.Tensor,
        position_ids: torch.Tensor,
        cos_sin: torch.Tensor,
        wq: torch.Tensor,
        weight_scale: float,
        q_fp8_out: torch.Tensor,
        weights_out: torch.Tensor,
        *,
        is_neox: bool,
        compiled: CompiledIndexerQSide,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Fresh `(q_fp8, weights)`; `q_fp8_out`, `weights_out` and `compiled` are not read.

        The rope FMA is evaluated in fp64 and rounded to fp32 once, which is what
        a fused multiply-add does; every later step is an exact fp32 or bf16
        rounding, so the result is bit-exact rather than "close".
        """
        tokens = q.shape[0]
        pe_bf16 = q[..., :_ROPE_DIM]
        x = pe_bf16.float()
        bits = pe_bf16.contiguous().view(torch.int16)
        subnormal = (bits & 0x7F80) == 0
        x = torch.where(subnormal, torch.copysign(torch.zeros_like(x), x), x)
        nope = q[..., _ROPE_DIM:].float()

        table = cos_sin[position_ids.long()]  # [T, 64]
        cos, sin = table[:, :_HALF], table[:, _HALF:]
        if is_neox:
            partner = torch.cat([x[..., _HALF:], x[..., :_HALF]], dim=-1)
            cos_e = torch.cat([cos, cos], dim=-1)
            sin_e = torch.cat([sin, sin], dim=-1)
            sign = torch.cat([-torch.ones(_HALF), torch.ones(_HALF)])
        else:
            pairs = x.reshape(tokens, _HEADS, _HALF, 2)
            partner = pairs.flip(-1).reshape(tokens, _HEADS, _ROPE_DIM)
            cos_e = cos.repeat_interleave(2, dim=-1)
            sin_e = sin.repeat_interleave(2, dim=-1)
            sign = torch.tensor([-1.0, 1.0]).repeat(_HALF)
        sign = sign.to(device=q.device, dtype=torch.float32)
        cos_e, sin_e = cos_e[:, None, :], sin_e[:, None, :]
        partner_term = (partner * sin_e) * sign  # one fp32 rounding, then an exact sign flip
        rotated = (x.double() * cos_e.double() + partner_term.double()).float()
        rotated = rotated.to(torch.bfloat16).float()

        row = torch.cat([rotated, nope], dim=-1)  # [T, 32, 128]
        amax = row.abs().amax(dim=-1)
        floor = torch.tensor(_MIN_AMAX, dtype=torch.float32, device=q.device)
        inv_max = torch.tensor(1.0 / _E4M3_MAX, dtype=torch.float32, device=q.device)
        raw_scale = torch.maximum(amax, floor) * inv_max
        scale, inv_scale = _ue8m0_round_up(raw_scale)
        q_fp8 = (row * inv_scale[..., None]).to(torch.float8_e4m3fn)
        c = torch.tensor(weight_scale, dtype=torch.float32, device=q.device)
        weights = (wq * scale) * c
        return q_fp8, weights

    def is_valid(
        self,
        q: torch.Tensor,
        position_ids: torch.Tensor,
        cos_sin: torch.Tensor,
        wq: torch.Tensor,
        weight_scale: float,
        q_fp8_out: torch.Tensor,
        weights_out: torch.Tensor,
        *,
        is_neox: bool,
        compiled: CompiledIndexerQSide,
    ) -> None:
        # The launcher's rope mode is fixed at compile time and the wrapper does
        # not check it against `is_neox`: the other mode returns plausible wrong
        # codes (max abs ~7.8 against 2.4e-4).
        assert compiled.is_neox == bool(is_neox), (
            f"compiled for is_neox={compiled.is_neox}, called with is_neox={is_neox}"
        )
        # Positions are read on the device with no bound.
        if position_ids.numel():
            lo, hi = int(position_ids.min()), int(position_ids.max())
            assert 0 <= lo and hi < cos_sin.shape[0], (
                f"position_ids span [{lo}, {hi}] but the rope table has "
                f"{cos_sin.shape[0]} rows; the kernel reads past it"
            )


def _ue8m0_round_up(scale: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """`(2**ceil(log2 scale), 2**-ceil(log2 scale))` by exponent arithmetic, as the kernel does."""
    bits = scale.contiguous().view(torch.int32)
    exp_bits = bits & 0x7F800000
    exp_bits = torch.where((bits & 0x007FFFFF) != 0, exp_bits + 0x00800000, exp_bits)
    inv_bits = 0x7F000000 - exp_bits
    return exp_bits.view(torch.float32), inv_bits.view(torch.float32)


cute_dsl_indexer_q_side_fused = CuteDslIndexerQSideFused()
