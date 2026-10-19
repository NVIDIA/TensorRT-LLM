# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Mamba2 SSD prefill: the chunked scan over a packed batch of context
sequences, returning every sequence's final state."""

from __future__ import annotations

from typing import Optional

import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper
from tensorrt_llm._torch.modules.mamba import ssd_combined as _ssd_combined

#: Per-rank geometry at tep4: 256 heads / 4, head_dim 64, 8 groups / 4, d_state 128.
NHEADS, HEADDIM, NGROUPS, DSTATE = 64, 64, 2, 128
#: At dep4 (attention DP) every rank runs the whole mixer: 256 heads, 8 groups.
DEP4_NHEADS, DEP4_NGROUPS = 256, 8
#: At tep8 (the BF16 checkpoint over 8 ranks) a rank runs an eighth of the mixer: 32
#: heads over one group, the same 32 heads per group as at the other two.
TEP8_NHEADS, TEP8_NGROUPS = 32, 1
#: `config.chunk_size`; the engine hands it to Mamba2Metadata as `mamba_chunk_size`.
CHUNK_SIZE = 128
#: The engine's SSM pool for this checkpoint (V2 hybrid manager, max_batch_size 32): 35 slots,
#: every slot holding all 48 Mamba layers' states side by side, so a layer's view
#: `[35, 64, 64, 128]` has slot stride 48 * 64 * 64 * 128 and dense inner dims. At dep4
#: (the manager uses tp=1 for Mamba under attention DP) measured: `[35, 256, 64, 128]`,
#: slot stride 48 * 256 * 64 * 128 -- slots from 22 on lie past 2^31 elements.
ENGINE_SLOTS = 35
ENGINE_SLOT_LAYERS = 48

_LENGTHS_32 = tuple(1 + (i * 37) % 300 for i in range(32))
_DEP4 = dict(nheads=DEP4_NHEADS, ngroups=DEP4_NGROUPS)
_TEP8 = dict(nheads=TEP8_NHEADS, ngroups=TEP8_NGROUPS)


class MambaChunkScanCombined(OpWrapper):
    """The SSD chunked scan of a packed varlen batch (batch dim 1), one call for
    all context rows of a step, fp32 inside, bf16 operands to the tensor cores.

    For sequence `s` over tokens `[b, e)` of `cu_seqlens`, per head `h` (group
    `g = h // (nheads / ngroups)`), with `dt' = softplus(dt + dt_bias)` clamped
    to `dt_limit` and `S = initial_states[s]` (zeros when `initial_states` is
    None):
    `S = exp(dt'[t,h] * A[h]) * S + dt'[t,h] * outer(x[t,h], B[t,g])`,
    `out[t,h] = S @ C[t,g] + D[h] * x[t,h]` (the D skip included, no z gate),
    and the return is the stacked final `S` per sequence, `[nseq, H, P, N]` in
    `state_dtype`.

    Fusion boundary: dt bias + softplus, the chunked recurrence, the D skip and
    the per-sequence final state are inside. Gathering the initial states from
    the pool and scattering the final states back are the caller's (see `note`).

    Writes the output into `out` (required, `x.shape`) and returns only the
    states.
    """

    ARCHS = frozenset({Arch.SM_100, Arch.SM_107})

    # spec keys: lengths (tokens per context row), initial (has_initial_state per row),
    # pool_dtype (the SSM cache, default fp16), state_dtype (default fp32),
    # pool_slots / pool_layers (the pool as `[slots, layers, H, P, N]` viewed at one
    # layer; default max(8, rows + 3) x 2), layout ("token_major" | "channel_major"),
    # param_dtype (D / dt_bias, default bf16 as stored), nheads / ngroups (default
    # the tep4 64 / 2; head_dim 64 and d_state 128 at both).
    CELLS: tuple[Cell, ...] = (
        Cell(
            why="gate prompts: 3 fresh rows, boundaries off the chunk grid, no carry",
            spec=dict(lengths=(100, 300, 37), initial=(False, False, False)),
        ),
        Cell(
            why="chunked-prefill continuation: every row carries fp16-pool state",
            spec=dict(lengths=(128, 77, 300), initial=(True, True, True)),
        ),
        Cell(
            why="fresh rows zero-filled beside carried rows, fp32 pool",
            spec=dict(
                lengths=(5, 200, 1, 129, 64),
                initial=(True, False, True, False, True),
                pool_dtype=torch.float32,
            ),
        ),
        Cell(
            why="rows starting on chunk boundaries (offset 0) with carried state",
            spec=dict(lengths=(128, 256, 128), initial=(True, True, False)),
        ),
        Cell(
            why="single-token context rows, carried and fresh",
            spec=dict(lengths=(1, 1, 1, 1), initial=(True, False, True, True)),
        ),
        Cell(
            why="32 rows (the batch ceiling) at the engine's 35-slot x 48-layer pool",
            spec=dict(
                lengths=_LENGTHS_32,
                initial=tuple(i % 3 != 0 for i in range(32)),
                pool_slots=ENGINE_SLOTS,
                pool_layers=ENGINE_SLOT_LAYERS,
            ),
        ),
        Cell(
            why="one fresh 8192-token prompt (the token ceiling): 64 chunks",
            spec=dict(lengths=(8192,), initial=(False,)),
        ),
        Cell(
            why="one 8192-token continuation chunk carrying fp32-pool state",
            spec=dict(lengths=(8192,), initial=(True,), pool_dtype=torch.float32),
        ),
        Cell(
            why="state_dtype fp16 = the fp16 pool's dtype, fp16 initial states",
            spec=dict(
                lengths=(100, 300, 37), initial=(True, False, True), state_dtype=torch.float16
            ),
        ),
        Cell(
            why="channel-major x/B/C: views of a [4608, T] conv output",
            spec=dict(lengths=(100, 300, 37), initial=(True, False, True), layout="channel_major"),
        ),
        Cell(
            why="fp32 D and dt_bias instead of the checkpoint's bf16",
            spec=dict(lengths=(64, 100), initial=(False, True), param_dtype=torch.float32),
        ),
        Cell(
            why="dep4 256 heads / 8 groups: 3 fresh rows off the chunk grid",
            spec=dict(_DEP4, lengths=(100, 300, 37), initial=(False, False, False)),
        ),
        Cell(
            why="dep4: fresh rows beside carried rows, fp32 pool",
            spec=dict(
                _DEP4,
                lengths=(5, 200, 1, 129, 64),
                initial=(True, False, True, False, True),
                pool_dtype=torch.float32,
            ),
        ),
        Cell(
            why="dep4: 32 rows at the engine's 35-slot x 48-layer fp16 pool",
            spec=dict(
                _DEP4,
                lengths=_LENGTHS_32,
                initial=tuple(i % 3 != 0 for i in range(32)),
                pool_slots=ENGINE_SLOTS,
                pool_layers=ENGINE_SLOT_LAYERS,
            ),
        ),
        Cell(
            why="dep4: one fresh 8192-token prompt (the token ceiling)",
            spec=dict(_DEP4, lengths=(8192,), initial=(False,)),
        ),
        Cell(
            why="dep4: an 8192-token continuation carrying fp32-pool state",
            spec=dict(_DEP4, lengths=(8192,), initial=(True,), pool_dtype=torch.float32),
        ),
        # The BF16 checkpoint's SSM state is fp32 (config `mamba_ssm_cache_dtype`), so the
        # tep8 cells that read the pool read an fp32 one.
        Cell(
            why="tep8 32 heads / 1 group: 3 fresh rows off the chunk grid",
            spec=dict(_TEP8, lengths=(100, 300, 37), initial=(False, False, False)),
        ),
        Cell(
            why="tep8: fresh rows beside carried rows, fp32 pool",
            spec=dict(
                _TEP8,
                lengths=(5, 200, 1, 129, 64),
                initial=(True, False, True, False, True),
                pool_dtype=torch.float32,
            ),
        ),
        Cell(
            why="tep8: 32 rows at the engine's 35-slot x 48-layer fp32 pool",
            spec=dict(
                _TEP8,
                lengths=_LENGTHS_32,
                initial=tuple(i % 3 != 0 for i in range(32)),
                pool_slots=ENGINE_SLOTS,
                pool_layers=ENGINE_SLOT_LAYERS,
                pool_dtype=torch.float32,
            ),
        ),
        Cell(
            why="tep8: one fresh 8192-token prompt (the token ceiling)",
            spec=dict(_TEP8, lengths=(8192,), initial=(False,)),
        ),
        Cell(
            why="tep8: an 8192-token continuation carrying fp32-pool state",
            spec=dict(_TEP8, lengths=(8192,), initial=(True,), pool_dtype=torch.float32),
        ),
    )

    note = """
    Call site (PrefillTarget, the context rows of a step; n = md.num_contexts,
    T = md.num_ctx_tokens, mm = md.mamba_metadata, pool = this layer's
    `[slots, 64, 64, 128]` SSM view):

        si, hi = mm.state_indices[:n], mm.has_initial_states[:n]
        init = None
        if mm.use_initial_states:                         # host bool
            init = pool[si].to(torch.float32)             # gather; dtype == state_dtype
            init.masked_fill_(~hi.view(n, 1, 1, 1), 0)    # fresh rows start at 0
        y = torch.empty(1, T, 64, 64, dtype=torch.bfloat16)
        final = mamba_chunk_scan_combined(
            x.view(1, T, 64, 64), dt.view(1, T, 64), A, B.view(1, T, 2, 128),
            C.view(1, T, 2, 128), 128, D, None, dt_bias, init, mm.seq_idx,
            mm.chunk_indices, mm.chunk_offsets, mm.cu_seqlens[:n + 1], True,
            (0.0, float("inf")), y, False, True, torch.float32)
        pool[si] = final.to(pool.dtype)                   # scatter: the one rounding

    At dep4 the same call with 256 heads and 8 groups: x `(1, T, 256, 64)`,
    dt `(1, T, 256)`, B/C `(1, T, 8, 128)`, A / D / dt_bias `[256]`, y
    `(1, T, 256, 64)`, x/B/C views of the `[T, 18432]` conv output, the pool
    `[slots, 256, 64, 128]` (the `dep4` cells).

    x/B/C are views of the token-major `[T, 4608]` conv output (row stride 4608
    is fine: x needs unit stride on its last dim or its token dim, B/C on their
    last dim, else the wrapper copies them), dt a view of the in_proj output
    (any strides).
    A is fp32 `-exp(A_log)`; D and dt_bias `[64]` in bf16 or fp32.

    What the caller owns:
      - zero-filling the rows without initial state. The slot holds whatever
        the previous occupant left; the op cannot tell, and reads it.
      - `initial_states.dtype == state_dtype` (or C's dtype when None): the
        scan kernel refuses a mismatch at compile time (CompilationError).
      - passing `initial_states=None` exactly when `chunk_indices` is None.
        With states but no chunk metadata the op asserts; the converse
        (metadata, no states) is legal and ignores the metadata.
      - `state_dtype`: fp32 is certified (the final state is rounded once, at
        the scatter). fp16 is certified too (its cell: the carried chunk
        states then round to fp16 inside). None means C's dtype, bf16:
        uncertified.

    Metadata consumed, as Mamba2Metadata.prepare builds it: `cu_seqlens`
    int32 (`[0, cumsum(context lens)]`, then +1 per decode row: slice
    `[:n + 1]`), `seq_idx` int32 `[1, T]` (`repeat_interleave(arange(n),
    lens)`), `chunk_indices` / `chunk_offsets` int32 (one entry per logical
    chunk: ceil(T / 128) + one per context start off the 128 grid; only built
    when `use_initial_states`, else None), `use_initial_states` (host
    `any(has_initial_states[:n])`, `num_cached_tokens > 0`), `state_indices`
    int32 (contexts first). `chunk_size` must be `mm.chunk_size` (128).

    Host behaviour: no host syncs on the Triton path (`.item()`-free given the
    metadata; verified under `torch.cuda.set_sync_debug_mode("error")`, with
    the gather and scatter above). No Python loop over sequences. Six Triton
    launches. The first call per autotune key benchmarks every config
    (synchronizes: 5 s measured, and 23 s more the first time states are
    carried -- a new key). Triton also specializes integer arguments (== 1,
    divisible by 16) -- the token count, the chunk count, the logical-chunk
    count -- and compiles each new variant on first use, about 1 s apiece
    measured; warmup should at least cover both HAS_INITSTATES variants
    (Mamba2Metadata.force_initial_states_for_warmup exists for that). Scratch
    is allocated per call: at T = 8192 with fp32 states about 270 MiB (chunk
    states, carried states, dt/dA, C.B) plus 2 MiB per row. Prefill is not
    CUDA-graph captured. Measured warm: 0.3 ms for one 8192-token row.

    Numerics: the kernel feeds bf16 operands to its dots -- the intra-chunk
    `C.B * decay * dt` weights, the scaled B of the chunk states, the carried
    state before `C @ S` -- so its error is a bf16 rounding of the *terms*:
    out within 2 U (U = 2^-8) of the sum of the terms' magnitudes, final
    states within 1.3 U. Where the D skip cancels `C @ S` (D runs negative in
    this checkpoint) that is a large multiple of the output itself.

    `TRTLLM_USE_MAMBA_FI_SSD=1` reroutes to FlashInfer's SSD, which raises on
    sm_107 ("requires ... SM100/SM103/SM110"); leave it unset.

    Not certified: z (gating inside), `return_final_states`, D `[H, P]`,
    fp32 / fp16 activations (tl.dot then runs TF32 / fp16), batch > 1
    without cu_seqlens, zero-length rows.

    Hard limits
      chunk_size  power of 2  asserted by the wrapper
      nheads % ngroups == 0   asserted by the wrapper
    """

    def __call__(
        self,
        x: torch.Tensor,
        dt: torch.Tensor,
        A: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
        chunk_size: int,
        D: Optional[torch.Tensor],
        z: Optional[torch.Tensor],
        dt_bias: Optional[torch.Tensor],
        initial_states: Optional[torch.Tensor],
        seq_idx: Optional[torch.Tensor],
        chunk_indices: Optional[torch.Tensor],
        chunk_offsets: Optional[torch.Tensor],
        cu_seqlens: Optional[torch.Tensor],
        dt_softplus: bool,
        dt_limit: tuple[float, float],
        out: torch.Tensor,
        return_final_states: bool,
        return_varlen_states: bool,
        state_dtype: Optional[torch.dtype],
    ):
        return _ssd_combined.mamba_chunk_scan_combined(
            x,
            dt,
            A,
            B,
            C,
            chunk_size,
            D=D,
            z=z,
            dt_bias=dt_bias,
            initial_states=initial_states,
            seq_idx=seq_idx,
            chunk_indices=chunk_indices,
            chunk_offsets=chunk_offsets,
            cu_seqlens=cu_seqlens,
            dt_softplus=dt_softplus,
            dt_limit=dt_limit,
            out=out,
            return_final_states=return_final_states,
            return_varlen_states=return_varlen_states,
            state_dtype=state_dtype,
        )

    def reference(
        self,
        x: torch.Tensor,
        dt: torch.Tensor,
        A: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
        chunk_size: int,
        D: Optional[torch.Tensor],
        z: Optional[torch.Tensor],
        dt_bias: Optional[torch.Tensor],
        initial_states: Optional[torch.Tensor],
        seq_idx: Optional[torch.Tensor],
        chunk_indices: Optional[torch.Tensor],
        chunk_offsets: Optional[torch.Tensor],
        cu_seqlens: Optional[torch.Tensor],
        dt_softplus: bool,
        dt_limit: tuple[float, float],
        out: torch.Tensor,
        return_final_states: bool,
        return_varlen_states: bool,
        state_dtype: Optional[torch.dtype],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """`(out, final_states)` as the call leaves them; mutates nothing.

        The exact token-by-token recurrence in fp32, one sequence at a time,
        rounded once: `out` to its dtype, the states to `state_dtype` (C's
        dtype when None). `chunk_size`, `seq_idx`, `chunk_indices` and
        `chunk_offsets` are blocking metadata the recurrence does not need;
        `is_valid` checks they agree with `cu_seqlens`. Only the certified
        call is mirrored: no z, varlen states returned, final states not.
        """
        assert z is None, "z gating is not certified"
        assert return_varlen_states and not return_final_states, (
            "only the per-sequence final states are certified"
        )
        assert x.shape[0] == 1, "only the packed varlen batch (batch dim 1) is certified"
        _, total, nheads, headdim = x.shape
        rep = nheads // B.shape[2]
        xf = x[0].float()
        dtf = dt[0].float() + (0.0 if dt_bias is None else dt_bias.float())
        if dt_softplus:
            dtf = torch.where(dtf <= 20.0, torch.log1p(torch.exp(dtf)), dtf)
        dtf = dtf.clamp(dt_limit[0], dt_limit[1])
        decay = torch.exp(dtf * A.float())  # [T, H]
        bf = B[0].float().repeat_interleave(rep, dim=1)  # [T, H, N]
        cf = C[0].float().repeat_interleave(rep, dim=1)
        if D is None:
            df = torch.zeros(nheads, headdim, device=x.device)
        else:
            df = D.float() if D.dim() == 2 else D.float()[:, None].expand(nheads, headdim)
        y = torch.empty(total, nheads, headdim, device=x.device)
        bounds = (cu_seqlens if cu_seqlens is not None else torch.tensor([0, total])).tolist()
        finals = []
        for s in range(len(bounds) - 1):
            if initial_states is None:
                state = torch.zeros(nheads, headdim, B.shape[3], device=x.device)
            else:
                state = initial_states[s].float()
            for t in range(bounds[s], bounds[s + 1]):
                inc = (dtf[t][:, None] * xf[t])[:, :, None] * bf[t][:, None, :]
                state = state * decay[t][:, None, None] + inc
                y[t] = (state * cf[t][:, None, :]).sum(-1) + df * xf[t]
            finals.append(state)
        dtype = C.dtype if state_dtype is None else state_dtype
        return y.unsqueeze(0).to(out.dtype), torch.stack(finals).to(dtype)

    def is_valid(
        self,
        x: torch.Tensor,
        dt: torch.Tensor,
        A: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
        chunk_size: int,
        D: Optional[torch.Tensor],
        z: Optional[torch.Tensor],
        dt_bias: Optional[torch.Tensor],
        initial_states: Optional[torch.Tensor],
        seq_idx: Optional[torch.Tensor],
        chunk_indices: Optional[torch.Tensor],
        chunk_offsets: Optional[torch.Tensor],
        cu_seqlens: Optional[torch.Tensor],
        dt_softplus: bool,
        dt_limit: tuple[float, float],
        out: torch.Tensor,
        return_final_states: bool,
        return_varlen_states: bool,
        state_dtype: Optional[torch.dtype],
    ) -> None:
        if cu_seqlens is None or not return_varlen_states:
            return  # the kernel ignores cu_seqlens then: one sequence per batch row
        total = x.shape[1]
        bounds = cu_seqlens.tolist()
        assert bounds[0] == 0 and bounds[-1] == total, (
            f"cu_seqlens must span the packed tokens exactly, [0 .. {total}]; got "
            f"[{bounds[0]} .. {bounds[-1]}]"
        )
        # Every kernel but the final-state one splits rows by seq_idx alone.
        # Without it (or with stale labels) the chunk states and the C.B products
        # run straight across a row boundary -- measured: outputs off by 58%
        # (59% stale) of the batch's largest output, final states by 13% (30%).
        # With carried states and no seq_idx the op asserts on its own.
        lengths = [e - b for b, e in zip(bounds[:-1], bounds[1:])]
        if seq_idx is None:
            assert initial_states is not None or len(lengths) == 1, (
                "packed rows need seq_idx; without it the kernels run them together"
            )
            return
        expected = torch.repeat_interleave(
            torch.arange(len(lengths)), torch.tensor(lengths, dtype=torch.long)
        )
        assert torch.equal(seq_idx.reshape(-1).long().cpu(), expected), (
            "seq_idx does not label the tokens of cu_seqlens; the kernels split rows by seq_idx"
        )
        if initial_states is None or chunk_indices is None or chunk_offsets is None:
            return  # no carried states: metadata ignored; carried states without it: op asserts
        # With carried states the scan walks the logical chunks chunk_indices /
        # chunk_offsets list, as given (measured, for a batch whose boundaries
        # moved by one token: outputs 13% off, no error). A logical chunk is a run
        # of one row inside one physical chunk: physical chunk c, plus one entry
        # per row start strictly inside it.
        want_idx, want_off = [], []
        for c in range(-(-total // chunk_size)):
            lo, hi = c * chunk_size, min((c + 1) * chunk_size, total)
            want_idx.append(c)
            want_off.append(0)
            for start in bounds[1:-1]:
                if lo < start < hi:
                    want_idx.append(c)
                    want_off.append(start - lo)
        assert chunk_indices.tolist() == want_idx and chunk_offsets.tolist() == want_off, (
            f"chunk_indices / chunk_offsets do not describe these cu_seqlens at chunk_size "
            f"{chunk_size}; the scan reads them as given"
        )


mamba_chunk_scan_combined = MambaChunkScanCombined()
