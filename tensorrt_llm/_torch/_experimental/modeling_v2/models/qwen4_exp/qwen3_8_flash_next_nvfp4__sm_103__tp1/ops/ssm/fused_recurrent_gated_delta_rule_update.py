# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Gated DeltaNet recurrent core over a slot pool, several tokens per sequence:
the speculative verify step's recurrence.

Wraps the in-tree module callable
``tensorrt_llm._torch.modules.fla.fused_recurrent.fused_recurrent_gated_delta_rule_update``
(tier-2 sourcing: the gated delta rule has no op boundary). One launch of the
vendored Triton kernel ``fused_recurrent_gated_delta_rule_update_fwd_kernel`` --
``triton.heuristics`` + ``triton.jit``, no autotuner.
"""

from __future__ import annotations

from typing import Optional

import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper
from tensorrt_llm._torch.modules.fla.fused_recurrent import (
    fused_recurrent_gated_delta_rule_update as _fused_recurrent_gated_delta_rule_update,
)

#: This checkpoint's GDN geometry at tp1.
NUM_K_HEADS, NUM_V_HEADS, HEAD_K_DIM, HEAD_V_DIM = 16, 48, 128, 128

#: Rows of the engine's per-step buffer (`max_num_sequences`), as the
#: runtime-value dumps record `intermediate_ssm_states`: [32, p, 48, 128, 128].
ENGINE_VERIFY_ROWS = 32
#: The recurrent pool's slot count the cache manager allocates (10 .. 461 over
#: the dumps). The target gathers a batch-scoped copy before this call, so the
#: write-back mode is bracketed here rather than driven.
ENGINE_SLOTS = 461
#: Pool and buffer rows of the small cells.
SLOTS = 8


def _verify(n, p, why, **spec):
    return Cell(why=why, spec=dict(mode="verify", n=n, p=p, **spec))


def _write_back(n, p, why, **spec):
    return Cell(why=why, spec=dict(mode="write_back", n=n, p=p, **spec))


class FusedRecurrentGatedDeltaRuleUpdate(OpWrapper):
    """Advance each sequence's `[K, V]` recurrent state by *all* its tokens, in
    order. Per token and v-head `hv` (q/k head `hv // (HV // H)`):

        q, k = q / (||q|| + 1e-6), k / (||k|| + 1e-6)    # if use_qk_l2norm_in_kernel
        h *= exp(g);  u = (v - h^T k) * beta;  h += k u^T;  o = h^T (scale * q)

    in fp32 throughout. `h` starts at `initial_state_source[idx]` (`[S, HV, V,
    K]`, K innermost), or zero when `idx < 0` or there is no pool. `scale=None`
    is `K ** -0.5`. `beta` with `v`'s rank is read per channel.

    Two independent sinks for `h`:
    `disable_state_update=False` stores the final state back into slot `idx`;
    `intermediate_states_buffer` (`[S', cache_steps, HV, V, K]`) receives the
    state after every token at `[idx, step]`. The verify call is the second
    alone: the pool must not move until the sampler has accepted a prefix.

    Fusion boundary: L2 norm, scale, the indexed gather and both scatters are
    inside. The gates are *not*: `g = -exp(A_log) * softplus(a + dt_bias)` and
    `beta = sigmoid(b)` arrive evaluated. The conv and q/k/v split are the
    caller's too.

    Returns the `[N, P, HV, V]` output (`[1, T, HV, V]` with `cu_seqlens`) in
    `output`'s dtype, else `q`'s; it *is* `output` when one is given. The pool
    and the per-step buffer are written in place and not returned.
    """

    ARCHS = frozenset({Arch.SM_103})

    # spec: mode ("verify": disable_state_update + buffer with cache_steps = p
    # | "write_back" | "both"), n sequences x p tokens (or lengths for the
    # varlen form), pool_slots (None: no pool), cache_rows, slots (None:
    # arange(n)), pool_dtype / cache_dtype, beta ("fp32" | "bf16": sigmoid of
    # a bf16 projection, as the target passes it | "channel"), qkv ("views"
    # over a packed [T, 10240] row block | "contiguous", as the target passes
    # them), output (None | "prefix" of a longer buffer | "fp32"),
    # index_dtype, cu_dtype, pool_view ("layer_major").
    CELLS: tuple[Cell, ...] = (
        tuple(
            _verify(
                n,
                p,
                f"target verify: n={n} p={p}, n-row gather, 32-row buffer",
                pool_slots=n,
                cache_rows=ENGINE_VERIFY_ROWS,
                beta="bf16",
                qkv="contiguous",
            )
            for n, p in ((1, 2), (8, 2), (16, 2), (32, 2), (8, 3), (8, 4))
        )
        + (
            _verify(
                4,
                2,
                "verify: rows 0/31/17/3 of the 32-row buffer",
                pool_slots=ENGINE_VERIFY_ROWS,
                cache_rows=ENGINE_VERIFY_ROWS,
                slots=(0, 31, 17, 3),
            ),
            _verify(
                4,
                2,
                "verify: -1-padded rows naming 31 and 0",
                pool_slots=ENGINE_VERIFY_ROWS,
                cache_rows=ENGINE_VERIFY_ROWS,
                slots=(31, -1, 0, -1),
            ),
        )
        + tuple(
            _verify(n, p, f"verify P={p}, n={n}: a state cached per token")
            for n, p in ((1, 1), (3, 1), (3, 2), (3, 3), (3, 4), (3, 8))
        )
        + tuple(_verify(n, 4, f"verify batch width n={n}, P=4") for n in (1, 2, 4, 8))
        + (
            _verify(4, 3, "verify scattered slots 7/1/4/0", slots=(7, 1, 4, 0)),
            _verify(3, 4, "verify slot -1: zero start, no cache row", slots=(2, -1, 5)),
            _verify(2, 3, "verify into a bf16 per-step buffer", cache_dtype=torch.bfloat16),
            _verify(2, 3, "verify with no pool: every sequence from zero", pool_slots=None),
            _write_back(
                3,
                2,
                "write-back into a 461-slot pool: slots 0, 460, 230",
                pool_slots=ENGINE_SLOTS,
                slots=(0, ENGINE_SLOTS - 1, 230),
            ),
        )
        + tuple(
            _write_back(4, p, f"write-back P={p}, scattered slots 7/1/4/0", slots=(7, 1, 4, 0))
            for p in (1, 2, 3, 4, 8, 16)
        )
        + (
            _write_back(3, 4, "write-back into a prefix of a longer output", output="prefix"),
            _write_back(2, 1, "write-back P=1 into a prefix of an output", output="prefix"),
            _write_back(3, 3, "write-back slot -1: no slot written", slots=(2, -1, 5)),
            _write_back(2, 4, "write-back, every slot -1", slots=(-1, -1)),
            _write_back(3, 4, "write-back into a bf16 pool", pool_dtype=torch.bfloat16),
            _write_back(2, 1, "write-back P=1 into a bf16 pool", pool_dtype=torch.bfloat16),
            _write_back(3, 1, "bf16 beta (the mixer's dtype), P=1", slots=(2, 5, 0), beta="bf16"),
            _write_back(3, 4, "bf16 beta (the mixer's dtype), P=4", slots=(2, 5, 0), beta="bf16"),
            _write_back(2, 3, "per-channel beta [N, P, HV, V]", slots=(1, 2), beta="channel"),
            _write_back(2, 3, "fp32 output buffer against bf16 q", slots=(1, 2), output="fp32"),
            _write_back(2, 3, "int64 initial_state_indices", slots=(1, 2), index_dtype=torch.int64),
            _write_back(
                2,
                3,
                "a [layers, S, ...] pool's middle layer",
                slots=(1, 4),
                pool_view="layer_major",
            ),
            Cell(why="both sinks at once", spec=dict(mode="both", n=2, p=4)),
            _write_back(3, None, "varlen: ragged lengths 1/5/2", lengths=(1, 5, 2)),
            _write_back(
                3,
                None,
                "varlen: uniform lengths, scattered slots",
                lengths=(4, 4, 4),
                slots=(2, 6, 3),
            ),
            _write_back(2, None, "varlen: int32 cu_seqlens", lengths=(2, 4), cu_dtype=torch.int32),
        )
    )

    note = """
    Metadata consumed: none; every piece of state is an argument.

    Call site (`_gdn_verify`), all positional: `(q, k, v, g, beta, src,
    int_idx, None, True, None, True, False, int_ssm, p, None)` -- q/k/v the
    contiguous `[n, p, 16|48, 128]` reshapes of `pack_gdn_decode_qkv`'s
    outputs (so the in-kernel L2 norm is required), `g` fp32 from
    `fused_gdn_gating`, `beta = sigmoid(b)` bf16, `src` an `[n, 48, 128, 128]`
    fp32 gather of the live slots, `int_idx = intermediate_state_indices[:n]`
    (int32 arange), `int_ssm` the layer's `[32, p, 48, 128, 128]` per-step
    buffer. The buffer's rows are the batch rows `update_mamba_states` reads,
    not slot ids, and its step axis is `tokens_per_gen_step`: `p` (as
    `cache_steps`) must equal it.

    Preconditions not checkable from metadata: `HV % H == 0` (q/k heads are
    repeat_interleaved, not tiled); indices in `[0, S)` and `[0, S')` or
    negative (the device check is `idx >= 0` only; out of range is an
    out-of-bounds access, not probed); distinct indices when writing back --
    measured at two sequences on one slot: outputs exact, the slot a
    per-(v-tile, head) mixture 0.83 .. 0.89 from a sequential application; in the
    varlen form `cache_steps >= max(seq_len)`; q/k unit-norm-conditioned (pass
    `use_qk_l2norm_in_kernel=True` or normalize upstream). `g` is log-space.

    Raise loudly, so not guarded: `cu_seqlens` with `q.shape[0] > 1` and an
    index count other than `len(cu_seqlens) - 1` (ValueError); a `None` pool
    with the write-back on (triton CompilationError).

    `is_valid` guards what is accepted and silently wrong, each measured:
    a strided `output` (the one tensor not made contiguous) is written with
    packed arithmetic -- 7.04 off and the neighbouring half overwritten;
    `output` with `disable_output_calculation=True` is never written and the
    return is a [1, 1, 1, 1] scratch; a strided pool or per-step buffer is
    silently replaced by a contiguous copy, so the writes land in the copy and
    the caller's tensor comes back bitwise unchanged; `cache_steps` below the
    token count (or None, i.e. 0) overwrites the next row's steps -- 0.850 off
    at cache_steps 2, P 4; `cache_steps` other than the buffer's step axis
    misplaces every row past 0 against the buffer's own indexing.

    Deterministic; the first call in a fresh process is bitwise the second.

    Not certified: `disable_output_calculation=True` alone (its cached states
    were never checked); `K` or `V` other than 128; fp16/fp32 q/k/v; P > 16.

    Hard limits
      N * HV  gridDim.z  boundary 65535  probed at N=1300 (62400): runs;
              N=1400 (67200): RuntimeError "invalid argument", context survives
    """

    def __call__(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        initial_state_source: Optional[torch.Tensor],
        initial_state_indices: torch.Tensor,
        scale: Optional[float] = None,
        use_qk_l2norm_in_kernel: bool = False,
        cu_seqlens: Optional[torch.Tensor] = None,
        disable_state_update: bool = False,
        disable_output_calculation: bool = False,
        intermediate_states_buffer: Optional[torch.Tensor] = None,
        cache_steps: Optional[int] = None,
        output: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return _fused_recurrent_gated_delta_rule_update(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            scale=scale,
            initial_state_source=initial_state_source,
            initial_state_indices=initial_state_indices,
            cu_seqlens=cu_seqlens,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
            disable_state_update=disable_state_update,
            disable_output_calculation=disable_output_calculation,
            intermediate_states_buffer=intermediate_states_buffer,
            cache_steps=cache_steps,
            output=output,
        )

    def reference(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        initial_state_source: Optional[torch.Tensor],
        initial_state_indices: torch.Tensor,
        scale: Optional[float] = None,
        use_qk_l2norm_in_kernel: bool = False,
        cu_seqlens: Optional[torch.Tensor] = None,
        disable_state_update: bool = False,
        disable_output_calculation: bool = False,
        intermediate_states_buffer: Optional[torch.Tensor] = None,
        cache_steps: Optional[int] = None,
        output: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """`(out, initial_state_source, intermediate_states_buffer)` as the call
        leaves them; mutates nothing. Token-sequential, fp32 throughout, each
        store rounded once. `disable_output_calculation` and `cache_steps` are
        ignored (the output is always computed; the buffer is indexed by its
        own step axis)."""
        H, K = k.shape[2], k.shape[3]
        HV, V = v.shape[2], v.shape[3]
        rep = HV // H
        scale = K**-0.5 if scale is None else scale
        if cu_seqlens is None:
            spans = [(b, 0, q.shape[1]) for b in range(q.shape[0])]
        else:
            cu = cu_seqlens.tolist()
            spans = [(0, cu[i], cu[i + 1]) for i in range(len(cu) - 1)]
        out = torch.zeros(v.shape, dtype=torch.float32, device=v.device)
        pool = None if initial_state_source is None else initial_state_source.clone()
        cache = None if intermediate_states_buffer is None else intermediate_states_buffer.clone()
        per_channel = beta.dim() == v.dim()
        for i, (b, start, end) in enumerate(spans):
            slot = int(initial_state_indices[i])
            if slot >= 0 and initial_state_source is not None:
                h = initial_state_source[slot].float().transpose(1, 2)  # [HV, K, V]
            else:
                h = torch.zeros(HV, K, V, dtype=torch.float32, device=v.device)
            for step, t in enumerate(range(start, end)):
                q_t = q[b, t].float().repeat_interleave(rep, dim=0)  # [HV, K]
                k_t = k[b, t].float().repeat_interleave(rep, dim=0)
                if use_qk_l2norm_in_kernel:
                    q_t = q_t / (q_t.square().sum(-1, keepdim=True).sqrt() + 1e-6)
                    k_t = k_t / (k_t.square().sum(-1, keepdim=True).sqrt() + 1e-6)
                q_t = q_t * scale
                h = h * torch.exp(g[b, t].float())[:, None, None]
                u = v[b, t].float() - (h * k_t[:, :, None]).sum(1)  # [HV, V]
                u = u * (beta[b, t].float() if per_channel else beta[b, t].float()[:, None])
                h = h + k_t[:, :, None] * u[:, None, :]
                out[b, t] = (h * q_t[:, :, None]).sum(1)
                if cache is not None and slot >= 0:
                    cache[slot, step] = h.transpose(1, 2).to(cache.dtype)
            if pool is not None and not disable_state_update and slot >= 0:
                pool[slot] = h.transpose(1, 2).to(pool.dtype)
        return out.to(q.dtype if output is None else output.dtype), pool, cache

    def is_valid(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        initial_state_source: Optional[torch.Tensor],
        initial_state_indices: torch.Tensor,
        scale: Optional[float] = None,
        use_qk_l2norm_in_kernel: bool = False,
        cu_seqlens: Optional[torch.Tensor] = None,
        disable_state_update: bool = False,
        disable_output_calculation: bool = False,
        intermediate_states_buffer: Optional[torch.Tensor] = None,
        cache_steps: Optional[int] = None,
        output: Optional[torch.Tensor] = None,
    ) -> None:
        assert not (output is not None and disable_output_calculation), (
            "output and disable_output_calculation=True are mutually exclusive: with the "
            "output calculation off the buffer is never written and the return value is a "
            "[1, 1, 1, 1] scratch tensor"
        )
        if output is not None:
            assert output.is_contiguous(), (
                f"output must be contiguous; got strides {output.stride()} for shape "
                f"{tuple(output.shape)} -- the kernel writes it with packed [B, T, HV, V] "
                "arithmetic, misplacing values and clobbering whatever lies between"
            )
        if initial_state_source is not None and not disable_state_update:
            assert initial_state_source.is_contiguous(), (
                f"initial_state_source must be contiguous when disable_state_update is False; "
                f"got strides {initial_state_source.stride()} -- a strided pool is copied "
                "before the call and the final state is written into the copy"
            )
        if intermediate_states_buffer is not None:
            assert intermediate_states_buffer.is_contiguous(), (
                f"intermediate_states_buffer must be contiguous; got strides "
                f"{intermediate_states_buffer.stride()} -- a strided buffer is copied before "
                "the call and every per-step state is written into the copy"
            )
            if cu_seqlens is None:
                assert cache_steps is not None and cache_steps >= q.shape[1], (
                    f"cache_steps must be at least the per-sequence token count when an "
                    f"intermediate_states_buffer is given; got cache_steps={cache_steps} for "
                    f"{q.shape[1]} tokens per sequence -- a smaller stride makes each "
                    "sequence's tail steps overwrite the next row's"
                )
            assert cache_steps == intermediate_states_buffer.shape[1], (
                f"cache_steps={cache_steps} must equal intermediate_states_buffer's step axis "
                f"({intermediate_states_buffer.shape[1]}) -- the kernel strides rows by "
                "cache_steps, so any other value misplaces every row past 0"
            )


fused_recurrent_gated_delta_rule_update = FusedRecurrentGatedDeltaRuleUpdate()
