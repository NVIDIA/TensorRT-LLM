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
#: Default pool and buffer rows of the test's builds.
SLOTS = 8


def _verify(n, p, why, **spec):
    return Cell(why=why, spec=dict(mode="verify", n=n, p=p, **spec))


class FusedRecurrentGatedDeltaRuleUpdate(OpWrapper):
    """Advance each sequence's `[K, V]` recurrent state by *all* its tokens, in
    order. Per token and v-head `hv` (q/k head `hv // (HV // H)`):

        q, k = q / sqrt(q.q + 1e-6), k / sqrt(k.k + 1e-6)    # if use_qk_l2norm_in_kernel
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
    # | "write_back"), n sequences x p tokens, pool_slots, cache_rows, slots
    # (None: arange(n)), pool_dtype / cache_dtype, beta ("fp32" | "bf16":
    # sigmoid of a bf16 projection, as the target passes it), qkv ("views"
    # over a packed [T, 10240] row block | "contiguous", as the target passes
    # them).
    CELLS: tuple[Cell, ...] = tuple(
        _verify(
            n,
            p,
            f"{label}target verify: n={n} p={p}, n-row gather, 32-row buffer",
            pool_slots=n,
            cache_rows=ENGINE_VERIFY_ROWS,
            beta="bf16",
            qkv="contiguous",
            pool_dtype=dtype,
            cache_dtype=dtype,
        )
        for dtype, label, n, p in (
            (torch.float32, "", 32, 2),
            (torch.float32, "", 8, 4),
            (torch.bfloat16, "bf16 state: ", 32, 2),
        )
    )

    note = """
    Metadata consumed: none; every piece of state is an argument.

    Call site (`_gdn_verify`, the legacy verify path of `TRTLLM_USE_GDN_REPLAY=0`, `p` up
    to 4), all positional: `(q, k, v, g, beta, src, int_idx, None, True, None, True, False,
    int_ssm, p, None)` -- q/k/v the contiguous `[n, p, 16|48, 128]` reshapes of
    `pack_gdn_decode_qkv`'s outputs (so the in-kernel L2 norm is required), `g` fp32 from
    `fused_gdn_gating`, `beta = sigmoid(b)` bf16, `src` an `[n, 48, 128, 128]` gather of
    the live slots and `int_ssm` the layer's `[32, p, 48, 128, 128]` per-step buffer, both
    at the pool's dtype (fp32 or bf16, as `mamba_ssm_cache_dtype` allocated it; the buffer's
    states are rounded once to it), `int_idx = intermediate_state_indices[:n]` (int32
    arange). The buffer's rows are the batch rows `update_mamba_states` reads, not slot
    ids, and its step axis is `tokens_per_gen_step`: `p` (as `cache_steps`) must equal it.

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

    The first call in a fresh process is bitwise the second (measured on the
    write-back sink; the kernel has no autotuner).

    Certified: the call site's form alone -- the per-step buffer sink with the pool
    untouched, at 32 sequences of P = 2 and 8 of P = 4 over an fp32 pool and buffer, and
    32 of P = 2 over a bf16 one. Not certified: the write-back sink
    (`disable_state_update=False`), alone or with the buffer; the varlen form
    (`cu_seqlens`); negative, scattered or int64 indices; no pool; fp32 or per-channel
    `beta`; q/k/v as views of the packed rows; an `output` buffer;
    `disable_output_calculation=True`; `K` or `V` other than 128; fp16/fp32 q/k/v; P = 3
    and P outside 2..4; N above 32.

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
                    q_t = q_t / torch.sqrt(q_t.square().sum(-1, keepdim=True) + 1e-6)
                    k_t = k_t / torch.sqrt(k_t.square().sum(-1, keepdim=True) + 1e-6)
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
