# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""FlashInfer GDN prefill adapter for ``Qwen3NextGatedDeltaNet.forward_extend``.

Exposes ``chunk_gated_delta_rule`` with a signature call-compatible with the
vendored Triton ``tensorrt_llm._torch.modules.fla.chunk.chunk_gated_delta_rule``.

Differences vs the Triton path are absorbed inside this wrapper:
  * Layout: TRT-LLM ``[1, T, H, D]`` -> FlashInfer packed ``[T, H, D]``.
  * Forget-gate space: TRT-LLM's Triton ``chunk_gated_delta_rule`` consumes
    ``g`` in **log space** (``fla/chunk.py`` doc: "(forget) gating tensor (in
    log space!)"); FlashInfer's ``chunk_gated_delta_rule`` consumes the same
    quantity in **linear space** (default 1.0 = no decay; alpha = exp(log_g)).
    The wrapper converts ``g_linear = exp(g_log)`` before calling FlashInfer.
  * Pre-L2-normalize Q/K when ``use_qk_l2norm_in_kernel=True``
    (the FlashInfer prefill kernel does NOT apply L2 norm internally; the
    ``use_qk_l2norm_in_kernel`` parameter on ``flashinfer.chunk_gated_delta_rule``
    is currently a dead arg, see ``flashinfer/gdn_prefill.py:317-356``).
  * Pre-gather and post-scatter of indexed SSM state into FlashInfer's packed
    ``[num_seqs, H, V, K]`` layout. TRT-LLM's GDN state pool uses the same
    ``[N, H, V, K]`` logical layout, so the adapter gathers/scatters without
    transposing the last two dims. The SM100/SM103 kernel carries the recurrent
    state in fp32 in TMEM regardless of the initial/output-state I/O dtype, so
    the round-trip stays in the native pool dtype (bf16/fp16) there with no
    precision change; only SM90/SM120 need an fp32 up-cast/down-cast.

This module is only imported when ``TLLM_USE_FLASHINFER_GDN_PREFILL=1`` is set
at process start; do not import it lazily inside hot paths.
"""

from typing import Optional, Tuple

import torch

from tensorrt_llm._torch.modules.fla.fused_state_io import (
    cast_scatter_fp32_vk_to_vk,
    gather_cast_vk_to_fp32_vk,
)
from tensorrt_llm._torch.modules.fla.l2norm import l2norm_fwd
from tensorrt_llm._utils import is_sm_100f


# Mirror the @torch.compiler.disable on the legacy Triton wrapper
# (chunk.py:119): Dynamo must not trace this wrapper because it imports
# `flashinfer` lazily and calls into FI's CuTe-DSL kernels, neither of
# which compiles cleanly.
@torch.compiler.disable
def chunk_gated_delta_rule(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: Optional[float] = None,
    initial_state: Optional[torch.Tensor] = None,
    initial_state_indices: Optional[torch.Tensor] = None,
    inplace_indexed_state_update: bool = False,
    output_final_state: bool = False,
    cu_seqlens: Optional[torch.Tensor] = None,
    head_first: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
    output: Optional[torch.Tensor] = None,
    state_workspace: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
    cu_seqlens_cpu: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Adapter for FlashInfer's ``chunk_gated_delta_rule``.

    Gathers the indexed SSM state into FlashInfer's packed layout, runs the
    kernel, then scatters the final state back (or returns it). ``final_state``
    is non-``None`` only with ``output_final_state=True`` and no in-place
    update; an in-place update reports through ``initial_state``.
    ``state_workspace`` optionally supplies an (input, output) tensor pair,
    each shaped ``[capacity, H, V, K]`` in the state I/O dtype.
    Returned states never alias this scratch. A batch with more sequences than
    ``capacity`` runs in groups of at most ``capacity`` sequences, which needs
    ``cu_seqlens_cpu``: a host copy of ``cu_seqlens``.
    """
    # FlashInfer is imported lazily so importing this module on a non-FlashInfer
    # build does not error until the function is actually called.
    import flashinfer

    # --- Step 1: pre-flight asserts --------------------------------------
    assert head_first is False, "head_first=True is not supported by this wrapper"
    assert q.dim() == 4 and q.shape[0] == 1, f"q must be [1, T, H_q, D_k], got {tuple(q.shape)}"
    assert k.shape[2] == q.shape[2], (
        f"num_q_heads ({q.shape[2]}) must equal num_k_heads ({k.shape[2]})"
    )
    assert q.dtype in (torch.bfloat16, torch.float16), f"q dtype must be bf16/fp16, got {q.dtype}"
    assert g.dtype == torch.float32, f"g must be fp32, got {g.dtype}"
    assert cu_seqlens is not None, "cu_seqlens is required (varlen mode)"
    assert initial_state is not None, "initial_state is required"
    if inplace_indexed_state_update:
        assert initial_state_indices is not None, (
            "inplace_indexed_state_update=True requires initial_state_indices"
        )

    # --- Step 2: layout [1, T, H, D] -> [T, H, D] ------------------------
    # q/k/v are slices of mixed_qkv produced by torch.split on the last dim,
    # so the squeeze(0) view is *non-contiguous*. The .contiguous() here pays
    # a real copy and is required (FlashInfer reads contiguous TMA tiles).
    q3 = q.squeeze(0).contiguous()
    k3 = k.squeeze(0).contiguous()
    v3 = v.squeeze(0).contiguous()
    # Convert g from Triton's log-space convention to FlashInfer's linear-space
    # alpha (default 1.0 = no decay). FlashInfer's prefill kernel multiplies the
    # SSM state by ``g`` directly each chunk; passing log-space values produces
    # NaN. ``torch.exp`` always returns a fresh contiguous tensor, so no
    # explicit .contiguous() is needed.
    g2 = torch.exp(g.squeeze(0))
    # ``.to(torch.float32)`` allocates a fresh contiguous fp32 tensor when the
    # source is bf16 (the common case), or returns the (already contiguous)
    # input view when beta is already fp32 — no .contiguous() needed.
    beta2 = beta.squeeze(0).to(torch.float32)

    # --- Step 3: emulate use_qk_l2norm_in_kernel -------------------------
    # Use the fused Triton l2norm_fwd (eps=1e-6 matches the Triton/FI decode
    # kernels) instead of F.normalize, which incurs extra kernel launches and
    # an intermediate buffer at short ISL.
    if use_qk_l2norm_in_kernel:
        q3 = l2norm_fwd(q3)
        k3 = l2norm_fwd(k3)

    # --- Step 4: state I/O dtype ----------------------------------------
    # TRT-LLM's GDN kernels and FlashInfer both use [N, H, V, K] state layout.
    # The SM100/SM103 kernel carries the recurrent state in fp32 in TMEM
    # regardless of the initial/output-state I/O dtype (the state tensors are
    # only the gmem load/store format), so passing bf16/fp16 state there is
    # numerically identical to the fp32 round-trip while moving half the bytes.
    # SM90/SM120 still require fp32 state.
    state_dtype = initial_state.dtype if is_sm_100f() else torch.float32
    num_seqs = cu_seqlens.shape[0] - 1
    if state_workspace is not None:
        assert len(state_workspace) == 2
        # Both buffers hold one group of sequences, so they share one capacity.
        assert state_workspace[0].shape == state_workspace[1].shape
        for buffer in state_workspace:
            assert buffer.shape[0] >= 1
            assert buffer.shape[1:] == initial_state.shape[1:]
            assert buffer.dtype == state_dtype
            assert buffer.device == initial_state.device

    # FI 0.6.10 accepts `output=` / `output_state=`; pre-allocating skips its
    # internal `torch.empty` per call. `num_o_heads` per FI docstring is
    # max(num_q_heads, num_v_heads) — equivalently num_v_heads for GVA.
    # Only allocate / request final state when a caller actually consumes it
    # (either inplace scatter back to the SSM pool, or return to the caller);
    # otherwise FI skips the final-state aggregation entirely.
    total_seq_len = q3.shape[0]
    num_o_heads = max(q3.shape[1], v3.shape[1])
    head_size = q3.shape[2]
    need_state = inplace_indexed_state_update or output_final_state
    output_buf = (
        output.squeeze(0)
        if output is not None
        else q3.new_empty(total_seq_len, num_o_heads, head_size)
    )
    final_to_return: Optional[torch.Tensor] = None
    if output_final_state and not inplace_indexed_state_update:
        final_to_return = torch.empty(
            num_seqs,
            num_o_heads,
            head_size,
            head_size,
            dtype=initial_state.dtype,
            device=initial_state.device,
        )

    # A workspace smaller than the batch is used for consecutive groups of
    # sequences. Sequences are independent, so each group gathers, runs and
    # scatters its own states; group token ranges come from host offsets.
    if state_workspace is None or num_seqs <= state_workspace[0].shape[0]:
        groups = [(0, num_seqs, 0, total_seq_len)]
    else:
        assert cu_seqlens_cpu is not None, (
            "cu_seqlens_cpu is required when the batch exceeds the state workspace"
        )
        assert cu_seqlens_cpu.shape == cu_seqlens.shape
        # Split evenly: as few groups as fit, without a small remainder group.
        num_groups = -(-num_seqs // state_workspace[0].shape[0])
        size, extra = divmod(num_seqs, num_groups)
        bounds = [0]
        for group in range(num_groups):
            bounds.append(bounds[-1] + size + (group < extra))
        offsets = cu_seqlens_cpu.tolist()
        groups = [(lo, hi, offsets[lo], offsets[hi]) for lo, hi in zip(bounds[:-1], bounds[1:])]

    for seq_lo, seq_hi, tok_lo, tok_hi in groups:
        whole = len(groups) == 1
        group_seqs = seq_hi - seq_lo
        if whole:
            group_state, group_indices = initial_state, initial_state_indices
        elif initial_state_indices is None:
            group_state, group_indices = initial_state[seq_lo:seq_hi], None
        else:
            group_state, group_indices = initial_state, initial_state_indices[seq_lo:seq_hi]
        group_cu = cu_seqlens if whole else cu_seqlens[seq_lo : seq_hi + 1] - tok_lo
        q_g, k_g, v_g = q3[tok_lo:tok_hi], k3[tok_lo:tok_hi], v3[tok_lo:tok_hi]
        g_g, beta_g = g2[tok_lo:tok_hi], beta2[tok_lo:tok_hi]
        output_g = output_buf[tok_lo:tok_hi]

        # --- Step 5: gather initial state (+ cast dtype only when required)
        # Fuse gather (+ optional cast) and contiguous into a single Triton kernel.
        gathered_init = gather_cast_vk_to_fp32_vk(
            group_state,
            group_indices,
            out_dtype=state_dtype,
            output=state_workspace[0][:group_seqs] if state_workspace is not None else None,
        )

        # --- Step 6: call FlashInfer with pre-allocated output/state buffers
        if need_state:
            # Match the initial-state dtype (native bf16/fp16 on SM100/SM103, else
            # fp32); FlashInfer writes the final state in this dtype and the scatter
            # below adapts to the destination pool dtype without an extra cast.
            state_buf = (
                state_workspace[1][:group_seqs]
                if state_workspace is not None
                else q3.new_empty(group_seqs, num_o_heads, head_size, head_size, dtype=state_dtype)
            )
            _, out_state = flashinfer.chunk_gated_delta_rule(
                q=q_g,
                k=k_g,
                v=v_g,
                g=g_g,
                beta=beta_g,
                scale=scale,
                initial_state=gathered_init,
                output_final_state=True,
                cu_seqlens=group_cu,
                use_qk_l2norm_in_kernel=False,  # dead param in FlashInfer; we already normalized
                output=output_g,
                output_state=state_buf,
            )
        else:
            # FI returns a single tensor (not a tuple) when output_final_state=False.
            flashinfer.chunk_gated_delta_rule(
                q=q_g,
                k=k_g,
                v=v_g,
                g=g_g,
                beta=beta_g,
                scale=scale,
                initial_state=gathered_init,
                output_final_state=False,
                cu_seqlens=group_cu,
                use_qk_l2norm_in_kernel=False,
                output=output_g,
            )
            out_state = None

        # --- Step 7: cast state back (if needed), scatter / return -----
        # Fuse cast (out_state.dtype -> destination dtype; a no-op on SM100/SM103
        # where both are the native pool dtype) + optional indexed scatter into a
        # single Triton pass, mirroring Step 5. The inplace branch writes only the
        # slots named by ``initial_state_indices`` and leaves the rest untouched.
        if inplace_indexed_state_update:
            cast_scatter_fp32_vk_to_vk(out_state, group_state, group_indices)
        elif output_final_state:
            cast_scatter_fp32_vk_to_vk(out_state, final_to_return[seq_lo:seq_hi], None)

    out_packed = output_buf

    # --- Step 8: restore output layout ----------------------------------
    out = out_packed.unsqueeze(0)

    # --- Step 9: return -------------------------------------------------
    return out, final_to_return
