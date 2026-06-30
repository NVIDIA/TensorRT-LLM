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
  * Pre-gather and post-scatter of indexed SSM state (FlashInfer requires
    packed ``[num_seqs, H, V, K]`` fp32 initial/output state). TRT-LLM's GDN
    state pool uses the same ``[N, H, V, K]`` logical layout, so the adapter
    casts/gathers/scatters without transposing the last two dims.

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
    use_qk_l2norm_in_kernel: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Adapter for FlashInfer's chunk_gated_delta_rule."""
    # FlashInfer is imported lazily so importing this module on a non-FlashInfer
    # build does not error until the function is actually called.
    import flashinfer

    # --- Step 1: pre-flight asserts --------------------------------------
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

    # --- Step 4: gather initial state and cast dtype ---------------------
    # TRT-LLM's GDN kernels and FlashInfer both use [N, H, V, K] state layout.
    # Fuse gather + cast-to-fp32 + contiguous into a single Triton kernel.
    gathered_init = gather_cast_vk_to_fp32_vk(initial_state, initial_state_indices)

    # --- Step 5+6: call FlashInfer with pre-allocated output/state buffers
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
    output_buf = q3.new_empty(total_seq_len, num_o_heads, head_size)
    if need_state:
        num_seqs = cu_seqlens.shape[0] - 1
        state_buf = q3.new_empty(num_seqs, num_o_heads, head_size, head_size, dtype=torch.float32)
        out_packed, out_state = flashinfer.chunk_gated_delta_rule(
            q=q3,
            k=k3,
            v=v3,
            g=g2,
            beta=beta2,
            scale=scale,
            initial_state=gathered_init,
            output_final_state=True,
            cu_seqlens=cu_seqlens,
            use_qk_l2norm_in_kernel=False,  # dead param in FlashInfer; we already normalized
            output=output_buf,
            output_state=state_buf,
        )
    else:
        # FI returns a single tensor (not a tuple) when output_final_state=False.
        out_packed = flashinfer.chunk_gated_delta_rule(
            q=q3,
            k=k3,
            v=v3,
            g=g2,
            beta=beta2,
            scale=scale,
            initial_state=gathered_init,
            output_final_state=False,
            cu_seqlens=cu_seqlens,
            use_qk_l2norm_in_kernel=False,
            output=output_buf,
        )
        out_state = None

    # --- Step 7: cast state back, scatter / return ---------------------
    # Fuse cast (fp32 -> initial_state.dtype) + optional indexed scatter into a
    # single Triton pass, mirroring Step 4. The inplace branch writes only the
    # slots named by ``initial_state_indices`` and leaves the rest untouched.
    if inplace_indexed_state_update:
        cast_scatter_fp32_vk_to_vk(out_state, initial_state, initial_state_indices)
        final_to_return: Optional[torch.Tensor] = None
    elif output_final_state:
        num_seqs_out, num_h_out, v_out, k_out = out_state.shape
        final_to_return = torch.empty(
            num_seqs_out,
            num_h_out,
            v_out,
            k_out,
            dtype=initial_state.dtype,
            device=out_state.device,
        )
        cast_scatter_fp32_vk_to_vk(out_state, final_to_return, None)
    else:
        final_to_return = None

    # --- Step 8: restore output layout ----------------------------------
    out = out_packed.unsqueeze(0)

    # --- Step 9: return -------------------------------------------------
    return out, final_to_return
