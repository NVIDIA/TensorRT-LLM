# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fast fused SiTU MoE backend for Kimi K3 routed experts.

This module wraps the private-fork FlashInfer trtllm-gen fused MoE with
the SiTU activation (``ActivationType.Situ == 9``) for the K3 routed
expert compute. It uses the **native MXFP4 path**
(``trtllm_fp4_block_scale_routed_moe`` with ``MxE2m1`` weights × BF16
activations) — the checkpoint's ``mxfp4-pack-quantized`` expert weights
(E2M1 nibbles + per-32-group E8M0 scales) are the kernel's native
encoding, so **no mxfp4→mxint4 value conversion is needed**, only a
layout shuffle done once at load time.

Verified facts about the private SiTU cubin pool
(``local_cubins/20260617_v0613rc1_situ_v0611_barrierfix``):

* ``Bmm_Bfloat16_MxE2m1Bfloat16_castBfloat16_*siTuGlu*`` cubins exist
  (sm100f — family compatible with sm_103/GB300 — plus sm103a builds).
* SiTuGlu kernel formula (``GemmGatedActOptions.h``)::

      left  = alpha * tanh(x0 / alpha) * sigmoid(x0)   # SiTU gate
      right = beta  * tanh(x1 / beta)                  # "lin" up
      out   = left * right

  ``alpha``/``beta`` come from per-expert fp32 ``gemm1_alpha`` /
  ``gemm1_beta`` tensors; K3 uses alpha=4.0, beta=25.0.

Routing: K3's noaux_tc gate (sigmoid + selection-only
``e_score_correction_bias``, renormalize over raw sigmoid scores,
``routed_scaling_factor``) is computed on the host by the caller
(:class:`~.kimi_k3_moe_gate.KimiK3MoEGate`) — this backend consumes
**precomputed** ``(topk_idx, topk_weights)`` via the kernel's
``UnpackedPrecomputed`` routing mode, whose contract
(``RoutingKernel.h::DataBase``) applies the provided weights verbatim
(no in-kernel re-normalization).

EP: the kernel natively supports a contiguous local expert slice via
``local_expert_offset`` / ``local_num_experts``; global expert ids are
passed unchanged in ``topk_idx`` and tokens whose experts are all
remote contribute zeros to the output (caller allreduces partials).

Environment requirements (see ``scripts/moe_fast/RESULTS.md`` and
``setup_snapshot_env.sh`` for the full story):

* ``FLASHINFER_PRIVATE_CUBIN_DIR`` must point at the SiTU pool
  **before** ``import flashinfer`` (jit_env snapshots it at import).
* ``import flashinfer`` must resolve to the private-fork snapshot
  (``exisiting_optimization_work/trtllmgen_MOE``) — either via
  ``PYTHONPATH`` or :func:`ensure_snapshot_flashinfer` **before**
  anything else (including tensorrt_llm) imports flashinfer.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Optional, Tuple

import torch

# K3 SiTU activation constants (activation_situ_beta / _linear_beta).
K3_SITU_ALPHA = 4.0
K3_SITU_LINEAR_BETA = 25.0

_EPILOGUE_TILE_M = 128


# ---------------------------------------------------------------------------
# Snapshot flashinfer import management.
# ---------------------------------------------------------------------------

def _default_snapshot_dir() -> str:
    """Snapshot location: explicit env > derived from the optimization-work
    checkout (``KIMI_K3_OPT_WORK_DIR``, see examples/kimi_k3/setup_env.sh)."""
    explicit = os.environ.get("KIMI_K3_FLASHINFER_SNAPSHOT_DIR")
    if explicit:
        return explicit
    opt_work = os.environ.get("KIMI_K3_OPT_WORK_DIR", "")
    return os.path.join(opt_work, "trtllmgen_MOE") if opt_work else ""


DEFAULT_SNAPSHOT_DIR = _default_snapshot_dir()


def ensure_snapshot_flashinfer(snapshot_dir: Optional[str] = None) -> "module":
    """Import (or validate) the private-fork flashinfer snapshot.

    Must run before anything else imports ``flashinfer``. If flashinfer
    is already imported, verifies it is the snapshot (has ``is_private``
    support) and raises otherwise.

    Returns the imported ``flashinfer`` module.
    """
    snapshot_dir = snapshot_dir or DEFAULT_SNAPSHOT_DIR
    if "flashinfer" in sys.modules:
        fi = sys.modules["flashinfer"]
    else:
        if snapshot_dir and os.path.isdir(snapshot_dir) and snapshot_dir not in sys.path:
            sys.path.insert(0, snapshot_dir)
        import flashinfer as fi  # noqa: F401
    import inspect

    sig = inspect.signature(fi.fused_moe.trtllm_fp4_block_scale_routed_moe)
    if "is_private" not in sig.parameters:
        raise RuntimeError(
            "imported flashinfer lacks is_private support — it is the stock "
            f"package, not the private snapshot. Loaded from: {fi.__file__}. "
            f"Put the snapshot ({snapshot_dir}) first on sys.path/PYTHONPATH "
            "before flashinfer is first imported."
        )
    return fi


# ---------------------------------------------------------------------------
# Weight preparation: checkpoint MXFP4 → kernel shuffled layout.
# ---------------------------------------------------------------------------


@dataclass
class FusedSituExpertWeights:
    """Shuffled kernel-layout weights for a contiguous local expert slice.

    * ``gemm1_weights``  uint8 ``[E_local, 2*I, H//2]`` (shuffled rows)
    * ``gemm1_scales``   float8_e4m3fn-view of UE8M0 bytes,
      ``[E_local, 2*I, H//32]`` (shuffled + block-interleaved)
    * ``gemm2_weights``  uint8 ``[E_local, H, I//2]``
    * ``gemm2_scales``   float8_e4m3fn view, ``[E_local, H, I//32]``
    """

    gemm1_weights: torch.Tensor
    gemm1_scales: torch.Tensor
    gemm2_weights: torch.Tensor
    gemm2_scales: torch.Tensor
    hidden_size: int
    intermediate_size: int
    num_local_experts: int
    gate_first: bool


def prepare_fused_situ_weights(
    w1_packed: torch.Tensor,
    w1_scales: torch.Tensor,
    w3_packed: torch.Tensor,
    w3_scales: torch.Tensor,
    w2_packed: torch.Tensor,
    w2_scales: torch.Tensor,
    *,
    gate_first: bool = False,
    device: Optional[torch.device] = None,
) -> FusedSituExpertWeights:
    """Convert KimiK3RoutedExpertBank-shaped MXFP4 buffers to kernel layout.

    Inputs (checkpoint / bank encoding, ``group_size=32``):

    * ``w1_packed`` / ``w3_packed``: uint8 ``[E, I, H//2]`` e2m1 nibble
      pairs (low nibble = even element) — w1 is the SiTU gate, w3 the
      linear up projection (HF ``KimiBlockSparseMLP`` naming).
    * ``w1_scales`` / ``w3_scales``: uint8 ``[E, I, H//32]`` E8M0.
    * ``w2_packed``: uint8 ``[E, H, I//2]``; ``w2_scales`` uint8
      ``[E, H, I//32]``.

    ``gate_first=False`` (default, VERIFIED on GB300 against the fp32
    reference: cos=0.999994) places w3 (up/linear) rows in the first
    half and w1 (SiTU gate) rows in the second half of the fused gemm1
    matrix — the same convention as SwiGlu in the snapshot tests
    (activation applied to the second half). The kernel applies
    ``reorder_rows_for_gated_act_gemm`` interleaving on top.

    Layout work (per expert): gated-act row reorder + shuffle for
    transposed-MMA epilogue (weights and scales) + block-scale
    interleave of the scale matrix. Values are copied bit-exact — no
    re-quantization.
    """
    fi = ensure_snapshot_flashinfer()
    from flashinfer.fp4_quantization import block_scale_interleave  # type: ignore
    from flashinfer.fused_moe.core import (  # type: ignore
        _maybe_get_cached_w3_w1_permute_indices,
        get_w2_permute_indices_with_cache,
    )

    assert w1_packed.dtype == torch.uint8 and w2_packed.dtype == torch.uint8
    E, I, H_half = w1_packed.shape
    H = H_half * 2
    assert w2_packed.shape == (E, H, I // 2), (w2_packed.shape, (E, H, I // 2))
    assert w1_scales.shape == (E, I, H // 32)
    assert w2_scales.shape == (E, H, I // 32)

    if device is None:
        device = torch.device("cuda")

    # Fused gemm1 = [gate; up] (or swapped) along the row dim.
    if gate_first:
        gemm1_w = torch.cat([w1_packed, w3_packed], dim=1)
        gemm1_s = torch.cat([w1_scales, w3_scales], dim=1)
    else:
        gemm1_w = torch.cat([w3_packed, w1_packed], dim=1)
        gemm1_s = torch.cat([w3_scales, w1_scales], dim=1)

    cache: dict = {}
    g1w_out = torch.empty(E, 2 * I, H_half, dtype=torch.uint8, device=device)
    g1s_out = torch.empty(E, 2 * I, H // 32, dtype=torch.uint8, device=device)
    g2w_out = torch.empty(E, H, I // 2, dtype=torch.uint8, device=device)
    g2s_out = torch.empty(E, H, I // 32, dtype=torch.uint8, device=device)

    for e in range(E):
        w = gemm1_w[e].to(device, non_blocking=True)
        s = gemm1_s[e].to(device, non_blocking=True)
        p_w = _maybe_get_cached_w3_w1_permute_indices(cache, w, _EPILOGUE_TILE_M)
        g1w_out[e] = w[p_w.to(device)]
        p_s = _maybe_get_cached_w3_w1_permute_indices(
            cache, s, _EPILOGUE_TILE_M, num_elts_per_sf=16
        )
        g1s_out[e] = block_scale_interleave(s[p_s.to(device)].contiguous()).reshape(
            2 * I, H // 32
        )

        w2 = w2_packed[e].to(device, non_blocking=True)
        s2 = w2_scales[e].to(device, non_blocking=True)
        p_w2 = get_w2_permute_indices_with_cache(cache, w2, _EPILOGUE_TILE_M)
        g2w_out[e] = w2[p_w2.to(device)]
        p_s2 = get_w2_permute_indices_with_cache(
            cache, s2, _EPILOGUE_TILE_M, num_elts_per_sf=16
        )
        g2s_out[e] = block_scale_interleave(s2[p_s2.to(device)].contiguous()).reshape(
            H, I // 32
        )

    return FusedSituExpertWeights(
        gemm1_weights=g1w_out,
        gemm1_scales=g1s_out.view(torch.float8_e4m3fn),
        gemm2_weights=g2w_out,
        gemm2_scales=g2s_out.view(torch.float8_e4m3fn),
        hidden_size=H,
        intermediate_size=I,
        num_local_experts=E,
        gate_first=gate_first,
    )


# ---------------------------------------------------------------------------
# Forward.
# ---------------------------------------------------------------------------


def fused_situ_moe_forward(
    hidden_states: torch.Tensor,
    topk_idx: torch.Tensor,
    topk_weights: torch.Tensor,
    weights: FusedSituExpertWeights,
    *,
    num_experts: int,
    top_k: int,
    local_expert_offset: int = 0,
    situ_alpha: float = K3_SITU_ALPHA,
    situ_linear_beta: float = K3_SITU_LINEAR_BETA,
    output: Optional[torch.Tensor] = None,
    tune_max_num_tokens: int = 8192,
) -> torch.Tensor:
    """Fused MXFP4×BF16 SiTU MoE over the local expert slice.

    Parameters
    ----------
    hidden_states
        bf16 ``[T, hidden]`` — K3 routed latent (hidden = 3584).
    topk_idx
        int32 ``[T, top_k]`` **global** expert ids from KimiK3MoEGate.
    topk_weights
        fp32 (or bf16) ``[T, top_k]`` final combine weights — already
        renormalized and multiplied by ``routed_scaling_factor``.
        Applied verbatim by the kernel (cast to bf16).
    weights
        Prepared local-slice weights (:func:`prepare_fused_situ_weights`).
    num_experts / top_k
        Global expert count (896) and top-k (16).
    local_expert_offset
        Global id of this rank's first local expert
        (= ep_rank * num_local_experts for a contiguous slice).

    Returns
    -------
    bf16 ``[T, hidden]`` partial output covering local experts only
    (zeros for tokens with no local expert); caller allreduces.
    """
    fi = ensure_snapshot_flashinfer()
    from flashinfer.tllm_enums import ActivationType  # type: ignore

    assert hidden_states.dtype == torch.bfloat16, hidden_states.dtype
    assert hidden_states.dim() == 2 and hidden_states.shape[1] == weights.hidden_size
    T = hidden_states.shape[0]
    assert topk_idx.shape == (T, top_k) and topk_weights.shape == (T, top_k)

    dev = hidden_states.device
    E_local = weights.num_local_experts
    ones = torch.ones(E_local, dtype=torch.float32, device=dev)
    alpha = torch.full((E_local,), float(situ_alpha), dtype=torch.float32, device=dev)
    beta = torch.full(
        (E_local,), float(situ_linear_beta), dtype=torch.float32, device=dev
    )

    out_list = fi.fused_moe.trtllm_fp4_block_scale_routed_moe(
        (topk_idx.to(torch.int32), topk_weights.to(torch.bfloat16)),
        None,  # routing_bias (selection bias already applied host-side)
        hidden_states,
        None,  # hidden_states_scale — bf16 activations
        weights.gemm1_weights,
        weights.gemm1_scales,
        None,  # gemm1_bias
        alpha,  # gemm1_alpha → SiTuGlu alpha (4.0)
        beta,  # gemm1_beta → SiTuGlu beta (25.0)
        None,  # gemm1_clamp_limit
        weights.gemm2_weights,
        weights.gemm2_scales,
        None,  # gemm2_bias
        ones,  # output1_scale_scalar
        ones,  # output1_scale_gate_scalar
        ones,  # output2_scale_scalar
        num_experts,
        top_k,
        None,  # n_group
        None,  # topk_group
        weights.intermediate_size,
        local_expert_offset,
        E_local,
        None,  # routed_scaling_factor — already folded into topk_weights
        1,  # routing_method_type: Renormalize (ignored for precomputed weights)
        True,  # do_finalize
        None,  # enable_pdl (auto)
        int(ActivationType.Situ.value),
        None,  # per_token_scale
        output,
        tune_max_num_tokens,
        True,  # is_private — select the SiTU cubin pool
    )
    return output if output is not None else out_list[0]


# ---------------------------------------------------------------------------
# Pure-torch batched fallback (no flashinfer): grouped dequant + bmm over
# ACTIVE local experts only. ~vectorized replacement for the per-expert
# python loop in kimi_k3_moe_block._moe_infer.
# ---------------------------------------------------------------------------


def _dequant_mxfp4_bf16(packed: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """Vectorized MXFP4 dequant to bf16. packed [..., N//2], scales [..., N//32]."""
    lut = torch.tensor(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
        dtype=torch.float32,
        device=packed.device,
    )
    lo = (packed & 0x0F).to(torch.long)
    hi = (packed >> 4).to(torch.long)
    vals = torch.stack([lut[lo], lut[hi]], dim=-1).reshape(*packed.shape[:-1], -1)
    scale = torch.exp2(scales.to(torch.float32) - 127.0)
    n = vals.shape[-1]
    g = n // scales.shape[-1]
    vals = vals.reshape(*scales.shape, g) * scale.unsqueeze(-1)
    return vals.reshape(*packed.shape[:-1], n).to(torch.bfloat16)


def batched_torch_situ_moe_forward(
    hidden_states: torch.Tensor,
    topk_idx: torch.Tensor,
    topk_weights: torch.Tensor,
    w1_packed: torch.Tensor,
    w1_scales: torch.Tensor,
    w3_packed: torch.Tensor,
    w3_scales: torch.Tensor,
    w2_packed: torch.Tensor,
    w2_scales: torch.Tensor,
    *,
    local_expert_offset: int = 0,
    situ_alpha: float = K3_SITU_ALPHA,
    situ_linear_beta: float = K3_SITU_LINEAR_BETA,
) -> torch.Tensor:
    """Fallback: grouped dequant+GEMM over active local experts only.

    Same contract as :func:`fused_situ_moe_forward` but takes the raw
    bank buffers (local slice, ``[E_local, ...]``). Dequantizes only
    experts that received tokens this step and runs one segment-GEMM
    pass. bf16 math, fp32 activation, fp32 combine.
    """
    T, H = hidden_states.shape
    E_local = w1_packed.shape[0]
    device = hidden_states.device

    flat_idx = topk_idx.reshape(-1).to(torch.long) - local_expert_offset
    flat_w = topk_weights.reshape(-1).to(torch.float32)
    valid = (flat_idx >= 0) & (flat_idx < E_local)
    out = torch.zeros(T, H, dtype=torch.float32, device=device)
    if not bool(valid.any()):
        return out.to(torch.bfloat16)

    sel = valid.nonzero(as_tuple=True)[0]
    tok = sel // topk_idx.shape[1]
    exp = flat_idx[sel]
    order = torch.argsort(exp)
    tok, exp, wts = tok[order], exp[order], flat_w[sel][order]

    active, counts = torch.unique_consecutive(exp, return_counts=True)
    x = hidden_states[tok]  # [S, H] grouped by expert

    # Dequant only active experts.
    w1 = _dequant_mxfp4_bf16(w1_packed[active], w1_scales[active])  # [A, I, H]
    w3 = _dequant_mxfp4_bf16(w3_packed[active], w3_scales[active])
    w2 = _dequant_mxfp4_bf16(w2_packed[active], w2_scales[active])  # [A, H, I]

    starts = torch.cumsum(counts, 0) - counts
    seg_out = torch.empty(x.shape[0], H, dtype=torch.float32, device=device)
    starts_l = starts.tolist()
    counts_l = counts.tolist()
    for a in range(len(active)):
        s, c = starts_l[a], counts_l[a]
        xs = x[s : s + c]
        gate = (xs @ w1[a].t()).to(torch.float32)
        up = (xs @ w3[a].t()).to(torch.float32)
        act = (
            situ_alpha
            * torch.tanh(gate / situ_alpha)
            * torch.sigmoid(gate)
            * (situ_linear_beta * torch.tanh(up / situ_linear_beta))
        )
        seg_out[s : s + c] = (act.to(torch.bfloat16) @ w2[a].t()).to(torch.float32)

    out.index_add_(0, tok, seg_out * wts.unsqueeze(-1))
    return out.to(torch.bfloat16)
