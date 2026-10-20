# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Qwen3.5 MTP layer: routed experts quantized at load time for the TRTLLM-gen MoE backend.

The Qwen3.5-397B-A17B NVFP4 checkpoint keeps the MTP layer in BF16 (512 BF16 experts, 25.2 MB each). With
``TRTLLM_Q35_MTP_MOE_QUANT`` (default unset) the MTP layer's experts are built quantized on TRTLLM-gen instead of
BF16 CUTLASS, from the checkpoint's BF16 weights, without calibration data:

* ``fp8block``: FP8 E4M3 weights with one FP32 scale per 128 x 128 block (the DeepSeek-V3 layout,
  ``QuantAlgo.FP8_BLOCK_SCALES``); activations are quantized per 1 x 128 group at run time. Half the expert bytes.
* ``mxfp4``: MXFP4 (E2M1 values, one E8M0 power-of-two scale per 32 elements) weights with BF16 activations
  (``QuantAlgo.W4A16_MXFP4``). A quarter of the expert bytes.
* ``nvfp4``: NVFP4 weights and activations (``QuantAlgo.NVFP4``, the target experts' format: E2M1 values, one E4M3
  scale per 16 elements, one FP32 scale per tensor; ModelOpt's recipe, weight scales from the weights' amax). The
  experts then run exactly like the target layers' (trtllm-gen FP4 MoE, so the Q35 router GEMM + NVFP4 quant,
  TRTLLM_Q35_MOE_PREROUTE tables, FIN_FOLD and MOEGEMM_PIN apply to the MTP layer too). The activation global scales
  (the checkpoint's ``input_scale``s, which the MIXED_PRECISION export does not have for the BF16 MTP layer) come
  from TRTLLM_Q35_MTP_MOE_NVFP4_AMAX (FC1 input, FC2 input amax; calibrated defaults 256 / 2048, see below).

Only the draft model's weights change (the target is untouched), so outputs stay those of the target; the draft's
proposals - and with them the real acceptance length - can change: gate with the unforced AL checks.

The conversion happens in the experts' ``load_weights``: the checkpoint's fused BF16 tensors (``gate_up_proj``
[E, H, 2I], ``down_proj`` [E, I, H]) of this rank's experts are quantized and handed on in the per-expert (VANILLA)
key layout the quantized MoE methods load (``{e}.w1/w3/w2.weight`` + ``weight_scale_inv`` / ``weight_scale``).
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

import torch

ENV_MTP_MOE_QUANT = "TRTLLM_Q35_MTP_MOE_QUANT"
# Optional cap (default 0 = off): MoE inputs above this many tokens take a BF16 twin built the stock way (CUTLASS).
ENV_MTP_MOE_QUANT_MAX_M = "TRTLLM_Q35_MTP_MOE_QUANT_MAX_M"
DEFAULT_MTP_MOE_QUANT_MAX_M = 0
# Small-tile (tileN 8/16) FP8 block-scale MoE tactics are excluded (known issue): with fp8block, the runner never
# profiles or runs tileN < 32 (TRTLLM_Q35_FP8_MOE_MIN_TILEN, custom_ops/trtllm_gen_custom_ops.py); its cache-miss
# fallback already takes the widest valid tile.
#
# Guard slot (fp8block only; TRTLLM_Q35_MTP_MOE_QUANT_GUARD, default "w2" with fp8block, "0" = off): one zero expert
# slot is appended after the last local expert of the listed backend tensors and the parameters become views of the
# first n_local slots (same shapes and values; the memory right after the last expert is mapped and zero). This is a
# mapped-memory backstop for the FP8 block-scale MoE GEMMs at small per-rank intermediate sizes (known issue), not a
# kernel fix. Cost: one expert slot per guarded tensor (w2: H x I bytes) plus a transient copy at load.
ENV_MTP_MOE_QUANT_GUARD = "TRTLLM_Q35_MTP_MOE_QUANT_GUARD"
DEFAULT_MTP_MOE_QUANT_GUARD = "w2"
_GUARD_TENSORS = {
    "w31": "w3_w1_weight",
    "w31s": "w3_w1_weight_scaling_factor",
    "w2": "w2_weight",
    "w2s": "w2_weight_scaling_factor",
}
_MODES = ("fp8block", "mxfp4", "nvfp4")
FP8_BLOCK = 128
MXFP4_BLOCK = 32
NVFP4_BLOCK = 16
# NVFP4 global scale convention (ModelOpt / TRT-LLM): amax maps to E2M1 max (6) x E4M3 max (448).
_NVFP4_RANGE = 6.0 * 448.0
# TRTLLM_Q35_MTP_MOE_NVFP4_AMAX="<fc1>,<fc2>": the activation amax the MTP experts' NVFP4 input quantization is built
# for (FC1 input = the MoE input after post_attention_layernorm; FC2 input = the SwiGLU output). An activation above
# it saturates its block scale (E4M3 448) and is clipped; one far below it only gets a smaller block scale (E4M3
# keeps 3 mantissa bits down to 2^-6 of the range), so the defaults carry a wide margin over the calibrated maxima.
# Calibration (TRTLLM_Q35_MTP_MOE_CALIB on the BF16 MTP layer, TEP8 TP4xEP2, max over ranks, prefill + decode calls)
# over short math prompts and long-context multi-turn agentic prompts: FC1 input amax ~85, routed SwiGLU output amax
# ~400, with only a tiny fraction of rows near those maxima. The same probe reproduces the checkpoint's
# ModelOpt-calibrated FC1 amax of all 60 target layers within a few percent. Defaults: ~3x / ~5x margin.
ENV_MTP_MOE_NVFP4_AMAX = "TRTLLM_Q35_MTP_MOE_NVFP4_AMAX"
DEFAULT_MTP_MOE_NVFP4_AMAX = (256.0, 2048.0)
_E2M1_MAX = 6.0
_FP8_MAX = 448.0
# E2M1 magnitudes in code order 0..7.
_E2M1_VALUES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def mtp_moe_quant_mode() -> Optional[str]:
    mode = os.environ.get(ENV_MTP_MOE_QUANT, "").strip().lower()
    if not mode or mode in ("0", "off", "none"):
        return None
    if mode not in _MODES:
        raise ValueError(f"{ENV_MTP_MOE_QUANT}={mode!r}: expected one of {_MODES}")
    return mode


# ----------------------------------------------------------------------------------------------
# Calibration probe (debug, default off): TRTLLM_Q35_MTP_MOE_CALIB=<json path; "{rank}" -> global rank>. Records,
# with graph-capturable ops (so CUDA-graph decode replays count), the running maxima of every MoE layer's input and,
# for layers with BF16 experts (the stock MTP layer), of the SwiGLU output silu(w1 x) * (w3 x) of the routed (token,
# expert) pairs on this rank's experts / intermediate slice, plus exceedance counts for candidate global ranges.
# The max over ranks is the layer's amax (EP: other experts; MoE TP: other intermediate rows).
# ----------------------------------------------------------------------------------------------
ENV_MTP_MOE_CALIB = "TRTLLM_Q35_MTP_MOE_CALIB"
_CALIB_X_THR = (8.0, 16.0, 32.0, 64.0, 128.0, 256.0)
_CALIB_H_THR = (16.0, 32.0, 64.0, 128.0, 256.0, 512.0, 1024.0, 2048.0)
_CALIB_RING = 1 << 20
_calib_state: Dict = {}


def mtp_moe_calib_path() -> Optional[str]:
    v = os.environ.get(ENV_MTP_MOE_CALIB, "").strip()
    return v or None


def _calib_buffers(device, layer_idx: int):
    key = (device, layer_idx)
    st = _calib_state.get(key)
    if st is None:
        if torch.cuda.is_current_stream_capturing():
            return None
        # [x amax, routed h amax, all-local h amax, calls, rows, routed pairs, x rows over thr..., pairs over thr...]
        st = torch.zeros(
            6 + len(_CALIB_X_THR) + len(_CALIB_H_THR), dtype=torch.float64, device=device
        )
        _calib_state[key] = st
        if (
            "thr",
            device,
        ) not in _calib_state:  # created outside graph capture (no H2D copy while capturing)
            _calib_state[("thr", device)] = (
                torch.tensor(_CALIB_X_THR, dtype=torch.float64, device=device),
                torch.tensor(_CALIB_H_THR, dtype=torch.float64, device=device),
            )
        if not _calib_state.get("_atexit"):
            import atexit

            atexit.register(mtp_moe_calib_dump)
            _calib_state["_atexit"] = True
    return st


def mtp_moe_calib_record(
    layer_idx: Optional[int],
    experts,
    x: torch.Tensor,
    router_logits: Optional[torch.Tensor],
    top_k: int,
) -> None:
    """Record one MoE call (``x`` [M, H] the MoE input; ``router_logits`` [M, E] or None = input only)."""
    if layer_idx is None or x.dim() != 2 or x.shape[0] == 0:
        return
    st = _calib_buffers(x.device, int(layer_idx))
    if st is None:
        return
    nx = len(_CALIB_X_THR)
    thr_x, thr_h = _calib_state[("thr", x.device)]
    xr = x.abs().amax(dim=-1).double()  # [M]
    st[0] = torch.maximum(st[0], xr.amax())
    st[3] += 1
    st[4] += x.shape[0]
    st[6 : 6 + nx] += (xr[:, None] > thr_x).sum(dim=0)
    backend = getattr(experts, "backend", experts)
    w31 = getattr(backend, "w3_w1_weight", None)
    if router_logits is None or w31 is None or w31.dtype != torch.bfloat16 or w31.dim() != 3:
        return
    e_local, two_i, h = w31.shape
    inter = two_i // 2
    start = int(getattr(backend, "slot_start", 0))
    w = w31.view(e_local * two_i, h)
    for c in range(0, x.shape[0], 128):
        xc = x[c : c + 128]
        gu = torch.matmul(xc, w.t()).float().view(xc.shape[0], e_local, 2, inter)
        # rows [w3; w1]: up first, gate second
        hmax = (
            (torch.nn.functional.silu(gu[:, :, 1]) * gu[:, :, 0]).abs().amax(dim=-1)
        )  # [m, E_local]
        lid = router_logits[c : c + 128].float().topk(top_k, dim=-1).indices - start
        valid = (lid >= 0) & (lid < e_local)
        routed = (hmax.gather(1, lid.clamp(0, e_local - 1)) * valid).double()  # [m, K]
        st[1] = torch.maximum(st[1], routed.amax())
        st[2] = torch.maximum(st[2], hmax.amax().double())
        st[5] += valid.sum()
        st[6 + nx :] += (routed.reshape(-1)[:, None] > thr_h).sum(dim=0)
        # per-call ring of (rows, x amax, routed h amax) for the BF16-expert layers (distribution; the warmup's
        # calls come first and can be dropped offline)
        ring = _calib_state.get(("ring", x.device, int(layer_idx)))
        if ring is None and not torch.cuda.is_current_stream_capturing():
            ring = (
                torch.zeros((_CALIB_RING, 3), dtype=torch.float32, device=x.device),
                torch.zeros((1,), dtype=torch.int64, device=x.device),
            )
            _calib_state[("ring", x.device, int(layer_idx))] = ring
        if ring is not None:
            buf, cnt = ring
            row = torch.stack(
                [
                    torch.full((), float(xc.shape[0]), device=x.device),
                    xr[c : c + 128].amax().float(),
                    routed.amax().float(),
                ]
            )
            buf.index_copy_(0, torch.remainder(cnt, _CALIB_RING), row[None])
            cnt.add_(1)


def mtp_moe_calib_dump() -> None:
    path = mtp_moe_calib_path()
    if path is None:
        return
    import json

    rank = os.environ.get("OMPI_COMM_WORLD_RANK", os.environ.get("SLURM_PROCID", "0"))
    nx = len(_CALIB_X_THR)
    out = {"x_thresholds": list(_CALIB_X_THR), "h_thresholds": list(_CALIB_H_THR), "layers": {}}
    for key, st in list(_calib_state.items()):
        if not isinstance(key, tuple) or key[0] in ("thr", "ring"):
            continue
        v = st.tolist()
        out["layers"][str(key[1])] = {
            "x_amax": v[0],
            "h_amax_routed": v[1],
            "h_amax_all_local": v[2],
            "calls": v[3],
            "rows": v[4],
            "routed_pairs": v[5],
            "x_rows_over": v[6 : 6 + nx],
            "h_pairs_over": v[6 + nx :],
        }
    try:
        with open(path.replace("{rank}", str(rank)), "w") as f:
            json.dump(out, f)
        rings = {
            str(k[2]): (b.cpu(), int(c.item()))
            for k, (b, c) in _calib_state.items()
            if isinstance(k, tuple) and k[0] == "ring"
        }
        if rings:
            torch.save(rings, path.replace("{rank}", str(rank)).rsplit(".", 1)[0] + "_ring.pt")
    except (OSError, RuntimeError):
        pass


def mtp_moe_quant_max_m() -> int:
    return int(os.environ.get(ENV_MTP_MOE_QUANT_MAX_M, str(DEFAULT_MTP_MOE_QUANT_MAX_M)) or 0)


def mtp_moe_quant_guard(mode: Optional[str]) -> List[str]:
    """Backend tensor names that get a zero guard expert slot (``TRTLLM_Q35_MTP_MOE_QUANT_GUARD``): only for
    ``fp8block``; default ``w2``; ``0`` / ``off`` / ``none`` = no guard; ``all`` (or ``1``) = all four; else a comma
    list of ``w31``, ``w31s``, ``w2``, ``w2s``."""
    if mode != "fp8block":
        return []
    v = os.environ.get(ENV_MTP_MOE_QUANT_GUARD, DEFAULT_MTP_MOE_QUANT_GUARD).strip().lower()
    if v in ("", "0", "off", "none"):
        return []
    if v in ("1", "all"):
        return list(_GUARD_TENSORS.values())
    names = []
    for t in (t.strip() for t in v.split(",")):
        if not t:
            continue
        if t not in _GUARD_TENSORS:
            raise ValueError(
                f"{ENV_MTP_MOE_QUANT_GUARD}={v!r}: unknown tensor {t!r}, expected a comma list of "
                f"{tuple(_GUARD_TENSORS)} or 0 / all"
            )
        if _GUARD_TENSORS[t] not in names:
            names.append(_GUARD_TENSORS[t])
    return names


def guard_pad_expert_params(backend, names: List[str]) -> List[str]:
    """Append one zero expert slot after the last local expert of each named parameter of ``backend``; the parameter
    keeps its object, shape and values but its data becomes a view of the first n_local slots of the padded buffer.
    Returns the names padded (missing / 0-d parameters are skipped)."""
    padded_names = []
    for name in names:
        p = getattr(backend, name, None)
        if p is None or p.dim() < 1:
            continue
        padded = torch.zeros((p.shape[0] + 1, *p.shape[1:]), dtype=p.dtype, device=p.device)
        padded[:-1].copy_(p.data)
        p.data = padded[:-1]
        padded_names.append(name)
    return padded_names


def experts_unquantized(model_config, expert_quant_config) -> bool:
    """Whether a layer's routed experts load unquantized: no per-layer quant entry and a global quant config
    that does not quantize them (MIXED_PRECISION exports such as the Qwen3.5 NVFP4 checkpoint's MTP layer)."""
    from tensorrt_llm.quantization.mode import QuantAlgo

    if expert_quant_config is not None:
        return not expert_quant_config.layer_quant_mode.has_any_quant(exclude_kv_cache=True)
    qc = getattr(model_config, "quant_config", None)
    if qc is None or qc.quant_algo in (None, QuantAlgo.MIXED_PRECISION):
        return True
    return not qc.layer_quant_mode.has_any_quant(exclude_kv_cache=True)


def quant_config_for(mode: str, kv_cache_quant_algo=None):
    """The experts' ``QuantConfig`` for ``mode``."""
    from tensorrt_llm.models.modeling_utils import QuantConfig
    from tensorrt_llm.quantization.mode import QuantAlgo

    if mode == "nvfp4":
        return QuantConfig(
            quant_algo=QuantAlgo.NVFP4,
            kv_cache_quant_algo=kv_cache_quant_algo,
            group_size=NVFP4_BLOCK,
        )
    algo = {"fp8block": QuantAlgo.FP8_BLOCK_SCALES, "mxfp4": QuantAlgo.W4A16_MXFP4}[mode]
    return QuantConfig(quant_algo=algo, kv_cache_quant_algo=kv_cache_quant_algo)


def mtp_moe_nvfp4_amax() -> Tuple[float, float]:
    """(FC1 input amax, FC2 input amax) for the NVFP4 MTP experts' activation global scales."""
    v = os.environ.get(ENV_MTP_MOE_NVFP4_AMAX, "").strip()
    if not v:
        return DEFAULT_MTP_MOE_NVFP4_AMAX
    parts = [float(t) for t in v.split(",")]
    if len(parts) != 2 or not all(p > 0 for p in parts):
        raise ValueError(
            f"{ENV_MTP_MOE_NVFP4_AMAX}={v!r}: expected two positive floats '<fc1>,<fc2>'"
        )
    return parts[0], parts[1]


# ----------------------------------------------------------------------------------------------
# Quantizers ([N, K] BF16 -> quantized values + scales)
# ----------------------------------------------------------------------------------------------
def quantize_fp8_block(w: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """``w`` [N, K] -> (E4M3 [N, K], FP32 ``weight_scale_inv`` [N/128, K/128]) with ``w ~ q * scale`` per
    128 x 128 block (block absolute max mapped to 448). N and K must be multiples of 128."""
    n, k = w.shape
    assert n % FP8_BLOCK == 0 and k % FP8_BLOCK == 0, (n, k)
    blocks = w.float().view(n // FP8_BLOCK, FP8_BLOCK, k // FP8_BLOCK, FP8_BLOCK)
    amax = blocks.abs().amax(dim=(1, 3)).clamp_min(1e-12)
    scale = amax / _FP8_MAX
    q = (blocks / scale[:, None, :, None]).clamp(-_FP8_MAX, _FP8_MAX).to(torch.float8_e4m3fn)
    return q.view(n, k), scale.contiguous()


def dequantize_fp8_block(q: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    n, k = q.shape
    blocks = q.float().view(n // FP8_BLOCK, FP8_BLOCK, k // FP8_BLOCK, FP8_BLOCK)
    return (blocks * scale[:, None, :, None]).view(n, k)


def quantize_mxfp4(w: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """``w`` [N, K] -> (packed E2M1 uint8 [N, K/2], E8M0 uint8 scales [N, K/32]): per 32-element block the
    power-of-two scale ``2^ceil(log2(amax / 6))`` (no clipping), values rounded to the nearest E2M1 magnitude
    (ties to the even code). Element ``2j`` goes to the low nibble of byte ``j``."""
    n, k = w.shape
    assert k % MXFP4_BLOCK == 0, k
    blocks = w.float().view(n, k // MXFP4_BLOCK, MXFP4_BLOCK)
    amax = blocks.abs().amax(dim=-1).clamp_min(2.0**-120)
    exponent = torch.ceil(torch.log2(amax / _E2M1_MAX)).clamp(-127, 127)
    scaled = blocks / torch.exp2(exponent)[..., None]
    mag = scaled.abs().clamp(max=_E2M1_MAX)
    grid = torch.tensor(_E2M1_VALUES, device=w.device)
    # nearest magnitude; ties -> even code (the lower code when the code is odd)
    dist = (mag[..., None] - grid).abs()
    code = dist.argmin(dim=-1)  # first minimum = lower code on ties
    tie = torch.isclose(
        dist.gather(-1, code[..., None]).squeeze(-1),
        dist.gather(-1, (code + 1).clamp(max=7)[..., None]).squeeze(-1),
    ) & (code < 7)
    code = torch.where(tie & (code % 2 == 1), code + 1, code)
    code = code | ((scaled < 0) & (code > 0)).to(code.dtype) << 3
    code = code.to(torch.uint8).view(n, k)
    packed = (code[:, 0::2] | (code[:, 1::2] << 4)).contiguous()
    scales = (exponent + 127).to(torch.uint8).contiguous()
    return packed, scales


def dequantize_mxfp4(packed: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    n, half = packed.shape
    k = 2 * half
    lo = (packed & 0x0F).to(torch.int64)
    hi = (packed >> 4).to(torch.int64)
    code = torch.stack([lo, hi], dim=-1).view(n, k)
    grid = torch.tensor(_E2M1_VALUES, device=packed.device)
    val = grid[code & 7] * torch.where(code >= 8, -1.0, 1.0)
    scale = torch.exp2(scales.float() - 127.0)
    return (val.view(n, k // MXFP4_BLOCK, MXFP4_BLOCK) * scale[..., None]).view(n, k)


def _e2m1_codes(mag: torch.Tensor) -> torch.Tensor:
    """Magnitudes (>= 0, <= 6) -> E2M1 magnitude codes 0..7 (uint8), round to nearest, ties to the even code (what
    cvt.rn.satfinite.e2m1x2 does): boundaries 0.25 / 0.75 / 1.25 / 1.75 / 2.5 / 3.5 / 5, ties at odd-code midpoints
    go down (0.25, 1.25, 2.5, 5) and at even-code midpoints up (0.75, 1.75, 3.5)."""
    code = (mag > 0.25).to(torch.uint8)
    for thr, inclusive in (
        (0.75, True),
        (1.25, False),
        (1.75, True),
        (2.5, False),
        (3.5, True),
        (5.0, False),
    ):
        code += (mag >= thr if inclusive else mag > thr).to(torch.uint8)
    return code


# TRTLLM_Q35_MTP_MOE_NVFP4_MSE=1 (default off): per 16-element block, pick the E4M3 block scale among
# e4m3(block_amax / 6 / weight_scale_2 * f), f in _NVFP4_MSE_FACTORS, with the least squared weight error (the amax
# scale, f = 1, never clips; a slightly smaller one usually fits the E2M1 grid better). Weights only (activations
# are quantized by the kernels).
ENV_MTP_MOE_NVFP4_MSE = "TRTLLM_Q35_MTP_MOE_NVFP4_MSE"
_NVFP4_MSE_FACTORS = (1.0, 0.96, 0.92, 0.88, 0.84, 0.8, 0.76, 0.72)


def _nvfp4_codes(blocks: torch.Tensor, eff: torch.Tensor) -> torch.Tensor:
    """Signed E2M1 codes (uint8, sign in bit 3) of ``blocks`` [..., 16] for per-block multipliers ``eff`` [...]. A
    negative value that rounds to zero keeps its sign (-0, code 8), as cvt.rn.satfinite.e2m1x2 and the TRT-LLM
    quantization kernel produce it (same value as +0)."""
    scaled = torch.where(
        eff[..., None] > 0, blocks / eff[..., None].clamp_min(1e-30), torch.zeros_like(blocks)
    )
    code = _e2m1_codes(scaled.abs().clamp(max=_E2M1_MAX))
    return code | ((scaled < 0).to(torch.uint8) << 3)


def quantize_nvfp4(
    w: torch.Tensor, amax: Optional[float] = None, mse: Optional[bool] = None
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``w`` [N, K] -> (packed E2M1 uint8 [N, K/2], E4M3 block scales [N, K/16], FP32 ``weight_scale_2`` 0-d) in
    the ModelOpt NVFP4 checkpoint layout: ``weight_scale_2 = amax / (6 * 448)`` (``amax`` defaults to the tensor's),
    block scale ``e4m3(block_amax / 6 / weight_scale_2)``, ``w ~ e2m1 * block_scale * weight_scale_2``; element ``2j``
    in the low nibble of byte ``j``. ``mse`` (default: TRTLLM_Q35_MTP_MOE_NVFP4_MSE): least-squares block scales."""
    n, k = w.shape
    assert k % NVFP4_BLOCK == 0, k
    if mse is None:
        mse = os.environ.get(ENV_MTP_MOE_NVFP4_MSE, "0") == "1"
    wf = w.float()
    a = wf.abs().amax() if amax is None else torch.tensor(float(amax), device=w.device)
    ws2 = (a / _NVFP4_RANGE).clamp_min(1e-30).to(torch.float32)
    blocks = wf.view(n, k // NVFP4_BLOCK, NVFP4_BLOCK)
    base = blocks.abs().amax(dim=-1) / 6.0 / ws2
    sf = base.clamp(max=448.0).to(torch.float8_e4m3fn)
    code = _nvfp4_codes(blocks, sf.float() * ws2)
    if mse:
        grid = torch.tensor(_E2M1_VALUES + tuple(-v for v in _E2M1_VALUES), device=w.device)
        best = None
        for f in _NVFP4_MSE_FACTORS:
            sf_f = (base * f).clamp(max=448.0).to(torch.float8_e4m3fn)
            eff = sf_f.float() * ws2
            code_f = _nvfp4_codes(blocks, eff)
            deq = grid[code_f.long()] * eff[..., None]
            err = (deq - blocks).square().sum(dim=-1)
            if best is None:
                best, sf, code = err, sf_f, code_f
            else:
                better = err < best
                best = torch.where(better, err, best)
                sf = torch.where(better, sf_f.view(torch.uint8), sf.view(torch.uint8)).view(
                    torch.float8_e4m3fn
                )
                code = torch.where(better[..., None], code_f, code)
    code = code.view(n, k)
    packed = (code[:, 0::2] | (code[:, 1::2] << 4)).contiguous()
    return packed, sf.contiguous(), ws2.reshape(())


def dequantize_nvfp4(
    packed: torch.Tensor, scales: torch.Tensor, weight_scale_2: torch.Tensor
) -> torch.Tensor:
    n, half = packed.shape
    k = 2 * half
    lo = (packed & 0x0F).to(torch.int64)
    hi = (packed >> 4).to(torch.int64)
    code = torch.stack([lo, hi], dim=-1).view(n, k)
    grid = torch.tensor(_E2M1_VALUES, device=packed.device)
    val = grid[code & 7] * torch.where(code >= 8, -1.0, 1.0)
    scale = scales.view(torch.float8_e4m3fn).float() * weight_scale_2.float()
    return (val.view(n, k // NVFP4_BLOCK, NVFP4_BLOCK) * scale[..., None]).view(n, k)


def _quantize(w: torch.Tensor, mode: str) -> Tuple[torch.Tensor, torch.Tensor, str]:
    if mode == "fp8block":
        q, s = quantize_fp8_block(w)
        return q, s, "weight_scale_inv"
    q, s = quantize_mxfp4(w)
    return q, s, "weight_scale"


def convert_bf16_experts(
    weights: Dict, expert_ids: List[int], mode: str, device: torch.device
) -> Dict[str, torch.Tensor]:
    """Per-expert (VANILLA) quantized weights of ``expert_ids`` from the checkpoint's BF16 expert tensors, either
    fused (``gate_up_proj`` [E, H, 2I] gate first, ``down_proj`` [E, I, H], as the Qwen3.5 weight mapper hands
    them over) or per expert (``{e}.w1.weight`` / ``w3`` [I, H], ``{e}.w2.weight`` [H, I])."""
    out: Dict[str, torch.Tensor] = {}
    fused = "gate_up_proj" in weights
    if mode == "nvfp4":
        fc1_amax, fc2_amax = mtp_moe_nvfp4_amax()
        in1 = torch.tensor(fc1_amax / _NVFP4_RANGE, dtype=torch.float32)
        in2 = torch.tensor(fc2_amax / _NVFP4_RANGE, dtype=torch.float32)
    for e in expert_ids:
        if fused:
            gu = weights["gate_up_proj"][e].to(device=device, dtype=torch.bfloat16).transpose(0, 1)
            w1, w3 = gu.chunk(2, dim=0)  # gate, up: [I, H]
            w2 = weights["down_proj"][e].to(device=device, dtype=torch.bfloat16).transpose(0, 1)
        else:
            w1, w3, w2 = (
                weights[f"{e}.{n}.weight"].to(device=device, dtype=torch.bfloat16)
                for n in ("w1", "w3", "w2")
            )
        if mode == "nvfp4":
            # w1 / w3 share one per-tensor scale (the fused FC1 GEMM has one alpha per expert); the loader shards
            # the full-expert tensors and keeps the per-tensor scales.
            a13 = torch.maximum(w1.abs().amax(), w3.abs().amax()).float()
            for name, w, a, in_scale in (
                ("w1", w1, a13, in1),
                ("w3", w3, a13, in1),
                ("w2", w2, None, in2),
            ):
                q, sf, ws2 = quantize_nvfp4(w.contiguous(), amax=None if a is None else float(a))
                out[f"{e}.{name}.weight"] = q
                out[f"{e}.{name}.weight_scale"] = sf
                out[f"{e}.{name}.weight_scale_2"] = ws2.cpu()
                out[f"{e}.{name}.input_scale"] = in_scale.clone()
            continue
        for name, w in (("w1", w1), ("w3", w3), ("w2", w2)):
            q, s, scale_key = _quantize(w.contiguous(), mode)
            out[f"{e}.{name}.weight"] = q
            out[f"{e}.{name}.{scale_key}"] = s
    return out


def _is_bf16_expert_dict(weights: Dict) -> bool:
    if "gate_up_proj" in weights:
        return weights["gate_up_proj"].dtype == torch.bfloat16
    w1 = next((v for k, v in weights.items() if k.endswith(".w1.weight")), None)
    return w1 is not None and w1.dtype == torch.bfloat16


def wrap_experts_loader(experts, mode: str, bf16_twin=None) -> None:
    """Make the experts' ``load_weights`` take the checkpoint's BF16 expert tensors and load this rank's experts
    quantized (``mode``) in the per-expert (VANILLA) layout. ``experts`` must have been built with the matching
    quant config (``quant_config_for``).

    Both the module and, for a ``ConfigurableMoE``, its ``backend`` are wrapped: the model loader walks
    ``named_modules()`` and hands the expert weights to the backend module directly (the weight mapper's special
    MoE handling), while direct callers use the wrapper; converted (non-BF16) weights pass through unchanged, so
    a wrapper that delegates to its wrapped backend converts once."""
    backend = getattr(experts, "backend", experts)
    for target in (experts,) if backend is experts else (experts, backend):
        _wrap_load_weights(target, experts, backend, mode, bf16_twin)


def _wrap_load_weights(target, experts, backend, mode: str, bf16_twin=None) -> None:
    from .fused_moe.interface import MoEWeightLoadingMode

    load = target.load_weights

    def load_weights(weights: List[Dict], allow_partial_loading: bool = False):
        assert len(weights) == 1, len(weights)
        w = weights[0]
        if not _is_bf16_expert_dict(w):  # converted already, or a reload of quantized weights
            return load(weights, allow_partial_loading=allow_partial_loading)
        if bf16_twin is not None:
            # The BF16 twin (inputs above the M cap) loads the checkpoint tensors as they are, in the layout
            # the weight mapper set on this module.
            mode_in = getattr(target, "weight_loading_mode", None)
            twin_backend = getattr(bf16_twin, "backend", bf16_twin)
            if mode_in is not None:
                bf16_twin.weight_loading_mode = mode_in
                twin_backend.weight_loading_mode = mode_in
            twin_backend.load_weights([dict(w)], allow_partial_loading=allow_partial_loading)
        if getattr(backend, "layer_load_balancer", None):
            raise NotImplementedError(f"{ENV_MTP_MOE_QUANT} does not support the MoE load balancer")
        device = torch.device("cuda", torch.cuda.current_device())
        converted = convert_bf16_experts(w, list(backend.initial_local_expert_ids), mode, device)
        # The weight mapper sets the checkpoint's (fused) layout on the module; the converted
        # tensors use the per-expert one.
        experts.weight_loading_mode = MoEWeightLoadingMode.VANILLA
        backend.weight_loading_mode = MoEWeightLoadingMode.VANILLA
        result = load([converted], allow_partial_loading=allow_partial_loading)
        # The converting call is the only one that pads (a delegating wrapper's inner call passes through).
        guarded = guard_pad_expert_params(backend, mtp_moe_quant_guard(mode))
        if guarded:
            from tensorrt_llm.logger import logger

            logger.info(
                f"{ENV_MTP_MOE_QUANT}={mode}: zero guard expert slot after the last local expert of "
                f"{', '.join(guarded)} ({ENV_MTP_MOE_QUANT_GUARD})"
            )
        return result

    target.load_weights = load_weights


__all__ = [
    "ENV_MTP_MOE_QUANT",
    "ENV_MTP_MOE_QUANT_GUARD",
    "ENV_MTP_MOE_QUANT_MAX_M",
    "convert_bf16_experts",
    "guard_pad_expert_params",
    "mtp_moe_quant_guard",
    "mtp_moe_quant_max_m",
    "experts_unquantized",
    "dequantize_fp8_block",
    "dequantize_mxfp4",
    "dequantize_nvfp4",
    "ENV_MTP_MOE_NVFP4_AMAX",
    "ENV_MTP_MOE_NVFP4_MSE",
    "mtp_moe_nvfp4_amax",
    "quantize_nvfp4",
    "ENV_MTP_MOE_CALIB",
    "mtp_moe_calib_dump",
    "mtp_moe_calib_path",
    "mtp_moe_calib_record",
    "mtp_moe_quant_mode",
    "quant_config_for",
    "quantize_fp8_block",
    "quantize_mxfp4",
    "wrap_experts_loader",
]
