# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""DSV4 fused q_a path: replace [kv_a_proj q-slice GEMM -> q_a_layernorm -> fp8_quant] with a
single fused fp8-out GEMM that emits (qr_fp8, qr_sf) directly for q_b_proj (and the indexer's wq_b).

Enabled by env TRTLLM_DSV4_FUSED_QA=1. The fp8-out GEMM kernel (2-CTA cluster, raw PTX) is JIT-built
from dsv4_fused_qa_csrc/*.cu via torch cpp_extension on first use. gamma (q_a_layernorm weight) is folded
offline into the q-slice weight; the RMS normalization is dropped (validated lossless, GSM8K 96.10).

Kernel I/O (validated byte-exact vs fp8_quantize_1x128_packed_ue8m0):
  fused_qa_fp8_out(A_fp8[M,K], A_sf, B_fp8[N,K], B_sf) -> (D_fp8[M,N], D_sf_packed[num_packed_sf_k, m_aligned])
  A_sf / B_sf: packed UE8M0 int32, physical [sf_k, lead] (data_ptr = buffer start). Requires M%4==0, N%4==0.
"""

import os

import torch
import torch.nn.functional as F

_EXT = None
_DIR = os.path.join(os.path.dirname(__file__), "dsv4_fused_qa_csrc")


def _load_ext():
    global _EXT
    if _EXT is None:
        from torch.utils.cpp_extension import load

        # Build for both SM100 (B200) and SM103 (B300) — the 2-CTA cluster/tcgen05 cubin must
        # stamp the running arch or cudaFuncSetAttribute fails with "no kernel image".
        _EXT = load(
            name="dsv4_fused_qa_op",
            sources=[os.path.join(_DIR, "fp8_gemm_quant_op.cu")],
            extra_cuda_cflags=[
                "-O3",
                "--use_fast_math",
                "-gencode",
                "arch=compute_100a,code=sm_100a",
                "-gencode",
                "arch=compute_103a,code=sm_103a",
            ],
            extra_ldflags=["-lcuda"],
            verbose=False,
        )
    return _EXT


# ---------------------------------------------------------------------------
# torch custom op (graph-capturable). Returns (qr_fp8 [M,N], qr_sf physical [num_packed_sf_k, m_aligned]).
# ---------------------------------------------------------------------------
@torch.library.custom_op("dsv4_fuse::fused_qa_fp8_out", mutates_args=())
def fused_qa_fp8_out(
    a_fp8: torch.Tensor, a_sf: torch.Tensor, b_fp8: torch.Tensor, b_sf: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    d_fp8, d_sf = _load_ext().fp8_gemm_quant_out(a_fp8, a_sf, b_fp8, b_sf)
    return d_fp8, d_sf


@fused_qa_fp8_out.register_fake
def _(a_fp8, a_sf, b_fp8, b_sf):
    M, K = a_fp8.shape
    N = b_fp8.shape[0]
    num_n_blocks = (N + 127) // 128
    num_packed_sf_k = (num_n_blocks + 3) // 4
    m_aligned = (M + 3) // 4 * 4
    d_fp8 = a_fp8.new_empty((M, N), dtype=torch.float8_e4m3fn)
    d_sf = a_fp8.new_empty((num_packed_sf_k, m_aligned), dtype=torch.int32)
    return d_fp8, d_sf


# ---------------------------------------------------------------------------
# Weight prep (offline / lazy at first forward). All UE8M0, matching deep_gemm.
# ---------------------------------------------------------------------------
def _ceil_log2_pow2_e8m0(amax: torch.Tensor):
    """amax [..] -> (e8m0 byte uint8, quant_scale=2^e fp32) for smallest 2^e >= amax/448 (no saturation)."""
    s = (amax.float() / 448.0).clamp_min(1e-10)
    e = torch.ceil(torch.log2(s))  # smallest integer e with 2^e >= s
    byte = (e + 127.0).clamp(0, 255).to(torch.uint8)
    return byte, torch.exp2(e)


def requant_128x128_ue8m0(w_bf16: torch.Tensor):
    """w [N,K] bf16 -> (fp8 [N,K], e8m0 byte [nb, kb])  (128x128 block UE8M0, deep_gemm weight format)."""
    N, K = w_bf16.shape
    nb, kb = (N + 127) // 128, (K + 127) // 128
    wp = F.pad(w_bf16.float(), (0, kb * 128 - K, 0, nb * 128 - N))
    amax = wp.view(nb, 128, kb, 128).abs().amax(dim=(1, 3))  # [nb, kb]
    byte, scale = _ceil_log2_pow2_e8m0(amax)  # [nb, kb]
    scale_full = scale.repeat_interleave(128, 0).repeat_interleave(128, 1)[:N, :K]
    fp8 = (w_bf16.float() / scale_full).to(torch.float8_e4m3fn)
    return fp8, byte


def weight_scale_128x128_to_fused_sfb(byte_nbkb: torch.Tensor, N: int, K: int) -> torch.Tensor:
    """e8m0 [nb,kb] -> kernel sfb [sfb_k, N] int32 (broadcast 128x128 -> per-N, pack 4 K-blocks/uint32)."""
    nb, kb = byte_nbkb.shape
    sfb_k = (kb + 3) // 4
    per_n = byte_nbkb.repeat_interleave(128, 0)[:N].to(torch.int32)  # [N, kb]
    packed = torch.zeros((N, sfb_k), dtype=torch.int32, device=byte_nbkb.device)
    for j in range(kb):
        packed[:, j // 4] |= (per_n[:, j] & 0xFF) << ((j % 4) * 8)
    return packed.t().contiguous()  # [sfb_k, N]


def dequant_weight_128x128(w_fp8: torch.Tensor, byte_nbkb: torch.Tensor) -> torch.Tensor:
    """fp8 [N,K] + e8m0 [nb,kb] -> bf16 [N,K]."""
    N, K = w_fp8.shape
    scale = torch.exp2(byte_nbkb.float() - 127.0)
    scale_full = scale.repeat_interleave(128, 0).repeat_interleave(128, 1)[:N, :K]
    return (w_fp8.float() * scale_full).bfloat16()


def deep_gemm_nt_out(
    a_fp8: torch.Tensor,
    a_sf: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    out_dtype,
    M_valid: int,
    disable_ue8m0_cast: bool = True,
) -> torch.Tensor:
    """deep_gemm.fp8_gemm_nt with a PRE-quantized activation (a_fp8 [Mfull,K], a_sf packed-UE8M0 view),
    against an fp8 module weight + its scale. Returns out[:M_valid] [.., N]. Used to feed q_b_proj /
    indexer.wq_b the fused qr fp8 directly (skip their internal _fp8_quantize)."""
    from tensorrt_llm import deep_gemm

    Mfull, N = a_fp8.shape[0], weight.shape[0]
    out = a_fp8.new_empty((Mfull, N), dtype=out_dtype)
    deep_gemm.fp8_gemm_nt(
        (a_fp8, a_sf), (weight, weight_scale), out, disable_ue8m0_cast=disable_ue8m0_cast
    )
    return out[:M_valid]


def dequant_qr_to_bf16(qr_fp8: torch.Tensor, qr_sf_view: torch.Tensor, K: int) -> torch.Tensor:
    """fp8 [M,K] + packed-UE8M0 a_sf view [M, num_packed] -> bf16 [M,K] (graph-capturable)."""
    num_kb = (K + 127) // 128
    kb = torch.arange(num_kb, device=qr_fp8.device)
    packed = qr_sf_view[:, kb // 4].to(torch.int64)
    byte = (packed >> ((kb % 4) * 8).to(torch.int64)) & 0xFF
    scale = torch.exp2(byte.float() - 127.0)
    return (qr_fp8.float() * scale.repeat_interleave(128, dim=1)[:, :K]).bfloat16()


def run_fused_qa_qpath(hidden_bf16: torch.Tensor, W_q_fp8: torch.Tensor, W_q_sfb: torch.Tensor):
    """Compute the gamma-folded q-slice as fp8 directly. Pads M to a multiple of 256 (2-CTA cluster
    granularity — required for correctness), runs the fused fp8-out GEMM, slices back.

    Returns (qr_fp8 [M, q_lora_rank], qr_sf_view [M, num_packed] strided over physical [num_packed, m_aligned],
    hidden_fp8 [Mpad, K], hidden_sf) — the quantized (padded) hidden is returned for reuse by the kv GEMM.
    """
    from tensorrt_llm._torch.custom_ops.torch_custom_ops import _fp8_quantize_1x128_ue8m0

    M = hidden_bf16.shape[0]
    Mpad = (M + 255) // 256 * 256
    hidden_p = hidden_bf16 if Mpad == M else F.pad(hidden_bf16, (0, 0, 0, Mpad - M))
    hp_fp8, hp_sf = _fp8_quantize_1x128_ue8m0(hidden_p, tactic=0)
    qr_fp8_p, qr_sf_p = torch.ops.dsv4_fuse.fused_qa_fp8_out(hp_fp8, hp_sf, W_q_fp8, W_q_sfb)
    qr_fp8 = qr_fp8_p[:M].contiguous()
    # physical [num_packed, m_aligned(Mpad)] -> deep_gemm A-scale view [M, num_packed] stride (1, Mpad)
    qr_sf_view = qr_sf_p[:, :M].t()
    return qr_fp8, qr_sf_view, hp_fp8, hp_sf, Mpad


def build_fused_qa_weights(
    kv_a_weight_fp8: torch.Tensor,
    kv_a_weight_scale: torch.Tensor,
    gamma: torch.Tensor,
    q_lora_rank: int,
):
    """Split kv_a_proj weight at q_lora_rank, fold gamma into the q-slice (drop RMS), and produce both
    the fused-op B operands (W_q_folded fp8 + sfb) and the bf16 kv-slice GEMM weight.

    kv_a_weight_scale is the deep_gemm 128x128 UE8M0 byte tensor [nb, kb] for the full [2112, K] weight.
    Returns dict with W_q fp8/sfb (for fused op) and W_kvrope fp8 + scale (for the bf16 kv GEMM).
    """
    N_full, K = kv_a_weight_fp8.shape
    q_nb = q_lora_rank // 128  # 1536/128 = 12 (block-aligned)
    # --- q-slice: dequant, fold gamma per row, requant 128x128 ---
    W_q_fp8 = kv_a_weight_fp8[:q_lora_rank]
    W_q_byte = kv_a_weight_scale[:q_nb]
    W_q_bf16 = dequant_weight_128x128(W_q_fp8, W_q_byte)
    W_q_folded = (W_q_bf16.float() * gamma.float().unsqueeze(1)).bfloat16()
    Wqf_fp8, Wqf_byte = requant_128x128_ue8m0(W_q_folded)
    Wqf_sfb = weight_scale_128x128_to_fused_sfb(Wqf_byte, q_lora_rank, K)
    # --- kv+rope slice: keep as-is (bf16 GEMM) ---
    W_kvrope_fp8 = kv_a_weight_fp8[q_lora_rank:].contiguous()
    W_kvrope_byte = kv_a_weight_scale[q_nb:].contiguous()
    return {
        "W_q_fp8": Wqf_fp8.contiguous(),
        "W_q_sfb": Wqf_sfb,
        "W_kvrope_fp8": W_kvrope_fp8,
        "W_kvrope_scale": W_kvrope_byte,
    }
