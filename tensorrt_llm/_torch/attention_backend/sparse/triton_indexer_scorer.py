# Triton fused MXFP4 lightning-indexer MQA-logits scorer for SM120 (perf step 2).
# Replaces the pure-torch _torch_(paged_)mqa_logits_sm120 einsum, which
# materializes a [Nq, n_heads, Nk] fp32 accumulator (~64 GiB at 32K -> OOM) and
# a dequantized k_f. This kernel dequantizes MXFP4 inline and reduces over heads
# on the fly, emitting only the [Nq, Nk] logits in tiles.
#
# In-kernel MXFP4: byte p (0..63) holds logical dims 2p (low nibble) and 2p+1
# (high nibble), both in scale-block p//16, so
#   q.k = sum_p q_lo[p]*k_lo[p] + sum_p q_hi[p]*k_hi[p]   (two [.,64]@[64,.] dots)
# with each nibble's value = LUT[code&7] * (code&8 ? -1:1) * 2^(scale[p//16]-127).
# Validated (accuracy_eval/scorer_triton_mxfp4.py) vs the _dequant_mxfp4 reference:
# top-512 index overlap 100%, rel ~1e-6, kv up to 15000.
import torch
import triton
import triton.language as tl

_LUT = [0., 0.5, 1., 1.5, 2., 3., 4., 6.]  # E2M1 magnitudes for code&7

# Per-device cache of the E2M1 magnitude LUT as a device tensor. Building it via
# torch.tensor([...]) is a host->device copy that is ILLEGAL inside a CUDA-graph
# capture (same hazard the _dequant_mxfp4 LUT hit, fixed in 7f1d6de39). The cache
# is populated on the first (eager) call -- the eager prefill/decode warmup that
# precedes capture -- so the captured decode only indexes into an existing tensor.
_LUT_CACHE = {}


def _get_lut(device):
    t = _LUT_CACHE.get(device)
    if t is None:
        t = torch.tensor(_LUT, device=device, dtype=torch.float32)
        _LUT_CACHE[device] = t
    return t


@triton.jit
def _mqa_logits_fused_mxfp4(
        qc_ptr, qs_ptr,            # q codes [Nq,H,64] u8, q scale [Nq,H,4] u8
        kc_ptr, ksc_ptr,           # k codes [Nk,64] u8,   k scale [Nk,4] u8
        w_ptr, ksb_ptr, keb_ptr,   # weights [Nq,H] f32, bounds [Nq] int32
        lut_ptr, out_ptr,          # lut [8] f32, out [Nq,Nk] f32
        Nq, Nk, H,
        HALF: tl.constexpr, BLOCK_Q: tl.constexpr, BLOCK_S: tl.constexpr):
    pid_q = tl.program_id(0)
    pid_s = tl.program_id(1)
    offq = pid_q * BLOCK_Q + tl.arange(0, BLOCK_Q)
    offs = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
    offp = tl.arange(0, HALF)          # byte positions 0..63
    sblk = offp // 16                  # scale-block per byte (0..3)
    qm = offq < Nq
    sm = offs < Nk

    kb = tl.load(kc_ptr + offs[:, None] * HALF + offp[None, :],
                 mask=sm[:, None], other=0).to(tl.int32)
    klo_c = kb & 0xF
    khi_c = (kb >> 4) & 0xF
    klo = tl.load(lut_ptr + (klo_c & 7))
    khi = tl.load(lut_ptr + (khi_c & 7))
    klo = tl.where((klo_c & 8) != 0, -klo, klo)
    khi = tl.where((khi_c & 8) != 0, -khi, khi)
    ksb = tl.load(ksc_ptr + offs[:, None] * 4 + sblk[None, :],
                  mask=sm[:, None], other=0).to(tl.int32)
    ksf = (ksb << 23).to(tl.float32, bitcast=True)          # 2^(b-127)
    klo_t = tl.trans(klo * ksf)                             # [64, BS]
    khi_t = tl.trans(khi * ksf)

    logits = tl.zeros((BLOCK_Q, BLOCK_S), dtype=tl.float32)
    for h in range(H):
        qb = tl.load(qc_ptr + offq[:, None] * (H * HALF) + h * HALF + offp[None, :],
                     mask=qm[:, None], other=0).to(tl.int32)
        qlo_c = qb & 0xF
        qhi_c = (qb >> 4) & 0xF
        qlo = tl.load(lut_ptr + (qlo_c & 7))
        qhi = tl.load(lut_ptr + (qhi_c & 7))
        qlo = tl.where((qlo_c & 8) != 0, -qlo, qlo)
        qhi = tl.where((qhi_c & 8) != 0, -qhi, qhi)
        qsb = tl.load(qs_ptr + offq[:, None] * (H * 4) + h * 4 + sblk[None, :],
                      mask=qm[:, None], other=0).to(tl.int32)
        qsf = (qsb << 23).to(tl.float32, bitcast=True)
        dot = (tl.dot(qlo * qsf, klo_t, input_precision="ieee")
               + tl.dot(qhi * qsf, khi_t, input_precision="ieee"))
        dot = tl.maximum(dot, 0.0)
        wh = tl.load(w_ptr + offq * H + h, mask=qm, other=0.0)
        logits += wh[:, None] * dot
    ksv = tl.load(ksb_ptr + offq, mask=qm, other=0)
    kev = tl.load(keb_ptr + offq, mask=qm, other=0)
    keep = (offs[None, :] >= ksv[:, None]) & (offs[None, :] < kev[:, None])
    logits = tl.where(keep, logits, float("-inf"))
    tl.store(out_ptr + offq[:, None] * Nk + offs[None, :], logits,
             mask=qm[:, None] & sm[None, :])


def mqa_logits_mxfp4(qc, qs, kc, ksc, weights, ks, ke, BLOCK_Q=64, BLOCK_S=32):
    """Fused MXFP4 weighted-ReLU-dot logits. qc [Nq,H,64] u8, qs [Nq,H,4] u8,
    kc [Nk,64] u8, ksc [Nk,4] u8, weights [Nq,H], ks/ke [Nq] int32.
    Returns [Nq, Nk] f32 with -inf outside [ks, ke)."""
    Nq, H, _ = qc.shape
    Nk = kc.shape[0]
    lut = _get_lut(qc.device)  # cached; no per-call H2D copy (capture-safe)
    out = torch.empty(Nq, Nk, device=qc.device, dtype=torch.float32)
    grid = (triton.cdiv(Nq, BLOCK_Q), triton.cdiv(Nk, BLOCK_S))
    # num_stages=1: the H loop holds the k tiles + transposes live; pipelining
    # exceeds the ~101 KB SM120 shared-memory budget.
    _mqa_logits_fused_mxfp4[grid](
        qc.contiguous(), qs.contiguous(), kc.contiguous(), ksc.contiguous(),
        weights.contiguous().float(), ks.int(), ke.int(), lut, out,
        Nq, Nk, H, HALF=qc.shape[-1], BLOCK_Q=BLOCK_Q, BLOCK_S=BLOCK_S,
        num_stages=1, num_warps=4)
    return out
