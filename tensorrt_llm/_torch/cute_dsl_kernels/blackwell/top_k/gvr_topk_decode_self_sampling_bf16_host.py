# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Native BF16 self-sampling decode dispatch and CUDA-graph-safe launchers.

FP32 uses the existing host and device modules. BF16 preserves the count-verified
GVR threshold ladder, with packed loads, exact BF16 refinement and a separate
launch cache. No input conversion is performed here.
"""

import operator
import threading
from collections.abc import Sequence
from types import ModuleType

import torch

from . import gvr_topk_decode_self_sampling_host as _fp32
from .gvr_topk_decode_self_sampling_host import (
    BLKC,
    CMPC,
    NB,
    QUADC,
    default_workspace,
    kernel_view,
    route_streaming,
    validate_run_ws,
)

_F32 = torch.float32
_I32 = torch.int32
_TENSOR = torch.Tensor
_index = operator.index
_is_capturing = torch.cuda.is_current_stream_capturing
_GVR_MAX_DEV = _fp32.GVR_MAX_DEV
_ws_hot = _fp32._ws_keep
_VARLEN_CACHE_BF16: dict[tuple[int, ...], tuple] = {}
_WARMUP_DONE: set[tuple[int, ...]] = set()
_COMPILE_LOCK = threading.RLock()


def _device() -> ModuleType:
    from . import gvr_topk_decode_self_sampling

    return gvr_topk_decode_self_sampling


def _bf16_halve_u(plan: dict) -> dict:
    """bf16 streaming arms read 8-element 16B vectors, so one vector covers
    what two fp32 float4s did: halve the per-thread vector batch U (tuple slot
    1) to keep per-tile ELEMENT coverage — and the 32-bit classify mask —
    identical while halving the load-instruction count."""
    tpl = list(plan["tpl"])
    tpl[1] = max(int(tpl[1]) // 2, 1)
    plan["tpl"] = tuple(tpl)
    return plan


def _gvr_main_gate(blk: int, kpt: int, split: bool = True) -> tuple[int, int]:
    """(SCPB, CMPB) constexpr mirror of GvrMainKernel.__init__ (gvr_topk_ss.py
    :1551-1552).  Kept here so a route rung can check what it does to the
    in-kernel degeneracy gate before changing BLK/KPT."""
    kbig = kpt >= 2 and kpt * blk >= 2048
    scpb = (8192 if split else 16384) if blk >= 1024 else 8192 if kbig else 4096
    cmpb = (4096 if kbig else 2048) if blk >= 1024 else 1024
    return (scpb, cmpb)


def _degen_gate_ok(blk_old: int, kpt_old: int, blk_new: int, kpt_new: int) -> bool:
    """True iff moving (blk_old, kpt_old) -> (blk_new, kpt_new) does not shrink
    either half of the degeneracy gate (OP56 G1)."""
    s_o, c_o = _gvr_main_gate(blk_old, kpt_old)
    s_n, c_n = _gvr_main_gate(blk_new, kpt_new)
    return s_n >= s_o and c_n >= c_o


def route_bf16(
    b: int, n: int, npad: int, k: int, num_sms: int = 148, sm_version: int = 100
) -> dict[str, object]:
    """bf16 dispatch table. route() is a pure function of shape, so the fp32
    table is the correct starting point; bf16-specific re-tunes (16B-vector U
    halving, register-family capacity fitting, bin-count halving) are applied
    to the returned copy only."""
    plan = _fp32.route(b, n, npad, k, num_sms, sm_version)
    if plan["kernel"] in ("reg", "regimg") and k == 512 and (n <= 1280) and (b > 148):
        plan["tpl"] = (256, 1, 8, 1, True, True, False, 512)
        plan["block"] = 256
    pk16 = npad <= 65536 and n <= 65536 and (b > 15)
    if plan["kernel"] == "main" and (not plan["tpl"][5]) and (b > 148):
        n4q = n >> 2
        if 1024 < n4q <= 1152:
            CMP = n if n < 2560 else 2560
            QC = QUADC
            CURE = not (n < 2 * k and b > 148)
            DEGE = n <= 3 * k or n <= 4 * k + 64
            if DEGE and CMP < n:
                CMP = n
            vptx = 4
            while vptx > 1 and (512 * (vptx // 2) >= n4q or 512 * (vptx // 2) * 4 + 512 >= n):
                vptx //= 2
            if DEGE:
                tpl = (512, vptx, 4, 1, CURE, True, False, 512)
            else:
                kpt = 1 if k <= 512 else 2
                tpl = (512, vptx, 4, kpt, CURE, False, False, 512)
            if b > 296 and 2 * vptx <= 4 and (k <= 1024):
                tpl = (256, 2 * vptx, 8) + tuple(tpl[3:])
            return {
                "kernel": "reg",
                "tpl": tpl,
                "rt": {"n": n, "npad": npad, "k": k, "CMP": CMP, "IMGOFF": 2 * NB, "QC": QC},
                "grid": (b, 1),
                "cluster": 1,
                "block": tpl[0],
                "smem": (tpl[7] + (CMP if pk16 else 2 * CMP)) * 4,
                "pk16": pk16,
                "ws": False,
            }
    if (
        plan["kernel"] == "main"
        and b <= 148
        and (1024 * 4 * 4 + 1024 >= n)
        and (n >> 2 > 1152)
        and (not (b <= 2 and k > BLKC and (2 * BLKC * 8 + BLKC >= n)))
    ):
        n4q = n >> 2
        CMP = n if n < 2560 else 2560
        QC = QUADC
        CURE = not (n < 2 * k and b > 148)
        DEGE = n <= 3 * k or n <= 4 * k + 64
        if DEGE and CMP < n:
            CMP = n
        kpt = 1 if k <= 1024 else 2
        vptx = 4
        while vptx > 1 and (1024 * (vptx // 2) >= n4q or 1024 * (vptx // 2) * 4 + 1024 >= n):
            vptx //= 2
        tpl = (1024, vptx, 1, kpt, CURE, DEGE, False, 1024)
        return {
            "kernel": "reg",
            "tpl": tpl,
            "rt": {"n": n, "npad": npad, "k": k, "CMP": CMP, "IMGOFF": 2 * NB, "QC": QC},
            "grid": (b, 1),
            "cluster": 1,
            "block": tpl[0],
            "smem": (tpl[7] + (CMP if pk16 else 2 * CMP)) * 4,
            "pk16": pk16,
            "ws": False,
        }
    if plan["kernel"] == "main" and plan["tpl"][5] and (b <= 32) and (k <= 2 * BLKC):
        av = 148 // (b if b > 0 else 1)
        amax = 1
        while amax << 1 <= av and amax < 16:
            amax <<= 1
        vsel = 0
        cs = 0
        g4_skip = False
        if amax >= 2 and (not g4_skip):
            for v in (1, 2):
                c = 1
                while c * BLKC * v * 8 + BLKC < n:
                    c <<= 1
                if c == 8 and b > 15:
                    continue
                if c == 16 and (b < 2 or b > 8 or v < 2 or (k < 1024)):
                    continue
                if c >= 16 and n >= 131072 and (b >= 8):
                    continue
                if c <= amax:
                    vsel = v
                    cs = c
                    break
        if vsel and cs >= 2:
            return {
                "kernel": "reg_clus",
                "tpl": (BLKC, vsel, cs),
                "rt": {"n": n, "npad": npad, "k": k},
                "grid": (cs, b),
                "cluster": cs,
                "block": BLKC,
                "smem": (3 * NB + 2 * CMPC) * 4,
                "ws": False,
            }
    if plan["kernel"] in ("main", "clus"):
        plan = _bf16_halve_u(plan)
    elif plan["kernel"] == "reg_clus":
        av = 148 // (b if b > 0 else 1)
        amax = 1
        while amax << 1 <= av and amax < 8:
            amax <<= 1
        vsel = 0
        cs = 0
        if amax >= 2:
            for v in (1, 2, 4):
                c = 1
                while c * BLKC * v * 8 + BLKC < n:
                    c <<= 1
                if c == 8 and b > 15:
                    continue
                if c <= amax:
                    vsel = v
                    cs = c
                    break
        if vsel and cs >= 2:
            plan["tpl"] = (BLKC, vsel, cs)
            plan["grid"] = (cs, b)
            plan["cluster"] = cs
        else:
            plan = _bf16_halve_u(plan)
    elif plan["kernel"] in ("reg", "regimg"):
        tpl = list(plan["tpl"])
        n4r = n >> 2
        while tpl[1] > 1 and (
            tpl[0] * (tpl[1] // 2) >= n4r or tpl[0] * (tpl[1] // 2) * 4 + tpl[0] >= n
        ):
            tpl[1] //= 2
        target_nbh = 512 if tpl[0] == 512 else 1024
        if tpl[7] > target_nbh:
            tpl[7] = target_nbh
        if b > 296 and k <= 1024 and (tpl[0] == 512) and (tpl[2] == 4) and (tpl[1] <= 2):
            tpl[0] = 256
            tpl[1] *= 2
            tpl[2] = 8
            if tpl[1] <= 2 and k > 512:
                tpl[7] = 256
            plan["block"] = 256
        plan["tpl"] = tuple(tpl)
        if pk16 and plan["kernel"] == "reg":
            plan["smem"] = (tpl[7] + plan["rt"]["CMP"]) * 4
            plan["pk16"] = True
        else:
            plan["smem"] = (tpl[7] + 2 * plan["rt"]["CMP"]) * 4
    return plan


def route_streaming_bf16(
    b: int, n: int, npad: int, k: int, force_main: bool = False
) -> dict[str, object]:
    """bf16 twin of route_streaming (see route_bf16).

    Split-aware vector-width pick: the fp32 table derives the split count R so
    each CTA's chunk (~n/R) fills one fp32 tile of BLK*U*4 elements. When
    U == 1 the 16-byte bf16 tile cannot shrink with U and spans BLK*8 -- a
    ~BLK*4 chunk then idles half the threads and halves the outstanding-load
    parallelism exactly in the latency-bound deep-split regime. Route those
    plans to the 8-byte-vector engine (fp32 tile geometry, all threads
    active); keep the 16-byte engine when the chunk actually fills >= 3/4 of
    the wider tile (fewer load instructions at full thread activity)."""
    plan = route_streaming(b, n, npad, k, force_main=force_main)
    if plan["kernel"] in ("main", "clus"):
        tpl = plan["tpl"]
        if (
            plan["kernel"] == "main"
            and tpl[5]
            and (int(tpl[0]) == 1024)
            and (int(tpl[1]) == 1)
            and (n >= 65536)
            and (k <= 2048)
        ):
            r512 = ((n >> 2) + 511) // 512
            kpt = 1 if k <= 512 else 2 if k <= 1024 else 4
            g1_ok = not True or k < 1024 or _degen_gate_ok(int(tpl[0]), int(tpl[4]), 512, kpt)
            if r512 > int(plan["rt"]["R"]) and b * r512 <= 148 and g1_ok:
                tpl = (512, 1, 1, tpl[3], kpt, True, tpl[6])
                plan["tpl"] = tpl
                plan["grid"] = (r512, b)
                plan["block"] = 512
                plan["rt"]["R"] = r512
                plan["rt"]["Q"] = ((n >> 2) + r512 - 1) // r512
        if (
            plan["kernel"] == "main"
            and tpl[5]
            and (b <= 32)
            and (k <= 1024)
            and (int(tpl[1]) == 1)
            and ("rt" in plan)
            and (int(plan["rt"].get("R", 1)) > 1)
            and (b * int(plan["rt"].get("R", 1)) >= 148)
            and ((n + int(plan["rt"]["R"]) - 1) // int(plan["rt"]["R"]) < 6144)
        ):
            r1_8 = max(148 // b, 1)
            r2_8 = max(((n >> 3) + 1023) // 1024, 1)
            r_n8 = max(min(r1_8, r2_8), 1)
            if r_n8 > 1 and r_n8 != int(plan["rt"]["R"]):
                q8 = ((n >> 3) + r_n8 - 1) // r_n8
                per8 = q8 >> 10
                u8 = 8 if per8 >= 8 else 4 if per8 >= 4 else 2 if per8 >= 2 else 1
                if u8 > 4:
                    u8 = 4
                kpt8 = 1 if k <= 1024 else 2 if k <= 2048 else 4 if k <= 4096 else 8
                tshg8 = b > 15 and k <= 1024 and (n >> 2 <= 32768)
                plan["tpl"] = (1024, u8, 1, tpl[3], kpt8, True, tshg8)
                plan["rt"]["R"] = r_n8
                plan["rt"]["Q"] = ((n >> 2) + r_n8 - 1) // r_n8
                plan["grid"] = (r_n8, b)
                return plan
        if (
            plan["kernel"] == "main"
            and tpl[5]
            and (int(tpl[1]) == 1)
            and (n >= 16384)
            and ("rt" in plan)
            and (int(plan["rt"].get("R", 1)) > 1)
            and (b * int(plan["rt"].get("R", 1)) < 148)
        ):
            r_const = int(plan["rt"]["R"])
            chunk = (n + r_const - 1) // r_const
            if chunk < int(tpl[0]) * 8 * 3 // 4:
                plan["vec4"] = True
                return plan
        if plan["kernel"] == "main" and (not tpl[5]) and (int(tpl[1]) == 4):
            blk = int(tpl[0])
            rolls_full = -(-n // (blk * 4 * 8))
            rolls_half = -(-n // (blk * 2 * 8))
            if rolls_full < rolls_half:
                return plan
        if plan["kernel"] == "main" and tpl[5] and (int(tpl[1]) == 4) and ("rt" in plan):
            r_c = int(plan["rt"].get("R", 1))
            chunk = (n + r_c - 1) // r_c
            rolls_full_s = -(-chunk // (int(tpl[0]) * 4 * 8))
            rolls_half_s = -(-chunk // (int(tpl[0]) * 2 * 8))
            if rolls_full_s < rolls_half_s:
                return plan
        plan = _bf16_halve_u(plan)
        tpl = plan["tpl"]
        if plan["kernel"] == "main" and tpl[5] and (int(tpl[0]) == 1024) and (int(tpl[1]) < 4):
            r_const = int(plan["rt"]["R"])
            r_pow2 = 1 << r_const.bit_length() - 1
            u_const = int(tpl[1])
            u_pow2 = min(2 * u_const, 4)
            chunk_now = (n + r_const - 1) // r_const
            chunk_pow2 = (n + r_pow2 - 1) // r_pow2
            rolls_now = (chunk_now + 1024 * u_const * 8 - 1) // (1024 * u_const * 8)
            rolls_pow2 = (chunk_pow2 + 1024 * u_pow2 * 8 - 1) // (1024 * u_pow2 * 8)
            if r_pow2 < r_const and b * r_pow2 >= 128 and (rolls_pow2 < rolls_now):
                plan["tpl"] = (tpl[0], u_pow2) + tuple(tpl[2:])
                plan["rt"]["R"] = r_pow2
                plan["rt"]["Q"] = ((n >> 2) + r_pow2 - 1) // r_pow2
                plan["grid"] = (r_pow2, b)
    return plan


def _varlen_launcher_bf16(
    num_rows: int,
    npad: int,
    k: int,
    n_env: int,
    next_n: int,
    cr: int,
    device_index: int,
    device_profile: int,
) -> tuple:
    """bf16 twin of _varlen_launcher: same capture-stable family tiers, the
    bf16 route table, and the BFloat16 compiled variants."""
    key = (num_rows, npad, k, n_env, next_n, cr, device_index, device_profile)
    hit = _VARLEN_CACHE_BF16.get(key)
    if hit is not None:
        return hit
    n_kernel = min(n_env, npad)
    n_route = max(n_kernel, k + 1)
    cr_shift = 0 if cr == 1 else 2
    if k in (512, 1024, 2048) and 0 < n_kernel - k <= 3:
        from . import gvr_topk_decode_self_sampling_bf16_complement as complement

        fn = complement.get_compiled(k, n_kernel, next_n, cr_shift)
        lc = ("complement", fn, n_kernel)
        _VARLEN_CACHE_BF16[key] = lc
        return lc
    dev = _device()
    num_sms, sm_version = _fp32._unpack_device_profile(device_profile)
    plan_free = route_bf16(num_rows, n_route, npad, k, num_sms, sm_version)
    if plan_free["kernel"] == "reg_clus":
        cluster_options = {}
        if (
            num_rows == 1
            and next_n == 1
            and (cr == 1)
            and (k == 2048)
            and (65536 <= n_kernel <= 132096)
            and (tuple(plan_free["tpl"]) in ((1024, 1, 8), (1024, 2, 8)))
        ):
            cluster_options["hybrid"] = True
            cluster_options["oneq_enabled"] = True
        fn = dev.get_compiled__regclus(
            tuple(plan_free["tpl"]),
            varlen=True,
            next_n=next_n,
            cr_shift=cr_shift,
            hint_free=True,
            dtype="bf16",
            nbh=512 if k > 1024 or (k in (512, 1024) and n_env >= 65536) else 1024,
            quadc=96 if k > 1024 and num_rows > 1 and (plan_free["tpl"][2] >= 8) else 384,
            **cluster_options,
        )
        lc = ("reg_clus", fn, n_kernel)
        _VARLEN_CACHE_BF16[key] = lc
        return lc
    if plan_free["kernel"] in ("reg", "regimg"):
        reg_options = {}
        if k == 2048 and num_rows <= 148 and (n_kernel > 2048):
            reg_options["binproof"] = True
            if plan_free["kernel"] == "reg" and plan_free["tpl"][4]:
                tpl = list(plan_free["tpl"])
                old_bins = tpl[7]
                if old_bins < 1024:
                    tpl[7] = 1024
                    plan_free["tpl"] = tuple(tpl)
                    plan_free["smem"] += 4 * (1024 - old_bins)
        if n_kernel <= 2048:
            reg_options["packed_prefetch"] = False
        fn = dev.get_compiled__reg(
            tuple(plan_free["tpl"]),
            varlen=True,
            next_n=next_n,
            cr_shift=cr_shift,
            hint_free=True,
            dtype="bf16",
            pk16=bool(plan_free.get("pk16", False)),
            **reg_options,
        )
        rt_f = plan_free["rt"]
        lc = ("reg", fn, (n_kernel, rt_f["CMP"], rt_f["QC"], dev.STATIC_BYTES + plan_free["smem"]))
        _VARLEN_CACHE_BF16[key] = lc
        return lc
    if plan_free["kernel"] == "clus":
        rt_f = plan_free["rt"]
        fn = dev.get_compiled__clus(
            tuple(plan_free["tpl"]),
            scap=rt_f["SCAP"],
            cmp_=rt_f["CMP"],
            varlen=True,
            next_n=next_n,
            cr_shift=cr_shift,
            hint_free=True,
            dtype="bf16",
        )
        lc = ("clus", fn, (n_kernel, npad, k, rt_f["SCAP"], rt_f["CMP"], 0, 0, 0, 0, 0))
        _VARLEN_CACHE_BF16[key] = lc
        return lc
    plan = route_streaming_bf16(num_rows, n_route, npad, k, force_main=True)
    tpl = tuple(plan["tpl"])
    rt = plan["rt"]
    r_const = rt["R"]
    if plan.get("vec4"):
        fn = dev.get_compiled(
            tpl[:6] + (False,) + (next_n, cr_shift, r_const),
            hint_free=True,
            dtype="bf16",
            vector_elems=4,
        )
    else:
        v16 = not tpl[5] and int(tpl[0]) < 512 and (npad <= 65536)
        fn = dev.get_compiled(
            tpl[:6] + (False,) + (next_n, cr_shift, r_const),
            hint_free=True,
            dtype="bf16",
            v16=v16,
            dense=k == 2048,
        )
    big = num_rows * r_const <= 148
    if r_const > 2 and k > 1024:
        split_aim = 13 * k // 8 if num_rows > 8 else 7 * k // 4
        split_sfac = 48
    else:
        split_aim = 2 * k if k > 1024 or r_const == 2 else 7 * k // 2
        split_sfac = 32 if r_const == 2 else 48 if k > 1024 else 16
    aim_base = (
        (
            (3 * k // 2 if k >= 1024 and (k == 1024 or n_env >= 131072) else 2 * k)
            if r_const == 1
            else split_aim
        )
        if big
        else 11 * k // 8
        if k >= 1024
        else 3 * k // 2
    )
    sfac = split_sfac if r_const > 1 else 64 if k >= 1024 else 32
    amin = 3 * k if r_const == 2 else split_aim
    sd_en = 1 if k > 1024 and (not big) else 0
    tsh_en = 1 if tpl[5] else 0
    pre = (0, npad, k, rt["SCAP_"], rt["CMP_"], r_const, 0, 0, 0, 0, 0)
    tail = (aim_base, sfac, amin, sd_en, tsh_en)
    lc = ("main", fn, pre, tail)
    _VARLEN_CACHE_BF16[key] = lc
    return lc


def run_varlen_bf16(
    logits: torch.Tensor,
    kv_lens: torch.Tensor,
    indices: torch.Tensor,
    next_n: int = 1,
    compress_ratio: int = 1,
    values: torch.Tensor | None = None,
    max_seq_len: int | None = None,
    workspace: torch.Tensor | None = None,
) -> None:
    """Native-bf16 twin of ``run_varlen`` (same per-row varlen contract;
    bfloat16 logits read directly by the device code and widened only in
    registers). Optional values are CUDA float32, with the same flattened
    output view and -FLT_MAX padding as the frozen PR head. Supplying
    max_seq_len avoids device-to-host reads; use the exact uncompressed
    envelope, not the padded storage width, to preserve dispatch."""
    if logits.dtype is not torch.bfloat16:
        raise RuntimeError(f"logits must be bfloat16 (got {logits.dtype}); fp32 -> run_varlen")
    if not (isinstance(kv_lens, _TENSOR) and kv_lens.is_cuda):
        raise RuntimeError("kv_lens must be a CUDA tensor")
    if kv_lens.dtype is not _I32:
        raise RuntimeError("kv_lens must be int32")
    if kv_lens.dim() != 1:
        raise RuntimeError("kv_lens must be 1-D")
    nn = _index(next_n)
    cr = _index(compress_ratio)
    if nn < 1:
        raise RuntimeError(f"next_n must be >= 1, got {nn}")
    if cr not in (1, 4):
        raise RuntimeError(f"compress_ratio must be 1 (DSv3.2) or 4 (DSv4), got {cr}")
    if len(logits.shape) != 2:
        raise RuntimeError("logits must be 2-D")
    num_rows = logits.shape[0]
    if num_rows == 0:
        return
    if not logits.is_cuda:
        raise RuntimeError("logits must be CUDA")
    if num_rows % nn:
        raise RuntimeError(f"num_rows {num_rows} not divisible by next_n {nn}")
    batch = num_rows // nn
    if kv_lens.shape[0] != batch:
        raise RuntimeError(f"kv_lens length {kv_lens.shape[0]} != num_rows/next_n = {batch}")
    if kv_lens.device != logits.device or indices.device != logits.device:
        raise RuntimeError("logits, kv_lens and indices must be on the same CUDA device")
    if values is not None and values.device != logits.device:
        raise RuntimeError("values must be on the same CUDA device as logits")
    d = logits.get_device()
    if not 0 <= d < _GVR_MAX_DEV:
        raise RuntimeError(f"device index out of range: {d}")
    if workspace is not None:
        validate_run_ws(workspace, logits)
        ws = kernel_view(workspace)
    else:
        ws = _ws_hot.get(d)
        if ws is None:
            ws = default_workspace(logits)
    if not (logits.is_cuda and indices.is_cuda):
        raise RuntimeError("all tensors must be CUDA")
    if indices.dtype is not _I32:
        raise RuntimeError("indices must be int32")
    if len(indices.shape) != 2 or indices.shape[0] != num_rows:
        raise RuntimeError(
            f"indices must be [num_rows={num_rows}, >=k], got {tuple(indices.shape)}"
        )
    k = indices.shape[1]
    if not (indices.is_contiguous() and kv_lens.is_contiguous()):
        raise RuntimeError("indices/kv_lens must be contiguous")
    if logits.stride(1) != 1:
        raise RuntimeError("logits inner stride must be 1")
    npad = logits.stride(0) if num_rows > 1 else logits.shape[1]
    lg = logits
    if not logits.is_contiguous():
        need = logits.storage_offset() + num_rows * npad
        if logits.untyped_storage().size() // 2 < need:
            raise RuntimeError("logits view storage too small to widen to its row stride")
        lg = logits.as_strided((num_rows, npad), (npad, 1), logits.storage_offset())
    if npad & 7:
        raise RuntimeError(f"npad (logits row stride) must be a multiple of 8, got {npad}")
    if lg.data_ptr() & 15:
        raise RuntimeError("logits base must be 16-byte aligned")
    if values is not None:
        if not values.is_cuda or values.dtype is not _F32:
            raise RuntimeError("values must be CUDA float32")
        if (
            len(values.shape) != 2
            or values.shape[0] != num_rows
            or values.shape[1] < k
            or (not values.is_contiguous())
        ):
            raise RuntimeError(
                f"values must be contiguous [num_rows={num_rows}, >=k], got {tuple(values.shape)}"
            )
    cshift = 0 if cr == 1 else 2
    if max_seq_len is not None:
        n_env = int(max_seq_len) >> cshift
    else:
        if _is_capturing():
            raise RuntimeError(
                "run_varlen_bf16 without max_seq_len reads kv_lens.max() on host; "
                "pass max_seq_len (a capture-stable engine constant)"
            )
        n_env = int(kv_lens.max().item()) >> cshift
        n_env = 1 << max(n_env - 1, 1).bit_length()
    n_env = min(max(n_env, 1), npad)
    profile = _fp32._device_profile_key(d)
    key = (num_rows, npad, k, n_env, nn, cr, d, profile)
    lc = _VARLEN_CACHE_BF16.get(key)
    if lc is None:
        if _is_capturing():
            raise RuntimeError(
                "varlen launcher not compiled for this shape — warm up before CUDA graph capture"
            )
        # Serialize cold compilation across BF16 families. Shared-memory layouts
        # are instance constants; warmed launches never take the lock.
        with _COMPILE_LOCK, torch.cuda.device(d):
            lc = _varlen_launcher_bf16(num_rows, npad, k, n_env, nn, cr, d, profile)
    idx = indices
    if idx.shape[1] != k:
        idx = idx.reshape(-1)[: num_rows * k].view(num_rows, k)
    vals = values
    if vals is not None and vals.shape[1] != k:
        vals = vals.reshape(-1)[: num_rows * k].view(num_rows, k)
    pre_arg = idx
    if lc[0] in ("reg_clus", "complement"):
        lc[1](lg, pre_arg, kv_lens, idx, lc[2])
    elif lc[0] == "reg":
        lc[1](lg, pre_arg, kv_lens, idx, *lc[2])
    elif lc[0] == "clus":
        lc[1](lg, pre_arg, kv_lens, idx, *lc[2])
    else:
        _, fn, pre, tail = lc
        fn(lg, pre_arg, idx, ws, *pre, kv_lens, *tail)
    if vals is not None:
        idx64 = idx.to(torch.int64)
        vals.copy_(lg.gather(1, idx64.clamp_min(0)))
        vals.masked_fill_(idx < 0, torch.finfo(_F32).min)
    return


def warmup_varlen(
    top_k: int,
    max_seq_len: int,
    compress_ratio: int = 1,
    next_n: int = 1,
    num_rows_list: Sequence[int] = (1,),
    row_stride: int | None = None,
) -> None:
    """Compile and launch the requested BF16 row counts before graph capture.

    The input arena must use the same element row stride as serving. All
    allocations and initialization launches occur here, outside timed decode.
    """
    k = operator.index(top_k)
    nn = operator.index(next_n)
    cr = operator.index(compress_ratio)
    envelope = operator.index(max_seq_len)
    if nn < 1 or cr not in (1, 4):
        raise RuntimeError("next_n must be positive and compress_ratio must be 1 or 4")
    if k < 1 or envelope < 0:
        raise RuntimeError("top_k must be positive and max_seq_len must be nonnegative")
    rows_list = sorted({max(operator.index(r) // nn * nn, nn) for r in num_rows_list})
    if not rows_list:
        return
    n_env = max(envelope // cr, 1)
    npad = (n_env + 63) // 64 * 64 if row_stride is None else operator.index(row_stride)
    if npad < n_env or npad % 8:
        raise RuntimeError(f"row_stride must be a multiple of 8 >= {n_env}, got {npad}")
    device = torch.cuda.current_device()
    profile = _fp32._device_profile_key(device)
    for rows in rows_list:
        key = (rows, npad, k, n_env, nn, cr, device, profile)
        if key in _WARMUP_DONE and key in _VARLEN_CACHE_BF16 and device in _ws_hot:
            continue
        logits = torch.zeros((rows, npad), dtype=torch.bfloat16, device=device)
        lengths = torch.full((rows // nn,), envelope, dtype=torch.int32, device=device)
        indices = torch.empty((rows, k), dtype=torch.int32, device=device)
        run_varlen_bf16(
            logits, lengths, indices, next_n=nn, compress_ratio=cr, max_seq_len=envelope
        )
        torch.cuda.synchronize(device)
        _WARMUP_DONE.add(key)
