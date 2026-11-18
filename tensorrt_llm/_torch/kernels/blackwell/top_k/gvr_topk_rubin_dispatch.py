# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shape-based FP32 GVR dispatch for Rubin."""

import math
from collections.abc import Callable
from typing import NamedTuple


class _Tuning(NamedTuple):
    family: str
    block: int = 0
    min_blocks: int = 1
    unroll: int = 1
    cluster: int = 1
    parts: int = 1
    bins: int = 1024
    options: tuple[tuple[str, int | bool], ...] = ()
    expected_family: str = ""


_REFERENCE_SMS = 208
# Inclusive (K, N minimum, N maximum, reference rows minimum/maximum, tuning).
_REGIONS: tuple[tuple[int, int, int, int, int, _Tuning], ...] = (
    (512, 513, 2047, 3, 4, _Tuning(family="reg", block=256, min_blocks=3, bins=1024)),
    (512, 2048, 4095, 3, 4, _Tuning(family="reg", block=512, min_blocks=2, bins=1024)),
    (512, 4096, 65535, 3, 256, _Tuning(family="blackwell_options")),
    (512, 65536, 260095, 3, 16, _Tuning(family="reg_clus", cluster=8)),
    (
        512,
        260096,
        262143,
        3,
        16,
        _Tuning(
            family="main", block=1024, min_blocks=1, unroll=4, parts=16, options=(("amin_num", 4),)
        ),
    ),
    (
        512,
        65536,
        262143,
        65,
        128,
        _Tuning(
            family="main",
            block=1024,
            min_blocks=1,
            unroll=4,
            parts=1,
            options=(("warp_rank", True),),
        ),
    ),
    (
        512,
        65536,
        262143,
        129,
        256,
        _Tuning(
            family="main", block=512, min_blocks=2, unroll=4, parts=1, options=(("aim_num", 4),)
        ),
    ),
    (512, 513, 2047, 257, 1024, _Tuning(family="default", options=(("bins", 512),))),
    (
        512,
        2048,
        8191,
        257,
        1024,
        _Tuning(family="reg", block=256, min_blocks=4, bins=1024, options=(("warp_rank", True),)),
    ),
    (
        512,
        8192,
        131071,
        257,
        512,
        _Tuning(family="main", block=256, min_blocks=4, unroll=8, parts=1),
    ),
    (512, 8192, 32767, 513, 1024, _Tuning(family="blackwell_options")),
    (
        512,
        32768,
        131071,
        513,
        1024,
        _Tuning(
            family="main", block=512, min_blocks=2, unroll=4, parts=1, options=(("aim_num", 4),)
        ),
    ),
    (512, 131072, 262143, 257, 512, _Tuning(family="default", options=(("aim_num", 8),))),
    (
        512,
        131072,
        262143,
        513,
        1024,
        _Tuning(
            family="main", block=1024, min_blocks=1, unroll=4, parts=1, options=(("aim_num", 8),)
        ),
    ),
    (1024, 1025, 2047, 1, 4, _Tuning(family="blackwell_options", options=(("bins", 512),))),
    (1024, 1025, 2047, 5, 1024, _Tuning(family="blackwell_options", options=(("bins", 256),))),
    (1024, 2048, 16383, 1, 128, _Tuning(family="blackwell_options", options=(("QC", 80),))),
    (
        1024,
        2048,
        4095,
        129,
        256,
        _Tuning(family="blackwell_options", options=(("bracket_sample_lanes", 8),)),
    ),
    (
        1024,
        4096,
        8191,
        129,
        256,
        _Tuning(family="reg", block=512, min_blocks=2, bins=1024, options=(("static_npad", True),)),
    ),
    (
        1024,
        2048,
        8191,
        257,
        1024,
        _Tuning(family="reg", block=256, min_blocks=4, bins=1024, options=(("warp_rank", True),)),
    ),
    (
        1024,
        8192,
        16383,
        129,
        512,
        _Tuning(family="blackwell_options", options=(("stage_output", False),)),
    ),
    (1024, 8192, 16383, 513, 1024, _Tuning(family="reg", block=256, min_blocks=4, bins=1024)),
    (
        1024,
        16384,
        32767,
        1,
        128,
        _Tuning(family="blackwell_options", options=(("stage_output", True),)),
    ),
    (1024, 32768, 260095, 1, 16, _Tuning(family="reg_clus", cluster=8)),
    (
        1024,
        260096,
        262143,
        1,
        16,
        _Tuning(
            family="main",
            block=1024,
            min_blocks=1,
            unroll=2,
            parts=52,
            options=(("warp_rank", True),),
        ),
    ),
    (
        1024,
        32768,
        65535,
        65,
        128,
        _Tuning(
            family="reg", block=1024, min_blocks=1, bins=2048, options=(("static_npad", True),)
        ),
    ),
    (
        1024,
        65536,
        262143,
        65,
        128,
        _Tuning(
            family="main", block=1024, min_blocks=1, unroll=4, parts=1, options=(("aim_num", 4),)
        ),
    ),
    (
        1024,
        16384,
        262143,
        129,
        256,
        _Tuning(family="main", block=512, min_blocks=2, unroll=4, parts=1),
    ),
    (1024, 65536, 260095, 257, 512, _Tuning(family="default", options=(("sfac", 48),))),
    (
        1024,
        65536,
        131071,
        513,
        1024,
        _Tuning(family="main", block=512, min_blocks=2, unroll=4, parts=1, options=(("sfac", 48),)),
    ),
    (
        1024,
        131072,
        260095,
        513,
        1024,
        _Tuning(
            family="main", block=1024, min_blocks=1, unroll=4, parts=1, options=(("sfac", 48),)
        ),
    ),
    (
        1024,
        260096,
        262143,
        257,
        1024,
        _Tuning(
            family="main", block=1024, min_blocks=1, unroll=4, parts=1, options=(("aim_num", 8),)
        ),
    ),
    (
        2048,
        4096,
        16383,
        1,
        1024,
        _Tuning(family="blackwell_options", options=(("warp_rank", True),)),
    ),
    (2048, 16384, 163839, 1, 1, _Tuning(family="blackwell_options")),
    (
        2048,
        16384,
        32767,
        2,
        4,
        _Tuning(
            family="reg", block=1024, min_blocks=1, bins=1024, options=(("static_npad", True),)
        ),
    ),
    (
        2048,
        16384,
        32767,
        5,
        8,
        _Tuning(
            family="reg", block=1024, min_blocks=1, bins=1024, options=(("dense_bracket", True),)
        ),
    ),
    (
        2048,
        16384,
        32767,
        9,
        128,
        _Tuning(
            family="reg", block=1024, min_blocks=1, bins=1024, options=(("static_npad", True),)
        ),
    ),
    (2048, 32768, 131071, 2, 16, _Tuning(family="reg_clus", cluster=8)),
    (
        2048,
        131072,
        161791,
        2,
        16,
        _Tuning(family="main", block=1024, min_blocks=1, unroll=2, parts=208),
    ),
    (2048, 32768, 161791, 17, 32, _Tuning(family="blackwell_options")),
    (
        2048,
        161792,
        163839,
        2,
        32,
        _Tuning(
            family="main", block=1024, min_blocks=1, unroll=2, parts=40, options=(("amin_num", 2),)
        ),
    ),
    (
        2048,
        32768,
        65535,
        33,
        128,
        _Tuning(
            family="main", block=512, min_blocks=1, unroll=8, parts=1, options=(("aim_num", 3),)
        ),
    ),
    (
        2048,
        65536,
        131071,
        33,
        128,
        _Tuning(
            family="main", block=1024, min_blocks=1, unroll=4, parts=1, options=(("aim_num", 4),)
        ),
    ),
    (
        2048,
        131072,
        163839,
        33,
        128,
        _Tuning(
            family="main", block=1024, min_blocks=1, unroll=4, parts=1, options=(("aim_num", 3),)
        ),
    ),
    (2048, 16384, 163839, 129, 512, _Tuning(family="blackwell_options")),
    (
        2048,
        16384,
        163839,
        513,
        1024,
        _Tuning(family="main", block=512, min_blocks=2, unroll=4, parts=1),
    ),
)


def select_tuning(rows: int, n: int, k: int, num_sms: int, sm_version: int) -> _Tuning | None:
    """Select continuous resource-normalized regions independently of allocation."""
    if sm_version != 107 or n <= k or rows < 1 or num_sms < 1:
        return None
    reference_rows = (rows * _REFERENCE_SMS + num_sms - 1) // num_sms
    tuning = next(
        (
            t
            for kk, lo, hi, blo, bhi, t in _REGIONS
            if kk == k and lo <= n <= hi and blo <= reference_rows <= bhi
        ),
        None,
    )
    if tuning is None:
        return None
    if tuning.family == "reg":
        vpt = max(1, ((n >> 2) + tuning.block - 1) // tuning.block)
        if vpt > 8 or k > 4 * tuning.block:
            return None
    if tuning.family == "reg_clus" and (n >> 2) > 4 * 1024 * tuning.cluster:
        return None
    if tuning.family in ("main", "clus") and n <= 4096:
        return None
    if tuning.family == "clus" and k > 1024:
        return None
    if tuning.family == "main" and tuning.parts > 1 and rows > 160:
        return None
    return tuning


def make_plan(
    tuning: _Tuning,
    rows: int,
    n: int,
    npad: int,
    k: int,
    num_sms: int,
    baseline_route: Callable,
    streaming_route: Callable,
    dynamic_route: Callable,
) -> dict:
    """Build one existing GVR plan and capture-stable option overrides."""
    family = tuning.family
    if family in ("default", "blackwell_options"):
        plan = baseline_route(
            rows, n, npad, k, num_sms, 100 if family == "blackwell_options" else 107
        )
        if tuning.expected_family and plan["kernel"] != tuning.expected_family:
            return baseline_route(rows, n, npad, k, num_sms, 107)
    elif family == "reg":
        block = tuning.block
        vpt = max(1, ((n >> 2) + block - 1) // block)
        degenerate = n <= 4 * k + 64
        cmp_ = n if degenerate else min(n, 2560)
        kpt = 1 if degenerate or k <= block else 2 if k <= 2 * block else 4
        plan = dict(
            kernel="reg",
            tpl=(
                block,
                vpt,
                tuning.min_blocks,
                kpt,
                not (n < 2 * k and rows > num_sms),
                degenerate,
                False,
                tuning.bins,
            ),
            rt=dict(n=n, npad=npad, k=k, CMP=cmp_, IMGOFF=tuning.bins, QC=160),
            grid=(rows, 1),
            cluster=1,
            block=block,
            smem=4 * (tuning.bins + 2 * cmp_),
            ws=False,
        )
    elif family == "reg_clus":
        need = max(1, math.ceil((n >> 2) / (1024 * tuning.cluster)))
        vpt = next(v for v in (1, 2, 4) if v >= need)
        plan = dict(
            kernel="reg_clus",
            tpl=(1024, vpt, tuning.cluster, k),
            rt=dict(n=n, npad=npad, k=k),
            grid=(tuning.cluster, rows),
            cluster=tuning.cluster,
            block=1024,
            smem=(3 * 1024 + 2 * 4096) * 4,
            ws=False,
        )
    elif family == "clus":
        plan = dict(
            kernel="clus",
            tpl=(1024, tuning.unroll, 1, 256, tuning.cluster),
            rt=dict(n=n, npad=npad, k=k, SCAP=8192, CMP=2048),
            grid=(tuning.cluster, rows),
            cluster=tuning.cluster,
            block=1024,
            smem=256 * 8 + (8192 + 4) * 8 + 2048 * 8,
            ws=False,
        )
    else:
        assert family == "main"
        plan = streaming_route(rows, n, npad, k, force_main=True, num_sms=num_sms, sm_version=107)
        block = tuning.block
        parts = tuning.parts
        # Keep one-wave split geometry inside the actual stream's SM resources.
        if parts > 1:
            parts = max(1, min(parts, num_sms // rows, max(1, math.ceil(n / 4096))))
        kpt = 1 if k <= block else 2 if k <= 2 * block else 4 if k <= 4 * block else 8
        scap = (16384 if parts == 1 else 8192) if block == 1024 else 8192 if k >= 2048 else 4096
        cmp_ = (4096 if k >= 2048 else 2048) if block == 1024 else 1024
        plan.update(
            tpl=(block, tuning.unroll, tuning.min_blocks, 256, kpt, parts > 1, False),
            block=block,
            grid=(parts, rows),
            smem=(scap + 4) * (8 if parts > 1 or block >= 512 else 4) + (cmp_ + 1) * 8,
        )
        plan["rt"].update(R=parts, SCAP_=scap, CMP_=cmp_)
    for name, value in tuning.options:
        register_options = {
            "stage_output",
            "static_npad",
            "rematerialize_bins",
            "dense_bracket",
            "bracket_sample_lanes",
            "QC",
            "bins",
        }
        streaming_options = {"staged_k", "aim_num", "sfac", "amin_num", "sd_en", "tsh_en"}
        if (
            (name in register_options and plan["kernel"] not in ("reg", "regimg"))
            or (name in streaming_options and plan["kernel"] != "main")
            or (name in ("scap", "cmp_") and plan["kernel"] != "clus")
            or (name == "warp_rank" and plan["kernel"] not in ("reg", "regimg", "main"))
        ):
            return baseline_route(rows, n, npad, k, num_sms, 107)
        if name == "bins" and (value < plan["block"] or value % plan["block"]):
            return baseline_route(rows, n, npad, k, num_sms, 107)
        if name == "staged_k" and value and plan["rt"].get("R") != 1:
            return baseline_route(rows, n, npad, k, num_sms, 107)
    compile_options = {}
    runtime_options = {}
    for name, value in tuning.options:
        if name == "QC":
            plan["rt"]["QC"] = value
        elif name == "bins":
            tpl = list(plan["tpl"])
            plan["smem"] += 4 * (value - tpl[7])
            tpl[7] = value
            plan["tpl"] = tuple(tpl)
            plan["rt"]["IMGOFF"] = value
        elif name in ("scap", "cmp_"):
            plan["rt"]["SCAP" if name == "scap" else "CMP"] = value
        elif name in ("aim_num", "sfac", "amin_num", "sd_en", "tsh_en"):
            runtime_options[name] = value
        elif name == "static_npad":
            compile_options[name] = npad if value else None
        else:
            compile_options[name] = value
    if plan["kernel"] in ("main", "clus"):
        if plan["kernel"] == "clus":
            plan["smem"] = 256 * 8 + (plan["rt"]["SCAP"] + 4) * 8 + plan["rt"]["CMP"] * 8
        dynamic, _ = dynamic_route(plan, n)
        plan["rt"].update(dynamic)
    plan.update(
        _rubin_tuning=True,
        _rubin_options=compile_options,
        _rubin_runtime=runtime_options,
        _rubin_blackwell_options=family == "blackwell_options",
    )
    return plan
