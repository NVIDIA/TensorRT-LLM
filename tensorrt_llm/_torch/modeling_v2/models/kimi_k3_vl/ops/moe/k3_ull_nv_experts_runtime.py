# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Host side of the NVFP4 TP16 Rubin experts (``nv_experts.RubinNvExpertsKernel``).

* :func:`scratch`: the experts' workspace (images,
  partials, counters, route mailbox, staged rows, grouped hidden images, peer slots, generations;
  the hidden images of both kernels armed with 0x88, everything else with 0xFF), the latent image
  plan of 8-row calls (``nv_experts.IMAGE_BYTES``) and the grouped plan of 9-64-row calls
  (``nv_experts.PLAN_BYTES``).
* :func:`route_mailbox`: the route mailbox words (int32) the routing kernel writes in polled mode
  (128 ids, then the image count, which reaches ``2 * 8 - 1`` when the image is complete).
"""

import functools

import cutlass
import cutlass.cute as cute
import torch
from cutlass.cute.runtime import make_fake_stream

# isort: off
from ...kernels.k3r_moe.ull import nv_experts as nv
from ..common import k3_ull_cute_host as _host
from ...kernels.k3r_moe.ull import experts as _experts
from ...kernels.k3r_moe.ull import experts_rubin as _rubin
from ...kernels.k3r_moe.ull import experts_rubin64 as _rubin64
# isort: on

BATCH = 8


def slots_for(device, polled=True):
    """Per-CTA workspace entries: vLLM's (SMs rounded up to 32, at least 160); the route mailbox
    needs ``4 * slots >= 772`` bytes of ``ws_xflag`` (224 on 152-SM parts)."""
    sms = torch.cuda.get_device_properties(device).multi_processor_count
    slots = max(160, -(-sms // 32) * 32)
    return max(slots, 224) if polled else slots


def workspace_bytes(slots):
    geometry = _experts.ExpertsGeometry(16, slots=slots)
    size = _rubin.peers_offset(geometry.workspace_bytes, _rubin64.CENSUS_BYTES)
    return size + _rubin.peers_bytes(slots) + _rubin.generation_bytes(slots)


def arm(workspace, slots):
    """(Re-)arm a workspace: 0xFF everywhere, the hidden images of both call parities 0x88 (the
    8-row kernel's [2, 128] and the grouped kernel's [2, 1024] units)."""
    geometry = _experts.ExpertsGeometry(16, slots=slots)
    unit = geometry.h_compact_bytes
    workspace.fill_(255)
    images = workspace[: geometry.ws_partial].view(2 * 128, unit)
    images[:, : nv.H_UNIT_BYTES] = nv.ARM_BYTE
    grouped = workspace[geometry.ws_units : geometry.ws_units + 2 * 1024 * unit].view(2048, unit)
    grouped[:, : nv.H_UNIT_BYTES] = nv.ARM_BYTE


def scratch(device, slots):
    """The experts' workspace (2 MiB aligned, armed), the latent image plan and the grouped plan."""
    size = workspace_bytes(slots)
    chunk = 2 << 20
    raw = torch.empty((size + chunk,), device=device, dtype=torch.uint8)
    start = -raw.data_ptr() % chunk
    workspace = raw[start : start + size]
    arm(workspace, slots)
    plan = torch.zeros(nv.IMAGE_BYTES, device=device, dtype=torch.uint8)
    rows = torch.zeros(nv.PLAN_BYTES, device=device, dtype=torch.uint8)
    return {"workspace": workspace, "plan": plan, "rows": rows, "slots": slots}


def route_mailbox(workspace, slots):
    geometry = _experts.ExpertsGeometry(16, slots=slots)
    start = geometry.ws_xflag + _rubin.ROUTE_IDS
    return workspace[start : start + 4 * _rubin.ROUTE_WORDS].view(torch.int32)


def to_cute(tensor, align=16):
    return _host.to_cute(tensor.reshape(-1), align)


@functools.cache
def compile_experts(
    experts, grid, slots, device, publish=False, polled=False, static=True, fast=True, stamp=0,
    options="",
):  # fmt: skip
    """The kernel for one configuration (``publish``: False / True / "push"); ``options``: CuTe DSL
    compile options (e.g. "--keep-ptx")."""
    moe = nv.RubinNvExpertsKernel(
        experts, grid, slots, publish=publish, polled=polled, static_groups=static,
        fast_exchange=fast, stamp=stamp,
    )  # fmt: skip
    flat = lambda n, dtype=cutlass.Uint8, align=16: _host.fake((n,), (1,), dtype, align)  # noqa: E731
    out = flat(BATCH * 16 * (1 if publish else 3584), cutlass.BFloat16)
    tensors = (flat(nv.IMAGE_BYTES), flat(BATCH * 16, cutlass.Int32))
    tensors += (flat(2 * (experts + 1), cutlass.Float32), flat(experts * nv.W13_SCALE_BYTES))
    tensors += (flat(experts * nv.W2_SCALE_BYTES), flat(moe.workspace_bytes), out)
    box = (cutlass.Int64(0), cutlass.Int32(0), cutlass.Int32(0)) if publish else ()
    if publish == "push":
        box += (flat(BATCH * 7168, cutlass.BFloat16), _host.fake((16,), None, cutlass.Int64))
        box += (cutlass.Int32(0),)
    scalars = (cutlass.Int32(BATCH), *map(cutlass.Float32, (100, 0.25, 0.04)), *box)
    scalars += (cutlass.Int64(0), cutlass.Int64(0))
    with torch.cuda.device(device):
        if options:
            return moe, cute.compile(moe, *tensors, *scalars, make_fake_stream(), options=options)
        return moe, cute.compile(moe, *tensors, *scalars, make_fake_stream())


# FC2(B) tail pool zone of the grouped kernel by rows (vLLM ``POOL_ZONE`` for TP16; else 12).
POOL_ZONE = ((16, 0), (32, 6))


def pool_zone(rows):
    return next((zone for top, zone in POOL_ZONE if rows <= top), 12)


@functools.cache
def compile_grouped(experts, batch, grid, slots, device, publish=False, static=True, options=""):
    """``RubinNvGroupedExpertsKernel`` at ``batch`` (9-64) rows (``publish``: False / True /
    "push" / "twoshot": vLLM's two-shot latent push, box (receive slots [3 x 16, 64, 224], the
    owners' bases, this rank's slot), the parent's finalize unchanged); static groups and the pool
    zone as vLLM's verify steps."""
    assert 8 < batch <= nv.PLAN_ROWS
    moe = nv.RubinNvGroupedExpertsKernel(
        experts, grid, slots, publish=publish, static_groups=static, pool=(pool_zone(batch), 6)
    )
    flat = lambda n, dtype=cutlass.Uint8, align=16: _host.fake((n,), (1,), dtype, align)  # noqa: E731
    out = flat(batch * 16 * (1 if publish else 3584), cutlass.BFloat16)
    tensors = (flat(nv.PLAN_BYTES), flat(batch * 16, cutlass.Int32))
    tensors += (flat(2 * (experts + 1), cutlass.Float32), flat(experts * nv.W13_SCALE_BYTES))
    tensors += (flat(experts * nv.W2_SCALE_BYTES), flat(moe.workspace_bytes), out)
    box = (cutlass.Int64(0), cutlass.Int32(0), cutlass.Int32(0)) if publish else ()
    if publish == "push":
        box += (flat(batch * 7168, cutlass.BFloat16), _host.fake((16,), None, cutlass.Int64))
        box += (cutlass.Int32(0),)
    if publish == "twoshot":  # receive slots, owners' bases, parity x TP + rank (vLLM's)
        box = (_host.fake((48, 64, 224)), _host.fake((16,), None, cutlass.Int64), cutlass.Int32(0))
    scalars = (cutlass.Int32(batch), *map(cutlass.Float32, (100, 0.25, 0.04)), *box)
    scalars += (cutlass.Int64(0), cutlass.Int64(0))
    with torch.cuda.device(device):
        if options:
            return moe, cute.compile(moe, *tensors, *scalars, make_fake_stream(), options=options)
        return moe, cute.compile(moe, *tensors, *scalars, make_fake_stream())


@functools.cache
def compile_plan(device, batch):
    flat = _host.fake((nv.PLAN_BYTES,), (1,), cutlass.Uint8)
    x = _host.fake((batch, 3584))
    with torch.cuda.device(device):
        return cute.compile(
            nv.PlanNvKernel(batch), x, flat, cutlass.Float32(1.0), make_fake_stream()
        )


def write_plan(latent, plan, x_gsf):
    """``PlanNvKernel``: the BF16 latent rows [batch <= 64, 3584] as the NVFP4 grouped plan."""
    code = compile_plan(latent.device.index, latent.shape[0])
    code(
        _host.to_cute(latent, 16),
        to_cute(plan),
        cutlass.Float32(float(x_gsf)),
        _host.stream(latent.device),
    )


@functools.cache
def compile_image(device, batch=BATCH):
    flat = _host.fake((nv.IMAGE_BYTES,), (1,), cutlass.Uint8)
    x = _host.fake((batch, 3584))
    with torch.cuda.device(device):
        return cute.compile(
            nv.ImagePlanNvKernel(batch), x, flat, cutlass.Float32(1.0), make_fake_stream()
        )


def write_image(latent, plan, x_gsf):
    """``ImagePlanNvKernel``: the BF16 latent [8, 3584] as the NVFP4 image into ``plan``."""
    code = compile_image(latent.device.index, latent.shape[0])
    code(
        _host.to_cute(latent, 16),
        to_cute(plan),
        cutlass.Float32(float(x_gsf)),
        _host.stream(latent.device),
    )
