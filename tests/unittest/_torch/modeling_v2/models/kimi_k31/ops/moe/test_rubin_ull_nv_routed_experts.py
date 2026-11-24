# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the ULL target's nv_routed_experts entry: every cell's launch prepared as the
target prepares it (``moe.NvExperts``, all 896 experts), driven against the FP64 reference within
a band derived from the kernel's arithmetic, with the published latent read back from a multicast
mailbox bound on this device and the pushed rows from local peer buffers."""

import zlib

import cutlass
import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Cell
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.moe.rubin_ull_nv_routed_experts import (  # noqa: E501
    LATENT,
    TOP_K,
    dequantized_experts,
    e2m1_round,
    hidden_scales,
    latent,
    quantize_hidden,
    situ,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.moe.rubin_ull_nv_routed_experts import (  # noqa: E501
    nv_routed_experts as op,
)

__extra_import_path__ = ["../../../.."]
from _cells import certified_cells  # noqa: E402 — needs the path declared above
from _validating import validating  # noqa: E402 — needs the path declared above

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10,
    reason="the tcgen05 NVFP4 experts need an SM10x GPU",
)

TARGET_CELLS: dict[str, tuple[Cell, ...]] = {}
CELLS = certified_cells(op, TARGET_CELLS)

EXPERTS, TP, CAP = 896, 16, 64
PARITY, RANK, ORDINAL = 1, 5, 1  # the box this rank publishes into
SENTINEL = -0x80000000  # the mailboxes' arm word (two BF16 -0)


# --------------------------------------------------------------------------------------------
# the bank, the scratch and the mailboxes
# --------------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def target():
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.kimi_k31_pro_nvfp4__sm_107__helix16_moetp16ep1 import (  # noqa: E501
        modeling as moe,
    )
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.kimi_k31_pro_nvfp4__sm_107__helix16_moetp16ep1 import (  # noqa: E501
        weights,
    )

    return moe, weights


@pytest.fixture(scope="module")
def bank(target):
    """A random publish-ordered bank of all 896 experts: E2M1 codes uniform over the 16 nibbles,
    UE4M3 block scales in [1/16, 1/4), per-expert global scales 2^-6..2^-4 (gate = up), unit
    activation scales, as the checkpoint's."""
    _, weights = target
    g = torch.Generator(device="cuda").manual_seed(31)

    def u8(*shape, low=0, high=256):
        return torch.randint(low, high, shape, generator=g, device="cuda", dtype=torch.uint8)

    s13 = 2.0 ** torch.randint(-6, -3, (EXPERTS,), generator=g, device="cuda").float()
    s2 = 2.0 ** torch.randint(-6, -3, (EXPERTS,), generator=g, device="cuda").float()
    raw = dict(
        w1=u8(EXPERTS, 192, LATENT // 2),
        w3=u8(EXPERTS, 192, LATENT // 2),
        w2=u8(EXPERTS, LATENT, 96),
        w1_sf=u8(EXPERTS, 192, LATENT // 16, low=0x18, high=0x28),
        w3_sf=u8(EXPERTS, 192, LATENT // 16, low=0x18, high=0x28),
        w2_sf=u8(EXPERTS, LATENT, 12, low=0x18, high=0x28),
        w1_scale2=s13,
        w3_scale2=s13.clone(),
        w2_scale2=s2,
        fc1_input_scale=torch.ones(EXPERTS, device="cuda"),
        fc2_input_scale=torch.ones(EXPERTS, device="cuda"),
    )
    return weights.pack_routed_experts_tp16(**raw, publish=True)


@pytest.fixture(scope="module")
def scratch(target):
    moe, _ = target
    return moe.NvScratch(torch.device("cuda", torch.cuda.current_device()))


_KEEP = []  # driver allocations of the multicast mailbox: alive for the process


def _check(result, step=""):
    from cuda.bindings import driver as cu

    err, *rest = result
    if err != cu.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"{step}: {err.name}")
    return rest[0] if len(rest) == 1 else rest


class _Pointer:
    def __init__(self, address, numel):
        self.__cuda_array_interface__ = dict(
            shape=(numel,), typestr="<i2", data=(address, False), strides=None, version=3
        )


def _multicast_mailbox(shape):
    """A BF16 tensor ``shape`` on the current device and the multicast address of its memory (a
    multicast object over the first ``n`` devices, each with its own copy bound, ``n`` the
    fewest the driver takes), or why the driver gives none."""
    from cuda.bindings import driver as cu

    index = torch.cuda.current_device()
    numel = 1
    for n in shape:
        numel *= n
    handles = cu.CUmemAllocationHandleType
    device = _check(cu.cuDeviceGet(index), "cuDeviceGet")
    attribute = cu.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MULTICAST_SUPPORTED
    reasons = [f"multicast supported={_check(cu.cuDeviceGetAttribute(attribute, device))}"]
    counts = [n for n in (1, 2) if n <= torch.cuda.device_count()]
    for count in counts:
        members = [index] + [d for d in range(torch.cuda.device_count()) if d != index]
        members = members[:count]
        for kind in (
            handles.CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR,
            handles.CU_MEM_HANDLE_TYPE_FABRIC,
            handles.CU_MEM_HANDLE_TYPE_NONE,
        ):
            try:
                prop = cu.CUmulticastObjectProp()
                prop.numDevices = count
                prop.handleTypes = kind
                prop.size = numel * 2
                flag = cu.CUmulticastGranularity_flags.CU_MULTICAST_GRANULARITY_RECOMMENDED
                gran = _check(cu.cuMulticastGetGranularity(prop, flag), "granularity")
                size = -(-numel * 2 // gran) * gran
                prop.size = size
                mc = _check(cu.cuMulticastCreate(prop), "cuMulticastCreate")
                for d in members:
                    member = _check(cu.cuDeviceGet(d), "cuDeviceGet")
                    _check(cu.cuMulticastAddDevice(mc, member), "cuMulticastAddDevice")
                memories = []
                for d in members:
                    alloc = cu.CUmemAllocationProp()
                    alloc.type = cu.CUmemAllocationType.CU_MEM_ALLOCATION_TYPE_PINNED
                    alloc.location.type = cu.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
                    alloc.location.id = d
                    alloc.requestedHandleTypes = kind
                    memory = _check(cu.cuMemCreate(size, alloc, 0), "cuMemCreate")
                    _check(cu.cuMulticastBindMem(mc, 0, memory, 0, size, 0), "cuMulticastBindMem")
                    memories.append(memory)
                access = cu.CUmemAccessDesc()
                access.location.type = cu.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
                access.location.id = index
                access.flags = cu.CUmemAccess_flags.CU_MEM_ACCESS_FLAGS_PROT_READWRITE
                addresses = []
                for handle in (memories[0], mc):
                    address = _check(cu.cuMemAddressReserve(size, gran, 0, 0), "reserve")
                    _check(cu.cuMemMap(address, size, 0, handle, 0), "cuMemMap")
                    _check(cu.cuMemSetAccess(address, size, [access], 1), "cuMemSetAccess")
                    addresses.append(int(address))
            except (RuntimeError, AttributeError, TypeError) as error:
                reasons.append(f"{count} device(s) {kind.name}: {error}")
                continue
            _KEEP.append((mc, memories))
            tensor = torch.as_tensor(_Pointer(addresses[0], numel), device="cuda")
            return tensor.view(torch.bfloat16).view(shape), addresses[1]
    return "; ".join(reasons)


@pytest.fixture(scope="module")
def mailbox():
    """The tail's latent mailbox [3, TP, CAP, 3584] (multicast) as the experts publish into it."""
    made = _multicast_mailbox((3, TP, CAP, LATENT))
    if isinstance(made, str):
        pytest.skip(f"no multicast object: {made}")
    return made


# --------------------------------------------------------------------------------------------
# one call
# --------------------------------------------------------------------------------------------


def _ids(rows, shared, g):
    if shared:
        one = torch.randperm(EXPERTS, generator=g, device="cuda")[:TOP_K]
        return one.repeat(rows).to(torch.int32)
    picks = [torch.randperm(EXPERTS, generator=g, device="cuda")[:TOP_K] for _ in range(rows)]
    return torch.cat(picks).to(torch.int32)


def _call(spec, seed, target, bank, scratch, mailbox_or_none):
    """Prepare the cell's launch as ``prepare_moe`` does, write its plan (and the route mailbox
    of a polled launch) as the routing kernel would, and launch through the entry."""
    moe, _ = target
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.kernels.moe import (
        rubin_ull_experts_nv_experts_runtime as rt,  # noqa: E501
    )

    rows, mode = spec["rows"], spec["mode"]
    g = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(rows, LATENT, generator=g, device="cuda").bfloat16()
    ids = _ids(rows, spec["shared_ids"], g)
    weights = (torch.rand(rows * TOP_K, generator=g, device="cuda") * 0.3 + 0.02).bfloat16()
    out = torch.full((rows * TOP_K, LATENT), float("nan"), dtype=torch.bfloat16, device="cuda")
    extra = {}
    publish = ()
    if mode in ("publish", "push"):
        box, mc = mailbox_or_none
        box.view(torch.int32).fill_(SENTINEL)
        publish = (cutlass.Int64(mc), cutlass.Int32(PARITY), cutlass.Int32(RANK))
        extra["mailbox"] = box
    if mode == "push":
        shared = torch.randn(rows, 7168, generator=g, device="cuda").bfloat16()
        shared[:, ::97] = -0.0
        peers = []  # every peer's RS mailbox [3, TP, CAP, 7168 / TP]
        for _ in range(TP):
            rs = torch.empty((3, TP, CAP, 7168 // TP), dtype=torch.bfloat16, device="cuda")
            rs.view(torch.int16).fill_(-0x8000)
            peers.append(rs)
        addresses = torch.tensor([p.data_ptr() for p in peers], dtype=torch.int64, device="cuda")
        publish += (rt.to_cute(shared), rt._host.to_cute(addresses, 16), cutlass.Int32(CAP))
        extra.update(shared=shared, peers=peers, addresses=addresses)
    if mode == "twoshot":
        width = LATENT // TP
        owners = []
        for _ in range(TP):
            recv = torch.empty((3 * TP, CAP, width), dtype=torch.bfloat16, device="cuda")
            recv.view(torch.int16).fill_(-0x8000)
            owners.append(recv)
        addresses = torch.tensor([r.data_ptr() for r in owners], dtype=torch.int64, device="cuda")
        slot = ORDINAL % 3 * TP + RANK
        publish = (
            rt._host.to_cute(owners[RANK]),
            rt._host.to_cute(addresses, 16),
            cutlass.Int32(slot),
        )
        extra.update(owners=owners, addresses=addresses, slot=slot)
    experts = moe.NvExperts(bank, scratch, beta=4.0, linear_beta=25.0, q=8, rows=(rows,))
    experts.prepare(rows, None, out, publish, True, mode == "twoshot", True)
    launch = experts.prepared[rows]["launch"]
    gsf = float(bank["x_gsf"].reshape(-1)[0])
    if launch.image:
        rt.write_image(x, scratch.image, gsf)
    else:
        rt.write_plan(x, scratch.rows, gsf)
    if launch.polled:
        scratch.routes[: rows * TOP_K] = ids
        scratch.routes[TOP_K * 8] = 2 * 8 - 1
    stream = rt._host.stream(x.device)
    with validating(op):
        returned = op(launch, ids, weights, rows, stream)
    torch.cuda.synchronize()
    return launch, ids, weights, returned, extra


# --------------------------------------------------------------------------------------------
# the band
# --------------------------------------------------------------------------------------------


def _rows_band(launch, ids, rows):
    """Per element of the [16 rows, 3584] expert rows, how far the kernel may sit from the FP64
    reference, from its arithmetic:

    - FC1 (K 3584: 56 K64 MMAs, then the K-split partials) and FC2 (K 192: 3 MMAs) add exact
      E2M1 x E2M1 x UE4M3 x UE4M3 products into FP32 accumulators: one rounding per MMA step
      and partial (<= 64 of 2^-24 each, 2^-18 of the summed magnitudes); 2^-14 / 2^-15 used,
      16x for the MMA's own internal accumulation;
    - SiTU runs ``tanh.approx`` / ``ex2.approx`` / ``rcp.approx`` (each <= 2^-10.9 relative):
      the hidden is taken within 2^-9 of its value at the corners of the FC1 interval;
    - the hidden image re-rounds wherever that interval straddles an E2M1 boundary, and a block
      whose amax interval straddles a UE4M3 boundary may take either neighbouring scale: the
      code is any value of either scale's rounding of the interval;
    - the rows are BF16 (round to nearest, 2^-8)."""
    x = latent(launch.plan, rows, launch.image)
    flat = ids.reshape(-1)[: rows * TOP_K].long()
    gsf = float(launch.packed["alphas"][-1, 0])
    band = torch.empty(rows * TOP_K, LATENT, dtype=torch.float64, device=x.device)
    for start in range(0, flat.numel(), 32):
        pairs = flat[start : start + 32]
        n = pairs.numel()
        w1, w3, w2, a1, a2 = dequantized_experts(launch.packed, pairs)
        xt = x[torch.arange(start, start + n, device=x.device) // TOP_K][:, :, None]
        gate = a1[:, None] * torch.bmm(w1, xt)[..., 0]
        up = a1[:, None] * torch.bmm(w3, xt)[..., 0]
        eg = 2.0**-14 * a1[:, None] * torch.bmm(w1.abs(), xt.abs())[..., 0]
        eu = 2.0**-14 * a1[:, None] * torch.bmm(w3.abs(), xt.abs())[..., 0]
        del w1, w3
        corners = torch.stack(
            [situ(gate + sg * eg, up + su * eu, launch.situ) for sg in (-1, 1) for su in (-1, 1)]
        )
        widen = 2.0**-9 * corners.abs().amax(0)
        lo, hi = corners.amin(0) - widen, corners.amax(0) + widen
        q0 = quantize_hidden(situ(gate, up, launch.situ), gsf).view(n, -1, 16)
        lo, hi = lo.view(n, -1, 16), hi.view(n, -1, 16)
        top = torch.maximum(lo.abs(), hi.abs()).amax(-1, keepdim=True)
        straddle = (lo <= 0) & (hi >= 0)
        bottom = torch.where(straddle, 0.0, torch.minimum(lo.abs(), hi.abs())).amax(
            -1, keepdim=True
        )
        s_lo = hidden_scales(bottom * (1 - 2.0**-20), gsf)
        s_hi = hidden_scales(top * (1 + 2.0**-20), gsf)
        q_lo = q_hi = None
        for s in (s_lo, s_hi):
            m = torch.where(s > 0, gsf / s, 0.0)
            vlo, vhi = lo * m, hi * m
            a = e2m1_round(vlo - 2.0**-20 * vlo.abs()) * s
            b = e2m1_round(vhi + 2.0**-20 * vhi.abs()) * s
            q_lo = a if q_lo is None else torch.minimum(q_lo, a)
            q_hi = b if q_hi is None else torch.maximum(q_hi, b)
        # More than two scales possible (never seen): any code of the largest, +-1 step.
        after = (s_lo.to(torch.float8_e4m3fn).view(torch.uint8) + 1).view(torch.float8_e4m3fn)
        wide = s_hi > after.double()
        q_lo = torch.where(wide, lo - s_hi, q_lo)
        q_hi = torch.where(wide, hi + s_hi, q_hi)
        dev = torch.maximum(q_hi - q0, q0 - q_lo).clamp_min(0).view(n, -1)
        mag = torch.maximum(q_lo.abs(), q_hi.abs()).view(n, -1)
        aw2 = w2.abs()
        b = a2[:, None] * (
            torch.bmm(aw2, dev[:, :, None])[..., 0]
            + 2.0**-15 * torch.bmm(aw2, mag[:, :, None])[..., 0]
        )
        row = a2[:, None] * torch.bmm(w2, q0.view(n, -1, 1))[..., 0]
        band[start : start + n] = b + 2.0**-8 * (row.abs() + b)
    return band


def _tight(got, band, ref):
    """The band has to reject a wrong answer, or it certifies nothing: the kernel's own values
    against a decoy reference (each row compared with its neighbour's: another expert, or
    another token) must leave the band almost everywhere."""
    decoy = torch.roll(ref, 1, dims=0)
    caught = ((got.double() - decoy).abs() > band).double().mean().item()
    assert caught > 0.9, f"a neighbouring row passes the band at {1 - caught:.1%} of elements"


@pytest.mark.parametrize("cell", CELLS, ids=[c.why[:44] for c in CELLS])
def test_certified_cells(cell, target, bank, scratch, request) -> None:
    spec = cell.spec
    box = request.getfixturevalue("mailbox") if spec["mode"] in ("publish", "push") else None
    seed = zlib.crc32(cell.why.encode()) % 2**31
    launch, ids, weights, returned, extra = _call(spec, seed, target, bank, scratch, box)
    rows = spec["rows"]
    assert returned is launch.out
    ref = op.reference(launch, ids, weights, rows, None)
    band = _rows_band(launch, ids, rows)
    if launch.polled:  # the last CTA re-armed the route mailbox
        assert bool((scratch.routes == -1).all()), "route mailbox not re-armed"
    if not launch.published:
        err = (launch.out.double() - ref).abs()
        bad = err > band
        rms = (err.pow(2).mean() / ref.pow(2).mean()).sqrt().item()
        assert not bool(bad.any()), (
            f"{int(bad.sum())} elements outside the band; worst {err.max().item():.3e}, "
            f"relative RMS {rms:.2e}"
        )
        _tight(launch.out, band, ref)
        return
    # Published: the BF16 staged rows' routing-weighted FP32 sum in slot order, then BF16.
    w = weights[: rows * TOP_K].double().view(rows, TOP_K, 1).abs()
    rows_ref = op.reference(
        launch.__class__(**{**launch.__dict__, "box": ()}), ids, weights, rows, None
    ).view(rows, TOP_K, LATENT)
    fma = 16 * 2.0**-24 * (w * rows_ref.abs()).sum(1)
    lat_band = (w * band.view(rows, TOP_K, LATENT)).sum(1) + fma
    lat_band += 2.0**-8 * (ref.abs() + lat_band)
    # The rows' errors (BF16 rounding, hidden codes re-rounded either way) are zero-mean and
    # independent across the 16 rows: in aggregate they add in quadrature.
    quad = (w.pow(2) * band.view(rows, TOP_K, LATENT).pow(2)).sum(1).sqrt() + fma
    quad += 2.0**-8 * ref.abs()
    if spec["mode"] == "twoshot":  # owner c: columns [224 c, 224 c + 224) at the rank's slot
        got = torch.cat([o[extra["slot"], :rows] for o in extra["owners"]], dim=1)
        for c, owner in enumerate(extra["owners"]):
            rest = torch.ones(owner.shape[:2], dtype=torch.bool, device="cuda")
            rest[extra["slot"], :rows] = False
            assert bool((owner[rest].view(torch.int16) == -0x8000).all()), f"owner {c} overrun"
    else:
        box = extra["mailbox"]
        got = box[PARITY, RANK, :rows]
        written = torch.zeros(box.shape[:3], dtype=torch.bool, device="cuda")
        written[PARITY, RANK, :rows] = True
        assert bool((box[~written].view(torch.int32) == SENTINEL).all()), "wrote past its slot"
    err = (got.double() - ref).abs()
    bad = err > lat_band
    rms = (err.pow(2).mean() / ref.pow(2).mean()).sqrt().item()
    assert not bool(bad.any()), (
        f"{int(bad.sum())} latent elements outside the band; worst {err.max().item():.3e}, "
        f"relative RMS {rms:.2e}"
    )
    # Aggregate: the RMS error within the quadrature band's RMS, which a neighbouring token's
    # latent (the decoy) must exceed several times over.
    limit = quad.pow(2).mean().sqrt().item()
    measured = (got.double() - ref).pow(2).mean().sqrt().item()
    decoy = (got.double() - torch.roll(ref, 1, dims=0)).pow(2).mean().sqrt().item()
    assert measured <= limit, f"RMS error {measured:.3e} above the quadrature band {limit:.3e}"
    assert decoy > 4 * limit, f"a neighbouring token's latent ({decoy:.3e}) passes {limit:.3e}"
    if spec["mode"] == "push":
        shard = 7168 // TP
        shared = extra["shared"].view(torch.int16)
        shared = torch.where(shared == -0x8000, torch.zeros_like(shared), shared)  # -0 made +0
        for p, peer in enumerate(extra["peers"]):
            want = shared[:, p * shard : (p + 1) * shard]
            assert torch.equal(peer[PARITY, RANK, :rows].view(torch.int16), want), f"peer {p}"
            rest = torch.ones(peer.shape[:3], dtype=torch.bool, device="cuda")
            rest[PARITY, RANK, :rows] = False
            assert bool((peer[rest].view(torch.int16) == -0x8000).all()), f"peer {p} overrun"


def test_is_valid_refuses_what_the_kernel_would_read_wrongly(target, bank, scratch) -> None:
    spec = dict(rows=1, mode="direct", shared_ids=False)
    launch, ids, weights, _, _ = _call(spec, 7, target, bank, scratch, None)
    with pytest.raises(ValueError, match="prepared for"):
        op.is_valid(launch, ids, weights, 2, None)
    twice = ids.clone()
    twice[1] = twice[0]
    with pytest.raises(ValueError, match="twice"):
        op.is_valid(launch, twice, weights, 1, None)
    outside = ids.clone()
    outside[3] = EXPERTS
    with pytest.raises(ValueError, match="outside"):
        op.is_valid(launch, outside, weights, 1, None)
    with pytest.raises(ValueError, match="int32"):
        op.is_valid(launch, ids.long(), weights, 1, None)
