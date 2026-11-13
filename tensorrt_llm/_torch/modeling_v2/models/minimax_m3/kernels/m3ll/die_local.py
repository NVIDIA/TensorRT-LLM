# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""DIE_LOCAL: die-local roles for the fused MoE engine on a target's own m3_moe plan, and the routed
expert weights placed on the die whose SMs stream them. Placement and scheduling only: bitwise.

An SM107 GPU has two locality domains (dies) of 100 SMs plus 12 remainder SMs. A weight buffer from the caching
allocator is striped over both dies, so every routed CTA reads half of its bytes across the die link. The reference (its
``expert_local``) cuts the experts into per-die shards and gives the dies a per-die plan. Our engine cannot take that
plan bit for bit: the per-die plan moves the FC1 split units (a split unit sums head + tail, an unsplit unit its
quarters once; moe_engine_kernel.py ``partials``), and a die-local plan that keeps every split point of the m3_moe
plan streams up to 1.39x the bytes on its busiest CTA at 8 rows (desk, the TP2 plan). This module keeps the plan:

* roles: each routed CTA of the plan goes to the die holding most of the bytes it streams (FC1 units by expert-row
  half, FC2 output blocks by owner), 94 per die (:func:`assign_dies`); the engine reads its role from a ``%smid`` ->
  role table (engine trace key ``dl_tab``, vendoring edit 8) built on the driver's SM -> domain probe
  (``LocalityMap.logical_map``): the plan, the unit decode and every sum are the plain engine's;
* placement: the routed buffers ``e_w31`` [E, 2 I, H / 2] and ``e_w2`` [E, H, I / 2] keep their shape, layout and
  virtual addresses' contents; each 2 MiB granule (the VMM allocation granule) is backed by the die that wants most of
  its bytes (:func:`granule_runs`), so the wide path (trtllm-gen at 16..32 rows) reads the same bytes.

At the TP2 8-row plan the model gives ~77 % of the routed bytes die-local (FC1 98.7 % of the bytes on the wanted die x
88.9 % granule majority; FC2 55 %) against 50 % striped.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

GRANULE = (
    2 << 20
)  # capacity: fixed: the SM107 VMM allocation granule (the default; callers pass LocalityMap's)
ROW_BLOCK = (
    32  # capacity: fixed: rows of one FC1 unit / one FC2 output block in the engine (32-row blocks)
)
_PROBE_SRC = r"""
extern "C" __global__ void smid_one_per_sm(int* out, unsigned* arrived, long long budget) {
    extern __shared__ int pad[];
    if (threadIdx.x == 0) {
        unsigned s;
        asm volatile("mov.u32 %0, %%smid;" : "=r"(s));
        pad[0] = (int)s;
        atomicAdd(arrived, 1u);
        long long t0 = clock64();
        // stay resident until every CTA has arrived (co-resident CTAs are on distinct SMs: the shared memory admits
        // one per SM); give up after the budget (the host then sees an id twice or -1)
        while (atomicAdd(arrived, 0u) < gridDim.x && clock64() - t0 < budget) {
        }
        out[blockIdx.x] = (atomicAdd(arrived, 0u) >= gridDim.x) ? pad[0] : -1;
    }
}
"""


# ---- roles ----------------------------------------------------------------------------------------------------------


def cta_die_bytes(
    plan,
    num_ctas: int,
    m_max: int,
    *,
    units: int,
    s1: int,
    item_bytes: int,
    blk_bytes: int,
    top_k: int = 4,
    gs: Optional[Sequence[int]] = None,
) -> List[List[float]]:
    """Per routed CTA of ``plan`` (int32 [4 m + 1, num_ctas, m + 1]: per group i0 | i1 << 16, then b0 | nb << 16), the
    bytes it streams per call summed over the distinct-expert counts ``gs`` (default every G): [FC1 bytes of units
    < units / 2, FC1 bytes of units >= units / 2, FC2 bytes of blocks in the first half, FC2 bytes in the second]."""
    p = plan.tolist() if hasattr(plan, "tolist") else plan
    gs = list(gs) if gs is not None else list(range(top_k, top_k * m_max + 1))
    half_u = units // 2
    nb_total = 0
    for c in range(num_ctas):
        w = int(p[gs[0]][c][m_max])
        nb_total = max(nb_total, (w & 0xFFFF) + (w >> 16))
    half_b = nb_total // 2
    out = [[0.0, 0.0, 0.0, 0.0] for _ in range(num_ctas)]
    for g in gs:
        ng = (g + 3) // 4
        for c in range(num_ctas):
            row = p[g][c]
            for ph in range(ng):
                w = int(row[ph])
                i0, i1 = w & 0xFFFF, w >> 16
                for it in range(i0, max(i0, i1)):
                    u = (it // s1) % units
                    out[c][0 if u < half_u else 1] += item_bytes
            w = int(row[m_max])
            b0, nb = w & 0xFFFF, w >> 16
            for b in range(b0, b0 + nb):
                out[c][2 if b < half_b else 3] += g * blk_bytes
    return out


def assign_dies(die_bytes: Sequence[Sequence[float]], per_die: int) -> List[int]:
    """Die 0 / 1 per routed CTA, exactly ``per_die`` each: the CTAs that stream the most more first-half than
    second-half bytes go to die 0 (ties by CTA index, so every rank derives the same assignment)."""
    n = len(die_bytes)
    if n != 2 * per_die:
        raise ValueError(f"die_local: {n} routed CTAs are not 2 x {per_die}")
    pref = sorted(
        range(n),
        key=lambda c: (
            -((die_bytes[c][0] + die_bytes[c][2]) - (die_bytes[c][1] + die_bytes[c][3])),
            c,
        ),
    )
    die = [1] * n
    for c in pref[:per_die]:
        die[c] = 0
    return die


def role_table(lmap: Sequence[int], die: Sequence[int], num_ctas: int, num_roles: int) -> List[int]:
    """The ``%smid`` -> role table: ``lmap`` (LocalityMap.logical_map: SM -> logical block, logical [d P, (d + 1) P) on
    domain d's SMs, P = num_ctas / 2, then the extra blocks) with logical block d P + i -> the i-th routed CTA ``die``
    puts on die d, and extra block 2 P + j -> role num_ctas + j (the shared CTAs). Raises unless a bijection of the SMs
    onto [0, num_roles)."""
    per = num_ctas // 2
    by_die = [[c for c in range(num_ctas) if die[c] == d] for d in range(2)]
    if len(by_die[0]) != per or len(by_die[1]) != per:
        raise ValueError(
            f"die_local: {len(by_die[0])} / {len(by_die[1])} routed CTAs per die, not {per}"
        )
    perm = by_die[0] + by_die[1] + list(range(num_ctas, num_roles))
    tab = [perm[int(lb)] if 0 <= int(lb) < len(perm) else -1 for lb in lmap]
    if len(tab) != num_roles or sorted(tab) != list(range(num_roles)):
        raise ValueError(
            f"die_local: the SM -> role table is not a bijection onto {num_roles} roles"
        )
    return tab


def check_table(tab: Sequence[int], num_sms: int, num_roles: int) -> None:
    """Fail closed unless ``tab`` maps SMs 0 .. num_sms - 1 one to one onto roles 0 .. num_roles - 1 (the engine's grid
    is one CTA per SM: a duplicate role would leave a role unrun and its consumers spinning)."""
    if (
        num_roles != num_sms
        or len(tab) < num_sms
        or sorted(int(x) for x in tab[:num_sms]) != list(range(num_roles))
    ):
        raise ValueError(
            f"die_local: role table is not a bijection of {num_sms} SMs onto {num_roles} roles"
        )


# ---- placement ------------------------------------------------------------------------------------------------------


def unit_want(
    plan, die: Sequence[int], num_ctas: int, m_max: int, *, units: int, s1: int, top_k: int = 4
) -> List[int]:
    """The die that streams most of FC1 unit u's rows (any expert, any slot) over every G under ``die``."""
    p = plan.tolist() if hasattr(plan, "tolist") else plan
    cnt = [[0, 0] for _ in range(units)]
    for g in range(top_k, top_k * m_max + 1):
        for c in range(num_ctas):
            for ph in range((g + 3) // 4):
                w = int(p[g][c][ph])
                i0, i1 = w & 0xFFFF, w >> 16
                for it in range(i0, max(i0, i1)):
                    cnt[(it // s1) % units][die[c]] += 1
    return [0 if a >= b else 1 for a, b in cnt]


def block_want(
    plan, die: Sequence[int], num_ctas: int, m_max: int, nblocks: int, top_k: int = 4
) -> List[int]:
    """The die of each FC2 32-row output block's owner CTA (the plan's FC2 ranges are the same for every G)."""
    p = plan.tolist() if hasattr(plan, "tolist") else plan
    want = [-1] * nblocks
    for c in range(num_ctas):
        w = int(p[top_k][c][m_max])
        for b in range(w & 0xFFFF, (w & 0xFFFF) + (w >> 16)):
            want[b] = die[c]
    if min(want) < 0:
        raise ValueError("die_local: an FC2 block has no owner CTA")
    return want


def granule_runs(
    n_experts: int, piece_bytes: int, want: Sequence[int], granule: int = GRANULE
) -> Tuple[List[Tuple[int, int]], float]:
    """The (bytes, domain) runs of a buffer of ``n_experts`` x len(want) pieces of ``piece_bytes`` (piece k of every
    expert wanted by die want[k]): each granule goes to the die wanting most of its bytes (ties: die 0), consecutive
    granules of one die coalesce. Returns (runs, fraction of the bytes on their wanted die). The size must be a whole
    number of granules."""
    total = n_experts * len(want) * piece_bytes
    if total % granule:
        raise ValueError(f"die_local: {total} B is not a whole number of {granule} B granules")
    ng = total // granule
    cnt = [[0, 0] for _ in range(ng)]
    off = 0
    for _ in range(n_experts):
        for d in want:
            a, end = off, off + piece_bytes
            while a < end:
                g = a // granule
                b = min(end, (g + 1) * granule)
                cnt[g][d] += b - a
                a = b
            off = end
    runs: List[Tuple[int, int]] = []
    local = 0
    for a, b in cnt:
        d = 0 if a >= b else 1
        local += max(a, b)
        if runs and runs[-1][1] == d:
            runs[-1] = (runs[-1][0] + granule, d)
        else:
            runs.append((granule, d))
    return runs, local / total


def place(locmap, t, runs: Sequence[Tuple[int, int]]):
    """A copy of ``t`` (contiguous) with the same shape, dtype and contents in one virtual range backed per ``runs``
    ((bytes, domain) in address order, granule multiples summing to t's bytes), each run's first and last byte checked
    in its domain. The returned tensor's storage holds the mapping."""
    import torch

    from .locality import _Mapping

    if not t.is_contiguous():
        raise ValueError("die_local.place: needs a contiguous tensor")
    total = t.numel() * t.element_size()
    if sum(n for n, _ in runs) != total:
        raise ValueError(
            f"die_local.place: runs cover {sum(n for n, _ in runs)} B, the tensor {total} B"
        )
    m = _Mapping(locmap.device, list(runs))
    flat = torch.as_tensor(m, device=f"cuda:{locmap.device}")
    out = flat[:total].view(t.dtype).view(t.shape)
    out.copy_(t)
    out._locality_mapping = m
    off = 0
    base = out.data_ptr()
    for n, d in runs:
        for q in (base + off, base + off + n - 1):
            if locmap.domain_of(q) != d:
                raise RuntimeError(f"die_local.place: byte {q:#x} is not in domain {d}")
        off += n
    return out


# ---- device checks --------------------------------------------------------------------------------------------------


def probe_sm_ids(device_index: int, num_sms: int, smem_bytes: int = 160 * 1024) -> List[int]:
    """%smid of every CTA of a one-CTA-per-SM grid (``smem_bytes`` of dynamic shared memory each, so no two share an
    SM; every CTA waits for all to arrive, so they are co-resident), sorted: the ids a role table must cover (-1: a
    CTA that gave up waiting). NVRTC + driver, as LocalityMap's probe."""
    import ctypes

    import torch
    from cuda.bindings import driver as d
    from cuda.bindings import nvrtc

    def ck(res, what):
        err = res[0] if isinstance(res, tuple) else res
        if int(err) != 0:
            raise RuntimeError(f"die_local: {what} failed: {err}")
        return (
            res[1]
            if isinstance(res, tuple) and len(res) == 2
            else (res[1:] if isinstance(res, tuple) else None)
        )

    dev = ck(d.cuDeviceGet(device_index), "cuDeviceGet")
    major = ck(
        d.cuDeviceGetAttribute(
            d.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, dev
        ),
        "cc",
    )
    minor = ck(
        d.cuDeviceGetAttribute(
            d.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, dev
        ),
        "cc",
    )
    prog = ck(
        nvrtc.nvrtcCreateProgram(_PROBE_SRC.encode(), b"smid_one_per_sm.cu", 0, [], []),
        "nvrtc create",
    )
    opts = [f"--gpu-architecture=sm_{major}{minor}".encode()]
    if int(nvrtc.nvrtcCompileProgram(prog, len(opts), opts)[0]) != 0:
        raise RuntimeError("die_local: SM id probe compile failed")
    n = ck(nvrtc.nvrtcGetCUBINSize(prog), "cubin size")
    cubin = b" " * n
    ck(nvrtc.nvrtcGetCUBIN(prog, cubin), "cubin")
    torch.cuda.set_device(device_index)
    torch.cuda.synchronize(device_index)
    buf = torch.full((num_sms,), -1, dtype=torch.int32, device=f"cuda:{device_index}")
    mod = ck(d.cuModuleLoadData(cubin), "probe module")
    try:
        fn = ck(d.cuModuleGetFunction(mod, b"smid_one_per_sm"), "probe fn")
        ck(
            d.cuFuncSetAttribute(
                fn,
                d.CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                smem_bytes,
            ),
            "smem attribute",
        )
        arrived = torch.zeros(1, dtype=torch.int32, device=f"cuda:{device_index}")
        args = (
            (
                ctypes.c_void_p(buf.data_ptr()),
                ctypes.c_void_p(arrived.data_ptr()),
                ctypes.c_longlong(1 << 32),
            ),
            (ctypes.c_void_p, ctypes.c_void_p, ctypes.c_longlong),
        )
        stream = torch.cuda.current_stream(device_index).cuda_stream
        ck(
            d.cuLaunchKernel(fn, num_sms, 1, 1, 32, 1, 1, smem_bytes, stream, args, 0),
            "probe launch",
        )
        torch.cuda.synchronize(device_index)
    finally:
        d.cuModuleUnload(mod)
    return sorted(int(x) for x in buf.tolist())


class DieLocal:
    """One rank's die-local setup: the role table (int32 device tensor), the placement report and the engagement
    fields."""

    def __init__(self, tab, info: Dict[str, object]):
        self.tab = tab
        self.info = info


def build_roles(
    device,
    plan,
    num_ctas: int,
    num_roles: int,
    m_max: int,
    *,
    units: int,
    s1: int,
    item_bytes: int,
    blk_bytes: int,
):
    """(LocalityMap, die per routed CTA, SM -> role table list, model dict) for an engine of ``num_ctas`` routed +
    ``num_roles - num_ctas`` shared CTAs on every SM of ``device``. Raises (fail closed) when the device has no two
    locality domains, the grid is not one CTA per SM, a die cannot hold half the routed CTAs, or the SM ids a
    one-CTA-per-SM grid lands on are not 0 .. SMs - 1."""
    import torch

    from .locality import LocalityMap

    idx = torch.device(device).index
    idx = torch.cuda.current_device() if idx is None else idx
    lm = LocalityMap.get(idx)
    if lm.num_sms != num_roles:
        raise ValueError(
            f"die_local: {num_roles} CTAs, {lm.num_sms} SMs (roles by SM need one CTA per SM)"
        )
    if num_ctas % 2 or num_ctas // 2 > lm.sms_per_domain:
        raise ValueError(
            f"die_local: {num_ctas} routed CTAs do not split over two {lm.sms_per_domain}-SM domains"
        )
    ids = probe_sm_ids(idx, lm.num_sms)
    if ids != list(range(lm.num_sms)):
        raise ValueError(
            f"die_local: a one-CTA-per-SM grid ran on SM ids {ids[:4]}..{ids[-4:]}, not 0..{lm.num_sms - 1}"
        )
    db = cta_die_bytes(
        plan, num_ctas, m_max, units=units, s1=s1, item_bytes=item_bytes, blk_bytes=blk_bytes
    )
    die = assign_dies(db, num_ctas // 2)
    lmap, _ = lm.logical_map(num_ctas // 2, lm.num_sms - num_ctas)
    tab = role_table(lmap.tolist(), die, num_ctas, num_roles)
    check_table(tab, lm.num_sms, num_roles)
    loc = sum(db[c][die[c]] + db[c][2 + die[c]] for c in range(num_ctas))
    allb = sum(sum(v) for v in db)
    return lm, die, tab, {"role_local": loc / allb if allb else 0.0}


def prepare(
    ours_state,
    expert_params: Sequence[Tuple[object, object]],
    *,
    hidden: int = 6144,
    top_k: int = 4,
    num_experts: int = 128,
) -> DieLocal:
    """One rank's DIE_LOCAL for the engine on ``ours_state`` (the target's m3_moe state whose plan / CTA split the
    engine runs): the SM -> role table (int32 device tensor), then every (e_w31, e_w2) parameter pair of
    ``expert_params`` re-placed in place (``param.data`` swapped for :func:`place`'s copy). Placement stops (roles kept,
    already placed layers kept: same bytes) at the first tensor whose old storage is not freed to the driver (another
    reference or a shared arena would otherwise double the weights); the reason is in ``info["place_stop"]``. Raises
    (nothing placed, the caller runs the plain roles) when the roles cannot be built. Before CUDA-graph capture and
    before the engines' layers are built (they bind the buffers' addresses)."""
    import torch

    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("die_local.prepare: not during CUDA-graph capture")
    dev = ours_state.device
    num_ctas, nsh, m = int(ours_state.num_ctas), int(ours_state.mod.NSH), int(ours_state.m)
    inter = int(ours_state.config["i_tp"])
    units, s1 = 2 * inter // ROW_BLOCK, int(getattr(ours_state.mod, "S1", 6))
    item_bytes = ROW_BLOCK * (hidden // 2 + hidden // 16) // s1
    blk_bytes = ROW_BLOCK * (inter // 2 + inter // 16)
    plan = ours_state.plan.cpu()
    lm, die, tab, model = build_roles(
        dev,
        plan,
        num_ctas,
        num_ctas + nsh,
        m,
        units=units,
        s1=s1,
        item_bytes=item_bytes,
        blk_bytes=blk_bytes,
    )
    uw = unit_want(plan, die, num_ctas, m, units=units, s1=s1, top_k=top_k)
    bw = block_want(plan, die, num_ctas, m, hidden // ROW_BLOCK, top_k=top_k)
    runs31, f31 = granule_runs(num_experts, ROW_BLOCK * hidden // 2, uw, lm.granularity)
    runs2, f2 = granule_runs(num_experts, ROW_BLOCK * inter // 2, bw, lm.granularity)
    shapes = ((num_experts, 2 * inter, hidden // 2), (num_experts, hidden, inter // 2))
    placed, nbytes, stop = 0, 0, None
    for li, pair in enumerate(expert_params):
        for prm, runs, shape in zip(pair, (runs31, runs2), shapes):
            t = prm.data
            if tuple(t.shape) != shape or t.element_size() != 1 or not t.is_contiguous():
                stop = f"pair {li}: {tuple(t.shape)} {t.dtype} is not the contiguous {shape} byte buffer"
                break
            nb = t.numel()
            torch.cuda.synchronize(dev)
            alloc0 = torch.cuda.memory_allocated(dev)
            new = place(lm, t, runs)
            torch.cuda.synchronize(dev)
            free0 = torch.cuda.mem_get_info(dev)[0]
            prm.data = new
            del new, t
            alloc1 = torch.cuda.memory_allocated(dev)
            torch.cuda.empty_cache()
            free1 = torch.cuda.mem_get_info(dev)[0]
            if alloc0 - alloc1 < nb or free1 - free0 < nb - (64 << 20):
                stop = (
                    f"pair {li}: the old storage was not released (allocated -{(alloc0 - alloc1) >> 20} MiB, driver "
                    f"free +{(free1 - free0) >> 20} MiB, buffer {nb >> 20} MiB)"
                )
                break
            nbytes += nb
        if stop is not None:
            break
        placed += 1
    tab_t = torch.tensor(tab, dtype=torch.int32, device=dev)
    info = dict(model)
    info.update(
        {
            "per_die": num_ctas // 2,
            "shared_roles": nsh,
            "fc1_place_local": f31,
            "fc2_place_local": f2,
            "runs": (len(runs31), len(runs2)),
            "placed_layers": placed,
            "layers": len(expert_params),
            "placed_gib": nbytes / 2**30,
            "place_stop": stop,
        }
    )
    w1, w2 = hidden // 2 * 2 * inter, inter // 2 * hidden
    db = cta_die_bytes(
        plan, num_ctas, m, units=units, s1=s1, item_bytes=item_bytes, blk_bytes=blk_bytes
    )
    fc1_role = sum(db[c][die[c]] for c in range(num_ctas)) / max(1.0, sum(v[0] + v[1] for v in db))
    info["fc1_role_local"] = fc1_role
    info["model_local"] = (w1 * fc1_role * f31 + w2 * f2) / (w1 + w2)
    return DieLocal(tab_t, info)


__all__ = [
    "GRANULE",
    "ROW_BLOCK",
    "DieLocal",
    "assign_dies",
    "block_want",
    "build_roles",
    "check_table",
    "cta_die_bytes",
    "granule_runs",
    "place",
    "prepare",
    "probe_sm_ids",
    "role_table",
    "unit_want",
]
