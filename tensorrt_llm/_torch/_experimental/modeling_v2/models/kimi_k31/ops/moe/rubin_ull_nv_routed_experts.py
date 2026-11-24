# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The NVFP4 routed experts of one MoE layer at TP16 EP1:
``kernels/moe/rubin_ull_experts_nv_experts.py``'s ``RubinNvExpertsKernel`` (1-8 rows, the latent
image) and ``RubinNvGroupedExpertsKernel`` (9-64 rows, the grouped plan), vLLM MR8's Rubin experts
engines on the checkpoint's NVFP4 weights."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import cutlass
import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper

from ...kernels.moe import rubin_ull_experts_nv_experts as _nv
from ...kernels.moe import rubin_ull_experts_nv_experts_runtime as _rt

LATENT = 3584
SLICE = 192  # this rank's intermediate channels (3072 / TP16)
TOP_K = 16
IMAGE_ROWS = _rt.BATCH  # rows on the image engine; above: the grouped engine
PLAN_ROWS = _nv.PLAN_ROWS
#: The routing kernel's image count once a polled 8-row image is complete.
IMAGE_COUNT = 2 * IMAGE_ROWS - 1
#: 1 / 6 as the kernels' FP32 constant (``nv_scale``: ``0f3E2AAAAB``).
SIXTH = 0.1666666716337204
E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
_E2M1_BOUNDS = (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0)


@dataclass(frozen=True, eq=False)
class NvExpertsLaunch:
    """One row count's prepared launch (``moe.NvExperts._prepare``, at load, outside capture).

    ``code``: the compiled kernel; ``views``: CuTe views of (plan, alphas, w13 scales, w2 scales,
    workspace); ``output``: the CuTe view the unpublished rows land in (``raw`` when they are
    gathered into ``out``, else ``out``); ``box``: the publish box (CuTe scalars and views, () when
    the rows are not published); ``addresses``: (w13, w2) as Int64; ``scalars``: the SiTU
    constants as Float32. The torch tensors those views and addresses were made from follow
    (``plan``, ``packed``, ``workspace``, ``routes``: the route mailbox a polled launch reads),
    with ``situ`` the SiTU constants as floats, for ``reference`` and ``is_valid`` only.
    ``mode``: False (direct rows), True (latent multicast), "push" (also the shared rows into the
    peers' RS slots) or "twoshot" (the latent chunks into their owners' receive slots).
    """

    code: Any
    views: tuple
    output: Any
    box: tuple
    addresses: tuple
    scalars: tuple
    out: torch.Tensor
    raw: torch.Tensor | None
    direct_index: torch.Tensor | None
    rows: int
    polled: bool
    mode: Any
    plan: torch.Tensor
    packed: Mapping[str, torch.Tensor]
    workspace: torch.Tensor
    routes: torch.Tensor
    situ: tuple[float, float, float]

    @property
    def image(self) -> bool:
        return self.rows <= IMAGE_ROWS

    @property
    def published(self) -> bool:
        return bool(self.box)


def _cell(why: str, rows: int, mode: str = "direct", shared_ids: bool = False) -> Cell:
    """``mode``: "direct" (unpublished rows), "publish", "push" or "twoshot"; ``shared_ids``: every
    token routed to the same 16 experts (else 16 distinct random experts per token)."""
    return Cell(why=why, spec=dict(rows=rows, mode=mode, shared_ids=shared_ids))


class NvRoutedExperts(OpWrapper):
    """Each (token ``t``, top-k slot ``k``) row of expert ``e = ids[16 t + k]``:
    ``alpha2_e * W2_e . nvfp4(SiTU(alpha1_e * W1_e . x_t, alpha1_e * W3_e . x_t))``, from one
    persistent launch over this rank's 192 intermediate channels of all 896 experts.

    ``x_t`` is token ``t``'s NVFP4 latent [3584] in ``launch.plan`` (the routing kernel's image at
    1-8 rows, its grouped plan at 9-64); W1 (gate) / W3 (up) / W2 are the NVFP4 weights of
    ``weights.pack_routed_experts_tp16(..., publish=True)``; ``nvfp4`` is the hidden image
    (16-channel blocks, UE4M3 scales under ``alphas[E, 0]``); SiTU(g, u) = beta tanh(g / beta)
    sigmoid(g) * lb tanh(u / lb). ``ids`` int32 [>= 16 rows] (16 distinct experts per token; a
    polled launch reads them from ``launch.routes`` instead). Unpublished, the rows land in
    ``launch.out`` [16 rows, 3584] BF16 in model column order; published, each token's
    ``sum_k weights[16 t + k] * row`` (BF16 rows, FP32 slot-order sum) goes to the latent mailbox
    (``launch.box``) and ``launch.out`` is not written. Returns ``launch.out``.
    """

    ARCHS = frozenset({Arch.SM_103})

    CELLS: tuple[Cell, ...] = (
        _cell("b1 verify on GB300: 8 rows, polled route mailbox, direct rows + gather", 8),
        _cell("q1 decode: 1 row on the image engine (not polled)", 1),
        _cell("4 rows: the most Rubin keeps unpublished (q1 b4)", 4),
        _cell("b1 verify on Rubin: 8 rows published + shared rows pushed to the peers", 8, "push"),
        _cell("b2 verify: 16 rows on the grouped engine, latent multicast", 16, "publish"),
        _cell("b4 verify on GB300: 32 rows, latent multicast", 32, "publish"),
        _cell("b4 verify on Rubin: 32 rows, two-shot push to the owners' slots", 32, "twoshot"),
        _cell(
            "every token on the same 16 experts: units of 8 rows split over 16 rows",
            16,
            "publish",
            shared_ids=True,
        ),
    )

    note = """
    The routed experts of the K3.1 ULL generation target (MoE TP16 EP1, 896 experts, top 16, latent
    3584, 192 intermediate channels a rank). Persistent grids whose CTAs wait on one another: the
    whole grid must be co-resident, and the workspace (``nv_experts_runtime.scratch``) is shared by
    every layer and keeps its call parity, so launches on one workspace run in stream order. The
    latent plan is the routing kernel's (``SharedFront``, ``RouteSitu`` with the NVFP4 writers); a
    polled 8-row launch reads the ids and the image count from the route mailbox before its
    dependency wait, and its last CTA re-arms the mailbox. Compiled at ``prepare``, outside graph
    capture. The cells drive all 896 experts (the served bank's size) on one GB300 through the
    Rubin engines; the publish cells point the latent box at a one-device multicast object and the
    push / two-shot peers at local buffers.
    """

    def __call__(
        self,
        launch: NvExpertsLaunch,
        ids: torch.Tensor,
        weights: torch.Tensor | None,
        rows: int,
        stream: Any,
    ) -> torch.Tensor:
        output = _rt.to_cute(weights) if launch.box else launch.output
        plan, alphas, w13_sf, w2_sf, workspace = launch.views
        args = (plan, _rt.to_cute(ids), alphas, w13_sf, w2_sf, workspace, output)
        args += (cutlass.Int32(rows), *launch.scalars, *launch.box, *launch.addresses)
        launch.code(*args, stream)
        if launch.raw is not None:
            torch.index_select(launch.raw, -1, launch.direct_index, out=launch.out)
        return launch.out

    def reference(
        self,
        launch: NvExpertsLaunch,
        ids: torch.Tensor,
        weights: torch.Tensor | None,
        rows: int,
        stream: Any,
    ) -> torch.Tensor:
        """FP64. Unpublished: the [16 rows, 3584] expert rows in model column order. Published:
        the [rows, 3584] latent partial (each token's routing-weighted sum of its 16 rows) the
        launch publishes. The hidden image is quantized as the kernel does it, from the FP64
        hidden. ``stream`` is not read; the pushed shared rows are a copy of the caller's."""
        x = latent(launch.plan, rows, launch.image)
        expert_rows = torch.empty(rows * TOP_K, LATENT, dtype=torch.float64, device=x.device)
        flat = ids.reshape(-1)[: rows * TOP_K].long()
        h_gsf = float(launch.packed["alphas"][-1, 0])
        for start in range(0, flat.numel(), 32):
            pairs = flat[start : start + 32]
            w1, w3, w2, a1, a2 = dequantized_experts(launch.packed, pairs)
            xt = x[torch.arange(start, start + pairs.numel(), device=x.device) // TOP_K]
            gate = a1[:, None] * torch.bmm(w1, xt[:, :, None])[..., 0]
            up = a1[:, None] * torch.bmm(w3, xt[:, :, None])[..., 0]
            hidden = quantize_hidden(situ(gate, up, launch.situ), h_gsf)
            rows_out = a2[:, None] * torch.bmm(w2, hidden[:, :, None])[..., 0]
            expert_rows[start : start + pairs.numel()] = rows_out
        if not launch.published:
            return expert_rows
        w = weights.reshape(-1)[: rows * TOP_K].double().view(rows, TOP_K, 1)
        return (expert_rows.view(rows, TOP_K, LATENT) * w).sum(1)

    def is_valid(
        self,
        launch: NvExpertsLaunch,
        ids: torch.Tensor,
        weights: torch.Tensor | None,
        rows: int,
        stream: Any,
    ) -> None:
        """Raise where the kernel reads past or misreads its inputs without failing: another row
        count than the launch was prepared for, ids it would read as other experts (wrong dtype,
        out of range, an expert twice in a token), a published launch without BF16 weights, a
        direct launch whose ``out`` is not its [16 rows, 3584] rows, or a polled launch whose
        route mailbox does not hold the ids and the complete image count (it would spin)."""
        if rows != launch.rows:
            raise ValueError(
                f"nv_routed_experts: prepared for {launch.rows} rows, called with {rows}"
            )
        experts = launch.packed["w13"].shape[0]
        if ids.dtype != torch.int32 or not ids.is_contiguous() or ids.numel() < rows * TOP_K:
            raise ValueError("nv_routed_experts: ids must be contiguous int32 [>= 16 rows]")
        mine = ids.reshape(-1)[: rows * TOP_K].view(rows, TOP_K)
        if bool(((mine < 0) | (mine >= experts)).any()):
            raise ValueError("nv_routed_experts: an expert id outside the bank")
        if bool((mine.sort(1).values.diff(dim=1) == 0).any()):
            raise ValueError("nv_routed_experts: an expert twice in one token's top 16")
        if launch.published:
            if (
                weights is None
                or weights.dtype != torch.bfloat16
                or not weights.is_contiguous()
                or weights.numel() < rows * TOP_K
            ):
                raise ValueError("nv_routed_experts: published rows need BF16 weights [16 rows]")
        elif launch.out.shape != (rows * TOP_K, LATENT) or launch.out.dtype != torch.bfloat16:
            raise ValueError("nv_routed_experts: out must be the BF16 [16 rows, 3584] rows")
        if launch.polled:
            routes = launch.routes
            if (
                not torch.equal(routes[: rows * TOP_K], mine.reshape(-1))
                or int(routes[TOP_K * IMAGE_ROWS]) != IMAGE_COUNT
            ):
                raise ValueError("nv_routed_experts: the route mailbox does not hold this call")


nv_routed_experts = NvRoutedExperts()


# --------------------------------------------------------------------------------------------
# FP64 mirrors of the layouts (reference, and the test's band)
# --------------------------------------------------------------------------------------------


def _e2m1(codes: torch.Tensor) -> torch.Tensor:
    """Packed E2M1 bytes [..., n] (element 2 i in the low nibble) as FP64 [..., 2 n]."""
    table = torch.tensor(E2M1 + tuple(-v for v in E2M1), dtype=torch.float64, device=codes.device)
    nibbles = torch.stack((codes & 15, codes >> 4), dim=-1).reshape(*codes.shape[:-1], -1)
    return table[nibbles.long()]


def _ue4m3(scales: torch.Tensor) -> torch.Tensor:
    return scales.contiguous().view(torch.float8_e4m3fn).double()


def latent(plan: torch.Tensor, rows: int, image: bool) -> torch.Tensor:
    """The tokens' NVFP4 latent [rows, 3584] in FP64 (code x UE4M3 scale, i.e. under the
    activation global scale): the image's [14 stages][8 tokens][128 B swizzled] + [56][8][16 B]
    scale atoms, or the grouped plan's [64][1792 B] + [64][56][16 B] scale cells."""
    device = plan.device
    t = torch.arange(rows, device=device)[:, None]
    j = torch.arange(LATENT // 2, device=device)[None, :]
    block = torch.arange(LATENT // 16, device=device)[None, :]
    if image:
        chunk = j // 4
        unit = (chunk & 31) >> 2
        offset = (chunk >> 5) * _nv.IMAGE_STAGE + t * 128 + ((unit ^ t) & 7) * 16
        offset = offset + (chunk & 3) * 4 + j % 4
        scale = _nv.IMAGE_XSF + (block // 4) * 128 + t * 16 + block % 4
    else:
        offset = t * _nv.PLAN_ROW_BYTES + j
        scale = _nv.PLAN_XSF + t * _nv.PLAN_SCALE_ROW + (block // 4) * 16 + block % 4
    values = _e2m1(plan[offset])
    return values * _ue4m3(plan[scale]).repeat_interleave(16, dim=1)


def dequantized_experts(packed: Mapping[str, torch.Tensor], experts: torch.Tensor):
    """The experts ``experts`` (one per pair) of a publish-ordered bank in FP64: gate W1 / up W3
    [P, 192, 3584] and W2 [P, 3584, 192] in model column order (code x UE4M3 scale), and
    (alpha1, alpha2) [P]."""
    from ...kimi_k31_pro_nvfp4__sm_107__helix16_moetp16ep1.weights import (
        _unatoms,
        w2_row_order,
        w13_row_order,
    )

    device = packed["w13"].device
    inverse = w13_row_order().to(device).argsort()
    columns = w2_row_order(True).to(device).argsort()
    w13 = _e2m1(packed["w13"][experts][:, inverse])
    sf13 = _unatoms(packed["w13_sf"][experts], 3, LATENT // 64)[:, inverse]
    w13 = w13 * _ue4m3(sf13).repeat_interleave(16, dim=2)
    w2 = _e2m1(packed["w2"][experts][:, columns])
    sf2 = _unatoms(packed["w2_sf"][experts], LATENT // 128, SLICE // 64)[:, columns]
    w2 = w2 * _ue4m3(sf2).repeat_interleave(16, dim=2)
    alphas = packed["alphas"][experts].double()
    return w13[:, SLICE:], w13[:, :SLICE], w2, alphas[:, 0], alphas[:, 1]


def situ(gate: torch.Tensor, up: torch.Tensor, constants: tuple[float, float, float]):
    """``beta lb tanh(gate / beta) sigmoid(gate) tanh(up / lb)``; constants (beta lb, 1 / beta,
    1 / lb) as the kernel takes them."""
    beta_lb, inv_beta, inv_lb = constants
    return beta_lb * torch.tanh(gate * inv_beta) * torch.sigmoid(gate) * torch.tanh(up * inv_lb)


def e2m1_round(values: torch.Tensor) -> torch.Tensor:
    """Round to the nearest E2M1 value (ties to even, saturating at 6), sign kept."""
    bounds = torch.tensor(_E2M1_BOUNDS, dtype=values.dtype, device=values.device)
    table = torch.tensor(E2M1, dtype=values.dtype, device=values.device)
    magnitude = values.abs()
    index = torch.bucketize(magnitude, bounds)
    tie = magnitude == bounds[index.clamp_max(len(_E2M1_BOUNDS) - 1)]
    index = index + (tie & (index % 2 == 1)).long()
    return torch.copysign(table[index], values)


def e4m3_round(values: torch.Tensor) -> torch.Tensor:
    return values.to(torch.float8_e4m3fn).double()


def hidden_scales(amax: torch.Tensor, gsf: float) -> torch.Tensor:
    """The UE4M3 block scale of blocks whose absolute maximum is ``amax`` (``nv_scale``)."""
    return e4m3_round(amax * SIXTH * gsf)


def quantize_hidden(hidden: torch.Tensor, gsf: float) -> torch.Tensor:
    """The hidden image's values (code x scale) of FP64 ``hidden`` [..., 192]: 16-channel blocks,
    UE4M3 scale of amax / 6 under ``gsf``, E2M1 of ``hidden * gsf / scale``."""
    blocks = hidden.reshape(*hidden.shape[:-1], -1, 16)
    scale = hidden_scales(blocks.abs().amax(-1, keepdim=True), gsf)
    multiplier = torch.where(scale > 0, gsf / scale, torch.finfo(torch.float32).max)
    return (e2m1_round(blocks * multiplier) * scale).reshape(hidden.shape)
