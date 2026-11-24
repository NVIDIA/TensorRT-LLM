# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""dep16's MoE front levers on one sm_107 GPU: each keeps the bytes at every row count a dep16
generation graph step runs.

A dep16 generation rank's CUDA graph steps run ``_moe_dep_gen_streams`` on 8 rows per request
(k7: the request's token and seven drafts), 8 to 96 rows (b1 to b12). The levers checked here are
generation constants whose rows are module bounds, ``(first, last)``:

* ``megamoe_quantize_pdl`` (``_MOE_QUANT_PDL_M``): the NVFP4 quantization into MegaMoE's bucket
  launches programmatically (PDL) behind latent_down on the latent stream;
* ``moe_widen_after_fork`` (``_MOE_WIDEN_LATE_M``): where the mix stored no FP32 router input (up
  to 64 rows), the router input's widening is issued after the latent fork;
* ``moe_router_tf32_splitk`` (``_MOE_ROUTER_TF32_M``): the router GEMM writes the 8 K-slice
  partials of cuBLAS's TF32 split-K GEMM and the bucket's routing sums them in slice order. It
  runs where ``_moe_router_tf32_check`` passes on the layer's router weight under the bounds (the
  runtime's verdict, ``rt.moe_router_tf32``): at the committed 96 rows; with the bounds patched to
  every row count the check fails (cuBLAS cuts 1 to 16 rows otherwise) and nothing changes.

At each of those row counts, with the lever's bounds as committed, patched to every row count and
patched to none, the graph path's step runs eagerly (the graph runner's warm-up), then a CUDA graph
is captured on one set of inputs and replayed on two others; every run equals today's in-order step
(``cuda_graph=False``: every kernel on the current stream) on the same inputs, bit for bit.

Every kernel is real but MegaMoE's. Its launch (``_moe_bucket_launch``, and ``rubin_megamoe`` on the
staging path of 32 rows) is a stand-in that copies out what MegaMoE reads (this rank's NVFP4 rows and
their scales, the expert ids and weights of every row of the bucket) and returns rows computed from
them, so the norm and latent_up run on what the front produced. The bytes compared: what MegaMoE
reads, its stand-in's rows, latent_up's output and the shared experts' output.
"""

from __future__ import annotations

import functools
import importlib
from types import SimpleNamespace

import pytest
import torch

_DEP16 = (
    "tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31."
    "kimi_k31_pro_nvfp4__sm_107__dep16.modeling"
)
_HIDDEN, _LATENT, _EXPERTS, _TOP_K, _SHARED = 7168, 3584, 896, 16, 6144
_RANKS = 16
# A dep16 generation step's rows on one rank: 8 per request (k7), b1 to b12.
_ROWS = tuple(range(8, 97, 8))
# MegaMoE's bucket ladder up to the largest step.
_LADDER = (32, 64, 128)
# The SMs the shared gate_up GEMM leaves to MegaMoE's exchange on a graph step (EP16's exchange
# CTAs and the spares; the value does not change its output).
_RESERVE = 37
# Each lever: the module bound its graph step reads.
_BOUNDS = {
    "megamoe_quantize_pdl": "_MOE_QUANT_PDL_M",
    "moe_widen_after_fork": "_MOE_WIDEN_LATE_M",
    "moe_router_tf32_splitk": "_MOE_ROUTER_TF32_M",
}
_SETTINGS = {"committed": None, "every_row": (1, 4096), "none": (1, 0)}

_ON_SM107 = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 7),
    reason="dep16's kernels run on sm_107",
)


def _module():
    return importlib.import_module(_DEP16)


@functools.lru_cache(maxsize=None)
def _layer() -> SimpleNamespace:
    """One latent-MoE layer at dep16's shapes, seeded: the FP32 router, latent_down as the MXFP8
    lever holds it, the whole shared experts in MXFP8. MegaMoE's own weights are not read."""
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.gemm.rubin_mxfp8_gemm import (  # noqa: E501
        mxfp8_scales_to_r128c4,
    )

    mod = _module()
    gen = torch.Generator(device="cuda").manual_seed(2026)

    def randn(*shape, scale=1.0):
        return torch.randn(*shape, generator=gen, device="cuda") * scale

    def mx(n, k):
        w = (randn(n, k) * 0.5).to(torch.float8_e4m3fn).contiguous()
        e = torch.randint(120, 124, (n, k // 32), generator=gen, device="cuda", dtype=torch.int32)
        return w, mxfp8_scales_to_r128c4(e.to(torch.uint8)).contiguous()

    gu_w, gu_s = mx(2 * _SHARED, _HIDDEN)
    dn_w, dn_s = mx(_HIDDEN, _SHARED)
    return SimpleNamespace(
        # The checkpoint's BF16 router weight, loaded in FP32 (TF32-exact).
        router=randn(_EXPERTS, _HIDDEN, scale=_HIDDEN**-0.5)
        .to(torch.bfloat16)
        .float()
        .contiguous(),
        router_bias=randn(_EXPERTS, scale=0.05),
        latent_down=mod._Mx(*mx(_LATENT, _HIDDEN)),
        latent_up=randn(_HIDDEN, _LATENT, scale=_LATENT**-0.5).to(torch.bfloat16).contiguous(),
        latent_norm=torch.ones(_LATENT, dtype=torch.bfloat16, device="cuda"),
        shared_gu_w=gu_w,
        shared_gu_s=gu_s,
        shared_dn_w=dn_w,
        shared_dn_s=dn_s,
        fc31_in_scale=torch.full((1,), 96.0, device="cuda"),
        fc1_w=None,
        fc1_s=None,
        fc2_w=None,
        fc2_s=None,
        fc31_alpha=None,
        fc2_alpha=None,
        fc1_norm_const=None,
        expert_offset=0,
    )


def _experts(x4, sf, ids, weights) -> torch.Tensor:
    """MegaMoE's stand-in rows ``[rows, latent]`` BF16, computed from what it reads."""
    v = x4.float().repeat_interleave(2, dim=1) + sf.float().repeat_interleave(16, dim=1)
    return ((v + (ids.float() * weights).sum(-1, keepdim=True)) * 2.0**-8).to(torch.bfloat16)


class _Rig:
    """One rank's MoE step at ``rows`` rows (every rank holds as many) on the real dep16 module,
    MegaMoE's launch replaced by the stand-in (``monkeypatch``)."""

    def __init__(self, mod, rows: int, monkeypatch):
        self.mod, self.rows, self.L = mod, rows, _layer()
        u8 = dict(dtype=torch.uint8, device="cuda")
        self.buckets = {
            t: SimpleNamespace(
                max_tokens=t,
                activation=torch.full((t, _LATENT // 2), 0x5A, **u8),
                activation_sf=torch.full((t, _LATENT // 16), 0x5A, **u8),
                topk_idx=torch.full((t, _TOP_K), 7, dtype=torch.int32, device="cuda"),
                topk_weights=torch.full((t, _TOP_K), 0.5, device="cuda"),
            )
            for t in _LADDER
        }
        self.kept: dict = {}
        self.x = torch.empty(rows, _HIDDEN, dtype=torch.bfloat16, device="cuda")
        self.xf = torch.empty(rows, _HIDDEN, dtype=torch.float32, device="cuda")
        g = mod._CONSTANTS["generation"]
        # The post-attention mix stores the router's FP32 input above the PDL mix's rows.
        self.mix_f32 = bool(g["attn_res_router_f32"]) and rows > int(g["attn_res_pdl_max_m"])
        events = ("shared_fork", "shared_join", "latent_fork", "latent_ready", "memset_ready")
        self.rt = SimpleNamespace(
            moe_comm=SimpleNamespace(
                latent_stream=torch.cuda.Stream(),
                memset_stream=torch.cuda.Stream(),
                **{name: torch.cuda.Event() for name in events},
            ),
            megamoe=SimpleNamespace(ep_rank=0, chunk_tokens=4096),
            aux_stream=torch.cuda.Stream(),
            shared_gate_up_reserve=_RESERVE,
            moe_back_planes=None,  # MegaMoE is a stand-in that returns its rows reduced
            moe_router_tf32=False,
        )
        monkeypatch.setattr(mod, "megamoe_bucket", self._bucket)
        monkeypatch.setattr(mod, "_moe_bucket_launch", self._bucket_launch)
        monkeypatch.setattr(mod, "rubin_megamoe", self._staged_launch)

    def _bucket(self, ws, chunk_max, rows):
        return self.buckets[min(t for t in _LADDER if t >= chunk_max)]

    def _keep(self, name: str, t: torch.Tensor) -> None:
        raw = t.contiguous().view(torch.uint8)
        if name not in self.kept:
            self.kept[name] = torch.empty_like(raw)
        self.kept[name].copy_(raw)

    def _launch(self, x4, sf, ids, weights, ids_all, weights_all) -> torch.Tensor:
        for name, t in (("x4", x4), ("sf", sf), ("ids", ids_all), ("weights", weights_all)):
            self._keep(f"MegaMoE reads {name}", t)
        out = _experts(x4, sf, ids, weights)
        self._keep("MegaMoE stand-in rows", out)
        return out

    def _bucket_launch(self, L, b, rows, rt):
        bufs = (b.activation[:rows], b.activation_sf[:rows], b.topk_idx[:rows])
        return self._launch(*bufs, b.topk_weights[:rows], b.topk_idx, b.topk_weights)

    def _staged_launch(self, x4, sf, ids, weights, *weights_ws_chunk_rows):
        return self._launch(x4, sf.view(x4.shape[0], -1), ids, weights, ids, weights)

    def inputs(self, seed: int) -> torch.Tensor:
        """``[rows, hidden]`` BF16: a unit normal at a scale per row of 2^-6..2^6, a zero row and a
        tiny row (seed-dependent order)."""
        gen = torch.Generator(device="cuda").manual_seed(seed)
        x = torch.randn(self.rows, _HIDDEN, generator=gen, device="cuda")
        exponent = (torch.arange(self.rows, device="cuda") * seed % 13 - 6).float()
        x *= torch.pow(2.0, exponent)[:, None]
        x[seed % self.rows] = 0.0
        x[(seed + 3) % self.rows] *= 2.0**-120
        return x.to(torch.bfloat16)

    def load(self, x: torch.Tensor) -> None:
        self.x.copy_(x)
        self.xf.copy_(x.float())

    def run(self, cuda_graph: bool):
        st = SimpleNamespace(cuda_graph=cuda_graph, all_rank_num_tokens=(self.rows,) * _RANKS)
        xf = self.xf if self.mix_f32 else None
        return self.mod.latent_moe(self.L, 1, self.x, st, self.rt, xf)

    def snapshot(self, out) -> dict:
        snap = {name: t.clone() for name, t in self.kept.items()}
        snap["latent_up output"] = out.routed.view(torch.uint8).clone()
        snap["shared experts output"] = out.shared.view(torch.uint8).clone()
        return snap


def _differ(got: dict, want: dict) -> list:
    assert got.keys() == want.keys(), (sorted(got), sorted(want))
    return [name for name in want if not torch.equal(got[name], want[name])]


@_ON_SM107
@pytest.mark.parametrize("rows", _ROWS)
@pytest.mark.parametrize("setting", sorted(_SETTINGS))
@pytest.mark.parametrize("lever", sorted(_BOUNDS))
def test_a_lever_keeps_todays_bytes_eager_and_on_graph_replays(lever, setting, rows, monkeypatch):
    mod = _module()
    if _SETTINGS[setting] is not None:
        monkeypatch.setattr(mod, _BOUNDS[lever], _SETTINGS[setting])
    rig = _Rig(mod, rows, monkeypatch)
    if lever == "moe_router_tf32_splitk":  # the runtime's verdict under these bounds
        why = mod._moe_router_tf32_check([rig.L.router])
        assert (why is None) == (setting != "every_row"), why
        rig.rt.moe_router_tf32 = why is None
    a, b, c = (rig.inputs(seed) for seed in (1, 2, 3))
    want = []
    for x in (b, c):  # today's in-order step
        rig.load(x)
        out = rig.run(False)
        torch.cuda.synchronize()
        want.append(rig.snapshot(out))
    for x, w in zip((b, c), want):  # the graph path, eager: the graph runner's warm-up
        rig.load(x)
        out = rig.run(True)
        torch.cuda.synchronize()
        differ = _differ(rig.snapshot(out), w)
        assert not differ, f"{lever} ({setting}) at {rows} rows, eager: {differ}"
    rig.load(a)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = rig.run(True)
    torch.cuda.synchronize()
    for x, w in zip((b, c), want):
        rig.load(x)
        graph.replay()
        torch.cuda.synchronize()
        differ = _differ(rig.snapshot(out), w)
        assert not differ, f"{lever} ({setting}) at {rows} rows, graph replay: {differ}"
    assert int(rig.kept["MegaMoE reads x4"].numel()) == rows * _LATENT // 2


@_ON_SM107
@pytest.mark.parametrize("rows", _ROWS)
def test_the_router_tf32_check_at_each_generation_row_count(rows, monkeypatch):
    """Bounds of one row count: the check passes where cuBLAS cuts the TF32 GEMM in the 8 slices
    of 896 (24 rows and up) and the partials' sum is its logits, and refuses 8 and 16 rows
    (another cut); a weight with low mantissa bits is refused."""
    mod = _module()
    monkeypatch.setattr(mod, "_MOE_ROUTER_TF32_M", (rows, rows))
    router = _layer().router
    why = mod._moe_router_tf32_check([router])
    assert (why is None) == (rows >= 24), why
    noisy = router.clone()
    noisy[0, 0] = noisy[0, 0] * (1.0 + 2.0**-20)
    assert mod._moe_router_tf32_check([router, noisy]) == "a router weight is not TF32-exact FP32"
