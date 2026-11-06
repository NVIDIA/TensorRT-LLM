# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The megakernel's BF16 MoE experts stage on one rank (``stages/bf16Experts.cuh``, the unquantized MoE layers 0
and 61): from the router logits and the gathered pool output of the 8 verify rows to this rank's ``[8, 4096]`` BF16
partial of the expert output (the finalize all-reduce's input).

Input / output contract: the NVFP4 stage's (``experts.py``): ``logits`` FP32 ``[8, 256]`` and ``pool`` BF16
``[8, 4096]`` as Lamport words, ``out`` BF16 ``[8, 4096]``; with ``MOE_FLAG_MH_F2`` the F2' / finalize areas of the
MoE hop workspace instead. The pre-expert norm is the stock BF16 layers' ``RMSNorm(use_gemma=True)``: ``gain_pre``
here is the raw ``pre_expert_norm.weight`` (the stage applies ``1 + w`` in FP32), not the NVFP4 stage's
``bf16(w + 1)``.

Weights are the stock tensors of the two-window stack ``[own | r - 1]`` (``replicated_experts`` / ``ring_buddy``),
read in place: ``w3_w1_weight`` BF16 ``[64, 64, 8192, 64]`` and ``w2_weight`` BF16 ``[64, 64, 4096, 64]``
(``BF16TRTLLMGenFusedMoEMethod``: gate/up interleave, 32-row shuffle, BlockMajorK), each through a 2-D tensor map of
``[64 * 64 * R rows][128 B]``.
"""

from __future__ import annotations

import ctypes
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch

from . import experts, library, schedule_bf16_experts

DESC = experts.DESC
DESC_FIELDS = experts.DESC_FIELDS
WINDOW = 64
HIDDEN = 4096
INTER = 4096
EXPERTS = 256
TOP_K = 8
PRE_EXPERT_EPS = experts.PRE_EXPERT_EPS
POST_EXPERT_EPS = experts.POST_EXPERT_EPS
RENORM_EPS = experts.RENORM_EPS
DBG_X = 1280

lamport_words = experts.lamport_words


def _bind_abi(lib: library.Library) -> None:
    c = lib.lib
    if getattr(c, "_z1_bound", False):
        return
    for name in ("msoMkBfWsBytes", "msoMkBfDbgBytes", "msoMkBfN", "msoMkBfCompiled"):
        getattr(c, name).restype = ctypes.c_int
        getattr(c, name).argtypes = []
    c.msoMkBfMagic.restype = ctypes.c_int64
    c.msoMkBfMagic.argtypes = []
    c.msoMkBfTmap.restype = ctypes.c_int
    c.msoMkBfTmap.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_ulonglong,
        ctypes.c_ulonglong,
    ]
    c.msoMkBfUnitKb.restype = ctypes.c_int
    c.msoMkBfUnitKb.argtypes = []
    c.msoMkBfStages.restype = ctypes.c_int
    assert c.msoMkBfCompiled() == 1, (
        f"mode {lib.mode} does not compile the BF16 experts stage (use a *_z1 mode)"
    )
    c._z1_bound = True


def tensor_map(lib: library.Library, stack: torch.Tensor) -> torch.Tensor:
    """The device copy (128 B) of the SW128 tensor map of a BlockMajorK BF16 stack ``[E, K / 64, R, 64]``."""
    _bind_abi(lib)
    assert (
        stack.dtype == torch.bfloat16
        and stack.is_contiguous()
        and stack.dim() == 4
        and stack.shape[-1] == 64
    )
    rows = stack.shape[2]
    planes = stack.shape[0] * stack.shape[1]
    host = torch.zeros(128, dtype=torch.uint8)
    rc = lib.lib.msoMkBfTmap(
        ctypes.c_void_p(host.data_ptr()),
        ctypes.c_void_p(stack.data_ptr()),
        ctypes.c_ulonglong(rows),
        ctypes.c_ulonglong(planes),
    )
    assert rc == 0, f"cuTensorMapEncodeTiled: {rc}"
    return host.to(stack.device)


@dataclass
class Bf16Window:
    """One BF16 layer's two-window expert stack on this rank (the stock tensors, in place)."""

    w13: torch.Tensor  # BF16 [64, 64, 8192, 64]
    w2: torch.Tensor  # BF16 [64, 64, 4096, 64]

    def check(self) -> None:
        assert (
            tuple(self.w13.shape) == (WINDOW, HIDDEN // 64, 2 * INTER, 64)
            and self.w13.is_contiguous()
        )
        assert tuple(self.w2.shape) == (WINDOW, INTER // 64, HIDDEN, 64) and self.w2.is_contiguous()
        assert self.w13.dtype == torch.bfloat16 and self.w2.dtype == torch.bfloat16


@dataclass
class LayerInputs:
    """Per-layer buffers the descriptor points at (graph-stable)."""

    logits: torch.Tensor  # FP32 [8, 256], Lamport words
    pool: torch.Tensor  # BF16 [8, 4096], Lamport words
    out: torch.Tensor  # BF16 [8, 4096]
    dbg: Optional[torch.Tensor] = None  # u8 [kBfDbgBytes] (check)
    rows: Optional[torch.Tensor] = None  # BF16 [64, 4096] FC2 rows (row t * 8 + k) (check)


class Bf16ExpertsStage:
    """The BF16 experts stage on its own: one ``Megakernel`` of ``bf16_experts_tables`` and per layer its parameter
    blocks. ``mode`` must be a ``*_z1`` library mode."""

    def __init__(
        self,
        mode: str = "check_z1",
        copies: int = 3,
        workspace: Optional[library.Workspace] = None,
        timeout_ms: int = 20000,
    ):
        self.lib = library.library(mode)
        _bind_abi(self.lib)
        self.ws = workspace or library.Workspace(self.lib, copies)
        dev = self.ws.arena.device
        c = self.lib.lib
        self.ws_bytes = c.msoMkBfWsBytes()
        self.dbg_bytes = c.msoMkBfDbgBytes()
        self.magic = int(c.msoMkBfMagic())
        self.mma_n = c.msoMkBfN()
        empty = lambda nbytes: torch.full((nbytes // 4,), -0.0, dtype=torch.float32, device=dev)  # noqa: E731
        self.moe_ws = empty(self.ws.copies * self.ws_bytes)
        experts._bind_abi(self.lib)
        self.part = empty(c.msoMkMoePartBytes())
        self.tables = schedule_bf16_experts.bf16_experts_tables(self.lib.ctas)
        self.mk = library.Megakernel(self.lib, self.ws, self.tables, timeout_ms=timeout_ms)

    def inputs(self, check: bool = False) -> LayerInputs:
        dev = self.ws.arena.device
        return LayerInputs(
            logits=torch.zeros(8, EXPERTS, dtype=torch.float32, device=dev),
            pool=torch.zeros(8, HIDDEN, dtype=torch.bfloat16, device=dev),
            out=torch.zeros(8, HIDDEN, dtype=torch.bfloat16, device=dev),
            dbg=torch.zeros(self.dbg_bytes, dtype=torch.uint8, device=dev) if check else None,
            rows=torch.zeros(64, HIDDEN, dtype=torch.bfloat16, device=dev) if check else None,
        )

    def make_layer(
        self,
        window: Bf16Window,
        bias: torch.Tensor,
        w_pre: torch.Tensor,
        w_post: torch.Tensor,
        rank: int,
        io: LayerInputs,
        tile_cut: bool = False,
        stamps: bool = False,
        extra_flags: int = 0,
        own_copy: Optional[Tuple[int, int]] = None,
        nvfp4_ws: Optional[torch.Tensor] = None,
    ):
        """One layer's parameter blocks. ``bias`` FP32 ``[256]``; ``w_pre`` BF16 ``[4096]`` the raw
        ``pre_expert_norm.weight`` (gain ``1 + w``); ``w_post`` BF16 ``[4096]`` (``post_expert_norm_weight``).

        ``own_copy = (copy, copies)``: the layer uses that copy of the stage's workspace and clears copy ``copy + 1``
        (the BF16 layers of a forward that also runs NVFP4 layers: they are not every launch of the rotation);
        default: the launch rotation. ``nvfp4_ws``: the NVFP4 stage's workspace, whose copy ``n + 1`` this launch
        then clears in that stage's place."""
        window.check()
        dev = self.ws.arena.device
        assert bias.dtype == torch.float32 and bias.numel() == EXPERTS and bias.is_contiguous()
        assert w_pre.dtype == torch.bfloat16 and w_pre.numel() == HIDDEN and w_pre.is_contiguous()
        assert (
            w_post.dtype == torch.bfloat16 and w_post.numel() == HIDDEN and w_post.is_contiguous()
        )
        t13 = tensor_map(self.lib, window.w13)
        t2 = tensor_map(self.lib, window.w2)
        flags = (experts.MOE_FLAG_TILE_CUT if tile_cut else 0) | int(extra_flags)
        vals = dict(
            Magic=self.magic,
            Logits=io.logits.data_ptr(),
            Pool=io.pool.data_ptr(),
            Bias=bias.data_ptr(),
            GainPre=w_pre.data_ptr(),
            Gs=0,
            TmapW13=t13.data_ptr(),
            SfW13=0,
            TmapW2=t2.data_ptr(),
            SfW2=0,
            Alpha1=0,
            ScaleC=0,
            Alpha2=0,
            GainPost=w_post.data_ptr(),
            Ws=self.moe_ws.data_ptr(),
            Part=self.part.data_ptr(),
            Rank=int(rank),
            Out=io.out.data_ptr(),
            Dbg=0 if io.dbg is None else io.dbg.data_ptr(),
            Eps=experts._f32_bits(PRE_EXPERT_EPS) | (experts._f32_bits(POST_EXPERT_EPS) << 32),
            RenormEps=experts._f32_bits(RENORM_EPS),
            Flags=flags,
            Rows=0 if io.rows is None else io.rows.data_ptr(),
            Reserved=0,
        )
        if own_copy is not None:
            cp, ncp = own_copy
            assert (
                0 <= cp < ncp <= 8 and ncp >= 2 and self.moe_ws.numel() * 4 >= ncp * self.ws_bytes
            )
            vals["Reserved"] = (1 + int(cp)) | (int(ncp) << 8)
        if nvfp4_ws is not None:
            vals["Gs"] = nvfp4_ws.data_ptr()
        host = torch.zeros(len(DESC_FIELDS), dtype=torch.int64)
        for name, v in vals.items():
            host[DESC[name]] = int(v) if int(v) < 2**63 else int(v) - 2**64
        desc = host.to(dev)
        keep = (desc, t13, t2, window, bias, w_pre, w_post, io)
        return self.mk.make_layer(
            dict(Moe=desc.data_ptr()), flags=library.FLAG_EVICT_FIRST, stamps=stamps, keep=keep
        )

    def __call__(self, layer, pdl: bool = True):
        return self.mk.launch(layer, library.KernelIo(), pdl=pdl)


def debug_view(dbg: torch.Tensor) -> Dict[str, torch.Tensor]:
    """The debug block's fields (host tensors)."""
    d = dbg.cpu()
    i32 = lambda off, n: d[off : off + 4 * n].view(torch.int32).clone()  # noqa: E731
    E = int(i32(experts.DBG_SLOTS, 1)[0])
    return dict(
        selected=i32(experts.DBG_SEL, 64).view(8, 8),
        weights=d[experts.DBG_W : experts.DBG_W + 256].view(torch.float32).clone().view(8, 8),
        local_ids=i32(experts.DBG_LOCAL, 64).view(8, 8),
        loads=i32(experts.DBG_LOADS, 16),
        E=torch.tensor(E),
        slots=i32(experts.DBG_SLOTS + 4, 64)[:E],
        x=d[DBG_X : DBG_X + 8 * HIDDEN * 2].clone().view(torch.bfloat16).view(8, HIDDEN),
    )


# ---- stock references (single GPU) -----------------------------------------------------------------------------------
def stock_placement(selected: torch.Tensor, weights: torch.Tensor, rank: int):
    """``ring_buddy_assign`` (the served Triton kernel):
    ``(local ids int32 [8, 8], weights BF16 [8, 8], stats [16])``."""
    from ...ops.moe.ring_buddy_assign import ring_buddy_assign

    return ring_buddy_assign(selected.contiguous(), weights.contiguous(), rank)


_NORMS: Dict[int, torch.nn.Module] = {}


def stock_norm(pool: torch.Tensor, w_pre: torch.Tensor) -> torch.Tensor:
    """The stock BF16 layers' pre-expert norm: ``RMSNorm(use_gemma=True)`` (flashinfer), BF16 ``[8, 4096]``. The
    module is made once per weight tensor (outside any graph capture on first use)."""
    from tensorrt_llm._torch.modules.rms_norm import RMSNorm

    key = w_pre.data_ptr()
    if key not in _NORMS:
        norm = RMSNorm(
            hidden_size=HIDDEN, eps=PRE_EXPERT_EPS, dtype=torch.bfloat16, use_gemma=True
        ).to(pool.device)
        with torch.no_grad():
            norm.weight.copy_(w_pre)
        _NORMS[key] = norm
    with torch.no_grad():
        return _NORMS[key](pool)


_ENUMS = []


def _flashinfer_enums():
    if not _ENUMS:
        from tensorrt_llm._torch.moe.fused_moe.moe_op_backend import FlashinferOpBackend
        from tensorrt_llm._torch.moe.fused_moe.routing import RoutingMethodType

        ob = FlashinferOpBackend()
        _ENUMS.extend(
            [ob.cvt_routing_method_type(RoutingMethodType.MiniMax2), ob.cvt_activation_type(0)]
        )
    return _ENUMS[0], _ENUMS[1]


def stock_experts(
    x: torch.Tensor, local_ids: torch.Tensor, weights_bf16: torch.Tensor, window: Bf16Window
):
    """``bf16_routed_experts`` (FlashInfer TRTLLM-gen BF16, the served call) on the same input: ``(rows [64, 4096] BF16
    indexed t * 8 + k (zero where the pair is not local), gemm2 output, expanded_idx_to_permuted_idx)``."""
    from tensorrt_llm._torch.moe.fused_moe.quantization import (
        TRTLLM_GEN_WEIGHT_LAYOUT_BLOCK_MAJOR_K,
    )

    from ...ops.moe.bf16_routed_experts import bf16_routed_experts

    rmt, act = _flashinfer_enums()
    out = bf16_routed_experts(
        x.contiguous(),
        window.w13,
        window.w2,
        local_ids.to(torch.int32).contiguous(),
        weights_bf16.to(torch.bfloat16).contiguous(),
        INTER,
        0,
        WINDOW,
        rmt,
        act,
        True,
        int(TRTLLM_GEN_WEIGHT_LAYOUT_BLOCK_MAJOR_K),
        EXPERTS,
        TOP_K,
    )
    gemm2, _, index = out[0], out[1], out[2]
    index = index.reshape(-1).to(torch.int64)
    rows = torch.zeros(8 * TOP_K, HIDDEN, dtype=torch.bfloat16, device=x.device)
    sel = (index >= 0).nonzero().reshape(-1)
    rows[sel] = gemm2.view(-1, HIDDEN)[index[sel]].to(torch.bfloat16)
    return rows, gemm2, index


def stock_combine(
    rows: torch.Tensor, local_ids: torch.Tensor, weights: torch.Tensor, w_post: torch.Tensor
):
    """The served weighted post-norm sum (``post_expert_norm_combine``, FP32) on rows indexed ``t * 8 + k``, rounded
    to BF16 with -0 cleared: this rank's partial."""
    from tensorrt_llm._torch.models.modeling_musespark_open import post_expert_norm_combine

    valid = (local_ids.reshape(-1) >= 0) & (local_ids.reshape(-1) < WINDOW)
    index = torch.where(
        valid, torch.arange(64, device=rows.device), torch.full_like(valid, -1, dtype=torch.int64)
    )
    out = post_expert_norm_combine(
        rows,
        index.to(torch.int32),
        weights,
        w_post,
        POST_EXPERT_EPS,
        num_tokens=8,
        top_k=TOP_K,
        hidden_size=HIDDEN,
    )
    b = out.to(torch.bfloat16)
    return torch.where(b == 0, torch.zeros_like(b), b)


# ---- float64 references ----------------------------------------------------------------------------------------------
def norm_exact(
    pool: torch.Tensor, w_pre: torch.Tensor, eps: float = PRE_EXPERT_EPS
) -> torch.Tensor:
    """``x rstd (1 + w)`` in float64."""
    x = pool.double()
    rstd = torch.rsqrt(x.square().mean(-1, keepdim=True) + eps)
    return x * rstd * (1.0 + w_pre.double())


def experts_exact(x: torch.Tensor, local_ids: torch.Tensor, window: Bf16Window) -> torch.Tensor:
    """FP64 rows ``[64, 4096]`` (``t * 8 + k``; zero where not local) of ``W2 (silu(x W1^T) (x W3^T))`` on the float64
    input ``x`` (no intermediate rounding), from the unshuffled stock stacks."""
    from ...ops.moe.bf16_routed_experts import _unshuffled

    rows = torch.zeros(8 * TOP_K, HIDDEN, dtype=torch.float64, device=x.device)
    ids = local_ids.reshape(-1).tolist()
    cache: Dict[int, tuple] = {}
    for tk, e in enumerate(ids):
        if not 0 <= e < WINDOW:
            continue
        if e not in cache:
            w13, w2 = _unshuffled(window.w13[e : e + 1], window.w2[e : e + 1])
            cache[e] = (w13[0].double(), w2[0].double())
        w13, w2 = cache[e]
        h = w13 @ x[tk // TOP_K].double()
        up, gate = h[:INTER], h[INTER:]
        rows[tk] = w2 @ (gate * torch.sigmoid(gate) * up)
    return rows


def combine_exact(
    rows: torch.Tensor, local_ids: torch.Tensor, weights: torch.Tensor, w_post: torch.Tensor
):
    """The weighted post-norm sum in float64 from given rows: ``sum_k w_k rsqrt(mean(rho_k^2) + eps) rho_k``,
    ``rho = y (w_post + 1)``."""
    gain = w_post.double() + 1.0
    y = rows.double().view(8, TOP_K, HIDDEN) * gain
    valid = (local_ids >= 0) & (local_ids < WINDOW)
    rstd = torch.rsqrt(y.square().mean(-1) + POST_EXPERT_EPS)
    scale = torch.where(valid.to(rows.device), weights.double() * rstd, torch.zeros_like(rstd))
    return (scale[..., None] * y).sum(1)


# ---- one real BF16 layer from the checkpoint -------------------------------------------------------------------------
def stock_bf16_stack(up_gate_rows: torch.Tensor, w2_rows: torch.Tensor):
    """One expert's ``[w3; w1]`` ``[8192, 4096]`` and ``w2`` ``[4096, 4096]`` (BF16, unpermuted) in the leaf's layout
    (``BF16TRTLLMGenFusedMoEMethod.process_weights_after_loading``): permuted rows, BlockMajorK."""
    from tensorrt_llm._torch.moe.fused_moe.quantization import BF16TRTLLMGenFusedMoEMethod

    m = BF16TRTLLMGenFusedMoEMethod()
    p13 = m._get_w3_w1_permute_indices(up_gate_rows)
    p2 = m._get_w2_permute_indices(w2_rows)
    return (
        m._prepare_bf16_weight_for_trtllm_gen(up_gate_rows, p13),
        m._prepare_bf16_weight_for_trtllm_gen(w2_rows, p2),
    )


def load_real_bf16_layer(ckpt: str, layer: int, rank: int, device) -> Dict[str, object]:
    """Layer ``layer``'s (0 or 61) MoE tensors for rank ``rank``'s two-window stack ``[own | r - 1]``, converted to
    the leaf's layout. The checkpoint holds ``gate_up_proj [256, 8192, 4096]`` BF16 (rows ``[0, 4096)`` gate = w1,
    ``[4096, 8192)`` up = w3) and ``down_proj [256, 4096, 4096]``; the leaf's per-expert ``w3_w1`` is ``[w3; w1]``."""
    import json
    import os

    from safetensors import safe_open

    pre = f"model.language_model.layers.{layer}.mlp."
    with open(os.path.join(ckpt, "model.safetensors.index.json")) as f:
        wmap = json.load(f)["weight_map"]
    handles: Dict[str, object] = {}

    def get(name, sl=None):
        fn = wmap[pre + name]
        if fn not in handles:
            handles[fn] = safe_open(os.path.join(ckpt, fn), framework="pt", device="cpu")
        h = handles[fn]
        return h.get_slice(pre + name)[sl] if sl is not None else h.get_tensor(pre + name)

    blocks = [rank % 8, (rank - 1) % 8]  # [own | r - 1]
    w13 = torch.empty(WINDOW, HIDDEN // 64, 2 * INTER, 64, dtype=torch.bfloat16, device=device)
    w2 = torch.empty(WINDOW, INTER // 64, HIDDEN, 64, dtype=torch.bfloat16, device=device)
    for j, b in enumerate(blocks):
        for e0 in range(0, 32, 8):
            gu = get("experts.gate_up_proj", slice(32 * b + e0, 32 * b + e0 + 8)).to(device)
            dn = get("experts.down_proj", slice(32 * b + e0, 32 * b + e0 + 8)).to(device)
            assert gu.dtype == torch.bfloat16 and tuple(gu.shape[1:]) == (2 * INTER, HIDDEN), (
                gu.shape
            )
            for e in range(gu.shape[0]):
                ug = torch.cat([gu[e, INTER:], gu[e, :INTER]], dim=0).contiguous()
                a, c = stock_bf16_stack(ug, dn[e].contiguous())
                w13[32 * j + e0 + e].copy_(a)
                w2[32 * j + e0 + e].copy_(c)
            del gu, dn
    return dict(
        window=Bf16Window(w13=w13, w2=w2),
        bias=get("gate.e_score_correction_bias").float().to(device).contiguous(),
        w_pre=get("pre_expert_norm.weight").to(torch.bfloat16).to(device).contiguous(),
        w_post=get("experts.post_expert_norm.weight").to(torch.bfloat16).to(device).contiguous(),
    )
