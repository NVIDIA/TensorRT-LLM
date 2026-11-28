# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU certification matrix for the MegaMoE entries on four GPUs (EP4, attention-DP rows).

The kernel exchanges tokens over NVLink, so it cannot run in one process: this script is its own
launcher. Run it plainly and it re-executes itself under `mpirun`, one rank per device named in
CUDA_VISIBLE_DEVICES, under a deadline the parent enforces by killing the whole process group (a
wedged NVLink barrier hangs rather than raising).

    CUDA_VISIBLE_DEVICES=0,1,2,3 python _cute_dsl_megamoe_nvfp4_op_matrix.py

Entries covered: `cute_dsl_megamoe_nvfp4` (the kernel; every cell of `CuteDslMegaMoeNvfp4.CELLS` is driven, the
matrix asserts it at the end), `megamoe_workspace` (symmetric buffers and EP group), and with them the placement
helpers of `megamoe_expert_slots`, `megamoe_pack_experts`, `megamoe_shared_expert` and `megamoe_stage_inputs` as the
composition a target runs (the routing launch's slot lookup and MXFP8 of the shared expert's input as plain torch,
`_megamoe_reference`). The collected entry point is `test_modeling_v2_cute_dsl_megamoe_nvfp4.py`; neither this file's
name nor its function names match pytest's collection patterns, because the checks are one fixed sequence in a single
4-rank job.

Reference: the weights of expert `e` are a pure function of `e` (every rank can regenerate any
expert), the activation is the exact dequantization of what the kernel was fed, and the math is the
checkpoint's: clamped SwiGLU in real units, the routing weight applied before the FC2 requantization,
NVFP4 intermediate with the expert's own global scale. Tolerance follows the engine's MegaMoE tests:
elementwise `|out - ref| <= atol + rtol * |ref|` for >= 92% of elements at this error accumulation.
"""

import os
import signal
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # _megamoe_reference
from _megamoe_reference import megamoe_slots, mxfp8_rows, mxfp8_scale_atoms  # noqa: E402

assert torch.cuda.is_available(), "the MegaMoE kernel requires CUDA devices"

_WORKER_FLAG = "--rank-worker"
DEADLINE_S = 2000

HIDDEN = 5120
INTER = 2304
EXPERTS = 128
TOP_K = 6
CLAMP = 5.0  # low enough that the clamp bites on ~10% of the activations (checkpoint: 10)
MAX_T = 256
ROWS = [64, 17, 0, 1]  # per-rank rows: a busy rank, a short one, an idle one, a single row
ROUTED_SCALE = 1.5
SF_BYTES = HIDDEN // 16
# The engine's MegaMoE tolerance at this error accumulation (intermediate * top_k > 10000).
RTOL, ATOL, PERCENT = 0.1, 0.15, 0.92

E2M1 = None  # bound in the rank body (needs a device)
RANK = 0
WORLD = 1
DEV = "cuda"
GROUP = None
M: Dict[str, object] = {}  # entries, bound in the rank body
DRIVEN = (
    set()
)  # (world, bucket, per-rank rows, local slots, shared expert, combine format) of every launch made


def _say(msg: str) -> None:
    if RANK == 0:
        print(f"[matrix] {msg}", flush=True)


# ── expert weights: a pure function of the expert id ──────────────────────────


def _rand_nvfp4(gen, n: int, k: int):
    codes = torch.randint(0, 16, (n, k), dtype=torch.uint8, device=DEV, generator=gen)
    exps = torch.randint(-3, 1, (n, k // 16), device=DEV, generator=gen, dtype=torch.int32).float()
    sf = torch.exp2(exps).to(torch.float8_e4m3fn)
    packed = (codes[:, 0::2] | (codes[:, 1::2] << 4)).contiguous()
    return packed, sf, codes


def _dequant(codes: torch.Tensor, sf: torch.Tensor, scale2: float) -> torch.Tensor:
    return E2M1[codes.long()] * sf.float().repeat_interleave(16, dim=1) * scale2


_EXPERT_CACHE: Dict[int, dict] = {}


def expert(e: int) -> dict:
    """Expert `e`: packed codes, e4m3 scales and the checkpoint scalars. Identical on every rank."""
    cached = _EXPERT_CACHE.get(e)
    if cached is not None:
        return cached
    gen = torch.Generator(device=DEV).manual_seed(7000 + e)
    w1, s1, c1 = _rand_nvfp4(gen, INTER, HIDDEN)
    w3, s3, c3 = _rand_nvfp4(gen, INTER, HIDDEN)
    w2, s2, c2 = _rand_nvfp4(gen, HIDDEN, INTER)
    u = torch.rand(4, generator=torch.Generator().manual_seed(9000 + e)).tolist()
    ws1 = 0.05 * (0.8 + 0.4 * u[0])
    # every fifth / seventh expert has a different weight_scale_2 for up than for gate (a power of two, so
    # scaling the smaller half's block scales down onto the larger is exact)
    ws3 = ws1 * (2.0 if e % 5 == 0 else (0.5 if e % 7 == 0 else 1.0))
    ws2 = 0.01 * (0.8 + 0.4 * u[1])
    isc2 = 0.04 * (0.8 + 0.4 * u[2])  # per-expert FC2 activation scale
    out = dict(
        w1=w1,
        s1=s1,
        c1=c1,
        w3=w3,
        s3=s3,
        c3=c3,
        w2=w2,
        s2=s2,
        c2=c2,
        ws1=ws1,
        ws3=ws3,
        ws2=ws2,
        isc2=isc2,
    )
    if len(_EXPERT_CACHE) < 40:  # a rank's experts at most; the reference regenerates the rest
        _EXPERT_CACHE[e] = out
    return out


INPUT_SCALE_FC1 = 0.0017  # layer-wide: max over every expert's w1/w3 input_scale (amax 4.6 / 2688)


def build_stacks(assignments: List[int], rank: int, local: int):
    """This rank's packed kernel operands for the experts its slots hold."""
    ids = assignments[rank * local : (rank + 1) * local]
    ex = [expert(e) for e in ids]
    t = lambda key: torch.stack([x[key] for x in ex])  # noqa: E731
    f = lambda key: torch.tensor([x[key] for x in ex], dtype=torch.float32, device=DEV)  # noqa: E731
    scalars = M["pack"].megamoe_expert_scalars(
        torch.tensor(INPUT_SCALE_FC1, device=DEV), f("isc2"), f("ws1"), f("ws3"), f("ws2")
    )
    fc1, fc1_sf, fc2, fc2_sf = M["pack"].megamoe_pack_experts(
        t("w1"), t("s1"), t("w3"), t("s3"), t("w2"), t("s2"),
        w3_scale_fold=scalars["w3_scale_fold"], w1_scale_fold=scalars["w1_scale_fold"],
    )  # fmt: skip
    return dict(fc1=fc1, fc1_sf=fc1_sf, fc2=fc2, fc2_sf=fc2_sf, scalars=scalars)


# ── shared expert (FP8 weights, UE8M0 block-32 scales) ────────────────────────


def make_shared():
    gen = torch.Generator(device=DEV).manual_seed(4242)

    def proj(n, k):
        w = (
            (torch.randn(n, k, device=DEV, generator=gen) * 1.0)
            .clamp(-6, 6)
            .to(torch.float8_e4m3fn)
        )
        exps = torch.randint(
            119, 124, (n // 32, k // 32), device=DEV, generator=gen, dtype=torch.int32
        )
        return w, exps.to(torch.uint8)

    w1, s1 = proj(INTER, HIDDEN)
    w3, s3 = proj(INTER, HIDDEN)
    w2, s2 = proj(HIDDEN, INTER)
    packed = M["shared"].megamoe_pack_shared_expert(w1, s1, w3, s3, w2, s2, CLAMP, (32, 32))

    def real(w, s):
        scale = torch.exp2(s.float() - 127.0).repeat_interleave(32, 0).repeat_interleave(32, 1)
        return w.float() * scale

    return packed, (real(w1, s1), real(w3, s3), real(w2, s2))


def mxfp8_roundtrip(x: torch.Tensor) -> torch.Tensor:
    """The kernel's MXFP8 quantization of the shared expert's input, as plain torch."""
    rows, k = x.shape
    b = x.float().view(rows, k // 32, 32)
    amax = b.abs().amax(-1)
    bits = (amax * (1.0 / 448.0)).view(torch.int32)
    e8 = torch.clamp((bits >> 23) + ((bits & 0x7FFFFF) != 0).to(torch.int32), max=254)
    scale = torch.exp2(e8.float() - 127.0)[..., None]
    q = (b / scale).to(torch.float8_e4m3fn).float()
    return (q * scale).view(rows, k)


# ── reference ────────────────────────────────────────────────────────────────


def e2m1_rne(a: torch.Tensor) -> torch.Tensor:
    step = torch.where(a.abs() < 2.0, 0.5, torch.where(a.abs() < 4.0, 1.0, 2.0))
    return torch.sign(a) * torch.clamp(torch.round(a.abs() / step) * step, max=6.0)


def nvfp4_roundtrip(a: torch.Tensor, g: float) -> torch.Tensor:
    """NVFP4 quantize + dequantize under global scale `g` (16-element e4m3 block scales)."""
    rows, k = a.shape
    b = a.reshape(rows, k // 16, 16)
    vmax = b.abs().amax(-1, keepdim=True)
    sf = (g * vmax / 6.0).clamp(max=448.0).to(torch.float8_e4m3fn).float()
    inv = torch.where(sf == 0, torch.zeros_like(sf), g / sf)
    return (e2m1_rne(b * inv) * sf / g).reshape(rows, k)


def unpack_activation(data: torch.Tensor, sf: torch.Tensor, g1: float) -> torch.Tensor:
    rows = data.shape[0]
    codes = torch.stack([data & 0x0F, data >> 4], dim=-1).reshape(rows, HIDDEN).long()
    scale = sf.view(torch.float8_e4m3fn).reshape(rows, HIDDEN // 16).float()
    return E2M1[codes] * scale.repeat_interleave(16, dim=1) / g1


def swiglu(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.silu(gate.clamp(max=CLAMP)) * up.clamp(-CLAMP, CLAMP)


def _expert_real(e: int):
    ex = expert(e)
    return (
        _dequant(ex["c1"], ex["s1"], ex["ws1"]),
        _dequant(ex["c3"], ex["s3"], ex["ws3"]),
        _dequant(ex["c2"], ex["s2"], ex["ws2"]),
        1.0 / ex["isc2"],
    )


def reference_routed(xhat, ids, w):
    return M["op"].cute_dsl_megamoe_nvfp4.reference(xhat, ids, w, _expert_real, CLAMP)


def reference_shared(x, real):
    w1, w3, w2 = real
    zero = torch.zeros((x.shape[0], 1), device=DEV)
    return M["op"].cute_dsl_megamoe_nvfp4.reference(
        torch.zeros_like(x),
        torch.zeros((x.shape[0], 1), dtype=torch.long, device=DEV),
        zero,
        lambda e: (
            torch.zeros_like(w1),
            torch.zeros_like(w3),
            torch.zeros(HIDDEN, INTER, device=DEV),
            1.0,
        ),
        CLAMP,
        shared=(w1, w3, w2, x, CLAMP),
    )


def rel_rms(value: torch.Tensor, ref: torch.Tensor) -> float:
    return (
        (value.float() - ref).pow(2).mean().sqrt() / ref.pow(2).mean().sqrt().clamp(min=1e-12)
    ).item()


def within(value: torch.Tensor, ref: torch.Tensor) -> float:
    ok = (value.float() - ref).abs() <= ATOL + RTOL * ref.abs()
    return ok.float().mean().item()


# ── composition under test ───────────────────────────────────────────────────


class Layer:
    """One MoE layer's state: what a target's module holds, built from the entries."""

    def __init__(self, assignments: List[int], shared, combine_format: str, ep_group):
        self.assignments = list(assignments)
        self.local = len(assignments) // WORLD
        self.combine_format = combine_format
        self.shared = shared
        stacks = build_stacks(self.assignments, RANK, self.local)
        self.stacks = stacks
        self.table = M["slots"].megamoe_slot_table(self.assignments, EXPERTS, RANK, DEV)
        self.shared_bytes, _ = M["ws"].megamoe_shared_workspace_bytes(
            world_size=WORLD,
            num_topk=TOP_K,
            num_experts_per_rank=self.local,
            hidden_size=HIDDEN,
            intermediate_size=INTER,
            max_tokens_per_rank=MAX_T,
            gate_up_clamp=CLAMP,
            combine_format=combine_format,
        )
        self.regions = M["ws"].megamoe_symm_regions(
            process_group=ep_group,
            world_size=WORLD,
            rank=RANK,
            hidden_size=HIDDEN,
            max_tokens_per_rank=MAX_T,
            num_topk=TOP_K,
            combine_format=combine_format,
            shared_workspace_bytes=self.shared_bytes,
        )
        self.idx = M["ws"].megamoe_topk_idx_buffer(MAX_T, TOP_K, DEV)
        if shared is not None:
            self.x8, self.x8_sf = M["shared"].megamoe_shared_activation_buffers(MAX_T, HIDDEN, DEV)

    def set_placement(self, assignments: List[int]) -> None:
        """Move experts in place: the weights, scalars and the routing table keep their addresses."""
        self.assignments = list(assignments)
        new = build_stacks(self.assignments, RANK, self.local)
        for key in ("fc1", "fc1_sf", "fc2", "fc2_sf"):
            self.stacks[key].copy_(new[key])
        for key, val in new["scalars"].items():
            self.stacks["scalars"][key].copy_(val)
        self.table.copy_(M["slots"].megamoe_slot_table(self.assignments, EXPERTS, RANK, DEV))
        # Every rank's weights and table must have moved before any rank launches: a rank on the
        # new table sends rows to slots whose weights a slower rank has not replaced yet.
        import torch.distributed as dist

        torch.cuda.synchronize()
        dist.barrier()

    def forward(self, x, ids, w, rows: List[int]) -> torch.Tensor:
        """x [n, H] bf16, router ids [n, K] int32 and weights [n, K] f32 -> [n, H] bf16."""
        n = x.shape[0]
        bucket = M["ws"].megamoe_pick_bucket([MAX_T], max(rows))
        assert bucket == MAX_T
        slots = megamoe_slots(self.table, ids, EXPERTS)
        if n > 0:
            x_fp4, x_sf = M["quant"].fp4_quantize(
                x, self.stacks["scalars"]["input_global_scale"], 16, False, False
            )
            x_sf = x_sf.view(n, SF_BYTES)
            if self.shared is not None:
                q8, e8 = mxfp8_rows(x)
                self.x8[:n].copy_(q8)
                atoms = mxfp8_scale_atoms(e8)
                self.x8_sf[: atoms.numel()].copy_(atoms)
        else:
            x_fp4 = torch.empty((0, HIDDEN // 2), dtype=torch.uint8, device=DEV)
            x_sf = torch.empty((0, SF_BYTES), dtype=torch.uint8, device=DEV)
        r = self.regions
        M["stage"].megamoe_stage_inputs(
            slots, w, self.idx, r.topk_weights, x_fp4, x_sf, r.activation, r.activation_sf
        )
        kwargs = {}
        if self.shared is not None:
            s = self.shared
            kwargs = dict(
                shared_fc1_weight=s["fc1_weight"],
                shared_fc1_weight_sf=s["fc1_weight_sf"],
                shared_fc2_weight=s["fc2_weight"],
                shared_fc2_weight_sf=s["fc2_weight_sf"],
                shared_activation=self.x8,
                shared_activation_sf=self.x8_sf,
                shared_intermediate_size=s["intermediate_size"],
                shared_gate_up_clamp=s["gate_up_clamp"],
            )
        sc = self.stacks["scalars"]
        op = M["op"].cute_dsl_megamoe_nvfp4
        args = (
            r.activation,
            r.activation_sf,
            self.idx,
            r.topk_weights,
            self.stacks["fc1"],
            self.stacks["fc1_sf"],
            self.stacks["fc2"],
            self.stacks["fc2_sf"],
            sc["fc1_alpha"],
            sc["fc2_alpha"],
            sc["fc1_norm_const"],
            r.shared_workspace,
            r.peer_offsets,
        )
        options = dict(
            world_size=WORLD,
            local_rank=RANK,
            max_tokens_per_rank=MAX_T,
            num_tokens=n,
            gate_up_clamp=CLAMP,
            combine_format=self.combine_format,
            **kwargs,
        )
        op.is_valid(
            *args, **options
        )  # the guard `validating` interposes in the single-process tests
        DRIVEN.add(
            (WORLD, bucket, tuple(rows), self.local, self.shared is not None, self.combine_format)
        )
        return op(*args, **options)


def make_inputs(seed: int, rows: List[int]):
    """This rank's x, router ids and weights; also every rank's, as the reference needs no peers."""
    gen = torch.Generator(device=DEV).manual_seed(seed * 31 + RANK)
    n = rows[RANK]
    x = (torch.randn(n, HIDDEN, device=DEV, generator=gen) * 1.0).to(torch.bfloat16)
    logits = torch.randn(n, EXPERTS, device=DEV, generator=gen)
    # a few hot experts, so a replicated expert's rows are spread over its copies
    logits[:, [0, 1, 2, 3]] += 2.0
    w, ids = torch.topk(logits, TOP_K, dim=-1)
    w = torch.softmax(w, dim=-1) * ROUTED_SCALE
    return x, ids.to(torch.int32).contiguous(), w.float().contiguous()


def reference(x, ids, w, layer: Layer, with_shared, real):
    n = x.shape[0]
    if n == 0:
        return torch.zeros((0, HIDDEN), dtype=torch.float32, device=DEV)
    x_fp4, x_sf = M["quant"].fp4_quantize(
        x, layer.stacks["scalars"]["input_global_scale"], 16, False, False
    )
    g1 = 1.0 / INPUT_SCALE_FC1
    xhat = unpack_activation(x_fp4, x_sf.view(n, SF_BYTES), g1)
    out = reference_routed(xhat, ids.long(), w)
    if with_shared:
        out = out + reference_shared(x.float(), real)
    return out


# ── checks ───────────────────────────────────────────────────────────────────

ASSIGN_ONE = list(range(EXPERTS))  # one copy, contiguous
SLOTS_REPL = EXPERTS + 8  # two extra slots per rank at 4 ranks
CHECKS = []


def check(fn):
    CHECKS.append(fn)
    return fn


def _all_ok(ok: bool, what: str) -> None:
    import torch.distributed as dist

    flag = torch.tensor([0 if ok else 1], device=DEV)
    dist.all_reduce(flag)
    assert flag.item() == 0, f"{what}: failed on some rank"


def _compare(out, ref, label, min_within=PERCENT, max_rms=0.06):
    if tuple(out.shape) != tuple(
        ref.shape
    ):  # an empty result passes only where the reference is empty too
        print(
            f"[rank {RANK}] {label}: shape {tuple(out.shape)} != reference {tuple(ref.shape)}",
            flush=True,
        )
        return False
    if out.shape[0] == 0:
        return True
    pct, rms = within(out, ref), rel_rms(out, ref)
    print(
        f"[rank {RANK}] {label}: rows {out.shape[0]}  within {pct:.4f}  rel-RMS {rms:.4f}",
        flush=True,
    )
    return pct >= min_within and rms <= max_rms


STATE: Dict[str, object] = {}


@check
def check_placements_and_ladder():
    """Pure host logic the targets rely on, at the real expert counts."""
    for n_exp, n_slots, ep in (
        (384, 416, 32),
        (384, 384, 8),
        (384, 384, 32),
        (EXPERTS, SLOTS_REPL, WORLD),
    ):
        a = M["slots"].megamoe_default_placement(n_exp, n_slots, ep)
        M["slots"].megamoe_check_placement(a, n_exp, ep)
        assert len(a) == n_slots
    a = M["slots"].megamoe_default_placement(384, 416, 32)
    assert a[:13] == list(range(13)) and a[13:26] == list(range(12, 25)), (
        "rank r holds 12r .. 12r+12"
    )
    assert a[-13:] == list(range(372, 384)) + [0], "the last rank wraps to expert 0"
    assert M["ws"].megamoe_token_buckets(150) == [150]
    assert M["ws"].megamoe_token_buckets(1280) == [256, 1024, 1280]
    assert M["ws"].megamoe_token_buckets(0) == [256, 1024, 4096]
    assert M["ws"].megamoe_pick_bucket([256, 1024, 1280], 257) == 1024
    # rows of a replicated expert spread over its copies (expert 0 is held by ranks 3 and 0)
    a = M["slots"].megamoe_default_placement(EXPERTS, SLOTS_REPL, WORLD)
    table = M["slots"].megamoe_slot_table(a, EXPERTS, RANK, DEV)
    picked = megamoe_slots(table, torch.zeros((64, 1), dtype=torch.int32, device=DEV), EXPERTS)
    assert len(set(picked.flatten().tolist())) == 2, "both copies of a replicated expert get rows"


@check
def check_one_copy_routed_matches_reference():
    """One copy per expert, no shared expert: every rank's rows (incl. an idle rank) vs the reference."""
    from tensorrt_llm.mapping import Mapping

    mapping = Mapping(
        world_size=WORLD,
        rank=RANK,
        gpus_per_node=WORLD,
        tp_size=WORLD,
        moe_ep_size=WORLD,
        enable_attention_dp=True,
    )
    STATE["group"] = M["ws"].megamoe_ep_process_group(mapping)
    layer = Layer(ASSIGN_ONE, None, "bf16", STATE["group"])
    x, ids, w = make_inputs(1, ROWS)
    out = layer.forward(x, ids, w, ROWS).clone()
    torch.cuda.synchronize()
    ref = reference(x, ids, w, layer, False, None)
    STATE["x"], STATE["ids"], STATE["w"], STATE["out_one"], STATE["ref"] = x, ids, w, out, ref
    _all_ok(_compare(out, ref, "one-copy routed"), "one-copy routed vs reference")
    # repeat: the kernel's self-resetting counters must give the same answer (up to bf16 add order)
    again = layer.forward(x, ids, w, ROWS).clone()
    _all_ok(_compare(again, out.float(), "one-copy repeat", 0.999, 0.02), "one-copy repeat")
    STATE["layer_one"] = layer


@check
def check_replicated_placement_matches_one_copy():
    """Extra slots: copies of experts, spread by the row hash; same answer as one copy per expert."""
    a = M["slots"].megamoe_default_placement(EXPERTS, SLOTS_REPL, WORLD)
    assert len(set(a)) == EXPERTS and len(a) == SLOTS_REPL
    layer = Layer(a, None, "bf16", STATE["group"])
    x, ids, w = STATE["x"], STATE["ids"], STATE["w"]
    slots = megamoe_slots(layer.table, ids, EXPERTS)
    assert torch.equal(torch.tensor(a, device=DEV)[slots.long()], ids.long()), (
        "a slot must hold its expert"
    )
    out = layer.forward(x, ids, w, ROWS).clone()
    torch.cuda.synchronize()
    _all_ok(_compare(out, STATE["ref"], "replicated routed"), "replicated routed vs reference")
    _all_ok(
        _compare(out, STATE["out_one"].float(), "replicated vs one-copy", 0.99, 0.02),
        "replicated vs one-copy",
    )
    STATE["layer_repl"] = layer


@check
def check_permuted_placement_and_inplace_rewrite_under_graph():
    """The deployed layer (replicated placement + fused shared expert) in a CUDA graph; then an arbitrary
    valid placement applied to it in place. One graph only: a captured workspace pairing cannot be
    re-fenced eagerly, and every variant here shares the symmetric buffers."""
    layer = STATE["layer_shared"]
    real = STATE["shared_ref"][1]
    x, ids, w = STATE["x"], STATE["ids"], STATE["w"]
    x_s, ids_s, w_s = x.clone(), ids.clone(), w.clone()
    eager = layer.forward(x_s, ids_s, w_s, ROWS).clone()  # also the warm-up the capture needs
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        layer.forward(x_s, ids_s, w_s, ROWS)
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()
    with torch.cuda.graph(graph):
        captured = layer.forward(x_s, ids_s, w_s, ROWS)
    graph.replay()
    torch.cuda.synchronize()
    _all_ok(_compare(captured, eager.float(), "graph replay vs eager", 0.99, 0.02), "graph replay")
    # new data into the same static buffers
    x2, ids2, w2 = make_inputs(2, ROWS)
    x_s.copy_(x2), ids_s.copy_(ids2), w_s.copy_(w2)
    graph.replay()
    torch.cuda.synchronize()
    ref2 = reference(x2, ids2, w2, layer, True, real)
    _all_ok(
        _compare(captured, ref2, "graph replay, new rows"), "graph replay new rows vs reference"
    )
    # move experts: a seeded shuffle of slots, kept valid (no expert twice on a rank)
    base = M["slots"].megamoe_default_placement(EXPERTS, SLOTS_REPL, WORLD)
    gen = torch.Generator().manual_seed(99)
    for _ in range(1000):
        perm = torch.randperm(len(base), generator=gen).tolist()
        moved = [base[i] for i in perm]
        try:
            M["slots"].megamoe_check_placement(moved, EXPERTS, WORLD)
            break
        except ValueError:
            continue
    else:  # pragma: no cover
        raise AssertionError("no valid shuffle found")
    assert moved != base
    layer.set_placement(moved)
    graph.replay()
    torch.cuda.synchronize()
    _all_ok(
        _compare(captured, ref2, "graph replay after placement change"),
        "graph after placement change",
    )
    # the replay staged slots of the new placement: every pick's slot holds its expert under `moved` (under the old
    # table some would not, since the shuffle moved experts between slots)
    n2 = x2.shape[0]
    staged = layer.idx[:n2].long()
    holds = torch.tensor(moved, dtype=torch.int64, device=DEV)[staged.clamp(min=0)]
    _all_ok(
        bool((staged >= 0).all()) and torch.equal(holds, ids2.long()),
        "replay stages the new placement's slots",
    )
    after = layer.forward(x2, ids2, w2, ROWS).clone()
    _all_ok(
        _compare(after, captured.float(), "eager vs replay after change", 0.99, 0.02),
        "eager after change",
    )
    layer.set_placement(base)
    del graph


@check
def check_fused_shared_expert():
    """The FP8 shared expert fused into the kernel: routed + shared vs reference, and vs routed only."""
    shared, real = make_shared()
    a = M["slots"].megamoe_default_placement(EXPERTS, SLOTS_REPL, WORLD)
    layer = Layer(a, shared, "bf16", STATE["group"])
    x, ids, w = STATE["x"], STATE["ids"], STATE["w"]
    out = layer.forward(x, ids, w, ROWS).clone()
    torch.cuda.synchronize()
    ref = reference(x, ids, w, layer, True, real)
    only = reference(x, ids, w, layer, False, None)
    shared_part = reference_shared(x.float(), real) if x.shape[0] else only
    if x.shape[0]:
        print(
            f"[rank {RANK}] shared share of output RMS: "
            f"{shared_part.pow(2).mean().sqrt().item() / ref.pow(2).mean().sqrt().item():.3f}",
            flush=True,
        )
    ok = _compare(out, ref, "fused shared + routed")
    if x.shape[0]:
        # the shared expert must really be in there: routed-only reference is much further away
        far = rel_rms(out, only)
        print(f"[rank {RANK}] fused vs routed-only reference rel-RMS {far:.4f}", flush=True)
        ok = ok and far > 3 * rel_rms(out, ref)
    _all_ok(ok, "fused shared expert")
    # a rank that is idle still takes part (rows 0 on rank 2)
    STATE["layer_shared"] = layer
    STATE["shared_ref"] = (shared, real)


@check
def check_low_precision_combine_formats():
    """The kernel's wire formats for the cross-rank combine. Not usable with the fused shared expert."""
    x, ids, w = STATE["x"], STATE["ids"], STATE["w"]
    a = M["slots"].megamoe_default_placement(EXPERTS, SLOTS_REPL, WORLD)
    for fmt, bound in (("32e4m3xe8m0", 0.055), ("16e2m1xbf16", 0.19)):
        layer = Layer(a, None, fmt, STATE["group"])
        out = layer.forward(x, ids, w, ROWS).clone()
        torch.cuda.synchronize()
        pct, rms = (
            (within(out, STATE["ref"]), rel_rms(out, STATE["ref"])) if x.shape[0] else (1.0, 0.0)
        )
        print(f"[rank {RANK}] combine {fmt}: within {pct:.4f} rel-RMS {rms:.4f}", flush=True)
        _all_ok(rms <= bound, f"combine {fmt}")
    # with the fused shared expert the wrapper refuses a low-precision combine
    refused = False
    try:
        shared, _ = STATE["shared_ref"]
        Layer(a, shared, "32e4m3xe8m0", STATE["group"]).forward(x, ids, w, ROWS)
    except (
        AssertionError
    ) as error:  # the entry's own diagnostic, not any assertion on the way to it
        refused = "the fused shared expert needs the bf16 combine" in str(error)
    _all_ok(refused, "fused shared + low-precision combine must be refused")


@check
def check_bench_layer_latency():
    """Per-layer latency of this matrix's composition (route-to-slots and MXFP8 quantize in plain torch + NVFP4
    quantize + stage + kernel with the fused shared expert, replicated placement), CUDA-graph replayed, EP4.
    Printed only; runs when MEGAMOE_BENCH=1. Rows per rank use one bucket (256)."""
    if os.environ.get("MEGAMOE_BENCH") != "1":
        return
    import torch.distributed as dist

    layer = STATE["layer_shared"]
    for n in (25, 150, 250):
        rows = [n] * WORLD
        x, ids, w = make_inputs(5, rows)
        layer.forward(x, ids, w, rows)  # eager warm-up
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            layer.forward(x, ids, w, rows)
        torch.cuda.current_stream().wait_stream(side)
        torch.cuda.synchronize()
        with torch.cuda.graph(graph):
            layer.forward(x, ids, w, rows)
        for _ in range(20):
            graph.replay()
        torch.cuda.synchronize()
        dist.barrier()
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        reps = 200
        start.record()
        for _ in range(reps):
            graph.replay()
        end.record()
        torch.cuda.synchronize()
        ms = start.elapsed_time(end) / reps
        t = torch.tensor([ms], device=DEV)
        dist.all_reduce(t, op=dist.ReduceOp.MAX)
        if RANK == 0:
            print(
                f"[bench] megamoe layer (shared fused, 34 slots/rank, EP4) rows/rank {n}: "
                f"{t.item() * 1000:.1f} us per layer",
                flush=True,
            )
        del graph


# Graph captures go last: once a graph has captured a (local, shared) workspace pairing, the op refuses
# any later eager first-launch (barrier-reset fence) on that shared workspace, and every variant here
# shares the same symmetric buffers. So every eager variant is launched before the first capture.
_ORDER = [
    "check_placements_and_ladder",
    "check_one_copy_routed_matches_reference",
    "check_replicated_placement_matches_one_copy",
    "check_fused_shared_expert",
    "check_low_precision_combine_formats",
    "check_permuted_placement_and_inplace_rewrite_under_graph",
    "check_bench_layer_latency",
]
CHECKS[:] = sorted(CHECKS, key=lambda f: _ORDER.index(f.__name__))


def _run_one_rank() -> int:
    global RANK, WORLD, E2M1, M
    from mpi4py import MPI

    from tensorrt_llm._torch.modeling_v2.catalog.quantization import fp4_quantize
    from tensorrt_llm._torch.modeling_v2.models.deepseek_v41.ops.moe import (
        cute_dsl_megamoe_nvfp4,
        megamoe_expert_slots,
        megamoe_pack_experts,
        megamoe_shared_expert,
        megamoe_stage_inputs,
        megamoe_workspace,
    )

    comm = MPI.COMM_WORLD
    RANK, WORLD = comm.Get_rank(), comm.Get_size()
    assert WORLD == 4, "this matrix is written for 4 ranks"
    torch.cuda.set_device(RANK)
    torch.backends.cuda.matmul.allow_tf32 = False
    E2M1 = torch.tensor(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
        dtype=torch.float32,
        device=DEV,
    )
    M.update(
        op=cute_dsl_megamoe_nvfp4,
        slots=megamoe_expert_slots,
        pack=megamoe_pack_experts,
        shared=megamoe_shared_expert,
        stage=megamoe_stage_inputs,
        ws=megamoe_workspace,
        quant=fp4_quantize,
    )
    for check_fn in CHECKS:
        try:
            if RANK == 0:
                print(f"[matrix] {check_fn.__name__}", flush=True)
            check_fn()
            comm.Barrier()
        except BaseException:
            import traceback

            print(f"[rank {RANK}] FAILED {check_fn.__name__}", flush=True)
            traceback.print_exc()
            sys.stdout.flush()
            sys.stderr.flush()
            comm.Abort(1)
    comm.Barrier()
    declared = {
        (
            c.spec["world"],
            c.spec["bucket"],
            tuple(c.spec["rows"]),
            c.spec["slots"],
            c.spec["shared"],
            c.spec["combine"],
        )
        for c in cute_dsl_megamoe_nvfp4.CuteDslMegaMoeNvfp4.CELLS
    }
    assert declared <= DRIVEN, (
        f"cells of CuteDslMegaMoeNvfp4.CELLS the matrix never launched: {sorted(declared - DRIVEN, key=str)}"
    )
    print(f"[rank {RANK}] {len(CHECKS)} checks passed", flush=True)
    return 0


def _spawn_ranks() -> None:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    assert visible, "set CUDA_VISIBLE_DEVICES to the devices this run owns, e.g. 0,1,2,3"
    world_size = len([d for d in visible.split(",") if d.strip()])
    assert world_size == 4, (
        f"this matrix runs on 4 devices; CUDA_VISIBLE_DEVICES names {world_size}"
    )
    command = [
        "mpirun",
        "--oversubscribe",
        "-n",
        str(world_size),
        sys.executable,
        str(Path(__file__).resolve()),
        _WORKER_FLAG,
    ]
    print(f"[launcher] {' '.join(command)}", flush=True)
    process = subprocess.Popen(command, start_new_session=True)
    try:
        code = process.wait(timeout=DEADLINE_S)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()
        raise AssertionError(
            f"the {world_size}-rank run did not finish in {DEADLINE_S}s (wedged)"
        ) from None
    assert code == 0, f"the {world_size}-rank run exited {code}"


if __name__ == "__main__":
    if _WORKER_FLAG in sys.argv:
        sys.exit(_run_one_rank())
    _spawn_ranks()
    print("OK")
