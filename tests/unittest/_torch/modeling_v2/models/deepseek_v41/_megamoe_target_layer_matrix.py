# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU matrix of a generation target's MegaMoE layer (`megamoe._MegaMoe` + `weights.load_megamoe_*`) on four GPUs (EP4).

Like `_cute_dsl_megamoe_nvfp4_op_matrix.py` this is its own launcher: the kernel exchanges tokens over NVLink, so it
cannot run in one process. `TARGET=dep python _megamoe_target_layer_matrix.py` re-executes itself under
`mpirun`, one rank per device of CUDA_VISIBLE_DEVICES, under a deadline.

What runs is the TARGET's code, end to end for one layer: `resolve`-style config, `MegaMoeBuild` (every bucket's
symmetric regions), the checkpoint loaders on a synthetic checkpoint in the checkpoint's own names (128 experts at
H=5120, I=2304, a pure function of the expert id), `prime`, then `_MegaMoe.forward` on rows per rank `[64, 17, 0, 1]` (a
busy rank, a short one, an idle one, a single row). Checked against a plain-torch fp32 reference that dequantizes the
checkpoint tensors (clamped SwiGLU in real units, routing weight before the FC2 requantization, MXFP8 shared expert),
with the engine MegaMoE tolerance (`|d| <= 0.15 + 0.1 |ref|` for >= 92% of elements): routed-only (relative RMS 0.0023
at the op), with the fused FP8 shared expert (0.017), and with copies against one copy (identical bytes: same answer).
"""

import importlib
import os
import signal
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace as NS

import torch

assert torch.cuda.is_available(), "the MegaMoE kernel requires CUDA devices"

_WORKER_FLAG = "--rank-worker"
DEADLINE_S = 2000
_FAMILY = "tensorrt_llm._torch.modeling_v2.models.deepseek_v41"
HIDDEN, INTER, EXPERTS, TOP_K, CLAMP = 5120, 2304, 128, 6, 5.0
ROWS = [64, 17, 0, 1]
PREFIX = "layers.9.ffn."
RTOL, ATOL, PERCENT = 0.1, 0.15, 0.92
DEV = "cuda"
RANK = WORLD = 0
E2M1 = None


def _say(msg):
    if RANK == 0:
        print(f"[layer-matrix] {msg}", flush=True)


class _Checkpoint(dict):
    """A checkpoint whose expert `e` is a pure function of `e` (every
    rank regenerates any expert), in the checkpoint's names."""

    def __missing__(self, key):
        parts = key[len(PREFIX) :].split(".")
        if parts[0] == "shared_experts":
            self._shared()
            return self[key]
        e, name, what = int(parts[1]), parts[2], parts[3]
        if what in ("input_scale", "weight_scale_2"):
            u = torch.rand(8, generator=torch.Generator().manual_seed(9000 + e)).tolist()
            ws1 = 0.05 * (0.8 + 0.4 * u[0])
            vals = {
                ("w1", "weight_scale_2"): ws1,
                ("w3", "weight_scale_2"): ws1 * (2.0 if e % 5 == 0 else 1.0),
                ("w2", "weight_scale_2"): 0.01 * (0.8 + 0.4 * u[1]),
                ("w1", "input_scale"): 0.0017 * (0.5 + 0.5 * u[2]),
                ("w3", "input_scale"): 0.0017 * (0.5 + 0.5 * u[3]),
                ("w2", "input_scale"): 0.04 * (0.8 + 0.4 * u[4]),
            }
            self[key] = torch.tensor(vals[(name, what)], dtype=torch.float32)
            return self[key]
        gen = torch.Generator(device=DEV).manual_seed(
            7000 + e * 3 + {"w1": 0, "w3": 1, "w2": 2}[name]
        )
        n, k = (HIDDEN, INTER) if name == "w2" else (INTER, HIDDEN)
        codes = torch.randint(0, 16, (n, k), dtype=torch.uint8, device=DEV, generator=gen)
        sf = (
            torch.randint(-3, 1, (n, k // 16), device=DEV, generator=gen, dtype=torch.int32)
            .float()
            .exp2()
        )
        base = f"{PREFIX}experts.{e}.{name}."
        if len(self) > 120:  # a rank's experts at most
            for stale in [
                k_ for k_ in self if "experts" in k_ and not k_.endswith(("scale_2", "input_scale"))
            ][:6]:
                del self[stale]
        self[base + "weight"] = (codes[:, 0::2] | (codes[:, 1::2] << 4)).contiguous()
        self[base + "weight_scale"] = sf.to(torch.float8_e4m3fn)
        self[base + "codes"] = codes
        return self[key]

    def _shared(self):
        gen = torch.Generator(device=DEV).manual_seed(4242)
        for name, (n, k) in (
            ("w1", (INTER, HIDDEN)),
            ("w3", (INTER, HIDDEN)),
            ("w2", (HIDDEN, INTER)),
        ):
            self[f"{PREFIX}shared_experts.{name}.weight"] = (
                torch.randn(n, k, device=DEV, generator=gen).clamp(-6, 6).to(torch.float8_e4m3fn)
            )
            self[f"{PREFIX}shared_experts.{name}.scale"] = torch.randint(
                119, 124, (n // 32, k // 32), device=DEV, generator=gen, dtype=torch.int32
            ).to(torch.uint8)

    def real(self, e):
        """(w_gate, w_up, w_down) of expert `e` in f32 real units, as the reference needs them."""
        out = []
        for name in ("w1", "w3", "w2"):
            q = f"{PREFIX}experts.{e}.{name}."
            _ = self[q + "weight"]
            codes = self[q + "codes"]
            sf = self[q + "weight_scale"].float().repeat_interleave(16, dim=1)
            out.append(E2M1[codes.long()] * sf * float(self[q + "weight_scale_2"]))
        return out


def _real_shared(ck):
    ck._shared()
    out = []
    for name in ("w1", "w3", "w2"):
        w = ck[f"{PREFIX}shared_experts.{name}.weight"].float()
        s = (
            torch.exp2(ck[f"{PREFIX}shared_experts.{name}.scale"].float() - 127.0)
            .repeat_interleave(32, 0)
            .repeat_interleave(32, 1)
        )
        out.append(w * s)
    return out


def _within(v, ref):
    return ((v.float() - ref).abs() <= ATOL + RTOL * ref.abs()).float().mean().item()


def _rms(v, ref):
    return (
        (v.float() - ref).pow(2).mean().sqrt() / ref.pow(2).mean().sqrt().clamp(min=1e-12)
    ).item()


def _all_ok(ok, what):
    import torch.distributed as dist

    flag = torch.tensor([0 if ok else 1], device=DEV)
    dist.all_reduce(flag)
    assert flag.item() == 0, f"{what}: failed on some rank"


def _build_layer(target, slots, fused, ck, mapping):
    mega = importlib.import_module(f"{_FAMILY}.flash_nvfp4__sm_103__{target}.megamoe")
    wts = importlib.import_module(f"{_FAMILY}.flash_nvfp4__sm_103__{target}.weights")
    mc = mega.MegaMoeConfig(
        ep_size=WORLD, ep_rank=RANK, num_experts=EXPERTS, num_slots=slots, top_k=TOP_K, hidden=HIDDEN, inter=INTER,
        clamp=CLAMP, combine_format="bf16", fused_shared=fused, max_tokens=256, initial_assignments=None,
    )  # fmt: skip
    cfg = NS(routed_scaling_factor=1.5, swiglu_limit=CLAMP)
    build = mega.MegaMoeBuild(mc, mapping, torch.device(DEV))
    layer = mega._MegaMoe(cfg, mc, 0)
    g = torch.Generator(device=DEV).manual_seed(123)
    layer.gate_weight = torch.nn.Parameter(
        (torch.randn(EXPERTS, HIDDEN, device=DEV, generator=g) * 0.05).bfloat16(),
        requires_grad=False,
    )
    layer.gate_bias = torch.nn.Parameter(torch.zeros(EXPERTS, device=DEV), requires_grad=False)
    layer.build = build
    wts.load_megamoe_experts(layer, ck, PREFIX + "experts.", DEV)
    if fused:
        wts.load_megamoe_shared(layer, ck, PREFIX, DEV)
    mega.prime(NS(layers=[NS(moe=layer)], mega_cfg=mc, mega_build=build))
    return layer


def _inputs():
    g = torch.Generator(device=DEV).manual_seed(77 * 31 + RANK)
    return (torch.randn(ROWS[RANK], HIDDEN, device=DEV, generator=g)).to(torch.bfloat16)


def _reference(layer, x, ck, with_shared):
    from tensorrt_llm._torch.modeling_v2.models.deepseek_v41.ops.moe import (
        cute_dsl_megamoe_nvfp4 as op_mod,
    )

    n = x.shape[0]
    if n == 0:
        return torch.zeros((0, HIDDEN), device=DEV)
    w, ids = layer._route(x)
    xq, xsf = layer_quant(x, layer)
    codes = torch.stack([xq & 0x0F, xq >> 4], dim=-1).reshape(n, HIDDEN).long()
    xhat = (
        E2M1[codes]
        * xsf.view(torch.float8_e4m3fn)
        .reshape(n, HIDDEN // 16)
        .float()
        .repeat_interleave(16, dim=1)
        / layer.input_global_scale.item()
    )
    cache = {}

    def expert(e):
        if e not in cache:
            wg, wu, wd = ck.real(e)
            cache[e] = (wg, wu, wd, 1.0 / float(ck[f"{PREFIX}experts.{e}.w2.input_scale"]))
        return cache[e]

    ref = op_mod.cute_dsl_megamoe_nvfp4.reference(xhat, ids.long(), w, expert, CLAMP)
    if with_shared:
        w1, w3, w2 = _real_shared(ck)
        xm = op_mod._mxfp8_roundtrip(x)
        ref = ref + op_mod._swiglu(xm @ w1.t(), xm @ w3.t(), CLAMP) @ w2.t()
    return ref


def layer_quant(x, layer):
    from tensorrt_llm._torch.modeling_v2.catalog.quantization.fp4_quantize import fp4_quantize

    q, sf = fp4_quantize(x, layer.input_global_scale, 16, False, False)
    return q, sf


def _run(target):
    from tensorrt_llm.mapping import Mapping

    mapping = Mapping(
        world_size=WORLD,
        rank=RANK,
        gpus_per_node=WORLD,
        tp_size=WORLD,
        moe_ep_size=WORLD,
        enable_attention_dp=True,
    )
    ck = _Checkpoint()
    x = _inputs()
    outs = {}
    for label, slots, fused in (
        ("routed, one copy", 128, False),
        ("routed + fused shared, copies", 136, True),
        ("routed, copies", 136, False),
    ):
        _say(label)
        layer = _build_layer(target, slots, fused, ck, mapping)
        out = layer(x, ROWS).clone()
        torch.cuda.synchronize()
        ref = _reference(layer, x, ck, fused)
        ok = True
        if x.shape[0]:
            pct, rms = _within(out, ref), _rms(out, ref)
            print(
                f"[rank {RANK}] {label}: rows {x.shape[0]} within {pct:.4f} rel-RMS {rms:.4f}",
                flush=True,
            )
            ok = pct >= PERCENT and rms <= (0.06 if fused else 0.02)
        else:
            assert out.shape == (0, HIDDEN)
        _all_ok(ok, f"{label} vs reference")
        again = layer(
            x, ROWS
        ).clone()  # the kernel's self-resetting counters: the same answer twice
        _all_ok(
            x.shape[0] == 0 or _rms(again, out.float()) <= 0.02, f"{label} repeat"
        )  # collective: every rank calls it
        outs[label] = out
    # copies hold identical bytes: the same function as one copy (up to the bf16 add order)
    _all_ok(
        x.shape[0] == 0 or _rms(outs["routed, copies"], outs["routed, one copy"].float()) <= 0.02,
        "copies vs one copy",
    )


def _run_one_rank() -> int:
    global RANK, WORLD, E2M1
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    RANK, WORLD = comm.Get_rank(), comm.Get_size()
    assert WORLD == 4, "this matrix is written for 4 ranks"
    torch.cuda.set_device(RANK)
    torch.backends.cuda.matmul.allow_tf32 = False
    E2M1 = torch.tensor(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
        device=DEV,
    )
    try:
        _run(os.environ["TARGET"])
    except BaseException:
        import traceback

        print(f"[rank {RANK}] FAILED", flush=True)
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        comm.Abort(1)
    comm.Barrier()
    print(f"[rank {RANK}] layer matrix passed", flush=True)
    return 0


def _spawn_ranks() -> None:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    assert visible, "set CUDA_VISIBLE_DEVICES to the devices this run owns, e.g. 0,1,2,3"
    world_size = len([d for d in visible.split(",") if d.strip()])
    assert world_size == 4 and os.environ.get("TARGET") == "dep"
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
