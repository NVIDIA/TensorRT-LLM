# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Placement invariance of a generation target's MegaMoE layer (`megamoe._MegaMoe` + `weights.load_megamoe_*`) on four
GPUs (EP4), derived from `_megamoe_target_layer_matrix.py`.

Like `_cute_dsl_megamoe_nvfp4_op_matrix.py` this is its own launcher: the kernel exchanges tokens over NVLink, so it
cannot run in one process. `TARGET=dep python _megamoe_placement_invariance.py` re-executes itself under
`mpirun`, one rank per device of CUDA_VISIBLE_DEVICES, under a deadline.

What runs is the TARGET's code for one layer: `MegaMoeBuild`, the checkpoint loaders on a synthetic checkpoint in the
checkpoint's own names (128 experts at H=5120, I=2304, a pure function of the expert id), `prime`, then
`_MegaMoe.forward` on rows per rank `[64, 17, 0, 1]`, once per placement of the same experts (the default, a rotation
that moves every expert to another rank and, with one copy, a shuffle) for 128 slots routed-only, 128 slots with the
fused shared expert and 136 slots (copies). The check is bitwise: every placement's output equals the default
placement's (`torch.equal`), as stock's MegaMoE answers the same at any placement. Numerical agreement with a reference
is `_megamoe_target_layer_matrix.py`'s.
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
ROWS = [int(r) for r in os.environ.get("PLACE_ROWS", "64,17,0,1").replace("+", ",").split(",")]
PREFIX = "layers.9.ffn."
DEV = "cuda"
RANK = WORLD = 0
E2M1 = None


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


def _rms(v, ref):
    return (
        (v.float() - ref).pow(2).mean().sqrt() / ref.pow(2).mean().sqrt().clamp(min=1e-12)
    ).item()


def _all_ok(ok, what):
    import torch.distributed as dist

    flag = torch.tensor([0 if ok else 1], device=DEV)
    dist.all_reduce(flag)
    assert flag.item() == 0, f"{what}: failed on some rank"


def _build_layer(target, slots, fused, ck, mapping, assign=None):
    mega = importlib.import_module(f"{_FAMILY}.flash_nvfp4__sm_103__{target}.megamoe")
    wts = importlib.import_module(f"{_FAMILY}.flash_nvfp4__sm_103__{target}.weights")
    mc = mega.MegaMoeConfig(
        ep_size=WORLD, ep_rank=RANK, num_experts=EXPERTS, num_slots=slots, top_k=TOP_K, hidden=HIDDEN, inter=INTER,
        clamp=CLAMP, combine_format="bf16", fused_shared=fused,
        max_tokens=int(os.environ.get("PLACE_MAX_TOKENS", "256")),
        initial_assignments=None if assign is None else {0: list(assign)},
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


def _placements(slots):
    """Slot -> expert lists: the default, a rotation that moves every expert to another rank, and a shuffle."""
    from tensorrt_llm._torch.modeling_v2.models.deepseek_v41.ops.moe.megamoe_expert_slots import (
        megamoe_default_placement,
    )

    base = megamoe_default_placement(EXPERTS, slots, WORLD)
    per = slots // WORLD
    rotated = base[per:] + base[:per]
    out = {"default": base, "rotated by a rank": rotated}
    if slots == EXPERTS:  # one copy: any permutation is a placement
        g = torch.Generator().manual_seed(5)
        out["shuffled"] = [base[i] for i in torch.randperm(slots, generator=g).tolist()]
    return out


def _run(target):
    """The layer's output for this rank's rows under several placements of the same experts: the same function, so the
    same bits (stock's MegaMoE answers the same at any placement)."""
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
    results = []
    for slots, fused in ((128, False), (128, True), (136, False)):
        outs = {}
        for name, assign in _placements(slots).items():
            layer = _build_layer(target, slots, fused, ck, mapping, assign)
            outs[name] = layer(x, ROWS).clone()
            torch.cuda.synchronize()
            del layer
        base = outs["default"]
        for name, out in outs.items():
            if x.shape[0]:
                diff = (out.float() - base.float()).abs()
                bad_rows = int((diff.amax(dim=1) > 0).sum())
                print(
                    f"[rank {RANK}] slots {slots} shared {fused} {name}: rows differing {bad_rows}/{x.shape[0]} "
                    f"max |d| {diff.max().item():.3e} rel-RMS {_rms(out, base.float()):.2e}",
                    flush=True,
                )
        ok = x.shape[0] == 0 or all(torch.equal(o, base) for o in outs.values())
        print(f"[rank {RANK}] slots {slots} shared {fused}: placement invariant {ok}", flush=True)
        results.append(ok)
    _all_ok(all(results), "placement invariance")


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
