# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test: what the routed pool kernel reads before ``griddepcontrol.wait``, per warp role.

``ops/comm/pool_gather_route``'s kernel runs behind the pool GEMM and the
router shard with programmatic dependent launch. Its routing warps load the 8
expert-bias words (a parameter) first, then wait, then read the router's logits
``logits_local``. Under ``router_fork`` the router is a plain launch on a side
stream and the current stream joins it right before this kernel; in a CUDA graph
that join into a launch with the programmatic attribute is a programmatic edge, so
the kernel's wait is the only thing that orders its logit reads after the router's
stores. A read moved before the wait would not change a bit in any run: the plain
router has no trigger, so the kernel only starts once the router's CTAs exited.
Hence this check of the machine code rather than of values.

Every variant ``pool_gather_route`` compiles on the decode path (programmatic
dependent launch on, the gathered row stored) is compiled into a directory of its
own -- the CuTe DSL names a dump after the kernel class, not the variant, so a
shared directory keeps only the last compile -- disassembled with
``cuobjdump -sass`` and walked per warp role, from the entry to that role's first
wait (``_sass_walk.per_warp``, here also counting generic
``LD`` / ``ST`` / ``ATOM`` as global accesses, on the listing without the
``.reuse`` operand-cache hints, which the walk would otherwise not read as
registers and so could not tell the warp roles apart). The variants: every placement
kind, every peer rank 0..7, and for the placed kinds also the early-id
publication at each trigger point (exit, after the logit exchange, after the
wait). Pass, for every variant:

* warps 0-3 (pool gather + quantization): no global access before their wait;
* warps 4-7 (logit exchange, routing, placement): exactly the 8 expert-bias words
  (8 x ``LDG.E``, 32-bit loads) before their wait;
* no store, no wider load, no exit, no trigger and no indirect or unresolved
  branch before any role's wait;
* the kernel does read 128-bit words (the logits and the inbox) somewhere, so the
  "no wider load before the wait" rule above is not vacuous.

And per kind, as a check of the walk itself on this code: with every wait removed
from the listing, the first routing warp does reach a 128-bit load.

A change that moves any access across a wait, or that adds a pre-wait load,
fails here and needs its own argument (and this test's update). A new kernel
that reads ``logits_local`` behind the router-fork join belongs in this file too.

Needs one sm_103 or sm_107 GPU, the CuTe DSL and ``cuobjdump``; skips otherwise.
"""

import glob
import os
import re
import shutil
import subprocess

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep.kernels.replicated_experts.ring_buddy import (  # noqa: E501
    STATIC_FROM,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep.ops.comm import (  # noqa: E501
    pool_gather_route as route,
)

__extra_import_path__ = ["../.."]
import _sass_walk as sass_walk  # noqa: E402

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.get_device_capability() not in ((10, 3), (10, 7))
    or not route.available()
    or shutil.which("cuobjdump") is None,
    reason="needs an sm_103 or sm_107 GPU, the CuTe DSL and cuobjdump",
)

#: The CTA: 4 quantizing warps, then 4 routing warps (``QUANT_WARPS``, ``ROUTE_THREADS``).
THREADS = route.QUANT_THREADS + route.ROUTE_THREADS
#: The expert-bias words each routing lane loads before its wait (one per peer block).
BIAS_LOADS = ["LDG.E"] * route.WORLD
#: Generic-address memory instructions (they can read or write global memory).
GENERIC = ("LD", "ST", "ATOM")
#: Placeholder compile-time addresses of the early-id publication (compiled, never launched).
PUBLISH_WORDS = 0x7F0000001000
PUBLISH_EPOCH = 0x7F0000002000
#: The early-id trigger points (``early_expert_ids.TRIGGER_EXIT`` / ``_EXCHANGE`` / ``_ENTRY``).
EARLY_IDS_TRIGGERS = (0, 1, 2)
PLACES = {
    "none": route.PLACE_NONE,
    "owner": route.PLACE_OWNER,
    "balance": route.PLACE_BALANCE,
    "static": route.PLACE_STATIC,
    "balance3": route.PLACE_BALANCE3,
}
#: (placement name, early-id trigger or None for no publication).
KINDS = [(name, None) for name in PLACES] + [
    (name, trigger)
    for name, place in PLACES.items()
    if place in route.EARLY_PLACES
    for trigger in EARLY_IDS_TRIGGERS
]


def _kind_id(kind) -> str:
    name, trigger = kind
    return name if trigger is None else f"{name}-early{trigger}"


@pytest.fixture
def dumping_compile(monkeypatch):
    """``cute.compile`` writing its cubin to ``target["dir"]``."""
    cute = route.cute
    original = cute.compile
    target = {"dir": None}

    def compile_with_dump(*args, **kwargs):
        options = kwargs.pop("options", "")
        kwargs["options"] = f"{options} --keep-cubin --dump-dir={target['dir']}".strip()
        return original(*args, **kwargs)

    monkeypatch.setattr(cute, "compile", compile_with_dump)
    monkeypatch.setattr(sass_walk, "GLOBAL", tuple(sass_walk.GLOBAL) + GENERIC)
    return target


def _compile_variant(target, directory, peer, place, trigger) -> str:
    """Compile one variant into ``directory`` (new, empty); return its kernel's SASS."""
    os.makedirs(directory)
    target["dir"] = directory
    extra = (
        {}
        if trigger is None
        else dict(publish=PUBLISH_WORDS, epoch_word=PUBLISH_EPOCH, trigger=trigger, stamps=0)
    )
    static_from = STATIC_FROM if place == route.PLACE_STATIC else route.LOCAL_EXPERTS
    # Past the op's compile cache: every variant is compiled, and dumped, here.
    route._compile.__wrapped__(
        torch.cuda.current_device(), peer, place, static_from, True, True, **extra
    )
    cubins = glob.glob(os.path.join(directory, "*.cubin"))
    assert len(cubins) == 1, f"{directory}: expected one cubin, found {cubins}"
    listing = subprocess.run(
        ["cuobjdump", "-sass", cubins[0]], check=True, capture_output=True, text=True
    ).stdout
    functions = [
        "Function :" + chunk
        for chunk in listing.split("Function :")[1:]
        if "PoolGatherRouteKernel" in chunk.split("\n", 1)[0]
    ]
    assert len(functions) == 1, f"{cubins[0]}: {len(functions)} PoolGatherRouteKernel functions"
    return re.sub(r"\.reuse\b", "", functions[0])


def _problems(sass: str):
    """The pre-wait rule's violations in one variant's SASS, per warp role."""
    problems = []
    instructions, _ = sass_walk.parse(sass)
    if not any(ins.opcode.startswith("LDG") and ".128" in ins.opcode for ins in instructions):
        problems.append("no 128-bit global load anywhere (the logit read changed form)")
    for warps, found in sass_walk.per_warp(sass, THREADS):
        accesses = [ins.opcode for ins in found["accesses"]]
        role = f"warps {warps}"
        if found["exits"] or found["triggers"] or found["indirect"] or found["unresolved"]:
            problems.append(f"{role}: {sass_walk.summary(found)}")
        if max(warps) < route.QUANT_WARPS:
            if accesses:
                problems.append(f"{role} (quantization) access before the wait: {accesses}")
        elif min(warps) >= route.QUANT_WARPS:
            if accesses != BIAS_LOADS:
                problems.append(f"{role} (routing) before the wait: {accesses}, not {BIAS_LOADS}")
        else:
            problems.append(f"{role}: quantization and routing warps share one walk")
    return problems


@pytest.mark.parametrize("kind", KINDS, ids=_kind_id)
def test_the_routed_pool_kernel_reads_the_logits_only_after_its_wait(
    kind, dumping_compile, tmp_path
):
    name, trigger = kind
    found = {}
    for peer in range(route.WORLD):
        directory = str(tmp_path / f"{_kind_id(kind)}-peer{peer}")
        sass = _compile_variant(dumping_compile, directory, peer, PLACES[name], trigger)
        problems = _problems(sass)
        if problems:
            found[peer] = problems
        if peer == 0:
            # The walk's sensitivity on this very code: without its waits the first
            # routing warp reaches the 128-bit logit reads.
            unwaited = sass_walk.pre_wait(sass.replace("ACQBULK", "NOP"), route.QUANT_THREADS)
            wide = [ins.opcode for ins in unwaited["accesses"] if ".128" in ins.opcode]
            assert wide, sass_walk.summary(unwaited)
    assert not found, f"{_kind_id(kind)}: pre-wait violations by peer rank: {found}"
