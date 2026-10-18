# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test: what the direct routed pool kernel reads before ``griddepcontrol.wait``, per warp role.

``ops/comm/pool_direct_route``'s kernel (``pool_into_route``) runs behind the pool GEMM and
the router shard's publish twin with programmatic dependent launch: under ``router_fork`` the
twin is a plain launch on a side stream joined right before this kernel, and in a CUDA
graph that join into a launch with the programmatic attribute is a programmatic edge.
Its transport words are ordered by their sentinel test, not by the wait, but the kernel
must still read nothing a predecessor produces before its wait: the expert bias (a
parameter) is all its routing warps may load first, as in ``pool_gather_route``.

Every variant ``pool_direct_route`` compiles on the decode path (programmatic dependent
launch on, the gathered row stored, the default poll guard) is compiled into a directory
of its own (the CuTe DSL names a dump after the kernel class, not the variant),
disassembled with ``cuobjdump -sass`` and checked twice:

* The whole kernel with ``_sass_wait_audit.audit``: a warp leader of CTA 0 waits on
  every path before it can exit, and the only exit without a wait is the launch-uniform
  dry-launch exit on the kernel's last parameter (``PoolDirectWorkspace.kernel`` launches
  ``dry = 1`` once per variant, eagerly, outside any capture; ``pool_direct_route``
  always passes ``dry = 0``).
* Per warp role, with that parameter read as 0 (the decode path's value):
  ``_sass_walk.per_warp`` from the entry to the role's first wait, also counting
  generic ``LD`` / ``ST`` / ``ATOM`` as global accesses, on the listing without the
  ``.reuse`` operand-cache hints. Warps 0-3 (pool poll + quantization): no global access
  before their wait. Warps 4-7 (logit poll, routing, placement): exactly the 8
  expert-bias words (8 x ``LDG.E``). No store, no wider load, no exit, no trigger and no
  indirect or unresolved branch before any role's wait; the kernel does read 128-bit
  words (its polls) somewhere, so the "no wider load" rule is not vacuous.

The variants: every placement kind, every peer rank 0..7, and for the placed kinds also
the early-id publication at each trigger point. Per kind, as a check of the walk on
this code: with every wait removed from the listing, the first routing warp reaches a
128-bit load.

The publish twin (``router_shard_publish``) is the kernel in front of that join. It is
launched without the programmatic attribute and must carry no ``griddepcontrol``
instruction: then it waits for the attention hop in full (the fork's edge), and the
programmatic join into this kernel resolves only when the twin has completed. Checked on
the twin's PTX and SASS for every row count and both warp counts it is launched with.

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
    pool_direct_route as direct,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep.ops.comm import (  # noqa: E501
    pool_gather_route as route,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep.ops.moe import (  # noqa: E501
    router_shard as shard,
)

__extra_import_path__ = ["../.."]
import _sass_walk as sass_walk  # noqa: E402
from _sass_wait_audit import audit, parse  # noqa: E402

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.get_device_capability() not in ((10, 3), (10, 7))
    or not direct.available()
    or shutil.which("cuobjdump") is None,
    reason="needs an sm_103 or sm_107 GPU, the CuTe DSL and cuobjdump",
)

#: The CTA: 4 quantizing warps, then 4 routing warps.
THREADS = direct.QUANT_THREADS + direct.ROUTE_THREADS
#: The expert-bias words each routing lane loads before its wait (one per peer block).
BIAS_LOADS = ["LDG.E"] * route.WORLD
#: Generic-address memory instructions (they can read or write global memory).
GENERIC = ("LD", "ST", "ATOM")
#: The kernel's last parameter, ``dry`` (``LAUNCH_UNIFORM_EXITS`` of the schedule test).
DRY_PARAM = "c[0x0][0x420]"
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
    publish, epoch, trigger_point = (0, 0, 0)
    if trigger is not None:
        publish, epoch, trigger_point = PUBLISH_WORDS, PUBLISH_EPOCH, trigger
    static_from = STATIC_FROM if place == route.PLACE_STATIC else route.LOCAL_EXPERTS
    # Past the op's compile cache: every variant is compiled, and dumped, here; the
    # served poll guard, as PoolDirectWorkspace compiles it.
    direct._compile.__wrapped__(
        torch.cuda.current_device(),
        peer,
        place,
        static_from,
        True,
        True,
        publish,
        epoch,
        trigger_point,
        0,
        direct.DIRECT_GUARD_NS,
        False,
    )
    cubins = glob.glob(os.path.join(directory, "*.cubin"))
    assert len(cubins) == 1, f"{directory}: expected one cubin, found {cubins}"
    listing = subprocess.run(
        ["cuobjdump", "-sass", cubins[0]], check=True, capture_output=True, text=True
    ).stdout
    functions = [
        "Function :" + chunk
        for chunk in listing.split("Function :")[1:]
        if "PoolDirectRouteKernel" in chunk.split("\n", 1)[0]
    ]
    assert len(functions) == 1, f"{cubins[0]}: {len(functions)} PoolDirectRouteKernel functions"
    return re.sub(r"\.reuse\b", "", functions[0])


def _wet(sass: str) -> str:
    """The listing with the dry flag read as 0: its loads become moves of zero, its other
    reads the zero register (the decode path's launches all pass ``dry = 0``)."""
    out = []
    for line in sass.splitlines():
        if DRY_PARAM in line:
            loaded = re.search(r"\bLDCU?(?:\.\w+)*\s+(U?R\d+),\s*" + re.escape(DRY_PARAM), line)
            if loaded:
                register = loaded.group(1)
                zero = "URZ" if register.startswith("U") else "RZ"
                move = "UMOV" if register.startswith("U") else "MOV"
                line = line.replace(loaded.group(0), f"{move} {register}, {zero}")
            else:
                line = line.replace(DRY_PARAM, "RZ")
        out.append(line)
    return "\n".join(out)


def _whole_kernel_problems(sass: str):
    """The wait-before-exit audit of the whole kernel: ok, and the dry exit the only early one."""
    problems = []
    functions = parse(sass)
    assert len(functions) == 1, list(functions)
    ((instructions, labels),) = functions.values()
    result = audit(instructions, labels, THREADS)
    if not result["ok"]:
        problems.append(f"no wait on every path before an exit: {result}")
    reads = sorted({r for early in result["launch_uniform_exits"] for r in early["reads"]})
    if reads != [DRY_PARAM]:
        problems.append(f"launch-uniform exits {result['launch_uniform_exits']}, want the dry flag")
    if result["global_stores_before_wait"]:
        problems.append(f"stores before the wait: {result['global_stores_before_wait']}")
    return problems


def _role_problems(sass: str):
    """The pre-wait rule's violations in one variant's SASS, per warp role (dry = 0)."""
    problems = []
    instructions, _ = sass_walk.parse(sass)
    if not any(ins.opcode.startswith("LDG") and ".128" in ins.opcode for ins in instructions):
        problems.append("no 128-bit global load anywhere (the polls changed form)")
    for warps, found in sass_walk.per_warp(_wet(sass), THREADS):
        accesses = [ins.opcode for ins in found["accesses"]]
        role = f"warps {warps}"
        if found["exits"] or found["triggers"] or found["indirect"] or found["unresolved"]:
            problems.append(f"{role}: {sass_walk.summary(found)}")
        if max(warps) < direct.QUANT_WARPS:
            if accesses:
                problems.append(f"{role} (quantization) access before the wait: {accesses}")
        elif min(warps) >= direct.QUANT_WARPS:
            if accesses != BIAS_LOADS:
                problems.append(f"{role} (routing) before the wait: {accesses}, not {BIAS_LOADS}")
        else:
            problems.append(f"{role}: quantization and routing warps share one walk")
    return problems


@pytest.mark.parametrize("kind", KINDS, ids=_kind_id)
def test_the_direct_routed_pool_kernel_reads_nothing_produced_before_its_wait(
    kind, dumping_compile, tmp_path
):
    name, trigger = kind
    found = {}
    for peer in range(route.WORLD):
        directory = str(tmp_path / f"{_kind_id(kind)}-peer{peer}")
        sass = _compile_variant(dumping_compile, directory, peer, PLACES[name], trigger)
        problems = _whole_kernel_problems(sass) + _role_problems(sass)
        if problems:
            found[peer] = problems
        if peer == 0:
            # The walk's sensitivity on this very code: without its waits the first
            # routing warp reaches the 128-bit polls.
            unwaited = sass_walk.pre_wait(
                _wet(sass).replace("ACQBULK", "NOP"), direct.QUANT_THREADS
            )
            wide = [ins.opcode for ins in unwaited["accesses"] if ".128" in ins.opcode]
            assert wide, sass_walk.summary(unwaited)
    assert not found, f"{_kind_id(kind)}: pre-wait violations by peer rank: {found}"


#: SASS of ``griddepcontrol.wait`` / ``griddepcontrol.launch_dependents``.
PDL_SASS = ("ACQBULK", "PREEXIT")


@pytest.mark.parametrize("warps", (4, 16))
@pytest.mark.parametrize("rows", range(1, shard.MAX_ROWS + 1))
@torch.inference_mode()
def test_the_publish_twin_is_a_plain_launch_without_pdl_instructions(
    rows, warps, monkeypatch, tmp_path
):
    launches = []
    run = shard._router_shard_publish_kernel.run

    def recording_run(*args, **kwargs):
        kernel = run(*args, **kwargs)
        launches.append((kwargs.get("launch_pdl"), kernel))
        return kernel

    monkeypatch.setattr(shard._router_shard_publish_kernel, "run", recording_run)
    hidden = torch.randn(rows, shard.HIDDEN, device="cuda").bfloat16()
    weight = torch.randn(shard.EXPERTS, shard.HIDDEN, device="cuda") * 0.02
    boxes = torch.empty(shard.publish_copies(rows), shard.PUBLISH_BOX, device="cuda")
    shard.router_shard_publish(hidden, weight, 5, 0.0110485, warps, boxes)
    torch.cuda.synchronize()
    assert len(launches) == 1, f"{len(launches)} launches of the twin"
    ((launch_pdl, kernel),) = launches
    assert launch_pdl is False, f"the twin is launched with launch_pdl={launch_pdl!r}"
    assert "griddepcontrol" not in kernel.asm["ptx"], "griddepcontrol in the twin's PTX"
    cubin = tmp_path / "twin.cubin"
    cubin.write_bytes(kernel.asm["cubin"])
    sass = subprocess.run(
        ["cuobjdump", "-sass", str(cubin)], check=True, capture_output=True, text=True
    ).stdout
    assert "_router_shard_publish_kernel" in sass, "the twin's function is not in its cubin"
    found = sorted({op for op in PDL_SASS for line in sass.splitlines() if op in line})
    assert not found, f"PDL instructions in the twin's SASS: {found}"
