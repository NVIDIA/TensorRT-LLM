# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""dep16's MegaMoE token path, ``kernels/moe/megamoe_token_path.py`` (CPU, from source).

The target's subclasses of the engine's generation-phase MegaMoE kernel and its token exchange carry
the three token-path levers (``topk_sentinel``, ``sort_local_prefix``, ``refine_sf_first``), and
``bind_token_path_levers`` makes the kernel subclass the one the engine's op builds. With stand-ins
for what needs the GPU stack:

1. every engine method the subclasses override, and the device-workspace build the kernel subclass
   runs again, is pinned to its source at the MR base (sha256), so an engine change fails here and
   names the method: the copy must follow it; the subclasses override nothing else, and what they
   add the engine does not have;
2. the kernel subclass ships the levers off, hands them to its exchange and tags any that is on in
   ``name()``; the exchange reads them off its descriptor and refuses the sentinel inbox with the
   FC1 score plane; past the device-workspace build the subclass only records its combine plane
   and, under ``defer_topk_reduce`` (off as shipped, tagged in ``name()``), drops the TopkReduce;
3. the binding sets the constants and puts the subclass in the package's place only when a lever is
   on, and refuses a process that already runs another setting;
4. the engine's op builds the class the package names when it builds: with the levers on the
   subclass, off the engine class (``_construct_megamoe_kernel`` run from its source over stand-ins);
   it imports the name inside the function, so a binding at import would fail here;
5. the sort CTAs' local segment bases are the helper's: the helper's prefix and ring order, worked
   in Python over random count rows and slot orders, against the sort CTAs' per-source formula.
"""

from __future__ import annotations
import __future__

import ast
import hashlib
import importlib.util
import random
import sys
from pathlib import Path
from types import MappingProxyType, ModuleType, SimpleNamespace

import pytest

pytestmark = pytest.mark.cpu_only

_PKG_ROOT = Path(importlib.util.find_spec("tensorrt_llm").submodule_search_locations[0]).resolve()
_FAMILY = _PKG_ROOT / "_torch/_experimental/modeling_v2/models/kimi_k31"
_TARGET = _FAMILY / "kimi_k31_pro_nvfp4__sm_107__dep16"
_OURS = _TARGET / "kernels/moe/megamoe_token_path.py"
_MEGA = _PKG_ROOT / "_torch/moe/kernels/cutedsl_megamoe/kernel_src/rubin/inference/mega"
_ENGINE = {
    "GenphaseTokenComm": _MEGA / "token_comm_gen_specialized.py",
    "BlockScaledSwapAbGenphaseMoeKernel": (
        _MEGA / "block_scaled_swap_ab_mega_moe_kernel_gen_specialized.py"
    ),
}
_OP = _PKG_ROOT / "_torch/moe/custom_ops/cute_dsl_megamoe_custom_op.py"
_PACKAGE = "tensorrt_llm._torch.moe.kernels.cutedsl_megamoe"
_SWITCHES = ("topk_sentinel", "sort_local_prefix", "refine_sf_first")
_CONSTANTS = {s: f"token_comm_{s}" for s in _SWITCHES}
# Our class -> (its engine base, the engine methods it overrides, what it adds).
_SUBCLASSES = {
    "RubinTokenPathComm": (
        "GenphaseTokenComm",
        {
            "impl_desc_require",
            "__init__",
            "_validate_configuration",
            "_register_symmetric_regions",
            "_build_smem_workspace",
            "_run_exchange",
            "_canonicalize_wire_indices",
            "_run_sort",
            "_run_refine",
        },
        {
            "wire_index_bias",
            "_stage_local_segment_bases",
            "sort_scan_input_region",
            "sort_scan_prefix_region",
        },
    ),
    "RubinTokenPathMegaMoE": (
        "BlockScaledSwapAbGenphaseMoeKernel",
        {"__init__", "name"},
        set(_CONSTANTS.values()) | {"defer_topk_reduce", "combine_plane_offsets"},
    ),
}
#: sha256 of each pinned engine method at the MR base, over its source lines from its first
#: decorator to its last line: what the subclasses override, and the device-workspace build that
#: RubinTokenPathMegaMoE.__init__ runs again after the engine constructor.
_ENGINE_PINS = {
    ("GenphaseTokenComm", "impl_desc_require"): (
        "854eb611276df4ca7a1c05b1af6e447a1a9114babc8b8e9029665daf3d79aec1"
    ),
    ("GenphaseTokenComm", "__init__"): (
        "41b5ad9ad0fc2b3e107d9aec9f66e780a93f2bb35d55f06a02fb2b7c3ecd9b48"
    ),
    ("GenphaseTokenComm", "_validate_configuration"): (
        "92e248ed1d8c3633743d145c6f1ad4075cab4fda1018b9c413c7e1e8cdd648aa"
    ),
    ("GenphaseTokenComm", "_register_symmetric_regions"): (
        "44a4a4a706d85a28744f7c662a332aaed3f7c772366687ea92ab48a817120acb"
    ),
    ("GenphaseTokenComm", "_build_smem_workspace"): (
        "68b3387c9d9620a7f5ed3876cb9fb4a329beac912dc364367ab85315429b00d4"
    ),
    ("GenphaseTokenComm", "_run_exchange"): (
        "7013a18d450adad1d94de335affdce30da54093095b5f4b639307347ffda0fb4"
    ),
    ("GenphaseTokenComm", "_canonicalize_wire_indices"): (
        "744389a29c820e7d4e9f28345264030ac0914a7625868c324417c2abef77697a"
    ),
    ("GenphaseTokenComm", "_run_sort"): (
        "b753200169263bce2fc5791107bd885e9ccf1219b1f2cef3d011f085949d2814"
    ),
    ("GenphaseTokenComm", "_run_refine"): (
        "638668ad10de98e7aa31f067e73fba75c827fbe3585fdf183b759b0db2f1cde7"
    ),
    ("BlockScaledSwapAbGenphaseMoeKernel", "__init__"): (
        "8bc09890b82ee98f663f8aede37b583ab493c011b3cace8b0ac3a95267b1efc9"
    ),
    ("BlockScaledSwapAbGenphaseMoeKernel", "_build_device_workspace"): (
        "6d629b9d47bc5d8f103587c7a985cfabddba712e31c59c90e334f773d78e0288"
    ),
    ("BlockScaledSwapAbGenphaseMoeKernel", "name"): (
        "0e5bf11ca881c62c1fe09219cf5b813c8906aa85ec1d9426b816997f127afc4d"
    ),
}


def _tree(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"))


def _class(path: Path, name: str) -> ast.ClassDef:
    (cls,) = [n for n in _tree(path).body if isinstance(n, ast.ClassDef) and n.name == name]
    return cls


def _members(cls: ast.ClassDef) -> dict:
    """The methods and annotated class attributes a class body defines, by name."""
    found = {}
    for node in cls.body:
        if isinstance(node, ast.FunctionDef):
            found[node.name] = node
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            found[node.target.id] = node
    return found


def _source_sha256(path: Path, cls_name: str, method: str) -> str:
    lines = path.read_text(encoding="utf-8").split("\n")
    node = _members(_class(path, cls_name))[method]
    start = min([d.lineno for d in node.decorator_list] + [node.lineno])
    return hashlib.sha256("\n".join(lines[start - 1 : node.end_lineno]).encode()).hexdigest()


def _definitions(path: Path, names: set, ns: dict | None = None) -> dict:
    """The module-level functions and (annotated) assignments of ``path`` named in ``names``, run
    from its source (annotations unevaluated, as in the module)."""
    wanted = set(names)
    nodes = []
    for node in _tree(path).body:
        if isinstance(node, ast.FunctionDef):
            found = {node.name}
        elif isinstance(node, ast.Assign):
            found = {t.id for t in node.targets if isinstance(t, ast.Name)}
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            found = {node.target.id}
        else:
            continue
        if found & wanted:
            nodes.append(node)
            wanted -= found
    assert not wanted, f"{path.name} no longer defines {sorted(wanted)}"
    ns = {} if ns is None else ns
    flags = __future__.annotations.compiler_flag
    module = ast.Module(body=nodes, type_ignores=[])
    exec(compile(module, str(path), "exec", flags=flags, dont_inherit=True), ns)
    return ns


# ---------------------------------------------------------------------------------------------
# 1. The engine methods the copies follow
# ---------------------------------------------------------------------------------------------


def test_the_engine_methods_are_the_ones_the_subclasses_copy():
    changed = [
        f"{cls}.{method}"
        for (cls, method), pinned in sorted(_ENGINE_PINS.items())
        if _source_sha256(_ENGINE[cls], cls, method) != pinned
    ]
    assert not changed, (
        f"the engine changed {changed}: port the change into kernels/moe/megamoe_token_path.py "
        "(or check the kernel's constructor still needs no more than its device workspace again), "
        "then update the pin"
    )


def test_the_subclasses_override_the_pinned_methods_only():
    for name, (base, overrides, adds) in _SUBCLASSES.items():
        ours = _class(_OURS, name)
        assert [ast.unparse(b) for b in ours.bases] == [base]
        engine = _members(_class(_ENGINE[base], base))
        members = _members(ours)
        assert set(members) == overrides | adds, (name, sorted(set(members) ^ (overrides | adds)))
        assert overrides <= set(engine), (name, sorted(overrides - set(engine)))
        assert not adds & set(engine), (name, sorted(adds & set(engine)))
    pinned = {(base, m) for base, overrides, _ in _SUBCLASSES.values() for m in overrides}
    assert pinned <= set(_ENGINE_PINS), sorted(pinned - set(_ENGINE_PINS))


# ---------------------------------------------------------------------------------------------
# 2. The kernel's constants and their hand-off; the exchange's refusal
# ---------------------------------------------------------------------------------------------


def test_the_kernel_ships_the_levers_off_and_hands_them_to_its_exchange():
    kernel = _class(_OURS, "RubinTokenPathMegaMoE")
    members = _members(kernel)
    assert {c: ast.literal_eval(members[c].value) for c in _CONSTANTS.values()} == {
        c: False for c in _CONSTANTS.values()
    }
    init = members["__init__"]
    (build,) = [
        n
        for n in ast.walk(init)
        if isinstance(n, ast.Call) and ast.unparse(n.func) == "RubinTokenPathComm"
    ]
    handed = {
        k.value: ast.unparse(v)
        for node in ast.walk(build)
        if isinstance(node, ast.Dict)
        for k, v in zip(node.keys, node.values)
        if isinstance(k, ast.Constant) and k.value in _SWITCHES
    }
    assert handed == {s: f"self.{c}" for s, c in _CONSTANTS.items()}
    statements = [ast.unparse(s) for s in init.body]
    assert statements[0] == "super().__init__(problem_desc, impl_desc)"
    build = statements.index("self._device_workspace = self._build_device_workspace()")
    (tail,) = statements[build + 1 :]
    assert tail.startswith("if not self.reduce_topk_in_kernel:"), tail
    assert "self._topk_reduce = _no_topk_reduce" in tail and "combine_plane_offsets[key]" in tail
    assert ast.literal_eval(members["defer_topk_reduce"].value) is False
    name = ast.unparse(members["name"])
    assert all(c in name for c in (*_CONSTANTS.values(), "defer_topk_reduce"))


def test_the_exchange_reads_the_levers_and_refuses_the_sentinel_with_the_fc1_scores():
    members = _members(_class(_OURS, "RubinTokenPathComm"))
    guards = [
        ast.unparse(node.test)
        for node in ast.walk(members["_validate_configuration"])
        if isinstance(node, ast.If) and any(isinstance(b, ast.Raise) for b in node.body)
    ]
    assert "self.topk_sentinel and self.apply_topk_at_fc1" in guards
    init = ast.unparse(members["__init__"]).replace("'", '"')
    required = ast.unparse(members["impl_desc_require"]).replace("'", '"')
    for switch in _SWITCHES:
        assert f'impl_desc.get("{switch}", False)' in init
        assert f'"{switch}": OptionalRequirement(bool)' in required


# ---------------------------------------------------------------------------------------------
# 3. The binding; 4. the class the engine's op builds
# ---------------------------------------------------------------------------------------------


class _Built:
    """A stand-in kernel class: keeps the descriptors it was built with."""

    token_comm_topk_sentinel = False
    token_comm_sort_local_prefix = False
    token_comm_refine_sf_first = False

    def __init__(self, problem_desc, impl_desc):
        self.problem_desc, self.impl_desc = problem_desc, impl_desc


@pytest.fixture
def stand_ins(monkeypatch):
    """The package and the engine's op (``_construct_megamoe_kernel`` and the bucket rule it reads,
    from source) over stand-ins for cutlass and the kernel package; the binding from source, its
    kernel subclass a stand-in."""
    engine = type("BlockScaledSwapAbGenphaseMoeKernel", (_Built,), {})
    ours = type("RubinTokenPathMegaMoE", (engine,), {})
    generic = type("RubinInferenceMegaMoE", (_Built,), {})
    package = ModuleType(_PACKAGE)
    package.RubinInferenceGenphaseMegaMoE = engine
    package.RubinInferenceMegaMoE = generic
    package.BlackwellInferenceMegaMoE = type("BlackwellInferenceMegaMoE", (_Built,), {})
    api = ModuleType(f"{_PACKAGE}.api")
    api.ImplDesc = api.ProblemDesc = dict
    quant = ModuleType(f"{_PACKAGE}.quant_def")
    quant.CombineFormat = SimpleNamespace(parse=lambda name: name)
    quant.QuantKind = SimpleNamespace(nvfp4=SimpleNamespace(instruction_k=lambda mode: 64))
    kernels = ModuleType(_PACKAGE.rsplit(".", 1)[0])
    kernels.cutedsl_megamoe = package
    nvgpu = ModuleType("cutlass.cute.nvgpu")
    nvgpu.OperandMajorMode = SimpleNamespace(K="K")
    for module in (kernels, package, api, quant, nvgpu):
        monkeypatch.setitem(sys.modules, module.__name__, module)
    cutlass = SimpleNamespace(Int32="Int32", Int64="Int64")
    op = _definitions(
        _OP,
        {
            "_SM107_GENPHASE_TACTICS",
            "_SM107_GENPHASE_MAX_TOKENS_PER_RANK",
            "_SM107_GENERIC_KERNEL_BUCKETS",
            "_sm107_tactic_bucket",
            "_construct_megamoe_kernel",
        },
        {"cutlass": cutlass},
    )
    binding = _definitions(
        _OURS,
        {"_TOKEN_LEVERS", "bind_token_path_levers", "_install_generation_kernel"},
        {
            "MappingProxyType": MappingProxyType,
            "BlockScaledSwapAbGenphaseMoeKernel": engine,
            "RubinTokenPathMegaMoE": ours,
        },
    )
    assert dict(binding["_TOKEN_LEVERS"]) == _CONSTANTS
    common = dict(
        static_expert_shape=(56, 6144, 3584),
        world_size=16,
        gate_up_clamp=None,
        swiglu_alpha=None,
        swiglu_beta=None,
        situ_beta=4.0,
        situ_linear_beta=25.0,
        num_topk=16,
        max_tokens_per_rank=96,
        apply_topk_in_fc1=False,
        mma_tiler_mnk=(256, 256, 256),
        cluster_shape_mnk=(4, 1, 1),
        use_2cta_instrs=True,
        schedule_policy=("phase_interleave", 3),
        token_padding_block=64,
        sf_padding_block=128,
        work_id_mode="atomic_counter",
        non_ubulk_fc2_store=False,
        epi_flag_batch=(1, 4),
        launch_cluster_count=46,
        fallback_cluster_shape_mn=None,
        preferred_cluster_count=None,
        fallback_cluster_count=None,
        flag_batch=1,
        token_back_mode="epi_warps",
        in_kernel_fc2_reduce=False,
        fc2_tma_stages=1,
    )

    def build():
        kernel, rejection = op["_construct_megamoe_kernel"](
            sm_version=107, common=common, combine_format="bf16"
        )
        assert rejection is None
        return kernel

    return SimpleNamespace(
        bind=binding["bind_token_path_levers"],
        install=binding["_install_generation_kernel"],
        build=build,
        package=package,
        engine=engine,
        ours=ours,
    )


def _state(cls) -> dict:
    return {s: getattr(cls, c) for s, c in _CONSTANTS.items()}


def test_off_leaves_the_engine_kernel_in_place(stand_ins):
    stand_ins.bind(**{s: False for s in _SWITCHES})
    assert stand_ins.package.RubinInferenceGenphaseMegaMoE is stand_ins.engine
    assert _state(stand_ins.ours) == {s: False for s in _SWITCHES}
    assert type(stand_ins.build()) is stand_ins.engine


@pytest.mark.parametrize("switch", _SWITCHES)
def test_a_switch_turns_its_constant_on_and_only_that_one(stand_ins, switch):
    stand_ins.bind(**{s: s == switch for s in _SWITCHES})
    assert _state(stand_ins.ours) == {s: s == switch for s in _SWITCHES}
    assert _state(stand_ins.engine) == {s: False for s in _SWITCHES}
    assert stand_ins.package.RubinInferenceGenphaseMegaMoE is stand_ins.ours
    stand_ins.bind(**{s: s == switch for s in _SWITCHES})  # the same setting again
    others = [s for s in _SWITCHES if s != switch]
    with pytest.raises(ValueError, match=f"MegaMoE lever {switch}"):
        stand_ins.bind(**{s: s == others[0] for s in _SWITCHES})


def test_the_op_builds_the_subclass_once_the_levers_are_on(stand_ins):
    assert type(stand_ins.build()) is stand_ins.engine
    stand_ins.bind(**{s: True for s in _SWITCHES})
    kernel = stand_ins.build()
    assert type(kernel) is stand_ins.ours
    # the op's import-time copy of the name serves an isinstance check, which the subclass passes
    assert isinstance(kernel, stand_ins.engine)
    assert _state(type(kernel)) == {s: True for s in _SWITCHES}


def test_another_generation_kernel_in_place_is_refused(stand_ins):
    stand_ins.package.RubinInferenceGenphaseMegaMoE = type("Other", (stand_ins.engine,), {})
    with pytest.raises(RuntimeError, match="the generation kernel is already"):
        stand_ins.install(stand_ins.ours)


def test_the_op_imports_the_kernel_name_when_it_builds():
    """``_construct_megamoe_kernel`` imports ``RubinInferenceGenphaseMegaMoE`` from the package in
    its body, so it builds what the package names at that moment."""
    (fn,) = [
        n
        for n in _tree(_OP).body
        if isinstance(n, ast.FunctionDef) and n.name == "_construct_megamoe_kernel"
    ]
    imports = [
        n
        for n in ast.walk(fn)
        if isinstance(n, ast.ImportFrom)
        and n.module == _PACKAGE
        and "RubinInferenceGenphaseMegaMoE" in [a.name for a in n.names]
    ]
    assert imports and all(a.asname is None for n in imports for a in n.names)


# ---------------------------------------------------------------------------------------------
# 5. The sort CTAs' local segment bases
# ---------------------------------------------------------------------------------------------


def _helper_bases(counts, slot_to_expert, local_rank, block):
    """The helper's source segment bases: the exclusive prefix of padded loads in slot order,
    then each source after the earlier ones in ring order from this rank."""
    world, experts = len(counts), len(counts[0])
    loads = [sum(counts[r][e] for r in range(world)) for e in range(experts)]
    data_base, running = [0] * experts, 0
    for slot in range(experts):
        expert = slot_to_expert[slot]
        data_base[expert] = running
        running += -(-loads[expert] // block) * block
    bases = [[0] * experts for _ in range(world)]
    for e in range(experts):
        begin = data_base[e]
        for k in range(world):
            source = (local_rank + k) % world
            bases[source][e] = begin
            begin += counts[source][e]
    return bases


def _sort_cta_bases(counts, slot_to_expert, local_rank, source, block):
    """One sort CTA's local formula (``_stage_local_segment_bases``)."""
    world, experts = len(counts), len(counts[0])
    loads_by_slot = [
        sum(counts[r][slot_to_expert[s]] for r in range(world)) for s in range(experts)
    ]
    padded = [-(-load // block) * block for load in loads_by_slot]
    prefix = [sum(padded[:s]) for s in range(experts)]
    ring = (source - local_rank + world) % world
    bases = [0] * experts
    for slot in range(experts):
        expert = slot_to_expert[slot]
        begin = prefix[slot]
        for earlier in range(world - 1):
            if earlier < ring:
                begin += counts[(local_rank + earlier) % world][expert]
        bases[expert] = begin
    return bases


@pytest.mark.parametrize("world", [4, 16])
def test_the_sort_ctas_local_bases_are_the_helpers(world):
    rng = random.Random(world)
    for _ in range(50):
        experts = 56
        counts = [
            [rng.choice([0, 0, rng.randint(0, 40)]) for _ in range(experts)] for _ in range(world)
        ]
        slot_to_expert = list(range(experts))
        rng.shuffle(slot_to_expert)
        local_rank = rng.randrange(world)
        want = _helper_bases(counts, slot_to_expert, local_rank, 32)
        for source in range(world):
            assert _sort_cta_bases(counts, slot_to_expert, local_rank, source, 32) == want[source]
