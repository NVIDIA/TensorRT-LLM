# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""CPU coverage of the MegaMoE heuristic-v2 default tactics and autotune shortlists.

Execute production host definitions with CUDA imports replaced by metadata stubs (as in
test_megamoe_combine_dedup_tactic.py). For every priority (model, EP, capacity) the BF16-combine default
must come from the keyed rule (never the token-bucket or non-bulk fallback), pass the tactic
validator, build the intended kernel class without falling back, and keep one epilogue warpgroup on
the SM107 gen-phase buckets; the shortlist the autotuner gets must hold the default first and only
tactics the op accepts.
"""

from __future__ import annotations

import ast
import dataclasses
import functools
import inspect
import sys
import types
import weakref
from pathlib import Path
from types import SimpleNamespace

import pytest

_ROOT = Path(__file__).resolve().parents[4]
_OP = _ROOT / "tensorrt_llm/_torch/moe/custom_ops/cute_dsl_megamoe_custom_op.py"

# Priority MoE shapes: (experts, top-k, hidden, intermediate, top-k at FC1, gate/up clamp,
# SiTU beta, SiTU linear beta, SwiGLU alpha, SwiGLU beta).
_MODELS = {
    "deepseek_v4_pro": (384, 6, 7168, 3072, True, 10.0, None, None, None, None),
    "deepseek_v4_1_flash": (384, 6, 5120, 2304, True, 10.0, None, None, None, None),
    "kimi_k3_1_pro": (896, 16, 3584, 3072, False, None, 4.0, 25.0, None, None),
    "glm_5_3_flash": (288, 8, 4096, 2048, False, 10.0, None, None, None, None),
    "glm_5_3": (256, 8, 6144, 2048, False, None, None, None, None, None),
    "minimax_m3_1_flash": (128, 4, 6144, 3072, False, 7.0, None, None, 1.702, 1.0),
    "inkling_2_small": (384, 8, 4608, 2304, False, None, 4.0, 25.0, None, None),
    "mistral_large_4": (128, 4, 6144, 6144, False, None, None, None, None, None),
    "reflection_beam": (256, 8, 5120, 2560, False, None, None, None, None, None),
    "glm_5_2": (256, 8, 6144, 2048, True, None, None, None, None, None),
}
_SM107_CAPACITIES = (32, 64, 128, 192, 256, 384, 512, 1024, 2048, 4096, 8192, 16384, 32768)
_BLACKWELL_CAPACITIES = (256, 1024, 2048, 4096, 8192, 16384, 32768)


class _Kernel:
    """Kernel metadata stub: records the problem and implementation it was built with."""

    def __init__(self, problem: dict, implementation: dict) -> None:
        self.problem = problem
        self.implementation = implementation

    @classmethod
    def problem_desc_require(cls) -> dict:
        return {"swiglu_alpha": True}

    def get_workspace_sizes(self) -> tuple[int, int]:
        rows = self.problem["world_size"] if self.implementation.get("combine_dedup") else 6
        return 384, rows * 128


class _GenericKernel(_Kernel):
    pass


class _GenphaseKernel(_Kernel):
    def __init__(self, problem: dict, implementation: dict) -> None:
        super().__init__(problem, implementation)
        hidden = problem["hidden_size"]
        self.token_comm = SimpleNamespace(
            plain_activation_row_elements=hidden,
            plain_activation_sf_row_elements=_GenphaseKernel.sf_row_bytes(hidden),
        )

    sf_row_bytes = staticmethod(lambda hidden: hidden)


@pytest.fixture
def op(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
    """Load host definitions without importing TRT-LLM or any CUDA package."""
    tree = ast.parse(_OP.read_text())
    selected = (ast.Assign, ast.AnnAssign, ast.FunctionDef, ast.ClassDef)
    nodes = [ast.parse("from __future__ import annotations").body[0]]
    for node in tree.body:
        if isinstance(node, selected):
            nodes.append(node)
        elif (
            isinstance(node, ast.If)
            and isinstance(node.test, ast.Name)
            and node.test.id == "IS_MEGAMOE_OP_AVAILABLE"
        ):
            for child in node.body:
                # The torch op registration needs real torch; the host definitions do not.
                if isinstance(child, ast.Assign) and ast.unparse(child.value).startswith(
                    "torch.library."
                ):
                    continue
                if isinstance(child, selected):
                    if isinstance(child, ast.FunctionDef) and child.name == "_":
                        child.decorator_list = []
                    nodes.append(child)
    module = types.ModuleType("_megamoe_heuristic_v2_cpu")
    monkeypatch.setitem(sys.modules, module.__name__, module)
    module.__dict__.update(
        dataclasses=dataclasses,
        functools=functools,
        inspect=inspect,
        weakref=weakref,
        torch=SimpleNamespace(bfloat16="torch.bfloat16"),
        cutlass=SimpleNamespace(BFloat16="torch.bfloat16", Int32="int32", Int64="int64"),
        TunableRunner=object,
        logger=SimpleNamespace(debug=lambda *args: None, warning_once=lambda *args, **kw: None),
        get_sm_version=lambda: 107,
        ceil_div=lambda a, b: (a + b - 1) // b,
        pad_up=lambda a, b: (a + b - 1) // b * b,
    )
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(_OP), "exec"), module.__dict__)
    module.IS_MEGAMOE_OP_AVAILABLE = True
    module.RubinInferenceGenphaseMegaMoE = _GenphaseKernel
    module.RubinInferenceMegaMoE = _GenericKernel
    monkeypatch.setattr(_GenphaseKernel, "sf_row_bytes", module.megamoe_activation_sf_bytes_per_row)
    # Measured SM107 cluster occupancy (212 SMs; GPC packing gives 46 four-CTA clusters).
    occupancy = {1: 212, 2: 106, 4: 46, 8: 22}
    monkeypatch.setattr(module, "_max_active_clusters", lambda size, *args: occupancy[size])
    package = "tensorrt_llm._torch.moe.kernels.cutedsl_megamoe"
    imports = {
        "cutlass.cute.nvgpu": {"OperandMajorMode": SimpleNamespace(K="K")},
        "tensorrt_llm._torch.moe.kernels.megamoe_kernels": {
            "BlackwellInferenceMegaMoE": _GenericKernel,
            "RubinInferenceMegaMoE": _GenericKernel,
            "RubinInferenceGenphaseMegaMoE": _GenphaseKernel,
        },
        package + ".api": {"ImplDesc": dict, "ProblemDesc": dict},
        package + ".quant_def": {
            "CombineFormat": SimpleNamespace(parse=lambda value: value),
            "QuantKind": SimpleNamespace(nvfp4=SimpleNamespace(instruction_k=lambda mode: 128)),
        },
    }
    for name, values in imports.items():
        stub = types.ModuleType(name)
        stub.__dict__.update(values)
        monkeypatch.setitem(sys.modules, name, stub)
    return module


def _problem(op: types.ModuleType, sm: int, model: str, ep: int, capacity: int) -> dict:
    experts, topk, hidden, inter, fc1_topk, clamp, situ, situ_lin, alpha, beta = _MODELS[model]
    in_kernel = sm == 107 and op._sm107_use_in_kernel_fc2_reduce(capacity, fc1_topk)
    return dict(
        world_size=ep,
        num_topk=topk,
        num_experts_per_rank=experts // ep,
        hidden_size=hidden,
        intermediate_size_per_partition=inter,
        expand_intermediate_size_per_partition=2 * inter,
        max_tokens_per_rank=capacity,
        apply_topk_in_fc1=fc1_topk,
        gate_up_clamp=clamp,
        situ_beta=situ,
        situ_linear_beta=situ_lin,
        swiglu_alpha=alpha,
        swiglu_beta=beta,
        in_kernel_fc2_reduce=in_kernel,
    )


def _priority_keys():
    for model in _MODELS:
        for ep in (4, 8, 16):
            for capacity in _SM107_CAPACITIES:
                yield 107, model, ep, capacity
            for sm in (100, 103):
                for capacity in _BLACKWELL_CAPACITIES:
                    yield sm, model, ep, capacity


def _runner(op: types.ModuleType, monkeypatch: pytest.MonkeyPatch, problem: dict, sm: int):
    monkeypatch.setattr(op, "get_sm_version", lambda: sm)
    return op.Sm100MegaMoENvfp4Runner(local_rank=0, output_dtype="torch.bfloat16", **problem)


def test_every_priority_key_gets_a_keyed_tactic_that_builds_its_kernel(
    op: types.ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    checked = 0
    for sm, model, ep, capacity in _priority_keys():
        problem = _problem(op, sm, model, ep, capacity)
        runner = _runner(op, monkeypatch, problem, sm)
        key = runner._tactic_lookup_key()
        default = op._default_megamoe_tactic_for_key(key, num_tokens=capacity)
        bucket = op._sm107_tactic_bucket(capacity) if sm == 107 else None
        # Tuned, never the token-bucket or non-bulk fallback.
        keyed = (
            op._sm107_genphase_shape_tactic(key) or op._SM107_GENPHASE_TACTICS[bucket]
            if bucket is not None
            else op._heuristic_megamoe_tactic(key)
        )
        assert default == op._unpack_tactic(keyed), (sm, model, ep, capacity)
        # A measured prefill shape row may coincide with the non-bulk fallback (GLM 5.2 at EP16).
        shape_row = op._sm107_prefill_shape_tactic(key) if sm == 107 else None
        assert shape_row is not None or default != op._unpack_tactic(
            op._MEGAMOE_NONBULK_STANDALONE_TACTIC
        )
        op.validate_megamoe_tactic(default, sm_version=sm)
        # The default builds without falling back, on the intended kernel class.
        kernel, resolved = op._resolve_megamoe_kernel_tactic(
            default, allow_fallback=True, build=runner._build_kernel_once
        )
        assert resolved == default, (sm, model, ep, capacity)
        genphase = bucket is not None and default[3][0] == "decode"
        assert isinstance(kernel, _GenphaseKernel if genphase else _GenericKernel)
        assert genphase is (bucket is not None), (sm, model, ep, capacity)
        # Two epilogue warpgroups only on the SM107 generic kernel, never on a gen-phase bucket.
        assert default[12] == 1 or (sm == 107 and bucket is None)
        checked += 1
    assert checked == len(_MODELS) * 3 * (len(_SM107_CAPACITIES) + 2 * len(_BLACKWELL_CAPACITIES))


def test_shortlists_offer_the_default_first_and_only_accepted_tactics(
    op: types.ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    for sm, model, ep, capacity in _priority_keys():
        problem = _problem(op, sm, model, ep, capacity)
        runner = _runner(op, monkeypatch, problem, sm)
        key = runner._tactic_lookup_key()
        default = op._default_megamoe_tactic_for_key(key, num_tokens=capacity)
        tactics = runner.get_valid_tactics(
            [SimpleNamespace(shape=(capacity, problem["hidden_size"]))], None
        )
        assert len(set(map(repr, tactics))) == len(tactics)
        if problem["in_kernel_fc2_reduce"]:
            # The curated enumeration covers the in-kernel FC2 reduce, with the default appended.
            assert default in tactics
            continue
        assert tactics[0] == default
        shortlist = op._megamoe_shortlist_for_key(key)
        assert len(tactics) <= 1 + len(shortlist)
        bucket = op._sm107_tactic_bucket(capacity) if sm == 107 else None
        for tactic in tactics[1:]:
            assert tactic in [op._unpack_tactic(t) for t in shortlist]
            op.validate_megamoe_tactic(tactic, sm_version=sm)
            assert tactic[12] == 1 or (sm == 107 and bucket is None)
            runner._build_kernel_once(tactic)
        if sm == 107:
            # Every SM107 member builds here, so none is dropped.
            assert len(tactics) == 1 + len(
                [t for t in shortlist if op._unpack_tactic(t) != default]
            )
        else:
            # Without combine dedup on SM100 / SM103 the two dedup members are dropped.
            assert all(t[11] is False for t in tactics)


def test_rule_regions_follow_the_runner_key(op: types.ModuleType) -> None:
    def key(sm: int, model: str, ep: int, capacity: int, **overrides: object) -> tuple:
        values = dict(
            _problem(op, sm, model, ep, capacity), sm_version=sm, output_dtype="torch.bfloat16"
        )
        values.update(overrides)
        return op._megamoe_problem_key(combine_format="bf16", **values)

    # Top-k at FC1 dedups up to the world-size cap (past EP8 up to 16384 tokens per rank, measured
    # shapes only); rows per rank picks the epilogue otherwise.
    for ep, capacity in ((8, 16384), (8, 32768)):
        assert op._heuristic_megamoe_tactic(key(107, "deepseek_v4_pro", ep, capacity)) == (
            op._SM107_PREFILL_COMBINE_DEDUP_TACTIC
        ), (ep, capacity)
    for model, ep, capacity, expected in (
        ("deepseek_v4_pro", 16, 4096, op._SM107_PREFILL_TACTIC),  # 24576 rows per rank
        ("deepseek_v4_pro", 16, 16384, op._SM107_PREFILL_TWO_WARPGROUP_TACTIC),  # 98304
        ("deepseek_v4_pro", 16, 32768, op._SM107_PREFILL_TWO_WARPGROUP_TACTIC),  # 196608
        ("deepseek_v4_1_flash", 32, 4096, op._SM107_PREFILL_TACTIC),  # 24576
        ("glm_5_3_flash", 8, 4096, op._SM107_PREFILL_TACTIC),  # 32768
        ("kimi_k3_1_pro", 4, 4096, op._SM107_PREFILL_TWO_WARPGROUP_TACTIC),  # 65536
        ("mistral_large_4", 4, 16384, op._SM107_PREFILL_TWO_WARPGROUP_TACTIC),  # 65536
    ):
        assert op._heuristic_megamoe_tactic(key(107, model, ep, capacity)) == expected, (
            model,
            ep,
            capacity,
        )
    assert op._heuristic_megamoe_tactic(
        key(107, "deepseek_v4_pro", 4, 8192, helper_expert_count=2)
    ) == (op._SM107_PREFILL_TACTIC)
    for sm in (100, 103):
        assert (
            op._heuristic_megamoe_tactic(key(sm, "kimi_k3_1_pro", 4, 4096)) == op._BLACKWELL_TACTIC
        )
    # The one exact-key override: SM103 DSv4.1-Flash DEP4 at 256 tokens per rank.
    exact = key(103, "deepseek_v4_1_flash", 4, 256)
    assert op._heuristic_megamoe_tactic(exact) == op._MEGAMOE_EXACT_KEY_TACTICS[exact]
    assert list(op._MEGAMOE_EXACT_KEY_TACTICS) == [exact]
    assert (
        op._heuristic_megamoe_tactic(key(103, "deepseek_v4_1_flash", 8, 256))
        == op._BLACKWELL_TACTIC
    )


def _key(op: types.ModuleType, model: str, ep: int, capacity: int, **overrides: object) -> tuple:
    values = dict(
        _problem(op, 107, model, ep, capacity), sm_version=107, output_dtype="torch.bfloat16"
    )
    values.update(overrides)
    return op._megamoe_problem_key(combine_format="bf16", **values)


def _rows_rule(op: types.ModuleType, topk: int, capacity: int) -> tuple:
    """The rows-per-rank rule's non-dedup prefill tactic."""
    if capacity * topk <= op._SM107_PREFILL_TWO_WARPGROUP_MIN_ROWS_PER_RANK:
        return op._SM107_PREFILL_TACTIC
    return op._SM107_PREFILL_TWO_WARPGROUP_TACTIC


def test_query_sizes_exactly_the_offered_tactics(
    op: types.ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Tuning and workspace sizing share one offered list per runner configuration: the query sizes
    the maximum over exactly the tactics get_valid_tactics offers (and, where combine dedup makes a
    load-balanced launch run another default, that launch's list too)."""
    fields = ("mma_tiler_mnk", "cluster_shape_mnk", "schedule_policy", "token_back_mode")
    fields += ("combine_dedup", "epilogue_warpgroups", "non_ubulk_fc2_store", "flag_batch")

    def sizes(kernel: _Kernel) -> tuple[int, int]:
        merged = {**kernel.problem, **kernel.implementation}
        text = repr([merged.get(name) for name in fields])
        return 384, 64 * (sum(map(ord, text)) % 9973)

    monkeypatch.setattr(_Kernel, "get_workspace_sizes", sizes)
    cases = (
        (107, "deepseek_v4_1_flash", 16, 4096),  # dedup default past EP8: two launch variants
        (107, "glm_5_2", 4, 8192),  # dedup default at EP4
        (107, "kimi_k3_1_pro", 8, 4096),  # plain default
        (107, "glm_5_2", 16, 16384),  # shape row
        (103, "deepseek_v4_1_flash", 4, 4096),  # Blackwell, refused dedup members dropped
    )
    for sm, model, ep, capacity in cases:
        op._MEGAMOE_OFFERED_TACTICS_CACHE.clear()
        problem = _problem(op, sm, model, ep, capacity)
        runner = _runner(op, monkeypatch, problem, sm)
        key = runner._tactic_lookup_key()
        offered = runner.get_valid_tactics(
            [SimpleNamespace(shape=(capacity, problem["hidden_size"]))], None
        )
        assert tuple(offered) == op._megamoe_offered_tactics(key, build=runner._build_kernel_once)
        assert offered[0] == op._default_megamoe_tactic_for_key(key, num_tokens=capacity)
        variants = op._megamoe_launch_variant_keys(key)
        assert len(variants) == (2 if offered[0][11] else 1), (sm, model, ep, capacity)
        tactics = [
            t
            for k in variants
            for t in op._megamoe_offered_tactics(k, build=runner._build_kernel_once)
        ]
        expected = max(sizes(runner._build_kernel_once(t))[1] for t in tactics)
        query = {k: v for k, v in problem.items() if k != "in_kernel_fc2_reduce"}
        query["in_kernel_fc2_reduce"] = problem["in_kernel_fc2_reduce"]
        assert op.query_megamoe_kernel_properties(**query, tactic_autotune=True)[0] == expected
        untuned = max(
            sizes(
                runner._build_kernel_once(
                    op._megamoe_offered_tactics(k, build=runner._build_kernel_once)[0]
                )
            )[1]
            for k in variants
        )
        assert op.query_megamoe_kernel_properties(**query)[0] == untuned


def test_query_tries_each_tactic_once(
    op: types.ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The workspace query tries each tactic once: the offered list resolves a refused default and
    the probe is that resolution, a shortlist member refused for one launch variant is not tried
    again for the other, and a query on a warm offered-list cache does not retry a refused default."""
    tried, warnings = [], []
    refused = {("decode", 22), ("grouped", 512)}
    init = _Kernel.__init__

    def counting_init(self, problem: dict, implementation: dict) -> None:
        schedule = tuple(implementation["schedule_policy"])
        tried.append((type(self).__name__, repr(sorted(implementation.items(), key=str))))
        if schedule in refused:
            raise ValueError(f"{schedule} refused here")
        init(self, problem, implementation)

    monkeypatch.setattr(_Kernel, "__init__", counting_init)
    monkeypatch.setattr(op.logger, "warning_once", lambda msg, *, key: warnings.append(msg))
    monkeypatch.setattr(op, "_MEGAMOE_OFFERED_TACTICS_CACHE", {})

    # A refused decode default (the generation-phase kernel) runs its generic-kernel twin.
    problem = _problem(op, 107, "deepseek_v4_pro", 4, 128)
    for tactic_autotune, warm in ((False, False), (True, False), (False, True)):
        tried.clear()
        warnings.clear()
        _, direct = op.query_megamoe_kernel_properties(**problem, tactic_autotune=tactic_autotune)
        assert not direct
        assert len(tried) == len(set(tried)), tried
        genphase_default = sum(
            1 for kind, impl in tried if kind == "_GenphaseKernel" and "'decode', 22" in impl
        )
        assert genphase_default == (0 if warm else 1)
        assert len(warnings) == (0 if warm else 1)
    # A dedup key past EP8 sizes two launch variants; the grouped shortlist member of both is tried
    # once and dropped from both.
    problem = _problem(op, 107, "deepseek_v4_1_flash", 16, 4096)
    tried.clear()
    op.query_megamoe_kernel_properties(**problem, tactic_autotune=True)
    assert len(tried) == len(set(tried)), tried
    assert sum(1 for _, impl in tried if "'grouped', 512" in impl) == 1


def test_glm52_prefill_defaults(op: types.ModuleType) -> None:
    """GLM 5.2 high throughput (top-k weights at FC1) runs its prefill buckets on the shared runner.

    At DEP4 the dedup tactic measured 30% / 44% faster than the non-dedup rule tactic and 39% / 75%
    faster than the token-bucket default at 4096 / 16384 tokens per rank on SM107. At DEP16 the
    shape row (grouped schedule, standalone token-back warps) measured 2-8% faster than the rule
    from 4096 to 32768 tokens per rank and applies there only; at 2048 the rule was faster.
    """

    def default(ep: int, capacity: int, combine_dedup: bool = True) -> tuple:
        return op._default_megamoe_tactic_for_key(
            _key(op, "glm_5_2", ep, capacity), num_tokens=capacity, combine_dedup=combine_dedup
        )

    ((ep16_row, spans),) = op._SM107_PREFILL_SHAPE_TACTICS.values()
    assert spans == ((4096, 32768),)
    for capacity in (2048, 4096, 8192, 16384, 32768):
        assert default(4, capacity) == op._SM107_PREFILL_COMBINE_DEDUP_TACTIC, capacity
        expected = ep16_row if capacity >= 4096 else _rows_rule(op, 8, capacity)
        assert default(16, capacity) == expected, capacity
        assert default(16, capacity, combine_dedup=False) == expected, capacity
    for capacity in (4095, 32769):
        assert default(16, capacity) == _rows_rule(op, 8, capacity), capacity
    for capacity in (4096, 4097, 32768):
        assert default(16, capacity) == ep16_row, capacity
    # The row is keyed on the whole shape: EP8 (32 experts per rank) keeps the rule.
    assert default(8, 4096) == op._SM107_PREFILL_COMBINE_DEDUP_TACTIC


def test_ep16_dedup_takes_the_allowlisted_shapes_only(op: types.ModuleType) -> None:
    """Past EP8 combine dedup is an allowlist: DSv4.1-Flash at EP16 measured 15-24% faster than
    the rule from 4096 to 16384 tokens per rank. At 32768 it stays off in the default and the
    shortlist on purpose (23-25% slower on balanced full batches, 3.1 GiB more workspace per rank), and
    DSv4-Pro at EP16 is not listed. Default, shortlist and workspace sizing share one predicate, so
    nothing else sizes a dedup workspace."""
    dedup = op._SM107_PREFILL_COMBINE_DEDUP_TACTIC
    flash = "deepseek_v4_1_flash"

    def check(key: tuple, capacity: int, wanted: bool) -> None:
        assert op._sm107_combine_dedup_allowed(key) is wanted, (key, capacity)
        default = op._default_megamoe_tactic_for_key(key, num_tokens=capacity)
        assert default[11] is wanted, (key, capacity)
        assert (dedup in op._megamoe_shortlist_for_key(key)) is (wanted and key[1] > 8)

    for capacity in (4096, 8192, 16384):
        check(_key(op, flash, 16, capacity), capacity, True)
        check(_key(op, flash, 16, capacity, swiglu_alpha=1.0, swiglu_beta=0.0), capacity, True)
    # Outside the measured span: below 4096 and just above 16384.
    for capacity in (2048, 4095, 16385, 20000, 32768):
        check(_key(op, flash, 16, capacity), capacity, False)
    for capacity in (4096, 16384):
        # A changed clamp, helper experts, another FC1-top-k shape, DSv4-Pro and GLM 5.2 at EP16.
        check(_key(op, flash, 16, capacity, gate_up_clamp=7.0), capacity, False)
        check(_key(op, flash, 16, capacity, helper_expert_count=2), capacity, False)
        check(_key(op, flash, 16, capacity, hidden_size=4096), capacity, False)
        check(_key(op, "deepseek_v4_pro", 16, capacity), capacity, False)
        check(_key(op, "glm_5_2", 16, capacity), capacity, False)
        # EP8 and below keep dedup for every top-k-at-FC1 shape without helpers.
        check(_key(op, "deepseek_v4_pro", 8, capacity), capacity, True)
        check(_key(op, flash, 8, capacity, hidden_size=4096), capacity, True)
        check(_key(op, flash, 8, capacity, helper_expert_count=2), capacity, False)
    pro = _key(op, "deepseek_v4_pro", 16, 16384)
    assert op._default_megamoe_tactic_for_key(pro, num_tokens=16384) == _rows_rule(op, 6, 16384)


def test_selector_precedence_exact_then_shape_then_region(op: types.ModuleType) -> None:
    """Through the public selector: an exact-key row beats a shape row, which beats the region rule."""
    key = _key(op, "glm_5_2", 16, 4096)
    ((row, _),) = op._SM107_PREFILL_SHAPE_TACTICS.values()
    assert op._default_megamoe_tactic_for_key(key, num_tokens=4096) == row
    exact = op._unpack_tactic(op._SM107_PREFILL_SHORTLIST[0])
    op._MEGAMOE_EXACT_KEY_TACTICS[key] = exact
    try:
        assert op._default_megamoe_tactic_for_key(key, num_tokens=4096) == exact
    finally:
        del op._MEGAMOE_EXACT_KEY_TACTICS[key]
    # Outside the row's spans the region rule applies.
    assert op._default_megamoe_tactic_for_key(
        _key(op, "glm_5_2", 16, 2048), num_tokens=2048
    ) == _rows_rule(op, 8, 2048)


def test_defaults_and_shortlists_follow_the_measured_rules(op: types.ModuleType) -> None:
    """Rules that held with no exception on every measured key (SM107 and GB200 / GB300 scans): a guard
    against table edits, not a runtime filter.

    - SM107 gen phase, up to 6144 rows per rank: the decode schedule (smallest margin 2.1%, 43 keys).
    - SM107 gen phase and SM100 / SM103: [4, 1, 1] clusters (1.0% / 0.8%, 51 / 32 keys).
    - SM100 / SM103: the FC2 bulk store (4.5%, 8 keys).
    - SM107 prefill with the top-k weights at FC1 and EP >= top-k: combine dedup (2.9%, 8 keys); past
      EP8 only DSv4.1-Flash at EP16 from 4096 to 16384 tokens per rank (the measured allowlist).
    """
    checked = 0
    for sm, model, ep, capacity in _priority_keys():
        values = dict(
            _problem(op, sm, model, ep, capacity), sm_version=sm, output_dtype="torch.bfloat16"
        )
        if values["in_kernel_fc2_reduce"]:
            continue
        key = op._megamoe_problem_key(combine_format="bf16", **values)
        default = op._default_megamoe_tactic_for_key(key, num_tokens=capacity)
        tactics = [default] + [op._unpack_tactic(t) for t in op._megamoe_shortlist_for_key(key)]
        topk = values["num_topk"]
        genphase = sm == 107 and op._sm107_tactic_bucket(capacity) is not None
        for tactic in tactics:
            where = (sm, model, ep, capacity, tactic)
            if genphase and capacity * topk <= 6144:
                assert tactic[3][0] == "decode", where
            if genphase or sm in (100, 103):
                assert tactic[1] == [4, 1, 1], where
            if sm in (100, 103):
                assert tactic[6] is True, where
        if sm == 107 and not genphase and values["apply_topk_in_fc1"] and ep >= topk:
            wanted = ep <= 8 or (model == "deepseek_v4_1_flash" and 4096 <= capacity <= 16384)
            assert default[11] is wanted, (sm, model, ep, capacity)
        checked += 1
    assert checked > 0


def test_unmeasured_prefill_shapes_take_the_rule(op: types.ModuleType) -> None:
    """An SM107 prefill shape outside the measured models (192 experts, top-8, hidden 4096,
    intermediate 1536) takes the region rule from its runner key alone: combine dedup for top-k
    weights at FC1 up to EP8, otherwise the rows-per-rank rule; past EP8 the dedup allowlist does
    not name it. Gen-phase buckets keep the gen-phase table."""

    def key(fc1_topk: bool, ep: int, capacity: int) -> tuple:
        return op._megamoe_problem_key(
            sm_version=107,
            world_size=ep,
            num_topk=8,
            num_experts_per_rank=192 // ep,
            hidden_size=4096,
            intermediate_size_per_partition=1536,
            expand_intermediate_size_per_partition=3072,
            max_tokens_per_rank=capacity,
            output_dtype="torch.bfloat16",
            apply_topk_in_fc1=fc1_topk,
            gate_up_clamp=None,
            situ_beta=None,
            situ_linear_beta=None,
            in_kernel_fc2_reduce=False,
            combine_format="bf16",
        )

    def default(fc1_topk: bool, ep: int, capacity: int, combine_dedup: bool = True) -> tuple:
        return op._default_megamoe_tactic_for_key(
            key(fc1_topk, ep, capacity), num_tokens=capacity, combine_dedup=combine_dedup
        )

    for ep in (4, 8, 16):
        for capacity in (2048, 4096, 8192, 16384, 32768):
            rule = _rows_rule(op, 8, capacity)
            assert default(False, ep, capacity) == rule, (ep, capacity)
            dedup = op._SM107_PREFILL_COMBINE_DEDUP_TACTIC if ep <= 8 else rule
            assert default(True, ep, capacity) == dedup, (ep, capacity)
            assert default(True, ep, capacity, combine_dedup=False) == rule, (ep, capacity)
            op.validate_megamoe_tactic(default(True, ep, capacity), sm_version=107)
    # Gen-phase buckets keep the gen-phase table.
    assert default(False, 8, 512)[3][0] == "decode"


def test_genphase_shape_rows_win_over_the_bucket_rule(op: types.ModuleType) -> None:
    def default(ep: int, capacity: int) -> tuple:
        values = dict(
            _problem(op, 107, "glm_5_2", ep, capacity),
            sm_version=107,
            output_dtype="torch.bfloat16",
        )
        problem_key = op._megamoe_problem_key(combine_format="bf16", **values)
        return op._default_megamoe_tactic_for_key(problem_key, num_tokens=capacity)

    def bucket_entry(capacity: int) -> tuple:
        return op._unpack_tactic(op._SM107_GENPHASE_TACTICS[capacity])

    rows = {repr(op._unpack_tactic(t)) for t in op._SM107_GENPHASE_SHAPE_TACTICS.values()}
    assert rows
    # GLM 5.2 keeps its measured rows at 384 and 512 at EP4 and EP16; other buckets and EP8 take the
    # bucket entry.
    for capacity in (384, 512):
        for ep in (4, 16):
            assert repr(default(ep, capacity)) in rows and default(ep, capacity) != bucket_entry(
                capacity
            )
        assert default(8, capacity) == bucket_entry(capacity)
    assert default(4, 256) == bucket_entry(256)


def test_genphase_decode_shortlist_offers_the_n256_tile(
    op: types.ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Up to 6144 rows per rank the gen-phase shortlist offers decode tactics only: its N128 lead-14
    member and the N256 lead-14 tile, which was the best tactic of a 288-expert top-8 shape at EP4 and
    384 tokens per rank. Above 6144 rows the shortlist is unchanged; where the default already is the
    N256 tile, the offered list holds it once."""
    n256 = op._unpack_tactic(op._SM107_N256_LEAD14)
    small = [
        op._unpack_tactic(t)
        for t in op._megamoe_shortlist_for_key(_key(op, "glm_5_3_flash", 4, 384))
    ]
    assert [t[3] for t in small] == [("decode", 14), ("decode", 14)] and small[-1] == n256
    # 512 tokens x top-16 = 8192 rows per rank: the full gen-phase shortlist.
    assert (
        op._megamoe_shortlist_for_key(_key(op, "kimi_k3_1_pro", 4, 512))
        == op._SM107_GENPHASE_SHORTLIST
    )
    for model, ep, capacity in (("glm_5_3_flash", 4, 384), ("deepseek_v4_1_flash", 4, 1024)):
        problem = _problem(op, 107, model, ep, capacity)
        runner = _runner(op, monkeypatch, problem, 107)
        offered = runner.get_valid_tactics(
            [SimpleNamespace(shape=(capacity, problem["hidden_size"]))], None
        )
        assert offered.count(n256) == 1, (model, ep, capacity, offered)
        assert all(t[3][0] == "decode" for t in offered), (model, ep, capacity)
