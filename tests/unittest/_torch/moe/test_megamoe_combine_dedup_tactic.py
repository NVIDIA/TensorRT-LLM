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
"""CPU checks of the MegaMoE ``combine_dedup`` tactic field.

Execute production host definitions with CUDA imports replaced by metadata
stubs. These checks establish the tactic format, validation, consumer
selection, cache keys and the default rule; the kernel enforces the
problem-level preconditions (top-k at FC1, BF16 combine, world size) itself.
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

# A generic-bucket tactic that meets every tactic-local dedup condition.
_DEDUP = (
    [256, 256, 256],
    [4, 1, 1],
    None,
    ("phase_interleave", 4),
    "atomic_counter",
    "reuse_dispatch_warps",
    True,
    2,
    1,
    (1, 4),
    "expert",
    True,
)


class _KernelMetadata:
    def __init__(self, problem: dict | None = None, implementation: dict | None = None) -> None:
        self.problem = problem
        self.implementation = implementation

    def get_workspace_sizes(self) -> tuple[int, int]:
        return 384, 1024 if self.implementation.get("combine_dedup") else 768


class _GenphaseMetadata(_KernelMetadata):
    @classmethod
    def problem_desc_require(cls) -> dict:
        return {}


class _GenericMetadata(_KernelMetadata):
    @classmethod
    def problem_desc_require(cls) -> dict:
        return {}


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
    module = types.ModuleType("_megamoe_combine_dedup_cpu")
    monkeypatch.setitem(sys.modules, module.__name__, module)
    module.__dict__.update(
        dataclasses=dataclasses,
        functools=functools,
        inspect=inspect,
        weakref=weakref,
        torch=SimpleNamespace(bfloat16="torch.bfloat16"),
        cutlass=SimpleNamespace(BFloat16="torch.bfloat16", Int32="int32", Int64="int64"),
        TunableRunner=object,
        logger=SimpleNamespace(debug=lambda *args: None),
        get_sm_version=lambda: 107,
        ceil_div=lambda a, b: (a + b - 1) // b,
        pad_up=lambda a, b: (a + b - 1) // b * b,
    )
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(_OP), "exec"), module.__dict__)
    module.IS_MEGAMOE_OP_AVAILABLE = True
    module.RubinInferenceGenphaseMegaMoE = _GenphaseMetadata
    module.RubinInferenceMegaMoE = _GenericMetadata
    monkeypatch.setattr(module, "_max_active_clusters", lambda *args: 46)
    package = "tensorrt_llm._torch.moe.kernels.cutedsl_megamoe"
    imports = {
        "cutlass.cute.nvgpu": {"OperandMajorMode": SimpleNamespace(K="K")},
        "tensorrt_llm._torch.moe.kernels.megamoe_kernels": {
            "BlackwellInferenceMegaMoE": _GenericMetadata,
            "RubinInferenceMegaMoE": _GenericMetadata,
            "RubinInferenceGenphaseMegaMoE": _GenphaseMetadata,
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


def _problem(**overrides: object) -> dict:
    values = dict(
        world_size=4,
        local_rank=0,
        num_topk=6,
        num_experts_per_rank=96,
        hidden_size=7168,
        intermediate_size_per_partition=3072,
        expand_intermediate_size_per_partition=6144,
        max_tokens_per_rank=8192,
        output_dtype="torch.bfloat16",
    )
    values.update(overrides)
    return values


def test_unpack_defaults_legacy_tactics_to_no_dedup(op: types.ModuleType) -> None:
    assert op._unpack_tactic(_DEDUP) == _DEDUP
    for length in (11, 10):
        assert op._unpack_tactic(_DEDUP[:length])[11] is False
    legacy = ([256, 128, 256], [2, 1, 1], 512, "static", "epi_warps", True, 1, (1, 1))
    assert op._unpack_tactic(legacy)[11] is False
    for tokens in (64, 4096, 16384):
        assert op.default_megamoe_tactic(tokens)[11] is False


def test_validation_rejects_tactics_dedup_cannot_run(op: types.ModuleType) -> None:
    op.validate_megamoe_tactic(_DEDUP, sm_version=107)
    for index, value, reason in (
        (5, "epi_warps", "token_back_mode='epi_warps'"),
        (10, "token_tile", "token_back_ready_granularity='token_tile'"),
        (11, 1, "combine_dedup must be bool"),
    ):
        tactic = list(_DEDUP)
        tactic[index] = value
        with pytest.raises(ValueError, match=reason):
            op.validate_megamoe_tactic(tuple(tactic), sm_version=107)
    no_bulk = list(_DEDUP)
    no_bulk[6], no_bulk[7] = False, None
    with pytest.raises(ValueError, match="use_bulk_fc2_store=False"):
        op.validate_megamoe_tactic(tuple(no_bulk), sm_version=107)
    with pytest.raises(ValueError, match="SM100"):
        op.validate_megamoe_tactic(_DEDUP, sm_version=100)


def test_query_and_runner_build_the_same_dedup_kernel(
    op: types.ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = []
    original = op._construct_megamoe_kernel

    def construct(**kwargs: object) -> tuple:
        calls.append(kwargs)
        return original(**kwargs)

    monkeypatch.setattr(op, "_construct_megamoe_kernel", construct)
    problem = _problem()
    query = {
        key: value for key, value in problem.items() if key not in {"local_rank", "output_dtype"}
    }
    # The probe sizes the shared workspace from the dedup kernel itself.
    assert op.query_megamoe_kernel_properties(**query, tactic=_DEDUP) == (1024, False)
    runner = op.Sm100MegaMoENvfp4Runner(**problem)
    kernel = runner._build_kernel_once(_DEDUP)
    assert isinstance(kernel, _GenericMetadata)
    assert kernel.implementation["combine_dedup"] is True
    assert len(calls) == 2
    for call in calls:
        call["common"].setdefault("helper_expert_count", 0)
    assert calls[0] == calls[1]
    plain = (*_DEDUP[:11], False)
    assert "combine_dedup" not in runner._build_kernel_once(plain).implementation
    assert runner._tactic_cache_key(_DEDUP) != runner._tactic_cache_key(plain)


@pytest.mark.parametrize("sm_version, capacity", [(107, 256), (100, 8192)])
def test_dedup_never_reaches_a_kernel_without_the_option(
    op: types.ModuleType, sm_version: int, capacity: int
) -> None:
    common = dict(
        static_expert_shape=(96, 6144, 7168),
        world_size=4,
        num_topk=6,
        max_tokens_per_rank=capacity,
        apply_topk_in_fc1=True,
        gate_up_clamp=None,
        swiglu_alpha=None,
        swiglu_beta=None,
        situ_beta=None,
        situ_linear_beta=None,
        mma_tiler_mnk=(256, 256, 256),
        cluster_shape_mnk=(4, 1, 1),
        use_2cta_instrs=True,
        schedule_policy=("phase_interleave", 4),
        token_padding_block=64,
        sf_padding_block=128,
        work_id_mode="atomic_counter",
        non_ubulk_fc2_store=False,
        epi_flag_batch=(1, 4),
        launch_cluster_count=46,
        fallback_cluster_shape_mn=(2, 1),
        preferred_cluster_count=44,
        fallback_cluster_count=4,
        flag_batch=1,
        token_back_mode="reuse_dispatch_warps",
        in_kernel_fc2_reduce=False,
        fc2_tma_stages=2,
        combine_dedup=True,
    )
    # A gen-phase bucket and the Blackwell kernel would drop the undeclared option.
    with pytest.raises(ValueError, match="SM107 generic MegaMoE kernel only"):
        op._construct_megamoe_kernel(sm_version=sm_version, common=common, combine_format="bf16")


def test_shape_defaults_and_autotune_space_never_select_dedup(op: types.ModuleType) -> None:
    for tokens in (32, 256, 1024, 4096, 8192, 16384):
        for in_kernel in (False, True):
            tactic = op._default_megamoe_tactic_for_problem(
                sm_version=107,
                max_tokens_per_rank=tokens,
                num_tokens=tokens,
                apply_topk_in_fc1=True,
                in_kernel_fc2_reduce=in_kernel,
                combine_format="bf16",
            )
            assert len(tactic) == 12 and tactic[11] is False
    for table in (op._SM107_DSV4_PRO_DEFAULT_TACTICS.values(), op._SM107_GENPHASE_TACTICS.values()):
        assert all(op._unpack_tactic(tactic)[11] is False for tactic in table)
    # The provider is sized for the default and its non-dedup base only, so the
    # tuning space never proposes any other dedup tactic.
    for sm_version in (100, 107):
        for tokens in (64, 4096, 16384):
            for load_balance in (False, True):
                candidates = op.enumerate_megamoe_candidate_tactics(
                    tokens, sm_version, load_balance=load_balance
                )
                assert candidates and all(len(t) == 12 and t[11] is False for t in candidates)


def _dedup_cells(op: types.ModuleType) -> list[tuple[int, int, int]]:
    return [
        (world, experts, tokens)
        for (world, experts), (low, high) in op._SM107_DSV4_PRO_COMBINE_DEDUP_TOKENS.items()
        for tokens in sorted({low, (low + high) // 2, high})
    ]


def test_default_turns_dedup_on_only_in_the_measured_ranges(op: types.ModuleType) -> None:
    on = set(_dedup_cells(op))
    assert on
    for world, experts in ((4, 96), (8, 48), (16, 24), (32, 12)):
        for tokens in (256, 1024, 2048, 4096, 6144, 8192, 12288, 16384, 32768):
            key = op._sm107_dsv4_pro_problem_key(world, experts, tokens)
            tactic = op._default_megamoe_tactic_for_key(key, num_tokens=tokens)
            low, high = op._SM107_DSV4_PRO_COMBINE_DEDUP_TOKENS.get((world, experts), (1, 0))
            assert tactic[11] is (low <= tokens <= high)
            base = op._default_megamoe_tactic_for_key(key, num_tokens=tokens, combine_dedup=False)
            assert base[11] is False
            if not tactic[11]:
                assert tactic == base
                continue
            op.validate_megamoe_tactic(tactic, sm_version=107)
            # Same tactic as the measured reuse + dedup arm: only the token-back fields move.
            assert tactic[0][1] == 256 and tactic[5:7] == ("reuse_dispatch_warps", True)
            assert tactic[8] == 1 and tactic[7] == (base[7] or 1)
            assert tactic[1:5] == base[1:5] and tactic[9:11] == base[9:11]
    assert all(world <= 32 and tokens > 1024 for world, _, tokens in on)


@pytest.mark.parametrize(
    "index, value",
    [
        (0, 100),  # SM100 kernel
        (2, 8),  # top-k
        (4, 4096),  # hidden size
        (9, False),  # top-k at FC2
        (10, None),  # gate/up clamp
        (13, True),  # in-kernel FC2 reduce
        (14, "32e4m3xe8m0"),  # quantized combine
        (17, 2),  # helper experts
    ],
)
def test_default_keeps_dedup_off_outside_the_measured_problem(
    op: types.ModuleType, index: int, value: object
) -> None:
    for world, experts, tokens in _dedup_cells(op):
        key = list(op._sm107_dsv4_pro_problem_key(world, experts, tokens))
        key[index] = value
        assert op._sm107_default_combine_dedup(tuple(key)) is False


def test_query_sizes_the_provider_for_dedup_and_its_base(
    op: types.ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    def sizes(kernel: _KernelMetadata) -> tuple[int, int]:
        rows = kernel.problem["world_size"] if kernel.implementation.get("combine_dedup") else 6
        return 384, rows * 128

    monkeypatch.setattr(_KernelMetadata, "get_workspace_sizes", sizes)
    # The shape defaults launch [4, 1, 1] clusters with a [2, 1, 1] fallback.
    monkeypatch.setattr(op, "_max_active_clusters", lambda size, *args: 184 // size)
    for world, experts, tokens in _dedup_cells(op):
        query = dict(
            world_size=world,
            num_topk=6,
            num_experts_per_rank=experts,
            hidden_size=7168,
            intermediate_size_per_partition=3072,
            expand_intermediate_size_per_partition=6144,
            max_tokens_per_rank=tokens,
            gate_up_clamp=10.0,
        )
        assert op.query_megamoe_kernel_properties(**query) == (max(world, 6) * 128, False)
        key = op._sm107_dsv4_pro_problem_key(world, experts, tokens)
        runner = op.Sm100MegaMoENvfp4Runner(**_problem(**query, output_dtype="torch.bfloat16"))
        assert runner.unique_id() == key
        default = op._default_megamoe_tactic_for_key(key, num_tokens=tokens)
        assert runner._build_kernel_once(default).implementation["combine_dedup"] is True
        inputs = [SimpleNamespace(shape=(tokens, 7168))]
        tactics = runner.get_valid_tactics(inputs, None)
        assert [t for t in tactics if op._unpack_tactic(t)[11]] == [default]
