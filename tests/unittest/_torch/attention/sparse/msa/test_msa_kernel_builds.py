# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""msa_kernel_builds: fmha_sm100's plan, run and variant lookup routed through KernelBuilder.

A stand-in package picks one of two variants by the longest query segment, so nothing is
compiled and no build reaches a worker process.
"""

import types

import pytest
import torch

from tensorrt_llm._torch.attention.backends.sparse.minimax_m3.kernels import msa_kernel_builds
from tensorrt_llm._torch.kernel_builder import KernelBuilder
from tensorrt_llm._torch.rank_worker import RankWorker

_OP = msa_kernel_builds._OP
# The stand-in's variants: a 64-row query tile up to 64 rows, a 128-row one above.
_SHORT, _LONG = "v64_0", "v128_0"


@pytest.fixture
def package(monkeypatch, tmp_path):
    """The stand-in package's module, routed through this process's KernelBuilder, whose waits
    and the package's own variant lookups are recorded in order (``events``), and the query
    length at which a plan raises (``fail``)."""
    events = []
    fail = {"qo_len": None}
    manager = types.SimpleNamespace(
        _compile_only=None,
        _load_templates=None,
        _loaded={},
        _all_module=None,
        _ensure_all_module=lambda: None,
    )
    jit = types.SimpleNamespace(
        _variant_manager=manager,
        _variant_key_from_runtime=lambda tile, split: (f"v{tile}_{int(split)}", {"tile": tile}),
        _built_dir=lambda name, library: tmp_path / name,
        _jit_base=lambda: tmp_path,
        _acquire_file_lock=None,
        _release_file_lock=None,
        _do_compile_plan=None,
        _do_compile_reduction=None,
        _do_compile_sparse_topk=None,
    )
    api = types.SimpleNamespace(
        _prefill_qlen_threshold=lambda sparse: 1024,
        get_fmha_variant=lambda *args: events.append(("lookup", args)) or args,
        get_reduction_module=lambda: None,
    )

    def plan(qo_segment_lens, kv_segment_lens, num_kv_splits=1, **options):
        qo_len = int(qo_segment_lens.max())
        if qo_len == fail["qo_len"]:
            raise KeyError("a plan that fails")
        return (None, None, None, {"qo_len": qo_len, "splits": num_kv_splits})

    def run(q, k, v, plan_info, **options):
        sub_plan = plan_info[3]
        tile = 64 if sub_plan["qo_len"] <= 64 else 128
        return api.get_fmha_variant(tile, sub_plan["splits"] > 1)

    module = types.SimpleNamespace(fmha_sm100_plan=plan, fmha_sm100=run)
    monkeypatch.setattr(msa_kernel_builds, "_package", None)
    msa_kernel_builds._route(module, api, msa_kernel_builds._Package(module, api, jit))
    builder = KernelBuilder(enabled=True)
    monkeypatch.setattr(KernelBuilder, "_instance", builder)
    # The variants take long enough for a worker; none is started here.
    monkeypatch.setattr(RankWorker, "worth_a_worker", lambda self, family, expected_s=None: True)
    monkeypatch.setattr(builder, "wait", lambda op, key: events.append(("wait", key)) or 0.0)
    return types.SimpleNamespace(module=module, builder=builder, events=events, fail=fail)


def _call(module, longest):
    """Plan and run a batch of two query segments, the longest ``longest`` rows."""
    lens = torch.tensor([3, longest], dtype=torch.int32)
    q = torch.zeros(3 + longest, 2, 8)
    k = v = torch.zeros(16, 2, 8)
    return module.fmha_sm100(q, k, v, module.fmha_sm100_plan(lens, lens.clone()))


@pytest.fixture
def build_mode(package, monkeypatch):
    """KernelBuilder in build mode, with the builds submitted recorded, not run."""
    submitted = []
    monkeypatch.setattr(package.builder, "active", lambda: True)
    monkeypatch.setattr(
        package.builder,
        "submit",
        lambda op, key, task, config=None: submitted.append((key, task)) or True,
    )
    return submitted


def _variants(submitted):
    return {key for key, task in submitted if task.function.endswith(":build_variant")}


def test_a_run_in_build_mode_submits_the_variants_of_the_query_lengths_it_sweeps(
    package, build_mode
):
    _call(package.module, 100)
    # Query lengths 1 to 100 reach both tiles.
    assert _variants(build_mode) == {_SHORT, _LONG}
    modules = {key for key, task in build_mode if task.function.endswith(":build_module")}
    assert modules == {"plan", "reduction", "sparse_topk"}
    # The replays looked nothing up; the call's own lookup waited for its build first.
    assert package.events == [("wait", _LONG), ("lookup", (128, False))]

    # A call of the same signature and lengths lists nothing new.
    count = len(build_mode)
    _call(package.module, 100)
    assert len(build_mode) == count


def test_a_replay_that_raises_leaves_the_variants_to_their_first_use(package, build_mode):
    package.fail["qo_len"] = 7
    _call(package.module, 100)
    assert _variants(build_mode) == set()
    assert msa_kernel_builds._replay.found is None
    assert package.events == [("wait", _LONG), ("lookup", (128, False))]


def test_in_inference_mode_a_lookup_waits_for_its_build_once(package, monkeypatch):
    submitted = []
    monkeypatch.setattr(
        package.builder, "submit", lambda op, key, task, config=None: submitted.append(key) or True
    )
    _call(package.module, 40)
    _call(package.module, 40)
    assert submitted == []
    assert package.events == [
        ("wait", _SHORT),
        ("lookup", (64, False)),
        ("lookup", (64, False)),
    ]
