# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""msa_kernel_builds: fmha_sm100's plan, run and variant lookup routed through KernelBuilder.

A stand-in package picks one of two variants by the longest query segment, so nothing is
compiled and no build reaches a worker process.
"""

import importlib.util
import os
import sys
import types
from pathlib import Path

import pytest
import torch
import torch.utils.cpp_extension as cpp_extension

from tensorrt_llm._torch.attention.backends.sparse.minimax_m3.kernels import msa_kernel_builds
from tensorrt_llm._torch.build_pool import BuildPool
from tensorrt_llm._torch.kernel_builder import BuildStrategy, KernelBuilder

_OP = msa_kernel_builds._OP
# The stand-in's variants: a 64-row query tile up to 64 rows, a 128-row one above.
_SHORT, _LONG = "v64_0", "v128_0"


@pytest.fixture
def package(monkeypatch, tmp_path):
    """The stand-in package's module, routed through this process's KernelBuilder, with the
    package's own variant lookups recorded in order (``events``), the query length at which a
    plan raises (``fail``), and the libraries built ahead of time (``prebuilt``, paths relative
    to their directory)."""
    events = []
    fail = {"qo_len": None}
    prebuilt = set()
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
        _prebuilt=lambda relative: tmp_path / "libs" / relative if relative in prebuilt else None,
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
    # Workers take builds; none is started here.
    monkeypatch.setattr(BuildPool, "accepting", lambda self: True)
    return types.SimpleNamespace(
        module=module, builder=builder, events=events, fail=fail, prebuilt=prebuilt
    )


def _call(module, longest):
    """Plan and run a batch of two query segments, the longest ``longest`` rows."""
    lens = torch.tensor([3, longest], dtype=torch.int32)
    q = torch.zeros(3 + longest, 2, 8)
    k = v = torch.zeros(16, 2, 8)
    return module.fmha_sm100(q, k, v, module.fmha_sm100_plan(lens, lens.clone()))


@pytest.fixture
def build_mode(package, monkeypatch):
    """KernelBuilder in build mode, with the builds asked for recorded, not run."""
    built = []
    monkeypatch.setattr(package.builder, "active", lambda: True)
    monkeypatch.setattr(package.builder, "build", lambda op, builds: built.extend(builds))
    return built


def _variants(built):
    return {key for key, task, _ in built if task.function.endswith(":build_variant")}


def test_a_run_in_build_mode_builds_the_variants_of_the_query_lengths_it_sweeps(
    package, build_mode
):
    _call(package.module, 100)
    # Query lengths 1 to 100 reach both tiles.
    assert _variants(build_mode) == {_SHORT, _LONG}
    modules = {key for key, task, _ in build_mode if task.function.endswith(":build_module")}
    assert modules == {"plan", "reduction", "sparse_topk"}
    # The ranks split the variants, the modules and the k2q CSR extension: the variants and
    # modules over the package's build directory, the extension over PyTorch's.
    configs = {key: config for key, _, config in build_mode}
    assert configs[_SHORT] is configs["plan"] is msa_kernel_builds._CONFIG
    k2q = configs[msa_kernel_builds._K2Q_CSR_MODULE]
    assert k2q is msa_kernel_builds._K2Q_CONFIG
    assert k2q.strategy is BuildStrategy.SHARED
    assert k2q.share_scope is msa_kernel_builds._k2q_csr_scope
    # The replays looked nothing up; the call's own lookup is the package's.
    assert package.events == [("lookup", (128, False))]

    # A call of the same signature and lengths lists nothing new.
    count = len(build_mode)
    _call(package.module, 100)
    assert len(build_mode) == count


def test_a_run_in_build_mode_skips_what_the_package_has_built_ahead_of_time(package, build_mode):
    """A variant, module or extension the package loads from its ahead-of-time build
    (BUILD_MSA_AOT) is not built in the warmup."""
    package.prebuilt.update(
        {
            f"{_SHORT}/{_SHORT}.so",
            "plan/fmha_sm100_plan.so",
            f"k2q/{msa_kernel_builds._K2Q_CSR_EXTENSION}.so",
        }
    )
    _call(package.module, 100)
    assert _variants(build_mode) == {_LONG}
    modules = {key for key, task, _ in build_mode if task.function.endswith(":build_module")}
    assert modules == {"reduction", "sparse_topk"}
    assert msa_kernel_builds._K2Q_CSR_MODULE not in {key for key, _, _ in build_mode}


def test_a_replay_that_raises_leaves_the_kernels_to_their_first_use(package, build_mode):
    package.fail["qo_len"] = 7
    _call(package.module, 100)
    assert build_mode == []
    assert msa_kernel_builds._replay.found is None
    assert package.events == [("lookup", (128, False))]


def test_in_inference_mode_a_run_builds_nothing(package, monkeypatch):
    built = []
    monkeypatch.setattr(package.builder, "build", lambda op, builds: built.extend(builds))
    _call(package.module, 40)
    _call(package.module, 40)
    assert built == []
    assert package.events == [("lookup", (64, False)), ("lookup", (64, False))]


@pytest.mark.parametrize("builds", ["variants", "k2q_csr"])
def test_ranks_split_builds_only_over_one_cache_directory_of_one_host(
    monkeypatch, tmp_path, builds
):
    """The variants and modules land in the package's build directory, the k2q CSR extension in
    PyTorch's: the ranks that read one such directory on one host split those builds."""
    cache, elsewhere = tmp_path / "cache", tmp_path / "elsewhere"
    cache.mkdir()
    elsewhere.mkdir()
    (tmp_path / "link").symlink_to(cache)

    def scope(base, host="node-0"):
        monkeypatch.setattr(os, "uname", lambda: types.SimpleNamespace(nodename=host))
        if builds == "k2q_csr":
            monkeypatch.setattr(
                cpp_extension, "_get_build_directory", lambda name, verbose: str(base)
            )
            return msa_kernel_builds._k2q_csr_scope()
        jit = types.SimpleNamespace(_jit_base=lambda: base)
        monkeypatch.setattr(msa_kernel_builds, "_package", types.SimpleNamespace(jit=jit))
        return msa_kernel_builds._cache_scope()

    # One directory, however a rank names it.
    assert scope(cache) == scope(tmp_path / "link")
    assert scope(cache) != scope(elsewhere)
    assert scope(cache) != scope(cache, host="node-1")


def test_a_worker_reports_the_k2q_csr_extension_built_only_when_its_import_compiled_it(
    monkeypatch, tmp_path
):
    """The worker that claimed the extension's build imports its module, which compiles the
    extension or loads it built: the build is reported built only when the extension's library
    changed during the import, not by how long the import took."""
    directory = tmp_path / "torch_extensions" / msa_kernel_builds._K2Q_CSR_EXTENSION
    directory.mkdir(parents=True)
    library = directory / f"{msa_kernel_builds._K2Q_CSR_EXTENSION}.so"
    monkeypatch.setattr(cpp_extension, "_get_build_directory", lambda name, verbose: str(directory))
    monkeypatch.setattr(msa_kernel_builds, "_require_package", lambda: None)
    # The package's module: its import compiles the extension unless it is built.
    (tmp_path / "k2q_csr_module.py").write_text(
        "import os\n"
        f"if not os.path.exists({str(library)!r}):\n"
        f"    open({str(library)!r}, 'w').close()\n"
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setattr(msa_kernel_builds, "_K2Q_CSR_MODULE", "k2q_csr_module")

    def import_in_a_worker():
        # Each worker process imports the module anew.
        monkeypatch.delitem(sys.modules, "k2q_csr_module", raising=False)
        return msa_kernel_builds.import_k2q_csr()

    assert import_in_a_worker() is True and library.exists()
    # A later worker, of this launch or another, loads it.
    assert import_in_a_worker() is False


def test_the_packaged_fmha_sm100_has_what_the_builds_ahead_use() -> None:
    """The builds ahead reach fmha_sm100 through names of its own (``_Package``); a package that
    lacks one keeps its kernels on their first use, with one warning. The package TensorRT-LLM
    ships must keep them, so that a version of it that drops one fails here."""
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("SM100 (Blackwell) required")
    from tensorrt_llm._torch.attention.backends.sparse.minimax_m3.kernels.msa_utils import (
        msa_package_available,
    )

    if not msa_package_available():
        pytest.skip("fmha_sm100 (MSA) not importable")
    import fmha_sm100
    from fmha_sm100 import api, jit

    msa_kernel_builds._Package(fmha_sm100, api, jit)
    # A worker imports the k2q CSR extension's module by this name: found without importing it,
    # which would compile the extension. The module builds the extension of this name, in the
    # directory PyTorch's cpp_extension.load builds it in.
    spec = importlib.util.find_spec(msa_kernel_builds._K2Q_CSR_MODULE)
    assert spec is not None
    assert f'"{msa_kernel_builds._K2Q_CSR_EXTENSION}"' in Path(spec.origin).read_text()
    assert os.path.isdir(msa_kernel_builds._k2q_csr_directory())
