# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared helpers of the Kimi K3.1 family tests: skip markers, the deterministic sample of an entry's certified
cells, and the MPI-pool helpers.

An SM version is the CUDA device's compute capability as ``major * 10 + minor``: 107 is sm_107
(VR200), 103 is sm_103 (GB300). A builder reads the device when it is called, that is when the
test module that calls it is imported, as an inline ``pytest.mark.skipif`` condition would.

- ``on_sm(*sms)``: the test needs CUDA and a device of one of ``sms``. It skips on a host without
  CUDA.
- ``cpu_or_sm(*sms)``: the test runs on a host without CUDA, or on a device of one of ``sms``.
- ``on_ranks(world, *sms)``: the marks of an MPI-pool test of ``world`` ranks, which needs ``world``
  devices of one of ``sms``.

The CPU stage runs ``-m cpu_only`` and collects only the files whose text holds
``pytest.mark.cpu_only`` (``tests/unittest/conftest.py::pytest_ignore_collect``). A file that runs
there states that mark itself, beside the builder:
``pytestmark = [pytest.mark.cpu_only, cpu_or_sm(107)]``.

The family's MPI-pool test helpers: pickling, one task per rank, and what rank functions share.

The pool's workers cannot import a test module or the helper modules beside it, so a test module
calls ``pickle_by_value(__name__)`` once: the pool then pickles with cloudpickle, which sends that
module's functions, and this module's, to the workers by value. The helpers a rank function calls
inside a worker (``dep_mapping``, ``tray_allreduce``, ``bf16_deviation``) live here for that reason;
single-process tests use ``dep_mapping`` too.

``mpi_pool_executor.map`` and ``.submit`` hand each task to whichever worker is free; they do not
pin one task to each of the WORLD workers. If a rank function returns before its collective has
finished, its worker can take a second task while another worker takes none, and the case (or the
next one) waits on a rank that never joins. Collecting the results through ``one_per_rank`` (which
``run_per_rank`` does) turns that into an immediate failure that names the ranks the pool ran.
"""

import sys
import types
from collections.abc import Callable, Iterable

import pytest
import torch

__extra_import_path__ = ["../.."]
from _validating import ULP  # noqa: E402 — needs the path declared above


def _sm_version() -> int | None:
    """The CUDA device's SM version, or ``None`` on a host without CUDA."""
    if not torch.cuda.is_available():
        return None
    major, minor = torch.cuda.get_device_capability()
    return major * 10 + minor


def _names(sms: tuple[int, ...]) -> str:
    """``sm_107``, ``sm_103 or sm_107``, ``sm_100, sm_103 or sm_107``."""
    names = [f"sm_{sm}" for sm in sms]
    return names[0] if len(names) == 1 else f"{', '.join(names[:-1])} or {names[-1]}"


def on_sm(
    *sms: int, gpus: int = 1, cutedsl: bool = False, reason: str | None = None
) -> pytest.MarkDecorator:
    """Skip unless CUDA is available, the device is one of ``sms`` and the host has ``gpus`` devices.

    Args:
        sms: the SM versions the test runs on.
        gpus: the number of devices the test needs.
        cutedsl: the test also needs the CuTe DSL (``IS_CUTLASS_DSL_AVAILABLE``).
        reason: the skip reason, when the test says more than which devices it needs, such as
            which kernel needs them.
    """
    skip = _sm_version() not in sms or (gpus > 1 and torch.cuda.device_count() < gpus)
    if cutedsl:
        from tensorrt_llm._torch.cute_dsl_utils import IS_CUTLASS_DSL_AVAILABLE

        skip = skip or not IS_CUTLASS_DSL_AVAILABLE
    if reason is None:
        reason = f"needs an {_names(sms)} GPU" if gpus == 1 else f"needs {gpus} {_names(sms)} GPUs"
        if cutedsl:
            reason += " and the CuTe DSL"
    return pytest.mark.skipif(skip, reason=reason)


def cpu_or_sm(*sms: int, reason: str | None = None) -> pytest.MarkDecorator:
    """Skip on a CUDA device that is not one of ``sms``; a host without CUDA runs the test.

    Args:
        sms: the SM versions the test runs on, besides a host without CUDA.
        reason: the skip reason, when the test says more than which devices it runs on.
    """
    sm = _sm_version()
    return pytest.mark.skipif(
        sm is not None and sm not in sms,
        reason=reason or f"runs on a CPU host or an {_names(sms)} GPU",
    )


def on_ranks(
    world: int, *sms: int, cutedsl: bool = False, reason: str | None = None
) -> list[pytest.MarkDecorator]:
    """The marks of an MPI-pool test of ``world`` ranks: ``on_sm(*sms, gpus=world)``, no thread-leak check.

    The pool is reused across cases and its manager thread outlives the first one, which the
    thread-leak check would report as a leak.

    Args:
        world: the number of ranks, one device each.
        sms: the SM versions the test runs on.
        cutedsl: the test also needs the CuTe DSL (``IS_CUTLASS_DSL_AVAILABLE``).
        reason: the skip reason, when the test says more than which devices it needs.
    """
    return [
        pytest.mark.threadleak(enabled=False),
        on_sm(*sms, gpus=world, cutedsl=cutedsl, reason=reason),
    ]


def sample_cells(cells, size_key=None, group_keys=()):
    """A deterministic sample of an entry's certified cells: per group of equal ``group_keys`` spec values (dtype,
    tile, ...), the cells with the smallest and the largest ``size_key`` value (the boundaries) and the median one (a
    typical cell); without ``size_key``, the first, middle and last cell of the group in declared order.

    Args:
        cells: the entry's ``CELLS``.
        size_key: the spec key that orders a group (rows, tokens, m, ...), or None.
        group_keys: the spec keys whose values split the cells into groups.
    """
    groups = {}
    for cell in cells:
        groups.setdefault(tuple(repr(cell.spec.get(k)) for k in group_keys), []).append(cell)
    picked = []
    for group in groups.values():
        if size_key is not None:
            group = sorted(group, key=lambda c: c.spec[size_key])
        for cell in (group[0], group[len(group) // 2], group[-1]):
            if all(cell is not p for p in picked):
                picked.append(cell)
    return tuple(picked)


#: The ``sys.modules`` entry that keeps a worker's ``tray_allreduce`` modules across tasks.
_TRAY_REGISTRY = "_k31_tray_allreduce"


def pickle_by_value(module_name: str) -> None:
    """Make the MPI pool send the functions of ``module_name`` and of this module by value.

    Args:
        module_name: the calling test module's ``__name__``.
    """
    import pickle

    import cloudpickle
    from mpi4py import MPI

    cloudpickle.register_pickle_by_value(sys.modules[module_name])
    cloudpickle.register_pickle_by_value(sys.modules[__name__])
    MPI.pickle.__init__(cloudpickle.dumps, cloudpickle.loads, pickle.HIGHEST_PROTOCOL)


def _rank_key(result) -> int:
    return result["rank"]


def one_per_rank(results: Iterable, world: int, rank_of: Callable | None = None) -> list:
    """The pool's results sorted by rank, after asserting they cover ranks ``0 .. world - 1`` once.

    ``rank_of`` maps one result to the rank that produced it; by default a result is a dict with a
    ``"rank"`` key.
    """
    key = _rank_key if rank_of is None else rank_of
    results = list(results)
    ranks = sorted(key(r) for r in results)
    assert ranks == list(range(world)), f"the pool ran ranks {ranks}, not one task per rank"
    return sorted(results, key=key)


def run_per_rank(executor, fn: Callable, *args, world: int) -> list:
    """Run ``fn(*args)`` as ``world`` pool tasks; return their results through ``one_per_rank``.

    Args:
        executor: the ``mpi_pool_executor`` fixture's pool.
        fn: the rank function; it returns a dict with a ``"rank"`` key.
        *args: the arguments every task gets.
        world: the number of tasks, one per rank.
    """
    return one_per_rank(executor.starmap(fn, [args] * world), world)


def dep_mapping(world: int, rank: int, **kwargs):
    """The attention-DP pure-EP ``Mapping`` of ``world`` ranks (TP = MoE EP = ``world``, MoE TP 1).

    Args:
        world: the group size.
        rank: this rank.
        **kwargs: further ``Mapping`` arguments, e.g. ``gpus_per_node``.
    """
    from tensorrt_llm.mapping import Mapping

    return Mapping(
        world_size=world,
        rank=rank,
        tp_size=world,
        moe_ep_size=world,
        moe_tp_size=1,
        enable_attention_dp=True,
        **kwargs,
    )


def tray_allreduce(rank: int, world: int, **kwargs) -> tuple:
    """The TP ``Mapping`` of ``world`` ranks and its engine MNNVL ``AllReduce`` (BF16), in a worker.

    A rank function is pickled by value, so its globals are fresh on every task: the module (and its
    workspace) is built once per worker process and kept in a ``sys.modules`` registry.

    Args:
        rank: this rank.
        world: the TP size.
        **kwargs: further ``Mapping`` arguments, e.g. ``gpus_per_node``.

    Returns:
        ``(mapping, allreduce)``.
    """
    from tensorrt_llm._torch.distributed import AllReduce, AllReduceStrategy
    from tensorrt_llm.mapping import Mapping

    mapping = Mapping(world_size=world, tp_size=world, rank=rank, **kwargs)
    registry = sys.modules.setdefault(_TRAY_REGISTRY, types.ModuleType(_TRAY_REGISTRY))
    built = vars(registry).setdefault("by_mapping", {})
    if mapping not in built:
        built[mapping] = AllReduce(
            mapping=mapping, strategy=AllReduceStrategy.MNNVL, dtype=torch.bfloat16
        )
    return mapping, built[mapping]


def bf16_deviation(out: torch.Tensor, ref: torch.Tensor) -> tuple[float, float]:
    """(worst element deviation against its row's scale, relative RMS deviation), in bf16 ulp."""
    o, r = out.float(), ref.float()
    row = r.abs().amax(dim=-1, keepdim=True).clamp_min(1e-9)
    element = ((o - r).abs() / row).max().item() / ULP[torch.bfloat16]
    rms = ((o - r).pow(2).mean().sqrt() / r.pow(2).mean().sqrt().clamp_min(1e-9)).item()
    return element, rms / ULP[torch.bfloat16]
