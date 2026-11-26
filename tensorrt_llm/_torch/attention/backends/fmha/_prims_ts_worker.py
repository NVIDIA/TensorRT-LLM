# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Standalone PrimTS compiler. Only import GPU libraries inside pool workers."""

import json
import multiprocessing
import os
import signal
import sys
import threading
import time
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from importlib.machinery import ModuleSpec
from pathlib import Path
from types import ModuleType
from typing import Any

_BACKENDS = "tensorrt_llm._torch.attention.backends"
_VENDOR = f"{_BACKENDS}.prims_ts"
_ROOT = Path(__file__).resolve().parents[4]
# Packages served from their directories without running their ``__init__`` (TensorRT-LLM's
# initialization imports MPI): PrimTS, its JIT helper and the CSA2 PrimTS kernels' host code,
# which imports ``tensorrt_llm.logger`` and ``tensorrt_llm._torch.flashinfer_utils``.
_PACKAGES = {
    "tensorrt_llm": _ROOT,
    "tensorrt_llm._torch": _ROOT / "_torch",
    _BACKENDS: _ROOT / "_torch" / "attention" / "backends",
    f"{_BACKENDS}.sparse": _ROOT / "_torch" / "attention" / "backends" / "sparse",
    f"{_BACKENDS}.sparse.csa2": _ROOT / "_torch" / "attention" / "backends" / "sparse" / "csa2",
    _VENDOR: _ROOT / "_torch" / "attention" / "backends" / "prims_ts",
}


class _VendorPackages:
    """Resolve PrimTS and its JIT helper without running TensorRT-LLM initialization."""

    def find_spec(
        self, fullname: str, path: Sequence[str] | None = None, target: ModuleType | None = None
    ) -> ModuleSpec | None:
        if fullname in _PACKAGES:
            locations = [str(_PACKAGES[fullname])]
        elif any(package.startswith(fullname + ".") for package in _PACKAGES):
            # Empty namespace parents bypass TensorRT-LLM's MPI initialization.
            locations = []
        else:
            return None
        spec = ModuleSpec(fullname, loader=None, is_package=True)
        spec.submodule_search_locations = locations
        return spec


def _init_worker(search_path: list[str]) -> None:
    # FlashInfer fingerprints the compiler distributions on the rank's sys.path.
    sys.path[:] = search_path
    sys.meta_path.insert(0, _VendorPackages())


def _plan_mla_decode(device: Any, payload: dict[str, Any]) -> bool:
    import torch

    from tensorrt_llm._torch.attention.backends.prims_ts import mla_decode

    kwargs = payload["plan_kwargs"].copy()
    kwargs["q_dtype"] = getattr(torch, kwargs["q_dtype"])
    wrapper = mla_decode.BatchSparseMLADecodePagedTSWrapper()
    wrapper.plan(device, payload["rows"], payload["heads"], **kwargs)
    return wrapper.is_kernel_cached()


def _plan_csa2_packed(device: Any, payload: dict[str, Any]) -> bool:
    from tensorrt_llm._torch.attention.backends.sparse.csa2 import _prims_ts_packed

    return _prims_ts_packed.compile_packed_sparse_mla_rows(
        device, payload["rows"], payload["heads"], **payload["plan_kwargs"]
    )


# Each payload kind compiles one kernel and reports whether the persistent cache now holds it.
_PLANNERS = {"mla_decode": _plan_mla_decode, "csa2_packed": _plan_csa2_packed}


def _plan(payload: dict[str, Any]) -> None:
    import torch

    device = torch.device("cuda", payload["device_index"])
    torch.cuda.set_device(device)
    try:
        cached = _PLANNERS[payload.get("kind", "mla_decode")](device, payload)
    except Exception as error:
        # Native compilation failures need the specialization that triggered them.
        raise RuntimeError(f"PrimTS worker could not compile {payload!r}: {error}") from error
    if not cached:
        raise RuntimeError(f"PrimTS worker did not persist its kernel for {payload!r}")


def _watch_rank(parent_pid: int) -> None:
    # The supervisor leads its own session. Its pool must also exit if the
    # serving rank dies before its timeout/exception cleanup can run.
    while os.getppid() == parent_pid:
        time.sleep(1)
    os.killpg(os.getpid(), signal.SIGKILL)


if __name__ == "__main__":
    request = json.load(sys.stdin)
    if os.getpgrp() != os.getpid():
        raise RuntimeError("PrimTS precompile supervisor requires a dedicated process group")
    threading.Thread(target=_watch_rank, args=(request["parent_pid"],), daemon=True).start()
    with ProcessPoolExecutor(
        max_workers=int(sys.argv[1]),
        mp_context=multiprocessing.get_context("spawn"),
        initializer=_init_worker,
        initargs=(request["sys_path"],),
    ) as pool:
        list(pool.map(_plan, request["payloads"]))
