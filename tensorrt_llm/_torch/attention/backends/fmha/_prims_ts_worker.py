# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Standalone PrimTS compiler. Only import GPU libraries inside pool workers."""

import json
import multiprocessing
import sys
from concurrent.futures import ProcessPoolExecutor
from importlib.machinery import ModuleSpec
from pathlib import Path
from typing import Any

_BACKENDS = "tensorrt_llm._torch.attention.backends"
_VENDOR = f"{_BACKENDS}.prims_ts"


class _VendorPackages:
    """Resolve PrimTS and its JIT helper without running TensorRT-LLM initialization."""

    def find_spec(self, fullname: str, path=None, target=None) -> ModuleSpec | None:
        if fullname == _VENDOR:
            locations = [str(Path(__file__).resolve().parents[1] / "prims_ts")]
        elif fullname == _BACKENDS:
            locations = [str(Path(__file__).resolve().parents[1])]
        elif _VENDOR.startswith(fullname + "."):
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


def _plan(payload: dict[str, Any]) -> None:
    import torch

    device = torch.device("cuda", payload["device_index"])
    torch.cuda.set_device(device)
    from tensorrt_llm._torch.attention.backends.prims_ts import mla_decode

    kwargs = payload["plan_kwargs"]
    kwargs["q_dtype"] = getattr(torch, kwargs["q_dtype"])
    mla_decode.BatchSparseMLADecodePagedTSWrapper().plan(
        device, payload["rows"], payload["heads"], **kwargs
    )


if __name__ == "__main__":
    request = json.load(sys.stdin)
    with ProcessPoolExecutor(
        max_workers=int(sys.argv[1]),
        mp_context=multiprocessing.get_context("spawn"),
        initializer=_init_worker,
        initargs=(request["sys_path"],),
    ) as pool:
        list(pool.map(_plan, request["payloads"]))
