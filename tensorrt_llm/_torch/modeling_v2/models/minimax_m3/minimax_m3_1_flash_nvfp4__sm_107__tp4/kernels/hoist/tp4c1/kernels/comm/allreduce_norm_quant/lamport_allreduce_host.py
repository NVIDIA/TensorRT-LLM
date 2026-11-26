# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Lamport all-reduce workspaces: per row bucket (8, 16, 48) an exact-mode exchange that producers (o_proj GEMV, MoE
engines, fronts) push their partials into and the prepushed boundaries read."""

import enum
import os

import torch
import torch.distributed as dist

from ....runtime.knobs import KNOBS as _K
from ....toolchain.paths import csrc_dir

G3 = _K["G3_BUILD"]
if os.path.isdir(
    "/usr/local/cuda-13.5"
):  # tp4c1: nvcc / ninja children get these through an explicit env=
    from ....toolchain.procenv import select_cuda

    select_cuda(
        "/usr/local/cuda-13.5"
    )  # + the CCCL include fix when the side toolkit's include/cuda escapes it
EPS = 1e-6
_st = {}


class LaunchMode(enum.IntEnum):
    """Exact-mode launch arm of the lamport all-reduce (ws_create's opt_mode)."""

    CARVEOUT = 0
    NO_CARVEOUT = 1  # shipped: the max-smem carveout starves neighbouring split-K GEMMs
    LATE_TRIGGER_CARVEOUT = 2
    LATE_TRIGGER = 3


SRC = csrc_dir("comm", "allreduce_norm_quant")


def build_pr(build_root):
    from torch.utils.cpp_extension import load

    arch = _K.get("L7_ARCH", "100")
    flags = ["-O3", "-std=c++20", f"-gencode=arch=compute_{arch},code=sm_{arch}"]
    bdir = os.path.join(build_root, f"prar-{arch}")
    os.makedirs(bdir, exist_ok=True)
    return load(
        name=f"pr_ar_{arch}",
        sources=[os.path.join(SRC, "lamport_allreduce_norm_ext.cu")],
        extra_include_paths=[SRC],
        extra_cuda_cflags=flags,
        build_directory=bdir,
        verbose=False,
    )


def _cpu():
    if "cpu" not in _st:
        _st["cpu"] = dist.new_group(backend="gloo")
    return _st["cpu"]


def _gather_open(ext, ws, hdl):
    world = dist.get_world_size()
    hs = [None] * world
    dist.all_gather_object(hs, hdl.tolist(), group=_cpu())
    ext.ws_open(ws, [torch.tensor(h, dtype=torch.uint8) for h in hs])


class LamportWorkspaces:
    """One lamport exchange per row bucket for this TP group; owns the extension handles until close()."""

    BUCKETS = (8, 16, 48)

    def __init__(self, ext, exact: int = 1, mode: LaunchMode = LaunchMode.NO_CARVEOUT):
        self.ext, self.exact, self.mode = ext, exact, LaunchMode(mode)
        self.handles = {}
        for b in (
            self.BUCKETS
        ):  # create + open bucket by bucket: the gloo gathers must pair up across ranks
            hdl = torch.empty(64, dtype=torch.uint8)
            w = ext.ws_create(dist.get_rank(), dist.get_world_size(), b, hdl, exact, int(self.mode))
            _gather_open(ext, w, hdl)
            self.handles[b] = w

    @staticmethod
    def bucket(rows: int) -> int:
        return 8 if rows <= 8 else 16 if rows <= 16 else 48

    def handle(self, rows: int) -> int:
        return self.handles[self.bucket(rows)]

    def push_params(self, rows: int) -> torch.Tensor:
        """The raw PushParams bytes a producer extension copies into its kernel argument."""
        return self.ext.push_params_bytes(self.handle(rows))

    def close(self):
        for w in self.handles.values():
            self.ext.ws_destroy(w)
        self.handles = {}

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def _ext():
    if "prext" not in _st:
        _K.setdefault("L7_ARCH", "100")
        root = os.path.join(G3, "prport", "build-g7")
        if dist.get_rank() == 0:
            _st["prext"] = build_pr(root)
        dist.barrier()
        if dist.get_rank() != 0:
            _st["prext"] = build_pr(root)
    return _st["prext"]


def workspaces(exact: int = 1, mode: LaunchMode = LaunchMode.NO_CARVEOUT) -> LamportWorkspaces:
    """The process's LamportWorkspaces for (exact, mode), created on first use (a collective)."""
    key = f"pr{exact}" if mode == LaunchMode.NO_CARVEOUT else f"pr{exact}m{int(mode)}"
    if key not in _st:
        _st[key] = LamportWorkspaces(_ext(), exact, mode)
    return _st[key]


def _pr(exact=1, mode: LaunchMode = LaunchMode.NO_CARVEOUT):
    """(ext, {bucket: handle}) for callers that predate LamportWorkspaces."""
    w = workspaces(exact, mode)
    return w.ext, w.handles
