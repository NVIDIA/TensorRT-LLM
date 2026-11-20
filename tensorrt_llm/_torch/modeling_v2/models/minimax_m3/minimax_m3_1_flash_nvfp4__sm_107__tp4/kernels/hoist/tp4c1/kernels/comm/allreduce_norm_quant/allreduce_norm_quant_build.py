# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tagged-word boundary extension (allreduce_norm_quant_ext.cu): one build per process, one BoundaryWorkspace
per TP group."""

import os
import tempfile

import torch
import torch.distributed as dist

from ....runtime.knobs import KNOBS as _K
from ....toolchain.paths import csrc_dir

SRC = os.path.join(csrc_dir("comm", "allreduce_norm_quant"), "allreduce_norm_quant_ext.cu")
_st = {}


def ext():
    if "ext" not in _st:
        from torch.utils.cpp_extension import load

        bd = os.path.join(_K.get("TMPDIR", tempfile.gettempdir()), "h9lb-build")
        os.makedirs(bd, exist_ok=True)

        def _load():
            return load(
                name="h9_lb",
                sources=[SRC],
                build_directory=bd,
                verbose=False,
                extra_cuda_cflags=["-O3", "-std=c++20", "-gencode=arch=compute_107a,code=sm_107a"],
                extra_cflags=["-O3", "-std=c++20"],
            )

        if dist.is_initialized():
            if dist.get_rank() == 0:
                _st["ext"] = _load()
            dist.barrier()
            if dist.get_rank() != 0:
                _st["ext"] = _load()
        else:
            _st["ext"] = _load()
    return _st["ext"]


class BoundaryWorkspace:
    """The tagged-word exchange of this TP group: one receive area per call site; owns it until close()."""

    def __init__(self, ext, nsites: int = 64):
        self.ext, self.nsites, self.used = ext, nsites, 0
        h = torch.empty(64, dtype=torch.uint8)
        self.id = ext.ws_create(dist.get_rank(), dist.get_world_size(), nsites, h)
        cpu = dist.new_group(backend="gloo")
        hs = [None] * dist.get_world_size()
        dist.all_gather_object(hs, h.tolist(), group=cpu)
        ext.ws_open(self.id, [torch.tensor(x, dtype=torch.uint8) for x in hs])
        dist.barrier()

    def new_site(self) -> int:
        assert self.used < self.nsites, "out of boundary sites"
        self.used += 1
        return self.used - 1

    def close(self):
        if self.id is not None:
            self.ext.ws_destroy(self.id)
            self.id = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def workspace(nsites: int = 64) -> BoundaryWorkspace:
    """The process's BoundaryWorkspace, created on first use (a collective)."""
    if "ws" not in _st:
        _st["ws"] = BoundaryWorkspace(ext(), nsites)
    return _st["ws"]


def ws(nsites=64):
    return workspace(nsites).id


def new_site():
    return workspace().new_site()
