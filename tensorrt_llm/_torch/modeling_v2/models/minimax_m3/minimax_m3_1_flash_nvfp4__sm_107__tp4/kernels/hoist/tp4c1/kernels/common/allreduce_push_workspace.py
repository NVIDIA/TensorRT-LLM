# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The Lamport workspace of the row bucket a producer (o_proj GEMV, MoE engine) pushes its partial into and the
prepushed boundary all-reduce reads, plus the packed push spec the CuTe GEMVs take."""

from __future__ import annotations

import torch

from ...runtime.knobs import KNOBS as _K

_st = {}


def _g3():
    from ..comm.allreduce_norm_quant import lamport_allreduce_host as gb

    return gb


def _bucket(t):
    return 8 if t <= 8 else 16 if t <= 16 else 48


def push_workspace(t):
    """(all-reduce ext, workspace) a pushing producer and its boundary share; H3_OWN_WS=1 gives the pair a private
    workspace (same ext, exact mode, unicast) so no other producer/consumer pair interleaves on its bucket slots."""
    gb = _g3()
    ext, wss = gb._pr(1)
    b = _bucket(t)
    if _K.get("H3_OWN_WS", "0") != "1":
        return ext, wss[b]
    key = f"own{b}"
    if key not in _st:
        import torch.distributed as dist

        hdl = torch.empty(64, dtype=torch.uint8)
        w = ext.ws_create(dist.get_rank(), dist.get_world_size(), b, hdl, 1)
        gb._gather_open(ext, w, hdl)
        _st[key] = w
    return ext, _st[key]


def push_spec(t):
    import struct

    ext, w = push_workspace(t)
    key = f"prx{id(w)}"
    if key not in _st:
        b = bytes(ext.push_params_bytes(w).tolist())
        peers = struct.unpack_from("8q", b, 0)
        flags = struct.unpack_from("q", b, 64)[0]
        rank, world, maxt, cluster, ndst = struct.unpack_from("5i", b, 72)
        if cluster != 4 or ndst != world:
            raise RuntimeError(
                f"push workspace params: cluster {cluster} (want 4), ndst {ndst} (want world {world})"
            )
        pt = torch.tensor(
            peers[:world],
            dtype=torch.int64,
            device=torch.device("cuda", torch.cuda.current_device()),
        )
        _st[key] = ((pt.data_ptr(), flags, rank, world, maxt, 4), pt, list(peers[:world]))
    return _st[key][0]


prx_ws, _prx_spec = push_workspace, push_spec  # names the comm/moe/attention hosts still call
