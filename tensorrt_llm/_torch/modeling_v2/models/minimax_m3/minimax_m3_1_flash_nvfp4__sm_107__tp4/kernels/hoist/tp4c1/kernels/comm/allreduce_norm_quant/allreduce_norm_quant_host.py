# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Boundary factories on the allreduce-norm-quant extension: all-reduce + residual + Gemma RMSNorm (+ MXFP8),
reading the partials a producer pushed into the lamport workspace."""

import torch

from ....runtime.knobs import KNOBS as _K
from . import allreduce_norm_quant_build as lbcore

EPS = 1e-6
CL = int(_K.get("H9_CFG", "0"))  # 0: per-row-count specialization; else cl*10 + rows per CTA


def _call(blk, x, residual, gamma, res_out, norm_out, mxq=None, mxs=None):
    e, w = lbcore.ext(), lbcore.ws()
    site = lbcore.new_site()
    t = blk.t

    def fn():
        e.tagged_allreduce_norm(
            w, site, t, x(), None, residual(), gamma, EPS, 1, norm_out, res_out, mxq, mxs, CL, True
        )

    return fn


def _prx_ws(t):
    """(extension, workspace) of the lamport bucket the producer pushed rows `t` into (allreduce_push_workspace)."""
    from ...common import allreduce_push_workspace as k2

    return k2.prx_ws(t)


def _post_prepushed_mx(blk, cl):
    """boundary_post on prepushed lamport slots that also writes MXFP8 into post_mxfp8_buffers (the shared expert's
    input)."""
    from ...common import post_mxfp8_buffers as h10

    e = lbcore.ext()
    t = blk.t
    pext, w = _prx_ws(t)
    pp = pext.push_params_bytes(w)
    q, s = h10._mx_bufs(blk)
    qv, sv = q.view(torch.uint8), s.view(torch.uint8).view(-1)

    def fn():
        e.prepushed_allreduce_norm(
            pp,
            t,
            blk.residual_mid,
            blk.aw.post_norm,
            EPS,
            1,
            blk.normed2,
            blk.residual_out,
            qv,
            sv,
            cl,
        )

    return fn


def make_post_prepushed_mx(blk):
    return _post_prepushed_mx(blk, 8)
