# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Activation hand-off slot types; targets own independent slot instances."""


class PrequantSlot:
    def __init__(self):
        self.pending = None  # (data_ptr, rows, width, e4m3 [rows, width], scales)

    def publish(self, x, xq, xsf):
        self.pending = (x.data_ptr(), x.shape[0], x.shape[-1], xq, xsf)

    def clear(self):
        self.pending = None

    def take(self, x):
        """(e4m3 [x.shape[0], width], scales) if x is the published tensor, else None."""
        pending, self.pending = self.pending, None
        if (
            pending is None
            or x.dim() != 2
            or x.data_ptr() != pending[0]
            or x.shape[0] != pending[1]
            or x.shape[1] != pending[2]
        ):
            return None
        return pending[3], pending[4]


class ArPushSlot:
    """The attention boundary's all-reduce exchange for the o_proj GEMV to push into.

    The fast forward arms ``target`` (m3_mx_gemv ``ar`` tuple) around the attention call; the o_proj marks the output
    it pushed in ``pushed``; the boundary then runs the all-reduce with ``prepushed`` = the ``ar`` tuple's mode."""

    def __init__(self):
        self.target = None
        self.pushed = None  # data_ptr of the pushed output

    def arm(self, target):
        self.target = target
        self.pushed = None

    def take_target(self):
        target, self.target = self.target, None
        return target

    def take_pushed(self, x):
        pushed, self.pushed = self.pushed, None
        return pushed is not None and x.data_ptr() == pushed


class MoePushSlot:
    """The MoE boundary's all-reduce exchange for its two producers to push into: the routed-expert engine (set A,
    ``engine`` = its ``arp`` descriptor) and the shared expert's down projection (set B, ``shared`` = m3_mx_gemv
    ``ar`` tuple). Each marks the output it pushed; the boundary then runs the all-reduce with ``prepushed`` = 3."""

    def __init__(self):
        self.engine = None
        self.shared = None
        self.engine_pushed = None  # data_ptr of the pushed routed output
        self.shared_pushed = None  # data_ptr of the pushed shared output

    def arm(self, engine, shared):
        self.engine, self.shared = engine, shared
        self.engine_pushed = self.shared_pushed = None

    def disarm(self):
        self.engine = self.shared = None

    def take_engine(self):
        engine, self.engine = self.engine, None
        return engine

    def take_shared(self):
        shared, self.shared = self.shared, None
        return shared

    def take_pushed(self, shared, routed):
        """(routed pushed, shared pushed) for these outputs; clears the marks."""
        e = routed is not None and self.engine_pushed == routed.data_ptr()
        s = shared is not None and self.shared_pushed == shared.data_ptr()
        self.engine_pushed = self.shared_pushed = None
        return e, s
