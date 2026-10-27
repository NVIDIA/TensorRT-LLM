# SPDX-FileCopyrightText: Copyright (c) 2026 MiniMax
# SPDX-License-Identifier: MIT

import enum


class NamedBarrierFwdSm100(enum.IntEnum):
    Epilogue = enum.auto()  # starts from 1 as barrier 0 is reserved for sync_threads()
    TmemPtr = enum.auto()
    SoftmaxStatsW0 = enum.auto()
    SoftmaxStatsW1 = enum.auto()
    SoftmaxStatsW2 = enum.auto()
    SoftmaxStatsW3 = enum.auto()
    SoftmaxStatsW4 = enum.auto()
    SoftmaxStatsW5 = enum.auto()
    SoftmaxStatsW6 = enum.auto()
    SoftmaxStatsW7 = enum.auto()
    LoadWG = enum.auto()
    StoreEpilogue = enum.auto()
    KvLoad = enum.auto()
    KvDequantK = enum.auto()
    KvDequantV = enum.auto()
