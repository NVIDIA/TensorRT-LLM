# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Target-local activation hand-off state."""

from ...ops.gemm.prequant import ArPushSlot, MoePushSlot, PrequantSlot

SLOT = PrequantSlot()
AR_PUSH = ArPushSlot()
MOE_PUSH = MoePushSlot()
