# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""B1 / B2 skinny GEMM stand-in. The replica only runs B8 (one request x q8): providers.SkinnyProjection builds
configs for M 1 / 2 but never runs them at B8, so these are inert (``run`` at M <= 2 would raise)."""


class SkinnyGemmConfig:
    def __init__(self, *args):
        self.args = args


def shape_dynamic_skinny_gemm(x, weight, cfg, bias):
    if x.shape[0] > 2:
        raise RuntimeError("skinny GEMM stand-in called at M > 2")
    return None
