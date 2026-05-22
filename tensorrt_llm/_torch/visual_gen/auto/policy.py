# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Rewrite policy for the auto path.

A `RewritePolicy` is what a `VisGenFamilyAdapter` returns to declare which
fusions/lowerings the rewrite pipeline should run on the captured FX graph.
The full set of toggles will grow as rewrites land; for the S1 skeleton we
keep only what dispatch needs to identify a policy object exists.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict
from pydantic import Field as PydanticField


class RewritePolicy(BaseModel):
    """Per-family declarations for the FX rewrite pipeline.

    Today the matchers wire `attention_backend` because the family adapter and
    pipeline shell need to know which backend to bind. Other toggles are
    placeholders that will gain rewrite-pass meaning in later stages.
    """

    model_config = ConfigDict(extra="forbid")

    attention_backend: str = PydanticField(
        "TRTLLM",
        description=(
            "Name of the visual_gen attention backend to lower SDPA to. "
            "Any string registered in `attention_backend/utils.py:get_visual_gen_attention_backend` "
            "is accepted — VANILLA, TRTLLM, FA4 today; future Ulysses / Ring / "
            "Star / Attention2D backends become reachable here without "
            "auto-path code changes."
        ),
    )
    fuse_qkv: bool = PydanticField(
        True,
        description="Run same-input GEMM fusion on the QKV projection.",
    )
    fuse_qk_rope: bool = PydanticField(
        True,
        description=(
            "Lower the (QK-RMSNorm + RoPE) cluster to TRT-LLM's "
            "`fused_dit_qk_norm_rope` kernel (single + dual-stream sites). "
            "Default True. Disable for exact-match parity vs Diffusers eager "
            "(the kernel has sub-BF16-ULP precision differences that compound "
            "over multi-step inference; perceptually invisible but breaks "
            "pixel-equality tests). Env var `VISGEN_AUTO_DISABLE_QKROPE=1` "
            "overrides this to False without code changes."
        ),
    )
    fuse_silu_mul: bool = PydanticField(
        True,
        description="Run SwiGLU/SiLU*gate fusion in the MLP.",
    )
    fuse_add_rmsnorm: bool = PydanticField(
        True,
        description="Fuse add-residual + RMSNorm where AutoDeploy's pass matches.",
    )
