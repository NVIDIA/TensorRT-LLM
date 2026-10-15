# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""AutoTuner adapter for the dense PrimsTS GEMM.

``N`` and ``K`` stay fixed. Only ``M`` is bucketed, on a fixed list from
1024 to 65536. A runtime ``M`` rounds up to the next bucket and uses that
bucket's tactic; below 1024 uses 1024, and above 65536 uses 65536.
``autotune(tuning_buckets=..., round_up=True)`` replaces the bucket list;
lookup uses the same ceil rule only while that override is still active.
Dtype and epilogue select the cache entry; they are not tactics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from flashinfer.autotuner import (
    AutoTuner,
    ConstraintSpec,
    DynamicTensorSpec,
    TuningConfig,
    TunableRunner,
)

from .support import nvfp4_128x4_numel
from .tactics import (
    config_from_tactic,
    fallback_config,
    legal_tactics,
)

# 24 points: step 1024 through 8192, step 2048 through 32768, step 8192
# through 65536. Fixed, so a tune call covers the range rather than stopping
# at the M of that call.
_M_BUCKETS = (
    1024,
    2048,
    3072,
    4096,
    5120,
    6144,
    7168,
    8192,
    10240,
    12288,
    14336,
    16384,
    18432,
    20480,
    22528,
    24576,
    26624,
    28672,
    30720,
    32768,
    40960,
    49152,
    57344,
    65536,
)


def _map_m_bucket(m: int) -> int:
    """Smallest default bucket at or above ``m``, clamped to the list."""
    if m <= _M_BUCKETS[0]:
        return _M_BUCKETS[0]
    for bucket in _M_BUCKETS:
        if bucket >= m:
            return bucket
    return _M_BUCKETS[-1]


_M_SPEC = DynamicTensorSpec(
    (0,),
    (0,),
    _M_BUCKETS,
    _map_m_bucket,
)


def _output_rows(shapes: tuple[tuple[int, ...], ...]) -> int:
    return shapes[0][0]


def _activation_scale_rows(shapes: tuple[tuple[int, ...], ...]) -> int:
    """NVFP4 activation scale length for this profile's ``M`` and packed ``K``."""
    return nvfp4_128x4_numel(shapes[0][0], shapes[0][1] * 2)


def _swiglu_output_scale_rows(shapes: tuple[tuple[int, ...], ...]) -> int:
    return nvfp4_128x4_numel(shapes[0][0], shapes[1][0] // 2)


def _linear_output_scale_rows(shapes: tuple[tuple[int, ...], ...]) -> int:
    return nvfp4_128x4_numel(shapes[0][0], shapes[1][0])


def _arange_positions(shapes, dtype, device):
    """RoPE positions must stay inside the synthesized cos/sin table."""
    return torch.arange(shapes[0], device=device, dtype=dtype)


def _constraints(output_scale_rows) -> tuple[ConstraintSpec, ...]:
    # Indices match launch_dense_gemm's input list. Absent tensors stay None;
    # a constraint on them only rewrites the profile key.
    return (
        ConstraintSpec(2, 0, _output_rows),
        ConstraintSpec(4, 0, _output_rows),
        ConstraintSpec(8, 0, _output_rows),
        ConstraintSpec(9, 0, _output_rows),
        ConstraintSpec(10, 0, _activation_scale_rows),
        ConstraintSpec(12, 0, output_scale_rows),
    )


def _tuning_config(output_scale_rows) -> TuningConfig:
    return TuningConfig(
        use_cuda_graph=True,
        use_cold_l2_cache=True,
        dynamic_tensor_specs=(_M_SPEC,),
        constraint_specs=_constraints(output_scale_rows),
        tensor_initializers=((9, _arange_positions),),
    )


_LINEAR_TUNING_CONFIG = _tuning_config(_linear_output_scale_rows)
_SWIGLU_TUNING_CONFIG = _tuning_config(_swiglu_output_scale_rows)


def dense_gemm_op_name(operand_format: str, epilogue: str) -> str:
    family = "fp8" if operand_format == "fp8_e4m3" else "fp4"
    kind = {"linear": "linear", "swiglu": "swiglu", "qkv_qknorm_rope": "qkv"}[epilogue]
    return f"prims_ts_{family}_{kind}"


def tuning_config_for(epilogue: str) -> TuningConfig:
    if epilogue == "swiglu":
        return _SWIGLU_TUNING_CONFIG
    return _LINEAR_TUNING_CONFIG


@dataclass(frozen=True)
class GemmIdentity:
    """What selects a kernel family. Not a tactic."""

    arch: int
    operand_format: str
    output_format: str
    epilogue: str
    has_bias: bool
    head_dim: Optional[int]
    is_neox: Optional[bool]
    has_qkv_scale: bool


class PrimsTsGemmRunner(TunableRunner):
    """One runner per public function. Tactics are configs, not dtypes."""

    def __init__(self, op_name: str, identity: GemmIdentity) -> None:
        self.op_name = op_name
        self.identity = identity

    def get_valid_tactics(self, inputs, profile):
        del inputs, profile
        return legal_tactics(
            self.identity.arch,
            self.identity.operand_format,
            self.identity.output_format,
            self.identity.epilogue,
        )

    def get_cache_key_extras(self, inputs) -> tuple:
        del inputs
        identity = self.identity
        return (
            identity.arch,
            identity.operand_format,
            identity.output_format,
            identity.epilogue,
            identity.has_bias,
            identity.head_dim,
            identity.is_neox,
            identity.has_qkv_scale,
            # Tactic tuples gained use_tma_store. Keep this so a winner saved
            # before that field is not replayed.
            "tma_store",
        )

    def precompile_tactics(self, inputs, tactics, profile, **kwargs) -> bool:
        del profile
        for index, tactic in enumerate(tactics):
            try:
                self.forward(inputs, tactic=tactic, **kwargs)
            except Exception:
                # The default is first. If that kernel cannot launch, the
                # failure is real. Other tactics are skipped and the profiler
                # records them as unsuccessful.
                if index == 0:
                    raise
        return True

    def forward(self, inputs, tactic=-1, do_preparation=False, scale=1.0, **kwargs):
        del do_preparation, kwargs
        from .api import _launch

        (
            a,
            weight,
            output,
            bias,
            a_scale,
            weight_scale,
            q_norm,
            k_norm,
            cos_sin,
            positions,
            sfa,
            sfb,
            sf_c,
            scale_c,
            scale_gate,
            qkv_scale,
        ) = inputs
        identity = self.identity
        if tactic == -1:
            config = fallback_config(
                arch=identity.arch,
                operand_format=identity.operand_format,
                output_format=identity.output_format,
                epilogue=identity.epilogue,
                has_bias=identity.has_bias,
                head_dim=identity.head_dim,
                is_neox=identity.is_neox,
                has_qkv_scale=identity.has_qkv_scale,
            )
        else:
            config = config_from_tactic(
                arch=identity.arch,
                operand_format=identity.operand_format,
                output_format=identity.output_format,
                epilogue=identity.epilogue,
                has_bias=identity.has_bias,
                head_dim=identity.head_dim,
                is_neox=identity.is_neox,
                has_qkv_scale=identity.has_qkv_scale,
                tactic=tactic,
            )
        packed = identity.operand_format == "nvfp4_e2m1"
        _launch(
            config,
            a,
            weight,
            output,
            (a.shape[0], weight.shape[0], a.shape[1] * (2 if packed else 1)),
            bias=bias,
            x_scale=a_scale,
            weight_scale=weight_scale if not packed else None,
            q_norm=q_norm,
            k_norm=k_norm,
            cos_sin=cos_sin,
            positions=positions,
            sfa=sfa,
            sfb=sfb,
            sf_c=sf_c,
            scale=float(scale),
            scale_c=scale_c,
            scale_gate=scale_gate,
            qkv_scale=qkv_scale,
        )
        return output


def launch_dense_gemm(
    op_name: str,
    *,
    arch: int,
    operand_format: str,
    output_format: str,
    epilogue: str,
    has_bias: bool,
    head_dim: Optional[int],
    is_neox: Optional[bool],
    has_qkv_scale: bool,
    a: torch.Tensor,
    weight: torch.Tensor,
    output: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    a_scale: Optional[torch.Tensor] = None,
    weight_scale: Optional[torch.Tensor] = None,
    q_norm: Optional[torch.Tensor] = None,
    k_norm: Optional[torch.Tensor] = None,
    cos_sin: Optional[torch.Tensor] = None,
    positions: Optional[torch.Tensor] = None,
    sfa: Optional[torch.Tensor] = None,
    sfb: Optional[torch.Tensor] = None,
    sf_c: Optional[torch.Tensor] = None,
    scale: float = 1.0,
    scale_c: Optional[torch.Tensor] = None,
    scale_gate: Optional[torch.Tensor] = None,
    qkv_scale: Optional[torch.Tensor] = None,
) -> None:
    """Pick a cached tactic, or the historical default, then launch."""
    identity = GemmIdentity(
        arch,
        operand_format,
        output_format,
        epilogue,
        has_bias,
        head_dim,
        is_neox,
        has_qkv_scale,
    )
    runner = PrimsTsGemmRunner(op_name, identity)
    inputs = [
        a,
        weight,
        output,
        bias,
        a_scale,
        weight_scale,
        q_norm,
        k_norm,
        cos_sin,
        positions,
        sfa,
        sfb,
        sf_c,
        scale_c,
        scale_gate,
        qkv_scale,
    ]
    chosen, tactic = AutoTuner.get().choose_one(
        op_name,
        [runner],
        tuning_config_for(epilogue),
        inputs,
        scale=scale,
    )
    chosen(inputs, tactic=tactic, scale=scale)
