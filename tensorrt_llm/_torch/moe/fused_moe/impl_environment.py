# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Collect and freeze machine inputs used for MoE selection."""

import os
from contextlib import contextmanager
from enum import Enum
from typing import Callable, Dict, Optional, Tuple

from tensorrt_llm.logger import logger

from .impl_contract import MoEEnvironment


class MoEDep(str, Enum):
    """Optional dependencies that affect MoE selection."""

    #: ``import flashinfer`` succeeds. Gates the SM120/SM121 NVFP4 decode
    #: backend (``CuteDslB12xFusedMoE``).
    FLASHINFER = "flashinfer"
    #: FlashInfer additionally exposes ``trtllm_bf16_moe`` /
    #: ``trtllm_bf16_routed_moe``. Strictly stronger than :attr:`FLASHINFER`
    #: and gates the TRTLLM-Gen unquantized BF16 path.
    FLASHINFER_BF16_MOE = "flashinfer_bf16_moe"
    #: The bundled DeepGEMM build exposes the ``fp8_fp4_mega_moe`` kernel.
    DEEPGEMM_MEGAMOE = "deepgemm_megamoe"
    #: ``nvidia-cutlass-dsl[cu13]`` is new enough for the MegaMoE CuteDSL ABI.
    MEGAMOE_CUTEDSL_RUNTIME = "megamoe_cutedsl_runtime"
    #: The ``trtllm::cute_dsl_megamoe_nvfp4_*`` custom ops are registered.
    MEGAMOE_CUTEDSL_OP = "megamoe_cutedsl_op"


class MoEEnvFlag(str, Enum):
    """Environment variables that MoE selection is allowed to read."""

    #: Opt-in to the FlashInfer provider for quantized TRTLLM-Gen. Changes the
    #: routing split, which is why load-balancer eligibility depends on it.
    TRTLLM_GEN_USE_FLASHINFER = "TRTLLM_GEN_FUSED_MOE_USE_FLASHINFER"


# Probe details are logged but excluded from the stable fingerprint.
DepProbe = Callable[[], Tuple[bool, str]]


def _probe_flashinfer() -> Tuple[bool, str]:
    try:
        import flashinfer  # noqa: F401
    except Exception as exc:  # noqa: BLE001 - any import failure means absent
        return False, f"import flashinfer failed: {exc}"
    return True, ""


def _probe_flashinfer_bf16_moe() -> Tuple[bool, str]:
    try:
        from flashinfer.fused_moe import core as _core
    except Exception as exc:  # noqa: BLE001 - any import failure means absent
        return False, f"import flashinfer.fused_moe.core failed: {exc}"
    missing = [
        symbol
        for symbol in ("trtllm_bf16_moe", "trtllm_bf16_routed_moe")
        if not hasattr(_core, symbol)
    ]
    if missing:
        return False, f"flashinfer.fused_moe.core lacks {', '.join(missing)}"
    return True, ""


def _probe_deepgemm_megamoe() -> Tuple[bool, str]:
    from .quantization import _import_deep_gemm, _MegaMoEUnavailable

    try:
        _import_deep_gemm()
    except _MegaMoEUnavailable as exc:
        return False, str(exc)
    return True, ""


def _probe_megamoe_cutedsl_runtime() -> Tuple[bool, str]:
    from .mega_moe.mega_moe_cute_dsl import is_megamoe_cute_dsl_runtime_available

    available, reason = is_megamoe_cute_dsl_runtime_available()
    return bool(available), "" if available else str(reason)


def _probe_megamoe_cutedsl_op() -> Tuple[bool, str]:
    # Read the module because registration updates this flag after import.
    from ..custom_ops import cute_dsl_megamoe_custom_op as megamoe_op

    if megamoe_op.IS_MEGAMOE_OP_AVAILABLE:
        return True, ""
    return False, str(megamoe_op.MEGAMOE_OP_UNAVAILABLE_REASON)


_DEP_PROBES: Dict[MoEDep, DepProbe] = {
    MoEDep.FLASHINFER: _probe_flashinfer,
    MoEDep.FLASHINFER_BF16_MOE: _probe_flashinfer_bf16_moe,
    MoEDep.DEEPGEMM_MEGAMOE: _probe_deepgemm_megamoe,
    MoEDep.MEGAMOE_CUTEDSL_RUNTIME: _probe_megamoe_cutedsl_runtime,
    MoEDep.MEGAMOE_CUTEDSL_OP: _probe_megamoe_cutedsl_op,
}

# Preserve prior defaults when environment variables are unset.
_ENV_FLAG_DEFAULTS: Dict[MoEEnvFlag, str] = {
    MoEEnvFlag.TRTLLM_GEN_USE_FLASHINFER: "0",
}

_CACHED_ENVIRONMENT: Optional[MoEEnvironment] = None
_OVERRIDE_ENVIRONMENT: Optional[MoEEnvironment] = None


def _run_probe(dep: MoEDep, probe: DepProbe) -> bool:
    try:
        available, detail = probe()
    except Exception as exc:  # noqa: BLE001 - a broken probe means "absent"
        # Treat broken probes as unavailable while keeping the failure visible.
        logger.warning(f"MoE dependency probe {dep.value} raised {type(exc).__name__}: {exc}")
        return False
    if not available:
        logger.debug(f"MoE dependency {dep.value} unavailable: {detail}")
    return available


def collect_moe_environment(force: bool = False) -> MoEEnvironment:
    """Collect and cache the frozen MoE selection environment."""
    global _CACHED_ENVIRONMENT
    if _OVERRIDE_ENVIRONMENT is not None:
        return _OVERRIDE_ENVIRONMENT
    if _CACHED_ENVIRONMENT is not None and not force:
        return _CACHED_ENVIRONMENT

    from tensorrt_llm._utils import get_sm_version

    available = tuple(
        sorted(dep.value for dep, probe in _DEP_PROBES.items() if _run_probe(dep, probe))
    )
    env_flags = tuple(
        sorted(
            (flag.value, os.environ.get(flag.value, default))
            for flag, default in _ENV_FLAG_DEFAULTS.items()
        )
    )
    environment = MoEEnvironment(sm=get_sm_version(), available_deps=available, env_flags=env_flags)
    logger.debug(
        f"collected MoE environment: sm={environment.sm} deps={available} "
        f"flags={env_flags} ({environment.fingerprint()})"
    )
    _CACHED_ENVIRONMENT = environment
    return environment


def reset_moe_environment_cache() -> None:
    """Drop the cached probe result. For tests that change probe outcomes."""
    global _CACHED_ENVIRONMENT
    _CACHED_ENVIRONMENT = None


@contextmanager
def override_moe_environment(environment: MoEEnvironment):
    """Temporarily override the collected MoE selection environment."""
    global _OVERRIDE_ENVIRONMENT
    previous = _OVERRIDE_ENVIRONMENT
    _OVERRIDE_ENVIRONMENT = environment
    try:
        yield environment
    finally:
        _OVERRIDE_ENVIRONMENT = previous
