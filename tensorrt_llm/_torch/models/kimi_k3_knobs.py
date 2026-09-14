# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Resolution of the Kimi K3 FP8 weight-read knobs.

These knobs decide whether a replicated K3 projection is read from an FP8
(e4m3, 128x128 block-scale) copy of its weights instead of BF16. They are
consumed on the checkpoint-loading path — ``load_weights`` keeps the FP8
checkpoint pairs only when the read is enabled, and the post-load conversion
swaps the modules — so they live on
:class:`~tensorrt_llm.models.modeling_utils.QuantConfig`.

They used to be read straight from the ``KIMI_K3_*`` environment variables
inside :mod:`modeling_kimi_linear`. Resolution precedence for every knob:

1. an explicit config value (not ``None``) wins;
2. else the deprecated environment variable, if set, is honored (emitting a
   one-time deprecation warning); otherwise
3. the historical default (some are computed at runtime from the arch /
   parallelism).

This keeps the shipped env-var behavior working for back-compat while making the
config surface authoritative. Kept intentionally lightweight (no torch model
imports) so the resolution logic is unit-testable on CPU.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Callable, Optional

from ..._utils import is_sm_100f
from ...logger import logger

# Deprecated env var -> the config path that now owns the knob. Used only to
# make the one-time deprecation warning actionable.
_ENV_TO_CONFIG_PATH = {
    "KIMI_K3_FP8_WEIGHT_READ": "quant_config.kimi_k3_fp8_weight_read",
    "KIMI_K3_FP8_WEIGHT_READ_KDA": "quant_config.kimi_k3_fp8_weight_read_kda",
    "KIMI_K3_FP8_WEIGHT_READ_MLA": "quant_config.kimi_k3_fp8_weight_read_mla",
    "KIMI_K3_FP8_WEIGHT_READ_GATE_UP": "quant_config.kimi_k3_fp8_weight_read_gate_up",
    "KIMI_K3_KDA_GLUE_FP8": "quant_config.kimi_k3_kda_glue_fp8",
}

# FP8 weight-read knob field names on ``QuantConfig``. These are carried from the
# user-facing ``llm_args.quant_config`` onto the checkpoint-derived
# ``model_config.quant_config`` by :func:`carry_user_quant_knobs`.
KIMI_K3_QUANT_KNOB_FIELDS = (
    "kimi_k3_fp8_weight_read",
    "kimi_k3_fp8_weight_read_kda",
    "kimi_k3_fp8_weight_read_mla",
    "kimi_k3_fp8_weight_read_gate_up",
    "kimi_k3_kda_glue_fp8",
)


def _resolve(
    config_value: Optional[Any], env_name: str, env_parser: Callable[[str], Any], default: Any
) -> Any:
    """Resolve one knob: config wins; else deprecated env (warn-once); else default.

    ``config_value`` is ``None`` when the knob was not set on the config surface.
    Whenever the deprecated env var is present a one-time warning is emitted,
    even if a config value overrides it, so users learn to migrate.
    """
    env_raw = os.environ.get(env_name)
    if env_raw is not None:
        logger.warning_once(
            f"Environment variable '{env_name}' is deprecated and will be "
            f"removed; set '{_ENV_TO_CONFIG_PATH[env_name]}' on the config "
            f"surface instead (via extra_llm_api_options). It is still honored "
            f"for now, but the config value takes precedence when both are set.",
            key=f"kimi_k3_deprecated_env::{env_name}",
        )
    if config_value is not None:
        return config_value
    if env_raw is not None:
        return env_parser(env_raw)
    return default


def _knob(config: Optional[Any], name: str) -> Optional[Any]:
    """Read ``name`` off a (possibly ``None``) config object; ``None`` if absent."""
    if config is None:
        return None
    return getattr(config, name, None)


@dataclass(frozen=True)
class Fp8WeightReadGates:
    """Resolved FP8 weight-read gates.

    ``master`` folds in the ``is_sm_100f()`` arch gate: it is ``False`` off
    Blackwell regardless of the requested value. The sub-gates only ever narrow
    an enabled master, so all of them are ``False`` when ``master`` is ``False``.
    """

    master: bool
    kda: bool
    kda_glue: bool
    mla: bool
    gate_up: bool


def resolve_fp8_weight_read_gates(
    quant_config: Optional[Any], *, enable_attention_dp: bool
) -> Fp8WeightReadGates:
    """Resolve the 5 FP8 weight-read gates from ``quant_config`` (+ deprecated env).

    The master switch is opt-in (FP8 weight reads are lossy relative to BF16, so
    a default run keeps BF16 and matches the published accuracy numbers) and is
    additionally gated on ``is_sm_100f()`` — the DeepGEMM ``fp8_swap_ab_gemm``
    kernel is Blackwell-only. The KDA / KDA-glue / MLA / gate-up sub-gates only
    narrow an already-enabled master and stay default-on.
    """
    master_requested = _resolve(
        _knob(quant_config, "kimi_k3_fp8_weight_read"),
        "KIMI_K3_FP8_WEIGHT_READ",
        lambda s: s not in ("", "0"),
        False,
    )
    master = bool(is_sm_100f() and master_requested)

    kda = master and bool(
        _resolve(
            _knob(quant_config, "kimi_k3_fp8_weight_read_kda"),
            "KIMI_K3_FP8_WEIGHT_READ_KDA",
            lambda s: s != "0",
            True,
        )
    )
    kda_glue = kda and bool(
        _resolve(
            _knob(quant_config, "kimi_k3_kda_glue_fp8"),
            "KIMI_K3_KDA_GLUE_FP8",
            lambda s: s != "0",
            True,
        )
    )
    mla = master and bool(
        _resolve(
            _knob(quant_config, "kimi_k3_fp8_weight_read_mla"),
            "KIMI_K3_FP8_WEIGHT_READ_MLA",
            lambda s: s != "0",
            True,
        )
    )
    gate_up = master and bool(
        _resolve(
            _knob(quant_config, "kimi_k3_fp8_weight_read_gate_up"),
            "KIMI_K3_FP8_WEIGHT_READ_GATE_UP",
            lambda s: s != "0",
            enable_attention_dp,
        )
    )
    return Fp8WeightReadGates(master=master, kda=kda, kda_glue=kda_glue, mla=mla, gate_up=gate_up)


def carry_user_quant_knobs(
    src_quant_config: Optional[Any], dst_quant_config: Optional[Any]
) -> None:
    """Copy the K3 FP8 weight-read knobs from the user's ``quant_config``
    (``llm_args.quant_config``) onto the checkpoint-derived
    ``model_config.quant_config``.

    ``model_config.quant_config`` is built from the checkpoint's
    ``hf_quant_config.json`` and does not carry the user's ``extra_llm_api_options``
    ``quant_config`` values, so the K3 FP8-read knobs are threaded across here.
    A no-op for knobs the user did not set (``None``) and for non-K3 runs.
    """
    if src_quant_config is None or dst_quant_config is None:
        return
    for name in KIMI_K3_QUANT_KNOB_FIELDS:
        value = getattr(src_quant_config, name, None)
        if value is not None:
            setattr(dst_quant_config, name, value)
