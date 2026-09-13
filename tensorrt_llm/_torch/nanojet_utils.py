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

"""Lazy access to NanoJet and its TensorRT-LLM integration contract."""

import platform
from typing import TYPE_CHECKING, Optional

from ..logger import logger

if TYPE_CHECKING:
    from .model_config import ModelConfig

_CONTRACT = None
_AVAILABLE: Optional[bool] = None


def is_nanojet_available() -> bool:
    """Whether nanojet and its integration contract can be loaded. Cached after the first call."""
    global _AVAILABLE, _CONTRACT
    if _AVAILABLE is not None:
        return _AVAILABLE

    _AVAILABLE = False
    if platform.system() == "Windows":
        return _AVAILABLE
    try:
        import nanojet_kernels
        from nanojet_kernels.interface_contract import trtllm as contract

        _CONTRACT = contract
        _AVAILABLE = True
        logger.info(f"nanojet is available: {nanojet_kernels.__version__}")
    except ImportError:
        logger.debug("nanojet is not importable; nanojet passes will not apply")
    except AttributeError:
        # Installed but without the integration contract: too old to drive from here.
        logger.warning(
            "nanojet is installed but exposes no interface_contract.trtllm; skipping nanojet ops"
        )
    return _AVAILABLE


_MODELS_WITH_TUNE_CONFIGS_APPLIED: set = set()


def _register_compilation_passes() -> None:
    from .compilation.patterns import register_custom_pass_registrar
    from .compilation.patterns.nanojet import register_nanojet_fusions

    register_custom_pass_registrar("nanojet", register_nanojet_fusions)


def initialize_nanojet(model_config: "ModelConfig") -> None:
    """Initialize NanoJet for a compatible checkpoint when it is available."""
    from tensorrt_llm.models.modeling_utils import QuantAlgo

    if (model_config.get_quant_config().quant_algo != QuantAlgo.FP8
            or not is_nanojet_available()):
        return

    from .custom_ops.nanojet import register_nanojet_ops

    if not register_nanojet_ops():
        return
    _register_compilation_passes()
    model_config.extra_attrs["nanojet_enabled"] = True
    ensure_tune_configs(model_config)


def ensure_tune_configs(model_config) -> None:
    """Load NanoJet's tuned CUTLASS tiles for a model config. Idempotent."""
    pretrained_config = model_config.pretrained_config
    get_text_config = getattr(pretrained_config, "get_text_config", None)
    if callable(get_text_config):
        pretrained_config = get_text_config()
    head_dim = getattr(pretrained_config, "head_dim", None)
    if not isinstance(head_dim, int):
        head_dim = pretrained_config.hidden_size // pretrained_config.num_attention_heads
    shape = {
        "hidden_size": pretrained_config.hidden_size,
        "intermediate_size": pretrained_config.intermediate_size,
        "head_dim": head_dim,
        "num_attention_heads": pretrained_config.num_attention_heads,
        "num_key_value_heads": pretrained_config.num_key_value_heads,
    }
    model_identity = (
        pretrained_config.model_type,
        tuple(sorted(shape.items())),
    )
    if model_identity in _MODELS_WITH_TUNE_CONFIGS_APPLIED:
        return
    _MODELS_WITH_TUNE_CONFIGS_APPLIED.add(model_identity)
    _CONTRACT.apply_tune_configs(pretrained_config.model_type, "fp8", shape)


def nanojet_supports(op: str, **constraints) -> bool:
    """Ask nanojet whether it accepts a concrete configuration of ``op``.

    Graph transforms call this instead of restating nanojet's dispatch tables, so a nanojet
    release that widens (or narrows) the shapes it handles takes effect here without any
    change on our side.
    """
    if not is_nanojet_available():
        return False
    return _CONTRACT.supports(op, **constraints)
