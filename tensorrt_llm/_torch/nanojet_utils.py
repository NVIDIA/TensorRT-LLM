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

"""Lazy access to nanojet.

Nothing here imports nanojet at module load. TensorRT LLM scans its custom-op and transform
packages unconditionally, so an eager import would mean every install pays for nanojet —
attempting it, failing, and logging — whether or not the user asked for a nanojet pass. The
import happens on the first call, which only comes from a nanojet transform or backend that
was explicitly enabled.
"""

import platform
from typing import Optional

from ..logger import logger

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
        logger.warning("nanojet requested but not importable; nanojet passes will not apply")
    except AttributeError:
        # Installed but without the integration contract: too old to drive from here.
        logger.warning(
            "nanojet is installed but exposes no interface_contract.trtllm; skipping nanojet ops"
        )
    return _AVAILABLE


_MODELS_WITH_TUNE_CONFIGS_APPLIED: set = set()

# nanojet files its tuned tiles under these names; TensorRT LLM spells the algorithm its own
# way in ``quant_algo``.
_QUANT_ALGO_TO_NANOJET = {"FP8": "fp8", "FP8_BLOCK_SCALES": "blockwise_fp8"}


def ensure_tune_configs(factory) -> None:
    """Load nanojet's tuned CUTLASS tiles for the model this factory built. Idempotent.

    Called from every nanojet transform so no combination of enabled passes can miss it.
    A missing config or unrecognized quantization leaves the kernels on default tiles.
    """
    if not is_nanojet_available():
        return
    try:
        model_config, _ = factory._get_model_config()
        quantization = _QUANT_ALGO_TO_NANOJET.get(factory.get_quant_config().get("quant_algo"))
        if quantization is None:
            return
        shape = dict(
            hidden_size=model_config.hidden_size,
            intermediate_size=model_config.intermediate_size,
            head_dim=getattr(
                model_config,
                "head_dim",
                model_config.hidden_size // model_config.num_attention_heads,
            ),
            num_attention_heads=model_config.num_attention_heads,
            num_key_value_heads=model_config.num_key_value_heads,
        )
        model_identity = (model_config.model_type, quantization, tuple(sorted(shape.items())))
        if model_identity in _MODELS_WITH_TUNE_CONFIGS_APPLIED:
            return
        _MODELS_WITH_TUNE_CONFIGS_APPLIED.add(model_identity)
        _CONTRACT.apply_tune_configs(model_config.model_type, quantization, shape)
    except Exception as error:
        logger.warning(f"could not apply nanojet tune configs: {type(error).__name__}: {error}")


def nanojet_supports(op: str, **constraints) -> bool:
    """Ask nanojet whether it accepts a concrete configuration of ``op``.

    Graph transforms call this instead of restating nanojet's dispatch tables, so a nanojet
    release that widens (or narrows) the shapes it handles takes effect here without any
    change on our side.
    """
    if not is_nanojet_available():
        return False
    return _CONTRACT.supports(op, **constraints)
