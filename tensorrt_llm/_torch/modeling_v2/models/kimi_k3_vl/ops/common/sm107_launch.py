# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Launch constants of the family's SM107 code.

``USE_PDL``: the family's SM107 kernels launch with programmatic dependent launch (on SM90 and
newer), as the served configurations run them. It is a constant, not read from
``TRTLLM_ENABLE_PDL``: ``routing.py`` declines the SM107 targets when that engine-wide switch
turns PDL off, so a deployment that sets it falls back instead of running these launches anyway.

``log_constants``: each rank logs, once at INFO when it builds a target, the target and its
constants (``K3Sm107Constants``):

    Kimi K3 SM107 constants: kimi_k3_nvfp4__sm_107__<layout> (<n>): <field>=<value>, ...
"""

import dataclasses
from typing import Any, Final

from tensorrt_llm.logger import logger

USE_PDL: Final[bool] = True


def _value(value: Any) -> str:
    return getattr(value, "__name__", None) or repr(value)


def log_constants(target_module: str, constants: Any) -> None:
    """Log the target ``<family>.<target>.modeling`` builds and its constants dataclass."""
    target = target_module.rsplit(".", 2)[-2]
    fields = dataclasses.fields(constants)
    values = ", ".join(f"{f.name}={_value(getattr(constants, f.name))}" for f in fields)
    logger.info(f"Kimi K3 SM107 constants: {target} ({len(fields)}): {values}")


__all__ = ["USE_PDL", "log_constants"]
