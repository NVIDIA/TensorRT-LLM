# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The only os.environ reader in tp4c1 (checks/env_lint.py). M31_TP4C1_DEBUG_* is visible only to debug
variants, and only when the config enables debug; production bytes never read env."""

from __future__ import annotations

import os

PREFIX = "M31_TP4C1_DEBUG_"
OLD_PREFIX = (
    "M3_LLK_DEBUG_"  # pre-rename prefix, still honoured; PREFIX wins when both name one knob
)


def _knobs(src) -> dict[str, str]:
    out = {k[len(OLD_PREFIX) :]: v for k, v in src.items() if k.startswith(OLD_PREFIX)}
    out.update({k[len(PREFIX) :]: v for k, v in src.items() if k.startswith(PREFIX)})
    return out


class DebugDisabled(PermissionError):
    pass


class Debug:
    def __init__(self, enabled: bool = False, environ=None):
        self.enabled = enabled
        src = os.environ if environ is None else environ
        self._vals = _knobs(src) if enabled else {}

    def get(
        self, name: str, default: str | None = None, *, variant_is_debug: bool = False
    ) -> str | None:
        if not (self.enabled and variant_is_debug):
            raise DebugDisabled(
                f"debug knob {PREFIX}{name} read outside a debug variant/debug config"
            )
        return self._vals.get(name, default)

    def snapshot(self) -> dict[str, str]:
        return dict(self._vals)


HOST_DEBUG_ENV = ("H1_DBG", "H1_DIE", "H1_NOSPEC")  # = checks/allow_host_env.txt


def host_env_set(names=HOST_DEBUG_ENV) -> dict[str, str]:
    """Host debug knobs read by byte-identical .cu copies via getenv; production requires all unset."""
    return {k: os.environ[k] for k in names if k in os.environ}


def stray_debug_env() -> list[str]:
    """M31_TP4C1_DEBUG_* (or the old prefix) set while debug is off: reported in the receipt, never honoured."""
    return sorted(k for k in os.environ if k.startswith((PREFIX, OLD_PREFIX)))
