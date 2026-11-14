# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Install the TP8 state factory without changing any digested engine source."""

from .m31_local_moe_ws_receipt import receipt


def install(model, *, rank, make_state):
    """Bind the proven factory and emit one INFO receipt per new engine state.

    The caller owns target routing and pre-traffic dual-residency admission.
    Stock fallback and LOCAL_MOE kill semantics belong to make_state unchanged.
    """
    if hasattr(model, "_local_moe_state_factory"):
        raise RuntimeError("TP8 locality state factory already installed")
    if model._ENGINE_STATES:
        raise RuntimeError("TP8 locality installation follows engine state build")
    if type(rank) is not int or not 0 <= rank < 8:
        raise ValueError("TP8 locality receipt rank invalid")

    def build_state(*args, **kwargs):
        state = make_state(*args, **kwargs)
        receipt(state, rank, getattr(state, "locality", None))
        return state

    model.M3MoeState = build_state
    model._local_moe_state_factory = build_state
    return build_state
