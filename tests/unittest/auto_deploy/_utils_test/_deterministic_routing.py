# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
"""Context-manager intercept that forces MoE routing to pick experts ``[0..top_k-1]``.

Why this exists
---------------

The sharding-IR equivalence test runs an unsharded and a sharded copy of the
same exported graph and compares per-token logits. MoE top-k routing is a
non-smooth argmax: under bf16 reduction-order noise the two sides can pick
*different* experts for the same token, producing per-token errors that look
like sharding bugs but are really finite-precision artifacts.

Suppressing that without mutating the model
-------------------------------------------

Routing across the 17 router/gate classes under
``tensorrt_llm/_torch/auto_deploy/models/custom/`` ultimately funnels into
one of three ops:

  1. ``torch.ops.aten.topk.default`` -- the PyTorch top-k. Used directly by
     Qwen3.5, Gemma4, DeepSeek-V2, Mistral4, Llama4, KimiK2, and the bare
     ``nn.Linear`` gates whose parent MoE block does its own ``torch.topk``.

  2. ``torch.ops.trtllm.noaux_tc_op`` -- the fused
     ``sigmoid + bias + group-top-k + normalize`` kernel. Used by DeepSeek-V3,
     Nemotron-H, and the Glm4/HunYuan family.

  3. ``torch.ops.auto_deploy.torch_moe_router.default`` -- AD's reference
     router op (linear -> topk -> softmax -> scatter). Used by GptOss and
     Granite.

A ``TorchDispatchMode`` intercept on those three ops is sufficient to force
deterministic routing for every router we ship, without knowing anything
about the router classes. Compared to the previous helper
(``fix_moe_routers_deterministic``) it has no class-name dispatch table, no
``weight``/``bias`` mutation, no ``forward`` monkey-patch, and gracefully
handles routers we have not yet enumerated -- as long as their routing goes
through one of the three ops above.

Precedent: ``_MoeExpertProbe`` in
``tensorrt_llm/_torch/auto_deploy/export/export.py:108`` uses the same
``TorchDispatchMode`` pattern to discover expert lists during export. We
follow the same fall-through style (any op we do not intercept is delegated
to ``func(*args, **kwargs)``) so unrelated ops are unaffected.

Usage
-----

::

    with DeterministicMoeRoutingMode():
        y_unsharded = gm_unsharded(input_ids=..., position_ids=...)
        y_sharded = gm_sharded(input_ids=..., position_ids=...)

The mode is active only inside the ``with`` block; both forwards see the
same deterministic routing decisions, so per-token expert assignments are
bit-identical across precisions and reduction orders.
"""

from typing import Any, Sequence

import torch
from torch.utils._python_dispatch import TorchDispatchMode


class DeterministicMoeRoutingMode(TorchDispatchMode):
    """Force MoE routing to deterministically pick experts ``[0..top_k-1]``.

    Intercepts three routing-relevant ops. Any op not in that set is delegated
    unchanged. Each handler also falls through to the original op if its
    arguments do not match the expected schema -- never silently corrupt a
    call that happens to share a name but not a contract.
    """

    def __torch_dispatch__(self, func, types, args: Sequence[Any] = (), kwargs=None):
        kwargs = kwargs or {}

        if func is torch.ops.aten.topk.default:
            return _patch_aten_topk(func, args, kwargs)

        # Custom ops may not be registered in all builds (e.g. when TRT-LLM C++
        # extensions are unavailable). Resolve lazily and only intercept if the
        # op actually exists.
        noaux_tc_op = _try_get_op("trtllm", "noaux_tc_op")
        if noaux_tc_op is not None and func is noaux_tc_op:
            return _patch_noaux_tc_op(func, args, kwargs)

        moe_router_op = _try_get_op("auto_deploy", "torch_moe_router")
        if moe_router_op is not None and func is moe_router_op:
            return _patch_torch_moe_router(func, args, kwargs)

        return func(*args, **kwargs)


# -----------------------------------------------------------------------------
# Op handlers
# -----------------------------------------------------------------------------


def _patch_aten_topk(func, args, kwargs):
    """Replace ``aten.topk`` output with deterministic indices ``[0..k-1]``.

    Schema (PyTorch): ``topk(input, k, dim=-1, largest=True, sorted=True) -> (values, indices)``.

    Returns the same ``(values, indices)`` shape pair the caller expects, with
    ``indices`` always pointing at the first ``k`` positions along ``dim`` and
    ``values`` gathered from the original input at those positions. The fall-
    through guards keep us safe against unexpected ``k`` / ``dim`` / shape
    combinations (e.g. ``k`` larger than the dim size, scalar inputs).
    """
    if not args:
        return func(*args, **kwargs)
    input_t = args[0]
    if not isinstance(input_t, torch.Tensor) or input_t.ndim == 0:
        return func(*args, **kwargs)

    k = args[1] if len(args) > 1 else kwargs.get("k")
    if not isinstance(k, int):
        return func(*args, **kwargs)
    dim = args[2] if len(args) > 2 else kwargs.get("dim", -1)
    if not isinstance(dim, int):
        return func(*args, **kwargs)
    if dim < 0:
        dim = input_t.ndim + dim
    if not (0 <= dim < input_t.ndim):
        return func(*args, **kwargs)
    if k <= 0 or k > input_t.shape[dim]:
        return func(*args, **kwargs)

    # Build indices = arange(k) broadcast to input_t.shape with `dim` replaced by k.
    out_shape = list(input_t.shape)
    out_shape[dim] = k
    view_shape = [1] * input_t.ndim
    view_shape[dim] = k
    indices = (
        torch.arange(k, device=input_t.device, dtype=torch.long)
        .view(view_shape)
        .expand(out_shape)
        .contiguous()
    )
    values = input_t.gather(dim, indices)
    return values, indices


def _patch_noaux_tc_op(func, args, kwargs):
    """Replace ``trtllm.noaux_tc_op`` output with deterministic ``(weights, indices)``.

    Schema observed in callers (e.g. ``modeling_deepseek.py``,
    ``modeling_nemotron_h.py``)::

        noaux_tc_op(
            router_logits,                  # [B, num_experts] -- arg 0
            e_score_correction_bias,        # [num_experts]    -- arg 1
            n_group,                        # int              -- arg 2
            topk_group,                     # int              -- arg 3
            top_k,                          # int              -- arg 4
            routed_scaling_factor,          # float            -- arg 5
        ) -> (topk_weights, topk_indices)   # both [B, top_k]

    We return uniform weights (sum-to-one) and indices ``[0..top_k-1]``. Both
    are simple constants; the downstream MoE compute is then a deterministic
    function of ``hidden_states``.
    """
    if len(args) < 5:
        return func(*args, **kwargs)
    router_logits = args[0]
    top_k = args[4]
    if not (isinstance(router_logits, torch.Tensor) and router_logits.ndim == 2):
        return func(*args, **kwargs)
    if not isinstance(top_k, int) or top_k <= 0 or top_k > router_logits.shape[1]:
        return func(*args, **kwargs)

    B = router_logits.shape[0]
    device = router_logits.device
    dtype = router_logits.dtype
    indices = (
        torch.arange(top_k, device=device, dtype=torch.long)
        .unsqueeze(0)
        .expand(B, top_k)
        .contiguous()
    )
    weights = torch.full((B, top_k), 1.0 / top_k, device=device, dtype=dtype)
    return weights, indices


def _patch_torch_moe_router(func, args, kwargs):
    """Replace ``auto_deploy.torch_moe_router`` output with a monotonic scores tensor.

    Schema (``custom_ops/linear/torch_router.py``)::

        torch_moe_router(
            hidden_states,  # [B, S, H] or [B*S, H] -- arg 0
            weight,         # [E, H]                -- arg 1
            bias,           # [E]                   -- arg 2
            top_k,          # int                   -- arg 3
        ) -> router_scores  # [T, E]

    The op normally returns scattered-top-k softmax scores; the parent block
    then runs ``torch.topk(scores, top_k, ...)`` on them (which would hit
    ``_patch_aten_topk`` above). We bypass the noise by returning a constant
    monotonic-decreasing scores tensor -- positions ``[0..top_k-1]`` get the
    highest values, so any downstream top-k picks them deterministically.
    """
    if len(args) < 2:
        return func(*args, **kwargs)
    hidden_states = args[0]
    weight = args[1]
    if not isinstance(hidden_states, torch.Tensor) or not isinstance(weight, torch.Tensor):
        return func(*args, **kwargs)
    if weight.ndim != 2:
        return func(*args, **kwargs)

    E = weight.shape[0]
    if hidden_states.ndim == 3:
        T = hidden_states.shape[0] * hidden_states.shape[1]
    elif hidden_states.ndim == 2:
        T = hidden_states.shape[0]
    else:
        return func(*args, **kwargs)

    device = hidden_states.device
    dtype = hidden_states.dtype
    # Monotonic decreasing along the expert dim: position 0 highest, E-1 lowest.
    base = torch.arange(E, 0, -1, dtype=torch.float32, device=device) / float(E)
    scores = base.to(dtype).unsqueeze(0).expand(T, E).contiguous()
    return scores


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


def _try_get_op(namespace: str, name: str):
    """Resolve ``torch.ops.<namespace>.<name>.default`` safely.

    Returns ``None`` if the namespace/op is not registered in the running
    build, so the dispatch mode no-ops gracefully on environments that lack
    the C++ extensions (e.g. local development on a build without
    ``libtensorrt_llm``).
    """
    ns = getattr(torch.ops, namespace, None)
    if ns is None:
        return None
    op = getattr(ns, name, None)
    if op is None:
        return None
    return getattr(op, "default", None)
