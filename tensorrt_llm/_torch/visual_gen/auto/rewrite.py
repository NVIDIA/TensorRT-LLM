# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""FX rewrite pipeline for the auto path.

The first pass rewrites `aten::scaled_dot_product_attention.*` calls to
`torch.ops.visgen_auto.sdpa` (see `ops.py`). The pattern matcher validates
that the SDPA call is non-causal and unmasked — DiT-invariant — and refuses
to rewrite anything else (e.g., an LLM-shaped causal attention with a
`bool` mask). Subsequent passes (QKV-GEMM fusion in ``fusion.py``,
QK-norm+RoPE fusion in ``qk_rope_fusion.py``, FP8/NVFP4-contig insertion,
``_assert_tensor_metadata`` strip) run in canonical order via the
``PassManager`` in ``pass_manager.py``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch
import torch.fx as fx

from tensorrt_llm.logger import logger

from . import ops  # noqa: F401 — registers torch.ops.visgen_auto.sdpa

if TYPE_CHECKING:
    from .adapter import VisGenFamilyAdapter
    from .pass_manager import PassManager
    from .policy import RewritePolicy


_SDPA_TARGETS = (
    torch.ops.aten.scaled_dot_product_attention.default,
    torch.ops.aten._scaled_dot_product_cudnn_attention.default,
    torch.ops.aten._scaled_dot_product_efficient_attention.default,
    torch.ops.aten._scaled_dot_product_flash_attention.default,
)


def _rewrite_sdpa_to_visgen(graph_module: fx.GraphModule, backend: str = "VANILLA") -> int:
    """Replace SDPA call nodes with `torch.ops.visgen_auto.sdpa`.

    DiT invariant: SDPA is called as ``sdpa(q, k, v)`` (or with explicit
    ``scale=`` and/or ``is_causal=False``) — no attn_mask, no dropout,
    non-causal. Calls violating that are skipped (the pass stays
    conservative; an upstream layer can lower them differently later).

    Args:
        graph_module: target GraphModule (mutated in place).
        backend: attention-backend selection written into the inserted
            `visgen_auto.sdpa` node (one of ``"VANILLA"``, ``"TRTLLM"``,
            ``"FA4"``). Pulled from `RewritePolicy.attention_backend`.

    Returns the number of nodes rewritten.
    """
    target_op = torch.ops.visgen_auto.sdpa.default
    n_rewritten = 0
    n_skipped = 0
    g = graph_module.graph

    for node in list(g.nodes):
        if node.op != "call_function" or node.target not in _SDPA_TARGETS:
            continue

        if not _is_dit_compatible_sdpa(node):
            n_skipped += 1
            continue

        q, k, v = node.args[0], node.args[1], node.args[2]
        scale = node.kwargs.get("scale", None)

        new_kwargs: dict = {"backend": backend}
        if scale is not None:
            new_kwargs["scale"] = scale

        # Force BHSD contiguous on q/k/v. Diffusers DiT blocks reach SDPA via
        # `view(B,S,H,D).transpose(1,2)` — a non-contiguous view. Inductor is
        # free to "realize" that view as a contig copy in its codegen (which it
        # does silently at 4-rank Ulysses on PixArt/LTX-Video at S=1024), but
        # the `assert_size_stride` check on the SDPA input still references
        # the captured-time (transpose-view) strides → AssertionError at
        # runtime. Inserting `.contiguous()` explicitly pins both planning and
        # realization to the same layout. No-op when already contig.
        #
        # Use the `call_function(aten.contiguous.default, ...)` form, NOT
        # `call_method("contiguous", ...)`. The qk_rope_fusion pattern matcher
        # walks back from each SDPA's q/k input through `_TRANSPARENT_OPS`
        # (which contains `aten.contiguous.default`) looking for an
        # `aten.rms_norm` ancestor; a `call_method` node breaks the walk and
        # silently disables the QK-rope fusion on every attention site.
        with g.inserting_before(node):
            q_c = g.call_function(torch.ops.aten.contiguous.default, args=(q,))
            k_c = g.call_function(torch.ops.aten.contiguous.default, args=(k,))
            v_c = g.call_function(torch.ops.aten.contiguous.default, args=(v,))
            new_node = g.call_function(
                target_op,
                args=(q_c, k_c, v_c),
                kwargs=new_kwargs,
            )

        # Some SDPA overloads return a tuple ``(out, ...)``; aten default
        # returns the tensor directly. If users of `node` index into it,
        # they'll need to be re-pointed at `new_node` (single-output).
        # The DiT case captured here yields a single-tensor result on every
        # overload we've observed, so we replace uses 1:1.
        node.replace_all_uses_with(new_node)
        g.erase_node(node)
        n_rewritten += 1

    if n_rewritten or n_skipped:
        graph_module.graph.lint()
        graph_module.recompile()
    logger.info(
        f"VisGen-Auto rewrite: SDPA → visgen_auto.sdpa "
        f"(rewritten={n_rewritten}, skipped={n_skipped})"
    )
    return n_rewritten


def _is_dit_compatible_sdpa(node: fx.Node) -> bool:
    """Validate that an SDPA call matches the DiT invariant.

    Accepts: 3-arg call (q, k, v), or with ``scale=`` and/or
    ``is_causal=False``. Rejects everything else.
    """
    if len(node.args) > 3:
        # Positional attn_mask / dropout_p / is_causal / scale.
        # SDPA signature: (q, k, v, attn_mask=None, dropout_p=0.0,
        #                  is_causal=False, scale=None, enable_gqa=False).
        attn_mask = node.args[3] if len(node.args) > 3 else None
        dropout_p = node.args[4] if len(node.args) > 4 else 0.0
        is_causal = node.args[5] if len(node.args) > 5 else False
        if attn_mask is not None:
            return False
        if dropout_p:
            return False
        if is_causal:
            return False

    if node.kwargs:
        if node.kwargs.get("attn_mask") is not None:
            return False
        if node.kwargs.get("dropout_p", 0.0):
            return False
        if node.kwargs.get("is_causal", False):
            return False

    return True


def _ensure_contiguous_for_fp8_quant(graph_module: fx.GraphModule) -> int:
    """Prepend `.contiguous()` to inputs of every FP8/NVFP4 activation-quant
    call in the captured graph.

    The underlying ops (`fp8Op.cpp:e4m3_quantize_helper` for FP8,
    `fp4Quantize.cpp:fp4_quantize` for NVFP4, plus their GEMM downstream
    callers) require contiguous input. `Linear.apply_linear` upstream does
    `input = input.reshape(-1, input.shape[-1])` without forcing contiguity
    — so when the caller-side `hidden_states` is itself non-contiguous
    (e.g. Diffusers' `_pack_latents` returns a `permute().reshape()` chain
    that produces a non-contiguous view at FLUX.2 input dims), the reshape
    is also non-contiguous and the quant op rejects it.

    `.contiguous()` is a cheap no-op when the input is already contiguous,
    so this is safe to apply unconditionally.
    """
    quant_targets: tuple = (
        torch.ops.tensorrt_llm.quantize_e4m3_per_tensor.default,
        torch.ops.tensorrt_llm.static_quantize_e4m3_per_tensor.default,
    )
    # NVFP4 quant ops live under `torch.ops.trtllm` (different namespace from
    # FP8). Guarded by hasattr so the rewrite still compiles on builds without
    # the NVFP4 build flag.
    if hasattr(torch.ops, "trtllm"):
        for opname in ("fp4_quantize", "tunable_fp4_quantize"):
            op = getattr(torch.ops.trtllm, opname, None)
            if op is not None:
                quant_targets = quant_targets + (op.default,)
    n = 0
    g = graph_module.graph
    for node in list(g.nodes):
        if node.op != "call_function" or node.target not in quant_targets:
            continue
        if not node.args:
            continue
        x = node.args[0]
        # Skip if we've already inserted a contiguous() call for this input.
        if (
            isinstance(x, fx.Node)
            and x.op == "call_function"
            and x.target is torch.ops.aten.contiguous.default
        ):
            continue
        with g.inserting_before(node):
            contig = g.call_function(torch.ops.aten.contiguous.default, args=(x,))
        new_args = (contig,) + tuple(node.args[1:])
        node.args = new_args
        n += 1
    if n:
        g.lint()
        graph_module.recompile()
    return n


def _strip_assert_tensor_metadata(graph_module: fx.GraphModule) -> int:
    """Remove all `aten._assert_tensor_metadata` nodes from the captured graph.

    `torch.export` inserts these as runtime guards on tensor dtype / device /
    layout matching what was seen at capture time. They're useful during
    development but routinely fire false positives when the captured graph
    re-runs with values that have the right semantics but differ by a fixed
    detail (e.g., quant scales whose dtype slightly differs at capture vs
    runtime). For our auto-path use cases (replay captured graphs with the
    same shapes), the asserts add no correctness value and block
    legitimate runs.
    """
    n = 0
    g = graph_module.graph
    for node in list(g.nodes):
        if (
            node.op == "call_function"
            and node.target is torch.ops.aten._assert_tensor_metadata.default
        ):
            g.erase_node(node)
            n += 1
    if n:
        g.lint()
        graph_module.recompile()
    return n


def _build_default_pass_manager(policy: "RewritePolicy") -> "PassManager":
    """Construct the default visgen-auto pass pipeline from a `RewritePolicy`.

    Pass order is canonical and load-bearing — qk_rope fusion depends on
    QKV fusion having already produced the same-input narrow markers it
    walks back to. Adapters that need different ordering should override
    `customize_passes` and re-order via the manager primitives.

    Pass names (stable; family adapter hooks reference them):
        sdpa_rewrite          → `_rewrite_sdpa_to_visgen`
        fuse_qkv              → `_fuse_same_input_linears` (if policy.fuse_qkv)
        qk_rope_single        → `fuse_qk_norm_rope` (if enabled)
        qk_rope_dual          → `fuse_qk_norm_rope_dual_stream` (if enabled)
        contig_for_fp8_quant  → `_ensure_contiguous_for_fp8_quant`
        strip_assert_metadata → `_strip_assert_tensor_metadata`
    """
    import os

    from .pass_manager import Pass, PassManager

    pm = PassManager()
    pm.append(
        Pass(
            "sdpa_rewrite",
            lambda gm: _rewrite_sdpa_to_visgen(gm, backend=policy.attention_backend),
        )
    )
    if policy.fuse_qkv:
        from .fusion import _fuse_same_input_linears

        pm.append(Pass("fuse_qkv", _fuse_same_input_linears))
        qk_rope_enabled = policy.fuse_qk_rope and not os.environ.get("VISGEN_AUTO_DISABLE_QKROPE")
        if qk_rope_enabled:
            from .qk_rope_fusion import fuse_qk_norm_rope, fuse_qk_norm_rope_dual_stream

            pm.append(Pass("qk_rope_single", fuse_qk_norm_rope))
            pm.append(Pass("qk_rope_dual", fuse_qk_norm_rope_dual_stream))
    pm.append(Pass("contig_for_fp8_quant", _ensure_contiguous_for_fp8_quant))
    # `_strip_assert_tensor_metadata` is always last — export-time shape/dtype
    # guards spuriously fail at runtime on FP8/NVFP4 graphs, and DCE can't
    # collapse the old qk_rope chains while these asserts hold them alive.
    pm.append(Pass("strip_assert_metadata", _strip_assert_tensor_metadata))
    return pm


def apply_rewrites(
    graph_module: fx.GraphModule,
    policy: "RewritePolicy",
    adapter: "Optional[VisGenFamilyAdapter]" = None,
) -> fx.GraphModule:
    """Apply the rewrite pipeline declared by `policy`.

    Build a default `PassManager` from the policy, give `adapter` a chance
    to splice in family-specific passes via `customize_passes(pm)`, then
    run. Returning the same GraphModule (mutated in place).

    `adapter` is optional for backward compatibility with callers that
    only have a `RewritePolicy` (e.g. existing tests). When absent, the
    pipeline is exactly what `RewritePolicy` declares — no family hooks.
    """
    pm = _build_default_pass_manager(policy)
    if adapter is not None:
        adapter.customize_passes(pm)
    pm.run(graph_module)
    return graph_module
