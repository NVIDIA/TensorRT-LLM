# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""iHC hyper-connection mapping: the gate projections over this checkpoint's
four residual streams and the collapse of those streams into a layer's input.

A macro entry composed from native torch -- one fp32 ``F.linear``, an
unweighted RMS inverse, the gate sigmoid and the stream-weighted sum -- it
launches no TensorRT-LLM kernel. Both forms the checkpoint uses are the same
expression; they differ only in how many gate rows the projection has and in
the magnitude on the second half:

* connection (``model.layers.N.hc_{attn,mlp}_layer.hc_pre``): ``fn`` ``[2N,
  N*H]``, ``base`` ``[2N]``, ``scale`` ``[2]``. The first ``N`` gate rows map
  the streams into the layer's input (pre); the last ``N`` are the weights the
  caller applies to the block's output (post), which is why they are returned.
* head (``model.hc_head``): ``fn`` ``[N, N*H]``, ``base`` ``[N]``, ``scale``
  ``[1]``. One gate set, collapsed to the single stream ``model.norm`` takes.

The semantics are HuggingFace ``HYV4FlashHyperConnection`` and
``HYV4FlashHyperHead`` (``modeling_hy_v4_flash.py``), with ``N = hc_mult`` and
the pre-GEMM norm ``HYV4FlashUnweightedRMSNorm(eps=config.rms_norm_eps)``::

    flat = streams.flatten(-2).float()  # [..., N*H]
    logits = F.linear(flat, fn) * rsqrt(flat.pow(2).mean(-1, keepdim=True) + norm_eps)
    pre = sigmoid(logits[..., :N] * scale[0] + base[:N]) + eps
    post = post_magnitude * sigmoid(logits[..., N:] * scale[1] + base[N:]) + eps
    out = (pre.unsqueeze(-1) * streams).sum(-2)

and, for the head, ``gates = sigmoid(logits * scale + base) + eps`` with the
same ``out``. Everything from ``flat`` on is fp32; ``out`` is rounded once to
``streams.dtype``, and the gates come back in fp32 because the caller applies
``post`` itself.

Fusion boundary: the pre-GEMM norm, the projection, both gate halves and the
pre-weighted stream collapse. The post-mapping back into the residual streams
(``post.unsqueeze(-1) * block.unsqueeze(-2) + residual``, the decoder layer's
line after the block) is not in: it takes the block's output, which does not
exist yet at this call. The ``eps`` added after the sigmoid is the gate's
``hc_eps``, not the norm's -- the checkpoint carries 1e-6 and 1e-5.

Prior art, and why this is the composition rather than a kernel: vLLM
``vllm/models/hy_v4/nvidia/hc.py`` computes the same math eagerly, and its
``triton_ihc.py`` (``triton_ihc_pre``, adapted there from SGLang PR #36805)
splits the pre half over two kernels and re-reads ``fn`` per token; the
checkpoint's own ``ihc_kernel_backend`` is ``naive``, i.e. this form. Nothing
in ``torch.ops.trtllm``, the catalog or another target's ``ops/`` computes this
hyper-connection: ``glm5_next``'s ``mhc_*`` is the DeepSeek-V4 form (24 mixes
and a Sinkhorn comb) and ``qwen4_exp``'s ``hc_*`` mixers are a different
formula. The torch composition is the fastest thing that exists for it today.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from tensorrt_llm._torch.modeling_v2.catalog._op import Arch, Cell, OpWrapper, true_fp32_matmul


class IhcMapping(OpWrapper):
    """Gate the ``N`` residual streams into one, and return the post weights.

    ``(out, gates) = ihc_mapping(streams, fn, base, scale, post_magnitude,
    eps, norm_eps)``. ``out`` is ``[..., H]`` in ``streams.dtype``, the input to
    the layer's norm; ``gates`` is ``[..., R]`` in fp32, ``R`` being ``N`` (head)
    or ``2N`` (connection). ``post_magnitude`` scales rows ``N:`` only and has
    no effect when ``R == N``; the head ignores it, as HF does.
    """

    #: Driven on the run's sm_103 device only; this target routes sm_103 alone.
    ARCHS = frozenset({Arch.SM_103})

    # The Hy4-Flash geometry: four streams of 3072, so the connection projects
    # to 8 gate rows and the head to 4. `fn_rms` and `stream_sigma` are the
    # checkpoint's own element statistics (85 hc_fn/hc_head_fn tensors, all
    # fp32, rms 0.0130..0.0493; layer-0 attn 0.0133, the head 0.0130), read
    # from the shards. The row counts are the ends this target runs: one decode
    # token, the largest padded decode graph batch, and a 5120-token prefill
    # chunk.
    CELLS: tuple[Cell, ...] = (
        Cell(
            why="connection, decode: 1 row, 4 streams, 8 gate rows",
            spec=dict(
                rows=1,
                streams=4,
                hidden=3072,
                groups=2,
                dtype=torch.bfloat16,
                post_magnitude=2.0,
                eps=1e-6,
                norm_eps=1e-5,
                fn_rms=0.0133,
                stream_sigma=8.0,
            ),
        ),
        Cell(
            why="connection, 64 rows: the largest padded decode graph batch",
            spec=dict(
                rows=64,
                streams=4,
                hidden=3072,
                groups=2,
                dtype=torch.bfloat16,
                post_magnitude=2.0,
                eps=1e-6,
                norm_eps=1e-5,
                fn_rms=0.0133,
                stream_sigma=8.0,
            ),
        ),
        Cell(
            why="connection, prefill: 5120 rows at the model's widest hc_fn",
            spec=dict(
                rows=5120,
                streams=4,
                hidden=3072,
                groups=2,
                dtype=torch.bfloat16,
                post_magnitude=2.0,
                eps=1e-6,
                norm_eps=1e-5,
                fn_rms=0.0493,
                stream_sigma=8.0,
            ),
        ),
        Cell(
            why="head, decode: 1 row, 4 streams, 4 gate rows",
            spec=dict(
                rows=1,
                streams=4,
                hidden=3072,
                groups=1,
                dtype=torch.bfloat16,
                post_magnitude=2.0,
                eps=1e-6,
                norm_eps=1e-5,
                fn_rms=0.0130,
                stream_sigma=8.0,
            ),
        ),
        Cell(
            why="head, 64 rows: the largest padded decode graph batch",
            spec=dict(
                rows=64,
                streams=4,
                hidden=3072,
                groups=1,
                dtype=torch.bfloat16,
                post_magnitude=2.0,
                eps=1e-6,
                norm_eps=1e-5,
                fn_rms=0.0130,
                stream_sigma=8.0,
            ),
        ),
        Cell(
            why="head, prefill: 5120 rows at the model's widest hc_fn",
            spec=dict(
                rows=5120,
                streams=4,
                hidden=3072,
                groups=1,
                dtype=torch.bfloat16,
                post_magnitude=2.0,
                eps=1e-6,
                norm_eps=1e-5,
                fn_rms=0.0493,
                stream_sigma=8.0,
            ),
        ),
    )

    note = """
    Macro entry (tier 4): `__call__` composes native torch and launches no
    TensorRT-LLM kernel. Searched for a fused replacement: `torch.ops.trtllm`
    (no hyper-connection op), the catalog (none), another target's `ops/`
    (`glm5_next` `mhc_*` is the DSv4 24-mix Sinkhorn form, `qwen4_exp` `hc_*`
    are different formulas), vLLM (`hy_v4/nvidia/hc.py` eager is this same
    composition; `triton_ihc.py`'s pre half is two kernels and re-reads the
    weight matrix per token) and SGLang (the PR vLLM adapted). None is both
    this math and faster than cuBLAS plus pointwise at these shapes.

    Precision: this device's torch defaults `matmul.fp32_precision` to `tf32`
    with `allow_tf32=True` (measured here), so an unwrapped fp32 product is
    TF32. Measured at the op's own shape (5120 rows, 8 gate rows, K 12288),
    TF32 moves the projection by 9776 / 1741 ulp -- 30x the test's gate band,
    so dropping the wrapper would fail the cells; at one row the product is a
    gemv, where the policy does not apply (1.7 ulp). `__call__` runs the
    projection under `catalog._op.true_fp32_matmul()`, so the entry's result is
    the same number whatever the ambient policy is, and `reference` accumulates
    in float64 -- an fp32 reference would be the less accurate side.

    Bands, measured on sm_103 (worst cell in parentheses): `out` 2.5 / 0.05 ulp
    (1.652 / 0.01125, connection at 5120 rows); gates 320 / 16 ulp
    (149.38 / 5.619, the same cell's pre half, which runs the model's widest
    `hc_fn` at rms 0.0493). The gate band is wider than torch's fp32 default
    (~53 ulp of these rows), so the test also measures a plausible wrong
    variant -- the pre-GEMM norm left out -- and the band rejects it by more
    than 100x.

    Geometry, measured from the checkpoint's shards: `hc_pre.hc_fn` is
    `[8, 12288]` fp32 with rms 0.0133 (layer-0 attn) and 0.0222 (layer-0 mlp),
    `hc_head.hc_head_fn` `[4, 12288]` fp32 rms 0.0130, across all 85 such
    tensors rms 0.0130..0.0493; `hc_base` `[8]` / `hc_head_base` `[4]` fp32
    (layer-0 attn base rms 1.79, the head's 1.11); `hc_scale` `[2]` fp32
    (1.52 / 0.03 at layer-0 attn, 0.039 / 0.039 at layer-0 mlp), `hc_head_scale`
    `[1]` = 0.158. Config: hc_mult 4, hc_eps 1e-6, hc_magnitude 2.0,
    rms_norm_eps 1e-5, `ihc_kernel_backend` `naive`.

    The gate logits are invariant to the streams' scale -- the pre-GEMM RMS
    norm divides out whatever `flat` is scaled by, up to `norm_eps` -- so a
    cell's `stream_sigma` is not a band axis; the cells use 8.0, not 1.0, so
    that dropping the norm would change the logits by that factor rather than
    leaving them alone.

    Not certified: stream scales other than 8.0; `streams` dtypes other than
    bf16 (fp16/fp32 are accepted and cast back, not measured); a `fn`, `base`
    or `scale` that is not fp32 (a non-fp32 `fn` is loud in `F.linear`, a
    non-fp32 `base`/`scale` would promote silently and lose precision);
    `N != 4` or `H != 3072`; row counts above 5120; `fn` element rms above
    0.0493 (the widest of the checkpoint's 85); `post_magnitude` other than
    2.0 (0.0 is also driven, in the differential test) and `eps`/`norm_eps`
    other than the config's 1e-6/1e-5; the batched `[B, S, N, H]` form, which
    takes the same `flatten(-2)` path but is never driven; the post-mapping
    back into the streams, which is outside this op.
    """

    def __call__(
        self,
        streams: torch.Tensor,
        fn: torch.Tensor,
        base: torch.Tensor,
        scale: torch.Tensor,
        post_magnitude: float,
        eps: float,
        norm_eps: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        streams_per_row = streams.shape[-2]
        with true_fp32_matmul():
            flat = streams.flatten(-2).float()
            inv_rms = torch.rsqrt(flat.pow(2).mean(dim=-1, keepdim=True) + norm_eps)
            logits = F.linear(flat, fn) * inv_rms
        gate_rows = logits.shape[-1]
        if gate_rows == streams_per_row:
            gates = torch.sigmoid(logits * scale + base) + eps
        elif gate_rows == 2 * streams_per_row:
            pre = (
                torch.sigmoid(logits[..., :streams_per_row] * scale[0] + base[:streams_per_row])
                + eps
            )
            post = (
                post_magnitude
                * torch.sigmoid(logits[..., streams_per_row:] * scale[1] + base[streams_per_row:])
                + eps
            )
            gates = torch.cat((pre, post), dim=-1)
        else:
            raise ValueError(
                f"fn has {gate_rows} rows for {streams_per_row} streams: expected "
                f"{streams_per_row} (head) or {2 * streams_per_row} (connection)"
            )
        out = (gates[..., :streams_per_row].unsqueeze(-1) * streams).sum(dim=-2)
        return out.to(streams.dtype), gates

    def reference(
        self,
        streams: torch.Tensor,
        fn: torch.Tensor,
        base: torch.Tensor,
        scale: torch.Tensor,
        post_magnitude: float,
        eps: float,
        norm_eps: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """float64 throughout, rounded once at the end.

        An fp32 accumulation would share the entry's rounding and, on this
        device, would be TF32 on top of it; float64 is the independent oracle.
        """
        streams_per_row = streams.shape[-2]
        flat = streams.double().flatten(-2)
        logits = F.linear(flat, fn.double()) * torch.rsqrt(
            flat.pow(2).mean(dim=-1, keepdim=True) + norm_eps
        )
        gate_rows = logits.shape[-1]
        if gate_rows == streams_per_row:
            gates = torch.sigmoid(logits * scale.double() + base.double()) + eps
        elif gate_rows == 2 * streams_per_row:
            pre = (
                torch.sigmoid(
                    logits[..., :streams_per_row] * scale[0].double()
                    + base[:streams_per_row].double()
                )
                + eps
            )
            post = (
                post_magnitude
                * torch.sigmoid(
                    logits[..., streams_per_row:] * scale[1].double()
                    + base[streams_per_row:].double()
                )
                + eps
            )
            gates = torch.cat((pre, post), dim=-1)
        else:
            raise ValueError(
                f"fn has {gate_rows} rows for {streams_per_row} streams: expected "
                f"{streams_per_row} (head) or {2 * streams_per_row} (connection)"
            )
        out = (gates[..., :streams_per_row].unsqueeze(-1) * streams.double()).sum(dim=-2)
        return out.to(streams.dtype), gates.float()

    # No `is_valid`: every stream layout the op can be handed raises from the
    # arithmetic rather than answering plausibly -- the flattened `[rows, N*H]`
    # form is a 2-D tensor, so `flatten(-2)` collapses it to one row of
    # `rows*N*H` and the projection rejects it; a stream axis that is not `N`
    # with a matching width leaves the gate-row count unequal to both `N` and
    # `2N` and the branch check raises. Both are measured in the test. The
    # base class's documented default is the no-op, so nothing is overridden.


ihc_mapping = IhcMapping()
