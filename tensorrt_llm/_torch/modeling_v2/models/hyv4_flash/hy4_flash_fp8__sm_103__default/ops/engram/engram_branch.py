# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The Engram branch: the gated memory signal Hy4-Flash adds to its residual streams.

``HYV4FlashEngramLayer.forward`` in
``transformers/models/hy_v4_flash/modeling_hy_v4_flash.py`` turns the looked-up n-gram
embeddings into one residual per hyper-connection stream. Per stream ``hc``, with ``H``
the hidden size and every tensor in the hidden dtype::

    key    = RMSNorm(key_projs[hc](embeddings))              # gamma: norm_key[hc]
    query  = RMSNorm(hidden_states[..., hc, :])              # gamma: norm_query[hc]
    g      = (key * query).sum(-1) * H**-0.5
    gate   = sigmoid(sign(g) * sqrt(|g|.clamp_min(1e-6)))
    signal = pre_gamma * RMSNorm(value_proj(embeddings))     # no affine
    out    = gate[..., None] * signal + hc_scales[hc] * signal
    out    = out.masked_fill(~engram_mask[..., None, None], 0.0)

``out`` is ``[..., hc_mult, H]`` and is the *residual*: HF's decoder layer adds it
(``hidden_states = hidden_states + self.engram_layer(...)``), and so does this entry's
caller. vLLM (``models/hy_v4_flash/nvidia/engram.py``) and SGLang
(``kernels/ops/embeddings/hy4_flash_engram.py``) fold that add into their own branch,
which is why their functions return the sum instead.

The four key projections are HF's own per-stream ``F.linear`` calls rather than one
``[rows, 4H] x [4H, E]`` GEMM. The fused form is the same product with the same FLOPs,
but cuBLAS picks its own reduction order at ``N = 4H``, while HF's per-stream calls make
this entry bit-identical to HF's module -- the test transcribes
``HYV4FlashEngramLayer.forward`` with ``nn.Linear``/``nn.RMSNorm`` and gets the same
tensor element for element.

The norm eps is ``torch.finfo(torch.float32).eps``. HF builds all three norm families as
``nn.RMSNorm(size)`` with no ``eps``, and on this device's torch that resolves to the
fp32 eps: on bf16 input at rms 1 down to 1e-3, ``nn.RMSNorm`` matches an fp32
computation with this eps bit for bit, and differs from the same computation with
``finfo(bfloat16).eps`` by 1.5625e-2 at rms 1 and 4.1384 at rms 1e-3, where the eps
dominates. SGLang's combine kernel hardcodes the bf16 eps and so computes a different
norm; vLLM calls the same ``nn.RMSNorm`` modules HF does.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from tensorrt_llm._torch.modeling_v2.catalog._op import Arch, Cell, OpWrapper

#: The eps HF's Engram norms get: ``nn.RMSNorm(size)`` with no ``eps``, which this
#: device's torch resolves to the fp32 eps (measured; see the module docstring).
_NORM_EPS = torch.finfo(torch.float32).eps

#: HF's floor on the gate magnitude, applied before the signed square root.
_GATE_FLOOR = 1e-6


class EngramBranch(OpWrapper):
    """``hidden_states`` ``[..., HC, H]`` and ``embeddings`` ``[..., E]`` -> ``[..., HC, H]``.

    Returns the residual, ``gate * signal + hc_scales * signal`` per stream, zeroed where
    ``engram_mask`` is false; the caller adds it to ``hidden_states``.
    """

    #: Driven on the run's sm_103 device only; this target routes sm_103 alone.
    ARCHS = frozenset({Arch.SM_103})

    # The checkpoint's two Engram-branch extremes, read from the shards (all tensors
    # bf16): layer 31 carries the widest branch (``unified_pre_gamma`` rms 1.882, the
    # four ``unified_per_hc_scales`` rms 1.104 / 0.185 / 0.271 / 0.127) and layer 1 the
    # dimmest (0.534, and 0.0102 / 0.0151 / 0.0103 / 0.0273). The row counts are the
    # ends this target runs: one decode token, a six-token DFly2 verify step, the
    # largest padded decode graph batch, a 512-token chunked-prefill chunk, the widest
    # prefill chunk and the 8192-token ``max_num_tokens`` envelope.
    CELLS: tuple[Cell, ...] = (
        Cell(why="decode: one token at the widest branch", spec=dict(rows=1, layer=31)),
        Cell(why="a six-token DFly2 verify step", spec=dict(rows=6, layer=31)),
        Cell(why="the largest padded decode graph batch", spec=dict(rows=64, layer=31)),
        Cell(why="a 512-token chunked-prefill chunk", spec=dict(rows=512, layer=31)),
        Cell(why="the widest prefill chunk: 5120 tokens", spec=dict(rows=5120, layer=31)),
        Cell(why="the 8192-token max_num_tokens envelope", spec=dict(rows=8192, layer=31)),
        Cell(why="the dimmest branch: layer 1 at 64 tokens", spec=dict(rows=64, layer=1)),
    )

    note = """
    Macro entry (tier 4): `__call__` composes native torch and launches no TensorRT-LLM
    kernel -- four bf16 key `F.linear` calls, one value `F.linear`, nine `F.rms_norm`
    calls, the signed-sqrt gate and the per-stream scale, all in the hidden dtype.
    Searched for a fused replacement: `torch.ops.trtllm` (no Engram branch op), the
    catalog (`norm/flashinfer_rmsnorm` is one norm launch; nothing computes the gate,
    the per-stream scale or the mask), the in-tree Engram implementation
    (`_torch/modules/engram/functional.py` `engram_gate` is the DeepSeek-V4.1 variant:
    it folds both RMS norms into the dot product, has no unified value norm, no
    `pre_gamma`, no per-stream scale and no mask, so it computes a different function;
    `_torch/kernels/blackwell/engram/packed_projection.py` `engram_packed_projection`
    is an MXFP8 CuTe GEMM over a packed WKV weight and this checkpoint's projections
    are bf16), another target's `ops/` (`qwen4_exp`'s PLE is the host-table lookup and
    n-gram embed, not this branch), FlashInfer (rmsnorm and activation kernels only),
    vLLM (`hy_v4_flash/nvidia/engram.py` computes the branch eagerly: the same
    composition, with the signal and the per-stream scale in fp32, the `(gate + scale)`
    fold and the residual add, so its rounding points are not HF's) and SGLang
    (`kernels/ops/embeddings/hy4_flash_engram.py` `_engram_combine_kernel` fuses the
    post-projection half -- norms, dot, gate, scales, mask and the add -- into one
    Triton launch, with the bf16 eps for the key/query norms and the folded
    `(gate + scale) * signal` rounding, so it too is a different function; sourcing a
    kernel that reproduces HF's rounding points is a separate task).

    Geometry and precision, measured from the checkpoint's 148 shards: `hidden_size`
    3072, `hc_mult` 4, `engram_size` 1536 (two orders of 768), 16 hash heads of 96;
    `key_projs.{i}.weight` bf16 `[3072, 1536]` rms 0.0184..0.0233, `.bias` rms
    0.84..1.69; `value_proj.weight` rms 0.0206..0.0234, `.bias` rms 0.229..0.453;
    `norm_key.{i}.weight` and `norm_query.{i}.weight` bf16 `[3072]` rms 0.287..1.260
    (the two are the same vector to three decimals); `unified_pre_gamma` bf16 `[3072]`;
    `unified_per_hc_scales.{i}` bf16 `[3072]`. The cell's `layer` selects the measured
    statistics of that layer's four Engram weights; the test states them.

    The gate is invariant to the residual streams' scale -- the query norm divides it
    out -- so `hidden_states` rms is not a band axis; the test builds the streams at
    rms 8.0 so that dropping the query norm would change the gate by that factor.

    Measured on this run's sm_103 device, over the seven cells above: against a
    transcription of `HYV4FlashEngramLayer.forward` built from `nn.Linear`/`nn.RMSNorm`
    modules the entry is bit-for-bit equal at (layer 31, 1 row), (layer 31, 512 rows)
    and (layer 1, 64 rows); against the float64 reference the cells sit at 3.51..27.15
    element ulp of each row's largest magnitude (worst: 5120 rows) and 1.181..2.527
    relative rms ulp (worst: one row), cosine 0.9999753..0.9999894. The test's band is
    64 element / 4 rms ulp, 2.4x and 1.6x the worst cell. Four computations that are
    not this op sit outside it: dropping the signed square root 348.0 / 30.447 ulp,
    dropping the query norm 1401.2 / 124.583, dropping `hc_scales` 1525.4 / 341.549,
    and normalizing `pre_gamma * value` 127.1 / 119.928 -- the last two separate in
    different metrics, which is why the test's separation is an "either" and its
    measured worst case is 7.6x the rms band (`flat_gate`).

    Not certified: `hidden_states` dtypes other than bf16 (fp16/fp32 would promote or
    raise in the norms and projections); weights other than the checkpoint's bf16
    (an fp32 `pre_gamma`/`hc_scales` would promote the branch to fp32 silently);
    `hidden_size` other than 3072, `hc_mult` other than 4 or `engram_size` other than
    1536; row counts above 8192; a batched `[B, S, HC, H]` form, which takes the same
    `[..., hc, :]` path but is never driven; and a caller that folds the residual add
    into this op, which is outside it.
    """

    def __call__(
        self,
        hidden_states: torch.Tensor,
        embeddings: torch.Tensor,
        key_weights: torch.Tensor,
        key_biases: torch.Tensor,
        key_norm_weights: torch.Tensor,
        query_norm_weights: torch.Tensor,
        value_weight: torch.Tensor,
        value_bias: torch.Tensor,
        pre_gamma: torch.Tensor,
        hc_scales: torch.Tensor,
        engram_mask: torch.Tensor,
    ) -> torch.Tensor:
        streams = hidden_states.shape[-2]
        hidden = hidden_states.shape[-1]
        if streams != key_weights.shape[0]:
            raise ValueError(
                f"hidden_states carries {streams} streams and key_weights "
                f"{key_weights.shape[0]}: the Engram layer has one key projection per stream"
            )
        if engram_mask.shape != hidden_states.shape[:-2]:
            raise ValueError(
                f"engram_mask {tuple(engram_mask.shape)} does not cover the "
                f"{tuple(hidden_states.shape[:-2])} positions of hidden_states"
            )
        gates = []
        for hc in range(streams):
            key = F.linear(embeddings, key_weights[hc], key_biases[hc])
            key = F.rms_norm(key, (hidden,), key_norm_weights[hc], _NORM_EPS)
            query = F.rms_norm(
                hidden_states[..., hc, :], (hidden,), query_norm_weights[hc], _NORM_EPS
            )
            gate = (key * query).sum(dim=-1) * hidden**-0.5
            gate = gate.abs().clamp_min(_GATE_FLOOR).sqrt() * gate.sign()
            gates.append(gate.sigmoid())
        gates = torch.stack(gates, dim=-1).unsqueeze(-1)
        signal = pre_gamma * F.rms_norm(
            F.linear(embeddings, value_weight, value_bias), (hidden,), None, _NORM_EPS
        )
        out = gates * signal.unsqueeze(-2) + hc_scales * signal.unsqueeze(-2)
        return out.masked_fill(~engram_mask[..., None, None], 0.0)

    def reference(
        self,
        hidden_states: torch.Tensor,
        embeddings: torch.Tensor,
        key_weights: torch.Tensor,
        key_biases: torch.Tensor,
        key_norm_weights: torch.Tensor,
        query_norm_weights: torch.Tensor,
        value_weight: torch.Tensor,
        value_bias: torch.Tensor,
        pre_gamma: torch.Tensor,
        hc_scales: torch.Tensor,
        engram_mask: torch.Tensor,
    ) -> torch.Tensor:
        """HF's formula in float64, rounded once to ``hidden_states.dtype`` at the end.

        An fp32 accumulation would share the entry's rounding; float64 is the
        independent oracle. The norms are written out as ``x * rsqrt(mean(x**2) + eps)
        * gamma``, which is what ``F.rms_norm`` computes, with the same eps.
        """
        streams = hidden_states.shape[-2]
        hidden = hidden_states.shape[-1]
        if streams != key_weights.shape[0]:
            raise ValueError(
                f"hidden_states carries {streams} streams and key_weights "
                f"{key_weights.shape[0]}: the Engram layer has one key projection per stream"
            )
        if engram_mask.shape != hidden_states.shape[:-2]:
            raise ValueError(
                f"engram_mask {tuple(engram_mask.shape)} does not cover the "
                f"{tuple(hidden_states.shape[:-2])} positions of hidden_states"
            )
        h = hidden_states.double()
        e = embeddings.double()
        gates = []
        for hc in range(streams):
            key = F.linear(e, key_weights[hc].double(), key_biases[hc].double())
            key = (
                key
                * torch.rsqrt(key.pow(2).mean(dim=-1, keepdim=True) + _NORM_EPS)
                * key_norm_weights[hc].double()
            )
            query = h[..., hc, :]
            query = (
                query
                * torch.rsqrt(query.pow(2).mean(dim=-1, keepdim=True) + _NORM_EPS)
                * query_norm_weights[hc].double()
            )
            gate = (key * query).sum(dim=-1) * hidden**-0.5
            gate = gate.abs().clamp_min(_GATE_FLOOR).sqrt() * gate.sign()
            gates.append(torch.sigmoid(gate))
        gates = torch.stack(gates, dim=-1).unsqueeze(-1)
        value = F.linear(e, value_weight.double(), value_bias.double())
        signal = (
            pre_gamma.double()
            * value
            * torch.rsqrt(value.pow(2).mean(dim=-1, keepdim=True) + _NORM_EPS)
        )
        out = gates * signal.unsqueeze(-2) + hc_scales.double() * signal.unsqueeze(-2)
        return out.masked_fill(~engram_mask[..., None, None], 0.0).to(hidden_states.dtype)

    # No `is_valid`: the two layouts a caller can get wrong -- a stream axis that is not
    # the number of key projections, and a mask that does not cover the hidden states'
    # positions -- raise from the checks above, and every dtype or width mismatch raises
    # from the projections and norms. There is no input the op answers silently wrongly.


engram_branch = EngramBranch()
