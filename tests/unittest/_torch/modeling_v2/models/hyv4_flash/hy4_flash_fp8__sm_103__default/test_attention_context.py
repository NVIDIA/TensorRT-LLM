# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A prefill's K/V spans each request's cached prefix, not only the step's rows (CPU only)."""

import torch
from _target import ContextStep, mla_layer, requires_checkpoint

from tensorrt_llm._torch.modeling_v2.models.hyv4_flash.hy4_flash_fp8__sm_103__default import (  # noqa: E501
    modeling as M,
)


class _Recorded:
    """The layer's context kernels replaced by recorders returning rows numbered by position."""

    def __init__(self, layer) -> None:
        self.width = layer.geom.heads_local * (layer.d.nope + layer.d.v_dim)
        self.append = None
        self.gather = None
        self.attn = None
        self.k_pe = None
        self._saved: dict = {}

    def _append(self, *args, **kwargs):
        self.append = args

    def _load(self, **kwargs):
        self.gather = kwargs
        rows = int(kwargs["num_ctx_kv_tokens"])
        dtype = kwargs["out_dtype"]
        ckv = torch.arange(rows * kwargs["kv_lora_rank"], dtype=dtype).view(
            rows, kwargs["kv_lora_rank"]
        )
        self.k_pe = torch.arange(rows * kwargs["qk_rope_head_dim"], dtype=dtype).view(
            rows, kwargs["qk_rope_head_dim"]
        )
        return ckv, self.k_pe

    def _mm(self, a, _b):
        rows = a.shape[0]
        return torch.arange(rows * self.width, dtype=a.dtype).view(rows, self.width)

    def _attention(self, **kwargs):
        self.attn = kwargs

    def __enter__(self) -> "_Recorded":
        patched = {
            "mla_rope_append_paged_kv_assign_q": self._append,
            "load_paged_kv_cache_for_mla": self._load,
            "cublas_mm": self._mm,
            "thop_attention": self._attention,
        }
        for name, fn in patched.items():
            self._saved[name] = getattr(M, name)
            setattr(M, name, fn)
        return self

    def __exit__(self, *exc) -> None:
        for name, value in self._saved.items():
            setattr(M, name, value)


@requires_checkpoint
def test_a_context_kv_spans_the_cached_prefix_as_well_as_the_steps_rows():
    layer = mla_layer(0)
    md = ContextStep(cached=(4, 0), new=(6, 6))
    step_rows, total = md.rows, int(md.host_total_kv_lens[0])
    assert (step_rows, total) == (12, 16)

    q = torch.zeros(step_rows, layer.geom.heads_local, layer.d.qk_dim)
    latent = torch.zeros(step_rows, layer.d.kv_lora + layer.d.rope_dim)
    with _Recorded(layer) as rec:
        out = layer.context_rows(q, latent, md, {"rotary_cos_sin": None})

    assert rec.gather["num_ctx_kv_tokens"] == total
    assert rec.gather["cu_ctx_kv_lens"] is md.ctx_kv_indptr
    # The append is this step's rows, at the engine's offsets.
    assert rec.append[0].shape[0] == step_rows
    assert rec.append[1].shape[0] == step_rows
    assert rec.append[3] is md.ctx_cached_token_indptr
    assert rec.append[4] is md.ctx_kv_indptr

    hl, nope, qk_dim = layer.geom.heads_local, layer.d.nope, layer.d.qk_dim
    assert rec.attn["q"].shape[0] == step_rows
    assert out.shape[0] == step_rows
    assert rec.attn["num_ctx_tokens"] == step_rows
    assert rec.attn["host_total_kv_lens"] is md.host_total_kv_lens
    # Every pool row reaches K and V in order: K's no-PE half and V from the packed GEMM, K's
    # rope half from the gather.
    k = rec.attn["k"].view(total, hl, qk_dim)
    packed = torch.arange(total * rec.width, dtype=rec.k_pe.dtype).view(total, rec.width)
    assert torch.equal(k[..., :nope], packed[:, : hl * nope].view(total, hl, nope))
    assert torch.equal(k[..., nope:], rec.k_pe.unsqueeze(1).expand(total, hl, layer.d.rope_dim))
    assert torch.equal(rec.attn["v"], packed[:, hl * nope :])
