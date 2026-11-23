# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""What a layer's prefill hands the MLA context entry.

CPU only: the geometry comes from the checkpoint's `config.json` and the entries
are called with plain tensors, so no weight and no device are read.

A context step's K/V is not the step's own rows. The gather the layer calls
(`load_paged_kv_cache_for_mla`) copies positions `[0, L_s)` of every context
sequence, so a request whose prefix is already in the pool -- the second chunk of
a chunked prefill, or a block `enable_block_reuse` lent from an earlier request --
contributes its cached rows as well as the rows this step appends. The engine's
`host_total_kv_lens[0]` is that total and `ctx_kv_indptr` the same quantity as a
per-request cumsum, while the query, the query's rope output and the entry's
output hold this step's rows only.

The two counts coincide when nothing is cached, which is why a first prefill
chunk cannot tell them apart. Sized by the step's rows, the gather is told to
write fewer rows than `cu_ctx_kv_lens` sums to, and the K/V the entry receives
stops short of the cached prefix -- rows of the pool the step's requests own and
must attend over.
"""

import torch

# The checkpoint stand-in this directory's tests share; pytest puts the test
# file's own directory on the path.
from _target import ContextStep, mla_layer, requires_checkpoint

from tensorrt_llm._torch.modeling_v2.models.hyv4_flash.hy4_flash_fp8__sm_103__default import (  # noqa: E501
    modeling as M,
)

#: The kernels one context call reaches, replaced while a recorder is installed.
_PATCHED = (
    "mla_rope_append_paged_kv_assign_q",
    "load_paged_kv_cache_for_mla",
    "cublas_mm",
    "thop_attention",
)


class _Recorded:
    """The layer's context call with its four kernels replaced by recorders.

    The layer is handed tensors and metadata only, so each kernel can be swapped
    for one that keeps what it was given and returns rows numbered by their pool
    position -- so a pool row that never reaches K or V is visible in an
    assertion, not just in a row count.
    """

    def __init__(self, layer) -> None:
        self.layer = layer
        self.width = layer.geom.heads_local * (layer.d.nope + layer.d.v_dim)
        self.append = None
        self.gather = None
        self.attn = None
        self.ckv = None
        self.k_pe = None
        self._saved: dict = {}

    def _append(self, *args, **kwargs):
        self.append = args
        return None

    def _load(self, **kwargs):
        self.gather = kwargs
        rows = int(kwargs["num_ctx_kv_tokens"])
        dtype = kwargs["out_dtype"]
        self.ckv = torch.arange(rows * kwargs["kv_lora_rank"], dtype=dtype).view(
            rows, kwargs["kv_lora_rank"]
        )
        self.k_pe = torch.arange(rows * kwargs["qk_rope_head_dim"], dtype=dtype).view(
            rows, kwargs["qk_rope_head_dim"]
        )
        return self.ckv, self.k_pe

    def _mm(self, a, _b):
        rows = a.shape[0]
        return torch.arange(rows * self.width, dtype=a.dtype).view(rows, self.width)

    def _attention(self, **kwargs):
        self.attn = kwargs
        return None

    def packed(self, rows: int) -> torch.Tensor:
        """The buffer the replaced GEMM returns: `rows` rows, numbered by position."""
        return torch.arange(rows * self.width, dtype=self.ckv.dtype).view(rows, self.width)

    def __enter__(self) -> "_Recorded":
        for name in _PATCHED:
            self._saved[name] = getattr(M, name)
        M.mla_rope_append_paged_kv_assign_q = self._append
        M.load_paged_kv_cache_for_mla = self._load
        M.cublas_mm = self._mm
        M.thop_attention = self._attention
        return self

    def __exit__(self, *exc) -> None:
        for name, value in self._saved.items():
            setattr(M, name, value)


def _context_call(layer, md):
    """One prefill call with this step's rows as queries and a recorder installed."""
    step_rows = md.rows
    q = torch.zeros(step_rows, layer.geom.heads_local, layer.d.qk_dim)
    latent = torch.zeros(step_rows, layer.d.kv_lora + layer.d.rope_dim)
    with _Recorded(layer) as rec:
        out = layer.context_rows(q, latent, md, {"rotary_cos_sin": None})
    return rec, out


@requires_checkpoint
def test_a_context_kv_spans_the_cached_prefix_as_well_as_the_steps_rows():
    """Request 0's four cached rows are among the context's K/V rows, not beyond them."""
    layer = mla_layer(0)
    md = ContextStep(cached=(4, 0), new=(6, 6))

    step_rows, total = md.rows, int(md.host_total_kv_lens[0])
    assert (step_rows, total, md.num_contexts) == (12, 16, 2)
    assert int(md.ctx_kv_indptr[-1]) == total
    assert int(md.ctx_cached_token_indptr[-1]) == 4

    rec, out = _context_call(layer, md)

    # The gather is asked for every row the contexts own, with the same lengths
    # the engine's cumsum names.
    assert rec.gather["num_ctx_kv_tokens"] == total
    assert rec.gather["num_contexts"] == md.num_contexts
    assert rec.gather["cu_ctx_kv_lens"] is md.ctx_kv_indptr
    assert int(rec.gather["cu_ctx_kv_lens"][-1]) == total

    # The append prologue is still this step's rows, at the engine's offsets.
    assert rec.append[0].shape[0] == step_rows
    assert rec.append[1].shape[0] == step_rows
    assert rec.append[2] == md.num_contexts
    assert rec.append[3] is md.ctx_cached_token_indptr
    assert rec.append[4] is md.ctx_kv_indptr

    # K and V carry the whole context; the query and the output stay this step's.
    hl, nope, qk_dim = layer.geom.heads_local, layer.d.nope, layer.d.qk_dim
    assert rec.attn["q"].shape[0] == step_rows
    assert rec.attn["output"].shape[0] == step_rows
    assert out.shape[0] == step_rows
    assert rec.attn["num_ctx_tokens"] == step_rows
    assert rec.attn["host_total_kv_lens"] is md.host_total_kv_lens
    assert rec.attn["k"].shape == (total, hl * qk_dim)
    assert rec.attn["v"].shape == (total, hl * layer.d.v_dim)

    # Every pool row reached K and V, in order, and both are the gather's own
    # rows: K's no-PE half and V out of the packed buffer, K's rope half direct.
    k = rec.attn["k"].view(total, hl, qk_dim)
    packed = rec.packed(total)
    assert torch.equal(k[..., :nope], packed[:, : hl * nope].view(total, hl, nope))
    assert torch.equal(k[..., nope:], rec.k_pe.unsqueeze(1).expand(total, hl, layer.d.rope_dim))
    assert torch.equal(rec.attn["v"], packed[:, hl * nope :])


@requires_checkpoint
def test_a_first_chunk_is_the_step_whose_two_row_counts_agree():
    """With nothing cached the context's rows are exactly the step's: the case that hid this."""
    layer = mla_layer(0)
    md = ContextStep(cached=(0, 0), new=(6, 6))

    assert md.rows == int(md.host_total_kv_lens[0]) == 12

    rec, out = _context_call(layer, md)

    assert rec.gather["num_ctx_kv_tokens"] == md.rows
    assert rec.attn["k"].shape[0] == md.rows
    assert rec.attn["v"].shape[0] == md.rows
    assert rec.attn["host_total_kv_lens"] is md.host_total_kv_lens
    assert out.shape[0] == md.rows
