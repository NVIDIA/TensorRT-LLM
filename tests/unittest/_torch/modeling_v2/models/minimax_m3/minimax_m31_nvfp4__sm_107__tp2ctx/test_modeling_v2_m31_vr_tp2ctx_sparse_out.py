# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""TP2CTX_SPARSE_OUT (R8, #20629 NVFP4_OUT_IN_PLACE): the NVFP4 sparse prefill writes the attention output in place.

The shared msa_prefill hook is inert unless an attention layer sets ``msa_sparse_out_in_place``; the tp2ctx target sets
it on every layer (kill switch TRTLLM_M31_DISABLE_TP2CTX_SPARSE_OUT), and the caller-owned output is passed only to an
MSA build whose sparse_atten_nvfp4_kv_func accepts ``out`` (otherwise the result is copied as before). CPU.
"""

from pathlib import Path

import torch

import tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3 as _family
from tensorrt_llm._torch.attention.backends.fmha import msa_prefill

_TARGET = Path(_family.__file__).resolve().parent / "minimax_m31_nvfp4__sm_107__tp2ctx"


def test_the_out_argument_is_detected_once_per_function():
    def with_out(q, *, schedule=None, out=None):
        return q

    def without_out(q, *, schedule=None):
        return q

    assert msa_prefill._sparse_func_takes_out(with_out)
    assert not msa_prefill._sparse_func_takes_out(without_out)
    assert msa_prefill._SPARSE_FUNC_TAKES_OUT[with_out] is True


def test_an_output_that_aliases_its_inputs_is_detected():
    qkv = torch.empty(64, 3, 8)
    q, k = qkv[:, 0], qkv[:, 1]
    assert msa_prefill._overlaps(qkv[:, 0], qkv[:, 0])
    assert msa_prefill._overlaps(q, k)  # interleaved views of one buffer share its byte range
    assert not msa_prefill._overlaps(torch.empty(64, 8), torch.empty(64, 8))


def test_the_shared_hook_defaults_to_the_copy_path():
    src = Path(msa_prefill.__file__).read_text()
    assert 'in_place=getattr(attn, "msa_sparse_out_in_place", False),' in src
    assert (
        "write_in_place = in_place and _sparse_func_takes_out(sparse.sparse_atten_nvfp4_kv_func)"
        in src
    )
    assert "    if not write_in_place:\n        out.copy_(result)" in src


def test_tp2ctx_opts_every_layer_in_with_a_kill_switch():
    src = (_TARGET / "modeling.py").read_text()
    assert 'if not m31_disabled("TP2CTX_SPARSE_OUT"):' in src
    assert "layer.self_attn.attn.msa_sparse_out_in_place = True" in src
