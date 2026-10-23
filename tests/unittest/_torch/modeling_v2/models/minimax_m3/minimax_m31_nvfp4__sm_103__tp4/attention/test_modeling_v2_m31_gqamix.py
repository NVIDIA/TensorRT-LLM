# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GQAMIX: a mixed step's sparse generation rows (MsaDecodeFmha) through ops/gqamix_sparse_decode.

The copy must be the in-tree kernel module except for the two decode kernels' specialization (no row count, no slice
alignment) and the one import it cannot make relatively, and must write the in-tree kernel's output bit for bit on a
mixed step's [num_ctx:] slices. The shared MsaDecodeFmha hook must change nothing while it is unset.
"""

import inspect
import os
import subprocess
import sys
import types
from pathlib import Path

import pytest
import torch

import tensorrt_llm._torch.attention.backends.sparse.minimax_m3.kernels.triton_sparse_decode as in_tree
from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m31_nvfp4__sm_103__tp4 import (
    modeling as m,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m31_nvfp4__sm_103__tp4.ops import (
    gqamix_sparse_decode as copy,
)
from tensorrt_llm._torch.attention.backends.fmha.msa_decode import MsaDecodeFmha

HEADS = 16  # query heads per rank at tp4, one KV head
D, PAGE, TOPK, QLEN, SF_VEC = 128, 128, 16, 4, 16

_JIT_OLD = '@triton.jit(do_not_specialize=["decode_query_len"])'
_JIT_NEW = (
    "@triton.jit(\n"
    '    do_not_specialize=["decode_query_len", "total_q"],\n'
    '    do_not_specialize_on_alignment=["block_table_ptr", "seq_lens"],\n'
    ")"
)
_IMPORT_OLD = "from .msa_utils import check_decode_span_shape"
_IMPORT_NEW = (
    "from tensorrt_llm._torch.attention.backends.sparse.minimax_m3.kernels.msa_utils import (\n"
    "    check_decode_span_shape,\n"
    ")"
)


def _code_lines(text: str) -> list:
    return [line for line in text.splitlines() if line.strip()]


def test_the_copy_is_the_in_tree_module_but_for_its_specialization() -> None:
    """Line for line (blank lines aside: the import sorts into a different block) the in-tree module."""
    ours = Path(copy.__file__).read_text()
    theirs = Path(in_tree.__file__).read_text()
    assert (
        ours.count(_JIT_NEW) == 2
        and ours.count(_IMPORT_NEW) == 1
        and theirs.count(_IMPORT_OLD) == 1
    )
    ours = ours.replace(_JIT_NEW, _JIT_OLD).replace(_IMPORT_NEW, "")
    assert _code_lines(ours) == _code_lines(theirs.replace(_IMPORT_OLD, "")), (
        "ops/gqamix_sparse_decode.py must stay a copy of the in-tree triton_sparse_decode.py: re-copy it"
    )


def test_register_sets_the_hook_once_and_only_where_it_exists() -> None:
    lib = types.SimpleNamespace(sparse_decode_hook=None)
    other_lib = types.SimpleNamespace()  # an FMHA library without the hook
    manager = types.SimpleNamespace(fmha_libs=[other_lib, lib])
    m._gqamix_register(types.SimpleNamespace(_fmha_manager=manager))
    assert lib.sparse_decode_hook is m._gqamix_decode
    assert not hasattr(other_lib, "sparse_decode_hook")
    mine = object()
    lib.sparse_decode_hook = mine
    m._gqamix_register(types.SimpleNamespace(_fmha_manager=manager))
    assert lib.sparse_decode_hook is mine
    m._gqamix_register(types.SimpleNamespace())  # a backend without an FMHA manager


def test_the_kill_switch_leaves_the_pre_hook_decode() -> None:
    """TRTLLM_M31_DISABLE_GQAMIX=1: GQAMIX is off, so no layer registers the hook and MsaDecodeFmha runs the in-tree
    decode (whose output the copy matches bit for bit below)."""
    target = "tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m31_nvfp4__sm_103__tp4"
    probe = f"import {target}.modeling as m; print(m.GQAMIX)"
    env = dict(os.environ, TRTLLM_M31_DISABLE_GQAMIX="1")
    out = subprocess.run(
        [sys.executable, "-c", probe], env=env, capture_output=True, text=True, check=True
    )
    assert out.stdout.strip().splitlines()[-1] == "False"
    source = inspect.getsource(m.PrefillTarget)
    assert "if GQAMIX and sparse:\n            _gqamix_register(core.attn[i])" in source
    assert source.count("_gqamix_register(") == 1


class _Attn:
    is_mla_enable = False
    kv_lora_rank = None
    v_head_dim = None
    head_dim = D
    num_heads = HEADS
    layer_idx = 3
    q_scaling = 1.0


def _params(attn, num_tokens):
    return types.SimpleNamespace(
        attn=attn,
        num_tokens=num_tokens,
        meta=types.SimpleNamespace(kv_cache_manager=object()),
        fwd=types.SimpleNamespace(kv_scale_quant_orig=None),
        query_input=torch.zeros(num_tokens, HEADS * D),
        output=torch.zeros(num_tokens, HEADS * D),
        token_offset=2,
        input_seq_length=QLEN,
    )


@pytest.mark.parametrize("hooked", [False, True], ids=["unset", "registered"])
def test_the_shared_decode_hook_is_inert_unset(monkeypatch, hooked) -> None:
    from tensorrt_llm._torch.attention.backends.sparse.minimax_m3.kernels import msa_utils

    calls = []
    monkeypatch.setattr(msa_utils, "msa_paged_kv", lambda manager, layer: ("k", "v"))
    monkeypatch.setattr(
        in_tree, "minimax_m3_sparse_attn_decode", lambda *a, **k: calls.append(("in_tree", a, k))
    )
    attn = _Attn()
    fmha = MsaDecodeFmha(attn)
    assert fmha.sparse_decode_hook is None
    if hooked:
        fmha.sparse_decode_hook = lambda *a, **k: calls.append(("hook", a, k))
    indexes = torch.zeros(10, 1, TOPK, dtype=torch.int32)
    fmha._run_sparse(_params(attn, QLEN), indexes, "bt", "sl")
    assert [c[0] for c in calls] == ["hook" if hooked else "in_tree"]
    _, args, kwargs = calls[0]
    assert args[1:3] == ("k", "v") and args[4:] == ("bt", "sl")
    assert args[3].shape == (1, QLEN, TOPK)
    assert kwargs["decode_query_len"] == QLEN and kwargs["sm_scale"] == D**-0.5


def _mixed_step(nreq, seq_len, row_first, seed):
    """A mixed step's generation rows: NVFP4 caches, FP8 q, and block-table / length slices past row_first."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    nb = seq_len // PAGE
    pages = (nreq + row_first) * nb
    block_table = torch.randperm(pages, device="cuda", generator=g).to(torch.int32).view(-1, nb)
    seq_lens = torch.full((nreq + row_first,), seq_len, device="cuda", dtype=torch.int32)
    total_q = nreq * QLEN
    q = torch.randn(total_q, HEADS, D, device="cuda", generator=g).to(torch.float8_e4m3fn)
    kv = [
        torch.randint(
            0, 256, (pages, 1, PAGE, D // 2), device="cuda", generator=g, dtype=torch.uint8
        )
        for _ in "kv"
    ]
    sc = [
        (torch.rand(pages, 1, PAGE, D // SF_VEC, device="cuda", generator=g) + 0.5).to(
            torch.float8_e4m3fn
        )
        for _ in "kv"
    ]
    topk = torch.stack(
        [torch.randperm(nb, device="cuda", generator=g)[:TOPK] for _ in range(total_q)]
    )
    args = (q, kv[0], kv[1], topk.to(torch.int32).view(total_q, 1, TOPK).permute(1, 0, 2))
    args += (block_table[row_first:], seq_lens[row_first:])
    kwargs = dict(
        sm_scale=D**-0.5,
        decode_query_len=QLEN,
        k_block_scale=sc[0],
        v_block_scale=sc[1],
        k_global_scale=torch.tensor([0.02], device="cuda"),
        v_global_scale=torch.tensor([0.03], device="cuda"),
    )
    return args, kwargs, total_q


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
@pytest.mark.parametrize("nreq,row_first", [(1, 0), (1, 1), (3, 2), (5, 3), (8, 1)])
@pytest.mark.parametrize("seq_len", [16384, 131072])
def test_the_copy_writes_the_in_tree_output_on_a_mixed_step(nreq, row_first, seq_len) -> None:
    args, kwargs, total_q = _mixed_step(nreq, seq_len, row_first, 31 * nreq + row_first)
    ref = torch.empty(total_q, HEADS, D, device="cuda", dtype=torch.bfloat16)
    out = torch.empty_like(ref)
    in_tree.minimax_m3_sparse_attn_decode(*args, output=ref, **kwargs)
    copy.minimax_m3_sparse_attn_decode(*args, output=out, **kwargs)
    torch.cuda.synchronize()
    assert torch.isfinite(ref.float()).all()
    assert torch.equal(out.view(torch.int16), ref.view(torch.int16))
