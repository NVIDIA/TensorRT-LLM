# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The absorbed query's head layout on the target's own scratch, per kernel that reads it.

A rank holds 16 of the model's 64 query heads at tep4 (four ranks) and 8 at tep8 (eight ranks
over two nodes of four). FlashMLA's sm100 sparse prefill takes 64 or 128 query heads, so the
prefill target keeps the absorbed query in a `[rows, 64, 576]` bf16 scratch, zeroed once per
step, and writes only its rank's heads 0..h-1 (the absorption BMM into the 512-wide head, the
rope tail written or zeroed by the SSMax op, the owner's rope op in place). The generation
rows' sparse decode, `cute_dsl_ds_mla_sparse_decode`, runs the rank's h heads unpadded: the
decode target's scratch is `[rows, h, 576]`, every element written by every layer (the SSMax
op's tail write included), and the prefill target's generation rows hand it the 64-head
scratch, of which it reads heads 0..h-1. Both kernels write only those h heads of their output
into the compact `[rows, h, 512]` latent output the V-B BMM reads. That the kernels leave the
live heads unchanged under zero dead heads, and read h of a wider query, is their own
certification (`flash_mla_sparse_prefill`'s and `cute_dsl_ds_mla_sparse_decode`'s cells); this
checks the layout around them, through `PrefillTarget._scratch` and `DecodeTarget._scratch`,
at both world sizes:

* the prefill scratch's query is 64 heads of 576, bf16, every byte zero (+0.0, not -0.0); the
  decode scratch's is the rank's h heads of 576, bf16, contiguous;
* the rank's heads are a view of heads 0..h-1 with contiguous heads, the layout the rope op and
  the kernels address, so writing them never reaches heads h..63;
* the latent output is h heads wide, a NoPE owner's latent rows are 576 wide and zero (the
  fused glue's; the unfused glue, the core's `unfused_glue` (`_UNFUSED_GLUE`), builds them per
  call), and no FlashMLA scheduler rides on the step.

`_GEN_ATTN_FLASH_MLA` (the core's `gen_attn_flash_mla`) puts the generation rows back on
FlashMLA's 64-head decode for A/B verification; its layout (the decode scratch at 64 heads,
zeroed, one scheduler shared by the step's layers, and a latent output only for the unfused
glue: the fused glue reads FlashMLA's live heads in place) is checked too, under both glues, and
that the shipped default is the native kernel. Neither switch depends on max_seq_len.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.mistral_large4.mistral_large_4_nvfp4__sm_103__tep4 import (  # noqa: E501
    modeling as M,
)

pytestmark = pytest.mark.cpu_only

DEV = "cpu"
KV_LORA, ROPE, V_DIM = 512, 64, 128
LAT = KV_LORA + ROPE
#: World size -> the query heads a rank holds: the model's 64 over the ranks.
HEADS = {4: 16, 8: 8}
GLUES = pytest.mark.parametrize("unfused_glue", [False, True], ids=["fused", "unfused"])


def _core(heads: int, flash_mla: bool, unfused_glue: bool) -> SimpleNamespace:
    """The widths the targets' scratch reads off the core, with `heads` of the 64 on the rank,
    its generation rows' kernel (`flash_mla`: `_GEN_ATTN_FLASH_MLA`) and its glue
    (`unfused_glue`: `_UNFUSED_GLUE`)."""
    return SimpleNamespace(
        heads=heads,
        v_dim=V_DIM,
        lat_dim=LAT,
        kv_lora=KV_LORA,
        rope_dim=ROPE,
        gen_attn_flash_mla=flash_mla,
        unfused_glue=unfused_glue,
    )


def _target(cls, heads: int, flash_mla: bool, unfused_glue: bool):
    target = object.__new__(cls)
    target.core = _core(heads, flash_mla, unfused_glue)
    return target


def _check_scratch(s, rows: int, heads: int, width: int, lat_out: bool, unfused_glue: bool) -> None:
    assert s.fused_q.dtype == torch.bfloat16
    assert tuple(s.fused_q.shape) == (rows, width, LAT)
    if width > heads:
        assert not s.fused_q.view(torch.int16).any(), "a head is not +0.0"
    if lat_out:
        assert tuple(s.lat_out.shape) == (rows, heads, KV_LORA)
    else:
        assert s.lat_out is None, "decode reads FlashMLA's output in place"
    assert tuple(s.attn_out.shape) == (rows, heads * V_DIM)
    if unfused_glue:
        assert s.nope_lat is None, "the unfused glue builds a NoPE owner's latent per call"
    else:
        # A NoPE owner's latent rows: RMSNorm(c_kv) lands in the first 512 columns, the rope
        # key stays +0.0 for the step.
        assert s.nope_lat.dtype == torch.bfloat16 and tuple(s.nope_lat.shape) == (rows, LAT)
        assert not s.nope_lat.view(torch.int16).any(), "a NoPE owner's rope key is not +0.0"

    # The rank's heads as the attention block takes them: heads 0..h-1, contiguous heads.
    fq = s.fused_q[:, :heads]
    assert fq.stride(-1) == 1 and fq.stride(1) == LAT
    assert fq.data_ptr() == s.fused_q.data_ptr()
    g = torch.Generator().manual_seed(rows)
    fq[..., :KV_LORA].copy_(torch.randn(rows, heads, KV_LORA, generator=g))
    fq[..., KV_LORA:].copy_(torch.randn(rows, heads, ROPE, generator=g))
    if width > heads:
        assert not s.fused_q[:, heads:].view(torch.int16).any(), "a dead head was written"
    else:
        assert s.fused_q.is_contiguous()


def test_the_native_decode_is_the_shipped_default():
    assert M._GEN_ATTN_FLASH_MLA is False
    assert M._FLASH_MLA_HEADS == 64


@GLUES
@pytest.mark.parametrize("world", sorted(HEADS))
@pytest.mark.parametrize("rows", [1, 2, 7, 16])
def test_decode_scratch(rows, world, unfused_glue):
    heads = HEADS[world]
    s = _target(M.DecodeTarget, heads, False, unfused_glue)._scratch(rows, torch.bfloat16, DEV)
    _check_scratch(s, rows, heads, heads, lat_out=True, unfused_glue=unfused_glue)
    assert s.sched is None


@GLUES
@pytest.mark.parametrize("world", sorted(HEADS))
@pytest.mark.parametrize("rows", [1, 16])
def test_decode_scratch_under_flash_mla(rows, world, unfused_glue):
    heads = HEADS[world]
    s = _target(M.DecodeTarget, heads, True, unfused_glue)._scratch(rows, torch.bfloat16, DEV)
    _check_scratch(
        s, rows, heads, M._FLASH_MLA_HEADS, lat_out=unfused_glue, unfused_glue=unfused_glue
    )
    assert s.sched is not None


@GLUES
@pytest.mark.parametrize("flash_mla", [False, True])
@pytest.mark.parametrize("world", sorted(HEADS))
@pytest.mark.parametrize("tc, gen", [(5, 0), (31, 1), (1031, 15), (2048, 16)])
def test_prefill_scratch(tc, gen, world, flash_mla, unfused_glue):
    rows = tc + gen
    heads = HEADS[world]
    target = _target(M.PrefillTarget, heads, flash_mla, unfused_glue)
    s = target._scratch(rows, torch.bfloat16, DEV)
    _check_scratch(s, rows, heads, M._FLASH_MLA_HEADS, lat_out=True, unfused_glue=unfused_glue)
    # The context and generation rows are row slices of the one buffer: the dead heads of
    # either slice stay zero after the other is written.
    assert s.fused_q[:tc].data_ptr() == s.fused_q.data_ptr()
    assert (s.sched is not None) == flash_mla
