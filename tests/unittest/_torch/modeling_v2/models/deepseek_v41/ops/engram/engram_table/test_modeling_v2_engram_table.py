# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Test for the host-resident engram table: a lookup through
`ShardedFp8MultiHeadEmbedding` against a dense dequantization.

The table lives in pinned host memory and is read by the gather kernel through UVA; its bucket offsets are a buffer on
the device. Row shards (heads not divisible by the table's ranks) return the rows the rank owns and zero elsewhere; head
shards return only the rank's own heads.
"""

import pytest
import torch
from _v41_util import DEV

__extra_import_path__ = ["../../../../.."]

from tensorrt_llm._torch.modeling_v2.models.deepseek_v41.ops.engram.engram_table.table import (
    ShardedFp8MultiHeadEmbedding,
)

pytestmark = pytest.mark.skipif(
    not (torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 10),
    reason="the engram lookup is a Blackwell (sm_10x) kernel",
)

DIM, BLOCK = 128, 32
BUCKETS = [37, 41, 43, 47]  # one bucket per head


def _checkpoint_rows():
    gen = torch.Generator(device="cpu").manual_seed(7)
    rows = sum(BUCKETS)
    weight = (torch.randn(rows, DIM, generator=gen) * 4).to(torch.float8_e4m3fn)
    scale = torch.randint(118, 134, (rows, DIM // BLOCK), generator=gen, dtype=torch.uint8).view(
        torch.float8_e8m0fnu
    )
    return weight, scale


def _dense(weight, scale):
    values = weight.float().unflatten(-1, (-1, BLOCK))
    exponent = (scale.view(torch.uint8).float() - 127).exp2().unsqueeze(-1)
    return (values * exponent).flatten(-2).to(torch.bfloat16)


# (tp_size, tp_rank) -> (owned heads, owned global rows), stated from BUCKETS (168 rows) rather than read from the
# module: 2 ranks shard the 4 heads (37 + 41 = 78 rows below head 2), 3 ranks shard rows (ceil(168 / 3) = 56 each).
_OWNERSHIP = {
    (1, 0): ((0, 4), (0, 168)),
    (2, 0): ((0, 2), (0, 78)),
    (2, 1): ((2, 4), (78, 168)),
    (3, 0): ((0, 4), (0, 56)),
    (3, 2): ((0, 4), (112, 168)),
}


@pytest.mark.parametrize("tp_size,tp_rank", list(_OWNERSHIP), ids=str)
def test_lookup_matches_dense_dequantization(tp_size, tp_rank):
    weight, scale = _checkpoint_rows()
    table = ShardedFp8MultiHeadEmbedding(
        list_of_N=BUCKETS,
        D=DIM,
        block_size=BLOCK,
        dtype=torch.bfloat16,
        tp_size=tp_size,
        tp_rank=tp_rank,
    )
    (head_lo, head_hi), (row_lo, row_hi) = _OWNERSHIP[(tp_size, tp_rank)]
    assert table.shard_heads == (tp_size == 2)
    assert (table.head_start, table.head_start + table.num_local_heads) == (head_lo, head_hi)
    assert (table.vocab_start_idx, table.vocab_end_idx) == (row_lo, row_hi)
    table.to(DEV)  # what a target does once the module exists: the offsets buffer goes to the GPU
    table.load_weights({"weight": weight, "scale": scale})
    gen = torch.Generator(device="cpu").manual_seed(11)
    ids = torch.stack([torch.randint(0, n, (64,), generator=gen) for n in BUCKETS], dim=1).to(DEV)
    got = table(ids)

    dense = _dense(weight, scale).to(DEV)
    offsets = torch.tensor([0, *torch.tensor(BUCKETS).cumsum(0)[:-1].tolist()], device=DEV)
    rows = ids + offsets  # global rows, [64, heads]
    expect = dense[rows]  # [64, heads, DIM]
    if tp_size == 2:  # head shard: only the owned heads come back
        expect = expect[:, head_lo:head_hi]
    else:  # row shard: every head, unowned rows zero
        owned = (rows >= row_lo) & (rows < row_hi)
        expect = expect.masked_fill(~owned.unsqueeze(-1), 0)
    assert got.shape == expect.shape and got.dtype == torch.bfloat16
    torch.testing.assert_close(got, expect, rtol=0, atol=0)


def test_mxfp4_table_stores_e2m1_pairs_and_reads_their_values():
    """An MXFP4 checkpoint table (stock's `engram_embed_quant == "mxfp4"`): the shard holds `D / 2` bytes a row (E2M1
    pairs, element 2k in the low nibble), node-shareable bytes included, and a lookup returns the widened values."""
    from tensorrt_llm._torch.modeling_v2.models.deepseek_v41.ops.engram.engram_lookup import (
        e2m1_to_e4m3_bytes,
    )

    gen = torch.Generator(device="cpu").manual_seed(13)
    rows = sum(BUCKETS)
    packed = torch.randint(0, 256, (rows, DIM // 2), generator=gen, dtype=torch.uint8)
    _, scale = _checkpoint_rows()
    table = ShardedFp8MultiHeadEmbedding(
        list_of_N=BUCKETS,
        D=DIM,
        block_size=BLOCK,
        dtype=torch.bfloat16,
        tp_size=1,
        tp_rank=0,
        mxfp4=True,
    )
    assert (
        table.mxfp4 and table.weight.dtype == torch.uint8 and table.weight.shape == (rows, DIM // 2)
    )
    assert table._table_bytes() == rows * (DIM // 2 + DIM // BLOCK)
    table.to(DEV)
    table.load_weights({"weight": packed, "scale": scale})
    ids = torch.stack([torch.randint(0, n, (64,), generator=gen) for n in BUCKETS], dim=1).to(DEV)
    got = table(ids)
    wide = e2m1_to_e4m3_bytes(packed).view(torch.float8_e4m3fn)
    offsets = torch.tensor([0, *torch.tensor(BUCKETS).cumsum(0)[:-1].tolist()], device=DEV)
    expect = _dense(wide, scale).to(DEV)[ids + offsets]
    torch.testing.assert_close(got, expect, rtol=0, atol=0)


def test_table_format_follows_the_quantization_config():
    """`engram_table_mxfp4`: stock's `engram_table_is_mxfp4` on the checkpoint's quantization config, for the layer of
    the checkpoint prefix (fp8 tables without `engram_embed_quant`)."""
    from types import SimpleNamespace as NS

    from tensorrt_llm._torch.modeling_v2.models.deepseek_v41.ops.engram.engram_table.table import (
        engram_table_mxfp4,
    )

    mx = {"quant_method": "fp8", "engram_embed_quant": "mxfp4", "engram_embed_block_size": 32,
          "engram_embed_scale_dtype": "ue8m0", "engram_embed_layers": [1, 14]}  # fmt: skip
    assert engram_table_mxfp4(NS(_v41_quantization_config=mx), "layers.14.engram.")
    assert engram_table_mxfp4(NS(quantization_config=mx), "layers.1.engram.")
    assert not engram_table_mxfp4(
        NS(quantization_config={"quant_method": "fp8"}), "layers.1.engram."
    )
    assert not engram_table_mxfp4(NS(), "layers.1.engram.")
    with pytest.raises(ValueError):  # a layer the checkpoint does not list: mixed table formats
        engram_table_mxfp4(NS(quantization_config=mx), "layers.2.engram.")


# Four heads over three ranks shard rows: ceil(170 / 3) = 57 rows per rank, the final rank owning [114, 170) (56 rows
# of its 57).
UNEVEN = [37, 41, 43, 49]


def _rows(rows, scale_cols=DIM // BLOCK, weight_cols=DIM):
    gen = torch.Generator(device="cpu").manual_seed(3)
    weight = (torch.randn(rows, weight_cols, generator=gen) * 4).to(torch.float8_e4m3fn)
    scale = torch.randint(118, 134, (rows, scale_cols), generator=gen, dtype=torch.uint8).view(
        torch.float8_e8m0fnu
    )
    return {"weight": weight, "scale": scale}


def _final_row_shard():
    table = ShardedFp8MultiHeadEmbedding(
        list_of_N=UNEVEN, D=DIM, block_size=BLOCK, dtype=torch.bfloat16, tp_size=3, tp_rank=2
    )
    assert not table.shard_heads and (table.vocab_start_idx, table.vocab_end_idx) == (114, 170)
    assert table.part_num_embeddings == 57
    return table


def test_complete_final_row_shard_loads_and_zeroes_only_its_padding():
    weights = _rows(sum(UNEVEN))
    table = _final_row_shard()
    table.load_weights(weights)
    for name in ("weight", "scale"):
        got = getattr(table, name).data.view(torch.uint8)
        assert torch.equal(got[:56], weights[name][114:170].view(torch.uint8))
        assert not got[56:].any()


@pytest.mark.parametrize(
    "weights, name",
    [
        (
            _rows(sum(UNEVEN) - 1),
            "weight",
        ),  # the final shard yields 55 of its 56 rows: row 169 would read padding
        (
            _rows(sum(UNEVEN), scale_cols=1),
            "scale",
        ),  # one code where 4 are needed: copy_ would broadcast it
        (_rows(sum(UNEVEN), weight_cols=DIM // 2), "weight"),
    ],
    ids=["short-final-shard", "one-column-scale", "narrow-weight"],
)
def test_incomplete_or_narrow_checkpoint_rows_are_rejected(weights, name):
    with pytest.raises(ValueError, match=f"Engram {name}"):
        _final_row_shard().load_weights(weights)


def test_narrow_scale_is_rejected_on_a_full_shard():
    table = ShardedFp8MultiHeadEmbedding(
        list_of_N=BUCKETS, D=DIM, block_size=BLOCK, dtype=torch.bfloat16, tp_size=1, tp_rank=0
    )
    with pytest.raises(ValueError, match="Engram scale"):
        table.load_weights(_rows(sum(BUCKETS), scale_cols=1))


def test_node_share_name_tells_mxfp4_from_fp8_tables():
    """One checkpoint file key, two table formats: an MXFP4 table and an FP8 table of the same shape never map the same
    node-shared copy (their row widths differ), while equal formats do."""

    def table(mxfp4):
        return ShardedFp8MultiHeadEmbedding(
            list_of_N=BUCKETS, D=DIM, block_size=BLOCK, dtype=torch.bfloat16, tp_size=1, tp_rank=0,
            share_key="/ckpt/model-00002.safetensors:1:2:3:4:layer1", mxfp4=mxfp4,
        )  # fmt: skip

    fp8, mx = table(False), table(True)
    assert fp8._share_name != mx._share_name
    assert table(True)._share_name == mx._share_name
    assert mx.weight.shape == (sum(BUCKETS), DIM // 2) and fp8.weight.shape == (sum(BUCKETS), DIM)


@pytest.mark.parametrize("target", ["dep2", "dep", "tep8_moe_tp4_ep2"])
def test_targets_engram_module_requires_regular_checkpoint_loading(target):
    """A target builds its table in `load`, after the loader's post-init check, so its `_Engram` module carries the
    marker the loader reads (`_requires_regular_checkpoint_loading`): a load format other than AUTO, which cannot fill
    the pinned host table in place, is refused before the load instead of failing in it."""
    import importlib
    from types import SimpleNamespace

    from torch import nn

    from tensorrt_llm._errors import UnsupportedError
    from tensorrt_llm._torch.pyexecutor.model_loader import _check_post_init_model_compat
    from tensorrt_llm.llmapi.llm_args import LoadFormat

    modeling = importlib.import_module(
        f"tensorrt_llm._torch.modeling_v2.models.deepseek_v41.flash_nvfp4__sm_103__{target}.modeling"
    )
    assert modeling._Engram._requires_regular_checkpoint_loading is True
    # The marker is the class's, so a bare module (no config) carries it.
    engram = modeling._Engram.__new__(modeling._Engram)
    nn.Module.__init__(engram)
    model = nn.ModuleDict({"engram": engram})
    loader = SimpleNamespace(checkpoint_format="HF")
    _check_post_init_model_compat(model, LoadFormat.AUTO, loader)
    with pytest.raises(UnsupportedError, match="load_format=AUTO"):
        _check_post_init_model_compat(model, LoadFormat.DUMMY, loader)
