# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""What a layer's decode hands the MLA generation entry.

CPU only: the geometry comes from the checkpoint's `config.json` and the entries
are called with plain tensors, so no weight and no device are read.

A decode row reaches the generation entry in one of two shapes, and which shape
a layer uses is its geometry:

* a **sparse** layer passes its indexer's top-k key positions as
  `sparse_attn_indices`, and the entry attends exactly those pool rows;
* a **windowed** layer passes none. Its decode is the dense windowed entry: the
  window is a constant and each step's length comes from the device sequence
  lengths.

The second half is why this is pinned rather than left implicit. A captured
CUDA graph replays every device tensor at its captured address, so anything a
step recomputes arrives fresh -- but a scalar the entry derives from the *host*
KV length is frozen at capture. A windowed layer that handed the sparse entry a
static selection took that frozen bound: on the decode steps measured, its
window came back empty and the layer's attention output was exact zeros, while
the same step run eagerly attended correctly.

`_recent_rows` is the selection's layout, still used where the indexer names
keys by position: a valid prefix followed by `-1` padding, request-local.

A **draft block** hands the same entry `block_width` rows per request, over a
selection whose live prefix is the request's picks and whose tail is the block's
own slots: there the valid prefix is *not* the whole buffer, and the count is
per row (`DflyBlockSelection.lengths`, the writer's), not the request's device
length. The entry derives each row's key count from a per-request bound
(`bound - width + 1 + j`), so the bound is the writer's count through the block's
*last* row -- the one row the block's `width` and the selection's live prefix
agree on -- while the step's device length, the pool's end, parts from it once a
request's picks fill the selection.
"""

import torch

# The checkpoint stand-in this directory's tests share; pytest puts the test
# file's own directory on the path.
from _target import BlockStep, Step, core, drafter, mla_layer, requires_checkpoint

from tensorrt_llm._torch.modeling_v2.models.hyv4_flash.hy4_flash_fp8__sm_103__default import (  # noqa: E501
    draft as draft_module,
)
from tensorrt_llm._torch.modeling_v2.models.hyv4_flash.hy4_flash_fp8__sm_103__default import (  # noqa: E501
    modeling as modeling_module,
)
from tensorrt_llm._torch.modeling_v2.models.hyv4_flash.hy4_flash_fp8__sm_103__default.dims import (  # noqa: E501
    dims,
)
from tensorrt_llm._torch.modeling_v2.models.hyv4_flash.hy4_flash_fp8__sm_103__default.draft import (  # noqa: E501
    DflyBlockSelection,
    Hyv4FlashDraft,
)
from tensorrt_llm._torch.modeling_v2.models.hyv4_flash.hy4_flash_fp8__sm_103__default.modeling import (  # noqa: E501
    MlaLayer,
    _decode_selection,
    _recent_rows,
)
from tensorrt_llm._torch.modeling_v2.models.hyv4_flash.hy4_flash_fp8__sm_103__default.ops.spec.dfly_block_prepare import (  # noqa: E501
    dfly_block_indices,
)


@requires_checkpoint
def test_only_a_sparse_layer_owns_a_decode_selection():
    """Every windowed layer hands `None` and carries the window the dense entry masks with."""
    c = core()
    d = dims(c)
    window = int(c.model_config.pretrained_config.text_config.sliding_window)

    # Even handed one, only a sparse layer's selection reaches the entry.
    selection = torch.zeros(1, d.index_topk, dtype=torch.int32)
    owned = tuple(
        i for i, geom in enumerate(d.layers) if _decode_selection(geom, selection) is not None
    )
    assert owned == d.sparse_layers
    assert len(owned) == 12 and len(d.layers) - len(owned) == 30

    for i, geom in enumerate(d.layers):
        if i in owned:
            assert geom.window is None, f"sparse layer {i} carries a window"
        else:
            # No selection, so the decode is the dense windowed entry; the mask
            # it applies is this constant, never a captured step's length.
            assert geom.window == window, f"windowed layer {i} carries {geom.window}, not {window}"


def test_recent_rows_are_a_valid_prefix_followed_by_padding():
    """Position 0 is one valid column; a row below the document start is no column."""
    rows = _recent_rows(torch.tensor([0, 1, 2, 7], dtype=torch.int64), 4)
    assert rows.dtype == torch.int32
    assert rows.tolist() == [
        [0, -1, -1, -1],
        [1, 0, -1, -1],
        [2, 1, 0, -1],
        [7, 6, 5, 4],
    ]


@requires_checkpoint
def test_a_generation_step_hands_the_entry_a_bound_that_does_not_move():
    """The entry's host KV length is a launch scalar a capture bakes, so no step may set it.

    A sparse layer reads exactly the rows its selection names, a windowed one is
    carried by its window: both bounds are properties of the layer and the pool,
    and the same short and long step must hand the entry the same scalar.
    """
    d = dims(core())
    max_seq_len = 5120
    rope = {"rotary_cos_sin": None}
    sparse = mla_layer(d.sparse_layers[2])
    windowed = mla_layer(next(i for i in range(len(d.layers)) if i not in d.sparse_layers))
    selection = torch.zeros((1, d.index_topk), dtype=torch.int32)

    for kv in (6, max_seq_len - 1):
        md = Step(kv, max_seq_len)
        sparse_bound = sparse._generation_args(md, rope, selection)["host_past_key_value_lengths"]
        window_bound = windowed._generation_args(md, rope, None)["host_past_key_value_lengths"]
        assert sparse_bound.dtype == md.kv_lens_runtime.dtype
        assert sparse_bound.tolist() == [d.index_topk]
        assert window_bound.tolist() == [max_seq_len]
        # A windowed layer past its window is on the sliding mask only above it.
        assert window_bound.item() > windowed.geom.window


@requires_checkpoint
def test_a_draft_blocks_rows_are_counted_by_the_writer_not_the_device_length(monkeypatch):
    """A block row's key width is its own, and the bound the block call hands the entry says so.

    A draft row's selection is the request's picks compacted, then the block's
    own slots up to and including that row, `-1` past the live prefix:
    `DflyBlockIndices` -- the op the engine's worker runs -- leaves the columns
    in `DflyBlockSelection.packed` and the prefix's width in
    `DflyBlockSelection.lengths`. The entry derives row `j`'s own key count from
    its per-request bound (`bound - width + 1 + j`, the dense causal bound read
    as a count), so the bound has to be the writer's count through the block's
    *last* row -- the one row where the block's `width` and the selection's live
    prefix agree -- while the step's token-space length, the block's exclusive
    end, is the pool's end: past `anchors > index_topk` a row's live width stops
    at `index_topk + j + 1`, and a bound at the pool's end reads the `-1` tail
    into the softmax.

    The kernels under the call are plain torch: which tensors reach the entry is
    the subject here, not the arithmetic on them. The rope prologue is measured
    beside it, because the step's device lengths are what place the block's rows
    in the pool; the writer's counts are the entry's bound, not its placement.
    """
    d = drafter()
    bundle = d.dims
    gens, num_contexts, width = 1, 1, bundle.block_width
    rows = gens * width
    end = 5134  # the pool's length here: the block's exclusive end
    anchors = end - width
    device_lengths = (3072, end)  # a context request of the step, then the block's
    md = BlockStep(device_lengths, num_contexts=num_contexts, max_seq_len=end)

    # The picks state the worker hands `DflyBlockIndices`: one request whose every
    # pick is valid, so its selection is full -- the regime the counts part in.
    history = torch.arange(anchors - bundle.index_topk, anchors, dtype=torch.int32)
    history = history.expand(bundle.num_layers, 1, bundle.index_topk).contiguous()
    block_slots = torch.arange(anchors, end, dtype=torch.int32).expand(gens, width).contiguous()
    packed = torch.empty(bundle.num_layers, rows, bundle.index_width, dtype=torch.int32)
    lengths = torch.empty(bundle.num_layers, rows, dtype=torch.int32)
    dfly_block_indices.reference(
        history,
        torch.zeros(gens, dtype=torch.int64),
        block_slots,
        torch.ones(gens, dtype=torch.bool),
        packed,
        lengths,
    )

    # What the writer wrote: row `j` lives for `index_topk + j + 1` columns, and
    # every column past its prefix is padding, `-1`.
    assert lengths[0].tolist() == [bundle.index_topk + j + 1 for j in range(width)]
    for j in range(width):
        assert packed[0, j, : bundle.index_topk].min() >= 0
        assert packed[0, j, lengths[0, j] - 1].item() == block_slots[0, j]
        assert (packed[0, j, lengths[0, j] :] == -1).all()
    assert lengths[0, -1] < device_lengths[-1], "the two counts have parted"

    placement = DflyBlockSelection(
        packed=packed,
        lengths=lengths,
        slots=block_slots,
        positions=torch.arange(anchors, end, dtype=torch.int64),
        active=torch.ones(gens, dtype=torch.bool),
        attn_metadata=md,
    )

    def plain_gemm(q, w, scale, w_scale, output_dtype=None):
        """`block_fp8_gemm` with its dequantization dropped: this is wiring, not math."""
        return (q.float() @ w.float().t()).to(output_dtype or q.dtype)

    def plain_bmm(a, b, out):
        """`bmm_out`'s write-in-place signature over `torch.bmm`."""
        return out.copy_(torch.bmm(a.float(), b.float()).to(out.dtype))

    def rope_table(self, md_, device):
        """The target sparse layer's rope table, as the entry's args read it."""
        return {"rotary_cos_sin": None}

    seen = []
    prologue = []

    def record(**kwargs):
        seen.append(kwargs)
        kwargs["output"].zero_()
        return kwargs["output"]

    def record_rope(*args):
        """The rope prologue, called positionally: keep the call, do not append."""
        prologue.append(args)

    monkeypatch.setattr(draft_module, "_rms", lambda x, weight, eps: x)
    monkeypatch.setattr(draft_module, "cublas_mm", lambda a, b: a @ b.to(a.dtype))
    monkeypatch.setattr(draft_module, "fp8_quantize_1x128", lambda x: (x, x))
    monkeypatch.setattr(draft_module, "block_fp8_gemm", plain_gemm)
    monkeypatch.setattr(draft_module, "bmm_out", plain_bmm)
    monkeypatch.setattr(modeling_module, "bmm_out", plain_bmm)
    monkeypatch.setattr(draft_module, "mla_rope_generation", record_rope)
    monkeypatch.setattr(draft_module, "thop_attention", record)
    monkeypatch.setattr(Hyv4FlashDraft, "_rope_table", rope_table)
    monkeypatch.setattr(Hyv4FlashDraft, "_mlp", lambda self, x, i: torch.zeros_like(x))
    monkeypatch.setattr(MlaLayer, "epilogue", lambda self, attn, x: torch.zeros_like(x))

    hidden = d.dflash_forward(
        noise_embedding=torch.zeros(gens, width, bundle.hidden, dtype=bundle.dtype),
        query_positions=torch.zeros(gens, width, dtype=torch.long),
        num_ctx_per_req=None,
        ctx_k_cache=None,
        ctx_v_cache=None,
        ctx_cache_batch_idx=None,
        ctx_kv_cache=None,
        ctx_page_table=None,
        block_placement=placement,
    )

    assert hidden.shape == (rows, bundle.hidden)

    # The prologue is called positionally; the two arguments pinned here are the
    # step's device lengths (`kv_lens_cuda`), which place the block's rows, and the
    # block width (`predicted_tokens_per_seq`), which says how many rows a request
    # appended.
    kv_lens_cuda, predicted_tokens_per_seq = 10, 23
    assert len(prologue) == bundle.num_layers
    for args in prologue:
        assert args[kv_lens_cuda].tolist() == list(device_lengths)
        assert args[predicted_tokens_per_seq] == width

    assert len(seen) == bundle.num_layers
    for i, kwargs in enumerate(seen):
        bound = kwargs["sequence_length"]
        assert bound.dtype == torch.int32
        # The context request ahead of the block keeps the step's own length; the
        # block's request is handed the writer's count through the block's last row.
        assert bound.tolist() == [device_lengths[0], lengths[i, width - 1].item()]
        assert bound[-1] < device_lengths[-1], "the bound is the pool's end, not the count"
        # The kernel's own derivation, row by row: the one bound this call hands the
        # entry turns into the writer's count for every row of the block.
        for j in range(width):
            assert bound[-1].item() - width + 1 + j == lengths[i, j].item()
        # And not the per-row-counts path, which no kernel of this arch serves.
        assert kwargs["sparse_attn_kv_lens"] is None
        # The block's own layout: `width` rows of one request, the target layer's
        # pool row, the selection's width for the top-k and not the request's length.
        assert kwargs["sparse_attn_indices"].tolist() == packed[i].tolist()
        assert kwargs["num_sparse_topk"] == bundle.index_width
        assert kwargs["predicted_tokens_per_seq"] == width
        assert kwargs["local_layer_idx"] == bundle.target_kv_layers[i]
        assert kwargs["latent_cache"].shape == (rows, bundle.kv_lora + bundle.rope_dim)
        assert kwargs["q_pe"].shape == (rows, bundle.heads_local, bundle.rope_dim)
        assert kwargs["attention_window_size"] == md.max_seq_len
