# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The Nemotron-3 Ultra BF16 tp8ep2 weight table slices the checkpoint the way its docstring says.

Eight ranks each take their own share of one 1.1 TB checkpoint, so a slice that is off by a segment
-- a head, a group, an expert's width -- loads without an error and produces plausible logits. What the
accuracy gate can see of that is one number on one dataset; this test sees the slicing itself, by
index arithmetic on the checkpoint's published tensor shapes, so it needs no weights and no GPU:

* every rank fills every declared weight exactly once, source and destination volumes agreeing;
* across the 8 ranks each sharded checkpoint tensor is read exactly once, the two KV heads exactly
  four times each (replicated), and everything else replicated by every rank;
* the segments of `in_proj` and the conv channels sit where the model lays them out (z | x | B | C |
  dt, x | B | C), in the order the forward reads them;
* rank r holds the 256 routed experts of window r % 2, each cut to its quarter of the intermediate
  width: up projection rows and down projection columns [1280 t, 1280 t + 1280) of 5120 with
  t = r // 2 -- the rank layout the engine's own `Mapping` gives moe_ep 2 x moe_tp 4, which is what
  the ranks are built from here;
* what a rank leaves unread is exactly the MTP module, any layer past the configured stack and the
  other window's experts, which is the assertion `load` makes at the end.

The shapes below are the checkpoint's own (the BF16 release's headers), written from its geometry
rather than read back from the table under test.
"""

from __future__ import annotations

import itertools
import re
from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.nemotron_h.nemotron3_ultra_bf16__sm_100__tp8ep2 import (  # noqa: E501
    weights as W,
)
from tensorrt_llm.mapping import Mapping

TP, MOE_EP, MOE_TP = 8, 2, 4
HIDDEN, VOCAB = 8192, 131072
HEADS, SSM_HEAD_DIM, GROUPS, STATE, CONV_K = 256, 64, 8, 128, 4
D_INNER = HEADS * SSM_HEAD_DIM  # 16384
GN = GROUPS * STATE  # 1024: B (and C) over every group
Q_HEADS, KV_HEADS, HEAD_DIM = 64, 2, 128
EXPERTS, MOE_INTER, LATENT, SHARED = 512, 5120, 2048, 10240
# One of each block kind, mamba and moe twice: every per-layer weight role, and a layer index that
# is not the first. Depth is not part of what the table slices.
KINDS = ["mamba", "moe", "attention", "mamba", "moe"]

# (checkpoint key pattern, shape, how many of the 8 ranks read each element)
_SHARDED, _KV, _REPLICATED = 1, TP // KV_HEADS, TP
_CHECKPOINT = [
    (r"backbone\.embeddings\.weight", (VOCAB, HIDDEN), _REPLICATED),
    (r"backbone\.norm_f\.weight", (HIDDEN,), _REPLICATED),
    (r"backbone\.layers\.\d+\.norm\.weight", (HIDDEN,), _REPLICATED),
    # mamba
    (
        r"backbone\.layers\.\d+\.mixer\.in_proj\.weight",
        (2 * D_INNER + 2 * GN + HEADS, HIDDEN),
        _SHARDED,
    ),
    (r"backbone\.layers\.\d+\.mixer\.conv1d\.weight", (D_INNER + 2 * GN, 1, CONV_K), _SHARDED),
    (r"backbone\.layers\.\d+\.mixer\.conv1d\.bias", (D_INNER + 2 * GN,), _SHARDED),
    (r"backbone\.layers\.\d+\.mixer\.(A_log|D|dt_bias)", (HEADS,), _SHARDED),
    (r"backbone\.layers\.\d+\.mixer\.norm\.weight", (D_INNER,), _SHARDED),
    (r"backbone\.layers\.\d+\.mixer\.out_proj\.weight", (HIDDEN, D_INNER), _SHARDED),
    # attention
    (r"backbone\.layers\.\d+\.mixer\.q_proj\.weight", (Q_HEADS * HEAD_DIM, HIDDEN), _SHARDED),
    (r"backbone\.layers\.\d+\.mixer\.[kv]_proj\.weight", (KV_HEADS * HEAD_DIM, HIDDEN), _KV),
    (r"backbone\.layers\.\d+\.mixer\.o_proj\.weight", (HIDDEN, Q_HEADS * HEAD_DIM), _SHARDED),
    # LatentMoE
    (r"backbone\.layers\.\d+\.mixer\.gate\.weight", (EXPERTS, HIDDEN), _REPLICATED),
    (r"backbone\.layers\.\d+\.mixer\.gate\.e_score_correction_bias", (EXPERTS,), _REPLICATED),
    (r"backbone\.layers\.\d+\.mixer\.fc1_latent_proj\.weight", (LATENT, HIDDEN), _REPLICATED),
    (r"backbone\.layers\.\d+\.mixer\.fc2_latent_proj\.weight", (HIDDEN, LATENT), _REPLICATED),
    (r"backbone\.layers\.\d+\.mixer\.shared_experts\.up_proj\.weight", (SHARED, HIDDEN), _SHARDED),
    (
        r"backbone\.layers\.\d+\.mixer\.shared_experts\.down_proj\.weight",
        (HIDDEN, SHARED),
        _SHARDED,
    ),
    (r"backbone\.layers\.\d+\.mixer\.experts\.\d+\.up_proj\.weight", (MOE_INTER, LATENT), _SHARDED),
    (
        r"backbone\.layers\.\d+\.mixer\.experts\.\d+\.down_proj\.weight",
        (LATENT, MOE_INTER),
        _SHARDED,
    ),
]


def _core(rank: int) -> SimpleNamespace:
    """The few fields `dims` reads, at Ultra's numbers, for one rank of an 8-rank tp8ep2 group. The
    mapping is the engine's own: which window and which slice a rank gets is its decision."""
    cfg = SimpleNamespace(
        layers_block_type=list(KINDS),
        hidden_size=HIDDEN,
        vocab_size=VOCAB,
        mamba_num_heads=HEADS,
        mamba_head_dim=SSM_HEAD_DIM,
        n_groups=GROUPS,
        ssm_state_size=STATE,
        conv_kernel=CONV_K,
        num_attention_heads=Q_HEADS,
        num_key_value_heads=KV_HEADS,
        head_dim=HEAD_DIM,
        n_routed_experts=EXPERTS,
        num_experts_per_tok=22,
        routed_scaling_factor=5.0,
        moe_intermediate_size=MOE_INTER,
        moe_latent_size=LATENT,
        moe_shared_expert_intermediate_size=SHARED,
    )
    mapping = Mapping(
        world_size=TP,
        rank=rank,
        gpus_per_node=4,
        tp_size=TP,
        moe_ep_size=MOE_EP,
        moe_tp_size=MOE_TP,
    )
    model_config = SimpleNamespace(
        pretrained_config=cfg, mapping=mapping, torch_dtype=torch.bfloat16
    )
    return SimpleNamespace(model_config=model_config)


def _checkpoint_shape(key: str) -> tuple[int, ...]:
    for pattern, shape, _ in _CHECKPOINT:
        if re.fullmatch(pattern, key):
            return shape
    raise AssertionError(f"the table reads a checkpoint key this test does not know: {key}")


def _declared_shapes(d: SimpleNamespace) -> dict[str, tuple[int, ...]]:
    shapes = {}
    for entry in W.MODEL_WEIGHTS.WEIGHTS:
        keys = (
            [entry.name]
            if entry.layers is None
            else [f"l{i}_{entry.name}" for i in entry.layers(d)]
        )
        for key in keys:
            shapes[key] = tuple(entry.shape(d))
    return shapes


def _box(index, shape: tuple[int, ...]) -> tuple[tuple[int, int], ...]:
    """The index (None, an int, a slice or a tuple of them) as a half-open range per dimension."""
    if index is None:
        index = ()
    elif not isinstance(index, tuple):
        index = (index,)
    box = []
    for dim, size in enumerate(shape):
        sel = index[dim] if dim < len(index) else slice(None)
        if isinstance(sel, int):
            box.append((sel, sel + 1))
        else:
            start, stop, step = sel.indices(size)
            assert step == 1, sel
            box.append((start, stop))
    return tuple(box)


def _volume(box) -> int:
    n = 1
    for lo, hi in box:
        n *= hi - lo
    return n


def _overlap(a, b) -> bool:
    return all(max(x[0], y[0]) < min(x[1], y[1]) for x, y in zip(a, b))


@pytest.fixture(scope="module")
def ranks():
    """`(dims, declared shapes, manifest)` of each of the 8 ranks."""
    out = []
    for rank in range(TP):
        core = _core(rank)
        d = W.MODEL_WEIGHTS.dims(core)
        out.append((d, _declared_shapes(d), W.MODEL_WEIGHTS.manifest(core)))
    return out


def test_every_rank_fills_every_declared_weight_exactly_once(ranks):
    for rank, (_, shapes, manifest) in enumerate(ranks):
        assert set(manifest) == set(shapes), f"rank {rank}: manifest and declaration disagree"
        for param, rows in manifest.items():
            sources = [source for source, _ in rows]
            dst_boxes = [_box(dst, shapes[param]) for _, _, dst in sources]
            where = f"rank {rank} {param} {shapes[param]}"
            for (key, src, _), dst_box in zip(sources, dst_boxes):
                src_box = _box(src, _checkpoint_shape(key))
                assert _volume(src_box) == _volume(dst_box), (
                    f"{where}: {key} {src_box} fills {dst_box}: the volumes differ"
                )
            for a, b in itertools.combinations(dst_boxes, 2):
                assert not _overlap(a, b), f"{where}: two sources fill the same elements"
            whole = tuple((0, n) for n in shapes[param])
            assert sum(map(_volume, dst_boxes)) == _volume(whole), (
                f"{where}: the sources do not cover the weight"
            )


def _multiplicity(boxes, shape) -> set[int]:
    """How many boxes cover each element, over the cells the boxes' edges cut the tensor into."""
    cuts = [
        sorted({0, n} | {b[d][0] for b in boxes} | {b[d][1] for b in boxes})
        for d, n in enumerate(shape)
    ]
    counts = set()
    for cell in itertools.product(*(range(len(c) - 1) for c in cuts)):
        lo = [cuts[d][i] for d, i in enumerate(cell)]
        hi = [cuts[d][i + 1] for d, i in enumerate(cell)]
        counts.add(
            sum(
                all(b[d][0] <= lo[d] and hi[d] <= b[d][1] for d in range(len(shape))) for b in boxes
            )
        )
    return counts


def test_the_checkpoint_is_tiled_across_the_ranks_as_documented(ranks):
    reads: dict[str, list] = {}
    for _, _, manifest in ranks:
        for rows in manifest.values():
            for (key, src, _), _ in rows:
                reads.setdefault(key, []).append(_box(src, _checkpoint_shape(key)))
    for key, boxes in sorted(reads.items()):
        expected = next(n for p, _, n in _CHECKPOINT if re.fullmatch(p, key))
        got = _multiplicity(boxes, _checkpoint_shape(key))
        assert got == {expected}, (
            f"{key}: each element is read by {sorted(got)} of the {TP} ranks, expected {expected}"
        )


def test_in_proj_and_conv_segments_sit_where_the_model_lays_them_out(ranks):
    """`in_proj` rows are z | x | B | C | dt and the conv channels x | B | C. Each rank takes its
    own slice of every segment and packs them in that order, which is the order the forward's
    split reads them."""
    d_inner_r, gn_r, heads_r = D_INNER // TP, GN // TP, HEADS // TP
    for rank, (_, _, manifest) in enumerate(ranks):
        rows = [r for r, _ in manifest["l0_in_proj_w"]]
        starts = [
            rank * d_inner_r,  # z
            D_INNER + rank * d_inner_r,  # x
            2 * D_INNER + rank * gn_r,  # B
            2 * D_INNER + GN + rank * gn_r,  # C
            2 * D_INNER + 2 * GN + rank * heads_r,  # dt
        ]
        lengths = [d_inner_r, d_inner_r, gn_r, gn_r, heads_r]
        assert [(s[1][0].start, s[1][0].stop - s[1][0].start) for s in rows] == list(
            zip(starts, lengths)
        ), f"rank {rank}: in_proj segments"
        dst = [s[2][0].start for s in rows]
        assert dst == [0, d_inner_r, 2 * d_inner_r, 2 * d_inner_r + gn_r, 2 * d_inner_r + 2 * gn_r]
        rows = [r for r, _ in manifest["l0_conv_w"]]
        starts = [rank * d_inner_r, D_INNER + rank * gn_r, D_INNER + GN + rank * gn_r]
        lengths = [d_inner_r, gn_r, gn_r]
        assert [(s[1][0].start, s[1][0].stop - s[1][0].start) for s in rows] == list(
            zip(starts, lengths)
        ), f"rank {rank}: conv segments"


def test_attention_heads_and_the_replicated_kv_head(ranks):
    q_r = Q_HEADS // TP * HEAD_DIM
    for rank, (d, _, manifest) in enumerate(ranks):
        assert d.kv_head == rank // (TP // KV_HEADS), f"rank {rank}: reads kv head {d.kv_head}"
        (q, k, v) = [r for r, _ in manifest["l2_qkv_w"]]
        assert q[1][0] == slice(rank * q_r, (rank + 1) * q_r) and q[2][0] == slice(0, q_r)
        for src, dst_start in ((k, q_r), (v, q_r + HEAD_DIM)):
            assert src[1][0] == slice(d.kv_head * HEAD_DIM, (d.kv_head + 1) * HEAD_DIM)
            assert src[2][0] == slice(dst_start, dst_start + HEAD_DIM)


def _expert_window(rank: int) -> range:
    """The experts rank `rank` runs: window rank % 2 of two 256-expert windows."""
    local = EXPERTS // MOE_EP
    return range((rank % MOE_EP) * local, (rank % MOE_EP + 1) * local)


def test_every_rank_holds_its_window_and_its_slice_of_every_expert_in_it(ranks):
    width = MOE_INTER // MOE_TP
    for rank, (d, _, manifest) in enumerate(ranks):
        experts = _expert_window(rank)
        t = rank // MOE_EP
        assert (d.ep_rank, d.moe_tp_rank) == (rank % MOE_EP, t), f"rank {rank}: the mapping moved"
        cut = slice(t * width, (t + 1) * width)
        for param in ("l1_ex_fc1_w", "l1_ex_fc2_w", "l4_ex_fc1_w", "l4_ex_fc2_w"):
            layer, up = int(param[1]), "fc1" in param
            rows = [r[0] for r in manifest[param]]
            tensor = "up" if up else "down"
            assert [key for key, _, _ in rows] == [
                f"backbone.layers.{layer}.mixer.experts.{e}.{tensor}_proj.weight" for e in experts
            ], f"rank {rank} {param}"
            # Up projection [I, H]: rows of the slice; down projection [H, I]: its columns.
            expected = (cut,) if up else (slice(None), cut)
            assert all(src == expected for _, src, _ in rows), f"rank {rank} {param}"
            assert [dst for _, _, dst in rows] == [(e,) for e in range(len(experts))]


def test_what_a_rank_leaves_unread_is_the_mtp_module_and_the_other_window(ranks):
    """`load`'s closing assertion, on the whole key set of a checkpoint carrying an MTP module and a
    layer past the stack: leftover = `expected_unconsumed`, for every rank, and that is those
    extras plus the routed experts of the window the rank does not run."""
    union = {row[0][0] for _, _, manifest in ranks for rows in manifest.values() for row in rows}
    extras = {
        "mtp.layers.0.mixer.q_proj.weight",
        "mtp.layers.1.mixer.experts.0.up_proj.weight",
        "backbone.layers.99.mixer.in_proj.weight",
    }
    checkpoint = union | extras | {"lm_head.weight"}
    for rank, (d, _, manifest) in enumerate(ranks):
        consumed = {row[0][0] for rows in manifest.values() for row in rows} | {"lm_head.weight"}
        leftover = checkpoint - consumed
        assert leftover == W.NemotronUltraBf16Tp8ep2Weights.expected_unconsumed(
            d, dict.fromkeys(checkpoint)
        )
        mine = _expert_window(rank)
        other_window = {
            key
            for key in union
            if (m := re.fullmatch(r"backbone\.layers\.\d+\.mixer\.experts\.(\d+)\..+", key))
            and int(m.group(1)) not in mine
        }
        assert other_window, f"rank {rank}: the other window's experts are not in the checkpoint"
        assert leftover == extras | other_window, f"rank {rank}: unexpected leftovers"
