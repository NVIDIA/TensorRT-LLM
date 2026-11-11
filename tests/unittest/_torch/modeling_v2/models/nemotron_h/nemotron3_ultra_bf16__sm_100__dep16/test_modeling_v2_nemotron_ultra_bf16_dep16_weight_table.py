# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The Nemotron-3 Ultra BF16 dep16 weight table reads the checkpoint the way its docstring says.

Sixteen ranks each take their share of one 1.1 TB checkpoint: everything but the routed experts whole,
and a 32-expert window of those, fed to the engine's MoE layer. A window off by one expert, or a
tensor sliced where it should be whole, loads without an error and produces plausible logits; this
test sees the reading itself, by index arithmetic on the checkpoint's published tensor shapes, so it
needs no weights and no GPU:

* every rank fills every declared weight exactly once, source and destination volumes agreeing;
* every checkpoint tensor the table reads is read whole by each of the 16 ranks;
* the table declares no routed expert, and across the ranks the engine feed hands each expert of
  each MoE layer to exactly one rank, in its window, by global id;
* the fused qkv operand is every q row, then both k heads, then both v heads;
* what a rank leaves unread is exactly the MTP module, the experts outside its window and any
  layer past the configured stack, which is the assertion `load` makes at the end.

The shapes below are the checkpoint's own (the BF16 release's headers), written from its geometry
rather than read back from the table under test.
"""

from __future__ import annotations

import itertools
import re
from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.nemotron_h.nemotron3_ultra_bf16__sm_100__dep16 import (  # noqa: E501
    weights as W,
)

pytestmark = pytest.mark.cpu_only

RANKS = 16
HIDDEN, VOCAB = 8192, 131072
HEADS, SSM_HEAD_DIM, GROUPS, STATE, CONV_K = 256, 64, 8, 128, 4
D_INNER = HEADS * SSM_HEAD_DIM  # 16384
GN = GROUPS * STATE  # 1024: B (and C) over every group
Q_HEADS, KV_HEADS, HEAD_DIM = 64, 2, 128
EXPERTS, MOE_INTER, LATENT, SHARED = 512, 5120, 2048, 10240
WINDOW = EXPERTS // RANKS  # 32
# One of each block kind, mamba and moe twice: every per-layer weight role, and a layer index that
# is not the first. Depth is not part of what the table reads.
KINDS = ["mamba", "moe", "attention", "mamba", "moe"]
MOE_LAYERS = [1, 4]

# (checkpoint key pattern, shape) of every tensor the table reads whole.
_CHECKPOINT = [
    (r"backbone\.embeddings\.weight", (VOCAB, HIDDEN)),
    (r"backbone\.norm_f\.weight", (HIDDEN,)),
    (r"backbone\.layers\.\d+\.norm\.weight", (HIDDEN,)),
    # mamba
    (r"backbone\.layers\.\d+\.mixer\.in_proj\.weight", (2 * D_INNER + 2 * GN + HEADS, HIDDEN)),
    (r"backbone\.layers\.\d+\.mixer\.conv1d\.weight", (D_INNER + 2 * GN, 1, CONV_K)),
    (r"backbone\.layers\.\d+\.mixer\.conv1d\.bias", (D_INNER + 2 * GN,)),
    (r"backbone\.layers\.\d+\.mixer\.(A_log|D|dt_bias)", (HEADS,)),
    (r"backbone\.layers\.\d+\.mixer\.norm\.weight", (D_INNER,)),
    (r"backbone\.layers\.\d+\.mixer\.out_proj\.weight", (HIDDEN, D_INNER)),
    # attention
    (r"backbone\.layers\.\d+\.mixer\.q_proj\.weight", (Q_HEADS * HEAD_DIM, HIDDEN)),
    (r"backbone\.layers\.\d+\.mixer\.[kv]_proj\.weight", (KV_HEADS * HEAD_DIM, HIDDEN)),
    (r"backbone\.layers\.\d+\.mixer\.o_proj\.weight", (HIDDEN, Q_HEADS * HEAD_DIM)),
    # LatentMoE, except the routed experts
    (r"backbone\.layers\.\d+\.mixer\.gate\.weight", (EXPERTS, HIDDEN)),
    (r"backbone\.layers\.\d+\.mixer\.gate\.e_score_correction_bias", (EXPERTS,)),
    (r"backbone\.layers\.\d+\.mixer\.fc1_latent_proj\.weight", (LATENT, HIDDEN)),
    (r"backbone\.layers\.\d+\.mixer\.fc2_latent_proj\.weight", (HIDDEN, LATENT)),
    (r"backbone\.layers\.\d+\.mixer\.shared_experts\.up_proj\.weight", (SHARED, HIDDEN)),
    (r"backbone\.layers\.\d+\.mixer\.shared_experts\.down_proj\.weight", (HIDDEN, SHARED)),
]


def _core(rank: int, moe_backend: str = "MEGAMOE_CUTEDSL", **mapping) -> SimpleNamespace:
    """The few fields `dims` reads, at Ultra's numbers, for one rank of the 16-rank dep16 group."""
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
    fields = dict(enable_attention_dp=True, moe_tp_size=1, moe_ep_size=RANKS, moe_ep_rank=rank)
    fields.update(mapping)
    model_config = SimpleNamespace(
        pretrained_config=cfg,
        mapping=SimpleNamespace(**fields),
        torch_dtype=torch.bfloat16,
        moe_backend_requested=moe_backend,
    )
    return SimpleNamespace(model_config=model_config)


def _checkpoint_shape(key: str) -> tuple[int, ...]:
    for pattern, shape in _CHECKPOINT:
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
    """`(dims, declared shapes, manifest)` of each of the 16 ranks."""
    out = []
    for rank in range(RANKS):
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


def test_every_rank_reads_every_tensor_whole(ranks):
    """Attention DP: no tensor the table reads is sliced, and all 16 ranks read the same set."""
    keys = None
    for rank, (_, _, manifest) in enumerate(ranks):
        read = set()
        for rows in manifest.values():
            for (key, src, _), _ in rows:
                assert src is None, f"rank {rank} reads a slice {src} of {key}"
                read.add(key)
        assert keys is None or read == keys, f"rank {rank} reads another set of tensors"
        keys = read


def test_the_table_declares_no_routed_expert(ranks):
    for d, shapes, manifest in ranks:
        assert not any(".experts." in row[0][0] for rows in manifest.values() for row in rows)
        assert not any(k.startswith(tuple(f"l{i}_ex_" for i in MOE_LAYERS)) for k in shapes)


def test_the_engine_feed_is_this_ranks_window_and_tiles_the_experts(ranks):
    """Each MoE block's engine layer gets every expert of the rank's window once, by global id;
    across the 16 ranks each expert of each layer is fed exactly once."""
    fed = {i: [] for i in MOE_LAYERS}
    for rank, (d, _, _) in enumerate(ranks):
        assert (d.expert_offset, d.local_experts) == (WINDOW * rank, WINDOW)
        for i in MOE_LAYERS:
            feed = W.MODEL_WEIGHTS.engine_expert_feed(d, i)
            p = f"backbone.layers.{i}.mixer.experts"
            assert feed == [
                (f"{p}.{e}.up_proj.weight", f"{p}.{e}.down_proj.weight", str(e))
                for e in range(WINDOW * rank, WINDOW * (rank + 1))
            ], f"rank {rank} layer {i}"
            fed[i].extend(int(e) for _, _, e in feed)
    for i, experts in fed.items():
        assert sorted(experts) == list(range(EXPERTS)), f"layer {i}: experts fed {len(experts)}"


def test_the_per_rank_widths_are_the_whole_ones(ranks):
    d = ranks[3][0]
    assert d.in_proj_r == 2 * D_INNER + 2 * GN + HEADS == 35072
    assert d.conv_r == D_INNER + 2 * GN == 18432
    assert (d.heads_r, d.groups_r, d.gn_r, d.d_inner_r) == (HEADS, GROUPS, GN, D_INNER)
    assert d.norm_group == D_INNER // GROUPS == 2048
    assert (d.q_heads_r, d.kv_heads, d.qkv_r) == (Q_HEADS, KV_HEADS, 8704)
    assert d.shared_r == SHARED
    assert (d.ep_size, d.ep_rank) == (RANKS, 3)


def test_qkv_is_every_q_row_then_both_k_heads_then_both_v_heads(ranks):
    q, kv = Q_HEADS * HEAD_DIM, KV_HEADS * HEAD_DIM
    for _, _, manifest in ranks:
        (qr, kr, vr) = [r for r, _ in manifest["l2_qkv_w"]]
        assert [r[0] for r in (qr, kr, vr)] == [
            "backbone.layers.2.mixer.q_proj.weight",
            "backbone.layers.2.mixer.k_proj.weight",
            "backbone.layers.2.mixer.v_proj.weight",
        ]
        assert [r[2][0] for r in (qr, kr, vr)] == [
            slice(0, q),
            slice(q, q + kv),
            slice(q + kv, q + 2 * kv),
        ]


def test_what_a_rank_leaves_unread_is_the_mtp_module_and_the_other_windows(ranks):
    """`load`'s closing assertion, on the whole key set of a checkpoint carrying an MTP module and a
    layer past the stack: leftover = `expected_unconsumed`, for every rank."""
    table = {row[0][0] for _, _, manifest in ranks for rows in manifest.values() for row in rows}
    experts = {
        f"backbone.layers.{i}.mixer.experts.{e}.{proj}_proj.weight"
        for i in MOE_LAYERS
        for e in range(EXPERTS)
        for proj in ("up", "down")
    }
    extras = {
        "mtp.layers.0.mixer.q_proj.weight",
        "mtp.layers.1.mixer.experts.0.up_proj.weight",
        "backbone.layers.99.mixer.in_proj.weight",
    }
    checkpoint = table | experts | extras | {"lm_head.weight"}
    for rank, (d, _, manifest) in enumerate(ranks):
        fed = {
            k
            for i in MOE_LAYERS
            for up, down, _ in W.MODEL_WEIGHTS.engine_expert_feed(d, i)
            for k in (up, down)
        }
        consumed = (
            {row[0][0] for rows in manifest.values() for row in rows} | fed | {"lm_head.weight"}
        )
        leftover = checkpoint - consumed
        assert leftover == W.NemotronUltraBf16Dep16Weights.expected_unconsumed(
            d, dict.fromkeys(checkpoint)
        )
        outside = {
            k
            for k in leftover
            if ".experts." in k
            and k.startswith("backbone.")
            and not WINDOW * rank <= int(k.split(".")[5]) < WINDOW * (rank + 1)
        }
        assert leftover == outside | extras, f"rank {rank}: unexpected leftovers"


@pytest.mark.parametrize(
    "requested, engine",
    [
        ("AUTO", "CUTLASS"),
        ("CUTLASS", "CUTLASS"),
        ("cutlass", "CUTLASS"),
        ("MEGAMOE_CUTEDSL", "MEGAMOE_CUTEDSL"),
        ("TRTLLM", "TRTLLM"),
    ],
)
def test_every_admitted_request_runs_on_an_engine_backend(requested, engine):
    assert W.engine_moe_backend(SimpleNamespace(moe_backend_requested=requested)) == engine


def test_a_request_routing_never_admits_is_refused():
    with pytest.raises(AssertionError, match="routing admits"):
        W.engine_moe_backend(SimpleNamespace(moe_backend_requested="DEEPGEMM"))


@pytest.mark.parametrize(
    "mapping, missed",
    [
        (dict(enable_attention_dp=False), "attention-data-parallel"),
        (dict(moe_tp_size=2, moe_ep_size=8), "moe_tp 1"),
    ],
    ids=["no-attention-dp", "moe-tp2"],
)
def test_dims_refuse_another_layout(mapping, missed):
    with pytest.raises(AssertionError, match=missed):
        W.MODEL_WEIGHTS.dims(_core(0, **mapping))
