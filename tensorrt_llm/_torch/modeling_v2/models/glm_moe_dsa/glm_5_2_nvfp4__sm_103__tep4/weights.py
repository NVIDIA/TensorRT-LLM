# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Weight table and loader: glm-5.2-nvfp4 / sm_103 / tep4.

Every rank is handed the whole checkpoint dict, so every split is this table's:

* **tensor-parallel slices** -- `q_b_proj`, `kv_b_proj`, the dense and shared
  `gate_up` by output rows; `o_proj` and the two `down` by input columns; the
  shell's `lm_head` by vocabulary rows. The slice is taken on the lazy
  checkpoint entry, before the copy to the device, so a rank moves only its
  own bytes. That is the `shard` column of `W`.
* **expert windows** -- rank `r` loads routed experts `[64r, 64r+64)`.
* **replicated** -- the fused `q_a | kv_a` projection and its two norms, the
  whole indexer, the router, the embedding, every norm, and all 256 experts'
  per-tensor NVFP4 scalars (`post_load_weights` asserts them over every expert).

Only the routed experts of the trunk are NVFP4. The MTP module at layer
`num_hidden_layers` is bf16 throughout and ships no `embed_tokens` or
`shared_head.head` of its own: the draft model reads the trunk's.

Three relayouts no checkpoint stores:

* `kv_b_proj` rows regrouped `[k_nope | v]` per head -> `[H*nope ; H*v]`, over
  this rank's heads, after the row slice;
* the indexer's `wk` and `weights_proj` stacked into one fp32 block with
  LayerNorm's mean-centring folded into `wk` (the prefill projection), and the
  same pair uncentred in bf16 (the decode split-output projection, whose fused
  k entry computes the LayerNorm itself);
* the routed experts in the vendored grouped GEMMs' layout: FC1 rows in 64-row
  blocks alternating up and gate, and the block scales in the 512-byte
  `(32, 4, 4)` atom. This is not the trtllm-gen runner's layout, so the trunk
  experts have no stock MoE fallback. With `TRTLLM_GLM52_MOE_FC1_KBLK=1` (the
  default; the model reads it once at build as `fc1_kblk`) each expert's FC1
  weight is then stored k-block (`moe_fc1_kblock`): `[N/128][K/256][128 rows][128 B]`,
  so each FC1 GEMM's 128-row x 128 B TMA box is one contiguous 16 KB read instead of
  128 B pieces at a 3 KB pitch (decode GG1 -4.0 .. -4.6 %). Same shape, same bytes
  per expert and per 128-row block, scales unchanged. `TRTLLM_GLM52_MOE_FC2_KBLK=1`
  (also the default; `fc2_kblk`) does the same to each expert's FC2 weight, whose boxes
  were 128 B pieces at a 1 KB pitch (prefill-pair GG2 -2.1 .. -2.9 %, decode FC2
  -1.2 .. -1.5 %).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Callable

import torch
import torch.nn as nn

from .kernels.moe import moe_fc1_kblock as _fc1_kblock
from .kernels.moe import mtp_moe_gather_grouped_gemm_act_fusion as _mtp_fc1

# FC1 pairs up and gate in 64-row blocks; the MTP kernel's `epi_subtile_n` is the same 64.
_PAIR_BLOCK = 64
_SF_ATOM_M0, _SF_ATOM_M1, _SF_ATOM_K = 32, 4, 4
#: The scale atom the vendored grouped GEMMs address: 128 rows by 64 K elements in 512 bytes.
SF_ATOM_MN, SF_ATOM_K, SF_ATOM = 128, 64, 512


# ---------------------------------------------------------------------------
# Decode MoE hot experts (replica slots).
#
# Decode MoE under EP4 waits in the post-MoE all-reduce for the rank with the most distinct
# active experts. With `TRTLLM_GLM52_DEC_EXPERT_REPLICAS=R` (default 8; 0 disables it;
# at most `DEC_HOT_MAX_REPLICAS`) every rank also holds each trunk MoE layer's `R` hottest
# experts in local slots `64 .. 64 + R - 1` of its FC1/FC2 stacks, and the decode top-k
# (`moe_decode_topk_routing_rep`, 9..64 rows) assigns each active hot expert to the EP
# rank with the fewest active experts. R = 8 on B300, realtext ISL 1k / OSL 8k /
# concurrency 32 with MTP: decode step -0.27 ms (-1.4 %), +9.3 GB of weights per rank;
# R = 16 measured no further gain.
#
# `DEC_HOT[layer]`: global expert ids by descending decode selection frequency (ties: lower
# id), measured over 32 MMLU prompts (35,129 teacher-forced tokens, top-8 per layer); the
# first `R` are the layer's replicas. The MTP layer has none.
# ---------------------------------------------------------------------------
#: The longest replica list a layer carries in `DEC_HOT`.
DEC_HOT_MAX_REPLICAS = 16

DEC_HOT: dict[int, tuple[int, ...]] = {
    3: (62, 15, 77, 152, 250, 141, 60, 158, 236, 231, 142, 166, 90, 83, 7, 11),
    4: (209, 241, 115, 136, 248, 148, 107, 217, 198, 114, 94, 186, 176, 252, 177, 53),
    5: (192, 221, 148, 3, 210, 100, 92, 60, 175, 89, 19, 153, 52, 62, 33, 18),
    6: (241, 215, 165, 220, 197, 6, 160, 166, 120, 81, 103, 249, 15, 111, 164, 115),
    7: (16, 211, 177, 52, 146, 72, 25, 41, 143, 77, 106, 140, 88, 169, 190, 224),
    8: (14, 39, 92, 254, 141, 113, 145, 19, 44, 195, 180, 186, 148, 112, 26, 250),
    9: (48, 145, 36, 195, 249, 222, 49, 144, 93, 167, 164, 188, 154, 212, 186, 64),
    10: (223, 150, 125, 141, 50, 235, 222, 168, 92, 192, 252, 133, 175, 231, 205, 52),
    11: (45, 106, 221, 40, 27, 67, 61, 122, 236, 161, 96, 175, 209, 207, 41, 181),
    12: (145, 192, 251, 24, 185, 249, 69, 190, 155, 121, 85, 205, 170, 115, 25, 46),
    13: (250, 140, 205, 35, 57, 54, 105, 195, 183, 138, 208, 242, 234, 192, 84, 220),
    14: (199, 19, 228, 130, 236, 208, 219, 131, 92, 129, 94, 198, 124, 238, 71, 64),
    15: (15, 28, 207, 20, 127, 169, 37, 172, 154, 117, 219, 137, 96, 176, 251, 180),
    16: (211, 234, 185, 213, 130, 79, 40, 10, 123, 232, 116, 60, 61, 35, 94, 11),
    17: (127, 171, 106, 152, 78, 100, 191, 224, 168, 10, 45, 110, 209, 109, 21, 146),
    18: (241, 76, 40, 110, 53, 62, 224, 38, 122, 220, 68, 22, 208, 199, 120, 33),
    19: (218, 62, 239, 130, 142, 140, 77, 181, 56, 72, 15, 191, 32, 3, 38, 44),
    20: (114, 220, 23, 181, 195, 41, 141, 105, 162, 75, 229, 171, 44, 11, 131, 35),
    21: (197, 231, 244, 87, 13, 119, 136, 191, 219, 245, 97, 228, 77, 236, 169, 117),
    22: (34, 133, 221, 186, 209, 222, 177, 8, 57, 115, 117, 67, 56, 32, 127, 139),
    23: (253, 86, 119, 90, 195, 235, 52, 32, 155, 70, 26, 115, 145, 33, 68, 72),
    24: (198, 100, 94, 159, 53, 214, 168, 107, 206, 248, 221, 7, 141, 176, 3, 39),
    25: (58, 119, 253, 205, 156, 175, 230, 127, 143, 215, 121, 24, 108, 248, 71, 207),
    26: (166, 123, 164, 44, 4, 99, 133, 135, 124, 159, 176, 184, 127, 61, 37, 233),
    27: (181, 117, 243, 196, 26, 240, 124, 140, 127, 80, 135, 5, 157, 56, 86, 76),
    28: (23, 159, 246, 148, 175, 194, 244, 9, 98, 240, 102, 155, 46, 62, 29, 149),
    29: (39, 151, 115, 128, 235, 34, 163, 203, 183, 122, 141, 65, 7, 253, 16, 189),
    30: (94, 169, 112, 45, 125, 122, 185, 67, 144, 237, 115, 16, 89, 8, 109, 60),
    31: (110, 5, 145, 177, 224, 122, 6, 221, 44, 227, 242, 86, 141, 84, 233, 172),
    32: (23, 83, 12, 126, 188, 47, 241, 232, 214, 115, 233, 195, 42, 145, 80, 44),
    33: (201, 106, 250, 132, 171, 25, 11, 111, 61, 183, 17, 133, 169, 161, 20, 236),
    34: (44, 133, 170, 40, 82, 89, 97, 179, 214, 101, 212, 173, 31, 150, 93, 110),
    35: (248, 100, 99, 227, 17, 131, 222, 50, 176, 228, 61, 57, 28, 231, 182, 168),
    36: (105, 82, 70, 104, 106, 200, 133, 60, 157, 206, 31, 180, 249, 205, 113, 38),
    37: (217, 182, 222, 105, 144, 162, 47, 150, 79, 78, 96, 249, 215, 99, 65, 216),
    38: (51, 41, 117, 98, 231, 152, 94, 73, 134, 0, 23, 255, 207, 156, 4, 33),
    39: (190, 79, 104, 199, 118, 196, 182, 247, 211, 5, 244, 253, 33, 227, 68, 108),
    40: (247, 46, 205, 11, 123, 146, 126, 120, 130, 173, 34, 202, 13, 221, 176, 132),
    41: (146, 177, 105, 2, 200, 235, 133, 122, 63, 7, 251, 241, 135, 164, 228, 71),
    42: (22, 234, 28, 192, 230, 82, 55, 164, 27, 140, 158, 207, 90, 255, 44, 126),
    43: (228, 173, 102, 234, 211, 254, 103, 27, 187, 183, 115, 81, 189, 145, 180, 66),
    44: (55, 184, 205, 167, 208, 155, 124, 64, 181, 25, 10, 161, 217, 58, 199, 211),
    45: (103, 170, 73, 45, 93, 25, 154, 219, 130, 226, 185, 63, 204, 29, 52, 116),
    46: (102, 196, 60, 117, 97, 73, 112, 192, 121, 214, 183, 83, 161, 129, 33, 67),
    47: (87, 232, 55, 184, 144, 126, 49, 96, 250, 0, 181, 141, 188, 243, 136, 148),
    48: (55, 121, 142, 63, 23, 175, 58, 223, 81, 170, 117, 219, 191, 212, 64, 176),
    49: (133, 110, 172, 94, 189, 37, 255, 106, 102, 231, 41, 32, 236, 188, 119, 31),
    50: (160, 171, 208, 200, 169, 139, 103, 153, 185, 99, 51, 242, 203, 250, 158, 64),
    51: (165, 191, 149, 236, 55, 84, 164, 181, 14, 35, 197, 199, 50, 254, 13, 125),
    52: (178, 219, 26, 19, 223, 99, 128, 22, 65, 202, 53, 137, 167, 61, 102, 214),
    53: (233, 14, 29, 212, 145, 115, 61, 50, 15, 182, 144, 219, 13, 246, 56, 156),
    54: (255, 69, 26, 137, 216, 29, 189, 196, 95, 32, 124, 17, 192, 51, 128, 115),
    55: (64, 254, 136, 86, 155, 21, 196, 7, 45, 158, 221, 237, 101, 176, 29, 198),
    56: (113, 35, 30, 123, 72, 2, 164, 49, 124, 132, 73, 165, 28, 58, 144, 23),
    57: (3, 128, 193, 160, 108, 65, 11, 237, 51, 194, 116, 181, 191, 82, 206, 114),
    58: (27, 63, 253, 79, 74, 31, 54, 85, 105, 254, 242, 16, 125, 216, 122, 145),
    59: (89, 127, 184, 223, 246, 27, 5, 161, 80, 214, 70, 58, 122, 176, 146, 55),
    60: (164, 14, 62, 246, 194, 53, 192, 73, 81, 98, 37, 233, 28, 118, 161, 47),
    61: (155, 56, 34, 224, 27, 255, 232, 229, 141, 67, 30, 210, 227, 176, 244, 39),
    62: (87, 120, 113, 239, 149, 74, 29, 163, 91, 148, 225, 31, 13, 8, 48, 57),
    63: (70, 59, 222, 246, 99, 23, 90, 186, 96, 80, 88, 206, 9, 212, 18, 143),
    64: (252, 183, 71, 20, 220, 101, 31, 99, 43, 94, 49, 115, 27, 53, 217, 151),
    65: (126, 2, 249, 173, 253, 183, 116, 214, 175, 127, 156, 237, 67, 160, 65, 38),
    66: (44, 181, 53, 130, 49, 247, 153, 162, 122, 167, 248, 76, 46, 126, 61, 221),
    67: (182, 147, 48, 163, 69, 22, 106, 250, 154, 43, 135, 11, 82, 57, 168, 213),
    68: (37, 154, 109, 134, 5, 91, 127, 203, 45, 163, 242, 230, 213, 48, 26, 161),
    69: (170, 86, 97, 73, 198, 206, 22, 227, 66, 24, 180, 40, 244, 171, 234, 208),
    70: (98, 149, 51, 155, 153, 35, 34, 159, 220, 64, 29, 55, 5, 200, 74, 58),
    71: (26, 160, 90, 185, 32, 61, 206, 246, 80, 245, 155, 210, 153, 222, 251, 15),
    72: (3, 53, 16, 134, 222, 70, 131, 187, 188, 161, 246, 114, 30, 196, 111, 15),
    73: (151, 119, 124, 112, 91, 3, 106, 48, 147, 23, 62, 235, 109, 93, 170, 13),
    74: (9, 14, 113, 170, 75, 242, 88, 184, 169, 227, 89, 235, 177, 252, 246, 39),
    75: (167, 92, 215, 150, 224, 107, 60, 5, 35, 69, 22, 221, 64, 132, 213, 170),
    76: (53, 143, 141, 227, 41, 152, 229, 230, 61, 106, 1, 34, 0, 80, 66, 154),
    77: (201, 170, 150, 249, 138, 133, 160, 24, 139, 95, 65, 50, 1, 237, 49, 172),
}


def dec_replicas() -> int:
    """`TRTLLM_GLM52_DEC_EXPERT_REPLICAS`, validated: 0 (off) .. `DEC_HOT_MAX_REPLICAS`; default 8."""
    r = int(os.environ.get("TRTLLM_GLM52_DEC_EXPERT_REPLICAS", "8"))
    if not 0 <= r <= DEC_HOT_MAX_REPLICAS:
        raise ValueError(
            f"TRTLLM_GLM52_DEC_EXPERT_REPLICAS must be in 0..{DEC_HOT_MAX_REPLICAS}, got {r}"
        )
    return r


def dec_hot(layer: int, r: int) -> tuple[int, ...]:
    """Layer `layer`'s `r` replica experts in slot order (empty for a layer without any)."""
    return DEC_HOT.get(int(layer), ())[:r] if r else ()


#: `layers=ALL_LAYERS` is the common case and the default. A sentinel rather
#: than a lambda so the table reads as data, and rather than `True` so the
#: field has one type of answer: which layers.
ALL_LAYERS = object()


@dataclass(frozen=True)
class W:
    """One weight role: what it is, where its bytes come from, what happens on the way.

    `shape` takes the dimension bundle and returns the declared shape.

    `src` says where the bytes come from: a checkpoint key template with `{p}`
    for the layer prefix, or a callable `(d, i) -> [(key, index), ...]`. A
    `key` may be a tuple, meaning the transform is handed those tensors
    together.

    `layers` says which layers carry this weight: `ALL_LAYERS`, `None` for a
    weight that is not per-layer, or a callable `d -> iterable[int]`.

    `shard` takes the dimension bundle and returns a function of the lazy
    checkpoint entry: this rank's tensor-parallel slice. It runs before the
    copy to the device and before `transform`, which is what `kv_b_proj`
    relies on: the regroup reads this rank's head count.

    `transform` runs on the source tensor before the copy, on the
    destination's device.
    """

    name: str
    shape: Callable[[Any], tuple[int, ...]]
    src: str | Callable
    dtype: torch.dtype | None = None
    transform: Callable | None = None
    layers: object | None = ALL_LAYERS
    shard: Callable | None = None

    def layer_indices(self, d: SimpleNamespace) -> list[int] | None:
        """The layers this weight exists on, or None if it is not per-layer."""
        if self.layers is None:
            return None
        if self.layers is ALL_LAYERS:
            return list(range(d.num_layers))
        return list(self.layers(d))

    def sources(self, d: SimpleNamespace, i: int | None, prefix: str) -> list[tuple[Any, Any]]:
        """`[(checkpoint key or key tuple, index into the param | None), ...]`."""
        if callable(self.src):
            return list(self.src(d, i))
        return [(self.src.format(p=prefix), None)]


def _rows(start, width):
    return lambda d: lambda e: e[start(d) : start(d) + width(d)]


def _cols(start, width):
    return lambda d: lambda e: e[:, start(d) : start(d) + width(d)]


def _pair_perm(rows: int, device) -> torch.Tensor:
    """`[up ; gate]` -> 64-row blocks alternating up, gate."""
    half, b = rows // 2, _PAIR_BLOCK
    up = torch.arange(half).reshape(half // b, b)
    gate = torch.arange(half, rows).reshape(half // b, b)
    return torch.stack([up, gate], dim=1).reshape(-1).to(device)


def _sf_atoms(s: torch.Tensor) -> torch.Tensor:
    """Row-major `[N, K/16]` scale bytes -> `[N/128, K/64, 512]` atoms."""
    n, sf_k = s.shape
    m0, m1, ka = _SF_ATOM_M0, _SF_ATOM_M1, _SF_ATOM_K
    return (
        s.reshape(n // (m0 * m1), m1, m0, sf_k // ka, ka)
        .permute(0, 3, 2, 1, 4)
        .contiguous()
        .reshape(n // (m0 * m1), sf_k // ka, m0 * m1 * ka)
    )


def _fc1_w(t: tuple[torch.Tensor, ...], core) -> torch.Tensor:
    """One expert's `(up, gate)` -> 64-row blocks alternating up, gate; then, when the model
    was built with `fc1_kblk`, the k-block layout both FC1 GEMMs are compiled for. Per
    expert, so the replica slots (`DEC_HOT`) get the same layout as the home ones."""
    w = torch.cat(t, dim=0)
    w = torch.index_select(w, 0, _pair_perm(w.shape[0], w.device)).contiguous()
    return _fc1_kblock.to_kblock(w) if core.fc1_kblk else w


def _fc1_s(t: tuple[torch.Tensor, ...], core) -> torch.Tensor:
    s = torch.cat([x.view(torch.uint8) for x in t], dim=0)
    return _sf_atoms(torch.index_select(s, 0, _pair_perm(s.shape[0], s.device)))


def _fc2_w(t: torch.Tensor, core) -> torch.Tensor:
    """One expert's FC2 weight, row-major as stored; when the model was built with
    `fc2_kblk`, the k-block layout the three FC2 GEMMs are compiled for. Per expert, so the
    replica slots get the same layout as the home ones."""
    return _fc1_kblock.to_kblock(t.contiguous()) if core.fc2_kblk else t


def _fc2_s(t: torch.Tensor, core) -> torch.Tensor:
    return _sf_atoms(t.view(torch.uint8))


def _mtp_fc1_w(t: tuple[torch.Tensor, ...], core) -> torch.Tensor:
    # The unpermuted layout produces wrong numbers at the right shape, so the kernel's own
    # permutation is used rather than restated.
    return _mtp_fc1.interleave_w13(torch.cat(t, dim=0), _PAIR_BLOCK, dim=0).contiguous()


def _cat_rows(t: tuple[torch.Tensor, ...], core) -> torch.Tensor:
    return torch.cat(t, dim=0).contiguous()


def _reorder_kv_b(t: torch.Tensor, core) -> torch.Tensor:
    """`[H*(nope+v), C]` per-head-interleaved -> `[H*nope ; H*v]`, over this rank's heads."""
    cfg = core.model_config.pretrained_config
    h = cfg.num_attention_heads // core.model_config.mapping.tp_size
    n, v, c = cfg.qk_nope_head_dim, cfg.v_head_dim, cfg.kv_lora_rank
    t3 = t.reshape(h, n + v, c)
    return torch.cat(
        [t3[:, :n, :].reshape(h * n, c), t3[:, n:, :].reshape(h * v, c)], dim=0
    ).contiguous()


def _indexer_fused(t: tuple[torch.Tensor, ...], core) -> torch.Tensor:
    """fp32 `[wk - mean(wk) ; weights_proj]`. Centring makes `RMSNorm(wk' h) * w + b` the
    module's LayerNorm. fp32 is also the only `weights` dtype `fp8_mqa_logits` takes: fp16
    selects an accumulator that overflows here."""
    wk, wp = (x.to(torch.float32) for x in t)
    return torch.cat([wk - wk.mean(dim=0, keepdim=True), wp], dim=0).contiguous()


def _indexer_fused_bf16(t: tuple[torch.Tensor, ...], core) -> torch.Tensor:
    """bf16 `[wk ; weights_proj]`, uncentred: the fused k entry computes the LayerNorm, which
    is invariant to the per-row constant centring removes."""
    return torch.cat([x.to(torch.bfloat16) for x in t], dim=0).contiguous()


def _to_fp32(t: torch.Tensor, core) -> torch.Tensor:
    return t.to(torch.float32).contiguous()


def _materialize(entry, shard) -> torch.Tensor:
    """`[:]` on a lazy slice, or this rank's slice of it. A 0-dim scalar rejects `[:]`."""
    if getattr(entry, "ndim", 1) == 0:
        return entry
    return entry[:] if shard is None else shard(entry)


def _trunk_layers(d):
    """Every trunk layer, plus the MTP module when it is live: its attention block is a
    trunk layer's, one index further."""
    return range(d.num_layers + (1 if d.mtp_enabled else 0))


def _indexer_layers(d):
    """Layers that own an indexer. The rest reuse the most recent one's selection. The MTP
    module owns a full one."""
    return [i for i in _trunk_layers(d) if i == d.num_layers or d.full_indexer[i]]


def _dense_layers(d):
    return range(d.dense_layers)


def _moe_layers(d):
    """Router and shared expert: the MoE trunk layers and MTP. All bf16, so one key."""
    return range(d.dense_layers, d.num_layers + (1 if d.mtp_enabled else 0))


def _routed_layers(d):
    """The NVFP4 routed experts: MoE trunk layers only."""
    return range(d.dense_layers, d.num_layers)


def _mtp_only(d):
    return [d.num_layers] if d.mtp_enabled else []


def _one(template):
    return lambda d, i, t=template: [(t.format(i=i), None)]


def _pair(a, b):
    return lambda d, i, x=a, y=b: [((x.format(i=i), y.format(i=i)), None)]


def _expert_window(suffix, *, pair_with=None):
    """This rank's routed experts, one source per local expert; then, with decode expert
    replicas, layer `i`'s hot experts in slots `64 + h` (`DEC_HOT`)."""

    def build(d, i):
        slots = [(local, d.expert_offset + local) for local in range(d.local_experts)]
        slots += [(d.local_experts + h, e) for h, e in enumerate(d.dec_hot.get(i, ()))]
        out = []
        for local, expert in slots:
            q = f"model.layers.{i}.mlp.experts.{expert}"
            key = f"{q}.{suffix}" if pair_with is None else (f"{q}.{suffix}", f"{q}.{pair_with}")
            out.append((key, (local,)))
        return out

    return build


def _expert_scalars(suffix):
    return lambda d, i: [
        (f"model.layers.{i}.mlp.experts.{e}.{suffix}", (e,)) for e in range(d.num_experts)
    ]


#: The routed experts' per-tensor scalars, (parameter suffix, checkpoint suffix). The
#: checkpoint stores reciprocals.
_EXPERT_SCALARS = (
    ("isc1", "gate_proj.input_scale"),
    ("isc1_up", "up_proj.input_scale"),
    ("ws2_1", "gate_proj.weight_scale_2"),
    ("ws2_1_up", "up_proj.weight_scale_2"),
    ("isc2", "down_proj.input_scale"),
    ("ws2_2", "down_proj.weight_scale_2"),
)


def _mlp(prefix, inter, layers):
    """A bf16 SwiGLU pair split over the intermediate dimension: `gate_up` by output rows,
    gate first (the half order `flashinfer_silu_and_mul` reads), `down` by input columns.
    The same slice applies to gate and to up before they are stacked."""
    start = lambda d: d.tp_rank * inter(d)  # noqa: E731
    return (
        W(
            "mlp_gu",
            shape=lambda d: (2 * inter(d), d.hidden),
            src=_pair(f"{prefix}.gate_proj.weight", f"{prefix}.up_proj.weight"),
            transform=_cat_rows,
            layers=layers,
            shard=_rows(start, inter),
        ),
        W(
            "mlp_dn",
            shape=lambda d: (d.hidden, inter(d)),
            src=_one(f"{prefix}.down_proj.weight"),
            layers=layers,
            shard=_cols(start, inter),
        ),
    )


class GlmMoeDsaTep4Weights:
    """glm-5.2-nvfp4 / sm_103 / tep4."""

    RELEASE_AFTER = ("_fc2_s", "_fc2")

    WEIGHTS: tuple[W, ...] = (
        # --- attention: every trunk layer, and MTP ---
        W(
            "norm1",
            shape=lambda d: (d.hidden,),
            src=_one("model.layers.{i}.input_layernorm.weight"),
            layers=_trunk_layers,
        ),
        # q_a_proj and kv_a_proj_with_mqa read the same input and are both replicated, so
        # they are one [q_lora + latent, hidden] GEMM.
        W(
            "qkva",
            shape=lambda d: (d.q_lora + d.lat_dim, d.hidden),
            src=_pair(
                "model.layers.{i}.self_attn.q_a_proj.weight",
                "model.layers.{i}.self_attn.kv_a_proj_with_mqa.weight",
            ),
            transform=_cat_rows,
            layers=_trunk_layers,
        ),
        W(
            "q_norm",
            shape=lambda d: (d.q_lora,),
            src=_one("model.layers.{i}.self_attn.q_a_layernorm.weight"),
            layers=_trunk_layers,
        ),
        W(
            "kv_norm",
            shape=lambda d: (d.kv_lora,),
            src=_one("model.layers.{i}.self_attn.kv_a_layernorm.weight"),
            layers=_trunk_layers,
        ),
        W(
            "qb",
            shape=lambda d: (d.heads * d.qk_dim, d.q_lora),
            src=_one("model.layers.{i}.self_attn.q_b_proj.weight"),
            layers=_trunk_layers,
            shard=_rows(lambda d: d.tp_rank * d.heads * d.qk_dim, lambda d: d.heads * d.qk_dim),
        ),
        W(
            "kvb",
            shape=lambda d: (d.heads * (d.nope + d.v_dim), d.kv_lora),
            src=_one("model.layers.{i}.self_attn.kv_b_proj.weight"),
            transform=_reorder_kv_b,
            layers=_trunk_layers,
            shard=_rows(
                lambda d: d.tp_rank * d.heads * (d.nope + d.v_dim),
                lambda d: d.heads * (d.nope + d.v_dim),
            ),
        ),
        W(
            "o",
            shape=lambda d: (d.hidden, d.heads * d.v_dim),
            src=_one("model.layers.{i}.self_attn.o_proj.weight"),
            layers=_trunk_layers,
            shard=_cols(lambda d: d.tp_rank * d.heads * d.v_dim, lambda d: d.heads * d.v_dim),
        ),
        W(
            "norm2",
            shape=lambda d: (d.hidden,),
            src=_one("model.layers.{i}.post_attention_layernorm.weight"),
            layers=_trunk_layers,
        ),
        # --- the DSA indexer: replicated, on the layers that own one ---
        W(
            "ix_fused",
            shape=lambda d: (d.idx_dim + d.idx_heads, d.hidden),
            dtype=torch.float32,
            src=_pair(
                "model.layers.{i}.self_attn.indexer.wk.weight",
                "model.layers.{i}.self_attn.indexer.weights_proj.weight",
            ),
            transform=_indexer_fused,
            layers=_indexer_layers,
        ),
        W(
            "ix_fused_bf16",
            shape=lambda d: (d.idx_dim + d.idx_heads, d.hidden),
            src=_pair(
                "model.layers.{i}.self_attn.indexer.wk.weight",
                "model.layers.{i}.self_attn.indexer.weights_proj.weight",
            ),
            transform=_indexer_fused_bf16,
            layers=_indexer_layers,
        ),
        W(
            "ix_qb",
            shape=lambda d: (d.idx_heads * d.idx_dim, d.q_lora),
            src=_one("model.layers.{i}.self_attn.indexer.wq_b.weight"),
            layers=_indexer_layers,
        ),
        W(
            "ix_kn_w",
            shape=lambda d: (d.idx_dim,),
            dtype=torch.float32,
            src=_one("model.layers.{i}.self_attn.indexer.k_norm.weight"),
            transform=_to_fp32,
            layers=_indexer_layers,
        ),
        W(
            "ix_kn_b",
            shape=lambda d: (d.idx_dim,),
            dtype=torch.float32,
            src=_one("model.layers.{i}.self_attn.indexer.k_norm.bias"),
            transform=_to_fp32,
            layers=_indexer_layers,
        ),
        # --- bf16 MLPs: the dense layers' own and the shared expert, one key ---
        *_mlp("model.layers.{i}.mlp", lambda d: d.dense_inter, _dense_layers),
        *_mlp("model.layers.{i}.mlp.shared_experts", lambda d: d.shared_inter, _moe_layers),
        # --- routing ---
        W(
            "router",
            shape=lambda d: (d.num_experts, d.hidden),
            src=_one("model.layers.{i}.mlp.gate.weight"),
            layers=_moe_layers,
        ),
        W(
            "router_bias",
            shape=lambda d: (d.num_experts,),
            dtype=torch.float32,
            src=_one("model.layers.{i}.mlp.gate.e_score_correction_bias"),
            layers=_moe_layers,
        ),
        # --- this rank's NVFP4 routed experts, in the vendored grouped GEMM layout ---
        W(
            "fc1_w",
            shape=lambda d: (d.local_experts + d.dec_replicas, 2 * d.moe_inter, d.hidden // 2),
            dtype=torch.uint8,
            src=_expert_window("up_proj.weight", pair_with="gate_proj.weight"),
            transform=_fc1_w,
            layers=_routed_layers,
        ),
        W(
            "fc1_s",
            shape=lambda d: (
                d.local_experts + d.dec_replicas,
                2 * d.moe_inter // SF_ATOM_MN,
                d.hidden // SF_ATOM_K,
                SF_ATOM,
            ),
            dtype=torch.uint8,
            src=_expert_window("up_proj.weight_scale", pair_with="gate_proj.weight_scale"),
            transform=_fc1_s,
            layers=_routed_layers,
        ),
        W(
            "fc2_w",
            shape=lambda d: (d.local_experts + d.dec_replicas, d.hidden, d.moe_inter // 2),
            dtype=torch.uint8,
            src=_expert_window("down_proj.weight"),
            transform=_fc2_w,
            layers=_routed_layers,
        ),
        W(
            "fc2_s",
            shape=lambda d: (
                d.local_experts + d.dec_replicas,
                d.hidden // SF_ATOM_MN,
                d.moe_inter // SF_ATOM_K,
                SF_ATOM,
            ),
            dtype=torch.uint8,
            src=_expert_window("down_proj.weight_scale"),
            transform=_fc2_s,
            layers=_routed_layers,
        ),
        *(
            W(
                f"e_{n}",
                shape=lambda d: (d.num_experts,),
                dtype=torch.float32,
                src=_expert_scalars(s),
                layers=_routed_layers,
            )
            for n, s in _EXPERT_SCALARS
        ),
        # --- the MTP module's own weights ---
        W(
            "enorm",
            shape=lambda d: (d.hidden,),
            src=_one("model.layers.{i}.enorm.weight"),
            layers=_mtp_only,
        ),
        W(
            "hnorm",
            shape=lambda d: (d.hidden,),
            src=_one("model.layers.{i}.hnorm.weight"),
            layers=_mtp_only,
        ),
        W(
            "eh",
            shape=lambda d: (d.hidden, 2 * d.hidden),
            src=_one("model.layers.{i}.eh_proj.weight"),
            layers=_mtp_only,
        ),
        W(
            "head_norm",
            shape=lambda d: (d.hidden,),
            src=_one("model.layers.{i}.shared_head.norm.weight"),
            layers=_mtp_only,
        ),
        W(
            "fc1",
            shape=lambda d: (d.local_experts, 2 * d.moe_inter, d.hidden),
            src=_expert_window("up_proj.weight", pair_with="gate_proj.weight"),
            transform=_mtp_fc1_w,
            layers=_mtp_only,
        ),
        W(
            "fc2",
            shape=lambda d: (d.local_experts, d.hidden, d.moe_inter),
            src=_expert_window("down_proj.weight"),
            layers=_mtp_only,
        ),
        # --- not per-layer ---
        W("final_norm", shape=lambda d: (d.hidden,), src="model.norm.weight", layers=None),
        W(
            "embed",
            shape=lambda d: (d.vocab, d.hidden),
            src="model.embed_tokens.weight",
            layers=None,
        ),
    )

    def declare(self, d: SimpleNamespace) -> nn.ParameterDict:
        """Allocate every weight the table declares.

        Meta-init intercepts `torch.empty` here -- real CUDA storage arrives
        when the engine materializes the registry.
        """
        w = nn.ParameterDict()
        for entry in self.WEIGHTS:
            dtype = entry.dtype or d.dtype
            for key, _, _ in self._targets(entry, d):
                w[key] = nn.Parameter(
                    torch.empty(*entry.shape(d), dtype=dtype), requires_grad=False
                )
        return w

    def _targets(self, entry: W, d: SimpleNamespace) -> list[tuple[str, int | None, str]]:
        """`[(param key, layer index | None, checkpoint layer prefix), ...]`.

        One place decides which parameters an entry declares, so `declare` and
        `manifest` cannot disagree about whether a weight exists.
        """
        layers = entry.layer_indices(d)
        if layers is None:
            return [(entry.name, None, "")]
        return [(f"l{i}_{entry.name}", i, f"model.layers.{i}") for i in layers]

    def manifest(self, core) -> dict:
        """param key -> [(checkpoint key or key tuple, index | None, transform, shard)]."""
        d = self.dims(core)
        rows: dict = {}
        for entry in self.WEIGHTS:
            shard = entry.shard(d) if entry.shard else None
            for key, i, prefix in self._targets(entry, d):
                rows[key] = [
                    (ckpt, index, entry.transform, shard)
                    for ckpt, index in entry.sources(d, i, prefix)
                ]
        return rows

    def expected_unconsumed(self, core, weights) -> set:
        """This rank's off-window experts (a layer's replicated hot experts are loaded on
        every rank), and the MTP module whole when it is not live."""
        d = self.dims(core)
        lo, hi = d.expert_offset, d.expert_offset + d.local_experts
        keys = set()
        for i in _routed_layers(d):
            for e in range(d.num_experts):
                if lo <= e < hi or e in d.dec_hot.get(i, ()):
                    continue
                for proj in ("gate_proj", "up_proj", "down_proj"):
                    keys.add(f"model.layers.{i}.mlp.experts.{e}.{proj}.weight")
                    keys.add(f"model.layers.{i}.mlp.experts.{e}.{proj}.weight_scale")
        mtp = f"model.layers.{d.num_layers}."
        if not d.mtp_enabled:
            return keys | {k for k in weights if k.startswith(mtp)}
        for e in range(d.num_experts):
            if not lo <= e < hi:
                for proj in ("gate_proj", "up_proj", "down_proj"):
                    keys.add(f"{mtp}mlp.experts.{e}.{proj}.weight")
        return keys

    def load(self, model, weights) -> None:
        """Fill every declared weight from the checkpoint: the shard column, multi-key rows,
        0-dim scalars, the vocab-parallel `lm_head`, and coverage asserted in both directions."""
        core = model.model
        manifest = self.manifest(core)
        consumed: set = set()

        def fill(param, ckpt_key, index, transform, shard) -> None:
            keys = ckpt_key if isinstance(ckpt_key, tuple) else (ckpt_key,)
            for key in keys:
                assert key in weights, f"checkpoint key missing: {key}"
            dst = param.data if index is None else param.data[index]
            # A column shard is a strided view: make it dense before a transform reads it.
            src_all = tuple(
                _materialize(weights[key], shard).to(dst.device, non_blocking=True).contiguous()
                for key in keys
            )
            if transform is not None:
                src = transform(src_all if isinstance(ckpt_key, tuple) else src_all[0], core)
            else:
                assert len(src_all) == 1, "a multi-key row needs a transform"
                src = src_all[0]
            assert dst.shape == src.shape, (ckpt_key, tuple(dst.shape), tuple(src.shape))
            assert src.dtype == dst.dtype, (ckpt_key, src.dtype, dst.dtype)
            dst.copy_(src, non_blocking=True)
            consumed.update(keys)

        assert set(manifest.keys()) == set(core.w.keys()), (
            "manifest/parameter drift",
            sorted(set(manifest.keys()) ^ set(core.w.keys()))[:12],
        )
        for param_key, sources in manifest.items():
            for ckpt_key, index, transform, shard in sources:
                fill(core.w[param_key], ckpt_key, index, transform, shard)
            # The expert relayouts allocate scratch per expert; release it per layer so peak
            # load memory stays one layer deep.
            if param_key.endswith(self.RELEASE_AFTER):
                torch.cuda.empty_cache()

        # The shell builds lm_head vocab-parallel under TP: this rank's block of rows.
        d = self.dims(core)
        shard = d.vocab // d.tp_size
        fill(
            model.lm_head.weight,
            "lm_head.weight",
            None,
            None,
            _rows(lambda _: d.tp_rank * shard, lambda _: shard)(d),
        )
        shell = {n for n, _ in model.named_parameters() if not n.startswith("model.w.")}
        assert shell == {"lm_head.weight"}, f"unfed shell parameters: {sorted(shell)}"

        torch.cuda.synchronize()
        leftover = set(weights.keys()) - consumed
        expected = self.expected_unconsumed(core, weights)
        assert leftover == expected, (
            "checkpoint coverage: unexpected "
            f"{sorted(leftover - expected)[:8]}, missed {sorted(expected - leftover)[:8]}"
        )

    def dims(self, core) -> SimpleNamespace:
        cfg = core.model_config.pretrained_config
        mapping = core.model_config.mapping
        tp = mapping.tp_size
        local_experts = cfg.n_routed_experts // mapping.moe_ep_size
        replicas = dec_replicas()
        return SimpleNamespace(
            num_layers=cfg.num_hidden_layers,
            hidden=cfg.hidden_size,
            heads=cfg.num_attention_heads // tp,
            nope=cfg.qk_nope_head_dim,
            rope_dim=cfg.qk_rope_head_dim,
            v_dim=cfg.v_head_dim,
            kv_lora=cfg.kv_lora_rank,
            q_lora=cfg.q_lora_rank,
            qk_dim=cfg.qk_nope_head_dim + cfg.qk_rope_head_dim,
            lat_dim=cfg.kv_lora_rank + cfg.qk_rope_head_dim,
            vocab=cfg.vocab_size,
            dense_layers=cfg.first_k_dense_replace,
            num_experts=cfg.n_routed_experts,
            moe_inter=cfg.moe_intermediate_size,
            shared_inter=cfg.moe_intermediate_size * cfg.n_shared_experts // tp,
            dense_inter=cfg.intermediate_size // tp,
            local_experts=local_experts,
            expert_offset=local_experts * mapping.moe_ep_rank,
            # Decode expert replicas: slots 64.. of the routed stacks, per trunk MoE layer.
            dec_replicas=replicas,
            dec_hot={i: dec_hot(i, replicas) for i in DEC_HOT} if replicas else {},
            tp_rank=mapping.tp_rank,
            tp_size=tp,
            idx_heads=cfg.index_n_heads,
            idx_dim=cfg.index_head_dim,
            full_indexer=tuple(t == "full" for t in cfg.indexer_types),
            mtp_enabled=getattr(core.model_config, "spec_config", None) is not None,
            dtype=core.model_config.torch_dtype,
        )


MODEL_WEIGHTS = GlmMoeDsaTep4Weights()
