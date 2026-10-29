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
  experts have no stock MoE fallback.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Callable

import torch
import torch.nn as nn

from .kernels.moe import mtp_moe_gather_grouped_gemm_act_fusion as _mtp_fc1

# FC1 pairs up and gate in 64-row blocks; the MTP kernel's `epi_subtile_n` is the same 64.
_PAIR_BLOCK = 64
_SF_ATOM_M0, _SF_ATOM_M1, _SF_ATOM_K = 32, 4, 4
#: The scale atom the vendored grouped GEMMs address: 128 rows by 64 K elements in 512 bytes.
SF_ATOM_MN, SF_ATOM_K, SF_ATOM = 128, 64, 512


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
    return lambda d: (lambda e: e[start(d) : start(d) + width(d)])


def _cols(start, width):
    return lambda d: (lambda e: e[:, start(d) : start(d) + width(d)])


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
    w = torch.cat(t, dim=0)
    return torch.index_select(w, 0, _pair_perm(w.shape[0], w.device)).contiguous()


def _fc1_s(t: tuple[torch.Tensor, ...], core) -> torch.Tensor:
    s = torch.cat([x.view(torch.uint8) for x in t], dim=0)
    return _sf_atoms(torch.index_select(s, 0, _pair_perm(s.shape[0], s.device)))


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
    """This rank's routed experts, one source per local expert."""

    def build(d, i):
        out = []
        for local in range(d.local_experts):
            q = f"model.layers.{i}.mlp.experts.{d.expert_offset + local}"
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
            shape=lambda d: (d.local_experts, 2 * d.moe_inter, d.hidden // 2),
            dtype=torch.uint8,
            src=_expert_window("up_proj.weight", pair_with="gate_proj.weight"),
            transform=_fc1_w,
            layers=_routed_layers,
        ),
        W(
            "fc1_s",
            shape=lambda d: (
                d.local_experts,
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
            shape=lambda d: (d.local_experts, d.hidden, d.moe_inter // 2),
            dtype=torch.uint8,
            src=_expert_window("down_proj.weight"),
            layers=_routed_layers,
        ),
        W(
            "fc2_s",
            shape=lambda d: (
                d.local_experts,
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
        """This rank's off-window experts, and the MTP module whole when it is not live."""
        d = self.dims(core)
        lo, hi = d.expert_offset, d.expert_offset + d.local_experts
        keys = set()
        for i in _routed_layers(d):
            for e in range(d.num_experts):
                if lo <= e < hi:
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
            tp_rank=mapping.tp_rank,
            tp_size=tp,
            idx_heads=cfg.index_n_heads,
            idx_dim=cfg.index_head_dim,
            full_indexer=tuple(t == "full" for t in cfg.indexer_types),
            mtp_enabled=getattr(core.model_config, "spec_config", None) is not None,
            dtype=core.model_config.torch_dtype,
        )


MODEL_WEIGHTS = GlmMoeDsaTep4Weights()
