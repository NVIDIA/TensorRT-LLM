# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Weight table: Nemotron-3 Ultra 550B-A55B BF16 / sm_100 / tp8ep2.

One `W` per weight role, carrying its shape, checkpoint sources, dtype and
load-time transform. Every rank is handed the whole checkpoint dict and takes
its own share of it:

* **mamba** (48 layers): tensor-parallel by head. `in_proj` rows are split
  *per segment* -- z | x | B | C | dt -- not contiguously, and the conv
  channels the same way (x | B | C); with 8 groups over 8 ranks every rank owns
  exactly one B/C group and one norm group. `out_proj` is split along K and its
  partial sums are all-reduced by the forward.
* **attention** (12 layers): 8 of the 64 query heads per rank; the 2 KV heads
  are replicated, rank r reading KV head r // 4. `o_proj` split along K.
* **LatentMoE** (48 layers): the routed experts are split two ways, as the
  engine's mapping lays out moe_ep 2 x moe_tp 4 -- rank r holds the 256 experts
  of window e = r % 2 (experts [256 e, 256 e + 256)), each cut to its quarter of
  the intermediate width, t = r // 2 (FC1 rows and FC2 columns
  [1280 t, 1280 t + 1280) of 5120). The shared expert is tensor-parallel over all
  8 ranks (up rows, down columns); everything else the MoE reads is replicated.
  Both partial sums join in a single all-reduce.
* embeddings, final norm: replicated; `lm_head` is the shell's vocab-parallel
  shard (rows [16384 r, 16384 r + 16384)).

Every weight is bf16 as published, with two exceptions that are exact
conversions: the SSM parameters become fp32 (`A = -exp(A_log)`, `D`,
`dt_bias`), and the router's correction bias is fp32 in the checkpoint already.
The checkpoint has no quantization scales of any kind.

The checkpoint's MTP module (`mtp.*`) is not loaded: this target implements no
draft path, and routing refuses speculative decoding.
"""

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Callable

import torch
import torch.nn as nn

# A source row: (checkpoint key, index into the checkpoint tensor | None,
# index into the destination parameter | None).
Source = tuple[str, Any, Any]


@dataclass(frozen=True)
class W:
    """One weight role.

    `shape` takes the dimension bundle and returns the declared shape. `src`
    takes the bundle and the layer index (None for a non-layer weight) and
    returns the `Source` rows that fill the parameter. `layers` takes the
    bundle and returns the layer indices that carry this weight, or is None
    for a weight that is not per-layer. `transform` runs on each source tensor
    on the destination's device before the copy.
    """

    name: str
    shape: Callable[[SimpleNamespace], tuple[int, ...]]
    src: Callable[[SimpleNamespace, int | None], list[Source]]
    layers: Callable[[SimpleNamespace], list[int]] | None
    dtype: torch.dtype | None = None
    transform: Callable | None = None


# ----------------------------------------------------------------------------
# Source helpers. `{i}` is the layer index.
# ----------------------------------------------------------------------------


def _whole(template: str):
    return lambda d, i: [(template.format(i=i), None, None)]


def _row_segments(template: str, segments: Callable):
    """Rows taken from several ranges of one checkpoint tensor, laid side by side.

    `segments(d)` returns `[(src_start, length), ...]`; the destination rows
    are packed in that order.
    """

    def build(d, i):
        out, dst = [], 0
        for start, length in segments(d):
            out.append(
                (template.format(i=i), (slice(start, start + length),), (slice(dst, dst + length),))
            )
            dst += length
        return out

    return build


def _cols(template: str, start: Callable, length: Callable):
    """A column range of one checkpoint tensor, whole destination."""
    return lambda d, i: [
        (template.format(i=i), (slice(None), slice(start(d), start(d) + length(d))), None)
    ]


def _expert_split(suffix: str, axis: int):
    """This rank's routed experts (its expert window), one source per local expert into row
    `e`, each cut to the rank's `moe_inter_r`-wide slice of the intermediate width: rows of
    the up projection (`axis` 0, `[I, H]`) or columns of the down projection (`axis` 1,
    `[H, I]`)."""

    def build(d, i):
        lo = d.moe_tp_rank * d.moe_inter_r
        window = slice(lo, lo + d.moe_inter_r)
        index = (window,) if axis == 0 else (slice(None), window)
        return [
            (f"backbone.layers.{i}.mixer.experts.{d.expert_offset + e}.{suffix}", index, (e,))
            for e in range(d.local_experts)
        ]

    return build


# ----------------------------------------------------------------------------
# Segment maps of the TP split.
# ----------------------------------------------------------------------------


def _in_proj_segments(d):
    """z | x | B | C | dt rows of `in_proj` owned by this rank, in that order."""
    D, GN, r = d.d_inner, d.n_groups * d.ssm_state, d.tp_rank
    return [
        (r * d.d_inner_r, d.d_inner_r),  # z
        (D + r * d.d_inner_r, d.d_inner_r),  # x
        (2 * D + r * d.gn_r, d.gn_r),  # B
        (2 * D + GN + r * d.gn_r, d.gn_r),  # C
        (2 * D + 2 * GN + r * d.heads_r, d.heads_r),  # dt
    ]


def _conv_segments(d):
    """x | B | C conv channels owned by this rank, in that order."""
    D, GN, r = d.d_inner, d.n_groups * d.ssm_state, d.tp_rank
    return [
        (r * d.d_inner_r, d.d_inner_r),  # x
        (D + r * d.gn_r, d.gn_r),  # B
        (D + GN + r * d.gn_r, d.gn_r),  # C
    ]


def _head_segment(d):
    return [(d.tp_rank * d.heads_r, d.heads_r)]


def _gnorm_segment(d):
    return [(d.tp_rank * d.d_inner_r, d.d_inner_r)]


def _qkv_build(d, i):
    """q rows of this rank's heads, then its replicated KV head's k and v rows."""
    p = f"backbone.layers.{i}.mixer"
    q0 = d.tp_rank * d.q_width_r
    kv0 = d.kv_head * d.head_dim
    q, kv = d.q_width_r, d.head_dim
    return [
        (f"{p}.q_proj.weight", (slice(q0, q0 + q),), (slice(0, q),)),
        (f"{p}.k_proj.weight", (slice(kv0, kv0 + kv),), (slice(q, q + kv),)),
        (f"{p}.v_proj.weight", (slice(kv0, kv0 + kv),), (slice(q + kv, q + 2 * kv),)),
    ]


# ----------------------------------------------------------------------------
# Transforms.
# ----------------------------------------------------------------------------


def _conv_weight(t: torch.Tensor, core) -> torch.Tensor:
    """`[C, 1, K]` depthwise conv weight -> `[C, K]`."""
    return t.reshape(t.shape[0], t.shape[-1]).contiguous()


def _neg_exp_fp32(t: torch.Tensor, core) -> torch.Tensor:
    """`A_log` (bf16) -> `A = -exp(A_log)` in fp32."""
    return -torch.exp(t.float())


def _fp32(t: torch.Tensor, core) -> torch.Tensor:
    return t.float()


# ----------------------------------------------------------------------------
# The table.
# ----------------------------------------------------------------------------


def _all(d):
    return list(range(d.num_layers))


def _mamba(d):
    return d.mamba_layers


def _attn(d):
    return d.attn_layers


def _moe(d):
    return d.moe_layers


_L = "backbone.layers.{i}"
_M = "backbone.layers.{i}.mixer"


class NemotronUltraBf16Tp8ep2Weights:
    """Nemotron-3 Ultra BF16 / sm_100 / tp8ep2.

    `declare` allocates every weight the table names, `manifest` maps checkpoint
    keys onto them, `load` fills them. `declare` and `manifest` share one "which
    parameters exist" decision (`_targets`), so they cannot disagree.
    """

    WEIGHTS: tuple[W, ...] = (
        W("embed", lambda d: (d.vocab, d.hidden), _whole("backbone.embeddings.weight"), None),
        W("final_norm", lambda d: (d.hidden,), _whole("backbone.norm_f.weight"), None),
        W("norm", lambda d: (d.hidden,), _whole(f"{_L}.norm.weight"), _all),
        # --- mamba ---
        W(
            "in_proj_w",
            lambda d: (d.in_proj_r, d.hidden),
            _row_segments(f"{_M}.in_proj.weight", _in_proj_segments),
            _mamba,
        ),
        W(
            "conv_w",
            lambda d: (d.conv_r, d.conv_kernel),
            lambda d, i: [
                (s, (src[0], slice(None), slice(None)), dst)
                for s, src, dst in _row_segments(f"{_M}.conv1d.weight", _conv_segments)(d, i)
            ],
            _mamba,
            transform=_conv_weight,
        ),
        W(
            "conv_b",
            lambda d: (d.conv_r,),
            _row_segments(f"{_M}.conv1d.bias", _conv_segments),
            _mamba,
        ),
        W(
            "A",
            lambda d: (d.heads_r,),
            _row_segments(f"{_M}.A_log", _head_segment),
            _mamba,
            dtype=torch.float32,
            transform=_neg_exp_fp32,
        ),
        W(
            "D",
            lambda d: (d.heads_r,),
            _row_segments(f"{_M}.D", _head_segment),
            _mamba,
            dtype=torch.float32,
            transform=_fp32,
        ),
        W(
            "dt_bias",
            lambda d: (d.heads_r,),
            _row_segments(f"{_M}.dt_bias", _head_segment),
            _mamba,
            dtype=torch.float32,
            transform=_fp32,
        ),
        W(
            "gnorm",
            lambda d: (d.d_inner_r,),
            _row_segments(f"{_M}.norm.weight", _gnorm_segment),
            _mamba,
        ),
        W(
            "out_proj_w",
            lambda d: (d.hidden, d.d_inner_r),
            _cols(
                f"{_M}.out_proj.weight", lambda d: d.tp_rank * d.d_inner_r, lambda d: d.d_inner_r
            ),
            _mamba,
        ),
        # --- attention ---
        W("qkv_w", lambda d: (d.qkv_r, d.hidden), _qkv_build, _attn),
        W(
            "o_w",
            lambda d: (d.hidden, d.q_width_r),
            _cols(f"{_M}.o_proj.weight", lambda d: d.tp_rank * d.q_width_r, lambda d: d.q_width_r),
            _attn,
        ),
        # --- LatentMoE: router, latent projections (replicated) ---
        W(
            "gate_w",
            lambda d: (d.num_experts, d.hidden),
            _whole(f"{_M}.gate.weight"),
            _moe,
        ),
        W(
            "gate_b",
            lambda d: (d.num_experts,),
            _whole(f"{_M}.gate.e_score_correction_bias"),
            _moe,
            dtype=torch.float32,
        ),
        W(
            "fc1_lat_w",
            lambda d: (d.latent, d.hidden),
            _whole(f"{_M}.fc1_latent_proj.weight"),
            _moe,
        ),
        W(
            "fc2_lat_w",
            lambda d: (d.hidden, d.latent),
            _whole(f"{_M}.fc2_latent_proj.weight"),
            _moe,
        ),
        # --- LatentMoE: shared expert, tensor-parallel ---
        W(
            "sh_up_w",
            lambda d: (d.shared_r, d.hidden),
            _row_segments(
                f"{_M}.shared_experts.up_proj.weight",
                lambda d: [(d.tp_rank * d.shared_r, d.shared_r)],
            ),
            _moe,
        ),
        W(
            "sh_down_w",
            lambda d: (d.hidden, d.shared_r),
            _cols(
                f"{_M}.shared_experts.down_proj.weight",
                lambda d: d.tp_rank * d.shared_r,
                lambda d: d.shared_r,
            ),
            _moe,
        ),
        # --- LatentMoE: this rank's routed experts. Relu2 is element-wise, so FC1 is the
        # up projection alone ([I, H] per expert, no gate half) and FC2 the down projection
        # ([H, I]); both are stacked as the checkpoint stores them. ---
        W(
            "ex_fc1_w",
            lambda d: (d.local_experts, d.moe_inter_r, d.latent),
            _expert_split("up_proj.weight", 0),
            _moe,
        ),
        W(
            "ex_fc2_w",
            lambda d: (d.local_experts, d.latent, d.moe_inter_r),
            _expert_split("down_proj.weight", 1),
            _moe,
        ),
    )

    # Release the allocator's cached scratch after each layer's last expert operand so peak
    # load memory stays one layer deep.
    RELEASE_AFTER = "_ex_fc2_w"

    def dims(self, core) -> SimpleNamespace:
        cfg = core.model_config.pretrained_config
        mapping = core.model_config.mapping
        kinds = list(cfg.layers_block_type)
        tp, r = mapping.tp_size, mapping.tp_rank
        heads, hd = cfg.mamba_num_heads, cfg.mamba_head_dim
        groups, state = cfg.n_groups, cfg.ssm_state_size
        d_inner = heads * hd
        assert heads % tp == 0 and groups % tp == 0, (heads, groups, tp)
        heads_r, d_inner_r, gn_r = heads // tp, d_inner // tp, groups // tp * state
        assert cfg.num_attention_heads % tp == 0 and tp % cfg.num_key_value_heads == 0
        q_heads_r = cfg.num_attention_heads // tp
        local_experts = cfg.n_routed_experts // mapping.moe_ep_size
        moe_tp = mapping.moe_tp_size
        assert cfg.moe_intermediate_size % moe_tp == 0, (cfg.moe_intermediate_size, moe_tp)
        return SimpleNamespace(
            num_layers=len(kinds),
            kinds=kinds,
            mamba_layers=[i for i, k in enumerate(kinds) if k == "mamba"],
            attn_layers=[i for i, k in enumerate(kinds) if k == "attention"],
            moe_layers=[i for i, k in enumerate(kinds) if k == "moe"],
            hidden=cfg.hidden_size,
            vocab=cfg.vocab_size,
            tp=tp,
            tp_rank=r,
            # mamba
            d_inner=d_inner,
            d_inner_r=d_inner_r,
            heads_r=heads_r,
            head_dim_ssm=hd,
            n_groups=groups,
            groups_r=groups // tp,
            # The gated norm's group: one SSM group's channels (32 heads x 64).
            norm_group=d_inner // groups,
            ssm_state=state,
            gn_r=gn_r,
            conv_r=d_inner_r + 2 * gn_r,
            in_proj_r=2 * d_inner_r + 2 * gn_r + heads_r,
            conv_kernel=cfg.conv_kernel,
            # attention
            head_dim=cfg.head_dim,
            q_heads_r=q_heads_r,
            q_width_r=q_heads_r * cfg.head_dim,
            kv_head=r * cfg.num_key_value_heads // tp,
            qkv_r=(q_heads_r + 2) * cfg.head_dim,
            # moe
            num_experts=cfg.n_routed_experts,
            top_k=cfg.num_experts_per_tok,
            routed_scale=float(cfg.routed_scaling_factor),
            local_experts=local_experts,
            ep_size=mapping.moe_ep_size,
            ep_rank=mapping.moe_ep_rank,
            expert_offset=local_experts * mapping.moe_ep_rank,
            moe_inter=cfg.moe_intermediate_size,
            moe_tp=moe_tp,
            moe_tp_rank=mapping.moe_tp_rank,
            moe_inter_r=cfg.moe_intermediate_size // moe_tp,
            latent=cfg.moe_latent_size,
            shared_r=cfg.moe_shared_expert_intermediate_size // tp,
            dtype=core.model_config.torch_dtype,
        )

    def _targets(self, entry: W, d: SimpleNamespace) -> list[tuple[str, int | None]]:
        """`[(param key, layer index | None), ...]` -- the one place that decides
        which parameters an entry declares."""
        if entry.layers is None:
            return [(entry.name, None)]
        return [(f"l{i}_{entry.name}", i) for i in entry.layers(d)]

    def declare(self, d: SimpleNamespace) -> nn.ParameterDict:
        """Allocate every weight the table declares (meta-init intercepts `torch.empty`)."""
        w = nn.ParameterDict()
        for entry in self.WEIGHTS:
            dtype = entry.dtype or d.dtype
            for key, _ in self._targets(entry, d):
                w[key] = nn.Parameter(
                    torch.empty(*entry.shape(d), dtype=dtype), requires_grad=False
                )
        return w

    def manifest(self, core) -> dict[str, list[tuple[Source, Callable | None]]]:
        """param key -> [(source row, transform | None)]."""
        d = self.dims(core)
        rows: dict[str, list[tuple[Source, Callable | None]]] = {}
        for entry in self.WEIGHTS:
            for key, i in self._targets(entry, d):
                rows[key] = [(s, entry.transform) for s in entry.src(d, i)]
        return rows

    @staticmethod
    def _take(entry, index) -> torch.Tensor:
        """Realize one checkpoint entry, or a slice of it. Works on lazy
        safetensors slices and on tensors alike; a 0-d scalar rejects `[:]`."""
        if index is not None:
            return entry[index]
        return entry if getattr(entry, "ndim", 1) == 0 else entry[:]

    def load(self, model, weights) -> None:
        core = model.model
        d = self.dims(core)
        manifest = self.manifest(core)
        consumed: set[str] = set()
        assert set(manifest) == set(core.w.keys()), (
            "manifest/parameter drift",
            set(manifest) ^ set(core.w.keys()),
        )

        def fill(param: torch.Tensor, row: Source, transform) -> None:
            key, src_index, dst_index = row
            assert key in weights, f"checkpoint key missing: {key}"
            dst = param.data if dst_index is None else param.data[dst_index]
            src = self._take(weights[key], src_index).to(dst.device, non_blocking=True)
            if transform is not None:
                src = transform(src, core)
            assert dst.shape == src.shape, (key, tuple(dst.shape), tuple(src.shape))
            assert src.dtype == dst.dtype, (key, src.dtype, dst.dtype)
            dst.copy_(src, non_blocking=True)
            consumed.add(key)

        for param_key, sources in manifest.items():
            for row, transform in sources:
                fill(core.w[param_key], row, transform)
            if param_key.endswith(self.RELEASE_AFTER):
                torch.cuda.empty_cache()

        # Shell-registered exception: the base class owns lm_head (untied), and under
        # tensor parallelism without attention DP it builds a vocab-parallel shard.
        vocab_r = d.vocab // d.tp
        assert tuple(model.lm_head.weight.shape) == (vocab_r, d.hidden), (
            f"tp{d.tp}ep{d.ep_size} expects the shell's vocab-parallel lm_head [{vocab_r}, {d.hidden}], "
            f"got {tuple(model.lm_head.weight.shape)}"
        )
        fill(
            model.lm_head.weight,
            ("lm_head.weight", (slice(d.tp_rank * vocab_r, (d.tp_rank + 1) * vocab_r),), None),
            None,
        )
        shell = {n for n, _ in model.named_parameters() if not n.startswith("model.w.")}
        assert shell == {"lm_head.weight"}, f"unfed shell parameters: {sorted(shell)}"

        torch.cuda.synchronize()

        leftover = set(weights.keys()) - consumed
        expected = self.expected_unconsumed(d, weights)
        assert leftover == expected, (
            "checkpoint coverage: leftover keys are not exactly the MTP module plus this "
            f"rank's off-window experts; unexpected {sorted(leftover - expected)[:8]}, "
            f"missed {sorted(expected - leftover)[:8]}"
        )

    @staticmethod
    def expected_unconsumed(d, weights) -> set[str]:
        """The MTP module, every routed-expert tensor outside this rank's window, and
        any layer past the configured stack (a checkpoint directory holding a prefix of
        the layers can still carry a later layer in a shared shard)."""
        lo, hi = d.expert_offset, d.expert_offset + d.local_experts
        out = set()
        for key in weights.keys():
            if key.startswith("mtp."):
                out.add(key)
                continue
            parts = key.split(".")
            if len(parts) > 2 and parts[1] == "layers" and int(parts[2]) >= d.num_layers:
                out.add(key)
                continue
            # backbone.layers.{i}.mixer.experts.{e}.<proj>.<tensor>
            if len(parts) > 5 and parts[3] == "mixer" and parts[4] == "experts":
                if not lo <= int(parts[5]) < hi:
                    out.add(key)
        return out


MODEL_WEIGHTS = NemotronUltraBf16Tp8ep2Weights()
