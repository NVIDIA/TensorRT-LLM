# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Weight table: Nemotron-3 Ultra 550B-A55B BF16 / sm_100 / dep16.

One `W` per weight role, carrying its shape, checkpoint sources, dtype and
load-time transform. dep16 is attention data-parallel over 16 ranks: every rank
runs its own requests through a whole copy of everything except the routed
experts.

* **mamba** (48 layers), **attention** (12 layers), the LatentMoE router, latent
  projections and shared expert, embeddings, final norm and `lm_head`:
  replicated, every checkpoint tensor whole on every rank -- about 63 GiB.
* **routed experts** (48 layers x 512): expert-parallel, rank r holds experts
  [32 r, 32 r + 32) -- about 60 GiB. They live in the engine's MoE layer
  (`create_moe`), which owns their parameters, so the table declares none of
  them: `load` hands each layer this rank's window of the checkpoint in the
  engine's per-expert key format, `{e}.w1.weight` = `up_proj`, `{e}.w3.weight`
  empty (the experts are non-gated Relu2) and `{e}.w2.weight` = `down_proj`, by
  global expert id -- the feed Modeling V1's NemotronH weight mapper builds.

The latent projections and the router are whole on every rank, as in Modeling
V1 (its `fc1_latent_proj` / `fc2_latent_proj` and router gate are never
sharded): the MoE layer receives the full 2048-wide latent of this rank's own
tokens and exchanges it between ranks itself.

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

# The moe_config.backend requests routing admits, and the engine MoE backend each one runs the
# routed experts on. AUTO is what the engine resolves it to for this checkpoint (CUTLASS, the
# Modeling V1 default); every request runs in the engine's MoE layer, whose all-to-all is the
# only expert exchange this target has.
MOE_BACKENDS = {
    "AUTO": "CUTLASS",
    "CUTLASS": "CUTLASS",
    "MEGAMOE_CUTEDSL": "MEGAMOE_CUTEDSL",
    "TRTLLM": "TRTLLM",
}


def engine_moe_backend(model_config) -> str:
    """The engine MoE backend the routed experts run on, from the user's request."""
    requested = str(model_config.moe_backend_requested).upper()
    assert requested in MOE_BACKENDS, (
        f"moe_config.backend={model_config.moe_backend_requested!r}: routing admits "
        f"{sorted(MOE_BACKENDS)} only"
    )
    return MOE_BACKENDS[requested]


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


def _qkv_build(d, i):
    """Every q row, then every k row, then every v row: the fused qkv operand."""
    p = f"backbone.layers.{i}.mixer"
    q, kv = d.q_width_r, d.kv_width
    return [
        (f"{p}.q_proj.weight", None, (slice(0, q),)),
        (f"{p}.k_proj.weight", None, (slice(q, q + kv),)),
        (f"{p}.v_proj.weight", None, (slice(q + kv, q + 2 * kv),)),
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


class NemotronUltraBf16Dep16Weights:
    """Nemotron-3 Ultra BF16 / sm_100 / dep16.

    `declare` allocates every weight the table names, `manifest` maps checkpoint
    keys onto them, `load` fills them and feeds the engine's MoE layers. `declare`
    and `manifest` share one "which parameters exist" decision (`_targets`), so
    they cannot disagree.
    """

    WEIGHTS: tuple[W, ...] = (
        W("embed", lambda d: (d.vocab, d.hidden), _whole("backbone.embeddings.weight"), None),
        W("final_norm", lambda d: (d.hidden,), _whole("backbone.norm_f.weight"), None),
        W("norm", lambda d: (d.hidden,), _whole(f"{_L}.norm.weight"), _all),
        # --- mamba: z | x | B | C | dt rows of in_proj and x | B | C conv channels, whole ---
        W("in_proj_w", lambda d: (d.in_proj_r, d.hidden), _whole(f"{_M}.in_proj.weight"), _mamba),
        W(
            "conv_w",
            lambda d: (d.conv_r, d.conv_kernel),
            _whole(f"{_M}.conv1d.weight"),
            _mamba,
            transform=_conv_weight,
        ),
        W("conv_b", lambda d: (d.conv_r,), _whole(f"{_M}.conv1d.bias"), _mamba),
        W(
            "A",
            lambda d: (d.heads_r,),
            _whole(f"{_M}.A_log"),
            _mamba,
            dtype=torch.float32,
            transform=_neg_exp_fp32,
        ),
        W(
            "D",
            lambda d: (d.heads_r,),
            _whole(f"{_M}.D"),
            _mamba,
            dtype=torch.float32,
            transform=_fp32,
        ),
        W(
            "dt_bias",
            lambda d: (d.heads_r,),
            _whole(f"{_M}.dt_bias"),
            _mamba,
            dtype=torch.float32,
            transform=_fp32,
        ),
        W("gnorm", lambda d: (d.d_inner_r,), _whole(f"{_M}.norm.weight"), _mamba),
        W(
            "out_proj_w",
            lambda d: (d.hidden, d.d_inner_r),
            _whole(f"{_M}.out_proj.weight"),
            _mamba,
        ),
        # --- attention: all 64 q heads and both kv heads ---
        W("qkv_w", lambda d: (d.qkv_r, d.hidden), _qkv_build, _attn),
        W("o_w", lambda d: (d.hidden, d.q_width_r), _whole(f"{_M}.o_proj.weight"), _attn),
        # --- LatentMoE: router, latent projections, shared expert ---
        W("gate_w", lambda d: (d.num_experts, d.hidden), _whole(f"{_M}.gate.weight"), _moe),
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
        W(
            "sh_up_w",
            lambda d: (d.shared_r, d.hidden),
            _whole(f"{_M}.shared_experts.up_proj.weight"),
            _moe,
        ),
        W(
            "sh_down_w",
            lambda d: (d.hidden, d.shared_r),
            _whole(f"{_M}.shared_experts.down_proj.weight"),
            _moe,
        ),
    )

    def dims(self, core) -> SimpleNamespace:
        cfg = core.model_config.pretrained_config
        mapping = core.model_config.mapping
        kinds = list(cfg.layers_block_type)
        # Attention DP: everything but the routed experts is whole on every rank, so each
        # per-rank width (`*_r`) below is the whole one.
        assert mapping.enable_attention_dp, "dep16 is the attention-data-parallel layout"
        assert mapping.moe_tp_size == 1, "dep16 runs every expert whole (moe_tp 1)"
        heads, hd = cfg.mamba_num_heads, cfg.mamba_head_dim
        groups, state = cfg.n_groups, cfg.ssm_state_size
        d_inner = heads * hd
        gn = groups * state
        q_heads = cfg.num_attention_heads
        kv_heads = cfg.num_key_value_heads
        local_experts = cfg.n_routed_experts // mapping.moe_ep_size
        assert local_experts * mapping.moe_ep_size == cfg.n_routed_experts, (
            cfg.n_routed_experts,
            mapping.moe_ep_size,
        )
        return SimpleNamespace(
            num_layers=len(kinds),
            kinds=kinds,
            mamba_layers=[i for i, k in enumerate(kinds) if k == "mamba"],
            attn_layers=[i for i, k in enumerate(kinds) if k == "attention"],
            moe_layers=[i for i, k in enumerate(kinds) if k == "moe"],
            hidden=cfg.hidden_size,
            vocab=cfg.vocab_size,
            # mamba
            d_inner=d_inner,
            d_inner_r=d_inner,
            heads_r=heads,
            head_dim_ssm=hd,
            n_groups=groups,
            groups_r=groups,
            # The gated norm's group: one SSM group's channels (32 heads x 64).
            norm_group=d_inner // groups,
            ssm_state=state,
            gn_r=gn,
            conv_r=d_inner + 2 * gn,
            in_proj_r=2 * d_inner + 2 * gn + heads,
            conv_kernel=cfg.conv_kernel,
            # attention
            head_dim=cfg.head_dim,
            q_heads_r=q_heads,
            q_width_r=q_heads * cfg.head_dim,
            kv_heads=kv_heads,
            kv_width=kv_heads * cfg.head_dim,
            qkv_r=(q_heads + 2 * kv_heads) * cfg.head_dim,
            # moe
            num_experts=cfg.n_routed_experts,
            top_k=cfg.num_experts_per_tok,
            routed_scale=float(cfg.routed_scaling_factor),
            local_experts=local_experts,
            ep_size=mapping.moe_ep_size,
            ep_rank=mapping.moe_ep_rank,
            expert_offset=local_experts * mapping.moe_ep_rank,
            moe_inter=cfg.moe_intermediate_size,
            latent=cfg.moe_latent_size,
            shared_r=cfg.moe_shared_expert_intermediate_size,
            dtype=core.model_config.torch_dtype,
            # The engine MoE backend the routed experts run on (CUTLASS for AUTO).
            engine_moe=engine_moe_backend(core.model_config),
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
    def engine_expert_feed(d: SimpleNamespace, i: int) -> list[tuple[str, str, str]]:
        """Layer `i`'s routed experts for the engine's MoE layer: `(checkpoint up key,
        checkpoint down key, engine expert prefix)` per expert of this rank's window.

        The engine's loader takes `{e}.w1.weight` (FC1), `{e}.w3.weight` (the gate half,
        empty for the non-gated Relu2) and `{e}.w2.weight` (FC2), by global expert id.
        """
        p = f"backbone.layers.{i}.mixer.experts"
        return [
            (f"{p}.{e}.up_proj.weight", f"{p}.{e}.down_proj.weight", str(e))
            for e in range(d.expert_offset, d.expert_offset + d.local_experts)
        ]

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

        # The routed experts, one engine MoE layer at a time, the scratch released after each so
        # peak load memory stays one layer deep.
        assert set(core.routed_experts.keys()) == {str(i) for i in d.moe_layers}, (
            "engine MoE layers / MoE blocks drift",
            sorted(core.routed_experts.keys()),
        )
        for i in d.moe_layers:
            fed = {}
            for up_key, down_key, e in self.engine_expert_feed(d, i):
                assert up_key in weights and down_key in weights, (
                    f"checkpoint key missing: {up_key} / {down_key}"
                )
                up = self._take(weights[up_key], None)
                fed[f"{e}.w1.weight"] = up
                fed[f"{e}.w3.weight"] = up[:0]
                fed[f"{e}.w2.weight"] = self._take(weights[down_key], None)
                consumed.update((up_key, down_key))
            core.routed_experts[str(i)].load_weights([fed])
            torch.cuda.empty_cache()

        # Shell-registered exception: the base class owns lm_head (untied), and under
        # attention DP it builds it whole -- every rank holds different tokens, so a
        # vocab shard could not be completed by a collective over rows no other rank
        # computed.
        assert tuple(model.lm_head.weight.shape) == (d.vocab, d.hidden), (
            f"dep16 expects the shell's replicated lm_head [{d.vocab}, {d.hidden}], "
            f"got {tuple(model.lm_head.weight.shape)}"
        )
        fill(model.lm_head.weight, ("lm_head.weight", None, None), None)
        # The engine MoE layers' parameters were fed above, layer by layer.
        shell = {
            n
            for n, _ in model.named_parameters()
            if not n.startswith(("model.w.", "model.routed_experts."))
        }
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


MODEL_WEIGHTS = NemotronUltraBf16Dep16Weights()
