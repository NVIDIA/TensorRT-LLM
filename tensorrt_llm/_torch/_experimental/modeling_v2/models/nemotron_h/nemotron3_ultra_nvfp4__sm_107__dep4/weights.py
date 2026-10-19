# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Weight table: Nemotron-3 Ultra 550B-A55B NVFP4 / sm_107 / dep4.

One `W` per weight role, carrying its shape, checkpoint sources, dtype and
load-time transform. dep4 is attention data-parallel: every rank runs its own
requests through a whole copy of everything except the routed experts.

* **mamba** (48 layers), **attention** (12 layers), the LatentMoE router, latent
  projections and shared expert, embeddings, final norm and `lm_head`: replicated,
  every checkpoint tensor whole on every rank.
* **routed experts** (48 layers x 512): expert-parallel -- rank r holds experts
  [128 r, 128 r + 128), and the forward gathers every rank's tokens to run them.

The checkpoint's MTP module (`mtp.*`) is not loaded: this target implements no
draft path, and routing refuses speculative decoding.

Scalar formats differ by kind in this checkpoint (FP8 `weight_scale` is 0-d,
FP8 `input_scale` and the KV scales are `[1]`, NVFP4 scalars are 0-d); every
scalar lands in element 0 of a one-element fp32 parameter.
"""

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Callable

import torch
import torch.nn as nn

SF_BLOCK = 16  # NVFP4: one e4m3 scale per 16 elements along K

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


def _scalar(template: str):
    """A per-tensor scalar (0-d or [1]) into element 0 of a one-element parameter."""
    return lambda d, i: [(template.format(i=i), None, (0,))]


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


def _expert_window(suffix: str):
    """This rank's routed experts, one source per local expert, into row `local`."""

    def build(d, i):
        return [
            (f"backbone.layers.{i}.mixer.experts.{d.expert_offset + e}.{suffix}", None, (e,))
            for e in range(d.local_experts)
        ]

    return build


def _expert_window_scalar(suffix: str):
    """One per-tensor scalar per local expert, into element `local`."""

    def build(d, i):
        return [
            (f"backbone.layers.{i}.mixer.experts.{d.expert_offset + e}.{suffix}", None, (e,))
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


def _qkv_segments_q(d):
    return [(d.tp_rank * d.q_width_r, d.q_width_r)]


def _qkv_build(d, i):
    """Every q row, then every k row, then every v row: the fused qkv operand."""
    p = f"backbone.layers.{i}.mixer"
    q0 = 0
    kv0 = 0
    q, kv = d.q_width_r, d.kv_width
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


def _scalar_fp32(t: torch.Tensor, core) -> torch.Tensor:
    """A 0-d or one-element scale -> a one-element fp32 tensor."""
    return t.reshape(1).float()


def _router_bf16(t: torch.Tensor, core) -> torch.Tensor:
    """The router weight is stored fp32 but every value is bf16-exact, so the
    bf16 operand with an fp32-output GEMM computes the fp32 product exactly."""
    b = t.to(torch.bfloat16)
    assert torch.equal(b.float(), t.float()), "router weight is not bf16-representable"
    return b


def _block32_perm(rows: int, device) -> torch.Tensor:
    """Gather index of the 32-row block shuffle the trtllm-gen MoE operands
    take: inside each aligned block of 32 rows, source row `4u + v` (0<=u<8,
    0<=v<4) lands at destination row `8v + u`."""
    assert rows % 32 == 0, rows
    src = torch.arange(32)
    dst = (src % 4) * 8 + src // 4
    idx = torch.empty(32, dtype=torch.long)
    idx[dst] = src
    blocks = rows // 32
    return (idx.repeat(blocks) + torch.arange(blocks).repeat_interleave(32) * 32).to(device)


def _swizzle(x: torch.Tensor) -> torch.Tensor:
    """128x4 block-scale swizzle of a `[M, C]` uint8 scale matrix (M % 128 == 0 and
    C % 4 == 0 at this geometry, so the result has exactly `M * C` bytes)."""
    return torch.ops.trtllm.block_scale_interleave(x.contiguous())


def _expert_w(t: torch.Tensor, core) -> torch.Tensor:
    """One expert's packed nibbles `[N, K/2]` -> its kernel operand.

    Both FC1 (`up_proj`, non-gated: no gate half to interleave) and FC2
    (`down_proj`) take the 32-row block shuffle only.
    """
    return torch.index_select(t, 0, _block32_perm(t.shape[0], t.device)).contiguous()


def _expert_s(t: torch.Tensor, core) -> torch.Tensor:
    """One expert's e4m3 block scales `[N, K/16]` (linear) -> its kernel operand:
    the same row shuffle as the weight, then the 128x4 swizzle."""
    s = t.view(torch.uint8)
    rows, cols = s.shape
    s = torch.index_select(s, 0, _block32_perm(rows, s.device))
    return _swizzle(s).reshape(rows, cols)


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


class NemotronUltraDep4Weights:
    """Nemotron-3 Ultra NVFP4 / sm_107 / dep4.

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
            dtype=torch.float8_e4m3fn,
        ),
        W(
            "in_proj_ws",
            lambda d: (1,),
            _scalar(f"{_M}.in_proj.weight_scale"),
            _mamba,
            dtype=torch.float32,
            transform=_scalar_fp32,
        ),
        W(
            "in_proj_is",
            lambda d: (1,),
            _scalar(f"{_M}.in_proj.input_scale"),
            _mamba,
            dtype=torch.float32,
            transform=_scalar_fp32,
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
            dtype=torch.float8_e4m3fn,
        ),
        W(
            "out_proj_ws",
            lambda d: (1,),
            _scalar(f"{_M}.out_proj.weight_scale"),
            _mamba,
            dtype=torch.float32,
            transform=_scalar_fp32,
        ),
        W(
            "out_proj_is",
            lambda d: (1,),
            _scalar(f"{_M}.out_proj.input_scale"),
            _mamba,
            dtype=torch.float32,
            transform=_scalar_fp32,
        ),
        # --- attention ---
        W("qkv_w", lambda d: (d.qkv_r, d.hidden), _qkv_build, _attn),
        W(
            "o_w",
            lambda d: (d.hidden, d.q_width_r),
            _cols(f"{_M}.o_proj.weight", lambda d: d.tp_rank * d.q_width_r, lambda d: d.q_width_r),
            _attn,
        ),
        W(
            "k_scale",
            lambda d: (1,),
            _scalar(f"{_M}.k_proj.k_scale"),
            _attn,
            dtype=torch.float32,
            transform=_scalar_fp32,
        ),
        W(
            "v_scale",
            lambda d: (1,),
            _scalar(f"{_M}.v_proj.v_scale"),
            _attn,
            dtype=torch.float32,
            transform=_scalar_fp32,
        ),
        # --- LatentMoE: router, latent projections (replicated) ---
        W(
            "gate_w",
            lambda d: (d.num_experts, d.hidden),
            _whole(f"{_M}.gate.weight"),
            _moe,
            transform=_router_bf16,
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
        # --- LatentMoE: shared expert, tensor-parallel FP8 ---
        W(
            "sh_up_w",
            lambda d: (d.shared_r, d.hidden),
            _row_segments(
                f"{_M}.shared_experts.up_proj.weight",
                lambda d: [(d.tp_rank * d.shared_r, d.shared_r)],
            ),
            _moe,
            dtype=torch.float8_e4m3fn,
        ),
        W(
            "sh_up_ws",
            lambda d: (1,),
            _scalar(f"{_M}.shared_experts.up_proj.weight_scale"),
            _moe,
            dtype=torch.float32,
            transform=_scalar_fp32,
        ),
        W(
            "sh_up_is",
            lambda d: (1,),
            _scalar(f"{_M}.shared_experts.up_proj.input_scale"),
            _moe,
            dtype=torch.float32,
            transform=_scalar_fp32,
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
            dtype=torch.float8_e4m3fn,
        ),
        W(
            "sh_down_ws",
            lambda d: (1,),
            _scalar(f"{_M}.shared_experts.down_proj.weight_scale"),
            _moe,
            dtype=torch.float32,
            transform=_scalar_fp32,
        ),
        W(
            "sh_down_is",
            lambda d: (1,),
            _scalar(f"{_M}.shared_experts.down_proj.input_scale"),
            _moe,
            dtype=torch.float32,
            transform=_scalar_fp32,
        ),
        # --- LatentMoE: this rank's routed experts, NVFP4 ---
        W(
            "ex_fc1_w",
            lambda d: (d.local_experts, d.moe_inter, d.latent // 2),
            _expert_window("up_proj.weight"),
            _moe,
            dtype=torch.uint8,
            transform=_expert_w,
        ),
        W(
            "ex_fc1_s",
            lambda d: (d.local_experts, d.moe_inter, d.latent // SF_BLOCK),
            _expert_window("up_proj.weight_scale"),
            _moe,
            dtype=torch.uint8,
            transform=_expert_s,
        ),
        W(
            "ex_fc2_w",
            lambda d: (d.local_experts, d.latent, d.moe_inter // 2),
            _expert_window("down_proj.weight"),
            _moe,
            dtype=torch.uint8,
            transform=_expert_w,
        ),
        W(
            "ex_fc2_s",
            lambda d: (d.local_experts, d.latent, d.moe_inter // SF_BLOCK),
            _expert_window("down_proj.weight_scale"),
            _moe,
            dtype=torch.uint8,
            transform=_expert_s,
        ),
        W(
            "ex_fc1_ws2",
            lambda d: (d.local_experts,),
            _expert_window_scalar("up_proj.weight_scale_2"),
            _moe,
            dtype=torch.float32,
            transform=_fp32,
        ),
        W(
            "ex_fc2_ws2",
            lambda d: (d.local_experts,),
            _expert_window_scalar("down_proj.weight_scale_2"),
            _moe,
            dtype=torch.float32,
            transform=_fp32,
        ),
        # Activation scales are one value per layer shared by every expert; each local
        # expert's copy is read so `load` can assert that instead of trusting it.
        W(
            "ex_fc1_is",
            lambda d: (d.local_experts,),
            _expert_window_scalar("up_proj.input_scale"),
            _moe,
            dtype=torch.float32,
            transform=_fp32,
        ),
        W(
            "ex_fc2_is",
            lambda d: (d.local_experts,),
            _expert_window_scalar("down_proj.input_scale"),
            _moe,
            dtype=torch.float32,
            transform=_fp32,
        ),
    )

    # Expert relayouts allocate per-expert scratch; release it after each layer's
    # last expert operand so peak load memory stays one layer deep.
    RELEASE_AFTER = "_ex_fc2_s"

    def dims(self, core) -> SimpleNamespace:
        cfg = core.model_config.pretrained_config
        mapping = core.model_config.mapping
        kinds = list(cfg.layers_block_type)
        # Attention DP: everything but the routed experts is whole on every rank.
        assert mapping.enable_attention_dp, "dep4 is the attention-data-parallel layout"
        tp, r = 1, 0
        heads, hd = cfg.mamba_num_heads, cfg.mamba_head_dim
        groups, state = cfg.n_groups, cfg.ssm_state_size
        d_inner = heads * hd
        assert heads % tp == 0 and groups % tp == 0, (heads, groups, tp)
        heads_r, d_inner_r, gn_r = heads // tp, d_inner // tp, groups // tp * state
        assert cfg.num_attention_heads % tp == 0
        q_heads_r = cfg.num_attention_heads // tp
        local_experts = cfg.n_routed_experts // mapping.moe_ep_size
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
            kv_heads=cfg.num_key_value_heads,
            kv_width=cfg.num_key_value_heads * cfg.head_dim,
            qkv_r=(q_heads_r + 2 * cfg.num_key_value_heads) * cfg.head_dim,
            # moe
            num_experts=cfg.n_routed_experts,
            top_k=cfg.num_experts_per_tok,
            routed_scale=float(cfg.routed_scaling_factor),
            local_experts=local_experts,
            expert_offset=local_experts * mapping.moe_ep_rank,
            moe_inter=cfg.moe_intermediate_size,
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
            if dst.dim() == 0 or (dst_index is not None and src.dim() == 1 and src.numel() == 1):
                src = src.reshape(dst.shape)
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
        # attention DP it builds it whole -- every rank holds different tokens, so a
        # vocab shard could not be completed by a collective over rows no other rank
        # computed.
        assert tuple(model.lm_head.weight.shape) == (d.vocab, d.hidden), (
            f"dep4 expects the shell's replicated lm_head [{d.vocab}, {d.hidden}], "
            f"got {tuple(model.lm_head.weight.shape)}"
        )
        fill(model.lm_head.weight, ("lm_head.weight", None, None), None)
        shell = {n for n, _ in model.named_parameters() if not n.startswith("model.w.")}
        assert shell == {"lm_head.weight"}, f"unfed shell parameters: {sorted(shell)}"

        torch.cuda.synchronize()
        self._check_loaded(core, d)

        leftover = set(weights.keys()) - consumed
        expected = self.expected_unconsumed(d, weights)
        assert leftover == expected, (
            "checkpoint coverage: leftover keys are not exactly the MTP module plus this "
            f"rank's off-window experts; unexpected {sorted(leftover - expected)[:8]}, "
            f"missed {sorted(expected - leftover)[:8]}"
        )

    @staticmethod
    def _check_loaded(core, d) -> None:
        """Facts the forward relies on but the checkpoint only states implicitly."""
        w = core.w
        for i in d.attn_layers:
            # The FP8 KV cache is run with no scale tensors, which reads as 1.0.
            for s in ("k_scale", "v_scale"):
                assert float(w[f"l{i}_{s}"][0]) == 1.0, (i, s, float(w[f"l{i}_{s}"][0]))
        for i in d.moe_layers:
            # One activation scale per layer and projection, shared by every expert.
            for s in ("ex_fc1_is", "ex_fc2_is"):
                v = w[f"l{i}_{s}"]
                assert bool((v == v[0]).all()), f"layer {i}: {s} differs across experts"

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


MODEL_WEIGHTS = NemotronUltraDep4Weights()
