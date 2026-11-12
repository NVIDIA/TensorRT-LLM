# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Weight table and loader: Qwen3.8-27B-NVFP4 / sm_120 / tp1.

One `W` per weight role (shape, checkpoint source, dtype, load-time transform);
`Qwen3827BTp1Weights` unrolls the table into declaration, manifest and load, then
`derive` builds the decode and prefill copies the forward runs on. tp1 means no
per-rank slicing. Storage is HF `[out, in]`.

The checkpoint is ModelOpt MIXED_PRECISION: the gated-attention q/k/v/o and the
Gated-DeltaNet in_proj_qkv / in_proj_z / out_proj are FP8 with per-tensor weight and
static input scales; the dense MLP and lm_head are NVFP4 (group 16); in_proj_a/b,
norms, conv, A_log, dt_bias and the embedding are bf16. Neither the MTP module (no
speculative decoding) nor the vision tower (the target serves text) is loaded.

Load-time transforms reproduce the built-in loader bit for bit, because every copy
`derive` builds is a function of these bytes:

* Gated-DeltaNet `in_proj_qkvz` is `[q ; k ; v ; z]` in checkpoint row order (the
  packing the built-in mapper does and the dense permutation it undoes again). Both
  halves are rescaled onto the larger weight scale in fp32 and rounded back to e4m3;
  the fused input scale is the larger one. `in_proj_ba` is `[b ; a]`.
* Attention `qkv_proj` is `[q ; k ; v]` (q interleaves q and the output gate per head),
  every shard rescaled onto the largest weight scale through bf16 in-place multiply and
  divide on the device, as the per-tensor FP8 fused-QKV loader does; the fused input
  scale is the largest one.
* MLP `gate_up` is `[gate ; up]` with the two scale-factor blocks concatenated and then
  swizzled; gate and up share the input scale and global scale. Activation scales are
  stored the way the NVFP4 GEMM reads them: `input_scale = 1 / ckpt input_scale`,
  `alpha = ckpt input_scale * weight_scale_2`.

`derive` (after load) builds, per FP8 projection, the NVFP4 W4A16 values the decode
paths read in two layouts (Marlin and the CUDA-core GEMV), and the NVFP4 W4A4 copy the
prefill-sized rows read; permutes `gate_up` rows for the SwiGLU-epilogue GEMM; and packs
lm_head for Marlin beside its checkpoint W4A4 tensors. It then drops the FP8 bytes no
path reads: all of in_proj_qkvz, out_proj and o_proj, and the q/gate rows of qkv_proj
(its K/V rows stay FP8 for the prefill-sized rows).

`load(model, weights)` fills every declared parameter once and asserts coverage both
ways: every key is consumed or in the non-load set.
"""

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Callable

import torch
from torch import nn

from .ops.quantization import _nvfp4_recipes as recipes

ALL_LAYERS = object()

_LM = "model.language_model"
_MTP = "mtp"
_VISUAL = "model.visual"

#: Elements per NVFP4 scale factor.
SF_VEC = 16

e4m3 = torch.float8_e4m3fn
f32 = torch.float32
u8 = torch.uint8


@dataclass(frozen=True)
class W:
    """One weight role: what it is, where its bytes come from, what happens on the way.

    `shape` takes the dimension bundle and returns the declared shape. `src` is a
    checkpoint key template (`{p}` is the layer prefix) or a tuple of templates handed to
    `transform` together. `layers` says which layers carry it: `ALL_LAYERS`, `None` (not
    per-layer) or a callable `d -> iterable[int]`. `transform` runs on the source tensor(s)
    on the destination's device; `None` means the checkpoint tensor is the declared shape.
    """

    name: str
    shape: Callable[[Any], tuple[int, ...]]
    src: str | tuple[str, ...]
    dtype: torch.dtype | None = None
    transform: Callable | None = None
    layers: object | None = ALL_LAYERS

    def layer_indices(self, d: SimpleNamespace) -> list[int] | None:
        if self.layers is None:
            return None
        if self.layers is ALL_LAYERS:
            return list(range(d.num_layers))
        return list(self.layers(d))

    def keys(self, prefix: str) -> tuple[str, ...]:
        templates = self.src if isinstance(self.src, tuple) else (self.src,)
        return tuple(t.format(p=prefix) for t in templates)


def _materialize(entry) -> torch.Tensor:
    """Realize one checkpoint entry: `[:]` on a lazy safetensors slice; a 0-dim scale
    rejects that index."""
    return entry if getattr(entry, "ndim", 1) == 0 else entry[:]


def _attn_layers(d):
    return [i for i in range(d.num_layers) if d.is_attn[i]]


def _gdn_layers(d):
    return [i for i in range(d.num_layers) if not d.is_attn[i]]


# ------------------------------------------------------------- transforms ---


def _qkvz_weight(t: tuple[torch.Tensor, ...]) -> torch.Tensor:
    """`[qkv ; z]` FP8 rescaled onto the larger per-tensor weight scale in fp32."""
    qkv, z, qkv_ws, z_ws = t
    smax = torch.stack([qkv_ws.to(f32), z_ws.to(f32)]).max()
    parts = []
    for w, s in ((qkv, qkv_ws.to(f32)), (z, z_ws.to(f32))):
        if not torch.equal(s, smax):
            w = (w.to(f32) * (s / smax)).to(e4m3).contiguous()
        parts.append(w)
    return torch.cat(parts, 0)


def _max_scalar(t: tuple[torch.Tensor, ...]) -> torch.Tensor:
    return torch.stack([s.to(f32) for s in t]).max().reshape([])


def _qkv_weight(t: tuple[torch.Tensor, ...]) -> torch.Tensor:
    """`[q ; k ; v]` FP8, every shard rescaled onto the largest weight scale the way the
    per-tensor FP8 fused-QKV loader does it: FP8 -> bf16, in-place multiply by the shard's
    scale, in-place divide by the largest, back to FP8 (all on the device)."""
    q, k, v, q_ws, k_ws, v_ws = t
    scales = torch.stack([q_ws.to(f32), k_ws.to(f32), v_ws.to(f32)])
    smax = scales.max()
    out = []
    for i, w in enumerate((q, k, v)):
        rescaled = w.to(torch.bfloat16).mul_(scales[i])
        out.append(rescaled.div_(smax).to(e4m3))
    return torch.cat(out, 0)


def _ba_weight(t: tuple[torch.Tensor, ...]) -> torch.Tensor:
    b, a = t
    return torch.cat([b, a], 0)


def _conv_weight(w: torch.Tensor) -> torch.Tensor:
    """`[conv_dim, 1, width]` -> `[conv_dim, width]`, contiguous."""
    return w.reshape(w.shape[0], w.shape[2]).contiguous()


def _to_f32(w: torch.Tensor) -> torch.Tensor:
    return w.to(f32)


def _gate_up_weight(t: tuple[torch.Tensor, ...]) -> torch.Tensor:
    gate, up = t
    return torch.cat([gate, up], 0)


def _swizzled_sf(t) -> torch.Tensor:
    """One or more `[rows, K/16]` e4m3 scale blocks, concatenated along rows, as the
    128x4-swizzled 1-D buffer the NVFP4 GEMMs read."""
    blocks = t if isinstance(t, tuple) else (t,)
    sf = torch.cat([b.view(u8) for b in blocks], 0)
    return torch.ops.trtllm.block_scale_interleave(sf)


def _shared_scalar(t: tuple[torch.Tensor, ...]) -> torch.Tensor:
    """One scalar both fused shards carry; the loader refuses shards that disagree."""
    first = t[0].to(f32)
    for other in t[1:]:
        assert torch.allclose(first, other.to(f32)), f"fused shards disagree: {t}"
    return first.reshape([])


def _act_scale(t) -> torch.Tensor:
    """`1 / input_scale`: the checkpoint stores amax / (448 * 6). One element, as the NVFP4
    quantize and GEMMs take it."""
    s = _shared_scalar(t if isinstance(t, tuple) else (t,))
    return (1.0 / s).reshape([1])


def _alpha(t: tuple[torch.Tensor, ...]) -> torch.Tensor:
    """`input_scale * weight_scale_2` for the shards `(input scales..., weight_scale_2s...)`."""
    half = len(t) // 2
    input_scale = _shared_scalar(t[:half])
    weight_scale_2 = _shared_scalar(t[half:])
    return (input_scale.float() * weight_scale_2.float()).reshape([1])


def _scalar(w: torch.Tensor) -> torch.Tensor:
    return w.to(f32).reshape([])


_SCALAR = lambda d: ()  # noqa: E731 - a 0-dim parameter
_ONE = lambda d: (1,)  # noqa: E731 - a one-element parameter


def _fp8_projection(tag: str, module: str, rows, cols, layers) -> tuple[W, ...]:
    """A per-tensor FP8 projection loaded as stored: weight, weight scale, input scale."""
    return (
        W(
            f"{tag}_w",
            shape=lambda d: (rows(d), cols(d)),
            dtype=e4m3,
            src=f"{{p}}.{module}.weight",
            layers=layers,
        ),
        W(
            f"{tag}_ws",
            shape=_SCALAR,
            dtype=f32,
            src=f"{{p}}.{module}.weight_scale",
            transform=_scalar,
            layers=layers,
        ),
        W(
            f"{tag}_is",
            shape=_SCALAR,
            dtype=f32,
            src=f"{{p}}.{module}.input_scale",
            transform=_scalar,
            layers=layers,
        ),
    )


def _mlp(tag: str, names: tuple[str, ...], rows, cols) -> tuple[W, ...]:
    """An NVFP4 projection (or the gate|up pair): packed weight, swizzled scale factors,
    activation scale and alpha, as the W4A4 GEMMs read them."""
    mods = tuple(f"{{p}}.mlp.{n}" for n in names)
    return (
        W(
            f"{tag}_w",
            shape=lambda d: (rows(d), cols(d) // 2),
            dtype=u8,
            src=tuple(f"{m}.weight" for m in mods),
            transform=_gate_up_weight if len(mods) > 1 else None,
        ),
        W(
            f"{tag}_sf",
            shape=lambda d: (_sf_bytes(rows(d), cols(d)),),
            dtype=u8,
            src=tuple(f"{m}.weight_scale" for m in mods),
            transform=_swizzled_sf,
        ),
        W(
            f"{tag}_is",
            shape=_ONE,
            dtype=f32,
            src=tuple(f"{m}.input_scale" for m in mods),
            transform=_act_scale,
        ),
        W(
            f"{tag}_alpha",
            shape=_ONE,
            dtype=f32,
            src=tuple(f"{m}.input_scale" for m in mods)
            + tuple(f"{m}.weight_scale_2" for m in mods),
            transform=_alpha,
        ),
    )


def _sf_bytes(rows: int, cols: int) -> int:
    """Bytes of a 128x4-swizzled scale-factor buffer for an `[rows, cols]` NVFP4 weight."""
    pad = lambda x, y: (x + y - 1) // y * y  # noqa: E731
    return pad(rows, 128) * pad(cols // SF_VEC, 4)


class Qwen3827BTp1Weights:
    """Qwen3.8-27B-NVFP4 / sm_120 / tp1.

    `declare` allocates every weight the table names, `manifest` maps checkpoint keys
    onto them, `load` fills them and `derive` builds the copies. `declare` and `manifest`
    share one "which parameters exist" decision (`_targets`).
    """

    WEIGHTS: tuple[W, ...] = (
        # --- every layer: the two gemma norms and the dense NVFP4 MLP ---
        W("in_norm", shape=lambda d: (d.hidden,), src="{p}.input_layernorm.weight"),
        W("post_norm", shape=lambda d: (d.hidden,), src="{p}.post_attention_layernorm.weight"),
        *_mlp("gate_up", ("gate_proj", "up_proj"), lambda d: 2 * d.inter, lambda d: d.hidden),
        *_mlp("down", ("down_proj",), lambda d: d.hidden, lambda d: d.inter),
        # --- the gated-attention layers ---
        W(
            "qkv_w",
            shape=lambda d: (d.qkv_rows, d.hidden),
            dtype=e4m3,
            src=tuple(
                f"{{p}}.self_attn.{n}_proj.{s}"
                for s in ("weight", "weight_scale")
                for n in ("q", "k", "v")
            ),
            transform=_qkv_weight,
            layers=_attn_layers,
        ),
        W(
            "qkv_ws",
            shape=_SCALAR,
            dtype=f32,
            src=tuple(f"{{p}}.self_attn.{n}_proj.weight_scale" for n in ("q", "k", "v")),
            transform=_max_scalar,
            layers=_attn_layers,
        ),
        W(
            "qkv_is",
            shape=_SCALAR,
            dtype=f32,
            src=tuple(f"{{p}}.self_attn.{n}_proj.input_scale" for n in ("q", "k", "v")),
            transform=_max_scalar,
            layers=_attn_layers,
        ),
        *_fp8_projection(
            "o", "self_attn.o_proj", lambda d: d.hidden, lambda d: d.q_width, _attn_layers
        ),
        W(
            "q_norm",
            shape=lambda d: (d.head_dim,),
            src="{p}.self_attn.q_norm.weight",
            layers=_attn_layers,
        ),
        W(
            "k_norm",
            shape=lambda d: (d.head_dim,),
            src="{p}.self_attn.k_norm.weight",
            layers=_attn_layers,
        ),
        # --- the Gated-DeltaNet layers ---
        W(
            "qkvz_w",
            shape=lambda d: (d.qkvz_rows, d.hidden),
            dtype=e4m3,
            src=(
                "{p}.linear_attn.in_proj_qkv.weight",
                "{p}.linear_attn.in_proj_z.weight",
                "{p}.linear_attn.in_proj_qkv.weight_scale",
                "{p}.linear_attn.in_proj_z.weight_scale",
            ),
            transform=_qkvz_weight,
            layers=_gdn_layers,
        ),
        W(
            "qkvz_ws",
            shape=_SCALAR,
            dtype=f32,
            src=(
                "{p}.linear_attn.in_proj_qkv.weight_scale",
                "{p}.linear_attn.in_proj_z.weight_scale",
            ),
            transform=_max_scalar,
            layers=_gdn_layers,
        ),
        W(
            "qkvz_is",
            shape=_SCALAR,
            dtype=f32,
            src=(
                "{p}.linear_attn.in_proj_qkv.input_scale",
                "{p}.linear_attn.in_proj_z.input_scale",
            ),
            transform=_max_scalar,
            layers=_gdn_layers,
        ),
        W(
            "ba_w",
            shape=lambda d: (2 * d.num_v_heads, d.hidden),
            src=("{p}.linear_attn.in_proj_b.weight", "{p}.linear_attn.in_proj_a.weight"),
            transform=_ba_weight,
            layers=_gdn_layers,
        ),
        W(
            "conv_w",
            shape=lambda d: (d.conv_dim, d.conv_kernel),
            src="{p}.linear_attn.conv1d.weight",
            transform=_conv_weight,
            layers=_gdn_layers,
        ),
        W(
            "A_log",
            shape=lambda d: (d.num_v_heads,),
            dtype=f32,
            src="{p}.linear_attn.A_log",
            transform=_to_f32,
            layers=_gdn_layers,
        ),
        W(
            "dt_bias",
            shape=lambda d: (d.num_v_heads,),
            dtype=f32,
            src="{p}.linear_attn.dt_bias",
            transform=_to_f32,
            layers=_gdn_layers,
        ),
        W(
            "gnorm",
            shape=lambda d: (d.head_v_dim,),
            src="{p}.linear_attn.norm.weight",
            layers=_gdn_layers,
        ),
        *_fp8_projection(
            "out", "linear_attn.out_proj", lambda d: d.hidden, lambda d: d.value_dim, _gdn_layers
        ),
        # --- not per-layer ---
        W(
            "embed",
            shape=lambda d: (d.vocab, d.hidden),
            src=f"{_LM}.embed_tokens.weight",
            layers=None,
        ),
        W("norm", shape=lambda d: (d.hidden,), src=f"{_LM}.norm.weight", layers=None),
        W(
            "lm_w",
            shape=lambda d: (d.vocab, d.hidden // 2),
            dtype=u8,
            src="lm_head.weight",
            layers=None,
        ),
        W(
            "lm_sf",
            shape=lambda d: (d.vocab, d.hidden // SF_VEC),
            dtype=e4m3,
            src="lm_head.weight_scale",
            layers=None,
        ),
        W(
            "lm_ws2",
            shape=_SCALAR,
            dtype=f32,
            src="lm_head.weight_scale_2",
            transform=_scalar,
            layers=None,
        ),
        W(
            "lm_is",
            shape=_SCALAR,
            dtype=f32,
            src="lm_head.input_scale",
            transform=_scalar,
            layers=None,
        ),
    )

    def dims(self, core) -> SimpleNamespace:
        cfg = core.model_config.pretrained_config
        key_dim = cfg.linear_num_key_heads * cfg.linear_key_head_dim
        value_dim = cfg.linear_num_value_heads * cfg.linear_value_head_dim
        q_width = cfg.num_attention_heads * cfg.head_dim
        kv_width = cfg.num_key_value_heads * cfg.head_dim
        return SimpleNamespace(
            num_layers=cfg.num_hidden_layers,
            is_attn=[t == "full_attention" for t in cfg.layer_types],
            hidden=cfg.hidden_size,
            inter=cfg.intermediate_size,
            vocab=cfg.vocab_size,
            head_dim=cfg.head_dim,
            q_width=q_width,
            # q_proj interleaves q and the output gate per head: twice q's width.
            qkv_rows=2 * q_width + 2 * kv_width,
            num_v_heads=cfg.linear_num_value_heads,
            head_v_dim=cfg.linear_value_head_dim,
            value_dim=value_dim,
            conv_dim=2 * key_dim + value_dim,
            conv_kernel=cfg.linear_conv_kernel_dim,
            qkvz_rows=2 * key_dim + 2 * value_dim,
            dtype=core.model_config.torch_dtype,
        )

    def expected_unconsumed(self, weights) -> set:
        """The MTP module (no speculative decoding) and the vision tower (the target serves
        text)."""
        return {k for k in weights if k.startswith((f"{_MTP}.", f"{_VISUAL}."))}

    def declare(self, d: SimpleNamespace) -> nn.ParameterDict:
        """Allocate every weight the table declares (meta-init intercepts `torch.empty`)."""
        w = nn.ParameterDict()
        for entry in self.WEIGHTS:
            dtype = entry.dtype or d.dtype
            for key, _, _ in self._targets(entry, d):
                w[key] = nn.Parameter(
                    torch.empty(tuple(entry.shape(d)), dtype=dtype), requires_grad=False
                )
        return w

    def _targets(self, entry: W, d: SimpleNamespace) -> list[tuple[str, int | None, str]]:
        layers = entry.layer_indices(d)
        if layers is None:
            return [(entry.name, None, "")]
        return [(f"l{i}_{entry.name}", i, f"{_LM}.layers.{i}") for i in layers]

    def manifest(self, core) -> dict[str, tuple[tuple[str, ...], Callable | None, bool]]:
        """param key -> (checkpoint keys, transform, whether the transform takes a tuple)."""
        d = self.dims(core)
        rows = {}
        for entry in self.WEIGHTS:
            for key, _, prefix in self._targets(entry, d):
                rows[key] = (entry.keys(prefix), entry.transform, isinstance(entry.src, tuple))
        return rows

    def load(self, model, weights) -> None:
        """Fill every declared parameter and assert coverage in both directions."""
        core = model.model
        manifest = self.manifest(core)
        assert set(manifest) == set(core.w.keys()), (
            "manifest/parameter drift",
            set(manifest) ^ set(core.w.keys()),
        )
        consumed: set = set()
        for param_key, (keys, transform, multi) in manifest.items():
            param = core.w[param_key]
            dst = param.data
            for key in keys:
                assert key in weights, f"checkpoint key missing: {key}"
            src_all = tuple(
                _materialize(weights[key]).to(dst.device, non_blocking=True) for key in keys
            )
            if transform is not None:
                src = transform(src_all if multi else src_all[0])
            else:
                assert len(src_all) == 1, "a multi-key weight needs a transform"
                src = src_all[0]
            assert tuple(dst.shape) == tuple(src.shape), (
                param_key,
                tuple(dst.shape),
                tuple(src.shape),
            )
            assert src.dtype == dst.dtype, (param_key, src.dtype, dst.dtype)
            dst.copy_(src, non_blocking=True)
            consumed.update(keys)
        torch.cuda.synchronize()
        leftover = set(weights.keys()) - consumed
        expected = self.expected_unconsumed(weights)
        assert leftover == expected, (
            "checkpoint coverage: leftover keys are not exactly the MTP module and the "
            f"vision tower; unexpected {sorted(leftover - expected)[:8]}, "
            f"missed {sorted(expected - leftover)[:8]}"
        )

    def derive(self, core) -> None:
        """Build the copies the forward runs on, then drop the FP8 bytes no path reads.

        Registered as new parameters of `core.w` (after materialization, so they take
        device storage directly). Per FP8 projection: `<tag>_mw` (Marlin W4A16), `<tag>_gv`
        (CUDA-core GEMV layout of the same NVFP4 values) and `<tag>_a4` (W4A4 copy).
        """
        d = self.dims(core)
        w = core.w
        derived: dict[str, Any] = {}

        def fp8_copies(tag: str, i: int) -> None:
            weight, ws, inp = w[f"l{i}_{tag}_w"], w[f"l{i}_{tag}_ws"], w[f"l{i}_{tag}_is"]
            values = recipes.fp8_to_nvfp4_values(weight.data, ws.data)
            derived[f"l{i}_{tag}_mw"] = recipes.marlin_pack(values)
            derived[f"l{i}_{tag}_gv"] = recipes.gemv_layout(values)
            derived[f"l{i}_{tag}_a4"] = recipes.fp8_to_nvfp4_w4a4(weight.data, ws.data, inp.data)

        for i in _gdn_layers(d):
            fp8_copies("qkvz", i)
            fp8_copies("out", i)
        for i in _attn_layers(d):
            fp8_copies("qkv", i)
            fp8_copies("o", i)
        for i in range(d.num_layers):
            pw, psf = recipes.permute_gate_up_for_swiglu_epilogue(
                w[f"l{i}_gate_up_w"].data, w[f"l{i}_gate_up_sf"].data, 2 * d.inter, d.hidden
            )
            w[f"l{i}_gate_up_w"].data = pw
            w[f"l{i}_gate_up_sf"].data = psf
        derived["lm_mw"] = recipes.lm_head_marlin(w["lm_w"].data, w["lm_sf"].data, w["lm_ws2"].data)
        derived["lm_a4"] = recipes.lm_head_w4a4(
            w["lm_w"].data, w["lm_sf"].data, w["lm_ws2"].data, w["lm_is"].data
        )
        core.copies = derived

        # The FP8 bytes no path reads. qkv keeps only its K/V rows, which the
        # prefill-sized rows recompute in FP8 over the W4A4 GEMM's K/V columns.
        empty = torch.empty(0, dtype=e4m3, device=w["embed"].device)
        for i in _gdn_layers(d):
            w[f"l{i}_qkvz_w"].data = empty
            w[f"l{i}_out_w"].data = empty
        for i in _attn_layers(d):
            kv = w[f"l{i}_qkv_w"].data[2 * d.q_width :].contiguous()
            w[f"l{i}_qkv_w"].data = kv
            w[f"l{i}_o_w"].data = empty
        torch.cuda.synchronize()
        torch.cuda.empty_cache()


MODEL_WEIGHTS = Qwen3827BTp1Weights()
