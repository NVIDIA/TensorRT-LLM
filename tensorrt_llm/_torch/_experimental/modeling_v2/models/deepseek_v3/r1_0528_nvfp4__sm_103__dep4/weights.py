# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Weight manifest and loader: deepseek-r1-0528-nvfp4 / sm_103 / dep4.

MANIFEST is a data table: target parameter -> the checkpoint key(s) that fill
it, each with an optional destination index into the parameter and an
optional source transform.

Every rank is handed the **whole** checkpoint dict, so any split is entirely
this table's — and under attention data parallelism there is only one:

* **expert windows** — rank `r` lists only routed experts `[64r, 64r+64)` and
  writes each at local index `e - offset`. The off-window experts' six
  weight/scale keys are one of the two families a rank does not consume, and
  the coverage assert names them exactly rather than being relaxed;
* **replicated** — everything else, and that is the point of the segment: the
  topology divides the *requests*, not the weights. Both norms per layer, the
  whole 128-head MLA attention block including the q-LoRA pair, the fp8 KV
  scales, the dense MLPs of layers 0-2 and the shared-expert pair at full
  width, the router and its bias, the embedding, the final norm, the shell's
  `lm_head`, and **all 256 experts' per-tensor NVFP4 scalars** (6 fp32 values
  per expert, kept whole so the shared-expert activation-scale assert in
  `modeling.derive_after_load` still sees the max over every routed expert;
  the window is sliced there).

So this table has no per-rank row or column slices anywhere, which is exactly
what makes the `src`-transform column a tensor-parallel target needs
unnecessary here.

**The second predicted non-load is layer 61, and how much of it is non-load
depends on the config.** The checkpoint ships a bf16 multi-token-prediction
module at layer index `num_hidden_layers` — its own 256 experts,
`embed_tokens`, `eh_proj`, two extra norms and a `shared_head.head`, 790 keys.

* Under the target's identity config all 790 are an expected non-load, listed
  rather than swept under a relaxed assert.
* Under an MTP variant (`speculative_config` set) the module is loaded, and this table
  gains its rows: **212 keys per rank** are consumed (the whole front end and
  attention block, the router, the shared expert, and this rank's 64-expert
  window), leaving 578 — the 192 off-window experts' 576 weight tensors, which
  join the first non-load family above, plus exactly two that stay non-load on
  *every* rank. `embed_tokens.weight` and `shared_head.head.weight` are
  `torch.equal` to `model.embed_tokens.weight` and `lm_head.weight`, so the
  draft-model container points at the trunk's and saves 1.85 GB per rank.

The module's expert stacks land in `fused_moe`'s layout — `[E, 2I, H]` with
the **up** rows first, `[E, H, I]` for FC2 — which needs a plain concatenation
and none of the interleave / 32-row shuffle / swizzle the trtllm-gen runner's
NVFP4 stacks need: `hf_quant_config.json` excludes `model.layers.61*` from
quantization wholesale, so every weight here is bf16.

Storage is HF `[out, in]` row-major, so attention/router/embedding copies and
every NVFP4 weight-byte copy are layout-preserving. Three families need a
transform, all of them relayouts a kernel demands and no checkpoint stores:

* **`kv_b_proj` row reorder.** The checkpoint interleaves per head
  `[k_nope | v]` (row `h*(nope+v)+j`). The MLA context FMHA hard-codes V's
  row stride as the full packed width `H*(nope+v)` and reads V's per-head
  stride as `v_head_dim`, i.e. it addresses the `[.., H*nope:]` **column
  block** of the projection output. So the rows are re-grouped once at load
  into `[all heads' k_nope ; all heads' v]`; the two absorption operands
  (`k_b [H, nope, C]`, `v_b [H, v, C]`) are then plain views of the halves.

* **NVFP4 dense scales.** `nvfp4_gemm` consumes the 128x4-swizzled scale
  order; the checkpoint stores row-major `[N, K/16]`. For the fused
  `gate_up` linear the two halves' scale tensors are concatenated **before**
  the single `block_scale_interleave` — interleaving them separately and
  concatenating after produces a different byte order.

* **NVFP4 expert stacks.** Per expert, per the MoE runner's contract:
  concat `[up ; gate]` (up rows first), interleave (dest row `2i` = up `i`,
  `2i+1` = gate `i`), 32-row block shuffle (src `4u+v` -> dest `8v+u`) of
  the weight *and* scale bytes, then the 128x4 swizzle of the scales.
  FC2 skips the interleave. Expert parallelism splits the stack whole, so
  each expert's preparation is exactly the single-rank one — at this geometry
  (H=7168, I=2048) no padding is needed anywhere.

The per-tensor NVFP4 scalars (`input_scale`, `weight_scale_2`) are loaded
raw, one parameter per checkpoint key including the `up_proj` duplicates the
fused GEMM assumes equal to `gate_proj`'s; `modeling.derive_after_load`
asserts that equality and folds them into the kernels' `alpha` / `g` /
three-scalar forms. The per-layer `k_scale` / `v_scale` are loaded for the
same reason: the fp8 latent pool is only correct at a KV scaling factor of
1.0 and this target passes no scale tensors, so the value is checked there
rather than assumed here.

Loading contract: `load(model, weights)` consumes the engine-provided
checkpoint dict, fills every declared parameter exactly once, and asserts
bidirectional coverage — every target parameter written, and every checkpoint
key either consumed or in the two predicted non-load sets above.
"""

from types import SimpleNamespace

import torch

from tensorrt_llm._torch._experimental.modeling_v2._weights import ModelWeights, W

SF_BLOCK = 16  # NVFP4: one e4m3 scale per 16 elements along K


def _block32_perm(rows: int, device) -> torch.Tensor:
    """Gather index of the 32-row block shuffle: inside each aligned block of
    32 rows, source row `4u + v` (0<=u<8, 0<=v<4) lands at destination row
    `8v + u`."""
    assert rows % 32 == 0, rows
    src = torch.arange(32)
    dst = (src % 4) * 8 + src // 4
    idx = torch.empty(32, dtype=torch.long)
    idx[dst] = src
    blocks = rows // 32
    return (idx.repeat(blocks) + torch.arange(blocks).repeat_interleave(32) * 32).to(device)


def _interleave_perm(rows: int, device) -> torch.Tensor:
    """Gather index that interleaves the `[up | gate]` halves of a `2*I` row
    stack into up0, gate0, up1, gate1, ..."""
    p = torch.empty(rows, dtype=torch.long)
    p[0::2] = torch.arange(0, rows // 2)
    p[1::2] = torch.arange(rows // 2, rows)
    return p.to(device)


def _fc1_perm(rows: int, device) -> torch.Tensor:
    """Interleave then 32-row block shuffle, composed into one gather."""
    return _interleave_perm(rows, device)[_block32_perm(rows, device)]


def _swizzle(x: torch.Tensor) -> torch.Tensor:
    """128x4 block-scale swizzle of a `[M, C]` uint8 scale matrix, returned
    flat (`M % 128 == 0` and `C % 4 == 0` hold at this geometry, so the
    result is exactly `M * C` bytes)."""
    return torch.ops.trtllm.block_scale_interleave(x.contiguous())


def _reorder_kv_b(t: torch.Tensor, core) -> torch.Tensor:
    """`[H*(nope+v), C]` per-head-interleaved -> `[H*nope ; H*v]` blocks."""
    cfg = core.model_config.pretrained_config
    h, n, v, c = cfg.num_attention_heads, cfg.qk_nope_head_dim, cfg.v_head_dim, cfg.kv_lora_rank
    t3 = t.reshape(h, n + v, c)
    return torch.cat(
        [t3[:, :n, :].reshape(h * n, c), t3[:, n:, :].reshape(h * v, c)], dim=0
    ).contiguous()


def _cat_rows(t: tuple[torch.Tensor, ...], core) -> torch.Tensor:
    """Fused `gate_up` weight bytes: gate rows first (the half order
    `flashinfer_silu_and_mul` reads)."""
    return torch.cat(t, dim=0).contiguous()


def _cat_rows_swizzle(t: tuple[torch.Tensor, ...], core) -> torch.Tensor:
    """Fused `gate_up` scale bytes: concatenate, then one swizzle."""
    return _swizzle(torch.cat([x.view(torch.uint8) for x in t], dim=0))


def _swizzle_only(t: torch.Tensor, core) -> torch.Tensor:
    """`down_proj` scale bytes: swizzle in place of the row-major order."""
    return _swizzle(t.view(torch.uint8))


def _expert_fc1_w(t: tuple[torch.Tensor, ...], core) -> torch.Tensor:
    """(up, gate) packed nibbles -> one expert's FC1 operand."""
    up, gate = t
    w = torch.cat([up, gate], dim=0)
    return torch.index_select(w, 0, _fc1_perm(w.shape[0], w.device)).contiguous()


def _expert_fc1_s(t: tuple[torch.Tensor, ...], core) -> torch.Tensor:
    """(up, gate) e4m3 scales -> one expert's FC1 scale operand."""
    up, gate = t
    s = torch.cat([up.view(torch.uint8), gate.view(torch.uint8)], dim=0)
    rows, cols = s.shape
    s = torch.index_select(s, 0, _fc1_perm(rows, s.device))
    return _swizzle(s).reshape(rows, cols)


def _expert_fc2_w(t: torch.Tensor, core) -> torch.Tensor:
    """`down_proj` packed nibbles -> one expert's FC2 operand (no interleave)."""
    return torch.index_select(t, 0, _block32_perm(t.shape[0], t.device)).contiguous()


def _expert_fc2_s(t: torch.Tensor, core) -> torch.Tensor:
    """`down_proj` e4m3 scales -> one expert's FC2 scale operand."""
    s = t.view(torch.uint8)
    rows, cols = s.shape
    s = torch.index_select(s, 0, _block32_perm(rows, s.device))
    return _swizzle(s).reshape(rows, cols)


def _mtp_fc1(t: tuple[torch.Tensor, ...], core) -> torch.Tensor:
    """One MTP expert's FC1 operand: `[up ; gate]` rows, up first — the half
    order `fused_moe` reads, the opposite of the dense gate_up linear's."""
    return torch.cat(t, dim=0).contiguous()


def _materialize(entry) -> torch.Tensor:
    """Realize one checkpoint entry: `[:]` on a lazy safetensors slice, but a
    0-dim tensor (every NVFP4 per-tensor scalar and every KV scale) rejects
    that index."""
    return entry if getattr(entry, "ndim", 1) == 0 else entry[:]


def _trunk_layers(d):
    """Every trunk layer, plus the MTP module when it is live.

    The MTP module is layer `num_hidden_layers` of the same checkpoint and of the same
    KV pool, and eleven of its weights are a trunk layer's exactly -- same shape, same
    key template, same transform. They are those weights, one layer further along, not
    a second family that happens to look alike.
    """
    return range(d.num_layers + (1 if d.mtp_enabled else 0))


def _moe_layers(d):
    """Layers with a router: the trunk's MoE layers, plus MTP when live."""
    return range(d.dense_layers, d.num_layers + (1 if d.mtp_enabled else 0))


def _mtp_only(d):
    """The MTP module alone, and nothing at all when it is not live."""
    return [d.num_layers] if d.mtp_enabled else []


def _dense_mlp_layers(d):
    """The leading dense layers, whose MLP is the checkpoint's own `mlp`."""
    return range(d.dense_layers)


def _shared_expert_layers(d):
    """MoE trunk layers, whose `mlp` of the same shape is the shared expert.

    MTP is excluded: its shared expert is bf16, not NVFP4, so it is a different weight
    rather than this one at another layer.
    """
    return range(d.dense_layers, d.num_layers)


def _one(template):
    """A single checkpoint key, no index into the destination."""
    return lambda d, i, t=template: [(t.format(i=i), None)]


def _scalar(template):
    """A per-tensor scalar, written to element 0 of a one-element parameter."""
    return lambda d, i, t=template: [(t.format(i=i), (0,))]


def _pair(a, b):
    """Two checkpoint tensors handed to the transform together."""
    return lambda d, i, x=a, y=b: [((x.format(i=i), y.format(i=i)), None)]


def _expert_window(suffix, *, pair_with=None):
    """This rank's slice of the routed experts, one source per local expert.

    The checkpoint key depends on the rank -- `expert_offset` is this rank's first
    expert -- which is why the source is computed rather than templated.
    """

    def build(d, i):
        out = []
        for local in range(d.local_experts):
            q = f"model.layers.{i}.mlp.experts.{d.expert_offset + local}"
            if pair_with is None:
                out.append((f"{q}.{suffix}", (local,)))
            else:
                out.append(((f"{q}.{pair_with}", f"{q}.{suffix}"), (local,)))
        return out

    return build


def _expert_scalars(suffix):
    """A per-tensor scalar replicated over the whole routing space, not windowed.

    `derive_after_load` asserts the shared expert's activation scale against the max
    over all routed experts, so every one of them has to be present on every rank.
    """
    return lambda d, i: [
        (f"model.layers.{i}.mlp.experts.{e}.{suffix}", (e,)) for e in range(d.num_experts)
    ]


#: The six per-tensor NVFP4 scalars every MLP and every routed expert carries, as
#: (parameter suffix, checkpoint suffix). The checkpoint stores reciprocals.
_MLP_SCALARS = (
    ("isc1", "gate_proj.input_scale"),
    ("isc1_up", "up_proj.input_scale"),
    ("ws2_1", "gate_proj.weight_scale_2"),
    ("ws2_1_up", "up_proj.weight_scale_2"),
    ("isc2", "down_proj.input_scale"),
    ("ws2_2", "down_proj.weight_scale_2"),
)

_EXPERT_SCALARS = _MLP_SCALARS


def _nvfp4_mlp(key, prefix, inter, layers):
    """The ten rows of one NVFP4 SwiGLU MLP: a fused gate_up, a down, and six scalars.

    Used twice over disjoint layer ranges -- the dense layers' own `mlp` and the MoE
    layers' shared expert -- because the two have the same shape in terms of their own
    intermediate width, and different widths.
    """
    return (
        W(
            f"{key}_gu_w",
            shape=lambda d: (2 * inter(d), d.hidden // 2),
            dtype=torch.uint8,
            src=_pair(f"{prefix}.gate_proj.weight", f"{prefix}.up_proj.weight"),
            transform=_cat_rows,
            layers=layers,
        ),
        W(
            f"{key}_gu_s",
            shape=lambda d: (2 * inter(d) * (d.hidden // SF_BLOCK),),
            dtype=torch.uint8,
            src=_pair(f"{prefix}.gate_proj.weight_scale", f"{prefix}.up_proj.weight_scale"),
            transform=_cat_rows_swizzle,
            layers=layers,
        ),
        W(
            f"{key}_dn_w",
            shape=lambda d: (d.hidden, inter(d) // 2),
            dtype=torch.uint8,
            src=_one(f"{prefix}.down_proj.weight"),
            layers=layers,
        ),
        W(
            f"{key}_dn_s",
            shape=lambda d: (d.hidden * (inter(d) // SF_BLOCK),),
            dtype=torch.uint8,
            src=_one(f"{prefix}.down_proj.weight_scale"),
            transform=_swizzle_only,
            layers=layers,
        ),
        *(
            W(
                f"{key}_{n}",
                shape=lambda d: (1,),
                dtype=torch.float32,
                src=_scalar(f"{prefix}.{s}"),
                layers=layers,
            )
            for n, s in _MLP_SCALARS
        ),
    )


class DeepseekV3Dep4Weights(ModelWeights):
    """deepseek-r1-0528-nvfp4 / sm_103 / dep4."""

    RELEASE_AFTER = "_fc2_s"

    WEIGHTS: tuple[W, ...] = (
        # --- attention and the two layer norms: every trunk layer, and MTP ---
        W(
            "norm1",
            shape=lambda d: (d.hidden,),
            src=_one("model.layers.{i}.input_layernorm.weight"),
            layers=_trunk_layers,
        ),
        W(
            "qa",
            shape=lambda d: (d.q_lora, d.hidden),
            src=_one("model.layers.{i}.self_attn.q_a_proj.weight"),
            layers=_trunk_layers,
        ),
        W(
            "q_norm",
            shape=lambda d: (d.q_lora,),
            src=_one("model.layers.{i}.self_attn.q_a_layernorm.weight"),
            layers=_trunk_layers,
        ),
        W(
            "qb",
            shape=lambda d: (d.heads * d.qk_dim, d.q_lora),
            src=_one("model.layers.{i}.self_attn.q_b_proj.weight"),
            layers=_trunk_layers,
        ),
        W(
            "kva",
            shape=lambda d: (d.lat_dim, d.hidden),
            src=_one("model.layers.{i}.self_attn.kv_a_proj_with_mqa.weight"),
            layers=_trunk_layers,
        ),
        W(
            "kv_norm",
            shape=lambda d: (d.kv_lora,),
            src=_one("model.layers.{i}.self_attn.kv_a_layernorm.weight"),
            layers=_trunk_layers,
        ),
        W(
            "kvb",
            shape=lambda d: (d.heads * (d.nope + d.v_dim), d.kv_lora),
            src=_one("model.layers.{i}.self_attn.kv_b_proj.weight"),
            transform=_reorder_kv_b,
            layers=_trunk_layers,
        ),
        W(
            "o",
            shape=lambda d: (d.hidden, d.heads * d.v_dim),
            src=_one("model.layers.{i}.self_attn.o_proj.weight"),
            layers=_trunk_layers,
        ),
        # The fp8 KV-cache scales sit under the k/v projections this MLA checkpoint does
        # not otherwise have.
        W(
            "k_scale",
            shape=lambda d: (1,),
            dtype=torch.float32,
            src=_scalar("model.layers.{i}.self_attn.k_proj.k_scale"),
            layers=_trunk_layers,
        ),
        W(
            "v_scale",
            shape=lambda d: (1,),
            dtype=torch.float32,
            src=_scalar("model.layers.{i}.self_attn.v_proj.v_scale"),
            layers=_trunk_layers,
        ),
        W(
            "norm2",
            shape=lambda d: (d.hidden,),
            src=_one("model.layers.{i}.post_attention_layernorm.weight"),
            layers=_trunk_layers,
        ),
        # --- the NVFP4 MLP: the same parameter key over two layer families ---
        # Layers 0..first_k_dense_replace-1 have the checkpoint's own `mlp`; the MoE
        # layers have a shared expert of the same shape under `mlp.shared_experts`.
        # Two entries over disjoint layers rather than one: the intermediate width
        # differs, so these are different weights that share a key.
        *_nvfp4_mlp("mlp", "model.layers.{i}.mlp", lambda d: d.dense_inter, _dense_mlp_layers),
        *_nvfp4_mlp(
            "mlp",
            "model.layers.{i}.mlp.shared_experts",
            lambda d: d.shared_inter,
            _shared_expert_layers,
        ),
        # --- routing: MoE trunk layers, and MTP ---
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
        # --- this rank's routed experts, NVFP4 ---
        W(
            "fc1_w",
            shape=lambda d: (d.local_experts, 2 * d.moe_inter, d.hidden // 2),
            dtype=torch.uint8,
            src=_expert_window("gate_proj.weight", pair_with="up_proj.weight"),
            transform=_expert_fc1_w,
            layers=_shared_expert_layers,
        ),
        W(
            "fc1_s",
            shape=lambda d: (d.local_experts, 2 * d.moe_inter, d.hidden // SF_BLOCK),
            dtype=torch.uint8,
            src=_expert_window("gate_proj.weight_scale", pair_with="up_proj.weight_scale"),
            transform=_expert_fc1_s,
            layers=_shared_expert_layers,
        ),
        W(
            "fc2_w",
            shape=lambda d: (d.local_experts, d.hidden, d.moe_inter // 2),
            dtype=torch.uint8,
            src=_expert_window("down_proj.weight"),
            transform=_expert_fc2_w,
            layers=_shared_expert_layers,
        ),
        W(
            "fc2_s",
            shape=lambda d: (d.local_experts, d.hidden, d.moe_inter // SF_BLOCK),
            dtype=torch.uint8,
            src=_expert_window("down_proj.weight_scale"),
            transform=_expert_fc2_s,
            layers=_shared_expert_layers,
        ),
        # --- the routed experts' per-tensor scalars, over the whole routing space ---
        *(
            W(
                f"e_{n}",
                shape=lambda d: (d.num_experts,),
                dtype=torch.float32,
                src=_expert_scalars(s),
                layers=_shared_expert_layers,
            )
            for n, s in _EXPERT_SCALARS
        ),
        # --- the MTP module's own weights: the ones a trunk layer has no counterpart for ---
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
        # bf16, not NVFP4: the checkpoint excludes `model.layers.61*` from the export,
        # which is also why these go to `fused_moe` rather than the trtllm-gen runner.
        W(
            "sh_gu",
            shape=lambda d: (2 * d.shared_inter, d.hidden),
            src=_pair(
                "model.layers.{i}.mlp.shared_experts.gate_proj.weight",
                "model.layers.{i}.mlp.shared_experts.up_proj.weight",
            ),
            transform=_cat_rows,
            layers=_mtp_only,
        ),
        W(
            "sh_dn",
            shape=lambda d: (d.hidden, d.shared_inter),
            src=_one("model.layers.{i}.mlp.shared_experts.down_proj.weight"),
            layers=_mtp_only,
        ),
        W(
            "fc1",
            shape=lambda d: (d.local_experts, 2 * d.moe_inter, d.hidden),
            src=_expert_window("gate_proj.weight", pair_with="up_proj.weight"),
            transform=_mtp_fc1,
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

    def expected_unconsumed(self, core, weights) -> set:
        return _offwindow_expert_keys(core) | _mtp_keys(core, weights)

    def load(self, model, weights) -> None:
        """Overrides the base: multi-key rows feed their transform several tensors at
        once, the per-tensor scalars are 0-dim and reject `[:]`, and this target asserts
        coverage in both directions rather than only checkpoint-side."""
        core = model.model
        manifest = self.manifest(core)
        consumed: set = set()

        def fill(param: torch.nn.Parameter, ckpt_key, index, transform) -> None:
            keys = ckpt_key if isinstance(ckpt_key, tuple) else (ckpt_key,)
            for key in keys:
                assert key in weights, f"checkpoint key missing: {key}"
            dst = param.data if index is None else param.data[index]
            # Materialize the checkpoint entries where the destination lives: the
            # expert relayouts are row gathers over ~15 MB per expert and the
            # concatenations allocate scratch of the same order.
            src_all = tuple(
                _materialize(weights[key]).to(dst.device, non_blocking=True) for key in keys
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
            set(manifest.keys()) ^ set(core.w.keys()),
        )
        for param_key, sources in manifest.items():
            for ckpt_key, index, transform in sources:
                fill(core.w[param_key], ckpt_key, index, transform)
            # The expert relayouts allocate per-expert scratch 64 times per layer;
            # release it before the next parameter so peak load memory stays one
            # layer deep.
            if param_key.endswith(self.RELEASE_AFTER):
                torch.cuda.empty_cache()

        # Shell-registered exception: the base class owns lm_head (untied), and
        # under attention DP it builds it **whole** -- `[vocab, hidden]` on every
        # rank. That is the opposite of a tensor-parallel target, where the same
        # shell builds a vocab-parallel `[vocab/tp, hidden]`, and it is the only
        # consistent choice here: each rank holds different tokens, so a vocab
        # shard could not be completed by a collective over rows no other rank
        # computed. Asserted rather than adapted -- a shape change is a different
        # logits path.
        cfg = core.model_config.pretrained_config
        expected_lm_head_shape = (cfg.vocab_size, cfg.hidden_size)
        assert tuple(model.lm_head.weight.shape) == expected_lm_head_shape, (
            f"attention DP replicates lm_head; the shell built "
            f"{tuple(model.lm_head.weight.shape)} instead of {expected_lm_head_shape}"
        )
        fill(model.lm_head.weight, "lm_head.weight", None, None)
        # Parameter-side coverage, the other direction: the shell's construction
        # is topology-dependent, so name every registered parameter outside the
        # target's own ParameterDict instead of assuming lm_head is the only one.
        shell = {n for n, _ in model.named_parameters() if not n.startswith("model.w.")}
        assert shell == {"lm_head.weight"}, f"unfed shell parameters: {sorted(shell)}"

        torch.cuda.synchronize()
        leftover = set(weights.keys()) - consumed
        expected = self.expected_unconsumed(core, weights)
        assert leftover == expected, (
            "checkpoint coverage: leftover keys are not exactly this rank's "
            f"off-window experts plus the MTP layers; unexpected "
            f"{sorted(leftover - expected)[:8]}, missed {sorted(expected - leftover)[:8]}"
        )

    def dims(self, core) -> SimpleNamespace:
        cfg = core.model_config.pretrained_config
        mapping = core.model_config.mapping
        local_experts = cfg.n_routed_experts // mapping.moe_ep_size
        return SimpleNamespace(
            num_layers=cfg.num_hidden_layers,
            hidden=cfg.hidden_size,
            heads=cfg.num_attention_heads,
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
            shared_inter=cfg.moe_intermediate_size * cfg.n_shared_experts,
            dense_inter=cfg.intermediate_size,
            local_experts=local_experts,
            expert_offset=local_experts * mapping.moe_ep_rank,
            mtp_enabled=getattr(core.model_config, "spec_config", None) is not None,
            dtype=core.model_config.torch_dtype,
        )


MODEL_WEIGHTS = DeepseekV3Dep4Weights()


def _offwindow_expert_keys(core) -> set:
    """The first predicted non-load: the weight/scale tensors of every routed
    expert outside this rank's EP window. Their per-tensor scalars are consumed
    on every rank, so nothing else of theirs is left over.

    The trunk's MoE layers store six such tensors per expert (NVFP4 data plus
    block scales); the MTP module's are bf16, so its off-window experts leave
    three each and no `weight_scale` at all."""
    cfg = core.model_config.pretrained_config
    num_layers = cfg.num_hidden_layers
    lo, hi = core.expert_offset, core.expert_offset + core.local_experts
    layers = list(range(cfg.first_k_dense_replace, num_layers))
    if core.mtp_enabled:
        layers += list(range(num_layers, num_layers + core.mtp_layers))
    keys = set()
    for i in layers:
        quantized = i < num_layers
        for e in range(cfg.n_routed_experts):
            if lo <= e < hi:
                continue
            q = f"model.layers.{i}.mlp.experts.{e}"
            for proj in ("gate_proj", "up_proj", "down_proj"):
                keys.add(f"{q}.{proj}.weight")
                if quantized:
                    keys.add(f"{q}.{proj}.weight_scale")
    return keys


def _mtp_keys(core, weights) -> set:
    """The second predicted non-load: what the multi-token-prediction layers
    the checkpoint ships past `num_hidden_layers` leave behind. Read off the
    checkpoint by layer index rather than enumerated, so a key belonging to a
    real layer can never land here.

    With MTP off that is those layers whole — on this checkpoint layer 61's 790
    keys: its own 256 experts, embedding, `eh_proj`, norms and output head.
    With MTP on the manifest consumes all but two per rank: `embed_tokens` and
    `shared_head.head` are bitwise copies of the trunk's embedding and
    `lm_head`, which the draft-model container points at instead of loading
    them twice. (The off-window experts are left over too, but they belong to
    the family above and are named there.)"""
    num_layers = core.model_config.pretrained_config.num_hidden_layers
    prefixes = tuple(f"model.layers.{i}." for i in range(num_layers, num_layers + core.mtp_layers))
    if not prefixes:
        return set()
    keys = {k for k in weights if k.startswith(prefixes)}
    if not core.mtp_enabled:
        return keys
    aliased = (".embed_tokens.weight", ".shared_head.head.weight")
    return {k for k in keys if k.endswith(aliased)}
