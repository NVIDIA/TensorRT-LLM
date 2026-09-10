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
* Under a `configs/mtp*.yaml` variant the module is loaded, and this table
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

import torch

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
    h, n, v, c = core.heads, core.nope, core.v_dim, core.kv_lora
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


def _mtp_rows(rows: dict, core) -> None:
    """The MTP module's manifest rows, at layer index `num_hidden_layers`.
    Everything here is bf16 and stored HF `[out, in]`, so the only transforms
    are the same `kv_b_proj` row regroup the trunk's attention needs and two
    plain row concatenations."""
    p = f"model.layers.{core.num_layers}"
    rows["mtp_enorm"] = [(f"{p}.enorm.weight", None, None)]
    rows["mtp_hnorm"] = [(f"{p}.hnorm.weight", None, None)]
    rows["mtp_eh"] = [(f"{p}.eh_proj.weight", None, None)]
    rows["mtp_norm1"] = [(f"{p}.input_layernorm.weight", None, None)]
    rows["mtp_qa"] = [(f"{p}.self_attn.q_a_proj.weight", None, None)]
    rows["mtp_q_norm"] = [(f"{p}.self_attn.q_a_layernorm.weight", None, None)]
    rows["mtp_qb"] = [(f"{p}.self_attn.q_b_proj.weight", None, None)]
    rows["mtp_kva"] = [(f"{p}.self_attn.kv_a_proj_with_mqa.weight", None, None)]
    rows["mtp_kv_norm"] = [(f"{p}.self_attn.kv_a_layernorm.weight", None, None)]
    rows["mtp_kvb"] = [(f"{p}.self_attn.kv_b_proj.weight", None, _reorder_kv_b)]
    rows["mtp_o"] = [(f"{p}.self_attn.o_proj.weight", None, None)]
    rows["mtp_k_scale"] = [(f"{p}.self_attn.k_proj.k_scale", (0,), None)]
    rows["mtp_v_scale"] = [(f"{p}.self_attn.v_proj.v_scale", (0,), None)]
    rows["mtp_norm2"] = [(f"{p}.post_attention_layernorm.weight", None, None)]
    rows["mtp_head_norm"] = [(f"{p}.shared_head.norm.weight", None, None)]
    rows["mtp_router"] = [(f"{p}.mlp.gate.weight", None, None)]
    rows["mtp_router_bias"] = [(f"{p}.mlp.gate.e_score_correction_bias", None, None)]
    s = f"{p}.mlp.shared_experts"
    rows["mtp_sh_gu"] = [((f"{s}.gate_proj.weight", f"{s}.up_proj.weight"), None, _cat_rows)]
    rows["mtp_sh_dn"] = [(f"{s}.down_proj.weight", None, None)]
    fc1, fc2 = [], []
    for local in range(core.local_experts):
        q = f"{p}.mlp.experts.{core.expert_offset + local}"
        fc1.append(((f"{q}.up_proj.weight", f"{q}.gate_proj.weight"), (local,), _mtp_fc1))
        fc2.append((f"{q}.down_proj.weight", (local,), None))
    rows["mtp_fc1"] = fc1
    rows["mtp_fc2"] = fc2


def _dense_mlp_rows(rows: dict, key: str, prefix: str) -> None:
    """Manifest rows shared by the dense MLPs of layers 0-2 and the shared
    experts: one fused gate_up NVFP4 linear plus one down NVFP4 linear. Both
    are replicated and run over this rank's own tokens."""
    rows[f"{key}_gu_w"] = [
        ((f"{prefix}.gate_proj.weight", f"{prefix}.up_proj.weight"), None, _cat_rows)
    ]
    rows[f"{key}_gu_s"] = [
        (
            (f"{prefix}.gate_proj.weight_scale", f"{prefix}.up_proj.weight_scale"),
            None,
            _cat_rows_swizzle,
        )
    ]
    rows[f"{key}_dn_w"] = [(f"{prefix}.down_proj.weight", None, None)]
    rows[f"{key}_dn_s"] = [(f"{prefix}.down_proj.weight_scale", None, _swizzle_only)]
    rows[f"{key}_isc1"] = [(f"{prefix}.gate_proj.input_scale", (0,), None)]
    rows[f"{key}_isc1_up"] = [(f"{prefix}.up_proj.input_scale", (0,), None)]
    rows[f"{key}_ws2_1"] = [(f"{prefix}.gate_proj.weight_scale_2", (0,), None)]
    rows[f"{key}_ws2_1_up"] = [(f"{prefix}.up_proj.weight_scale_2", (0,), None)]
    rows[f"{key}_isc2"] = [(f"{prefix}.down_proj.input_scale", (0,), None)]
    rows[f"{key}_ws2_2"] = [(f"{prefix}.down_proj.weight_scale_2", (0,), None)]


def _materialize(entry) -> torch.Tensor:
    """Realize one checkpoint entry: `[:]` on a lazy safetensors slice, but a
    0-dim tensor (every NVFP4 per-tensor scalar and every KV scale) rejects
    that index."""
    return entry if getattr(entry, "ndim", 1) == 0 else entry[:]


def _manifest(core) -> dict:
    """target param key -> list of (ckpt key or key tuple, index into the
    param | None, source transform | None)."""
    rows: dict = {}
    for i in range(core.num_layers):
        p = f"model.layers.{i}"
        rows[f"l{i}_norm1"] = [(f"{p}.input_layernorm.weight", None, None)]
        rows[f"l{i}_qa"] = [(f"{p}.self_attn.q_a_proj.weight", None, None)]
        rows[f"l{i}_q_norm"] = [(f"{p}.self_attn.q_a_layernorm.weight", None, None)]
        rows[f"l{i}_qb"] = [(f"{p}.self_attn.q_b_proj.weight", None, None)]
        rows[f"l{i}_kva"] = [(f"{p}.self_attn.kv_a_proj_with_mqa.weight", None, None)]
        rows[f"l{i}_kv_norm"] = [(f"{p}.self_attn.kv_a_layernorm.weight", None, None)]
        rows[f"l{i}_kvb"] = [(f"{p}.self_attn.kv_b_proj.weight", None, _reorder_kv_b)]
        rows[f"l{i}_o"] = [(f"{p}.self_attn.o_proj.weight", None, None)]
        # The fp8 KV-cache scales sit under the k/v projections the MLA
        # checkpoint does not otherwise have.
        rows[f"l{i}_k_scale"] = [(f"{p}.self_attn.k_proj.k_scale", (0,), None)]
        rows[f"l{i}_v_scale"] = [(f"{p}.self_attn.v_proj.v_scale", (0,), None)]
        rows[f"l{i}_norm2"] = [(f"{p}.post_attention_layernorm.weight", None, None)]
        if i < core.dense_layers:
            _dense_mlp_rows(rows, f"l{i}_mlp", f"{p}.mlp")
            continue
        _dense_mlp_rows(rows, f"l{i}_mlp", f"{p}.mlp.shared_experts")
        rows[f"l{i}_router"] = [(f"{p}.mlp.gate.weight", None, None)]
        rows[f"l{i}_router_bias"] = [(f"{p}.mlp.gate.e_score_correction_bias", None, None)]
        fc1_w, fc1_s, fc2_w, fc2_s = [], [], [], []
        for local in range(core.local_experts):
            q = f"{p}.mlp.experts.{core.expert_offset + local}"
            fc1_w.append(
                (
                    (f"{q}.up_proj.weight", f"{q}.gate_proj.weight"),
                    (local,),
                    _expert_fc1_w,
                )
            )
            fc1_s.append(
                (
                    (f"{q}.up_proj.weight_scale", f"{q}.gate_proj.weight_scale"),
                    (local,),
                    _expert_fc1_s,
                )
            )
            fc2_w.append((f"{q}.down_proj.weight", (local,), _expert_fc2_w))
            fc2_s.append((f"{q}.down_proj.weight_scale", (local,), _expert_fc2_s))
        rows[f"l{i}_fc1_w"] = fc1_w
        rows[f"l{i}_fc1_s"] = fc1_s
        rows[f"l{i}_fc2_w"] = fc2_w
        rows[f"l{i}_fc2_s"] = fc2_s
        # The per-tensor scalars are replicated over the whole routing space,
        # not windowed: derive_after_load asserts the shared-expert
        # activation scale against the max over all 256 routed experts.
        isc1, isc1_up, ws2_1, ws2_1_up, isc2, ws2_2 = [], [], [], [], [], []
        for e in range(core.num_experts):
            q = f"{p}.mlp.experts.{e}"
            isc1.append((f"{q}.gate_proj.input_scale", (e,), None))
            isc1_up.append((f"{q}.up_proj.input_scale", (e,), None))
            ws2_1.append((f"{q}.gate_proj.weight_scale_2", (e,), None))
            ws2_1_up.append((f"{q}.up_proj.weight_scale_2", (e,), None))
            isc2.append((f"{q}.down_proj.input_scale", (e,), None))
            ws2_2.append((f"{q}.down_proj.weight_scale_2", (e,), None))
        rows[f"l{i}_e_isc1"] = isc1
        rows[f"l{i}_e_isc1_up"] = isc1_up
        rows[f"l{i}_e_ws2_1"] = ws2_1
        rows[f"l{i}_e_ws2_1_up"] = ws2_1_up
        rows[f"l{i}_e_isc2"] = isc2
        rows[f"l{i}_e_ws2_2"] = ws2_2
    rows["final_norm"] = [("model.norm.weight", None, None)]
    rows["embed"] = [("model.embed_tokens.weight", None, None)]
    if core.mtp_enabled:
        _mtp_rows(rows, core)
    return rows


def _offwindow_expert_keys(core) -> set:
    """The first predicted non-load: the weight/scale tensors of every routed
    expert outside this rank's EP window. Their per-tensor scalars are consumed
    on every rank, so nothing else of theirs is left over.

    The trunk's MoE layers store six such tensors per expert (NVFP4 data plus
    block scales); the MTP module's are bf16, so its off-window experts leave
    three each and no `weight_scale` at all."""
    lo, hi = core.expert_offset, core.expert_offset + core.local_experts
    layers = list(range(core.dense_layers, core.num_layers))
    if core.mtp_enabled:
        layers += list(range(core.num_layers, core.num_layers + core.mtp_layers))
    keys = set()
    for i in layers:
        quantized = i < core.num_layers
        for e in range(core.num_experts):
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
    prefixes = tuple(
        f"model.layers.{i}." for i in range(core.num_layers, core.num_layers + core.mtp_layers)
    )
    if not prefixes:
        return set()
    keys = {k for k in weights if k.startswith(prefixes)}
    if not core.mtp_enabled:
        return keys
    aliased = (".embed_tokens.weight", ".shared_head.head.weight")
    return {k for k in keys if k.endswith(aliased)}


def load(model, weights) -> None:
    core = model.model
    manifest = _manifest(core)
    consumed: set = set()

    def fill(param: torch.nn.Parameter, ckpt_key, index, transform) -> None:
        keys = ckpt_key if isinstance(ckpt_key, tuple) else (ckpt_key,)
        for key in keys:
            assert key in weights, f"checkpoint key missing: {key}"
        dst = param.data if index is None else param.data[index]
        # Materialize the checkpoint entries where the destination lives: the
        # expert relayouts are row gathers over ~15 MB per expert and the
        # concatenations allocate scratch of the same order. The per-tensor
        # scalars are stored 0-dim, which `[:]` rejects.
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
        # The expert relayouts allocate per-expert scratch 64 times per
        # layer; release it before the next parameter so peak load memory
        # stays one layer deep.
        if param_key.endswith("_fc2_s"):
            torch.cuda.empty_cache()

    # Shell-registered exception: the base class owns lm_head (untied), and
    # under attention DP it builds it **whole** — `[vocab, hidden]` on every
    # rank. That is the opposite of a tensor-parallel target, where the same
    # shell builds a vocab-parallel `[vocab/tp, hidden]`, and it is the only
    # consistent choice here: each rank holds different tokens, so a vocab
    # shard could not be completed by a collective over rows no other rank
    # computed. Asserted rather than adapted — a shape change is a different
    # logits path.
    assert tuple(model.lm_head.weight.shape) == (core.vocab, core.hidden), (
        f"attention DP replicates lm_head; the shell built "
        f"{tuple(model.lm_head.weight.shape)} instead of {(core.vocab, core.hidden)}"
    )
    fill(model.lm_head.weight, "lm_head.weight", None, None)
    # Parameter-side coverage, the other direction: the shell's construction
    # is topology-dependent, so name every registered parameter outside the
    # target's own ParameterDict instead of assuming lm_head is the only one.
    shell = {n for n, _ in model.named_parameters() if not n.startswith("model.w.")}
    assert shell == {"lm_head.weight"}, f"unfed shell parameters: {sorted(shell)}"

    torch.cuda.synchronize()
    leftover = set(weights.keys()) - consumed
    expected = _offwindow_expert_keys(core) | _mtp_keys(core, weights)
    assert leftover == expected, (
        "checkpoint coverage: leftover keys are not exactly this rank's "
        f"off-window experts plus the MTP layers; unexpected "
        f"{sorted(leftover - expected)[:8]}, missed {sorted(expected - leftover)[:8]}"
    )
