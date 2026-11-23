# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""The family DSpark drafter's FP8 E4M3 latent pool and its context-KV-only form.

* The drafter declares an FP8 E4M3 draft KV pool (``draft_kv_dtype``) and reads either an FP8 or a
  BF16 one.
* One layer's attention over an FP8 pool (``_mla_paged_attention``: the block store, then the
  cute-dsl FP8 decode) against an explicit composition in float32: the context and block latents
  as stored (``to_draft_fp8``), the query quantized the same way, unit dequant scales, the whole
  block visible from every block token. The same for a BF16 pool and the FP16 decode. Both the
  fused block store (``spec_tail_fusion``) and the indexed one.
* The whole block decode over an FP8 pool stores ``to_draft_fp8`` of the first layer's block
  latents, stays within a sanity bound of the BF16 pool's result, and replays the same in a CUDA graph.
* A context-KV-only drafter keeps only ``fc``, ``hidden_norm`` and each layer's
  ``kv_a_proj_with_mqa`` / ``kv_a_layernorm``, loads only those from a full checkpoint, computes the
  full drafter's context latents bit for bit, has no LM head and refuses to draft.
"""

import importlib
from types import SimpleNamespace

import pytest
import torch

__extra_import_path__ = ["."]
import _k3_dspark_drafter_inputs as parity  # noqa: E402

NEW = parity.NEW
BLOCK, PAGE = parity.BLOCK, parity.PAGE

_sm100f = pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.get_device_capability() not in ((10, 0), (10, 3), (10, 7)),
    reason="the cute-dsl MLA block decode runs on SM100, SM103 and SM107",
)


def _drafter(path_fields=None, context_kv_only=False):
    from tensorrt_llm._torch.configs import K3DsparkConfig
    from tensorrt_llm._torch.model_config import ModelConfig
    from tensorrt_llm.mapping import Mapping

    mod = importlib.import_module(f"{NEW}.drafter")
    cfg = K3DsparkConfig(**parity.drafter_config_dict())
    cfg.torch_dtype = torch.bfloat16
    config = ModelConfig(pretrained_config=cfg, mapping=Mapping(), max_seq_len=parity.MAX_SEQ)
    spec = SimpleNamespace(attention_backend="AUTO", draft_is_embedded_in_target=False)
    with torch.device("cuda"):
        drafter = mod.build_k3_dspark_drafter(
            config,
            spec,
            path=mod.K3DSparkPath(**(path_fields or {})),
            context_kv_only=context_kv_only,
        )
    drafter.load_weights(weights=parity.drafter_weights())
    return drafter


def _reference_attention(query, blk_kv, pool, fixup, ctx_lens, softmax_scale, fp8):
    """``[B, block, heads, 512]`` float32: every block token attends context + block."""
    to_fp8 = importlib.import_module(f"{NEW}.primitives").to_draft_fp8

    def stored(x):
        return to_fp8(x).float() if fp8 else x.float()

    q = stored(query)
    out = torch.empty(*query.shape[:3], parity.KV_LORA, dtype=torch.float32, device=query.device)
    for b, n in enumerate(ctx_lens):
        table = fixup.page_tables_i32[b].long()
        pos = torch.arange(n, device=query.device)
        ctx = pool[table[pos // PAGE], 0, 0, pos % PAGE].float()  # as stored (FP8 or BF16)
        keys = torch.cat([ctx, stored(blk_kv[b])])  # [n + block, 576]
        scores = torch.einsum("shd,td->sht", q[b], keys) * softmax_scale
        out[b] = torch.einsum("sht,td->shd", scores.softmax(-1), keys[:, : parity.KV_LORA])
    return out


@_sm100f
@pytest.mark.parametrize("fp8", [True, False], ids=["fp8", "bf16"])
@pytest.mark.parametrize("fused_store", [True, False], ids=["fused_store", "indexed_store"])
def test_one_layer_attention_matches_the_explicit_composition(fp8, fused_store):
    with torch.inference_mode():
        drafter = _drafter({"spec_tail_fusion": True} if fused_store else None)
        assert drafter.draft_kv_dtype == torch.float8_e4m3fn and drafter.context_kv_fp8
        inp = parity.Inputs(drafter, torch.float8_e4m3fn if fp8 else torch.bfloat16)
        projected = drafter.project_target_hidden(inp.captured)
        latent, _ = drafter.precompute_context_kv(projected, inp.positions)
        pool = inp.pools(latent)[0]
        fixup = importlib.import_module(f"{NEW}.drafter")._build_mla_block_fixup(
            inp.ctx, inp.table.long(), BLOCK, PAGE
        )
        query = parity.normalish((inp.batch, BLOCK, parity.HEADS, 576), 31, 1.0)
        blk_kv = parity.normalish((inp.batch, BLOCK, 576), 32, 1.0)
        before = pool.clone()
        out = drafter._mla_paged_attention(query, blk_kv, pool, fixup, BLOCK)
        torch.cuda.synchronize()

        to_fp8 = importlib.import_module(f"{NEW}.primitives").to_draft_fp8
        pages, offsets = fixup.pages.reshape(-1), fixup.offsets.reshape(-1)
        want_rows = to_fp8(blk_kv.reshape(-1, 576)) if fp8 else blk_kv.reshape(-1, 576)
        assert torch.equal(
            pool[pages, 0, 0, offsets].view(torch.uint8), want_rows.view(torch.uint8)
        )
        untouched = torch.ones(pool.shape[0], PAGE, dtype=torch.bool, device=pool.device)
        untouched[pages, offsets] = False
        assert torch.equal(
            pool[:, 0, 0][untouched].view(torch.uint8), before[:, 0, 0][untouched].view(torch.uint8)
        )

        ref = _reference_attention(
            query, blk_kv, pool, fixup, parity.CTX_LENS, drafter.softmax_scale, fp8
        )
        assert out.dtype == torch.bfloat16
        err = (out.float() - ref).norm() / ref.norm()
        assert err < 1e-2, f"relative L2 {err:.3e}"


@_sm100f
def test_block_decode_over_an_fp8_pool():
    with torch.inference_mode():
        drafter = _drafter({"spec_tail_fusion": True})
        to_fp8 = importlib.import_module(f"{NEW}.primitives").to_draft_fp8
        runs = {}
        for name, dtype in (("bf16", torch.bfloat16), ("fp8", torch.float8_e4m3fn)):
            inp = parity.Inputs(drafter, dtype)
            projected = drafter.project_target_hidden(inp.captured)
            latent, _ = drafter.precompute_context_kv(projected, inp.positions)
            embed = drafter.draft_model_full.model.embed_tokens(inp.ids.reshape(-1))
            noise = embed.view(inp.batch, BLOCK, parity.HIDDEN)
            fixup = importlib.import_module(f"{NEW}.drafter")._build_mla_block_fixup(
                inp.ctx, inp.table.long(), BLOCK, PAGE
            )
            parity.forward(drafter, inp, inp.pools(latent), noise.clone(), "fixup", fixup, None)
            pools = inp.pools(latent)
            hidden = parity.forward(drafter, inp, pools, noise.clone(), "fixup", fixup, None)
            pages, offsets = fixup.pages.reshape(-1), fixup.offsets.reshape(-1)
            runs[name] = (hidden, pools[0][:, 0, 0][pages, offsets])
            if name == "fp8":
                graph_pools = inp.pools(latent)
                static_noise = noise.clone()
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g):
                    out_g = parity.forward(
                        drafter, inp, graph_pools, static_noise, "fixup", fixup, None
                    )
                for p, ref in zip(graph_pools, inp.pools(latent)):
                    p.copy_(ref)
                g.replay()
                torch.cuda.synchronize()
                assert torch.equal(out_g.view(torch.int16), hidden.view(torch.int16))
        torch.cuda.synchronize()
        (h16, rows16), (h8, rows8) = runs["bf16"], runs["fp8"]
        assert rows8.dtype == torch.float8_e4m3fn
        # The first layer's block latents depend on the noise alone: stored as to_draft_fp8.
        assert torch.equal(rows8.view(torch.uint8), to_fp8(rows16).view(torch.uint8))
        assert torch.isfinite(h8.float()).all()
        err = (h8.float() - h16.float()).norm() / h16.float().norm()
        # A sanity bound: the per-layer arithmetic is pinned by the explicit composition above;
        # here E4M3 rounding of the query and of every latent compounds over the layers.
        print(f"relative L2 of the FP8 pool's hidden states against the BF16 pool's: {err:.3e}")
        assert err < 0.2, f"relative L2 of the FP8 pool's hidden states {err:.3e}"


@_sm100f
def test_context_kv_only_drafter():
    with torch.inference_mode():
        full = _drafter()
        ctx = _drafter(context_kv_only=True)
        backbone = ctx.draft_model_full.model
        assert backbone.embed_tokens is None and backbone.norm is None
        for layer in backbone.layers:
            attn = layer.self_attn
            assert layer.mlp is None and layer.input_layernorm is None
            assert layer.post_attention_layernorm is None
            assert all(
                getattr(attn, n) is None
                for n in ("q_a_proj", "q_a_layernorm", "q_b_proj", "kv_b_proj", "o_proj")
            )
            assert attn.kv_a_proj_with_mqa is not None and attn.kv_a_layernorm is not None
        assert ctx.markov_w1 is None and ctx.markov_w2 is None
        # Parameters: fc, hidden_norm, and two per layer.
        names = sorted(n for n, _ in ctx.named_parameters())
        assert len(names) == 2 + 2 * parity.LAYERS, names

        inp = parity.Inputs(full)
        for drafter in (full, ctx):
            drafter.load_weights_from_target_model(SimpleNamespace(lm_head=torch.nn.Linear(1, 1)))
        assert ctx.lm_head is None and full.lm_head is not None
        got, want = {}, {}
        for drafter, out in ((full, want), (ctx, got)):
            projected = drafter.project_target_hidden(inp.captured)
            out["projected"] = projected
            out["latent"] = drafter.precompute_context_kv(projected, inp.positions)[0]
            out["layer_major"] = drafter.precompute_context_kv_layer_major(projected, inp.positions)
        for name in want:
            assert torch.equal(got[name].view(torch.int16), want[name].view(torch.int16)), name
        with pytest.raises(RuntimeError, match="does not draft"):
            parity.forward(ctx, inp, inp.pools(want["latent"]), None, "build", None, None)


def test_the_drafter_declares_its_draft_kv():
    mod = importlib.import_module(f"{NEW}.drafter")
    assert mod.K3DSparkDrafter.draft_kv_dtype == torch.float8_e4m3fn
    assert mod.K3DSparkDrafter._paged_ctx_cache is True
    drafter = SimpleNamespace(_helix_mapping=None)
    for tokens_per_block, reads in ((32, True), (64, True), (128, True), (1, False), (48, False)):
        assert mod.K3DSparkDrafter.reads_managed_ctx_pages(drafter, tokens_per_block) is reads
    assert mod.K3DSparkDrafter.reads_helix_sharded_ctx_pages.fget(drafter) is False
    drafter._helix_mapping = object()
    assert mod.K3DSparkDrafter.reads_helix_sharded_ctx_pages.fget(drafter) is True
    with pytest.raises(ValueError, match="tail_v2"):
        mod.K3DSparkPath(tail_v2=True)
