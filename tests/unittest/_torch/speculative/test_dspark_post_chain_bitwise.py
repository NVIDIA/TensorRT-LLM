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
"""Old vs new MLA DSpark drafter post chain, bit for bit, at Kimi K3 per-rank shapes.

``old`` is the recipe's drafter path: TLLM_SPEC_TAIL_FUSION=1 (fused generation-step
preparation, block latents scattered into the pool), with the fused q_a / kv_a GEMM of
the fused all-reduce path (a local stand-in for the all-reduce on one GPU). ``new`` adds
TLLM_DFLASH_FUSED_PROLOGUE=1 (shape-only block-decode operands reused) and
TLLM_DSPARK_MLA_SPLIT_QKV_STREAMS=1 (each layer's kv path beside its q path, the block
latents stored by the RMSNorm/RoPE kernel itself).

Each case runs decode steps of ``prepare_1st_drafter_inputs`` + ``dflash_forward`` +
the draft logits on the same inputs, eagerly and through a captured CUDA graph (warmup
and capture with multi-stream on, as the CUDA graph runner does), and compares the
drafter inputs (noise embedding, query positions, advertised context, anchor tokens,
batch rows), the block-decode operands, the drafter output and logits, every stored
latent (the whole context pool), ``ctx_len`` and ``seq_lens``.
"""

import contextlib
import gc
import os
from types import SimpleNamespace

import pytest
import torch

SPLIT_ENV = "TLLM_DSPARK_MLA_SPLIT_QKV_STREAMS"
PROLOGUE_ENV = "TLLM_DFLASH_FUSED_PROLOGUE"
TAIL_ENV = "TLLM_SPEC_TAIL_FUSION"

# Kimi K3 DSpark drafter, per rank of a TP8 GEN: 64 heads / 8. Hidden, latent and
# head dims are the checkpoint's; INTER and N_CAPTURE only size GEMMs that both
# variants run identically.
HIDDEN = 7168
HEADS = 8
Q_LORA = 1536
KV_LORA = 512
NOPE = 128
ROPE = 64
V_DIM = 128
LATENT = KV_LORA + ROPE
INTER = 2304
LAYERS = 5
N_CAPTURE = 3
VOCAB = 4096
RANK = 64
PAGE = 64


def _sm10x() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 10


pytestmark = pytest.mark.skipif(not _sm10x(), reason="needs an SM10x GPU (CuTe DSL MLA decode)")


def _config(block_size: int, max_positions: int):
    from transformers import PretrainedConfig

    cfg = PretrainedConfig(
        architectures=["K3DSparkModel"],
        model_type="k3_dspark",
        hidden_size=HIDDEN,
        intermediate_size=INTER,
        num_hidden_layers=LAYERS,
        num_attention_heads=HEADS,
        num_key_value_heads=HEADS,
        q_lora_rank=Q_LORA,
        kv_lora_rank=KV_LORA,
        qk_nope_head_dim=NOPE,
        qk_rope_head_dim=ROPE,
        v_head_dim=V_DIM,
        vocab_size=VOCAB,
        rms_norm_eps=1e-6,
        max_position_embeddings=max_positions,
        rope_theta=50000.0,
        block_size=block_size,
        mask_token_id=VOCAB - 3,
        target_layer_ids=list(range(N_CAPTURE)),
        markov_rank=RANK,
        markov_head_type="vanilla",
        enable_confidence_head=True,
        confidence_head_with_markov=True,
        tie_word_embeddings=False,
        rope_parameters={
            "rope_type": "yarn",
            "factor": 32.0,
            "original_max_position_embeddings": 32768,
            "rope_theta": 50000.0,
            "beta_fast": 32,
            "beta_slow": 1,
            "mscale": 1.0,
            "mscale_all_dim": 1.0,
        },
    )
    cfg.dflash_config = {"mask_token_id": VOCAB - 3, "target_layer_ids": list(range(N_CAPTURE))}
    cfg.torch_dtype = torch.bfloat16
    return cfg


def _weights(seed: int) -> dict:
    gen = torch.Generator(device="cuda").manual_seed(seed)

    def rnd(*shape, scale=0.02):
        return (torch.randn(*shape, generator=gen, device="cuda") * scale).to(torch.bfloat16)

    def norm(n):
        return rnd(n, scale=0.05).abs() + 1.0

    w = {
        "context_proj.weight": rnd(HIDDEN, HIDDEN * N_CAPTURE),
        "context_norm.weight": norm(HIDDEN),
        "final_norm.weight": norm(HIDDEN),
        "embed_tokens.weight": rnd(VOCAB, HIDDEN, scale=1.0),
        "markov_head.markov_w1.weight": rnd(VOCAB, RANK),
        "markov_head.markov_w2.weight": rnd(VOCAB, RANK),
        "confidence_head.proj.weight": rnd(1, HIDDEN + RANK),
        "confidence_head.proj.bias": rnd(1),
    }
    for i in range(LAYERS):
        p = f"layers.{i}."
        w[p + "input_layernorm.weight"] = norm(HIDDEN)
        w[p + "post_attention_layernorm.weight"] = norm(HIDDEN)
        w[p + "self_attn.q_a_proj.weight"] = rnd(Q_LORA, HIDDEN)
        w[p + "self_attn.q_a_layernorm.weight"] = norm(Q_LORA)
        w[p + "self_attn.q_b_proj.weight"] = rnd(HEADS * (NOPE + ROPE), Q_LORA)
        w[p + "self_attn.kv_a_proj_with_mqa.weight"] = rnd(LATENT, HIDDEN)
        w[p + "self_attn.kv_a_layernorm.weight"] = norm(KV_LORA)
        w[p + "self_attn.kv_b_proj.weight"] = rnd(HEADS * (NOPE + V_DIM), KV_LORA)
        w[p + "self_attn.o_proj.weight"] = rnd(HIDDEN, HEADS * V_DIM)
        w[p + "mlp.gate_proj.weight"] = rnd(INTER, HIDDEN)
        w[p + "mlp.up_proj.weight"] = rnd(INTER, HIDDEN)
        w[p + "mlp.down_proj.weight"] = rnd(HIDDEN, INTER)
    return w


class _LocalFusedAllReduce:
    """One-GPU stand-in for the drafter's MNNVL all-reduce fused with residual + RMSNorm.

    Its presence selects the fused-all-reduce shape of ``dflash_forward`` (o_proj and
    down_proj partials, one q_a / kv_a GEMM), which is the recipe's
    (TLLM_DSPARK_DRAFT_FUSION=1 at TP8). Deterministic, so both variants see the same.
    """

    mnnvl_allreduce = True

    def __call__(self, partial, all_reduce_params):
        p = all_reduce_params
        hidden = partial + p.residual
        hf = hidden.float()
        normed = hf * torch.rsqrt(hf.square().mean(-1, keepdim=True) + p.eps)
        return (normed * p.norm_weight.float()).to(hidden.dtype), hidden


def build_drafter(block_size: int, max_positions: int, fused_a: bool, seed: int = 5):
    from tensorrt_llm._torch.model_config import ModelConfig
    from tensorrt_llm._torch.models.modeling_dspark import MLADSparkForCausalLM

    model_config = ModelConfig(
        pretrained_config=_config(block_size, max_positions), attn_backend="TRTLLM"
    )
    drafter = MLADSparkForCausalLM(model_config, dflash_attention_backend="CUTEDSL").to("cuda")
    drafter.load_weights(_weights(seed))
    drafter._runtime_position_ceiling = max_positions
    if fused_a:
        stand_in = _LocalFusedAllReduce()
        drafter._draft_fused_all_reduce = lambda: stand_in
    return drafter


class PostChain:
    """One DFlash worker + MLA drafter state and its decode step (prepare, forward, logits)."""

    def __init__(self, drafter, k: int, batch: int, ctx_tokens: int, steps: int, seed: int):
        from tensorrt_llm._torch.speculative.dflash import DFlashWorker
        from tensorrt_llm.llmapi import DFlashDecodingConfig
        from tensorrt_llm.mapping import Mapping

        gen = torch.Generator(device="cuda").manual_seed(seed)
        self.drafter, self.k, self.batch = drafter, k, batch
        width = k + 1
        num_slots = batch + 3
        # Room for the context, every step's accepted tokens and the block itself.
        blocks = (ctx_tokens + steps * width + k + 2 * PAGE) // PAGE + 1
        num_pages = (num_slots + 1) * blocks
        self.storage = torch.randn(
            LAYERS, num_pages, 1, 1, PAGE, LATENT, generator=gen, device="cuda"
        ).to(torch.bfloat16)
        worker = DFlashWorker(DFlashDecodingConfig(max_draft_len=k), Mapping())
        worker._ctx_paged = True
        worker._ctx_page_size = PAGE
        worker._resolved_block_size = k
        worker._resolved_mask_token_id = VOCAB - 3
        worker._dummy_slot = num_slots
        worker._max_ctx = blocks * PAGE - 2
        worker._ctx_kv_buf = [self.storage[layer] for layer in range(LAYERS)]
        pages = torch.randperm(num_pages, generator=gen, device="cuda").to(torch.int32)
        # Rows are batch positions (the manager's per-request block table).
        worker._ctx_block_tables = pages.view(num_slots + 1, blocks)
        worker._ctx_block_counts = torch.full(
            (num_slots + 1,), blocks, dtype=torch.long, device="cuda"
        )
        lens = ctx_tokens - torch.randint(
            0, max(ctx_tokens // 8, 1), (num_slots + 1,), generator=gen, device="cuda"
        )
        worker._ctx_len = lens.long()
        worker._ctx_len[num_slots] = 0
        worker._ctx_k_buf = None
        worker._ctx_v_buf = None
        worker._batch_to_slot = torch.randperm(num_slots, generator=gen, device="cuda")[
            :batch
        ].long()
        self.worker = worker
        total = batch * width
        self.total = total
        self.accepted = torch.zeros(batch, width, dtype=torch.int32, device="cuda")
        self.num_accepted = torch.ones(batch, dtype=torch.int32, device="cuda")
        captured = torch.zeros(total, HIDDEN * N_CAPTURE, dtype=torch.bfloat16, device="cuda")
        self.captured = captured
        # The getter closes over the buffer, not the chain: no reference cycle, so a dropped
        # chain (and its graph) is freed at once, never by a collection during a capture.
        self.spec_metadata = SimpleNamespace(
            runtime_draft_len=k,
            hidden_size=HIDDEN,
            get_hidden_states=lambda n: captured[:n],
        )
        self.attn_metadata = SimpleNamespace(
            num_contexts=0,
            num_seqs=batch,
            num_ctx_tokens=0,
            _seq_lens_cuda=torch.full((batch + 2,), 3, dtype=torch.int32, device="cuda"),
            _seq_lens=torch.full((batch + 2,), 3, dtype=torch.int32),
        )
        self.dummy_hidden = torch.empty(1, HIDDEN, dtype=torch.bfloat16, device="cuda")
        self.graph = None
        self.outputs = None

    def load_inputs(self, step_inputs):
        accepted, num_accepted, captured = step_inputs
        self.accepted.copy_(accepted)
        self.num_accepted.copy_(num_accepted)
        self.captured.copy_(captured)

    def step(self):
        inputs = self.worker.prepare_1st_drafter_inputs(
            input_ids=None,
            position_ids=None,
            hidden_states=self.dummy_hidden,
            accepted_tokens=self.accepted,
            num_accepted_tokens=self.num_accepted,
            attn_metadata=self.attn_metadata,
            spec_metadata=self.spec_metadata,
            draft_model=self.drafter,
            total_target_tokens=self.total,
        )
        hidden_out = self.drafter.dflash_forward(
            noise_embedding=inputs["noise_embedding"],
            query_positions=inputs["query_positions"],
            num_ctx_per_req=inputs["num_ctx_per_req"],
            ctx_k_cache=inputs["ctx_k_cache"],
            ctx_v_cache=inputs["ctx_v_cache"],
            ctx_cache_batch_idx=inputs["ctx_cache_batch_idx"],
            ctx_kv_cache=inputs["ctx_kv_cache"],
            ctx_page_table=inputs["ctx_page_table"],
            block_fixup=inputs.get("mla_block_fixup"),
        )
        logits = torch.nn.functional.linear(hidden_out, self.drafter.lm_head.weight)
        return inputs, hidden_out, logits

    def capture(self, warmup_inputs, warmup: int = 2):
        from tensorrt_llm._torch.modules.multi_stream_utils import with_multi_stream

        self.load_inputs(warmup_inputs)
        with with_multi_stream(True):
            for _ in range(warmup):
                self.step()
            torch.cuda.synchronize()
            # Collect outside the capture: a graph destroyed while another is captured
            # invalidates that capture.
            gc.collect()
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph):
                self.outputs = self.step()
        torch.cuda.synchronize()

    def replay(self, step_inputs):
        self.load_inputs(step_inputs)
        self.graph.replay()
        return self.outputs


def snapshot(chain: PostChain, outputs) -> dict:
    inputs, hidden_out, logits = outputs
    fixup = inputs.get("mla_block_fixup")
    snap = {
        "noise_embedding": inputs["noise_embedding"],
        "query_positions": inputs["query_positions"],
        "num_ctx_per_req": inputs["num_ctx_per_req"],
        "first_prev_tokens": inputs["first_prev_tokens"],
        "ctx_cache_batch_idx": inputs["ctx_cache_batch_idx"],
        "hidden_out": hidden_out,
        "logits": logits,
        "ctx_len": chain.worker._ctx_len,
        "seq_lens_cuda": chain.attn_metadata._seq_lens_cuda,
        "pool": chain.storage,
    }
    for field in fixup._fields if fixup is not None else ():
        snap[f"fixup.{field}"] = getattr(fixup, field)
    return {key: value.detach().clone() for key, value in snap.items()}


def step_inputs(k: int, batch: int, steps: int, seed: int):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    width = k + 1
    out = []
    for _ in range(steps):
        accepted = torch.randint(0, VOCAB - 3, (batch, width), generator=gen, device="cuda").to(
            torch.int32
        )
        num_accepted = torch.randint(1, width + 1, (batch,), generator=gen, device="cuda").to(
            torch.int32
        )
        captured = torch.randn(batch * width, HIDDEN * N_CAPTURE, generator=gen, device="cuda").to(
            torch.bfloat16
        )
        out.append((accepted, num_accepted, captured))
    return out


@contextlib.contextmanager
def variant_env(new: bool):
    saved = {key: os.environ.get(key) for key in (SPLIT_ENV, PROLOGUE_ENV, TAIL_ENV)}
    os.environ[TAIL_ENV] = "1"
    os.environ[SPLIT_ENV] = "1" if new else "0"
    os.environ[PROLOGUE_ENV] = "1" if new else "0"
    try:
        yield
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def run_variant(new, drafter, k, batch, ctx_tokens, inputs, mode, seed=11):
    """Snapshots after every step of one variant ("eager" or "graph")."""
    with variant_env(new):
        chain = PostChain(drafter, k, batch, ctx_tokens, len(inputs) + 2, seed)
        snaps = []
        if mode == "graph":
            chain.capture(inputs[0])
            for step in inputs:
                snaps.append(snapshot(chain, chain.replay(step)))
        else:
            for step in inputs:
                chain.load_inputs(step)
                snaps.append(snapshot(chain, chain.step()))
        torch.cuda.synchronize()
    return snaps


def assert_bitwise(old_snaps, new_snaps):
    assert len(old_snaps) == len(new_snaps)
    for step, (old, new) in enumerate(zip(old_snaps, new_snaps)):
        assert old.keys() == new.keys()
        for key in old:
            a, b = old[key], new[key]
            assert a.dtype == b.dtype and a.shape == b.shape, (step, key)
            if a.is_floating_point():
                # Bit patterns, so a NaN or a signed zero must match too.
                a = a.view(torch.int16) if a.element_size() == 2 else a.view(torch.int32)
                b = b.view(torch.int16) if b.element_size() == 2 else b.view(torch.int32)
            assert torch.equal(a, b), f"step {step}: {key} differs"


_DRAFTERS = {}


def _drafter(k: int, fused_a: bool):
    key = (k, fused_a)
    if key not in _DRAFTERS:
        _DRAFTERS[key] = build_drafter(k, max_positions=1 << 16, fused_a=fused_a)
    return _DRAFTERS[key]


@pytest.mark.parametrize("mode", ["graph", "eager"])
@pytest.mark.parametrize("fused_a", [True, False], ids=["fused_a", "separate_a"])
@pytest.mark.parametrize(
    "k,batch", [(6, 1), (6, 2), (6, 5), (5, 1), (5, 3), (5, 8)], ids=lambda v: str(v)
)
@torch.inference_mode()
def test_post_chain_old_vs_new_bitwise(k, batch, fused_a, mode):
    drafter = _drafter(k, fused_a)
    ctx_tokens = 3000
    inputs = step_inputs(k, batch, steps=4, seed=100 * k + batch)
    old = run_variant(False, drafter, k, batch, ctx_tokens, inputs, mode)
    new = run_variant(True, drafter, k, batch, ctx_tokens, inputs, mode)
    assert_bitwise(old, new)


@pytest.mark.parametrize("mode", ["graph", "eager"])
@pytest.mark.parametrize("fused_a", [True, False], ids=["fused_a", "separate_a"])
@torch.inference_mode()
def test_post_chain_new_takes_the_split_paths(fused_a, mode, monkeypatch):
    """The comparison above is only evidence if ``new`` ran the new code: the
    pool-write kernel instead of the scatter, and the kv path on the aux stream where
    multi-stream is on (graph warmup and capture), one stream otherwise (eager).
    """
    from tensorrt_llm._torch.models import modeling_dspark

    calls = {"pool_write": 0, "scatter": 0, "aux": 0}
    real_pool_write = modeling_dspark._rmsnorm_rope_pool_write
    real_scatter = modeling_dspark.scatter_rows_to_pool
    real_stream = modeling_dspark.MLADSparkForCausalLM._split_qkv_stream

    def pool_write(*args, **kwargs):
        done = real_pool_write(*args, **kwargs)
        calls["pool_write"] += int(done)
        return done

    def scatter(*args, **kwargs):
        calls["scatter"] += 1
        return real_scatter(*args, **kwargs)

    def stream(self, device):
        aux = real_stream(self, device)
        calls["aux"] += int(aux is not None)
        return aux

    monkeypatch.setattr(modeling_dspark, "_rmsnorm_rope_pool_write", pool_write)
    monkeypatch.setattr(modeling_dspark, "scatter_rows_to_pool", scatter)
    monkeypatch.setattr(modeling_dspark.MLADSparkForCausalLM, "_split_qkv_stream", stream)
    drafter = _drafter(6, fused_a)
    inputs = step_inputs(6, 2, steps=2, seed=3)
    # graph: two warmup steps and one capture; eager: one pass per step. Five layers each.
    passes = 3 if mode == "graph" else len(inputs)
    run_variant(True, drafter, 6, 2, 2000, inputs, mode)
    aux = passes * LAYERS if mode == "graph" else 0
    assert calls == {"pool_write": passes * LAYERS, "scatter": 0, "aux": aux}
    calls.update(pool_write=0, scatter=0, aux=0)
    run_variant(False, drafter, 6, 2, 2000, inputs, mode)
    assert calls == {"pool_write": 0, "scatter": passes * LAYERS, "aux": 0}
