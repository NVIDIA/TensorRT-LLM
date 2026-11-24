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
"""FP8 storage of the standalone MLA DSpark drafter's context KV.

``speculative_config.context_kv_dtype="fp8"`` stores the drafter's context
latents in the unified draft cache as FP8 E4M3 at a unit per-tensor scale
(``to_draft_fp8``): the draft pool, its share of the KV cache and its transfer
pages are sized at 576 instead of 1152 bytes per token and layer, and the
generation server's block decode reads them with the FP8 CuTe DSL MLA decode.
The default (``"auto"``) keeps BF16 and the existing tests cover it unchanged.
"""

import math
import types
from types import SimpleNamespace

import pytest
import test_dspark_unified_draft_cache as base
import torch
import torch.nn.functional as F

from tensorrt_llm._torch.disaggregation.native.auxiliary import (
    _decode_draft_history,
    _encode_draft_history,
)
from tensorrt_llm._torch.models.modeling_dspark import (
    _MLA_SPLIT_PLAN_PIECE_COST_TILES,
    _MLA_SPLIT_PLAN_PIECE_COST_TILES_FP8,
    MLADSparkForCausalLM,
    _build_mla_block_fixup,
    _yarn_get_mscale,
)
from tensorrt_llm._torch.pyexecutor.resource_manager import CacheTypeCpp, DataType
from tensorrt_llm._torch.speculative.dflash_attention import to_draft_fp8
from tensorrt_llm._torch.speculative.dspark import DSparkWorker
from tensorrt_llm._torch.speculative.spec_tail_kernels import (
    scatter_rows_to_pool,
    store_mla_context_latents,
)
from tensorrt_llm.llmapi.llm_args import DSparkDecodingConfig, KvCacheConfig
from tensorrt_llm.logger import logger
from tensorrt_llm.mapping import Mapping

FP8 = torch.float8_e4m3fn
PAGE = base.PAGE
LATENT = base.LATENT
LAYERS = base.LAYERS
BLOCK = base.BLOCK
HIDDEN = base.HIDDEN
NUM_CAPTURE = base.NUM_CAPTURE
VOCAB = base.VOCAB
MAX_SEQ_LEN = base.MAX_SEQ_LEN

needs_sm100f = base.needs_sm100f


def _fp8_capable() -> bool:
    # The E4M3 conversions of the FP8 stores need SM89 or newer.
    return torch.cuda.is_available() and torch.cuda.get_device_capability() >= (8, 9)


needs_fp8_cuda = pytest.mark.skipif(not _fp8_capable(), reason="FP8 E4M3 conversion needs SM89")


def _bytes(t: torch.Tensor) -> torch.Tensor:
    """A tensor's bytes; a view (writes go through) when it is contiguous."""
    return t.contiguous().view(torch.uint8)


def _pool_bytes(manager, layer) -> torch.Tensor:
    """Byte view of a draft layer's pool pages ``[pages, 1, 1, page, bytes]``."""
    return manager.get_draft_buffers(layer, kv_layout="HND").view(torch.uint8)


def _error(actual: torch.Tensor, expected: torch.Tensor) -> dict:
    """max abs / relative L2 / cosine of ``actual`` against ``expected``."""
    a = actual.float().reshape(-1)
    e = expected.float().reshape(-1)
    diff = a - e
    return {
        "max_abs": diff.abs().max().item(),
        "rel_l2": (diff.norm() / e.norm()).item(),
        "cosine": F.cosine_similarity(a, e, dim=0).item(),
    }


def _report(tag: str, stats: dict) -> None:
    logger.info(
        f"[fp8-numerics] {tag}: max_abs={stats['max_abs']:.4g} "
        f"rel_l2={stats['rel_l2']:.4g} cosine={stats['cosine']:.6f}"
    )


def _build_drafter(weights, *, backend="VANILLA", context_kv_only=False, context_kv_dtype="auto"):
    from tensorrt_llm._torch.model_config import ModelConfig

    model_config = ModelConfig(pretrained_config=base._tiny_mla_config(), attn_backend="TRTLLM")
    drafter = MLADSparkForCausalLM(
        model_config,
        dflash_attention_backend=backend,
        context_kv_only=context_kv_only,
        context_kv_dtype=context_kv_dtype,
    ).to("cuda")
    drafter.load_weights(dict(weights))
    return drafter


def _manager(draft_dtype, tokens_per_block=PAGE, max_tokens=1024):
    from tensorrt_llm._torch.pyexecutor.kv_cache_manager_v2 import KVCacheManagerV2

    return KVCacheManagerV2(
        KvCacheConfig(max_tokens=max_tokens, enable_block_reuse=False),
        CacheTypeCpp.SELF,
        num_layers=1,
        num_kv_heads=1,
        head_dim=64,
        tokens_per_block=tokens_per_block,
        max_seq_len=MAX_SEQ_LEN,
        max_batch_size=4,
        mapping=Mapping(),
        dtype=DataType.HALF,
        is_disagg=True,
        draft_num_layers=LAYERS,
        draft_num_kv_heads=1,
        draft_head_dim=LATENT,
        draft_dtype=draft_dtype,
        draft_kv_factor=1,
        draft_total_num_kv_heads=1,
        draft_scratch_tokens=BLOCK + 1,
        draft_attention_backend="CUTEDSL",
    )


def _worker(backend, context_kv_only=False, context_kv_dtype="auto"):
    config = DSparkDecodingConfig(
        max_draft_len=BLOCK - 1,
        speculative_model="unused",
        attention_backend=backend,
        context_kv_only=context_kv_only,
        context_kv_dtype=context_kv_dtype,
    )
    return DSparkWorker(config, Mapping())


def _draft_page_bytes(manager, request_id, layer):
    """Bytes of a request's draft pages of one layer, as ``[tokens, latent]``."""
    table = manager.get_draft_block_table([request_id])[0].long()
    pages = table[: manager.get_draft_num_blocks(request_id)].cuda()
    pool = _pool_bytes(manager, layer)
    return pool[pages, 0, 0].reshape(-1, pool.shape[-1])


# ---------------------------------------------------------------------------
# Configuration: opt-in, and rejected where it is not honoured.
# ---------------------------------------------------------------------------


def test_context_kv_dtype_defaults_to_the_drafter_dtype():
    config = DSparkDecodingConfig(max_draft_len=2, speculative_model="unused")
    assert config.context_kv_dtype == "auto"
    fp8 = DSparkDecodingConfig(max_draft_len=2, speculative_model="unused", context_kv_dtype="fp8")
    assert fp8.context_kv_dtype == "fp8"
    with pytest.raises(ValueError):
        DSparkDecodingConfig(max_draft_len=2, speculative_model="unused", context_kv_dtype="nvfp4")


def test_fp8_context_kv_is_rejected_for_a_drafter_that_cannot_honour_it():
    from tensorrt_llm._torch.models.modeling_dspark import _build_dspark_draft

    for embedded, kv_lora_rank in ((False, None), (True, base.KV_LORA)):
        spec = SimpleNamespace(
            context_kv_only=False,
            context_kv_dtype="fp8",
            attention_backend="AUTO",
            draft_is_embedded_in_target=embedded,
        )
        pretrained = SimpleNamespace(kv_lora_rank=kv_lora_rank, qk_rope_head_dim=base.ROPE)
        with pytest.raises(ValueError, match="context_kv_dtype is implemented for standalone MLA"):
            _build_dspark_draft(
                SimpleNamespace(spec_config=spec),
                SimpleNamespace(pretrained_config=pretrained),
                None,
                None,
            )


@needs_fp8_cuda
def test_fp8_context_kv_needs_the_cutedsl_block_decode(monkeypatch):
    weights = base._tiny_mla_weights()
    with pytest.raises(ValueError, match="CUTEDSL"):
        _build_drafter(weights, backend="VANILLA", context_kv_dtype="fp8")
    # A context server never runs the block decode, so any backend stores it.
    ctx = _build_drafter(weights, backend="VANILLA", context_kv_only=True, context_kv_dtype="fp8")
    assert ctx.context_kv_fp8
    assert not _build_drafter(weights).context_kv_fp8


@needs_sm100f
def test_fp8_drafter_decode_has_no_split_cap(monkeypatch):
    """TLLM_DSPARK_MLA_MAX_SPLIT_KV caps the BF16 decode only; an FP8 drafter that
    reads the pages must not silently ignore it."""
    weights = base._tiny_mla_weights()
    monkeypatch.setenv("TLLM_DSPARK_MLA_MAX_SPLIT_KV", "1")
    with pytest.raises(ValueError, match="TLLM_DSPARK_MLA_MAX_SPLIT_KV"):
        _build_drafter(weights, backend="CUTEDSL", context_kv_dtype="fp8")
    # BF16 keeps the cap, and a context server never decodes.
    assert _build_drafter(weights, backend="CUTEDSL")._mla_max_split_kv == 1
    assert _build_drafter(
        weights, backend="VANILLA", context_kv_only=True, context_kv_dtype="fp8"
    ).context_kv_fp8


# ---------------------------------------------------------------------------
# Pool sizing and transfer layout use the FP8 bytes.
# ---------------------------------------------------------------------------


def test_unified_draft_cache_kwargs_size_the_fp8_latent(monkeypatch):
    """The draft kwargs the executor builds make the manager size the layers at FP8."""
    from tensorrt_llm._torch.pyexecutor._util import KvCacheCreator
    from tensorrt_llm._torch.pyexecutor.kv_cache_manager_v2 import _get_draft_layer_sizes

    monkeypatch.setattr(KvCacheCreator, "_is_embedded_dspark", lambda self: False)

    def kwargs(fp8):
        creator = object.__new__(KvCacheCreator)
        drafter = SimpleNamespace(
            _kv_factor=1,
            _num_attn_layers=5,
            _head_dim=576,
            fc=SimpleNamespace(weight=torch.empty(1, dtype=torch.bfloat16)),
            dflash_attention_backend="CUTEDSL",
            config=None,
            context_kv_fp8=fp8,
        )
        creator._model_engine = SimpleNamespace(model=SimpleNamespace(draft_model=drafter))
        creator._draft_config = SimpleNamespace(
            pretrained_config=SimpleNamespace(), mapping=Mapping()
        )
        creator._speculative_config = SimpleNamespace(attention_backend="AUTO", max_draft_len=6)
        return creator._get_unified_draft_cache_kwargs()

    bf16, fp8 = kwargs(False), kwargs(True)
    assert bf16["draft_dtype"] == DataType.BF16 and fp8["draft_dtype"] == DataType.FP8
    assert {k: v for k, v in bf16.items() if k != "draft_dtype"} == {
        k: v for k, v in fp8.items() if k != "draft_dtype"
    }
    # Kimi K3 DSpark: 5 layers x 576 latent, 5760 -> 2880 B per token.
    assert _get_draft_layer_sizes(**bf16) == (1152,) * 5
    assert _get_draft_layer_sizes(**fp8) == (576,) * 5


@needs_fp8_cuda
def test_fp8_draft_pool_pages_and_transfer_layout():
    """The pool holds FP8 pages at half the bytes, the transfer page table copies
    them at that size, and peers that store different dtypes are rejected both
    at registration and by the per-request identity."""
    from tensorrt_llm._torch.disaggregation.native.mixers.attention.peer import AttentionPolicy
    from tensorrt_llm._torch.disaggregation.resource.kv_extractor import (
        build_page_table_from_manager,
    )
    from tensorrt_llm._torch.pyexecutor.kv_cache_manager_v2 import Role

    managers = {
        "bf16": _manager(DataType.BF16),
        "fp8": _manager(DataType.FP8),
        "fp8_2x_page": _manager(DataType.FP8, tokens_per_block=2 * PAGE),
    }
    try:
        bf16, fp8 = managers["bf16"], managers["fp8"]
        for mgr, dtype, itemsize in ((bf16, torch.bfloat16, 2), (fp8, FP8, 1)):
            buf = mgr.get_draft_buffers(0, kv_layout="HND")
            assert buf.dtype == dtype and tuple(buf.shape[1:]) == (1, 1, PAGE, LATENT)
            assert mgr._draft_layer_sizes == (LATENT * itemsize,) * LAYERS
            local = mgr.layer_offsets[mgr.draft_layer_ids[0]]
            assert mgr.get_layer_bytes_per_token(local, Role.ALL) == LATENT * itemsize
        assert bf16.get_cache_bytes_per_token() - fp8.get_cache_bytes_per_token() == (
            LAYERS * LATENT
        )
        assert fp8.get_draft_transfer_identity()["dtype"] == "torch.float8_e4m3fn"
        assert bf16.get_draft_transfer_identity()["dtype"] == "torch.bfloat16"

        tables = {name: build_page_table_from_manager(mgr) for name, mgr in managers.items()}

        def draft_page_bytes(table):
            return {
                view.bytes_per_layer
                for group in table.layer_groups
                if getattr(group, "cp_as_tp", False)
                for view in group.pool_views
            }

        assert draft_page_bytes(tables["fp8"]) == {LATENT * PAGE}
        assert draft_page_bytes(tables["fp8_2x_page"]) == {LATENT * 2 * PAGE}
        assert draft_page_bytes(tables["bf16"]) == {2 * LATENT * PAGE}
        AttentionPolicy.validate_peer_compatible(tables["fp8"], tables["fp8"], PAGE, PAGE)
        # Different page sizes, same bytes per token.
        AttentionPolicy.validate_peer_compatible(
            tables["fp8"], tables["fp8_2x_page"], PAGE, 2 * PAGE
        )
        with pytest.raises(ValueError, match="context_kv_dtype"):
            AttentionPolicy.validate_peer_compatible(tables["fp8"], tables["bf16"], PAGE, PAGE)
        with pytest.raises(ValueError, match="context_kv_dtype"):
            AttentionPolicy.validate_peer_compatible(tables["bf16"], tables["fp8"], PAGE, PAGE)

        sent = {"valid_length": 8, "position": 8, "layout": fp8.get_draft_transfer_identity()}
        sent = _decode_draft_history(_encode_draft_history(sent))
        with pytest.raises(ValueError, match="does not match the receiving worker"):
            bf16.restore_draft_history(1, sent)
    finally:
        for mgr in managers.values():
            mgr.shutdown()


# ---------------------------------------------------------------------------
# Stores: every write path quantizes the same way.
# ---------------------------------------------------------------------------


@needs_fp8_cuda
def test_fp8_latent_stores_match_to_draft_fp8():
    torch.manual_seed(5)
    num_tokens, num_layers, num_pages = 9, 3, 4
    latent = (torch.randn(num_tokens, num_layers, LATENT, device="cuda") * 3).to(torch.bfloat16)
    # Saturating, in range, and subnormal values.
    latent[0, :, :4] = torch.tensor([600.0, -1000.0, 447.0, 1e-3], device="cuda").to(torch.bfloat16)
    flat = torch.randperm(num_pages * PAGE, device="cuda")[:num_tokens]
    pages, offsets = flat // PAGE, flat % PAGE
    mask = torch.tensor([1, 1, 0, 1, 1, 1, 1, 0, 1], dtype=torch.int8, device="cuda")

    pools = [torch.full((num_pages, 1, 1, PAGE, LATENT), 7.0, device="cuda").to(FP8)]
    pools += [p.clone() for p in pools * (num_layers - 1)]
    expected = [_bytes(p).clone() for p in pools]
    for layer in range(num_layers):
        # A masked-out token stores a zeroed latent, as the eager path does.
        rows = to_draft_fp8(latent[:, layer] * mask.to(torch.bfloat16)[:, None])
        expected[layer][:, 0, 0][pages, offsets] = _bytes(rows)
    ptrs = torch.tensor([p.data_ptr() for p in pools], dtype=torch.int64, device="cuda")
    store_mla_context_latents(latent, pools, ptrs, pages, offsets, mask)
    for layer in range(num_layers):
        assert torch.equal(_bytes(pools[layer]), expected[layer])

    pool = torch.zeros(num_pages, PAGE, LATENT, device="cuda").to(FP8)
    src = latent[:, 0].contiguous()
    scatter_rows_to_pool(src, pool, pages, offsets)
    reference = torch.zeros_like(_bytes(pool))
    reference[pages, offsets] = _bytes(to_draft_fp8(src))
    assert torch.equal(_bytes(pool), reference)
    assert pool[pages[0], offsets[0], :3].float().tolist() == [448.0, -448.0, 448.0]
    assert not pool.float().isnan().any()


@needs_fp8_cuda
def test_fp8_stores_refuse_a_pool_of_another_dtype():
    latent = torch.zeros(2, 1, LATENT, device="cuda", dtype=torch.bfloat16)
    pages = offsets = torch.zeros(2, dtype=torch.long, device="cuda")
    pool = torch.zeros(1, 1, 1, PAGE, LATENT, device="cuda", dtype=torch.float16)
    ptrs = torch.tensor([pool.data_ptr()], dtype=torch.int64, device="cuda")
    mask = torch.ones(2, dtype=torch.int8, device="cuda")
    with pytest.raises(AssertionError):
        store_mla_context_latents(latent, [pool], ptrs, pages, offsets, mask)
    with pytest.raises(AssertionError):
        scatter_rows_to_pool(latent[:, 0], pool[:, 0, 0], pages, offsets)


@needs_fp8_cuda
def test_context_server_stores_saturated_fp8_latents():
    """CTX: the stored pages are the BF16 latents through to_draft_fp8, bit for bit."""
    weights = base._tiny_mla_weights()
    drafter = _build_drafter(weights, context_kv_only=True, context_kv_dtype="fp8")
    reference = _build_drafter(weights)
    manager = _manager(DataType.FP8)
    worker = _worker("VANILLA", context_kv_only=True, context_kv_dtype="fp8")
    worker.set_draft_model(drafter)
    try:
        prompt_len = 45
        torch.manual_seed(1)
        captured = torch.randn(prompt_len, HIDDEN * NUM_CAPTURE, device="cuda").bfloat16()
        base._context_step(worker, drafter, manager, base._request(101, prompt_len), captured)
        history = manager.get_draft_history(101)
        assert (history.valid_length, history.position) == (prompt_len, prompt_len)
        positions = torch.arange(prompt_len, device="cuda")
        expected = base._context_latents(reference, captured, positions)
        for layer in range(LAYERS):
            stored = _draft_page_bytes(manager, 101, layer)[:prompt_len]
            assert torch.equal(stored, _bytes(to_draft_fp8(expected[:, layer, 0])))
            dequant = stored.view(FP8).float()
            exact = expected[:, layer, 0].float()
            stats = _error(dequant, exact)
            _report(f"ctx latents layer {layer}", stats)
            # E4M3 round to nearest: half an ulp, 2^-4 relative, 2^-10 absolute
            # in the subnormal range.
            assert ((dequant - exact).abs() <= exact.abs() * 2**-4 + 2**-10).all()
            assert stats["rel_l2"] < 0.03
    finally:
        manager.shutdown()


# ---------------------------------------------------------------------------
# CTX -> transfer -> GEN, end to end through the worker and the block decode.
# ---------------------------------------------------------------------------


def _ctx_to_gen_draft(monkeypatch, context_kv_dtype, mode):
    """CTX stores the prompt, the pages are byte-copied, GEN drafts one block.

    Returns the drafted hidden states, GEN's context page bytes per layer and
    the BF16 latents of the same hidden states.
    """
    monkeypatch.setenv("TLLM_SPEC_TAIL_FUSION", "0" if mode == "eager" else "1")
    weights = base._tiny_mla_weights()
    ctx_drafter = _build_drafter(
        weights, backend="CUTEDSL", context_kv_only=True, context_kv_dtype=context_kv_dtype
    )
    gen_drafter = _build_drafter(weights, backend="CUTEDSL", context_kv_dtype=context_kv_dtype)
    draft_dtype = DataType.FP8 if context_kv_dtype == "fp8" else DataType.BF16
    ctx_manager, gen_manager = _manager(draft_dtype), _manager(draft_dtype)
    ctx_worker = _worker("CUTEDSL", context_kv_only=True, context_kv_dtype=context_kv_dtype)
    gen_worker = _worker("CUTEDSL", context_kv_dtype=context_kv_dtype)
    ctx_worker.set_draft_model(ctx_drafter)
    gen_worker.set_draft_model(gen_drafter)
    width = BLOCK
    try:
        prompt_len = 70
        torch.manual_seed(2)
        captured = torch.randn(prompt_len, HIDDEN * NUM_CAPTURE, device="cuda").bfloat16()
        base._context_step(
            ctx_worker, ctx_drafter, ctx_manager, base._request(5, prompt_len), captured
        )

        gen_request = base._request(9, prompt_len)
        assert gen_manager.prepare_context(gen_request)
        assert gen_manager.resize_context(gen_request, prompt_len + 2 * width)
        meta = _decode_draft_history(_encode_draft_history(ctx_manager.export_draft_history(5)))
        gen_manager.restore_draft_history(9, meta)
        num_pages = -(-prompt_len // PAGE)
        src_table = ctx_manager.get_draft_block_table([5])[0][:num_pages].long().cuda()
        dst_table = gen_manager.get_draft_block_table([9])[0][:num_pages].long().cuda()
        for layer in range(LAYERS):
            # The transfer copies page bytes.
            gen_pool = _pool_bytes(gen_manager, layer)
            gen_pool[dst_table] = _pool_bytes(ctx_manager, layer)[src_table]

        num_accepted = 2
        gen_captured = torch.randn(width, HIDDEN * NUM_CAPTURE, device="cuda").bfloat16()
        spec = base._Captured([9], gen_captured)
        attn = SimpleNamespace(
            num_seqs=1,
            num_contexts=0,
            num_ctx_tokens=0,
            max_seq_len=MAX_SEQ_LEN,
            _seq_lens=torch.zeros(1, dtype=torch.int32),
            _seq_lens_cuda=torch.zeros(1, dtype=torch.int32, device="cuda"),
        )
        gen_worker.prepare_managed_draft_cache(
            gen_drafter, spec, attn, base._resources(gen_manager)
        )
        accepted = torch.randint(0, VOCAB - 4, (1, width), device="cuda", dtype=torch.int32)
        inputs = gen_worker.prepare_1st_drafter_inputs(
            input_ids=torch.zeros(width, dtype=torch.long, device="cuda"),
            position_ids=torch.arange(width, device="cuda"),
            hidden_states=torch.zeros(width, HIDDEN, device="cuda"),
            accepted_tokens=accepted,
            num_accepted_tokens=torch.tensor([num_accepted], device="cuda", dtype=torch.int32),
            attn_metadata=attn,
            spec_metadata=spec,
            draft_model=gen_drafter,
            total_target_tokens=width,
        )
        assert ("mla_block_fixup" in inputs) == (mode != "eager")
        context_len = prompt_len + num_accepted
        drafted = gen_drafter.dflash_forward(
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
        torch.cuda.synchronize()
        positions = torch.arange(context_len, device="cuda")
        all_captured = torch.cat([captured, gen_captured[:num_accepted]])
        latents = base._context_latents(gen_drafter, all_captured, positions)
        pages = [_draft_page_bytes(gen_manager, 9, layer)[:context_len] for layer in range(LAYERS)]
        return drafted, pages, latents
    finally:
        ctx_manager.shutdown()
        gen_manager.shutdown()


@needs_sm100f
@pytest.mark.parametrize("mode", ["eager", "fused"])
def test_generation_server_drafts_from_an_fp8_context(monkeypatch, mode):
    """GEN pages hold the FP8 latents of the prompt (CTX) and of the accepted
    tokens (GEN), and the draft follows the BF16-context draft within FP8
    tolerance."""
    drafted_fp8, pages_fp8, latents = _ctx_to_gen_draft(monkeypatch, "fp8", mode)
    drafted_bf16, pages_bf16, latents_bf16 = _ctx_to_gen_draft(monkeypatch, "auto", mode)
    assert torch.equal(latents, latents_bf16)
    for layer in range(LAYERS):
        assert torch.equal(pages_fp8[layer], _bytes(to_draft_fp8(latents[:, layer, 0])))
        assert torch.equal(pages_bf16[layer], _bytes(latents[:, layer, 0]))
    stats = _error(drafted_fp8, drafted_bf16)
    _report(f"tiny drafter output, {mode}", stats)
    assert stats["cosine"] > 0.999 and stats["rel_l2"] < 0.05


# ---------------------------------------------------------------------------
# The GEN drafter attention at Kimi K3 shapes: FP8 against BF16.
# ---------------------------------------------------------------------------

K3_PAGE, K3_KV_LORA, K3_ROPE = 64, 512, 64
K3_LATENT = K3_KV_LORA + K3_ROPE


def _k3_block_decoder(heads):
    """What ``MLADSparkForCausalLM._cute_dsl_block_decode`` reads, K3 DSpark geometry."""
    cls = MLADSparkForCausalLM
    stub = SimpleNamespace(
        kv_lora_rank=K3_KV_LORA,
        qk_rope_head_dim=K3_ROPE,
        _num_heads=heads,
        _cute_dsl_checked_shape=None,
        _mla_workspace=None,
        _mla_fp8_scales=None,
        _mla_max_split_kv=0,
        _mla_split_plan=None,
        _mla_split_plan_enabled=False,
    )
    # K3 DSpark YaRN: factor 32, mscale_all_dim 1; qk_nope 128 + rope 64.
    mscale = _yarn_get_mscale(32.0, 1.0)
    stub.softmax_scale = mscale * mscale / math.sqrt(128 + K3_ROPE)
    for name in (
        "_assert_cute_dsl_can_implement",
        "_mla_decode_workspace",
        "_mla_fp8_decode_scales",
        "_mla_split_kv_plan",
        "_cute_dsl_block_decode",
    ):
        setattr(stub, name, types.MethodType(getattr(cls, name), stub))
    return stub


def _k3_operands(ctx_lens, block_size, heads, seed):
    """Paged K3-shaped latents and a block query with drafter-like magnitudes.

    Latents: the RMSNorm'd 512 (unit RMS times a norm weight around 1) and a
    larger k_pe; the absorbed query has unit variance, which with the K3
    softmax scale gives logits with a standard deviation of about 3.
    """
    gen = torch.Generator(device="cuda").manual_seed(seed)
    batch = len(ctx_lens)
    ctx = torch.tensor(ctx_lens, device="cuda")
    last = ctx + block_size - 1
    pages_needed = last // K3_PAGE + 1
    total = int(pages_needed.sum())
    perm = torch.randperm(total, device="cuda", generator=gen)
    table = torch.zeros(batch, int(pages_needed.max()), dtype=torch.long, device="cuda")
    start = 0
    for b, n in enumerate(pages_needed.tolist()):
        table[b, :n] = perm[start : start + n]
        start += n
    norm_weight = 0.5 + torch.rand(K3_KV_LORA, device="cuda", generator=gen)
    pool = torch.empty(total, 1, 1, K3_PAGE, K3_LATENT, dtype=torch.bfloat16, device="cuda")
    for first in range(0, total, 2048):
        rows = pool[first : first + 2048]
        values = torch.randn(rows.shape, device="cuda", generator=gen)
        values[..., :K3_KV_LORA] *= norm_weight
        values[..., K3_KV_LORA:] *= 2.0
        rows.copy_(values)
    query = torch.randn(
        batch, block_size, heads, K3_LATENT, device="cuda", generator=gen
    ).bfloat16()
    fixup = _build_mla_block_fixup(ctx, table, block_size, K3_PAGE)
    return fixup, pool, query


def _reference_block_decode(stub, query, pool, fixup):
    """FP32 attention of every block token over its request's whole KV."""
    batch = query.shape[0]
    flat = pool[:, 0, 0].reshape(-1, K3_LATENT)
    outs = []
    for b in range(batch):
        length = int(fixup.seq_lens_i32[b])
        pos = torch.arange(length, device=query.device)
        rows = fixup.page_tables_i32[b].long()[pos // K3_PAGE] * K3_PAGE + pos % K3_PAGE
        kv = flat[rows].float()
        scores = torch.einsum("shc,tc->sht", query[b].float(), kv) * stub.softmax_scale
        outs.append(torch.einsum("sht,tc->shc", scores.softmax(-1), kv[:, :K3_KV_LORA]))
    return torch.stack(outs)


@needs_sm100f
@pytest.mark.parametrize("heads", [64, 8])
@pytest.mark.parametrize("block_size", [5, 7])
def test_fp8_block_decode_follows_bf16_at_k3_shapes(heads, block_size):
    """64 heads (attention DP: all heads on a rank) and 8 (TP8): 576 latent, page 64,
    contexts up to 137k.

    The FP8 decode reads to_draft_fp8 of the BF16 pool and query. Against an
    FP32 reference on its own FP8 operands it is as close as the BF16 decode is
    to one on BF16 operands, which pins the plumbing (scales, layout, page
    table, bounds, padding); what is left against the BF16 decode is the E4M3
    rounding of q and K, about 9% relative L2 here, where unit-variance queries
    give logits with a standard deviation of about 3.7.
    """
    stub = _k3_block_decoder(heads)
    fixup, pool, query = _k3_operands(
        [3000, 17000, 64000, 137000], block_size, heads, heads * 100 + block_size
    )
    out_bf16 = stub._cute_dsl_block_decode(query, pool, fixup, block_size)
    pool_fp8 = to_draft_fp8(pool)
    out_fp8 = stub._cute_dsl_block_decode(query, pool_fp8, fixup, block_size)
    reference = _reference_block_decode(stub, query, pool, fixup)
    fp8_reference = _reference_block_decode(
        stub, to_draft_fp8(query).to(torch.bfloat16), pool_fp8.to(torch.bfloat16), fixup
    )
    stats = {
        "FP8 vs BF16 decode": _error(out_fp8, out_bf16),
        "FP8 decode vs FP32 reference": _error(out_fp8, reference),
        "BF16 decode vs FP32 reference": _error(out_bf16, reference),
        "FP8 decode vs FP32 on its FP8 operands": _error(out_fp8, fp8_reference),
    }
    for tag, value in stats.items():
        _report(f"K3 heads {heads} block {block_size}: {tag}", value)
    assert stats["BF16 decode vs FP32 reference"]["rel_l2"] < 0.005
    assert stats["FP8 decode vs FP32 on its FP8 operands"]["rel_l2"] < 0.01
    fp8 = stats["FP8 vs BF16 decode"]
    assert fp8["cosine"] > 0.993 and fp8["rel_l2"] < 0.12


@needs_sm100f
def test_fp8_block_decode_needs_a_bf16_query():
    """The FP8 decode writes BF16: an FP16 drafter is refused by the shape check, before the op."""
    stub = _k3_block_decoder(8)
    fixup, pool, query = _k3_operands([3000, 17000], 5, 8, 1)

    with pytest.raises(ValueError, match="BF16 drafter"):
        stub._cute_dsl_block_decode(query.half(), to_draft_fp8(pool), fixup, 5)


# ---------------------------------------------------------------------------
# The block decode's per-request split-KV plan (TLLM_DSPARK_MLA_SPLIT_PLAN).
# ---------------------------------------------------------------------------


def _k3_lengths_fixup(ctx_lens, block_size):
    """A block fixup for these context lengths; only its lengths matter here."""
    ctx = torch.tensor(ctx_lens, device="cuda")
    pages = (max(ctx_lens) + block_size) // K3_PAGE + 1
    table = torch.zeros(len(ctx_lens), pages, dtype=torch.long, device="cuda")
    return _build_mla_block_fixup(ctx, table, block_size, K3_PAGE)


@needs_sm100f
@pytest.mark.parametrize("heads", [64, 8])
@pytest.mark.parametrize("fp8", [False, True], ids=["bf16", "fp8"])
def test_block_decode_split_plan_is_the_planner_mirrors(heads, fp8):
    """What the block decode launches is plan_mla_split_kv at the decode's
    geometry and the pool dtype's piece cost, from one request to a full batch."""
    from tensorrt_llm._torch.attention_backend.fmha import cute_dsl_mla
    from tensorrt_llm._torch.custom_ops.cute_dsl_custom_ops import CuteDSLNVMlaDecodeBlackwellRunner

    block_size = 5
    stub = _k3_block_decoder(heads)
    rows, wave = CuteDSLNVMlaDecodeBlackwellRunner.get_split_plan_geometry(heads, block_size)
    cost = _MLA_SPLIT_PLAN_PIECE_COST_TILES_FP8 if fp8 else _MLA_SPLIT_PLAN_PIECE_COST_TILES
    for lengths in ([137000], [137000] * 5, [3000, 17000, 64000, 137000, 90000], [30000] * 16):
        fixup = _k3_lengths_fixup(lengths, block_size)
        plan = stub._mla_split_kv_plan(fixup, block_size, fp8)
        expected = cute_dsl_mla.plan_mla_split_kv(
            fixup.seq_lens_i32.tolist(),
            rows_per_request=rows,
            wave_clusters=wave,
            piece_cost=cost,
        )
        assert plan.tolist() == expected, lengths


@needs_sm100f
@pytest.mark.parametrize("heads", [64, 8])
@pytest.mark.parametrize("fp8", [False, True], ids=["bf16", "fp8"])
def test_block_decode_with_a_split_plan_follows_the_uniform_split(heads, fp8):
    """A request planned onto another split sums its partial results in another
    order: the BF16 output stays the uniform launch's within BF16 rounding.

    The FP8 decode rounds the softmax probabilities to E4M3 for the PV product, so the
    split structure also moves its rounding error. There the planned decode is held to the
    FP32 reference on the FP8 operands (the bar of the uniform decode's test, with a wider
    band) and to the uniform output within the same band."""
    block_size = 5
    stub = _k3_block_decoder(heads)
    fixup, pool, query = _k3_operands(
        [3000, 17000, 64000, 137000, 90000], block_size, heads, heads + 7
    )
    if fp8:
        pool = to_draft_fp8(pool)
    uniform = stub._cute_dsl_block_decode(query, pool, fixup, block_size)
    stub._mla_split_plan_enabled = True
    planned = stub._cute_dsl_block_decode(query, pool, fixup, block_size)
    plan = stub._mla_split_plan[2]
    assert plan.min() < plan.max(), plan.tolist()
    tag = f"K3 heads {heads} {'fp8' if fp8 else 'bf16'}"
    stats = _error(planned, uniform)
    _report(f"{tag}: planned vs uniform split", stats)
    if not fp8:
        assert stats["cosine"] > 0.9999 and stats["rel_l2"] < 0.005
        return
    fp8_reference = _reference_block_decode(
        stub, to_draft_fp8(query).to(torch.bfloat16), pool.to(torch.bfloat16), fixup
    )
    reference_stats = _error(planned, fp8_reference)
    _report(f"{tag}: planned decode vs FP32 on its FP8 operands", reference_stats)
    assert reference_stats["cosine"] > 0.9995 and reference_stats["rel_l2"] < 0.03
    assert stats["cosine"] > 0.9995 and stats["rel_l2"] < 0.03


@needs_sm100f
@pytest.mark.parametrize("heads", [64, 8])
@pytest.mark.parametrize("fp8", [False, True], ids=["bf16", "fp8"])
def test_block_decode_plans_inside_a_cuda_graph(heads, fp8):
    """A captured step launches its own plan: every replay recomputes it from the
    step's lengths, and the output is the eager one bit for bit."""
    block_size = 5
    stub = _k3_block_decoder(heads)
    stub._mla_split_plan_enabled = True
    fixup, pool, query = _k3_operands([3000, 17000, 64000, 137000], block_size, heads, heads + 21)
    if fp8:
        pool = to_draft_fp8(pool)
    eager = stub._cute_dsl_block_decode(query, pool, fixup, block_size)
    eager_plan = stub._mla_split_plan[2].clone()
    assert eager_plan.max() > 1
    step = fixup._replace()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = stub._cute_dsl_block_decode(query, pool, step, block_size)
    assert stub._mla_split_plan[0] is step
    plan = stub._mla_split_plan[2]
    plan.fill_(1)
    captured.fill_(float("nan"))
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(plan, eager_plan)
    assert torch.equal(captured.view(torch.int16), eager.view(torch.int16))
