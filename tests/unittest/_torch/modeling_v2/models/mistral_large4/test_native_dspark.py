# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Native DSpark math independently of weights, CUDA kernels and runtime sampling."""

import importlib
import json
import struct
from types import MethodType, SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

pytestmark = pytest.mark.cpu_only


@pytest.fixture(scope="module")
def native_checkpoint():
    from utils.llm_data import llm_models_root

    root = llm_models_root()
    path = None if root is None else root / "Mistral-Large-4-1T-A52B-NVFP4"
    if path is None or not (path / "params.json").is_file():
        pytest.skip("the NVFP4 checkpoint is not under LLM_MODELS_ROOT")
    index = json.loads((path / "consolidated.safetensors.index.json").read_text())["weight_map"]
    keys = {key for key in index if key.startswith("mtp_dspark.")}
    headers = {}
    for shard in {index[key] for key in keys}:
        with (path / shard).open("rb") as stream:
            size = struct.unpack("<Q", stream.read(8))[0]
            headers.update(json.loads(stream.read(size)))
    return path, {key: headers[key] for key in keys}


@pytest.fixture(params=["tep4", "dep4", "tp4"])
def modeling(request):
    return importlib.import_module(
        "tensorrt_llm._torch._experimental.modeling_v2.models.mistral_large4."
        f"mistral_large_4_nvfp4__sm_103__{request.param}.modeling"
    )


def _draft(modeling, core):
    spec = SimpleNamespace(
        _draft_model_config={"n_layers": 2, "ragged_attention": "4,4"},
        block_size=3,
        mask_token_id=3,
        target_layer_ids=[0, 1],
    )
    shell = SimpleNamespace(model=core, spec_config=spec)
    return modeling.DraftModel(shell, SimpleNamespace(pretrained_config=SimpleNamespace()))


@pytest.mark.parametrize("next_n", [1, 4])
@pytest.mark.parametrize("mixed", [False, True])
@pytest.mark.parametrize("max_seq_len", [32768, 1048576])
@pytest.mark.parametrize("unfused", [False, True])
def test_indexer_preserves_request_geometry(
    modeling, monkeypatch, next_n, mixed, max_seq_len, unfused
):
    """Both indexer paths retain per-request lengths and the selector's compiled width."""
    batch, heads, dim, topk = 3, 2, 4, 2
    tc = 3 if mixed else 0
    rows = tc + batch * next_n
    table = torch.zeros(16, 4)
    compiled = object() if max_seq_len == 32768 else None
    core = SimpleNamespace(
        idx_dim=dim,
        idx_heads=heads,
        idx_topk=topk,
        idx_eps=1e-5,
        idx_weight_scale=1.0,
        rope_dim=4,
        _rope=SimpleNamespace(indexer=table),
        _ix_k={True: object()},
        _ix_k_checked=True,
        _ix_q=object(),
        _ix_topk=compiled,
        _max_ctx_len=max_seq_len,
        unfused_glue=unfused,
    )
    ix = SimpleNamespace(
        rope=True,
        wk_wp=None,
        k_norm_w=None,
        k_norm_b=None,
        q_norm_w=None,
        q_norm_b=None,
        wq_b=None,
        wq_b_s=None,
        ssmax_s=None,
        ssmax_b=None,
    )
    lengths = torch.tensor(([17] if mixed else []) + [31, 2049, max_seq_len])
    blocks = torch.arange(lengths.numel()).view(-1, 1)
    step = dict(
        num_contexts=int(mixed),
        num_seqs=batch + int(mixed),
        ix_pools={0: torch.empty(1)},
        slot_mapping_fp8=None,
        slot_mapping_scale=None,
        kv_lens_cuda=lengths,
        indexer_k_cache_block_offsets=blocks,
        scheduler_metadata_buffer=object(),
        radix_aux_indices=object(),
        radix_aux_logits=object(),
        indexer_prefill_q_split=None,
    )
    monkeypatch.setattr(modeling, "cublas_mm", lambda *a: torch.zeros(rows, dim + heads))
    monkeypatch.setattr(modeling, "cute_dsl_indexer_k_cache_update", lambda *a, **kw: a[5])
    monkeypatch.setattr(modeling, "layer_norm", lambda x, *a: x)
    scale = Mock(side_effect=lambda q, *a: q)
    fused = Mock(side_effect=lambda *a: (a[7], a[8]))
    q_side = Mock(side_effect=lambda *a, **kw: (a[5], a[6]))
    monkeypatch.setattr(modeling, "ssmax_scaling", scale)
    monkeypatch.setattr(modeling, "ssmax_indexer_q_side_fused", fused)
    monkeypatch.setattr(modeling, "cute_dsl_indexer_q_side_fused", q_side)
    scorer, generic, radix = Mock(), Mock(), Mock()
    monkeypatch.setattr(modeling, "cute_dsl_fp8_paged_mqa_logits", scorer)
    monkeypatch.setattr(modeling, "indexer_topk_decode", generic)
    monkeypatch.setattr(modeling, "cute_dsl_radix_topk_decode", radix)
    target = SimpleNamespace(
        core=core, _linear_fp8=lambda *a: torch.zeros(rows, heads * dim), _select_ctx=Mock()
    )
    output = torch.full((rows, topk), -1, dtype=torch.int32)
    args = (
        target,
        step,
        torch.zeros(rows, dim),
        torch.arange(rows),
        SimpleNamespace(index=0, indexer=ix),
        output,
    )
    if mixed:
        modeling.PrefillTarget._indexer(*args, tc, False, False)
        target._select_ctx.assert_called_once()
    else:
        modeling.DecodeTarget._indexer(*args, False)
    query, _, weights, kv_lens, page_table, *_ = scorer.call_args.args
    assert query.shape == (batch, next_n, heads, dim)
    assert weights.shape == (batch * next_n, heads)
    torch.testing.assert_close(kv_lens, lengths[int(mixed) :])
    torch.testing.assert_close(page_table, blocks[int(mixed) :])
    assert scale.call_count == q_side.call_count == int(unfused)
    assert fused.call_count == int(not unfused)
    if next_n == 1 and compiled is not None:
        radix.assert_called_once()
        generic.assert_not_called()
        assert radix.call_args.kwargs == dict(next_n=1, compiled=compiled)
        selected = radix.call_args.args[2]
    else:
        generic.assert_called_once()
        radix.assert_not_called()
        assert generic.call_args.args[3] == next_n
        selected = generic.call_args.args[2]
    assert selected.shape == (batch * next_n, topk)
    assert selected.data_ptr() == output[tc:].data_ptr()


@pytest.mark.parametrize("with_images", [False, True])
def test_target_image_embeddings_preserve_worker_token_ids(modeling, monkeypatch, with_images):
    ids = torch.tensor([1, 2, 3])
    embed = torch.arange(20, dtype=torch.float32).reshape(5, 4)
    image_rows = torch.full((2, 4), -7.0)
    hidden = torch.ones(3, 4)
    logits = torch.ones(3, 5)
    core = Mock(return_value=hidden)
    core.w = {"embed": embed}
    worker = Mock(return_value={"logits": logits})
    shell = SimpleNamespace(
        model=core,
        _vision=object(),
        spec_worker=worker,
        draft_model=object(),
        softcap=8.0,
        lm_head=object(),
        logits_processor=Mock(return_value=logits),
    )
    monkeypatch.setattr(modeling, "_image_rows", lambda *args: image_rows)
    monkeypatch.setattr(modeling, "tanh_softcap", lambda x, cap: cap * torch.tanh(x / cap))
    family = modeling.__name__.split("__")[-1].split(".")[0].capitalize()
    target = getattr(modeling, f"ModelingV2MistralLarge4Nvfp4Sm103{family}")
    result = target.forward(
        shell,
        attn_metadata=object(),
        input_ids=ids,
        spec_metadata=SimpleNamespace(gather_ids=torch.arange(3)),
        multimodal_params=[object()] if with_images else [],
        mm_token_indices=torch.tensor([1, 2]),
    )
    assert result is worker.return_value
    assert worker.call_args.kwargs["input_ids"] is ids
    assert worker.call_args.kwargs["hidden_states"] is hidden
    if with_images:
        assert core.call_args.kwargs["input_ids"] is None
        torch.testing.assert_close(
            core.call_args.kwargs["inputs_embeds"], torch.cat((embed[1:2], image_rows))
        )
    else:
        assert core.call_args.kwargs["input_ids"] is ids
        assert core.call_args.kwargs["inputs_embeds"] is None


@pytest.mark.parametrize("enabled", [False, True])
def test_native_shell_connects_guided_decoder_to_worker(modeling, enabled):
    from tensorrt_llm._torch.speculative.interface import SpecWorkerBase

    decoder = object()
    worker = SimpleNamespace(guided_decoder=None)
    worker.set_guided_decoder = MethodType(SpecWorkerBase.set_guided_decoder, worker)
    shell = SimpleNamespace(spec_worker=worker if enabled else None)
    family = modeling.__name__.split("__")[-1].split(".")[0].capitalize()
    target = getattr(modeling, f"ModelingV2MistralLarge4Nvfp4Sm103{family}")
    assert target.set_guided_decoder(shell, decoder) is enabled
    assert worker.guided_decoder is (decoder if enabled else None)


def test_rnn_uses_hidden_and_resets_each_proposal(modeling, monkeypatch):
    monkeypatch.setattr(
        modeling, "cublas_mm", lambda x, w, bias, dtype: x @ w + (0 if bias is None else bias)
    )
    # Rank-one head with explicit scalar equations: the candidate reads the
    # token and h0, the output reads the previous state, token and h1.
    weights = {
        "draft_markov_embedding": torch.tensor([[0.2], [0.7], [-0.5]], dtype=torch.float64),
        "draft_markov_output": torch.tensor([[-1.0], [0.0], [1.0]], dtype=torch.float64),
        "draft_joint": torch.tensor(
            [[0, 0, 0, 0], [1, 1, 1, 0], [2, -1, 0, 3]], dtype=torch.float64
        ),
        "draft_joint_bias": torch.zeros(3, dtype=torch.float64),
    }
    draft = _draft(modeling, SimpleNamespace(w=weights, heads=1, lat_dim=4))
    hidden = torch.tensor(
        [[[0.1, 0.3], [0.4, -0.2], [-0.5, 0.1]], [[0.3, -0.4], [-0.1, 0.6], [0.2, 0.2]]],
        dtype=torch.float64,
    )
    base = torch.tensor([[[0, 2, 0], [0, 0, 0], [0, 0, 0]]] * 2, dtype=torch.float64)
    anchors = torch.tensor([0, 2])
    expected = torch.empty_like(base)
    for b in range(2):
        state, previous = torch.tensor(0.0, dtype=torch.float64), int(anchors[b])
        for i in range(3):
            token = (0.2, 0.7, -0.5)[previous]
            emitted = torch.tanh(2 * state - token + 3 * hidden[b, i, 1])
            state = (state + torch.tanh(state + token + hidden[b, i, 0])) / 2
            expected[b, i] = base[b, i] + torch.stack((-emitted, emitted * 0, emitted))
            previous = int(expected[b, i].argmax())
    for _ in range(2):
        actual = draft.apply_markov_chain_logits(
            base,
            anchors,
            hidden_states=hidden,
            argmax_fn=lambda logits: logits.argmax(-1),
        )
        torch.testing.assert_close(actual, expected)
    # Rows carry independent recurrent state, even when request order changes.
    reversed_rows = draft.apply_markov_chain_logits(
        base.flip(0),
        anchors.flip(0),
        hidden_states=hidden.flip(0),
        argmax_fn=lambda logits: logits.argmax(-1),
    )
    torch.testing.assert_close(reversed_rows, expected.flip(0))


def test_position_specific_masks_do_not_index_the_vocabulary(modeling):
    weights = {
        "embed": torch.arange(12).reshape(3, 4),
        "draft_mask": torch.tensor([[10, 11, 12, 13], [20, 21, 22, 23]]),
    }
    draft = _draft(modeling, SimpleNamespace(w=weights, heads=1, lat_dim=4))
    actual = draft.embed_draft_block(torch.tensor([2, 0]), 3)
    assert actual.shape == (2, 3, 4)
    torch.testing.assert_close(actual[:, 0], weights["embed"][[2, 0]])
    torch.testing.assert_close(actual[:, 1:], weights["draft_mask"].expand(2, -1, -1))
    padded = draft.embed_draft_block(torch.tensor([-1, 3]), 3)
    assert padded.shape == (2, 3, 4)
    assert torch.count_nonzero(padded[:, 0]) == 0
    torch.testing.assert_close(padded[:, 1:], weights["draft_mask"].expand(2, -1, -1))


@pytest.mark.parametrize("layout", ["context", "block", "query"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_rotary_positions_match_complex_rotation(modeling, layout, dtype):
    generator = torch.Generator().manual_seed(941)
    angles = torch.arange(32)[:, None] * torch.tensor([1.0, 0.3, 0.1, 0.03])
    cos, sin = angles.cos().to(dtype).float(), angles.sin().to(dtype).float()
    draft = _draft(modeling, SimpleNamespace(w={}, heads=2, lat_dim=8))
    draft._rope = SimpleNamespace(indexer=torch.cat((cos, sin), dim=-1))
    positions = torch.tensor([[0, 4, 11], [7, 2, 15]])
    if layout == "context":
        positions = positions.flatten()
    shape = (*positions.shape, *((2,) if layout == "query" else ()), 8)
    x = torch.randn(shape, generator=generator).to(dtype)
    phase = torch.complex(cos[positions], sin[positions])
    if layout == "query":
        phase = phase.unsqueeze(-2)
    pairs = torch.view_as_complex(x.float().reshape(*shape[:-1], 4, 2))
    expected = torch.view_as_real(pairs * phase).flatten(-2).to(dtype)
    actual = draft._rotate(x, positions)
    assert actual.dtype == dtype
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("recurrent", [False, True])
@pytest.mark.parametrize("sharded", [False, True])
def test_worker_forwards_backbone_hidden_only_to_recurrent_head(recurrent, sharded):
    from tensorrt_llm._torch.speculative.dspark import DSparkWorker

    hidden = torch.randn(2, 3, 4)
    logits = torch.randn(2, 3, 4 if sharded else 8)
    anchors = torch.tensor([0, 1])
    metadata = object()
    global_tokens = torch.tensor([6, 7])
    sampler = Mock(return_value=global_tokens)
    worker = SimpleNamespace(
        mapping=SimpleNamespace(enable_attention_dp=False, tp_size=2, tp_rank=1),
        greedy_sample_draft_with_tp_gather=sampler,
    )
    worker._apply_dspark_markov_bias = MethodType(DSparkWorker._apply_dspark_markov_bias, worker)

    def head(base, first, *, argmax_fn, vocab_slice, **kwargs):
        assert base is logits and first is anchors
        assert vocab_slice == (slice(4, 8) if sharded else None)
        if recurrent:
            assert kwargs["hidden_states"] is hidden
        else:
            assert not kwargs
        assert argmax_fn(base[:, 0]) is global_tokens
        return base + 1

    draft = SimpleNamespace(
        has_markov_head=True,
        has_recurrent_markov_head=recurrent,
        markov_w2=torch.empty(8, 2),
        apply_markov_chain_logits=head,
    )
    actual = DSparkWorker._refine_block_logits(
        worker, draft, logits, {"first_prev_tokens": anchors}, metadata, hidden
    )
    torch.testing.assert_close(actual, logits + 1)
    assert sampler.call_count == 1
    assert sampler.call_args.args[1] is metadata
    if recurrent:
        with pytest.raises(ValueError, match="requires the draft backbone hidden states"):
            DSparkWorker._refine_block_logits(
                worker, draft, logits, {"first_prev_tokens": anchors}, metadata
            )


@pytest.mark.parametrize("phase", ["PrefillTarget", "DecodeTarget"])
def test_block_attention_matches_explicit_kv_and_obeys_window(modeling, monkeypatch, phase):
    # Compare the absorbed latent path with ordinary per-head K/V attention.
    # Poison old history, unused pages and padding: any unintended read must
    # either change the result or propagate a NaN.
    generator = torch.Generator().manual_seed(901)

    def rand(*shape):
        return torch.randn(*shape, generator=generator, dtype=torch.float64) * 0.1

    def norm(x, weight, eps):
        return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + eps) * weight

    monkeypatch.setattr(modeling, "cublas_mm", lambda x, w, bias, dtype: x @ w)
    monkeypatch.setattr(modeling, "bmm_out", lambda a, b, out: out.copy_(torch.bmm(a, b)))
    monkeypatch.setattr(
        modeling,
        "flashinfer_silu_and_mul",
        lambda x: torch.nn.functional.silu(x.chunk(2, -1)[0]) * x.chunk(2, -1)[1],
    )
    monkeypatch.setattr(modeling, "flashinfer_rmsnorm", norm)
    if hasattr(modeling, "cute_dsl_allreduce_add_rmsnorm"):

        def add_norm(partial, residual, weight, out, workspace):
            residual.add_(partial)
            return norm(residual, weight, 1e-5)

        monkeypatch.setattr(modeling, "cute_dsl_allreduce_add_rmsnorm", add_norm)
    core = SimpleNamespace(
        heads=2,
        lat_dim=6,
        hidden=4,
        nope=2,
        rope_dim=2,
        kv_lora=4,
        v_dim=2,
        qk_dim=4,
        eps=1e-5,
        softmax_scale=0.5,
        _lamport=None,
        rope_rows=32,
        w={},
    )
    draft = _draft(modeling, core)
    draft._rope = SimpleNamespace(rows=32, indexer=torch.tensor([[1.0, 0.0]] * 32))
    draft._layers = [
        SimpleNamespace(
            attn_norm=torch.ones(4),
            ffn_norm=torch.ones(4),
            next_norm=torch.ones(4),
            q=rand(4, 8),
            kv_a=rand(4, 6),
            kv_norm=torch.ones(4),
            k_b=rand(2, 2, 4),
            v_b=rand(2, 2, 4),
            out=rand(4, 4),
            gate=rand(4, 8),
            up=rand(4, 8),
            down=rand(8, 4),
        )
        for _ in range(2)
    ]
    core.w.update(embed=rand(2, 4), draft_mask=rand(2, 4))
    noise = draft.embed_draft_block(torch.tensor([1, 0]), draft.block_size)
    lengths = torch.tensor([7, 2])
    # Request-to-slot and logical-to-physical pages are both permuted.
    slots = torch.tensor([2, 0])
    table = torch.tensor([[4, 2, 7, 1], [0, 0, 0, 0], [5, 3, 6, 8]])
    contexts = [rand(2, 7, 6) for _ in range(2)]
    pools = [torch.full((9, 1, 1, 2, 6), float("nan"), dtype=torch.float64) for _ in range(2)]
    for layer in range(2):
        for b, length in enumerate(lengths.tolist()):
            for pos in range(max(0, length - 4), length):
                pools[layer][table[slots[b], pos // 2], 0, 0, pos % 2] = contexts[layer][b, pos]
    target = getattr(modeling, phase).__new__(getattr(modeling, phase))
    target.core = core
    actual = target.dspark_forward(
        draft,
        noise,
        torch.tensor([[7, 8, 9], [2, 3, 4]]),
        lengths,
        None,
        None,
        slots,
        pools,
        table,
    ).reshape(2, 3, 4)
    reference = []
    for b, length in enumerate(lengths.tolist()):
        residual = noise[b].clone()
        x = norm(residual, draft._layers[0].attn_norm, core.eps)
        for i, layer in enumerate(draft._layers):
            q = (x @ layer.q).reshape(3, 2, 4)
            kv = x @ layer.kv_a
            block_kv = torch.cat((norm(kv[:, :4], layer.kv_norm, core.eps), kv[:, 4:]), -1)
            ctx = contexts[i][b, max(0, length - 4) : length]
            latent = torch.cat((ctx, block_kv), 0)
            k = torch.cat(
                (
                    torch.einsum("tc,hdc->thd", latent[:, :4], layer.k_b),
                    latent[:, None, 4:].expand(-1, 2, -1),
                ),
                -1,
            )
            v = torch.einsum("tc,hvc->thv", latent[:, :4], layer.v_b)
            scores = torch.einsum("shd,thd->sht", q, k) * 0.5
            values = torch.einsum("sht,thv->shv", scores.softmax(-1), v)
            residual += values.flatten(1) @ layer.out
            mlp_in = norm(residual, layer.ffn_norm, core.eps)
            residual += (
                torch.nn.functional.silu(mlp_in @ layer.gate) * (mlp_in @ layer.up)
            ) @ layer.down
            x = norm(residual, layer.next_norm, core.eps)
        reference.append(x)
    torch.testing.assert_close(actual, torch.stack(reference), rtol=2e-6, atol=2e-7)


@pytest.mark.parametrize("world", [4, 8])
def test_native_draft_manifest_matches_checkpoint_headers(modeling, native_checkpoint, world):
    from tensorrt_llm._torch.models.checkpoints.mistral.config_loader import MistralConfigLoader
    from tensorrt_llm.llmapi import DSparkDecodingConfig
    from tensorrt_llm.mapping import Mapping

    path, headers = native_checkpoint
    # The configuration the engine adapts from the checkpoint's params.json.
    cfg = MistralConfigLoader().load_pretrained_config(str(path))
    spec = DSparkDecodingConfig(max_draft_len=3)
    from tensorrt_llm._torch.models.checkpoints.dspark_config import resolve_dspark_checkpoint

    resolve_dspark_checkpoint(spec, str(path))
    target = modeling.__name__.split(".")[-2]
    tp = target.endswith("__tp4")
    dep = target.endswith("__dep4")
    table = modeling._weights.MODEL_WEIGHTS
    for rank in range(world):
        mapping = Mapping(
            world_size=world,
            rank=rank,
            gpus_per_node=4,
            tp_size=world,
            moe_tp_size=world if tp else 1,
            moe_ep_size=1 if tp else world,
            enable_attention_dp=dep,
        )
        core = SimpleNamespace(
            model_config=SimpleNamespace(
                pretrained_config=cfg, mapping=mapping, torch_dtype=torch.bfloat16, spec_config=spec
            )
        )
        manifest = table.manifest(core)
        with torch.device("meta"):
            declared = table.declare(table.dims(core))
        consumed = set()
        for name, rows in manifest.items():
            if not name.startswith("draft_") and "_draft_" not in name:
                continue
            assert len(rows) == 1
            key, index, transform, *sharding = rows[0]
            assert index is None and transform is None
            assert headers[key]["dtype"] == "BF16"
            source = torch.empty(headers[key]["shape"], device="meta", dtype=torch.bfloat16)
            if sharding and sharding[0] is not None:
                source = sharding[0](source)
            assert source.shape == declared[name].shape, (rank, name, key)
            consumed.add(key)
        assert consumed == headers.keys()
        assert len(consumed) == 38
