# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""ChatGLM3-6B ``_torch`` modeling tests."""

import functools
import json
import os
from copy import deepcopy
from dataclasses import dataclass

import pytest
import torch
from safetensors.torch import load_file

import tensorrt_llm
from tensorrt_llm import LLM
from tensorrt_llm._torch.attention_backend.interface import RopeParams
from tensorrt_llm._torch.attention_backend.utils import get_attention_backend
from tensorrt_llm._torch.metadata import KVCacheParams
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.models.modeling_chatglm import ChatGLMForCausalLM, normalize_chatglm_config
from tensorrt_llm._torch.models.modeling_utils import MODEL_CLASS_MAPPING, get_model_architecture
from tensorrt_llm._torch.modules.rotary_embedding import RotaryEmbedding
from tensorrt_llm._torch.pyexecutor.kv_cache_manager_v2 import KVCacheManagerV2
from tensorrt_llm.llmapi import CudaGraphConfig, KvCacheConfig, SamplingParams
from tensorrt_llm.mapping import Mapping

CHATGLM3_CKPT = os.environ.get(
    "CHATGLM3_CKPT",
    "/lustre/fs1/portfolios/coreai/projects/coreai_comparch_trtllm/users/kleinc/hf_data/chatglm3-6b",
)

EXPECTED = dict(
    num_hidden_layers=28,
    hidden_size=4096,
    intermediate_size=13696,
    vocab_size=65024,
    num_attention_heads=32,
    num_key_value_heads=2,
    head_dim=128,
    max_position_embeddings=8192,
    partial_rotary_factor=0.5,
)

PROMPT_IDS = [64790, 64792, 790, 30951, 517, 269, 30, 54761, 31211, 30910]

PROMPTS = [
    "The capital of France is",
    "1 + 1 =",
    "The opposite of hot is",
    "Water is made of hydrogen and",
    "The sun rises in the",
]

COS_TOL = 0.999
MEAN_ABS_TOL = 2e-2
MAX_ABS_TOL = 2.5e-1
# Deep residual-stream boundaries carry large fp16 magnitudes; a single-element abs
# outlier can graze the flat cap at cosine~=1, so allow a magnitude-relative fallback
# there (cosine + mean_abs remain the strict aggregate correctness gates).
REL_MAX_ABS_TOL = 1e-2

skip_no_ckpt = pytest.mark.skipif(
    not os.path.isdir(CHATGLM3_CKPT), reason=f"ChatGLM3 checkpoint not found at {CHATGLM3_CKPT}"
)
skip_no_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")


@dataclass(repr=False)
class RuntimeCfg:
    cuda_graph: bool
    overlap_scheduler: bool

    def __repr__(self) -> str:
        return f"cuda_graph:{self.cuda_graph}-overlap:{self.overlap_scheduler}"


BASELINE = RuntimeCfg(cuda_graph=False, overlap_scheduler=False)
ENABLED = RuntimeCfg(cuda_graph=True, overlap_scheduler=True)
CFGS = [pytest.param(BASELINE, id="baseline"), pytest.param(ENABLED, id="cuda_graph")]


@pytest.fixture(autouse=True)
def _force_single_process_worker(monkeypatch):
    monkeypatch.setenv("TLLM_WORKER_USE_SINGLE_PROCESS", "1")


def _patch_chatglm3_hf_tied_compat():
    from transformers.dynamic_module_utils import get_class_from_dynamic_module

    cls = get_class_from_dynamic_module(
        "modeling_chatglm.ChatGLMForConditionalGeneration", CHATGLM3_CKPT
    )
    if not hasattr(cls, "all_tied_weights_keys"):
        cls.all_tied_weights_keys = {}


@functools.lru_cache(maxsize=1)
def _load_checkpoint_weights(ckpt: str) -> dict:
    index = os.path.join(ckpt, "model.safetensors.index.json")
    weights = {}
    with open(index) as f:
        shards = set(json.load(f)["weight_map"].values())
    for shard in sorted(shards):
        weights.update(load_file(os.path.join(ckpt, shard)))
    return weights


def _metrics(a: torch.Tensor, b: torch.Tensor) -> dict:
    a = a.float().reshape(-1)
    b = b.float().reshape(-1)
    return dict(
        max_abs=(a - b).abs().max().item(),
        mean_abs=(a - b).abs().mean().item(),
        cosine=torch.nn.functional.cosine_similarity(a, b, dim=0).item(),
    )


def _make_kv_cache_manager_v2(config, num_blocks=8, tokens_per_block=128):
    return KVCacheManagerV2(
        KvCacheConfig(max_tokens=num_blocks * tokens_per_block),
        tensorrt_llm.bindings.internal.batch_manager.CacheType.SELF,
        num_layers=config.num_hidden_layers,
        num_kv_heads=config.num_key_value_heads,
        head_dim=config.head_dim,
        tokens_per_block=tokens_per_block,
        max_seq_len=num_blocks * tokens_per_block,
        max_batch_size=1,
        mapping=Mapping(world_size=1, tp_size=1, rank=0),
        dtype=tensorrt_llm.bindings.DataType.HALF,
    )


class _HFRef:
    _model = None
    _tok = None

    @classmethod
    def get(cls):
        if cls._model is None:
            from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

            hf_config = AutoConfig.from_pretrained(CHATGLM3_CKPT, trust_remote_code=True)
            hf_config.torch_dtype = torch.float16
            if getattr(hf_config, "max_length", None) is None:
                hf_config.max_length = 8192
            _patch_chatglm3_hf_tied_compat()
            cls._tok = AutoTokenizer.from_pretrained(CHATGLM3_CKPT, trust_remote_code=True)
            cls._model = (
                AutoModelForCausalLM.from_pretrained(
                    CHATGLM3_CKPT,
                    config=hf_config,
                    trust_remote_code=True,
                    torch_dtype=torch.float16,
                )
                .cuda()
                .eval()
            )
        return cls._model, cls._tok

    @classmethod
    def config(cls):
        model, _ = cls.get()
        return model.config

    @classmethod
    def encode(cls, prompt: str):
        _, tok = cls.get()
        return tok(prompt, return_tensors="pt").input_ids[0].tolist()

    @classmethod
    @torch.inference_mode()
    def next_logits(cls, token_ids):
        model, _ = cls.get()
        ids = torch.tensor([token_ids], device="cuda")
        out = model(input_ids=ids, use_cache=True, return_dict=True)
        return out.logits[0, -1].float()

    @classmethod
    @torch.inference_mode()
    def greedy_generate(cls, token_ids, max_new_tokens, suppress_ids=()):
        model, _ = cls.get()
        device = "cuda"
        suppress = list(suppress_ids)
        prompt_len = len(token_ids)
        ids = torch.tensor([token_ids], device=device)
        pos = torch.arange(prompt_len, device=device).unsqueeze(0)
        out = model(input_ids=ids, position_ids=pos, use_cache=True, return_dict=True)
        past = out.past_key_values
        cur = out.logits[0, -1].float()

        tokens, step_logits = [], []
        cur_pos = prompt_len
        for _ in range(max_new_tokens):
            step_logits.append(cur.cpu())
            masked = cur.clone()
            for sid in suppress:
                masked[sid] = float("-inf")
            nxt = int(masked.argmax())
            tokens.append(nxt)
            step_out = model(
                input_ids=torch.tensor([[nxt]], device=device),
                position_ids=torch.tensor([[cur_pos]], device=device),
                past_key_values=past,
                use_cache=True,
                return_dict=True,
            )
            past = step_out.past_key_values
            cur = step_out.logits[0, -1].float()
            cur_pos += 1
        return tokens, step_logits


def _build_llm(cfg: RuntimeCfg) -> LLM:
    kv_cache_config = KvCacheConfig(
        free_gpu_memory_fraction=0.5,
        enable_block_reuse=False,
        use_kv_cache_manager_v2=True,
    )
    return LLM(
        model=CHATGLM3_CKPT,
        trust_remote_code=True,
        attn_backend="TRTLLM",
        dtype="float16",
        kv_cache_config=kv_cache_config,
        cuda_graph_config=CudaGraphConfig() if cfg.cuda_graph else None,
        disable_overlap_scheduler=not cfg.overlap_scheduler,
        max_batch_size=8,
        max_num_tokens=8192,
    )


def _find_cuda_graph_runner(llm: LLM):
    from tensorrt_llm._torch.pyexecutor.cuda_graph_runner import CUDAGraphRunner

    engine = getattr(getattr(llm, "_executor", None), "engine", None)
    runner = getattr(getattr(engine, "model_engine", None), "cuda_graph_runner", None)
    return runner if isinstance(runner, CUDAGraphRunner) else None


def _assert_cuda_graph_hard_path(llm: LLM, cfg: RuntimeCfg) -> None:
    runner = _find_cuda_graph_runner(llm)
    if cfg.cuda_graph:
        assert runner is not None, (
            "enabled config: could not reach the in-process CUDAGraphRunner "
            "(TLLM_WORKER_USE_SINGLE_PROCESS=1 is required for this introspection)"
        )
        graphs = runner.graphs
        print(
            f"[cuda_graph_hard_path] cfg={cfg} enabled={runner.enabled} "
            f"num_captured_graphs={len(graphs)} keys={list(graphs)[:8]}"
        )
        assert runner.enabled, "enabled config: CUDAGraphRunner.enabled is False (silent fallback)"
        assert len(graphs) >= 1, (
            "enabled config captured 0 CUDA graphs -> silent eager fallback, not the hard path"
        )
        assert all(isinstance(g, torch.cuda.CUDAGraph) for g in graphs.values()), (
            "captured graph objects are not all torch.cuda.CUDAGraph"
        )
    elif runner is not None:
        print(f"[cuda_graph_hard_path] cfg={cfg} enabled={runner.enabled}")
        assert not runner.enabled, "baseline config: CUDAGraphRunner.enabled unexpectedly True"
        assert len(runner.graphs) == 0, "baseline config unexpectedly captured CUDA graphs"


def _assert_v2_and_backend(llm: LLM, cfg: RuntimeCfg) -> None:
    args = llm.args
    assert args.kv_cache_config.use_kv_cache_manager_v2 is True
    assert str(getattr(args, "attn_backend", "TRTLLM")).upper() == "TRTLLM"
    if cfg.cuda_graph:
        assert args.cuda_graph_config is not None
    assert args.disable_overlap_scheduler is (not cfg.overlap_scheduler)
    _assert_cuda_graph_hard_path(llm, cfg)


def _greedy_params(max_tokens: int, logits: bool = False) -> SamplingParams:
    return SamplingParams(
        max_tokens=max_tokens, temperature=0.0, top_k=1, return_generation_logits=logits
    )


def test_chatglm3_config_architecture_registration():
    for arch in ("ChatGLMModel", "ChatGLMForConditionalGeneration", "ChatGLMForCausalLM"):
        assert MODEL_CLASS_MAPPING[arch] is ChatGLMForCausalLM


def test_chatglm3_config_normalization():
    class _Cfg:
        model_type = "chatglm"
        num_layers = 28
        padded_vocab_size = 65024
        vocab_size = 65024
        hidden_size = 4096
        ffn_hidden_size = 13696
        kv_channels = 128
        num_attention_heads = 32
        multi_query_attention = True
        multi_query_group_num = 2
        seq_length = 8192
        layernorm_epsilon = 1e-5
        add_qkv_bias = True
        add_bias_linear = False

    cfg = _Cfg()
    normalize_chatglm_config(cfg)
    for k, v in EXPECTED.items():
        assert getattr(cfg, k) == v, f"{k}: {getattr(cfg, k)} != {v}"
    assert cfg.rms_norm_eps == 1e-5
    assert cfg.attention_bias is True
    assert cfg.mlp_bias is False


@skip_no_cuda
@skip_no_ckpt
def test_chatglm3_config_and_weight_load():
    device = torch.device("cuda")

    model_config = ModelConfig.from_pretrained(CHATGLM3_CKPT, trust_remote_code=True)
    hf_config = model_config.pretrained_config
    for k, v in EXPECTED.items():
        assert getattr(hf_config, k) == v, f"{k}: {getattr(hf_config, k)} != {v}"
    assert hf_config.torch_dtype == torch.float16
    assert hf_config.rms_norm_eps == 1e-5

    model_cls, _ = get_model_architecture(hf_config)
    assert model_cls is ChatGLMForCausalLM

    with torch.device(device):
        model = ChatGLMForCausalLM(model_config).to(device).eval()

    weights = _load_checkpoint_weights(CHATGLM3_CKPT)
    assert "transformer.rotary_pos_emb.inv_freq" in weights

    model.load_weights(weights)

    for name, p in model.named_parameters():
        assert torch.isfinite(p).all(), f"non-finite parameter after load: {name}"

    assert model.lm_head.weight.data_ptr() != model.model.embed_tokens.weight.data_ptr()
    assert not torch.equal(
        model.lm_head.weight.detach().float(),
        model.model.embed_tokens.weight.detach().float(),
    )

    assert model_config.attn_backend == "TRTLLM"
    kv_cache_manager = _make_kv_cache_manager_v2(hf_config)
    assert isinstance(kv_cache_manager, KVCacheManagerV2)
    try:
        input_ids = torch.tensor(PROMPT_IDS, dtype=torch.int, device=device)
        request_ids, token_nums = [1], [input_ids.size(-1)]
        kv_cache_manager.add_dummy_requests(request_ids, token_nums)
        metadata_cls = get_attention_backend(model_config.attn_backend).Metadata
        attn_metadata = metadata_cls(
            seq_lens=torch.tensor([input_ids.size(-1)], dtype=torch.int),
            num_contexts=1,
            kv_cache_params=KVCacheParams(use_cache=True, num_cached_tokens_per_seq=[0]),
            max_num_requests=1,
            max_num_tokens=8192,
            kv_cache_manager=kv_cache_manager,
            request_ids=request_ids,
            prompt_lens=token_nums,
        )
        position_ids = torch.arange(input_ids.size(-1)).unsqueeze(0).to(device)
        with torch.inference_mode():
            attn_metadata.prepare()
            logits = model.forward(
                input_ids=input_ids, position_ids=position_ids, attn_metadata=attn_metadata
            )
        assert logits.shape[-1] == hf_config.vocab_size == 65024
        assert torch.isfinite(logits).all()
    finally:
        kv_cache_manager.shutdown()
        del model
        torch.cuda.empty_cache()


@skip_no_cuda
def test_chatglm3_partial_rope_boundary_and_theta():
    device = torch.device("cuda")
    head_dim, rotary_dim, seq = 128, 64, 6
    rope = RopeParams(dim=rotary_dim, theta=10000.0, max_positions=8192)
    rotary = RotaryEmbedding(rope, head_dim=head_dim, is_neox=False).to(device)

    q = torch.randn(1, seq, 4 * head_dim, device=device, dtype=torch.float32)
    pos = torch.arange(seq, device=device)
    (q_rot,) = rotary(pos, [q.clone()])

    q_v = q.view(1, seq, 4, head_dim)
    q_rot_v = q_rot.view(1, seq, 4, head_dim)
    torch.testing.assert_close(q_rot_v[..., rotary_dim:], q_v[..., rotary_dim:], atol=0.0, rtol=0.0)
    assert not torch.allclose(q_rot_v[:, 1:, :, :rotary_dim], q_v[:, 1:, :, :rotary_dim])

    cos = rotary.rotary_cos_sin[:, 0, :]
    inv_freq = torch.tensor(
        [10000.0 ** (-i / (rotary_dim // 2)) for i in range(rotary_dim // 2)], device=device
    )
    for p in (1, 3, 5):
        expected_cos = torch.cos(p * inv_freq).float()
        torch.testing.assert_close(cos[p].float(), expected_cos, atol=1e-2, rtol=1e-2)


@skip_no_cuda
@skip_no_ckpt
@pytest.mark.parametrize("cfg", CFGS)
@torch.no_grad()
def test_chatglm3_source_activation_replay(cfg: RuntimeCfg):
    device = torch.device("cuda")
    backend = "TRTLLM"

    hf_model, _ = _HFRef.get()
    hf_config = _HFRef.config()
    num_layers = hf_config.num_layers

    model_config = ModelConfig(pretrained_config=deepcopy(hf_config), attn_backend=backend)
    normalize_chatglm_config(model_config.pretrained_config)
    model = ChatGLMForCausalLM(model_config).to(torch.float16).to(device).eval()
    model.load_weights(
        {
            k: v
            for k, v in hf_model.state_dict().items()
            if not k.endswith("rotary_pos_emb.inv_freq")
        }
    )

    attn0 = model.model.layers[0].self_attn
    assert attn0.num_key_value_heads == 2
    assert attn0.num_heads == 32
    assert attn0.head_dim == 128
    assert attn0.rope_fusion is False
    assert model.model.layers[0].mlp.gate_up_proj.weight.shape[0] == 2 * hf_config.ffn_hidden_size

    rep_layers = sorted({0, num_layers // 2, num_layers - 1})

    hf_layer_out, hf_attn_out, hf_mlp_out = {}, {}, {}

    def _hf_hook(store, idx):
        def hook(_m, _inp, out):
            store[idx] = (out[0] if isinstance(out, tuple) else out).detach().float().squeeze(1)

        return hook

    hf_handles = []
    for i in rep_layers:
        block = hf_model.transformer.encoder.layers[i]
        hf_handles.append(block.register_forward_hook(_hf_hook(hf_layer_out, i)))
        hf_handles.append(block.self_attention.register_forward_hook(_hf_hook(hf_attn_out, i)))
        hf_handles.append(block.mlp.register_forward_hook(_hf_hook(hf_mlp_out, i)))

    input_ids_hf = torch.tensor([PROMPT_IDS], dtype=torch.long, device=device)
    position_ids = torch.arange(len(PROMPT_IDS), device=device).unsqueeze(0)
    with torch.inference_mode():
        hf_out = hf_model.forward(
            input_ids=input_ids_hf, position_ids=position_ids, use_cache=True, return_dict=True
        )
    for h in hf_handles:
        h.remove()

    trt_layer_out, trt_attn_out, trt_mlp_out = {}, {}, {}

    def _trt_hook(store, idx, extract=lambda out: out):
        def hook(_m, _inp, out):
            store[idx] = extract(out).detach().float()

        return hook

    trt_handles = []
    for i in rep_layers:
        layer = model.model.layers[i]
        trt_handles.append(
            layer.register_forward_hook(_trt_hook(trt_layer_out, i, lambda o: o[0] + o[1]))
        )
        trt_handles.append(layer.self_attn.register_forward_hook(_trt_hook(trt_attn_out, i)))
        trt_handles.append(layer.mlp.register_forward_hook(_trt_hook(trt_mlp_out, i)))

    num_blocks, tokens_per_block = 1, 128
    kv_cache_manager = _make_kv_cache_manager_v2(
        model.config, num_blocks=num_blocks, tokens_per_block=tokens_per_block
    )
    metadata_cls = get_attention_backend(backend).Metadata
    input_ids = torch.tensor(PROMPT_IDS, dtype=torch.int, device=device)
    try:
        request_ids, token_nums = [1], [input_ids.size(-1)]
        kv_cache_manager.add_dummy_requests(request_ids, token_nums)
        attn_metadata = metadata_cls(
            seq_lens=torch.tensor([input_ids.size(-1)], dtype=torch.int),
            num_contexts=1,
            kv_cache_params=KVCacheParams(use_cache=True, num_cached_tokens_per_seq=[0]),
            max_num_requests=1,
            max_num_tokens=8192,
            kv_cache_manager=kv_cache_manager,
            request_ids=request_ids,
            prompt_lens=token_nums,
        )
        with torch.inference_mode():
            attn_metadata.prepare()
            trt_logits = model.forward(
                input_ids=input_ids, position_ids=position_ids, attn_metadata=attn_metadata
            )
        for h in trt_handles:
            h.remove()

        for i in rep_layers:
            for name, trt_t, hf_t in (
                ("attention", trt_attn_out[i], hf_attn_out[i]),
                ("mlp", trt_mlp_out[i], hf_mlp_out[i]),
                ("layer", trt_layer_out[i], hf_layer_out[i]),
            ):
                m = _metrics(trt_t, hf_t)
                print(
                    f"[source_activation_replay] cfg={cfg} boundary={name} layer={i} "
                    f"backend={backend} max_abs={m['max_abs']:.4f} "
                    f"mean_abs={m['mean_abs']:.5f} cosine={m['cosine']:.6f}"
                )
                hf_abs_max = hf_t.float().abs().max().item()
                max_abs_ok = (
                    m["max_abs"] <= MAX_ABS_TOL or m["max_abs"] <= REL_MAX_ABS_TOL * hf_abs_max
                )
                assert m["cosine"] >= COS_TOL, f"{name} L{i} cosine {m['cosine']}"
                assert m["mean_abs"] <= MEAN_ABS_TOL, f"{name} L{i} mean_abs {m['mean_abs']}"
                assert max_abs_ok, f"{name} L{i} max_abs {m['max_abs']} (hf_abs_max {hf_abs_max})"

        hf_last = hf_out.logits[:, -1].float().squeeze(0)
        trt_last = trt_logits.float().reshape(-1)
        lm = _metrics(trt_last, hf_last)
        print(
            f"[source_activation_replay] cfg={cfg} boundary=logits "
            f"max_abs={lm['max_abs']:.4f} mean_abs={lm['mean_abs']:.5f} cosine={lm['cosine']:.6f}"
        )
        assert lm["cosine"] >= COS_TOL, f"logits cosine {lm['cosine']}"
        assert torch.argmax(hf_last) == torch.argmax(trt_last), "prefill logits argmax mismatch"

        next_id = int(torch.argmax(trt_last).item())
        gen_id = torch.tensor([next_id], dtype=torch.int, device=device)
        gen_pos = torch.tensor([len(PROMPT_IDS)], dtype=torch.int, device=device).unsqueeze(0)

        def _decode_md():
            return metadata_cls(
                seq_lens=torch.tensor([1], dtype=torch.int),
                num_contexts=0,
                kv_cache_params=KVCacheParams(
                    use_cache=True, num_cached_tokens_per_seq=[len(PROMPT_IDS)]
                ),
                max_num_requests=1,
                max_num_tokens=8192,
                kv_cache_manager=kv_cache_manager,
                request_ids=request_ids,
                prompt_lens=token_nums,
            )

        with torch.inference_mode():
            eager_md = _decode_md()
            eager_md.prepare()
            eager_dec = model.forward(
                input_ids=gen_id, position_ids=gen_pos, attn_metadata=eager_md
            )

        with torch.inference_mode():
            hf_dec = hf_model.forward(
                input_ids=torch.tensor([[next_id]], device=device),
                position_ids=gen_pos,
                past_key_values=hf_out.past_key_values,
                use_cache=True,
                return_dict=True,
            )
        assert torch.argmax(hf_dec.logits[:, -1].float().reshape(-1)) == torch.argmax(
            eager_dec.float().reshape(-1)
        ), "decode argmax mismatch"

        if cfg.cuda_graph:
            from _torch.helpers import create_mock_cuda_graph_runner

            runner = create_mock_cuda_graph_runner(1)
            graph_md = _decode_md().create_cuda_graph_metadata(1)
            inputs = {"input_ids": gen_id, "position_ids": gen_pos, "attn_metadata": graph_md}
            key = (1, 0, False)
            with torch.inference_mode():
                graph_md.prepare()
                runner.capture(key, lambda inp, _m=model: _m.forward(**inp), inputs)
                assert runner.enabled and len(runner.graphs) >= 1, "no CUDA graph captured"
                assert all(isinstance(g, torch.cuda.CUDAGraph) for g in runner.graphs.values())
                for _ in range(2):
                    graph_md.prepare()
                    graph_dec = runner.replay(key, inputs)
            gm = _metrics(graph_dec, eager_dec)
            print(
                f"[cuda_graph_hard_path] source_activation_replay decode graph-vs-eager "
                f"max_abs={gm['max_abs']:.5f} cosine={gm['cosine']:.6f}"
            )
            torch.testing.assert_close(graph_dec.float(), eager_dec.float(), atol=1e-2, rtol=1e-2)
            runner.clear()
    finally:
        kv_cache_manager.shutdown()
        del model
        torch.cuda.empty_cache()


@skip_no_cuda
@skip_no_ckpt
@pytest.mark.parametrize("cfg", CFGS)
@torch.no_grad()
def test_chatglm3_real_runtime_kv_cache_prefill_decode(cfg: RuntimeCfg):
    device = torch.device("cuda")
    backend = "TRTLLM"

    model_config = ModelConfig.from_pretrained(CHATGLM3_CKPT, trust_remote_code=True)
    assert model_config.attn_backend == backend, model_config.attn_backend
    hf_config = model_config.pretrained_config
    with torch.device(device):
        model = ChatGLMForCausalLM(model_config).to(device).eval()
    model.load_weights(_load_checkpoint_weights(CHATGLM3_CKPT))

    attn0 = model.model.layers[0].self_attn
    assert attn0.num_heads == 32, "expected 32 query heads"
    assert attn0.num_key_value_heads == 2, "expected compact 2 KV heads (MQA)"
    assert attn0.head_dim == 128

    metadata_cls = get_attention_backend(backend).Metadata
    tokens_per_block = 128
    kv_cache_manager = _make_kv_cache_manager_v2(
        hf_config, num_blocks=8, tokens_per_block=tokens_per_block
    )
    assert kv_cache_manager.num_kv_heads == hf_config.num_key_value_heads == 2

    prompt = list(PROMPT_IDS)
    n = len(prompt)
    try:
        req_ids, token_nums = [1], [n]
        kv_cache_manager.add_dummy_requests(req_ids, token_nums)
        prefill_md = metadata_cls(
            seq_lens=torch.tensor([n], dtype=torch.int),
            num_contexts=1,
            kv_cache_params=KVCacheParams(use_cache=True, num_cached_tokens_per_seq=[0]),
            max_num_requests=1,
            max_num_tokens=8192,
            kv_cache_manager=kv_cache_manager,
            request_ids=req_ids,
            prompt_lens=token_nums,
        )
        ctx_ids = torch.tensor(prompt, dtype=torch.int, device=device)
        ctx_pos = torch.arange(n, device=device).unsqueeze(0)
        with torch.inference_mode():
            prefill_md.prepare()
            ctx_logits = model.forward(
                input_ids=ctx_ids, position_ids=ctx_pos, attn_metadata=prefill_md
            )
        assert torch.isfinite(ctx_logits).all()
        next_id = int(torch.argmax(ctx_logits.float().reshape(-1)).item())

        gen_id = torch.tensor([next_id], dtype=torch.int, device=device)
        gen_pos = torch.tensor([[n]], dtype=torch.int, device=device)

        def _decode_md():
            return metadata_cls(
                seq_lens=torch.tensor([1], dtype=torch.int),
                num_contexts=0,
                kv_cache_params=KVCacheParams(use_cache=True, num_cached_tokens_per_seq=[n]),
                max_num_requests=1,
                max_num_tokens=8192,
                kv_cache_manager=kv_cache_manager,
                request_ids=req_ids,
                prompt_lens=token_nums,
            )

        with torch.inference_mode():
            dec_md = _decode_md()
            dec_md.prepare()
            dec_logits = model.forward(input_ids=gen_id, position_ids=gen_pos, attn_metadata=dec_md)
        assert torch.isfinite(dec_logits).all()
    finally:
        kv_cache_manager.shutdown()

    kv_ref = _make_kv_cache_manager_v2(hf_config, num_blocks=8, tokens_per_block=tokens_per_block)
    try:
        ext = prompt + [next_id]
        m = len(ext)
        kv_ref.add_dummy_requests([2], [m])
        ref_md = metadata_cls(
            seq_lens=torch.tensor([m], dtype=torch.int),
            num_contexts=1,
            kv_cache_params=KVCacheParams(use_cache=True, num_cached_tokens_per_seq=[0]),
            max_num_requests=1,
            max_num_tokens=8192,
            kv_cache_manager=kv_ref,
            request_ids=[2],
            prompt_lens=[m],
        )
        ext_ids = torch.tensor(ext, dtype=torch.int, device=device)
        ext_pos = torch.arange(m, device=device).unsqueeze(0)
        with torch.inference_mode():
            ref_md.prepare()
            ref_logits = model.forward(
                input_ids=ext_ids, position_ids=ext_pos, attn_metadata=ref_md
            )
        ref_last = ref_logits.float().reshape(-1)[-hf_config.vocab_size :]
        cache_reuse = _metrics(dec_logits.float().reshape(-1), ref_last)
        print(
            f"[real_runtime_kv_cache] cfg={cfg} backend={backend} "
            f"num_kv_heads={kv_cache_manager.num_kv_heads} num_q_heads={attn0.num_heads} "
            f"cache_reuse_cosine={cache_reuse['cosine']:.6f} max_abs={cache_reuse['max_abs']:.4f}"
        )
        assert torch.argmax(dec_logits.float().reshape(-1)) == torch.argmax(ref_last), (
            "decode-with-cache next-token != fresh-prefill next-token (KV cache reuse broken)"
        )
        assert cache_reuse["cosine"] >= COS_TOL
    finally:
        kv_ref.shutdown()

    if cfg.cuda_graph:
        from _torch.helpers import create_mock_cuda_graph_runner

        kv_graph = _make_kv_cache_manager_v2(
            hf_config, num_blocks=8, tokens_per_block=tokens_per_block
        )
        try:
            kv_graph.add_dummy_requests([3], [n])
            gen_id = torch.tensor([next_id], dtype=torch.int, device=device)
            gen_pos = torch.tensor([[n]], dtype=torch.int, device=device)
            graph_md = metadata_cls(
                seq_lens=torch.tensor([1], dtype=torch.int),
                num_contexts=0,
                kv_cache_params=KVCacheParams(use_cache=True, num_cached_tokens_per_seq=[n]),
                max_num_requests=1,
                max_num_tokens=8192,
                kv_cache_manager=kv_graph,
                request_ids=[3],
                prompt_lens=[n],
            ).create_cuda_graph_metadata(1)
            runner = create_mock_cuda_graph_runner(1)
            inputs = {"input_ids": gen_id, "position_ids": gen_pos, "attn_metadata": graph_md}
            key = (1, 0, False)
            with torch.inference_mode():
                graph_md.prepare()
                runner.capture(key, lambda inp, _m=model: _m.forward(**inp), inputs)
                assert runner.enabled and len(runner.graphs) >= 1, "no CUDA graph captured"
                assert all(isinstance(g, torch.cuda.CUDAGraph) for g in runner.graphs.values())
                graph_md.prepare()
                graph_dec = runner.replay(key, inputs)
            print(
                f"[cuda_graph_hard_path] real_runtime_kv_cache decode captured_graphs="
                f"{len(runner.graphs)}"
            )
            assert torch.isfinite(graph_dec.float()).all()
            runner.clear()
        finally:
            kv_graph.shutdown()

    del model
    torch.cuda.empty_cache()


@skip_no_cuda
@skip_no_ckpt
@pytest.mark.parametrize("cfg", CFGS)
def test_chatglm3_source_logit_replay(cfg: RuntimeCfg):
    llm = _build_llm(cfg)
    try:
        _assert_v2_and_backend(llm, cfg)
        prompt_ids = [_HFRef.encode(p) for p in PROMPTS[:3]]
        outputs = llm.generate(
            [{"prompt_token_ids": ids} for ids in prompt_ids], _greedy_params(1, logits=True)
        )
        for ids, out in zip(prompt_ids, outputs):
            trt_logits = out.outputs[0].generation_logits
            assert trt_logits is not None, "generation_logits missing"
            trt_logits = torch.as_tensor(trt_logits).float().reshape(-1).cpu()
            hf_logits = _HFRef.next_logits(ids).cpu()
            m = _metrics(trt_logits, hf_logits)
            trt_tok, hf_tok = int(trt_logits.argmax()), int(hf_logits.argmax())
            print(
                f"[source_logit_replay] cfg={cfg} max_abs={m['max_abs']:.4f} "
                f"mean_abs={m['mean_abs']:.5f} cosine={m['cosine']:.6f} "
                f"trt_tok={trt_tok} hf_tok={hf_tok}"
            )
            assert trt_tok == hf_tok, f"argmax mismatch {trt_tok} != {hf_tok}"
            assert m["cosine"] >= COS_TOL
    finally:
        llm.shutdown()


_EOS_TOKEN_ID = 2


@skip_no_cuda
@skip_no_ckpt
@pytest.mark.parametrize("cfg", CFGS)
def test_chatglm3_generation_parity(cfg: RuntimeCfg):
    max_new = 32
    params = SamplingParams(
        max_tokens=max_new,
        min_tokens=max_new,
        temperature=0.0,
        top_k=1,
        ignore_eos=True,
        return_generation_logits=True,
    )
    llm = _build_llm(cfg)
    try:
        _assert_v2_and_backend(llm, cfg)
        prompt_ids = [_HFRef.encode(p) for p in PROMPTS]
        assert len(prompt_ids) >= 5, "generation_parity requires >=5 prompts"
        outputs = llm.generate([{"prompt_token_ids": ids} for ids in prompt_ids], params)
        for i, (ids, out) in enumerate(zip(prompt_ids, outputs)):
            trt_tokens = list(out.outputs[0].token_ids)[:max_new]
            trt_gl = torch.as_tensor(out.outputs[0].generation_logits).float()
            trt_gl = trt_gl.reshape(-1, trt_gl.shape[-1])[:max_new]
            assert trt_gl.shape[0] >= max_new and len(trt_tokens) >= max_new, (
                f"prompt {i}: {len(trt_tokens)} tokens / {trt_gl.shape[0]} logit rows < {max_new}"
            )
            hf_tokens, hf_logits = _HFRef.greedy_generate(
                ids, max_new, suppress_ids=(_EOS_TOKEN_ID,)
            )

            tok_mismatch, min_cos, worst_step = [], 1.0, -1
            for j in range(max_new):
                cos = torch.nn.functional.cosine_similarity(
                    trt_gl[j].cpu(), hf_logits[j].cpu(), dim=0
                ).item()
                if cos < min_cos:
                    min_cos, worst_step = cos, j
                if trt_tokens[j] != hf_tokens[j]:
                    tok_mismatch.append((j, hf_tokens[j], trt_tokens[j]))
            print(
                f"[generation_parity] cfg={cfg} prompt={i} steps={max_new} "
                f"token_mismatches={len(tok_mismatch)} min_step_cosine={min_cos:.5f} "
                f"@step={worst_step} first_mismatches={tok_mismatch[:5]}"
            )
            assert not tok_mismatch, (
                f"prompt {i} per-step greedy-argmax token mismatch {tok_mismatch[:5]}: "
                f"trt={trt_tokens} hf={hf_tokens}"
            )
            assert min_cos >= COS_TOL, (
                f"prompt {i} min per-step cosine {min_cos} @step {worst_step}"
            )
    finally:
        llm.shutdown()
