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
"""ChatGLM3-6B attention backend tests."""

import os

import pytest
import torch

from tensorrt_llm._torch.attention_backend.trtllm import TrtllmAttention
from tensorrt_llm._torch.attention_backend.utils import get_attention_backend
from tensorrt_llm._torch.metadata import KVCacheParams
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.models.modeling_chatglm import ChatGLMForCausalLM
from tensorrt_llm._torch.modules.multi_stream_utils import with_multi_stream
from tensorrt_llm._torch.pyexecutor.config_utils import load_pretrained_config
from tensorrt_llm._torch.pyexecutor.kv_cache_manager_v2 import KVCacheManagerV2
from tensorrt_llm._torch.utils import piecewise_cuda_graph
from tensorrt_llm.mapping import Mapping

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="ChatGLM3 attention tests require CUDA"
)

PROMPT_IDS = [64790, 64792, 790, 30951, 517, 30910, 30939, 30996, 13, 30910]
REP_LAYERS = [0, 14, 27]


def _model_dir() -> str:
    d = os.environ.get("CHATGLM3_6B_MODEL_DIR")
    if d:
        return d
    from utils.llm_data import llm_models_root

    return str(llm_models_root() / "chatglm3-6b")


def _load_hf_chatglm(model_dir, dtype, device):
    from transformers import AutoConfig, AutoModelForCausalLM
    from transformers.dynamic_module_utils import get_class_from_dynamic_module

    cls = get_class_from_dynamic_module(
        "modeling_chatglm.ChatGLMForConditionalGeneration", model_dir
    )
    if not hasattr(cls, "all_tied_weights_keys"):
        cls.all_tied_weights_keys = {}
    config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
    if getattr(config, "max_length", None) is None:
        config.max_length = config.seq_length
    return (
        AutoModelForCausalLM.from_pretrained(
            model_dir, trust_remote_code=True, torch_dtype=dtype, config=config
        )
        .to(device)
        .eval()
    )


def _build_model(model_dir, dtype, device, backend="TRTLLM"):
    hf = _load_hf_chatglm(model_dir, dtype, device)
    pretrained_config = load_pretrained_config(model_dir, trust_remote_code=True)
    model_config = ModelConfig(pretrained_config=pretrained_config, attn_backend=backend)
    model = ChatGLMForCausalLM(model_config).to(dtype).to(device)
    model.load_weights(hf.state_dict())
    return hf, model, model_config.pretrained_config


def _build_kv_cache_manager(config, num_blocks=4, tokens_per_block=64, batch_size=1):
    import tensorrt_llm
    from tensorrt_llm.llmapi.llm_args import KvCacheConfig

    kv_dtype = (
        tensorrt_llm.bindings.DataType.HALF
        if config.torch_dtype == torch.half
        else tensorrt_llm.bindings.DataType.BF16
    )
    max_seq_len = num_blocks * tokens_per_block
    kv_cache_config = KvCacheConfig(max_tokens=max_seq_len, enable_block_reuse=False)
    return KVCacheManagerV2(
        kv_cache_config,
        tensorrt_llm.bindings.internal.batch_manager.CacheType.SELF,
        num_layers=config.num_hidden_layers,
        num_kv_heads=config.num_key_value_heads,
        head_dim=config.head_dim,
        tokens_per_block=tokens_per_block,
        max_seq_len=max_seq_len,
        max_batch_size=batch_size,
        mapping=Mapping(world_size=1, tp_size=1, rank=0),
        dtype=kv_dtype,
    )


def _context_metadata(backend, kv_cache_manager, seq_len, device):
    metadata_cls = get_attention_backend(backend).Metadata
    request_ids = [1]
    kv_cache_manager.add_dummy_requests(request_ids, [seq_len])
    attn_metadata = metadata_cls(
        seq_lens=torch.tensor([seq_len], dtype=torch.int),
        num_contexts=1,
        kv_cache_params=KVCacheParams(use_cache=True, num_cached_tokens_per_seq=[0]),
        max_num_requests=1,
        max_num_tokens=8192,
        kv_cache_manager=kv_cache_manager,
        request_ids=request_ids,
        prompt_lens=[seq_len],
    )
    position_ids = torch.arange(0, seq_len, dtype=torch.int32, device=device).unsqueeze(0)
    return attn_metadata, position_ids, request_ids


def _decode_metadata(backend, kv_cache_manager, request_ids, cached_len, device):
    metadata_cls = get_attention_backend(backend).Metadata
    attn_metadata = metadata_cls(
        seq_lens=torch.tensor([1], dtype=torch.int),
        num_contexts=0,
        kv_cache_params=KVCacheParams(use_cache=True, num_cached_tokens_per_seq=[cached_len]),
        max_num_requests=1,
        max_num_tokens=8192,
        kv_cache_manager=kv_cache_manager,
        request_ids=request_ids,
        prompt_lens=[cached_len],
    )
    position_ids = torch.tensor([[cached_len]], dtype=torch.int32, device=device)
    return attn_metadata, position_ids


def _cos(a, b):
    return torch.nn.functional.cosine_similarity(
        a.float().flatten(), b.float().flatten(), dim=0
    ).item()


def _replay_decode_logits_under_cuda_graph(model, gen_ids, dec_pos, dec_metadata):
    """Capture full-model decode in a CUDA graph and return replayed logits."""
    cg_md = dec_metadata.create_cuda_graph_metadata(1)
    cg_md.seq_lens = torch.tensor([1], dtype=torch.int)
    cg_md.num_contexts = 0
    cg_md.prepare()

    static_ids = torch.zeros_like(gen_ids)
    static_pos = torch.zeros_like(dec_pos)
    static_ids.copy_(gen_ids)
    static_pos.copy_(dec_pos)

    def _fwd():
        return model.forward(input_ids=static_ids, position_ids=static_pos, attn_metadata=cg_md)

    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph()
    with with_multi_stream(True), piecewise_cuda_graph(False):
        with torch.cuda.stream(side):
            for _ in range(2):
                _fwd()
        torch.cuda.current_stream().wait_stream(side)
        with torch.cuda.graph(graph):
            out = _fwd()
    graph.replay()
    torch.cuda.synchronize()
    return out.clone()


@torch.no_grad()
def test_source_activation_replay():
    model_dir = _model_dir()
    dtype = torch.float16
    device = torch.device("cuda")
    backend = "TRTLLM"

    hf, model, config = _build_model(model_dir, dtype, device, backend)

    hf_act, trt_act = {}, {}

    # Clone because the TRTLLM attention output buffer is reused across layers.
    def hf_hook(idx):
        def _h(_m, inp, out):
            hf_act[idx] = (inp[0].detach().clone(), out[0].detach().clone())

        return _h

    def trt_hook(idx):
        def _h(_m, args, kwargs, out):
            trt_act[idx] = (kwargs["hidden_states"].detach().clone(), out.detach().clone())

        return _h

    handles = []
    for i in REP_LAYERS:
        handles.append(
            hf.transformer.encoder.layers[i].self_attention.register_forward_hook(hf_hook(i))
        )
        handles.append(
            model.model.layers[i].self_attn.register_forward_hook(trt_hook(i), with_kwargs=True)
        )

    kv_cache_manager = _build_kv_cache_manager(config)
    try:
        input_ids = torch.tensor(PROMPT_IDS, dtype=torch.int32, device=device)
        seq_len = input_ids.size(-1)
        attn_metadata, position_ids, request_ids = _context_metadata(
            backend, kv_cache_manager, seq_len, device
        )

        assert isinstance(model.model.layers[0].self_attn.attn, TrtllmAttention)
        assert isinstance(kv_cache_manager, KVCacheManagerV2)

        with torch.inference_mode():
            attn_metadata.prepare()
            model.forward(
                input_ids=input_ids, position_ids=position_ids, attn_metadata=attn_metadata
            )
            hf_out = hf.forward(
                input_ids=input_ids.unsqueeze(0), position_ids=position_ids, use_cache=True
            )

        for i in REP_LAYERS:
            hf_in, hf_o = hf_act[i]
            trt_in, trt_o = trt_act[i]
            hf_in = hf_in.reshape(seq_len, -1)
            hf_o = hf_o.reshape(seq_len, -1)
            in_cos = _cos(trt_in, hf_in)
            out_cos = _cos(trt_o, hf_o)
            max_abs = (trt_o.float() - hf_o.float()).abs().max().item()
            mean_abs = (trt_o.float() - hf_o.float()).abs().mean().item()
            print(
                f"[source_activation_replay] layer={i} prompt_len={seq_len} "
                f"in_cos={in_cos:.5f} out_cos={out_cos:.5f} "
                f"max_abs={max_abs:.4f} mean_abs={mean_abs:.4f}"
            )
            assert in_cos > 0.99, f"layer {i} attention input diverged"
            assert out_cos > 0.99, f"layer {i} attention output diverged"

        trt_act.clear()
        hf_act.clear()
        gen_ids = torch.tensor([PROMPT_IDS[-1]], dtype=torch.int32, device=device)
        dec_metadata, dec_pos = _decode_metadata(
            backend, kv_cache_manager, request_ids, seq_len, device
        )
        with torch.inference_mode():
            dec_metadata.prepare()
            model.forward(input_ids=gen_ids, position_ids=dec_pos, attn_metadata=dec_metadata)
            hf.forward(
                input_ids=gen_ids.unsqueeze(0),
                position_ids=dec_pos,
                past_key_values=hf_out.past_key_values,
                use_cache=True,
            )
        dec_cos = _cos(trt_act[0][1], hf_act[0][1].reshape(1, -1))
        print(f"[source_activation_replay] decode layer=0 out_cos={dec_cos:.5f}")
        assert dec_cos > 0.99, "decode attention output diverged"
    finally:
        for h in handles:
            h.remove()
        kv_cache_manager.shutdown()

    _assert_cuda_graph_matches_hf(model, hf, config, device, backend)


def _assert_cuda_graph_matches_hf(model, hf, config, device, backend):
    kv_cache_manager = _build_kv_cache_manager(config)
    try:
        input_ids = torch.tensor(PROMPT_IDS, dtype=torch.int32, device=device)
        seq_len = input_ids.size(-1)
        attn_metadata, position_ids, request_ids = _context_metadata(
            backend, kv_cache_manager, seq_len, device
        )
        attn_metadata.prepare()
        model.forward(input_ids=input_ids, position_ids=position_ids, attn_metadata=attn_metadata)
        hf_out = hf.forward(
            input_ids=input_ids.unsqueeze(0), position_ids=position_ids, use_cache=True
        )

        gen_ids = torch.tensor([PROMPT_IDS[-1]], dtype=torch.int32, device=device)
        dec_metadata, dec_pos = _decode_metadata(
            backend, kv_cache_manager, request_ids, seq_len, device
        )
        trt_logits = _replay_decode_logits_under_cuda_graph(model, gen_ids, dec_pos, dec_metadata)
        hf_dec = hf.forward(
            input_ids=gen_ids.unsqueeze(0),
            position_ids=dec_pos,
            past_key_values=hf_out.past_key_values,
            use_cache=True,
        )
        assert trt_logits.argmax(-1).item() == hf_dec.logits[:, -1].argmax(-1).item(), (
            "cuda-graph decode token mismatch"
        )
        print("[source_activation_replay] cuda_graph=true hard path token OK")
    finally:
        kv_cache_manager.shutdown()


@pytest.mark.parametrize(
    "enable_cuda_graph", [False, True], ids=["cuda_graph=false", "cuda_graph=true"]
)
@torch.no_grad()
def test_real_runtime_cuda_graph_hard_path(enable_cuda_graph):
    model_dir = _model_dir()
    dtype = torch.float16
    device = torch.device("cuda")
    backend = "TRTLLM"

    hf, model, config = _build_model(model_dir, dtype, device, backend)
    kv_cache_manager = _build_kv_cache_manager(config)
    try:
        assert isinstance(model.model.layers[0].self_attn.attn, TrtllmAttention)
        assert isinstance(kv_cache_manager, KVCacheManagerV2)

        input_ids = torch.tensor(PROMPT_IDS, dtype=torch.int32, device=device)
        seq_len = input_ids.size(-1)
        attn_metadata, position_ids, request_ids = _context_metadata(
            backend, kv_cache_manager, seq_len, device
        )
        attn_metadata.prepare()
        model.forward(input_ids=input_ids, position_ids=position_ids, attn_metadata=attn_metadata)
        hf_out = hf.forward(
            input_ids=input_ids.unsqueeze(0), position_ids=position_ids, use_cache=True
        )

        gen_ids = torch.tensor([PROMPT_IDS[-1]], dtype=torch.int32, device=device)
        dec_metadata, dec_pos = _decode_metadata(
            backend, kv_cache_manager, request_ids, seq_len, device
        )

        if enable_cuda_graph:
            trt_logits = _replay_decode_logits_under_cuda_graph(
                model, gen_ids, dec_pos, dec_metadata
            )
        else:
            dec_metadata.prepare()
            trt_logits = model.forward(
                input_ids=gen_ids, position_ids=dec_pos, attn_metadata=dec_metadata
            )

        hf_dec = hf.forward(
            input_ids=gen_ids.unsqueeze(0),
            position_ids=dec_pos,
            past_key_values=hf_out.past_key_values,
            use_cache=True,
        )
        assert trt_logits.argmax(-1).item() == hf_dec.logits[:, -1].argmax(-1).item()
    finally:
        kv_cache_manager.shutdown()
