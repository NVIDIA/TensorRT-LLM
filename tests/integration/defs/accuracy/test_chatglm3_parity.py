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
"""ChatGLM3-6B greedy parity against the HF source."""

import gc
import os

import pytest
import torch

from tensorrt_llm import LLM, SamplingParams
from tensorrt_llm.llmapi import CudaGraphConfig, KvCacheConfig

EOS_ID = 2

SHORT_PROMPTS = [
    "1 + 1 =",
    "The capital of France is",
    "Question: What is 12 times 12? Answer:",
]

LONG_PROMPTS = [
    "1, 2, 3, 4, 5, 6, 7, 8, 9, 10,",
    "2, 4, 6, 8, 10, 12, 14,",
    "5, 10, 15, 20, 25, 30,",
    "100, 99, 98, 97, 96, 95,",
    "3, 6, 9, 12, 15, 18,",
]

CONFIGS = {
    "baseline": dict(cuda_graph=False, overlap=False),
    "enabled": dict(cuda_graph=True, overlap=True),
}


def _model_dir() -> str:
    d = os.environ.get("CHATGLM3_6B_MODEL_DIR")
    if d:
        return d
    from ..conftest import llm_models_root

    return f"{llm_models_root()}/chatglm3-6b"


def _tokenizer(model_dir):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)


def _load_hf(model_dir, dtype, device):
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


@torch.no_grad()
def _hf_final_logits(hf, input_ids, device):
    pos = torch.arange(input_ids.shape[1], device=device).unsqueeze(0)
    out = hf.forward(input_ids=input_ids, position_ids=pos, use_cache=True)
    return out.logits[:, -1].float().squeeze(0).cpu()  # [vocab]


@torch.no_grad()
def _hf_greedy_tokens(hf, input_ids, max_new_tokens, device):
    cur = input_ids
    past = None
    pos = torch.arange(input_ids.shape[1], device=device).unsqueeze(0)
    tokens = []
    for _ in range(max_new_tokens):
        out = hf.forward(input_ids=cur, position_ids=pos, past_key_values=past, use_cache=True)
        past = out.past_key_values
        nxt = int(out.logits[:, -1].argmax(-1).item())
        if nxt == EOS_ID:
            break
        tokens.append(nxt)
        cur = torch.tensor([[nxt]], dtype=input_ids.dtype, device=device)
        pos = pos[:, -1:] + 1
    return tokens


def _build_llm(model_dir, cfg, *, gather_logits=False):
    kwargs = dict(
        attn_backend="TRTLLM",
        kv_cache_config=KvCacheConfig(
            use_kv_cache_manager_v2=True, free_gpu_memory_fraction=0.5, enable_block_reuse=False
        ),
        disable_overlap_scheduler=not cfg["overlap"],
        max_batch_size=8,
        cuda_graph_config=CudaGraphConfig() if cfg["cuda_graph"] else None,
    )
    if gather_logits:
        kwargs["gather_generation_logits"] = True
    return LLM(model=model_dir, trust_remote_code=True, **kwargs)


@pytest.mark.parametrize("config_name", list(CONFIGS))
def test_source_logit_replay(config_name):
    model_dir = _model_dir()
    cfg = CONFIGS[config_name]
    prompts = SHORT_PROMPTS
    device = torch.device("cuda")

    tok = _tokenizer(model_dir)
    hf = _load_hf(model_dir, torch.float16, device)
    prompt_ids, hf_logits, hf_argmax = [], [], []
    for p in prompts:
        ids = tok([p], return_tensors="pt").input_ids.to(device)
        fl = _hf_final_logits(hf, ids, device)
        prompt_ids.append(ids[0].tolist())
        hf_logits.append(fl)
        hf_argmax.append(int(fl.argmax(-1).item()))
    del hf
    gc.collect()
    torch.cuda.empty_cache()

    llm = _build_llm(model_dir, cfg, gather_logits=True)
    sampling = SamplingParams(max_tokens=4, temperature=0.0, return_generation_logits=True)
    outs = llm.generate(prompt_ids, sampling)
    try:
        for i, out in enumerate(outs):
            trt_tok = int(out.outputs[0].token_ids[0])
            trt_logits = out.outputs[0].generation_logits[0].float().cpu()  # [vocab]
            cos = torch.nn.functional.cosine_similarity(hf_logits[i], trt_logits, dim=0).item()
            max_abs = (hf_logits[i] - trt_logits).abs().max().item()
            print(
                f"[source_logit_replay:{config_name}] prompt={prompts[i]!r} "
                f"hf_argmax={hf_argmax[i]} trt_tok={trt_tok} "
                f"final_logit_cos={cos:.5f} max_abs={max_abs:.3f}"
            )
            assert trt_tok == hf_argmax[i], (
                f"final-logit argmax mismatch: hf={hf_argmax[i]} trt={trt_tok}"
            )
            assert cos > 0.99, f"final-logit cosine {cos:.5f} below 0.99"
    finally:
        llm.shutdown()


@pytest.mark.parametrize("config_name", list(CONFIGS))
def test_generation_parity(config_name):
    model_dir = _model_dir()
    cfg = CONFIGS[config_name]
    prompts = LONG_PROMPTS
    assert len(prompts) >= 5
    max_new = 36
    device = torch.device("cuda")

    tok = _tokenizer(model_dir)
    hf = _load_hf(model_dir, torch.float16, device)
    prompt_ids, refs = [], []
    for p in prompts:
        ids = tok([p], return_tensors="pt").input_ids.to(device)
        prompt_ids.append(ids[0].tolist())
        refs.append(_hf_greedy_tokens(hf, ids, max_new, device))
    del hf
    gc.collect()
    torch.cuda.empty_cache()

    llm = _build_llm(model_dir, cfg)
    sampling = SamplingParams(max_tokens=max_new, temperature=0.0)
    outs = llm.generate(prompt_ids, sampling)
    try:
        for i, out in enumerate(outs):
            trt_tokens = [t for t in out.outputs[0].token_ids if t != EOS_ID]
            hf_tokens = refs[i]
            k = min(len(trt_tokens), len(hf_tokens))
            div = next((j for j in range(k) if trt_tokens[j] != hf_tokens[j]), None)
            print(
                f"[generation_parity:{config_name}] prompt={prompts[i]!r} "
                f"hf_len={len(hf_tokens)} trt_len={len(trt_tokens)} first_divergence={div}"
            )
            assert k >= 32, f"prompt {prompts[i]!r} produced only {k} tokens (<32) before EOS"
            assert trt_tokens[:k] == hf_tokens[:k], (
                f"generation diverged at step {div} for prompt {prompts[i]!r}"
            )
    finally:
        llm.shutdown()
