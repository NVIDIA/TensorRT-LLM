# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Accuracy tests for the llm.encode() path across encoder and decoder models.

These tests exercise the LLM(encode_only=True) / llm.encode() single-forward
prefill path and verify output correctness by direct logits comparison against
HuggingFace reference models.

Each decoder model is chosen as the *sole representative* of a distinct TRT-LLM
model architecture class (e.g. LlamaForCausalLM, Gemma3ForCausalLM).

Note: encode() is single-GPU only (no TP/PP). Every listed model is
architecturally required to fit on one GPU for these tests.
"""

import pytest
import torch

from tensorrt_llm import LLM
from tensorrt_llm.llmapi import EncodeCudaGraphConfig

from ..conftest import llm_models_root
from .accuracy_core import LlmapiAccuracyTestHarness

PROMPTS = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]

_TORCH_TO_LLM_DTYPE = {
    torch.bfloat16: "bfloat16",
    torch.float16: "float16",
    torch.float32: "float32",
}


def _resolve_checkpoint_dtype(model_path: str, trust_remote_code: bool = False):
    """Derive the checkpoint's native precision from its HF config."""
    from transformers import AutoConfig

    cfg = AutoConfig.from_pretrained(model_path, trust_remote_code=trust_remote_code)
    torch_dtype = getattr(cfg, "torch_dtype", None)
    if not isinstance(torch_dtype, torch.dtype):
        torch_dtype = torch.float32
    llm_dtype = _TORCH_TO_LLM_DTYPE.get(torch_dtype, "auto")
    return torch_dtype, llm_dtype


# --------------------------------------------------------------------------- #
# Encoder-only models (non-multimodal)
# --------------------------------------------------------------------------- #

CLASSIFICATION_MODELS = [
    pytest.param(
        "textattack/bert-base-uncased-yelp-polarity",
        f"{llm_models_root()}/bert/bert-base-uncased-yelp-polarity",
        id="bert-yelp",
    ),
]


# Qwen3-Embedding family. All variants are Qwen3ForCausalLM + a sentence-transformers
# last-token-pool + L2-normalize pipeline; one wrapper class (Qwen3ForTextEmbedding)
# serves all sizes. We cover both 0.6B (small/fast) and 8B (the large variant
# downstream users actually serve); 8B is memory-gated so it skips on small GPUs.
# The CI L0 list selects each variant by its `id=` below
# (tests/integration/test_lists/test-db/l0_l40s.yml) — the ids must stay in sync
# with that list or CI silently drops the test.
TEXT_EMBEDDING_MODELS = [
    pytest.param(
        "Qwen/Qwen3-Embedding-0.6B",
        f"{llm_models_root()}/Qwen3/Qwen3-Embedding-0.6B",
        id="qwen3-embedding-0.6b",
    ),
    pytest.param(
        "Qwen/Qwen3-Embedding-8B",
        f"{llm_models_root()}/Qwen3/Qwen3-Embedding-8B",
        marks=pytest.mark.skip_less_device_memory(32000),
        id="qwen3-embedding-8b",
    ),
]

# Encoder CUDA graph configs for parametrization. PROMPTS tokenize to short
# (~6-12 token) sequences, so we only need buckets that cover that range plus
# one small/larger pair to exercise dispatch + padding. Larger grids inflate
# warmup without adding coverage.
ENCODER_CUDA_GRAPH_CONFIGS = [
    pytest.param(None, id="eager"),
    pytest.param(
        dict(
            batch_sizes=[1, 4],
            num_tokens=[32, 64],
            seq_lens=[16, 32],
            enable_padding=True,
        ),
        id="cuda_graph",
    ),
]


class TestEncoderEncode(LlmapiAccuracyTestHarness):
    """HF logits-level accuracy for encoder-only (non-MM) architectures.

    Inherits LlmapiAccuracyTestHarness only for its class-scoped logger-level
    fixture. The harness' MODEL_NAME / MODEL_PATH attributes are not used
    here because the model differs per parametrize invocation.
    """

    @pytest.mark.parametrize("graph_kwargs", ENCODER_CUDA_GRAPH_CONFIGS)
    @pytest.mark.parametrize("model_name,model_path", CLASSIFICATION_MODELS)
    def test_encoder_encode_matches_huggingface_classification(
        self, model_name, model_path, graph_kwargs
    ):
        """Encoder classification heads: direct tensor compare on pooled logits.

        A classification head pools over the sequence (BERT: [CLS] token) and
        emits a single [num_classes] vector per prompt.

        Parametrized over the encoder CUDA graph path:
        - `eager`: cuda_graph_config=None, baseline.
        - `cuda_graph`: with a tight bucket grid; exercises capture/replay and
          padding logic.

        Under the `eager` ID we additionally verify the `return_raw_logits`
        flag: same forward, just a different output wrapping, so HF accuracy
        applies to the raw tensor as well.
        """
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        # Resolve the checkpoint's native precision.
        torch_dtype, llm_dtype = _resolve_checkpoint_dtype(model_path)
        cgc = None if graph_kwargs is None else EncodeCudaGraphConfig(**graph_kwargs)

        with LLM(model_path, encode_only=True, dtype=llm_dtype, cuda_graph_config=cgc) as llm:
            outs = llm.encode(PROMPTS)
            raw = llm.encode(PROMPTS, return_raw_logits=True) if cgc is None else None

        tokenizer = AutoTokenizer.from_pretrained(model_path)
        hf_model = (
            AutoModelForSequenceClassification.from_pretrained(model_path, torch_dtype=torch_dtype)
            .cuda()
            .eval()
        )
        with torch.inference_mode():
            inputs = tokenizer(PROMPTS, return_tensors="pt", padding="longest").to(hf_model.device)
            hf_logits = hf_model(**inputs).logits.float().cpu()
        tllm_logits = torch.stack([o.logits.cpu() for o in outs])

        torch.testing.assert_close(tllm_logits, hf_logits, rtol=1.5e-2, atol=1.5e-2)

        if raw is not None:
            assert isinstance(raw, torch.Tensor)
            assert raw.shape == hf_logits.shape
            torch.testing.assert_close(raw.cpu().float(), hf_logits, rtol=1.5e-2, atol=1.5e-2)

    @pytest.mark.parametrize("model_name,model_path", CLASSIFICATION_MODELS)
    def test_encoder_encode_cuda_graph_matches_eager_logits(self, model_name, model_path):
        """Tight numerical bound: graph replay must reproduce eager logits.

        This test compares graph output to eager output on the same model with a much
        tighter rtol=1e-3 to catch those.
        """
        _, llm_dtype = _resolve_checkpoint_dtype(model_path)

        with LLM(model_path, encode_only=True, dtype=llm_dtype) as llm_eager:
            eager_outs = llm_eager.encode(PROMPTS)
        cgc = EncodeCudaGraphConfig(
            batch_sizes=[1, 4],
            num_tokens=[64],
            seq_lens=[32],
            enable_padding=True,
        )
        with LLM(model_path, encode_only=True, dtype=llm_dtype, cuda_graph_config=cgc) as llm_graph:
            graph_outs = llm_graph.encode(PROMPTS)

        eager = torch.stack([o.logits.cpu() for o in eager_outs])
        graph = torch.stack([o.logits.cpu() for o in graph_outs])

        torch.testing.assert_close(graph, eager, rtol=1e-3, atol=1e-3)

    @pytest.mark.parametrize("model_name,model_path", TEXT_EMBEDDING_MODELS)
    def test_qwen3_text_embedding_matches_huggingface(self, model_name, model_path):
        """Decoder text-embedding: L2-normalized last-token hidden state vs HF."""
        import torch.nn.functional as F
        from transformers import AutoModel, AutoTokenizer

        torch_dtype, llm_dtype = _resolve_checkpoint_dtype(model_path)

        # Force the embedding wrapper class (the model's config declares
        # Qwen3ForCausalLM). encode() then returns the pooled+normalized vector.
        with LLM(
            model_path,
            encode_only=True,
            dtype=llm_dtype,
            model_kwargs={"architectures": ["Qwen3ForTextEmbedding"]},
        ) as llm:
            outs = llm.encode(PROMPTS)

        tokenizer = AutoTokenizer.from_pretrained(model_path)
        hf_model = AutoModel.from_pretrained(model_path, torch_dtype=torch_dtype).cuda().eval()

        for i, prompt in enumerate(PROMPTS):
            with torch.inference_mode():
                ids = tokenizer(prompt, return_tensors="pt").to(hf_model.device)
                last_hidden = hf_model(**ids).last_hidden_state  # [1, seq, hidden]
            # Last-token pool + L2 normalize (matches Qwen3ForTextEmbedding).
            hf_emb = F.normalize(last_hidden[0, -1].float(), p=2, dim=-1).cpu()

            tllm_emb = outs[i].logits.cpu().float()
            assert tllm_emb.shape == hf_emb.shape, (
                f"[{model_name}] prompt#{i} shape {tuple(tllm_emb.shape)} "
                f"!= HF {tuple(hf_emb.shape)}"
            )
            torch.testing.assert_close(tllm_emb, hf_emb, rtol=1.5e-2, atol=1.5e-2)
            # Embeddings must be unit-norm.
            assert abs(tllm_emb.norm().item() - 1.0) < 1e-2


# --------------------------------------------------------------------------- #
# Decoder models used in single-prefill mode
# --------------------------------------------------------------------------- #
#
# encode() on a decoder model runs a single prefill and returns logits without
# running the autoregressive loop. Use case: embedding extraction, reward /
# classification scoring on a causal LM backbone.
#
# One representative per distinct TRT-LLM architecture class:
#   LlamaForCausalLM   — TinyLlama (also covers Mistral, which aliases LlamaModel)
#   Qwen3ForCausalLM   — Qwen3-0.6B (QKNorm)
DECODER_MODELS = [
    # -- LlamaForCausalLM (covers Llama + Mistral family) --
    pytest.param(
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        f"{llm_models_root()}/llama-models-v2/TinyLlama-1.1B-Chat-v1.0",
        id="tinyllama-1.1b",
    ),
    # -- Qwen3ForCausalLM --
    pytest.param(
        "Qwen/Qwen3-0.6B",
        f"{llm_models_root()}/Qwen3/Qwen3-0.6B",
        id="qwen3-0.6b",
    ),
]


class TestDecoderEncode(LlmapiAccuracyTestHarness):
    """Validates encode() on decoder models used in single-prefill mode."""

    PROMPTS = [
        "The quick brown fox",
        "Hello, world! How are you today?",
        "In a distant galaxy, an advanced civilization discovered that light can be",
    ]

    # Top-K size used for the argmax-in-top-K containment / overlap checks.
    # This is robust to near-tie argmax flips under FP16/BF16 rounding.
    TOPK = 5
    TOPK_MIN_OVERLAP = 3

    @pytest.mark.threadleak(enabled=False)
    @pytest.mark.parametrize("model_name,model_path", DECODER_MODELS)
    def test_decoder_encode_matches_huggingface(self, model_name, model_path):
        """encode() last-token logits match HF causal-LM prefill.

        Two checks are performed:

        1. **Top-K semantic check** — top-1 on each side must appear in the
           other side's top-K, and the top-K sets must overlap by at least
           ``TOPK_MIN_OVERLAP``.

        2. **Focused numerical check** — ``torch.testing.assert_close`` is
           restricted to the union of both sides' top-K indices.
        """
        from transformers import AutoModelForCausalLM, AutoTokenizer

        # Resolve the checkpoint's native precision.
        torch_dtype, llm_dtype = _resolve_checkpoint_dtype(model_path)

        with LLM(model_path, encode_only=True, dtype=llm_dtype) as llm:
            outs = llm.encode(self.PROMPTS)

        tokenizer = AutoTokenizer.from_pretrained(model_path)
        hf_model = (
            AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch_dtype).cuda().eval()
        )

        # encode() routes decoder causal LMs through LogitsProcessor with
        # gather_context_logits=False, which returns last-token logits only
        # (shape [vocab_size] per prompt). Per-token logits would require a
        # separate encode() flag — tracked as follow-up work. See
        # tensorrt_llm/_torch/modules/logits_processor.py.
        for i, prompt in enumerate(self.PROMPTS):
            with torch.inference_mode():
                inputs = tokenizer(prompt, return_tensors="pt").to(hf_model.device)
                hf_last = hf_model(**inputs).logits[0, -1].float().cpu()

            tllm_last = outs[i].logits.cpu().float()

            tllm_topk = tllm_last.topk(self.TOPK).indices
            hf_topk = hf_last.topk(self.TOPK).indices
            tllm_top1 = tllm_topk[0].item()
            hf_top1 = hf_topk[0].item()
            tllm_topk_set = set(tllm_topk.tolist())
            hf_topk_set = set(hf_topk.tolist())
            overlap = len(tllm_topk_set & hf_topk_set)

            # (1) Semantic check — top-1 must be in the other side's top-K,
            # and the top-K sets must substantially overlap.
            assert tllm_top1 in hf_topk_set and hf_top1 in tllm_topk_set, (
                f"[{model_name}] prompt#{i} ({prompt!r}) top-1 not in the "
                f"other side's top-{self.TOPK}: "
                f"TLLM top-1={tllm_top1}, HF top-1={hf_top1}, "
                f"TLLM top-{self.TOPK}={sorted(tllm_topk_set)}, "
                f"HF top-{self.TOPK}={sorted(hf_topk_set)}"
            )
            assert overlap >= self.TOPK_MIN_OVERLAP, (
                f"[{model_name}] prompt#{i} ({prompt!r}) top-{self.TOPK} "
                f"overlap {overlap} < {self.TOPK_MIN_OVERLAP}: "
                f"TLLM={sorted(tllm_topk_set)}, HF={sorted(hf_topk_set)}"
            )

            # (2) Focused numerical check — compare logits only at the
            # union of both sides' top-K indices.
            important_idx = torch.unique(torch.cat([tllm_topk, hf_topk]))
            torch.testing.assert_close(
                tllm_last[important_idx],
                hf_last[important_idx],
                atol=0.4,
                rtol=0.4,
                msg=lambda m: (
                    f"[{model_name}] prompt#{i} ({prompt!r}) top-K logits "
                    f"differ beyond tolerance.\nTLLM={tllm_last[important_idx].tolist()}\n"
                    f"HF={hf_last[important_idx].tolist()}\n{m}"
                ),
            )

    @pytest.mark.threadleak(enabled=False)
    @pytest.mark.parametrize("model_name,model_path", DECODER_MODELS)
    def test_decoder_encode_cuda_graph_matches_eager_logits(self, model_name, model_path):
        """Tight numerical bound: decoder single-prefill graph replay must reproduce eager logits."""
        _, llm_dtype = _resolve_checkpoint_dtype(model_path)

        with LLM(model_path, encode_only=True, dtype=llm_dtype) as llm_eager:
            eager_outs = llm_eager.encode(self.PROMPTS)

        cgc = EncodeCudaGraphConfig(
            batch_sizes=[1, 4],
            num_tokens=[32, 64],
            seq_lens=[16, 32],
            enable_padding=True,
        )
        with LLM(model_path, encode_only=True, dtype=llm_dtype, cuda_graph_config=cgc) as llm_graph:
            graph_outs = llm_graph.encode(self.PROMPTS)

        eager = torch.stack([o.logits.cpu() for o in eager_outs])
        graph = torch.stack([o.logits.cpu() for o in graph_outs])
        torch.testing.assert_close(graph, eager, rtol=5e-2, atol=0.5)
