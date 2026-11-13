# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Steps of a Kimi K3 target's MLA layer (``mla.py``) on one GPU, for checks of the layer against itself.

``MlaLayer`` builds a target's ``KimiMLARuntime`` at one TP16 rank's shape (6 of the 96 heads) with a single-rank
``Mapping()`` and random weights, over one bf16 latent cache (page 64) of 16 requests. ``run`` runs a step eagerly on
the generic path or on the decode branch; ``run_breakable_graph`` and ``run_cuda_graph`` run it captured and replayed.
``snapshot`` / ``restore`` put the cache back, so two ways of running a step start from the same contents.
"""

from __future__ import annotations

import weakref
from contextlib import contextmanager
from dataclasses import dataclass
from typing import List

import torch

import tensorrt_llm.bindings
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.ops.attention.k3_mla_attn_workspace import (
    K3MlaAttnWorkspace,
)
from tensorrt_llm._torch.attention.backends.interface import AttentionRuntimeFeatures, KVCacheParams
from tensorrt_llm._torch.attention.backends.utils import get_attention_backend
from tensorrt_llm._torch.configs.kimi_linear import KimiLinearConfig
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import KVCacheManagerV2
from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequestState
from tensorrt_llm._torch.utils import AuxStreamType, model_extra_attrs
from tensorrt_llm.llmapi.llm_args import KvCacheConfig
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models.modeling_utils import QuantConfig

HEADS = 6  # one TP16 rank's share of K3's 96 heads
HIDDEN = 7168
TOKENS_PER_BLOCK = 64
LATENT = 512 + 64
NUM_REQUESTS = 16
MAX_SEQ_LEN = 512


def k3_config(num_heads: int = HEADS) -> KimiLinearConfig:
    """K3's MLA geometry at ``num_heads`` heads on one rank."""
    config = KimiLinearConfig(
        vocab_size=128,
        hidden_size=HIDDEN,
        intermediate_size=256,
        num_hidden_layers=1,
        attn_res_block_size=1,
        torch_dtype=torch.bfloat16,
        linear_attn_config={"kda_layers": [], "full_attn_layers": [1], "num_heads": num_heads},
        num_attention_heads=num_heads,
        num_key_value_heads=num_heads,
        q_lora_rank=1536,
        kv_lora_rank=512,
        qk_nope_head_dim=128,
        qk_rope_head_dim=64,
        v_head_dim=128,
        rms_norm_eps=1e-6,
        mla_use_nope=True,
        mla_use_output_gate=True,
        max_position_embeddings=4096,
    )
    config._attn_implementation = "eager"
    return config


@dataclass
class Step:
    """One step: the requests, their new tokens, cached tokens and prompt lengths, contexts first. A context
    request's prompt length is its new tokens, as the engine sets it for a chunk; a generation request's is its whole
    prompt."""

    name: str
    request_ids: List[int]
    seq_lens: List[int]
    cached: List[int]
    num_contexts: int
    prompt_lens: List[int]
    chunked_prefill: bool = True

    @property
    def num_tokens(self) -> int:
        return sum(self.seq_lens)


#: The prompts the first step writes (request i holds PROMPTS[i] tokens after it).
PROMPTS = [100, 37, 64, 1, 65, 128, 7, 90, 33, 64, 2, 50, 20, 20, 20, 20]


def fresh_prefill() -> Step:
    """Every request's prompt, as context requests without a cached prefix."""
    return Step(
        "fresh prefill",
        list(range(NUM_REQUESTS)),
        PROMPTS,
        [0] * NUM_REQUESTS,
        NUM_REQUESTS,
        PROMPTS,
    )


def prefix_step(chunked_prefill: bool) -> Step:
    """Three context requests continuing over a cached prefix (100, 37 and 128 tokens): with chunked prefill the
    prefix is read in 5 chunks of up to 64 tokens; without, in one piece."""
    return Step(
        "chunked prefill" if chunked_prefill else "cached prefix",
        [0, 1, 5],
        [50, 30, 9],
        [100, 37, 128],
        3,
        [50, 30, 9],
        chunked_prefill=chunked_prefill,
    )


def mixed(lens: List[int]) -> Step:
    """Request 12 as a fresh 20-token context request (its rows rewritten), then requests 0..9 generating one token."""
    gens = list(range(10))
    return Step("mixed", [12] + gens, [20] + [1] * 10, [0] + [lens[i] for i in gens], 1,
                [20] + [PROMPTS[i] for i in gens])  # fmt: skip


def generation(ids: List[int], tokens: int, lens: List[int]) -> Step:
    """Pure generation of ``tokens`` per request for ``ids`` whose caches hold ``lens[i]`` tokens."""
    return Step(f"generation {len(ids)}x{tokens}", ids, [tokens] * len(ids), [lens[i] for i in ids], 0,
                [PROMPTS[i] for i in ids])  # fmt: skip


class MlaLayer:
    """A target's MLA layer, its weights and its KV cache (see the module docstring)."""

    def __init__(self, mla_module, seed: int = 0) -> None:
        self.mla = mla_module
        torch.manual_seed(seed)
        self.mc = ModelConfig(
            pretrained_config=k3_config(),
            mapping=Mapping(),
            quant_config=QuantConfig(),
            attn_backend="TRTLLM",
        )
        aux = {AuxStreamType.Attention: torch.cuda.Stream()}
        with torch.device("cuda"):
            self.runtime = mla_module.KimiMLARuntime(self.mc.pretrained_config, 0, self.mc, aux)
        gen = torch.Generator(device="cuda").manual_seed(seed)
        with torch.no_grad():
            for _, p in self.runtime.mixer.named_parameters():
                w = torch.randn(p.shape, generator=gen, device="cuda", dtype=torch.float32)
                p.copy_((1.0 + 0.1 * w) if p.dim() == 1 else w / p.shape[1] ** 0.5)
        for module in self.runtime.modules():
            if callable(getattr(module, "post_load_weights", None)):
                module.post_load_weights()
        self.runtime.mixer.k3_workspace = K3MlaAttnWorkspace.create(
            torch.device("cuda"), HEADS // 6
        )
        self.manager = KVCacheManagerV2(
            KvCacheConfig(
                max_tokens=NUM_REQUESTS * MAX_SEQ_LEN, enable_block_reuse=False, host_cache_size=0
            ),
            tensorrt_llm.bindings.internal.batch_manager.CacheType.SELFKONLY,
            num_layers=1,
            num_kv_heads=1,
            head_dim=LATENT,
            tokens_per_block=TOKENS_PER_BLOCK,
            max_seq_len=MAX_SEQ_LEN,
            max_batch_size=NUM_REQUESTS,
            mapping=Mapping(),
            dtype=tensorrt_llm.bindings.DataType.BF16,
        )
        requests = self.manager.add_dummy_requests(
            list(range(NUM_REQUESTS)), [MAX_SEQ_LEN] * NUM_REQUESTS
        )
        assert requests is not None
        for req in requests:
            req.state = LlmRequestState.GENERATION_IN_PROGRESS
        self.pool().zero_()
        self._gen = torch.Generator(device="cuda").manual_seed(seed + 1)

    def shutdown(self) -> None:
        self.manager.shutdown()

    def pool(self) -> torch.Tensor:
        return self.manager.get_buffers(0)

    def snapshot(self) -> torch.Tensor:
        torch.cuda.synchronize()
        return self.pool().clone()

    def restore(self, pool: torch.Tensor) -> None:
        self.pool().copy_(pool)
        torch.cuda.synchronize()

    def hidden(self, step: Step) -> torch.Tensor:
        return torch.randn(
            step.num_tokens, HIDDEN, generator=self._gen, device="cuda", dtype=torch.bfloat16
        )

    def metadata(self, step: Step):
        md = get_attention_backend("TRTLLM").Metadata(
            seq_lens=torch.tensor(step.seq_lens, dtype=torch.int),
            request_ids=list(step.request_ids),
            max_num_requests=NUM_REQUESTS,
            num_contexts=step.num_contexts,
            prompt_lens=list(step.prompt_lens),
            max_num_tokens=1024,
            kv_cache_manager=self.manager,
            kv_cache_params=KVCacheParams(
                use_cache=True, num_cached_tokens_per_seq=list(step.cached)
            ),
            mapping=Mapping(),
            enable_context_mla_with_cached_kv=True,
            runtime_features=AttentionRuntimeFeatures(
                chunked_prefill=step.chunked_prefill, chunk_size=16
            ),
        )
        md.prepare()
        return md

    @contextmanager
    def _attrs(self, md):
        with model_extra_attrs({"attention_metadata": weakref.ref(md)}):
            yield

    def run(self, step: Step, hidden: torch.Tensor, decode_step=None) -> torch.Tensor:
        """``step`` eagerly: on the generic path, or with ``decode_step`` (a ``DecodeStep``) on the decode branch."""
        md = self.metadata(step)
        with self._attrs(md):
            if decode_step is None:
                out = self.runtime.forward_generic(hidden, md, reduce_output=False)
            else:
                assert self.runtime.will_run_decode_branch(md, decode_step), (
                    f"{step.name}: not the decode branch"
                )
                out = self.runtime(hidden, md, step=decode_step, reduce_output=False)
        torch.cuda.synchronize()
        return out[: step.num_tokens]

    def run_breakable_graph(self, step: Step, hidden: torch.Tensor) -> torch.Tensor:
        """``step`` on the generic path captured into a breakable CUDA graph (the attention core as one eager break,
        the projections and the gate in captured segments) and replayed, as the engine runs a prefill step under
        ``prefill_cuda_graph_backend=BREAKABLE``. The cache is restored before the replay, so the replay alone writes
        the step's rows."""
        from tensorrt_llm._torch.pyexecutor.breakable_cuda_graph import (
            BreakableCUDAGraph,
            BreakableCUDAGraphCapture,
            enable_breakable_cuda_graph,
        )

        md = self.metadata(step)
        before = self.snapshot()
        graph = BreakableCUDAGraph()
        static = hidden.clone()
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.inference_mode(), self._attrs(md), enable_breakable_cuda_graph():
            with torch.cuda.stream(stream):
                with BreakableCUDAGraphCapture(
                    graph, pool=torch.cuda.graph_pool_handle(), stream=stream
                ):
                    out = self.runtime.forward_generic(static, md, reduce_output=False)
            torch.cuda.current_stream().wait_stream(stream)
            assert graph.num_breaks == 1, (
                f"{step.name}: {graph.num_breaks} breaks, expected the attention core"
            )
            self.restore(before)
            out.fill_(0)
            graph.replay()
        torch.cuda.synchronize()
        return out[: step.num_tokens]

    def run_cuda_graph(self, step: Step, hidden: torch.Tensor, decode_step=None) -> torch.Tensor:
        """A pure generation ``step`` captured into a CUDA graph (after one eager warm-up) and replayed, as the
        engine runs decode steps: on the generic path, or with ``decode_step`` on the decode branch. The cache is
        restored before the warm-up and before the replay."""
        md = self.metadata(step)
        before = self.snapshot()
        static = hidden.clone()

        def body():
            if decode_step is None:
                return self.runtime.forward_generic(static, md, reduce_output=False)
            return self.runtime(static, md, step=decode_step, reduce_output=False)

        graph = torch.cuda.CUDAGraph()
        with torch.inference_mode(), self._attrs(md):
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                body()
            torch.cuda.current_stream().wait_stream(stream)
            self.restore(before)
            with torch.cuda.graph(graph):
                out = body()
            self.restore(before)
            out.fill_(0)
            graph.replay()
        torch.cuda.synchronize()
        return out[: step.num_tokens]


def rows_of(layer: MlaLayer, request_ids: List[int], lens: List[int]) -> torch.Tensor:
    """The cache rows ``[0, lens[k])`` of each request in ``request_ids``, concatenated, read through the manager's
    block ids."""
    pool = layer.pool()
    out = []
    for rid, n in zip(request_ids, lens):
        blocks = list(layer.manager.get_batch_cache_indices([rid])[0])
        rows = torch.cat(
            [pool[b, 0, :, 0] for b in blocks[: (n + TOKENS_PER_BLOCK - 1) // TOKENS_PER_BLOCK]]
        )
        out.append(rows[:n])
    return torch.cat(out)


def advance(lens: List[int], step: Step) -> List[int]:
    """The cache lengths after ``step``."""
    lens = list(lens)
    for rid, n in zip(step.request_ids, step.seq_lens):
        lens[rid] = step.cached[step.request_ids.index(rid)] + n
    return lens
