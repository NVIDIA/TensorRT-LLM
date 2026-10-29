# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Mistral-Large-4-1T-A52B-NVFP4 at tp4 in the checkpoint's own context: the engine at max_seq_len
1048576 with chunked prefill in 32768-token steps, as a served engine reaches it.

No GPU and no weights. The checkpoint goes through the engine's own configuration path with the
LLM-API options of the 1M core unit (four ranks, tensor parallel throughout, the routed experts cut
along their width, 16 requests and 32768 tokens per step, max_seq_len 1048576, chunked prefill on,
block reuse off, the KV cache at its default dtype), and in the spelling the quickstart gates use
(``--kv_cache_dtype=fp8``). On the configuration it resolves to, the engine's own host code is run
wherever a CPU reaches it:

* (1) the limits on their way to the executor and the attention metadata, the quantization key,
  the KV cache manager both spellings build (``MistralLarge4CacheManager``, the fp8_ds_mla latent
  cache), the DSA sparse config and the attention runtime features that turn on context MLA over
  cached KV;
* (2) the DSA indexer's prefill planning (``Indexer.prepare_for_chunked_prefill``, which calls
  ``_split_prefill_queries`` and ``plan_mla_chunked_prefill_chunks``) on each of the four ranks'
  host metadata: the query split across the TP ranks that the target must gather back;
* (3) the KV-capacity estimate (``KvCacheCreator``): its dummy requests, the reservations it makes
  at this config and the pool and admission cap it leaves of the 32K boot's budget; the
  warmup's steps and CUDA-graph decode rows; the target's held context workspace, sized from the
  engine's quantities, and the context-row path it takes on each of these steps and on a 1M
  prompt's chunks;
* (4) the DSA metadata buffers that grow with max_seq_len, and the per-step context gather
  mappings.

The delivered 32K configuration (pinned in test_modeling_v2_tp4_engine_config.py) is resolved
beside it as the contrast. The checkpoint is Mistral-native, as published (params.json,
tekken.json, consolidated safetensors), loaded with checkpoint_format mistral_large_4 in every
spelling. It is looked up under ``LLM_MODELS_ROOT`` (as ``Mistral-Large-4-1T-A52B-NVFP4`` or
``mistral-large-4-nvfp4``) by its params.json, and the tests are skipped where it is
absent. A CPU job may still see the node's GPUs: nothing here opens a CUDA context,
allocates pinned memory or reads a device property, and the SM is stated as sm_103 wherever the
engine asks for it.
"""

from __future__ import annotations

import copy
import importlib
import itertools
import json
import math
import threading
import types
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml
from test_common.llm_data import llm_models_root

from tensorrt_llm._torch._experimental.modeling_v2 import _router_index
from tensorrt_llm._torch._experimental.modeling_v2.models.mistral_large4 import routing
from tensorrt_llm._torch._experimental.modeling_v2.models.mistral_large4.mistral_large_4_nvfp4__sm_103__tp4 import (  # noqa: E501
    modeling as tp4_modeling,
)
from tensorrt_llm._torch.attention.backends import trtllm as trtllm_backend
from tensorrt_llm._torch.attention.backends.interface import AttentionRuntimeFeatures
from tensorrt_llm._torch.attention.backends.sparse.dsa import indexer as dsa_indexer
from tensorrt_llm._torch.attention.backends.sparse.dsa import metadata as dsa_metadata
from tensorrt_llm._torch.attention.backends.sparse.dsa.indexer import Indexer
from tensorrt_llm._torch.attention.backends.sparse.dsa.metadata import DSAtrtllmAttentionMetadata
from tensorrt_llm._torch.attention.backends.sparse.mistral_large4 import MistralLarge4CacheManager
from tensorrt_llm._torch.attention.backends.sparse.params import (
    get_indexer_mqa_logits_workspace_bytes,
)
from tensorrt_llm._torch.models.modeling_utils import DecoderModelForCausalLM
from tensorrt_llm._torch.pyexecutor import _util, py_executor_creator
from tensorrt_llm._torch.pyexecutor._util import KvCacheCreator
from tensorrt_llm._torch.pyexecutor.engine.metadata import build_attention_metadata
from tensorrt_llm._torch.pyexecutor.engine.runners.decoder.dummy_requests import DummyRequestFactory
from tensorrt_llm._torch.pyexecutor.engine.runners.decoder.warmup import (
    DecoderWarmup,
    WarmupAgreement,
)
from tensorrt_llm._torch.pyexecutor.kv_cache import kv_cache_manager_v2
from tensorrt_llm._torch.pyexecutor.model_engine import PyTorchModelEngine
from tensorrt_llm._torch.pyexecutor.model_loader import ModelLoader as RankModelLoader
from tensorrt_llm._torch.pyexecutor.py_executor import PyExecutor
from tensorrt_llm._torch.pyexecutor.resource_manager import ResourceManagerType
from tensorrt_llm.bindings.internal import thop
from tensorrt_llm.llmapi.llm_args import SeqLenAwareSparseAttentionConfig, TorchLlmArgs
from tensorrt_llm.llmapi.llm_utils import ModelLoader as FrontEndModelLoader
from tensorrt_llm.models.modeling_utils import QuantAlgo
from tensorrt_llm.sampling_params import SamplingParams

pytestmark = pytest.mark.cpu_only

_SM103 = (10, 3)
_RANKS = 4
_MAX_SEQ_LEN = 1048576
# max_num_tokens: the chunk, and the step budget context and decode tokens share.
_CHUNK = 32768
_MAX_BATCH = 16
_PAGE = 32
# A token's KV on every rank: a 656-byte fp8_ds_mla latent row (512 e4m3, four fp32 tile scales,
# 64 bf16 rope) on each of the 63 KV owners, a 128-byte fp8 indexer key and its fp32 scale on each
# of the 44 full-indexer layers.
_LATENT_ROW_BYTES = 656
_KV_OWNERS = 63
_INDEXER_KEY_BYTES = 132
_FULL_INDEXER_LAYERS = 44
_TOKEN_BYTES = 47136
_TOPK = 2048
_Q_SPLIT_THRESHOLD = 8192
# The pool each rank of a boot at this config on the fp8_ds_mla cache held, the native
# checkpoint's vision tower loaded, in tokens (66756 pages of 32): the budget handed to
# the estimate.
_BOOT_POOL_TOKENS = 2_136_192
# The last chunk of a 1048576-token request reads a cached prefix of 1015808 tokens.
_LAST_CHUNK_PREFIX = _MAX_SEQ_LEN - _CHUNK
# The envelope gate's long prompt: max_seq_len less its 16 output tokens and one.
_ENVELOPE_PROMPT = _MAX_SEQ_LEN - 16 - 1

_TP4 = dict(
    tensor_parallel_size=4,
    moe_tensor_parallel_size=4,
    moe_expert_parallel_size=1,
    enable_attention_dp=False,
)
# The quickstart gates (boot, envelope, witness) state only --tp_size=4, with the MoE sizes as the
# -1 sentinels, and run the KV pool at the quickstart's --kv_cache_fraction default 0.9.
_TP4_QUICKSTART = dict(
    tensor_parallel_size=4,
    moe_expert_parallel_size=-1,
    moe_tensor_parallel_size=-1,
    moe_cluster_parallel_size=-1,
    enable_attention_dp=False,
)


def _options(
    topology: dict, max_seq_len: int, chunked: bool, fraction: float = 0.85, **kv_dtype
) -> dict:
    """The LLM-API options of a Mistral Large 4 tp4 gate, the checkpoint in its Mistral-native format:
    the 1M core unit at 1048576 with chunked prefill, the delivered 32K engine at 32768 without
    it. The KV cache dtype is left at its default (auto) as the CI class leaves it, unless
    ``kv_dtype`` names one."""
    return dict(
        checkpoint_format="mistral_large_4",
        **topology,
        max_batch_size=_MAX_BATCH,
        max_num_tokens=_CHUNK,
        max_seq_len=max_seq_len,
        enable_chunked_prefill=chunked,
        cuda_graph_config=dict(enable_padding=True, max_batch_size=_MAX_BATCH),
        kv_cache_config=dict(
            enable_block_reuse=False, free_gpu_memory_fraction=fraction, **kv_dtype
        ),
    )


_NVFP4_NAMES = ("Mistral-Large-4-1T-A52B-NVFP4", "mistral-large-4-nvfp4")

# The GSM8K reference rows, and the key the harness reads them by.
_GSM8K_REFERENCES = (
    Path(__file__).resolve().parents[6] / "integration/defs/accuracy/references/gsm8k.yaml"
)
_MODEL_NAME = "mistralai/Mistral-Large-4-1T-A52B"

_TP4_TARGET = "ModelingV2MistralLarge4Nvfp4Sm103Tp4"


def _checkpoint_dir(names: tuple[str, ...]) -> Path:
    """The first of ``names`` under LLM_MODELS_ROOT holding a Mistral-native params.json."""
    root = llm_models_root()
    for name in names:
        if root is not None and (root / name / "params.json").is_file():
            return root / name
    pytest.skip(f"none of {names} is under LLM_MODELS_ROOT ({root})")


class _Routed:
    """What the recording `route` returns: no target is built here."""


def _resolve(checkpoint_dir: Path, options: dict) -> SimpleNamespace:
    """The configuration a served engine reaches for this checkpoint, front end then rank."""
    model = str(checkpoint_dir)
    llm_args = TorchLlmArgs(model=model, modeling_v2="require", **options)
    prequantized = FrontEndModelLoader(llm_args)._update_from_hf_quant_config()

    routed = []

    def route(pretrained_config, mapping, compute_capability, spec_config):
        routed.append(
            SimpleNamespace(
                pretrained_config=pretrained_config,
                quantization_config=copy.deepcopy(pretrained_config.quantization_config),
                mapping=mapping,
                compute_capability=compute_capability,
                spec_config=spec_config,
            )
        )
        return _Routed

    with pytest.MonkeyPatch.context() as mp:
        # The rank's config pass pins the CUDA memory fraction first; there is no device here.
        mp.setattr(torch.cuda, "set_per_process_memory_fraction", lambda *args, **kwargs: None)
        mp.setattr(_router_index, "_device_compute_capability", lambda: _SM103)
        mp.setattr(_router_index, "routing_module", lambda arch: SimpleNamespace(route=route))
        llm_args, checkpoint_loader = py_executor_creator._load_config_and_create_checkpoint_loader(
            llm_args, model
        )
        _, max_num_tokens, max_seq_len, max_batch_size = llm_args.get_runtime_sizes()
        mapping = llm_args.parallel_config.to_mapping()
        model_config = RankModelLoader(
            llm_args=llm_args,
            mapping=mapping,
            spec_config=llm_args.speculative_config,
            sparse_attention_config=llm_args.sparse_attention_config,
            max_num_tokens=max_num_tokens,
            max_seq_len=max_seq_len,
        )._load_and_validate_config(model, checkpoint_loader)
    return SimpleNamespace(
        path=checkpoint_dir,
        llm_args=llm_args,
        prequantized=prequantized,
        model_config=model_config,
        mapping=mapping,
        routed=routed,
        max_num_tokens=max_num_tokens,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )


@pytest.fixture(scope="module")
def tp4():
    """The 1M core unit, as the CI class builds it: the KV cache dtype at its default."""
    return _resolve(_checkpoint_dir(_NVFP4_NAMES), _options(_TP4, _MAX_SEQ_LEN, True))


@pytest.fixture(scope="module")
def tp4_fp8():
    """The 1M core unit as the gates' options file states it: the KV cache dtype fp8."""
    return _resolve(_checkpoint_dir(_NVFP4_NAMES), _options(_TP4, _MAX_SEQ_LEN, True, dtype="fp8"))


@pytest.fixture(scope="module")
def tp4_quickstart():
    """The 1M config as the quickstart gates build it (``--kv_cache_dtype=fp8``)."""
    return _resolve(
        _checkpoint_dir(_NVFP4_NAMES),
        _options(_TP4_QUICKSTART, _MAX_SEQ_LEN, True, fraction=0.9, dtype="fp8"),
    )


@pytest.fixture(scope="module")
def tp4_32k():
    """The delivered engine: 32768, unchunked."""
    return _resolve(_checkpoint_dir(_NVFP4_NAMES), _options(_TP4, _CHUNK, False))


def _anchor(llm_args) -> float:
    """The reference row `AccuracyTask.get_hypothesis_testing_params` selects for this engine."""
    quant = llm_args.quant_config
    acc_specs = dict(
        dtype=llm_args.dtype,
        quant_algo=quant.quant_algo,
        kv_cache_quant_algo=quant.kv_cache_quant_algo,
        spec_dec_algo=None,
        extra_acc_spec=None,
    )
    for entry in yaml.safe_load(_GSM8K_REFERENCES.read_text())[_MODEL_NAME]:
        if all(
            entry.get(key, "auto" if key == "dtype" else None) == value
            for key, value in acc_specs.items()
        ):
            return entry["accuracy"]
    raise AssertionError(f"no {_MODEL_NAME} GSM8K reference row for {acc_specs}")


def _rank_mapping(mapping, rank: int):
    rank_mapping = copy.deepcopy(mapping)
    rank_mapping.rank = rank
    return rank_mapping


def _pin_sm103(mp: pytest.MonkeyPatch) -> None:
    """The engine reads the SM off the device in these modules; state it as sm_103."""
    for module in (_util, trtllm_backend, dsa_indexer, dsa_metadata):
        mp.setattr(module, "get_sm_version", lambda: 103)


def _host_only(mp: pytest.MonkeyPatch) -> None:
    """Keep the engine's host-side preparation on the host: no pinned staging buffers, and the
    copies it hands to the device stay where they are."""
    for module in (dsa_indexer, dsa_metadata):
        mp.setattr(module, "prefer_pinned", lambda: False)
    mp.setattr(dsa_indexer, "maybe_pin_memory", lambda tensor: tensor)
    mp.setattr(torch.Tensor, "cuda", lambda self, *args, **kwargs: self)


def _block_table_width(max_seq_len: int) -> int:
    """``max_blocks_per_seq`` of ``KVCacheManagerV2``, the width of every block table the metadata
    sizes by it: the pages of max_seq_len plus the generation headroom, padded to a multiple of 4."""
    pages = math.ceil((max_seq_len + kv_cache_manager_v2.BASE_GENERATION_TOKEN_COUNT) / _PAGE)
    return (pages + 3) // 4 * 4


# ------------------------------------------------------- four ranks in one process


class _TpGroup:
    """The four ranks of the TP group in one process: rank r is a thread, and every exchange is a
    barrier, so each rank's call sees what every rank sent, as over MPI."""

    def __init__(self, size: int = _RANKS):
        self.size = size
        self._barrier = threading.Barrier(size, timeout=60)
        self._slots = [None] * size

    def exchange(self, rank: int, obj):
        self._slots[rank] = obj
        self._barrier.wait()
        gathered = list(self._slots)
        self._barrier.wait()
        return gathered

    def run(self, step):
        """``step(rank)`` on every rank at once; the per-rank results in rank order."""
        results, errors = [None] * self.size, [None] * self.size

        def body(rank):
            try:
                results[rank] = step(rank)
            except BaseException as error:  # noqa: BLE001 - re-raised below, after every rank
                errors[rank] = error
                self._barrier.abort()

        threads = [threading.Thread(target=body, args=(rank,)) for rank in range(self.size)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        # The rank that failed first, not a peer it left waiting at the barrier.
        raised = [error for error in errors if error is not None]
        for error in raised:
            if not isinstance(error, threading.BrokenBarrierError):
                raise error
        if raised:
            raise raised[0]
        return results


class _RankDist:
    """One rank's communicator: the world gather the warmup agrees its shapes over."""

    def __init__(self, group: _TpGroup, rank: int):
        self.group = group
        self.rank = self.tp_rank = rank
        self.world_size = self.tp_size = group.size

    def allgather(self, obj):
        return self.group.exchange(self.rank, obj)


class _KvPool:
    """The KV manager surface the warmup's dummy requests read: ``tokens`` tokens in pages of 32,
    and dummies recorded as the token counts they were asked for. The longest sequence it offers
    follows ``KVCacheManagerV2.get_num_available_tokens``: what is left once batch_size - 1
    one-page sequences are set aside."""

    tokens_per_block = _PAGE
    num_extra_kv_tokens = 0

    def __init__(self, tokens: int, max_seq_len: int):
        self.tokens = tokens
        self.max_seq_len = max_seq_len

    def get_num_available_tokens(
        self, *, token_num_upper_bound, batch_size=1, max_num_draft_tokens=0, **_
    ):
        pages = self.tokens // self.tokens_per_block - (batch_size - 1)
        return max(0, min(token_num_upper_bound, pages * self.tokens_per_block))

    def get_num_free_blocks(self) -> int:
        return self.tokens // self.tokens_per_block

    def add_dummy_requests(self, request_ids, token_nums=None, is_gen=False, **kwargs):
        if token_nums is None:
            token_nums = [1 + kwargs.get("max_num_draft_tokens", 0)] * len(request_ids)
        return [
            SimpleNamespace(
                py_request_id=request_id,
                token_num=token_num,
                is_gen=is_gen,
                is_last_context_chunk=True,
                is_attention_dp_dummy=False,
                py_draft_tokens=[],
                py_needs_onehot_draft_probs=False,
            )
            for request_id, token_num in zip(request_ids, token_nums)
        ]

    def free_resources(self, request) -> None:
        pass


class _Resources:
    def __init__(self, kv: _KvPool):
        self.kv = kv

    def get_resource_manager(self, key):
        return self.kv if key == ResourceManagerType.KV_CACHE_MANAGER else None


def _runner_config(resolved) -> SimpleNamespace:
    """The decoder runner's sizes for this engine. No speculative decoding: every draft width is
    zero, and beams are one wide."""
    graphs = resolved.llm_args.cuda_graph_config
    return SimpleNamespace(
        max_num_tokens=resolved.max_num_tokens,
        max_seq_len=resolved.max_seq_len,
        max_batch_size=resolved.max_batch_size,
        cuda_graph_batch_sizes=graphs.batch_sizes,
        max_beam_width=1,
        spec_config=None,
        use_mrope=False,
        max_draft_len=0,
        original_max_draft_len=0,
        max_total_draft_tokens=0,
        max_draft_loop_tokens=0,
    )


def _dummy_factory(resolved) -> DummyRequestFactory:
    return DummyRequestFactory(
        config=_runner_config(resolved),
        kv_cache_manager_key=ResourceManagerType.KV_CACHE_MANAGER,
        model=SimpleNamespace(model_config=resolved.model_config),
        get_runtime_tokens_per_gen_step=lambda draft_len: 1 + draft_len,
        release_request_state=None,
    )


# ------------------------------------------------------- the engine's attention metadata


class _RecordingBackend:
    """An attention backend whose metadata is the keyword arguments it is built with."""

    Metadata = SimpleNamespace


def _runtime_features(resolved) -> AttentionRuntimeFeatures:
    """``AttentionRuntimeFeatures`` as ``create_py_executor`` builds them for this engine: no
    speculative drafter, and the chunk is max_num_tokens."""
    llm_args = resolved.llm_args
    return AttentionRuntimeFeatures(
        chunked_prefill=llm_args.enable_chunked_prefill,
        cache_reuse=llm_args.kv_cache_config.enable_block_reuse,
        has_speculative_draft_tokens=False,
        chunk_size=resolved.max_num_tokens,
    )


def _engine_metadata(resolved, rank: int) -> SimpleNamespace:
    """What ``build_attention_metadata`` hands the attention backend's metadata on ``rank``."""
    return build_attention_metadata(
        resolved.model_config,
        max_batch_size=resolved.max_batch_size,
        max_num_tokens=resolved.max_num_tokens,
        max_beam_width=1,
        attention_backend=_RecordingBackend,
        attention_runtime_features=_runtime_features(resolved),
        mapping=_rank_mapping(resolved.mapping, rank),
        cache_indirection=None,
    )


def _host_metadata(resolved, rank: int, q_lens, cached, gen_kv=()) -> SimpleNamespace:
    """One rank's DSA metadata for one step, host side: the engine's metadata arguments for the
    rank, the step's requests, and what ``DSAtrtllmAttentionMetadata.prepare`` derives from them
    before it plans the indexer (the context and generation indptrs, the indexer-skip decision).
    ``q_lens`` and ``cached`` are the context rows' new and cached tokens; ``gen_kv`` the KV
    lengths of the decode rows behind them."""
    engine = _engine_metadata(resolved, rank)
    params = engine.sparse_metadata_params
    num_contexts, num_generations = len(q_lens), len(gen_kv)
    seq_lens = torch.tensor([*q_lens, *[1] * num_generations], dtype=torch.int32)
    cached_tokens = [*cached, *(kv - 1 for kv in gen_kv)]
    width = _block_table_width(resolved.max_seq_len)

    def indptr():
        return torch.zeros(engine.max_num_requests + 1, dtype=torch.int64)

    md = SimpleNamespace(**vars(engine))
    md.kv_cache_manager = SimpleNamespace(
        index_head_dim=params.index_head_dim,
        max_seq_len=resolved.max_seq_len,
        tokens_per_block=_PAGE,
    )
    md.indexer_max_chunk_size = params.indexer_max_chunk_size
    md.num_sparse_topk = params.max_sparse_topk
    md.enable_indexer_skip = params.enable_indexer_skip
    md.indexer_head_dim = params.index_head_dim
    md.indexer_quant_block_size = 128
    md._tokens_per_block = _PAGE
    md.compress_ratios = [1]
    md.num_contexts, md.num_generations = num_contexts, num_generations
    md.num_seqs = num_contexts + num_generations
    md.num_ctx_tokens = sum(q_lens)
    md.num_tokens = md.num_ctx_tokens + num_generations
    md.seq_lens = seq_lens
    md.request_ids = list(range(md.num_seqs))
    md.kv_cache_params = SimpleNamespace(
        num_cached_tokens_per_seq=cached_tokens, num_extra_kv_tokens=0
    )
    for name in ("ctx_cached_token", "ctx_kv", "gen_cached_token", "gen_kv"):
        setattr(md, f"{name}_indptr", indptr())
        setattr(md, f"host_{name}_indptr", indptr())
    md.cu_seqlen_ks = torch.empty(0, dtype=torch.int32)
    md.host_indexer_k_cache_block_offsets = torch.arange(
        engine.max_num_sequences * width, dtype=torch.int32
    ).view(engine.max_num_sequences, width)
    md.slot_mapping_fp8 = torch.empty(engine.max_num_tokens, dtype=torch.int64)
    md.slot_mapping_scale = torch.empty(engine.max_num_tokens, dtype=torch.int64)
    md.indexer_prefill_q_split = None
    md.indexer_prefill_chunks = None

    md.prepare_dense_topk_indices = types.MethodType(
        DSAtrtllmAttentionMetadata.prepare_dense_topk_indices, md
    )

    cached_lens = torch.tensor(cached_tokens, dtype=torch.int)
    kv_lens = cached_lens + seq_lens
    DSAtrtllmAttentionMetadata.prepare_for_mla_rope_append(md, cached_lens, kv_lens)
    DSAtrtllmAttentionMetadata.prepare_for_skip_indexer(md, kv_lens)
    return md


def _plan(resolved, rank: int, q_lens, cached, gen_kv=()) -> SimpleNamespace:
    """The indexer's prefill plan for one step on ``rank``, by the engine's own planning."""
    md = _host_metadata(resolved, rank, q_lens, cached, gen_kv)
    with pytest.MonkeyPatch.context() as mp:
        _host_only(mp)
        Indexer.prepare_for_chunked_prefill(md, Indexer.build_indexer_params(md))
    return md


def _target_ctx_kv_offsets(md, q_lens) -> dict:
    """The target's context offsets, `_ctx_kv_offsets` and `_ctx_host_offsets`, on a step `_plan`
    prepared, given the uncached prefix sums the engine's `TrtllmAttentionMetadata.prepare` fills
    beside the DSA ones."""
    md.host_ctx_uncached_token_indptr = torch.tensor(
        [0, *itertools.accumulate(q_lens)], dtype=torch.int64
    )
    return dict(
        **tp4_modeling._ctx_kv_offsets(md, md.num_contexts),
        **tp4_modeling._ctx_host_offsets(md, md.num_contexts),
    )


def _target_blocks(md, q_lens) -> list:
    """The target's indexer scoring blocks (`_indexer_blocks`) for a step `_plan` prepared on this
    engine, from the entries `PrefillTarget.step_args` hands it."""
    # The engine plans chunks under chunked prefill, so the whole-step fallback's entries
    # (num_ctx_kv_tokens, the step's windows) are never read here.
    assert md.indexer_prefill_chunks is not None
    step = dict(
        num_ctx_tokens=md.num_ctx_tokens,
        max_seq_len=md.kv_cache_manager.max_seq_len,
        **_target_ctx_kv_offsets(md, q_lens),
    )
    return tp4_modeling._indexer_blocks(md.indexer_prefill_chunks, step)


def _target_sizes(resolved) -> SimpleNamespace:
    """What the target sizes its buffers from on the engine's attention metadata
    (`_derive_first_step`): max_num_requests, max_num_tokens and runtime_features as
    ``build_attention_metadata`` hands them over (the features as ``create_py_executor`` builds
    them, `_runtime_features`: chunked prefill from enable_chunked_prefill, cache reuse from the
    KvCacheConfig's enable_block_reuse), and max_seq_len, which ``TrtllmAttentionMetadata`` reads
    off the KV cache manager the engine builds for it."""
    md = _engine_metadata(resolved, 0)
    return SimpleNamespace(
        max_num_requests=md.max_num_requests,
        max_num_tokens=md.max_num_tokens,
        max_seq_len=resolved.max_seq_len,
        runtime_features=md.runtime_features,
    )


def _spans(md) -> list[tuple[int, int, int, int]]:
    """Each planned chunk's query rows [token_start, token_end) and keys [k_start, k_end)."""
    return [
        (chunk.token_start, chunk.token_end, chunk.k_token_start, chunk.k_token_end)
        for chunk in md.indexer_prefill_chunks
    ]


# ------------------------------------------------------- (1) topology, limits, quantization


def test_tp4_mapping_as_at_base(tp4, tp4_quickstart, tp4_32k):
    """The tp4 mapping exactly as the delivered 32K engine has it, in both spellings: attention TP
    4 with no attention DP, the routed experts tensor parallel four ways and not expert-parallel,
    one pipeline stage, rank r MoE TP rank r and MoE EP rank 0. The router is handed it, and the
    family's routing claims the configuration with the tp4 target: max_seq_len is no routing
    criterion."""
    llm_args, mapping = tp4.llm_args, tp4.mapping
    assert (
        llm_args.tensor_parallel_size,
        llm_args.moe_tensor_parallel_size,
        llm_args.moe_expert_parallel_size,
        llm_args.enable_attention_dp,
    ) == (4, 4, 1, False)
    assert (mapping.world_size, mapping.tp_size, mapping.pp_size, mapping.cp_size) == (4, 4, 1, 1)
    assert (mapping.moe_tp_size, mapping.moe_ep_size, mapping.moe_cluster_size) == (4, 1, 1)
    assert mapping.enable_attention_dp is False and mapping.dp_size == 1
    assert mapping.attn_tp_size == 4
    assert mapping.to_dict() == tp4_32k.mapping.to_dict()
    assert tp4_quickstart.mapping.to_dict() == mapping.to_dict()
    for rank in range(_RANKS):
        rank_mapping = _rank_mapping(mapping, rank)
        assert (
            rank_mapping.tp_rank,
            rank_mapping.moe_tp_rank,
            rank_mapping.moe_ep_rank,
            rank_mapping.moe_cluster_rank,
        ) == (rank, rank, 0, 0)
        assert _rank_mapping(tp4_quickstart.mapping, rank) == rank_mapping
        assert _rank_mapping(tp4_32k.mapping, rank) == rank_mapping

    for resolved in (tp4, tp4_quickstart):
        seen = resolved.routed[0]
        assert seen.mapping is resolved.model_config.mapping
        assert seen.mapping.to_dict() == mapping.to_dict()
        assert seen.compute_capability == _SM103 and seen.spec_config is None
        target = routing.route(seen.pretrained_config, seen.mapping, _SM103, None)
        assert target.__name__ == _TP4_TARGET


def test_limits_reach_the_engine_unclamped(tp4, tp4_quickstart, monkeypatch):
    """max_seq_len 1048576 is the checkpoint's max_position_embeddings, and nothing on the way
    from the LLM API to the executor lowers it: the LLM args and the rank's ModelConfig carry it;
    ``PyTorchModelEngine`` keeps it, since the causal-LM shell's ``infer_max_seq_len`` reads the
    same 1048576 off this config; and ``create_py_executor`` keeps chunked context on for MLA on
    sm_103. 32768 tokens and 16 requests per step, chunked prefill on, block reuse off, in both
    spellings."""
    cfg = tp4.model_config.pretrained_config
    assert cfg.max_position_embeddings == _MAX_SEQ_LEN
    for resolved in (tp4, tp4_quickstart):
        llm_args = resolved.llm_args
        assert (llm_args.max_seq_len, llm_args.max_num_tokens, llm_args.max_batch_size) == (
            _MAX_SEQ_LEN,
            _CHUNK,
            _MAX_BATCH,
        )
        assert llm_args.enable_chunked_prefill is True
        assert llm_args.kv_cache_config.enable_block_reuse is False
        assert (resolved.max_num_tokens, resolved.max_seq_len, resolved.max_batch_size) == (
            _CHUNK,
            _MAX_SEQ_LEN,
            _MAX_BATCH,
        )
        model_config = resolved.model_config
        assert (model_config.max_num_tokens, model_config.max_seq_len) == (_CHUNK, _MAX_SEQ_LEN)
    assert tp4.llm_args.kv_cache_config.free_gpu_memory_fraction == 0.85
    assert tp4_quickstart.llm_args.kv_cache_config.free_gpu_memory_fraction == 0.9

    monkeypatch.delenv("TLLM_ALLOW_LONG_MAX_MODEL_LEN", raising=False)
    shell = SimpleNamespace(config=cfg)
    assert DecoderModelForCausalLM.infer_max_seq_len(shell) == _MAX_SEQ_LEN
    engine = SimpleNamespace(
        model=SimpleNamespace(
            infer_max_seq_len=lambda: DecoderModelForCausalLM.infer_max_seq_len(shell)
        ),
        max_seq_len=tp4.max_seq_len,
        max_num_tokens=tp4.max_num_tokens,
        batch_size=tp4.max_batch_size,
    )
    PyTorchModelEngine._init_max_seq_len(engine)
    PyTorchModelEngine._init_max_num_tokens(engine)
    assert (engine.max_seq_len, engine.max_num_tokens) == (_MAX_SEQ_LEN, _CHUNK)

    assert 103 in py_executor_creator._MLA_CHUNKED_PREFILL_SUPPORTED_SM_VERSIONS


def test_llm_api_admits_a_1m_prompt(tp4, tp4_32k):
    """``BaseLLM._check_arguments`` refuses a prompt longer than max_num_tokens only with chunked
    prefill off: the envelope gate's 1048559-token prompt passes it at 1M, where the delivered 32K
    engine refuses 32769 tokens."""
    llm_module = importlib.import_module("tensorrt_llm.llmapi.llm")

    def check(resolved, prompt_len):
        llm = SimpleNamespace(
            args=resolved.llm_args, _check_one_model_speculative_sampling=lambda params: None
        )
        llm_module.BaseLLM._check_arguments(
            llm, prompt_len, SamplingParams(max_tokens=16), is_gen_only=False
        )

    check(tp4, _ENVELOPE_PROMPT)
    check(tp4_32k, _CHUNK)
    with pytest.raises(llm_module.RequestError, match="should not exceed max_num_tokens"):
        check(tp4_32k, _CHUNK + 1)


def test_quantization_is_the_anchor_key(tp4):
    """At 1M as at 32K: MIXED_PRECISION with the fp8 KV cache (the checkpoint's fp8_kv_cache under
    the default dtype), the GSM8K anchor row's key, and the router handed params.json's
    quantization_config with its 475 quantized_layers intact."""
    assert tp4.prequantized is True
    for quant in (tp4.llm_args.quant_config, tp4.model_config.quant_config):
        assert quant.quant_algo == QuantAlgo.MIXED_PRECISION
        assert quant.kv_cache_quant_algo == QuantAlgo.FP8
    assert tp4.llm_args.kv_cache_config.dtype == "auto"
    assert _anchor(tp4.llm_args) == 91.926

    on_disk = json.loads((tp4.path / "params.json").read_text())["quantization_config"]
    assert len(tp4.routed) == 1
    seen = tp4.routed[0]
    assert seen.quantization_config == on_disk and len(on_disk["quantized_layers"]) == 475
    assert tp4.model_config.modeling_v2_target is _Routed


def test_kv_cache_is_fp8_ds_mla(tp4, tp4_fp8, tp4_quickstart):
    """Every spelling builds the fp8_ds_mla latent cache: the CI class's default dtype (auto, the
    checkpoint's fp8_kv_cache), the gates' options file's fp8 and the quickstart gates' fp8 each
    select
    ``MistralLarge4CacheManager`` (the sparse registry picks it by the MistralLarge4Config), and
    the engine hands it the FP8 element type. Per token on every rank: a 656-byte row on each of
    the 63 KV owners and the 132-byte indexer key on each of the 44 full-indexer layers, 47136 B,
    in pages of 32. The target refuses any other latent layout on its first step."""
    cfg = tp4.model_config.pretrained_config
    owners = sum(1 for i, s in enumerate(cfg.kv_source_layers) if s == i)
    assert (owners, cfg.index_topk_pattern.count("F")) == (_KV_OWNERS, _FULL_INDEXER_LAYERS)
    assert _TOKEN_BYTES == (
        _KV_OWNERS * _LATENT_ROW_BYTES + _FULL_INDEXER_LAYERS * _INDEXER_KEY_BYTES
    )
    for resolved, dtype in ((tp4, "auto"), (tp4_fp8, "fp8"), (tp4_quickstart, "fp8")):
        kv_cache_config = resolved.llm_args.kv_cache_config
        assert kv_cache_config.dtype == dtype
        assert kv_cache_config.tokens_per_block == _PAGE
        assert resolved.model_config.quant_config.quant_mode.has_fp8_kv_cache()
        manager = _util.get_kv_cache_manager_cls(resolved.model_config, kv_cache_config)
        assert manager is MistralLarge4CacheManager
        assert manager.latent_layout == "fp8_ds_mla"
        assert manager.latent_token_bytes == _LATENT_ROW_BYTES
        per_token = manager.get_cache_size_per_token(resolved.model_config, resolved.mapping)
        assert per_token == _TOKEN_BYTES


def test_dsa_sparse_config(tp4, tp4_32k, monkeypatch):
    """The DSA config the metadata is built from: the checkpoint's index_topk 2048, q_split_threshold
    at its default 8192 (nothing sets -1), no indexer_max_chunk_size (32768 in the metadata), the
    short-sequence indexer skip on, the shared-indexer layers seen, and neither the CuTe DSL top-k
    nor the heuristic top-k. The two logits caps the prefill planning reads are their defaults:
    2^31 elements per logits call, 2^28 packed per chunk. The target tiles its own indexer prefill
    logits under `_prefill_logits_budget` of this engine: the ceiling, 2^28 (256 rows at 1048576
    keys), where the delivered 32K engine's is 2^24 (512 rows at 32768 keys)."""
    cfg = tp4.model_config.pretrained_config
    sparse = tp4.model_config.sparse_attention_config
    assert sparse.algorithm == "dsa"
    assert sparse.q_split_threshold == _Q_SPLIT_THRESHOLD
    assert sparse.indexer_max_chunk_size is None
    assert sparse.skip_indexer_for_short_seqs is True
    assert sparse.use_cute_dsl_topk is False and sparse.enable_heuristic_topk is False
    assert cfg.index_topk == _TOPK

    params = sparse.to_sparse_metadata_params(pretrained_config=cfg)
    assert params.max_sparse_topk == _TOPK
    assert params.q_split_threshold == _Q_SPLIT_THRESHOLD
    assert params.indexer_max_chunk_size == 32768
    assert params.index_head_dim == 128
    assert params.enable_indexer_skip is True
    assert params.has_shared_indexer_layers is True

    user = tp4.llm_args.sparse_attention_config
    assert user is None or user.q_split_threshold == _Q_SPLIT_THRESHOLD

    assert dsa_indexer._INDEXER_MQA_LOGITS_ELEM_BUDGET == 1 << 31
    assert dsa_indexer._INDEXER_PACKED_LOGITS_ELEM_BUDGET == 1 << 28

    monkeypatch.delenv("TLLM_INDEXER_MQA_LOGITS_ELEM_BUDGET", raising=False)
    budget = tp4_modeling._prefill_logits_budget(tp4.max_num_tokens, tp4.max_seq_len)
    assert budget == tp4_modeling._PREFILL_LOGITS_CEILING == 1 << 28
    assert budget // _MAX_SEQ_LEN == 256
    assert tp4_modeling._prefill_logits_budget(tp4_32k.max_num_tokens, tp4_32k.max_seq_len) == (
        1 << 24
    )


def test_context_mla_over_cached_kv_turns_on(tp4, tp4_32k):
    """``create_py_executor`` builds the attention runtime features with chunked_prefill on and the
    chunk max_num_tokens, and ``build_attention_metadata`` turns on context MLA over cached KV for
    an MLA model with chunked prefill: every chunk after a request's first is a context request with
    a cached prefix. The delivered 32K engine (no chunking, no reuse) has it off."""
    for rank in range(_RANKS):
        md = _engine_metadata(tp4, rank)
        assert md.enable_context_mla_with_cached_kv is True
        features = md.runtime_features
        assert (features.chunked_prefill, features.cache_reuse, features.chunk_size) == (
            True,
            False,
            _CHUNK,
        )
        assert features.has_speculative_draft_tokens is False
        assert (md.max_num_requests, md.max_num_tokens, md.max_num_sequences) == (
            _MAX_BATCH,
            _CHUNK,
            _MAX_BATCH,
        )
        assert md.mapping.tp_rank == rank and md.mapping.enable_attention_dp is False
        assert md.sparse_metadata_params.q_split_threshold == _Q_SPLIT_THRESHOLD
        assert md.enable_flash_mla is False
        assert _engine_metadata(tp4_32k, rank).enable_context_mla_with_cached_kv is False


# ------------------------------------------------------- (2) the query split across the TP ranks

# Each step size is planned with the context rows composed four ways.
_SPLIT_STEPS = (8191, 8192, 32768, 32769)
# Per-rank query counts of a split step, in rank order; None below the threshold.
_SPLIT_SIZES = {
    8191: None,
    8192: [2048] * 4,
    32753: [8188, 8188, 8188, 8189],
    32768: [8192] * 4,
    32769: [8192, 8192, 8192, 8193],
}
# Decode rows behind the context rows, by KV length: one at max_seq_len, the rest spread.
_DECODE_KV = (
    1048576,
    1048575,
    524288,
    131073,
    65536,
    32769,
    32768,
    8192,
    2049,
    2048,
    2047,
    1024,
    33,
    32,
    2,
)


def _compositions(tc: int) -> dict:
    """The context rows of a ``tc``-token step: (new tokens, cached prefixes, decode KV lengths).
    One fresh prompt; one chunk over the longest cached prefix a 1048576-token request leaves it
    (1015808 below 32769 tokens); four requests over different prefixes, cut by the rank
    boundaries; one chunk with 15 decode rows behind it."""
    last = min(_LAST_CHUNK_PREFIX, _MAX_SEQ_LEN - tc)
    a, b, c = tc * 3 // 8, tc // 8, tc // 4
    return {
        "fresh": ([tc], [0], ()),
        "last_chunk": ([tc], [last], ()),
        "prefixes": ([a, b, c, tc - a - b - c], [131072, 0, 524288, 917504], ()),
        "behind_decode": ([tc], [262144], _DECODE_KV),
    }


def _check_plan(md, q_lens, cached, rank: int) -> tuple[int, int]:
    """One rank's plan against the rules it must keep; returns the query rows it covers."""
    tc = sum(q_lens)
    sizes = _SPLIT_SIZES[tc]
    if sizes is None:
        assert md.indexer_prefill_q_split is None
        begin, end = 0, tc
    else:
        begin = sum(sizes[:rank])
        end = begin + sizes[rank]
        assert md.indexer_prefill_q_split == (begin, end, sizes)

    chunks = md.indexer_prefill_chunks
    assert chunks, "a context step always plans its indexer chunks under chunked MLA prefill"
    assert chunks[0].token_start == begin and chunks[-1].token_end == end
    assert all(a.token_end == b.token_start for a, b in zip(chunks, chunks[1:]))

    q_starts = [sum(q_lens[:i]) for i in range(len(q_lens))]
    kv_lens = [c + q for c, q in zip(cached, q_lens)]
    for chunk in chunks:
        rows = chunk.token_end - chunk.token_start
        assert rows > 0
        assert chunk.cu_seqlen_ks.numel() == chunk.cu_seqlen_ke.numel() == rows
        first = max(i for i, start in enumerate(q_starts) if start <= chunk.token_start)
        last = max(i for i, start in enumerate(q_starts) if start < chunk.token_end)
        assert chunk.k_token_start == sum(kv_lens[:first])
        if first == last:
            # One request, whole or cut: its queries see its cached prefix and its own rows up to
            # the chunk's last one.
            seen = cached[first] + chunk.token_end - q_starts[first]
            assert chunk.k_token_end - chunk.k_token_start == seen
            assert int(chunk.cu_seqlen_ks.max()) == 0
        else:
            # Whole requests packed together: the keys of each in turn.
            assert chunk.token_start == q_starts[first]
            assert chunk.token_end == q_starts[last] + q_lens[last]
            assert chunk.k_token_end - chunk.k_token_start == sum(kv_lens[first : last + 1])
        assert int(chunk.cu_seqlen_ke[-1]) == chunk.k_token_end - chunk.k_token_start
        assert bool((chunk.cu_seqlen_ke > chunk.cu_seqlen_ks).all())
    return begin, end


@pytest.mark.parametrize("tc", _SPLIT_STEPS)
def test_query_split_tiles_the_step(tp4, tc):
    """The engine's own planning on each rank's host metadata, at 8191, 8192, 32768 and 32769
    context query tokens, each composed as one fresh prompt, one chunk over a cached prefix of
    1015808 (the last chunk of a 1048576-token request; 1015807 at 32769 tokens), four requests
    over different cached prefixes and one chunk with 15 decode rows behind it. Below 8192 tokens
    there is no split: every rank plans every row. From 8192 on each rank gets
    [tc r / 4, tc (r + 1) / 4) and the four ranges tile [0, tc) exactly once, in rank order; each
    rank's chunks tile its range in order, a request cut by a rank boundary is a chunk of its own
    over its cached prefix and its rows up to the chunk's end, and whole requests are grouped by
    the packed-logits budget. Decode rows neither move the split nor the chunks. A step of 32769
    context tokens exceeds max_num_tokens 32768 and is never scheduled; it is planned here for the
    uneven split's sake (32753 tokens behind 15 decode rows is the reachable one)."""
    for name, (q_lens, cached, gen_kv) in _compositions(tc).items():
        covered = []
        for rank in range(_RANKS):
            md = _plan(tp4, rank, q_lens, cached, gen_kv)
            covered.append(_check_plan(md, q_lens, cached, rank))
            if gen_kv:
                alone = _plan(tp4, rank, q_lens, cached)
                assert md.indexer_prefill_q_split == alone.indexer_prefill_q_split
                assert _spans(md) == _spans(alone)
        if _SPLIT_SIZES[tc] is None:
            assert covered == [(0, tc)] * _RANKS, name
        else:
            assert covered[0][0] == 0 and covered[-1][1] == tc, name
            assert all(a[1] == b[0] for a, b in zip(covered, covered[1:])), name


def test_query_split_contract(tp4, tp4_32k):
    """The plans the target is checked against, exactly. ``indexer_prefill_q_split`` is
    (q_begin, q_end, q_sizes) and each chunk (token_start, token_end, k_token_start, k_token_end):

    * one fresh 32768-token prompt: rank r scores rows [8192 r, 8192 r + 8192) against keys
      [0, 8192 r + 8192);
    * the last chunk of a 1048576-token request (32768 rows over 1015808 cached): rank r scores
      rows [8192 r, 8192 r + 8192) against keys [0, 1015808 + 8192 r + 8192), the first row's
      window ending at 1015808 + 8192 r + 1;
    * 32769 rows over 1015807 cached: sizes 8192, 8192, 8192, 8193;
    * 32753 rows behind 15 decode rows (a full 32768-token step): sizes 8188, 8188, 8188, 8189;
    * four requests (12288, 4096, 8192, 8192 rows over 131072, 0, 524288, 917504 cached): rank 0
      the first request's rows [0, 8192), rank 1 its rows [8192, 12288) and the second request
      whole, ranks 2 and 3 the third and the fourth whole;
    * 8191 rows over 1015808 cached, below the threshold: no split, one chunk of every row on
      every rank;
    * and the delivered 32K engine, with no context MLA over cached KV: no split and no chunks at
      32768 rows.
    """
    for rank in range(_RANKS):
        lo, hi = 8192 * rank, 8192 * (rank + 1)

        md = _plan(tp4, rank, [_CHUNK], [0])
        assert md.indexer_prefill_q_split == (lo, hi, [8192] * 4)
        assert _spans(md) == [(lo, hi, 0, hi)]

        md = _plan(tp4, rank, [_CHUNK], [_LAST_CHUNK_PREFIX])
        assert md.indexer_prefill_q_split == (lo, hi, [8192] * 4)
        assert _spans(md) == [(lo, hi, 0, _LAST_CHUNK_PREFIX + hi)]
        (chunk,) = md.indexer_prefill_chunks
        assert int(chunk.cu_seqlen_ke[0]) == _LAST_CHUNK_PREFIX + lo + 1
        assert int(chunk.cu_seqlen_ke[-1]) == _LAST_CHUNK_PREFIX + hi

        md = _plan(tp4, rank, [_CHUNK + 1], [_LAST_CHUNK_PREFIX - 1])
        sizes = [8192, 8192, 8192, 8193]
        assert md.indexer_prefill_q_split == (lo, lo + sizes[rank], sizes)
        assert _spans(md) == [(lo, lo + sizes[rank], 0, _LAST_CHUNK_PREFIX - 1 + lo + sizes[rank])]

        md = _plan(tp4, rank, [_CHUNK - 15], [0], _DECODE_KV)
        sizes = [8188, 8188, 8188, 8189]
        begin = 8188 * rank
        assert md.num_ctx_tokens + md.num_generations == _CHUNK
        assert md.indexer_prefill_q_split == (begin, begin + sizes[rank], sizes)
        assert _spans(md) == [(begin, begin + sizes[rank], 0, begin + sizes[rank])]

        md = _plan(tp4, rank, [8191], [_LAST_CHUNK_PREFIX])
        assert md.indexer_prefill_q_split is None
        assert _spans(md) == [(0, 8191, 0, _LAST_CHUNK_PREFIX + 8191)]

        md = _plan(tp4_32k, rank, [_CHUNK], [0])
        assert md.indexer_prefill_q_split is None and md.indexer_prefill_chunks is None

    q_lens, cached = [12288, 4096, 8192, 8192], [131072, 0, 524288, 917504]
    plans = [_spans(_plan(tp4, rank, q_lens, cached)) for rank in range(_RANKS)]
    assert plans == [
        [(0, 8192, 0, 139264)],
        [(8192, 12288, 0, 143360), (12288, 16384, 143360, 147456)],
        [(16384, 24576, 147456, 679936)],
        [(24576, 32768, 679936, 1605632)],
    ]


def test_query_split_follows_the_attention_mapping(tp4):
    """The split reads the attention mapping alone: tp_size 4 and tp_rank, attention DP off. The
    MoE split never reaches it, and with attention DP on (one rank's own requests) nothing is
    split."""
    for rank in range(_RANKS):
        mapping = _rank_mapping(tp4.mapping, rank)
        bounds = dsa_indexer._split_prefill_queries(_CHUNK, mapping, _Q_SPLIT_THRESHOLD)
        assert bounds == (8192 * rank, 8192 * (rank + 1), [8192] * 4)
        adp = SimpleNamespace(enable_attention_dp=True, tp_size=4, tp_rank=rank)
        assert dsa_indexer._split_prefill_queries(_CHUNK, adp, _Q_SPLIT_THRESHOLD) == (
            0,
            _CHUNK,
            None,
        )
    assert dsa_indexer._split_prefill_queries(_CHUNK, tp4.mapping, -1) == (0, _CHUNK, None)


# The steps whose indexer chunk packs past max_seq_len keys: (new, cached) tokens per context
# request. Two 5-token last chunks over 983040-token prefixes (10 context rows, no split: one
# chunk of 1966090 keys); a 50-token last chunk over a 1048480-token prefix beside a fresh
# 200-token prompt (one chunk of 1048730 keys).
_PACKED_STEPS = {
    "two-tails-over-983040": ([5, 5], [983040, 983040]),
    "tail-beside-a-prompt": ([50, 200], [1048480, 0]),
}


@pytest.mark.parametrize("name", list(_PACKED_STEPS))
def test_packed_chunks_are_scored_request_by_request(tp4, name):
    """`plan_mla_chunked_prefill_chunks` packs consecutive whole context requests into one indexer
    chunk while sum(q) x sum(kv) stays within 1 << 28, and the chunk's keys are every packed
    request's whole KV: past max_seq_len keys here, wider than the key scorer's and the prefill
    top-k's cells. `_indexer_blocks` cuts it at its request boundaries: each request's rows
    against its own keys only, its windows re-based to its first key, every block within
    max_seq_len keys and every window inside its block."""
    q_lens, cached = _PACKED_STEPS[name]
    md = _plan(tp4, 0, q_lens, cached)
    kv = [c + q for c, q in zip(cached, q_lens)]
    q_at = [0, *itertools.accumulate(q_lens)]
    kv_at = [0, *itertools.accumulate(kv)]
    assert md.indexer_prefill_q_split is None
    assert _spans(md) == [(0, q_at[-1], 0, kv_at[-1])]
    assert kv_at[-1] > _MAX_SEQ_LEN and q_at[-1] * kv_at[-1] <= 1 << 28
    (chunk,) = md.indexer_prefill_chunks
    blocks = _target_blocks(md, q_lens)
    assert [(b.token_start, b.token_end, b.k_token_start, b.k_token_end) for b in blocks] == [
        (q_at[r], q_at[r + 1], kv_at[r], kv_at[r + 1]) for r in range(len(q_lens))
    ]
    for r, block in enumerate(blocks):
        rows = slice(q_at[r], q_at[r + 1])
        n_k = block.k_token_end - block.k_token_start
        assert n_k == kv[r] <= _MAX_SEQ_LEN
        assert torch.equal(block.cu_seqlen_ks, chunk.cu_seqlen_ks[rows] - kv_at[r])
        assert torch.equal(block.cu_seqlen_ke, chunk.cu_seqlen_ke[rows] - kv_at[r])
        assert int(block.cu_seqlen_ks.min()) >= 0 and int(block.cu_seqlen_ke.max()) <= n_k


def test_chunks_within_max_seq_len_stay_whole_under_the_split(tp4):
    """A chunk of at most max_seq_len keys is one scoring block, unchanged, on every rank of the
    query split: the envelope gate's prompt's last chunk over its 1015808-token prefix (8192 rows a
    rank); and the two 983040-prefix tails beside a fresh 8182-token prompt, 8192 context rows
    split 2048 a rank, where rank 0's share packs both tails into a 1966090-key chunk -- cut at
    the tails -- and then the fresh prompt's first 2038 rows, a block of their own like every
    other rank's slice of it."""
    for rank in range(_RANKS):
        md = _plan(tp4, rank, [_CHUNK], [_LAST_CHUNK_PREFIX])
        assert md.indexer_prefill_q_split == (8192 * rank, 8192 * (rank + 1), [8192] * _RANKS)
        blocks = _target_blocks(md, [_CHUNK])
        assert len(blocks) == len(md.indexer_prefill_chunks)
        assert all(b is c for b, c in zip(blocks, md.indexer_prefill_chunks))

    q_lens, cached = [5, 5, 8182], [983040, 983040, 0]
    for rank in range(_RANKS):
        md = _plan(tp4, rank, q_lens, cached)
        q_begin, q_end = 2048 * rank, 2048 * (rank + 1)
        assert md.indexer_prefill_q_split == (q_begin, q_end, [2048] * _RANKS)
        blocks = _target_blocks(md, q_lens)
        spans = [(b.token_start, b.token_end, b.k_token_start, b.k_token_end) for b in blocks]
        if rank == 0:
            assert _spans(md) == [(0, 10, 0, 1966090), (10, 2048, 1966090, 1968128)]
            assert spans == [(0, 5, 0, 983045), (5, 10, 983045, 1966090), *_spans(md)[1:]]
            assert blocks[2] is md.indexer_prefill_chunks[1]
        else:
            assert spans == _spans(md) == [(q_begin, q_end, 1966090, 1966090 + q_end - 10)]
        assert all(b.k_token_end - b.k_token_start <= _MAX_SEQ_LEN for b in blocks)


# ------------------------------------------------------- (3) KV estimate, warmup


def _model_engine(resolved) -> SimpleNamespace:
    """The fields of ``PyTorchModelEngine`` the KV-cache creator reads, for this engine."""
    graphs = resolved.llm_args.cuda_graph_config
    return SimpleNamespace(
        model=SimpleNamespace(model_config=resolved.model_config),
        attn_runtime_features=_runtime_features(resolved),
        batch_size=resolved.max_batch_size,
        _max_cuda_graph_batch_size=max(graphs.batch_sizes),
        max_seq_len=resolved.max_seq_len,
        use_mrope=False,
        mm_encoder_cache=None,
        sparse_attention_config=resolved.llm_args.sparse_attention_config,
    )


def _kv_creator(resolved) -> KvCacheCreator:
    """``KvCacheCreator`` as ``create_py_executor`` builds it for this engine (no speculation, one
    beam, no connector, not disaggregated)."""
    llm_args = resolved.llm_args
    engine = _model_engine(resolved)
    kv_cache_config = copy.deepcopy(llm_args.kv_cache_config)
    return KvCacheCreator(
        model_engine=engine,
        draft_model_engine=None,
        mapping=resolved.mapping,
        net_max_seq_len=resolved.max_seq_len,
        kv_connector_manager=None,
        max_num_tokens=resolved.max_num_tokens,
        max_beam_width=1,
        tokens_per_block=kv_cache_config.tokens_per_block,
        max_seq_len=resolved.max_seq_len,
        max_batch_size=resolved.max_batch_size,
        kv_cache_config=kv_cache_config,
        llm_args=llm_args,
        speculative_config=None,
        sparse_attention_config=engine.sparse_attention_config,
        profiling_stage_data=None,
        is_disagg=False,
    )


def test_kv_estimation_runs_one_full_step(tp4, tp4_fp8, tp4_quickstart, tp4_32k):
    """The estimation pass sends one fresh 32768-token prompt at 1M (the delivered 32K engine: a
    32767-token and a 1-token prompt), with no chunk-aware MLA profile, which is skipped for any
    sparse algorithm but skip_softmax. Its throwaway KV pool (``MistralLarge4CacheManager``, a
    KVCacheManagerV2) is sized for the CUDA-graph warmup's longest decode row:
    ceil((1048576 + 1) / 32) + 15 pages, 1049088 tokens, where 32K needed 33280. The same in
    every KV dtype spelling."""

    def estimation(resolved):
        creator = _kv_creator(resolved)
        assert creator._kv_cache_manager_cls is MistralLarge4CacheManager
        tokens = creator._get_token_num_for_estimation()
        assert creator._mla_chunked_profile_length is None
        assert creator._get_mla_chunked_profile_length(resolved.max_seq_len - 1) is None
        return tokens, [len(request.input_token_ids) for request in creator._dummy_reqs]

    tokens, prompts = estimation(tp4)
    assert prompts == [_CHUNK]
    assert tokens == (math.ceil((_MAX_SEQ_LEN + 1) / _PAGE) + _MAX_BATCH - 1) * _PAGE == 1_049_088
    assert estimation(tp4_fp8) == estimation(tp4_quickstart) == (tokens, prompts)
    assert estimation(tp4_32k) == (33_280, [_CHUNK - 1, 1])


def test_kv_capacity_reserves_nothing_and_caps_nothing(tp4):
    """The estimate, handed the budget a boot at this config on the fp8_ds_mla cache measured
    (2136192 tokens of 47136 B per rank, 93.78 GiB), on sm_103:

    * the DeepSeek-V4 indexer-logits reserve applies to algorithm deepseek_v4 only, and this is
      dsa (it would set aside 8 GiB: min(2^31, 32768 x 1048576) fp32 logits);
    * the context-MLA workspace: the backend declares w = 0 B/token, since DSA on sm_103 runs
      sparse MLA and the engine stages nothing for it (the dense fp8 path would stage 16 heads x
      (64 + 64 + 128) = 4096 B/token); chunked prefill bounds it; with w = 0 there is no
      kv_len_cap, no reserve and no attended-KV admission cap. The target's own context
      workspace (bf16 [2**19, 576], 576 MiB, `_ctx_workspace_rows` of this engine) is not the
      engine's to know of and needs no reserve: the target allocates it on its first context
      step, the estimation's own, and holds it, so the measured peak includes it
      (test_context_rows_path_per_step);
    * so the KV budget leaves the estimate unchanged and the pool is 2136192 tokens per rank,
      one 1048576-token request with a chunk (two before every rank held the vision tower);
      the scheduler, with no cap, keeps a second context request beside the last chunk of a
      1M one.

    Were a staging rate declared here (the dense path's 4096 B/token), chunked prefill would not
    exempt it on sm_103 without a chunk-aware profile, and the cap it installed,
    budget / (47136 + 4096) = 1965403 tokens, would still admit a 1M request and a chunk."""
    creator = _kv_creator(tp4)
    model_config, mapping = tp4.model_config, tp4.mapping
    kv_cache_config = creator._kv_cache_config
    fraction = kv_cache_config.free_gpu_memory_fraction
    budget = _BOOT_POOL_TOKENS * _TOKEN_BYTES
    free = math.ceil(budget / fraction)
    total = free + 160 * 2**30

    with pytest.MonkeyPatch.context() as mp:
        _pin_sm103(mp)
        mp.setattr(torch.cuda, "empty_cache", lambda: None)
        mp.setattr(torch.cuda, "reset_peak_memory_stats", lambda *args, **kwargs: None)
        mp.setattr(torch.cuda, "mem_get_info", lambda *args, **kwargs: (free, total))
        mp.setattr(
            torch.cuda,
            "memory_stats",
            lambda *args, **kwargs: {"allocated_bytes.all.current": total - free},
        )
        w = _util.get_attention_workspace_bytes_per_token(model_config, mapping)
        bounded = _util.get_attention_workspace_is_chunked_prefill_bounded(model_config)
        assert creator._get_kv_size_per_token().slope == _TOKEN_BYTES
        creator.configure_kv_cache_capacity(None)

    assert model_config.sparse_attention_config.algorithm == "dsa"
    assert getattr(creator._sparse_attention_config, "algorithm", None) != "deepseek_v4"
    assert get_indexer_mqa_logits_workspace_bytes(_CHUNK, _MAX_SEQ_LEN) == 8 * 2**30

    staged = dict(
        num_attn_heads=model_config.pretrained_config.num_attention_heads // mapping.tp_size,
        qk_rope_head_dim=64,
        qk_nope_head_dim=64,
        v_head_dim=128,
        fp8_context_mla=True,
        separate_q_and_kv_input=True,
    )
    assert thop.get_context_mla_workspace_bytes_per_token(**staged, sparse_mla=False) == 4096
    assert thop.get_context_mla_workspace_bytes_per_token(**staged, sparse_mla=True) == 0
    assert w == 0 and bounded is True
    assert (
        _util.get_mla_context_workspace_kv_len_cap(
            kv_cache_config,
            _MAX_BATCH,
            _CHUNK,
            _MAX_SEQ_LEN,
            True,
            bounded,
            chunked_workspace_profiled=False,
            require_chunked_workspace_profile=False,
        )
        is None
    )
    assert creator._fp8_ctx_mla_kv_len_cap is None
    pool_bytes = kv_cache_config.max_gpu_total_bytes
    assert pool_bytes == int(free * fraction)
    pool = pool_bytes // _TOKEN_BYTES
    assert pool == _BOOT_POOL_TOKENS
    assert pool // (_MAX_SEQ_LEN + _CHUNK) == 1

    executor = SimpleNamespace(
        is_warmup=False,
        max_seq_len=tp4.max_seq_len,
        kv_cache_manager=SimpleNamespace(fp8_ctx_mla_kv_len_cap=creator._fp8_ctx_mla_kv_len_cap),
        _context_attended_kv_len=PyExecutor._context_attended_kv_len,
    )
    for name in ("_get_ctx_mla_kv_len_cap", "_warn_if_ctx_mla_kv_len_cap_degenerate"):
        setattr(executor, name, types.MethodType(getattr(PyExecutor, name), executor))
    last_chunk = SimpleNamespace(
        context_current_position=_LAST_CHUNK_PREFIX,
        is_first_context_chunk=False,
        estimated_reusable_tokens=0,
        context_chunk_size=_CHUNK,
        orig_prompt_len=_MAX_SEQ_LEN,
    )
    first_chunk = SimpleNamespace(
        context_current_position=0,
        is_first_context_chunk=True,
        estimated_reusable_tokens=0,
        context_chunk_size=_CHUNK,
        orig_prompt_len=_MAX_SEQ_LEN,
    )
    scheduled = PyExecutor._cap_context_by_total_kv_len(executor, [last_chunk, first_chunk])
    assert scheduled == [last_chunk, first_chunk]
    assert executor._ctx_mla_kv_len_cap is None

    reserve, cap = _util.get_mla_context_workspace_reserve(
        budget, _TOKEN_BYTES, 4096, _MAX_BATCH * _MAX_SEQ_LEN
    )
    assert (
        _util.get_mla_context_workspace_kv_len_cap(
            kv_cache_config,
            _MAX_BATCH,
            _CHUNK,
            _MAX_SEQ_LEN,
            True,
            True,
            chunked_workspace_profiled=False,
            require_chunked_workspace_profile=True,
        )
        == _MAX_BATCH * _MAX_SEQ_LEN
    )
    assert cap == budget // (_TOKEN_BYTES + 4096) == 1_965_403
    assert cap >= _MAX_SEQ_LEN + _CHUNK


def test_warmup_runs_one_32768_token_prompt(tp4, tp4_32k):
    """The warmup's shapes at 1M, agreed by the four ranks: 1-token context and decode, the
    largest context step of 32768 tokens, 16 decode rows, a 2-token context, as at 32K. A prompt
    may now be up to 1048575 tokens, so the largest step is one 32768-token prompt (at 32K: a
    32767-token and a 1-token one). It is a fresh prompt: from 8192 context tokens on, the warmup's
    and the estimation's largest step run the query split on every rank."""

    def max_context_prompts(resolved):
        config = _runner_config(resolved)
        group = _TpGroup()
        factory = _dummy_factory(resolved)

        def rank_warmup(rank):
            resources = _Resources(_KvPool(_BOOT_POOL_TOKENS, config.max_seq_len))
            warmup = DecoderWarmup.__new__(DecoderWarmup)
            warmup._config = config
            warmup.kv_cache_manager_key = ResourceManagerType.KV_CACHE_MANAGER
            agreement = WarmupAgreement(
                _RankDist(group, rank), _rank_mapping(resolved.mapping, rank)
            )
            shapes = agreement.agree_shapes(warmup._get_full_general_warmup_requests(resources))
            max_context = max(tokens for tokens, gen in shapes if gen == 0)
            batch = factory.create_warmup_request(resources, max_context, 0)
            return shapes, [request.token_num for request in batch.context_requests]

        return group.run(rank_warmup)

    for shapes, prompts in max_context_prompts(tp4):
        assert shapes == [(1, 0), (1, 1), (_CHUNK, 0), (_MAX_BATCH, _MAX_BATCH), (2, 0)]
        assert prompts == [_CHUNK]
    for shapes, prompts in max_context_prompts(tp4_32k):
        assert prompts == [_CHUNK - 1, 1]

    for rank in range(_RANKS):
        md = _plan(tp4, rank, [_CHUNK], [0])
        assert md.indexer_prefill_q_split == (8192 * rank, 8192 * (rank + 1), [8192] * 4)


def test_context_rows_path_per_step(tp4, tp4_fp8, tp4_quickstart, tp4_32k):
    """The target's held context workspace and its context-row path
    (``PrefillTarget._ctx_workspace``) on the engine's own host metadata for rank 0 at this
    config, in every KV dtype spelling. The held workspace's rows are `_ctx_workspace_rows` of the
    engine's quantities, off the runtime features the executor builds from the LLM-API options
    (chunked prefill on, block reuse off): under chunked prefill a step can attend up to 16 cached
    prefixes of max_seq_len, so 2**19 (`_CTX_WORKSPACE_ROWS`), where the delivered 32K engine
    (neither chunked prefill nor block reuse) holds twice max_num_tokens, 65536. The path reads a
    step's context KV as cached + new over its context requests (``_ctx_kv_offsets`` off the
    engine's indptrs, the host ``ctx_kv_indptr[num_contexts]``), never the engine's
    ``num_ctx_kv_tokens``, which counts the new tokens only:

    * the KV-capacity estimate's step, one fresh 32768-token prompt (N = 32768, split 8192 per
      rank), takes the workspace path, and the first context step allocates the target's bf16
      [2**19, 576] workspace (576 MiB), held from then on: the estimate measures its peak with the
      workspace in it, so it needs no reserve (w = 0, above);
    * a 1048576-token prompt's chunks, each a query-split step with the indexer running: through
      the 16th (N up to 2**19) on the workspace path, keeping layer 35's rows beside the step's
      through the 8th (2N up to 2**19), from the 17th on the decode path in 4 sub-batches of
      8192, the most FlashMLA's decode scheduler is driven at (it exits the process past 11622);
    * the envelope gate's 1048559-token prompt: its last chunk is 32751 rows behind 1015808 cached
      keys, in sub-batches of 8192 x 3 and 8175;
    * a step packing 16384 rows behind 1015808 with a fresh 16369-token prompt and 15 decode rows
      behind them (32768 rows) takes the decode path; the decode rows are not context rows.

    The workspace is never reallocated after the estimate's step.
    """
    rows_b = tp4_modeling._CTX_WORKSPACE_ROWS
    assert (rows_b, tp4_modeling._CTX_DECODE_ROWS) == (2**19, 8192)
    assert rows_b * 576 * torch.bfloat16.itemsize == 576 * 2**20
    sizes_32k = _target_sizes(tp4_32k)
    features = sizes_32k.runtime_features
    assert (features.chunked_prefill, features.cache_reuse) == (False, False)
    assert tp4_modeling._most_ctx_kv(sizes_32k) == _CHUNK
    assert tp4_modeling._ctx_workspace_rows(sizes_32k) == 2 * _CHUNK
    cpu = torch.device("cpu")
    for resolved in (tp4, tp4_fp8, tp4_quickstart):
        creator = _kv_creator(resolved)
        creator._get_token_num_for_estimation()
        assert [len(request.input_token_ids) for request in creator._dummy_reqs] == [_CHUNK]
        sizes = _target_sizes(resolved)
        llm_args = resolved.llm_args
        features = sizes.runtime_features
        assert (features.chunked_prefill, features.cache_reuse) == (
            llm_args.enable_chunked_prefill,
            llm_args.kv_cache_config.enable_block_reuse,
        )
        assert (features.chunked_prefill, features.cache_reuse) == (True, False)
        assert tp4_modeling._most_ctx_kv(sizes) == _MAX_BATCH * _MAX_SEQ_LEN
        assert tp4_modeling._ctx_workspace_rows(sizes) == rows_b
        target = tp4_modeling.PrefillTarget.__new__(tp4_modeling.PrefillTarget)
        target.core = SimpleNamespace(
            lat_dim=576, ctx_workspace_rows=tp4_modeling._ctx_workspace_rows(sizes)
        )

        def path(q_lens, cached, gen_kv=()):
            md = _plan(resolved, 0, q_lens, cached, gen_kv)
            step = dict(
                num_contexts=md.num_contexts,
                num_ctx_tokens=md.num_ctx_tokens,
                **_target_ctx_kv_offsets(md, q_lens),
            )
            n_kv = int(md.host_ctx_kv_indptr[md.num_contexts])
            assert n_kv == sum(q_lens) + sum(cached)
            assert step["ctx_cached_tokens"] + step["num_ctx_tokens"] == n_kv
            return md, n_kv, target._ctx_workspace(step, md.num_ctx_tokens, cpu)

        md, n_kv, ctx = path([_CHUNK], [0])
        held = target._ws_buf
        assert held.shape == (rows_b, 576) and held.dtype == torch.bfloat16
        assert ctx.ws_buf.shape == (_CHUNK, 576) and ctx.ws_buf.data_ptr() == held.data_ptr()
        assert ctx.kept.data_ptr() == held[_CHUNK].data_ptr()
        assert md.indexer_prefill_q_split == (0, 8192, [8192] * 4)

        for chunk in (2, 8, 9, 16, 17, 32):
            md, n_kv, ctx = path([_CHUNK], [(chunk - 1) * _CHUNK])
            assert n_kv == chunk * _CHUNK
            assert md.indexer_prefill_q_split == (0, 8192, [8192] * 4)
            assert not md.skip_indexer_for_ctx_reqs
            if chunk <= 16:
                assert ctx.ws_buf.shape[0] == n_kv and ctx.ws_buf.data_ptr() == held.data_ptr()
                assert (ctx.kept is not None) == (chunk <= 8)
            else:
                assert ctx.ws_buf is None
                assert ctx.sub_batches == [(lo, lo + 8192) for lo in range(0, _CHUNK, 8192)]

        last = _ENVELOPE_PROMPT - _LAST_CHUNK_PREFIX
        assert last == 32751
        md, n_kv, ctx = path([last], [_LAST_CHUNK_PREFIX])
        assert n_kv == _ENVELOPE_PROMPT and ctx.ws_buf is None
        assert [hi - lo for lo, hi in ctx.sub_batches] == [8192, 8192, 8192, 8175]
        assert sorted(ctx.sched) == [8175, 8192]

        md, n_kv, ctx = path([16384, 16369], [_LAST_CHUNK_PREFIX, 0], _DECODE_KV)
        assert md.num_tokens == _CHUNK and n_kv == _LAST_CHUNK_PREFIX + 32753 > rows_b
        assert ctx.ws_buf is None
        assert ctx.sub_batches[-1] == (24576, 32753)
        assert target._ws_buf is held


def test_cuda_graph_warmup_decode_rows(tp4):
    """The decode graphs (batches 1, 2, 4, 8, 16) are captured twice, the DSA config asking for
    separate short and long graphs at its index_topk: once at max_seq_len and once at 2047 (2048
    less the one token a decode step adds). Each capture batch is one longest dummy and batch - 1
    one-token rows: the longest is 1048575 tokens (KV length 1048575, 32768 pages, its decode
    token at position 1048574), or 2046 on the short pass. That holds in the estimation's
    1049088-token pool and in the served one."""
    sparse = tp4.model_config.sparse_attention_config
    assert isinstance(sparse, SeqLenAwareSparseAttentionConfig)
    assert sparse.needs_separate_short_long_cuda_graphs() is True
    assert sparse.seq_len_threshold == _TOPK
    capture_lengths = [_MAX_SEQ_LEN, sparse.seq_len_threshold - 1]

    graphs = sorted(tp4.llm_args.cuda_graph_config.batch_sizes, reverse=True)
    assert graphs == [16, 8, 4, 2, 1]
    factory = _dummy_factory(tp4)
    for pool_tokens in (1_049_088, _BOOT_POOL_TOKENS):
        for batch_size in graphs:
            for max_seq_len in capture_lengths:
                resources = _Resources(_KvPool(pool_tokens, tp4.max_seq_len))
                batch = factory.create_cuda_graph_warmup_request(
                    resources, batch_size, 0, max_seq_len
                )
                rows = [request.token_num for request in batch.generation_requests]
                assert rows == [max_seq_len - 1] + [1] * (batch_size - 1)
                assert all(request.is_gen for request in batch.generation_requests)
                assert len(batch.context_requests) == 0
    assert math.ceil((_MAX_SEQ_LEN - 1) / _PAGE) == 32768


# ------------------------------------------------------- (4) buffers sized by max_seq_len


def _indexer_buffers(resolved, width: int) -> tuple[dict, dict]:
    """The DSA metadata's indexer buffers as ``create_buffers_for_indexer`` allocates them for a
    KV manager ``width`` pages wide: name -> (shape, dtype), the device ones recorded on the meta
    device, the host ones as allocated. ``num_sms`` sizes only the DeepGEMM schedule buffers."""
    params = resolved.model_config.sparse_attention_config.to_sparse_metadata_params(
        pretrained_config=resolved.model_config.pretrained_config
    )
    device = {}

    def get_empty(buffers, tensor_shape, dtype, cache_name, capture_graph=False):
        device[cache_name] = (tuple(tensor_shape), dtype)
        return torch.empty(tensor_shape, dtype=dtype, device="meta")

    md = SimpleNamespace(
        sparse_metadata_params=params,
        cuda_graph_buffers=None,
        get_empty=get_empty,
        max_num_sequences=resolved.max_batch_size,
        max_num_tokens=resolved.max_num_tokens,
        kv_cache_manager=SimpleNamespace(max_blocks_per_seq=width),
        draft_kv_cache_manager=None,
        use_fp8_ds_mla=False,
        num_sparse_topk=params.max_sparse_topk,
        enable_indexer_skip=params.enable_indexer_skip,
        needs_gvr_prior=False,
        num_sms=1,
        _radix_rows_per_sequence=1,
        _draft_sizing_cap=0,
        enable_context_mla_with_cached_kv=True,
        enable_ragged_verification=False,
    )
    for name in (
        "_create_kv_lens_2d_buffer",
        "_create_radix_aux_buffers",
        "create_expanded_buffers",
    ):
        setattr(md, name, types.MethodType(getattr(DSAtrtllmAttentionMetadata, name), md))
    with pytest.MonkeyPatch.context() as mp:
        _host_only(mp)
        DSAtrtllmAttentionMetadata.create_buffers_for_indexer(md, capture_graph=False)
    host = {
        name: (tuple(value.shape), value.dtype)
        for name, value in vars(md).items()
        if name.startswith("host_") and isinstance(value, torch.Tensor)
    }
    return device, host


def test_metadata_buffers_sized_by_max_seq_len(tp4, tp4_32k):
    """What grows with max_seq_len per rank, on the engine's own allocation code:

    * the block tables, at KVCacheManagerV2's width ceil((max_seq_len + 1) / 32) padded to 4:
      32772 columns at 1M (1028 at 32K). The DSA metadata's indexer_k_cache_block_offsets,
      block_table and block_table_expanded are [16, 32772] int32, 2097408 B each, and the first and
      third have host copies of the same size; the latent pool's kv_cache_block_offsets
      (TrtllmAttentionMetadata) is [pools, 16, 2, 32772] int32, 4194816 B per pool;
    * nothing else of the indexer's: topk_indices_buffer and shared_topk_indices stay
      [32768, 2048] int32 (max_num_tokens), radix_aux_indices and radix_aux_logits
      [16, 10, 2048] (max_num_sequences), scheduler_metadata_buffer [num_sms + 1, 2];
    * the context gather mappings, built every context step over every context request's cached
      and new keys on every rank (the query split does not narrow them): the last chunk of a 1M
      request gathers 1048576 int64 offsets each for slot_mapping_fp8_fullkv and
      slot_mapping_scale_fullkv, 16 MiB together, plus their host staging; a step with no cached
      prefix reuses the per-step slot mappings."""
    width, width_32k = _block_table_width(_MAX_SEQ_LEN), _block_table_width(_CHUNK)
    assert (width, width_32k) == (32772, 1028)

    device, host = _indexer_buffers(tp4, width)
    device_32k, host_32k = _indexer_buffers(tp4_32k, width_32k)
    assert device.keys() == device_32k.keys() and host.keys() == host_32k.keys()
    grown = {name for name in device if device[name] != device_32k[name]}
    assert grown == {"indexer_k_cache_block_offsets", "block_table", "block_table_expanded"}
    for name in grown:
        assert device[name] == ((_MAX_BATCH, width), torch.int32)
        assert device_32k[name] == ((_MAX_BATCH, width_32k), torch.int32)
    assert _MAX_BATCH * width * 4 == 2_097_408
    grown_host = {name for name in host if host[name] != host_32k[name]}
    assert grown_host == {"host_indexer_k_cache_block_offsets", "host_block_table_expanded"}
    for name in grown_host:
        assert host[name] == ((_MAX_BATCH, width), torch.int32)
    assert _MAX_BATCH * 2 * width * 4 == 4_194_816

    assert device["topk_indices_buffer"] == ((_CHUNK, _TOPK), torch.int32)
    assert device["shared_topk_indices"] == ((_CHUNK, _TOPK), torch.int32)
    assert device["radix_aux_indices"] == ((_MAX_BATCH, 10, _TOPK), torch.int32)
    assert device["radix_aux_logits"] == ((_MAX_BATCH, 10, _TOPK), torch.float32)
    assert device["scheduler_metadata_buffer"] == ((2, 2), torch.int32)

    for rank in range(_RANKS):
        md = _plan(tp4, rank, [_CHUNK], [_LAST_CHUNK_PREFIX])
        for name in ("slot_mapping_fp8_fullkv", "slot_mapping_scale_fullkv"):
            mapping = getattr(md, name)
            assert (tuple(mapping.shape), mapping.dtype) == ((_MAX_SEQ_LEN,), torch.int64)
    assert 2 * _MAX_SEQ_LEN * 8 == 16 * 2**20
    fresh = _plan(tp4, 0, [_CHUNK], [0])
    assert fresh.slot_mapping_fp8_fullkv is fresh.slot_mapping_fp8
    assert fresh.slot_mapping_scale_fullkv is fresh.slot_mapping_scale
