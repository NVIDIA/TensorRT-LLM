# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Mistral-Large-4-1T-A52B-NVFP4 at tp4 and at tp8: the engine's configuration with the routed
experts tensor parallel, as a served engine reaches it, on four ranks of one GB300 node and on
eight ranks over two nodes of four.

No GPU and no weights. The checkpoint goes through the engine's own configuration path with the
LLM-API options of the tp4 and tp8 core units (four or eight ranks, tensor parallel throughout,
attention DP off, every routed expert cut along its width four or eight ways and none
expert-parallel, 32768 tokens and 16 requests over the whole engine, the latent KV cache at its
default -- the checkpoint's fp8_kv_cache, the fp8_ds_mla pool of MistralLarge4CacheManager):

* the LLM API front end, ``TorchLlmArgs`` and ``ModelLoader._update_from_hf_quant_config``
  (``llmapi/llm_utils.py``), which settle the key the accuracy harness looks its anchor up by;
* each rank, ``_load_config_and_create_checkpoint_loader`` (``py_executor_creator``) then the
  model loader's ``_load_and_validate_config``, which builds the ``ModelConfig`` and asks the
  modeling_v2 router for a target. The router's ``route`` is replaced by a recorder, so what it
  is handed is observed rather than judged.

The same deployment is reached from ``tensor_parallel_size`` alone: at tp4 as a gate command
carrying only ``--tp_size=4`` builds it, at tp8 as the boot gate's command (``--tp_size=8`` and
no MoE split) is parsed and turned into ``LLM`` keyword arguments by the tree's own
``examples/llm-api/quickstart_advanced.py`` (``parse_arguments`` and ``setup_llm``), plus the
``modeling_v2`` argument the harness adds. ``Mapping`` gives the MoE the whole TP width when
neither MoE split is stated.

No gate states ``gpus_per_node``: ``TorchLlmArgs`` takes it from ``torch.cuda.device_count()`` on
rank 0, which on a GB300 node of this cluster is 4. The count is set to 4 where the arguments are
built rather than read from the test host, which may see no device or a whole node. What the
engine reads of the MoE split and of the node layout on its way to a modeling_v2 target -- the
mapping's ranks and groups, the MoE token budget and load balancer, the autotuner and the FP4 MoE
runner's tuning, the weight-sharing qualification, the causal-LM shell's lm_head, and at tp8 the
MNNVL communicator splits and the device each rank selects -- and the largest step its warmup and
KV-capacity estimation send are pinned on the engine's own host code, the ranks simulated in one
process where they exchange anything.

The checkpoint is Mistral-native, as published (params.json, tekken.json, consolidated
safetensors), and is loaded with checkpoint_format mistral_large_4: the gates' LLM-API options
state it and the boot gate's command passes --checkpoint_format=mistral_large_4. The checkpoint
at tep4, at dep4 and at tep8 is checked to resolve and route as it did before this target. The
checkpoint is looked up under ``LLM_MODELS_ROOT`` (as ``Mistral-Large-4-1T-A52B-NVFP4`` or
``mistral-large-4-nvfp4``) by its params.json, and its tests are skipped where it is absent.
"""

from __future__ import annotations

import copy
import dataclasses
import importlib.util
import json
import math
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml
from test_common.llm_data import llm_models_root

from tensorrt_llm._ipc_utils import IpcMemory
from tensorrt_llm._torch._experimental.modeling_v2 import _router_index
from tensorrt_llm._torch._experimental.modeling_v2._router_index import NoTarget
from tensorrt_llm._torch._experimental.modeling_v2.models.mistral_large4 import routing
from tensorrt_llm._torch.attention.backends.sparse.mistral_large4 import MistralLarge4CacheManager
from tensorrt_llm._torch.autotuner import AutoTuner, DistributedTuningStrategy
from tensorrt_llm._torch.custom_ops.trtllm_gen_custom_ops import FP4BlockScaleMoERunner
from tensorrt_llm._torch.distributed import ops as distributed_ops
from tensorrt_llm._torch.distributed.communicator import Distributed, MPIDist
from tensorrt_llm._torch.distributed.mnnvl_memory import MnnvlMemory
from tensorrt_llm._torch.models.modeling_utils import DecoderModel, DecoderModelForCausalLM
from tensorrt_llm._torch.modules.linear import TensorParallelMode
from tensorrt_llm._torch.moe.fused_moe.moe_load_balancer import maybe_create_moe_load_balancer
from tensorrt_llm._torch.moe.fused_moe.routing import RoutingMethodType
from tensorrt_llm._torch.pyexecutor import py_executor_creator
from tensorrt_llm._torch.pyexecutor._util import CacheCost, KvCacheCreator, get_kv_cache_manager_cls
from tensorrt_llm._torch.pyexecutor.engine.runners.decoder.dummy_requests import DummyRequestFactory
from tensorrt_llm._torch.pyexecutor.engine.runners.decoder.warmup import (
    DecoderWarmup,
    WarmupAgreement,
)
from tensorrt_llm._torch.pyexecutor.model_loader import ModelLoader as RankModelLoader
from tensorrt_llm._torch.pyexecutor.resource_manager import ResourceManagerType
from tensorrt_llm._torch.pyexecutor.scheduler.adp_router import ADPRouter, DefaultADPRouter
from tensorrt_llm._torch.utils import ActType_TrtllmGen
from tensorrt_llm._torch.weight_sharing.post_transform_profiles import (
    PostTransformConfigIdentity,
    PostTransformQualificationReason,
    PostTransformRuntimeConfig,
)
from tensorrt_llm.executor import base_worker
from tensorrt_llm.executor.base_worker import BaseWorker
from tensorrt_llm.llmapi.llm_args import (
    AttentionDpConfig,
    LoadFormat,
    TorchLlmArgs,
    _ParallelConfig,
)
from tensorrt_llm.llmapi.llm_utils import ModelLoader as FrontEndModelLoader
from tensorrt_llm.mapping import Mapping, MpiTopology
from tensorrt_llm.models.modeling_utils import QuantAlgo
from tensorrt_llm.usage import usage_lib

pytestmark = pytest.mark.cpu_only

_SM103 = (10, 3)
# Devices per GB300 node on this cluster: one node holds the four ranks, two hold the eight.
_GPUS_PER_NODE = 4
# The gates' max_seq_len, which is also their max_num_tokens: 32768.
_MAX_SEQ_LEN = 32768

# The topologies. tp4 and tp8 are the core units' (spec-tp4/options/core.yaml,
# spec-tp8/options/core.yaml); a gate command carrying only --tp_size=4 states no MoE split and no
# attention DP, and quickstart_advanced.py spells that as the -1 sentinels. tep4 is where the
# tep4 target already serves, tep8 the same target on the same eight ranks as tp8
# (spec-tep8/options/core.yaml), dep4 the NVFP4 attention-DP target (spec-dep4/options/core.yaml).
_TP4 = dict(
    tensor_parallel_size=4,
    moe_tensor_parallel_size=4,
    moe_expert_parallel_size=1,
    enable_attention_dp=False,
)
_TP4_BARE = dict(tensor_parallel_size=4)
_TP4_QUICKSTART = dict(
    tensor_parallel_size=4,
    moe_expert_parallel_size=-1,
    moe_tensor_parallel_size=-1,
    moe_cluster_parallel_size=-1,
    enable_attention_dp=False,
)
_TP8 = dict(
    tensor_parallel_size=8,
    moe_tensor_parallel_size=8,
    moe_expert_parallel_size=1,
    enable_attention_dp=False,
)
_TEP4 = dict(tensor_parallel_size=4, moe_expert_parallel_size=4, enable_attention_dp=False)
_TEP8 = dict(tensor_parallel_size=8, moe_expert_parallel_size=8, enable_attention_dp=False)
_DEP4 = dict(tensor_parallel_size=4, moe_expert_parallel_size=4, enable_attention_dp=True)

# The tp8 boot gate's quickstart command after --model_dir (gates/geometry/geometry.core.tp8.json):
# the checkpoint's Mistral-native format, --tp_size=8 and no MoE split, the core unit's CUDA-graph
# capture list, task.yaml's runtime flags. No --kv_cache_dtype: the latent cache is the
# checkpoint's default (quickstart's "auto"), the fp8_ds_mla pool.
_GATE_ARGV = (
    "--checkpoint_format=mistral_large_4",
    "--max_tokens=2",
    "--tp_size=8",
    "--use_cuda_graph",
    "--cuda_graph_batch_sizes",
    "1",
    "2",
    "4",
    "8",
    "16",
    "--disable_kv_cache_reuse",
    "--max_seq_len=32768",
    "--max_batch_size=16",
    "--max_num_tokens=32768",
    "--cuda_graph_padding_enabled",
)
_QUICKSTART = Path(__file__).resolve().parents[7] / "examples/llm-api/quickstart_advanced.py"


def _options(topology: dict, max_seq_len: int) -> dict:
    """The LLM-API options of a Mistral Large 4 accuracy gate on ``topology``, the checkpoint in its
    Mistral-native format, max_num_tokens = max_seq_len, with the latent KV cache at its
    default: the checkpoint's fp8_kv_cache, the fp8_ds_mla pool."""
    return dict(
        checkpoint_format="mistral_large_4",
        **topology,
        max_batch_size=16,
        max_num_tokens=max_seq_len,
        max_seq_len=max_seq_len,
        enable_chunked_prefill=False,
        cuda_graph_config=dict(enable_padding=True, max_batch_size=16),
        kv_cache_config=dict(enable_block_reuse=False, free_gpu_memory_fraction=0.85),
    )


_NVFP4_NAMES = ("Mistral-Large-4-1T-A52B-NVFP4", "mistral-large-4-nvfp4")

# The GSM8K reference rows, and the key the harness reads them by.
_GSM8K_REFERENCES = (
    Path(__file__).resolve().parents[6] / "integration/defs/accuracy/references/gsm8k.yaml"
)
_MODEL_NAME = "mistralai/Mistral-Large-4-1T-A52B"

_TP4_TARGET = "ModelingV2MistralLarge4Nvfp4Sm103Tp4"
_OTHER_TARGETS = (
    "ModelingV2MistralLarge4Nvfp4Sm103Tep4",
    "ModelingV2MistralLarge4Nvfp4Sm103Dep4",
)

# By world size: the fixtures of the core unit as its gates spell it with the MoE split stated
# (core.yaml; at tp8 also the boot gate's command, which states none), and of the
# expert-parallel deployment on the same ranks.
_SPELLINGS = {4: ("tp4",), 8: ("tp8", "tp8_gate")}
_TEP = {4: "tep4", 8: "tep8"}
# By world size: the deployments on the same ranks whose KV cost the core unit's is compared with.
_KV_PEERS = {4: ("tep4", "dep4"), 8: ("tep8",)}
# By world size: the query heads a rank holds, and its width slice of every routed expert's 6144
# (w1/w3 rows, w2 columns) with w2's block-scale columns of 384, one per 16 inputs.
_HEADS = {4: 16, 8: 8}
_SLICE = {4: (1536, 96), 8: (768, 48)}
_WORLDS = pytest.mark.parametrize("world", [4, 8], ids=["tp4", "tp8"])


def _checkpoint_dir(names: tuple[str, ...]) -> Path:
    """The first of ``names`` under LLM_MODELS_ROOT holding a Mistral-native params.json."""
    root = llm_models_root()
    for name in names:
        if root is not None and (root / name / "params.json").is_file():
            return root / name
    pytest.skip(f"none of {names} is under LLM_MODELS_ROOT ({root})")


def _llm_args(devices: int = _GPUS_PER_NODE, **kwargs) -> TorchLlmArgs:
    """``TorchLlmArgs`` as the LLM constructor builds it on rank 0 of a node with ``devices``
    visible devices (outside Ray)."""
    with pytest.MonkeyPatch.context() as mp:
        mp.delenv("RAY_LOCAL_WORLD_SIZE", raising=False)
        mp.setattr(torch.cuda, "device_count", lambda: devices)
        return TorchLlmArgs(**kwargs)


def _gate_kwargs(checkpoint_dir: Path, argv: tuple[str, ...]) -> dict:
    """The ``LLM`` keyword arguments of a quickstart gate command: the tree's own
    quickstart_advanced.py parses ``argv`` and builds them in ``setup_llm``, and the gate harness
    adds ``modeling_v2``. Every KV-cache key of core.yaml has a quickstart flag, so the harness
    hands over no other."""
    if not _QUICKSTART.is_file():
        pytest.skip(f"{_QUICKSTART} is not in this checkout")
    spec = importlib.util.spec_from_file_location("quickstart_advanced", _QUICKSTART)
    quickstart = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(quickstart)
    built = {}

    def llm(**kwargs):
        built.update(kwargs)
        return SimpleNamespace()

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(sys, "argv", ["quickstart_advanced.py", f"--model_dir={checkpoint_dir}", *argv])
        args = quickstart.parse_arguments()
        mp.setattr(quickstart, "LLM", llm)
        quickstart.setup_llm(args)
    return dict(built, modeling_v2="require")


class _Routed:
    """What the recording `route` returns: no target is built here."""


def _resolve(checkpoint_dir: Path, llm_args: TorchLlmArgs) -> SimpleNamespace:
    """The configuration a served engine reaches for this checkpoint, front end then rank."""
    model = str(checkpoint_dir)
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
        checkpoint_loader=checkpoint_loader,
        prequantized=prequantized,
        model_config=model_config,
        mapping=mapping,
        routed=routed,
        max_num_tokens=max_num_tokens,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )


def _resolve_options(names: tuple[str, ...], topology: dict, max_seq_len: int) -> SimpleNamespace:
    path = _checkpoint_dir(names)
    options = _options(topology, max_seq_len)
    return _resolve(path, _llm_args(model=str(path), modeling_v2="require", **options))


@pytest.fixture(scope="module")
def tp4():
    """The tp4 accuracy and serve gates' engine: spec-tp4/options/core.yaml."""
    return _resolve_options(_NVFP4_NAMES, _TP4, _MAX_SEQ_LEN)


@pytest.fixture(scope="module")
def tp4_spellings():
    """``tensor_parallel_size`` 4 with no MoE or attention-DP option: bare, and as
    quickstart_advanced.py passes ``--tp_size=4``, the MoE sizes as -1."""
    return {
        "bare": _resolve_options(_NVFP4_NAMES, _TP4_BARE, _MAX_SEQ_LEN),
        "quickstart": _resolve_options(_NVFP4_NAMES, _TP4_QUICKSTART, _MAX_SEQ_LEN),
    }


@pytest.fixture(params=["bare", "quickstart"])
def tp4_bare(request, tp4_spellings):
    return tp4_spellings[request.param]


@pytest.fixture(scope="module")
def tp8():
    """The tp8 accuracy and serve gates' engine: spec-tp8/options/core.yaml."""
    return _resolve_options(_NVFP4_NAMES, _TP8, _MAX_SEQ_LEN)


@pytest.fixture(scope="module")
def tp8_gate():
    """The tp8 boot gate's engine: its quickstart command, --tp_size=8 and no MoE split."""
    path = _checkpoint_dir(_NVFP4_NAMES)
    return _resolve(path, _llm_args(**_gate_kwargs(path, _GATE_ARGV)))


@pytest.fixture(scope="module")
def tep4():
    return _resolve_options(_NVFP4_NAMES, _TEP4, _MAX_SEQ_LEN)


@pytest.fixture(scope="module")
def tep8():
    return _resolve_options(_NVFP4_NAMES, _TEP8, _MAX_SEQ_LEN)


@pytest.fixture(scope="module")
def dep4():
    return _resolve_options(_NVFP4_NAMES, _DEP4, _MAX_SEQ_LEN)


def _core_unit(request, world: int) -> list:
    """The core unit at ``world`` ranks, as each of its gates spells it with the MoE split."""
    return [request.getfixturevalue(name) for name in _SPELLINGS[world]]


def _unsplit(request, world: int) -> list:
    """The core unit at ``world`` ranks spelled with ``tensor_parallel_size`` alone: at tp4 bare
    and as quickstart_advanced.py passes ``--tp_size=4``, at tp8 the boot gate's command."""
    if world == 4:
        return list(request.getfixturevalue("tp4_spellings").values())
    return [request.getfixturevalue("tp8_gate")]


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


def _kv_cost(resolved, rank: int) -> CacheCost:
    """The engine's affine KV cost on ``rank``, as the KV-cache creator sizes the pool with it."""
    model_config, llm_args = resolved.model_config, resolved.llm_args
    kv_cache_config = llm_args.kv_cache_config
    manager = get_kv_cache_manager_cls(model_config, kv_cache_config)
    return CacheCost.from_raw(
        manager.get_cache_size_per_token(
            model_config,
            _rank_mapping(resolved.mapping, rank),
            tokens_per_block=kv_cache_config.tokens_per_block,
            max_seq_len=resolved.max_seq_len,
            max_batch_size=resolved.max_batch_size,
            max_num_tokens=resolved.max_num_tokens,
            kv_cache_config=kv_cache_config,
            spec_config=llm_args.speculative_config,
            is_draft=False,
        )
    )


def _summary(resolved) -> dict:
    """Everything the engine resolved for a checkpoint except its mapping."""
    pretrained = resolved.model_config.pretrained_config.to_dict()
    pretrained.pop("_name_or_path", None)
    llm_args, model_config = resolved.llm_args, resolved.model_config
    return dict(
        llm_quant=repr(llm_args.quant_config),
        quant=repr(model_config.quant_config),
        layers={k: repr(v) for k, v in (model_config.quant_config_dict or {}).items()},
        pretrained=pretrained,
        sparse=model_config.sparse_attention_config.model_dump(),
        routed=resolved.routed[0].quantization_config,
        sizes=(resolved.max_num_tokens, resolved.max_seq_len, resolved.max_batch_size),
        cuda_graphs=llm_args.cuda_graph_config.model_dump(),
        kv_cache=llm_args.kv_cache_config.model_dump(),
        moe=llm_args.moe_config.model_dump(),
        moe_backend=model_config.moe_backend,
        moe_max_num_tokens=model_config.moe_max_num_tokens,
    )


def _flat(value, prefix: str = "") -> dict:
    """``value`` as dotted keys to leaves, so two resolutions compare field by field."""
    if isinstance(value, dict):
        flat = {}
        for key, item in value.items():
            flat.update(_flat(item, f"{prefix}{key}."))
        return flat
    return {prefix[:-1]: value}


def _differs(a: dict, b: dict) -> set:
    a, b = _flat(a), _flat(b)
    return {key for key in a.keys() | b.keys() if a.get(key, a) != b.get(key, b)}


def _width_slice(resolved, rank: int) -> tuple[range, range]:
    """Rank ``rank``'s cut of every routed expert, from the mapping and the model config: the
    intermediate rows of w1/w3 (= the input columns of w2) and w2's NVFP4 block-scale columns,
    one per ``group_size`` inputs."""
    cfg = resolved.model_config.pretrained_config
    rank_mapping = _rank_mapping(resolved.mapping, rank)
    width = cfg.moe_intermediate_size // rank_mapping.moe_tp_size
    group = resolved.model_config.quant_config.group_size
    start = width * rank_mapping.moe_tp_rank
    return range(start, start + width), range(start // group, (start + width) // group)


# ------------------------------------------------------- the ranks in one process


class _Group:
    """The ranks in one process: rank r is a thread, and every exchange is a barrier, so each
    rank's call sees what every rank sent, as over MPI."""

    def __init__(self, size: int):
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

    def __init__(self, group: _Group, rank: int):
        self.group = group
        self.rank = self.tp_rank = rank
        self.world_size = self.tp_size = group.size

    def allgather(self, obj):
        return self.group.exchange(self.rank, obj)


class _KvPool:
    """The KV manager surface the warmup's dummy requests read: ``tokens`` free tokens in pages of
    32, and dummies recorded as the token counts they were asked for."""

    tokens_per_block = 32
    num_extra_kv_tokens = 0

    def __init__(self, tokens: int, max_seq_len: int):
        self.tokens = tokens
        self.max_seq_len = max_seq_len

    def get_num_available_tokens(self, *, token_num_upper_bound, max_num_draft_tokens=0, **_):
        return min(token_num_upper_bound, self.tokens)

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


def _shell(resolved, rank: int, cls=DecoderModelForCausalLM):
    """The causal-LM shell every target subclasses, on ``rank``, built on the meta device."""
    config = dataclasses.replace(
        resolved.model_config, mapping=_rank_mapping(resolved.mapping, rank)
    )
    cfg = config.pretrained_config
    with torch.device("meta"):
        return cls(
            DecoderModel(config),
            config=config,
            hidden_size=cfg.hidden_size,
            vocab_size=cfg.vocab_size,
        )


# ------------------------------------------------------- (1) topology, node layout, limits


@_WORLDS
def test_topology_and_limits(request, world):
    """Four ranks at tp4, eight at tp8, tensor parallel throughout: attention TP over every rank
    with no attention DP (16 of the 64 heads per rank at tp4, 8 at tp8), the routed experts tensor
    parallel as many ways and not expert-parallel, one pipeline stage -- at tp8 from core.yaml and
    from the boot gate's command alike. Rank r is TP rank r and MoE TP rank r, MoE EP rank 0: its
    window is all 128 experts at offset 0, and its share of each is the width slice [I r, I r + I)
    of 6144 (w1/w3 rows, w2 columns) with w2's block scales at columns [I/16 r, I/16 r + I/16) of
    384, I = 1536 at tp4 and 768 at tp8; the slices tile every expert once. One scheduler drives
    the ranks: 16 requests and 32768 tokens per step in all, unchunked."""
    core = request.getfixturevalue(f"tp{world}")
    llm_args = core.llm_args
    assert (
        llm_args.tensor_parallel_size,
        llm_args.moe_tensor_parallel_size,
        llm_args.moe_expert_parallel_size,
        llm_args.enable_attention_dp,
    ) == (world, world, 1, False)
    parallel = llm_args.parallel_config
    assert (parallel.tp_size, parallel.moe_tp_size, parallel.moe_ep_size, parallel.pp_size) == (
        world,
        world,
        1,
        1,
    )
    assert llm_args.attention_dp_config is None

    width, scale_width = _SLICE[world]
    for resolved in _core_unit(request, world):
        llm_args, mapping = resolved.llm_args, resolved.mapping
        parallel = llm_args.parallel_config
        assert (parallel.tp_size, parallel.pp_size, parallel.cp_size) == (world, 1, 1)
        assert llm_args.enable_attention_dp is False and parallel.enable_attention_dp is False

        assert (mapping.world_size, mapping.tp_size, mapping.pp_size, mapping.cp_size) == (
            world,
            world,
            1,
            1,
        )
        assert (mapping.moe_tp_size, mapping.moe_ep_size, mapping.moe_cluster_size) == (world, 1, 1)
        assert mapping.enable_attention_dp is False and mapping.dp_size == 1
        assert mapping.attn_tp_size == world and mapping.enable_lm_head_tp_in_adp is False
        assert mapping.has_moe_tp() and not mapping.has_moe_ep()
        assert resolved.model_config.mapping is mapping

        cfg = resolved.model_config.pretrained_config
        assert (cfg.num_attention_heads, cfg.n_routed_experts, cfg.num_experts_per_tok) == (
            64,
            128,
            4,
        )
        assert (cfg.moe_intermediate_size, resolved.model_config.quant_config.group_size) == (
            6144,
            16,
        )
        local_experts = cfg.n_routed_experts // mapping.moe_ep_size
        assert (cfg.num_attention_heads // mapping.tp_size, local_experts) == (_HEADS[world], 128)

        rows, scales = set(), set()
        for rank in range(world):
            rank_mapping = _rank_mapping(mapping, rank)
            assert (
                rank_mapping.tp_rank,
                rank_mapping.moe_tp_rank,
                rank_mapping.moe_ep_rank,
                rank_mapping.moe_cluster_rank,
            ) == (rank, rank, 0, 0)
            assert rank_mapping.ep_experts(cfg.n_routed_experts) == list(range(128))
            assert local_experts * rank_mapping.moe_ep_rank == 0
            assert rank_mapping.tp_group == rank_mapping.moe_tp_group == list(range(world))
            assert rank_mapping.moe_ep_group == [rank]
            rank_rows, scale_columns = _width_slice(resolved, rank)
            assert rank_rows == range(width * rank, width * rank + width)
            assert scale_columns == range(scale_width * rank, scale_width * rank + scale_width)
            assert rows.isdisjoint(rank_rows) and scales.isdisjoint(scale_columns)
            rows.update(rank_rows)
            scales.update(scale_columns)
        assert rows == set(range(6144)) and scales == set(range(6144 // 16))

        assert (resolved.max_batch_size, resolved.max_num_tokens, resolved.max_seq_len) == (
            16,
            32768,
            32768,
        )
        model_config = resolved.model_config
        assert (model_config.max_num_tokens, model_config.max_seq_len) == (32768, 32768)
        assert llm_args.enable_chunked_prefill is False


def test_tp_size_4_alone_builds_the_same_deployment(tp4, tp4_bare):
    """``tensor_parallel_size`` 4 with no MoE or attention-DP option -- also as
    quickstart_advanced.py passes ``--tp_size=4``, the MoE sizes as -1 -- resolves to the core
    unit's mapping on every rank and the same configuration throughout, and the router is handed
    an equal mapping. The only trace of the spelling is ``Mapping.moe_tp_ep_user_specified``,
    which neither ``to_dict`` nor equality carries."""
    assert tp4_bare.mapping.to_dict() == tp4.mapping.to_dict()
    for rank in range(4):
        bare, core = _rank_mapping(tp4_bare.mapping, rank), _rank_mapping(tp4.mapping, rank)
        assert bare == core
        assert (bare.moe_tp_rank, bare.moe_ep_rank) == (core.moe_tp_rank, core.moe_ep_rank)
    assert _summary(tp4_bare) == _summary(tp4)
    assert tp4_bare.routed[0].mapping == tp4.routed[0].mapping
    assert tp4_bare.routed[0].mapping.to_dict() == tp4.routed[0].mapping.to_dict()
    assert tp4.mapping.moe_tp_ep_user_specified is True
    assert tp4_bare.mapping.moe_tp_ep_user_specified is False


def test_gpus_per_node_is_the_visible_device_count(tp8, tp8_gate):
    """At tp8 neither gate states gpus_per_node. ``TorchLlmArgs.gpus_per_node`` defaults to None
    and its validator (outside Ray) replaces None with ``torch.cuda.device_count()`` of the
    process that builds the arguments, rank 0; each node's ranks see that node's four held
    devices, so it is 4. The mapping every rank builds from ``parallel_config`` carries it: rank r
    is local rank r % 4 on node r // 4, and the deployment is multi-node. The LLM constructor's
    device check reads it too: world_size_per_node is 4, which four visible devices satisfy.
    Stating gpus_per_node 4 builds the same mapping.

    The defaults of ``_ParallelConfig`` and ``Mapping`` are 8, not the device count: an
    eight-rank mapping built without it reads one node of eight -- local rank r, never multi-node
    -- and would demand eight devices on rank 0's node."""
    field = TorchLlmArgs.model_fields["gpus_per_node"]
    assert field.default is None and field.validate_default is True
    assert "gpus_per_node" not in _options(_TP8, _MAX_SEQ_LEN)
    assert "gpus_per_node" not in _gate_kwargs(tp8_gate.path, _GATE_ARGV)

    for resolved in (tp8, tp8_gate):
        llm_args, mapping = resolved.llm_args, resolved.mapping
        assert llm_args.gpus_per_node == _GPUS_PER_NODE == 4
        assert llm_args.parallel_config.gpus_per_node == 4
        assert mapping.gpus_per_node == 4 and resolved.model_config.mapping.gpus_per_node == 4
        assert llm_args.parallel_config.world_size_per_node == 4
        for rank in range(8):
            rank_mapping = _rank_mapping(mapping, rank)
            assert (rank_mapping.local_rank, rank_mapping.node_rank) == (rank % 4, rank // 4)
            assert rank_mapping.get_node_rank(rank) == rank // 4
            assert rank_mapping.is_multi_node() is True

    path = str(tp8.path)
    for devices in (4, 8):
        llm_args = _llm_args(devices, model=path, **_options(_TP8, _MAX_SEQ_LEN))
        assert llm_args.gpus_per_node == devices
        assert llm_args.parallel_config.world_size_per_node == devices
    stated = _llm_args(1, model=path, gpus_per_node=4, **_options(_TP8, _MAX_SEQ_LEN))
    assert stated.gpus_per_node == 4
    assert stated.parallel_config.to_mapping().to_dict() == tp8.mapping.to_dict()

    assert _ParallelConfig.model_fields["gpus_per_node"].default == 8
    assert _ParallelConfig(tp_size=8, moe_tp_size=8, moe_ep_size=1).world_size_per_node == 8
    unstated = Mapping(world_size=8, rank=5, tp_size=8, moe_tp_size=8, moe_ep_size=1)
    assert unstated.gpus_per_node == 8
    assert (unstated.local_rank, unstated.node_rank, unstated.is_multi_node()) == (5, 0, False)


def test_each_rank_selects_its_own_nodes_device(tp8):
    """The device each worker sets before building its executor
    (``BaseWorker._get_comm_ranks_device_id``) at tp8: its global MPI rank modulo the devices its
    node shows, so rank r drives device r % 4 of its node -- the mapping's local rank. The ranks
    allgather their global ranks and devices, and each pins its CPU affinity to its device."""
    group = _Group(8)
    local = threading.local()
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(torch.cuda, "device_count", lambda: _GPUS_PER_NODE)
        mp.setattr(torch.cuda, "set_device", lambda device: setattr(local, "device", device))
        mp.setattr(base_worker, "global_mpi_rank", lambda: local.rank)
        mp.setattr(
            base_worker,
            "mpi_comm",
            lambda: SimpleNamespace(allgather=lambda obj: group.exchange(local.rank, obj)),
        )
        mp.setattr(
            base_worker, "configure_cpu_affinity", lambda device: setattr(local, "cpus", device)
        )

        def step(rank):
            local.rank = rank
            ranks, devices = BaseWorker._get_comm_ranks_device_id(SimpleNamespace(global_rank=rank))
            return ranks, devices, local.device, local.cpus

        results = group.run(step)
    for rank, (ranks, devices, device, cpus) in enumerate(results):
        assert ranks == list(range(8))
        assert devices == [0, 1, 2, 3, 0, 1, 2, 3]
        assert device == cpus == rank % 4 == _rank_mapping(tp8.mapping, rank).local_rank


def test_core_yaml_and_boot_command_build_one_deployment(tp8, tp8_gate):
    """At tp8, core.yaml and the boot gate's command, which states --tp_size=8 and no MoE split,
    resolve to the same mapping on every rank and the same model, quantization, DSA config,
    limits, CUDA graphs, KV dtype, page size, block-reuse setting, KV manager and MoE config; the
    router is handed equal mappings. ``Mapping`` gives the MoE the whole TP width when neither MoE
    size is stated (moe_tp 8, moe_ep 1), so the command's -1 sentinels land where core.yaml's 8
    and 1 do.

    What differs, field by field, and nothing else:

    * the MoE sizes as stated: core.yaml's 8 and 1, the command's -1 and -1, in ``TorchLlmArgs``
      and in ``parallel_config`` -- and so in the usage telemetry, which reports them (8 and 1
      against 0 and 0) and nothing reads back;
    * the mapping's ``moe_tp_ep_user_specified``: True from core.yaml, False from the command;
      neither ``to_dict`` nor equality carries it;
    * the KV fraction: the quickstart's --kv_cache_fraction default, 0.9, against core.yaml's
      0.85, so the boot gate's pool is sized at 0.9 of the free memory and the accuracy gate's at
      0.85;
    * ``kv_cache_config.mamba_ssm_cache_dtype``: the quickstart's default 'bfloat16' against
      'auto', which the model loader resolves to the checkpoint's bfloat16 -- the same value in
      the quantization config, read only by a recurrent-state pool this model has none of;
    * ``attention_dp_config``: None from core.yaml, the quickstart's ``AttentionDpConfig`` from the
      command -- balancing and KV-aware routing off, so the executor builds the same default ADP
      router for either and never reads the quickstart's balancing waits, and attention DP is off
      besides."""
    assert tp8_gate.mapping.to_dict() == tp8.mapping.to_dict()
    assert tp8_gate.routed[0].mapping == tp8.routed[0].mapping
    assert tp8_gate.routed[0].mapping.to_dict() == tp8.routed[0].mapping.to_dict()
    for rank in range(8):
        gate, core = _rank_mapping(tp8_gate.mapping, rank), _rank_mapping(tp8.mapping, rank)
        assert gate == core
        assert (gate.moe_tp_rank, gate.moe_ep_rank) == (core.moe_tp_rank, core.moe_ep_rank)
    gate, core = _summary(tp8_gate), _summary(tp8)
    assert _differs(gate, core) == {
        "kv_cache.free_gpu_memory_fraction",
        "kv_cache.mamba_ssm_cache_dtype",
    }
    assert gate["moe_max_num_tokens"] == core["moe_max_num_tokens"] == 32768
    for resolved in (tp8, tp8_gate):
        graphs = resolved.llm_args.cuda_graph_config
        assert graphs.batch_sizes == [1, 2, 4, 8, 16] and graphs.enable_padding is True
        kv = resolved.llm_args.kv_cache_config
        assert (kv.dtype, kv.enable_block_reuse, kv.tokens_per_block) == ("auto", False, 32)
        assert kv.use_kv_cache_manager_v2 is True
        assert resolved.llm_args.disable_overlap_scheduler is False
    assert tp8.llm_args.kv_cache_config.free_gpu_memory_fraction == 0.85
    assert tp8_gate.llm_args.kv_cache_config.free_gpu_memory_fraction == 0.9

    # Field by field.
    stated = ("moe_tensor_parallel_size", "moe_expert_parallel_size")
    assert tuple(getattr(tp8.llm_args, name) for name in stated) == (8, 1)
    assert tuple(getattr(tp8_gate.llm_args, name) for name in stated) == (-1, -1)
    dumps = (
        args.model_dump(exclude={"attention_dp_config"})
        for args in (tp8_gate.llm_args, tp8.llm_args)
    )
    assert _differs(*dumps) == {
        "moe_tensor_parallel_size",
        "moe_expert_parallel_size",
        "kv_cache_config.free_gpu_memory_fraction",
        "kv_cache_config.mamba_ssm_cache_dtype",
    }
    parallel = tp8_gate.llm_args.parallel_config, tp8.llm_args.parallel_config
    assert _differs(*(p.model_dump() for p in parallel)) == {"moe_tp_size", "moe_ep_size"}
    assert [(p.moe_tp_size, p.moe_ep_size) for p in parallel] == [(-1, -1), (8, 1)]
    assert _differs(vars(tp8_gate.mapping), vars(tp8.mapping)) == {"moe_tp_ep_user_specified"}
    assert (tp8_gate.mapping.moe_tp_ep_user_specified, tp8.mapping.moe_tp_ep_user_specified) == (
        False,
        True,
    )
    telemetry = [usage_lib._extract_trtllm_config(r.llm_args) for r in (tp8_gate, tp8)]
    assert [(t["moe_tensor_parallel_size"], t["moe_expert_parallel_size"]) for t in telemetry] == [
        (0, 0),
        (8, 1),
    ]

    assert [r.llm_args.kv_cache_config.mamba_ssm_cache_dtype for r in (tp8_gate, tp8)] == [
        "bfloat16",
        "auto",
    ]
    for resolved in (tp8_gate, tp8):
        assert resolved.model_config.quant_config.mamba_ssm_cache_dtype == torch.bfloat16

    assert tp8.llm_args.attention_dp_config is None
    adp = tp8_gate.llm_args.attention_dp_config
    assert type(adp) is AttentionDpConfig
    assert (
        adp.enable_balance,
        adp.enable_kv_cache_aware_routing,
        adp.kv_cache_routing_conversation_affinity,
    ) == (False, False, False)
    # The quickstart's balancing waits (its flags' defaults, not the class's), read only when
    # balancing is on.
    assert (adp.timeout_iters, adp.batching_wait_iters) == (0, 0)
    for resolved in (tp8_gate, tp8):
        config = resolved.llm_args.attention_dp_config
        router = ADPRouter.create(
            dist=None, has_seq_slot_headroom=False, attention_dp_config=config
        )
        assert type(router) is DefaultADPRouter
        assert not (config is not None and config.enable_balance)


# ------------------------------------------------------- (2) quantization, the anchor's key


@_WORLDS
def test_quantization_is_the_anchor_key(request, world):
    """At tp4 and tp8 as at tep4, dep4 and tep8: MIXED_PRECISION with the FP8 KV cache, the GSM8K
    anchor row's key, and the router handed params.json's quantization_config with its 475
    quantized_layers intact, on the core unit's mapping -- at tp8 from core.yaml and from the boot
    gate's command alike."""
    resolutions = _core_unit(request, world)
    on_disk = json.loads((resolutions[0].path / "params.json").read_text())["quantization_config"]
    assert len(on_disk["quantized_layers"]) == 475
    for resolved in resolutions:
        assert resolved.prequantized is True
        for quant in (resolved.llm_args.quant_config, resolved.model_config.quant_config):
            assert quant.quant_algo == QuantAlgo.MIXED_PRECISION
            assert quant.kv_cache_quant_algo == QuantAlgo.FP8
            assert quant.group_size == 16
        assert resolved.llm_args.kv_cache_config.dtype == "auto"
        assert _anchor(resolved.llm_args) == 91.926

        assert len(resolved.routed) == 1
        seen = resolved.routed[0]
        assert seen.quantization_config == on_disk
        assert resolved.model_config.pretrained_config.quantization_config == on_disk
        assert seen.pretrained_config is resolved.model_config.pretrained_config
        assert resolved.model_config.modeling_v2_target is _Routed
        assert seen.compute_capability == _SM103 and seen.spec_config is None
        assert seen.mapping is resolved.model_config.mapping
        assert (seen.mapping.world_size, seen.mapping.tp_size, seen.mapping.pp_size) == (
            world,
            world,
            1,
        )
        assert (seen.mapping.moe_tp_size, seen.mapping.moe_ep_size) == (world, 1)
        assert seen.mapping.enable_attention_dp is False


# ------------------------------------------------------- (3) the KV pool, per rank


@_WORLDS
def test_kv_pool_as_at_tep(request, world):
    """MLA's latent and the indexer keys are not head-sharded, the MoE split never reaches the KV
    cost, and every rank holds all 16 requests: a token costs the same on every rank at tp4 as at
    tep4 and dep4, and on each of the eight ranks at tp8 as on each tep8 rank, an fp8_ds_mla
    latent of 656 bytes (512 e4m3, four fp32 tile scales, 64 bf16 rope) on each of the 63 KV
    owners plus a 128-byte fp8 indexer key and its fp32 scale on each of the 44 full-indexer
    layers, in pages of 32, in MistralLarge4CacheManager (the nine sharers keep no latent). A
    rank's pool for 16 requests of max_seq_len 32768 is that times 16 x 32768 tokens."""
    core = _core_unit(request, world)
    peers = [request.getfixturevalue(name) for name in _KV_PEERS[world]]
    for resolved in core + peers:
        model_config, kv_cache_config = resolved.model_config, resolved.llm_args.kv_cache_config
        assert get_kv_cache_manager_cls(model_config, kv_cache_config) is (
            MistralLarge4CacheManager
        )
        assert kv_cache_config.tokens_per_block == 32

    model_config = core[0].model_config
    cfg, sparse = model_config.pretrained_config, model_config.sparse_attention_config
    latent = 656  # fp8_ds_mla: 512 e4m3 + 4 fp32 tile scales + 64 bf16 rope
    owners = sum(1 for i, s in enumerate(cfg.kv_source_layers) if s == i)
    indexer_key = sparse.index_head_dim + sparse.index_head_dim // 128 * 4
    full_indexer_layers = cfg.index_topk_pattern.count("F")
    per_token = latent * owners + indexer_key * full_indexer_layers
    assert (latent, owners, indexer_key, full_indexer_layers) == (656, 63, 132, 44)
    assert per_token == 47136

    for rank in range(world):
        cost = _kv_cost(core[0], rank)
        assert (cost.slope, cost.intercept) == (per_token, 0)
        for resolved in core[1:] + peers:
            assert cost == _kv_cost(resolved, rank)
        assert cost.bytes_for_tokens(32) == 32 * per_token == 1_508_352

    tokens = core[0].max_batch_size * core[0].max_seq_len
    assert tokens == 16 * 32768
    pool = _kv_cost(core[0], 0).bytes_for_tokens(tokens)
    assert pool == tokens * per_token
    assert pool == 524288 * 47136 == 24_712_839_168
    assert math.isclose(pool / 2**30, 23.016, abs_tol=5e-4)


@_WORLDS
def test_indexer_chunk_holds_a_whole_prompt(request, world):
    """The indexer's [T, N] logits are cut at indexer_max_chunk_size: a whole 32K prompt, the
    largest unchunked step, is one chunk."""
    resolved = request.getfixturevalue(f"tp{world}")
    model_config = resolved.model_config
    params = model_config.sparse_attention_config.to_sparse_metadata_params(
        pretrained_config=model_config.pretrained_config
    )
    assert params.indexer_max_chunk_size >= 32768
    assert params.indexer_max_chunk_size >= resolved.max_num_tokens


# ------------------------------------------------------- (4) the largest step


@_WORLDS
def test_warmup_runs_max_num_tokens_on_every_rank(request, world):
    """The warmup's shapes at tp4 and at tp8. Each rank proposes its shapes from its own free KV,
    the ranks keep the ones all proposed, and every rank enters each shape's forward. The largest
    is max_num_tokens of context in one step -- a 32767-token and a 1-token prompt, since a prompt
    leaves room for one decode token -- run by all the ranks on the same tokens. Every rank's
    window is all 128 experts, so each rank's routed experts see all 32768 tokens and every one of
    their top-4 slots: 4 x 32768 = 131072 permuted rows on every rank, all on local experts."""
    resolved = request.getfixturevalue(f"tp{world}")
    config = _runner_config(resolved)
    model = SimpleNamespace(model_config=resolved.model_config)
    group = _Group(world)

    def rank_warmup(rank):
        # The pool holds at least max_batch_size requests of max_seq_len.
        resources = _Resources(
            _KvPool(config.max_batch_size * config.max_seq_len, config.max_seq_len)
        )
        warmup = DecoderWarmup.__new__(DecoderWarmup)
        warmup._config = config
        warmup.kv_cache_manager_key = ResourceManagerType.KV_CACHE_MANAGER
        agreement = WarmupAgreement(_RankDist(group, rank), _rank_mapping(resolved.mapping, rank))
        assert agreement.is_distributed_forward()
        shapes = agreement.agree_shapes(warmup._get_full_general_warmup_requests(resources))
        factory = DummyRequestFactory(
            config=config,
            kv_cache_manager_key=ResourceManagerType.KV_CACHE_MANAGER,
            model=model,
            get_runtime_tokens_per_gen_step=lambda draft_len: 1 + draft_len,
            release_request_state=None,
        )
        max_context = max(tokens for tokens, gen in shapes if gen == 0)
        batch = factory.create_warmup_request(resources, max_context, 0)
        enters = agreement.should_run_warmup_batch(batch, max_context, "max context")
        prompts = [request.token_num for request in batch.context_requests]
        return shapes, enters, prompts

    results = group.run(rank_warmup)
    assert len(results) == world
    for shapes, enters, prompts in results:
        assert shapes == [(1, 0), (1, 1), (32768, 0), (16, 16), (2, 0)]
        assert enters is True
        assert prompts == [32767, 1]

    cfg = resolved.model_config.pretrained_config
    rows = sum(results[0][2])
    assert rows == resolved.max_num_tokens == 32768
    for rank in range(world):
        window = _rank_mapping(resolved.mapping, rank).ep_experts(cfg.n_routed_experts)
        assert window == list(range(cfg.n_routed_experts))
    assert rows * cfg.num_experts_per_tok == 131072
    local_experts = cfg.n_routed_experts // resolved.mapping.moe_ep_size
    assert local_experts == cfg.n_routed_experts


@_WORLDS
def test_kv_estimation_runs_one_max_num_tokens_step(request, world, dep4):
    """Before the real pool exists the engine runs a profiling pass on dummy context requests as
    long as a request may be, net max_seq_len - 1. Without attention DP it submits max_num_tokens
    of them once (a 32767-token and a 1-token prompt) for the one scheduler the four ranks of tp4
    or the eight of tp8 share; dep4 submits a copy per rank. The DSA config keeps the pass off the
    chunked-MLA profile."""
    resolved = request.getfixturevalue(f"tp{world}")

    def lengths(resolved):
        creator = KvCacheCreator.__new__(KvCacheCreator)
        creator._model_engine = SimpleNamespace(
            model=SimpleNamespace(model_config=resolved.model_config), use_mrope=False
        )
        creator._max_num_tokens = resolved.max_num_tokens
        creator._max_beam_width = 1
        creator._mapping = resolved.mapping
        requests = creator._create_dummy_context_requests(max(1, resolved.max_seq_len - 1))
        assert creator._mla_chunked_profile_length is None
        return [len(request.input_token_ids) for request in requests]

    assert lengths(resolved) == [32767, 1]
    assert sum(lengths(resolved)) == resolved.max_num_tokens
    assert lengths(dep4) == [32767, 1] * 4


# ------------------------------------------------------- (5) the engine's reads of the MoE split


@_WORLDS
def test_engine_reads_of_the_moe_split(request, world):
    """What the engine derives from the MoE split before a target runs. ModelConfig sizes the MoE
    token budget from the attention-DP width, not the MoE split: max_num_tokens x dp_size = 32768
    at tp4 as at tep4, and at tp8 (from core.yaml and from the boot gate's command alike) as at
    tep8. No MoE load balancer is built on any rank: none is configured, and
    ``maybe_create_moe_load_balancer`` reads only the EP split, which is 1 at tp4 and tp8. The
    autotuner's tuning group is the TP group -- ``tp_size`` and ``tp_rank`` -- so it is every
    rank, (4, r) at tp4 and tep4 and (8, r) over both nodes at tp8 and tep8."""
    for resolved in _core_unit(request, world) + [request.getfixturevalue(_TEP[world])]:
        model_config = resolved.model_config
        assert model_config.moe_max_num_tokens == resolved.max_num_tokens * resolved.mapping.dp_size
        assert model_config.moe_max_num_tokens == 32768
        assert resolved.llm_args.moe_config.load_balancer is None
        assert model_config.moe_load_balancer is None
        for rank in range(world):
            rank_mapping = _rank_mapping(resolved.mapping, rank)
            with maybe_create_moe_load_balancer(model_config, rank_mapping) as balancer:
                assert balancer is None
            tuner = SimpleNamespace(mapping=rank_mapping)
            assert AutoTuner._tuning_group(tuner) == (world, rank)


@_WORLDS
def test_moe_autotuner_at_128_local_experts(request, world):
    """The NVFP4 MoE runner the target calls (128 experts, all 128 local at offset 0, width 1536
    at tp4 and 768 at tp8, top-4, ``tune_max_num_tokens`` = max_num_tokens, ``use_dp`` False)
    tunes INDEPENDENT, each rank alone, so tuning sends nothing between the ranks (or, at tp8, the
    nodes). Its expert-parallel size, num_experts // local_num_experts, is 1 here and the world
    size at tep4 and tep8, and without attention DP neither moves a bucket: the buckets and the
    token-to-bucket rule are tep's, 32768 the top bucket. The tuner's cache key (top_k,
    intermediate_size, local_num_experts, act_type) keeps the tp entries apart from the tep ones.
    The tuning group is the TP group, ``(tp_size, tp_rank)``, at tp and tep alike."""
    tp, tep = request.getfixturevalue(f"tp{world}"), request.getfixturevalue(_TEP[world])
    for rank in range(world):
        tuner = SimpleNamespace(mapping=_rank_mapping(tp.mapping, rank))
        assert AutoTuner._tuning_group(tuner) == (world, rank)
        tuner = SimpleNamespace(mapping=_rank_mapping(tep.mapping, rank))
        assert AutoTuner._tuning_group(tuner) == (world, rank)

    def runner(resolved):
        cfg, mapping = resolved.model_config.pretrained_config, resolved.mapping
        local_experts = cfg.n_routed_experts // mapping.moe_ep_size
        return FP4BlockScaleMoERunner(
            num_experts=cfg.n_routed_experts,
            top_k=cfg.num_experts_per_tok,
            n_group=None,
            topk_group=None,
            intermediate_size=cfg.moe_intermediate_size // mapping.moe_tp_size,
            local_expert_offset=local_experts * _rank_mapping(mapping, world - 1).moe_ep_rank,
            local_num_experts=local_experts,
            routed_scaling_factor=None,
            routing_method_type=RoutingMethodType.Renormalize,
            do_finalize=True,
            act_type=ActType_TrtllmGen.SwiGlu.value,
            tune_max_num_tokens=resolved.max_num_tokens,
        )

    width, _ = _SLICE[world]
    tp_runner, tep_runner = runner(tp), runner(tep)
    assert (tp_runner.local_expert_offset, tp_runner.local_num_experts) == (0, 128)
    assert tp_runner.intermediate_size == width
    assert tp_runner.num_experts // tp_runner.local_num_experts == 1
    assert tep_runner.num_experts // tep_runner.local_num_experts == world
    assert tp_runner.tuning_config is FP4BlockScaleMoERunner.get_tuning_config(
        1, tune_max_num_tokens=32768, use_dp=False
    )
    swiglu = ActType_TrtllmGen.SwiGlu.value
    assert tp_runner.unique_id() == (4, width, 128, swiglu)
    assert tep_runner.unique_id() == (4, 6144, 128 // world, swiglu)

    tp_tuning, tep_tuning = tp_runner.tuning_config, tep_runner.tuning_config
    assert tp_tuning.distributed_tuning_strategy == DistributedTuningStrategy.INDEPENDENT
    assert tp_tuning.tune_max_num_tokens == 32768
    (tp_spec,) = tp_tuning.dynamic_tensor_specs
    (tep_spec,) = tep_tuning.dynamic_tensor_specs
    assert tp_spec.gen_tuning_buckets == tep_spec.gen_tuning_buckets
    assert max(tp_spec.gen_tuning_buckets) == 32768
    for tokens in (1, 3, 16, 17, 4097, 8192, 8193, 16384, 32767, 32768):
        assert tp_spec.map_to_tuning_buckets(tokens) == tep_spec.map_to_tuning_buckets(tokens)
    assert tp_spec.map_to_tuning_buckets(32768) == 32768


def test_mnnvl_and_collectives_at_moe_tp_8(tp8, tep8):
    """Where the engine's collectives read the MoE split and the node layout at tp8, on each
    rank's mapping:

    * the rank's mapping is an ``MpiTopology`` and its communicator an ``MPIDist`` (the MPI
      launch leaves TLLM_DISABLE_MPI unset); ``DeviceMeshTopology`` -- whose mesh has no moe_tp
      dimension unless moe_ep_size > 1 -- and ``TorchDist`` are the non-MPI path;
    * ``MnnvlMemory.get_comm`` splits by pipeline, context and MoE TP rank, keyed by TP rank: at
      moe_tp 8 every rank is its own colour, eight one-rank communicators (tep8: one of all eight).
      Only the MoE all-to-all workspaces (and Helix's subclass) build an ``MnnvlMemory``, and the
      all-to-all is refused at moe_tp > 1;
    * the MNNVL AllReduce's workspace communicator (``_get_mnnvl_workspace_comm``) splits by
      pipeline and context rank only, keyed by TP rank: one communicator of all eight ranks, as at
      tep8. An engine ``AllReduce`` at the default AUTO strategy takes MNNVL only for a group that
      spans nodes (``MNNVLAllReduce.is_mnnvl`` asks ``mapping.is_multi_node()``: true here), and
      the one-node IPC workspace (``IpcMemory``) stays closed past one node's devices; an MNNVL
      buffer sits on device ``mapping.local_rank``. No userbuffers: the strategy is AUTO, not
      NCCL_SYMMETRIC, and torch.compile is off."""
    llm_args = tp8.llm_args
    assert llm_args.allreduce_strategy == "AUTO" and llm_args.torch_compile_config is None
    assert MnnvlMemory.comm_layout_signature(tp8.mapping) == (8, 1, 1, 8, 8, 1, 1)
    assert MnnvlMemory.comm_layout_signature(tep8.mapping) == (8, 1, 1, 8, 1, 8, 1)

    splits = []
    with pytest.MonkeyPatch.context() as mp:
        mp.delenv("TLLM_DISABLE_MPI", raising=False)
        mp.setattr(
            distributed_ops,
            "mpi_comm",
            lambda: SimpleNamespace(Split=lambda color, key: splits.append((color, key))),
        )
        for rank in range(8):
            rank_mapping = _rank_mapping(tp8.mapping, rank)
            assert type(rank_mapping) is MpiTopology
            assert type(Distributed.get(rank_mapping)) is MPIDist
            assert MnnvlMemory.comm_split_color_key(rank_mapping) == (rank, rank)
            assert MnnvlMemory.comm_split_color_key(_rank_mapping(tep8.mapping, rank)) == (0, rank)
            distributed_ops._get_mnnvl_workspace_comm(rank_mapping)
            assert distributed_ops._mnnvl_device_index(rank_mapping) == rank % 4
            assert rank_mapping.is_multi_node() is True
            assert IpcMemory(rank_mapping, 1024).open_ipc is False
    assert splits == [(0, rank) for rank in range(8)]


@_WORLDS
def test_weight_sharing_reads(request, world):
    """``pyexecutor/model_loader.py`` imports ``_torch/weight_sharing`` and reads the MoE split
    there on every load. ``ModelLoader.load`` qualifies the constructed model against the
    post-transform profile registry with a ``PostTransformRuntimeConfig`` that records moe_tp 4,
    moe_ep 1 on one node at tp4 and moe_tp 8, moe_ep 1 and multi-node at tp8: no profile names a
    modeling_v2 target's class, so the decision is ROOT_MODEL_CLASS_NOT_REGISTERED and nothing is
    shared. ``SourceIdentity``, which hashes moe_tp_size and each rank's moe_tp_rank
    (``check_weight_sharing_compatibility`` compares it), is built only for GMS or an MX
    checkpoint, and the gates load AUTO from this checkpoint's files."""

    class _Target(DecoderModelForCausalLM):
        """Stands for the target class the router returns: no profile registers it."""

    tp, tep = request.getfixturevalue(f"tp{world}"), request.getfixturevalue(_TEP[world])
    for resolved, moe_split in ((tp, (world, 1)), (tep, (1, world))):
        runtime = PostTransformRuntimeConfig.from_model_config(resolved.model_config)
        assert (runtime.tp_size, runtime.pp_size, runtime.cp_size) == (world, 1, 1)
        assert (runtime.moe_tp_size, runtime.moe_ep_size) == moe_split
        assert (runtime.attention_tp_size, runtime.attention_dp, runtime.multi_node) == (
            world,
            False,
            world > _GPUS_PER_NODE,
        )

        shell = _shell(resolved, 3, _Target)
        decision = RankModelLoader._qualify_post_transform_profile(
            shell,
            config_identity=PostTransformConfigIdentity.from_model_config(resolved.model_config),
            speculative_mode=None,
            loads_draft_weights=False,
        )
        assert decision.qualified is False and decision.profile is None
        assert decision.reason == PostTransformQualificationReason.ROOT_MODEL_CLASS_NOT_REGISTERED

        assert resolved.llm_args.load_format == LoadFormat.AUTO
        assert resolved.checkpoint_loader.checkpoint_format != "MX"
        assert (
            RankModelLoader._needs_source_identity(resolved.checkpoint_loader, LoadFormat.AUTO)
            is False
        )


@_WORLDS
def test_shell_lm_head_as_at_tep(request, world):
    """The causal-LM shell every target subclasses reads the TP width, not the MoE split: on each
    rank ``DecoderModelForCausalLM`` builds the same vocab-parallel lm_head at tp4 as at tep4 and
    at tp8 as at tep8, a quarter (51200 rows of 6144) or an eighth (25600 rows) of the
    204800-token vocabulary, gathered over the TP group of every rank and with no AllReduce of its
    own. Built on the meta device: no weights."""
    tp, tep = request.getfixturevalue(f"tp{world}"), request.getfixturevalue(_TEP[world])

    def lm_head(resolved, rank):
        shell = _shell(resolved, rank)
        head = shell.lm_head
        return dict(
            type=type(head),
            tp=(head.tp_size, head.tp_rank, head.tp_mode),
            shape=tuple(head.weight.shape),
            gather=(head.gather_output, shell.model_config.lm_head_gather_output),
            all_reduce=head.all_reduce,
            group=head.mapping.tp_group,
            quant=head.quant_config,
            pp=(shell.pp_rank, shell.pp_size),
        )

    vocab_rows = {4: 51200, 8: 25600}[world]
    for rank in range(world):
        head = lm_head(tp, rank)
        assert head == lm_head(tep, rank)
        assert head["tp"] == (world, rank, TensorParallelMode.COLUMN)
        assert head["shape"] == (204800 // world, 6144) == (vocab_rows, 6144)
        assert head["gather"] == (True, True) and head["all_reduce"] is None
        assert head["group"] == list(range(world)) and head["pp"] == (0, 1)


# ------------------------------------------------------- (6) tep4, dep4 and tep8 as before


def test_tep4_dep4_and_tep8_route_as_before(tep4, dep4, tep8):
    """The NVFP4 checkpoint at tep4, at dep4 and at tep8 resolves and routes as before this
    target: each claimed by its own target -- tep8 by the tep4 target, which serves eight ranks
    over two nodes of four too -- the routed experts expert-parallel and not split along their
    width (MoE TP rank 0 everywhere), each on the checkpoint's anchor and the same 47136-byte
    token."""
    cases = (
        (tep4, 4, False, "ModelingV2MistralLarge4Nvfp4Sm103Tep4"),
        (dep4, 4, True, "ModelingV2MistralLarge4Nvfp4Sm103Dep4"),
        (tep8, 8, False, "ModelingV2MistralLarge4Nvfp4Sm103Tep4"),
    )
    for resolved, world, attention_dp, target in cases:
        seen = resolved.routed[0]
        mapping = seen.mapping
        assert (mapping.world_size, mapping.tp_size, mapping.pp_size) == (world, world, 1)
        assert (mapping.moe_ep_size, mapping.moe_tp_size, mapping.enable_attention_dp) == (
            world,
            1,
            attention_dp,
        )
        assert mapping.gpus_per_node == 4 and mapping.is_multi_node() is (world > 4)
        for rank in range(world):
            rank_mapping = _rank_mapping(mapping, rank)
            assert (rank_mapping.moe_ep_rank, rank_mapping.moe_tp_rank) == (rank, 0)
            assert (rank_mapping.local_rank, rank_mapping.node_rank) == (rank % 4, rank // 4)
        assert routing.route(seen.pretrained_config, mapping, _SM103, None).__name__ == target
        assert resolved.llm_args.quant_config.quant_algo == QuantAlgo.MIXED_PRECISION
        assert resolved.llm_args.quant_config.kv_cache_quant_algo == QuantAlgo.FP8
        assert _anchor(resolved.llm_args) == 91.926
        assert (_kv_cost(resolved, 0).slope, _kv_cost(resolved, 0).intercept) == (47136, 0)
        assert (resolved.max_num_tokens, resolved.max_seq_len, resolved.max_batch_size) == (
            32768,
            32768,
            16,
        )


@_WORLDS
def test_tp_differs_from_tep_in_the_moe_split_only(request, world):
    """The same checkpoint at tp4 and at tep4, and at tp8 and at tep8, resolves to the same
    configuration -- quantization, per-layer quantization, model config, DSA config, limits, CUDA
    graphs, KV config, the MoE config and token budget, what the router reads -- except the
    mapping's MoE split. No tep4 or dep4 branch of the routing claims tp4 or tp8 in any
    spelling (core.yaml; ``tensor_parallel_size`` alone, at tp4 bare and as quickstart_advanced.py
    passes it, at tp8 the boot gate's command): it is refused, naming moe_ep=1 and moe_tp at the
    world size, or claimed by the tp4 target."""
    tp, tep = request.getfixturevalue(f"tp{world}"), request.getfixturevalue(_TEP[world])
    assert _summary(tp) == _summary(tep)
    tp_mapping, tep_mapping = tp.mapping.to_dict(), tep.mapping.to_dict()
    assert {k for k in tp_mapping if tp_mapping[k] != tep_mapping[k]} == {
        "moe_tp_size",
        "moe_ep_size",
    }
    deployment = f"world_size={world} tp={world} pp=1 moe_ep=1 moe_tp={world} attention_dp=False"
    for resolved in [tp] + _unsplit(request, world):
        seen = resolved.routed[0]
        try:
            target = routing.route(seen.pretrained_config, seen.mapping, _SM103, None)
        except NoTarget as refusal:
            assert deployment in str(refusal)
        else:
            assert target.__name__ == _TP4_TARGET
            assert target.__name__ not in _OTHER_TARGETS
