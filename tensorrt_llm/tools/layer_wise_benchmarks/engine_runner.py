# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The layer-wise benchmark's engine path, for models without per-layer modules.

A modeling_v2 target runs its layers as one forward over its own caches and metadata, so the
per-layer runner cannot call them. Here the first N layers (``layer_indices`` must be
``[0, N)``) are built through ``create_py_executor`` exactly as serving builds them -- the
executor's warmup autotunes and captures the CUDA graphs -- and each run pack replays one
generation step of a fixed batch through ``model_engine.forward``. The interface matches
``Runner``: ``create_run_pack(...)`` returns a callable, so ``run.py`` times it the same way.

Every run pack uses the same seeded input token ids and refills every cache pool (KV, indexer,
recurrent state) with the same seeded random bytes, so the work per step is identical across
runs; all-zero caches can send data-dependent kernels down other paths. DFlash2 maps every dummy
request to its drafter's dummy slot, whose history is zero: the drafter part of the step is
cheaper than in serving.
"""

from __future__ import annotations

import contextlib
import inspect
import json
import os
from typing import Optional

import nvtx
import torch

from tensorrt_llm._torch.pyexecutor.py_executor_creator import create_py_executor
from tensorrt_llm._torch.pyexecutor.resource_manager import ResourceManagerType
from tensorrt_llm._torch.pyexecutor.scheduler import ScheduledRequests
from tensorrt_llm._torch.speculative.utils import get_draft_kv_cache_manager
from tensorrt_llm._utils import TensorWrapper, convert_to_torch_tensor
from tensorrt_llm.llmapi.llm_args import TorchLlmArgs
from tensorrt_llm.logger import logger

from . import layer_truncation
from .runner import Runner


def select_backend(backend: str, pretrained_model_name_or_path: str, llm_args: dict) -> str:
    """``layers`` or ``engine`` for ``backend``.

    ``auto`` picks the engine for an architecture modeling_v2 routes: its targets have no
    per-layer modules.
    """
    if backend in ("layers", "engine"):
        return backend
    if backend != "auto":
        raise ValueError(f"backend {backend!r}: one of auto, layers, engine")
    from tensorrt_llm._torch._experimental.modeling_v2._router_index import (
        ModelingV2Mode,
        registered_routes,
    )

    if str(llm_args.get("modeling_v2", ModelingV2Mode.AUTO.value)) == ModelingV2Mode.OFF.value:
        return "layers"
    config_path = os.path.join(pretrained_model_name_or_path, "config.json")
    with open(config_path) as f:
        architectures = json.load(f).get("architectures") or [None]
    return "engine" if architectures[0] in registered_routes() else "layers"


def merge_llm_args(base: dict, override: dict) -> dict:
    """``base`` with ``override`` applied, nested dicts merged key by key."""
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = merge_llm_args(merged[key], value)
        else:
            merged[key] = value
    return merged


def cache_regions(kv_cache_manager) -> list[torch.Tensor]:
    """Byte views of every GPU cache pool: KV, indexer and recurrent state alike."""
    impl = getattr(kv_cache_manager, "impl", None)
    if impl is not None and hasattr(impl, "pool_group_descs"):  # KVCacheManagerV2 family
        return [
            convert_to_torch_tensor(
                TensorWrapper(
                    int(pool.base_address),
                    torch.uint8,
                    [int(group.num_slots) * int(pool.slot_bytes)],
                )
            )
            for group in impl.pool_group_descs
            for pool in group.pools
            if int(group.num_slots) * int(pool.slot_bytes)
        ]
    pool = kv_cache_manager.get_unique_primary_pool()
    return [pool.view(torch.uint8).view(-1)]


def fill_random_(regions: list[torch.Tensor], seed: int) -> None:
    """Fill ``regions`` with seeded random bytes in [0, 64).

    With the top two bits clear, every 1-, 2- and 4-byte float decodes to a finite value below
    2 and FP8 E4M3 never hits its NaN codes, so the pools stay readable whatever format a layer
    stores.
    """
    generator = torch.Generator(device="cuda").manual_seed(seed)
    for region in regions:
        region.random_(0, 64, generator=generator)


def prefix_num_layers(layer_indices: list[int]) -> int:
    """N for ``layer_indices == [0, N)``; the engine path cannot skip leading layers."""
    n = len(layer_indices)
    if list(layer_indices) != list(range(n)):
        raise ValueError(
            f"layer_indices {layer_indices}: the engine path builds a prefix of the model, so "
            f"they must be 0..N-1; measure a window [a, b) as the difference of two runs"
        )
    return n


class EngineRunner:
    """Build the first N layers through a real executor and replay fixed generation steps."""

    def __init__(
        self,
        pretrained_model_name_or_path: str,
        *,
        llm_args: TorchLlmArgs,
        layer_indices: list[int],
        seed: int = 0,
    ) -> None:
        self.layer_indices = list(layer_indices)
        n = prefix_num_layers(self.layer_indices)
        total = layer_truncation.checkpoint_num_layers(pretrained_model_name_or_path)
        num_layers = n if total is not None and n < total else None
        with Runner.truncate_layers_ctx(pretrained_model_name_or_path, num_layers):
            self.executor = create_py_executor(
                llm_args, checkpoint_dir=pretrained_model_name_or_path, start_worker=False
            )
        self.model_engine = self.executor.model_engine
        self.resource_manager = self.executor.resource_manager
        self.kv_cache_manager = self.resource_manager.get_resource_manager(
            ResourceManagerType.KV_CACHE_MANAGER
        )
        self.model = self.model_engine.model
        self.model_config = self.model.model_config
        self._cache_regions = cache_regions(self.kv_cache_manager)
        logger.info(
            f"Layer-wise benchmarks: {len(self._cache_regions)} cache pools, "
            f"{sum(r.numel() for r in self._cache_regions) / (1 << 30):.2f} GiB, seeded random fill"
        )
        self.seed = seed
        self._batch_stack = contextlib.ExitStack()
        self._input_ids: Optional[torch.Tensor] = None
        self._start_event: Optional[torch.cuda.Event] = None
        self._nvtx_message: Optional[str] = None
        self._replayed = None
        graph_runner = self.model_engine.cuda_graph_runner
        if graph_runner is not None and graph_runner.enabled:
            graph_runner.register_pre_replay_hook(self._before_replay)

    def _before_replay(self, key, num_tokens, static_tensors, current_inputs) -> None:
        self._replayed = key
        if self._input_ids is not None:
            static_tensors["input_ids"][:num_tokens].copy_(self._input_ids[:num_tokens])
        if self._start_event is not None:
            self._start_event.record()
        if self._nvtx_message is not None:
            nvtx.push_range(self._nvtx_message)

    def _build_batch(
        self, batch_size: int, request_id_begin: int, draft_len: int, seq_len_kv_cache: int
    ) -> ScheduledRequests:
        factory = self.model_engine._runner._dummy_requests
        config = factory._config
        width = factory.get_runtime_tokens_per_gen_step(draft_len) - 1
        add_kwargs = {}
        if (
            "materialize_history"
            in inspect.signature(self.kv_cache_manager.add_dummy_requests).parameters
        ):
            add_kwargs["materialize_history"] = True
        request_ids = list(range(request_id_begin, request_id_begin + batch_size))
        # token_num = history + the one input token: seq_len_kv_cache tokens are cached.
        requests = self.kv_cache_manager.add_dummy_requests(
            request_ids,
            token_nums=[seq_len_kv_cache + 1] * batch_size,
            is_gen=True,
            max_num_draft_tokens=width,
            kv_reserve_draft_tokens=config.max_draft_loop_tokens,
            use_mrope=config.use_mrope,
            max_beam_width=config.max_beam_width,
            draft_kv_cache_manager=get_draft_kv_cache_manager(
                config.spec_config, self.resource_manager
            ),
            **factory._dummy_request_kwargs(self.resource_manager, batch_size),
            **add_kwargs,
        )
        if requests is None:
            raise RuntimeError(
                f"no KV cache for {batch_size} requests of {seq_len_kv_cache} tokens; raise "
                "kv_cache_config in llm_args or lower the batch / KV length"
            )
        batch = ScheduledRequests()
        batch.generation_requests = requests
        batch = factory.finish_cuda_graph_warmup_request(
            batch, self.resource_manager, batch_size, draft_len
        )
        if batch is None:
            raise RuntimeError(f"could not add the dummy request resources for {batch_size}")
        return self._batch_stack.enter_context(
            factory.release_batch_context(batch, self.resource_manager)
        )

    def create_run_pack(
        self,
        run_type: str,
        *,
        batch_size: int,
        request_id_begin: int,
        seq_len_q: int,
        seq_len_kv_cache: int,
        kv_cache_manager=None,
        attn_workspace=None,
    ):
        """One generation step of ``batch_size`` requests; the arguments mirror ``Runner``.

        Each request has ``seq_len_kv_cache`` cached tokens and ``seq_len_q`` tokens per step
        (1 + the draft).
        """
        if run_type != "GEN":
            raise NotImplementedError("the engine path measures generation steps only")
        self.release_batch()
        draft_len = seq_len_q - 1
        batch = self._build_batch(batch_size, request_id_begin, draft_len, seq_len_kv_cache)
        # Every problem spec starts from the same cache contents; replays then update the
        # recurrent state in place, deterministically.
        fill_random_(self._cache_regions, self.seed)
        vocab_size = self.model_config.pretrained_config.vocab_size
        generator = torch.Generator(device="cuda").manual_seed(self.seed)
        self._input_ids = torch.randint(
            0,
            vocab_size,
            (batch_size * seq_len_q,),
            dtype=torch.int32,
            device="cuda",
            generator=generator,
        )
        enable_spec_decode = draft_len > 0 or (
            self.model_engine.spec_config is not None
            and self.model_engine.spec_config.spec_dec_mode.use_one_engine()
        )
        graph_runner = self.model_engine.cuda_graph_runner
        expect_graph = graph_runner is not None and graph_runner.enabled
        checked = [False]

        def run_pack(*, check=False, events=None, nvtx_message=None):
            # events=(start, end) and the NVTX range open just before the graph replay, after
            # the engine's host-side input preparation, so both cover the step's GPU work only.
            self._replayed = None
            self._start_event = events[0] if events is not None else None
            self._nvtx_message = nvtx_message if expect_graph else None
            with (
                nvtx.annotate(nvtx_message)
                if nvtx_message is not None and not expect_graph
                else contextlib.nullcontext()
            ):
                with torch.inference_mode():
                    outputs = self.model_engine.forward(
                        batch,
                        self.resource_manager,
                        None,
                        enable_spec_decode=enable_spec_decode,
                        runtime_draft_len=draft_len,
                    )
                if events is not None:
                    events[1].record()
            if self._nvtx_message is not None and self._replayed is not None:
                nvtx.pop_range()
            self._start_event = None
            self._nvtx_message = None
            if expect_graph and not checked[0]:
                # A miss runs eagerly (other MoE kernels plus launch overhead); a padded replay
                # runs a larger batch. Either measures another shape.
                key = self._replayed
                if key is None or (key.batch_size, key.draft_len) != (batch_size, draft_len):
                    raise RuntimeError(
                        f"batch_size {batch_size} draft_len {draft_len} replayed "
                        f"{'no CUDA graph' if key is None else key}; capture this batch size "
                        "(batch_size goes into llm_args.cuda_graph_config.batch_sizes)"
                    )
                checked[0] = True
            if check and isinstance(outputs, dict):
                for name, value in outputs.items():
                    if isinstance(value, torch.Tensor) and value.is_floating_point():
                        if not torch.isfinite(value).all():
                            raise ValueError(f"non-finite values in output {name}")
            return outputs

        return run_pack

    @contextlib.contextmanager
    def replace_routing_method_ctx(self, balance_method, balance_ratio):
        if balance_method.name != "NotModified":
            raise NotImplementedError(
                "the engine path runs the model's own routing; use balance_method NotModified"
            )
        yield

    def release_batch(self) -> None:
        self._batch_stack.close()
        self._batch_stack = contextlib.ExitStack()

    def shutdown(self) -> None:
        self.release_batch()
        self.executor.shutdown()
        logger.info("Layer-wise benchmarks: engine executor shut down")
