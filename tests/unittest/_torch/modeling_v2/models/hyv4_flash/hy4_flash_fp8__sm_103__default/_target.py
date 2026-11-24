# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU stand-ins shared by this target's tests: a `core` over the checkpoint's `config.json` and
one member's `Mapping`, a weightless MLA layer, and one step's attention metadata as plain values."""

from __future__ import annotations

import itertools
import json
import os
from collections import defaultdict
from types import SimpleNamespace

import pytest
import torch
from utils.llm_data import llm_models_root

from tensorrt_llm._torch.modeling_v2.models.hyv4_flash.hy4_flash_fp8__sm_103__default.dims import (  # noqa: E501
    dims,
)
from tensorrt_llm._torch.modeling_v2.models.hyv4_flash.hy4_flash_fp8__sm_103__default.modeling import (  # noqa: E501
    MlaLayer,
)

_MODELS_ROOT = llm_models_root()
CKPT = os.environ.get(
    "HYV4_FLASH_CKPT",
    str(_MODELS_ROOT / "Hy4-flash-FP8") if _MODELS_ROOT is not None else "",
)

requires_checkpoint = pytest.mark.skipif(
    not os.path.exists(os.path.join(CKPT, "config.json")),
    reason=f"checkpoint config.json not found at {CKPT}",
)


class Attr(dict):
    """A JSON object readable as attributes and as a dict, like the config class's nested blocks."""

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name) from None


def ns(value):
    if isinstance(value, dict):
        return Attr({k: ns(v) for k, v in value.items()})
    if isinstance(value, list):
        return [ns(v) for v in value]
    return value


def checkpoint_config():
    with open(os.path.join(CKPT, "config.json")) as f:
        return ns(json.load(f))


def core(tp=4, ep=4, attn_dp=False, world=4, rank=0, moe_tp=1, max_num_tokens=8192):
    """A `core` over the checkpoint's config and one member's `Mapping`."""
    mapping = SimpleNamespace(
        tp_size=tp,
        tp_rank=rank,
        enable_attention_dp=attn_dp,
        moe_ep_size=ep,
        moe_ep_rank=rank % ep,
        moe_tp_size=moe_tp,
        moe_tp_rank=0,
        world_size=world,
        rank=rank,
    )
    model_config = SimpleNamespace(
        pretrained_config=checkpoint_config(),
        mapping=mapping,
        torch_dtype="bfloat16",
        tokens_per_block=32,
        max_num_tokens=max_num_tokens,
        max_seq_len=4096,
        max_batch_size=8,
        max_context_length=4096,
    )
    return SimpleNamespace(model_config=model_config)


def mla_layer(i: int) -> MlaLayer:
    """One MLA layer with no weight behind it, on the un-speculated core."""
    d = dims(core())
    core_stub = type("Core", (), {})()
    core_stub.dims = d
    core_stub.w = defaultdict(lambda: torch.zeros((1,)))
    core_stub._pool_row = staticmethod(lambda md, layer: layer)
    core_stub.gen_width = 1
    return MlaLayer(core_stub, i)


class Step:
    """The metadata one generation call reads: one request with `kv` keys."""

    def __init__(self, kv: int, max_seq_len: int) -> None:
        one = (1,)
        self.num_contexts = 0
        self.max_seq_len = max_seq_len
        self.max_context_length = max_seq_len
        self.max_num_requests = 1
        self.kv_lens_cuda_runtime = torch.full(one, kv, dtype=torch.int32)
        self.kv_lens_runtime = torch.full(one, kv, dtype=torch.int32)
        self.host_total_kv_lens = torch.zeros((2,), dtype=torch.int32)
        self.prompt_lens_cuda_runtime = torch.zeros(one, dtype=torch.int32)
        self.prompt_lens_cpu_runtime = torch.zeros(one, dtype=torch.int32)
        self.host_request_types_runtime = torch.zeros(one, dtype=torch.int32)
        self.kv_cache_block_offsets = None
        self.host_kv_cache_pool_pointers = None
        self.host_kv_cache_pool_mapping = None
        self.effective_workspace = None


class ContextStep(Step):
    """A prefill step: request `s` has `cached[s]` pool rows and appends `new[s]` now.

    `host_total_kv_lens[0]` and `ctx_kv_indptr` count the cached prefix too, as the engine writes
    them; only the `new` rows are query rows.
    """

    def __init__(
        self, cached: tuple[int, ...], new: tuple[int, ...], max_seq_len: int = 5120
    ) -> None:
        assert len(cached) == len(new)
        total = [c + n for c, n in zip(cached, new)]
        super().__init__(sum(total), max_seq_len)
        self.rows = sum(new)
        self.num_contexts = len(cached)
        self.ctx_cached_token_indptr = torch.tensor(
            [0, *itertools.accumulate(cached)], dtype=torch.int32
        )
        self.ctx_kv_indptr = torch.tensor([0, *itertools.accumulate(total)], dtype=torch.int32)
        self.host_total_kv_lens = torch.tensor([sum(total)], dtype=torch.int32)
        self.kv_lens_cuda_runtime = torch.tensor(total, dtype=torch.int32)
        self.kv_lens_runtime = torch.tensor(total, dtype=torch.int32)
        self.prompt_lens_cuda_runtime = torch.tensor(total, dtype=torch.int32)
        self.prompt_lens_cpu_runtime = torch.tensor(total, dtype=torch.int32)
        self.max_ctx_seq_len = max(total)
        self.max_ctx_kv_len = max(total)
