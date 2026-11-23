# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The checkpoint stand-in the target's CPU tests read geometry from.

`dims` reads a `core` whose only inputs are the checkpoint's `config.json` and
one member's `Mapping`; these helpers build that pair without a weight or a
device, so a test asks the served checkpoint for its own numbers instead of
restating them. `requires_checkpoint` skips a test whose subject is the
checkpoint itself where the config is not on disk.

`mla_layer` and `Step` are the pair the MLA tests call a layer's entry points
with: a layer whose weights are never read, and one step's metadata as plain
values. `ContextStep` is the prefill one, whose requests carry a cached prefix,
and `BlockStep` the drafter's, a mixed batch whose generating requests carry the
block's own end. `drafter` is that stand-in turned into the target's own DFly2
drafter, for the tests of the drafter's module and of the worker that runs it.
"""

from __future__ import annotations

import functools
import itertools
import json
import os
from collections import defaultdict
from types import SimpleNamespace

import pytest
import torch
from utils.llm_data import llm_models_root

from tensorrt_llm._torch.modeling_v2.models.hyv4_flash_for_conditional_generation.hf_003107f_orig__sm_103__unified.dims import (  # noqa: E501
    dims,
)
from tensorrt_llm._torch.modeling_v2.models.hyv4_flash_for_conditional_generation.hf_003107f_orig__sm_103__unified.draft import (  # noqa: E501
    Hyv4FlashDraft,
)
from tensorrt_llm._torch.modeling_v2.models.hyv4_flash_for_conditional_generation.hf_003107f_orig__sm_103__unified.modeling import (  # noqa: E501
    MlaLayer,
)

#: The checkpoint: `HYV4_FLASH_CKPT`, else `Hy4-flash-FP8` under the models root, the
#: directory the accuracy tests load.
_MODELS_ROOT = llm_models_root()
CKPT = os.environ.get(
    "HYV4_FLASH_CKPT",
    str(_MODELS_ROOT / "Hy4-flash-FP8") if _MODELS_ROOT is not None else "",
)

#: Skip a test that reads `CKPT` where the checkpoint is not on disk.
requires_checkpoint = pytest.mark.skipif(
    not os.path.exists(os.path.join(CKPT, "config.json")),
    reason=f"checkpoint config.json not found at {CKPT}",
)


class Attr(dict):
    """A JSON object read both ways: `cfg.hidden_size` and `rope_parameters[layer_type]`.

    The config class exposes the checkpoint's nested blocks as dicts, so the
    test's stand-in must keep `.get` as well as attribute access.
    """

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name) from None


def ns(value):
    """`config.json`'s nested dicts as attribute objects, as the config class reads them."""
    if isinstance(value, dict):
        return Attr({k: ns(v) for k, v in value.items()})
    if isinstance(value, list):
        return [ns(v) for v in value]
    return value


def checkpoint_config():
    """The published checkpoint's `config.json`, read the way `dims` reads it."""
    with open(os.path.join(CKPT, "config.json")) as f:
        return ns(json.load(f))


def core(tp=4, ep=4, attn_dp=False, world=4, rank=0, moe_tp=1, max_num_tokens=8192):
    """A `core` over the checkpoint's config and one member's `Mapping`.

    `max_num_tokens` is the engine's per-step token budget, which `dims` reads
    for the MoE runner's autotuning range; the default is the one the gates'
    engines serve at.
    """
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
    """One MLA layer with no weight behind it: the entry args only read geometry."""
    d = dims(core())
    core_stub = type("Core", (), {})()
    core_stub.dims = d
    core_stub.w = defaultdict(lambda: torch.zeros((1,)))
    core_stub._pool_row = staticmethod(lambda md, layer: layer)
    # The un-speculated core: one generation row per request.
    core_stub.gen_width = 1
    return MlaLayer(core_stub, i)


@functools.lru_cache(maxsize=None)
def drafter() -> Hyv4FlashDraft:
    """A `Hyv4FlashDraft` over the checkpoint's config and the tep4 mapping, on CPU.

    The declared tensors are allocated but never filled, so this is the module's
    wiring and not its weights; one construction is shared by the tests below.
    `dims` reads the engine's `torch_dtype` as a dtype -- `declare` allocates
    with it -- while the stand-in keeps the JSON spelling, so it is converted
    here.
    """
    stand_in = core()
    stand_in.model_config.torch_dtype = torch.bfloat16
    return Hyv4FlashDraft(stand_in)


class Step:
    """The metadata one generation call reads, as plain values.

    Every field is a CPU tensor or an int, so a step can be built without an
    engine; the args a layer hands the entry read nothing else.
    """

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
    """A prefill step: request `s` owns `cached[s]` pool rows plus `new[s]` appended now.

    The engine's own quantities, as `attention/backends/trtllm.py` and the
    sparse MLA metadata write them: `host_total_kv_lens[0]` is the sum of the
    context requests' KV lengths -- the cached prefix included -- `ctx_kv_indptr`
    the same sum as a per-request cumsum, and `ctx_cached_token_indptr` the
    cached prefix alone. Only the `new` rows are this step's query rows.
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


class BlockStep(Step):
    """The step a draft block runs in: `num_contexts` context requests, then the block's.

    The engine hands the drafter the *target's* metadata, so a generating
    request's device KV length here is the block's own exclusive end -- what
    `spec_worker._write_block_kv_lens` leaves in `md.kv_lens_cuda_runtime`, the
    length the prologue places the block's rows with -- while the context
    requests keep the step's own. One request contributes `block_width` rows to
    the block, and how many of a row's selected columns are live is that row's
    count in `DflyBlockSelection.lengths`, not this tensor.
    """

    def __init__(self, lengths, num_contexts: int = 0, max_seq_len: int = 5120) -> None:
        super().__init__(int(lengths[-1]), max_seq_len)
        per_request = torch.tensor(lengths, dtype=torch.int32)
        self.num_contexts = num_contexts
        self.max_num_requests = len(lengths)
        self.kv_lens_cuda_runtime = per_request.clone()
        self.kv_lens_runtime = per_request.clone()
        self.prompt_lens_cuda_runtime = per_request.clone()
        self.prompt_lens_cpu_runtime = per_request.clone()
        self.host_total_kv_lens = torch.tensor(
            [sum(lengths[:num_contexts]), sum(lengths[num_contexts:])], dtype=torch.int32
        )
        self.host_request_types_runtime = torch.tensor(
            [1] * num_contexts + [0] * (len(lengths) - num_contexts), dtype=torch.int32
        )
