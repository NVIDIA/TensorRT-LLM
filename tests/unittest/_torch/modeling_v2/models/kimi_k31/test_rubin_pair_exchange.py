# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The specialized Kimi K3.1 Rubin targets keep the vocab-parallel pair exchange off the one-shot
all-reduce workspace.

A target's DFlash worker selects target tokens from this rank's vocab shard through the engine's
``VocabParallelHead``: one float32 all-reduce of ``(value, index)`` pairs. That all-reduce has the
LM head's mapping, so it shares the TP group's all-reduce workspaces with the target's own bf16
all-reduces. The one-shot Lamport kernel of the IPC workspace records the size to clear in
elements of its caller's dtype, and the next one-shot call clears that many of its own elements.
So a bf16 call after a float32 call clears half of the float32 call's bytes, and a later one-shot
call can read the other half as a peer's data. The MNNVL kernels record bytes to clear, and the
one-shot kernel never runs across nodes.

Where the exchange runs on these layouts:

* tep8: over the attention TP group, 8 ranks on two 4-GPU nodes, on both servers;
* helix8: over the Helix ranks taken as TP ranks, 8 ranks on two nodes;
* dep16: nowhere. Under attention DP the LM head is not vocab-parallel.
"""

from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import tensorrt_llm._torch.distributed.ops as dist_ops
import tensorrt_llm._torch.speculative.vocab_parallel as vocab_parallel
from tensorrt_llm._torch.distributed import AllReduceStrategy
from tensorrt_llm._torch.modules.embedding import LMHead
from tensorrt_llm._torch.modules.linear import TensorParallelMode
from tensorrt_llm._torch.modules.logits_processor import LogitsProcessor
from tensorrt_llm._torch.speculative.dflash import DFlashWorker
from tensorrt_llm.llmapi.llm_args import AdvancedSamplingMode
from tensorrt_llm.mapping import Mapping

pytestmark = pytest.mark.cpu_only

_FAMILY = (
    Path(vocab_parallel.__file__).resolve().parents[1] / "_experimental/modeling_v2/models/kimi_k31"
)
#: GPUs of one Rubin (VR200) node.
_GPUS_PER_NODE = 4
_VOCAB = 163840
#: The LM head's input width plays no part here.
_HIDDEN = 64
#: Each layout's mapping, as the family's routing table states it.
_LAYOUTS = {
    "tep8": dict(world_size=8, tp_size=8, moe_tp_size=1, moe_ep_size=8),
    "helix8": dict(
        world_size=8,
        tp_size=1,
        cp_size=8,
        cp_config={"cp_type": "HELIX"},
        moe_tp_size=2,
        moe_ep_size=4,
    ),
    "dep16": dict(world_size=16, tp_size=16, enable_attention_dp=True, moe_ep_size=16),
}


def _modeling(layout: str) -> ast.Module:
    """A specialized target's modeling.py."""
    path = _FAMILY / f"kimi_k31_pro_nvfp4__sm_107__{layout}" / "modeling.py"
    return ast.parse(path.read_text())


def _calls(node: ast.AST, name: str) -> list:
    """The calls of ``name``, as a bare name or as an attribute."""
    return [
        n
        for n in ast.walk(node)
        if isinstance(n, ast.Call)
        and name in (getattr(n.func, "id", None), getattr(n.func, "attr", None))
    ]


@pytest.mark.parametrize("layout", list(_LAYOUTS))
def test_worker_selects_through_the_engine_exchange(layout):
    """The worker builds no head, exchange or all-reduce: it reads the engine head's exchange."""
    tree = _modeling(layout)
    (worker,) = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "DFlashWorker"]
    methods = {n.name for n in worker.body if isinstance(n, ast.FunctionDef)}
    assert not methods & {"bind_target_lm_head", "setup_aliases", "target_logits"}
    for name in ("VocabParallelHead", "tp_pair_exchange", "_PairExchange", "AllReduce"):
        assert not _calls(worker, name), f"{layout}: the worker calls {name}"


@pytest.mark.parametrize("layout", _LAYOUTS)
def test_target_all_reduces_are_bf16_or_nccl(layout):
    """The TP group's one-shot workspace sees a single dtype from the target."""
    for call in _calls(_modeling(layout), "AllReduce"):
        kw = {k.arg: ast.unparse(k.value) for k in call.keywords}
        assert (
            kw.get("dtype") == "torch.bfloat16" or kw.get("strategy") == "AllReduceStrategy.NCCL"
        ), f"{layout}: line {call.lineno} builds AllReduce({kw})"


def _mapping(rank: int, **layout) -> Mapping:
    return Mapping(rank=rank, gpus_per_node=_GPUS_PER_NODE, **layout)


def _shell_lm_head(mapping: Mapping) -> LMHead:
    """The LM head ``DecoderModelForCausalLM`` builds over ``mapping`` (not tied to the embedding).

    helix8's shell passes its Helix ranks taken as TP ranks, so this takes that mapping too.
    """
    if mapping.has_cp_helix():
        mapping = mapping.repurpose_helix_cp_to_tp()
    mapping = mapping.mapping_for_full_rows()
    if mapping.enable_attention_dp and not mapping.enable_lm_head_tp_in_adp:
        return LMHead(_VOCAB, _HIDDEN, dtype=torch.bfloat16)
    return LMHead(
        _VOCAB,
        _HIDDEN,
        dtype=torch.bfloat16,
        mapping=mapping,
        tensor_parallel_mode=TensorParallelMode.COLUMN,
        gather_output=True,
        reduce_output=False,
    )


def _bind(lm_head: LMHead):
    """The engine's binding, which every specialized worker inherits, on a stand-in worker:
    greedy and temperature-only sampling, no penalties, no guided decoder."""
    worker = SimpleNamespace(
        guided_decoder=None,
        spec_config=SimpleNamespace(
            enable_penalty=False,
            use_rejection_sampling=False,
            advanced_sampling_mode=AdvancedSamplingMode.NO_TOPK_NO_TOPP,
        ),
        _vocab_parallel=None,
    )
    DFlashWorker.bind_target_lm_head(worker, lm_head, LogitsProcessor())
    return worker._vocab_parallel


def _rubin_is_mnnvl(mapping, dtype, explicitly_requested=False):
    """``MNNVLAllReduce.is_mnnvl`` on Rubin, where the platform checks (aarch64, MNNVL memory)
    pass; the rest is the engine's rule."""
    return (
        dtype in dist_ops.MNNVLAllReduce.get_supported_dtypes()
        and not mapping.has_cp()
        and (explicitly_requested or mapping.is_multi_node())
    )


class _RecordedAllReduce:
    """Stands in for the engine AllReduce: records how the exchange builds it, and whether the
    engine would give it the MNNVL all-reduce."""

    def __init__(self, mapping, strategy=AllReduceStrategy.AUTO, dtype=None):
        self.mapping = mapping
        self.strategy = strategy
        self.dtype = dtype
        mnnvl = strategy in (AllReduceStrategy.AUTO, AllReduceStrategy.MNNVL) and _rubin_is_mnnvl(
            mapping, dtype, explicitly_requested=strategy == AllReduceStrategy.MNNVL
        )
        self.mnnvl_allreduce = object() if mnnvl else None


@pytest.mark.parametrize("layout", list(_LAYOUTS))
@pytest.mark.parametrize("rank", [0, 5])
def test_layout_exchange_takes_mnnvl(monkeypatch, layout, rank):
    """tep8 and helix8 exchange over 8 ranks on two nodes, on the MNNVL all-reduce; dep16 has no
    exchange."""
    monkeypatch.setattr(dist_ops.MNNVLAllReduce, "is_mnnvl", staticmethod(_rubin_is_mnnvl))
    monkeypatch.setattr(vocab_parallel, "AllReduce", _RecordedAllReduce)

    selection = _bind(_shell_lm_head(_mapping(rank, **_LAYOUTS[layout])))

    if layout == "dep16":
        assert selection is None
        return
    exchange = selection.head._exchange
    all_reduce = exchange.all_reduce
    assert (exchange.tp_size, exchange.tp_rank) == (8, rank)
    assert all_reduce.dtype == torch.float32
    assert all_reduce.mapping.is_multi_node() and not all_reduce.mapping.has_cp()
    assert all_reduce.strategy == AllReduceStrategy.AUTO
    assert all_reduce.mnnvl_allreduce is not None


@pytest.mark.parametrize("tp_size", [2, 4])
def test_one_node_exchange_takes_no_workspace(monkeypatch, tp_size):
    """A TP group inside one node (no layout of these targets has one) gets the NCCL exchange: the
    real AllReduce takes neither the one-shot nor the MNNVL workspace."""

    def _no_workspace(*args, **kwargs):
        raise AssertionError("the pair exchange took the one-shot all-reduce workspace")

    monkeypatch.setattr(dist_ops.MNNVLAllReduce, "is_mnnvl", staticmethod(_rubin_is_mnnvl))
    monkeypatch.setattr(dist_ops, "get_allreduce_workspace", _no_workspace)

    selection = _bind(_shell_lm_head(_mapping(1, world_size=tp_size, tp_size=tp_size)))

    all_reduce = selection.head._exchange.all_reduce
    assert isinstance(all_reduce, dist_ops.AllReduce)
    assert all_reduce.strategy == AllReduceStrategy.NCCL
    assert all_reduce.workspace is None
    assert all_reduce.mnnvl_allreduce is None
