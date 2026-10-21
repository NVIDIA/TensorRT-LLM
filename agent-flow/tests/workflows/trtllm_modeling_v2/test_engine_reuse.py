# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Block reuse is attested by what the engine reused: a gate that asks for it fails on none."""

from __future__ import annotations

import sys
import textwrap
from pathlib import Path

from agent_flow.workflows.trtllm_modeling_v2.checks import gates
from agent_flow.workflows.trtllm_modeling_v2.infra.environment import RunEnvironment

from .test_engine_attestation import _FAKE_TRTLLM, _task

_FAKE_LLM = """
    import os
    import types

    from tensorrt_llm.executor.base_worker import BaseWorker


    class MambaStateConfig:
        def __init__(self, periodic_snapshot_interval=0,
                     additional_snapshot_offsets_from_start=(),
                     additional_snapshot_offsets_from_end=()):
            self.periodic_snapshot_interval = periodic_snapshot_interval
            self.additional_snapshot_offsets_from_start = list(
                additional_snapshot_offsets_from_start)
            self.additional_snapshot_offsets_from_end = list(
                additional_snapshot_offsets_from_end)


    class KvCacheConfig:
        def __init__(self, **fields):
            self._stated = set(fields)
            self.enable_block_reuse = fields.get("enable_block_reuse", True)
            self.dtype = fields.get("dtype", "auto")
            self.mamba_ssm_cache_dtype = fields.get("mamba_ssm_cache_dtype", "auto")
            self.free_gpu_memory_fraction = fields.get("free_gpu_memory_fraction", 0.9)
            self.tokens_per_block = fields.get("tokens_per_block", 32)
            state = fields.get("mamba_state_config") or {}
            if isinstance(state, dict):
                state = MambaStateConfig(**state)
            self.mamba_state_config = state
            self.max_gpu_total_bytes = 0

        @property
        def model_fields_set(self):
            return self._stated


    class Tokenizer:
        def encode(self, text, add_special_tokens=True):
            return [ord(c) for c in text]


    class RequestOutput:
        # As the LLM API's: the handle `generate_async` returns is the output itself.

        def __init__(self, ids, cached):
            self.prompt_token_ids = ids
            self.cached_tokens = cached
            self.outputs = [types.SimpleNamespace(
                token_ids=[11, 12], text=" Paris", finish_reason="length")]

        def __iter__(self):
            yield self

        def result(self):
            return self


    class LLM:
        # Resumes a prompt where an earlier one left a recurrent-state snapshot, as a
        # hybrid engine does; FAKE_REUSE=0 never resumes, FAKE_WORKER=0 sets its worker
        # up in another process.

        def __init__(self, model=None, **kw):
            kv = kw.get("kv_cache_config") or KvCacheConfig()
            if isinstance(kv, dict):
                kv = KvCacheConfig(**kv)
            parallel = types.SimpleNamespace(
                tp_size=kw.get("tensor_parallel_size", 1), pp_size=1,
                moe_ep_size=kw.get("moe_expert_parallel_size") or -1, moe_tp_size=-1,
                enable_attention_dp=kw.get("enable_attention_dp", False))
            self.args = types.SimpleNamespace(
                modeling_v2=kw.get("modeling_v2", "off"), kv_cache_config=kv,
                cuda_graph_config=kw.get("cuda_graph_config"), speculative_config=None,
                max_num_tokens=kw.get("max_num_tokens", 8192),
                max_batch_size=kw.get("max_batch_size", 2048),
                max_seq_len=kw.get("max_seq_len"),
                enable_chunked_prefill=kw.get("enable_chunked_prefill", False),
                disable_overlap_scheduler=False, parallel_config=parallel)
            self.tokenizer = Tokenizer()
            self.seen = []
            if os.environ.get("FAKE_WORKER", "1") == "1":
                BaseWorker(self.args).setup_engine()

        def get_kv_cache_capacity(self):
            return {"maxNumBlocks": 100, "tokensPerBlock": 32, "maxNumTokens": 3200}

        def _resumed(self, ids):
            kv = self.args.kv_cache_config
            if os.environ.get("FAKE_REUSE", "1") != "1" or not kv.enable_block_reuse:
                return 0
            shared = 0
            for earlier in self.seen:
                n = 0
                while n < min(len(ids), len(earlier)) and ids[n] == earlier[n]:
                    n += 1
                shared = max(shared, n)
            limit = min(shared, len(ids) - 1)
            state = kv.mamba_state_config
            points = [o for o in state.additional_snapshot_offsets_from_start if o <= limit]
            if state.periodic_snapshot_interval:
                points += list(range(state.periodic_snapshot_interval, limit + 1,
                                     state.periodic_snapshot_interval))
            return max(points, default=0)

        def generate_async(self, inputs, params=None, streaming=False, **kwargs):
            if isinstance(inputs, dict):
                ids = list(inputs["prompt_token_ids"])
            else:
                ids = self.tokenizer.encode(inputs)
            cached = self._resumed(ids)
            self.seen.append(ids)
            return RequestOutput(ids, cached)

        def shutdown(self):
            pass
"""

_FAKE_QUICKSTART = """
    import argparse
    import types

    from tensorrt_llm import LLM, KvCacheConfig


    def parse_arguments():
        parser = argparse.ArgumentParser()
        parser.add_argument("--model_dir")
        parser.add_argument("--max_tokens", type=int, default=64)
        parser.add_argument("--tp_size", type=int, default=1)
        parser.add_argument("--moe_ep_size", type=int, default=-1)
        parser.add_argument("--enable_attention_dp", action="store_true")
        parser.add_argument("--max_seq_len", type=int)
        parser.add_argument("--max_batch_size", type=int, default=2048)
        parser.add_argument("--max_num_tokens", type=int, default=8192)
        parser.add_argument("--kv_cache_fraction", type=float, default=0.9)
        parser.add_argument("--mamba_ssm_cache_dtype", default="auto")
        parser.add_argument("--kv_cache_dtype", default="auto")
        parser.add_argument("--disable_kv_cache_reuse", action="store_true")
        parser.add_argument("--enable_chunked_prefill", action="store_true")
        parser.add_argument("--use_cuda_graph", action="store_true")
        parser.add_argument("--cuda_graph_padding_enabled", action="store_true")
        parser.add_argument("--cuda_graph_batch_sizes", type=int, nargs="+")
        return parser.parse_args()


    def setup_llm(args):
        kv = KvCacheConfig(
            enable_block_reuse=not args.disable_kv_cache_reuse,
            free_gpu_memory_fraction=args.kv_cache_fraction,
            dtype=args.kv_cache_dtype,
            mamba_ssm_cache_dtype=args.mamba_ssm_cache_dtype,
        )
        graphs = None
        if args.use_cuda_graph:
            graphs = types.SimpleNamespace(
                batch_sizes=sorted(args.cuda_graph_batch_sizes or [1]),
                enable_padding=args.cuda_graph_padding_enabled)
        llm = LLM(
            model=args.model_dir, kv_cache_config=kv, cuda_graph_config=graphs,
            tensor_parallel_size=args.tp_size, moe_expert_parallel_size=args.moe_ep_size,
            enable_attention_dp=args.enable_attention_dp, max_seq_len=args.max_seq_len,
            max_batch_size=args.max_batch_size, max_num_tokens=args.max_num_tokens,
            enable_chunked_prefill=args.enable_chunked_prefill)
        return llm, types.SimpleNamespace(max_tokens=args.max_tokens, ignore_eos=False)
"""


def _fake_site(tmp_path: Path) -> tuple[Path, Path]:
    """``(site, repo)``: a fake ``tensorrt_llm`` whose engine reuses, and a tree with its quickstart."""
    site = tmp_path / "site"
    files = {**_FAKE_TRTLLM, "tensorrt_llm/__init__.py": _FAKE_LLM}
    for name, text in files.items():
        path = site / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(textwrap.dedent(text), encoding="utf-8")
    repo = tmp_path / "repo"
    script = repo / "examples" / "llm-api" / "quickstart_advanced.py"
    script.parent.mkdir(parents=True)
    script.write_text(textwrap.dedent(_FAKE_QUICKSTART), encoding="utf-8")
    return site, repo


def _real(site: Path, **env) -> RunEnvironment:
    return RunEnvironment(
        kind="local", python=sys.executable, world_size=1, env={"PYTHONPATH": str(site), **env}
    )


def _real_boot(tmp_path: Path, **env) -> gates.GateResult:
    site, repo = _fake_site(tmp_path)
    task = _task(tmp_path, trtllm_repo_path=str(repo))
    return gates.run_boot_gate(
        _real(site, **env), task_data=task, unit="core", member="tp1", workdir=tmp_path / "gates"
    )


def test_a_real_boot_fails_on_an_engine_that_reused_nothing(tmp_path):
    for worker in ("1", "0"):
        result = _real_boot(tmp_path / worker, FAKE_WORKER=worker, FAKE_REUSE="0")
        assert not result.passed, result.stdout_tail
        assert "the engine reused nothing: the reuse probe sent one 320-token prompt" in (
            result.detail
        )
