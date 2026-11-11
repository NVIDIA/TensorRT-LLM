# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Attestation: a deployment's options against what every rank's engine ran."""

from __future__ import annotations

import enum
import json
import sys
import textwrap
from pathlib import Path
from types import SimpleNamespace

import pytest

from agent_flow.workflows.trtllm_modeling_v2.checks import attest
from agent_flow.workflows.trtllm_modeling_v2.checks import boot as boot_checks
from agent_flow.workflows.trtllm_modeling_v2.infra.compute.environment import RunEnvironment
from agent_flow.workflows.trtllm_modeling_v2.spec.deployments import Deployment, Server

from ._helpers import TARGET_CLASS, rank_lines

OPTIONS = {
    "tensor_parallel_size": 2,
    "enable_attention_dp": True,
    "max_batch_size": 64,
    "cuda_graph_config": {},
    "speculative_config": None,
    "kv_cache_config": {"dtype": "nvfp4", "free_gpu_memory_fraction": 0.6},
    "a_dtype_option_added_tomorrow": "fp8",
}
#: What the engine resolved: every stated key, with the defaults it filled in around them.
RESOLVED = {
    **OPTIONS,
    "cuda_graph_config": {"batch_sizes": [1, 2], "enable_padding": True},
    "kv_cache_config": {"dtype": "nvfp4", "free_gpu_memory_fraction": 0.6, "tokens_per_block": 32},
    "max_seq_len": None,
}


def test_declared_equals_observed_on_every_rank_with_no_option_known_by_name():
    found = attest.attest(rank_lines(RESOLVED), OPTIONS, tree="/repo")
    assert found.status == attest.MATCH and found.failure() is None
    assert (found.ranks, found.keys) == (2, 8)
    assert "modeling_v2" in found.target and found.origin == "/repo/tensorrt_llm/__init__.py"


@pytest.mark.parametrize(
    "resolved,said",
    [
        ({"max_batch_size": 32}, "rank 0: max_batch_size is 32, declared 64"),
        ({"kv_cache_config": {"dtype": "fp8"}}, "kv_cache_config.dtype is 'fp8', declared 'nvfp4'"),
        ({"cuda_graph_config": None}, "cuda_graph_config is None, declared {}"),
        (
            {"speculative_config": {"decoding_type": "MTP"}},
            "speculative_config is {'decoding_type'",
        ),
        ({"a_dtype_option_added_tomorrow": "auto"}, "a_dtype_option_added_tomorrow is 'auto'"),
        # Text is compared as written: a path or a name that differs by case is another one.
        ({"a_dtype_option_added_tomorrow": "FP8"}, "a_dtype_option_added_tomorrow is 'FP8'"),
    ],
)
def test_a_key_the_engine_resolved_otherwise_fails_the_check(resolved, said):
    found = attest.attest(rank_lines({**RESOLVED, **resolved}), OPTIONS)
    assert found.status == attest.MISMATCH and said in found.failure()


def test_a_value_that_leaves_the_choice_to_the_engine_records_what_each_rank_ran():
    declared = {
        "cache_transceiver_config": {"backend": "DEFAULT"},
        "kv_cache_config": {"dtype": "Auto"},
        "max_batch_size": 4,
    }
    ran = {
        "cache_transceiver_config": {"backend": "NIXL", "max_tokens_in_buffer": 4096},
        "kv_cache_config": {"dtype": "fp8"},
        "max_batch_size": 4,
    }
    found = attest.attest(rank_lines(ran), declared)
    assert found.status == attest.MATCH and found.failure() is None
    assert found.resolved == (
        "cache_transceiver_config.backend: declared DEFAULT, ran NIXL",
        "kv_cache_config.dtype: declared Auto, ran fp8",
    )
    assert "target " in attest.describe([found.record()])
    assert attest.describe([found.record()]).endswith(
        "; cache_transceiver_config.backend: declared DEFAULT, ran NIXL; "
        "kv_cache_config.dtype: declared Auto, ran fp8]"
    )
    # Every other value is still compared exactly, and a block the engine has off is no choice.
    assert attest.attest(rank_lines({**ran, "max_batch_size": 8}), declared).status == (
        attest.MISMATCH
    )
    off = attest.attest(rank_lines({**ran, "cache_transceiver_config": None}), declared)
    assert "cache_transceiver_config.backend is None, declared 'DEFAULT'" in off.failure()
    named = {**declared, "cache_transceiver_config": {"backend": "UCX"}}
    assert "is 'NIXL', declared 'UCX'" in attest.attest(rank_lines(ran), named).failure()
    almost = attest.attest(rank_lines({"backend": "NIXL"}), {"backend": "defaults"})
    assert almost.status == attest.MISMATCH and not almost.resolved


def test_a_missing_rank_or_key_reads_unknown_and_never_passes():
    one_rank = attest.attest(rank_lines(RESOLVED, ranks=1), OPTIONS)
    assert one_rank.status == attest.UNKNOWN
    assert "rank 1: it printed no record" in one_rank.failure()
    assert "unknown is not a pass" in one_rank.failure()
    no_key = {k: v for k, v in RESOLVED.items() if k != "enable_attention_dp"}
    assert "rank 0: enable_attention_dp is not in the engine's configuration" in (
        attest.attest(rank_lines(no_key), OPTIONS).failure()
    )
    unread = attest.attest(rank_lines(RESOLVED, options=None, error="AttributeError: x"), OPTIONS)
    assert "rank 0: AttributeError: x" in unread.failure()
    assert attest.attest("", OPTIONS).status == attest.UNKNOWN


def test_the_model_class_and_the_import_origin_are_attested_on_every_rank():
    stock = rank_lines(RESOLVED, target="tensorrt_llm._torch.models.modeling_qwen3.Qwen3")
    assert "not a Modeling V2 target" in attest.attest(stock, OPTIONS).failure()
    # A control run on the stock path is not held to a Modeling V2 class.
    assert attest.attest(stock, OPTIONS, target=False).status == attest.MATCH
    assert (
        "the model class it built"
        in attest.attest(rank_lines(RESOLVED, target=None), OPTIONS).failure()
    )
    elsewhere = attest.attest(rank_lines(RESOLVED, tree="/opt/other"), OPTIONS, tree="/repo")
    assert "imported tensorrt_llm from /opt/other/tensorrt_llm/__init__.py" in elsewhere.failure()


def test_a_value_inside_a_list_and_every_rank_of_every_parallel_size_are_held():
    inputs = [{"name": "token_type_ids", "dtype": "int32"}]
    declared = {"cuda_graph_config": {"extra_model_inputs": inputs}}
    other = {"cuda_graph_config": {"extra_model_inputs": [{**inputs[0], "dtype": "float32"}]}}
    assert attest.attest(rank_lines(declared), declared).status == attest.MATCH
    assert attest.attest(rank_lines(other), declared).status == attest.MISMATCH
    # An engine's ranks are its tensor, pipeline and context parallel sizes, multiplied.
    wide = {"tensor_parallel_size": 2, "pipeline_parallel_size": 2}
    assert Server(options=wide).ranks == 4
    half = attest.attest(rank_lines(wide, ranks=2), wide)
    assert (half.status, half.ranks) == (attest.UNKNOWN, 4)
    assert attest.attest(rank_lines(wide, ranks=4), wide).status == attest.MATCH
    # The model class is in the Modeling V2 package, not one whose name contains it.
    lookalike = rank_lines({}, target="tensorrt_llm._torch.models.not_modeling_v2.Stock")
    assert attest.attest(lookalike, {}).status == attest.MISMATCH


def test_a_repeated_prompt_must_come_back_with_cached_tokens():
    assert attest.reuses({"kv_cache_config": {"enable_block_reuse": True}})
    assert not attest.reuses(
        {"kv_cache_config": {"enable_block_reuse": False}}
    ) and not attest.reuses({})
    # One repeat; with attention DP a request may land on any rank, so one more per rank.
    assert attest.reuse_sends({"tensor_parallel_size": 4}) == 2
    assert attest.reuse_sends({"tensor_parallel_size": 4, "enable_attention_dp": True}) == 5
    assert attest.reuse_shortfall([0, 448]) is None
    assert attest.reuse_shortfall([0, 0, 0, 448, 0]) is None
    assert "the engine reused nothing" in attest.reuse_shortfall([0, 0])
    assert "unknown is not a pass" in attest.reuse_shortfall([0, None])
    assert "the engine reused nothing" in attest.reuse_shortfall([0, True])  # no count of tokens


def test_a_rank_prints_every_field_of_its_configuration_under_each_name_it_answers_to(
    monkeypatch,
):
    hook = Path(attest.__file__).parent / "harness" / "rankhook" / "sitecustomize.py"
    monkeypatch.setattr(sys, "path", [])  # the hook hands on to no other sitecustomize here
    monkeypatch.delenv("MV2FLOW_RANK_FACTS", raising=False)
    module: dict = {"__name__": "rankhook", "__file__": str(hook)}
    exec(compile(hook.read_text(encoding="utf-8"), str(hook), "exec"), module)

    class Dtype(enum.Enum):
        FP8 = "fp8"

    class Field:
        def __init__(self, alias=None, choices=None):
            self.alias = alias
            self.validation_alias = SimpleNamespace(choices=choices) if choices else None

    class Kv:
        model_fields = {"dtype": Field(), "free_gpu_memory_fraction": Field()}
        dtype, free_gpu_memory_fraction, _private = Dtype.FP8, 0.9, "unseen"

    class Draft:
        model_fields = {"speculative_model": Field(choices=["speculative_model_dir"])}
        speculative_model = Path("/ckpt/draft")

    class Args:
        model_fields = {
            "kv_cache_config": Field(),
            "speculative_config": Field(),
            "n": Field("num"),
        }
        kv_cache_config, speculative_config, n = Kv(), Draft(), 4

    class Tokenizer:
        def __init__(self):
            self.vocab = {str(n): n for n in range(100_000)}

        def __str__(self):
            return "<tokenizer>"

    # Only the configuration is walked: an object that is none is named, never dumped.
    loose = SimpleNamespace(max_seq_len=8, tokenizer=Tokenizer(), table=list(range(1000)))
    assert module["_plain"](loose) == {
        "max_seq_len": 8,
        "tokenizer": "<tokenizer>",
        "table": "<list of 1000>",
    }
    resolved = module["_plain"](Args())
    assert resolved == {
        "kv_cache_config": {"dtype": "fp8", "free_gpu_memory_fraction": 0.9},
        "speculative_config": {
            "speculative_model": "/ckpt/draft",
            "speculative_model_dir": "/ckpt/draft",
        },
        "n": 4,
        "num": 4,
    }
    # What the comparison then reads of it: a key spelled by its alias is known.
    declared = {"speculative_config": {"speculative_model_dir": "/ckpt/draft"}, "num": 4}
    record = {"rank": 0, "options": resolved, "target": TARGET_CLASS, "origin": "/r/t/__init__.py"}
    line = f"MV2FLOW_RANK_ENGINE {json.dumps(record)}\n"
    assert attest.attest(line, declared).status == attest.MATCH


_FAKE_TRTLLM = {
    "tensorrt_llm/executor/__init__.py": "",
    "tensorrt_llm/executor/base_worker.py": """
        import types


        class Model:
            pass


        Model.__module__ = "tensorrt_llm._torch._experimental.modeling_v2.models.fake.model"


        class BaseWorker:
            def __init__(self, llm_args):
                self.llm_args = llm_args
                self.rank = 0
                self.engine = None

            def setup_engine(self):
                self.engine = types.SimpleNamespace(
                    model_engine=types.SimpleNamespace(model=Model()))
    """,
    "tensorrt_llm/__init__.py": """
        import os
        import types

        from tensorrt_llm.executor.base_worker import BaseWorker


        class SamplingParams:
            def __init__(self, **fields):
                self.__dict__.update(fields)


        class Output:
            def __init__(self, ids, cached):
                self.prompt_token_ids = ids
                self.cached_tokens = cached
                self.outputs = [types.SimpleNamespace(
                    token_ids=[11, 12], text=" Paris", finish_reason="length")]

            def __iter__(self):
                yield self


        class LLM:
            # Resumes a prompt it has seen (FAKE_REUSE=0: never); FAKE_DTYPE is the KV dtype
            # its engine resolves whatever was asked; FAKE_WORKER=0 sets no worker up here.

            def __init__(self, model=None, **options):
                if os.environ.get("FAKE_DTYPE"):
                    options["kv_cache_config"]["dtype"] = os.environ["FAKE_DTYPE"]
                self.seen = []
                self.tokenizer = None
                if os.environ.get("FAKE_WORKER", "1") == "1":
                    BaseWorker(types.SimpleNamespace(**options)).setup_engine()

            def generate_async(self, inputs, params=None, streaming=False, **kwargs):
                again = inputs in self.seen and os.environ.get("FAKE_REUSE", "1") == "1"
                self.seen.append(inputs)
                return Output(list(range(len(inputs.split()))), 448 if again else 0)

            def shutdown(self):
                pass
    """,
}


def _real_boot(tmp_path: Path, **env):
    """A boot through the real engine driver and rank hook, on a stand-in ``tensorrt_llm``."""
    repo = tmp_path / "repo"
    for name, text in _FAKE_TRTLLM.items():
        path = repo / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(textwrap.dedent(text), encoding="utf-8")
    options = {"kv_cache_config": {"dtype": "fp8", "enable_block_reuse": True}}
    deployment = Deployment(name="tp1", core=True, server=Server(options=options))
    task = {"checkpoint_path": "/ckpt/model", "trtllm_repo_path": str(repo)}
    environment = RunEnvironment(
        kind="local", python=sys.executable, world_size=1, env={"PYTHONPATH": str(repo), **env}
    )
    return boot_checks.run_boot_gate(
        environment, task_data=task, deployment=deployment, workdir=tmp_path / "gates"
    )


def test_a_real_boot_is_held_to_what_its_rank_printed_and_to_a_cache_that_reused(tmp_path):
    passed = _real_boot(tmp_path / "ok")
    assert passed.passed, passed.detail + passed.stdout_tail
    assert "a repeated prompt resumed 448 cached tokens" in passed.detail
    assert passed.engine[0]["status"] == attest.MATCH and passed.engine[0]["ranks"] == 1
    assert "[engine match: 1 rank(s) x" in passed.summary()

    other = _real_boot(tmp_path / "dtype", FAKE_DTYPE="auto")
    assert not other.passed
    assert "rank 0: kv_cache_config.dtype is 'auto', declared 'fp8'" in other.detail

    cold = _real_boot(tmp_path / "cold", FAKE_REUSE="0")
    assert not cold.passed and "the engine reused nothing" in cold.detail

    # An engine none of whose ranks said what it ran: unknown, not a pass.
    silent = _real_boot(tmp_path / "silent", FAKE_WORKER="0")
    assert not silent.passed and "rank 0: it printed no record" in silent.detail
