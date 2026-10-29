# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Every gate attests the engine it ran: what it reports is held to what the gate declared."""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from agent_flow.workflows.trtllm_modeling_v2.checks import acceptance as acceptance_checks
from agent_flow.workflows.trtllm_modeling_v2.checks import boot as boot_checks
from agent_flow.workflows.trtllm_modeling_v2.checks.harness import workload_io
from agent_flow.workflows.trtllm_modeling_v2.infra.compute.commands import CommandResult
from agent_flow.workflows.trtllm_modeling_v2.infra.compute.environment import RunEnvironment

#: A hybrid target's snapshot policy and runtime block, as a run's `task.yaml` states them.
_SNAPSHOTS = {
    "periodic_snapshot_interval": 2048,
    "additional_snapshot_offsets_from_start": [256, 512, 768],
}
_OPTIONS = {
    "kv_cache_config": {
        "free_gpu_memory_fraction": 0.6,
        "enable_block_reuse": True,
        "mamba_state_config": _SNAPSHOTS,
        "use_kv_cache_manager_v2": True,
    },
    "enable_chunked_prefill": True,
    "max_num_tokens": 8192,
    "cuda_graph_config": {"enable_padding": True, "max_batch_size": 128},
}
_RUNTIME = {
    "cuda_graph": True,
    "chunked_prefill": True,
    "overlap_scheduler": True,
    "kv_cache_reuse": True,
    "extra_flags": [
        "--max_seq_len=9216",
        "--max_batch_size=128",
        "--max_num_tokens=8192",
        "--kv_cache_fraction=0.6",
        "--mamba_ssm_cache_dtype=auto",
    ],
}
#: The 19 padded batches `CudaGraphConfig(enable_padding=True, max_batch_size=128)` captures.
_PADDED = [1, 2, 4, 8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120, 128]


def _task(tmp_path: Path, **extra) -> dict:
    options = tmp_path / "core.llm_api.yaml"
    options.write_text(yaml.safe_dump(_OPTIONS), encoding="utf-8")
    task = {
        "checkpoint_path": "/ckpt/model",
        "parallel": "unified",
        "trtllm_repo_path": str(tmp_path / "repo"),
        "features": [],
        "runtime": {"default": dict(_RUNTIME)},
        "accuracy": {"benchmark": "gsm8k", "llm_api_options": {"core": str(options)}},
    }
    task.update(extra)
    return task


def _env() -> RunEnvironment:
    return RunEnvironment(kind="local", python="/py", world_size=1, env={})


def _engine_config(**changes) -> dict:
    """What a correctly built core@tp1 engine reports."""
    config = {
        "modeling_v2": "require",
        "block_reuse": True,
        "mamba_snapshot": {
            **_SNAPSHOTS,
            "additional_snapshot_offsets_from_end": [],
        },
        "kv_cache_dtype": "auto",
        "kv_cache_quant": None,
        "mamba_ssm_cache_dtype": "auto",
        "mamba_ssm_cache_dtype_resolved": "float32",
        "cuda_graph": {"batch_sizes": list(_PADDED), "enable_padding": True},
        "cuda_graph_captured": list(_PADDED),
        "speculative": None,
        "max_num_tokens": 8192,
        "max_batch_size": 128,
        "max_seq_len": 9216,
        "chunked_prefill": True,
        "overlap_scheduler": True,
        "tp": 1,
        "pp": 1,
        "ep": 1,
        "attention_dp": False,
    }
    config.update(changes)
    return config


def _engine_line(config: dict | None = None) -> str:
    """The engine record a worker-side observer logs for ``config``."""
    config = _engine_config() if config is None else config
    record = {
        "argv": [],
        "worker": True,
        "observer": True,
        "config": config,
        "sources": dict.fromkeys(config, "worker"),
        "pools": {"kv_max_tokens": 262144},
        "memory": {
            "source": "nvml",
            "devices": [{"device": "0", "free_mib": 40960.0, "total_mib": 81920.0}],
        },
        "errors": {},
    }
    return f"{workload_io.ENGINE_MARKER} {json.dumps(record)}\n"


def _records(*texts: str) -> str:
    return "".join(
        f"{workload_io.RECORD_MARKER} {json.dumps({'index': n, 'text': t})}\n"
        for n, t in enumerate(texts)
    )


def _boot_runner(*outputs: str):
    seen: list[str] = []

    def run(command, env=None, timeout=0):
        seen.append(command)
        return CommandResult(returncode=0, stdout=outputs[min(len(seen), len(outputs)) - 1])

    run.seen = seen  # type: ignore[attr-defined]
    return run


def test_a_boot_gate_fails_on_an_engine_that_ran_another_configuration(tmp_path):
    task = _task(tmp_path)
    good = boot_checks.run_boot_gate(
        _env(), task_data=task, unit="core", member="tp1", workdir=tmp_path,
        runner=_boot_runner(_engine_line() + _records("Paris", "light")),
    )  # fmt: skip
    assert good.passed, good.detail
    [record] = good.engine
    assert record["reported"] and not record["mismatches"] and not good.warnings
    assert not record["not_attested"]
    assert "fields as declared" in good.summary()

    # Block reuse off with no warning in the log: only the attestation sees it.
    silent = _engine_line(_engine_config(block_reuse=False)) + _records("Paris", "light")
    bad = boot_checks.run_boot_gate(
        _env(), task_data=task, unit="core", member="tp1", workdir=tmp_path,
        runner=_boot_runner(silent),
    )  # fmt: skip
    assert not bad.passed
    assert bad.detail.startswith("the engine did not run the declared configuration -- ")
    assert "block_reuse: declared on, engine off (read from worker)" in bad.detail
    assert bad.engine[0]["mismatches"]


def test_the_acceptance_gate_holds_each_leg_to_its_own_mode(tmp_path):
    task = _task(tmp_path, features=["mtp"])
    options = {**_OPTIONS, "speculative_config": {"decoding_type": "MTP", "max_draft_len": 3}}
    Path(task["accuracy"]["llm_api_options"]["core"]).write_text(yaml.safe_dump(options))
    mtp = {"algorithm": "MTP", "max_draft_len": 3}
    legs = [
        _engine_line(_engine_config(speculative=mtp)),
        _engine_line(_engine_config(speculative=mtp, modeling_v2="require")),
    ]
    run = _boot_runner(*legs)
    result = acceptance_checks.run_acceptance_gate(
        _env(), task_data=task, runner=run, unit="core", member="tp1", workdir=tmp_path
    )
    assert len(run.seen) == 2 and not result.passed
    assert result.detail.startswith("the engine of the off leg did not run the declared")
    assert "modeling_v2: declared off, engine require" in result.detail
    assert [r["leg"] for r in result.engine] == ["require", "off"]


# A fake ``tensorrt_llm`` and ``pynvml``, for the harness runs of other tests.
_FAKE_TRTLLM = {
    "tensorrt_llm/__init__.py": "",
    "tensorrt_llm/executor/__init__.py": "",
    "tensorrt_llm/executor/base_worker.py": """
        import types


        class BaseWorker:
            def __init__(self, llm_args):
                self.llm_args = llm_args
                self.rank = 0
                self.engine = None

            def setup_engine(self):
                kv = self.llm_args.kv_cache_config
                state = kv.mamba_state_config
                # What the model loader does for a hybrid model, here without a word.
                if not (state.periodic_snapshot_interval
                        or state.additional_snapshot_offsets_from_start):
                    kv.enable_block_reuse = False
                kv.max_gpu_total_bytes = 1 << 30
                stats = types.SimpleNamespace(max_num_blocks=100, tokens_per_block=32)
                quant = types.SimpleNamespace(
                    kv_cache_quant_algo=None, mamba_ssm_cache_dtype="torch.float32")
                self.engine = types.SimpleNamespace(
                    model_engine=types.SimpleNamespace(
                        model=types.SimpleNamespace(
                            model_config=types.SimpleNamespace(quant_config=quant)),
                        _cuda_graph_batch_sizes=[1, 2],
                    ),
                    kv_cache_manager=types.SimpleNamespace(
                        enable_block_reuse=kv.enable_block_reuse,
                        tokens_per_block=32,
                        get_kv_cache_stats=lambda: stats,
                    ),
                )
    """,
    "pynvml.py": """
        import os
        import types


        class NVMLError(Exception):
            pass


        def nvmlInit():
            pass


        def nvmlShutdown():
            pass


        def nvmlDeviceGetCount():
            return 1


        def nvmlDeviceGetHandleByIndex(index):
            return index


        def nvmlDeviceGetMemoryInfo(handle):
            free = float(os.environ.get("FAKE_FREE_MIB", "40960")) * (1 << 20)
            return types.SimpleNamespace(free=int(free), total=80 << 30)
    """,
}
