#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Check the actual ON autotune helpers without importing TRT-LLM's GPU runtime."""

from __future__ import annotations

import argparse
import ast
import functools
import json
import math
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any, List, Optional, Tuple

FUNCTIONS = {
    "_autotune_pl_alpha",
    "synthesize_profiling_topk",
    "_token_back_ready_granularity_choices",
    "_megamoe_tuning_op_name",
    "_unpack_tactic",
    "_is_pow2_in_range",
    "validate_megamoe_tactic",
    "_epi_flag_batch_for_tokens",
    "_expand_megamoe_ready_tactics",
    "enumerate_megamoe_candidate_tactics",
}
CONSTANTS = {
    "_NVFP4_BLOCK_SIZE",
    "_SUPPORTED_MMA_TILE_M",
    "_SUPPORTED_MMA_TILE_N",
    "_AUTOTUNE_PL_SEED",
    "_DEFAULT_TOKEN_BACK_READY_GRANULARITY",
    "_TOKEN_BACK_READY_GRANULARITIES",
    "_TACTIC_LEN",
    "_TACTIC_LEN_V4",
    "_LEGACY_TACTIC_LEN",
    "_FLAG_BATCH_MAX",
    "_GEOMETRIES",
    "_TOKEN_BACK_MODES",
    "_TOKEN_BACK_STORE_BINDING",
    "_WORK_ID_MODE_CANDIDATES",
    "_GROUP_HINTS",
    "_FLAG_BATCHES",
    "_EPI_FLAG_BATCH_SMALL",
    "_EPI_FLAG_BATCH_LARGE",
    "_EPI_FLAG_BATCH_TOKEN_THRESHOLD",
}


def extract_helpers(path, torch):
    tree = ast.parse(path.read_text())
    nodes = [ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)]
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in FUNCTIONS:
            nodes.append(node)
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if any(isinstance(target, ast.Name) and target.id in CONSTANTS for target in targets):
                nodes.append(node)
    ns = dict(
        torch=torch,
        math=math,
        _os=os,
        functools=functools,
        Tuple=Tuple,
        List=List,
        Optional=Optional,
        Any=Any,
        logger=SimpleNamespace(debug=lambda *args: None),
    )
    module = ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[]))
    exec(compile(module, str(path), "exec"), ns)
    return ns


def check(repo, skip_torch=False):
    if skip_torch:
        torch = None
    else:
        import torch

        torch.set_num_threads(4)
    path = repo / "tensorrt_llm/_torch/moe/custom_ops/cute_dsl_megamoe_custom_op.py"
    ns = extract_helpers(path, torch)
    env_keys = ("MEGAMOE_AUTOTUNE_PL_ALPHA", "MEGAMOE_TOKEN_BACK_READY_GRANULARITY")
    saved = {key: os.environ.get(key) for key in env_keys}
    for key in env_keys:
        os.environ.pop(key, None)
    report = {"source": str(path), "torch_checks": "skipped" if skip_torch else "passed"}
    try:
        assert ns["_autotune_pl_alpha"](0) == 0.0
        assert ns["_autotune_pl_alpha"](4) == 0.8
        off_key = ns["_megamoe_tuning_op_name"](0)
        on_key = ns["_megamoe_tuning_op_name"](4)
        assert off_key == "trtllm::cute_dsl_megamoe_nvfp4_blackwell"
        assert on_key != off_key and "alpha=0.8" in on_key
        assert ns["_token_back_ready_granularity_choices"](load_balance=True) == (
            "expert",
            "token_tile",
        )
        candidates = {}
        for tokens in (1, 32, 256, 8192):
            off = ns["enumerate_megamoe_candidate_tactics"](tokens, 107)
            on = ns["enumerate_megamoe_candidate_tactics"](tokens, 107, load_balance=True)
            assert all(ns["_unpack_tactic"](t)[10] == "expert" for t in off)
            assert {ns["_unpack_tactic"](t)[10] for t in on} == {"expert", "token_tile"}
            assert any(t[2] is not None for t in on), "MixCGA not exposed to ON tuning"
            for tactic in on:
                ns["validate_megamoe_tactic"](tactic, sm_version=107)
            candidates[str(tokens)] = {"off": len(off), "on": len(on)}
        legacy = ([256, 128, 256], [2, 1, 1], 512, "static", "epi_warps", True, 1, (1, 1))
        assert ns["_unpack_tactic"](legacy)[10] == "expert"
        assert ns["_unpack_tactic"](ns["_unpack_tactic"](legacy)[:10])[10] == "expert"
        os.environ["MEGAMOE_AUTOTUNE_PL_ALPHA"] = "0.7"
        assert ns["_megamoe_tuning_op_name"](4) != on_key
        for invalid in ("nan", "inf", "-1"):
            os.environ["MEGAMOE_AUTOTUNE_PL_ALPHA"] = invalid
            try:
                ns["_autotune_pl_alpha"](4)
            except ValueError:
                pass
            else:
                raise AssertionError("Invalid exponent accepted: " + invalid)
        os.environ.pop("MEGAMOE_AUTOTUNE_PL_ALPHA")
        os.environ["MEGAMOE_TOKEN_BACK_READY_GRANULARITY"] = "token_tile"
        assert ns["_megamoe_tuning_op_name"](4) != on_key
        pinned = ns["enumerate_megamoe_candidate_tactics"](32, 107, load_balance=True)
        assert pinned and all(ns["_unpack_tactic"](t)[10] == "token_tile" for t in pinned)
        os.environ.pop("MEGAMOE_TOKEN_BACK_READY_GRANULARITY")
        report.update(off_cache_key=off_key, on_cache_key=on_key, candidates=candidates)

        backend = repo / "tensorrt_llm/_torch/moe/fused_moe/mega_moe/mega_moe_cute_dsl.py"
        tree = ast.parse(backend.read_text())
        method = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name == "is_rebalance_active"
        )
        state = SimpleNamespace(is_tuning_mode=False)
        method_ns = {"AutoTuner": SimpleNamespace(get=lambda: state)}
        exec(
            compile(
                ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[])),
                str(backend),
                "exec",
            ),
            method_ns,
        )
        state_cases = 0
        for slots in (0, 4):
            for armed in (False, True):
                for warmup in (False, True):
                    for tuning in (False, True):
                        for opt_in in (False, True):
                            state.is_tuning_mode = tuning
                            obj = SimpleNamespace(
                                _rebalance_slots_active=slots,
                                _rebalance_arm_open=armed,
                                _rebalance_warmup=warmup,
                                tactic_autotune=opt_in,
                            )
                            result = method_ns["is_rebalance_active"](obj)
                            expected = (
                                slots > 0
                                and armed
                                and ((tuning and opt_in) or (not tuning and not warmup))
                            )
                            assert result == expected
                            state_cases += 1
        report["rebalance_state_cases"] = state_cases

        # Reproduce the real executor setter: it marks EVERY ON backend as
        # warmup before AutoTuner enters profiling mode. Internal bucket
        # priming is a separate arm_open=False control, not this flag.
        executor = repo / "tensorrt_llm/_torch/pyexecutor/model_engine.py"
        executor_tree = ast.parse(executor.read_text())
        setter = next(
            node
            for node in ast.walk(executor_tree)
            if isinstance(node, ast.FunctionDef)
            and node.name == "is_warmup"
            and any(
                isinstance(dec, ast.Attribute) and dec.attr == "setter"
                for dec in node.decorator_list
            )
        )
        setter.decorator_list = []
        a2a_warmup = []
        setter_ns = {"_set_moe_a2a_warmup": a2a_warmup.append}
        exec(
            compile(
                ast.fix_missing_locations(ast.Module(body=[setter], type_ignores=[])),
                str(executor),
                "exec",
            ),
            setter_ns,
        )
        opted_in = SimpleNamespace(
            _rebalance_slots_active=4,
            _rebalance_arm_open=True,
            _rebalance_warmup=False,
            tactic_autotune=True,
        )
        opted_out = SimpleNamespace(
            _rebalance_slots_active=4,
            _rebalance_arm_open=True,
            _rebalance_warmup=False,
            tactic_autotune=False,
        )
        off = SimpleNamespace(
            _rebalance_slots_active=0,
            _rebalance_arm_open=True,
            _rebalance_warmup=False,
            tactic_autotune=True,
        )
        engine = SimpleNamespace(
            model=SimpleNamespace(modules=lambda: iter((opted_in, opted_out, off)))
        )
        setter_ns["is_warmup"](engine, True)
        assert opted_in._rebalance_warmup and opted_out._rebalance_warmup
        assert not off._rebalance_warmup
        assert engine._is_warmup and engine.moe_load_balancer_iter_info == (False, False)
        assert a2a_warmup == [True]
        state.is_tuning_mode = True
        assert method_ns["is_rebalance_active"](opted_in)
        assert not method_ns["is_rebalance_active"](opted_out)
        assert not method_ns["is_rebalance_active"](off)
        opted_in._rebalance_arm_open = False
        assert not method_ns["is_rebalance_active"](opted_in), "Internal prime must stay disarmed"
        opted_in._rebalance_arm_open = True
        state.is_tuning_mode = False
        assert not method_ns["is_rebalance_active"](opted_in), "Ordinary warmup must stay disarmed"
        setter_ns["is_warmup"](engine, False)
        assert not engine._is_warmup and engine.moe_load_balancer_iter_info == (True, True)
        assert a2a_warmup == [True, False]
        assert method_ns["is_rebalance_active"](opted_in)
        assert method_ns["is_rebalance_active"](opted_out)
        assert not method_ns["is_rebalance_active"](off)
        report["executor_warmup_integration"] = {
            "setter_source": str(executor),
            "tuning_opt_in_keeps_on": True,
            "tuning_opt_out_stays_disarmed": True,
            "ordinary_warmup_stays_disarmed": True,
            "internal_prime_arm_stays_disarmed": True,
            "serving_on_restored": True,
            "off_stays_disarmed": True,
            "main_all_to_all_warmup_preserved": True,
        }

        if torch is not None:
            workloads = []
            for tokens, topk, experts, ranks in (
                (0, 6, 52, 8),
                (1, 6, 52, 8),
                (3, 6, 52, 8),
                (1000, 5, 52, 8),
                (64, 10, 4, 4),
                (64, 6, 48, 1),
                (8192, 6, 52, 8),
            ):
                gen = ns["synthesize_profiling_topk"]
                kwargs = dict(
                    num_tokens=tokens,
                    num_topk=topk,
                    num_experts_per_rank=experts,
                    world_size=ranks,
                    alpha=0.8,
                    device=torch.device("cpu"),
                )
                aggregate = torch.zeros((ranks, experts), dtype=torch.long)
                sources = range(ranks) if tokens * topk % ranks else range(1)
                for source in sources:
                    idx = gen(**kwargs, source_rank=source)
                    assert idx.shape == (tokens, topk)
                    assert torch.equal(idx, gen(**kwargs, source_rank=source))
                    if tokens:
                        sorted_rows = torch.sort(idx, dim=1).values
                        assert bool((sorted_rows[:, 1:] > sorted_rows[:, :-1]).all())
                        assert int(idx.min()) >= 0 and int(idx.max()) < ranks * experts
                    counts = torch.bincount(idx.reshape(-1), minlength=ranks * experts).view(
                        ranks, experts
                    )
                    assert bool((counts[:, 1:] >= counts[:, :-1]).all())
                    per_rank = counts.sum(1)
                    assert int(per_rank.max() - per_rank.min()) <= 1
                    aggregate += counts
                if tokens * topk % ranks == 0:
                    aggregate *= ranks
                assert aggregate.sum(1).tolist() == [tokens * topk] * ranks
                if tokens:
                    assert bool((aggregate[:, -1] > 0).all()), (
                        "Highest helper slot was not exercised"
                    )
                uniform = gen(**dict(kwargs, alpha=0.0))
                expected = (torch.arange(tokens * topk) % (ranks * experts)).view(tokens, topk)
                assert torch.equal(uniform, expected)
                record = dict(
                    tokens_per_source=tokens,
                    topk=topk,
                    experts_per_rank=experts,
                    world_size=ranks,
                    destination_route_counts=aggregate.sum(1).tolist(),
                    expert_counts=aggregate.tolist(),
                )
                if tokens == 8192:
                    weights = torch.arange(experts, 0, -1, dtype=torch.float64).pow(-0.8)
                    probabilities = aggregate.to(torch.float64) / (tokens * topk)
                    target = weights / weights.sum()
                    error = torch.abs(probabilities - target).sum(1)
                    assert float(error.max()) < 0.12, error.tolist()
                    record["powerlaw_probability_l1_error"] = error.tolist()
                workloads.append(record)
            report["workloads"] = workloads
        report["status"] = "partial_no_torch" if skip_torch else "passed"
        return report
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--skip-torch", action="store_true")
    args = parser.parse_args()
    report = check(args.repo, args.skip_torch)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "workloads"}, indent=2))
