# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Say which target a configuration routes to, and why.

    python -m tensorrt_llm._torch.staircase.explain \
        --model /path/to/DeepSeek-R1-0528-NVFP4 --tp 4 --ep 4 --attention-dp

Prints the routing module's decision tree as it was actually evaluated, one
line per criterion, ending either in the target's class name and directory or
in the criterion that did not match. This is what a forward-reading decision
tree buys that a set of reverse predicates cannot: an answer to "why did I not
get the target I expected".

``--sm`` defaults to the local device but can be given explicitly, so a
configuration can be explained from a machine that has no GPU.
"""

from __future__ import annotations

import argparse
import sys
from typing import Optional, Tuple

from tensorrt_llm.mapping import Mapping

from ._router_index import STAIRCASE_ROUTERS, StaircaseContext, Trace, routing_module


def _sm(value: Optional[str]) -> Tuple[int, int]:
    if value is not None:
        major, _, minor = value.partition(".")
        return (int(major), int(minor))
    import torch

    assert torch.cuda.is_available(), (
        "no CUDA device visible; pass --sm (e.g. --sm 10.3) to explain a "
        "configuration from a host without one"
    )
    return torch.cuda.get_device_capability()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m tensorrt_llm._torch.staircase.explain",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--model", required=True, help="checkpoint directory")
    p.add_argument("--tp", type=int, default=1, help="tensor_parallel_size")
    p.add_argument("--pp", type=int, default=1, help="pipeline_parallel_size")
    p.add_argument("--ep", type=int, default=-1, help="moe_expert_parallel_size")
    p.add_argument("--moe-tp", type=int, default=-1, help="moe_tensor_parallel_size")
    p.add_argument("--attention-dp", action="store_true", help="enable_attention_dp")
    p.add_argument("--sm", default=None, help="SM version as major.minor; defaults to this device")
    return p


def main(argv: Optional[list] = None) -> int:
    args = build_parser().parse_args(argv)

    from tensorrt_llm._torch.pyexecutor.config_utils import load_pretrained_config

    pretrained_config = load_pretrained_config(args.model)
    world_size = args.tp * args.pp
    mapping = Mapping(
        world_size=world_size,
        tp_size=args.tp,
        pp_size=args.pp,
        moe_ep_size=args.ep,
        moe_tp_size=args.moe_tp,
        enable_attention_dp=args.attention_dp,
    )

    ctx = StaircaseContext(
        pretrained_config=pretrained_config,
        mapping=mapping,
        sm=_sm(args.sm),
        quant_config=None,
        spec_config=None,
        is_disagg=False,
    )

    arch = (pretrained_config.architectures or ["(none)"])[0]
    routing = routing_module(arch)
    if routing is None:
        print(f"{arch}  ->  no staircase routing module")
        print("  routed architectures: " + (", ".join(sorted(STAIRCASE_ROUTERS)) or "(none)"))
        return 1

    family = routing.__name__.rpartition(".")[0].rpartition(".")[2]
    print(f"{arch}  ->  models/{family}/routing.py")

    trace = Trace()
    target = routing.route(ctx, trace)
    for label, value, outcome in trace.steps:
        mark = "no match" if outcome is None else ("ok" if outcome is True else f"-> {outcome}")
        print(f"  {label:<10}{str(value):<52}{mark}")

    if target is None:
        print("  => no target")
        return 1

    module = routing.TARGET_MODULES[target]
    directory = module.rpartition(".")[0].replace(".", "/")
    print(f"  => {target}")
    print(f"     {directory}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
