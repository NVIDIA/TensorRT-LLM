# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Say which target a deployment routes to, and why.

    python -m tensorrt_llm._torch._experimental.modeling_v2.explain \\
        --model /path/to/DeepSeek-R1-0528-NVFP4 --tp 4 --ep 4 --attention-dp \\
        --config serve.yaml

Prints both stages of the decision as they were actually evaluated, one line
per criterion. The identity stage -- checkpoint shape, GPU architecture,
parallel topology -- ends in a target's class name or in the criterion that
did not match. The bounds stage then shows that target's ``within_bounds``
over the deployment's LLM API arguments, and whether it accepted them. This
is what a forward-reading decision tree buys that a set of reverse predicates
cannot: an answer to "why did I not get the target I expected".

The deployment is given the way ``trtllm-serve`` takes it: ``--config`` is
the same YAML of LLM API arguments, and ``--tp``, ``--pp``, ``--ep``,
``--moe-tp`` and ``--attention-dp`` are shorthands for the corresponding
arguments. ``--sm`` defaults to the local device but can be given explicitly,
so a configuration can be explained from a machine that has no GPU.
"""

from __future__ import annotations

import argparse
import sys
from typing import Optional, Tuple

import yaml

from ._router_index import MODELING_V2_ROUTERS, ModelingV2Context, Trace, routing_module

# argparse destination -> LLM API argument.
_SHORTHANDS = {
    "tp": "tensor_parallel_size",
    "pp": "pipeline_parallel_size",
    "ep": "moe_expert_parallel_size",
    "moe_tp": "moe_tensor_parallel_size",
    "attention_dp": "enable_attention_dp",
}


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
        prog="python -m tensorrt_llm._torch._experimental.modeling_v2.explain",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--model", required=True, help="checkpoint directory")
    p.add_argument(
        "--config",
        default=None,
        help="LLM API arguments as YAML, the file trtllm-serve --config takes",
    )
    p.add_argument("--tp", type=int, default=None, help="tensor_parallel_size")
    p.add_argument("--pp", type=int, default=None, help="pipeline_parallel_size")
    p.add_argument("--ep", type=int, default=None, help="moe_expert_parallel_size")
    p.add_argument("--moe-tp", type=int, default=None, help="moe_tensor_parallel_size")
    p.add_argument(
        "--attention-dp", action="store_const", const=True, default=None, help="enable_attention_dp"
    )
    p.add_argument("--sm", default=None, help="SM version as major.minor; defaults to this device")
    return p


def llm_args_from(args: argparse.Namespace):
    """The deployment as ``LLM(...)`` would have been given it.

    The YAML and the shorthands build one and the same ``TorchLlmArgs``, so
    ``within_bounds`` is handed exactly what the engine would hand it; a
    shorthand given explicitly wins over the same key in the file.
    """
    from tensorrt_llm.llmapi.llm_args import TorchLlmArgs

    fields = {}
    if args.config is not None:
        with open(args.config) as f:
            fields.update(yaml.safe_load(f) or {})
    for dest, field in _SHORTHANDS.items():
        value = getattr(args, dest)
        if value is not None:
            fields[field] = value
    return TorchLlmArgs(model=args.model, **fields)


def _print_steps(trace: Trace) -> None:
    for label, value, outcome in trace.steps:
        mark = "no match" if outcome is None else ("ok" if outcome is True else f"-> {outcome}")
        print(f"    {label:<14}{str(value):<48}{mark}")


def main(argv: Optional[list] = None) -> int:
    args = build_parser().parse_args(argv)

    from tensorrt_llm._torch.model_config import ModelConfig

    llm_args = llm_args_from(args)
    mapping = llm_args.parallel_config.to_mapping()

    # The engine's own loader, not a second reading of the checkpoint. It is
    # what fills `quant_config` from hf_quant_config.json, so a tree that gates
    # on quantization is explained against the same value the engine will route
    # on. It touches no CUDA, which is what keeps `--sm` usable off-GPU.
    model_config = ModelConfig.from_pretrained(
        args.model,
        mapping=mapping,
        moe_backend="AUTO",
    )
    ctx = ModelingV2Context.from_model_config(model_config, sm=_sm(args.sm))
    pretrained_config = model_config.pretrained_config

    arch = (pretrained_config.architectures or ["(none)"])[0]
    routing = routing_module(arch)
    if routing is None:
        print(f"{arch}  ->  no modeling_v2 routing module")
        print("  routed architectures: " + (", ".join(sorted(MODELING_V2_ROUTERS)) or "(none)"))
        return 1

    family = routing.__name__.rpartition(".")[0].rpartition(".")[2]
    print(f"{arch}  ->  models/{family}/routing.py")

    identity = Trace()
    target = routing.route(ctx, identity)
    print("  identity:")
    _print_steps(identity)
    if target is None:
        print("  => no target")
        return 1

    bounds = Trace()
    accepted = routing.within_bounds(target, llm_args, ctx, bounds)
    print(f"  bounds of {target}:")
    if bounds.steps:
        _print_steps(bounds)
    else:
        print("    (no criteria: every deployment the identity stage routes here is accepted)")
    if not accepted:
        print(
            f"  => outside the bounds of {target}; the built-in implementation serves this deployment"
        )
        return 1

    module = routing.TARGET_MODULES[target]
    directory = module.rpartition(".")[0].replace(".", "/")
    print(f"  => {target}")
    print(f"     {directory}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
