<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->
# Kimi K3 (`KimiK3ForConditionalGeneration`)

`routing.py` decides which target a deployment builds; read it first.

## Targets

| Target | GPU | Checkpoint | Layout |
|---|---|---|---|
| `kimi_k3_mxfp4__sm_100__tp16_moetp16ep1` | sm_100 | MXFP4 | TP16, MoE TP16 |
| `kimi_k3_mxfp4__sm_100__tp16_moetp4ep4` | sm_100 | MXFP4 | TP16, MoE TP4 x EP4 |
| `kimi_k3_nvfp4__sm_107__tp1` | sm_107 | NVFP4 | disaggregated context server: one rank, DWDP, `moe_config.backend: CUTEDSL` |
| `kimi_k3_nvfp4__sm_107__dep16` | sm_107 | NVFP4 | generation: attention DP over 16 ranks, MoE EP16, `MEGAMOE_CUTEDSL` |
| `kimi_k3_nvfp4__sm_107__helix8_moeep8` | sm_107 | NVFP4 | generation: Helix CP8 (TP1, no attention DP, NCCL off for the all-to-all), MoE EP8, `CUTEDSL_FC12` |
| `kimi_k3_nvfp4__sm_107__helix8_moetp2ep4` | sm_107 | NVFP4 | generation: Helix CP8 as above, MoE TP2 x EP4, `CUTEDSL_FC12` |

## Rubin targets (sm_107)

The NVFP4 checkpoint runs on sm_107 only, on four specialized targets, each one layout of the
recipe below. Routing admits exactly these layouts; anything else raises `NoTarget` with the facts
it saw (`modeling_v2="require"` turns that into an error). The layouts exclude each other, so the
target never needs the disaggregated role: no target behaves differently by role, and routing
records none.

The recipe:

- the NVFP4 checkpoint as stored (NVFP4 routed experts; the vision tower is left unread), with its
  `text_config` shape checked by routing;
- an NVFP4 latent KV cache in KV cache manager V2, the Python NIXL transceiver for disaggregated
  serving;
- DSpark speculative decoding (the drafter is part of the family, `dspark/`);
- programmatic dependent launch: the family's SM107 kernels launch with PDL as a constant
  (`ops/common/sm107_launch.py`), so routing declines a deployment whose workers see
  `TRTLLM_ENABLE_PDL` other than `1` (`llm_args.env_overrides` first, then the environment).

### Code layout

- A specialized target directory holds `__init__.py`, `modeling.py` and `weights.py`, nothing
  else. `modeling.py` starts with the target's constants (`K3Sm107Constants`, `CONSTANTS`), then
  the text config, the blocks (attention residual, KDA, MLA, MoE runtime), and the core and the
  shell with the engine hooks. `weights.py` holds the weight table and its transforms. The two
  helix8 targets each carry their own copy of the helix8 model.
- The targets share code only through the family directories:
  - `kernels/<kind>/`: the kernels. The SM107 ones are `k3r_*`; `k3_*` are the sm_100 targets'.
  - `ops/<kind>/`: their OpWrappers and the runtime pieces several targets and the drafter use
    (the SM107 linear, norm, embedding, MLP, short-conv and all-reduce layers, `sm107_*.py`), and
    `ops/cache/k3_sm107_cache_manager.py`, the KV cache manager the SM107 targets declare
    through `RuntimeHooks.kv_cache_manager_cls`.
  - `common/` (layout-independent glue, the KDA step metadata) and `dspark/` (the drafter).
- Kernels register their torch ops (`torch.library`), as the generic native-code guide describes;
  OpWrappers in `ops/` call them.

### Environment

The SM107 targets and the family code they use read no environment variable
(`test_kimi_k3_no_env_switches.py`): what a target runs is its constants. Engine-wide decisions
the targets follow (forced MNNVL all-reduce, the MNNVL test override) come from the engine's
helpers (`use_mnnvl_fabric_handle`, `MNNVLAllReduce.is_mnnvl`). Only `routing.py` reads the
deployment's environment, to decline.

### Log lines

Each rank logs, at INFO, the target it builds and its constants:

```
Kimi K3 SM107 constants: kimi_k3_nvfp4__sm_107__<layout> (<n>): <field>=<value>, ...
```

### Gates

- **Unit tests** under `tests/unittest/_torch/modeling_v2/models/kimi_k3_vl/`, by the code they
  cover; the single-GPU ones run in the GR100 stage, the CPU ones in `l0_cpu`.
- **Accuracy.** `tests/integration/defs/accuracy/modeling_v2/kimi_k3_vl/<target>/`: full GSM8K
  through the disaggregated perf-sanity harness, one deployment per case (context servers on the
  tp1 target, generation on one of the other three), listed in
  `tests/integration/test_lists/qa/llm_perf_multinode.txt`.
